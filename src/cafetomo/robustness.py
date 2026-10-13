"""Training-only conditional stability; frequencies are not accuracy probabilities.

World overlap is integrated exactly for piecewise-constant cubic voxels. Coverage
counts are equivalent fully covered replicas, fractional at lattice boundaries.
The supplied opacity-density contrast threshold must be declared from controls.
"""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from cafetomo.angular import AnalysisGrid
from cafetomo.bootstrap import resample
from cafetomo.config import Config
from cafetomo.forward import build_forward_model
from cafetomo.opacity import build_fit_data, opacity_sigma, poisson_sigma, solve_opacity
from cafetomo.prediction import _training_grid, fit_training_fold
from cafetomo.reconstruct import VoxelSolution, solve_voxels
from cafetomo.support import feature_mask_diagnostics, geometry_support, support_rows
from cafetomo.voxels import VoxelGrid


def validate_choices(choices):
    if not choices:
        raise ValueError("at least one declared choice required")
    for choice in choices:
        if choice.get("group") == "support" and "contrast_threshold" in choice:
            raise ValueError("support thresholds are angular significance, not density")
        if choice.get("weight_method") == "poisson" and not choice.get("convergence_evidence"):
            raise ValueError("Poisson weight choices require declared convergence evidence")
    thresholds = {
        c["contrast_threshold"] for c in choices if c.get("group") in {"algorithm", "weights"}
    }
    if len(thresholds) > 1:
        raise ValueError("algorithm and weight variants require one frozen contrast threshold")
    keys = [
        json.dumps({k: v for k, v in c.items() if k != "label"}, sort_keys=True) for c in choices
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate choice configurations")


def _overlap(source, target, axis):
    s = source.origin[axis] + np.arange(source.shape[axis]) * source.spacing
    t = target.origin[axis] + np.arange(target.shape[axis]) * target.spacing
    return csr_matrix(
        np.maximum(
            0,
            np.minimum(t[:, None] + target.spacing, s[None, :] + source.spacing)
            - np.maximum(t[:, None], s[None, :]),
        )
        / target.spacing
    )


def world_resample(values, source, target):
    """Integral over each reference cell, divided by reference cell volume."""
    result = np.asarray(values, float).reshape(source.shape)
    for axis in range(3):
        moved = np.moveaxis(result, axis, 0)
        mapped = _overlap(source, target, axis) @ moved.reshape(moved.shape[0], -1)
        result = np.moveaxis(mapped.reshape((target.shape[axis], *moved.shape[1:])), 0, axis)
    return result


def stability_map(
    volumes: list[VoxelSolution],
    thresholds: list[float],
    coverage: list[np.ndarray],
    *,
    reference_grid: VoxelGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Covered volume-weighted binary contrast frequency and exposure count.

    Independent replicas may yield identical outputs. Missing input replicas,
    invalid coverage and nonfinite measured density are errors, never discarded.
    """
    if not volumes or len(volumes) != len(thresholds) or len(volumes) != len(coverage):
        raise ValueError("matching nonempty replicas, thresholds and coverage required")
    numerator = np.zeros(reference_grid.shape)
    denominator = np.zeros(reference_grid.shape)
    for volume, threshold, mask in zip(volumes, thresholds, coverage, strict=True):
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.size != volume.grid.n_voxels:
            raise ValueError("coverage must be a boolean mask on the source grid")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("finite nonnegative frozen threshold required")
        rho = volume.rho3()
        mask = mask.reshape(volume.grid.shape)
        if np.any(~np.isfinite(rho[mask])):
            raise ValueError("nonfinite measured density")
        numerator += world_resample(mask & (rho > threshold), volume.grid, reference_grid)
        denominator += world_resample(mask, volume.grid, reference_grid)
    frequency = np.divide(
        numerator, denominator, out=np.full(reference_grid.shape, np.nan), where=denominator > 0
    )
    return frequency, denominator


def training_data(grid, live, fold, observation_mask=None):
    """Rebuild frozen weights deterministically and verify their archived hash."""
    clean, masks = _training_grid(grid, fold.cfg, observation_mask)
    cfg = fold.cfg
    sigma = opacity_sigma(
        clean,
        cfg,
        live,
        n_replicas=cfg.opacity.n_sigma_replicas,
        seed=cfg.reconstruction.seed,
        observation_mask=masks,
    )
    data = build_fit_data(solve_opacity(clean, cfg, live, observation_mask=masks), cfg, sigma)
    digest = hashlib.sha256(np.ascontiguousarray(data.w).tobytes()).hexdigest()
    if digest != fold.metadata["weight_hash"]:
        raise ValueError("rebuilt training weights differ from frozen fold")
    return data


def positive_weight_coverage(data, cfg, grid):
    forward = build_forward_model(data.rows, cfg, grid=grid)
    positive = np.isfinite(data.w) & (data.w > 0)
    return (np.asarray(forward.A[positive].sum(axis=0)).ravel() > 0).reshape(grid.shape)


def contrast_location(volume, coverage, threshold):
    """Density-weighted world centroid of declared contrast, not beam truth."""
    density = np.where(coverage & (volume.rho3() > threshold), volume.rho3(), 0)
    mass = density.sum()
    return [
        float(
            (
                density.sum(axis=tuple(a for a in range(3) if a != axis))
                * volume.grid.axis_centers(axis)
            ).sum()
            / mass
        )
        if mass > 0
        else float("nan")
        for axis in range(3)
    ]


def run_count_replicas(
    grid,
    live,
    cfg,
    *,
    n_replicas,
    seed,
    out_dir,
    contrast_threshold,
    pose_mode,
    observation_mask=None,
    frozen_fold=None,
):
    """Training Poisson replicas, fixed settings; failures propagate.

    fixed_pose uses explicitly supplied training-fold pose; each replica estimates
    its own weights. reestimated_pose refits nuisance pose on each replica. Run a
    five-replica pilot before committing resources to a larger pose campaign.
    """
    if n_replicas < 1 or pose_mode not in {"fixed_pose", "reestimated_pose"}:
        raise ValueError("positive replica count and declared pose mode required")
    if pose_mode == "fixed_pose" and frozen_fold is None:
        raise ValueError("fixed pose requires an explicit training fold")
    clean, masks = _training_grid(grid, cfg, observation_mask)
    rng = np.random.default_rng(seed)
    volumes, coverage = [], []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    replica_records = []
    for index in range(n_replicas):
        try:
            replica = resample(clean, rng)
            if pose_mode == "reestimated_pose":
                fold = fit_training_fold(
                    replica, live, cfg, observation_mask=masks, cache_dir=out / "cache"
                )
                data = training_data(replica, live, fold, masks)
                volume, replica_cfg = fold.volume, fold.cfg
            else:
                replica_cfg = replace(cfg, exposures=frozen_fold.cfg.exposures)
                sigma = opacity_sigma(
                    replica,
                    replica_cfg,
                    live,
                    n_replicas=cfg.opacity.n_sigma_replicas,
                    seed=cfg.reconstruction.seed,
                    observation_mask=masks,
                )
                data = build_fit_data(
                    solve_opacity(replica, replica_cfg, live, observation_mask=masks),
                    replica_cfg,
                    sigma,
                )
                volume = solve_voxels(
                    data, replica_cfg, grid=frozen_fold.volume.grid, holdouts=False
                )["full"]
            mask = positive_weight_coverage(data, replica_cfg, volume.grid)
            volume.save(out / f"replica-{index:03d}.npz")
            np.save(out / f"coverage-{index:03d}.npy", mask)
            volumes.append(volume)
            coverage.append(mask)
            replica_records.append(
                dict(
                    index=index,
                    status="success",
                    grid=asdict(volume.grid),
                    solver_info=volume.info,
                    pose_scope=pose_mode,
                )
            )
        except Exception as error:
            replica_records.append(dict(index=index, status="failed", error=str(error)))
            (out / "failures.json").write_text(
                json.dumps(
                    dict(
                        seed=seed,
                        pose_mode=pose_mode,
                        requested=n_replicas,
                        replicas=replica_records,
                    ),
                    indent=2,
                    default=str,
                )
            )
            raise
    frequency, exposure = stability_map(
        volumes, [contrast_threshold] * n_replicas, coverage, reference_grid=volumes[0].grid
    )
    np.savez_compressed(out / "stability.npz", frequency=frequency, exposure=exposure)
    metadata = dict(
        kind="conditional_count",
        pose_mode=pose_mode,
        seed=seed,
        n_replicas=n_replicas,
        replicas=replica_records,
        contrast_threshold=contrast_threshold,
        cfg=asdict(cfg),
        interpretation="conditional stability, not accuracy",
    )
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str))
    return metadata


def run_robustness(
    grid: AnalysisGrid,
    live: dict[str, float],
    cfg: Config,
    choices: list[dict],
    *,
    seed: int,
    out_dir: Path,
) -> dict:
    """One-factor choices using an explicit frozen training nuisance scope.

    Choice keys: group (algorithm/display), overrides (reconstruction/volume),
    contrast_threshold, observation_mask are not supported here: pass only the
    training grid. Pose and weights are fitted once from that grid, then frozen.
    A lattice_shift fraction shifts all world axes. Weight choices keep pose
    frozen and compare analytic sigma against declared Poisson n/2n draws.
    Support choices use angular significance, separately from density display.
    """
    validate_choices(choices)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fold = fit_training_fold(grid, live, cfg, observation_mask=None, cache_dir=out / "cache")
    data = training_data(grid, live, fold)
    grouped = {}
    records = []
    baseline_coverage = positive_weight_coverage(data, fold.cfg, fold.volume.grid)
    for index, choice in enumerate(choices):
        allowed = {
            "group",
            "overrides",
            "contrast_threshold",
            "lattice_shift",
            "label",
            "weight_method",
            "n_sigma_replicas",
            "convergence_evidence",
            "background_degree",
            "support_threshold",
            "tangent_sigma",
        }
        if set(choice) - allowed or choice["group"] not in {
            "algorithm",
            "display",
            "support",
            "weights",
        }:
            raise ValueError("unsupported choice or nuisance scope")
        overrides = choice.get("overrides", {})
        if (
            sum(len(values) for values in overrides.values()) + bool(choice.get("lattice_shift", 0))
            > 1
        ):
            raise ValueError("run one-factor choices before Cartesian sweeps")
        if set(overrides) - {"reconstruction", "volume"}:
            raise ValueError("only reconstruction/volume choices use frozen nuisance scope")
        variant = replace(
            fold.cfg,
            **{
                section: replace(getattr(fold.cfg, section), **values)
                for section, values in overrides.items()
            },
        )
        if choice["group"] == "display" and (overrides or choice.get("lattice_shift", 0)):
            raise ValueError("display choices cannot change reconstruction")
        lattice = None
        if choice.get("lattice_shift", 0):
            if variant.volume != fold.cfg.volume:
                raise ValueError("lattice shift and spacing must be separate one-factor choices")
            lattice = replace(
                fold.volume.grid,
                origin=tuple(
                    x + choice["lattice_shift"] * fold.volume.grid.spacing
                    for x in fold.volume.grid.origin
                ),
            )
        variant_data = data
        if choice["group"] == "weights":
            maps = solve_opacity(grid, variant, live)
            method = choice["weight_method"]
            if method == "analytic":
                sigma = poisson_sigma(grid, variant, live)
            elif method == "poisson":
                if choice["n_sigma_replicas"] < 2:
                    raise ValueError("Poisson weights require at least two replicas")
                sigma = opacity_sigma(
                    grid, variant, live, n_replicas=choice["n_sigma_replicas"], seed=seed
                )
            else:
                raise ValueError("unknown weight method")
            variant_data = build_fit_data(maps, variant, sigma)
            if method == "poisson":
                doubled = opacity_sigma(
                    grid, variant, live, n_replicas=2 * choice["n_sigma_replicas"], seed=seed
                )
                comparison = build_fit_data(maps, variant, doubled, rows=variant_data.rows)
                common_weights = (variant_data.w > 0) & (comparison.w > 0)
                relative = np.abs(
                    np.sqrt(variant_data.w[common_weights] / comparison.w[common_weights]) - 1
                )
                convergence = dict(
                    n=choice["n_sigma_replicas"],
                    doubled_n=2 * choice["n_sigma_replicas"],
                    n_common=int(common_weights.sum()),
                    median_relative_sigma_change=float(np.median(relative))
                    if relative.size
                    else float("nan"),
                    p95_relative_sigma_change=float(np.quantile(relative, 0.95))
                    if relative.size
                    else float("nan"),
                    interpretation="diagnostic only; caller declares acceptance tolerance",
                )
            else:
                convergence = None
        if choice["group"] == "support":
            diagnostics = feature_mask_diagnostics(
                data,
                background_degree=choice["background_degree"],
                threshold=choice["support_threshold"],
            )
            rows, masks = support_rows(data, diagnostics)
            support = geometry_support(
                rows, masks, variant, fold.volume.grid, tangent_sigma=choice["tangent_sigma"]
            )
            covered = support.jointly_covered
            volume = VoxelSolution(
                support.joint_support.ravel(),
                fold.volume.grid,
                {},
                cfg.position_ids,
                info={
                    "estimator": "geometric support fraction, not density",
                    "units": "dimensionless",
                },
            )
            threshold = 0.0
            np.savez_compressed(out / f"support-masks-{index:03d}.npz", **masks)
        else:
            volume = (
                fold.volume
                if choice["group"] == "display"
                else solve_voxels(variant_data, variant, grid=lattice, holdouts=False)["full"]
            )
            covered = positive_weight_coverage(variant_data, variant, volume.grid)
            threshold = choice["contrast_threshold"]
        volume.save(out / f"variant-{index:03d}.npz")
        np.save(out / f"coverage-{index:03d}.npy", covered)
        np.save(out / f"contrast-{index:03d}.npy", covered & (volume.rho3() > threshold))
        grouped.setdefault(choice["group"], []).append((volume, threshold, covered))
        records.append(
            dict(
                choice=choice,
                centroid_xyz_m=contrast_location(volume, covered, threshold),
                centroid_change_xyz_m=(
                    np.asarray(contrast_location(volume, covered, threshold))
                    - np.asarray(contrast_location(fold.volume, baseline_coverage, threshold))
                ).tolist()
                if choice["group"] != "support"
                else None,
                grid=asdict(volume.grid),
                solver_info=volume.info,
                weight_convergence=convergence if choice["group"] == "weights" else None,
                weight_hash=hashlib.sha256(
                    np.ascontiguousarray(variant_data.w).tobytes()
                ).hexdigest(),
            )
        )
    for group, items in grouped.items():
        volumes, thresholds, coverage = zip(*items, strict=True)
        frequency, exposure = stability_map(
            volumes, thresholds, coverage, reference_grid=fold.volume.grid
        )
        common = np.logical_and.reduce(
            [
                world_resample(m, v.grid, fold.volume.grid) >= 1 - 1e-12
                for v, m in zip(volumes, coverage, strict=True)
            ]
        )
        np.savez_compressed(
            out / f"{group}.npz", frequency=frequency, exposure=exposure, common_coverage=common
        )
    result = dict(
        seed=seed,
        source_train_hash=fold.train_hash,
        reference_grid=asdict(fold.volume.grid),
        fitted_pose=fold.metadata["pose"],
        contrast_units="effective opacity density 1/m; support group angular sigma",
        weight_hash=fold.metadata["weight_hash"],
        nuisance_scope="training pose and weights frozen",
        variants=records,
        interpretation="choice sensitivity, not probability or truth",
    )
    (out / "manifest.json").write_text(
        json.dumps(
            result, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)
        )
    )
    return result
