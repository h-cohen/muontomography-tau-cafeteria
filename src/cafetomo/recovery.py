"""Independent phenomenological transmission truth and retrospective recovery.

Truth dimensions are used only by generation and explicitly labelled scoring.
No material-density truth or calibrated interval coverage is claimed.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from cafetomo.angular import AnalysisGrid
from cafetomo.beamdepth import box_path_lengths, estimate_beam_depth
from cafetomo.config import Config, Pose
from cafetomo.opacity import build_fit_data, opacity_sigma, solve_opacity
from cafetomo.prediction import fit_training_fold
from cafetomo.raycast import bundle_offsets
from cafetomo.sky import detector_to_sky
from cafetomo.support import feature_mask_diagnostics, geometry_support, support_rows

TRUTH_CENTRES = (-1.7, 0.0, 1.7, 3.4)
TRUTH_POSE = Pose(1.78, 0.72, 0.0, 0.0)


@dataclass(frozen=True)
class RecoveryCase:
    case_id: str
    bottom_m: float
    thickness_m: float
    width_m: float
    truth_variant: str
    normal_column: float

    def __post_init__(self):
        if self.truth_variant not in (
            "nominal",
            "null",
            "extra_missing",
            "t_shape",
            "structured_response",
        ):
            raise ValueError("unknown truth variant")
        if (
            not np.all(
                np.isfinite([self.bottom_m, self.thickness_m, self.width_m, self.normal_column])
            )
            or min(self.thickness_m, self.width_m) <= 0
            or self.normal_column < 0
        ):
            raise ValueError("invalid truth dimensions or column")


def standard_cases():
    return [
        RecoveryCase(f"nominal_{h:g}", 6.4, h, 0.3, "nominal", 0.06) for h in (0.3, 0.6, 1.2, 1.8)
    ] + [
        RecoveryCase(v, 6.4, 0.6, 0.3, v, 0.06)
        for v in ("null", "extra_missing", "t_shape", "structured_response")
    ]


def truth_transmission(case, cfg, pid, directions, *, n_sub):
    """Fine analytic boxes, uniformly admitted subrays, mean exp(-opacity).

    Baseline smooth overburden has normal optical depth .07. Beams extend
    y=-5..5. T truth adds a flange connected at the beam tops. Extra/missing
    removes x=1.7 and adds x=-3.4. All mismatches start from nominal h=.6.
    """
    pose = TRUTH_POSE if pid == "pos1" else cfg.exposure(pid).pose
    d = np.asarray(directions)
    offsets = bundle_offsets(
        d,
        cfg.detector.aperture_m,
        n_sub,
        layer_dz_m=cfg.detector.layer_dz_cm / 100,
        az_deg=pose.az_deg,
    )
    starts = offsets + np.array([pose.x, pose.y, pose.z])
    dd = np.broadcast_to(d[:, None, :], starts.shape)
    tau = np.broadcast_to(0.07 / d[:, 2, None], starts.shape[:-1]).copy()
    centres = TRUTH_CENTRES if case.truth_variant != "extra_missing" else (-3.4, -1.7, 0.0, 3.4)
    if case.truth_variant != "null":
        for x in centres:
            lo = np.array([x - case.width_m / 2, -5.0, case.bottom_m])
            hi = np.array([x + case.width_m / 2, 5.0, case.bottom_m + case.thickness_m])
            tau += case.normal_column / case.thickness_m * box_path_lengths(starts, dd, lo, hi)
    if case.truth_variant == "t_shape":
        for x in centres:
            tau += 0.2 * box_path_lengths(
                starts,
                dd,
                np.array([x - 0.5, -5.0, case.bottom_m + case.thickness_m]),
                np.array([x + 0.5, 5.0, case.bottom_m + case.thickness_m + 0.2]),
            )
    if case.truth_variant == "structured_response":
        sx, sy = d[:, 0] / d[:, 2], d[:, 1] / d[:, 2]
        tau += (0.025 * (1 + np.sin(7 * sx) * np.cos(5 * sy)))[:, None]
        tau += (0.02 * (sx + 0.4) ** 2)[:, None]
    return np.exp(-tau).mean(axis=1)


def expected_counts(case, cfg, sky_mean, live, *, n_sub):
    sid = cfg.sky_reference.id
    ref = np.asarray(sky_mean.counts[sid], dtype=float)
    if (
        np.any(~np.isfinite(ref))
        or np.any(ref < 0)
        or any(not np.isfinite(live[p]) or live[p] <= 0 for p in (*cfg.position_ids, sid))
    ):
        raise ValueError("invalid reference means or exposure")
    tx, ty = sky_mean.tan_mesh()
    counts = {sid: ref.copy()}
    accepted = cfg.detector.acceptance(tx, ty) > 0
    for pid in cfg.position_ids:
        pose = TRUTH_POSE if pid == "pos1" else cfg.exposure(pid).pose
        sx, sy, above = detector_to_sky(tx, ty, pose)
        keep = accepted & above
        d = np.column_stack([sx[keep], sy[keep], np.ones(keep.sum())])
        d /= np.linalg.norm(d, axis=1)[:, None]
        transmission = np.zeros(ref.shape)
        # Bound memory independently of detector binning and fine quadrature.
        values = np.concatenate(
            [
                truth_transmission(case, cfg, pid, block, n_sub=n_sub)
                for block in np.array_split(d, max(1, (len(d) + 255) // 256))
            ]
        )
        transmission[keep] = values
        counts[pid] = live[pid] / live[sid] * ref * transmission
    return AnalysisGrid(sky_mean.edges.copy(), counts)


def simulate_counts(
    case: RecoveryCase,
    cfg: Config,
    sky_mean: AnalysisGrid,
    live: dict[str, float],
    *,
    seed: int,
    n_sub: int,
) -> AnalysisGrid:
    means = expected_counts(case, cfg, sky_mean, live, n_sub=n_sub)
    rng = np.random.default_rng(seed)
    return AnalysisGrid(means.edges, {p: rng.poisson(v) for p, v in means.counts.items()})


def expected_failure(exc):
    """Whitelist only documented statistical estimator failures; bugs propagate."""
    if isinstance(exc, ValueError):
        return str(
            exc
        ) == "no constrained sky directions: every lambda or sigma is non-finite" or any(
            str(exc).endswith(suffix)
            for suffix in (
                ": insufficient background coverage",
                ": background clipping did not converge",
            )
        )
    return isinstance(exc, RuntimeError) and str(exc).startswith(
        (
            "beam triangulation failed:",
            "pose fit failed:",
            "beam-depth fit failed (status ",
            "beam-depth fit: no fitted row crosses a beam",
        )
    )


def score_fit(case, fit):
    return dict(
        raw_fit=fit.to_json(),
        at_bound=bool(fit.at_bound),
        converged=bool(fit.converged),
        physical_thickness_measured=False,
        null_false_positive=case.truth_variant == "null",
        bias=None
        if case.truth_variant == "null"
        else dict(
            bottom_m=float(fit.z0 - case.bottom_m),
            thickness_m=float(fit.h - case.thickness_m),
            width_m=float(fit.w - case.width_m),
        ),
        truth_scoring_only=True,
    )


def _run_recovery_case(case, grid, live, analysis_cfg, *, seed, out_dir):
    """Full count pipeline; analysis_cfg is the unchanged declared rough prior."""
    start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(
        case=asdict(case),
        seed=seed,
        analysis_prior=asdict(analysis_cfg),
        truth_pose=asdict(TRUTH_POSE),
        status="started",
        physical_thickness_measured=False,
        interval_coverage=None,
        coverage_limitation="No calibrated intervals; conditional synthetic recovery only.",
    )
    (out_dir / "outcome.json").write_text(json.dumps(record, indent=2, default=_json_default))
    # Seed only random streams, never geometry.
    cfg = replace(analysis_cfg, reconstruction=replace(analysis_cfg.reconstruction, seed=seed))
    fold = fit_training_fold(grid, live, cfg, observation_mask=None, cache_dir=out_dir / "cache")
    record["training"] = fold.metadata
    (out_dir / "outcome.json").write_text(json.dumps(record, indent=2, default=_json_default))
    maps = solve_opacity(grid, fold.cfg, live)
    sigma = opacity_sigma(grid, fold.cfg, live, n_replicas=cfg.opacity.n_sigma_replicas, seed=seed)
    data = build_fit_data(maps, fold.cfg, sigma)
    weight_hash = hashlib.sha256(np.ascontiguousarray(data.w).tobytes()).hexdigest()
    if weight_hash != fold.metadata["weight_hash"]:
        raise RuntimeError("recovery weights differ from training fold")
    diag = feature_mask_diagnostics(data, background_degree=2, threshold=3.0)
    rows, masks = support_rows(data, diag)
    support = geometry_support(rows, masks, fold.cfg, fold.volume.grid, tangent_sigma=0.0)
    record["support"] = dict(
        jointly_covered_voxels=int(support.jointly_covered.sum()),
        positive_joint_voxels=int(np.count_nonzero(support.joint_support > 0)),
        significant_rows={p: int(m.sum()) for p, m in masks.items()},
    )
    # A full-volume opacity centroid is a localization diagnostic, not beam depth.
    rho = np.maximum(fold.volume.rho.reshape(fold.volume.grid.shape), 0)
    axes = [fold.volume.grid.axis_centers(a) for a in range(3)]
    centroid = [
        float(
            np.sum(rho * axes[a].reshape(tuple(len(axes[a]) if i == a else 1 for i in range(3))))
            / rho.sum()
        )
        if rho.sum() > 0
        else None
        for a in range(3)
    ]
    centres = (-3.4, -1.7, 0.0, 3.4) if case.truth_variant == "extra_missing" else TRUTH_CENTRES
    truth_centroid = (
        None
        if case.truth_variant == "null"
        else [float(np.mean(centres)), 0.0, case.bottom_m + case.thickness_m / 2]
    )
    record["voxel_beam_localization"] = voxel_localization(case, fold.volume)
    record["localization_xyz_m"] = dict(
        voxel_opacity_centroid=centroid,
        truth_beam_centroid=truth_centroid,
        error_xyz=None
        if truth_centroid is None
        else [None if v is None else v - t for v, t in zip(centroid, truth_centroid, strict=True)],
        limitation=(
            "Whole-volume centroid includes background and finite coverage; "
            "y extent is weakly constrained."
        ),
        truth_scoring_only=True,
    )
    try:
        beams, fit = estimate_beam_depth(data, fold.cfg, maps.sky)
    except RuntimeError as exc:
        if not expected_failure(exc):
            raise
        record.update(
            status="detection_failure"
            if str(exc).startswith("beam triangulation failed:")
            else "fit_failure",
            failure=str(exc),
        )
    else:
        record.update(score_fit(case, fit))
        record.update(
            status="boundary" if fit.at_bound else ("ok" if fit.converged else "fit_failure"),
            beams=beams,
        )
    record["runtime_s"] = time.perf_counter() - start
    (out_dir / "outcome.json").write_text(
        json.dumps(
            record,
            indent=2,
            default=_json_default,
        )
    )
    return record


def precision_budget(n_per_case=20):
    """Predeclared precision, not a claim of calibrated coverage."""
    if n_per_case < 1:
        raise ValueError("positive replica budget required")
    return dict(
        n_per_case=n_per_case,
        worst_case_binomial_se=0.5 / np.sqrt(n_per_case),
        calibrated_coverage=False,
        limitation="Five draws are pilot only; even twenty have wide binomial uncertainty.",
    )


def direct_controls(cfg, *, out_dir, n_sub=16):
    """Known-geometry noiseless controls, explicitly NOT full-count recovery."""
    from cafetomo.beamdepth import fit_beam_depth
    from cafetomo.fitdata import FitData
    from cafetomo.phantom import sky_rows
    from cafetomo.reconstruct import solve_voxels

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_cfg = cfg.with_pose("pos1", TRUTH_POSE)
    rows = sky_rows(cfg.position_ids, 0.85, 36)
    outputs = []
    for case in (standard_cases()[0], standard_cases()[3]):
        lambdas = []
        for quadrature in (n_sub, 2 * n_sub):
            lam = np.empty(rows.n_rows)
            for pid in rows.position_ids:
                keep = rows.mask_for(pid)
                lam[keep] = -np.log(
                    truth_transmission(
                        case, truth_cfg, pid, rows.directions()[keep], n_sub=quadrature
                    )
                )
            lambdas.append(lam)
        data = FitData(lambdas[-1], np.full(rows.n_rows, 1 / 0.002**2), rows)
        fit = fit_beam_depth(data, truth_cfg, xs_init=TRUTH_CENTRES, z0_init=case.bottom_m)
        volume = solve_voxels(data, truth_cfg, cache_dir=out_dir / "cache", holdouts=False)["full"]
        # Peak x per truth-local column; truth is used only for scoring.
        image = volume.rho.reshape(volume.grid.shape).sum(axis=(1, 2))
        x = volume.grid.axis_centers(0)
        errors = []
        for centre in TRUTH_CENTRES:
            use = np.abs(x - centre) < 0.7
            errors.append(float(x[use][np.argmax(image[use])] - centre))
        # Angular parallax precision h^2 * delta_t / baseline.
        scale = (
            (case.bottom_m + case.thickness_m / 2) ** 2
            * (1.7 / 36)
            / np.hypot(TRUTH_POSE.x, TRUTH_POSE.y)
        )
        outputs.append(
            dict(
                case=asdict(case),
                control="known_geometry_noiseless_not_count_pipeline",
                quadrature=[n_sub, 2 * n_sub],
                max_lambda_change=float(np.max(np.abs(lambdas[0] - lambdas[1]))),
                rms_lambda_change=float(np.sqrt(np.mean((lambdas[0] - lambdas[1]) ** 2))),
                voxel_localization=voxel_localization(case, volume),
                voxel_x_errors_m=errors,
                lateral_tolerance_m=volume.grid.spacing,
                lateral_pass=bool(np.max(np.abs(errors)) <= volume.grid.spacing),
                vertical_geometric_scale_m=float(scale),
                fit=score_fit(case, fit),
                vertical_pass=bool(abs(fit.z0 - case.bottom_m) <= scale),
            )
        )
    (out_dir / "direct-controls.json").write_text(json.dumps(outputs, indent=2))
    return outputs


def run_recovery_case(
    case: RecoveryCase,
    grid: AnalysisGrid,
    live: dict[str, float],
    analysis_cfg: Config,
    *,
    seed: int,
    out_dir: Path,
) -> dict:
    """Keep classified statistical failures; unexpected errors propagate."""
    start = time.perf_counter()
    try:
        return _run_recovery_case(case, grid, live, analysis_cfg, seed=seed, out_dir=out_dir)
    except (RuntimeError, ValueError) as exc:
        if not expected_failure(exc):
            raise
        record = dict(
            case=asdict(case),
            seed=seed,
            status="fit_failure",
            failure=str(exc),
            analysis_prior=asdict(analysis_cfg),
            physical_thickness_measured=False,
            runtime_s=time.perf_counter() - start,
            interval_coverage=None,
            coverage_limitation="No interval produced; failure retained.",
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        existing = Path(out_dir) / "outcome.json"
        if existing.exists():
            previous = json.loads(existing.read_text())
            previous.update(record)
            record = previous
        (Path(out_dir) / "outcome.json").write_text(
            json.dumps(record, indent=2, default=_json_default)
        )
        return record


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)}")


def voxel_localization(case, volume):
    """Truth-associated x windows only for scoring, never inference/segmentation.

    Subtract the per-y,z median across x as a simple declared background.
    Positive residuals span ALL reconstructed y,z; report xyz bias and RMS
    extent, including poor vertical localization rather than clipping to truth.
    """
    if case.truth_variant == "null":
        return None
    centres = (-3.4, -1.7, 0.0, 3.4) if case.truth_variant == "extra_missing" else TRUTH_CENTRES
    rho = volume.rho.reshape(volume.grid.shape)
    contrast = np.maximum(rho - np.median(rho, axis=0, keepdims=True), 0.0)
    axes = [volume.grid.axis_centers(a) for a in range(3)]
    results = []
    for centre in centres:
        weight = contrast * (np.abs(axes[0] - centre) < 0.65)[:, None, None]
        mass = weight.sum()
        position, spread = [], []
        for a in range(3):
            coordinate = axes[a].reshape(tuple(len(axes[a]) if a == i else 1 for i in range(3)))
            mean = float((weight * coordinate).sum() / mass) if mass > 0 else None
            position.append(mean)
            spread.append(
                float(np.sqrt((weight * (coordinate - mean) ** 2).sum() / mass))
                if mass > 0
                else None
            )
        target = [centre, 0.0, case.bottom_m + case.thickness_m / 2]
        results.append(
            dict(
                centroid_xyz_m=position,
                rms_extent_xyz_m=spread,
                error_xyz_m=[
                    None if v is None else v - t for v, t in zip(position, target, strict=True)
                ],
            )
        )
    return dict(
        truth_scoring_only=True,
        association_half_width_m=0.65,
        background="per-y,z median across x",
        beams=results,
        limitation="Truth-associated windows; conditional diagnostic, not blind detection.",
    )
