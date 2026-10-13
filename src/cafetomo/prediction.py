"""Frozen retrospective training fits and detector-coordinate predictive checks.

Scores are descriptive; repeated partitions reuse observations. Replica spread
is conditional counting uncertainty, including the shared roof and fitted pose.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.special import xlogy

from cafetomo.angular import AnalysisGrid
from cafetomo.config import Config
from cafetomo.fitdata import RowIndex
from cafetomo.forward import build_forward_model
from cafetomo.opacity import build_fit_data, opacity_sigma, solve_opacity
from cafetomo.reconstruct import VoxelSolution, solve_voxels
from cafetomo.selfcal import fit_pose
from cafetomo.sky import detector_to_sky
from cafetomo.validation_split import split_counts


@dataclass(frozen=True)
class FrozenFold:
    cfg: Config
    volume: VoxelSolution
    train_hash: str
    choice_hash: str
    metadata: dict = field(default_factory=dict)


def _training_grid(grid, cfg, observation_mask):
    masks = {}
    if observation_mask is not None and set(observation_mask) - set(cfg.position_ids):
        raise ValueError("masks may name only room positions")
    for pid in cfg.position_ids:
        mask = np.asarray(
            (observation_mask or {}).get(pid, np.ones((grid.n_bins, grid.n_bins), dtype=bool))
        )
        if mask.dtype != np.bool_ or mask.shape != (grid.n_bins, grid.n_bins):
            raise ValueError(f"invalid observation mask for {pid}")
        masks[pid] = mask.copy()
    counts = {
        p: np.where(masks[p], v, 0) if p in masks else np.array(v, copy=True)
        for p, v in grid.counts.items()
    }
    return AnalysisGrid(grid.edges.copy(), counts), masks


def _digest(cfg, grid=None, live=None, masks=None):
    h = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode())
    h.update(json.dumps(live, sort_keys=True).encode())
    if grid is not None:
        for name, value in [
            ("edges", grid.edges),
            *sorted(grid.counts.items()),
            *[(f"mask:{p}", v) for p, v in sorted(masks.items())],
        ]:
            arr = np.ascontiguousarray(value)
            h.update(name.encode())
            h.update(str((arr.dtype.str, arr.shape)).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def fit_training_fold(
    grid: AnalysisGrid,
    live: dict[str, float],
    cfg: Config,
    *,
    observation_mask: dict[str, np.ndarray] | None,
    cache_dir: Path,
) -> FrozenFold:
    clean, masks = _training_grid(grid, cfg, observation_mask)
    train_hash = _digest(cfg, clean, live, masks)
    fitted = fit_pose(clean, cfg, live, observation_mask=masks)
    trained_cfg = cfg.with_pose(cfg.selfcal.free_pose, fitted.pose)
    sigma = opacity_sigma(
        clean,
        trained_cfg,
        live,
        n_replicas=cfg.opacity.n_sigma_replicas,
        seed=cfg.reconstruction.seed,
        observation_mask=masks,
    )
    data = build_fit_data(
        solve_opacity(clean, trained_cfg, live, observation_mask=masks), trained_cfg, sigma
    )
    volume = solve_voxels(
        data, trained_cfg, cache_dir=Path(cache_dir) / train_hash, holdouts=False
    )["full"]
    return FrozenFold(
        trained_cfg,
        volume,
        train_hash,
        _digest(trained_cfg),
        dict(
            pose=asdict(fitted.pose),
            grid=asdict(volume.grid),
            n_training_bins={p: int(m.sum()) for p, m in masks.items()},
            n_rows=data.rows.n_rows,
            n_weighted_rows=int(np.count_nonzero(data.w)),
            weight_hash=hashlib.sha256(np.ascontiguousarray(data.w).tobytes()).hexdigest(),
            source_count_totals={p: int(v.sum()) for p, v in clean.counts.items()},
            live=dict(live),
        ),
    )


def predict_raw_counts(
    fold: FrozenFold,
    sky_grid: AnalysisGrid,
    sky_live: dict[str, float],
    target_live: dict[str, float],
) -> dict[str, np.ndarray]:
    """One independent forward row per original detector bin; NaN is no coverage.

    The reference is indexed in detector coordinates, never world aggregates.
    Offsets are fixed at the original measured open-sky zero point.
    """
    cfg = fold.cfg
    sid = cfg.sky_reference.id
    ref = np.asarray(sky_grid.counts[sid], dtype=float)
    tx, ty = sky_grid.tan_mesh()
    if ref.shape != tx.shape:
        raise ValueError("reference shape differs from detector grid")
    if not np.isfinite(sky_live[sid]) or sky_live[sid] <= 0:
        raise ValueError("reference live time must be finite and positive")
    if any(v != 0 for v in fold.volume.offsets.values()):
        raise ValueError("predictive folds require the measured zero-offset gauge")
    out = {}
    for k, pid in enumerate(cfg.position_ids):
        if not np.isfinite(target_live[pid]) or target_live[pid] <= 0:
            raise ValueError("target live times must be finite and positive")
        sx, sy, on_sky = detector_to_sky(tx, ty, cfg.exposure(pid).pose)
        keep = (
            on_sky
            & (cfg.detector.acceptance(tx, ty) > 0)
            & np.isfinite(ref)
            & (ref > 0)
            & (ref >= cfg.opacity.min_sky)
            & (np.abs(sx) <= cfg.opacity.max_tan)
            & (np.abs(sy) <= cfg.opacity.max_tan)
        )
        raw_flat = np.flatnonzero(keep)
        image = np.full(ref.shape, np.nan)
        if raw_flat.size:
            # sky_flat is merely a RowIndex identifier; scattering uses raw_flat.
            rows = RowIndex(
                cfg.position_ids,
                np.full(raw_flat.size, k, dtype=np.int64),
                sx.ravel()[raw_flat],
                sy.ravel()[raw_flat],
                np.arange(raw_flat.size, dtype=np.int64),
            )
            fwd = build_forward_model(rows, cfg, grid=fold.volume.grid)
            # Empty bundle rows have no geometric measurement support.
            supported = np.asarray(fwd.A.sum(axis=1)).ravel() > 0
            lam = fwd.predict(fold.volume.rho)[supported]
            mu = target_live[pid] / sky_live[sid] * ref.ravel()[raw_flat[supported]] * np.exp(-lam)
            if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
                raise ValueError("nonpositive or nonfinite predictive means")
            image.ravel()[raw_flat[supported]] = mu
        out[pid] = image
    return out


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    n, mu = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
    if (
        n.shape != mu.shape
        or np.any(~np.isfinite(n))
        or np.any(n < 0)
        or np.any(~np.isfinite(mu))
        or np.any(mu <= 0)
    ):
        raise ValueError(
            "finite nonnegative observations and positive means with matching shapes required"
        )
    return float(2 * np.sum(xlogy(n, n / mu) - n + mu))


def score_raw_counts(observed, predicted, *, observation_mask=None):
    """Maps, totals and coverage per position; no independent-pull interpretation."""
    out = {}
    for pid, mu in predicted.items():
        n = np.asarray(observed[pid])
        if n.shape != mu.shape:
            raise ValueError("observation/prediction shape mismatch")
        requested = np.asarray((observation_mask or {}).get(pid, np.ones(mu.shape, dtype=bool)))
        if requested.dtype != np.bool_ or requested.shape != mu.shape:
            raise ValueError("invalid scoring mask")
        covered = requested & np.isfinite(mu) & (mu > 0)
        out[pid] = dict(
            observed=n.copy(),
            predicted=mu.copy(),
            residual=np.where(covered, n - mu, np.nan),
            coverage=covered,
            n_requested=int(requested.sum()),
            n_covered=int(covered.sum()),
            observed_total=float(n[covered].sum()),
            predicted_total=float(mu[covered].sum()),
            deviance=poisson_deviance(n[covered], mu[covered]) if covered.any() else np.nan,
        )
    return out


def select_and_refit(
    grid, live, candidates, *, cache_dir, seed=0, fraction=0.5, observation_mask=None
):
    """Choose on an inner count split, then refit the complete outer training set.

    Candidates must be declared before outer evaluation. Angular exclusion masks
    persist through both inner fitting and scoring; sky is split once per inner
    count partition. Compare candidates on their common supported raw directions.
    Returns selected fold, selected index and descriptive scores.
    """
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one declared candidate required")
    clean, masks = _training_grid(grid, candidates[0], observation_mask)
    part = split_counts(clean, live, fraction=fraction, seed=seed)
    predictions = []
    for cfg in candidates:
        fold = fit_training_fold(
            part.train, part.train_live, cfg, observation_mask=masks, cache_dir=cache_dir
        )
        predicted = predict_raw_counts(fold, part.train, part.train_live, part.test_live)
        predictions.append(predicted)
    common_masks = {
        pid: masks[pid]
        & np.logical_and.reduce([np.isfinite(pred[pid]) & (pred[pid] > 0) for pred in predictions])
        for pid in masks
    }
    scores = []
    for predicted in predictions:
        records = score_raw_counts(part.test.counts, predicted, observation_mask=common_masks)
        vals = [v["deviance"] for v in records.values()]
        scores.append(float(sum(vals)) if all(np.isfinite(vals)) else np.inf)
    if not np.isfinite(scores).any():
        raise RuntimeError("all inner candidates failed coverage")
    chosen = int(np.argmin(scores))
    final_fold = fit_training_fold(
        clean, live, candidates[chosen], observation_mask=masks, cache_dir=cache_dir
    )
    metadata = dict(final_fold.metadata)
    metadata["inner_selection"] = dict(
        fraction=fraction,
        seed=seed,
        chosen=chosen,
        scores=scores,
        common_masks=common_masks,
        candidate_available_bins=[
            {
                pid: int(np.count_nonzero(masks[pid] & np.isfinite(pred[pid]) & (pred[pid] > 0)))
                for pid in masks
            }
            for pred in predictions
        ],
    )
    return replace(final_fold, metadata=metadata), chosen, scores


def training_prediction_replicas(
    grid, live, cfg, *, target_live, observation_mask=None, cache_dir, n_replicas, seed
):
    """Refit each training-only Poisson replica, including shared roof and pose.

    Return every replica prediction; exceptions propagate, never drop failed fits.
    Selection settings are fixed before replicas. No target counts are accepted.
    """
    if n_replicas < 1:
        raise ValueError("n_replicas must be positive")
    clean, masks = _training_grid(grid, cfg, observation_mask)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_replicas):
        replica = AnalysisGrid(
            clean.edges, {p: rng.poisson(clean.counts[p]) for p in sorted(clean.counts)}
        )
        fold = fit_training_fold(replica, live, cfg, observation_mask=masks, cache_dir=cache_dir)
        results.append(predict_raw_counts(fold, replica, live, target_live))
    return results


def summarize_prediction_replicas(replicas):
    """Raw-bin conditional mean/spread, with successful replica coverage counts.

    Fewer than two supported values cannot yield an uncertainty estimate.
    """
    if not replicas:
        raise ValueError("at least one replica required")
    out = {}
    for pid in replicas[0]:
        values = np.stack([replica[pid] for replica in replicas])
        supported = np.isfinite(values) & (values > 0)
        count = supported.sum(axis=0)
        total = np.where(supported, values, 0).sum(axis=0)
        mean = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
        squared = np.where(supported, values - mean, 0) ** 2
        variance = np.divide(
            squared.sum(axis=0), count - 1, out=np.full(total.shape, np.nan), where=count > 1
        )
        out[pid] = dict(
            mean=mean, std=np.sqrt(variance), n_supported=count, n_replicas=len(replicas)
        )
    return out
