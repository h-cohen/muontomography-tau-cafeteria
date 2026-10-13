"""Alternative consistency estimator sharing measurements and pose, not density.

Angular-bin envelopes are sampled at physical voxel centres. Tangent broadening
is a declared sensitivity assumption, not measured instrumental resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cafetomo.config import Config
from cafetomo.fitdata import FitData, RowIndex
from cafetomo.raycast import bundle_offsets
from cafetomo.voxels import VoxelGrid


@dataclass(frozen=True)
class FeatureMaskDiagnostics:
    masks: dict[str, np.ndarray]
    coefficients: dict[str, np.ndarray]
    excluded: dict[str, np.ndarray]
    eligible: dict[str, np.ndarray]
    powers: tuple[tuple[int, int], ...]
    background_degree: int
    threshold: float


@dataclass(frozen=True)
class SupportVolume:
    grid: VoxelGrid
    per_position: dict[str, np.ndarray]
    jointly_covered: np.ndarray
    joint_support: np.ndarray
    covered_support: dict[str, np.ndarray]
    significant_support: dict[str, np.ndarray]
    tangent_sigma: float
    angular_half_width: float
    n_aperture_sub: int


def feature_mask_diagnostics(
    data: FitData, *, background_degree: int, threshold: float
) -> FeatureMaskDiagnostics:
    """Weighted polynomial background, iteratively excluding positive outliers.

    Threshold is positive contrast in supplied per-row counting sigma units.
    Background clipping uses fixed 3-sigma positive residuals, independently of
    the selection threshold. All masks have the original row length. Polynomial
    terms are sx**i * sy**j in total-degree order; coefficients are unscaled.
    Settings must be frozen using null/development scenes or inner data.
    """
    if background_degree < 0 or not isinstance(background_degree, int):
        raise ValueError("background_degree must be a nonnegative integer")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    powers = tuple((i, d - i) for d in range(background_degree + 1) for i in range(d + 1))
    design = np.column_stack([data.rows.sx**i * data.rows.sy**j for i, j in powers])
    masks, coefficients, excluded, eligible = {}, {}, {}, {}
    for pid in data.rows.position_ids:
        valid = (
            data.rows.mask_for(pid)
            & np.isfinite(data.lam)
            & np.isfinite(data.w)
            & (data.w > 0)
            & np.isfinite(design).all(axis=1)
        )
        keep = valid.copy()
        coef = np.full(len(powers), np.nan)
        for _ in range(30):
            a = design[keep] * np.sqrt(data.w[keep, None])
            b = data.lam[keep] * np.sqrt(data.w[keep])
            if len(b) < len(powers) or np.linalg.matrix_rank(a) < len(powers):
                raise ValueError(f"{pid}: insufficient background coverage")
            coef = np.linalg.lstsq(a, b, rcond=None)[0]
            residual = data.lam - design @ coef
            new = keep & ~(residual * np.sqrt(np.maximum(data.w, 0)) > 3)
            if np.array_equal(new, keep):
                break
            keep = new
        else:
            raise ValueError(f"{pid}: background clipping did not converge")
        masks[pid] = valid & (residual * np.sqrt(np.maximum(data.w, 0)) > threshold)
        coefficients[pid] = coef
        excluded[pid] = valid & ~keep
        eligible[pid] = valid
    return FeatureMaskDiagnostics(
        masks, coefficients, excluded, eligible, powers, background_degree, threshold
    )


def feature_masks(data: FitData, *, background_degree: int, threshold: float) -> dict:
    """Per-position significant row masks; see feature_mask_diagnostics."""
    return feature_mask_diagnostics(
        data, background_degree=background_degree, threshold=threshold
    ).masks


def support_rows(data: FitData, diagnostics: FeatureMaskDiagnostics) -> tuple[RowIndex, dict]:
    """Restrict rows AND masks to measured training coverage for geometry_support."""
    keep = np.logical_or.reduce(list(diagnostics.eligible.values()))
    r = data.rows
    rows = RowIndex(r.position_ids, r.pos_of_row[keep], r.sx[keep], r.sy[keep], r.sky_flat[keep])
    return rows, {p: m[keep] for p, m in diagnostics.masks.items()}


def geometry_support(
    rows: RowIndex,
    masks: dict[str, np.ndarray],
    cfg: Config,
    grid: VoxelGrid,
    *,
    tangent_sigma: float,
) -> SupportVolume:
    """Minimum per-view significant/all angular-envelope support fraction.

    Every supplied row is measured coverage: callers MUST remove zero-weight or
    held-out rows (support_rows does this). Each mask is a full-length row mask,
    false outside its position. World sky bins have half-width sky_t_max/n_bins;
    tangent_sigma adds a one-sigma envelope padding, not a Gaussian probability.
    Admitted midpoint quadrature is averaged for both numerator and denominator.
    Arrays follow grid.shape, [x,y,z]. Only jointly covered centres are measured.
    Rasterization visits a row's slice bounding rectangle, avoiding a global
    voxel-by-row-by-subray array. No inverse matrix, rho or fitted centres enter.
    """
    if not np.isfinite(tangent_sigma) or tangent_sigma < 0:
        raise ValueError("tangent_sigma must be finite and nonnegative")
    if len(rows.position_ids) < 2:
        raise ValueError("joint support requires at least two positions")
    half = cfg.opacity.sky_t_max / cfg.opacity.sky_n_bins
    width = half + tangent_sigma
    axes = [grid.axis_centers(a) for a in range(3)]
    totals, signals, per = {}, {}, {}
    for pid in rows.position_ids:
        mask = np.asarray(masks[pid])
        if mask.shape != (rows.n_rows,) or mask.dtype != np.bool_:
            raise ValueError("feature masks must be full-length boolean row masks")
        position = rows.mask_for(pid)
        if np.any(mask & ~position):
            raise ValueError("feature mask selects another position")
        selected = np.flatnonzero(position)
        pose = cfg.exposure(pid).pose
        offsets = bundle_offsets(
            rows.directions()[selected],
            cfg.detector.aperture_m,
            cfg.volume.n_aperture_sub,
            layer_dz_m=cfg.detector.layer_dz_cm / 100,
            az_deg=pose.az_deg,
        )
        total, signal = np.zeros(grid.shape), np.zeros(grid.shape)
        for row, off in zip(selected, offsets, strict=True):
            for iz, z in enumerate(axes[2]):
                lever = z - pose.z
                if lever <= 0:
                    continue
                radius = width * lever
                cx = pose.x + rows.sx[row] * lever + off[:, 0]
                cy = pose.y + rows.sy[row] * lever + off[:, 1]
                ix0, ix1 = np.searchsorted(
                    axes[0], [cx.min() - radius, cx.max() + radius], side="left"
                )
                iy0, iy1 = np.searchsorted(
                    axes[1], [cy.min() - radius, cy.max() + radius], side="left"
                )
                ix1 = min(ix1 + 1, len(axes[0]))
                iy1 = min(iy1 + 1, len(axes[1]))
                if ix0 == ix1 or iy0 == iy1:
                    continue
                x, y = axes[0][ix0:ix1], axes[1][iy0:iy1]
                hit_x = (abs(x[:, None] - cx[None, :]) <= radius).astype(float)
                hit_y = (abs(y[:, None] - cy[None, :]) <= radius).astype(float)
                fraction = (hit_x @ hit_y.T) / len(off)
                total[ix0:ix1, iy0:iy1, iz] += fraction
                if mask[row]:
                    signal[ix0:ix1, iy0:iy1, iz] += fraction
        totals[pid], signals[pid] = total, signal
        per[pid] = np.divide(signal, total, out=np.full(grid.shape, np.nan), where=total > 0)
    covered = np.logical_and.reduce([v > 0 for v in totals.values()])
    joint = np.minimum.reduce(list(per.values()))
    joint[~covered] = np.nan
    return SupportVolume(
        grid, per, covered, joint, totals, signals, tangent_sigma, half, cfg.volume.n_aperture_sub
    )
