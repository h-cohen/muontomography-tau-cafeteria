"""Single-plane backprojection of the measured opacity: the model-free view.

Each measured lambda is placed where its own ray crosses a horizontal plane. No
solver, no regulariser, no grid prior. This is what the detector saw, in site
coordinates.

It is the reference the 3D reconstruction is read against: structure present in
the backprojection is data; structure present only in the reconstruction came
from the regulariser and must be described that way.
"""
from __future__ import annotations

import numpy as np

from cafetomo.config import Config
from cafetomo.fitdata import FitData


def plane_axes(cfg: Config, z_m: float, t_reach: float,
               res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres of a plane covering every position's footprint at z_m."""
    origins = cfg.origins()
    xs_b, ys_b = [], []
    for px, py, pz in origins.values():
        reach = t_reach * max(z_m - pz, 0.0)
        xs_b += [px - reach, px + reach]
        ys_b += [py - reach, py + reach]
    return (np.arange(min(xs_b), max(xs_b) + res_m, res_m),
            np.arange(min(ys_b), max(ys_b) + res_m, res_m))


def backproject_plane(data: FitData, cfg: Config, z_m: float,
                      xs: np.ndarray, ys: np.ndarray
                      ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Place each position's measured opacity onto the plane z = z_m.

    Each ray is binned into its single nearest pixel — no smoothing, no
    triangulated surface. A continuous interpolant (e.g. griddata) already adds
    a shape assumption between measured directions, which is exactly the kind
    of prior this anchor exists to be free of, and it is undefined besides for
    the single- or collinear-ray footprints every real position has near its
    edges.

    Rays landing more than half a pixel outside the grid are dropped, not
    clamped into edge pixels. A position at or above the plane (lever <= 0)
    contributes nothing.

    Returns ({position id: grid[len(xs), len(ys)]}, mean over covered positions).
    Pixels a position does not cover are NaN, never zero: "no ray went there"
    and "no material there" are different statements.
    """
    origins = cfg.origins()
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    half_x = 0.5 * float(np.median(np.diff(xs))) if xs.size > 1 else 0.0
    half_y = 0.5 * float(np.median(np.diff(ys))) if ys.size > 1 else 0.0

    per: dict[str, np.ndarray] = {}
    for pid in data.rows.position_ids:
        sel = data.rows.mask_for(pid) & (data.w > 0)
        ox, oy, oz = origins[pid]
        lever = z_m - oz
        total = np.zeros((xs.size, ys.size))
        count = np.zeros((xs.size, ys.size))
        if sel.any() and lever > 0:
            px = ox + data.rows.sx[sel] * lever
            py = oy + data.rows.sy[sel] * lever
            inside = ((px >= xs.min() - half_x) & (px <= xs.max() + half_x)
                      & (py >= ys.min() - half_y) & (py <= ys.max() + half_y))
            px, py, lam = px[inside], py[inside], data.lam[sel][inside]
            ix = np.argmin(np.abs(xs[:, None] - px[None, :]), axis=0)
            iy = np.argmin(np.abs(ys[:, None] - py[None, :]), axis=0)
            np.add.at(total, (ix, iy), lam)
            np.add.at(count, (ix, iy), 1.0)
        per[pid] = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)

    stack = np.stack([per[pid] for pid in data.rows.position_ids])
    n_ok = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.where(n_ok > 0, np.nansum(stack, axis=0) / np.maximum(n_ok, 1), np.nan)
    return per, mean
