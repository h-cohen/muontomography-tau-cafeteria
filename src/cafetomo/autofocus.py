"""Autofocus: the ceiling's height from two-view consistency.

For each candidate height z, a thin layer at z is fitted to ONE position and
scored on how well it predicts the OTHER (free offset on the held-out view).
Only at the true height can one layer explain both views. The finite room and
the physical forward model break the periodic-beam aliasing that fools plain
image correlation. Residuals are trimmed per view so isolated aliased bins
cannot spike the curve.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cafetomo.config import Config
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.inversion import solve
from cafetomo.voxels import VoxelGrid, auto_grid


@dataclass(frozen=True)
class FocusScan:
    zs: np.ndarray
    scores: np.ndarray
    z_best: float

    def to_json(self) -> dict:
        return {"z": self.z_best, "scan_z": self.zs.tolist(), "scan_score": self.scores.tolist()}


def layer_grid(cfg: Config, t_reach: float, z: float) -> VoxelGrid:
    s = cfg.autofocus
    vol = replace(cfg.volume, z_min_m=z - s.layer_thickness_m / 2,
                  z_max_m=z + s.layer_thickness_m / 2, spacing_m=s.spacing_m)
    return auto_grid(vol, cfg.origins(), t_reach, aperture_m=cfg.detector.aperture_m)


def layer_grids(cfg: Config, t_reach: float, z: float) -> tuple[VoxelGrid, VoxelGrid]:
    """The layer lattice at z and its twin shifted by half a voxel in x and y.

    The beams' parallax between the two views is a fraction of a voxel per
    height step, so the score also follows where the lattice cells sit under
    the beams: one phase alone moved the phantom's minimum by 0.12 m. Scoring
    both phases and averaging suppresses that. Each lattice gets one extra cell
    per horizontal axis so the shifted one still holds the full footprint."""
    g = layer_grid(cfg, t_reach, z)
    nx, ny, nz = g.shape
    shape = (nx + 1, ny + 1, nz)
    half = 0.5 * g.spacing
    return (VoxelGrid(origin=g.origin, spacing=g.spacing, shape=shape),
            VoxelGrid(origin=(g.origin[0] - half, g.origin[1] - half, g.origin[2]),
                      spacing=g.spacing, shape=shape))


def _require_rows(data: FitData) -> None:
    for pid in data.rows.position_ids:
        if not np.any(data.rows.mask_for(pid) & (data.w > 0)):
            raise ValueError(f"position {pid!r} has no positive-weight rows; "
                             "cross-position validation needs every position")


def _lattice_cv_score(data: FitData, cfg: Config, grid: VoxelGrid,
                      cache_dir: str | Path | None) -> float:
    fwd = build_forward_model(data.rows, cfg, grid=grid, cache_dir=cache_dir)
    rc = replace(cfg.reconstruction, algorithm="tv", tv_z_weight=0.0)
    scores = []
    for train in data.rows.position_ids:
        x, _ = solve(fwd, data.restricted(data.rows.mask_for(train)), rc, fit_offsets=True)
        pred = fwd.predict(x)
        for test in data.rows.position_ids:
            if test == train:
                continue
            sel = data.rows.mask_for(test) & (data.w > 0)
            w = data.w[sel]
            r = data.lam[sel] - pred[sel]
            r2 = w * (r - (w * r).sum() / w.sum()) ** 2
            scores.append(float(r2[r2 <= np.percentile(r2, cfg.autofocus.trim_pct)].mean()))
    return float(np.mean(scores))


def layer_cv_score(data: FitData, cfg: Config, z: float, *,
                   cache_dir: str | Path | None = None) -> float:
    """Trimmed held-out residual of a layer at z fitted to each position alone,
    averaged over both lattice phases (see layer_grids)."""
    _require_rows(data)
    return float(np.mean([_lattice_cv_score(data, cfg, g, cache_dir)
                          for g in layer_grids(cfg, data.rows.t_reach(), z)]))


def parabola_min(zs: np.ndarray, scores: np.ndarray) -> float:
    """Sub-grid minimum by a parabola through the lowest point and its
    neighbours, which may be unevenly spaced; the grid point itself at the scan
    edge. The first interior minimum has its left neighbour strictly higher
    and its right one no lower, so that triple is always strictly convex."""
    i = int(np.argmin(scores))
    if i == 0 or i == len(zs) - 1:
        return float(zs[i])
    z0, z1, z2 = (float(v) for v in zs[i - 1:i + 2])
    y0, y1, y2 = (float(v) for v in scores[i - 1:i + 2])
    h0, h1 = z1 - z0, z2 - z1
    num = h0**2 * (y1 - y2) - h1**2 * (y1 - y0)
    den = h0 * (y1 - y2) + h1 * (y1 - y0)
    return float(z1 - 0.5 * num / den)


def cv_height_scan(data: FitData, cfg: Config, *,
                   cache_dir: str | Path | None = None) -> FocusScan:
    """Coarse sweep over the configured band, then a fine sweep around its minimum."""
    s = cfg.autofocus
    scored: dict[float, float] = {}
    for z in np.round(np.arange(s.z_min_m, s.z_max_m + 1e-9, s.coarse_m), 3):
        scored[float(z)] = layer_cv_score(data, cfg, float(z), cache_dir=cache_dir)
    zc = min(scored, key=scored.get)
    for z in np.round(np.arange(zc - s.coarse_m, zc + s.coarse_m + 1e-9, s.fine_m), 3):
        z = float(z)
        if z not in scored and s.z_min_m <= z <= s.z_max_m:
            scored[z] = layer_cv_score(data, cfg, z, cache_dir=cache_dir)
    zs = np.array(sorted(scored))
    scores = np.array([scored[z] for z in zs])
    return FocusScan(zs=zs, scores=scores, z_best=parabola_min(zs, scores))
