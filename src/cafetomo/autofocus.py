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


def layer_cv_score(data: FitData, cfg: Config, z: float, *,
                   cache_dir: str | Path | None = None) -> float:
    fwd = build_forward_model(data.rows, cfg, grid=layer_grid(cfg, data.rows.t_reach(), z),
                              cache_dir=cache_dir)
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


def parabola_min(zs: np.ndarray, scores: np.ndarray) -> float:
    """Sub-grid minimum by a parabola through the lowest point and its
    neighbours; the grid point itself at the scan edge or a non-convex triple."""
    i = int(np.argmin(scores))
    if i == 0 or i == len(zs) - 1:
        return float(zs[i])
    a, b, c = scores[i - 1], scores[i], scores[i + 1]
    denom = a - 2 * b + c
    if denom <= 0:
        return float(zs[i])
    step = zs[i + 1] - zs[i]
    return float(zs[i] + 0.5 * (a - c) / denom * step)


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
