"""Synthetic ceilings with known geometry, projected through the real forward
model: the only ground truth this campaign has."""

import numpy as np

from cafetomo.fitdata import FitData, RowIndex
from cafetomo.forward import ForwardModel
from cafetomo.voxels import VoxelGrid


def sky_rows(position_ids: tuple[str, ...], t_max: float, n_bins: int) -> RowIndex:
    """A full regular (position, direction) row set, with none of the holes of real data."""
    edges = np.linspace(-t_max, t_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ii, jj = np.meshgrid(np.arange(n_bins), np.arange(n_bins), indexing="ij")
    n = len(position_ids)
    return RowIndex(position_ids=tuple(position_ids),
                    pos_of_row=np.repeat(np.arange(n, dtype=np.int64), ii.size),
                    sx=np.tile(centers[ii].ravel(), n), sy=np.tile(centers[jj].ravel(), n),
                    sky_flat=np.tile((ii * n_bins + jj).ravel().astype(np.int64), n))


def beam_ceiling(grid: VoxelGrid, *, xs, z0: float, w: float, h: float, kappa,
                 y_extent: tuple[float, float], slab_thickness: float = 0.2,
                 slab_kappa: float = 0.3) -> np.ndarray:
    """Rectangular beams along y (bottom face z0, width w, depth h, opacity
    density kappa_k) under a uniform slab whose underside is z0 + h."""
    x = grid.axis_centers(0)[:, None, None]
    y = grid.axis_centers(1)[None, :, None]
    z = grid.axis_centers(2)[None, None, :]
    in_y = (y >= y_extent[0]) & (y <= y_extent[1])
    vol = np.zeros(grid.shape)
    in_beam = np.zeros(grid.shape, dtype=bool)
    for xk, kk in zip(xs, kappa, strict=True):
        box = (np.abs(x - xk) <= w / 2) & in_y & (z >= z0) & (z <= z0 + h)
        vol = np.where(box, kk, vol)
        in_beam |= box
    slab = in_y & (z > z0 + h) & (z <= z0 + h + slab_thickness)
    return np.where(slab & ~in_beam, slab_kappa, vol)


def phantom_data(fwd: ForwardModel, truth: np.ndarray, like: FitData,
                 rng: np.random.Generator) -> FitData:
    """Truth projected onto `like`'s rows, plus Gaussian noise at `like`'s sigma.
    Rows with zero weight stay unmeasured."""
    clean = fwd.predict(truth)
    sigma = np.where(like.w > 0, 1.0 / np.sqrt(np.where(like.w > 0, like.w, 1.0)), 0.0)
    lam = np.where(like.w > 0, clean + rng.normal(size=clean.size) * sigma, 0.0)
    return FitData(lam=lam, w=like.w.copy(), rows=like.rows)
