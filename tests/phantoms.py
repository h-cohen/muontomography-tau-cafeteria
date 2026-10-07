"""The shared beam-ceiling phantom: the one ground truth the beam-depth,
measurement and validation tests all reconstruct, so they test one geometry."""

import numpy as np

from cafetomo.config import Config, Pose
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.phantom import beam_ceiling, phantom_data, sky_rows
from cafetomo.sky import SkyGrid, make_sky_grid
from cafetomo.voxels import VoxelGrid

T_MAX = 0.9
N_BINS = 36
XS = (-1.7, 0.0, 1.7, 3.4)
Z0 = 7.0
W = 0.3
KAPPA = (1.2,) * len(XS)


def phantom_sky() -> SkyGrid:
    """The sky grid whose flat indices the phantom's rows use."""
    return make_sky_grid(T_MAX, N_BINS)


def beam_phantom(cfg: Config, h_true: float, rng: np.random.Generator) -> tuple[Config, FitData]:
    """Four beams of depth `h_true` under a slab, seen from the measured 2-D baseline."""
    cfg = cfg.with_pose("pos1", Pose(1.78, 0.72, 0.0, 0.0))
    rows = sky_rows(cfg.position_ids, T_MAX, N_BINS)
    # x spans -9..10 m so every slab-bearing ray (|sx| <= 0.9 up to z = 8.45 m
    # from either position) stays inside the grid: a slab cut off by the grid
    # edge is a sharp background step the smooth background cannot follow.
    g = VoxelGrid(origin=(-9.0, -6.0, 6.5), spacing=0.05, shape=(380, 240, 70))
    truth = beam_ceiling(
        g, xs=XS, z0=Z0, w=W, h=h_true, kappa=KAPPA, y_extent=(-5.0, 5.0), slab_thickness=0.2
    )
    fwd = build_forward_model(rows, cfg, grid=g)
    like = FitData(lam=np.zeros(rows.n_rows), w=np.full(rows.n_rows, 1 / 0.02**2), rows=rows)
    return cfg, phantom_data(fwd, truth, like, rng)
