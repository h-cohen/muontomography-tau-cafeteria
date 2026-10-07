import numpy as np
import pytest

from cafetomo.autofocus import cv_height_scan, parabola_min
from cafetomo.config import Pose
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.phantom import beam_ceiling, phantom_data, sky_rows
from cafetomo.voxels import VoxelGrid


def test_parabola_min_subgrid():
    zs = np.array([6.8, 7.0, 7.2])
    assert parabola_min(zs, (zs - 7.05) ** 2) == pytest.approx(7.05, abs=1e-9)


def test_parabola_min_edge_returns_grid_point():
    zs = np.array([6.8, 7.0, 7.2])
    assert parabola_min(zs, np.array([1.0, 2.0, 3.0])) == 6.8


@pytest.mark.slow
def test_recovers_phantom_height(cfg):
    cfg = cfg.with_pose("pos1", Pose(1.78, 0.72, 0.0, 0.0))
    rows = sky_rows(cfg.position_ids, 0.9, 36)
    g = VoxelGrid(origin=(-6.0, -6.0, 6.0), spacing=0.1, shape=(140, 120, 26))
    truth = beam_ceiling(g, xs=(-1.7, 0.0, 1.7, 3.4), z0=7.0, w=0.3, h=0.4,
                         kappa=(1.5,) * 4, y_extent=(-4.0, 4.0))
    fwd = build_forward_model(rows, cfg, grid=g)
    like = FitData(lam=np.zeros(rows.n_rows), w=np.full(rows.n_rows, 1 / 0.03**2), rows=rows)
    data = phantom_data(fwd, truth, like, np.random.default_rng(0))
    scan = cv_height_scan(data, cfg)
    assert scan.z_best == pytest.approx(7.2, abs=0.25)   # layer centre: z0 + h/2
