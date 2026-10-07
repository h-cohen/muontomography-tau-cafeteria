import numpy as np
import pytest

from cafetomo.beamdepth import box_path_lengths, fit_beam_depth, zprofile_depth
from cafetomo.config import Pose
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.phantom import beam_ceiling, phantom_data, sky_rows
from cafetomo.reconstruct import VoxelSolution
from cafetomo.voxels import VoxelGrid


def test_box_path_vertical_and_oblique():
    starts = np.zeros((2, 3))
    dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]) / np.array([[1.0], [np.sqrt(2)]])
    L = box_path_lengths(starts, dirs, np.array([-0.5, -1, 7.0]), np.array([0.5, 1, 8.0]))
    assert L[0] == pytest.approx(1.0)
    assert L[1] == pytest.approx(0.0)        # passes x = 7..8 at z = 7..8: misses the box


def test_box_path_clips_corner():
    d = np.array([[0.1, 0.0, 1.0]]) / np.linalg.norm([0.1, 0, 1])
    L = box_path_lengths(np.zeros((1, 3)), d, np.array([0.65, -1, 7.0]), np.array([0.95, 1, 9.0]))
    # enters at z = 7 (x = .70), exits through x = .95 at z = 9.5 -> clipped by z = 9
    assert L[0] == pytest.approx(2.0 * np.sqrt(1.01), rel=1e-6)


def _phantom(cfg, h_true, rng):
    cfg = cfg.with_pose("pos1", Pose(1.78, 0.72, 0.0, 0.0))
    rows = sky_rows(cfg.position_ids, 0.9, 36)
    g = VoxelGrid(origin=(-6.0, -6.0, 6.5), spacing=0.05, shape=(280, 240, 70))
    truth = beam_ceiling(g, xs=(-1.7, 0.0, 1.7, 3.4), z0=7.0, w=0.3, h=h_true,
                         kappa=(1.2,) * 4, y_extent=(-5.0, 5.0), slab_thickness=0.2)
    fwd = build_forward_model(rows, cfg, grid=g)
    like = FitData(lam=np.zeros(rows.n_rows), w=np.full(rows.n_rows, 1 / 0.02**2), rows=rows)
    return cfg, phantom_data(fwd, truth, like, rng)


@pytest.mark.slow
@pytest.mark.parametrize("h_true", [0.6, 1.25])
def test_recovers_depth(cfg, h_true):
    c, data = _phantom(cfg, h_true, np.random.default_rng(3))
    fit = fit_beam_depth(data, c, xs_init=(-1.7, 0.0, 1.7, 3.4), z0_init=7.1)
    assert not fit.at_bound
    assert fit.h == pytest.approx(h_true, abs=0.15)
    assert fit.z0 == pytest.approx(7.0, abs=0.15)


def test_at_bound_flag(cfg):
    c, data = _phantom(cfg, 0.6, np.random.default_rng(4))
    from dataclasses import replace
    tight = replace(c, beamdepth=replace(c.beamdepth, h_max_m=0.3, h_init_m=0.2))
    assert fit_beam_depth(data, tight, xs_init=(-1.7, 0.0, 1.7, 3.4), z0_init=7.1).at_bound


def test_zprofile_measures_column_extent(cfg):
    g = VoxelGrid(origin=(-3.0, -3.0, 5.0), spacing=0.1, shape=(60, 60, 50))
    rho = beam_ceiling(g, xs=(0.0,), z0=7.0, w=0.3, h=1.2, kappa=(1.0,), y_extent=(-3, 3),
                       slab_thickness=0.0)
    sol = VoxelSolution(rho=rho.ravel(), grid=g, offsets={}, position_ids=("pos0", "pos1"))
    zp = zprofile_depth(sol, cfg, xs=(0.0,), w=0.3, z_ref=7.0)
    assert zp["bottom"] == pytest.approx(7.0, abs=0.1)
    assert zp["top"] == pytest.approx(8.2, abs=0.1)
