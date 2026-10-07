import numpy as np
import pytest

from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.phantom import beam_ceiling, phantom_data, sky_rows
from cafetomo.voxels import VoxelGrid


def test_beam_ceiling_geometry():
    g = VoxelGrid(origin=(-2.0, -2.0, 6.0), spacing=0.1, shape=(40, 40, 30))
    v = beam_ceiling(g, xs=(0.0,), z0=7.0, w=0.3, h=1.2, kappa=(1.0,),
                     y_extent=(-1.0, 1.0), slab_thickness=0.2, slab_kappa=0.5)
    zc = g.axis_centers(2)
    col = v[20, 20]
    assert col[(zc > 7.05) & (zc < 8.15)].min() == 1.0
    assert col[zc < 6.95].max() == 0.0
    assert col[(zc > 8.2) & (zc < 8.4)].min() == 0.5


def test_phantom_data_noise_matches_weights(cfg):
    rows = sky_rows(("pos0", "pos1"), 0.8, 32)
    g = VoxelGrid(origin=(-6.0, -6.0, 6.0), spacing=0.25, shape=(48, 48, 12))
    fwd = build_forward_model(rows, cfg, grid=g)
    truth = np.zeros(g.shape)
    like = FitData(lam=np.zeros(rows.n_rows), w=np.full(rows.n_rows, 1 / 0.05**2), rows=rows)
    d = phantom_data(fwd, truth, like, np.random.default_rng(0))
    assert abs(np.std(d.lam) - 0.05) < 0.005
    np.testing.assert_array_equal(d.w, like.w)


def test_zero_weight_rows_stay_unmeasured(cfg):
    rows = sky_rows(("pos0", "pos1"), 0.8, 8)
    g = VoxelGrid(origin=(-6.0, -6.0, 6.0), spacing=0.5, shape=(24, 24, 6))
    fwd = build_forward_model(rows, cfg, grid=g)
    w = np.ones(rows.n_rows)
    w[::3] = 0.0
    like = FitData(lam=np.zeros(rows.n_rows), w=w, rows=rows)
    d = phantom_data(fwd, np.full(g.shape, 0.1), like, np.random.default_rng(0))
    assert np.all(d.lam[w == 0] == 0.0)
    assert np.all(d.w[w == 0] == 0.0)


def test_a_zero_kappa_beam_is_not_overwritten_by_the_slab():
    g = VoxelGrid(origin=(-2.0, -2.0, 6.0), spacing=0.1, shape=(40, 40, 30))
    v = beam_ceiling(g, xs=(0.0,), z0=7.0, w=0.3, h=1.2, kappa=(0.0,),
                     y_extent=(-1.0, 1.0), slab_thickness=0.2, slab_kappa=0.5)
    zc = g.axis_centers(2)
    assert np.all(v[20, 20][(zc > 7.05) & (zc < 8.15)] == 0.0)


def test_beam_is_rasterised_symmetrically_about_its_centre():
    g = VoxelGrid(origin=(-7.0, -2.0, 6.5), spacing=0.1, shape=(140, 40, 15))
    xc = g.axis_centers(0)
    for xk in (0.0, 0.05, -1.7, 1.75):
        v = beam_ceiling(g, xs=(xk,), z0=7.0, w=0.3, h=0.3, kappa=(1.0,),
                         y_extent=(-1.0, 1.0), slab_kappa=0.0)
        col = v.sum(axis=(1, 2))
        assert float((col * xc).sum() / col.sum()) == pytest.approx(xk, abs=1e-9)
