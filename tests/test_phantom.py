import numpy as np

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
