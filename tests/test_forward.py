from dataclasses import replace

import numpy as np
import pytest

from cafetomo.fitdata import RowIndex
from cafetomo.forward import build_forward_model
from cafetomo.voxels import VoxelGrid


@pytest.fixture
def cfg2(cfg):
    """Real detector and two positions, a small 1-3 m slab so matrices stay tiny."""
    vol = replace(cfg.volume, z_min_m=1.0, z_max_m=3.0, spacing_m=0.5, n_aperture_sub=2)
    return replace(cfg, volume=vol)


def _rows():
    sx = np.array([0.0, 0.2, 0.0, -0.2])
    sy = np.array([0.0, 0.0, 0.1, 0.1])
    return RowIndex(
        position_ids=("pos0", "pos1"),
        pos_of_row=np.array([0, 0, 1, 1]),
        sx=sx,
        sy=sy,
        sky_flat=np.array([0, 1, 2, 3]),
    )


def test_build_sizes_the_grid_from_the_rows(cfg2):
    fwd = build_forward_model(_rows(), cfg2, cache_dir=None)
    assert fwd.A.shape == (4, fwd.grid.n_voxels)
    assert fwd.grid.extent(2) == pytest.approx((1.0, 3.0))
    assert fwd.n_rows == 4


def test_predict_of_a_uniform_volume_is_density_times_path_length(cfg2):
    fwd = build_forward_model(_rows(), cfg2, cache_dir=None)
    x = np.full(fwd.grid.n_voxels, 0.3)
    np.testing.assert_allclose(fwd.predict(x), 0.3 * np.asarray(fwd.A.sum(axis=1)).ravel())


def test_predict_adds_the_per_position_offsets(cfg2):
    fwd = build_forward_model(_rows(), cfg2, cache_dir=None)
    x = np.zeros(fwd.grid.n_voxels)
    got = fwd.predict(x, offsets={"pos0": 0.5, "pos1": -0.25})
    np.testing.assert_allclose(got, [0.5, 0.5, -0.25, -0.25])


def test_predict_accepts_a_3d_volume(cfg2):
    fwd = build_forward_model(_rows(), cfg2, cache_dir=None)
    x3 = np.full(fwd.grid.shape, 0.2)
    np.testing.assert_allclose(fwd.predict(x3), fwd.predict(x3.ravel()))


def test_an_explicit_grid_overrides_the_automatic_one(cfg2):
    g = VoxelGrid(origin=(-1.0, -1.0, 1.0), spacing=0.5, shape=(4, 4, 4))
    fwd = build_forward_model(_rows(), cfg2, grid=g, cache_dir=None)
    assert fwd.grid is g


def test_to_sky_image_places_values_at_their_sky_bins(cfg2):
    fwd = build_forward_model(_rows(), cfg2, cache_dir=None)
    img = fwd.to_sky_image(np.array([1.0, 2.0, 3.0, 4.0]), "pos1", n_sky=2)
    assert img.shape == (2, 2)
    np.testing.assert_allclose(img.ravel(), [np.nan, np.nan, 3.0, 4.0])


def test_the_matrix_is_reused_from_cache_across_calls(cfg2, tmp_path):
    cache = tmp_path / "cache"
    a = build_forward_model(_rows(), cfg2, cache_dir=cache)
    b = build_forward_model(_rows(), cfg2, cache_dir=cache)
    assert (a.A != b.A).nnz == 0
    assert len(list(cache.glob("A_*.npz"))) == 1
