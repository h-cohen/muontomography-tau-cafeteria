import numpy as np
import pytest

from cafetomo.config import Pose
from cafetomo.sky import detector_to_sky, make_sky_grid


def test_zero_azimuth_is_the_identity_map():
    tx = np.array([0.0, 0.3, -0.5])
    ty = np.array([0.0, -0.2, 0.4])
    sx, sy, valid = detector_to_sky(tx, ty, Pose(0, 0, 0, az_deg=0))
    assert valid.all()
    assert np.allclose(sx, tx)
    assert np.allclose(sy, ty)


def test_yaw_rotates_the_tangent_plane_but_preserves_zenith_angle():
    """A pure yaw cannot change how far from vertical a track is."""
    tx = np.array([0.3, 0.6, -0.4])
    ty = np.array([0.1, -0.2, 0.5])
    sx, sy, valid = detector_to_sky(tx, ty, Pose(0, 0, 0, az_deg=241))
    assert valid.all()
    assert np.allclose(np.hypot(sx, sy), np.hypot(tx, ty))
    assert not np.allclose(sx, tx)  # but the components do move


def test_sky_grid_holds_the_whole_untilted_acceptance():
    """The detector edge is at tan 1.25, so its corner sits at 1.77 < 2.5."""
    grid = make_sky_grid()
    c = np.linspace(-1.25, 1.25, 50)
    tx, ty = np.meshgrid(c, c, indexing="ij")
    sx, sy, valid = detector_to_sky(tx, ty, Pose(0, 0, 0, az_deg=241))
    _, in_grid = grid.bin_index(sx, sy)
    assert valid.all() and in_grid.all()


def test_sky_grid_matches_the_analysis_resolution():
    grid = make_sky_grid()
    assert grid.edges[1] - grid.edges[0] == pytest.approx(0.05)


def test_sky_grid_bin_index_round_trips():
    g = make_sky_grid(t_max=1.0, n_bins=20)
    centers = g.centers
    sx = np.array([centers[3], centers[19], 5.0])
    sy = np.array([centers[7], centers[0], 0.0])
    flat, ok = g.bin_index(sx, sy)
    assert ok[0] and ok[1] and not ok[2]
    assert flat[0] == 3 * 20 + 7
    assert flat[1] == 19 * 20 + 0


def test_sky_grid_flat_size():
    g = make_sky_grid(t_max=1.0, n_bins=20)
    assert g.flat_size == 400
