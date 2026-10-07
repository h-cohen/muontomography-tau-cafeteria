from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from cafetomo.fitdata import RowIndex
from cafetomo.forward import ForwardModel
from cafetomo.resolution import (
    alias_period,
    campaign_resolution,
    depth_resolution,
    position_baselines,
    rays_per_voxel,
    views_per_voxel,
)
from cafetomo.voxels import VoxelGrid


@pytest.fixture
def cfg2(cfg):
    """Two positions 1.75 m apart, volume 2-10 m."""
    return replace(cfg, volume=replace(cfg.volume, z_min_m=2.0, z_max_m=10.0, spacing_m=0.25))


def test_depth_resolution_grows_with_the_square_of_distance():
    a = depth_resolution(5.0, baseline_m=2.2, sigma_t=0.05)
    b = depth_resolution(10.0, baseline_m=2.2, sigma_t=0.05)
    assert b == pytest.approx(4 * a, rel=1e-9)


def test_depth_resolution_improves_with_a_longer_baseline():
    assert (depth_resolution(8.0, baseline_m=5.0, sigma_t=0.05)
            < depth_resolution(8.0, baseline_m=2.2, sigma_t=0.05))


def test_depth_resolution_matches_the_closed_form():
    """Two rays converging at z from detectors b apart: Delta_t = b/z, so
    dz = z^2/b * d(Delta_t), and the two views' angular errors add in quadrature."""
    z, b, s = 7.0, 2.2, 0.05
    assert depth_resolution(z, b, s) == pytest.approx(np.sqrt(2) * s * z**2 / b)


def test_a_zero_baseline_has_no_depth_resolution():
    assert depth_resolution(7.0, baseline_m=0.0, sigma_t=0.05) == float("inf")


def test_alias_period_is_the_closed_form():
    assert alias_period(7.0, baseline_m=2.2, feature_pitch_m=1.0) == pytest.approx(
        1.0 * 7.0 / 2.2)


def test_baselines_are_measured_between_positions(cfg2):
    b = position_baselines(cfg2)
    assert b == {("pos0", "pos1"): pytest.approx(1.75)}


def _toy_forward():
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 1, 1))
    # row 0 (pos0) hits voxel 0; rows 1 and 2 (pos1) both hit voxel 0; row 3 hits voxel 1
    A = sparse.csr_matrix(np.array([[1.0, 0.0],
                                    [2.0, 0.0],
                                    [0.5, 0.0],
                                    [0.0, 1.0]]))
    rows = RowIndex(position_ids=("pos0", "pos1"),
                    pos_of_row=np.array([0, 1, 1, 1]),
                    sx=np.zeros(4), sy=np.zeros(4), sky_flat=np.arange(4))
    return ForwardModel(A=A, grid=grid, rows=rows)


def test_views_per_voxel_counts_distinct_positions():
    fwd = _toy_forward()
    v = views_per_voxel(fwd)
    assert v.shape == fwd.grid.shape
    assert v.ravel().tolist() == [2, 1]


def test_rays_per_voxel_counts_every_measured_direction():
    """Views counts positions; rays counts rows. A voxel two pos1 rows cross
    has 1 view from pos1 but 2 rays -- the number the coverage gate uses."""
    fwd = _toy_forward()
    r = rays_per_voxel(fwd)
    assert r.shape == fwd.grid.shape
    assert r.ravel().tolist() == [3, 1]


def test_campaign_resolution_reports_the_baseline_and_depth_error(cfg2):
    r = campaign_resolution(cfg2, sigma_t=0.05, feature_pitch_m=1.0)
    assert r["max_baseline_m"] == pytest.approx(1.75)
    assert r["n_positions"] == 2
    assert set(r["depth_resolution_m"]) == {"z_min", "z_mid", "z_max"}
    # 10 m away on a ~2 m baseline is hopeless and the report must say so
    assert r["depth_resolution_m"]["z_max"] > 2.0
    assert r["z_range_m"] == pytest.approx((2.0, 10.0))


def test_campaign_resolution_flags_when_depth_is_unresolved(cfg2):
    r = campaign_resolution(cfg2, sigma_t=0.05, feature_pitch_m=1.0)
    assert r["depth_resolved"] is False
    assert "depth" in r["verdict"].lower()


def test_a_single_position_campaign_has_no_depth_information(cfg2):
    one = replace(cfg2, exposures=cfg2.exposures[:1])
    r = campaign_resolution(one, sigma_t=0.05, feature_pitch_m=1.0)
    assert r["max_baseline_m"] == 0.0
    assert r["depth_resolution_m"]["z_mid"] == float("inf")
    assert r["depth_resolved"] is False
