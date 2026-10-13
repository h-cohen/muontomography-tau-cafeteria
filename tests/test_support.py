from dataclasses import replace

import numpy as np

from cafetomo.config import Exposure, OpacitySettings, Pose, Volume
from cafetomo.detector import Detector
from cafetomo.fitdata import FitData, RowIndex
from cafetomo.support import feature_mask_diagnostics, feature_masks, geometry_support
from cafetomo.voxels import VoxelGrid


def scene(cfg):
    cfg = replace(
        cfg,
        detector=Detector(20, 10),
        exposures=(Exposure("a", "", Pose(-1, 0, 0)), Exposure("b", "", Pose(1, 0, 0))),
        opacity=OpacitySettings(sky_t_max=1, sky_n_bins=20),
        volume=Volume(n_aperture_sub=4),
    )
    rows = RowIndex(
        ("a", "b"),
        np.array([0, 0, 1, 1]),
        np.array([0.5, -0.5, -0.5, 0.5]),
        np.zeros(4),
        np.arange(4),
    )
    masks = {"a": np.array([1, 0, 0, 0], bool), "b": np.array([0, 0, 1, 0], bool)}
    return cfg, rows, masks


def test_intersecting_cones_missing_coverage_and_nonfeatures(cfg):
    cfg, rows, masks = scene(cfg)
    grid = VoxelGrid((-2.05, -0.05, 1.95), 0.1, (41, 1, 1))
    result = geometry_support(rows, masks, cfg, grid, tangent_sigma=0)
    assert result.joint_support[20, 0, 0] == 1
    assert np.isnan(result.joint_support[~result.jointly_covered]).all()
    assert not result.jointly_covered[0, 0, 0]
    both = result.jointly_covered
    assert np.all((result.joint_support[both] >= 0) & (result.joint_support[both] <= 1))
    masks["b"][:] = False
    result = geometry_support(rows, masks, cfg, grid, tangent_sigma=0)
    assert result.joint_support[20, 0, 0] == 0


def test_footprint_is_averaged_and_sigma_is_declared_sensitivity(cfg):
    cfg, rows, masks = scene(cfg)
    rows = replace(rows, sx=np.zeros(4))
    cfg = replace(
        cfg,
        exposures=tuple(replace(e, pose=Pose(0, 0, 0)) for e in cfg.exposures),
        opacity=OpacitySettings(sky_t_max=0.01, sky_n_bins=2),
    )
    grid = VoxelGrid((0.065, 0.020, 1.995), 0.01, (1, 1, 1))
    result = geometry_support(rows, masks, cfg, grid, tangent_sigma=0)
    # Coincident significant and nonfeature rows each contribute the same
    # fractional aperture coverage, hence half, never an any-graze binary mask.
    assert result.per_position["a"][0, 0, 0] == 0.5
    assert 0 < result.covered_support["a"][0, 0, 0] < 2
    broad = geometry_support(rows, masks, cfg, grid, tangent_sigma=0.1)
    assert broad.covered_support["a"][0, 0, 0] > result.covered_support["a"][0, 0, 0]


def test_robust_background_threshold_null_and_excluded_rows():
    x, y = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15), indexing="ij")
    x, y = x.ravel(), y.ravel()
    rows = RowIndex(("a",), np.zeros(len(x), int), x, y, np.arange(len(x)))
    base = 2 + 0.2 * x - 0.3 * y
    lam = base.copy()
    feature = (abs(x) < 0.15) & (abs(y) < 0.5)
    lam[feature] += 1
    w = np.full(len(x), 100.0)
    lam[0] = 1e6
    w[0] = 0
    data = FitData(lam, w, rows)
    out = feature_mask_diagnostics(data, background_degree=1, threshold=3)
    np.testing.assert_array_equal(out.masks["a"], feature)
    assert out.excluded["a"][feature].all()
    assert not out.eligible["a"][0]
    np.testing.assert_allclose(out.coefficients["a"], [2, -0.3, 0.2], atol=1e-10)
    assert not feature_masks(data, background_degree=1, threshold=20)["a"].any()
    assert not feature_masks(FitData(base, w, rows), background_degree=1, threshold=3)["a"].any()


def test_zero_weight_rows_removed_from_geometric_coverage(cfg):
    from cafetomo.support import support_rows

    cfg, rows, masks = scene(cfg)
    data = FitData(np.zeros(4), np.array([1, 1, 0, 1]), rows)
    diagnostics = feature_mask_diagnostics(data, background_degree=0, threshold=3)
    live, features = support_rows(data, diagnostics)
    assert live.n_rows == 3
    grid = VoxelGrid((-0.05, -0.05, 1.95), 0.1, (1, 1, 1))
    result = geometry_support(live, features, cfg, grid, tangent_sigma=0)
    assert not result.jointly_covered.any()
    assert np.isnan(result.joint_support).all()


def test_world_bin_envelope_includes_off_centre_voxel(cfg):
    cfg, rows, masks = scene(cfg)
    grid = VoxelGrid((0.035, -0.015, 1.995), 0.01, (1, 1, 1))
    # Both central rays meet x=0, but this world point is inside their angular bins.
    result = geometry_support(rows, masks, cfg, grid, tangent_sigma=0)
    assert result.joint_support.item() == 1
