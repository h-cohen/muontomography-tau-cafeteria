"""Pure helpers of the production 3D figure scripts."""

import sys

import numpy as np

sys.path.insert(0, "paper-draft/figures")


def test_layers_nearest_heights():
    from make_slices import nearest_layers

    centres = np.arange(1.1, 9.0, 0.2)
    assert nearest_layers(centres, [6.18, 6.9, 7.63]) == [25, 29, 33]


def test_band_average_by_unique_tangent():
    from make_arrayfit import band_average

    sx = np.array([0.1, 0.1, 0.2, 0.2])
    s, v = band_average(sx, np.array([1.0, 3.0, 5.0, 7.0]))
    assert s.tolist() == [0.1, 0.2] and v.tolist() == [2.0, 6.0]


def test_profile_axes_span_fitted_pose():
    from make_pose import scan_axes

    axes = scan_axes({"x": 1.79, "y": 0.62, "az": 0.77}, n=11)
    assert len(axes["x"]) == 11 and np.isclose(axes["x"][5], 1.79)
    assert np.isclose(axes["az"][0], 0.77 - 3) and np.isclose(axes["y"][-1], 0.62 + 0.30)


def test_half_maximum_level():
    from make_coverage import half_maximum

    assert half_maximum(np.array([0.0, 2.0, 1.0])) == 1.0


def test_projected_profiles_share_one_world_grid():
    from make_parallax import projected_profiles

    t = np.linspace(-0.5, 0.5, 11)
    prof = {"pos0": t**2, "pos1": t**2}
    origins = {"pos0": (0.0, 0.0, 0.0), "pos1": (1.0, 0.0, 0.0)}
    grid, out = projected_profiles(t, prof, origins, z=2.0, n=21)
    assert grid[0] == -1.0 and grid[-1] == 2.0 and set(out) == {"pos0", "pos1"}
    assert np.isnan(out["pos0"][-1]) and np.isfinite(out["pos1"][-1])


def test_calibrated_channels_drop_inactive():
    from make_channelcal import channel_points

    rows = [
        {"asic": 0, "channel": 1, "mpv_adc": 6000.0, "status": "ok"},
        {"asic": 0, "channel": 2, "mpv_adc": None, "status": "inactive"},
        {"asic": 1, "channel": 3, "mpv_adc": 7400.0, "status": "mpv_outlier"},
    ]
    pts = channel_points(rows)
    assert [(p["asic"], p["channel"]) for p in pts] == [(0, 1), (1, 3)]
    assert all("flagged" not in p for p in pts)


def test_map_extent_crops_to_display_range():
    from make_maps2d import cropped

    edges = np.linspace(-1.25, 1.25, 101)
    img = np.arange(100 * 100, dtype=float).reshape(100, 100)
    sub, extent = cropped(img, edges, 0.9)
    assert extent[0] >= -0.9 - 1e-9 and extent[1] <= 0.9 + 1e-9
    assert sub.shape[0] == sub.shape[1] == 72


def test_selected_iterate_is_first_minimum():
    from make_convergence import selected

    assert selected([5.0, 1.0, 0.7, 0.7, 0.9]) == 2


def test_depth_resolution_grows_with_height_squared():
    from make_coverage import resolution_curve

    z = np.array([1.0, 2.0])
    dz = resolution_curve(z, sigma_t=0.01, baseline=2.0)
    assert np.allclose(dz, np.sqrt(2) * 0.01 * z**2 / 2.0)


def test_score_normalized_to_its_minimum():
    from make_heights import relative_score

    assert relative_score(np.array([2.0, 1.0, 1.5])).tolist() == [1.0, 0.0, 0.5]


def test_detector_frame_swaps_names_points_and_top_views():
    import frame as style

    img = np.arange(6.0).reshape(2, 3)  # [i_x, j_y]
    style.set_frame("analysis")
    assert style.ax("x") == "x" and style.point(1.0, 2.0) == (1.0, 2.0)
    data, extent = style.top_view(img, (0, 2), (0, 3))
    assert data.shape == (3, 2) and extent == [0, 2, 0, 3]
    style.set_frame("detector")
    try:
        assert style.ax("x") == "y" and style.ax("y") == "x" and style.ax("z") == "z"
        assert style.point(1.0, 2.0) == (2.0, 1.0)
        data, extent = style.top_view(img, (0, 2), (0, 3))
        assert data.shape == (2, 3) and extent == [0, 3, 0, 2]
    finally:
        style.set_frame("analysis")
