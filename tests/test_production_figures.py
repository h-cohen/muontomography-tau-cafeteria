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
    from make_zprofile import half_maximum

    assert half_maximum(np.array([0.0, 2.0, 1.0])) == 1.0


def test_projected_profiles_share_one_world_grid():
    from make_parallax import projected_profiles

    t = np.linspace(-0.5, 0.5, 11)
    prof = {"pos0": t**2, "pos1": t**2}
    origins = {"pos0": (0.0, 0.0, 0.0), "pos1": (1.0, 0.0, 0.0)}
    grid, out = projected_profiles(t, prof, origins, z=2.0, n=21)
    assert grid[0] == -1.0 and grid[-1] == 2.0 and set(out) == {"pos0", "pos1"}
    assert np.isnan(out["pos0"][-1]) and np.isfinite(out["pos1"][-1])
