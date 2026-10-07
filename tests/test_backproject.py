import numpy as np
import pytest

from cafetomo.backproject import backproject_plane, plane_axes
from cafetomo.fitdata import FitData, RowIndex


def _data(sx, sy, lam, pos):
    sx, sy, lam = map(np.asarray, (sx, sy, lam))
    rows = RowIndex(
        position_ids=("pos0", "pos1"),
        pos_of_row=np.asarray(pos),
        sx=sx,
        sy=sy,
        sky_flat=np.arange(sx.size),
    )
    return FitData(lam=lam.astype(float), w=np.ones(sx.size), rows=rows)


def test_a_vertical_ray_lands_directly_above_its_detector(cfg):
    data = _data([0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [0, 1])
    x0, x1 = cfg.origins()["pos0"][0], cfg.origins()["pos1"][0]
    xs = np.array([-1.0, x0, 1.0, x1, 3.0, 5.0])
    ys = np.array([-1.0, 0.0, 1.0])
    per, _ = backproject_plane(data, cfg, z_m=5.0, xs=xs, ys=ys)

    i0 = int(np.nanargmax(np.nan_to_num(per["pos0"], nan=-np.inf)) // ys.size)
    i1 = int(np.nanargmax(np.nan_to_num(per["pos1"], nan=-np.inf)) // ys.size)
    assert xs[i0] == pytest.approx(x0)
    assert xs[i1] == pytest.approx(x1)


def test_a_tilted_ray_lands_at_the_lever_arm_offset(cfg):
    """A ray of tangent 0.4 from z = 0 reaches x = 0.4 * 5 = 2 m at z = 5."""
    data = _data([0.4], [0.0], [3.0], [0])
    xs = np.linspace(-1.0, 5.0, 13)
    ys = np.array([-0.5, 0.0, 0.5])
    per, _ = backproject_plane(data, cfg, z_m=5.0, xs=xs, ys=ys)
    hit = np.unravel_index(np.nanargmax(np.nan_to_num(per["pos0"], nan=-np.inf)), per["pos0"].shape)
    assert xs[hit[0]] == pytest.approx(2.0, abs=0.3)


def test_uncovered_pixels_are_nan_not_zero(cfg):
    data = _data([0.0], [0.0], [1.0], [0])
    xs = np.linspace(-1.0, 1.0, 5)
    ys = np.linspace(-1.0, 1.0, 5)
    per, mean = backproject_plane(data, cfg, z_m=5.0, xs=xs, ys=ys)
    assert np.isnan(per["pos1"]).all()
    assert np.isnan(mean).any()
    assert np.isfinite(per["pos0"]).any()
    np.testing.assert_array_equal(np.isfinite(mean), np.isfinite(per["pos0"]))


def test_rays_outside_the_grid_are_dropped_not_clamped(cfg):
    data = _data([0.0, 1.0], [0.0, 0.0], [1.0, 9.0], [0, 0])
    xs = np.linspace(-1.0, 1.0, 5)
    ys = np.linspace(-1.0, 1.0, 5)
    per, _ = backproject_plane(data, cfg, z_m=5.0, xs=xs, ys=ys)
    assert np.nanmax(per["pos0"]) == 1.0
    assert np.isfinite(per["pos0"]).sum() == 1


def test_the_mean_ignores_positions_with_no_coverage(cfg):
    data = _data([0.0, 0.05, -0.05], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [0, 0, 0])
    xs = np.linspace(-1.0, 1.0, 9)
    ys = np.linspace(-1.0, 1.0, 9)
    _, mean = backproject_plane(data, cfg, z_m=5.0, xs=xs, ys=ys)
    covered = mean[np.isfinite(mean)]
    assert covered.size > 0
    np.testing.assert_allclose(covered, 2.0, atol=1e-6)


def test_plane_axes_spans_every_footprint(cfg):
    xs, ys = plane_axes(cfg, z_m=5.0, t_reach=1.0, res_m=0.5)
    origins = cfg.origins().values()
    assert xs[0] <= min(ox for ox, _, _ in origins) - 5.0
    assert xs[-1] >= max(ox for ox, _, _ in origins) + 5.0
    assert np.allclose(np.diff(xs), 0.5)
