import json

import numpy as np
import pytest

from cafetomo.angular import AngularHist
from cafetomo.beamview import (
    local_contrast,
    opacity_view,
    render,
    smooth_weighted,
    without_floors,
)
from cafetomo.config import ViewSettings

EDGES = np.linspace(-1.0, 1.0, 101)  # 0.02 bins
VIEW = ViewSettings(rebin_factor=2, min_counts=20, min_sky_counts=20, smooth_bins=1.0)


def _hist(values, name="txty"):
    return AngularHist(np.asarray(values, dtype=np.int64), EDGES, EDGES, name)


def _scene(lam_stripe=0.4, ratio=0.25):
    sky = np.full((100, 100), 4000, dtype=np.int64)
    pos = np.round(sky * ratio).astype(np.int64)
    pos[48:52, :] = np.round(pos[48:52, :] * np.exp(-lam_stripe))
    return _hist(pos), _hist(sky), ratio


def test_opacity_is_minus_log_of_live_time_scaled_transmission():
    pos, sky, ratio = _scene()
    v = opacity_view(pos, sky, ratio, VIEW, t_floor=0.05)
    assert v.lam.shape == (50, 50)
    assert np.nanmedian(v.lam[:20]) == pytest.approx(0.0, abs=0.01)
    assert np.nanmean(v.lam[24:26]) == pytest.approx(0.4, abs=0.02)
    assert v.counts_pos.sum() == pos.total and v.counts_sky.sum() == sky.total


def test_sigma_is_poisson_of_both_counts():
    pos, sky, ratio = _scene()
    v = opacity_view(pos, sky, ratio, VIEW, t_floor=0.05)
    n_pos, n_sky = v.counts_pos[5, 5], v.counts_sky[5, 5]
    assert v.sigma[5, 5] == pytest.approx(np.sqrt(1 / n_pos + 1 / n_sky))


def test_bins_below_the_count_floors_are_not_measured():
    pos, sky, ratio = _scene()
    values = pos.values.copy()
    values[:2, :] = 0
    skyv = sky.values.copy()
    skyv[-2:, :] = 3
    v = opacity_view(_hist(values), _hist(skyv), ratio, VIEW, t_floor=0.05)
    assert np.isnan(v.lam[0, :]).all() and np.isnan(v.lam[-1, :]).all()
    assert np.isfinite(v.lam[10, 10])


def test_transmission_floor_caps_opacity():
    pos, sky, ratio = _scene(lam_stripe=6.0)
    v = opacity_view(pos, sky, ratio, VIEW, t_floor=0.05)
    assert np.nanmax(v.lam) <= -np.log(0.05) + 1e-9


def test_opacity_requires_matching_binning():
    pos, sky, ratio = _scene()
    other = AngularHist(sky.values[:50, :50], EDGES[:51], EDGES[:51], "txty")
    with pytest.raises(ValueError, match="binning"):
        opacity_view(pos, other, ratio, VIEW, t_floor=0.05)


def test_smoothing_keeps_a_constant_and_leaves_unmeasured_bins_nan():
    lam = np.full((30, 30), 0.2)
    w = np.ones((30, 30))
    lam[:, :3] = np.nan
    w[:, :3] = 0.0
    out = smooth_weighted(lam, w, 1.5)
    assert np.allclose(out[np.isfinite(out)], 0.2)
    assert np.isnan(out[:, 0]).all()


def test_smoothing_down_weights_noisy_bins():
    lam = np.zeros((9, 9))
    w = np.ones((9, 9))
    lam[4, 4], w[4, 4] = 10.0, 1e-6  # wild but nearly weightless
    assert smooth_weighted(lam, w, 1.0)[4, 4] == pytest.approx(0.0, abs=1e-4)


def test_contrast_is_flat_for_a_smooth_background_and_shows_a_stripe():
    x = np.linspace(-1, 1, 80)
    lam = 0.1 + 0.05 * x[:, None] ** 2 + 0 * x[None, :]
    w = np.ones_like(lam)
    flat = local_contrast(lam, w, smooth_bins=1.0, background_bins=15.0)
    assert np.nanmax(np.abs(flat[10:-10, 10:-10])) < 0.01
    lam[38:42, :] += 0.1
    c = local_contrast(lam, w, smooth_bins=1.0, background_bins=15.0)
    assert c[40, 40] > 0.06 and abs(c[20, 40]) < 0.02


def test_render_writes_figure_arrays_and_summary(tmp_path):
    pos, sky, ratio = _scene()
    v = opacity_view(pos, sky, ratio, VIEW, t_floor=0.05)
    full = opacity_view(pos, sky, ratio, without_floors(VIEW), t_floor=0.05)
    png = render(v, full, VIEW, tmp_path, "pos0")
    assert png.name == "pos0_beams2d.png" and png.stat().st_size > 5_000
    with np.load(tmp_path / "pos0_beams2d.npz") as d:
        assert d["lam"].shape == (50, 50) and {"smoothed", "contrast", "sigma"} <= set(d.files)
        assert {"lam_unfiltered", "smoothed_unfiltered", "contrast_unfiltered"} <= set(d.files)
    doc = json.loads((tmp_path / "pos0_beams2d.json").read_text())
    assert doc["measured_bins"] > 0 and doc["median_sigma"] > 0


def test_without_floors_draws_every_bin_with_counts_in_both():
    pos, sky, ratio = _scene()
    values = pos.values.copy()
    values[:4, :] = 3  # far below the floor, but counted
    values[4:6, :] = 0  # no counts: still not measured
    full = opacity_view(_hist(values), sky, ratio, without_floors(VIEW), t_floor=0.05)
    assert np.isfinite(full.lam[:2]).all() and np.isnan(full.lam[2]).all()
    assert np.isnan(opacity_view(_hist(values), sky, ratio, VIEW, t_floor=0.05).lam[0]).all()


def test_sky_difference_figure_and_its_noise(tmp_path):
    from cafetomo.beamview import render_sky_difference, sky_difference

    pos, sky, ratio = _scene()
    half = _hist(np.round(sky.values * 0.5).astype(np.int64))
    a = opacity_view(pos, half, 2 * ratio, VIEW, t_floor=0.05)
    b = opacity_view(pos, sky, ratio, VIEW, t_floor=0.05)
    d = sky_difference(a, b)
    assert np.nanmax(np.abs(d.delta)) < 1e-2  # same truth, rounding only
    n_a, n_b = a.counts_sky[5, 5], b.counts_sky[5, 5]
    assert d.noise[5, 5] == pytest.approx(np.sqrt(1 / n_a - 1 / n_b))
    png = render_sky_difference({"pos0": d}, VIEW, tmp_path / "sky_difference.png")
    assert png.stat().st_size > 5_000
