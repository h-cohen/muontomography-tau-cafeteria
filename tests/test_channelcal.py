import json

import numpy as np
import pytest
from scipy.stats import landau

from cafetomo.channelcal import _mode, fit_channel, langau, render, summarize
from cafetomo.config import CalibrationSettings
from cafetomo.rawdata import CHARGE_EDGES, PEDESTAL_EDGES

SET = CalibrationSettings()
RNG = np.random.default_rng(3)


# the observable peak of the smeared spectrum sits above the unsmeared Landau mode
TRUE_MPV = _mode(6000 + 0.4293 * 450, 450, 300)


def _spectrum(mpv=6000.0, xi=450.0, sigma=300.0, n=60_000, ped=2200.0):
    # scipy's standard Landau has its mode at -0.4293; place the mode at `mpv`
    q = landau.rvs(loc=mpv + 0.4293 * xi, scale=xi, size=n, random_state=RNG)
    q = q + RNG.normal(0, sigma, n) + ped
    return np.histogram(np.clip(q, 0, CHARGE_EDGES[-1] - 1), CHARGE_EDGES)[0]


def test_langau_is_the_landau_density_when_unsmeared():
    x = np.arange(0, 12000, 50.0) + 25
    loc, xi = 6000 + 0.4293 * 450, 450.0
    y = langau(x, loc=loc, xi=xi, sigma=1.0)
    assert x[y.argmax()] == pytest.approx(6000, abs=50)
    assert y == pytest.approx(landau.pdf(x, loc=loc, scale=xi), rel=0.02, abs=1e-6)


def test_fit_recovers_mpv_width_and_error_scale():
    f = fit_channel(_spectrum(), pedestal=2200.0, settings=SET)
    assert f.ok
    assert f.mpv == pytest.approx(TRUE_MPV, abs=40)
    assert f.xi == pytest.approx(450, rel=0.1)
    assert 0 < f.mpv_err < 25
    assert 0.5 < f.chi2_ndf < 2.0


def test_fit_error_covers_the_scatter_between_independent_spectra():
    mpvs = [fit_channel(_spectrum(n=8000), 2200.0, SET).mpv for _ in range(12)]
    errs = [fit_channel(_spectrum(n=8000), 2200.0, SET).mpv_err for _ in range(12)]
    assert np.std(mpvs) == pytest.approx(np.mean(errs), rel=0.6)


def test_saturated_tail_is_outside_the_fit_window():
    h = _spectrum().astype(float)
    h[np.searchsorted(CHARGE_EDGES, 2200 + SET.fit_max_adc + 100) :] += 5000  # pile-up
    assert fit_channel(h.astype(int), 2200.0, SET).mpv == pytest.approx(TRUE_MPV, abs=60)


def test_too_few_counts_is_flagged_not_fitted():
    f = fit_channel(_spectrum(n=300), 2200.0, SET)
    assert not f.ok and f.status == "low_stats" and np.isnan(f.mpv)


def test_noise_only_spectrum_fails_loudly_in_the_status():
    h = np.zeros(len(CHARGE_EDGES) - 1, dtype=np.int64)
    h[5] = 50_000  # all counts in one bin far below any Landau peak
    f = fit_channel(h, 2200.0, SET)
    assert not f.ok and np.isnan(f.mpv) and f.status in {"fit_failed", "low_stats"}


def _file_hists(mpv_by_channel, n=20_000):
    hit = np.zeros((4, 32, len(CHARGE_EDGES) - 1), dtype=np.int64)
    ped = np.zeros((4, 32, len(PEDESTAL_EDGES) - 1), dtype=np.int64)
    ped_bin = np.searchsorted(PEDESTAL_EDGES, 2200.0, side="right") - 1
    for a in range(4):
        for c in range(32):
            hit[a, c] = _spectrum(mpv_by_channel[a][c], n=n)
            ped[a, c, ped_bin] = 100_000
    return hit, ped


def test_summary_flags_the_outlier_channel_and_reports_day_spread():
    mpv = [[6000.0] * 32 for _ in range(4)]
    mpv[1][7] = 4200.0  # a low-gain channel
    days = [_file_hists(mpv, n=8000) for _ in range(2)]
    bars = tuple(tuple(range(23)) for _ in range(4))
    doc = summarize(
        [d[0] for d in days],
        [d[1] for d in days],
        n_flagged=np.full((4, 32), 70_000),
        n_valid_rows=100_000,
        channel_bars=bars,
        settings=SET,
    )
    rows = {(r["asic"], r["channel"]): r for r in doc["channels"]}
    assert rows[(1, 7)]["status"] == "mpv_outlier"
    assert rows[(1, 8)]["status"] == "ok"
    assert rows[(0, 3)]["mpv_adc"] == pytest.approx(TRUE_MPV, abs=40)
    assert rows[(0, 3)]["day_rms_adc"] < 60 and len(rows[(0, 3)]["day_mpv_adc"]) == 2
    assert rows[(0, 3)]["bar"] == 3 and rows[(0, 30)]["bar"] is None
    assert doc["asics"][0]["median_mpv_adc"] == pytest.approx(TRUE_MPV, abs=40)
    assert doc["n_ok"] == 127 and doc["n_flagged"] == 1


def test_render_writes_asic_grids_summary_and_json(tmp_path):
    mpv = [[6000.0 + 20 * c for c in range(32)] for _ in range(4)]
    h, p = _file_hists(mpv, n=6000)
    bars = tuple(tuple(range(23)) for _ in range(4))
    doc = summarize([h], [p], np.full((4, 32), 9000), 10_000, bars, SET)
    render(doc, [h], [p], tmp_path, "pos0", SET)
    for name in (
        "pos0_channelcal_asic0.png",
        "pos0_channelcal_asic3.png",
        "pos0_channelcal_summary.png",
    ):
        assert (tmp_path / name).stat().st_size > 5_000
    assert json.loads((tmp_path / "pos0_channelcal.json").read_text())["n_ok"] > 100


def test_fit_far_from_the_empirical_peak_is_flagged_poor_fit():
    mpv = [[6000.0] * 32 for _ in range(4)]
    plateau = np.zeros(len(CHARGE_EDGES) - 1, dtype=np.int64)
    lo, hi = (np.searchsorted(CHARGE_EDGES, v) for v in (2200 + 1500, 2200 + 5500))
    plateau[lo:hi] = 600  # broad flat spectrum, as for an end bar
    plateau[hi : hi + 8] = 1000
    days = [_file_hists(mpv, n=6000)]
    days[0][0][2, 4] = plateau
    bars = tuple(tuple(range(23)) for _ in range(4))
    doc = summarize([days[0][0]], [days[0][1]], np.full((4, 32), 9000), 10_000, bars, SET)
    row = {(r["asic"], r["channel"]): r for r in doc["channels"]}[(2, 4)]
    assert row["status"] in {"poor_fit", "fit_failed"}
    good = {(r["asic"], r["channel"]): r for r in doc["channels"]}[(2, 5)]
    assert good["peak_adc"] == pytest.approx(good["mpv_adc"], rel=0.05)
    assert 0 <= good["above_window_fraction"] < 0.2
