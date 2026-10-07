from types import SimpleNamespace

import numpy as np
import pytest

from cafetomo.config import Pose
from cafetomo.fitdata import FitData
from cafetomo.measure import measure
from cafetomo.phantom import sky_rows
from cafetomo.systematics import (
    background_shift,
    error_budget,
    mcs,
    mcs_dlam,
    mcs_shift,
    rms_theta,
    theta0,
)
from phantoms import KAPPA, N_BINS, T_MAX, XS, Z0, W, beam_phantom, phantom_sky


def test_highland_matches_hand_value():
    # 13.6 MeV/(beta c p) * sqrt(x) * (1 + 0.038 ln x) at x = 1, p = 1 GeV/c:
    # beta = 1/sqrt(1 + (0.10566)^2) = 0.99445, so theta0 = 13.68 mrad.
    assert theta0(np.array([1.0]), 0.1155)[0] == pytest.approx(13.68e-3, rel=0.005)


def test_rms_theta_by_hand_on_two_points():
    p, w = np.array([1.0, 2.0]), np.array([1.0, 1.0])
    expect = np.sqrt((theta0(p[:1], 0.5)[0] ** 2 + theta0(p[1:], 0.5)[0] ** 2) / 2)
    assert rms_theta(p, w, 0.5) == pytest.approx(expect, rel=1e-12)


def test_mcs_jitter_is_in_beam_displacement_over_lever(cfg):
    a = mcs(cfg, h_m=1.25, lever_m=7.0)
    assert a["displacement"] == pytest.approx(a["theta"] * 1.25 / np.sqrt(3), rel=1e-12)
    assert a["jitter_tan"] == pytest.approx(a["theta"] * 1.25 / (np.sqrt(3) * 7.0), rel=1e-12)
    assert mcs(cfg, h_m=1.25, lever_m=3.5)["jitter_tan"] == pytest.approx(2 * a["jitter_tan"])


def test_zero_jitter_injects_zero_dlam(cfg):
    c = cfg.with_pose("pos1", Pose(1.78, 0.72, 0.0, 0.0))
    rows = sky_rows(c.position_ids, T_MAX, N_BINS)
    data = FitData(lam=np.zeros(rows.n_rows), w=np.ones(rows.n_rows), rows=rows)
    depth = SimpleNamespace(z0=Z0, w=W, h=1.25, xs=XS, kappa=KAPPA)
    assert not np.any(mcs_dlam(data, c, depth, 0.0))
    assert np.any(mcs_dlam(data, c, depth, 0.02))


def test_budget_adds_in_quadrature_including_background():
    b = error_budget({"depth_h_sigma": 0.3}, {"depth_h": 0.0}, {"depth_h": 0.4},
                     {"depth_h": 0.0}, {"depth_h": 0.0}, ["depth_h"])
    assert b["depth_h_total"] == pytest.approx(0.5)
    b = error_budget({"depth_h_sigma": 0.3}, {"depth_h": 0.0}, {"depth_h": 0.0},
                     {"depth_h": 0.0}, {"depth_h": -0.4}, ["depth_h"])
    assert b["depth_h_bg"] == 0.4 and b["depth_h_total"] == pytest.approx(0.5)


def test_budget_keeps_nan_instead_of_zero():
    ok = {"a": 0.1, "b": 0.1}
    b = error_budget({"a_sigma": 0.1, "b_sigma": 0.1}, ok, ok, ok, {"a": np.nan, "b": 0.1},
                     ["a", "b"])
    assert np.isnan(b["a_bg"]) and np.isnan(b["a_total"])
    assert np.isfinite(b["b_total"])


@pytest.mark.slow
def test_shifts_on_the_phantom(cfg):
    c, data = beam_phantom(cfg, 1.25, np.random.default_rng(5))
    c, sky = c.fast(), phantom_sky()
    nominal = measure(data, c, sky, with_volume=False)
    bg = background_shift(data, c, nominal, sky=sky)
    jt = mcs(c, h_m=1.25, lever_m=7.0)["jitter_tan"]
    jit = mcs_shift(data, c, nominal, sky=sky, jitter_tan=jt)
    assert set(bg) == set(nominal.values) == set(jit)
    assert all(np.isfinite(v) for v in bg.values())
    assert abs(bg["depth_h"]) < 0.3
    assert np.isfinite(jit["depth_h"]) and abs(jit["depth_h"]) < 0.1
