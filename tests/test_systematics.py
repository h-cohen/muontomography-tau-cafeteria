import numpy as np
import pytest

from cafetomo.measure import measure
from cafetomo.systematics import background_shift, error_budget, mcs, mcs_shift, theta0
from phantoms import beam_phantom, phantom_sky


def test_highland_matches_hand_value():
    # 13.6 MeV/(beta c p) * sqrt(x) * (1 + 0.038 ln x) at x = 1, p = 1 GeV/c:
    # beta = 1/sqrt(1 + (0.10566)^2) = 0.99445, so theta0 = 13.68 mrad.
    assert theta0(np.array([1.0]), 0.1155)[0] == pytest.approx(13.68e-3, rel=0.005)


def test_mcs_blur_scales_with_lever(cfg):
    a, b = mcs(cfg, h_m=1.25, lever_m=7.0), mcs(cfg, h_m=1.25, lever_m=3.5)
    assert a["blur"] == pytest.approx(2 * b["blur"], rel=1e-9)
    assert a["jitter_tan"] == pytest.approx(a["theta"], rel=1e-9)


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
    jit = mcs_shift(data, c, nominal, sky=sky,
                    jitter_tan=mcs(c, h_m=1.25, lever_m=7.0)["jitter_tan"])
    assert set(bg) == set(nominal.values) == set(jit)
    assert all(np.isfinite(v) for v in bg.values())
    assert abs(bg["depth_h"]) < 0.3
    # The phantom has no scattering, so the smear only has to move the depth and
    # leave the geometry-only keys untouched.
    assert jit["depth_h"] != 0.0 and jit["autofocus_z"] == 0.0
