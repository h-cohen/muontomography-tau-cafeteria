import numpy as np
import pytest

from cafetomo import validation
from cafetomo.beamdepth import estimate_beam_depth
from cafetomo.validation import validate_autofocus, validate_depth
from phantoms import KAPPA, XS, Z0, W, beam_phantom, phantom_sky

H_TRUE = 1.25
NOMINAL = {"xs": list(XS), "w": W, "h": H_TRUE, "kappa": list(KAPPA), "zbottom": Z0}


@pytest.fixture(scope="module")
def phantom(cfg):
    c, data = beam_phantom(cfg, H_TRUE, np.random.default_rng(6))
    return c.fast(), data


@pytest.mark.slow
def test_validate_depth_runs_the_reported_estimator(phantom, monkeypatch):
    """Recovery goes through estimate_beam_depth, seeded by triangulating each
    realisation: validation has no route to the fit that could take truth seeds."""
    c, data = phantom
    calls = []

    def spy(d, cf, sky, **kw):
        calls.append(d)
        return estimate_beam_depth(d, cf, sky, **kw)

    monkeypatch.setattr(validation, "estimate_beam_depth", spy)
    assert not hasattr(validation, "fit_beam_depth")
    out = validate_depth(data, c, NOMINAL, sky=phantom_sky())
    n = len(c.validation.depth_h_true_m)
    assert n == 1 and len(calls) == n * c.validation.n_realizations
    assert all(len(out[k]) == n for k in ("depth_true", "depth_mean", "depth_spread"))
    assert np.isfinite(out["depth_max_bias"])
    assert out["depth_max_bias"] == pytest.approx(abs(out["depth_mean"][0] - out["depth_true"][0]))
    assert not any(ch.isdigit() for k in out for ch in k)


@pytest.mark.slow
def test_validate_autofocus_reports_signed_bias(phantom):
    c, data = phantom
    out = validate_autofocus(data, c, NOMINAL)
    err = np.array(out["focus_recovered"]) - np.array(out["focus_injected"])
    assert out["focus_injected"] == [z + H_TRUE / 2 for z in c.validation.focus_heights_m]
    assert out["focus_bias"] == pytest.approx(float(err.mean()))
    assert out["focus_max_error"] == pytest.approx(float(np.abs(err).max()))
    assert np.isfinite(out["focus_bias"])
    assert not any(ch.isdigit() for k in out for ch in k)
