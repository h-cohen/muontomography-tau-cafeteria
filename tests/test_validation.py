import numpy as np
import pytest

from cafetomo.validation import validate_autofocus, validate_depth
from phantoms import KAPPA, XS, Z0, W, beam_phantom

H_TRUE = 1.25
NOMINAL = {"xs": list(XS), "w": W, "h": H_TRUE, "kappa": list(KAPPA), "zbottom": Z0}


@pytest.fixture(scope="module")
def phantom(cfg):
    c, data = beam_phantom(cfg, H_TRUE, np.random.default_rng(6))
    return c.fast(), data


@pytest.mark.slow
def test_validate_depth_shapes(phantom):
    c, data = phantom
    out = validate_depth(data, c, NOMINAL)
    n = len(c.validation.depth_h_true_m)
    assert n == 1
    assert all(len(out[k]) == n for k in ("depth_true", "depth_mean", "depth_spread"))
    assert np.isfinite(out["depth_max_bias"])
    assert out["depth_max_bias"] == pytest.approx(
        abs(out["depth_mean"][0] - out["depth_true"][0]))
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
