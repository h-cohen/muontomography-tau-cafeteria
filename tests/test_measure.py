import numpy as np
import pytest

from cafetomo.measure import measure
from phantoms import XS, beam_phantom, phantom_sky

H_TRUE = 1.25
BASE_KEYS = {
    "autofocus_z",
    "beams_z",
    "beams_zx",
    "beams_pitch",
    "depth_h",
    "depth_w",
    "depth_zbottom",
    "depth_ztop",
}
VOLUME_KEYS = {"zprofile_fwhm", "zprofile_bottom", "zprofile_top"}


@pytest.fixture(scope="module")
def phantom(cfg):
    return beam_phantom(cfg, H_TRUE, np.random.default_rng(5))


@pytest.mark.slow
def test_measure_without_volume_recovers_the_phantom(phantom):
    c, data = phantom
    m = measure(data, c.fast(), phantom_sky(), with_volume=False)
    assert set(m.values) == BASE_KEYS and m.volume is None
    assert all(np.isfinite(v) for v in m.values.values())
    assert not any(ch.isdigit() for k in m.values for ch in k)
    assert m.values["depth_h"] == pytest.approx(H_TRUE, abs=0.15)
    assert m.values["depth_ztop"] == pytest.approx(m.values["depth_zbottom"] + m.values["depth_h"])
    assert m.values["beams_pitch"] == pytest.approx(XS[1] - XS[0], abs=0.15)
    depth = m.details["depth"]
    assert depth.density == c.physics.concrete_density_gcm3
    assert depth.kappa_mean > 0 and depth.overburden_mean > 0


@pytest.mark.slow
def test_measure_with_volume_adds_the_z_profile(phantom):
    c, data = phantom
    m = measure(data, c.fast(), phantom_sky(), with_volume=True)
    assert set(m.values) == BASE_KEYS | VOLUME_KEYS
    sol = m.details["solution"]
    assert m.volume.shape == sol.grid.shape
    assert np.isfinite(m.values["zprofile_bottom"])
