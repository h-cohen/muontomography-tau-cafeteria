import numpy as np
import pytest

from cafetomo import muonphysics as mp

THR = 0.03
MODEL = "guan"


def test_range_table_is_monotone():
    assert np.all(np.diff(mp._TABLE_T_MEV) > 0)
    assert np.all(np.diff(mp._TABLE_RANGE_GCM2) > 0)


def test_range_hits_the_table_nodes():
    # PDG muE_shielding_concrete: T = 1 GeV -> CSDA range 546.0 g/cm^2.
    assert mp.csda_range(1.0) == pytest.approx(546.0, rel=1e-6)
    with pytest.raises(ValueError, match="range table"):
        mp.csda_range(0.005)


def test_e_min_starts_at_threshold_and_rises():
    assert mp.e_min(0.0, THR) == pytest.approx(THR, rel=1e-9)
    x = np.array([0.0, 10.0, 100.0, 500.0, 2000.0])
    assert np.all(np.diff(mp.e_min(x, THR)) > 0)
    # ~2 MeV per g/cm^2 near minimum ionisation: 500 g/cm^2 needs ~1 GeV.
    assert mp.e_min(500.0, THR) == pytest.approx(0.95, abs=0.1)


@pytest.mark.parametrize("model", ["guan", "shukla"])
def test_vertical_flux_above_one_gev(model):
    """PDG "Cosmic Rays" (2022) section 30.3.1: vertical I(>1 GeV/c) at sea
    level is ~70 per m^2 s sr, with recent measurements 10-15% lower. A 20%
    tolerance covers both."""
    t1 = np.hypot(1.0, mp.M_MU_GEV) - mp.M_MU_GEV
    assert mp.integral_flux(t1, 1.0, model) == pytest.approx(70.0, rel=0.2)


def test_gaisser_fails_at_low_energy_and_guan_reduces_to_it_at_high():
    t1 = np.hypot(1.0, mp.M_MU_GEV) - mp.M_MU_GEV
    assert mp.integral_flux(t1, 1.0, "gaisser") > 5 * 70.0
    e = np.array([500.0, 2000.0])
    g = mp.differential_flux(e, 1.0, "gaisser")
    assert mp.differential_flux(e, 1.0, "guan") == pytest.approx(g, rel=0.03)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown muon flux model"):
        mp.differential_flux(1.0, 1.0, "nope")


def test_transmission_falls_from_one():
    x = np.array([0.0, 50.0, 200.0, 1000.0])
    t = mp.transmission(x, 1.0, threshold_gev=THR, model=MODEL)
    assert t[0] == pytest.approx(1.0)
    assert np.all(np.diff(t) < 0) and t[-1] > 0


@pytest.mark.parametrize("model", ["guan", "shukla"])
def test_overburden_round_trip(model):
    x = np.array([0.0, 5.0, 50.0, 150.0, 300.0, 1500.0])
    c = np.array([1.0, 0.9, 0.8, 0.7, 1.0, 0.6])
    lam = -np.log(mp.transmission(x, c, threshold_gev=THR, model=model))
    back = mp.overburden_from_lambda(lam, c, threshold_gev=THR, model=model)
    assert back == pytest.approx(x, abs=0.05)


def test_overburden_nan_and_non_positive():
    out = mp.overburden_from_lambda(
        np.array([np.nan, 0.0, -0.1, 0.1]), 1.0, threshold_gev=THR, model=MODEL
    )
    assert np.isnan(out[0]) and out[1] == 0.0 and out[2] == 0.0 and out[3] > 0


def test_kappa_is_density_times_slope():
    x = np.array([50.0, 150.0])
    k = mp.kappa_concrete(x, 1.0, 2.4, threshold_gev=THR, model=MODEL)
    assert mp.kappa_concrete(x, 1.0, 1.2, threshold_gev=THR, model=MODEL) == pytest.approx(k / 2)
    tr = mp.Transmission(np.ones(2), threshold_gev=THR, model=MODEL)
    slope = (tr.lam(x + 1.0) - tr.lam(x - 1.0)) / 2.0
    assert k == pytest.approx(240.0 * slope, rel=1e-3)


def _linearisation_error(model: str) -> float:
    worst = 0.0
    for c in (1.0, 0.7):
        tr = mp.Transmission(np.array([c]), threshold_gev=THR, model=model)
        for x in (50.0, 150.0, 300.0):
            k = tr.kappa(x, 2.4)[0]
            for length in np.linspace(0.05, 1.5, 30):
                exact = tr.lam(x + 240.0 * length)[0] - tr.lam(x)[0]
                worst = max(worst, abs(k * length - exact) / exact)
    return worst


def test_linearisation_error_exceeds_five_percent():
    """kappa(X) * L against the exact -ln[T(X + 100 rho L)/T(X)] for X in
    {50, 150, 300} g/cm^2, L <= 1.5 m, rho 2.4: the linear term is off by up
    to ~11%, beyond the 5% budget, which is why the beam-depth fit uses the
    exact (non-linear) beam term."""
    err = _linearisation_error(MODEL)
    assert 0.05 < err < 0.15
