"""Systematic uncertainties, each reported as a shift, never folded into sigma.

flux scale -- a real flux difference between the sky run and the position runs
              shifts every lambda by ln(1 + f).
MCS        -- the model uses straight rays; multiple Coulomb scattering in the
              beams (Highland) blurs the detected direction by theta0. Injected
              as an angular smear of the data, refit with the nominal model.
pose       -- the free position moved by its self-calibration sigma along x and y.
background -- the beam-depth fit removes a smooth background; an unmodelled
              sharp feature would bias the depth, so the polynomial degree is
              raised by one and the shift is the model-choice systematic.
density    -- the beams' opacity is pinned to concrete of the configured
              density; refit at density +- its systematic sigma.
flux model -- the pinned opacity rests on a sea-level muon spectrum; refit
              with the alternative published parametrisation.

A shift is NaN for a key when either measurement of it is NaN (for example a
z-profile face off the grid); it stays NaN through the budget instead of
being counted as zero.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from cafetomo.angular import AnalysisGrid
from cafetomo.beamdepth import beam_design
from cafetomo.config import Config
from cafetomo.fitdata import FitData, RowIndex
from cafetomo.measure import Measurement, measure
from cafetomo.opacity import OpacityMaps, build_fit_data, solve_opacity
from cafetomo.sky import SkyGrid

M_MU_GEV = 0.10566
X0_CONCRETE_M = 0.1155
P_MAX_GEV = 100.0
N_P_SAMPLES = 4000
MCS_SCAN = (0.5, 1.0, 2.0)


def theta0(p_gev: np.ndarray, L_m: float) -> np.ndarray:
    """Highland RMS projected scattering angle (radians) through L_m of concrete."""
    x = L_m / X0_CONCRETE_M
    beta = p_gev / np.sqrt(p_gev**2 + M_MU_GEV**2)
    return 13.6e-3 / (beta * p_gev) * np.sqrt(x) * (1 + 0.038 * np.log(x))


def spectrum_weight(p_gev: np.ndarray) -> np.ndarray:
    """Sea-level muon momentum spectrum shape: flat below a few GeV, ~p^-2.7
    above (mean ~4 GeV). Shape only (~30%), ample for a bound."""
    return 1.0 / (1.0 + (p_gev / 3.5) ** 2.7)


def rms_theta(p_gev: np.ndarray, weight: np.ndarray, L_m: float) -> float:
    """Spectrum-weighted RMS scattering angle, sqrt(<theta0^2>): the width of
    the Gaussian smear that independent scatters add up to."""
    th2 = theta0(p_gev, L_m) ** 2
    return float(np.sqrt(np.trapezoid(th2 * weight, p_gev) / np.trapezoid(weight, p_gev)))


def mcs(cfg: Config, *, h_m: float, lever_m: float) -> dict:
    """Scattering of a muon crossing the fitted beam depth `h_m` (the
    scattering length L, vertical crossing) and the image smear it causes.

    Scattering happens inside the beam and below it is air, so projecting the
    detected exit direction back is exact at the beam's bottom face; what is
    blurred is the lateral displacement accumulated inside the beam,
    theta_rms * h / sqrt(3) (metres). Seen from the detector at height
    `lever_m` (the beam bottom above it) that is a direction smear of
    `jitter_tan` = displacement / lever_m, in tan units.

    `mcs_p_min_gev` is a conservative lower momentum cut: the overburden above
    the beam is unknown, and a higher cut only lowers theta."""
    ps = np.geomspace(cfg.uncertainty.mcs_p_min_gev, P_MAX_GEV, N_P_SAMPLES)
    th = rms_theta(ps, spectrum_weight(ps), h_m)
    disp = th * h_m / np.sqrt(3.0)
    return {"theta": th, "displacement": disp, "jitter_tan": disp / lever_m}


def _delta(variant: Measurement, nominal: Measurement) -> dict[str, float]:
    return {k: float(variant.values[k] - nominal.values[k]) for k in variant.values}


def _worst(acc: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
    """Running max of |delta| per key; NaN once any term is NaN."""
    for k, v in delta.items():
        a = abs(v)
        acc[k] = (
            a if k not in acc else (np.nan if np.isnan(a) or np.isnan(acc[k]) else max(acc[k], a))
        )
    return acc


def flux_scale_shift(
    maps: OpacityMaps,
    cfg: Config,
    sigma: dict,
    rows: RowIndex,
    nominal: Measurement,
    *,
    sky: SkyGrid,
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    """Shift when every opacity is scaled by the configured flux fraction. A
    constant lambda offset can be absorbed by the background polynomial, but
    changes its inferred grammage in the concrete-pinned estimator. Depth can
    therefore shift or switch local solutions."""
    shifted = maps.shifted(float(np.log1p(cfg.uncertainty.flux_scale_frac)))
    m = measure(
        build_fit_data(shifted, cfg, sigma, rows=rows),
        cfg,
        sky,
        cache_dir=cache_dir,
        with_volume=False,
    )
    return _delta(m, nominal)


def mcs_dlam(data: FitData, cfg: Config, depth, jitter_tan: float) -> np.ndarray:
    """Change of lambda when the nominal fitted beams (`depth`) are seen through
    direction-smeared paths; the background cancels in the difference. The
    beams' opacity is taken at the fit's mean pinned density `kappa_mean`:
    linear in the path change, ample for the size of a blur bias."""
    s = cfg.beamdepth
    common = dict(
        aperture_m=cfg.detector.aperture_m,
        n_sub=s.n_sub,
        z0=depth.z0,
        w=depth.w,
        h=depth.h,
        xs=depth.xs,
        y_extent=s.y_extent_m,
    )
    origins = cfg.origins()
    return (
        beam_design(data.rows, origins, angle_jitter=jitter_tan, **common)
        - beam_design(data.rows, origins, **common)
    ).sum(axis=1) * depth.kappa_mean


def mcs_shift(
    data: FitData,
    cfg: Config,
    nominal: Measurement,
    *,
    sky: SkyGrid,
    jitter_tan: float,
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    """Bias multiple scattering causes in the nominal estimator.

    The blur is injected into the DATA (`mcs_dlam`) and the unsmeared
    estimator is refit; refitting with a smeared model instead would measure
    model mismatch, not the bias. The size depends on the momentum cut
    (`mcs_p_min_gev`) and on the scattering length, the fitted beam depth.
    The estimator has a few-centimetre instability floor, so each key quotes
    the largest |shift| over 0.5, 1 and 2 times `jitter_tan`; NaN if any is."""
    worst: dict[str, float] = {}
    for f in MCS_SCAN:
        dlam = mcs_dlam(data, cfg, nominal.details["depth"], f * jitter_tan)
        blurred = FitData(lam=data.lam + dlam * (data.w > 0), w=data.w, rows=data.rows)
        d = _delta(measure(blurred, cfg, sky, cache_dir=cache_dir, with_volume=False), nominal)
        worst = _worst(worst, d)
    return worst


def background_shift(
    data: FitData,
    cfg: Config,
    nominal: Measurement,
    *,
    sky: SkyGrid,
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    """Shift when the beam-depth background polynomial gains one degree."""
    s = replace(cfg.beamdepth, bg_degree=cfg.beamdepth.bg_degree + 1)
    m = measure(data, replace(cfg, beamdepth=s), sky, cache_dir=cache_dir, with_volume=False)
    return _delta(m, nominal)


def pose_shift(
    grid: AnalysisGrid,
    cfg: Config,
    live_time: dict[str, float],
    sigma: dict,
    pose_sigma: dict[str, float],
    nominal: Measurement,
    *,
    sky: SkyGrid,
    rows: RowIndex,
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    """Largest |shift| over the free position moved by +-1 sigma along x and y.
    A key that is NaN in any variant stays NaN. `rows` pins the row set to the
    nominal one, as in the bootstrap and flux paths."""
    free = cfg.selfcal.free_pose
    p = cfg.exposure(free).pose
    worst: dict[str, float] = {}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        c = cfg.with_pose(
            free, replace(p, x=p.x + dx * pose_sigma["x"], y=p.y + dy * pose_sigma["y"])
        )
        data = build_fit_data(solve_opacity(grid, c, live_time), c, sigma, rows=rows)
        worst = _worst(
            worst, _delta(measure(data, c, sky, cache_dir=cache_dir, with_volume=False), nominal)
        )
    return worst


def density_shift(
    data: FitData,
    cfg: Config,
    nominal: Measurement,
    *,
    sky: SkyGrid,
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    """Largest |shift| with the concrete density moved by +-1 sigma."""
    p = cfg.physics
    worst: dict[str, float] = {}
    for sign in (1, -1):
        rho = p.concrete_density_gcm3 + sign * p.concrete_density_sigma
        c = replace(cfg, physics=replace(p, concrete_density_gcm3=rho))
        m = measure(data, c, sky, cache_dir=cache_dir, with_volume=False)
        worst = _worst(worst, _delta(m, nominal))
    return worst


def flux_model_shift(
    data: FitData,
    cfg: Config,
    nominal: Measurement,
    *,
    sky: SkyGrid,
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    """Shift when the pinned opacity uses the alternative muon spectrum."""
    p = cfg.physics
    c = replace(cfg, physics=replace(p, flux_model=p.flux_model_alt))
    return _delta(measure(data, c, sky, cache_dir=cache_dir, with_volume=False), nominal)


def error_budget(stat: dict, shifts: dict[str, dict[str, float]], keys) -> dict:
    """Per key: `<k>_stat` (the bootstrap sigma), `<k>_<source>` for every
    systematic source in `shifts` (|shift|) and their quadrature `<k>_total`.
    The total is NaN when any component is NaN."""
    out = {}
    for k in keys:
        parts = {"stat": stat[f"{k}_sigma"]} | {src: d[k] for src, d in shifts.items()}
        parts = {src: abs(float(v)) for src, v in parts.items()}
        for src, v in parts.items():
            out[f"{k}_{src}"] = v
        out[f"{k}_total"] = float(np.sqrt(sum(v**2 for v in parts.values())))
    return out
