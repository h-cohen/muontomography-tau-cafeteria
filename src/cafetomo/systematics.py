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


def theta0(p_gev: np.ndarray, L_m: float) -> np.ndarray:
    """Highland RMS projected scattering angle (radians) through L_m of concrete."""
    x = L_m / X0_CONCRETE_M
    beta = p_gev / np.sqrt(p_gev**2 + M_MU_GEV**2)
    return 13.6e-3 / (beta * p_gev) * np.sqrt(x) * (1 + 0.038 * np.log(x))


def spectrum_weight(p_gev: np.ndarray) -> np.ndarray:
    """Sea-level muon momentum spectrum shape: flat below a few GeV, ~p^-2.7
    above (mean ~4 GeV). Shape only (~30%), ample for a bound."""
    return 1.0 / (1.0 + (p_gev / 3.5) ** 2.7)


def mcs(cfg: Config, *, h_m: float, lever_m: float) -> dict:
    """Flux-weighted theta0 for a vertical crossing of the fitted beam depth
    `h_m` (the scattering length L), and the lateral blur it causes at the
    ceiling `lever_m` away. `jitter_tan` is the tan-unit smear for the model."""
    ps = np.geomspace(cfg.uncertainty.mcs_p_min_gev, P_MAX_GEV, N_P_SAMPLES)
    w = spectrum_weight(ps)
    th = float(np.trapezoid(theta0(ps, h_m) * w, ps) / np.trapezoid(w, ps))
    return {"theta": th, "blur": th * lever_m, "jitter_tan": th}


def _delta(variant: Measurement, nominal: Measurement) -> dict[str, float]:
    return {k: float(variant.values[k] - nominal.values[k]) for k in variant.values}


def flux_scale_shift(maps: OpacityMaps, cfg: Config, sigma: dict, rows: RowIndex,
                     nominal: Measurement, *, sky: SkyGrid,
                     cache_dir: str | Path | None = None) -> dict[str, float]:
    """Shift when every opacity is scaled by the configured flux fraction."""
    shifted = maps.shifted(float(np.log1p(cfg.uncertainty.flux_scale_frac)))
    m = measure(build_fit_data(shifted, cfg, sigma, rows=rows), cfg, sky,
                cache_dir=cache_dir, with_volume=False)
    return _delta(m, nominal)


def mcs_shift(data: FitData, cfg: Config, nominal: Measurement, *, sky: SkyGrid,
              jitter_tan: float, cache_dir: str | Path | None = None) -> dict[str, float]:
    """Bias multiple scattering causes in the nominal estimator.

    The blur is injected into the DATA: the nominal fitted beams are seen
    through direction-smeared paths, the difference from the straight-path
    signal is added to lambda (the background cancels in it), and the unsmeared
    estimator is refit. Refitting with a smeared model instead would measure
    model mismatch, not the bias. The size depends on the momentum cut
    (`mcs_p_min_gev`) and on the scattering length, taken as the fitted beam
    depth."""
    d = nominal.details["depth"]
    s = cfg.beamdepth
    common = dict(aperture_m=cfg.detector.aperture_m, n_sub=s.n_sub, z0=d.z0, w=d.w, h=d.h,
                  xs=d.xs, y_extent=s.y_extent_m)
    origins = cfg.origins()
    kappa = np.asarray(d.kappa)
    dlam = (beam_design(data.rows, origins, angle_jitter=jitter_tan, **common)
            - beam_design(data.rows, origins, **common)) @ kappa
    blurred = FitData(lam=data.lam + dlam * (data.w > 0), w=data.w, rows=data.rows)
    m = measure(blurred, cfg, sky, cache_dir=cache_dir, with_volume=False)
    return _delta(m, nominal)


def background_shift(data: FitData, cfg: Config, nominal: Measurement, *, sky: SkyGrid,
                     cache_dir: str | Path | None = None) -> dict[str, float]:
    """Shift when the beam-depth background polynomial gains one degree."""
    s = replace(cfg.beamdepth, bg_degree=cfg.beamdepth.bg_degree + 1)
    m = measure(data, replace(cfg, beamdepth=s), sky, cache_dir=cache_dir, with_volume=False)
    return _delta(m, nominal)


def pose_shift(grid: AnalysisGrid, cfg: Config, live_time: dict[str, float], sigma: dict,
               pose_sigma: dict[str, float], nominal: Measurement, *, sky: SkyGrid,
               cache_dir: str | Path | None = None) -> dict[str, float]:
    """Largest |shift| over the free position moved by +-1 sigma along x and y.
    A key that is NaN in any variant stays NaN."""
    free = cfg.selfcal.free_pose
    p = cfg.exposure(free).pose
    worst: dict[str, float] = {}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        c = cfg.with_pose(free, replace(p, x=p.x + dx * pose_sigma["x"],
                                        y=p.y + dy * pose_sigma["y"]))
        data = build_fit_data(solve_opacity(grid, c, live_time), c, sigma)
        d = _delta(measure(data, c, sky, cache_dir=cache_dir, with_volume=False), nominal)
        for k, v in d.items():
            a = abs(v)
            worst[k] = a if k not in worst else (np.nan if np.isnan(a) or np.isnan(worst[k])
                                                 else max(worst[k], a))
    return worst


def error_budget(stat: dict, flux: dict, mcs_d: dict, pose: dict, bg: dict, keys) -> dict:
    """Per key, each source's magnitude and their quadrature total. The total
    is NaN when any component is NaN."""
    out = {}
    for k in keys:
        parts = {"stat": stat[f"{k}_sigma"], "flux": flux[k], "mcs": mcs_d[k],
                 "pose": pose[k], "bg": bg[k]}
        parts = {src: abs(float(v)) for src, v in parts.items()}
        for src, v in parts.items():
            out[f"{k}_{src}"] = v
        out[f"{k}_total"] = float(np.sqrt(sum(v**2 for v in parts.values())))
    return out
