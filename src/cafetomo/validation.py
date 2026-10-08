"""Ground truth for the two height measurements: phantoms with the REAL rows,
weights and positions of this campaign, at known geometry.

Phantoms match the selected estimator: effective linear beam opacities and
background for geometric fits, or material grammage and muon transmission for
optional concrete fits. Recovery tests numerical behavior within that model,
not the adequacy of the model for the real ceiling."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from cafetomo.autofocus import cv_height_scan
from cafetomo.beamdepth import estimate_beam_depth
from cafetomo.config import Config
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.phantom import beam_ceiling, phantom_data
from cafetomo.sky import SkyGrid
from cafetomo.voxels import VoxelGrid, auto_grid


def _slab_parameters(cfg: Config, nominal: dict) -> tuple[float, float, tuple]:
    if cfg.beamdepth.model == "geometry":
        background = float(nominal["background_mean"])
        kappa = tuple(nominal["kappa"])
        if not np.isfinite(background) or not np.all(np.isfinite(kappa)):
            raise ValueError("geometry validation needs finite fitted opacity and background")
        return 0.3, background / 0.3, kappa
    rho, overburden = nominal["density"], nominal["overburden_mean"]
    if not (np.isfinite(rho) and rho > 0 and np.isfinite(overburden) and overburden >= 0):
        raise ValueError("validation needs a finite fitted density and overburden")
    return overburden / (100.0 * rho), rho, (rho,) * len(nominal["xs"])


def _truth_grid(cfg: Config, data: FitData, nominal_depth: dict) -> VoxelGrid:
    """A fine lattice spanning every ray's footprint, so no injected slab is
    cut off by the grid edge (a step the fits' smooth background cannot follow).
    For the same reason the injected slab spans the grid in y; sensitivity to a
    sharp slab edge is a separate systematic, not part of this validation."""
    thickness, _, _ = _slab_parameters(cfg, nominal_depth)
    heights = (*cfg.validation.focus_heights_m, nominal_depth["zbottom"])
    depths = (*cfg.validation.depth_h_true_m, nominal_depth["h"])
    top = max(heights) + max(depths) + thickness
    vol = replace(
        cfg.volume,
        z_min_m=min(5.0, min(heights)),
        z_max_m=max(10.0, top + 0.05),
        spacing_m=0.05,
        xy_m=None,
    )
    return auto_grid(vol, cfg.origins(), data.rows.t_reach(), aperture_m=cfg.detector.aperture_m)


def _truth(g: VoxelGrid, cfg: Config, nominal_depth: dict, *, z0: float, h: float) -> np.ndarray:
    """Nominal beam footprint and model-matching slab background."""
    thickness, slab_kappa, kappa = _slab_parameters(cfg, nominal_depth)
    if cfg.beamdepth.model == "geometry":
        # Free opacity is not a measured material density. Hold the observed
        # normal column contrast fixed when changing the injected depth.
        kappa = tuple(value * nominal_depth["h"] / h for value in kappa)
    return beam_ceiling(
        g,
        xs=nominal_depth["xs"],
        z0=z0,
        w=nominal_depth["w"],
        h=h,
        kappa=kappa,
        y_extent=cfg.beamdepth.y_extent_m,
        slab_thickness=thickness,
        slab_kappa=slab_kappa,
        slab_y_extent=None,
    )


def validate_autofocus(
    data: FitData, cfg: Config, nominal_depth: dict, *, cache_dir: str | Path | None = None
) -> dict:
    """Inject the fitted beams at each configured height; recover it by the CV scan.

    The injected value is the beam layer's centre, z0 + h/2. `focus_bias` is
    the signed mean error: autofocus lands systematically below the centre,
    and only the sign tells a reader which way to correct."""
    g = _truth_grid(cfg, data, nominal_depth)
    fwd = build_forward_model(data.rows, cfg, grid=g)
    rng = np.random.default_rng(cfg.uncertainty.seed)
    injected, recovered = [], []
    for z0 in cfg.validation.focus_heights_m:
        truth = _truth(g, cfg, nominal_depth, z0=z0, h=nominal_depth["h"])
        physics = cfg.physics if cfg.beamdepth.model == "concrete" else None
        phantom = phantom_data(fwd, truth, data, rng, physics=physics)
        scan = cv_height_scan(phantom, cfg, cache_dir=cache_dir)
        injected.append(z0 + nominal_depth["h"] / 2)
        recovered.append(scan.z_best)
    err = np.array(recovered) - np.array(injected)
    return {
        "focus_injected": injected,
        "focus_recovered": recovered,
        "focus_bias": float(err.mean()),
        "focus_max_error": float(np.abs(err).max()),
    }


def validate_depth(data: FitData, cfg: Config, nominal_depth: dict, *, sky: SkyGrid) -> dict:
    """Inject beams of each configured depth; recover h over noise realisations.

    Geometry phantoms retain nominal normal column opacity as depth changes;
    Concrete phantoms retain their material density. Recovery runs the reported
    estimator (`estimate_beam_depth`): the seeds
    come from triangulating each phantom realisation, never from the truth,
    so the bias quoted is the measurement's own. `sky` is the grid the rows'
    sky indices refer to, needed by the triangulation."""
    g = _truth_grid(cfg, data, nominal_depth)
    fwd = build_forward_model(data.rows, cfg, grid=g)
    rng = np.random.default_rng(cfg.uncertainty.seed + 1)
    out = {"depth_true": [], "depth_mean": [], "depth_spread": []}
    for h in cfg.validation.depth_h_true_m:
        truth = _truth(g, cfg, nominal_depth, z0=nominal_depth["zbottom"], h=h)
        hs = []
        for _ in range(cfg.validation.n_realizations):
            physics = cfg.physics if cfg.beamdepth.model == "concrete" else None
            phantom = phantom_data(fwd, truth, data, rng, physics=physics)
            hs.append(estimate_beam_depth(phantom, cfg, sky)[1].h)
        out["depth_true"].append(h)
        out["depth_mean"].append(float(np.mean(hs)))
        out["depth_spread"].append(float(np.std(hs, ddof=1)) if len(hs) > 1 else float("nan"))
    bias = np.array(out["depth_mean"]) - np.array(out["depth_true"])
    out["depth_max_bias"] = float(np.max(np.abs(bias)))
    return out
