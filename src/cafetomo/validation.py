"""Ground truth for the two height measurements: phantoms with the REAL rows,
weights and positions of this campaign, at known geometry."""

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


def _truth_grid(cfg: Config, data: FitData) -> VoxelGrid:
    """A fine lattice spanning every ray's footprint, so no injected slab is
    cut off by the grid edge (a step the fits' smooth background cannot follow).
    For the same reason the injected slab spans the grid in y; sensitivity to a
    sharp slab edge is a separate systematic, not part of this validation."""
    vol = replace(cfg.volume, z_min_m=5.0, z_max_m=10.0, spacing_m=0.05)
    return auto_grid(vol, cfg.origins(), data.rows.t_reach(), aperture_m=cfg.detector.aperture_m)


def validate_autofocus(
    data: FitData, cfg: Config, nominal_depth: dict, *, cache_dir: str | Path | None = None
) -> dict:
    """Inject the fitted beams at each configured height; recover it by the CV scan.

    The injected value is the beam layer's centre, z0 + h/2. `focus_bias` is
    the signed mean error: autofocus lands systematically below the centre,
    and only the sign tells a reader which way to correct."""
    g = _truth_grid(cfg, data)
    fwd = build_forward_model(data.rows, cfg, grid=g)
    rng = np.random.default_rng(cfg.uncertainty.seed)
    injected, recovered = [], []
    for z0 in cfg.validation.focus_heights_m:
        truth = beam_ceiling(
            g,
            xs=nominal_depth["xs"],
            z0=z0,
            w=nominal_depth["w"],
            h=nominal_depth["h"],
            kappa=nominal_depth["kappa"],
            y_extent=cfg.beamdepth.y_extent_m,
            slab_y_extent=None,
        )
        scan = cv_height_scan(phantom_data(fwd, truth, data, rng), cfg, cache_dir=cache_dir)
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

    Recovery runs the reported estimator (`estimate_beam_depth`): the seeds
    come from triangulating each phantom realisation, never from the truth,
    so the bias quoted is the measurement's own. `sky` is the grid the rows'
    sky indices refer to, needed by the triangulation."""
    g = _truth_grid(cfg, data)
    fwd = build_forward_model(data.rows, cfg, grid=g)
    rng = np.random.default_rng(cfg.uncertainty.seed + 1)
    out = {"depth_true": [], "depth_mean": [], "depth_spread": []}
    for h in cfg.validation.depth_h_true_m:
        truth = beam_ceiling(
            g,
            xs=nominal_depth["xs"],
            z0=nominal_depth["zbottom"],
            w=nominal_depth["w"],
            h=h,
            kappa=nominal_depth["kappa"],
            y_extent=cfg.beamdepth.y_extent_m,
            slab_y_extent=None,
        )
        hs = [
            estimate_beam_depth(phantom_data(fwd, truth, data, rng), cfg, sky)[1].h
            for _ in range(cfg.validation.n_realizations)
        ]
        out["depth_true"].append(h)
        out["depth_mean"].append(float(np.mean(hs)))
        out["depth_spread"].append(float(np.std(hs, ddof=1)) if len(hs) > 1 else float("nan"))
    bias = np.array(out["depth_mean"]) - np.array(out["depth_true"])
    out["depth_max_bias"] = float(np.max(np.abs(bias)))
    return out
