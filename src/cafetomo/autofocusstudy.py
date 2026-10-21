"""Autofocus rerun after the layer-thickness correction.

Autofocus feeds no other result, so only its own pieces are rerun, with the
published estimator code: the nominal scan, the count replicas of the original
bootstrap (same random stream, so the spreads compare), the systematic shifts
of the error budget, and the phantom check. The output is a separate results
file; the archived autofocus results stay as they were reviewed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cafetomo.autofocus import FocusScan
from cafetomo.bootstrap import resample
from cafetomo.measure import measure
from cafetomo.opacity import build_fit_data, solve_opacity
from cafetomo.systematics import (
    background_shift,
    flux_scale_shift,
    mcs,
    mcs_shift,
    pose_shift,
)
from cafetomo.validation import validate_autofocus


def summarize(
    *,
    nominal: FocusScan,
    replicas: list[float],
    shifts: dict[str, float],
    validation: dict,
    layer_thickness: float,
) -> dict:
    """autofocuslayer.json: nominal height and scan, replica spread, shifts and budget."""
    z = np.asarray(replicas, dtype=float)
    finite = z[np.isfinite(z)]
    stat = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
    doc = {
        "z": float(nominal.z_best),
        "scan_z": nominal.zs.tolist(),
        "scan_score": nominal.scores.tolist(),
        "layer_thickness": float(layer_thickness),
        "z_mean": float(finite.mean()) if finite.size else float("nan"),
        "z_n": int(finite.size),
        "z_stat": stat,
        **{f"z_{k}": abs(float(v)) for k, v in shifts.items()},
    }
    doc["z_total"] = float(np.sqrt(stat**2 + sum(float(v) ** 2 for v in shifts.values())))
    doc["focus_bias"] = float(validation["focus_bias"])
    doc["focus_max_error"] = float(validation["focus_max_error"])
    return doc


def run_study(grid, live, cfg, maps, sigma, data, *, pose_sigma, nominal_depth, cache_dir=None):
    sky = maps.sky
    nominal = measure(data, cfg, sky, cache_dir=cache_dir, with_volume=False)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.uncertainty.seed, spawn_key=(1,)))
    replicas = []
    for r in range(cfg.uncertainty.n_replicas):
        sample = build_fit_data(
            solve_opacity(resample(grid, rng), cfg, live), cfg, sigma, rows=data.rows
        )
        replicas.append(
            measure(sample, cfg, sky, cache_dir=cache_dir, autofocus_only=True).values[
                "autofocus_z"
            ]
        )
        print(
            f"autofocus study: replica {r + 1}/{cfg.uncertainty.n_replicas}",
            file=sys.stderr,
            flush=True,
        )
    scat = mcs(cfg, h_m=nominal.values["depth_h"], lever_m=nominal.values["depth_zbottom"])
    kw = {"sky": sky, "cache_dir": cache_dir, "autofocus_only": True}
    shifts = {
        "flux": flux_scale_shift(maps, cfg, sigma, data.rows, nominal, **kw),
        "mcs": mcs_shift(data, cfg, nominal, jitter_tan=scat["jitter_tan"], **kw),
        "pose": pose_shift(grid, cfg, live, sigma, pose_sigma, nominal, rows=data.rows, **kw),
        "bg": background_shift(data, cfg, nominal, **kw),
    }
    return summarize(
        nominal=nominal.details["autofocus"],
        replicas=replicas,
        shifts={k: v["autofocus_z"] for k, v in shifts.items()},
        validation=validate_autofocus(data, cfg, nominal_depth, cache_dir=cache_dir),
        layer_thickness=cfg.autofocus.layer_thickness_m,
    )


def write(doc: dict, out: Path) -> Path:
    import json

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    return out
