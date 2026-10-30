"""Where the reconstruction puts opacity a ceiling does not have: coverage damping and the
two-view support, on a ceiling-only phantom through the real rows and on the data.

run.py --config C --baseline RUN/baseline --results RESULTS --out OUT.json

Settings compared: the previous production solve (coverage damping 0.05, every voxel), the
undamped solve on every voxel, and the undamped solve on the voxels both positions see (the
production config). Reported per setting:
- the phantom's reconstructed opacity below the air limit (z < phantom bottom - AIR_MARGIN_M),
  noise-free and with Gaussian noise of the per-row sigma, in percent of the total;
- the data's opacity below the same limit, its chi2 per row, and the cross-position
  prediction chi2 (fit one position's rows, score the other's).
The phantom is the measured ceiling: the triangulated beam centres and height and the
continued-array width and depth under a uniform slab; its opacity scale matches the data's
median lambda near the zenith.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from cafetomo.cli import _fit_data
from cafetomo.config import load_config
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.inversion import solve
from cafetomo.phantom import beam_ceiling
from cafetomo.reconstruct import VoxelSolution

AIR_MARGIN_M = 1.0  # opacity this far below the phantom's bottom face counts as air
ZENITH_TAN = 0.25  # rows within this tangent set the phantom's opacity scale
SLAB_M = 0.2  # phantom slab thickness above the beams
SEED = 1
SETTINGS = {
    "damped_all": {"coverage_damping": 0.05, "min_views": 1},
    "undamped_all": {"coverage_damping": 0.0, "min_views": 1},
    "undamped_twoview": {"coverage_damping": 0.0, "min_views": 2},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    cfg = load_config(a.config, a.results / "pose.json")
    _, _, data = _fit_data(cfg, a.baseline / "opacity")
    grid = VoxelSolution.load(a.baseline / "voxels" / "volume_full.npz").grid
    fwd = build_forward_model(data.rows, cfg, grid=grid, cache_dir=a.baseline / ".cache")
    beams = json.loads((a.results / "beams.json").read_text())
    arr = json.loads((a.results / "arraydepth.json").read_text())
    live = data.w > 0

    unit = beam_ceiling(
        grid,
        xs=beams["beams_x"],
        z0=beams["z"],
        w=arr["w"],
        h=arr["h"],
        kappa=[1.0] * len(beams["beams_x"]),
        y_extent=(float(grid.axis_centers(1)[0]), float(grid.axis_centers(1)[-1])),
        slab_thickness=SLAB_M,
        slab_kappa=1.0,
        slab_y_extent=None,
    ).ravel()
    near = live & (np.hypot(data.rows.sx, data.rows.sy) < ZENITH_TAN)
    clean = fwd.predict(unit)
    scale = float(np.median(data.lam[near]) / np.median(clean[near]))
    clean *= scale
    sigma = np.where(live, 1.0 / np.sqrt(np.where(live, data.w, 1.0)), 0.0)
    noisy = clean + np.random.default_rng(SEED).normal(size=clean.size) * sigma

    z = np.broadcast_to(grid.axis_centers(2)[None, None, :], grid.shape).ravel()
    air = z < beams["z"] - AIR_MARGIN_M

    def air_percent(x):
        return float(100.0 * x[air].sum() / x.sum())

    def with_lam(lam):
        return FitData(lam=np.where(live, lam, 0.0), w=data.w, rows=data.rows)

    doc = {
        "air_margin_m": AIR_MARGIN_M,
        "air_below_m": beams["z"] - AIR_MARGIN_M,
        "phantom_scale": scale,
        "damped_mu": SETTINGS["damped_all"]["coverage_damping"],
    }
    for name, change in SETTINGS.items():
        rc = replace(cfg.reconstruction, **change)
        for label, d in (("clean", with_lam(clean)), ("noisy", with_lam(noisy))):
            x, _ = solve(fwd, d, rc, fit_offsets=False)
            doc[f"{name}_phantom_{label}_air_percent"] = air_percent(x)
        x, info = solve(fwd, data, rc, fit_offsets=False)
        doc[f"{name}_data_air_percent"] = air_percent(x)
        doc[f"{name}_data_chisq"] = info["best_chi2"]
        for train in data.rows.position_ids:
            xt, _ = solve(fwd, data.restricted(data.rows.mask_for(train)), rc, fit_offsets=False)
            r = data.lam - fwd.predict(xt)
            other = live & ~data.rows.mask_for(train)
            doc[f"{name}_predict_from_{train}_chisq"] = float(
                np.sum((data.w * r**2)[other]) / other.sum()
            )
        print(name, {k: round(v, 4) for k, v in doc.items() if k.startswith(name)}, flush=True)
    a.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
