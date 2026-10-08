"""Conditional depth study with a continued beam array and transmitted-flux averaging.

Relative centres and the fitted pose are held fixed; a common transverse shift,
width, depth, bottom and independent column opacities remain free. Count spread
under these assumptions is not an unconditional physical-dimension uncertainty.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from cafetomo.beamdepth import _background, _band
from cafetomo.config import Config
from cafetomo.fitdata import FitData


def _transverse_paths(offsets, sx, sy, origin_z, *, z, w, h, centres):
    """Transverse box intersections averaged in transmission, then logged.

    The beam must span the full admitted footprint along y. Columns describe
    normal-incidence opacity, avoiding a singular density parameter as h falls.
    """
    slope = sx[:, None, None]
    parallel = np.abs(slope) < 1e-9
    safe = np.where(parallel, 1.0, slope)
    left = (centres[None, None, :] - w / 2 - offsets[:, :, None]) / safe
    right = (centres[None, None, :] + w / 2 - offsets[:, :, None]) / safe
    near, far = np.minimum(left, right), np.maximum(left, right)
    inside = np.abs(offsets[:, :, None] - centres[None, None, :]) <= w / 2
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), near)
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), far)
    lower = np.maximum(z - origin_z, 0.0)[:, None, None]
    upper = (z + h - origin_z)[:, None, None]
    path = np.maximum(np.minimum(far, upper) - np.maximum(near, lower), 0.0)
    path *= np.sqrt(1 + sx**2 + sy**2)[:, None, None]
    return path


def mean_flux_opacity(offsets, sx, sy, origin_z, *, z, w, h, centres, columns):
    """Aperture-averaged transmitted flux under full longitudinal coverage."""
    paths = _transverse_paths(offsets, sx, sy, origin_z, z=z, w=w, h=h, centres=centres)
    attenuation = np.einsum("rqb,b->rq", paths, columns / h)
    return -logsumexp(-attenuation, axis=1) + np.log(offsets.shape[1])


def continued_centres(centres):
    """Continue measured centre spacing at both ends, without a survey target."""
    centres = np.asarray(centres)
    pitch = np.median(np.diff(centres))
    return np.r_[
        centres[0] - pitch * np.arange(2, 0, -1),
        centres,
        centres[-1] + pitch * np.arange(1, 3),
    ]


def fit_array_depth(data: FitData, cfg: Config, *, centres, height_seed, n_sub=32):
    """Two common starts; free width starts at the configured rough estimate."""
    band = _band(data, cfg)
    rows = band.rows
    centres = np.asarray(centres)
    origins = np.array([cfg.origins()[rows.position_ids[i]] for i in rows.pos_of_row])
    poses = {e.id: e.pose for e in cfg.exposures}
    az = np.radians([poses[rows.position_ids[i]].az_deg for i in rows.pos_of_row])
    co, si = np.cos(az), np.sin(az)
    tx = rows.sx * co + rows.sy * si
    ty = -rows.sx * si + rows.sy * co
    spans = cfg.detector.aperture_m - np.abs(np.stack([tx, ty], axis=1)) * (
        cfg.detector.layer_dz_cm / 100
    )
    if np.any(spans <= 0):
        raise ValueError("conditional depth rows outside detector acceptance")
    q = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    x = origins[:, 0, None] + spans[:, 0, None] * co[:, None] * q
    offsets = (
        x[:, :, None]
        - spans[:, 1, None, None]
        * si[:, None, None]
        * (np.array([-1 / 3, 0, 1 / 3])[None, None, :])
    )
    offsets = offsets.reshape(rows.n_rows, -1)
    bg = _background(rows, cfg.beamdepth.bg_degree) * band.sw[:, None]
    target = band.lam * band.sw
    zlo = height_seed - cfg.beamdepth.h_max_m / 2 - 1
    zhi = height_seed + 1
    radius_y = (spans[:, 0] * np.abs(si) + spans[:, 1] * np.abs(co)) / 2
    yends = origins[:, 1, None] + rows.sy[:, None] * (
        np.array([zlo, zhi + cfg.beamdepth.h_max_m])[None, :] - origins[:, 2, None]
    )
    if np.any(yends.min(axis=1) - radius_y < cfg.beamdepth.y_extent_m[0]) or np.any(
        yends.max(axis=1) + radius_y > cfg.beamdepth.y_extent_m[1]
    ):
        raise ValueError("conditional transverse model cannot represent finite beam ends")
    lo = np.r_[zlo, 0.05, 0.05, -0.3, np.zeros(len(centres))]
    hi = np.r_[zhi, 1.0, cfg.beamdepth.h_max_m, 0.3, np.full(len(centres), 2.0)]

    cache = {}

    def residual(v):
        z, w, h, shift = v[:4]
        key = tuple(v[:4])
        if cache.get("key") != key:
            cache["paths"] = _transverse_paths(
                offsets,
                rows.sx,
                rows.sy,
                origins[:, 2],
                z=z,
                w=w,
                h=h,
                centres=centres + shift,
            )
            cache["key"] = key
        attenuation = np.einsum("rqb,b->rq", cache["paths"], v[4:] / h)
        lam = -logsumexp(-attenuation, axis=1) + np.log(offsets.shape[1])
        y = target - lam * band.sw
        coeff = np.linalg.lstsq(bg, y, rcond=None)[0]
        return y - bg @ coeff

    fits = []
    for hstart in (0.3, 1.8):
        v = np.r_[
            height_seed - hstart / 2, cfg.beamdepth.w_init_m, hstart, 0, np.full(len(centres), 0.06)
        ]
        fit = least_squares(
            residual, np.clip(v, lo, hi), bounds=(lo, hi), x_scale="jac", max_nfev=600
        )
        if not fit.success:
            raise RuntimeError(f"conditional array fit failed: {fit.message}")
        fits.append(fit)
    best = min(fits, key=lambda f: float(f.fun @ f.fun))
    z, w, h, shift = best.x[:4]
    dof = int(rows.n_rows - len(best.x) - np.linalg.matrix_rank(bg))
    return {
        "h": float(h),
        "bottom": float(z),
        "w": float(w),
        "shift": float(shift),
        "chi2": float(best.fun @ best.fun),
        "dof": dof,
        "chisq_per_dof": float(best.fun @ best.fun) / dof,
        "centres": (centres + shift).tolist(),
        "normal_columns": best.x[4:].tolist(),
        "lower_bounds": lo.tolist(),
        "upper_bounds": hi.tolist(),
        "n_sub": n_sub,
        "nfev": int(best.nfev),
        "starts": [{"h": float(f.x[2]), "chi2": float(f.fun @ f.fun)} for f in fits],
    }


def _with_boundary_flags(record):
    result = dict(record)
    if "lower_bounds" in result:
        values = np.array(
            [result["bottom"], result["w"], result["h"], result["shift"], *result["normal_columns"]]
        )
        lo, hi = np.array(result["lower_bounds"]), np.array(result["upper_bounds"])
        near = np.minimum(values - lo, hi - values) < 0.01 * (hi - lo)
        names = [
            "bottom",
            "width",
            "depth",
            "shift",
            *[f"column_{i}" for i in range(len(values) - 4)],
        ]
        result["bound_parameters"] = [name for name, flag in zip(names, near, strict=True) if flag]
        result["at_bound"] = bool(near.any())
        result["depth_at_bound"] = bool(near[2])
    if "n_sub" in result:
        result["quadrature_major"] = result["n_sub"]
        result["quadrature_minor"] = 3
    return result


def summarize_study(study):
    """Publish nominal conditional parameters and full-cohort count spreads."""
    replicas = [_with_boundary_flags(r) for r in study["replicas"]]
    arrays = {key: np.array([r[key] for r in replicas]) for key in ("h", "w", "bottom")}
    if any(np.any(~np.isfinite(a)) for a in arrays.values()):
        raise ValueError("non-finite conditional replica; no replicas may be dropped")
    nominal = _with_boundary_flags(study["nominal"])
    result = dict(nominal)
    result.update(
        {
            f"{key}_sigma": float(a.std(ddof=1)) if len(a) > 1 else float("nan")
            for key, a in arrays.items()
        }
    )
    result.update(
        {
            f"{key}_bootstrap_mean": float(a.mean()) if len(a) else float("nan")
            for key, a in arrays.items()
        }
    )
    result.update(
        {
            "conditional": True,
            "n_replicas": len(replicas),
            "n_depth_boundary_replicas": sum(r.get("depth_at_bound", False) for r in replicas),
            "uncertainty_scope": "count spread with centres, array and pose fixed",
            "replicas": replicas,
        }
    )
    return result


def main():
    from cafetomo.bootstrap import resample
    from cafetomo.cli import _fit_data, _grid
    from cafetomo.config import load_config
    from cafetomo.opacity import build_fit_data, solve_opacity

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "pose", "opacity", "ingest", "beams", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.pose)
    if args.fast:
        cfg = cfg.fast()
    n_replicas = cfg.uncertainty.n_replicas if args.replicas is None else args.replicas
    _, sigma, data = _fit_data(cfg, Path(args.opacity))
    beams = json.loads(Path(args.beams).read_text())
    centres = continued_centres(beams["beams_x"])
    nominal = fit_array_depth(data, cfg, centres=centres, height_seed=beams["z"])
    grid, live = _grid(cfg, Path(args.ingest))
    rng = np.random.default_rng(np.random.SeedSequence(cfg.uncertainty.seed, spawn_key=(1,)))
    replicas = []
    for i in range(n_replicas):
        sample = build_fit_data(
            solve_opacity(resample(grid, rng), cfg, live), cfg, sigma, rows=data.rows
        )
        replicas.append(fit_array_depth(sample, cfg, centres=centres, height_seed=beams["z"]))
        print(f"conditional array depth: {i + 1}/{n_replicas}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summarize_study({"nominal": nominal, "replicas": replicas}), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
