"""The viewer data contract.

volume.npy + meta.json keeps the viewer decoupled from the solver: the viewer
reads arrays and metadata, never the solver's internals, so either side can be
rebuilt without touching the other.

meta.json deliberately carries the resolution verdict. A viewer that can render
a crisp volume without being able to say that its height is poorly measured
would be a misleading instrument. For the same reason the fitted beam boxes
travel with their total uncertainty when the error budget has been run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cafetomo.config import Config
from cafetomo.reconstruct import VoxelSolution
from cafetomo.resolution import campaign_resolution


def _json_safe(obj):
    """Recursively strip numpy scalar/array types so json.dumps doesn't choke.

    `info` on a VoxelSolution is whatever the solver's internals happened to
    put there (chi2, iteration counts, ...), which is often numpy dtypes even
    though this module only ever produces plain Python values itself.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _beams_meta(results_dir: Path, cfg: Config) -> dict | None:
    """The fitted beam boxes, in the voxel grid's world frame (metres).

    The fit shares one depth h, one width w and one bottom face across beams,
    and treats every beam as running the full configured y extent; the boxes
    are drawn exactly as fitted. `h_sigma` is the TOTAL uncertainty from the
    error budget, null until that stage has run, so the viewer never shows a
    statistical-only error as if it were the full one.
    """
    fit_path = results_dir / "beamdepth.json"
    if not fit_path.is_file():
        return None
    fit = json.loads(fit_path.read_text())
    unc_path = results_dir / "uncertainty.json"
    h_sigma = (json.loads(unc_path.read_text())["depth_h_total"]
               if unc_path.is_file() else None)
    y0, y1 = (float(v) for v in cfg.beamdepth.y_extent_m)
    z0, h, w = float(fit["zbottom"]), float(fit["h"]), float(fit["w"])
    return {
        "boxes": [{"x": float(xk), "w": w, "zbottom": z0, "ztop": z0 + h,
                   "y_extent": [y0, y1]} for xk in fit["xs"]],
        "h": h,
        "h_sigma": None if h_sigma is None else float(h_sigma),
    }


def export_volume(voxels_dir: str | Path, cfg: Config, *,
                  bootstrap_dir: str | Path | None = None,
                  results_dir: str | Path | None = None,
                  out_dir: str | Path) -> Path:
    """Write volume.npy + meta.json (+ any optional layers found) for the viewer.

    Optional inputs are optional because a run may stop before the bootstrap
    or the beam-depth fit; whatever is absent is simply not offered, never
    filled with a placeholder. Returns the path of volume.npy.
    """
    voxels = Path(voxels_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vol = VoxelSolution.load(voxels / "volume_full.npz")
    rho = vol.rho3().astype(np.float32)
    np.save(out / "volume.npy", rho)
    layers = ["volume"]

    if bootstrap_dir is not None:
        with np.load(Path(bootstrap_dir) / "volume_stats.npz") as d:
            for name in ("sigma", "snr"):
                arr = d[name]
                if arr.shape != rho.shape:
                    raise ValueError(f"bootstrap {name} has shape {arr.shape}, "
                                     f"volume has {rho.shape}")
                np.save(out / f"{name}.npy", arr.astype(np.float32))
                layers.append(name)

    for name in ("views", "rays"):
        src = voxels / f"{name}.npy"
        if src.exists():
            np.save(out / f"{name}.npy", np.load(src))
            layers.append(name)

    pos = np.maximum(rho, 0.0)
    p99 = float(np.percentile(pos, 99)) if pos.max() > 0 else 1.0
    # The rms angular error of one sky-grid bin (width / sqrt 12), as the
    # beam-depth cross-check states its resolution, so the two agree.
    sigma_t = cfg.opacity.sky_t_max * 2 / cfg.opacity.sky_n_bins / np.sqrt(12.0)
    res = campaign_resolution(cfg, sigma_t=float(sigma_t),
                              feature_pitch_m=max(2.0, 4 * cfg.volume.spacing_m))

    meta = {
        "shape": list(rho.shape),
        "axis_order": "xyz",
        "origin_m": list(vol.grid.origin),
        "spacing_m": vol.grid.spacing,
        "units": "opacity density [1/m]",
        "value_range": [float(rho.min()), float(rho.max())],
        "suggested_iso": [round(0.3 * p99, 6), round(0.6 * p99, 6)],
        "run": voxels.name,
        "layers": layers,
        "inversion_version": vol.version,
        "offsets": vol.offsets,
        "fit_info": vol.info,
        "detectors": [
            {"id": e.id, "x": e.pose.x, "y": e.pose.y, "z": e.pose.z, "az_deg": e.pose.az_deg}
            for e in cfg.exposures
        ],
        "resolution": {
            "max_baseline_m": res["max_baseline_m"],
            "n_positions": res["n_positions"],
            "depth_resolution_m": res["depth_resolution_m"],
            "depth_resolved": res["depth_resolved"],
            "verdict": res["verdict"],
        },
    }
    if cfg.volume.viewer_crop_xy_m is not None:
        meta["viewer_crop_xy_m"] = [list(p) for p in cfg.volume.viewer_crop_xy_m]
    if results_dir is not None:
        beams = _beams_meta(Path(results_dir), cfg)
        if beams is not None:
            meta["beams"] = beams
    (out / "meta.json").write_text(json.dumps(_json_safe(meta), indent=2) + "\n")
    return out / "volume.npy"
