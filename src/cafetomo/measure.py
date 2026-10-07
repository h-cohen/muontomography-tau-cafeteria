"""Every scalar paper result from one measurement vector.

The nominal run, each bootstrap replica and each systematic variation call
this one function, so the uncertainty describes exactly the estimator that is
reported.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cafetomo.autofocus import cv_height_scan
from cafetomo.beamdepth import estimate_beam_depth, zprofile_depth
from cafetomo.config import Config
from cafetomo.fitdata import FitData
from cafetomo.reconstruct import solve_voxels
from cafetomo.sky import SkyGrid
from cafetomo.voxels import VoxelGrid


@dataclass(frozen=True)
class Measurement:
    values: dict[str, float]
    volume: np.ndarray | None
    details: dict = field(default_factory=dict, repr=False)


def measure(data: FitData, cfg: Config, sky: SkyGrid, *, vgrid: VoxelGrid | None = None,
            cache_dir: str | Path | None = None, angle_jitter: float = 0.0,
            with_volume: bool = True) -> Measurement:
    """Autofocus height, triangulated beam heights (joint and x-only, so the
    y family's pull stays visible), pitch, the parametric beam depth and,
    with `with_volume`, the voxel z-profile that cross-checks it.

    The beam depth comes from `estimate_beam_depth`, the same path the
    phantom validation runs, so its validated bias is this estimator's."""
    scan = cv_height_scan(data, cfg, cache_dir=cache_dir)
    beams, depth = estimate_beam_depth(data, cfg, sky, angle_jitter=angle_jitter)
    values = {"autofocus_z": scan.z_best, "beams_z": beams["z"], "beams_zx": beams["z_x"],
              "beams_pitch": beams["pitch"], "depth_h": depth.h, "depth_w": depth.w,
              "depth_zbottom": depth.z0, "depth_ztop": depth.z0 + depth.h}
    volume = None
    details = {"autofocus": scan, "beams": beams, "depth": depth}
    if with_volume:
        sol = solve_voxels(data, cfg, cache_dir=cache_dir, holdouts=False, grid=vgrid)["full"]
        zp = zprofile_depth(sol, cfg, xs=depth.xs, w=depth.w, z_ref=depth.z0)
        values |= {"zprofile_fwhm": zp["fwhm"], "zprofile_bottom": zp["bottom"],
                   "zprofile_top": zp["top"]}
        volume = sol.rho3()
        details |= {"solution": sol, "zprofile": zp}
    return Measurement(values=values, volume=volume, details=details)
