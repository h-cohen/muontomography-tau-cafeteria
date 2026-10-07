"""One subcommand per pipeline stage; the Makefile is the intended driver.

Every results/*.json is flat at the top level with digit-free keys: the paper's
number macros are generated from those keys, and a LaTeX macro name cannot
contain digits. Nested values (lists, per-position dicts) are kept for figures.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cafetomo.angular import AnalysisGrid, load_analysis_grid
from cafetomo.config import Config, load_config

# Position ids carry digits (pos0, pos1); data.json names them by their order
# in the config instead, so the keys stay valid macro names.
_ORDINALS = ("first", "second", "third", "fourth")

# The quantities the paper quotes with a full error budget.
_BUDGET_KEYS = ("autofocus_z", "beams_z", "beams_zx", "depth_h", "depth_zbottom")


def _cfg(args) -> Config:
    cfg = load_config(args.config, pose_file=getattr(args, "pose", None))
    return cfg.fast() if args.fast else cfg


def _plain(obj):
    """numpy scalars and arrays as the Python values json understands."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=_plain) + "\n")


def _grid(cfg: Config, ingest_dir: Path) -> tuple[AnalysisGrid, dict[str, float]]:
    from cafetomo.ingest import live_times
    ids = [*cfg.position_ids, cfg.sky_reference.id]
    grid = load_analysis_grid(ingest_dir, ids, cfg.opacity.rebin_factor)
    return grid, live_times(ingest_dir, ids)


def _source_names(cfg: Config) -> dict[str, str]:
    if len(cfg.position_ids) > len(_ORDINALS):
        raise ValueError(f"{len(cfg.position_ids)} positions; data.json names only "
                         f"{len(_ORDINALS)}")
    return dict(zip(cfg.position_ids, _ORDINALS, strict=False)) | {cfg.sky_reference.id: "sky"}


def cmd_ingest(args) -> None:
    """data.json: kept tracks and live hours per source, as `<ordinal>_tracks`
    / `<ordinal>_hours` (`first`, `second`, ... in config order, and `sky`)."""
    from cafetomo.ingest import ingest
    from cafetomo.provenance import write_meta
    cfg = _cfg(args)
    out = Path(args.out)
    res = ingest(cfg, out)
    # ingest records each source's live time in meta.json; provenance is added
    # to that record, not written over it.
    exposures = json.loads((out / "meta.json").read_text())["exposures"]
    write_meta(out, config_path=args.config, stage="ingest", extra={"exposures": exposures})
    names = _source_names(cfg)
    doc = {}
    for r in res:
        doc[f"{names[r.source_id]}_tracks"] = r.total_kept
        doc[f"{names[r.source_id]}_hours"] = r.live_time_s / 3600.0
    _write_json(Path(args.results) / "data.json", doc)


def cmd_selfcal(args) -> None:
    from cafetomo.selfcal import fit_pose, pose_bootstrap, pose_result
    cfg = _cfg(args)
    grid, live = _grid(cfg, Path(args.ingest))
    fit = fit_pose(grid, cfg, live)
    sigma = pose_bootstrap(grid, cfg, live, n=cfg.selfcal.n_bootstrap, seed=cfg.uncertainty.seed)
    _write_json(Path(args.out), pose_result(fit, sigma, cfg))


def cmd_opacity(args) -> None:
    from cafetomo.opacity import opacity_sigma, save_sigma, solve_opacity
    from cafetomo.provenance import write_meta
    cfg = _cfg(args)
    grid, live = _grid(cfg, Path(args.ingest))
    out = Path(args.out)
    solve_opacity(grid, cfg, live).save(out / "maps.npz")
    save_sigma(out / "sigma.npz", opacity_sigma(grid, cfg, live,
                                                 n_replicas=cfg.opacity.n_sigma_replicas,
                                                 seed=cfg.uncertainty.seed))
    write_meta(out, config_path=args.config, stage="opacity")


def _fit_data(cfg: Config, opacity_dir: Path):
    from cafetomo.opacity import OpacityMaps, build_fit_data, load_sigma
    maps = OpacityMaps.load(opacity_dir / "maps.npz")
    sigma = load_sigma(opacity_dir / "sigma.npz")
    return maps, sigma, build_fit_data(maps, cfg, sigma)


def cmd_reconstruct(args) -> None:
    """Volumes plus two coverage maps: views (positions per voxel, the depth
    information) and rays (rows per voxel, the viewer's coverage gate)."""
    from cafetomo.forward import build_forward_model
    from cafetomo.provenance import write_meta
    from cafetomo.reconstruct import solve_voxels
    from cafetomo.resolution import rays_per_voxel, views_per_voxel
    cfg = _cfg(args)
    _, _, data = _fit_data(cfg, Path(args.opacity))
    out = Path(args.out)
    fits = solve_voxels(data, cfg, cache_dir=args.cache)
    for name, sol in fits.items():
        sol.save(out / f"volume_{name}.npz")
    full = fits["full"]
    fwd = build_forward_model(data.rows, cfg, grid=full.grid, cache_dir=args.cache)
    np.save(out / "views.npy", views_per_voxel(fwd))
    np.save(out / "rays.npy", rays_per_voxel(fwd))
    _write_json(Path(args.results) / "reconstruction.json", {
        "chisq_per_row": full.info["best_chi2"], "iterations": full.info["n_iter_used"],
        "rows": full.info["n_rows_used"], "voxels": int(full.grid.n_voxels),
        "spacing": full.grid.spacing, "nx": full.grid.shape[0], "ny": full.grid.shape[1],
        "nz": full.grid.shape[2]})
    write_meta(out, config_path=args.config, stage="reconstruct")


def cmd_analyze(args) -> None:
    """autofocus.json, beams.json (+ the reconstruction gate as `gate_<k>`) and
    beamdepth.json (+ the voxel z-profile cross-check as `zprofile_<k>`)."""
    from cafetomo.beams import sky_images, verify_gate
    from cafetomo.measure import measure
    from cafetomo.reconstruct import VoxelSolution
    cfg = _cfg(args)
    maps, _, data = _fit_data(cfg, Path(args.opacity))
    sol = VoxelSolution.load(Path(args.voxels) / "volume_full.npz")
    m = measure(data, cfg, maps.sky, vgrid=sol.grid, cache_dir=args.cache)
    res = Path(args.results)
    _write_json(res / "autofocus.json", m.details["autofocus"].to_json())
    beams = m.details["beams"]
    gate = verify_gate(sky_images(data, maps.sky), cfg.origins(), maps.sky.centers,
                       beams["z"], sol, cfg.beams)
    _write_json(res / "beams.json", beams | {f"gate_{k}": v for k, v in gate.items()})
    depth = m.details["depth"]
    _write_json(res / "beamdepth.json", depth.to_json()
                | {f"zprofile_{k}": v for k, v in m.details["zprofile"].items()})
    if depth.at_bound:
        print("WARNING: beam-depth fit ended on a bound; h is not a measurement",
              file=sys.stderr)


def cmd_validate(args) -> None:
    from cafetomo.validation import validate_autofocus, validate_depth
    cfg = _cfg(args)
    maps, _, data = _fit_data(cfg, Path(args.opacity))
    nominal = json.loads((Path(args.results) / "beamdepth.json").read_text())
    doc = validate_autofocus(data, cfg, nominal, cache_dir=args.cache)
    doc |= validate_depth(data, cfg, nominal, sky=maps.sky)
    _write_json(Path(args.results) / "validation.json", doc)


def cmd_uncertainty(args) -> None:
    """uncertainty.json: bootstrap summary, MCS scale (`mcs_<k>`) and the error
    budget of every quoted quantity."""
    from cafetomo.bootstrap import run_bootstrap
    from cafetomo.measure import measure
    from cafetomo.reconstruct import VoxelSolution
    from cafetomo.systematics import (
        background_shift,
        error_budget,
        flux_scale_shift,
        mcs,
        mcs_shift,
        pose_shift,
    )
    if args.pose is None:
        raise ValueError("uncertainty needs --pose: the pose systematic moves by its sigma")
    cfg = _cfg(args)
    grid, live = _grid(cfg, Path(args.ingest))
    maps, sigma, data = _fit_data(cfg, Path(args.opacity))
    sky = maps.sky
    vgrid = VoxelSolution.load(Path(args.voxels) / "volume_full.npz").grid
    nominal = measure(data, cfg, sky, cache_dir=args.cache, with_volume=False)
    boot = run_bootstrap(grid, cfg, live_time=live, sigma=sigma, rows=data.rows, vgrid=vgrid,
                         sky=sky, cache_dir=args.cache)
    boot.save(args.out)
    stat = boot.summary()
    scat = mcs(cfg, h_m=nominal.values["depth_h"], lever_m=nominal.values["depth_zbottom"])
    flux = flux_scale_shift(maps, cfg, sigma, data.rows, nominal, sky=sky, cache_dir=args.cache)
    mcs_d = mcs_shift(data, cfg, nominal, sky=sky, jitter_tan=scat["jitter_tan"],
                      cache_dir=args.cache)
    pose_doc = json.loads(Path(args.pose).read_text())
    pose = pose_shift(grid, cfg, live, sigma, {"x": pose_doc["x_sigma"], "y": pose_doc["y_sigma"]},
                      nominal, sky=sky, rows=data.rows, cache_dir=args.cache)
    bg = background_shift(data, cfg, nominal, sky=sky, cache_dir=args.cache)
    _write_json(Path(args.results) / "uncertainty.json",
                {"replicas": cfg.uncertainty.n_replicas, **stat,
                 **{f"mcs_{k}": v for k, v in scat.items()},
                 **error_budget(stat, flux, mcs_d, pose, bg, _BUDGET_KEYS)})


def cmd_export(args) -> None:
    """`--results` carries the fitted beam boxes (and their total sigma) into
    the viewer."""
    from cafetomo.export import export_volume
    export_volume(args.voxels, _cfg(args), bootstrap_dir=args.bootstrap,
                  results_dir=args.results, out_dir=args.out)


def cmd_viewer(args) -> None:
    from cafetomo.viewerbuild import build
    build(Path("viewer"), out_path=Path(args.out), embed_dir=Path(args.export))


def cmd_numbers(args) -> None:
    from cafetomo.numbers import write_numbers
    write_numbers(Path(args.results), Path(args.out))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cafetomo", description=__doc__)
    ap.add_argument("--fast", action="store_true", help="cheap settings for CI and smoke runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def stage(name, fn, *opts):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        # Without --pose a stage runs on the config's prior pose; the Makefile
        # passes the fitted one to every stage after selfcal.
        for o in opts:
            p.add_argument(f"--{o}", required=o not in {"cache", "pose"})

    stage("ingest", cmd_ingest, "config", "out", "results")
    stage("selfcal", cmd_selfcal, "config", "ingest", "out")
    stage("opacity", cmd_opacity, "config", "pose", "ingest", "out")
    stage("reconstruct", cmd_reconstruct, "config", "pose", "opacity", "out", "results", "cache")
    stage("analyze", cmd_analyze, "config", "pose", "opacity", "voxels", "results", "cache")
    stage("validate", cmd_validate, "config", "pose", "opacity", "results", "cache")
    stage("uncertainty", cmd_uncertainty, "config", "pose", "ingest", "opacity", "voxels",
          "results", "out", "cache")
    stage("export", cmd_export, "config", "pose", "voxels", "bootstrap", "results", "out")
    stage("viewer", cmd_viewer, "export", "out")
    stage("numbers", cmd_numbers, "results", "out")
    args = ap.parse_args(argv)
    args.fn(args)
