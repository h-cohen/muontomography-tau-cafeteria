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
        raise ValueError(
            f"{len(cfg.position_ids)} positions; data.json names only {len(_ORDINALS)}"
        )
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


def cmd_ingest_raw(args) -> None:
    """An ingest directory in which each exposure with raw files takes its counts and
    live time from them (see `rawdata`); the other sources keep their ROOT histograms."""
    from cafetomo.ingest import ingest
    from cafetomo.provenance import write_meta
    from cafetomo.rawdata import ingest_raw

    cfg = _cfg(args)
    if cfg.raw is None:
        raise SystemExit(f"{args.config}: no raw section")
    out = Path(args.out)
    cache = Path(args.cache or out.parent / "cache")
    raw = {pid: ingest_raw(cfg, pid, cache, jobs=int(args.jobs or 1)) for pid in cfg.raw.exposures}
    ingest(cfg, out, raw)
    exposures = json.loads((out / "meta.json").read_text())["exposures"]
    write_meta(out, config_path=args.config, stage="ingest-raw", extra={"exposures": exposures})


def cmd_channelcal(args) -> None:
    """Per-channel Landau x Gauss charge calibration of each raw exposure (`channelcal`)."""
    from cafetomo.channelcal import render, summarize
    from cafetomo.provenance import write_meta
    from cafetomo.rawdata import ingest_raw

    cfg = _cfg(args)
    if cfg.raw is None:
        raise SystemExit(f"{args.config}: no raw section")
    out = Path(args.out)
    cache = Path(args.cache or out.parent / "cache")
    for pid in cfg.raw.exposures:
        files = ingest_raw(cfg, pid, cache, jobs=int(args.jobs or 1)).files
        hit = [f.hit_hist for f in files]
        ped = [f.ped_hist for f in files]
        doc = summarize(
            hit,
            ped,
            n_flagged=sum(f.n_flagged for f in files),
            n_valid_rows=sum(f.n_valid_rows for f in files),
            channel_bars=cfg.raw.channel_bars,
            settings=cfg.calibration,
        )
        render(doc, hit, ped, out, pid, cfg.calibration)
    write_meta(
        out,
        config_path=args.config,
        stage="channelcal",
        extra={"exposures": list(cfg.raw.exposures)},
    )


def cmd_rawcheck(args) -> None:
    """rawdata.json: the raw build against the ROOT histograms (counts, live time, opacity),
    the sky's stability across its days and later runs, and the channel calibration of
    every raw source. Keys are `<ordinal>_...` as in data.json."""
    from cafetomo.angular import AngularHist, load_counts
    from cafetomo.ingest import crop_counts, read_live_time
    from cafetomo.opacity import poisson_sigma, solve_opacity
    from cafetomo.rawcheck import count_match, epoch_scatter, lambda_shift
    from cafetomo.rawdata import TAN_EDGES, ingest_raw

    cfg = _cfg(args)
    raw_dir, root_dir, cal_dir = Path(args.ingest), Path(args.root_ingest), Path(args.channelcal)
    names = _source_names(cfg)
    meta = json.loads((raw_dir / "meta.json").read_text())["exposures"]
    roots = {e.id: e.root_file for e in cfg.exposures} | {
        cfg.sky_reference.id: cfg.sky_reference.root_file
    }
    doc: dict = {"clock_divisor": cfg.raw.clock_divisor}
    cal = {}
    for sid in cfg.raw.exposures:
        o = names[sid]
        raw, root = (load_counts(d / f"counts_{sid}.npz") for d in (raw_dir, root_dir))
        chi2, scale = count_match(raw, root)
        root_live = read_live_time(cfg.data_dir / roots[sid], interpolate=True)
        doc |= {
            f"{o}_raw_files": len(meta[sid]["raw_files"]),
            f"{o}_raw_tracks": int(meta[sid]["total_counts"]),
            f"{o}_raw_hours": meta[sid]["live_time_s"] / 3600.0,
            f"{o}_root_chi": chi2,
            f"{o}_root_count_ratio": scale,
            f"{o}_root_live_ratio": meta[sid]["live_time_s"] / root_live,
        }
        cal[sid] = json.loads((cal_dir / f"{sid}_channelcal.json").read_text())
        ok = [r for r in cal[sid]["channels"] if r["status"] == "ok"]
        drift = [r["day_rms_adc"] / r["mpv_adc"] for r in ok if r["day_rms_adc"] is not None]
        spread = [a["mpv_rms_frac"] for a in cal[sid]["asics"] if a["mpv_rms_frac"] is not None]
        doc |= {
            f"{o}_cal_ok": cal[sid]["n_ok"],
            f"{o}_cal_flagged": cal[sid]["n_flagged"],
            f"{o}_cal_inactive": cal[sid]["n_inactive"],
            f"{o}_cal_gain_spread": float(np.median(spread)) if spread else float("nan"),
            f"{o}_cal_day_drift": float(np.median(drift)) if drift else float("nan"),
        }
    first = cfg.position_ids[0]
    for sid in cal:
        if sid == first:
            continue
        rows = {(r["asic"], r["channel"]): r for r in cal[first]["channels"]}
        pairs = [
            (r["peak_adc"], rows[k]["peak_adc"], r["pedestal_adc"] - rows[k]["pedestal_adc"])
            for r in cal[sid]["channels"]
            if r["bar"] is not None
            and np.isfinite(r["peak_adc"])
            and np.isfinite(rows[k := (r["asic"], r["channel"])]["peak_adc"])
        ]
        gain = np.array([p / q for p, q, _ in pairs])
        o = names[sid]
        doc |= {
            f"{o}_first_gain_ratio": float(gain.mean()),
            f"{o}_first_gain_scatter": float(gain.std()),
            f"{o}_first_pedestal_shift": float(np.mean([d for *_, d in pairs])),
        }

    sky = cfg.sky_reference.id
    if sky in cfg.raw.exposures:
        days = [
            crop_counts(f.values, TAN_EDGES, TAN_EDGES, "txty", cfg.binning.edges())
            for f in ingest_raw(cfg, sky, Path(args.cache or raw_dir.parent / "cache")).files
        ]
        half = len(days) // 2
        doc["sky_halves_shape_scatter"], doc["sky_halves_shape_noise"] = epoch_scatter(
            days[:half], days[half:]
        )
        raw_sky, root_sky = (load_counts(d / f"counts_{sky}.npz") for d in (raw_dir, root_dir))
        later = AngularHist(
            root_sky.values - raw_sky.values, root_sky.xedges, root_sky.yedges, "txty"
        )
        doc["sky_late_shape_scatter"], doc["sky_late_shape_noise"] = epoch_scatter(
            [raw_sky], [later]
        )
        doc["sky_raw_live_fraction"] = meta[sky]["live_time_s"] / read_live_time(
            cfg.data_dir / cfg.sky_reference.root_file, interpolate=True
        )

    if sky in cfg.raw.exposures:
        from cafetomo.beamview import opacity_view, render_sky_difference, sky_difference

        raw_live = meta[sky]["live_time_s"]
        root_live = read_live_time(cfg.data_dir / cfg.sky_reference.root_file, interpolate=True)
        diffs = {}
        for pid in cfg.raw.exposures:
            if pid == sky:
                continue
            pos = load_counts(raw_dir / f"counts_{pid}.npz")
            views = [
                opacity_view(pos, s, meta[pid]["live_time_s"] / t, cfg.view, cfg.opacity.t_floor)
                for s, t in ((raw_sky, raw_live), (root_sky, root_live))
            ]
            diffs[pid] = sky_difference(*views)
            o = names[pid]
            doc |= {
                f"{o}_sky_delta_mean": float(np.nanmean(diffs[pid].delta)),
                f"{o}_sky_delta_rms": float(np.nanstd(diffs[pid].delta)),
                f"{o}_sky_delta_noise": float(np.sqrt(np.nanmean(diffs[pid].noise ** 2))),
            }
        render_sky_difference(diffs, cfg.view, Path(args.out).parent / "sky_difference.png")

    grid, live = _grid(cfg, raw_dir)
    lam = solve_opacity(grid, cfg, live).lam
    noise = poisson_sigma(grid, cfg, live)
    root_grid, root_live = _grid(cfg, root_dir)
    root_lam = solve_opacity(root_grid, cfg, root_live).lam
    for pid in cfg.position_ids:
        o = names[pid]
        shift, scatter, n = lambda_shift(lam[pid], root_lam[pid])
        doc |= {
            f"{o}_lam_shift": shift,
            f"{o}_lam_scatter": scatter,
            f"{o}_lam_noise": float(np.nanmedian(noise[pid])),
            f"{o}_lam_bins": n,
        }
    _write_json(Path(args.out), doc)


def cmd_beams2d(args) -> None:
    """The 2D angular opacity view (`beamview`) of each position, or of `--exposure`,
    from an ingest directory."""
    from cafetomo.angular import load_counts
    from cafetomo.beamview import opacity_view, render, without_floors
    from cafetomo.ingest import live_times
    from cafetomo.provenance import write_meta

    cfg = _cfg(args)
    ingest = Path(args.ingest)
    ids = [args.exposure] if args.exposure else list(cfg.position_ids)
    sky_id = cfg.sky_reference.id
    live = live_times(ingest, [sky_id, *ids])
    sky = load_counts(ingest / f"counts_{sky_id}.npz")
    for pid in ids:
        pos = load_counts(ingest / f"counts_{pid}.npz")
        ratio = live[pid] / live[sky_id]
        view, full = (
            opacity_view(pos, sky, ratio, v, cfg.opacity.t_floor)
            for v in (cfg.view, without_floors(cfg.view))
        )
        render(view, full, cfg.view, Path(args.out), pid)
    write_meta(args.out, config_path=args.config, stage="beams2d", extra={"exposures": ids})


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
    save_sigma(
        out / "sigma.npz",
        opacity_sigma(
            grid, cfg, live, n_replicas=cfg.opacity.n_sigma_replicas, seed=cfg.uncertainty.seed
        ),
    )
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
    _write_json(
        Path(args.results) / "reconstruction.json",
        {
            "chisq_per_row": full.info["best_chi2"],
            "iterations": full.info["n_iter_used"],
            "rows": full.info["n_rows_used"],
            "voxels": int(full.grid.n_voxels),
            "spacing": full.grid.spacing,
            "nx": full.grid.shape[0],
            "ny": full.grid.shape[1],
            "nz": full.grid.shape[2],
        },
    )
    write_meta(out, config_path=args.config, stage="reconstruct")


def cmd_analyze(args) -> None:
    """autofocus.json, beams.json (+ the reconstruction gate as `gate_<k>`) and
    beamdepth.json (+ the voxel z-profile cross-check as `zprofile_<k>`)."""
    from cafetomo.beamdepth import profile_beam_depth
    from cafetomo.beams import sky_images, verify_gate
    from cafetomo.measure import measure
    from cafetomo.reconstruct import VoxelSolution

    cfg = _cfg(args)
    maps, _, data = _fit_data(cfg, Path(args.opacity))
    sol = VoxelSolution.load(Path(args.voxels) / "volume_full.npz")
    m = measure(data, cfg, maps.sky, vgrid=sol.grid, cache_dir=args.cache)
    res = Path(args.results)
    beams = m.details["beams"]
    gate = verify_gate(
        sky_images(data, maps.sky), cfg.origins(), maps.sky.centers, beams["z"], sol, cfg.beams
    )
    depth = m.details["depth"]
    profile = {}
    if cfg.beamdepth.model == "geometry":
        heights = np.unique(
            np.r_[np.linspace(0.1, cfg.beamdepth.h_max_m, 21), cfg.beamdepth.h_starts_m, depth.h]
        )
        profile = {
            "depth_profile": profile_beam_depth(
                data,
                cfg,
                xs_init=beams["beams_x"],
                z0_init=beams["z"],
                heights=heights,
                nominal=depth,
            )
        }
    _write_json(res / "autofocus.json", m.details["autofocus"].to_json())
    _write_json(res / "beams.json", beams | {f"gate_{k}": v for k, v in gate.items()})
    depth_doc = (
        depth.to_json() | profile | {f"zprofile_{k}": v for k, v in m.details["zprofile"].items()}
    )
    if profile and profile["depth_profile"]["minimum_at_edge"]:
        depth_doc |= {
            "depth_resolved": False,
            "h_measurement": float("nan"),
            "zbottom_measurement": float("nan"),
        }
    _write_json(res / "beamdepth.json", depth_doc)
    if profile and profile["depth_profile"]["minimum_at_edge"]:
        print(
            "WARNING: depth profile minimum is at a sampled edge; depth is unresolved",
            file=sys.stderr,
        )
    if depth.at_bound:
        print("WARNING: beam-depth fit ended on a bound; h is not a measurement", file=sys.stderr)


def cmd_depthcheck(args) -> None:
    """Refit the same depth estimator using analytic count uncertainty weights.

    This diagnostic changes only the weights and preserves the nominal rows.
    It is not a calibrated additional contribution to the uncertainty budget.
    """
    from cafetomo.beamdepth import estimate_beam_depth
    from cafetomo.opacity import build_fit_data, poisson_sigma

    cfg = _cfg(args)
    maps, _, data = _fit_data(cfg, Path(args.opacity))
    grid, live = _grid(cfg, Path(args.ingest))
    analytic = build_fit_data(maps, cfg, poisson_sigma(grid, cfg, live), rows=data.rows)
    beams, depth = estimate_beam_depth(analytic, cfg, maps.sky)
    _write_json(
        Path(args.results) / "depthdiagnostics.json",
        {
            "analytic_h": depth.h,
            "analytic_zbottom": depth.z0,
            "analytic_chisq_per_dof": depth.chi2_per_dof,
            "analytic_at_bound": depth.at_bound,
            "analytic_beams_z": beams["z"],
            "analytic_kappa": list(depth.kappa),
        },
    )


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
    budget of every quoted quantity: `<k>_stat`, one `<k>_<source>` per
    systematic (flux scale, MCS, pose, background, concrete density `rho`,
    muon spectrum `model`) and `<k>_total`."""
    from cafetomo.bootstrap import run_bootstrap
    from cafetomo.measure import measure
    from cafetomo.reconstruct import VoxelSolution
    from cafetomo.systematics import (
        background_shift,
        density_shift,
        error_budget,
        flux_model_shift,
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
    boot = run_bootstrap(
        grid,
        cfg,
        live_time=live,
        sigma=sigma,
        rows=data.rows,
        vgrid=vgrid,
        sky=sky,
        cache_dir=args.cache,
    )
    boot.save(args.out)
    stat = boot.summary()
    scat = mcs(cfg, h_m=nominal.values["depth_h"], lever_m=nominal.values["depth_zbottom"])
    flux = flux_scale_shift(maps, cfg, sigma, data.rows, nominal, sky=sky, cache_dir=args.cache)
    mcs_d = mcs_shift(
        data, cfg, nominal, sky=sky, jitter_tan=scat["jitter_tan"], cache_dir=args.cache
    )
    pose_doc = json.loads(Path(args.pose).read_text())
    pose = pose_shift(
        grid,
        cfg,
        live,
        sigma,
        {"x": pose_doc["x_sigma"], "y": pose_doc["y_sigma"], "az_deg": pose_doc["az_sigma"]},
        nominal,
        sky=sky,
        rows=data.rows,
        cache_dir=args.cache,
    )
    bg = background_shift(data, cfg, nominal, sky=sky, cache_dir=args.cache)
    shifts = {"flux": flux, "mcs": mcs_d, "pose": pose, "bg": bg}
    if cfg.beamdepth.model == "concrete":
        shifts["rho"] = density_shift(data, cfg, nominal, sky=sky, cache_dir=args.cache)
        shifts["model"] = flux_model_shift(data, cfg, nominal, sky=sky, cache_dir=args.cache)
    _write_json(
        Path(args.results) / "uncertainty.json",
        {
            "replicas": cfg.uncertainty.n_replicas,
            **stat,
            **{f"mcs_{k}": v for k, v in scat.items()},
            **error_budget(stat, shifts, _BUDGET_KEYS),
        },
    )


def cmd_export(args) -> None:
    """`--results` carries the fitted beam boxes (and their total sigma) into
    the viewer."""
    from cafetomo.export import export_volume

    export_volume(
        args.voxels,
        _cfg(args),
        bootstrap_dir=args.bootstrap,
        results_dir=args.results,
        out_dir=args.out,
    )


def cmd_viewer(args) -> None:
    from cafetomo.viewerbuild import build

    build(Path("viewer"), out_path=Path(args.out), embed_dir=Path(args.export))


def cmd_numbers(args) -> None:
    from cafetomo.numbers import write_numbers

    write_numbers(Path(args.results), Path(args.out))


def cmd_config_numbers(args) -> None:
    from cafetomo.numbers import write_config_numbers

    write_config_numbers(Path(args.config), Path(args.volume), Path(args.out))


def cmd_production_sync(args) -> None:
    from cafetomo.production import sync

    sha = sync(
        Path(args.production),
        staged=Path(args.staged),
        results=Path(args.results),
        source_root=Path.cwd(),
    )
    print(f"{args.production} at {sha[:7]}; push with: make production-push")


def cmd_production_check(args) -> None:
    from cafetomo.production import check, literal_numbers

    root = Path(args.production)
    for warning in literal_numbers(root, args.lint.split(",")):
        print(f"warning: typed number {warning}")
    problems = check(root / "generated", results=Path(args.results))
    for p in problems:
        print(f"error: {p}")
    if problems:
        raise SystemExit(1)


def cmd_internal_validate(args) -> None:
    import hashlib

    import yaml

    from cafetomo.internal_validation import run_campaign

    cfg = _cfg(args)
    settings = yaml.safe_load(Path(args.settings).read_text())
    if args.fast:
        settings["mode"] = "smoke"
    grid, live = _grid(cfg, Path(args.ingest))
    provenance = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in (("config_file_hash", args.config), ("settings_file_hash", args.settings))
    }
    provenance["config_path"] = str(Path(args.config).resolve())
    run_campaign(
        grid,
        live,
        cfg,
        settings,
        out=args.out,
        cache=args.cache or str(Path(args.out) / "cache"),
        provenance=provenance,
    )


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
            p.add_argument(f"--{o}", required=o not in {"cache", "pose", "jobs", "exposure"})

    stage(
        "internal-validate", cmd_internal_validate, "config", "ingest", "settings", "out", "cache"
    )
    stage("ingest", cmd_ingest, "config", "out", "results")
    stage("ingest-raw", cmd_ingest_raw, "config", "out", "cache", "jobs")
    stage(
        "rawcheck",
        cmd_rawcheck,
        "config",
        "pose",
        "ingest",
        "root-ingest",
        "channelcal",
        "cache",
        "out",
    )
    stage("channelcal", cmd_channelcal, "config", "out", "cache", "jobs")
    stage("beams2d", cmd_beams2d, "config", "ingest", "out", "exposure")
    stage("selfcal", cmd_selfcal, "config", "ingest", "out")
    stage("opacity", cmd_opacity, "config", "pose", "ingest", "out")
    stage("reconstruct", cmd_reconstruct, "config", "pose", "opacity", "out", "results", "cache")
    stage("analyze", cmd_analyze, "config", "pose", "opacity", "voxels", "results", "cache")
    stage("depthcheck", cmd_depthcheck, "config", "pose", "ingest", "opacity", "results")
    stage("validate", cmd_validate, "config", "pose", "opacity", "results", "cache")
    stage(
        "uncertainty",
        cmd_uncertainty,
        "config",
        "pose",
        "ingest",
        "opacity",
        "voxels",
        "results",
        "out",
        "cache",
    )
    stage("export", cmd_export, "config", "pose", "voxels", "bootstrap", "results", "out")
    stage("viewer", cmd_viewer, "export", "out")
    stage("numbers", cmd_numbers, "results", "out")
    stage("config-numbers", cmd_config_numbers, "config", "volume", "out")
    stage("production-sync", cmd_production_sync, "production", "staged", "results")
    stage("production-check", cmd_production_check, "production", "results", "lint")
    args = ap.parse_args(argv)
    args.fn(args)
