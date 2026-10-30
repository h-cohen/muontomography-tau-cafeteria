"""3D systematics, time stability and depth controls (PREREG.md).

run.py run --config C --baseline B --results R --skyswap K --out O [--jobs N]
run.py summarize --out O --results R

Each variant is one task whose result is written to O/variants/<name>.json, so an interrupted
run resumes; `summarize` turns them into O/summary.json (shifts, envelopes, budgets, tests).
"""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from cafetomo.angular import save_counts
from cafetomo.arraydepth import array_model, continued_centres, fit_array_depth
from cafetomo.beams import find_beams
from cafetomo.cli import _fit_data, _grid
from cafetomo.config import RunSubset, load_config
from cafetomo.measure import measure
from cafetomo.opacity import build_fit_data, solve_opacity
from cafetomo.rawcheck import epoch_scatter

SKY_VARIANTS = {
    "sky_flip": "joint/ingest_m3_phi180",
    "sky_b_aligned": "joint/ingest_m3_phiB",
    "sky_m2": "joint/ingest_m2_phi0",
    "sky_plain_sum": "ingest_SKY",
}
RECON_VARIANTS = {
    "alpha_half": ("reconstruction", "tv_alpha", 0.5),
    "alpha_double": ("reconstruction", "tv_alpha", 2.0),
    "damping_half": ("reconstruction", "coverage_damping", 0.5),
    "damping_double": ("reconstruction", "coverage_damping", 2.0),
    "spacing_015": ("volume", "spacing_m", 0.15),
    "spacing_025": ("volume", "spacing_m", 0.25),
    "early_stop": ("reconstruction", "n_iter", 15),
}
RECOVERY_H = (0.6, 1.2, 1.8)
N_CONTROL = 20
SLAB = 0.1  # misspecified slab opacity at normal incidence
HEIGHTS = ("autofocus_z", "beams_z", "beams_zx", "beams_pitch")
IMAGE = ("zprofile_fwhm", "zprofile_bottom", "zprofile_top")
DEPTH = ("h", "w", "bottom")

A = None  # parsed arguments, set in every worker by `init`


def init(args):
    global A
    A = args
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def nominal_cfg(**changes):
    cfg = load_config(A.config, A.results / "pose.json")
    for section, key, value in changes.get("set", ()):
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **{key: value})})
    return cfg


def nominal_inputs(cfg):
    maps, sigma, data = _fit_data(cfg, A.baseline / "opacity")
    return maps, sigma, data


def data_from(cfg, ingest, sigma, rows):
    grid, live = _grid(cfg, Path(ingest))
    return build_fit_data(solve_opacity(grid, cfg, live), cfg, sigma, rows=rows)


def array_fit(data, cfg, beams, per_end=None, scale=1.0):
    xs = np.asarray(beams["beams_x"])
    xs = xs.mean() + (xs - xs.mean()) * scale
    centres = continued_centres(xs) if per_end is None else continued_centres(xs, per_end)
    return fit_array_depth(data, cfg, centres=centres, height_seed=beams["z"])


def evaluate(data, cfg, sky, name):
    """Every estimator of the registration on one dataset; the volume is kept for correlation."""
    m = measure(data, cfg, sky, cache_dir=A.out / "cache", with_volume=True)
    beams = m.details["beams"]
    arr = array_fit(data, cfg, beams)
    np.save(A.out / "volumes" / f"{name}.npy", m.volume)
    return {
        **{k: float(m.values[k]) for k in (*HEIGHTS, *IMAGE)},
        **{k: arr[k] for k in (*DEPTH, "chisq_per_dof", "dof")},
        "beams_x": beams["beams_x"],
        "array": arr,
    }


def half_ingests():
    """Early and late halves of the room runs, counted with the whole exposure's gains."""
    from cafetomo.channelcal import exposure_gains
    from cafetomo.rawdata import ingest_raw

    cfg = load_config(A.config)
    raw = cfg.raw
    cache = A.out / "rawcache"
    plain = replace(cfg, raw=replace(raw, gain_correction=False))
    gains = {
        pid: exposure_gains(
            ingest_raw(plain, pid, cache, A.jobs).files, raw.channel_bars, cfg.calibration
        )
        for pid in ("pos0", "pos1")
    }
    halves = {}
    files = sorted(Path(raw.exposures["pos0"]).glob("*.data"))
    for i, part in enumerate((files[: len(files) // 2], files[len(files) // 2 :])):
        folder = A.out / "raw" / f"pos0_h{i}"
        folder.mkdir(parents=True, exist_ok=True)
        for f in part:
            if not (folder / f.name).exists():
                (folder / f.name).symlink_to(f.resolve())
        c = replace(cfg, raw=replace(raw, exposures=raw.exposures | {"pos0": str(folder)}))
        halves[("pos0", i)] = ingest_raw(c, "pos0", cache, A.jobs, gains=gains["pos0"])
    n_seg = ingest_raw(plain, "pos1", cache, A.jobs).files[0].n_segments
    seg = {f"pos1_s{k}": RunSubset("pos1", (k, k), 0.0) for k in range(1, n_seg + 1)}
    c = replace(
        cfg, raw=replace(raw, first_run=raw.first_run | {"pos1": 1}, subsets=raw.subsets | seg)
    )
    live = np.array([ingest_raw(c, s, cache, A.jobs, gains=gains["pos1"]).live_time_s for s in seg])
    cut = int(np.argmin(np.abs(np.cumsum(live)[:-1] - live.sum() / 2))) + 1
    parts = {
        "pos1_h0": RunSubset("pos1", (1, cut), 0.0),
        "pos1_h1": RunSubset("pos1", (cut + 1, n_seg), 0.0),
    }
    c = replace(c, raw=replace(c.raw, subsets=c.raw.subsets | parts))
    for i in range(2):
        halves[("pos1", i)] = ingest_raw(c, f"pos1_h{i}", cache, A.jobs, gains=gains["pos1"])
    shape = {
        pid: dict(
            zip(
                ("scatter", "noise"),
                epoch_scatter([halves[(pid, 0)].hist], [halves[(pid, 1)].hist]),
                strict=True,
            )
        )
        for pid in ("pos0", "pos1")
    }
    split = {
        "pos0_files": [len(files) // 2, len(files) - len(files) // 2],
        "pos1_segments": [cut, n_seg - cut],
        "pos1_segment_live_s": live.tolist(),
    }
    for i in range(2):
        out = A.out / "ingest" / f"half{i}"
        if not out.exists():
            shutil.copytree(A.baseline / "ingest", out)
            meta = json.loads((out / "meta.json").read_text())
            for pid in ("pos0", "pos1"):
                h = halves[(pid, i)]
                save_counts(h.hist, out, pid, {})
                meta["exposures"][pid]["live_time_s"] = h.live_time_s
                meta["exposures"][pid]["total_counts"] = int(h.hist.values.sum())
                meta["exposures"][pid]["source"] = f"half {i} of the room runs"
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"shape": shape, "split": split}


def task(spec):
    name, kind, arg = spec
    path = A.out / "variants" / f"{name}.json"
    if path.exists():
        return name
    cfg = nominal_cfg()
    maps, sigma, data = nominal_inputs(cfg)
    sky = maps.sky
    if kind == "nominal":
        doc = evaluate(data, cfg, sky, name)
    elif kind == "sky":
        doc = evaluate(data_from(cfg, A.skyswap / arg, sigma, data.rows), cfg, sky, name)
    elif kind == "half":
        doc = evaluate(
            data_from(cfg, A.out / "ingest" / f"half{arg}", sigma, data.rows), cfg, sky, name
        )
    elif kind == "recon":
        section, key, factor = arg
        value = (
            factor
            if key in ("spacing_m", "n_iter")
            else getattr(getattr(cfg, section), key) * factor
        )
        doc = evaluate(data, nominal_cfg(set=[(section, key, value)]), sky, name)
        doc["value"] = value
    elif kind == "array":
        doc = array_variant(cfg, data, sigma, sky, arg)
    elif kind == "control":
        doc = control(cfg, data, sky, *arg)
    else:
        raise ValueError(kind)
    path.write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"done {name}", flush=True)
    return name


def nominal_beams(cfg, data, sky):
    return find_beams(data, cfg, sky)


def array_variant(cfg, data, sigma, sky, arg):
    kind, value = arg
    beams = nominal_beams(cfg, data, sky)
    if kind == "completeness":
        arr = array_fit(data, cfg, beams, per_end=value)
    elif kind == "centres":
        unc = json.loads((A.results / "uncertainty.json").read_text())
        arr = array_fit(
            data, cfg, beams, scale=1 + value * unc["beams_pitch_sigma"] / beams["pitch"]
        )
    elif kind == "background":
        c = replace(cfg, beamdepth=replace(cfg.beamdepth, bg_degree=value))
        arr = array_fit(data, c, beams)
    elif kind == "pose":
        axis, sign = value
        pose = json.loads((A.results / "pose.json").read_text())
        free = cfg.selfcal.free_pose
        p = cfg.exposure(free).pose
        step = {"x": pose["x_sigma"], "y": pose["y_sigma"], "az_deg": pose["az_sigma"]}[axis]
        c = cfg.with_pose(free, replace(p, **{axis: getattr(p, axis) + sign * step}))
        d = data_from(c, A.baseline / "ingest", sigma, data.rows)
        arr = array_fit(d, c, find_beams(d, c, sky))
    else:
        raise ValueError(kind)
    return {k: arr[k] for k in (*DEPTH, "chisq_per_dof", "dof")} | {"array": arr}


def control(cfg, data, sky, case, replica):
    """Array fit of simulated band opacities (beam model + nominal background + noise)."""
    beams = nominal_beams(cfg, data, sky)
    nominal = json.loads((A.results / "arraydepth.json").read_text())
    keep = (data.w > 0) & (np.abs(data.rows.sy) <= cfg.beamdepth.band_sy)
    background = array_model(data, cfg, nominal, height_seed=beams["z"])["background"]
    centre = nominal["bottom"] + nominal["h"] / 2
    h = 1.2 if case in ("null", "slab", "array") else case
    columns = np.array(nominal["normal_columns"])
    if case == "null":
        columns[:] = 0.0
    if case == "array":
        n = len(columns) - len(beams["beams_x"])
        columns[: n // 2] = 0.0
        columns[len(columns) - n // 2 :] = 0.0
    truth = nominal | {"h": h, "bottom": centre - h / 2, "normal_columns": columns.tolist()}
    beam = array_model(data, cfg, truth, height_seed=beams["z"])["beam"]
    sx, sy = data.rows.sx[keep], data.rows.sy[keep]
    lam = beam + background + (SLAB * np.sqrt(1 + sx**2 + sy**2) if case == "slab" else 0.0)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            0, spawn_key=(7, ["null", "slab", "array", *RECOVERY_H].index(case), replica)
        )
    )
    sim = data.lam.copy()
    sim[keep] = lam + rng.normal(size=lam.size) / np.sqrt(data.w[keep])
    fit = fit_array_depth(
        replace(data, lam=sim),
        cfg,
        centres=continued_centres(beams["beams_x"]),
        height_seed=beams["z"],
    )
    return {"case": case, "replica": replica, "true_h": h} | {
        k: fit[k]
        for k in (*DEPTH, "chisq_per_dof", "normal_columns", "lower_bounds", "upper_bounds")
    }


def specs():
    out = [("nominal", "nominal", None)]
    out += [(k, "sky", v) for k, v in SKY_VARIANTS.items()]
    out += [(f"half{i}", "half", i) for i in range(2)]
    out += [(k, "recon", v) for k, v in RECON_VARIANTS.items()]
    out += [(f"completeness_{n}", "array", ("completeness", n)) for n in (1, 3)]
    out += [(f"centres_{s:+d}", "array", ("centres", s)) for s in (-1, 1)]
    out += [(f"background_{d}", "array", ("background", d)) for d in (1, 3)]
    out += [
        (f"pose_{a}_{s:+d}", "array", ("pose", (a, s)))
        for a in ("x", "y", "az_deg")
        for s in (-1, 1)
    ]
    cases = ["null", "slab", "array", *RECOVERY_H]
    out += [(f"control_{c}_{r:02d}", "control", (c, r)) for c in cases for r in range(N_CONTROL)]
    return out


def run(args):
    init(args)
    for d in ("variants", "volumes", "ingest", "cache"):
        (A.out / d).mkdir(parents=True, exist_ok=True)
    if not (A.out / "views.npy").exists():
        shutil.copy(A.baseline / "voxels" / "views.npy", A.out / "views.npy")
    if not (A.out / "halves.json").exists():
        (A.out / "halves.json").write_text(json.dumps(half_ingests(), indent=2) + "\n")
    # the nominal first and alone: it fills the forward-model cache the variants share
    todo = specs()
    task(todo[0])
    with ProcessPoolExecutor(A.jobs, initializer=init, initargs=(args,)) as pool:
        for name in pool.map(task, todo[1:]):
            print(f"finished {name}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["run", "summarize"])
    for k in ("config", "baseline", "results", "skyswap", "out"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.step == "run":
        run(args)
    else:
        from summarize import summarize

        summarize(args.out, args.results)


if __name__ == "__main__":
    main()
