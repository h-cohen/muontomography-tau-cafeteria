"""Shifts, envelopes, budgets and tests of the registered 3D systematics study (PREREG.md)."""

import json
import math
from pathlib import Path

import numpy as np
from run import RECOVERY_H, SLAB

from cafetomo.beamdepth import AT_BOUND_FRACTION

HEIGHTS = ("autofocus_z", "beams_z", "beams_zx", "beams_pitch")
IMAGE = ("zprofile_fwhm", "zprofile_bottom", "zprofile_top")
DEPTH = ("h", "w", "bottom")
FAMILIES = {
    "sky": ("sky_flip", "sky_b_aligned", "sky_m2", "sky_plain_sum"),
    "reconstruction": (
        "alpha_half",
        "alpha_double",
        "damping_half",
        "damping_double",
        "spacing_015",
        "spacing_025",
        "early_stop",
    ),
    "completeness": ("completeness_1", "completeness_3"),
    "centres": ("centres_-1", "centres_+1"),
    "background": ("background_1", "background_3"),
    "pose": tuple(f"pose_{a}_{s}" for a in ("x", "y", "az_deg") for s in ("-1", "+1")),
}
SAME_GRID = ("alpha_half", "alpha_double", "damping_half", "damping_double", "early_stop")
CONTROL_NAMES = {
    "null": "null",
    "slab": "slab",
    "array": "array",
    0.6: "thin",
    1.2: "mid",
    1.8: "thick",
}
NOMINAL_TOL = 1e-6


def key(name: str) -> str:
    """A variant name as a macro-safe key (signs spelled out)."""
    return name.replace("+", "plus").replace("-", "minus")


def stat_sigmas(results: Path) -> dict:
    unc = json.loads((results / "uncertainty.json").read_text())
    af = json.loads((results / "autofocuslayer.json").read_text())
    arr = json.loads((results / "arraydepth.json").read_text())
    return {
        "autofocus_z": af["z_stat"],
        "beams_z": unc["beams_z_stat"],
        "beams_zx": unc["beams_zx_stat"],
        "beams_pitch": unc["beams_pitch_sigma"],
        **{k: arr[f"{k}_sigma"] for k in DEPTH},
    }


def check_nominal(nominal: dict, results: Path) -> None:
    """The recomputed nominal must reproduce the promoted production results."""
    af = json.loads((results / "autofocuslayer.json").read_text())
    beams = json.loads((results / "beams.json").read_text())
    arr = json.loads((results / "arraydepth.json").read_text())
    pairs = {
        "autofocus_z": af["z"],
        "beams_z": beams["z"],
        "beams_zx": beams["z_x"],
        "beams_pitch": beams["pitch"],
        **{k: arr[k] for k in DEPTH},
    }
    off = {k: nominal[k] - v for k, v in pairs.items() if abs(nominal[k] - v) > NOMINAL_TOL}
    if off:
        raise SystemExit(f"recomputed nominal differs from results: {off}")


def at_bound(record: dict, index: int) -> bool:
    lo, hi = record["lower_bounds"][index], record["upper_bounds"][index]
    value = [record["bottom"], record["w"], record["h"]][index]
    return min(value - lo, hi - value) < AT_BOUND_FRACTION * (hi - lo)


def summarize(out: Path, results: Path) -> dict:
    v = {p.stem: json.loads(p.read_text()) for p in (out / "variants").glob("*.json")}
    nominal = v["nominal"]
    check_nominal(nominal, results)
    stat = stat_sigmas(results)
    doc: dict = {}
    for family, names in FAMILIES.items():
        for q in (*HEIGHTS, *IMAGE, *DEPTH):
            shifts = [v[n][q] - nominal[q] for n in names if q in v[n]]
            if shifts:
                for n in names:
                    if q in v[n]:
                        doc[f"{q}_{family}_{key(n)}"] = v[n][q] - nominal[q]
                # an unmeasured variant (NaN) is counted, not taken as a zero shift
                doc[f"{q}_{family}"] = float(np.nanmax(np.abs(shifts)))
                doc[f"{q}_{family}_unmeasured"] = int(np.isnan(shifts).sum())

    halves = json.loads((out / "halves.json").read_text())
    for pid, s in halves["shape"].items():
        doc[f"stability_shape_scatter_{pid}"] = s["scatter"]
        doc[f"stability_shape_noise_{pid}"] = s["noise"]
    doc["stability_pos1_segments_early"], doc["stability_pos1_segments_late"] = halves["split"][
        "pos1_segments"
    ]
    for q in (*HEIGHTS, *IMAGE, *DEPTH):
        delta = v["half1"][q] - v["half0"][q]
        doc[f"{q}_halves_delta"] = delta
        if q in stat:
            ratio = abs(delta) / (2 * stat[q])
            doc[f"{q}_halves_ratio"] = ratio
            doc[f"{q}_stability"] = abs(delta) / 2 if ratio > 2 else 0.0

    nominal_array = json.loads((results / "arraydepth.json").read_text())
    scale = math.sqrt(max(nominal_array["chisq_per_dof"], 1.0))
    doc["depth_scale_factor"] = scale

    unc = json.loads((results / "uncertainty.json").read_text())
    af = json.loads((results / "autofocuslayer.json").read_text())
    previous = {
        "autofocus_z": af["z_total"],
        "beams_z": unc["beams_z_total"],
        "beams_zx": unc["beams_zx_total"],
        "beams_pitch": unc["beams_pitch_sigma"],
    }
    for q, old in previous.items():
        rows = [doc[f"{q}_sky"], doc[f"{q}_reconstruction"], doc[f"{q}_stability"]]
        doc[f"{q}_total"] = math.sqrt(old**2 + sum(r**2 for r in rows))
    for q in DEPTH:
        doc[f"{q}_stat_scaled"] = scale * stat[q]
        rows = [
            doc[f"{q}_{f}"]
            for f in ("centres", "completeness", "pose", "background", "sky", "reconstruction")
        ] + [doc[f"{q}_stability"]]
        doc[f"{q}_total"] = math.sqrt(doc[f"{q}_stat_scaled"] ** 2 + sum(r**2 for r in rows))

    views = np.load(out / "views.npy")
    seen = views >= 2
    ref = np.load(out / "volumes" / "nominal.npy")[seen]
    for n in SAME_GRID:
        vol = np.load(out / "volumes" / f"{n}.npy")[seen]
        doc[f"image_corr_{n}"] = float(np.corrcoef(ref, vol)[0, 1])
    doc["image_corr_min"] = min(doc[f"image_corr_{n}"] for n in SAME_GRID)

    controls = [x for k, x in v.items() if k.startswith("control_")]
    for case, label in CONTROL_NAMES.items():
        fits = [c for c in controls if c["case"] == case]
        h = np.array([c["h"] for c in fits])
        doc[f"control_{label}_n"] = len(fits)
        doc[f"control_{label}_h_mean"] = float(h.mean())
        doc[f"control_{label}_h_spread"] = float(h.std(ddof=1))
        doc[f"control_{label}_h_bias"] = float(h.mean() - fits[0]["true_h"])
        doc[f"control_{label}_depth_at_bound"] = sum(at_bound(c, 2) for c in fits)
        doc[f"control_{label}_column_mean"] = float(
            np.mean([np.mean(c["normal_columns"]) for c in fits])
        )
    doc["nominal_column_mean"] = float(np.mean(nominal_array["normal_columns"]))
    doc["control_null_column_ratio"] = doc["control_null_column_mean"] / doc["nominal_column_mean"]
    recovery = [CONTROL_NAMES[h] for h in RECOVERY_H]
    for label in recovery:
        doc[f"control_{label}_true_h"] = next(
            c["true_h"] for c in controls if CONTROL_NAMES[c["case"]] == label
        )
    doc["control_recovery_max_abs_bias"] = max(abs(doc[f"control_{x}_h_bias"]) for x in recovery)
    doc["control_recovery_max_spread"] = max(doc[f"control_{x}_h_spread"] for x in recovery)
    doc["control_slab_opacity"] = SLAB
    doc["reconstruction_spacing_fine"] = v["spacing_015"]["value"]
    doc["reconstruction_spacing_coarse"] = v["spacing_025"]["value"]
    # descriptive only, not a budget entry: the sky envelope without the half-turned roof A,
    # under which the beam centres are not triangulated
    for q in (*HEIGHTS, *DEPTH):
        rest = [v[n][q] - nominal[q] for n in FAMILIES["sky"] if n != "sky_flip"]
        doc[f"{q}_sky_without_flip"] = float(np.max(np.abs(rest)))
    (out / "summary.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc
