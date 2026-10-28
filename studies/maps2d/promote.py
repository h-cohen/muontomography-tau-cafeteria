"""Flat paper scalars of the 2D stripe study -> results/production/stripes.json.

The nominal hit selection is the pedestal-based neighbour selection; the flag-only values
are carried for the comparison. Axes are named as the detector does (beams along X).
Usage: promote.py MAPS2D_ROOT OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
docs = {k: json.loads((root / f"maps2d_{k}.json").read_text()) for k in ("pedestal", "flag")}
PIDS = ("pos0", "pos1")


def scalars(sel: str, prefix: str = "") -> dict:
    d = docs[sel]
    n, s = d["nominal"], d["sigma"]
    keys = ["shift", "pair_shift_mean", "transverse_shift"] + [
        f"{p}_{k}"
        for p in PIDS
        for k in ("period", "fwhm", "contrast", "transverse_centre", "transverse_contrast")
    ]
    doc = {}
    for k in keys:
        doc[prefix + k] = n[k]
        doc[prefix + k + "_sigma"] = s[k]
    return doc


def matched_ratio(key: str) -> tuple[float, float]:
    """Mean over resolved stripes present in both selections of pedestal / flag `key`, and
    the spread of that ratio over the stripes divided by sqrt(count)."""
    ratios = []
    for p in PIDS:
        flag = [s for s in docs["flag"]["stripes"][p]["beam"] if s["resolved"]]
        for n in (s for s in docs["pedestal"]["stripes"][p]["beam"] if s["resolved"]):
            near = [f for f in flag if abs(f["centre"] - n["centre"]) < 0.02]
            if near:
                f = min(near, key=lambda f: abs(f["centre"] - n["centre"]))
                ratios.append(n[key] / f[key])
    r = np.asarray(ratios)
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(r.size))


nom = docs["pedestal"]["nominal"]
doc = scalars("pedestal") | scalars("flag", "flag_")
doc |= {
    "n_pairs": nom["n_pairs"],
    "pair_shift_scatter": nom["pair_shift_scatter"],
    "shift_corr": nom["shift_corr"],
    "n_resolved_pos0": nom["pos0_n_resolved"],
    "n_resolved_pos1": nom["pos1_n_resolved"],
    "contrast_snr_pos0": nom["pos0_contrast"] / docs["pedestal"]["sigma"]["pos0_contrast"],
    "contrast_snr_pos1": nom["pos1_contrast"] / docs["pedestal"]["sigma"]["pos1_contrast"],
    "replicas": docs["pedestal"]["n_replicas"],
}
doc["fwhm_ratio"], doc["fwhm_ratio_sigma"] = matched_ratio("fwhm")
doc["contrast_ratio"], doc["contrast_ratio_sigma"] = matched_ratio("contrast")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print(json.dumps(doc, indent=1))
