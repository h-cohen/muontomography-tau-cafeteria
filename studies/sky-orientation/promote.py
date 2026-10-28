"""Flat paper scalars of the joint open-sky study -> results/production/skyjoint.json.

Inputs: the sky-swap root (skyswap.json, ingest metas) and the joint root (joint_sky.json,
compare.json). Shifts are maxima over the room positions; the nominal fit is m <= 3.
Usage: promote.py SKYSWAP_ROOT JOINT_ROOT OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

swap, joint, out = (Path(a) for a in sys.argv[1:4])
fit = json.loads((joint / "joint_sky.json").read_text())
nominal = fit["fits"]["m3"]
cmp = json.loads((joint / "compare.json").read_text())
before = json.loads((swap / "skyswap.json").read_text())
pids = [k for k in cmp]


def tracks(sky: str) -> int:
    meta = json.loads((swap / f"ingest_{sky}" / "meta.json").read_text())
    return meta["exposures"]["SKY"]["total_counts"]


def split_width(key: str) -> float:
    """Standard deviation of the roof-A share pulls per view bin (split.npz)."""
    v = np.load(joint / "split.npz")[key]
    return float(np.std(v[np.isfinite(v)]))


def worst(name: str, key: str) -> float:
    return max(abs(cmp[p][name][key]) for p in pids)


doc = {
    "a_hours": fit["live_h"]["SKY_A"],
    "b_hours": fit["live_h"]["SKY_B"],
    "hours": fit["live_h"]["SKY_A"] + fit["live_h"]["SKY_B"],
    "a_tracks": tracks("SKY_A"),
    "b_tracks": tracks("SKY_B"),
    "turn_deg": abs(fit["turn_deg"]),
    "harmonics": 3,
    "parameters": len(nominal["coefficients"]),
    "pull_width_before": max(before[p]["pull_std"] for p in pids),
    "chi2_per_dof": nominal["chi2_per_dof"],
    "dof": nominal["dof"],
    "rate_ratio": nominal["c"],
    "rate_ratio_sigma": nominal["c_err"],
    "flip_max": worst("orientation_phi180", "max_abs_smoothed"),
    "flip_mean": worst("orientation_phi180", "mean"),
    "b_aligned_max": max(
        worst("orientation_phiB", "max_abs_smoothed"),
        worst("orientation_phiB180", "max_abs_smoothed"),
    ),
    "model_max": worst("model_m2", "max_abs_smoothed"),
    "whole_run_max": worst("single_whole_run", "max_abs_smoothed"),
    "pull_width_split_uniform": split_width("pull_uniform"),
    "pull_width_split": split_width("pull"),
    "modulation_tau": nominal["modulation"]["tau"],
    "modulation_peak_pct": nominal["modulation"]["peak_to_peak_pct"],
    "modulation_peak_pct_sigma": nominal["modulation"]["peak_to_peak_pct_err"],
    "dipole_pct": nominal["modulation"]["m1_pct"],
    "dipole_pct_sigma": nominal["modulation"]["m1_pct_err"],
    "quadrupole_pct": nominal["modulation"]["m2_pct"],
    "quadrupole_pct_sigma": nominal["modulation"]["m2_pct_err"],
    "sigma_pos0": cmp["pos0"]["median_sigma_nominal"],
    "sigma_pos1": cmp["pos1"]["median_sigma_nominal"],
}
out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print(json.dumps(doc, indent=1))
