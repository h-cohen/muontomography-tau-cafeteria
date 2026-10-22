"""Corrected-weight comparison on the internal-validation pilot.

Refits the pilot's weight choices with the corrected opacity weights (zero
weight for exactly constant replicas) at the pilot's frozen pose and settings,
and compares the density and beam-excess persistence with the pilot's
original weights. Writes registration.json, comparison.json and choices/ into
a fresh output directory; the pilot is never modified.

    uv run python studies/weight-correction/compare.py --pilot DIR --out DIR
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from cafetomo.angular import AnalysisGrid
from cafetomo.config import load_config
from cafetomo.robustness import run_robustness

SETTINGS = Path(__file__).with_name("settings.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists: use a fresh output directory")
    a.out.mkdir(parents=True)
    pilot = a.pilot
    settings = json.loads(SETTINGS.read_text())
    stability = json.loads((pilot / "results/stability.json").read_text())
    manifest = json.loads((pilot / "manifest.json").read_text())
    counts = pilot / "folds/count-0/training-counts.npz"
    registration = {
        **settings,
        "original_root": str(pilot),
        "original_manifest_sha256": sha(pilot / "manifest.json"),
        "original_source_hash": manifest["source_hash"],
        "training_counts": str(counts),
        "training_sha256": sha(counts),
        "frozen_pose": stability["choices"]["fitted_pose"],
        "seed": stability["choices"]["seed"],
    }
    (a.out / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    (a.out / "comparison-config.yaml").write_text(yaml.safe_dump(registration["cfg"]))
    (a.out / "nuisance-config.yaml").write_text(
        yaml.safe_dump(stability["choices"]["nuisance_prior"])
    )
    cfg = load_config(a.out / "comparison-config.yaml")
    prior = load_config(a.out / "nuisance-config.yaml")
    d = np.load(counts)
    grid = AnalysisGrid(d["edges"], {k: d[k] for k in d.files if k != "edges"})
    live = json.loads((pilot / "folds/count-0/fold.json").read_text())["train_live"]
    result = run_robustness(
        grid,
        live,
        cfg,
        registration["choices"],
        seed=registration["seed"],
        out_dir=a.out / "choices",
        nuisance_cfg=prior,
    )
    comparison = {
        "original_weights": [
            v for v in stability["choices"]["variants"] if v["choice"]["group"] == "weights"
        ],
        "corrected_weights": result["variants"],
        "frozen_pose_reproduced": result["fitted_pose"] == registration["frozen_pose"],
        "baseline_weight_hash_reproduced": result["weight_hash"]
        == stability["choices"]["weight_hash"],
    }
    for group in ("weights", "beam-excess-weights"):
        old = np.load(pilot / "stability/choices" / f"{group}.npz")
        new = np.load(a.out / "choices" / f"{group}.npz")
        common = (old["exposure"] > 0) & (new["exposure"] > 0)
        delta = new["frequency"][common] - old["frequency"][common]
        comparison[group] = {
            "common_voxels": int(common.sum()),
            "changed_frequency_voxels": int(np.count_nonzero(delta)),
            "changed_frequency_voxels_above_1e-12": int((abs(delta) > 1e-12).sum()),
            "mean_absolute_frequency_change": float(np.mean(abs(delta))),
            "max_absolute_frequency_change": float(np.max(abs(delta))),
        }
    (a.out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    if not (comparison["frozen_pose_reproduced"] and comparison["baseline_weight_hash_reproduced"]):
        raise SystemExit("the pilot's frozen pose or baseline weights were not reproduced")


if __name__ == "__main__":
    main()
