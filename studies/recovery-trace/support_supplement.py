"""Separate reconstruction of original support diagnostics; no pilot edits."""

import hashlib
import json
from pathlib import Path

import numpy as np

from cafetomo.angular import AnalysisGrid
from cafetomo.config import Pose, load_config
from cafetomo.fitdata import FitData, RowIndex
from cafetomo.opacity import solve_opacity
from cafetomo.support import feature_mask_diagnostics

# Set from the command line: the supplement output and the pilot it reconstructs.
ROOT = PILOT = None
CONFIG = Path("configs/cafeteria.yaml")
FOLDS = [f"count-{i}" for i in range(3)] + [f"angular-{i}" for i in range(5)]
NAMES = [f"support-degree{d}-threshold{t}.npz" for d in (1, 2) for t in (2, 3, 4)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def register():
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "registration.json"
    if target.exists():
        raise FileExistsError(target)
    archive = {
        f"{fold}/{name}": sha(PILOT / "folds" / fold / name) for fold in FOLDS for name in NAMES
    }
    doc = dict(
        original_support_archive_sha256=archive,
        original_manifest_sha256=sha(PILOT / "manifest.json"),
        original_config_sha256=sha(CONFIG),
        scope=(
            "48 original support selections; reconstruct only omitted polynomial "
            "diagnostics, verify every original mask exactly"
        ),
        numerical_inputs=(
            "original training counts, live times, fitted pose, archived row weights "
            "and tangent coordinates"
        ),
        original_archives_immutable=True,
    )
    target.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    target.chmod(0o444)


def run():
    reg = json.loads((ROOT / "registration.json").read_text())
    assert sha(PILOT / "manifest.json") == reg["original_manifest_sha256"]
    assert sha(CONFIG) == reg["original_config_sha256"]
    cfg = load_config(CONFIG)
    records = []
    for fold in FOLDS:
        folder = PILOT / "folds" / fold
        meta = json.loads((folder / "fold.json").read_text())
        cfg_fit = cfg.with_pose(cfg.selfcal.free_pose, Pose(**meta["training"]["pose"]))
        with np.load(folder / "training-counts.npz") as d:
            grid = AnalysisGrid(d["edges"], {p: d[p] for p in ("pos0", "pos1", "SKY")})
        with np.load(folder / "training-weights.npz") as d:
            w, sx, sy, pos = [d[k] for k in ("w", "sx", "sy", "pos_of_row")]
        assert (
            hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()
            == meta["training"]["weight_hash"]
        )
        mask_file = folder / "masks.npz"
        if mask_file.exists():
            with np.load(mask_file) as archived_masks:
                masks = {pid: archived_masks[pid] for pid in cfg.position_ids}
        else:
            masks = None
        maps = solve_opacity(grid, cfg_fit, meta["train_live"], observation_mask=masks)
        flat, valid = maps.sky.bin_index(sx, sy)
        assert valid.all()
        centers = maps.sky.centers
        np.testing.assert_allclose(centers[flat // maps.sky.n_bins], sx, atol=1e-12)
        np.testing.assert_allclose(centers[flat % maps.sky.n_bins], sy, atol=1e-12)
        rows = RowIndex(cfg.position_ids, pos, sx, sy, flat)
        lam = np.array(
            [maps.lam[cfg.position_ids[int(k)]][int(j)] for k, j in zip(pos, flat, strict=True)]
        )
        data = FitData(np.where(np.isfinite(lam), lam, 0.0), w, rows)
        fold_out = ROOT / fold
        fold_out.mkdir(parents=True, exist_ok=True)
        for degree in (1, 2):
            for threshold in (2, 3, 4):
                name = f"support-degree{degree}-threshold{threshold}.npz"
                original = folder / name
                assert sha(original) == reg["original_support_archive_sha256"][f"{fold}/{name}"]
                diagnostic = feature_mask_diagnostics(
                    data, background_degree=degree, threshold=threshold
                )
                with np.load(original) as old:
                    mask_equal = {
                        pid: bool(np.array_equal(diagnostic.masks[pid], old[pid]))
                        for pid in cfg.position_ids
                    }
                np.savez_compressed(
                    fold_out / name,
                    powers=np.asarray(diagnostic.powers, dtype=np.int64),
                    **{f"mask_{p}": x for p, x in diagnostic.masks.items()},
                    **{f"coefficients_{p}": x for p, x in diagnostic.coefficients.items()},
                    **{f"excluded_{p}": x for p, x in diagnostic.excluded.items()},
                    **{f"eligible_{p}": x for p, x in diagnostic.eligible.items()},
                )
                records.append(
                    dict(
                        fold=fold,
                        setting=name,
                        mask_equal=mask_equal,
                        original_sha256=sha(original),
                        supplemental_sha256=sha(fold_out / name),
                    )
                )
        print("support", fold, flush=True)
    report = dict(
        comparisons=len(records),
        all_masks_equal=all(all(x["mask_equal"].values()) for x in records),
        records=records,
    )
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "support comparisons",
        len(records),
        "all masks equal",
        report["all_masks_equal"],
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["register", "run"])
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    ROOT, PILOT = a.out.resolve(), a.pilot
    {"register": register, "run": run}[a.step]()
