"""Separately registered deterministic recovery trace; never edits the original pilot."""

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cafetomo.cli import _grid
from cafetomo.config import load_config
from cafetomo.internal_validation import source_hash
from cafetomo.prediction import _digest
from cafetomo.recovery import direct_controls, run_recovery_case, simulate_counts, standard_cases

# Set from the command line: trace output, replayed pilot, baseline ingest and its config.
ROOT = PILOT = INGEST = CONFIG = None


def set_paths(root, pilot, ingest, config):
    # Workers are not forked on every platform, so they receive the paths explicitly.
    global ROOT, PILOT, INGEST, CONFIG
    ROOT, PILOT, INGEST, CONFIG = root, pilot, ingest, config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    if (ROOT / "registration.json").exists():
        raise FileExistsError("registration is immutable")
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PILOT / "manifest.json").read_text())
    summary = json.loads((PILOT / "results/recovery.json").read_text())
    cfg = load_config(CONFIG)
    grid, live = _grid(cfg, INGEST)
    assert _digest(cfg, grid, live, {}) == manifest["input_hash"]
    assert [x["case"]["case_id"] for x in summary["cases"]] == ["nominal_0.3"] * 5 + [
        "nominal_1.8"
    ] * 5
    cases = [
        dict(
            case_id=x["case"]["case_id"],
            replica=i % 5,
            seed=x["seed"],
            original=f"recovery/{x['case']['case_id']}/{i % 5}/outcome.json",
        )
        for i, x in enumerate(summary["cases"])
    ]
    record = dict(
        purpose=(
            "Supplemental trace of the original 10 noisy outcomes and 2 known-geometry "
            "controls; no estimator retuning."
        ),
        original_manifest_sha256=sha(PILOT / "manifest.json"),
        original_recovery_summary_sha256=sha(PILOT / "results/recovery.json"),
        original_input_hash=manifest["input_hash"],
        original_source_hash=manifest["source_hash"],
        source_revision=os.popen("git rev-parse HEAD").read().strip(),
        source_hash=source_hash(),
        config_sha256=sha(CONFIG),
        ingestion_meta_sha256=sha(INGEST / "meta.json"),
        source_order=list(grid.counts),
        live_times_s=live,
        analysis_prior=asdict(cfg),
        cases=cases,
        direct_controls=[
            dict(case_id="nominal_0.3", n_sub=16, quadrature=[16, 32]),
            dict(case_id="nominal_1.8", n_sub=16, quadrature=[16, 32]),
        ],
        comparison_contract=dict(
            recursive_original_keys=(
                "all keys including arrays, values and status, except elapsed runtime_s"
            ),
            numeric_rtol=1e-6,
            numeric_atol=1e-6,
            excluded=["runtime_s (wall-clock only)"],
            new_fields=["artifacts", "voxel_solver", "training.source_order", "training_weights"],
            mismatch_policy="retain all mismatches and failed outcomes; no rerun or retuning",
        ),
        worker_limit=3,
        thread_environment={
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    path = ROOT / "registration.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    path.chmod(0o444)
    print("registered", len(cases), "noisy cases + 2 direct controls", flush=True)


def one(case_record):
    reg = json.loads((ROOT / "registration.json").read_text())
    assert source_hash() == reg["source_hash"], "inference source changed after registration"
    cfg = load_config(CONFIG)
    grid, live = _grid(cfg, INGEST)
    assert _digest(cfg, grid, live, {}) == reg["original_input_hash"]
    assert list(grid.counts) == reg["source_order"]
    case = next(c for c in standard_cases() if c.case_id == case_record["case_id"])
    out = ROOT / "recovery" / case.case_id / str(case_record["replica"])
    out.mkdir(parents=True, exist_ok=False)
    synthetic = simulate_counts(case, cfg, grid, live, seed=case_record["seed"], n_sub=16)
    np.savez_compressed(out / "simulated-counts.npz", edges=synthetic.edges, **synthetic.counts)
    (out / "input.json").write_text(
        json.dumps(
            dict(
                case=asdict(case),
                seed=case_record["seed"],
                truth_subrays=16,
                source_order=list(synthetic.counts),
                live_times_s=live,
                analysis_prior=asdict(cfg),
                original_input_hash=reg["original_input_hash"],
            ),
            indent=2,
            default=str,
        )
        + "\n"
    )
    result = run_recovery_case(case, synthetic, live, cfg, seed=case_record["seed"], out_dir=out)
    return case.case_id, case_record["replica"], result["status"]


def controls():
    reg = json.loads((ROOT / "registration.json").read_text())
    assert source_hash() == reg["source_hash"]
    cfg = load_config(CONFIG)
    out = ROOT / "recovery" / "direct"
    out.mkdir(parents=True, exist_ok=False)
    results = direct_controls(cfg, out_dir=out, n_sub=16)
    return "direct", len(results), "complete"


def run():
    reg = json.loads((ROOT / "registration.json").read_text())
    assert source_hash() == reg["source_hash"]
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    jobs = [*reg["cases"], None]
    with ProcessPoolExecutor(
        max_workers=3, initializer=set_paths, initargs=(ROOT, PILOT, INGEST, CONFIG)
    ) as pool:
        pending = {
            pool.submit(controls if job is None else one, *([] if job is None else [job])): job
            for job in jobs
        }
        for future in as_completed(pending):
            try:
                print("completed", future.result(), flush=True)
            except Exception as exc:
                print("FAILED", pending[future], repr(exc), flush=True)
                raise


def compare_value(old, new, path, mismatches):
    if path.endswith(".runtime_s"):
        return
    if isinstance(old, dict):
        if not isinstance(new, dict):
            mismatches.append(dict(path=path, old=old, new=new))
            return
        for key, value in old.items():
            if key not in new:
                mismatches.append(dict(path=f"{path}.{key}", old=value, new="MISSING"))
            else:
                compare_value(value, new[key], f"{path}.{key}", mismatches)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            mismatches.append(dict(path=path, old=old, new=new))
            return
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            compare_value(a, b, f"{path}[{i}]", mismatches)
    elif isinstance(old, (float, int)) and not isinstance(old, bool):
        if (
            not isinstance(new, (float, int))
            or isinstance(new, bool)
            or not math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-6)
        ):
            mismatches.append(dict(path=path, old=old, new=new))
    elif old != new:
        mismatches.append(dict(path=path, old=old, new=new))


def compare():
    reg = json.loads((ROOT / "registration.json").read_text())
    original = json.loads((PILOT / "results/recovery.json").read_text())
    assert sha(PILOT / "manifest.json") == reg["original_manifest_sha256"]
    assert sha(PILOT / "results/recovery.json") == reg["original_recovery_summary_sha256"]
    report = []
    for item, old in zip(reg["cases"], original["cases"], strict=True):
        path = ROOT / item["original"]
        assert path.exists(), f"missing {path}"
        new = json.loads(path.read_text())
        mismatches = []
        compare_value(old, new, "case", mismatches)
        report.append(
            dict(
                case_id=item["case_id"],
                replica=item["replica"],
                seed=item["seed"],
                status=new["status"],
                compared_original_keys=len(old),
                mismatches=mismatches,
            )
        )
    new_controls = json.loads((ROOT / "recovery/direct/direct-controls.json").read_text())
    for old, new in zip(original["direct_controls"], new_controls, strict=True):
        mismatches = []
        compare_value(old, new, "direct_control", mismatches)
        report.append(
            dict(
                case_id=old["case"]["case_id"],
                control="direct",
                compared_original_keys=len(old),
                mismatches=mismatches,
            )
        )
    doc = dict(
        registration_sha256=sha(ROOT / "registration.json"),
        outcomes=report,
        compared=12,
        mismatch_count=sum(len(x["mismatches"]) for x in report),
    )
    (ROOT / "comparison.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("compared", doc["compared"], "outcomes;", doc["mismatch_count"], "mismatches", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["register", "run", "compare"])
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ingest", required=True, type=Path)
    ap.add_argument("--config", default=Path("configs/cafeteria.yaml"), type=Path)
    a = ap.parse_args()
    ROOT, PILOT, INGEST, CONFIG = a.out.resolve(), a.pilot, a.ingest, a.config
    {"register": register, "run": run, "compare": compare}[a.step]()
