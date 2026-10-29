"""Exercise rebuild scheduling without running the expensive scientific stages."""

import os
import subprocess
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def built_pipeline(tmp_path):
    (tmp_path / "Makefile").write_text((ROOT / "Makefile").read_text())
    inputs = [
        "configs/cafeteria.yaml",
        "configs/paper-inputs.json",
        "src/cafetomo/beamdepth.py",
        "pyproject.toml",
        "uv.lock",
        "viewer/shell.html",
    ]
    outputs = [
        "runs/fast/ingest/meta.json",
        "runs/fast/opacity/meta.json",
        "runs/fast/voxels/meta.json",
        *[
            f"runs/fast/results/{name}.json"
            for name in (
                "data",
                "pose",
                "reconstruction",
                "autofocus",
                "beams",
                "beamdepth",
                "depthdiagnostics",
                "arraydepth",
                "validation",
                "uncertainty",
                "inputs",
            )
        ],
        "runs/fast/generated/numbers.tex",
    ]
    now = time.time()
    for names, stamp in ((inputs, now - 20), (outputs, now - 10)):
        for name in names:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            os.utime(path, (stamp, stamp))
    return tmp_path


def _schedule(root, target):
    return subprocess.run(
        ["make", "--no-print-directory", "-n", "FAST=1", target],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.mark.parametrize("changed", ["src/cafetomo/beamdepth.py", "uv.lock"])
def test_code_or_dependency_change_rebuilds_analysis_and_uncertainty(built_pipeline, changed):
    assert "cafetomo" not in _schedule(built_pipeline, "uncertainty")
    (built_pipeline / changed).touch()
    scheduled = _schedule(built_pipeline, "uncertainty")
    assert "cafetomo --fast analyze " in scheduled
    assert "cafetomo --fast uncertainty " in scheduled


@pytest.mark.parametrize(
    "missing,stage",
    [
        ("data", "ingest"),
        ("reconstruction", "reconstruct"),
        ("beams", "analyze"),
        ("autofocus", "analyze"),
        ("depthdiagnostics", "depthcheck"),
    ],
)
def test_missing_result_is_regenerated_before_numbers(built_pipeline, missing, stage):
    (built_pipeline / f"runs/fast/results/{missing}.json").unlink()
    scheduled = _schedule(built_pipeline, "runs/fast/generated/numbers.tex")
    assert f"cafetomo --fast {stage} " in scheduled
    assert "cafetomo --fast numbers " in scheduled


def test_paper_inputs_are_copied_into_fast_results_without_touching_full_results(built_pipeline):
    source = built_pipeline / "configs/paper-inputs.json"
    source.write_text('{"measured_beam_depth": 1.2}')
    subprocess.run(
        ["make", "--no-print-directory", "FAST=1", "runs/fast/results/inputs.json"],
        cwd=built_pipeline,
        check=True,
        capture_output=True,
    )
    assert (built_pipeline / "runs/fast/results/inputs.json").read_text() == source.read_text()
    assert not (built_pipeline / "results").exists()


def test_arxiv_archive_places_bibliography_beside_main_tex(built_pipeline):
    files = {
        "paper-draft/main.tex": "\\bibliography{refs}",
        "paper-draft/sections/intro.tex": "text",
        "paper-draft/refs.bib": "bibliography source",
        "runs/fast/generated/main.bbl": "formatted bibliography",
        "runs/fast/generated/paper.pdf": "compiled draft",
    }
    for name in (
        "setup",
        "opacity",
        "height",
        "depth",
    ):
        files[f"runs/fast/generated/figures/{name}.pdf"] = "figure"
    for name, content in files.items():
        path = built_pipeline / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(
        ["make", "--no-print-directory", "FAST=1", "-o", "runs/fast/generated/paper.pdf", "arxiv"],
        cwd=built_pipeline,
        check=True,
        capture_output=True,
    )
    with tarfile.open(built_pipeline / "runs/fast/generated/arxiv.tar.gz") as archive:
        assert archive.extractfile("main.bbl").read() == b"formatted bibliography"
        assert "generated/numbers.tex" in archive.getnames()
        assert "generated/figures/depth.pdf" in archive.getnames()


def test_fast_all_only_schedules_viewer_and_explicit_paper_fails_helpfully(built_pipeline):
    scheduled = _schedule(built_pipeline, "all")
    assert "make paper" not in scheduled
    assert "internal-paper" not in scheduled
    result = subprocess.run(
        ["make", "--no-print-directory", "FAST=1", "paper"],
        cwd=built_pipeline,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FAST=1 only supports analysis/viewer smoke" in result.stderr
    (built_pipeline / "runs/fast/generated/paper.pdf").write_text("precompiled")
    result = subprocess.run(
        ["make", "--no-print-directory", "FAST=1", "paper"],
        cwd=built_pipeline,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "FAST=1 only supports analysis/viewer smoke" in result.stderr


def test_normal_paper_schedules_reviewed_input_preflight():
    scheduled = subprocess.run(
        ["make", "--no-print-directory", "-n", "paper"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--verify-selection" in scheduled
    assert scheduled.index("--verify-selection") < scheduled.index("cp runs/figure-review/")


def test_conditional_array_study_is_generated_before_paper_numbers(built_pipeline):
    (built_pipeline / "runs/fast/results/arraydepth.json").unlink()
    scheduled = _schedule(built_pipeline, "runs/fast/generated/numbers.tex")
    assert "python -m cafetomo.arraydepth --fast" in scheduled
    assert "--out runs/fast/results/arraydepth.json" in scheduled
    assert scheduled.index("python -m cafetomo.arraydepth") < scheduled.index(
        "cafetomo --fast numbers"
    )


def test_production_push_requires_check_and_sync_is_staged_from_baseline():
    out = subprocess.run(
        ["make", "--no-print-directory", "-n", "production-push"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.index("production-check") < out.index("push origin HEAD:main")


def test_production_check_skipped_without_submodule(tmp_path):
    out = subprocess.run(
        ["make", "--no-print-directory", "-n", "test", f"PRODUCTION={tmp_path / 'absent'}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "absent/generated/MANIFEST.json" in out  # guarded by test -f, so skipped at run time


def _dry(*args):
    return subprocess.run(
        ["make", "--no-print-directory", "-n", *args], capture_output=True, text=True, check=True
    ).stdout


def test_default_targets_keep_the_reviewed_archives():
    out = _dry("production-stage")
    assert "runs/internal-validation/pilot-2026-10-13-a" in out
    assert "runs/figure-review/selected-paper-publication-2026-10-18-fix3" in out
    assert "--runs runs/joint3d-2026-10-29/baseline" in out
    assert "numbers --results results/production/3d" in out
    assert "cp runs/figure-review/joint3d/internal-numbers.tex" in out
    assert "--results results/production/3d --runs runs/joint3d-2026-10-29/baseline" in out


def test_joint3d_internal_validates_the_joint_sky_run(tmp_path):
    out = _dry("joint3d-internal", f"RESULTS={tmp_path}/res")
    assert "--ingest runs/joint3d-2026-10-29/baseline/ingest" in out
    assert "--out runs/internal-validation/pilot-joint3d" in out
    assert f"--summary {tmp_path}/res/production/3d/internalvalidation.json" in out
    assert f"--baseline-results {tmp_path}/res/production/3d" in out


def test_joint3d_seeds_from_the_joint_sky_and_promotes_into_production(tmp_path):
    j = tmp_path / "j"
    out = _dry("joint3d", f"JOINT3DRUNS={j}", f"RESULTS={tmp_path}/res")
    assert f"seed.py configs/cafeteria.yaml runs/skyswap/joint/ingest_m3_phi0 {j}" in out
    assert f"RESULTS={j}/results" in out
    assert f"{j}/autofocus-study/autofocuslayer.json {tmp_path}/res/production/3d/" in out
    assert f"cp configs/paper-inputs.json {tmp_path}/res/production/3d/inputs.json" in out


def test_reproduce_writes_only_under_its_root(tmp_path):
    root = tmp_path / "repro"
    out = _dry("reproduce", f"REPRO={root}")
    for stage in (
        "internal-validate",
        "joint3d",
        "joint3d-internal",
        "make_internal_paper.py",
        "rawcheck",
        "weight-correction",
        "recovery-trace",
        "maps2d.py",
        "make_stripes.py",
        "production-stage",
    ):
        assert stage in out, stage
    for reviewed in (
        "pilot-2026-10-13-a",
        "weight-correction-2026-10-14",
        "response-aware-2026-10-07",
        "-fix3",
        "convergence-2026-10-21",
        "autofocus-study-2026-10-21",
        "--results results ",
        "--out results/",
        "paper-draft/generated",
    ):
        assert reviewed not in out, reviewed
    assert f"test ! -e {root}" in out


def test_tools_refuses_a_download_with_the_wrong_checksum(tmp_path):
    fake = tmp_path / "tectonic.tar.gz"
    fake.write_bytes(b"not tectonic")
    done = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "tools",
            f"TOOLS={tmp_path / 'tools'}",
            f"TECTONIC_URL=file://{fake}",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode != 0
    assert not (tmp_path / "tools" / "tectonic").exists()


def test_tools_pins_the_release_checksum():
    out = _dry("tools", "TOOLS=/tmp/claude-1000/never")
    assert "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7" in out
    assert "sha256sum -c" in out


def test_draft_build_regenerates_every_scripted_figure_and_finds_its_macros(tmp_path):
    root = tmp_path / "repro"
    out = _dry("internal-paper", f"GEN={root}/draft/generated", "RESULTS=results")
    assert "for f in height setup opacity; do" in out
    assert f"--out {root}/draft/generated/figures/$f.pdf" in out
    assert f"-Z search-path={root}/draft" in out


def test_stripes_builds_the_flag_only_reference_from_the_shared_sky_cache(tmp_path):
    out = _dry("stripes", f"STRIPERUNS={tmp_path}/s", f"SKYRUNS={tmp_path}/sky")
    assert f"--cache {tmp_path}/sky/cache" in out
    assert "--neighbour-sigmas 0" in out
    assert f"pedestal={tmp_path}/sky/joint/ingest_m3_phi0" in out
    assert f"flag={tmp_path}/s/skyswap-flag/joint/ingest_m3_phi0" in out
    assert "--pose results/pose.json" in out
    assert "flag-results/pose.json" not in out


def test_raw_macros_come_from_the_promoted_rawdata(tmp_path):
    out = _dry("rawdata-promote", f"RAWRUNS={tmp_path}/raw", f"RESULTS={tmp_path}/res")
    promoted = f"{tmp_path}/res/production/raw/rawdata.json"
    assert f"cp {tmp_path}/raw/results/rawdata.json {promoted}" in out
    out = _dry("reproduce-stages", f"REPRO={tmp_path}/r")
    assert f"{tmp_path}/r/results/production/raw/rawdata.json" in out
