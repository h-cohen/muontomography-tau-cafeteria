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
        "paper/main.tex": "\\bibliography{refs}",
        "paper/sections/intro.tex": "text",
        "paper/refs.bib": "bibliography source",
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
