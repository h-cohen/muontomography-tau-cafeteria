import subprocess
from pathlib import Path

import pytest

FIGS = [
    "setup",
    "opacity",
    "backprojection",
    "autofocus",
    "triangulation",
    "depth",
    "volume",
    "uncertainty",
]


@pytest.mark.slow
@pytest.mark.parametrize("name", FIGS)
def test_figure_builds_from_fast_run(name, tmp_path):
    if not Path("runs/fast/results/uncertainty.json").exists():
        pytest.skip("run `make uncertainty validation FAST=1` first")
    out = tmp_path / f"{name}.pdf"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            f"paper/figures/make_{name}.py",
            "--config",
            "configs/cafeteria.yaml",
            "--pose",
            "runs/fast/results/pose.json",
            "--results",
            "runs/fast/results",
            "--runs",
            "runs/fast",
            "--out",
            str(out),
        ],
        check=True,
    )
    assert out.stat().st_size > 1000
