import json
import subprocess

import pytest

from cafetomo.cli import main


def test_ingest_stage_writes_data_json(tmp_path):
    subprocess.run(["uv", "run", "cafetomo", "ingest", "--config", "configs/cafeteria.yaml",
                    "--out", str(tmp_path / "ingest"), "--results", str(tmp_path / "res")],
                   check=True)
    doc = json.loads((tmp_path / "res" / "data.json").read_text())
    assert set(doc) == {"first_tracks", "first_hours", "second_tracks", "second_hours",
                        "sky_tracks", "sky_hours"}
    assert doc["sky_tracks"] > 0 and doc["first_hours"] > 0
    meta = json.loads((tmp_path / "ingest" / "meta.json").read_text())
    # provenance joins the per-source record that live_times reads
    assert meta["stage"] == "ingest" and meta["exposures"]["pos0"]["live_time_s"] > 0


def test_unknown_stage_fails():
    r = subprocess.run(["uv", "run", "cafetomo", "nope"], capture_output=True)
    assert r.returncode != 0


def test_uncertainty_without_pose_fails(tmp_path):
    with pytest.raises(ValueError, match="--pose"):
        main(["uncertainty", "--config", "configs/cafeteria.yaml", "--ingest", "x",
              "--opacity", "x", "--voxels", "x", "--results", str(tmp_path), "--out", "x"])
