import json
import os

import pytest

from cafetomo.config import Pose, load_config

CONFIG = "configs/cafeteria.yaml"


def test_loads_campaign(cfg):
    assert cfg.position_ids == ("pos0", "pos1")
    assert cfg.detector.active_width_cm == 35.375
    assert cfg.sky_reference.id == "SKY"
    assert cfg.volume.viewer_crop_xy_m == ((-5.0, 7.0), (-5.0, 5.0))
    assert cfg.autofocus.spacing_m == 0.1


def test_data_dir_relative_to_config(tmp_path, cfg):
    here = os.getcwd()
    os.chdir(tmp_path)
    try:
        c = load_config(os.path.join(here, CONFIG))
    finally:
        os.chdir(here)
    assert (c.data_dir / "HistsOutSkyRoofRuns37-77.root").is_file()


def test_pose_file_overrides_free_pose(tmp_path):
    p = tmp_path / "pose.json"
    p.write_text(
        json.dumps({"free_pose": "pos1", "pose": {"x": 1.8, "y": 0.7, "z": 0.0, "az_deg": 0.1}})
    )
    c = load_config(CONFIG, pose_file=p)
    assert c.exposure("pos1").pose == Pose(1.8, 0.7, 0.0, 0.1)
    assert c.origins()["pos1"] == (1.8, 0.7, 0.0)


def test_unknown_key_raises(tmp_path):
    text = open(CONFIG).read().replace("tv_alpha: 0.03", "tv_alpha: 0.03\n  tv_alfa: 1")
    bad = tmp_path / "c.yaml"
    bad.write_text(text.replace("../data", os.path.abspath("data")))
    with pytest.raises(ValueError, match="tv_alfa"):
        load_config(bad)


def test_fast_shrinks_work(cfg):
    f = cfg.fast()
    assert f.uncertainty.n_replicas < cfg.uncertainty.n_replicas
    assert f.reconstruction.n_iter < cfg.reconstruction.n_iter
    assert f.volume.spacing_m > cfg.volume.spacing_m


def test_with_pose_unknown_raises(cfg):
    with pytest.raises(KeyError, match="nope"):
        cfg.with_pose("nope", Pose(0, 0, 0))


def test_physics_section(cfg):
    p = cfg.physics
    assert (p.concrete_density_gcm3, p.concrete_density_sigma) == (2.4, 0.1)
    assert p.flux_model != p.flux_model_alt
    assert cfg.fast().physics == p


def test_unknown_flux_model_raises(tmp_path):
    text = open(CONFIG).read().replace("flux_model: guan", "flux_model: nope")
    bad = tmp_path / "c.yaml"
    bad.write_text(text.replace("../data", os.path.abspath("data")))
    with pytest.raises(ValueError, match="nope"):
        load_config(bad)
