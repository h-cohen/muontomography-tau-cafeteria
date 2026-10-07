import json
from dataclasses import replace

import numpy as np
import pytest

from cafetomo.export import export_volume
from cafetomo.reconstruct import VoxelSolution
from cafetomo.resolution import campaign_resolution
from cafetomo.voxels import VoxelGrid

GRID = VoxelGrid(origin=(-1.0, -1.0, 5.0), spacing=0.5, shape=(4, 4, 4))

BEAMDEPTH = {"zbottom": 6.4, "w": 0.3, "h": 1.2, "ztop": 7.6, "xs": [-0.85, 0.85],
             "kappa": [1.0, 1.1], "chisq_per_dof": 1.05, "at_bound": False, "n_rows": 900,
             "profiles": {}, "converged": True, "n_eval": 40}


def _voxels_dir(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    sol = VoxelSolution(rho=np.abs(rng.normal(size=GRID.n_voxels)), grid=GRID,
                        offsets={"pos0": 0.1, "pos1": -0.1}, position_ids=("pos0", "pos1"),
                        info={"best_chi2": np.float64(1.2), "algorithm": "tv"})
    d = tmp_path / "voxels"
    sol.save(d / "volume_full.npz")
    return d


def _meta(out):
    return json.loads((out / "meta.json").read_text())


def test_export_writes_the_contract(tmp_path, cfg):
    out = tmp_path / "view"
    path = export_volume(_voxels_dir(tmp_path), cfg, out_dir=out)
    assert path == out / "volume.npy"
    assert np.load(path).shape == GRID.shape
    meta = _meta(out)
    assert meta["shape"] == list(GRID.shape)
    assert meta["axis_order"] == "xyz"
    assert meta["spacing_m"] == pytest.approx(0.5)
    assert meta["origin_m"] == list(GRID.origin)
    assert meta["layers"] == ["volume"]
    assert meta["fit_info"]["best_chi2"] == pytest.approx(1.2)


def test_meta_carries_the_detector_poses(tmp_path, cfg):
    out = tmp_path / "view"
    export_volume(_voxels_dir(tmp_path), cfg, out_dir=out)
    dets = _meta(out)["detectors"]
    assert [d["id"] for d in dets] == list(cfg.position_ids)
    assert all(set(d) == {"id", "x", "y", "z", "az_deg"} for d in dets)
    assert dets[1]["x"] == pytest.approx(cfg.exposures[1].pose.x)


def test_meta_carries_the_resolution_verdict(tmp_path, cfg):
    """The viewer must be able to show what the campaign cannot resolve without
    re-deriving it, and it must agree with the closed-form report."""
    out = tmp_path / "view"
    export_volume(_voxels_dir(tmp_path), cfg, out_dir=out)
    res = _meta(out)["resolution"]
    sigma_t = cfg.opacity.sky_t_max * 2 / cfg.opacity.sky_n_bins / np.sqrt(12.0)
    ref = campaign_resolution(cfg, sigma_t=sigma_t, feature_pitch_m=2.0)
    assert res["depth_resolved"] is ref["depth_resolved"]
    assert res["max_baseline_m"] == pytest.approx(ref["max_baseline_m"])
    assert res["verdict"] == ref["verdict"]


def test_optional_layers_are_exported_when_present(tmp_path, cfg):
    voxels = _voxels_dir(tmp_path)
    boot = tmp_path / "bootstrap"
    boot.mkdir()
    np.savez_compressed(boot / "volume_stats.npz",
                        mean=np.zeros(GRID.shape, dtype=np.float32),
                        sigma=np.full(GRID.shape, 0.5, dtype=np.float32),
                        snr=np.full(GRID.shape, 4.0, dtype=np.float32))
    np.save(voxels / "views.npy", np.full(GRID.shape, 2, dtype=np.int16))
    out = tmp_path / "view"
    export_volume(voxels, cfg, bootstrap_dir=boot, out_dir=out)
    assert _meta(out)["layers"] == ["volume", "sigma", "snr", "views"]
    assert np.load(out / "sigma.npy").dtype == np.float32
    assert np.all(np.load(out / "snr.npy") == 4.0)
    assert np.all(np.load(out / "views.npy") == 2)


def test_bootstrap_on_another_grid_fails_loudly(tmp_path, cfg):
    boot = tmp_path / "bootstrap"
    boot.mkdir()
    np.savez_compressed(boot / "volume_stats.npz", mean=np.zeros((2, 2, 2)),
                        sigma=np.ones((2, 2, 2)), snr=np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="sigma has shape"):
        export_volume(_voxels_dir(tmp_path), cfg, bootstrap_dir=boot, out_dir=tmp_path / "v")


def test_a_named_bootstrap_dir_without_stats_fails_loudly(tmp_path, cfg):
    with pytest.raises(FileNotFoundError):
        export_volume(_voxels_dir(tmp_path), cfg, bootstrap_dir=tmp_path / "missing",
                      out_dir=tmp_path / "v")


def test_viewer_crop_is_written_when_configured_and_absent_otherwise(tmp_path, cfg):
    voxels = _voxels_dir(tmp_path)
    out = tmp_path / "view"
    export_volume(voxels, cfg, out_dir=out)
    if cfg.volume.viewer_crop_xy_m is None:
        assert "viewer_crop_xy_m" not in _meta(out)

    crop = ((-1.0, 2.0), (-0.5, 1.5))
    export_volume(voxels, replace(cfg, volume=replace(cfg.volume, viewer_crop_xy_m=crop)),
                  out_dir=out)
    assert _meta(out)["viewer_crop_xy_m"] == [[-1.0, 2.0], [-0.5, 1.5]]


def test_beams_block_from_the_beam_depth_fit(tmp_path, cfg):
    results = tmp_path / "results"
    results.mkdir()
    (results / "beamdepth.json").write_text(json.dumps(BEAMDEPTH))
    (results / "uncertainty.json").write_text(json.dumps({"depth_h_total": 0.15,
                                                         "depth_h_sigma": 0.08}))
    out = tmp_path / "view"
    export_volume(_voxels_dir(tmp_path), cfg, results_dir=results, out_dir=out)
    beams = _meta(out)["beams"]
    y0, y1 = cfg.beamdepth.y_extent_m
    assert beams["h"] == pytest.approx(1.2)
    assert beams["h_sigma"] == pytest.approx(0.15)      # total, not statistical
    assert beams["boxes"] == [
        {"x": -0.85, "w": 0.3, "zbottom": 6.4, "ztop": pytest.approx(7.6), "y_extent": [y0, y1]},
        {"x": 0.85, "w": 0.3, "zbottom": 6.4, "ztop": pytest.approx(7.6), "y_extent": [y0, y1]},
    ]


def test_beams_without_an_error_budget_carry_a_null_sigma(tmp_path, cfg):
    results = tmp_path / "results"
    results.mkdir()
    (results / "beamdepth.json").write_text(json.dumps(BEAMDEPTH))
    out = tmp_path / "view"
    export_volume(_voxels_dir(tmp_path), cfg, results_dir=results, out_dir=out)
    beams = _meta(out)["beams"]
    assert beams["h_sigma"] is None
    assert len(beams["boxes"]) == 2


def test_no_beams_block_without_a_beam_depth_fit(tmp_path, cfg):
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "view"
    export_volume(_voxels_dir(tmp_path), cfg, results_dir=results, out_dir=out)
    assert "beams" not in _meta(out)
    export_volume(_voxels_dir(tmp_path), cfg, out_dir=out)
    assert "beams" not in _meta(out)
