from dataclasses import replace

import numpy as np
import pytest

from cafetomo import recovery
from cafetomo.angular import AnalysisGrid


def test_independent_truth_and_variants(cfg, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("inverse model used by truth")

    monkeypatch.setattr("cafetomo.forward.build_forward_model", forbidden)
    monkeypatch.setattr("cafetomo.reconstruct.solve_voxels", forbidden)
    grid = AnalysisGrid(np.linspace(-0.8, 0.8, 17), {"sky": np.full((16, 16), 10000.0)})
    live = {cfg.sky_reference.id: 100.0, **{p: 200.0 for p in cfg.position_ids}}
    grid = replace(grid, counts={cfg.sky_reference.id: grid.counts["sky"]})
    cases = recovery.standard_cases()
    means = [recovery.expected_counts(c, cfg, grid, live, n_sub=8) for c in cases]
    nominal = means[1].counts["pos0"]
    for idx in (4, 5, 6, 7):
        assert np.max(np.abs(means[idx].counts["pos0"] - nominal)) > 1
    doubled = recovery.expected_counts(cases[1], cfg, grid, {**live, "pos0": 400.0}, n_sub=8)
    assert np.allclose(doubled.counts["pos0"], 2 * nominal)
    assert np.all(means[4].counts["pos0"] >= nominal)


def test_column_fixed(cfg):
    d = np.array([[0.0, 0.0, 1.0]])
    for h in (0.3, 1.8):
        c = recovery.RecoveryCase("x", 6.4, h, 0.3, "nominal", 0.06)
        assert recovery.truth_transmission(c, cfg, "pos0", d, n_sub=16)[0] == pytest.approx(
            np.exp(-0.13), abs=0.025
        )


def test_failure_classifier():
    assert recovery.expected_failure(RuntimeError("beam triangulation failed: none"))
    assert not recovery.expected_failure(ValueError("bad input"))
    assert not recovery.expected_failure(RuntimeError("unexpected bug"))
    assert recovery.expected_failure(ValueError("pos0: insufficient background coverage"))
    assert recovery.expected_failure(ValueError("pos1: background clipping did not converge"))
    assert recovery.expected_failure(
        ValueError("no constrained sky directions: every lambda or sigma is non-finite")
    )
    assert not recovery.expected_failure(ValueError("insufficient background coverage"))
    assert not recovery.expected_failure(ValueError("pos0: insufficient background coverage extra"))
    assert not recovery.expected_failure(
        ValueError("no constrained sky directions: every lambda or sigma is non-finite extra")
    )


def test_volume_archive_survives_downstream_classified_failure(cfg, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from cafetomo.reconstruct import VoxelSolution
    from cafetomo.voxels import VoxelGrid

    volume = VoxelSolution(
        np.array([0.25]),
        VoxelGrid((0, 0, 1), 1, (1, 1, 1)),
        {"pos0": 0.0},
        cfg.position_ids,
        {"n_iter_used": 5, "best_iter": 2},
    )
    fold = SimpleNamespace(cfg=cfg, volume=volume, metadata={"weight_hash": "x"})
    monkeypatch.setattr(recovery, "fit_training_fold", lambda *a, **kw: fold)
    monkeypatch.setattr(
        recovery,
        "solve_opacity",
        lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("pos0: insufficient background coverage")
        ),
    )
    result = recovery.run_recovery_case(
        recovery.standard_cases()[0], None, {}, cfg, seed=12, out_dir=tmp_path
    )
    assert result["status"] == "fit_failure"
    assert result["artifacts"] == {"volume": "volume.npz", "solver_info": "solver-info.json"}
    np.testing.assert_allclose(VoxelSolution.load(tmp_path / "volume.npz").rho, volume.rho)
    assert (tmp_path / "solver-info.json").read_text().find('"best_iter": 2') > 0


def test_boundary_record():
    from types import SimpleNamespace

    f = SimpleNamespace(
        h=2.0,
        z0=6.0,
        w=0.4,
        xs=np.array([0.0]),
        at_bound=True,
        converged=True,
        to_json=lambda: {"at_bound": True},
    )
    r = recovery.score_fit(recovery.standard_cases()[0], f)
    assert r["at_bound"] and not r["physical_thickness_measured"]
    assert r["bias"]["thickness_m"] == pytest.approx(1.7)


def test_full_pipeline_preserves_rough_prior_and_failure(cfg, monkeypatch, tmp_path):
    received = []

    def fail(grid, live, prior, **kw):
        received.append(prior)
        raise RuntimeError("pose fit failed: iteration budget")

    monkeypatch.setattr(recovery, "fit_training_fold", fail)
    result = recovery.run_recovery_case(
        recovery.standard_cases()[0], None, {}, cfg, seed=12, out_dir=tmp_path
    )
    assert received[0].exposure("pos1").pose == cfg.exposure("pos1").pose
    assert received[0].exposure("pos1").pose != recovery.TRUTH_POSE
    assert result["status"] == "fit_failure"
    assert (tmp_path / "outcome.json").exists()

    def bug(*a, **kw):
        raise ValueError("programming error")

    monkeypatch.setattr(recovery, "fit_training_fold", bug)
    with pytest.raises(ValueError, match="programming error"):
        recovery.run_recovery_case(
            recovery.standard_cases()[0], None, {}, cfg, seed=12, out_dir=tmp_path
        )


def test_poisson_seed_and_shared_roof(cfg):
    sid = cfg.sky_reference.id
    grid = AnalysisGrid(np.linspace(-0.8, 0.8, 9), {sid: np.full((8, 8), 1000.0)})
    live = {p: 1.0 for p in (*cfg.position_ids, sid)}
    a = recovery.simulate_counts(recovery.standard_cases()[0], cfg, grid, live, seed=42, n_sub=4)
    b = recovery.simulate_counts(recovery.standard_cases()[0], cfg, grid, live, seed=42, n_sub=4)
    for p in a.counts:
        assert np.array_equal(a.counts[p], b.counts[p])
        assert np.issubdtype(a.counts[p].dtype, np.integer)


@pytest.mark.slow
def test_independent_noiseless_controls(cfg, tmp_path):
    outcomes = recovery.direct_controls(cfg, out_dir=tmp_path)
    assert len(outcomes) == 2
    for r in outcomes:
        assert r["lateral_pass"]
        assert r["vertical_pass"]
        assert r["rms_lambda_change"] < 0.002
        assert not r["fit"]["physical_thickness_measured"]


def test_transmission_average_not_average_path(cfg):
    d = np.array([[0.0, 0.0, 1.0]])
    c = recovery.RecoveryCase("mixed", 6.4, 0.3, 0.15, "nominal", 1.2)
    value = recovery.truth_transmission(c, cfg, "pos0", d, n_sub=16)[0]
    # Accepted midpoint sample positions along x; y is wholly within the beam.
    x = ((np.arange(16) + 0.5) / 16 - 0.5) * cfg.detector.aperture_m
    fraction = (np.abs(x) < 0.075).mean()
    assert value == pytest.approx(np.exp(-0.07) * ((1 - fraction) + fraction * np.exp(-1.2)))
    assert value > np.exp(-0.07 - fraction * 1.2)
    taller = replace(c, thickness_m=1.8)
    assert recovery.truth_transmission(taller, cfg, "pos0", d, n_sub=16)[0] == pytest.approx(value)


def test_null_scoring_has_no_dimension_bias():
    from types import SimpleNamespace

    f = SimpleNamespace(h=2.0, z0=6.0, w=0.4, at_bound=False, converged=True, to_json=lambda: {})
    r = recovery.score_fit(recovery.standard_cases()[4], f)
    assert r["null_false_positive"]
    assert r["bias"] is None


def test_voxel_localization_scores_xyz_without_truth_z_crop():
    from types import SimpleNamespace

    from cafetomo.voxels import VoxelGrid

    grid = VoxelGrid(origin=(-4.0, -2.0, 4.0), spacing=0.2, shape=(45, 20, 25))
    rho = np.zeros(grid.shape)
    for x in recovery.TRUTH_CENTRES:
        i = np.argmin(abs(grid.axis_centers(0) - x))
        rho[i, 10, 12] = 1.0
    volume = SimpleNamespace(grid=grid, rho=rho.ravel())
    result = recovery.voxel_localization(recovery.standard_cases()[0], volume)
    for beam in result["beams"]:
        assert np.max(np.abs(beam["error_xyz_m"])) <= 0.2
        assert np.max(beam["rms_extent_xyz_m"]) == 0.0
    assert recovery.voxel_localization(recovery.standard_cases()[4], volume) is None
