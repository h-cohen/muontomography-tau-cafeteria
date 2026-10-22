import json

import numpy as np
import pytest

from cafetomo.angular import AnalysisGrid


def test_protocol_scopes(cfg):
    from cafetomo.internal_validation import partitions, settings_document

    settings = settings_document({"mode": "smoke"})
    grid = AnalysisGrid(
        np.linspace(-0.2, 0.2, 3),
        {p: np.full((2, 2), 100) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {p: 100.0 for p in grid.counts}
    folds = list(partitions(grid, live, cfg, settings))
    count, angular = folds[0], folds[-1]
    assert count["train_live"][cfg.sky_reference.id] == 50
    assert angular["train_live"] == live
    assert angular["roof_scope"] == "full_reference_conditional"
    np.testing.assert_array_equal(
        angular["train"].counts[cfg.sky_reference.id], grid.counts[cfg.sky_reference.id]
    )


def test_manifest_failure_is_incomplete(cfg, tmp_path, monkeypatch):
    import cafetomo.internal_validation as v

    root = tmp_path / "run"
    grid = AnalysisGrid(
        np.linspace(-0.2, 0.2, 3),
        {p: np.full((2, 2), 100) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    monkeypatch.setattr(
        v, "select_and_refit", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("refused fit"))
    )
    with pytest.raises(RuntimeError, match="refused fit"):
        v.run_campaign(
            grid,
            {p: 1.0 for p in grid.counts},
            cfg,
            {"mode": "smoke"},
            out=root,
            cache=tmp_path / "cache",
        )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["failures"][0]["error"] == "refused fit"
    assert not (root / "results" / "prediction.json").exists()


def test_isolated_complete_campaign_and_test_perturbation(cfg, tmp_path, monkeypatch):
    import cafetomo.internal_validation as v
    from cafetomo.prediction import FrozenFold
    from cafetomo.reconstruct import VoxelSolution
    from cafetomo.voxels import VoxelGrid

    grid = AnalysisGrid(
        np.linspace(-0.2, 0.2, 11),
        {p: np.full((10, 10), 100) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {p: 100.0 for p in grid.counts}
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-20, -20, 1), 40.0, (1, 1, 1)), {}, cfg.position_ids
    )
    seen = []

    def select(train, times, candidates, **kw):
        assert kw["nuisance_cfg"] == cfg
        seen.append((v._digest(candidates[0], train, times, {}), kw["seed"], kw["full_roof_id"]))
        from dataclasses import replace

        from cafetomo.config import Pose

        trained = cfg.with_pose(cfg.selfcal.free_pose, Pose(5, 5, 0))
        trained = replace(
            trained, reconstruction=replace(trained.reconstruction, coverage_damping=0)
        )
        return FrozenFold(trained, volume, "train", "choice", {}), 0, [1.0]

    monkeypatch.setattr(v, "select_and_refit", select)
    monkeypatch.setattr(v, "run_support", lambda *a: {})
    monkeypatch.setattr(v, "run_recovery", lambda *a: v.write_json(a[-1] / "recovery.json", {}))
    monkeypatch.setattr(v, "run_stability", lambda *a: v.write_json(a[-1] / "stability.json", {}))

    def replica_spy(g, times, config, **kwargs):
        assert config.exposures == cfg.exposures
        assert config.selfcal == cfg.selfcal
        assert config.reconstruction.coverage_damping == 0
        assert kwargs["nuisance_cfg"] == cfg
        reference = kwargs["reference_scope"]
        assert reference["roof_live_time_s"] == times[cfg.sky_reference.id]
        if times[cfg.sky_reference.id] == live[cfg.sky_reference.id]:
            assert (
                reference["reference_counting_scope"]
                == "full-reference counting variation included"
            )
            assert (
                reference["point_score_reference_scope"]
                == "conditional on observed full roof calibration"
            )
        return [{p: np.ones((10, 10)) for p in cfg.position_ids}] * 2

    monkeypatch.setattr(v, "training_prediction_replicas", replica_spy)
    protected = tmp_path / "results"
    protected.mkdir()
    sentinel = protected / "paper.json"
    sentinel.write_bytes(b"unchanged paper input")
    v.run_campaign(grid, live, cfg, {"mode": "smoke"}, out=tmp_path / "a", cache=tmp_path / "cache")
    original = list(seen)
    seen.clear()
    original_partitions = v.partitions

    def perturbed(*args):
        for part in original_partitions(*args):
            from dataclasses import replace

            part["test"] = replace(
                part["test"], counts={p: x + 10000 for p, x in part["test"].counts.items()}
            )
            yield part

    monkeypatch.setattr(v, "partitions", perturbed)
    v.run_campaign(grid, live, cfg, {"mode": "smoke"}, out=tmp_path / "b", cache=tmp_path / "cache")
    assert seen == original
    assert sentinel.read_bytes() == b"unchanged paper input"
    assert json.loads((tmp_path / "a" / "manifest.json").read_text())["status"] == "complete"
    assert {p.name for p in (tmp_path / "a" / "results").iterdir()} == {
        "prediction.json",
        "support.json",
        "recovery.json",
        "stability.json",
    }


def test_cli_loads_isolated_output_and_fast_smoke(tmp_path, monkeypatch, cfg):
    import cafetomo.cli as cli
    import cafetomo.internal_validation as v

    settings = tmp_path / "settings.yaml"
    settings.write_text("mode: pilot\n")
    config = tmp_path / "config.yaml"
    config.write_text("config hash only\n")
    grid = AnalysisGrid(
        np.array([-0.1, 0.1]),
        {p: np.ones((1, 1), int) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    captured = {}
    monkeypatch.setattr(cli, "_cfg", lambda args: cfg)
    monkeypatch.setattr(cli, "_grid", lambda *args: (grid, {p: 1.0 for p in grid.counts}))

    def campaign(*args, **kw):
        captured.update(kw)
        assert args[3]["mode"] == "smoke"

    monkeypatch.setattr(v, "run_campaign", campaign)
    cli.main(
        [
            "--fast",
            "internal-validate",
            "--config",
            str(config),
            "--ingest",
            str(tmp_path),
            "--settings",
            str(settings),
            "--out",
            str(tmp_path / "new-run"),
            "--cache",
            str(tmp_path / "new-cache"),
        ]
    )
    assert captured["out"] == str(tmp_path / "new-run")
    assert captured["cache"] == str(tmp_path / "new-cache")


def test_shared_inner_pose_fit_and_angular_roof(cfg, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import cafetomo.prediction as p
    from cafetomo.config import Pose
    from cafetomo.reconstruct import VoxelSolution
    from cafetomo.voxels import VoxelGrid

    grid = AnalysisGrid(
        np.linspace(-0.2, 0.2, 3),
        {pid: np.full((2, 2), 100) for pid in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {pid: 100.0 for pid in grid.counts}
    calls = {"pose": [], "sigma": [], "opacity": [], "selection": []}

    def spy(stage, g, c, t, **kw):
        calls[stage].append((p._digest(c, g, t, kw["observation_mask"]), t.copy()))
        assert t[c.sky_reference.id] == 100.0
        np.testing.assert_array_equal(g.counts[c.sky_reference.id], grid.counts[c.sky_reference.id])
        if stage == "pose":
            assert c.exposures == cfg.exposures
            assert c.selfcal == cfg.selfcal
            assert c.reconstruction == cfg.reconstruction
        return SimpleNamespace(pose=Pose(5, 5, 0))

    monkeypatch.setattr(p, "fit_pose", lambda *a, **kw: spy("pose", *a, **kw))
    monkeypatch.setattr(p, "opacity_sigma", lambda *a, **kw: spy("sigma", *a, **kw))
    monkeypatch.setattr(p, "solve_opacity", lambda *a, **kw: spy("opacity", *a, **kw))
    monkeypatch.setattr(
        p,
        "build_fit_data",
        lambda *a: SimpleNamespace(rows=SimpleNamespace(n_rows=1), w=np.ones(1)),
    )
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-20, -20, 1), 40.0, (1, 1, 1)), {}, cfg.position_ids
    )
    monkeypatch.setattr(p, "solve_voxels", lambda *a, **kw: {"full": volume})
    from dataclasses import replace

    candidates = [cfg, replace(cfg, reconstruction=replace(cfg.reconstruction, tv_alpha=0.015))]
    fold, chosen, scores = p.select_and_refit(
        grid,
        live,
        candidates,
        cache_dir=tmp_path,
        shared_nuisance=True,
        full_roof_id=cfg.sky_reference.id,
        nuisance_cfg=cfg,
    )
    calls["selection"].append((chosen, list(scores), fold.choice_hash))
    assert len(calls["pose"]) == 2  # one shared inner fit plus outer-training refit
    before = {k: list(v) for k, v in calls.items()}
    mu = p.predict_raw_counts(fold, grid, live, live)
    p.score_raw_counts(grid.counts, mu)
    changed = {pid: counts + 10000 for pid, counts in grid.counts.items()}
    p.score_raw_counts(changed, mu)
    assert calls == before  # perturb TEST only, without repeating isolated fits


def test_completed_fit_cache_is_scientific(cfg, tmp_path, monkeypatch):
    import cafetomo.internal_validation as v
    from cafetomo.prediction import FrozenFold
    from cafetomo.reconstruct import VoxelSolution
    from cafetomo.voxels import VoxelGrid

    grid = AnalysisGrid(
        np.array([-0.1, 0.1]),
        {p: np.full((1, 1), 100) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {p: 1.0 for p in grid.counts}
    calls = []
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-1, -1, 1), 2.0, (1, 1, 1)), {}, cfg.position_ids
    )

    def fit(*args, **kw):
        calls.append(args[0])
        return FrozenFold(cfg, volume, "train", "choice")

    monkeypatch.setattr(v, "fit_training_fold", fit)
    cached = v.cached_fitter("source")
    for _ in range(2):
        cached(grid, live, cfg, observation_mask=None, cache_dir=tmp_path)
    assert len(calls) == 1
    grid.counts[cfg.position_ids[0]][0, 0] += 1
    cached(grid, live, cfg, observation_mask=None, cache_dir=tmp_path)
    assert len(calls) == 2
    v.cached_fitter("changed-source")(grid, live, cfg, observation_mask=None, cache_dir=tmp_path)
    assert len(calls) == 3
    reordered = AnalysisGrid(grid.edges, dict(reversed(list(grid.counts.items()))))
    cached(reordered, live, cfg, observation_mask=None, cache_dir=tmp_path)
    assert len(calls) == 4
    cached(reordered, live, cfg, observation_mask=None, cache_dir=tmp_path)
    assert len(calls) == 4
    records = [
        json.loads(p.read_text()) for p in tmp_path.glob("completed-training/*/complete.json")
    ]
    assert any(r["source_order"] == list(reordered.counts) for r in records)
    assert all(r["cache_scope"].startswith("ordered count sources") for r in records)


@pytest.mark.parametrize("protected", ["results", "paper-draft", "paper-production"])
def test_protect_checkout_outputs_from_other_cwd(cfg, tmp_path, monkeypatch, protected):
    from pathlib import Path

    import cafetomo.internal_validation as v

    checkout = Path(v.__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    grid = AnalysisGrid(
        np.array([-0.1, 0.1]),
        {p: np.ones((1, 1), int) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {p: 1.0 for p in grid.counts}
    with pytest.raises(ValueError, match="protected production"):
        v.run_campaign(
            grid,
            live,
            cfg,
            {"mode": "smoke"},
            out=checkout / protected / "do-not-create",
            cache=tmp_path / "cache",
        )
    assert not (checkout / protected / "do-not-create").exists()


def test_stability_preserves_original_nuisance_prior(cfg, tmp_path, monkeypatch):
    from dataclasses import replace

    import cafetomo.internal_validation as v
    import cafetomo.robustness as r
    from cafetomo.config import Pose
    from cafetomo.prediction import FrozenFold
    from cafetomo.reconstruct import VoxelSolution
    from cafetomo.voxels import VoxelGrid

    grid = AnalysisGrid(
        np.array([-0.1, 0.1]),
        {p: np.ones((1, 1), int) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    trained = cfg.with_pose(cfg.selfcal.free_pose, Pose(50, 50, 0))
    trained = replace(trained, reconstruction=replace(trained.reconstruction, coverage_damping=0))
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-1, -1, 1), 2.0, (1, 1, 1)), {}, cfg.position_ids
    )

    def spy(g, live, config, *args, **kwargs):
        assert config.exposures == cfg.exposures
        assert config.selfcal == cfg.selfcal
        assert config.reconstruction.coverage_damping == 0
        assert kwargs["nuisance_cfg"] == cfg
        return {}

    monkeypatch.setattr(r, "run_count_replicas", spy)
    monkeypatch.setattr(r, "run_robustness", spy)
    part = dict(
        name="count-0",
        original_cfg=cfg,
        train=grid,
        train_live={p: 1.0 for p in grid.counts},
        train_masks=None,
    )
    v.run_stability(
        (part, FrozenFold(trained, volume, "a", "b")),
        v.settings_document({"mode": "smoke"}),
        tmp_path,
        tmp_path / "results",
    )


def test_first_fit_failure_retains_all_registered_inputs(cfg, tmp_path, monkeypatch):
    import cafetomo.internal_validation as v

    grid = AnalysisGrid(
        np.linspace(-0.2, 0.2, 11),
        {p: np.full((10, 10), 100) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {p: 100.0 for p in grid.counts}

    def fail(*a, **kw):
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
        assert len(manifest["registered_folds"]) == 2
        angular = next(r for r in manifest["registered_folds"] if r["protocol"] == "angular")
        assert angular["train_live"] == live
        assert angular["roof_scope"] == "full_reference_conditional"
        assert angular["inner_selection_seed"] == v.seed(
            v.settings_document({"mode": "smoke"}), 3, 1
        )
        directory = tmp_path / "run" / "folds" / angular["name"]
        masks = np.load(directory / "masks.npz")
        counts = np.load(directory / "training-counts.npz")
        for p in cfg.position_ids:
            assert np.all(counts[p][~masks[p]] == 0)
        assert (directory / "test-masks.npz").exists()
        assert angular["source_hash"] == manifest["source_hash"]
        assert angular["training_input_hash"]
        raise RuntimeError("audit failure after registration")

    monkeypatch.setattr(v, "select_and_refit", fail)
    with pytest.raises(RuntimeError, match="audit failure"):
        v.run_campaign(
            grid, live, cfg, {"mode": "smoke"}, out=tmp_path / "run", cache=tmp_path / "cache"
        )


def test_angular_uncertainty_scope_is_declared(cfg):
    from cafetomo.internal_validation import uncertainty_reference_scope

    scope = uncertainty_reference_scope("angular", cfg.sky_reference.id, 123.0)
    assert scope["roof_live_time_s"] == 123.0
    assert scope["reference_counting_scope"] == "full-reference counting variation included"
    assert (
        scope["reference_sampling_scope"]
        == "one Poisson roof draw shared across all room views per replica"
    )
    assert scope["point_score_reference_scope"] == "conditional on observed full roof calibration"


@pytest.mark.parametrize(
    ("relative", "allowed"),
    [
        ("runs/internal-validation/pilot", True),
        ("runs/repro/internal-validation/pilot", True),
        ("runs/repro/baseline/pilot", False),
        ("runs/voxels/pilot", False),
    ],
)
def test_campaigns_inside_runs_need_an_internal_validation_folder(tmp_path, relative, allowed):
    from cafetomo.internal_validation import check_campaign_paths

    root = tmp_path / "checkout"
    out = root / relative
    cache = tmp_path / "cache"
    if allowed:
        check_campaign_paths(out, cache, {root})
    else:
        with pytest.raises(ValueError, match="inside runs must use"):
            check_campaign_paths(out, cache, {root})
