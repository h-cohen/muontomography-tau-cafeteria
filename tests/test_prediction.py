import numpy as np
import pytest

from cafetomo.angular import AnalysisGrid
from cafetomo.prediction import FrozenFold, poisson_deviance, predict_raw_counts
from cafetomo.reconstruct import VoxelSolution
from cafetomo.voxels import VoxelGrid


def test_zero_count_deviance():
    assert poisson_deviance(np.array([0]), np.array([2.0])) == 4
    with pytest.raises(ValueError):
        poisson_deviance(np.array([1]), np.array([0.0]))


def test_raw_alignment_and_zero_gauge(cfg):
    grid = AnalysisGrid(
        np.linspace(-0.2, 0.2, 4), {cfg.sky_reference.id: np.arange(9).reshape(3, 3) + 100}
    )
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-20, -20, 1), 40.0, (1, 1, 1)), {}, cfg.position_ids
    )
    fold = FrozenFold(cfg, volume, "train", "choice")
    out = predict_raw_counts(
        fold, grid, {cfg.sky_reference.id: 10}, {p: 20 for p in cfg.position_ids}
    )
    for p in cfg.position_ids:
        np.testing.assert_allclose(out[p], 2 * grid.counts[cfg.sky_reference.id])


def test_missing_reference_nan(cfg):
    grid = AnalysisGrid(np.array([-0.1, 0.1]), {cfg.sky_reference.id: np.zeros((1, 1))})
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-1, -1, 1), 2.0, (1, 1, 1)), {}, cfg.position_ids
    )
    assert all(
        np.isnan(v).all()
        for v in predict_raw_counts(
            FrozenFold(cfg, volume, "a", "b"),
            grid,
            {cfg.sky_reference.id: 1},
            {p: 1 for p in cfg.position_ids},
        ).values()
    )


def test_training_zeroes_exclusions_before_all_fits_and_hash(cfg, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import cafetomo.prediction as p
    from cafetomo.config import Pose

    masks = {pid: np.array([[True, False], [True, True]]) for pid in cfg.position_ids}
    counts = {pid: np.full((2, 2), 100) for pid in (*cfg.position_ids, cfg.sky_reference.id)}
    calls = []

    def spy(grid, config, live, **kwargs):
        calls.append(grid)
        for pid in cfg.position_ids:
            assert grid.counts[pid][0, 1] == 0
        assert not kwargs["observation_mask"][cfg.position_ids[0]][0, 1]
        return SimpleNamespace(pose=Pose(1, 2, 3))

    monkeypatch.setattr(p, "fit_pose", spy)
    monkeypatch.setattr(p, "opacity_sigma", lambda *a, **kw: spy(*a, **kw))
    monkeypatch.setattr(p, "solve_opacity", lambda *a, **kw: spy(*a, **kw))
    monkeypatch.setattr(
        p,
        "build_fit_data",
        lambda *a: SimpleNamespace(rows=SimpleNamespace(n_rows=1), w=np.ones(1)),
    )
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-1, -1, 1), 2.0, (1, 1, 1)), {}, cfg.position_ids
    )

    def solve(*args, **kwargs):
        assert kwargs["holdouts"] is False
        return {"full": volume}

    monkeypatch.setattr(p, "solve_voxels", solve)
    grid = AnalysisGrid(np.array([-0.2, 0, 0.2]), counts)
    live = {pid: 1 for pid in counts}
    a = p.fit_training_fold(grid, live, cfg, observation_mask=masks, cache_dir=tmp_path)
    for pid in cfg.position_ids:
        counts[pid][0, 1] += 1000
    b = p.fit_training_fold(grid, live, cfg, observation_mask=masks, cache_dir=tmp_path)
    assert a.train_hash == b.train_hash
    assert a.choice_hash == b.choice_hash
    np.testing.assert_array_equal(a.volume.rho, b.volume.rho)
    counts[cfg.position_ids[0]][0, 0] += 1
    c = p.fit_training_fold(grid, live, cfg, observation_mask=masks, cache_dir=tmp_path)
    assert a.train_hash != c.train_hash


def test_original_direction_rows_and_gauge(cfg, monkeypatch):
    from types import SimpleNamespace

    import cafetomo.prediction as p
    from cafetomo.sky import detector_to_sky

    grid = AnalysisGrid(
        np.linspace(-0.2, 0.2, 4), {cfg.sky_reference.id: np.arange(9).reshape(3, 3) + 100}
    )
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-1, -1, 1), 2.0, (1, 1, 1)), {}, cfg.position_ids
    )

    def build(rows, config, **kwargs):
        sx, sy, _ = detector_to_sky(
            *grid.tan_mesh(), cfg.exposure(rows.position_ids[rows.pos_of_row[0]]).pose
        )
        np.testing.assert_allclose(rows.sx, sx.ravel())
        np.testing.assert_allclose(rows.sy, sy.ravel())
        assert kwargs["grid"] is volume.grid
        from scipy import sparse

        return SimpleNamespace(
            A=sparse.csr_matrix(np.ones((9, 1))), predict=lambda rho: np.arange(9) / 10
        )

    monkeypatch.setattr(p, "build_forward_model", build)
    out = p.predict_raw_counts(
        FrozenFold(cfg, volume, "a", "b"),
        grid,
        {cfg.sky_reference.id: 1},
        {pid: 1 for pid in cfg.position_ids},
    )
    for mu in out.values():
        np.testing.assert_allclose(
            mu.ravel(), grid.counts[cfg.sky_reference.id].ravel() * np.exp(-np.arange(9) / 10)
        )
    bad = VoxelSolution(np.zeros(1), volume.grid, {cfg.position_ids[0]: 0.1}, cfg.position_ids)
    with pytest.raises(ValueError, match="gauge"):
        p.predict_raw_counts(
            FrozenFold(cfg, bad, "a", "b"),
            grid,
            {cfg.sky_reference.id: 1},
            {pid: 1 for pid in cfg.position_ids},
        )


def test_scoring_coverage_and_zero_observation():
    from cafetomo.prediction import score_raw_counts

    record = score_raw_counts({"p": np.array([[0, 99]])}, {"p": np.array([[2.0, np.nan]])})["p"]
    assert record["deviance"] == 4
    assert record["n_covered"] == 1 and record["n_requested"] == 2
    assert np.isnan(record["residual"][0, 1])


def test_inner_selection_then_full_training_refit(cfg, monkeypatch, tmp_path):
    from dataclasses import replace

    import cafetomo.prediction as p

    calls = []
    candidates = [cfg, replace(cfg, reconstruction=replace(cfg.reconstruction, tv_alpha=0.1))]
    grid = AnalysisGrid(
        np.array([-0.1, 0.1]),
        {pid: np.array([[100]]) for pid in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {pid: 10 for pid in grid.counts}

    def fit(g, live_times, c, **kw):
        calls.append((g, live_times, c))
        return FrozenFold(c, None, "a", "b")

    monkeypatch.setattr(p, "fit_training_fold", fit)
    monkeypatch.setattr(
        p,
        "predict_raw_counts",
        lambda fold, g, live_times, t: {pid: np.array([[50.0]]) for pid in cfg.position_ids},
    )
    _, chosen, scores = p.select_and_refit(grid, live, candidates, cache_dir=tmp_path, seed=10)
    assert chosen == 0 and scores[0] == scores[1]
    assert len(calls) == 3
    assert calls[0][1][cfg.sky_reference.id] == 5
    assert calls[2][1] == live
    for pid in grid.counts:
        np.testing.assert_array_equal(calls[2][0].counts[pid], grid.counts[pid])


def test_replica_rng_independent_of_excluded_counts(cfg, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import cafetomo.prediction as p

    grid = AnalysisGrid(
        np.array([-0.2, 0, 0.2]),
        {pid: np.full((2, 2), 100) for pid in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {pid: 1 for pid in grid.counts}
    masks = {pid: np.array([[True, False], [True, True]]) for pid in cfg.position_ids}
    captured = []

    def fit(g, live_times, c, **kw):
        captured.append({pid: v.copy() for pid, v in g.counts.items()})
        assert all(not kw["observation_mask"][pid][0, 1] for pid in cfg.position_ids)
        return SimpleNamespace(cfg=c)

    monkeypatch.setattr(p, "fit_training_fold", fit)
    monkeypatch.setattr(p, "predict_raw_counts", lambda *a: {})
    kwargs = dict(
        target_live=live, observation_mask=masks, cache_dir=tmp_path, n_replicas=2, seed=2
    )
    p.training_prediction_replicas(grid, live, cfg, **kwargs)
    for pid in cfg.position_ids:
        grid.counts[pid][0, 1] += 10000
    p.training_prediction_replicas(grid, live, cfg, **kwargs)
    for a, b in zip(captured[:2], captured[2:], strict=True):
        for pid in a:
            np.testing.assert_array_equal(a[pid], b[pid])


def test_inner_candidates_score_identical_common_directions(cfg, monkeypatch, tmp_path):
    from dataclasses import replace

    import cafetomo.prediction as p

    other = replace(cfg, reconstruction=replace(cfg.reconstruction, tv_alpha=0.1))
    grid = AnalysisGrid(
        np.array([-0.2, 0, 0.2]),
        {pid: np.full((2, 2), 100) for pid in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    live = {pid: 1 for pid in grid.counts}
    monkeypatch.setattr(p, "fit_training_fold", lambda g, t, c, **kw: FrozenFold(c, None, "a", "b"))

    def predict(fold, g, t, target):
        mu = np.full((2, 2), 50.0)
        mu[0, 0] = 10000 if fold.cfg is cfg else np.nan
        return {pid: mu.copy() for pid in cfg.position_ids}

    monkeypatch.setattr(p, "predict_raw_counts", predict)
    fold, chosen, scores = p.select_and_refit(grid, live, [cfg, other], cache_dir=tmp_path)
    assert scores[0] == scores[1] and chosen == 0
    metadata = fold.metadata["inner_selection"]
    assert all(not mask[0, 0] for mask in metadata["common_masks"].values())
    assert metadata["candidate_available_bins"][0][cfg.position_ids[0]] == 4
    assert metadata["candidate_available_bins"][1][cfg.position_ids[0]] == 3


def test_replica_summary_requires_two_supported_values():
    from cafetomo.prediction import summarize_prediction_replicas

    out = summarize_prediction_replicas(
        [{"p": np.array([1.0, np.nan, 2.0])}, {"p": np.array([3.0, np.nan, np.nan])}]
    )["p"]
    np.testing.assert_array_equal(out["n_supported"], [2, 0, 1])
    assert np.isclose(out["std"][0], np.sqrt(2))
    assert np.isnan(out["std"][1:]).all()


def test_geometric_coverage_distinguishes_missed_lattice_from_zero_density(cfg):
    from dataclasses import replace

    from cafetomo.config import Pose
    from cafetomo.prediction import score_raw_counts

    cfg = replace(cfg, exposures=tuple(replace(e, pose=Pose(0, 0, 0)) for e in cfg.exposures))
    grid = AnalysisGrid(np.array([-0.2, 0.2, 0.6]), {cfg.sky_reference.id: np.full((2, 2), 100)})
    volume = VoxelSolution(
        np.zeros(1), VoxelGrid((-0.5, -0.5, 10), 1.0, (1, 1, 1)), {}, cfg.position_ids
    )
    out = predict_raw_counts(
        FrozenFold(cfg, volume, "a", "b"),
        grid,
        {cfg.sky_reference.id: 1},
        {pid: 1 for pid in cfg.position_ids},
    )
    for pid, mu in out.items():
        assert mu[0, 0] == 100
        assert np.isnan(mu[1, 0])
        record = score_raw_counts({pid: np.full((2, 2), 100)}, {pid: mu})[pid]
        assert record["n_covered"] == 1
        assert record["coverage"][0, 0]
        assert not record["coverage"][1, 0]
        assert record["deviance"] == 0
