"""Retrospective partitions preserve counts and exclude missing observations."""

from dataclasses import replace

import numpy as np
import pytest

from cafetomo.angular import AnalysisGrid
from cafetomo.opacity import build_fit_data, opacity_sigma, poisson_sigma, solve_opacity


def _partition_grid():
    return AnalysisGrid(
        np.linspace(-0.5, 0.5, 5),
        {
            pid: np.arange(16).reshape(4, 4) + offset
            for pid, offset in (("pos1", 50), ("SKY", 100), ("pos0", 20))
        },
    )


def test_split_conserves_counts_live_times_and_shared_roof():
    from cafetomo.validation_split import split_counts

    grid = _partition_grid()
    originals = {pid: counts.copy() for pid, counts in grid.counts.items()}
    live = {"pos0": 13.0, "pos1": 21.0, "SKY": 37.0}
    part = split_counts(grid, live, fraction=0.5, seed=314159)
    again = split_counts(grid, live, fraction=0.5, seed=314159)
    reordered = AnalysisGrid(grid.edges, dict(reversed(list(grid.counts.items()))))
    reordered_part = split_counts(reordered, live, fraction=0.5, seed=314159)
    different = split_counts(grid, live, fraction=0.5, seed=314160)
    assert set(part.train.counts) == set(grid.counts) == set(part.test.counts)
    assert part.seed == 314159
    rng = np.random.default_rng(314159)
    for pid in sorted(grid.counts):
        np.testing.assert_array_equal(
            part.train.counts[pid] + part.test.counts[pid], originals[pid]
        )
        np.testing.assert_array_equal(grid.counts[pid], originals[pid])
        np.testing.assert_array_equal(part.train.counts[pid], again.train.counts[pid])
        np.testing.assert_array_equal(part.train.counts[pid], reordered_part.train.counts[pid])
        np.testing.assert_array_equal(part.train.counts[pid], rng.binomial(originals[pid], 0.5))
        assert part.train_live[pid] == part.test_live[pid] == live[pid] / 2
        assert part.train_live[pid] + part.test_live[pid] == live[pid]
        assert not np.shares_memory(part.train.counts[pid], grid.counts[pid])
        assert not np.shares_memory(part.test.counts[pid], grid.counts[pid])
    assert not np.array_equal(part.train.counts["SKY"], different.train.counts["SKY"])


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_split_rejects_invalid_fractions(fraction):
    from cafetomo.validation_split import split_counts

    with pytest.raises(ValueError):
        split_counts(
            _partition_grid(), {p: 1.0 for p in _partition_grid().counts}, fraction=fraction, seed=0
        )


@pytest.mark.parametrize(
    "bad",
    [
        np.full((4, 4), -1),
        np.full((4, 4), 0.5),
        np.full((4, 4), np.nan),
        np.ones((3, 4), dtype=int),
    ],
)
def test_split_rejects_bad_counts(bad):
    from cafetomo.validation_split import split_counts

    grid = _partition_grid()
    with pytest.raises(ValueError):
        split_counts(
            AnalysisGrid(grid.edges, {**grid.counts, "pos0": bad}),
            {p: 1.0 for p in grid.counts},
            fraction=0.5,
            seed=0,
        )


def test_non_half_fraction_scales_each_source_live_time():
    from cafetomo.validation_split import split_counts

    grid = _partition_grid()
    live = {"pos0": 3.0, "pos1": 7.0, "SKY": 19.0}
    part = split_counts(grid, live, fraction=0.3, seed=3)
    for pid in live:
        assert part.train_live[pid] == pytest.approx(0.3 * live[pid])
        assert part.test_live[pid] == pytest.approx(0.7 * live[pid])


def test_partial_tiles_and_guard_separation():
    from cafetomo.validation_split import angular_masks

    shape = (13, 11)
    ii, jj = np.indices(shape)
    tests = []
    for fold in range(5):
        train, test = angular_masks(shape, fold=fold)
        expected = ((ii // 5 + 2 * (jj // 5)) % 5) == fold
        np.testing.assert_array_equal(test, expected)
        assert train.dtype == test.dtype == bool
        assert not np.any(train & test)
        for i, j in zip(*np.nonzero(test), strict=True):
            assert not train[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2].any()
        tests.append(test)
    np.testing.assert_array_equal(np.sum(tests, axis=0), np.ones(shape))
    train, test = angular_masks(shape, fold=2, guard=0)
    np.testing.assert_array_equal(train, ~test)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fold": -1},
        {"fold": 5},
        {"fold": 0, "folds": 1},
        {"fold": 0, "tile": 0},
        {"fold": 0, "guard": -1},
    ],
)
def test_angular_masks_reject_invalid_settings(kwargs):
    from cafetomo.validation_split import angular_masks

    with pytest.raises(ValueError):
        angular_masks((13, 11), **kwargs)


def _opacity_grid():
    return AnalysisGrid(
        np.linspace(-0.1, 0.1, 3),
        {
            "SKY": np.full((2, 2), 5000, dtype=np.int64),
            "pos0": np.array([[0, 2500], [2500, 2500]]),
            "pos1": np.full((2, 2), 2500, dtype=np.int64),
        },
    )


LIVE = {"SKY": 2.0, "pos0": 1.0, "pos1": 1.0}


def test_measured_zero_is_finite_but_masked_bin_is_unmeasured(cfg):
    grid = _opacity_grid()
    nominal = solve_opacity(grid, cfg, LIVE)
    sig = poisson_sigma(grid, cfg, LIVE)
    rows = build_fit_data(nominal, cfg, sig).rows
    assert np.nanmax(nominal.lam["pos0"]) == pytest.approx(-np.log(cfg.opacity.t_floor))
    masks = {"pos0": np.zeros((2, 2), dtype=bool)}
    masked = solve_opacity(grid, cfg, LIVE, observation_mask=masks)
    masked_sig = poisson_sigma(grid, cfg, LIVE, observation_mask=masks)
    assert np.isnan(masked.lam["pos0"]).all()
    pinned = build_fit_data(masked, cfg, masked_sig, rows=rows)
    assert np.all(pinned.w[rows.mask_for("pos0")] == 0)
    np.testing.assert_array_equal(masked.lam["pos1"], nominal.lam["pos1"])
    boot = opacity_sigma(grid, cfg, LIVE, n_replicas=3, seed=2, observation_mask=masks)
    assert np.isnan(boot["pos0"]).all()


def test_mask_excludes_matched_roof_expectation_in_merged_world_bin(cfg):
    grid = _opacity_grid()
    c = replace(cfg, opacity=replace(cfg.opacity, sky_n_bins=1))
    mask = np.array([[False, True], [False, False]])
    maps = solve_opacity(grid, c, LIVE, observation_mask={"pos0": mask})
    assert maps.lam["pos0"][0] == pytest.approx(0.0)
    sigma = poisson_sigma(grid, c, LIVE, observation_mask={"pos0": mask})
    assert sigma["pos0"][0] == pytest.approx(np.sqrt(1 / 2500 + 1 / 5000))


def test_mask_rejects_shape_or_non_boolean_values(cfg):
    for mask in (np.ones((1, 2), dtype=bool), np.ones((2, 2), dtype=int)):
        with pytest.raises(ValueError):
            solve_opacity(_opacity_grid(), cfg, LIVE, observation_mask={"pos0": mask})


def test_pose_objective_cannot_consume_masked_room_counts(cfg, monkeypatch):
    from scipy.optimize import OptimizeResult

    from cafetomo import selfcal

    grid = _opacity_grid()
    masks = {pid: np.array([[False, True], [True, True]]) for pid in cfg.position_ids}
    c = replace(cfg, selfcal=replace(cfg.selfcal, baseline_m=None))
    captured = []

    def score(data, config, volume):
        captured.append((data.lam.copy(), data.w.copy(), data.rows.sky_flat.copy()))
        return float(np.sum(data.lam * data.w))

    monkeypatch.setattr(selfcal, "cross_position_score", score)

    def minimize(objective, initial, **kwargs):
        value = objective(initial)
        objective(initial + np.array([0.01, 0.01, 1.0]))
        return OptimizeResult(success=True, fun=value, x=initial, nfev=2)

    monkeypatch.setattr(selfcal.optimize, "minimize", minimize)
    first = selfcal.fit_pose(grid, c, LIVE, observation_mask=masks)
    before = captured.copy()
    captured.clear()
    changed = {pid: counts.copy() for pid, counts in grid.counts.items()}
    for pid in cfg.position_ids:
        changed[pid][~masks[pid]] = 10**8
    second = selfcal.fit_pose(AnalysisGrid(grid.edges, changed), c, LIVE, observation_mask=masks)
    assert first == second
    for a, b in zip(before, captured, strict=True):
        for left, right in zip(a, b, strict=True):
            np.testing.assert_array_equal(left, right)


def test_weight_replicas_do_not_use_masked_counts_for_random_stream(cfg):
    grid = _opacity_grid()
    masks = {pid: np.array([[False, True], [True, True]]) for pid in cfg.position_ids}
    changed = {pid: counts.copy() for pid, counts in grid.counts.items()}
    for pid in cfg.position_ids:
        changed[pid][~masks[pid]] = 10**8
    first = opacity_sigma(grid, cfg, LIVE, n_replicas=5, seed=3, observation_mask=masks)
    second = opacity_sigma(
        AnalysisGrid(grid.edges, changed), cfg, LIVE, n_replicas=5, seed=3, observation_mask=masks
    )
    for pid in cfg.position_ids:
        np.testing.assert_array_equal(first[pid], second[pid])


def test_pose_bootstrap_keeps_training_only_random_stream(cfg, monkeypatch):
    from scipy.optimize import OptimizeResult

    from cafetomo import selfcal

    grid = _opacity_grid()
    masks = {pid: np.array([[False, True], [True, True]]) for pid in cfg.position_ids}
    changed = {pid: counts.copy() for pid, counts in grid.counts.items()}
    for pid in cfg.position_ids:
        changed[pid][~masks[pid]] = 10**8
    c = replace(cfg, selfcal=replace(cfg.selfcal, baseline_m=None))
    monkeypatch.setattr(selfcal, "cross_position_score", lambda data, *_: float(data.lam.sum()))

    def minimize(objective, initial, **kwargs):
        value = objective(initial)
        result = initial.copy()
        result[0] += value * 0.001
        return OptimizeResult(success=True, fun=value, x=result, nfev=1)

    monkeypatch.setattr(selfcal.optimize, "minimize", minimize)
    first = selfcal.pose_bootstrap(grid, c, LIVE, n=3, seed=3, observation_mask=masks)
    second = selfcal.pose_bootstrap(
        AnalysisGrid(grid.edges, changed), c, LIVE, n=3, seed=3, observation_mask=masks
    )
    assert first == second


@pytest.mark.parametrize("pid", ["SKY", "pos_typo"])
def test_observation_masks_are_room_only_and_reject_unknown_sources(cfg, pid):
    with pytest.raises(ValueError, match="room"):
        solve_opacity(
            _opacity_grid(), cfg, LIVE, observation_mask={pid: np.ones((2, 2), dtype=bool)}
        )


@pytest.mark.parametrize(
    "live", [{"SKY": 1.0}, {"pos0": 0, "pos1": 1, "SKY": 1}, {"pos0": 1, "pos1": 1, "SKY": np.inf}]
)
def test_split_rejects_invalid_live_times(live):
    from cafetomo.validation_split import split_counts

    with pytest.raises(ValueError):
        split_counts(_partition_grid(), live, fraction=0.5, seed=0)
