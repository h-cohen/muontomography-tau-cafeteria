import numpy as np
import pytest

from cafetomo.reconstruct import VoxelSolution
from cafetomo.robustness import stability_map, validate_choices
from cafetomo.voxels import VoxelGrid


def solution(origin=(0, 0, 0), spacing=1, shape=(2, 1, 1)):
    return VoxelSolution(np.ones(np.prod(shape)), VoxelGrid(origin, spacing, shape), {}, ())


def test_identical_replicas_and_unmeasured():
    v = solution()
    f, n = stability_map([v, v], [0.5, 0.5], [np.array([1, 0], bool)] * 2, reference_grid=v.grid)
    assert f[0, 0, 0] == 1
    assert np.isnan(f[1, 0, 0])
    np.testing.assert_array_equal(n.ravel(), [2, 0])


def test_shifted_world_overlap():
    v = solution(origin=(0.5, 0, 0), shape=(1, 1, 1))
    f, n = stability_map([v], [0.5], [np.ones(1, bool)], reference_grid=solution().grid)
    np.testing.assert_allclose(f.ravel(), [1, 1])
    np.testing.assert_allclose(n.ravel(), [0.5, 0.5])


def test_missing_replicas_and_duplicates():
    with pytest.raises(ValueError):
        stability_map([], [], [], reference_grid=solution().grid)
    with pytest.raises(ValueError):
        stability_map([solution()], [], [], reference_grid=solution().grid)
    with pytest.raises(ValueError, match="duplicate"):
        validate_choices([{"tv_alpha": 0.03}, {"tv_alpha": 0.03}])


def test_coverage_uses_only_positive_weights(monkeypatch, cfg):
    from types import SimpleNamespace

    from scipy.sparse import csr_matrix

    import cafetomo.robustness as r

    monkeypatch.setattr(
        r, "build_forward_model", lambda *a, **kw: SimpleNamespace(A=csr_matrix([[1, 0], [0, 1]]))
    )
    data = SimpleNamespace(rows=None, w=np.array([1.0, 0]))
    np.testing.assert_array_equal(
        r.positive_weight_coverage(data, cfg, solution().grid).ravel(), [True, False]
    )


def test_spacing_overlap_and_partial_signal():
    v = solution(spacing=0.5, shape=(2, 2, 2))
    v.rho[:4] = 0
    f, n = stability_map(
        [v], [0.5], [np.ones(8, bool)], reference_grid=solution(shape=(1, 1, 1)).grid
    )
    assert f.item() == 0.5
    assert n.item() == 1


def test_bootstrap_replica_archive_is_opt_in():
    import inspect

    from cafetomo.bootstrap import run_bootstrap

    assert inspect.signature(run_bootstrap).parameters["replica_dir"].default is None


def test_support_choice_separate_from_density():
    from cafetomo.robustness import validate_choices

    with pytest.raises(ValueError, match="support"):
        validate_choices([dict(group="support", contrast_threshold=1)])


def test_poisson_weight_choice_requires_declared_convergence():
    from cafetomo.robustness import validate_choices

    with pytest.raises(ValueError, match="convergence"):
        validate_choices([dict(group="weights", weight_method="poisson", n_sigma_replicas=12)])


def test_count_failure_archived_and_not_silently_dropped(cfg, monkeypatch, tmp_path):
    import json

    import cafetomo.robustness as r
    from cafetomo.angular import AnalysisGrid

    grid = AnalysisGrid(
        np.array([-0.1, 0.1]),
        {p: np.ones((1, 1), dtype=int) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )

    def fail(*args, **kwargs):
        raise RuntimeError("invalid pose")

    monkeypatch.setattr(r, "fit_training_fold", fail)
    with pytest.raises(RuntimeError, match="invalid pose"):
        r.run_count_replicas(
            grid,
            {},
            cfg,
            n_replicas=5,
            seed=3,
            out_dir=tmp_path,
            contrast_threshold=0.1,
            pose_mode="reestimated_pose",
        )
    record = json.loads((tmp_path / "failures.json").read_text())
    assert record["requested"] == 5
    assert record["replicas"] == [dict(index=0, status="failed", error="invalid pose")]


def test_algorithm_threshold_frozen_across_variants():
    with pytest.raises(ValueError, match="frozen"):
        validate_choices(
            [
                dict(group="algorithm", contrast_threshold=0.1),
                dict(group="algorithm", contrast_threshold=0.2),
            ]
        )
