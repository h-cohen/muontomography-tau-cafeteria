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
        validate_choices([dict(group="algorithm", contrast_threshold=0.1)] * 2)


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


def test_rebuilt_weight_mismatch_rejected(monkeypatch, cfg):
    import hashlib
    from types import SimpleNamespace

    import cafetomo.robustness as r
    from cafetomo.angular import AnalysisGrid
    from cafetomo.prediction import FrozenFold

    grid = AnalysisGrid(
        np.array([-0.1, 0.1]),
        {p: np.ones((1, 1), int) for p in (*cfg.position_ids, cfg.sky_reference.id)},
    )
    original = np.array([1.0, 2.0])
    changed = np.array([1.0, 3.0])
    fold = FrozenFold(
        cfg,
        solution(),
        "train",
        "choice",
        {"weight_hash": hashlib.sha256(original.tobytes()).hexdigest()},
    )
    monkeypatch.setattr(r, "opacity_sigma", lambda *a, **kw: None)
    monkeypatch.setattr(r, "solve_opacity", lambda *a, **kw: None)
    monkeypatch.setattr(r, "build_fit_data", lambda *a: SimpleNamespace(w=changed))
    with pytest.raises(ValueError, match="rebuilt training weights differ"):
        r.training_data(grid, {p: 1.0 for p in grid.counts}, fold)


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


def test_beam_peak_association_retains_shifted_lost_and_new():
    from cafetomo.robustness import associate_beam_peaks

    baseline = [dict(x_m=1.0, z_centroid_m=5.0), dict(x_m=3.0, z_centroid_m=6.0)]
    variant = [dict(x_m=1.2, z_centroid_m=5.4), dict(x_m=6.0, z_centroid_m=7.0)]
    result = associate_beam_peaks(baseline, variant, match_tol_m=0.6)
    assert result["lost_baseline_indices"] == [1]
    assert result["new_variant_indices"] == [1]
    assert result["matches"][0]["delta_x_m"] == pytest.approx(0.2)
    assert result["matches"][0]["delta_z_centroid_m"] == pytest.approx(0.4)


def test_beam_diagnostics_frozen_prominence_and_height():
    from cafetomo.robustness import beam_diagnostics

    grid = VoxelGrid((0, 0, 4), 1.0, (5, 1, 3))
    rho = np.zeros(grid.shape)
    rho[1, 0, 1] = 2
    rho[3, 0, 2] = 3
    volume = VoxelSolution(rho.ravel(), grid, {}, ())
    result = beam_diagnostics(
        volume, np.ones(grid.shape, bool), 0.5, prominence=1.0, lateral_half_width_m=0.4
    )
    assert [p["x_m"] for p in result["peaks"]] == [1.5, 3.5]
    assert [p["z_centroid_m"] for p in result["peaks"]] == [5.5, 6.5]
    assert result["peaks"][0]["z_profile_extent_m"] == [5.0, 6.0]


@pytest.mark.parametrize(
    "choice",
    [
        dict(group="algorithm", contrast_threshold=0.1, weight_method="analytic"),
        dict(
            group="weights",
            contrast_threshold=0.1,
            weight_method="analytic",
            overrides={"reconstruction": {"tv_alpha": 0.06}},
        ),
        dict(
            group="support",
            background_degree=1,
            support_threshold=2,
            tangent_sigma=0,
            overrides={"volume": {"spacing_m": 0.25}},
        ),
        dict(group="display", contrast_threshold=0.1, lattice_shift=0.5),
        dict(group="display", contrast_threshold=0.1, support_threshold=2),
        dict(group="algorithm", contrast_threshold=0.1, overrides={"reconstruction": {"seed": 5}}),
        dict(
            group="weights", contrast_threshold=0.1, weight_method="analytic", n_sigma_replicas=12
        ),
    ],
)
def test_group_specific_rejects_confounded_or_ignored_options(choice):
    with pytest.raises(ValueError):
        validate_choices([choice])


@pytest.mark.parametrize(
    "alias",
    [
        dict(lattice_shift=0),
        dict(overrides={}),
        dict(overrides={"reconstruction": {}}),
        dict(overrides={"reconstruction": {"tv_alpha": 0.03}}),
        dict(overrides={"volume": {"spacing_m": 0.2}}),
        dict(label="different label"),
        dict(beam_background_rule="per_z_median"),
    ],
)
def test_effective_duplicate_aliases(alias):
    baseline = dict(group="algorithm", contrast_threshold=0.1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_choices([baseline, {**baseline, **alias}])


def test_effective_duplicates_use_supplied_cfg_and_ignore_evidence_prose(cfg):
    from dataclasses import replace

    custom = replace(cfg, reconstruction=replace(cfg.reconstruction, tv_alpha=0.06))
    with pytest.raises(ValueError, match="duplicate"):
        validate_choices(
            [
                dict(group="algorithm", contrast_threshold=0.1),
                dict(
                    group="algorithm",
                    contrast_threshold=0.1,
                    overrides={"reconstruction": {"tv_alpha": 0.06}},
                ),
            ],
            custom,
        )
    poisson = dict(
        group="weights",
        contrast_threshold=0.1,
        weight_method="poisson",
        n_sigma_replicas=24,
        convergence_evidence="predeclared criterion A",
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_choices([poisson, {**poisson, "convergence_evidence": "other prose"}])


def test_valid_distinct_choices_have_independent_groups():
    validate_choices(
        [
            dict(group="algorithm", contrast_threshold=0.1),
            dict(
                group="algorithm",
                contrast_threshold=0.1,
                overrides={"reconstruction": {"tv_alpha": 0.06}},
            ),
            dict(group="display", contrast_threshold=0.1),
            dict(group="weights", contrast_threshold=0.1, weight_method="analytic"),
            dict(group="support", background_degree=1, support_threshold=2, tangent_sigma=0),
        ]
    )


def test_frozen_beam_background_removes_uniform_slab_and_retains_excess():
    from cafetomo.robustness import beam_contrast_volume, freeze_beam_background

    grid = VoxelGrid((0, 0, 0), 1.0, (5, 1, 2))
    rho = np.ones(grid.shape)
    rho[2, 0, :] += 3
    volume = VoxelSolution(rho.ravel(), grid, {}, ())
    background = freeze_beam_background(volume, np.ones(grid.shape, bool))
    np.testing.assert_array_equal(background, np.ones(grid.shape))
    contrast = beam_contrast_volume(volume, background, grid)
    assert (contrast.rho > 0.1).sum() == 2
    null = VoxelSolution(np.ones(grid.n_voxels), grid, {}, ())
    assert np.all(beam_contrast_volume(null, background, grid).rho == 0)
    shifted = VoxelSolution(np.ones(grid.n_voxels), replace_grid(grid), {}, ())
    assert np.all(beam_contrast_volume(shifted, background, grid).rho == 0)


def replace_grid(grid):
    from dataclasses import replace

    return replace(grid, origin=(0.5, 0, 0))


def test_sweep_archives_separate_density_and_beam_excess_null(cfg, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import cafetomo.robustness as r
    from cafetomo.prediction import FrozenFold

    grid = VoxelGrid((0, 0, 0), 1.0, (5, 1, 2))
    volume = VoxelSolution(np.ones(grid.n_voxels), grid, {}, ())
    fold = FrozenFold(cfg, volume, "training", "choice", {"weight_hash": "weights", "pose": {}})
    data = SimpleNamespace(w=np.ones(1))
    monkeypatch.setattr(r, "fit_training_fold", lambda *a, **kw: fold)
    monkeypatch.setattr(r, "training_data", lambda *a, **kw: data)
    monkeypatch.setattr(r, "positive_weight_coverage", lambda *a, **kw: np.ones(grid.shape, bool))
    monkeypatch.setattr(r, "solve_voxels", lambda *a, **kw: {"full": volume})
    result = r.run_robustness(
        None, {}, cfg, [dict(group="algorithm", contrast_threshold=0.1)], seed=2, out_dir=tmp_path
    )
    assert result["variants"][0]["variant_beams"]["peaks"] == []
    assert "not beam-only" in result["density_frequency_interpretation"]
    np.testing.assert_array_equal(np.load(tmp_path / "algorithm.npz")["frequency"], 1)
    np.testing.assert_array_equal(np.load(tmp_path / "beam-excess-algorithm.npz")["frequency"], 0)


@pytest.mark.parametrize(
    "rho_value, measured", [(0.0, [False, True, False]), (2.0, [False, True, True, True, False])]
)
def test_beam_detection_does_not_turn_coverage_islands_into_peaks(rho_value, measured):
    from cafetomo.robustness import beam_diagnostics

    grid = VoxelGrid((0, 0, 0), 1.0, (len(measured), 1, 1))
    volume = VoxelSolution(np.full(grid.n_voxels, rho_value), grid, {}, ())
    if rho_value == 0:
        from cafetomo.robustness import beam_contrast_volume, freeze_beam_background

        slab = VoxelSolution(np.ones(grid.n_voxels), grid, {}, ())
        background = freeze_beam_background(slab, np.array(measured, bool).reshape(grid.shape))
        volume = beam_contrast_volume(slab, background, grid)
    result = beam_diagnostics(
        volume,
        np.array(measured, bool).reshape(grid.shape),
        0.1,
        prominence=0.0,
        lateral_half_width_m=0.4,
    )
    assert result["peaks"] == []


def test_localized_positive_beam_inside_measured_segment_near_gap():
    from cafetomo.robustness import beam_diagnostics

    grid = VoxelGrid((0, 0, 0), 1.0, (5, 1, 1))
    volume = VoxelSolution(np.array([0.0, 0.0, 3.0, 0.0, 0.0]), grid, {}, ())
    coverage = np.array([False, True, True, True, False]).reshape(grid.shape)
    result = beam_diagnostics(volume, coverage, 0.1, prominence=1.0, lateral_half_width_m=0.4)
    assert [p["x_m"] for p in result["peaks"]] == [2.5]
    assert result["peaks"][0]["z_centroid_m"] == 0.5
