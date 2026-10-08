import numpy as np
import pytest

from cafetomo.arraydepth import mean_flux_opacity


def test_partially_covered_aperture_averages_flux_before_log():
    offsets = ((np.arange(64) + 0.5) / 64 - 0.5)[None, :] * 0.4
    lam = mean_flux_opacity(
        offsets,
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        z=7.0,
        w=0.2,
        h=1.0,
        centres=np.array([0.0]),
        columns=np.array([0.2]),
    )
    expected = -np.log((1 + np.exp(-0.2)) / 2)
    assert lam[0] == pytest.approx(expected)
    assert lam[0] < 0.1


def test_full_coverage_retains_oblique_path_factor():
    lam = mean_flux_opacity(
        np.zeros((1, 3)),
        np.array([0.1]),
        np.array([0.2]),
        np.array([0.0]),
        z=1.0,
        w=1.0,
        h=1.0,
        centres=np.array([0.0]),
        columns=np.array([0.2]),
    )
    assert lam[0] == pytest.approx(0.2 * np.sqrt(1.05))


def test_beam_behind_tracker_has_no_forward_path():
    lam = mean_flux_opacity(
        np.zeros((1, 3)),
        np.zeros(1),
        np.zeros(1),
        np.array([3.0]),
        z=1.0,
        w=1.0,
        h=1.0,
        centres=np.array([0.0]),
        columns=np.array([0.2]),
    )
    assert lam[0] == pytest.approx(0.0)


def test_flux_kernel_agrees_with_independent_three_dimensional_intersections():
    from cafetomo.beamdepth import box_path_lengths

    rng = np.random.default_rng(8)
    sx = np.array([0.0, -0.3, 0.25])
    sy = np.array([0.2, 0.1, -0.2])
    oz = np.array([0.0, 0.1, 0.2])
    offsets = rng.uniform(-0.2, 0.2, (3, 32))
    centres = np.array([-1.0, 0.0, 1.0])
    columns = np.array([0.1, 0.2, 0.15])
    starts = np.zeros((3, 32, 3))
    starts[:, :, 0] = offsets
    starts[:, :, 2] = oz[:, None]
    dirs = np.column_stack([sx, sy, np.ones(3)])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = np.broadcast_to(dirs[:, None, :], starts.shape)
    attenuation = np.zeros((3, 32))
    for x, column in zip(centres, columns, strict=True):
        attenuation += (
            column
            / 0.8
            * box_path_lengths(
                starts, dirs, np.array([x - 0.2, -5, 3]), np.array([x + 0.2, 5, 3.8])
            )
        )
    expected = -np.log(np.exp(-attenuation).mean(axis=1))
    actual = mean_flux_opacity(
        offsets, sx, sy, oz, z=3.0, w=0.4, h=0.8, centres=centres, columns=columns
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_study_summary_keeps_all_replicas_and_count_scope():
    from cafetomo.arraydepth import summarize_study

    record = summarize_study(
        {
            "nominal": {"h": 1.4, "w": 0.3, "bottom": 6.4, "chisq_per_dof": 2.0},
            "replicas": [
                {"h": 1.2, "w": 0.2, "bottom": 6.3},
                {"h": 1.6, "w": 0.4, "bottom": 6.5},
            ],
        }
    )
    assert record["h"] == 1.4
    assert record["h_sigma"] == pytest.approx(np.sqrt(0.08))
    assert record["n_replicas"] == 2
    assert record["conditional"] is True
    assert record["uncertainty_scope"] == "count spread with centres, array and pose fixed"


def test_study_summary_rejects_nonfinite_replicas():
    from cafetomo.arraydepth import summarize_study

    with pytest.raises(ValueError, match="non-finite"):
        summarize_study(
            {
                "nominal": {"h": 1.4},
                "replicas": [
                    {"h": 1.2, "w": 0.3, "bottom": 6.4},
                    {"h": float("nan"), "w": 0.3, "bottom": 6.4},
                ],
            }
        )


@pytest.mark.slow
@pytest.mark.parametrize("depth", [0.6, 1.2, 1.8])
def test_noiseless_array_depth_recovers_independent_ray_box_truth(depth):
    import json

    from cafetomo.arraydepth import fit_array_depth
    from cafetomo.beamdepth import box_path_lengths
    from cafetomo.config import load_config
    from cafetomo.fitdata import FitData, RowIndex
    from cafetomo.raycast import bundle_offsets

    cfg = load_config("configs/cafeteria.yaml")
    tx, ty = np.meshgrid(np.linspace(-0.7, 0.7, 40), np.linspace(-0.15, 0.15, 7))
    sx, sy = tx.ravel(), ty.ravel()
    rows = RowIndex(
        tuple(cfg.position_ids),
        np.repeat([0, 1], len(sx)),
        np.tile(sx, 2),
        np.tile(sy, 2),
        np.arange(2 * len(sx)),
    )
    origins = np.array([cfg.origins()[rows.position_ids[i]] for i in rows.pos_of_row])
    starts = origins[:, None, :] + bundle_offsets(
        rows.directions(),
        cfg.detector.aperture_m,
        16,
        layer_dz_m=cfg.detector.layer_dz_cm / 100,
    )
    dirs = np.broadcast_to(rows.directions()[:, None, :], starts.shape)
    centres = np.array([-3.2, -1.6, 0.0, 1.6, 3.2, 4.8])
    attenuation = np.zeros(starts.shape[:2])
    for center in centres:
        attenuation += (
            0.06
            / depth
            * box_path_lengths(
                starts,
                dirs,
                np.array([center - 0.15, -5, 6.4]),
                np.array([center + 0.15, 5, 6.4 + depth]),
            )
        )
    lam = -np.log(np.exp(-attenuation).mean(axis=1)) + 0.1 + 0.02 * rows.sy**2
    result = fit_array_depth(
        FitData(lam, np.full(len(lam), 25000.0), rows), cfg, centres=centres, height_seed=7.0
    )
    assert result["h"] == pytest.approx(depth, abs=0.02)
    assert result["w"] == pytest.approx(0.3, abs=0.01)
    json.dumps(result)


def test_boundary_replicas_are_flagged_and_retained():
    from cafetomo.arraydepth import summarize_study

    base = {
        "h": 1.4,
        "w": 0.3,
        "bottom": 6.4,
        "shift": 0.0,
        "normal_columns": [0.1],
        "lower_bounds": [4.35, 0.05, 0.05, -0.3, 0.0],
        "upper_bounds": [7.85, 1.0, 3.0, 0.3, 2.0],
        "n_sub": 32,
    }
    record = summarize_study({"nominal": base, "replicas": [base | {"h": 0.05}, base]})
    assert record["n_replicas"] == 2
    assert record["n_depth_boundary_replicas"] == 1
    assert record["replicas"][0]["bound_parameters"] == ["depth"]
    assert record["quadrature_major"] == 32
    assert record["quadrature_minor"] == 3
