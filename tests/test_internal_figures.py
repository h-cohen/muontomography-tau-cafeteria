"""Complete synthetic archive; no campaign execution or paper promotion."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"
sys.path.insert(0, str(FIGURES))


def load(name):
    path = FIGURES / f"make_{name}.py"
    assert path.exists(), f"figure entry point missing: {path.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    def js(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    root = tmp_path / "archive"
    grid = dict(origin=[-2, -1, 6], spacing=1.0, shape=[4, 2, 3])
    fold = dict(
        name="count-0",
        protocol="count",
        training=dict(grid=grid),
        positions={"pos0": {}, "pos1": {}},
        point_score_reference_scope="conditional on observed training roof split",
        uncertainty_scope="representative first outer fold per protocol",
        uncertainty_replicas=2,
    )
    js(
        root / "manifest.json",
        dict(
            status="complete",
            mode="pilot",
            folds=[fold],
            settings=dict(stability=dict(contrast_threshold=0.1)),
        ),
    )
    directory = root / "folds" / "count-0"
    js(directory / "fold.json", fold)
    rho = np.arange(24).reshape(4, 2, 3) / 100
    np.save(directory / "rho.npy", rho)
    coverage = np.ones(rho.shape, bool)
    coverage[0] = False
    np.savez(
        directory / "support-degree2-threshold3.npz",
        joint=rho > 0.1,
        coverage=coverage,
        pos0=np.ones((3, 3), bool),
        pos1=np.ones((3, 3), bool),
    )
    observed = np.arange(9).reshape(3, 3) + 10
    covered = np.ones((3, 3), bool)
    covered[0, 0] = False
    predicted = observed - 1.0
    predicted[0, 0] = np.nan
    for pid in fold["positions"]:
        np.savez(
            directory / f"{pid}-prediction.npz",
            observed=observed,
            predicted=predicted,
            residual=np.where(covered, 1.0, np.nan),
            coverage=covered,
        )
        np.savez(
            directory / f"{pid}-conditional-uncertainty.npz",
            mean=predicted,
            std=np.where(covered, 2.0, np.nan),
            n_supported=covered * 2,
            n_replicas=2,
        )
    np.savez(directory / "test-counts.npz", edges=np.linspace(-1, 1, 4))
    js(root / "results" / "prediction.json", dict(folds=[fold]))
    js(
        root / "results" / "support.json",
        dict(
            folds=[
                dict(
                    fold="count-0", sensitivity=[dict(degree=2, threshold=3.0, jointly_covered=18)]
                )
            ]
        ),
    )
    cases = [
        dict(
            case=dict(case_id="nominal"),
            status="boundary",
            at_bound=True,
            voxel_beam_localization=dict(beams=[dict(error_xyz_m=[0.1, None, 0.4])]),
        ),
        dict(case=dict(case_id="null"), status="fit_failure", failure="insufficient counts"),
    ]
    js(root / "results" / "recovery.json", dict(cases=cases))
    frequency = np.where(coverage, 0.5, np.nan)
    exposure = coverage * 2
    for path in ("count/stability", "choices/algorithm", "choices/beam-excess-algorithm"):
        target = root / "stability" / f"{path}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, frequency=frequency, exposure=exposure)
    js(
        root / "stability" / "count" / "manifest.json",
        dict(replicas=[dict(grid=grid)], contrast_threshold=0.1, n_replicas=2),
    )
    js(
        root / "stability" / "choices" / "manifest.json",
        dict(
            reference_grid=grid,
            variants=[dict(group="algorithm", centroid_change_xyz_m=[0.1, 0.0, 0.2], choice={})],
        ),
    )
    js(root / "results" / "stability.json", dict(count_replicas=2))
    return root


@pytest.mark.parametrize("name", ["internal_prediction", "support", "recovery", "stability"])
def test_complete_archive_generates_figures(bundle, tmp_path, name):
    out = tmp_path / name
    load(name).generate(bundle, out)
    assert list(out.glob("*.pdf"))
    metadata = json.loads((out / "display-settings.json").read_text())
    assert metadata["source_archive"] == str(bundle.resolve())
    assert metadata["missing_coverage"] == "unmeasured"


@pytest.mark.parametrize("name", ["internal_prediction", "support", "recovery", "stability"])
def test_incomplete_campaign_never_creates_publishable_figure(bundle, tmp_path, name):
    manifest = bundle / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"complete"', '"incomplete"'))
    with pytest.raises(ValueError, match="complete"):
        load(name).generate(bundle, tmp_path / name)
    assert not (tmp_path / name).exists()


def test_missing_required_prediction_fails_without_stale_fallback(bundle, tmp_path):
    (bundle / "folds/count-0/pos1-prediction.npz").unlink()
    out = tmp_path / "prediction"
    with pytest.raises(FileNotFoundError):
        load("internal_prediction").generate(bundle, out)
    assert not out.exists()


def test_projection_preserves_missing_coverage_and_world_extent():
    load("support")
    from internal_figure_utils import extent, projection

    field = np.ones((4, 2, 3))
    mask = np.ones(field.shape, bool)
    mask[0] = False
    image = projection(field, mask, 2)
    assert np.isnan(image[:, 0]).all()
    assert np.all(image[:, 1:] == 1)
    assert extent(dict(origin=[-2, -1, 6], spacing=1.0, shape=[4, 2, 3]), [0, 2]) == [-2, 2, 6, 9]


def test_author_overlay_is_separate_and_original_bytes_preserved(bundle, tmp_path):
    from PIL import Image

    overlay = tmp_path / "author.jpeg"
    Image.new("RGB", (50, 30), "white").save(overlay)
    original = overlay.read_bytes()
    out = tmp_path / "support-overlay"
    load("support").generate(bundle, out, overlay=overlay)
    assert overlay.read_bytes() == original
    assert (out / "conditional-author-overlay.pdf").exists()
    assert (out / "count-0-training-voxel-only.pdf").exists()


def test_missing_support_sensitivity_artifact_is_not_skipped(bundle, tmp_path):
    (bundle / "folds/count-0/support-degree2-threshold3.npz").unlink()
    out = tmp_path / "support"
    with pytest.raises(FileNotFoundError):
        load("support").generate(bundle, out)
    assert not out.exists()


def test_smoke_campaign_is_not_publishable(bundle, tmp_path):
    path = bundle / "manifest.json"
    path.write_text(path.read_text().replace('"pilot"', '"smoke"'))
    with pytest.raises(ValueError, match="smoke"):
        load("recovery").generate(bundle, tmp_path / "recovery")


def test_main_volume_has_explicit_full_data_scope(bundle, tmp_path):
    volume = tmp_path / "full.npz"
    np.savez(volume, rho=np.ones((2, 2, 2)) * 0.2, origin=[0, 0, 5], spacing=1.0, shape=[2, 2, 2])
    coverage = tmp_path / "views.npy"
    np.save(coverage, np.full((2, 2, 2), 2, dtype=np.int16))
    out = tmp_path / "main"
    load("support").generate(bundle, out, main_volume=volume, main_coverage=coverage)
    metadata = json.loads((out / "display-settings.json").read_text())
    assert metadata["main_volume_scope"] == "nominal full-data reconstruction"
    assert metadata["main_volume"] == str(volume.resolve())
    assert (out / "nominal-full-data-voxel-only.pdf").exists()


def test_integer_main_views_require_both_positions():
    load("support")
    from make_support import main_coverage_mask

    assert main_coverage_mask(np.array([0, 1, 2], dtype=np.int16)).tolist() == [False, False, True]


def test_missing_representative_uncertainty_fails(bundle, tmp_path):
    (bundle / "folds/count-0/pos1-conditional-uncertainty.npz").unlink()
    with pytest.raises(FileNotFoundError):
        load("internal_prediction").generate(bundle, tmp_path / "missing-sd")
    assert not (tmp_path / "missing-sd").exists()
