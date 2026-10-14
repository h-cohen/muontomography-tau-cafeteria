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
            settings=dict(
                stability=dict(contrast_threshold=0.1),
                features=dict(background_degree=[2], signal_to_error=[3.0]),
                count=dict(train_fraction=0.5),
            ),
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
            variants=[dict(centroid_change_xyz_m=[0.1, 0.0, 0.2], choice=dict(group="algorithm"))],
        ),
    )
    js(root / "results" / "stability.json", dict(count_replicas=2, scope="count-0"))
    return root


@pytest.mark.parametrize("name", ["internal_prediction", "support", "recovery", "stability"])
def test_complete_archive_generates_figures(bundle, tmp_path, name):
    out = tmp_path / name
    load(name).generate(bundle, out)
    assert list(out.glob("*.pdf"))
    metadata = json.loads((out / "display-settings.json").read_text())
    assert metadata["source_archive"] == str(bundle.resolve())
    assert metadata["missing_coverage"] == "unmeasured"
    if name == "stability":
        assert metadata["training_fold_scope"] == "count-0"
        assert metadata["room_and_roof_train_fraction"] == 0.5
        assert metadata["world_grid_scopes"]["choices"]["origin"] == [-2, -1, 6]
    if name == "internal_prediction":
        assert metadata["summed_deviance_scope"] == "bookkeeping over reused observations"
    sources = {item["path"]: item for item in metadata["input_files"]}
    expected = {
        "internal_prediction": {
            "manifest.json",
            "results/prediction.json",
            "folds/count-0/fold.json",
            "folds/count-0/test-counts.npz",
            "folds/count-0/pos0-prediction.npz",
            "folds/count-0/pos1-prediction.npz",
            "folds/count-0/pos0-conditional-uncertainty.npz",
            "folds/count-0/pos1-conditional-uncertainty.npz",
        },
        "support": {
            "manifest.json",
            "results/support.json",
            "folds/count-0/fold.json",
            "folds/count-0/rho.npy",
            "folds/count-0/support-degree2-threshold3.npz",
        },
        "recovery": {"manifest.json", "results/recovery.json"},
        "stability": {
            "manifest.json",
            "results/stability.json",
            "stability/count/manifest.json",
            "stability/count/stability.npz",
            "stability/choices/manifest.json",
            "stability/choices/algorithm.npz",
            "stability/choices/beam-excess-algorithm.npz",
        },
    }
    assert set(sources) == expected[name]
    for relative, item in sources.items():
        import hashlib

        assert item["sha256"] == hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
        assert item["location"] == "archive"
        assert item["size_bytes"] == (bundle / relative).stat().st_size


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
    metadata = json.loads((out / "display-settings.json").read_text())
    assert any(
        item["path"] == str(overlay.resolve()) and item["location"] == "explicit"
        for item in metadata["input_files"]
    )
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
    sources = {item["path"]: item for item in metadata["input_files"]}
    assert sources[str(volume.resolve())]["sha256"] == metadata["main_volume_sha256"]
    assert sources[str(coverage.resolve())]["sha256"] == metadata["main_coverage_sha256"]
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


def test_consumed_input_hash_changes_when_prediction_changes(bundle, tmp_path):
    import hashlib

    module = load("internal_prediction")
    out = tmp_path / "before"
    module.generate(bundle, out)

    def inventory(directory):
        settings = json.loads((directory / "display-settings.json").read_text())
        assert "input_files" in settings, "settings must enumerate all consumed inputs"
        return {item["path"]: item["sha256"] for item in settings["input_files"]}

    before = inventory(out)
    relative = "folds/count-0/pos0-prediction.npz"
    path = bundle / relative
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["observed"] += 2
    arrays["residual"] = np.where(arrays["coverage"], arrays["residual"] + 2, np.nan)
    np.savez(path, **arrays)
    module.generate(bundle, tmp_path / "after")
    after = inventory(tmp_path / "after")
    assert before[relative] != after[relative]
    assert after[relative] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(after) == {
        "manifest.json",
        "results/prediction.json",
        "folds/count-0/fold.json",
        "folds/count-0/test-counts.npz",
        relative,
        "folds/count-0/pos1-prediction.npz",
        "folds/count-0/pos0-conditional-uncertainty.npz",
        "folds/count-0/pos1-conditional-uncertainty.npz",
    }


@pytest.mark.parametrize(
    "kind", ["empty", "omitted", "duplicate", "missing_sensitivity", "duplicate_sensitivity"]
)
def test_support_declared_folds_and_sensitivity_are_complete(bundle, tmp_path, kind):
    path = bundle / "results/support.json"
    result = json.loads(path.read_text())
    if kind == "empty":
        result["folds"] = []
    elif kind == "omitted":
        manifest = bundle / "manifest.json"
        document = json.loads(manifest.read_text())
        document["folds"].append(dict(document["folds"][0], name="angular-0"))
        manifest.write_text(json.dumps(document))
    elif kind == "duplicate":
        result["folds"] *= 2
    elif kind == "missing_sensitivity":
        result["folds"][0]["sensitivity"] = []
    else:
        result["folds"][0]["sensitivity"] *= 2
    path.write_text(json.dumps(result))
    out = tmp_path / kind
    with pytest.raises(ValueError, match="declared|duplicate|missing"):
        load("support").generate(bundle, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "kind",
    [
        "shape",
        "negative_sd",
        "nonfinite_sd",
        "negative_support",
        "too_many_support",
        "fractional_support",
        "unsupported_sd",
    ],
)
def test_uncertainty_validation_precedes_any_output(bundle, tmp_path, kind):
    path = bundle / "folds/count-0/pos1-conditional-uncertainty.npz"
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    if kind == "shape":
        arrays["n_supported"] = arrays["n_supported"][:1]
    elif kind in {"negative_sd", "nonfinite_sd"}:
        arrays["std"][1, 1] = -1 if kind == "negative_sd" else np.nan
    elif kind == "unsupported_sd":
        arrays["std"][0, 0] = 1
    else:
        arrays["n_supported"] = arrays["n_supported"].astype(float)
        arrays["n_supported"][1, 1] = {
            "negative_support": -1,
            "too_many_support": 3,
            "fractional_support": 1.5,
        }[kind]
    np.savez(path, **arrays)
    out = tmp_path / kind
    with pytest.raises(ValueError, match="uncertainty"):
        load("internal_prediction").generate(bundle, out)
    assert not out.exists()


def test_xyz_legend_survives_first_case_failure(bundle, tmp_path, monkeypatch):
    module = load("recovery")
    path = bundle / "results/recovery.json"
    result = json.loads(path.read_text())
    result["cases"].reverse()
    path.write_text(json.dumps(result))
    # Inspect real plotted artists at the save boundary; no replacement plotting logic.
    actual_save = module.save
    captured = {}

    def inspect_save(fig, out, name):
        captured["coordinates"] = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        captured["outcomes"] = [text.get_text() for text in fig.axes[1].get_legend().get_texts()]
        actual_save(fig, out, name)

    monkeypatch.setattr(module, "save", inspect_save)
    module.generate(bundle, tmp_path / "reversed")
    assert captured["coordinates"] == ["x", "y", "z"]
    assert set(captured["outcomes"]) == {"boundary", "fit_failure"}


def test_paper_summary_keeps_protocols_separate_and_all_outcomes():
    module = load("internal_paper")

    def fold(protocol, deviance, slab):
        return dict(
            protocol=protocol,
            slab_deviance=slab,
            positions=dict(pos0=dict(deviance=deviance, n_covered=10)),
        )

    prediction = dict(
        folds=[
            fold("count", 2, 20),
            fold("count", 24, 80),
            fold("angular", 12, 10),
            fold("angular", 8, 20),
        ]
    )
    recovery = dict(
        cases=[
            dict(
                case=dict(thickness_m=0.3),
                status="boundary",
                beams=dict(n_beams_x=5),
                voxel_beam_localization=dict(beams=[dict(error_xyz_m=[0.01, 0, -1])]),
            ),
            dict(
                case=dict(thickness_m=1.8),
                status="ok",
                beams=dict(n_beams_x=4),
                voxel_beam_localization=dict(beams=[dict(error_xyz_m=[0.02, 0, -2])]),
            ),
        ]
    )
    summary = module.summarize(prediction, recovery)
    assert summary["count_improvement_percent"] == pytest.approx(74)
    assert summary["angular_worse_folds"] == 1
    assert summary["count_covered_bins"] == 20
    assert summary["thin_boundary"] == 1
    assert summary["excess_peak_outcomes"] == 1
    assert summary["thin_z_bias"] == -1
    assert summary["scope"] == "retrospective pilot; conditional internal evidence"


def test_selected_macros_read_promoted_json_with_percent_precision(tmp_path):
    module = load("internal_paper")
    source = tmp_path / "internalvalidation.json"
    source.write_text(
        json.dumps(
            dict(
                count_improvement_percent=59.70758,
                thin_boundary=4,
                scope="conditional",
                input_files=[],
            )
        )
    )
    out = tmp_path / "internal-numbers.tex"
    module.write_summary_macros(source, out)
    assert r"\newcommand{\InternalCountImprovementPercent}{59.71}" in out.read_text()
    assert r"\newcommand{\InternalThinBoundary}{4}" in out.read_text()


def test_publication_binding_rejects_changed_baseline_numbers(tmp_path):
    module = load("internal_paper")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    reconstruction = dict(rows=12, nx=2, ny=2, nz=2, voxels=8, spacing=0.2, chisq_per_row=0.5)
    source = baseline / "reconstruction.json"
    source.write_text(json.dumps(reconstruction))
    nominal = tmp_path / "nominal.npz"
    np.savez(
        nominal,
        rho=np.ones((2, 2, 2)),
        shape=[2, 2, 2],
        spacing=0.2,
        meta=json.dumps(dict(info=dict(n_rows_used=12, best_chi2=0.5))),
    )
    summary = dict(count_improvement_percent=59.71)
    module.bind_baseline_inputs(summary, nominal, baseline)
    module.verify_baseline_inputs(summary, baseline)
    source.write_text(source.read_text().replace('"rows": 12', '"rows": 13'))
    with pytest.raises(ValueError, match="baseline"):
        module.verify_baseline_inputs(summary, baseline)
    with pytest.raises(ValueError, match="nominal"):
        module.bind_baseline_inputs({}, nominal, baseline)


def test_baseline_binding_is_portable_between_checkouts(tmp_path):
    import shutil

    module = load("internal_paper")
    old = tmp_path / "old"
    old.mkdir()
    (old / "reconstruction.json").write_text(
        json.dumps(dict(rows=1, nx=1, ny=1, nz=1, voxels=1, spacing=0.2, chisq_per_row=0.5))
    )
    nominal = tmp_path / "nominal.npz"
    np.savez(
        nominal,
        shape=[1, 1, 1],
        spacing=0.2,
        meta=json.dumps(dict(info=dict(n_rows_used=1, best_chi2=0.5))),
    )
    summary = {}
    module.bind_baseline_inputs(summary, nominal, old)
    relocated = tmp_path / "relocated"
    shutil.copytree(old, relocated)
    module.verify_baseline_inputs(summary, relocated)


def test_selected_input_guard_checks_every_source_and_arguments_after_relocation(
    tmp_path, monkeypatch
):
    import hashlib
    import shutil

    module = load("internal_paper")
    old = tmp_path / "old"
    old.mkdir()
    monkeypatch.chdir(old)
    paths = {
        "archive": Path("runs/original"),
        "correction": Path("runs/corrected"),
        "main_volume": Path("runs/nominal/volume.npz"),
        "main_coverage": Path("runs/nominal/views.npy"),
        "overlay": Path("paper/figures/author.jpeg"),
    }
    sources = [("archive", "manifest.json")]
    sources.extend(("archive", f"results/source-{i}.json") for i in range(9))
    sources.extend(("explicit", f"runs/corrected/source-{i}.json") for i in range(6))
    inventory = []
    for location, name in sources:
        path = paths["archive"] / name if location == "archive" else Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = name.encode()
        path.write_bytes(payload)
        inventory.append(
            dict(
                path=name,
                location=location,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    baseline = Path("results")
    baseline.mkdir()
    (baseline / "reconstruction.json").write_text("{}")
    summary = dict(
        input_files=inventory, selection_arguments=module.selection_arguments(*paths.values())
    )
    summary["baseline_inputs"] = [
        dict(path="reconstruction.json", sha256=hashlib.sha256(b"{}").hexdigest())
    ]
    module.verify_selected_inputs(summary, *paths.values(), baseline)
    relocated = tmp_path / "relocated"
    shutil.copytree(old, relocated)
    monkeypatch.chdir(relocated)
    module.verify_selected_inputs(summary, *paths.values(), baseline)
    (relocated / "runs/corrected/source-5.json").write_text("changed")
    with pytest.raises(ValueError, match="reviewed input changed"):
        module.verify_selected_inputs(summary, *paths.values(), baseline)
    (relocated / "runs/corrected/source-5.json").unlink()
    with pytest.raises(ValueError, match="reviewed input missing"):
        module.verify_selected_inputs(summary, *paths.values(), baseline)
    with pytest.raises(ValueError, match="selection arguments differ"):
        module.verify_selected_inputs(
            summary,
            paths["archive"],
            paths["correction"],
            paths["main_volume"],
            paths["main_coverage"],
            "paper/figures/other.jpeg",
            baseline,
        )
