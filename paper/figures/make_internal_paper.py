"""Select reviewed pilot panels and separate diagnostic scalars; never refit baseline data."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from internal_figure_utils import Archive, begin_output, check_volume, save, world_panel
from make_support import main_coverage_mask, voxel_panel
from style import plt


def summarize(prediction, recovery):
    result = dict(scope="retrospective pilot; conditional internal evidence")
    for protocol in ("count", "angular"):
        folds = [r for r in prediction["folds"] if r["protocol"] == protocol]
        if not folds:
            raise ValueError("missing declared prediction protocol")
        deviance = sum(v["deviance"] for r in folds for v in r["positions"].values())
        slab = sum(r["slab_deviance"] for r in folds)
        covered = sum(v["n_covered"] for r in folds for v in r["positions"].values())
        result.update(
            {
                f"{protocol}_folds": len(folds),
                f"{protocol}_deviance": deviance,
                f"{protocol}_slab_deviance": slab,
                f"{protocol}_covered_bins": covered,
                f"{protocol}_improvement_percent": 100 * (1 - deviance / slab),
                f"{protocol}_worse_folds": sum(
                    sum(v["deviance"] for v in r["positions"].values()) > r["slab_deviance"]
                    for r in folds
                ),
            }
        )
    cases = recovery["cases"]
    thicknesses = sorted({r["case"]["thickness_m"] for r in cases})
    if len(thicknesses) != 2:
        raise ValueError("selected pilot summary requires both registered recovery scenes")
    for label, thickness in zip(("thin", "thick"), thicknesses, strict=True):
        records = [r for r in cases if r["case"]["thickness_m"] == thickness]
        errors = np.array(
            [b["error_xyz_m"] for r in records for b in r["voxel_beam_localization"]["beams"]],
            dtype=float,
        )
        result.update(
            {
                f"{label}_replicas": len(records),
                f"{label}_truth_thickness": thickness,
                f"{label}_boundary": sum(r["status"] == "boundary" for r in records),
                f"{label}_z_bias": float(np.nanmean(errors[:, 2])),
            }
        )
    result["recovery_outcomes"] = len(cases)
    result["excess_peak_outcomes"] = sum(r["beams"]["n_beams_x"] > 4 for r in cases)
    return result


def bind_baseline_inputs(summary, nominal_path, baseline_results):
    """Bind unchanged baseline macro inputs to the explicit nominal field."""
    baseline_results = Path(baseline_results).resolve()
    reconstruction = json.loads((baseline_results / "reconstruction.json").read_text())
    with np.load(nominal_path, allow_pickle=False) as nominal:
        info = json.loads(str(nominal["meta"]))["info"]
        if (
            tuple(nominal["shape"]) != tuple(reconstruction[a] for a in ("nx", "ny", "nz"))
            or int(np.prod(nominal["shape"])) != reconstruction["voxels"]
            or float(nominal["spacing"]) != reconstruction["spacing"]
            or info["n_rows_used"] != reconstruction["rows"]
            or info["best_chi2"] != reconstruction["chisq_per_row"]
        ):
            raise ValueError("nominal volume and baseline reconstruction numbers differ")
    summary["baseline_inputs"] = [
        dict(
            path=path.name,
            source_path=logical_path(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(baseline_results.glob("*.json"))
        if path.name != "internalvalidation.json"
    ]


def verify_baseline_inputs(summary, baseline_results):
    baseline_results = Path(baseline_results).resolve()
    expected = {item["path"]: item["sha256"] for item in summary["baseline_inputs"]}
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in baseline_results.glob("*.json")
        if path.name != "internalvalidation.json"
    }
    if expected != actual:
        raise ValueError(
            "baseline macro inputs differ from the reviewed nominal selection; "
            "use explicit matching sources and a separately reviewed selection"
        )


def logical_path(path):
    """Use checkout-relative identities for sources inside the current checkout."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def selection_arguments(root, correction, main_volume, main_coverage, overlay):
    return {
        "archive": logical_path(root),
        "correction": logical_path(correction),
        "main_volume": logical_path(main_volume),
        "main_coverage": logical_path(main_coverage),
        "overlay": logical_path(overlay),
    }


def selected_input_files(archive):
    return [
        {**item, "path": logical_path(item["path"])}
        if item["location"] == "explicit"
        else item.copy()
        for item in (archive.input_files[name] for name in sorted(archive.input_files))
    ]


def verify_selected_inputs(
    summary, root, correction, main_volume, main_coverage, overlay, baseline_results
):
    """Validate every consumed byte and the chosen source roles before publication."""
    expected_arguments = selection_arguments(root, correction, main_volume, main_coverage, overlay)
    if summary.get("selection_arguments") != expected_arguments:
        raise ValueError("reviewed selection arguments differ; promote a fresh selection")
    expected = summary["input_files"]
    if len(expected) != 16 or len({(item["location"], item["path"]) for item in expected}) != 16:
        raise ValueError("reviewed selection must bind all 16 distinct input files")
    for item in expected:
        if item["location"] == "archive":
            path = Path(root) / item["path"]
        elif item["location"] == "explicit":
            path = Path(item["path"])
        else:
            raise ValueError("unknown reviewed input location")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"reviewed input missing: {path}") from exc
        if (
            len(payload) != item["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"reviewed input changed: {path}")
    verify_baseline_inputs(summary, baseline_results)


def write_summary_macros(source, out):
    """Render only promoted JSON scalars; percentages keep two decimal places."""
    document = json.loads(Path(source).read_text())
    lines = ["% Generated from results/internalvalidation.json; conditional pilot scalars."]
    for key, value in sorted(document.items()):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not np.isfinite(value):
            raise ValueError("promoted diagnostic scalar must be finite")
        name = "Internal" + "".join(part.capitalize() for part in key.split("_"))
        text = (
            str(value)
            if isinstance(value, int)
            else (f"{value:.2f}" if key.endswith(("percent", "z_bias")) else f"{value:.3g}")
        )
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    Path(out).write_text("\n".join(lines) + "\n")


def generate(
    root,
    correction,
    main_volume,
    main_coverage,
    overlay,
    out,
    summary_path,
    baseline_results="results",
):
    archive = Archive(root)
    prediction, recovery = archive.result("prediction"), archive.result("recovery")
    archive.result("support")
    stability = archive.result("stability")
    correction = Path(correction).resolve()
    registration = archive.json(correction / "registration.json")
    comparison = archive.json(correction / "comparison.json")
    if (
        registration["original_manifest_sha256"] != archive.input_files["manifest.json"]["sha256"]
        or not comparison["frozen_pose_reproduced"]
        or not comparison["baseline_weight_hash_reproduced"]
    ):
        raise ValueError("corrected comparison must reproduce the registered original pilot scope")
    choices = archive.json(correction / "choices/manifest.json")
    nominal = archive.npz(main_volume, ["rho", "origin", "spacing", "shape"])
    grid = dict(
        origin=nominal["origin"].tolist(),
        spacing=float(nominal["spacing"]),
        shape=nominal["shape"].tolist(),
    )
    coverage = main_coverage_mask(archive.npy(main_coverage))
    check_volume(nominal["rho"], coverage, grid)
    author_image = plt.imread(io.BytesIO(archive.bytes(overlay)), format=Path(overlay).suffix[1:])
    fold_path, fold, fold_grid = archive.fold(stability["scope"])
    support = archive.npz(fold_path / "support-degree2-threshold3.npz", ["joint", "coverage"])
    count_meta = archive.json(archive.root / "stability/count/manifest.json")
    count = archive.npz(archive.root / "stability/count/stability.npz", ["frequency", "exposure"])
    corrected = archive.npz(
        correction / "choices/beam-excess-weights.npz", ["frequency", "exposure"]
    )
    summary = summarize(prediction, recovery)
    summary.update(
        count_replicas=stability["count_replicas"],
        count_train_fraction=archive.manifest["settings"]["count"]["train_fraction"],
        prediction_uncertainty_replicas=fold["uncertainty_replicas"],
        corrected_analytic_objective=choices["variants"][0]["solver_info"]["best_chi2"],
        corrected_poisson_objective=choices["variants"][1]["solver_info"]["best_chi2"],
        corrected_weight_peaks=len(choices["variants"][1]["variant_beams"]["peaks"]),
        original_weight_objective=comparison["original_weights"][1]["solver_info"]["best_chi2"],
    )
    out = begin_output(
        archive,
        out,
        dict(
            main_volume_scope="nominal full-data",
            stability_scope=stability["scope"],
            corrected_weight_scope="separately registered frozen-pose comparison",
            input_files_scope="selected panels and promoted scalars only",
        ),
    )
    voxel_panel(
        nominal["rho"],
        coverage,
        grid,
        out,
        "internal-main",
        "Nominal full-data effective opacity",
        compact=True,
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.1), layout="constrained")
    ax.imshow(author_image)
    ax.axis("off")
    save(fig, out, "internal-overlay")
    panels = [
        (support["joint"], support["coverage"], fold_grid, "Geometric support", "dimensionless", 1),
        (
            count["frequency"],
            count["exposure"] > 0,
            count_meta["replicas"][0]["grid"],
            "Count + pose persistence",
            "frequency",
            stability["count_replicas"],
        ),
        (
            corrected["frequency"],
            corrected["exposure"] > 0,
            choices["reference_grid"],
            "Corrected weight beam excess",
            "frequency",
            2,
        ),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(7, 5.2), layout="constrained")
    for col, (field, covered, lattice, title, unit, denominator) in enumerate(panels):
        for row, axis in enumerate((2, 1)):
            world_panel(
                axes[row, col],
                field,
                covered,
                lattice,
                axis,
                title if row == 0 else "Side view",
                unit,
                [0, 1],
            )
        exposure = (
            support["coverage"].astype(float)
            if col == 0
            else (count if col == 1 else corrected)["exposure"]
        )
        world_panel(
            axes[2, col],
            exposure,
            np.ones(exposure.shape, bool),
            lattice,
            2,
            "Top-view coverage" if col == 0 else "Top-view denominator",
            "coverage" if col == 0 else "covered outcomes",
            [0, denominator],
        )
    for ax in axes.flat:
        ax.title.set_fontsize(7)
        ax.xaxis.label.set_fontsize(7)
        ax.yaxis.label.set_fontsize(7)
        ax.tick_params(labelsize=7)
    for ax in fig.axes:
        ax.tick_params(labelsize=7)
        ax.yaxis.label.set_fontsize(7)
    save(fig, out, "internal-support-stability")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), layout="constrained")
    for protocol, marker in (("count", "o"), ("angular", "s")):
        records = [r for r in prediction["folds"] if r["protocol"] == protocol]
        x = np.arange(len(records))
        values = [
            sum(v["deviance"] for v in r["positions"].values())
            / sum(v["n_covered"] for v in r["positions"].values())
            for r in records
        ]
        baseline = [
            r["slab_deviance"] / sum(v["n_covered"] for v in r["positions"].values())
            for r in records
        ]
        axes[0].plot(x, values, marker=marker, label=f"{protocol}: voxel")
        axes[0].plot(x, baseline, marker=marker, linestyle="--", label=f"{protocol}: slab")
    axes[0].set(
        xlabel="Retrospective fold",
        ylabel="Held-out deviance / covered bin",
        title="Prediction by protocol",
    )
    axes[0].legend(fontsize=6)
    for i, thickness in enumerate(sorted({r["case"]["thickness_m"] for r in recovery["cases"]})):
        errors = np.array(
            [
                b["error_xyz_m"]
                for r in recovery["cases"]
                if r["case"]["thickness_m"] == thickness
                for b in r["voxel_beam_localization"]["beams"]
            ],
            dtype=float,
        )
        for a, color in enumerate(("#1b6ca8", "#d1495b", "#65843c")):
            axes[1].scatter(
                np.full(len(errors), i + (a - 1) * 0.15), errors[:, a], s=8, alpha=0.5, color=color
            )
    from matplotlib.lines import Line2D

    axes[1].legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=c, label=a)
            for a, c in zip("xyz", ("#1b6ca8", "#d1495b", "#65843c"), strict=True)
        ],
        fontsize=6,
    )
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["0.3 m truth", "1.8 m truth"],
        ylabel="Centroid error (m)",
        title="Truth-associated recovery",
    )
    save(fig, out, "internal-prediction-recovery")
    summary["input_files"] = selected_input_files(archive)
    summary["selection_arguments"] = selection_arguments(
        root, correction, main_volume, main_coverage, overlay
    )
    bind_baseline_inputs(summary, main_volume, baseline_results)
    summary["original_archive"] = logical_path(archive.root)
    summary["corrected_comparison"] = logical_path(correction)
    summary_path = Path(summary_path)
    if summary_path.exists() and json.loads(summary_path.read_text()) != summary:
        raise ValueError("refusing to replace a different promoted diagnostic summary")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    settings_path = out / "display-settings.json"
    settings = json.loads(settings_path.read_text())
    settings["source_archive"] = logical_path(archive.root)
    settings["input_files"] = summary["input_files"]
    settings["selection_arguments"] = summary["selection_arguments"]
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    write_summary_macros(summary_path, out / "internal-numbers.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "archive",
        "correction",
        "main-volume",
        "main-coverage",
        "overlay",
        "out",
        "summary",
    ):
        parser.add_argument(f"--{name}", required=name != "out")
    parser.add_argument("--baseline-results", default="results")
    parser.add_argument("--verify-selection", action="store_true")
    args = parser.parse_args()
    if args.verify_selection:
        verify_selected_inputs(
            json.loads(Path(args.summary).read_text()),
            args.archive,
            args.correction,
            args.main_volume,
            args.main_coverage,
            args.overlay,
            args.baseline_results,
        )
    else:
        if args.out is None:
            parser.error("--out is required when generating a selection")
        generate(
            args.archive,
            args.correction,
            args.main_volume,
            args.main_coverage,
            args.overlay,
            args.out,
            args.summary,
            args.baseline_results,
        )
