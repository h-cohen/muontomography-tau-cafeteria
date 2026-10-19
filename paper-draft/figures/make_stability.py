"""Conditional counting persistence and choice sensitivity, with coverage denominators."""

import json

import numpy as np

from internal_figure_utils import (
    Archive,
    begin_output,
    check_volume,
    cli,
    save,
    world_panel,
)
from style import plt


def generate(root, out):
    archive = Archive(root)
    result = archive.result("stability")
    scope = result["scope"]
    count_folds = [fold["name"] for fold in archive.folds if fold["protocol"] == "count"]
    if not count_folds or scope != count_folds[0]:
        raise ValueError("stability scope must be the declared first count-training fold")
    fraction = archive.manifest["settings"]["count"]["train_fraction"]
    count_dir = archive.root / "stability" / "count"
    count = archive.json(count_dir / "manifest.json")
    choices_dir = archive.root / "stability" / "choices"
    choices = archive.json(choices_dir / "manifest.json")
    entries = [
        (
            "Conditional count: raw density persistence",
            count_dir / "stability.npz",
            count["replicas"][0]["grid"],
        )
    ]
    groups = sorted({r["choice"]["group"] for r in choices["variants"]})
    for group in groups:
        entries.append(
            (
                f"{group}: "
                + (
                    "geometric support persistence"
                    if group == "support"
                    else "raw density persistence"
                ),
                choices_dir / f"{group}.npz",
                choices["reference_grid"],
            )
        )
        if group != "support":
            entries.append(
                (
                    f"{group}: frozen-background beam excess",
                    choices_dir / f"beam-excess-{group}.npz",
                    choices["reference_grid"],
                )
            )
    loaded = []
    for name, path, grid in entries:
        data = archive.npz(path, ["frequency", "exposure"])
        coverage = data["exposure"] > 0
        check_volume(data["frequency"], coverage, grid)
        if np.any((data["frequency"][coverage] < 0) | (data["frequency"][coverage] > 1)):
            raise ValueError("frequency outside [0,1]")
        loaded.append((name, data, coverage, grid))
    out = begin_output(
        archive,
        out,
        dict(
            training_fold_scope=scope,
            room_and_roof_train_fraction=fraction,
            reconstruction_settings_scope="selected settings of first count-training fold",
            world_grid_scopes=dict(
                count=count["replicas"][0]["grid"], choices=choices["reference_grid"]
            ),
            count_density_threshold=count["contrast_threshold"],
            frequency="fraction of covered replicas/variants; not posterior confidence",
            beam_background="frozen baseline training per-z median; raw density separate",
        ),
    )
    for index, (name, data, coverage, grid) in enumerate(loaded):
        fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), layout="constrained")
        for col, axis in enumerate((2, 1)):
            view = "Top" if axis == 2 else "Side"
            world_panel(
                axes[0, col],
                data["frequency"],
                coverage,
                grid,
                axis,
                f"{view}: persistence",
                "covered frequency",
                [0, 1],
            )
            world_panel(
                axes[1, col],
                data["exposure"].astype(float),
                np.ones(coverage.shape, bool),
                grid,
                axis,
                f"{view}: denominator",
                "covered outcomes",
                [0, max(1, data["exposure"].max())],
            )
        fig.suptitle(
            f"{name}\nFirst count-training fold {scope}: {fraction:g} room/roof fraction\n"
            "Frequency is not posterior confidence; grey is unmeasured",
            fontsize=8,
        )
        save(fig, out, f"stability-{index:02d}")
    variants = choices["variants"]
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    for axis in range(3):
        changes = [
            np.nan
            if r.get("centroid_change_xyz_m") is None or r["centroid_change_xyz_m"][axis] is None
            else r["centroid_change_xyz_m"][axis]
            for r in variants
        ]
        ax.plot(np.arange(len(variants)), changes, "o-", label="xyz"[axis])
    ax.axhline(0, color="grey", lw=0.7)
    ax.set(
        ylabel="Centroid change from baseline (m)",
        xlabel="Predeclared variant index",
        title=(
            f"First count-training fold {scope}: parameter sensitivity\n"
            "NaN unmeasured; support variants have no density centroid"
        ),
    )
    ax.legend()
    save(fig, out, "choice-sensitivity")
    (out / "variants.json").write_text(json.dumps(variants, indent=2) + "\n")


if __name__ == "__main__":
    cli(generate)
