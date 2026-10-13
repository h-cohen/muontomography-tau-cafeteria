"""Conditional counting persistence and choice sensitivity, with coverage denominators."""

import json

import numpy as np

from internal_figure_utils import (
    Archive,
    begin_output,
    check_volume,
    cli,
    read_json,
    read_npz,
    save,
    world_panel,
)
from style import plt


def generate(root, out):
    archive = Archive(root)
    archive.result("stability")
    count_dir = archive.root / "stability" / "count"
    count = read_json(count_dir / "manifest.json")
    choices_dir = archive.root / "stability" / "choices"
    choices = read_json(choices_dir / "manifest.json")
    entries = [
        (
            "Conditional count: raw density persistence",
            count_dir / "stability.npz",
            count["replicas"][0]["grid"],
        )
    ]
    groups = sorted({r["group"] for r in choices["variants"]})
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
        data = read_npz(path, ["frequency", "exposure"])
        coverage = data["exposure"] > 0
        check_volume(data["frequency"], coverage, grid)
        if np.any((data["frequency"][coverage] < 0) | (data["frequency"][coverage] > 1)):
            raise ValueError("frequency outside [0,1]")
        loaded.append((name, data, coverage, grid))
    out = begin_output(
        archive,
        out,
        dict(
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
            name + "\nFrequency is not posterior confidence; grey is unmeasured", fontsize=8
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
        title="Parameter sensitivity; NaN is unmeasured, support variants have no density centroid",
    )
    ax.legend()
    save(fig, out, "choice-sensitivity")
    (out / "variants.json").write_text(json.dumps(variants, indent=2) + "\n")


if __name__ == "__main__":
    cli(generate)
