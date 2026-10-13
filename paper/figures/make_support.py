"""Voxel-only world view and separate dimensionless geometric support comparison."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from internal_figure_utils import (
    DISPLAY,
    Archive,
    begin_output,
    check_volume,
    cli,
    save,
    world_panel,
)
from style import SEQUENTIAL, plt


def archive_source_key(archive, path):
    path = Path(path).resolve()
    return str(path.relative_to(archive.root)) if path.is_relative_to(archive.root) else str(path)


def main_coverage_mask(views):
    """Integer archive views counts positions: jointly measured requires both."""
    if views.dtype == np.bool_:
        return views.copy()
    if not np.issubdtype(views.dtype, np.integer) or np.any((views < 0) | (views > 2)):
        raise ValueError("main coverage must be boolean or 0/1/2 position counts")
    return views >= 2


def generate(root, out, *, overlay=None, main_volume=None, main_coverage=None):
    archive = Archive(root)
    result = archive.result("support")
    declared_names = {fold["name"] for fold in archive.folds}
    summaries = result["folds"]
    names = [summary["fold"] for summary in summaries]
    if not names or len(names) != len(set(names)) or set(names) != declared_names:
        raise ValueError("support folds must match all declared folds without duplicates")
    features = archive.manifest["settings"].get("features")
    if not features:
        raise ValueError("missing declared feature sensitivity settings")
    declared_choices = {
        (degree, threshold)
        for degree in features["background_degree"]
        for threshold in features["signal_to_error"]
    }
    if not declared_choices:
        raise ValueError("missing declared feature sensitivity settings")
    folds = []
    for summary in result["folds"]:
        path, record, grid = archive.fold(summary["fold"])
        rho = archive.npy(path / "rho.npy")
        supports = []
        keys = [(choice["degree"], choice["threshold"]) for choice in summary["sensitivity"]]
        if not keys or len(keys) != len(set(keys)) or set(keys) != declared_choices:
            raise ValueError("support sensitivity must match declared choices without duplicates")
        for choice in summary["sensitivity"]:
            name = f"support-degree{choice['degree']}-threshold{choice['threshold']:g}"
            data = archive.npz(path / f"{name}.npz", ["joint", "coverage"])
            if data["coverage"].dtype != np.bool_:
                raise ValueError("support coverage must be boolean")
            check_volume(rho, data["coverage"], grid)
            check_volume(data["joint"], data["coverage"], grid)
            supports.append((name, data))
        if not supports:
            raise ValueError("missing support sensitivity artifacts")
        folds.append((record, grid, rho, supports))
    if (main_volume is None) != (main_coverage is None):
        raise ValueError("main-volume and main-coverage must be provided together")
    main = None
    if main_volume is not None:
        values = archive.npz(main_volume, ["rho", "origin", "spacing", "shape"])
        grid = dict(
            origin=values["origin"].tolist(),
            spacing=float(values["spacing"]),
            shape=values["shape"].tolist(),
        )
        coverage = main_coverage_mask(archive.npy(main_coverage))
        check_volume(values["rho"], coverage, grid)
        main = values["rho"], coverage, grid
    overlay_bytes = archive.bytes(overlay) if overlay is not None else None
    overlay_image = (
        plt.imread(io.BytesIO(overlay_bytes), format=Path(overlay).suffix[1:])
        if overlay is not None
        else None
    )
    out = begin_output(
        archive,
        out,
        dict(
            main_coverage_rule="boolean mask or integer position-count >= 2",
            main_volume_scope="nominal full-data reconstruction" if main is not None else None,
            main_volume=str(Path(main_volume).resolve()) if main is not None else None,
            main_volume_sha256=archive.input_files[archive_source_key(archive, main_volume)][
                "sha256"
            ]
            if main is not None
            else None,
            main_coverage=str(Path(main_coverage).resolve()) if main is not None else None,
            main_coverage_sha256=archive.input_files[archive_source_key(archive, main_coverage)][
                "sha256"
            ]
            if main is not None
            else None,
            support_fold_scopes=[
                dict(name=record["name"], grid=grid, scope="fitted training-fold counts and pose")
                for record, grid, _, _ in folds
            ],
            voxel_coverage="joint geometric support coverage",
            support="dimensionless alternative sharing training counts and fitted pose",
            overlay="conditional box overlays excluded; author image remains separate",
        ),
    )
    if main is not None:
        voxel_panel(*main, out, "nominal-full-data-voxel-only", "Nominal full-data reconstruction")
    if overlay_image is not None:
        fig, ax = plt.subplots(figsize=(7, 4.7), layout="constrained")
        ax.imshow(overlay_image)
        ax.axis("off")
        ax.set_title(
            "Author visualization with conditional box overlays\n"
            "Supporting model interpretation; separate from voxel-only reconstruction"
        )
        save(fig, out, "conditional-author-overlay")
        (out / "author-overlay-provenance.json").write_text(
            json.dumps(
                dict(
                    source=str(Path(overlay).resolve()),
                    sha256=hashlib.sha256(overlay_bytes).hexdigest(),
                ),
                indent=2,
            )
        )
    for record, grid, rho, supports in folds:
        coverage = supports[0][1]["coverage"]
        voxel_panel(
            rho,
            coverage,
            grid,
            out,
            f"{record['name']}-training-voxel-only",
            f"Training fold {record['name']}",
        )
        for name, data in supports:
            fig, axes = plt.subplots(3, 2, figsize=(7, 8), layout="constrained")
            for col, axis in enumerate((2, 1)):
                view = "Top" if axis == 2 else "Side"
                world_panel(
                    axes[0, col],
                    rho,
                    data["coverage"],
                    grid,
                    axis,
                    f"{view}: effective opacity",
                    "1/m",
                    DISPLAY["density_limits"],
                )
                world_panel(
                    axes[1, col],
                    data["joint"],
                    data["coverage"],
                    grid,
                    axis,
                    f"{view}: geometric intersection",
                    "dimensionless support",
                    [0, 1],
                )
                world_panel(
                    axes[2, col],
                    data["coverage"].astype(float),
                    np.ones(rho.shape, bool),
                    grid,
                    axis,
                    f"{view}: coverage",
                    "any joint coverage",
                    [0, 1],
                )
            fig.suptitle(
                f"{record['name']} / {name}\nShared training counts/pose; grey is unmeasured",
                fontsize=8,
            )
            save(fig, out, f"{record['name']}-{name}")


def voxel_panel(rho, coverage, grid, out, name, scope):
    fig = plt.figure(figsize=(7, 4.7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    shape = tuple(grid["shape"])
    edges = [grid["origin"][a] + np.arange(shape[a] + 1) * grid["spacing"] for a in range(3)]
    coordinates = np.meshgrid(*edges, indexing="ij")
    norm = Normalize(*DISPLAY["density_limits"])
    colors = SEQUENTIAL(norm(rho))
    colors[..., 3] = DISPLAY["voxel_alpha"]
    ax.voxels(
        *coordinates,
        coverage & (rho >= DISPLAY["density_threshold"]),
        facecolors=colors,
        edgecolor=None,
    )
    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="z (m)",
        title=(
            f"{scope}\nVoxel-only effective opacity; detector world coordinates\n"
            f"Display threshold {DISPLAY['density_threshold']:g} 1/m; extent is not physical depth"
        ),
    )
    for a, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(edges[a][0], edges[a][-1])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.set_box_aspect(np.array(shape))
    ax.view_init(
        **dict(elev=DISPLAY["voxel_camera"]["elevation"], azim=DISPLAY["voxel_camera"]["azimuth"])
    )
    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL),
        ax=ax,
        label="effective opacity density (1/m)",
        shrink=0.65,
    )
    save(fig, out, name)


if __name__ == "__main__":
    cli(generate, author_overlay=True)
