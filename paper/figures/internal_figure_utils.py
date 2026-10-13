"""Read archived internal-validation arrays; never reconstruct or consult baseline results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from style import SEQUENTIAL, apply, plt

DISPLAY = dict(
    missing_coverage="unmeasured",
    density_units="effective opacity density 1/m",
    projection="maximum over covered voxels; coverage shown separately",
    density_limits=[0.0, 0.05],
    density_threshold=0.015,
    frequency_limits=[0.0, 1.0],
    support_limits=[0.0, 1.0],
    voxel_camera=dict(elevation=22, azimuth=-58),
    voxel_alpha=0.8,
    interpretation="retrospective internal diagnostics; no external truth",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path, keys):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in keys}


class Archive:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest["status"] != "complete":
            raise ValueError("campaign must be complete before figure generation")
        if self.manifest["mode"] == "smoke":
            raise ValueError("smoke campaigns cannot produce publication figures")
        self.folds = self.manifest["folds"]
        if not self.folds:
            raise ValueError("complete campaign has no folds")

    def result(self, name):
        return read_json(self.root / "results" / f"{name}.json")

    def fold(self, name):
        path = self.root / "folds" / name
        record = read_json(path / "fold.json")
        return path, record, record["training"]["grid"]


def check_volume(field, coverage, grid):
    if field.shape != tuple(grid["shape"]) or coverage.shape != field.shape:
        raise ValueError("world-grid/array shape mismatch")
    if np.any(~np.isfinite(field[coverage])):
        raise ValueError("covered volume contains non-finite values")


def projection(field, coverage, axis):
    """Max projection retains NaN wherever no voxel on the ray is measured."""
    selected = np.where(coverage, field, -np.inf)
    image = np.max(selected, axis=axis)
    return np.where(np.any(coverage, axis=axis), image, np.nan).T


def extent(grid, axes):
    return [
        value
        for axis in axes
        for value in (
            grid["origin"][axis],
            grid["origin"][axis] + grid["spacing"] * grid["shape"][axis],
        )
    ]


def world_panel(ax, field, coverage, grid, axis, title, units, limits):
    check_volume(field, coverage, grid)
    axes = [a for a in range(3) if a != axis]
    image = ax.imshow(
        projection(field, coverage, axis),
        origin="lower",
        extent=extent(grid, axes),
        aspect="equal",
        cmap=SEQUENTIAL,
        vmin=limits[0],
        vmax=limits[1],
        interpolation="nearest",
    )
    ax.set(xlabel=f"{'xyz'[axes[0]]} (m)", ylabel=f"{'xyz'[axes[1]]} (m)", title=title)
    plt.colorbar(image, ax=ax, label=units, shrink=0.75)


def begin_output(archive, out, extra=None):
    out = Path(out)
    # A new directory makes partial/stale output unmistakable.
    out.mkdir(parents=True, exist_ok=False)
    document = dict(
        DISPLAY,
        source_archive=str(archive.root),
        manifest_sha256=hashlib.sha256((archive.root / "manifest.json").read_bytes()).hexdigest(),
    )
    document.update(extra or {})
    (out / "display-settings.json").write_text(json.dumps(document, indent=2) + "\n")
    apply()
    return out


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=160)
    plt.close(fig)


def cli(generate, *, author_overlay=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True, help="new output directory")
    if author_overlay:
        parser.add_argument("--overlay", help="optional author image, separately labelled")
        parser.add_argument("--main-volume", help="explicit nominal full-data NPZ")
        parser.add_argument(
            "--main-coverage", help="explicit matching views.npy or boolean coverage"
        )
    args = parser.parse_args()
    kwargs = (
        dict(overlay=args.overlay, main_volume=args.main_volume, main_coverage=args.main_coverage)
        if author_overlay
        else {}
    )
    generate(args.archive, args.out, **kwargs)
