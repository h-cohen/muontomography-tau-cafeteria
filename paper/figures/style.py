"""Shared figure style and command line for paper/figures/make_*.py."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from cafetomo.config import load_config  # noqa: E402
from cafetomo.reconstruct import VoxelSolution  # noqa: E402

COLUMN_IN = 3.4
FULL_IN = 7.0
POSITION_COLORS = {"pos0": "#1b6ca8", "pos1": "#d1495b"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# One-hue sequential ramp (blue, light to dark) for magnitudes: lambda, rho, sigma, counts.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
SEQUENTIAL.set_bad("#efeeea")
PANEL_LABEL_KW = {"fontweight": "bold", "va": "bottom", "ha": "left"}


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "lines.linewidth": 1.0,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    for name in ("config", "pose", "results", "runs", "out"):
        ap.add_argument(f"--{name}", required=True)
    return ap.parse_args(argv)


def setup(argv=None):
    """Parse the shared arguments, apply the style, and load the config."""
    a = args(argv)
    apply()
    return a, load_config(a.config, pose_file=a.pose)


def load_result(a: argparse.Namespace, name: str) -> dict:
    return json.loads((Path(a.results) / f"{name}.json").read_text())


def label_panels(axes) -> None:
    for ax, letter in zip(axes, "abcdef", strict=False):
        ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, **PANEL_LABEL_KW)


def save(fig, a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def volume_geometry(a: argparse.Namespace, beamdepth: dict):
    """Voxel centres and the slice indices shared by the volume and uncertainty figures.

    Returns (solution, z_index) where z_index is the layer at the beams' mid-height.
    """
    sol = VoxelSolution.load(Path(a.runs) / "voxels" / "volume_full.npz")
    z_mid = 0.5 * (beamdepth["zbottom"] + beamdepth["ztop"])
    z_index = int(np.argmin(np.abs(sol.grid.axis_centers(2) - z_mid)))
    return sol, z_index


def edges_extent(grid, axis_a: int, axis_b: int) -> list[float]:
    xa = grid.extent(axis_a)
    xb = grid.extent(axis_b)
    return [xa[0], xa[1], xb[0], xb[1]]
