"""Nominal voxel field: horizontal slices through the conditional box and an x-z section."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from cafetomo.reconstruct import VoxelSolution
from internal_figure_utils import DISPLAY


def nearest_layers(centres, heights) -> list[int]:
    return [int(np.argmin(np.abs(np.asarray(centres) - z))) for z in heights]


def main(argv=None) -> None:
    a, cfg = style.setup(argv)
    array = style.load_result(a, "arraydepth")
    voxels = Path(a.runs) / "voxels"
    sol = VoxelSolution.load(voxels / "volume_full.npz")
    q = sol.rho3().astype(float)
    q[np.load(voxels / "views.npy") < 2] = np.nan
    xc, yc, zc = (sol.grid.axis_centers(i) for i in range(3))
    heights = [array["bottom"], array["bottom"] + array["h"] / 2, array["bottom"] + array["h"]]
    layers = nearest_layers(zc, heights)
    vmin, vmax = DISPLAY["density_limits"]
    half = sol.grid.spacing / 2
    extent_xy = [xc[0] - half, xc[-1] + half, yc[0] - half, yc[-1] + half]

    fig = plt.figure(figsize=(style.FULL_IN, 4.6), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax, k in zip(axes, layers, strict=True):
        shown, extent_shown = style.top_view(q[:, :, k], extent_xy[:2], extent_xy[2:])
        im = ax.imshow(
            shown,
            origin="lower",
            extent=extent_shown,
            cmap=style.SEQUENTIAL,
            vmin=vmin,
            vmax=vmax,
        )
        for pid, (x, y, _) in cfg.origins().items():
            ax.plot(*style.point(x, y), "o", ms=3, color=style.POSITION_COLORS[pid])
        ax.set(
            title=rf"$z={zc[k]:.1f}$ m",
            xlabel=r"$x$ (m)",
            ylabel=r"$y$ (m)",
        )
    side = fig.add_subplot(grid[1, :])
    band = np.abs(yc) <= cfg.beamdepth.y_band_m[1]
    with np.errstate(invalid="ignore"):
        section = np.nanmean(q[:, band, :], axis=1)
    side.imshow(
        section.T,
        origin="lower",
        aspect="auto",
        cmap=style.SEQUENTIAL,
        vmin=vmin,
        vmax=vmax,
        extent=[xc[0] - half, xc[-1] + half, zc[0] - half, zc[-1] + half],
    )
    for z in (array["bottom"], array["bottom"] + array["h"]):
        side.axhline(z, color=style.INK_SECONDARY, lw=0.6, ls="--")
    side.set(xlabel=rf"${style.ax('x')}$ (m)", ylabel=r"$z$ (m)")
    fig.colorbar(im, ax=axes, label=r"$q$ (1/m)", shrink=0.8)
    style.label_panels([*axes, side])
    style.save(fig, a)


if __name__ == "__main__":
    main()
