"""Where the two views constrain the volume, and how vertical resolution falls with height."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from cafetomo.reconstruct import VoxelSolution
from cafetomo.resolution import depth_resolution


def resolution_curve(z, *, sigma_t: float, baseline: float) -> np.ndarray:
    return np.array([depth_resolution(float(v), baseline, sigma_t) for v in np.atleast_1d(z)])


def main(argv=None) -> None:
    a, cfg = frame.setup(argv)
    beams = style.load_result(a, "beams")
    array = style.load_result(a, "arraydepth")
    voxels = Path(a.runs) / "voxels"
    grid = VoxelSolution.load(voxels / "volume_full.npz").grid
    views = np.load(voxels / "views.npy")
    rays = np.load(voxels / "rays.npy").astype(float)
    xc, yc, zc = (grid.axis_centers(i) for i in range(3))
    half = grid.spacing / 2
    k = int(np.argmin(np.abs(zc - beams["z"])))
    band = np.abs(yc) <= cfg.beamdepth.y_band_m[1]

    fig, (ax, bx, cx) = plt.subplots(
        1,
        3,
        figsize=(style.FULL_IN, 2.5),
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.6, 1]},
    )
    shown, extent_shown = frame.top_view(
        views[:, :, k], (xc[0] - half, xc[-1] + half), (yc[0] - half, yc[-1] + half)
    )
    im = ax.imshow(
        shown,
        origin="lower",
        cmap=style.SEQUENTIAL,
        vmin=0,
        vmax=views.max(),
        extent=extent_shown,
    )
    for pid, (x, y, _) in cfg.origins().items():
        ax.plot(*frame.point(x, y), "o", ms=3, color=style.POSITION_COLORS[pid])
    ax.set(xlabel=r"$x$ (m)", ylabel=r"$y$ (m)")
    fig.colorbar(im, ax=ax, ticks=range(int(views.max()) + 1), label="positions")
    with np.errstate(invalid="ignore", divide="ignore"):
        section = np.log10(np.where(rays > 0, rays, np.nan)[:, band, :].mean(axis=1))
    im2 = bx.imshow(
        section.T,
        origin="lower",
        aspect="auto",
        cmap=style.SEQUENTIAL,
        extent=[xc[0] - half, xc[-1] + half, zc[0] - half, zc[-1] + half],
    )
    bx.set(xlabel=rf"${frame.ax('x')}$ (m)", ylabel=r"$z$ (m)")
    fig.colorbar(im2, ax=bx, label=r"$\log_{10}$ rows")
    z = np.linspace(zc[0], zc[-1], 200)
    cx.plot(
        resolution_curve(z, sigma_t=beams["sigma_t_assumed"], baseline=beams["baseline"]),
        z,
        color=style.INK,
    )
    cx.axhspan(
        array["bottom"], array["bottom"] + array["h"], color=style.GRID, label="conditional box"
    )
    cx.set(xlabel=r"$\delta z$ (m)", ylabel=r"$z$ (m)")
    cx.legend(loc="lower right")
    style.label_panels((ax, bx, cx))
    style.save(fig, a)


if __name__ == "__main__":
    main()
