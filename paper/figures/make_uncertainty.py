"""Figure 8: bootstrap sigma of the opacity density and the view coverage."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style


def main(argv=None) -> None:
    a, cfg = style.setup(argv)
    bd = style.load_result(a, "beamdepth")
    sol, iz = style.volume_geometry(a, bd)
    g = sol.grid
    sigma = np.load(Path(a.runs) / "bootstrap" / "volume_stats.npz")["sigma"]
    views = np.load(Path(a.runs) / "voxels" / "views.npy")
    (x0, x1), (y0, y1) = cfg.volume.viewer_crop_xy_m

    fig, axes = plt.subplots(1, 2, figsize=(style.FULL_IN, 2.8), gridspec_kw={"wspace": 0.35})
    ext = style.edges_extent(g, 0, 1)
    h = axes[0].imshow(
        sigma[:, :, iz].T, origin="lower", extent=ext, cmap=style.SEQUENTIAL, vmin=0.0
    )
    fig.colorbar(h, ax=axes[0], shrink=0.8, pad=0.03, label=r"bootstrap $\sigma_\rho$ (m$^{-1}$)")
    h = axes[1].imshow(
        views[:, :, iz].T,
        origin="lower",
        extent=ext,
        cmap=style.SEQUENTIAL,
        vmin=0,
        vmax=len(cfg.position_ids),
    )
    cb = fig.colorbar(h, ax=axes[1], shrink=0.8, pad=0.03, label="positions viewing the voxel")
    cb.set_ticks(range(len(cfg.position_ids) + 1))
    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"z = {g.axis_centers(2)[iz]:.1f} m", fontsize=8)
    style.label_panels(axes)
    style.save(fig, a)


if __name__ == "__main__":
    main()
