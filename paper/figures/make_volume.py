"""Figure 7: reconstructed opacity density in three orthogonal slices (SNR >= 2 only)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

SNR_MIN = 2.0


def main(argv=None) -> None:
    a, cfg = style.setup(argv)
    bd = style.load_result(a, "beamdepth")
    sol, iz = style.volume_geometry(a, bd)
    g = sol.grid
    snr = np.load(Path(a.runs) / "bootstrap" / "volume_stats.npz")["snr"]
    rho = np.ma.masked_where(~(snr >= SNR_MIN), sol.rho3())
    xs, ys = g.axis_centers(0), g.axis_centers(1)
    iy = int(np.argmin(np.abs(ys)))
    ix = int(np.argmin(np.abs(xs - np.sort(bd["xs"])[len(bd["xs"]) // 2])))
    (x0, x1), (y0, y1) = cfg.volume.viewer_crop_xy_m
    vmax = float(rho.max())

    fig, axes = plt.subplots(
        1, 3, figsize=(style.FULL_IN, 2.5), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1, 1]}
    )
    kw = {"origin": "lower", "cmap": style.SEQUENTIAL, "vmin": 0.0, "vmax": vmax}
    h = axes[0].imshow(rho[:, :, iz].T, extent=style.edges_extent(g, 0, 1), **kw)
    axes[0].set_xlim(x0, x1)
    axes[0].set_ylim(y0, y1)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title(f"z = {g.axis_centers(2)[iz]:.1f} m", fontsize=8)
    axes[1].imshow(rho[:, iy, :].T, extent=style.edges_extent(g, 0, 2), **kw)
    axes[1].set_xlim(x0, x1)
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    axes[1].set_title(f"y = {ys[iy]:.1f} m", fontsize=8)
    axes[2].imshow(rho[ix, :, :].T, extent=style.edges_extent(g, 1, 2), **kw)
    axes[2].set_xlim(y0, y1)
    axes[2].set_xlabel("y (m)")
    axes[2].set_ylabel("z (m)")
    axes[2].set_title(f"x = {xs[ix]:.1f} m", fontsize=8)
    fig.colorbar(h, ax=axes, shrink=0.8, pad=0.02, label=r"$\rho$ (m$^{-1}$)")
    style.label_panels(axes)
    style.save(fig, a)


if __name__ == "__main__":
    main()
