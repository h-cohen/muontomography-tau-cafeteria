"""Figure 2: measured opacity lambda per position, and the open-sky counts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import style
from cafetomo.angular import load_counts, rebin
from cafetomo.opacity import OpacityMaps


def main(argv=None) -> None:
    a, cfg = style.setup(argv)
    t = cfg.opacity.max_tan
    maps = OpacityMaps.load(Path(a.runs) / "opacity" / "maps.npz")
    sky = rebin(load_counts(Path(a.runs) / "ingest" / "counts_SKY.npz"), cfg.opacity.rebin_factor)

    fig, axes = plt.subplots(
        1, 3, figsize=(style.FULL_IN, 2.5), gridspec_kw={"wspace": 0.55}, sharex=True, sharey=True
    )
    edges = maps.sky.edges
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    lam = [maps.image(pid) for pid in maps.lam]
    vmax = float(np.nanmax(np.concatenate([im.ravel() for im in lam])))
    for ax, pid, im in zip(axes[:2], maps.lam, lam, strict=True):
        h = ax.imshow(
            im.T, origin="lower", extent=extent, cmap=style.SEQUENTIAL, vmin=0.0, vmax=vmax
        )
        ax.set_title(pid, fontsize=8, color=style.POSITION_COLORS[pid])
    fig.colorbar(h, ax=axes[:2], shrink=0.8, pad=0.02, label=r"opacity $\lambda$ (dimensionless)")

    counts = np.where(sky.values > 0, sky.values, np.nan)
    h = axes[2].imshow(
        counts.T,
        origin="lower",
        extent=[sky.xedges[0], sky.xedges[-1]] * 2,
        cmap=style.SEQUENTIAL,
        norm=LogNorm(),
    )
    axes[2].set_title("open sky", fontsize=8)
    fig.colorbar(h, ax=axes[2], shrink=0.8, pad=0.04, label="counts per bin")
    for ax in axes:
        ax.set_xlim(-t, t)
        ax.set_ylim(-t, t)
        ax.set_xlabel(r"$\tan\theta_x$ (dimensionless)")
    axes[0].set_ylabel(r"$\tan\theta_y$ (dimensionless)")
    style.label_panels(axes)
    style.save(fig, a)


if __name__ == "__main__":
    main()
