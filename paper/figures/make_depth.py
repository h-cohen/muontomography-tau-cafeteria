"""Figure 6: beam-depth fit, bootstrap distribution of the depth, and the voxel z profile."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style


def histogram_panel(ax, h_boot, h_nominal: float, h_measured: float) -> None:
    """Bootstrap depth histogram; non-finite replicas are dropped, none left gives a note."""
    h = np.asarray(h_boot, dtype=float)
    h = h[np.isfinite(h)]
    if h.size:
        ax.hist(h, bins=max(3, min(12, h.size)), color=style.POSITION_COLORS["pos0"])
    else:
        ax.text(0.5, 0.5, "no finite replicas", transform=ax.transAxes, ha="center", va="center")
    ax.axvline(h_nominal, color=style.INK, lw=0.8, label="nominal fit")
    ax.axvline(h_measured, color=style.INK_SECONDARY, lw=0.8, ls="--", label="on-site measurement")
    ax.set_xlabel("beam depth h (m)")
    ax.set_ylabel("bootstrap replicas")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylim(top=max(ax.get_ylim()[1], 1.0) * 1.5)
    ax.legend(loc="upper center")


def main(argv=None) -> None:
    a, _ = style.setup(argv)
    bd = style.load_result(a, "beamdepth")
    inputs = style.load_result(a, "inputs")
    h_boot = np.load(Path(a.runs) / "bootstrap" / "values.npz")["depth_h"]

    fig, (ax, bx, cx) = plt.subplots(
        1, 3, figsize=(style.FULL_IN, 2.7), gridspec_kw={"wspace": 0.75}
    )
    for pid, p in bd["profiles"].items():
        ax.plot(p["s"], p["data"], ".", ms=3, color=style.POSITION_COLORS[pid], label=f"{pid} data")
        ax.plot(p["s"], p["model"], "-", color=style.POSITION_COLORS[pid], label=f"{pid} fit")
    ax.set_xlabel(r"sky tangent $s$ (dimensionless)")
    ax.set_ylabel(r"opacity $\lambda$ (dimensionless)")
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.1, 1.0), ncols=2, columnspacing=0.8, handlelength=1.2
    )

    histogram_panel(bx, h_boot, bd["h"], inputs["measured_beam_depth"])

    cx.axvspan(bd["zbottom"], bd["ztop"], color=style.GRID, label="fitted box")
    cx.axvspan(
        bd["zprofile_bottom"],
        bd["zprofile_top"],
        ymax=0.06,
        color=style.POSITION_COLORS["pos1"],
        label="half maximum",
    )
    cx.plot(bd["zprofile_profile_z"], bd["zprofile_profile"], "o-", ms=3, color=style.INK)
    cx.axvline(
        inputs["measured_beam_bottom"],
        color=style.INK_SECONDARY,
        lw=0.8,
        ls="--",
        label="on-site bottom",
    )
    cx.set_xlabel("height z (m)")
    cx.set_ylabel(r"voxel opacity density $\rho$ (m$^{-1}$)")
    cx.set_ylim(top=cx.get_ylim()[1] * 1.35)
    cx.legend(loc="upper left")
    style.label_panels([ax, bx, cx])
    style.save(fig, a)


if __name__ == "__main__":
    main()
