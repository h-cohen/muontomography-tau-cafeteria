"""Figure 6: beam-depth fit, bootstrap distribution of the depth, and the voxel z profile."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

# Approximate on-site tape measurement of the beam depth (m). A plotted reference
# and an input of the paper, not a result of this analysis.
TAPE_DEPTH_M = 1.25


def main(argv=None) -> None:
    a, _ = style.setup(argv)
    bd = style.load_result(a, "beamdepth")
    h_boot = np.load(Path(a.runs) / "bootstrap" / "values.npz")["depth_h"]

    fig, (ax, bx, cx) = plt.subplots(
        1, 3, figsize=(style.FULL_IN, 2.7), gridspec_kw={"wspace": 0.75}
    )
    for pid, p in bd["profiles"].items():
        ax.plot(p["s"], p["data"], ".", ms=3, color=style.POSITION_COLORS[pid], label=f"{pid} data")
        ax.plot(p["s"], p["model"], "-", color=style.POSITION_COLORS[pid], label=f"{pid} fit")
    ax.set_xlabel(r"sky tangent $s$ (dimensionless)")
    ax.set_ylabel(r"opacity $\lambda$ (dimensionless)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend(loc="upper center", ncols=2, columnspacing=0.8, handlelength=1.2)

    bx.hist(h_boot, bins=max(3, min(12, len(h_boot))), color=style.POSITION_COLORS["pos0"])
    bx.axvline(bd["h"], color=style.INK, lw=0.8, label="nominal fit")
    bx.axvline(TAPE_DEPTH_M, color=style.INK_SECONDARY, lw=0.8, ls="--", label="tape (approx.)")
    bx.set_xlabel("beam depth h (m)")
    bx.set_ylabel("bootstrap replicas")
    bx.yaxis.get_major_locator().set_params(integer=True)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncols=1)

    cx.axvspan(bd["zbottom"], bd["ztop"], color=style.GRID, label="fitted box")
    cx.axvspan(
        bd["zprofile_bottom"],
        bd["zprofile_top"],
        ymax=0.06,
        color=style.POSITION_COLORS["pos1"],
        label="half maximum",
    )
    cx.plot(bd["zprofile_profile_z"], bd["zprofile_profile"], "o-", ms=3, color=style.INK)
    cx.set_xlabel("height z (m)")
    cx.set_ylabel(r"voxel opacity density $\rho$ (m$^{-1}$)")
    cx.legend(loc="upper left", bbox_to_anchor=(0, 1.28), ncols=1)
    style.label_panels([ax, bx, cx])
    style.save(fig, a)


if __name__ == "__main__":
    main()
