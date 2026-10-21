"""Beam-minus-gap voxel profile along z, its half-maximum width and the conditional box."""

import matplotlib.pyplot as plt
import numpy as np

import frame
import style


def half_maximum(profile) -> float:
    return float(np.nanmax(profile)) / 2


def main(argv=None) -> None:
    a, _ = frame.setup(argv)
    bd = style.load_result(a, "beamdepth")
    array = style.load_result(a, "arraydepth")
    z = np.asarray(bd["zprofile_profile_z"])
    p = np.asarray(bd["zprofile_profile"])

    fig, ax = plt.subplots(figsize=(style.COLUMN_IN, 2.6), layout="constrained")
    ax.axvspan(
        array["bottom"], array["bottom"] + array["h"], color=style.GRID, label="conditional box"
    )
    ax.plot(z, p, "o-", ms=2.5, color=style.INK, label=r"$\Pi(z)$")
    ax.axhline(half_maximum(p), color=style.MUTED, lw=0.8, ls=":")
    for key in ("zprofile_bottom", "zprofile_top"):
        ax.axvline(bd[key], color=style.INK_SECONDARY, lw=0.8, ls="--")
    ax.annotate(
        "",
        xy=(bd["zprofile_top"], half_maximum(p)),
        xytext=(bd["zprofile_bottom"], half_maximum(p)),
        arrowprops={"arrowstyle": "<->", "lw": 0.8},
    )
    ax.set(xlabel=r"$z$ (m)", ylabel=r"beam minus gap $q$ (1/m)")
    ax.legend(loc="upper left")
    style.save(fig, a)


if __name__ == "__main__":
    main()
