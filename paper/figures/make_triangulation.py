"""Figure 5: parallax correlation versus trial height and the triangulated height."""

import matplotlib.pyplot as plt

import style


def main(argv=None) -> None:
    a, _ = style.setup(argv)
    beams = style.load_result(a, "beams")
    unc = style.load_result(a, "uncertainty")

    fig, ax = plt.subplots(figsize=(style.COLUMN_IN, 2.4))
    ax.plot(beams["scan_z"], beams["scan_corr_x"], color=style.POSITION_COLORS["pos0"], label="x")
    ax.plot(beams["scan_z"], beams["scan_corr_y"], color=style.POSITION_COLORS["pos1"], label="y")
    ax.axvspan(
        beams["z"] - unc["beams_z_sigma"],
        beams["z"] + unc["beams_z_sigma"],
        color=style.GRID,
        label=r"$\pm\sigma$ (bootstrap)",
    )
    ax.axvline(beams["z"], color=style.INK, lw=0.8, ls="--", label="triangulated z")
    ax.set_xlabel("trial beam height z (m)")
    ax.set_ylabel("parallax correlation (dimensionless)")
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0), ncols=4, columnspacing=0.8, handlelength=1.2
    )
    style.save(fig, a)


if __name__ == "__main__":
    main()
