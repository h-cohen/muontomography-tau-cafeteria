"""Autofocus and transverse parallax in one height-diagnostic figure."""

import matplotlib.pyplot as plt

import style


def main(argv=None) -> None:
    a, _ = style.setup(argv)
    focus = style.load_result(a, "autofocus")
    beams = style.load_result(a, "beams")
    unc = style.load_result(a, "uncertainty")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(style.FULL_IN, 2.6), layout="constrained")
    ax.plot(focus["scan_z"], focus["scan_score"], "o-", ms=3, color=style.POSITION_COLORS["pos0"])
    ax.axvline(focus["z"], color=style.INK, ls="--", lw=0.8, label="autofocus")
    ax.axvspan(
        unc["autofocus_z_mean"] - unc["autofocus_z_sigma"],
        unc["autofocus_z_mean"] + unc["autofocus_z_sigma"],
        color=style.GRID,
        label="count-bootstrap spread",
    )
    ax.set(xlabel="trial layer height (m)", ylabel="prediction score")
    ax.legend(loc="upper right")
    bx.plot(beams["scan_z"], beams["scan_corr_x"], color=style.POSITION_COLORS["pos0"])
    bx.axvline(beams["z_x"], color=style.INK, ls="--", lw=0.8, label="transverse triangulation")
    bx.axvspan(
        beams["z_x"] - unc["beams_zx_sigma"],
        beams["z_x"] + unc["beams_zx_sigma"],
        color=style.GRID,
        label="count-bootstrap spread",
    )
    bx.set(xlabel="trial beam height (m)", ylabel="transverse parallax correlation")
    bx.legend(loc="lower left")
    style.label_panels((ax, bx))
    style.save(fig, a)


if __name__ == "__main__":
    main()
