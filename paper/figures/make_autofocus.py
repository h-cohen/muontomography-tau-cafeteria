"""Figure 4: autofocus score scan and injected-focus recovery."""

import matplotlib.pyplot as plt

import style


def main(argv=None) -> None:
    a, _ = style.setup(argv)
    af = style.load_result(a, "autofocus")
    unc = style.load_result(a, "uncertainty")
    val = style.load_result(a, "validation")
    c = style.POSITION_COLORS["pos0"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(style.FULL_IN, 2.5), gridspec_kw={"wspace": 0.3})
    ax.plot(af["scan_z"], af["scan_score"], "o-", color=c, ms=3, label="score")
    ax.axvspan(
        unc["autofocus_z_mean"] - unc["autofocus_z_sigma"],
        unc["autofocus_z_mean"] + unc["autofocus_z_sigma"],
        color=style.GRID,
        label="bootstrap mean $\\pm\\sigma$",
    )
    ax.axvline(af["z"], color=style.INK, lw=0.8, ls="--", label="parabola minimum")
    ax.set_xlabel("trial layer height z (m)")
    ax.set_ylabel("focus score (a.u.)")
    ax.legend(loc="upper right")

    lo = min(val["focus_injected"] + val["focus_recovered"]) - 0.3
    hi = max(val["focus_injected"] + val["focus_recovered"]) + 0.3
    bx.plot([lo, hi], [lo, hi], color=style.MUTED, lw=0.6, label="$y=x$")
    bx.plot(val["focus_injected"], val["focus_recovered"], "o", color=c, ms=4, label="recovered")
    bx.set_xlim(lo, hi)
    bx.set_ylim(lo, hi)
    bx.set_aspect("equal")
    bx.set_xlabel("injected height (m)")
    bx.set_ylabel("recovered height (m)")
    bx.legend(loc="lower right")
    style.label_panels([ax, bx])
    style.save(fig, a)


if __name__ == "__main__":
    main()
