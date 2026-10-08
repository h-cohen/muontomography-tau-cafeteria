"""Figure 1: measurement geometry, top view and side view."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import style


def main(argv=None) -> None:
    a, cfg = style.setup(argv)
    bd = style.load_result(a, "beamdepth")
    (x0, x1), (y0, y1) = cfg.volume.viewer_crop_xy_m
    ybeam = cfg.beamdepth.y_extent_m
    beams = style.load_result(a, "beams")
    pitch = beams["period"] * beams["z_x"]
    fitted_xs = np.asarray(bd["xs"])
    n_left = max(0, int(np.ceil((fitted_xs[0] - x0) / pitch)))
    n_right = max(0, int(np.ceil((x1 - fitted_xs[-1]) / pitch)))
    illustrative_xs = np.sort(
        np.r_[
            fitted_xs[0] - pitch * np.arange(1, n_left + 1),
            fitted_xs,
            fitted_xs[-1] + pitch * np.arange(1, n_right + 1),
        ]
    )
    illustrative_xs = illustrative_xs[(illustrative_xs >= x0) & (illustrative_xs <= x1)]
    zb, zt, w = bd["zbottom"], bd["ztop"], bd["w"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(style.FULL_IN, 2.6), gridspec_kw={"wspace": 0.3})
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=style.MUTED, lw=0.8, ls=":"))
    for xb in illustrative_xs:
        ax.add_patch(
            Rectangle((xb - w / 2, ybeam[0]), w, ybeam[1] - ybeam[0], fc=style.GRID, ec="none")
        )
    for pid, (px, py, _pz) in cfg.origins().items():
        ax.plot(px, py, "o", color=style.POSITION_COLORS[pid], ms=4, label=pid)
    px_all = [p[0] for p in cfg.origins().values()]
    py_all = [p[1] for p in cfg.origins().values()]
    reach = max(abs(x0), abs(x1), abs(y0), abs(y1))
    xlim = (min(px_all) - reach - 0.5, max(px_all) + reach + 0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(min(py_all) - reach - 0.5, max(py_all) + reach + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    for xb in illustrative_xs:
        bx.add_patch(Rectangle((xb - w / 2, zb), w, zt - zb, fc=style.INK_SECONDARY, ec="none"))
    xt = bd["xs"][1]
    for pid, (px, _, pz) in cfg.origins().items():
        bx.plot([px, xt], [pz, 0.5 * (zb + zt)], color=style.POSITION_COLORS[pid], lw=0.8)
        bx.plot(px, pz, "o", color=style.POSITION_COLORS[pid], ms=4, label=pid)
    bx.axhline(zb, color=style.MUTED, lw=0.5, ls=":")
    bx.set_xlim(*xlim)
    bx.set_ylim(-0.4, zt + 0.8)
    bx.set_xlabel("x (m)")
    bx.set_ylabel("z (m)")
    bx.text(
        xlim[1] - 0.2,
        zt + 0.1,
        "conditional box-fit layer",
        ha="right",
        va="bottom",
        color=style.INK_SECONDARY,
    )
    bx.legend(loc="center right", handlelength=0.8)
    style.label_panels([ax, bx])
    style.save(fig, a)


if __name__ == "__main__":
    main()
