"""The three ceiling-plane heights and how each is measured: rays along the beam shadows
crossing at the beam centres (z_x of the analysis frame), the same for the features of the
other axis that join the joint fit (z), and the data-driven autofocus score versus layer
height, with all three heights and their evaluated totals."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from cafetomo.beams import beam_peaks_subbin, match_peaks, profile, sky_images, triangulate
from cafetomo.cli import _fit_data

Z_TOP = 9.0  # rays are drawn from each detector up to this height (m)


def relative_score(score) -> np.ndarray:
    """Score excess over its minimum, as a fraction of the minimum."""
    s = np.asarray(score, dtype=float)
    return s / s.min() - 1.0


def shadow_rays(images, centers, origins, axis, band, z0, s):
    """Per matched feature, each position's shadow tangent along `axis` (match at z0)."""
    peaks = {
        pid: beam_peaks_subbin(*profile(img, centers, axis, band), s.prominence_sigmas)
        for pid, img in images.items()
    }
    return match_peaks(peaks, origins, axis, z0, s.match_tol_m)


def draw_rays(ax, feats, origins, axis, fitted, z, total, label):
    i = 0 if axis == "x" else 1
    zz = np.array([0.0, Z_TOP])
    for f in feats:
        for pid, t in f["obs"].items():
            o = origins[pid]
            ax.plot(o[i] + t * (zz - o[2]), zz, color=style.POSITION_COLORS[pid], lw=0.7)
    for pid, o in origins.items():
        ax.plot(o[i], o[2], "s", ms=4, color=style.POSITION_COLORS[pid], label=pid)
    ax.axhspan(z - total, z + total, color=style.GRID, zorder=0, label=r"$\pm$ total")
    ax.axhline(z, color=style.INK, lw=0.8, ls="--", label=label)
    ax.plot(fitted, np.full(len(fitted), z), "o", ms=4, mfc="none", color=style.INK)
    ax.set(
        xlabel=rf"${frame.ax(axis)}$ (m)",
        ylabel=r"$z$ (m)",
        ylim=(0, Z_TOP),
    )


def main(argv=None) -> None:
    a, cfg = frame.setup(argv)
    beams = style.load_result(a, "beams")
    sys = style.load_result(a, "systematics")
    af = json.loads((Path(a.results) / "autofocuslayer.json").read_text())
    maps, _, data = _fit_data(cfg, Path(a.runs) / "opacity")
    images = sky_images(data, maps.sky)
    centers = maps.sky.centers
    origins = cfg.origins()
    s = cfg.beams
    z0 = beams["parallax_x_z"]
    sigma_t = beams["sigma_t_assumed"]
    comb = shadow_rays(images, centers, origins, "x", s.band_x, z0, s)
    cross = shadow_rays(images, centers, origins, "y", s.band_y, z0, s)
    only_x = triangulate(images, origins, centers, z0, sigma_t, s, ("x",))

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(style.FULL_IN, 2.9), layout="constrained")
    zx = rf"$z_{frame.ax('x')}$"
    draw_rays(ax, comb, origins, "x", only_x["beams_x"], beams["z_x"], sys["beams_zx_total"], zx)
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("beam centres", fontsize=7, loc="right")
    draw_rays(
        bx, cross, origins, "y", beams["beams_y"], beams["z"], sys["beams_z_total"], "joint $z$"
    )
    bx.axhline(beams["z_x"], color=style.MUTED, lw=0.8, ls=":", label=zx)
    bx.legend(loc="lower left", fontsize=6)
    bx.set_title("joint fit", fontsize=7, loc="right")

    cx.plot(af["scan_z"], relative_score(af["scan_score"]), "o-", ms=3, color=style.INK)
    estimates = (
        (af["z"], sys["autofocus_z_total"], style.INK, "--", r"autofocus $\hat z$"),
        (beams["z_x"], sys["beams_zx_total"], style.POSITION_COLORS["pos1"], ":", zx),
        (beams["z"], sys["beams_z_total"], style.MUTED, "-.", "joint $z$"),
    )
    for k, (value, total, colour, ls, label) in enumerate(estimates):
        cx.axvline(value, color=colour, ls=ls, lw=0.9, label=label)
        cx.errorbar(
            value, -0.04 * (k + 1), xerr=total, fmt="o", ms=3, color=colour, capsize=2, lw=0.9
        )
    lo = min(v - t for v, t, *_ in estimates)
    hi = max(v + t for v, t, *_ in estimates)
    cx.set(
        xlabel="trial layer height (m)",
        ylabel=r"$\mathcal{C}/\mathcal{C}_{\min}-1$",
        xlim=(min(lo, af["scan_z"][0]) - 0.2, max(hi, af["scan_z"][-1]) + 0.2),
    )
    cx.legend(loc="upper right", fontsize=6)
    cx.set_title("autofocus", fontsize=7, loc="right")
    style.label_panels((ax, bx, cx))
    style.save(fig, a)


if __name__ == "__main__":
    main()
