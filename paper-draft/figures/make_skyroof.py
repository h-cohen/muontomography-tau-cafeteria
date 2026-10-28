"""Open-sky reference from both roof orientations (studies/sky-orientation/joint_sky.py):
the joint reference rate, the fitted azimuthal structure of the sky, and the split of
each bin's roof tracks between the orientations without and with that structure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from make_maps2d import cropped

HOUR = 3600.0
DIVERGING = plt.get_cmap("RdBu_r").with_extremes(bad="#efeeea")
PULL_RANGE = 4.0
DRAWS = 2000


def modulation_curve(fit: dict, tau: float, psi: np.ndarray, rng) -> tuple:
    """exp(s) - 1 in percent of its azimuthal mean on the circle `tau`, against the
    analysis-frame azimuth `psi`, with the 16-84 % band of coefficient draws."""
    m_max = len(fit["coefficients"]) // 6
    cols = np.stack(
        [
            tau ** (m + k) * f(m * psi)
            for m in range(1, m_max + 1)
            for k in range(3)
            for f in (np.cos, np.sin)
        ],
        axis=-1,
    )
    beta = np.asarray(fit["coefficients"])
    cov = np.asarray(fit["covariance"])[1:, 1:]

    def pct(b):
        s = cols @ b
        return 100 * (np.exp(s) / np.mean(np.exp(s)) - 1)

    draws = np.array([pct(b) for b in rng.multivariate_normal(beta, cov, DRAWS)])
    return pct(beta), *np.percentile(draws, [16, 84], axis=0)


def main(argv=None) -> None:
    a, cfg = frame.setup(argv)
    runs = Path(a.runs)
    d = dict(np.load(runs / "split.npz"))
    fit = json.loads((runs / "joint_sky.json").read_text())["fits"]["m3"]
    e, t_a, t_b = d["edges"], float(d["t_a"]), float(d["t_b"])
    used = d["used"]

    def masked(img):
        return np.where(used, img, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(style.FULL_IN, 4.6), layout="constrained")
    (ra, rb, rc), (rd, re, rf) = axes

    def show(axis, img, **kw):
        shown, extent = cropped(img, e, cfg.view.t_display)
        h = axis.imshow(shown, origin="lower", extent=extent, **kw)
        axis.set(xlabel=r"$\tan\theta_x$", ylabel=r"$\tan\theta_y$")
        return h

    rate = masked(d["reference"] / ((t_a + t_b) / HOUR))
    h = show(ra, rate, cmap=style.SEQUENTIAL, vmin=0, vmax=np.nanpercentile(rate, 99.5))
    fig.colorbar(h, ax=ra, shrink=0.85, label="tracks per hour per bin")
    ra.set_title("joint reference", fontsize=7, loc="right")

    sky = 100 * (np.exp(masked(d["sky_log"])) - 1)
    lim = np.nanmax(np.abs(sky))
    h = show(rb, sky, cmap=DIVERGING, vmin=-lim, vmax=lim)
    fig.colorbar(h, ax=rb, shrink=0.85, label="sky flux modulation (%)")
    rb.set_title("fitted sky (roof A)", fontsize=7, loc="right")

    tau = fit["modulation"]["tau"]
    psi = np.linspace(-np.pi, np.pi, 361)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.uncertainty.seed, spawn_key=(5,)))
    mid, lo, hi = modulation_curve(fit, tau, psi, rng)
    # azimuth from the shown x axis: the detector frame swaps the analysis axes
    shown = np.degrees(np.pi / 2 - psi if frame.ax("x") == "y" else psi)
    wrapped = (shown + 180) % 360 - 180
    order = np.argsort(wrapped)
    rc.fill_between(wrapped[order], lo[order], hi[order], color=style.GRID, lw=0)
    rc.plot(wrapped[order], mid[order], color=style.INK)
    rc.axhline(0, color=style.MUTED, lw=0.6)
    rc.set(
        xlim=(-180, 180),
        xticks=(-180, -90, 0, 90, 180),
        xlabel=r"azimuth from $+x$ (deg)",
        ylabel="sky flux modulation (%)",
    )
    rc.set_title(rf"$\tan\theta={tau:g}$", fontsize=7, loc="right")

    for axis, key, title in (
        (rd, "pull_uniform", "uniform sky"),
        (re, "pull", "fitted sky"),
    ):
        h = show(axis, d[key], cmap=DIVERGING, vmin=-PULL_RANGE, vmax=PULL_RANGE)
        axis.set_title(title, fontsize=7, loc="right")
    fig.colorbar(h, ax=[rd, re], shrink=0.85, label="roof-A share pull")

    bins = np.linspace(-6, 6, 61)
    x = 0.5 * (bins[1:] + bins[:-1])
    for key, colour, label in (
        ("pull_uniform", style.MUTED, "uniform sky"),
        ("pull", style.INK, "fitted sky"),
    ):
        v = d[key][np.isfinite(d[key])]
        rf.hist(v, bins, histtype="step", color=colour, label=label)
    gauss = v.size * (bins[1] - bins[0]) * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    rf.plot(x, gauss, color=style.POSITION_COLORS["pos0"], lw=0.8, ls="--", label="N(0, 1)")
    rf.set(
        xlabel="roof-A share pull",
        ylabel="view bins",
        xlim=(-6, 6),
        ylim=(0, 1.35 * rf.get_ylim()[1]),
    )
    rf.legend(frameon=False, fontsize=6, loc="upper left")
    style.label_panels(axes.ravel())
    style.save(fig, a)


if __name__ == "__main__":
    main()
