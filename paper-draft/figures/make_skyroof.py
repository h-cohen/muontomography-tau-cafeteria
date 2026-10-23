"""The open-sky reference from both roof orientations: rates, their ratio, the fitted sky
term and its residuals (studies/sky-orientation/joint_sky.py, split.npz)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from make_maps2d import cropped

HOUR = 3600.0


def log_ratio(n: np.ndarray, a: np.ndarray, t_a: float, t_b: float) -> np.ndarray:
    """ln of the roof-B over roof-A rate, given all roof tracks `n` and the A share `a`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(((n - a) / t_b) / (a / t_a))


def main(argv=None) -> None:
    a, cfg = frame.setup(argv)
    d = dict(np.load(Path(a.runs) / "split.npz"))
    t_a, t_b, used, e = float(d["t_a"]), float(d["t_b"]), d["used"], d["edges"]
    n, obs = d["n"], d["observed_a"]

    def masked(img):
        return np.where(used, img, np.nan)

    rates = {
        "roof A": masked(obs / (t_a / HOUR)),
        "roof B (turned)": masked((n - obs) / (t_b / HOUR)),
        "joint reference": masked(d["reference"] / ((t_a + t_b) / HOUR)),
    }
    ratios = {
        "measured ln(B/A)": masked(log_ratio(n, obs, t_a, t_b)),
        "fitted sky term": masked(log_ratio(n, d["expected_a"], t_a, t_b)),
    }
    fig, axes = plt.subplots(2, 3, figsize=(style.FULL_IN, 4.6), layout="constrained")
    vmax = np.nanpercentile(rates["roof A"], 99.5)

    def show(axis, img, title, **kw):
        shown, extent = cropped(img, e, cfg.view.t_display)
        h = axis.imshow(shown, origin="lower", extent=extent, **kw)
        axis.set_title(title, fontsize=7, loc="right")
        axis.set_xlabel(rf"$\tan\theta_{frame.ax('x')}$")
        axis.set_ylabel(rf"$\tan\theta_{frame.ax('y')}$")
        return h

    for axis, (title, img) in zip(axes[0], rates.items(), strict=True):
        h = show(axis, img, title, cmap=style.SEQUENTIAL, vmin=0, vmax=vmax)
    fig.colorbar(h, ax=axes[0], shrink=0.85, label="tracks per hour per bin")
    for axis, (title, img) in zip(axes[1, :2], ratios.items(), strict=True):
        h = show(axis, img, title, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    fig.colorbar(h, ax=axes[1, :2], shrink=0.85, label="log rate ratio")
    h = show(axes[1, 2], masked(d["pull"]), "residual pull", cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(h, ax=axes[1, 2], shrink=0.85, label="pull")
    style.label_panels(axes.ravel())
    style.save(fig, a)


if __name__ == "__main__":
    main()
