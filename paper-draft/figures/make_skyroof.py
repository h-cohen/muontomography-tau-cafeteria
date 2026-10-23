"""Open-sky track rates of both roof orientations and of their joint reference
(studies/sky-orientation/joint_sky.py, split.npz)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from make_maps2d import cropped

HOUR = 3600.0


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
    fig, axes = plt.subplots(1, 3, figsize=(style.FULL_IN, 2.4), layout="constrained")
    vmax = np.nanpercentile(rates["roof A"], 99.5)

    def show(axis, img, title, **kw):
        shown, extent = cropped(img, e, cfg.view.t_display)
        h = axis.imshow(shown, origin="lower", extent=extent, **kw)
        axis.set_title(title, fontsize=7, loc="right")
        axis.set_xlabel(rf"$\tan\theta_{frame.ax('x')}$")
        axis.set_ylabel(rf"$\tan\theta_{frame.ax('y')}$")
        return h

    for axis, (title, img) in zip(axes, rates.items(), strict=True):
        h = show(axis, img, title, cmap=style.SEQUENTIAL, vmin=0, vmax=vmax)
    fig.colorbar(h, ax=axes, shrink=0.85, label="tracks per hour per bin")
    style.label_panels(axes)
    style.save(fig, a)


if __name__ == "__main__":
    main()
