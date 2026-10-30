"""Rate of the open-sky reference from both roof orientations
(studies/sky-orientation/joint_sky.py)."""

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
    hours = (float(d["t_a"]) + float(d["t_b"])) / HOUR
    rate = np.where(d["used"], d["reference"] / hours, np.nan)
    fig, axis = plt.subplots(figsize=(style.COLUMN_IN, 2.8), layout="constrained")
    shown, extent = cropped(rate, d["edges"], cfg.view.t_display)
    h = axis.imshow(
        shown,
        origin="lower",
        extent=extent,
        cmap=style.SEQUENTIAL,
        vmin=0,
        vmax=np.nanpercentile(rate, 99.5),
    )
    axis.set(xlabel=r"$\tan\theta_x$", ylabel=r"$\tan\theta_y$")
    fig.colorbar(h, ax=axis, shrink=0.85, label="tracks per hour per bin")
    style.save(fig, a)


if __name__ == "__main__":
    main()
