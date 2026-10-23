"""Opacity of pos0 and pos1 against the joint open-sky reference of both roof orientations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from make_maps2d import cropped

NOMINAL = "beams2d_m3_phi0"


def main(argv=None) -> None:
    a, cfg = frame.setup(argv)
    view = cfg.view
    pids = list(cfg.position_ids)
    fig, axes = plt.subplots(1, len(pids), figsize=(style.FULL_IN, 3.0), layout="constrained")
    for axis, pid in zip(axes, pids, strict=True):
        d = np.load(Path(a.runs) / NOMINAL / f"{pid}_beams2d.npz")
        lam, extent = cropped(d["smoothed"], d["edges"], view.t_display)
        im = axis.imshow(
            lam,
            origin="lower",
            extent=extent,
            cmap=style.SEQUENTIAL,
            vmin=view.lam_range[0],
            vmax=view.lam_range[1],
        )
        axis.set(
            title=pid,
            xlabel=rf"$\tan\theta_{frame.ax('x')}$",
            ylabel=rf"$\tan\theta_{frame.ax('y')}$",
        )
    fig.colorbar(im, ax=axes, label=r"$\lambda$")
    style.label_panels(axes)
    style.save(fig, a)


if __name__ == "__main__":
    main()
