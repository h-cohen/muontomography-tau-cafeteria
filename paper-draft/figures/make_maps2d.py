"""Sky-divided angular opacity and local contrast per position (raw chain, count floors)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style


def cropped(img: np.ndarray, edges: np.ndarray, t: float):
    """The square |tan| <= t of an [i_x, j_y] image, oriented for imshow, and its extent."""
    centres = 0.5 * (edges[1:] + edges[:-1])
    keep = np.abs(centres) <= t
    lo, hi = edges[:-1][keep][0], edges[1:][keep][-1]
    return style.top_view(img[np.ix_(keep, keep)], (lo, hi), (lo, hi))


def main(argv=None) -> None:
    a, cfg = style.setup(argv)
    view = cfg.view
    pids = list(cfg.position_ids)
    fig, axes = plt.subplots(
        len(pids), 2, figsize=(style.FULL_IN, 3.0 * len(pids)), layout="constrained"
    )
    for row, pid in zip(axes, pids, strict=True):
        d = np.load(Path(a.runs) / "beams2d" / f"{pid}_beams2d.npz")
        lam, extent = cropped(d["smoothed"], d["edges"], view.t_display)
        con, _ = cropped(d["contrast"], d["edges"], view.t_display)
        im0 = row[0].imshow(
            lam,
            origin="lower",
            extent=extent,
            cmap=style.SEQUENTIAL,
            vmin=view.lam_range[0],
            vmax=view.lam_range[1],
        )
        im1 = row[1].imshow(
            con,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-view.contrast_range,
            vmax=view.contrast_range,
        )
        row[0].set(
            title=f"{pid}: opacity",
            xlabel=r"$\tan\theta_x$",
            ylabel=r"$\tan\theta_y$",
        )
        row[1].set(title=f"{pid}: local contrast", xlabel=r"$\tan\theta_x$")
        fig.colorbar(im0, ax=row[0], label=r"$\lambda$")
        fig.colorbar(im1, ax=row[1], label=r"$\Delta\lambda$")
    style.label_panels(axes.ravel())
    style.save(fig, a)


if __name__ == "__main__":
    main()
