"""Weighted residual per row of every SIRT-TV iterate and the selected (lowest) one."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from cafetomo.reconstruct import VoxelSolution

ZOOM_FROM = 20  # first iterate of the zoomed panel, past the initial descent


def selected(history) -> int:
    return int(np.argmin(history))


def main(argv=None) -> None:
    a, _ = frame.setup(argv)
    info = VoxelSolution.load(Path(a.runs) / "voxels" / "volume_full.npz").info
    history = np.asarray(info["chi2_history"])
    k = selected(history)
    if len(history) != info["n_iter_used"] or not np.isclose(
        history[k], style.load_result(a, "reconstruction")["chisq_per_row"], rtol=1e-12
    ):
        raise SystemExit("history does not belong to the reconstruction in results/")
    it = np.arange(len(history))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(style.FULL_IN, 2.4), layout="constrained")
    ax.semilogy(it, history, color=style.INK)
    ax.set(xlabel="iteration $k$", ylabel=r"$\chi^2_k$ per weighted row")
    tail = it >= ZOOM_FROM
    bx.plot(it[tail], history[tail], ".-", ms=2, color=style.INK)
    bx.axvline(k, color=style.POSITION_COLORS["pos1"], lw=0.8, ls="--", label="selected iterate")
    bx.set(xlabel="iteration $k$", ylabel=r"$\chi^2_k$ (zoom)")
    bx.legend(loc="upper right")
    style.label_panels((ax, bx))
    style.save(fig, a)


if __name__ == "__main__":
    main()
