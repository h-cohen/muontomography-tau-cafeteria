"""Corrected data-driven autofocus: cross-position prediction score versus layer height."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style


def relative_score(score) -> np.ndarray:
    """Score excess over its minimum, as a fraction of the minimum."""
    s = np.asarray(score, dtype=float)
    return s / s.min() - 1.0


def main(argv=None) -> None:
    a, _ = frame.setup(argv)
    af = json.loads((Path(a.results) / "autofocuslayer.json").read_text())
    beams = style.load_result(a, "beams")
    fig, ax = plt.subplots(figsize=(style.COLUMN_IN, 2.4), layout="constrained")
    ax.axvspan(
        af["z"] - af["z_total"], af["z"] + af["z_total"], color=style.GRID, label="evaluated total"
    )
    ax.plot(af["scan_z"], relative_score(af["scan_score"]), "o-", ms=3, color=style.INK)
    ax.axvline(af["z"], color=style.INK, ls="--", lw=0.8, label="autofocus")
    ax.axvline(
        beams["z_x"],
        color=style.POSITION_COLORS["pos1"],
        ls=":",
        lw=0.8,
        label=rf"triangulated $z_{frame.ax('x')}$",
    )
    ax.set(xlabel="trial layer height (m)", ylabel=r"$\mathcal{C}/\mathcal{C}_{\min}-1$")
    ax.legend(loc="upper center")
    style.save(fig, a)


if __name__ == "__main__":
    main()
