"""Continued-array conditional fit against the opacity data in the sky band, per position."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from cafetomo.arraydepth import array_model
from cafetomo.cli import _fit_data


def band_average(sx, values):
    s = np.unique(sx)
    return s, np.array([np.mean(values[sx == u]) for u in s])


def main(argv=None) -> None:
    a, cfg = style.setup(argv)
    record = style.load_result(a, "arraydepth")
    beams = style.load_result(a, "beams")
    _, _, data = _fit_data(cfg, Path(a.runs) / "opacity")
    m = array_model(data, cfg, record, height_seed=beams["z"])
    if abs(m["chi2"] / record["chi2"] - 1) > 1e-6:
        raise SystemExit("array model does not reproduce results/arraydepth.json")
    pos = np.asarray(m["pos"])
    model = m["beam"] + m["background"]

    pids = list(data.rows.position_ids)
    fig, axes = plt.subplots(
        2,
        len(pids),
        figsize=(style.FULL_IN, 3.6),
        sharex=True,
        height_ratios=[3, 1],
        layout="constrained",
    )
    for col, pid in enumerate(pids):
        sel = pos == pid
        s, lam = band_average(m["sx"][sel], m["lam"][sel])
        _, mod = band_average(m["sx"][sel], model[sel])
        _, bg = band_average(m["sx"][sel], m["background"][sel])
        top, bottom = axes[0, col], axes[1, col]
        top.plot(s, lam, ".", ms=3, color=style.POSITION_COLORS[pid], label="data")
        top.plot(s, mod, "-", color=style.INK, label="array + background")
        top.plot(s, bg, ":", color=style.MUTED, label="background")
        top.set(title=pid, ylabel=r"opacity $\lambda$ (dimensionless)")
        bottom.plot(s, lam - mod, ".", ms=3, color=style.POSITION_COLORS[pid])
        bottom.axhline(0, color=style.MUTED, lw=0.6)
        bottom.set(xlabel=r"sky tangent $s_x$ (dimensionless)", ylabel="residual")
    axes[0, 0].legend(loc="upper left")
    style.label_panels(axes[0])
    style.save(fig, a)


if __name__ == "__main__":
    main()
