"""Stripes of the 2D opacity maps (studies/maps2d): beam profiles of both positions with
pos1 shifted onto pos0, transverse profiles, and the stripe width and contrast with the
DAQ hit flag against the pedestal-based neighbour selection."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style

NOMINAL, FLAG = "pedestal", "flag"
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
HALF_WINDOW = 0.10  # as fitted (studies/maps2d)


def curve(st: dict) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(st["centre"] - HALF_WINDOW, st["centre"] + HALF_WINDOW, 81)
    s = st["fwhm"] / FWHM
    u = t - st["centre"]
    return t, st["contrast"] * np.exp(-0.5 * (u / s) ** 2) + st["background"] + st["slope"] * u


def matched(flag: list, nominal: list, tol: float = 0.02) -> list[tuple[dict, dict]]:
    """Resolved stripes of both selections at the same centre (within `tol`)."""
    out = []
    for n in (s for s in nominal if s["resolved"]):
        near = [f for f in flag if f["resolved"] and abs(f["centre"] - n["centre"]) < tol]
        if near:
            out.append((min(near, key=lambda f: abs(f["centre"] - n["centre"])), n))
    return out


def main(argv=None) -> None:
    a, _ = frame.setup(argv)
    docs = {k: json.loads((Path(a.runs) / f"maps2d_{k}.json").read_text()) for k in (NOMINAL, FLAG)}
    doc = docs[NOMINAL]
    pids = list(doc["profiles"])
    shift = doc["nominal"]["shift"]
    fig = plt.figure(figsize=(style.FULL_IN, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.15, 1))
    top = fig.add_subplot(grid[0, :])
    tr, wd, ct = (fig.add_subplot(grid[1, i]) for i in range(3))

    for pid in pids:
        colour = style.POSITION_COLORS[pid]
        dt = shift if pid == pids[1] else 0.0
        p = doc["profiles"][pid]["beam"]
        label = pid if dt == 0 else rf"{pid}, shifted by $\Delta t={dt:+.3f}$"
        top.errorbar(np.add(p["t"], dt), p["lam"], p["sigma"], fmt="o", ms=2, lw=0.6,
                     color=colour, label=label)  # fmt: skip
        for st in doc["stripes"][pid]["beam"]:
            if st["resolved"]:
                t, y = curve(st)
                top.plot(t + dt, y, color=colour, lw=1.0)
        q = doc["profiles"][pid]["transverse"]
        tr.errorbar(q["t"], q["lam"], q["sigma"], fmt="o", ms=1.5, lw=0.5, color=colour, label=pid)
        for st in doc["stripes"][pid]["transverse"]:
            if st["resolved"] and np.isclose(
                st["centre"], doc["nominal"][f"{pid}_transverse_centre"]
            ):
                t, y = curve(st)
                tr.plot(t, y, color=colour, lw=1.0)
    across, along = frame.ax("x"), frame.ax("y")
    top.set(
        xlabel=rf"$\tan\theta_{across}$ (pos0 frame)",
        ylabel=rf"$\lambda$, band $|\tan\theta_{along}|\leq 0.25$",
    )
    top.legend(frameon=False, fontsize=7, loc="upper right")
    tr.set(
        xlabel=rf"$\tan\theta_{along}$", ylabel=rf"$\lambda$, band $|\tan\theta_{across}|\leq 0.25$"
    )
    tr.legend(frameon=False, fontsize=6, loc="upper left")

    for axis, key, label in (
        (wd, "fwhm", "FWHM (tan)"),
        (ct, "contrast", r"contrast $\Delta\lambda$"),
    ):
        lo, hi = np.inf, -np.inf
        for pid in pids:
            pairs = matched(docs[FLAG]["stripes"][pid]["beam"], doc["stripes"][pid]["beam"])
            x = np.array([f[key] for f, _ in pairs])
            y = np.array([n[key] for _, n in pairs])
            xe = np.array([f[key + "_sigma"] for f, _ in pairs])
            ye = np.array([n[key + "_sigma"] for _, n in pairs])
            axis.errorbar(x, y, ye, xe, fmt="o", ms=2.5, lw=0.6, color=style.POSITION_COLORS[pid],
                          label=pid)  # fmt: skip
            if x.size:
                lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
        pad = 0.15 * (hi - lo)
        axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=style.MUTED, lw=0.6, ls="--")
        axis.set(xlabel=f"{label}, hit flag", ylabel=f"{label}, pedestal selection")
    wd.legend(frameon=False, fontsize=6, loc="upper left")
    style.label_panels([top, tr, wd, ct])
    style.save(fig, a)


if __name__ == "__main__":
    main()
