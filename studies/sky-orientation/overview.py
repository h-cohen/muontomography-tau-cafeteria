"""2D overview: the roof run in its two orientations, and pos0/pos1 after sky division.

Row 1: open-sky rate of SKY_A and SKY_B (tracks per hour per view bin, detector frame,
as divided) and the log ratio between them. Rows 2-3: opacity of each position against
SKY_A, against SKY_B, their difference, and the local contrast against the whole roof run.
Reads runs/skyswap/beams2d_{SKY_A,SKY_B,SKY} (`cafetomo beams2d`) and the ingest metas.
Usage: overview.py RUNROOT OUT.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper-draft" / "figures"))
import frame  # noqa: E402
import style  # noqa: E402
from cafetomo.config import load_config  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = load_config(root / "config.yaml", pose_file="results/pose.json")
view = cfg.view
frame.set_frame("detector")
style.apply()
SKIES = ("SKY_A", "SKY_B", "SKY")
SKY_LABEL = {"SKY_A": "roof A (runs 37-57)\n", "SKY_B": "roof B (runs 58-77)\nturned 90° cw"}


def load(sky: str, pid: str) -> dict:
    return dict(np.load(root / f"beams2d_{sky}" / f"{pid}_beams2d.npz"))


def sky_hours(sky: str) -> float:
    meta = json.loads((root / f"ingest_{sky}" / "meta.json").read_text())
    return meta["exposures"]["SKY"]["live_time_s"] / 3600


def show(axis, img, edges, title, cmap, vmin, vmax, label):
    centres = 0.5 * (edges[1:] + edges[:-1])
    keep = np.abs(centres) <= view.t_display
    lo, hi = edges[:-1][keep][0], edges[1:][keep][-1]
    shown, extent = frame.top_view(img[np.ix_(keep, keep)], (lo, hi), (lo, hi))
    im = axis.imshow(shown, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=8)
    axis.set_xlabel(rf"$\tan\theta_{frame.ax('x')}$")
    axis.set_ylabel(rf"$\tan\theta_{frame.ax('y')}$")
    plt.colorbar(im, ax=axis, label=label, shrink=0.85)


pids = list(cfg.position_ids)
fig, axes = plt.subplots(
    1 + len(pids), 4, figsize=(15, 3.6 * (1 + len(pids))), layout="constrained"
)
d = {s: {p: load(s, p) for p in pids} for s in SKIES}
edges = d["SKY_A"][pids[0]]["edges"]

rate = {}
for s in SKIES:
    n = d[s][pids[0]]["counts_sky"].astype(float)
    rate[s] = np.where(n >= view.min_sky_counts, n / sky_hours(s), np.nan)
vmax = np.nanpercentile(rate["SKY_A"], 99)
for i, s in enumerate(("SKY_A", "SKY_B")):
    hours = sky_hours(s)
    show(
        axes[0, i],
        rate[s],
        edges,
        f"{SKY_LABEL[s]}: {hours:.0f} h",
        style.SEQUENTIAL,
        0,
        vmax,
        "tracks / h / bin",
    )
show(
    axes[0, 2],
    np.log(rate["SKY_B"] / rate["SKY_A"]),
    edges,
    "ln(rate B / rate A)",
    "RdBu_r",
    -0.1,
    0.1,
    "log ratio",
)
show(
    axes[0, 3],
    rate["SKY"],
    edges,
    f"whole roof run: {sky_hours('SKY'):.0f} h",
    style.SEQUENTIAL,
    0,
    vmax,
    "tracks / h / bin",
)

lo, hi = view.lam_range
for row, p in zip(axes[1:], pids, strict=True):
    show(
        row[0], d["SKY_A"][p]["smoothed"], edges, f"{p}: λ vs roof A", style.SEQUENTIAL, lo, hi, "λ"
    )
    show(
        row[1], d["SKY_B"][p]["smoothed"], edges, f"{p}: λ vs roof B", style.SEQUENTIAL, lo, hi, "λ"
    )
    diff = d["SKY_B"][p]["smoothed"] - d["SKY_A"][p]["smoothed"]
    show(row[2], diff, edges, f"{p}: λ(B) − λ(A)", "RdBu_r", -0.06, 0.06, "Δλ")
    c = view.contrast_range
    show(
        row[3],
        d["SKY"][p]["contrast"],
        edges,
        f"{p}: local contrast vs whole roof",
        "RdBu_r",
        -c,
        c,
        "Δλ local",
    )
style.label_panels(axes.ravel())
fig.savefig(out, dpi=150)
