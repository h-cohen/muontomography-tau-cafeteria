"""λ shifts between joint-reference hypotheses and the single-orientation references.

Per room position, on view bins measured in both maps (unsmoothed λ): mean and rms of the
difference, and the median per-bin σ of the nominal map. Nominal: m ≤ 3, φ = 0.
Writes JOINT_ROOT/compare.json and a figure.
Usage: compare_joint.py SKYSWAP_ROOT JOINT_ROOT OUT.png
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

swap, joint, fig_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cfg = load_config(joint / "config.yaml", pose_file="results/pose.json")
view = cfg.view
pids = list(cfg.position_ids)
NOMINAL = "m3_phi0"
ALTERNATIVES = {
    "orientation_phi180": joint / "beams2d_m3_phi180",
    "orientation_phiB": joint / "beams2d_m3_phiB",
    "orientation_phiB180": joint / "beams2d_m3_phiB180",
    "model_m2": joint / "beams2d_m2_phi0",
    "single_SKY_A": swap / "beams2d_SKY_A",
    "single_SKY_B": swap / "beams2d_SKY_B",
    "single_whole_run": swap / "beams2d_SKY",
}


def load(d: Path, pid: str) -> dict:
    return dict(np.load(d / f"{pid}_beams2d.npz"))


def outer(edges: np.ndarray, t: float) -> np.ndarray:
    c = 0.5 * (edges[1:] + edges[:-1])
    tx, ty = np.meshgrid(c, c, indexing="ij")
    return np.hypot(tx, ty) > t


doc = {}
for pid in pids:
    nom = load(joint / f"beams2d_{NOMINAL}", pid)
    big = outer(nom["edges"], 0.5)
    doc[pid] = {"median_sigma_nominal": float(np.nanmedian(nom["sigma"]))}
    for name, d in ALTERNATIVES.items():
        diff = load(d, pid)["lam"] - nom["lam"]
        ok = np.isfinite(diff)
        doc[pid][name] = {
            "mean": float(diff[ok].mean()),
            "rms": float(diff[ok].std()),
            "rms_outer_t_gt_0.5": float(diff[ok & big].std()),
            "max_abs_smoothed": float(
                np.nanmax(np.abs(load(d, pid)["smoothed"] - nom["smoothed"]))
            ),
        }
(joint / "compare.json").write_text(json.dumps(doc, indent=1))
print(json.dumps(doc, indent=1))

frame.set_frame("detector")
style.apply()


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


fig, axes = plt.subplots(len(pids), 4, figsize=(15, 3.6 * len(pids)), layout="constrained")
lo, hi = view.lam_range
c = view.contrast_range
for row, pid in zip(axes, pids, strict=True):
    nom = load(joint / f"beams2d_{NOMINAL}", pid)
    e = nom["edges"]
    show(
        row[0], nom["smoothed"], e, f"{pid}: λ, joint roof A+B (φ=0)", style.SEQUENTIAL, lo, hi, "λ"
    )
    flip = load(joint / "beams2d_m3_phi180", pid)["smoothed"] - nom["smoothed"]
    show(row[1], flip, e, f"{pid}: λ(φ=180°) − λ(φ=0)", "RdBu_r", -0.06, 0.06, "Δλ")
    single = load(swap / "beams2d_SKY", pid)["smoothed"] - nom["smoothed"]
    show(row[2], single, e, f"{pid}: λ(whole run) − λ(joint)", "RdBu_r", -0.06, 0.06, "Δλ")
    show(row[3], nom["contrast"], e, f"{pid}: local contrast, joint", "RdBu_r", -c, c, "Δλ local")
style.label_panels(axes.ravel())
fig.savefig(fig_out, dpi=150)
