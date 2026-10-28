"""Stripe measurements on the 2D opacity maps (PREREG-maps2d.md).

Per hit selection: beam and transverse profiles of each position against the joint
open-sky reference, Gaussian-on-line fits of their stripes, the pos0-pos1 stripe shifts,
and Poisson-replica errors. Arrays are [i_x, j_y] in the analysis frame (beams along y);
the output names axes as the detector does (beams along X: analysis x is detector Y).

Usage: maps2d.py OUT_DIR NAME=JOINT_INGEST_DIR [NAME=JOINT_INGEST_DIR ...]
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from cafetomo.angular import AngularHist, load_counts
from cafetomo.beamview import opacity_view
from cafetomo.config import load_config
from cafetomo.ingest import live_times

BAND = 0.25  # |tan| across which a profile is averaged
REACH = 0.65  # |tan| along which a profile is kept
MIN_BINS = 5  # measured bins a profile point needs
PROMINENCE = 2.0  # stripe prominence in units of the profile point's sigma
HALF_WINDOW = 0.10  # +-tan of a stripe fit around its maximum
WIDTH_BOUNDS = (0.01, 0.30)  # FWHM bounds of a stripe fit
SHIFT_RANGE = (0.15, 0.40)  # pos1 shift searched for the beam matching
SHIFT_STEP = 0.001
N_REPLICAS = 200
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class Stripe:
    centre: float
    fwhm: float
    contrast: float
    flagged: bool


def profile(lam, sigma, centres, along_axis: int):
    """Weighted mean of λ over |t| <= BAND across the other axis, as a function of the
    tangent along `along_axis` (0 = analysis x = detector Y)."""
    lam, sigma = (np.moveaxis(a, along_axis, 0) for a in (lam, sigma))
    cross = np.abs(centres) <= BAND
    w = np.where(np.isfinite(lam), 1.0 / np.where(np.isfinite(sigma), sigma, 1.0) ** 2, 0.0)
    w = w[:, cross]
    n = (w > 0).sum(axis=1)
    sw = w.sum(axis=1)
    mean = (np.where(w > 0, lam[:, cross], 0.0) * w).sum(axis=1) / np.where(sw > 0, sw, 1.0)
    keep = (n >= MIN_BINS) & (np.abs(centres) <= REACH)
    return centres[keep], mean[keep], 1.0 / np.sqrt(sw[keep])


def _model(t, c, s, a, b0, b1):
    return a * np.exp(-0.5 * ((t - c) / s) ** 2) + b0 + b1 * (t - c)


def fit_stripe(t, y, e, t0) -> Stripe:
    win = np.abs(t - t0) <= HALF_WINDOW
    tw, yw, ew = t[win], y[win], e[win]
    lo_s, hi_s = WIDTH_BOUNDS[0] / FWHM, WIDTH_BOUNDS[1] / FWHM
    edge = np.r_[yw[:2], yw[-2:]].mean()
    p0 = (t0, 0.04 / FWHM, max(y[np.argmin(np.abs(t - t0))] - edge, 1e-3), edge, 0.0)
    bounds = (
        [t0 - HALF_WINDOW, lo_s, 0.0, -np.inf, -np.inf],
        [t0 + HALF_WINDOW, hi_s, np.inf, np.inf, np.inf],
    )
    p, _ = curve_fit(
        _model, tw, yw, p0=p0, sigma=ew, absolute_sigma=True, bounds=bounds, maxfev=20000
    )
    c, s, a = p[:3]
    at_bound = (
        np.isclose(c, t0 - HALF_WINDOW) or np.isclose(c, t0 + HALF_WINDOW)
        or np.isclose(s, lo_s, rtol=1e-3) or np.isclose(s, hi_s, rtol=1e-3)
    )  # fmt: skip
    return Stripe(float(c), float(s * FWHM), float(a), bool(at_bound))


def stripe_maxima(t, y, e) -> np.ndarray:
    idx, _ = find_peaks(y, prominence=PROMINENCE * float(np.median(e)))
    return t[idx]


def best_shift(t0, y0, t1, y1) -> tuple[float, float]:
    """Shift s (a pos1 point at t lands at t + s) maximizing the Pearson correlation of
    pos1's profile with pos0's on common tangents, over SHIFT_RANGE in either sign."""
    best = (np.nan, -np.inf)
    step = t0[1] - t0[0]
    grid = np.arange(SHIFT_RANGE[0], SHIFT_RANGE[1] + SHIFT_STEP / 2, SHIFT_STEP)
    for s in np.r_[-grid[::-1], grid]:
        y1s = np.interp(t0, t1 + s, y1, left=np.nan, right=np.nan)
        ok = np.isfinite(y1s)
        if ok.sum() < 0.25 / step:
            continue
        r = np.corrcoef(y0[ok], y1s[ok])[0, 1]
        if r > best[1]:
            best = (float(s), float(r))
    return best


def views(ingest: Path, cfg, rng=None):
    """Opacity view per position; with `rng`, from one Poisson replica of all counts."""
    sky_id = cfg.sky_reference.id
    pids = list(cfg.position_ids)
    live = live_times(ingest, [sky_id, *pids])

    def counts(sid):
        h = load_counts(ingest / f"counts_{sid}.npz")
        v = h.values if rng is None else rng.poisson(np.clip(h.values, 0, None))
        return AngularHist(v, h.xedges, h.yedges, h.name)

    sky = counts(sky_id)
    return {
        pid: opacity_view(counts(pid), sky, live[pid] / live[sky_id], cfg.view, cfg.opacity.t_floor)
        for pid in pids
    }


def measure(vs, nominal=None) -> dict:
    """Profiles, stripes and shifts; with `nominal`, refit its stripes at its maxima."""
    out = {}
    for pid, v in vs.items():
        c = 0.5 * (v.edges[1:] + v.edges[:-1])
        out[pid] = {}
        for name, axis in (("beam", 0), ("transverse", 1)):
            t, y, e = profile(v.lam, v.sigma, c, axis)
            maxima = stripe_maxima(t, y, e) if nominal is None else nominal[pid][name]["maxima"]
            out[pid][name] = {
                "t": t, "lam": y, "sigma": e, "maxima": np.asarray(maxima),
                "stripes": [fit_stripe(t, y, e, m) for m in maxima],
            }  # fmt: skip
    p0, p1 = (out[p]["beam"] for p in vs)
    s, r = best_shift(p0["t"], p0["lam"], p1["t"], p1["lam"])
    out["shift"], out["shift_corr"] = s, r
    return out


def summary(m: dict, pids) -> dict:
    """Scalars of one measurement (nominal or replica), with stripe pairing by the shift."""
    doc = {"shift": m["shift"], "shift_corr": m["shift_corr"]}
    for pid in pids:
        b = m[pid]["beam"]["stripes"]
        good = [s for s in b if not s.flagged]
        doc[f"{pid}_n_stripes"] = len(b)
        doc[f"{pid}_n_flagged"] = len(b) - len(good)
        doc[f"{pid}_fwhm"] = float(np.mean([s.fwhm for s in good]))
        doc[f"{pid}_contrast"] = float(np.mean([s.contrast for s in good]))
        cs = np.sort([s.centre for s in good])
        doc[f"{pid}_period"] = float(np.mean(np.diff(cs))) if cs.size > 1 else np.nan
        tr = m[pid]["transverse"]["stripes"]
        doc[f"{pid}_transverse_centre"] = tr[0].centre if len(tr) == 1 else np.nan
        doc[f"{pid}_transverse_fwhm"] = tr[0].fwhm if len(tr) == 1 else np.nan
        doc[f"{pid}_transverse_contrast"] = tr[0].contrast if len(tr) == 1 else np.nan
    a, b = (m[p]["beam"]["stripes"] for p in pids)
    pairs = []
    for s0 in a:
        if s0.flagged or not b:
            continue
        s1 = min(b, key=lambda s: abs(s.centre + m["shift"] - s0.centre))
        if not s1.flagged and abs(s1.centre + m["shift"] - s0.centre) < 0.5 * HALF_WINDOW:
            pairs.append(s0.centre - s1.centre)
    doc["n_pairs"] = len(pairs)
    doc["pair_shifts"] = pairs
    doc["pair_shift_mean"] = float(np.mean(pairs)) if pairs else np.nan
    doc["pair_shift_scatter"] = float(np.std(pairs, ddof=1)) if len(pairs) > 1 else np.nan
    doc["transverse_shift"] = (
        doc[f"{pids[0]}_transverse_centre"] - doc[f"{pids[1]}_transverse_centre"]
    )
    return doc


def run(name: str, ingest: Path, cfg, index: int) -> dict:
    pids = list(cfg.position_ids)
    nominal = measure(views(ingest, cfg))
    nom = summary(nominal, pids)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.uncertainty.seed, spawn_key=(3, index)))
    raw = [measure(views(ingest, cfg, rng), nominal) for _ in range(N_REPLICAS)]
    reps = [summary(r, pids) for r in raw]
    sig = {
        k: float(np.nanstd([r[k] for r in reps], ddof=1))
        for k, v in nom.items()
        if isinstance(v, float)
    }
    stripes = {
        pid: {
            part: [
                asdict(st)
                | {
                    f"{k}_sigma": float(
                        np.nanstd([getattr(r[pid][part]["stripes"][i], k) for r in raw], ddof=1)
                    )
                    for k in ("centre", "fwhm", "contrast")
                }
                for i, st in enumerate(nominal[pid][part]["stripes"])
            ]
            for part in ("beam", "transverse")
        }
        for pid in pids
    }
    profiles = {
        pid: {part: {q: nominal[pid][part][q].tolist() for q in ("t", "lam", "sigma")}
              for part in ("beam", "transverse")}
        for pid in pids
    }  # fmt: skip
    return {"selection": name, "ingest": str(ingest), "nominal": nom, "sigma": sig,
            "stripes": stripes, "profiles": profiles, "n_replicas": N_REPLICAS}  # fmt: skip


def main(argv) -> None:
    out = Path(argv[0])
    if out.exists():
        raise SystemExit(f"{out} exists: choose a fresh output directory")
    jobs = [a.split("=", 1) for a in argv[1:]]
    cfg = load_config("configs/cafeteria.yaml", pose_file="results/pose.json")
    docs = [run(n, Path(d), cfg, i) for i, (n, d) in enumerate(jobs)]
    out.mkdir(parents=True)
    for d in docs:
        (out / f"maps2d_{d['selection']}.json").write_text(
            json.dumps(d, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
