"""Joint open-sky reference from both roof orientations (PREREG-joint-sky.md).

Profiling the per-bin detector response leaves n_A ~ Binomial(n_A + n_B, p) with
logit p = beta0 + s(w_A) - s(w_B); s is the sky's azimuthal structure (m = 1..M harmonics,
quadratic radial coefficients) in the world frame. The fitted s rebuilds a reference for
a room position at world azimuth phi from all roof counts.
Usage: joint_sky.py SKYSWAP_ROOT OUT_ROOT
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

from cafetomo.angular import AngularHist, load_counts, rebin, save_counts
from cafetomo.config import Pose, load_config
from cafetomo.ingest import live_times
from cafetomo.sky import detector_to_sky

src, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = load_config(src / "config.yaml", pose_file="results/pose.json")
TURN = cfg.raw.subsets["SKY_B"].turn_deg - cfg.raw.subsets["SKY_A"].turn_deg
PHIS = {"phi0": 0.0, "phi180": 180.0, "phiB": TURN, "phiB180": TURN + 180.0}
REBIN = cfg.view.rebin_factor

hist = {s: load_counts(src / f"ingest_{s}" / "counts_SKY.npz") for s in ("SKY_A", "SKY_B")}
T = {s: live_times(src / f"ingest_{s}", ["SKY"])["SKY"] for s in hist}
n_a, n_b = (hist[s].values.astype(np.float64) for s in ("SKY_A", "SKY_B"))
xc = 0.5 * (hist["SKY_A"].xedges[1:] + hist["SKY_A"].xedges[:-1])
yc = 0.5 * (hist["SKY_A"].yedges[1:] + hist["SKY_A"].yedges[:-1])
tx, ty = np.meshgrid(xc, yc, indexing="ij")
use = (cfg.detector.acceptance(tx, ty) > 0) & (n_a + n_b >= 1)


def block_sum(v: np.ndarray) -> np.ndarray:
    """Sum over each view bin's REBIN x REBIN fine bins, in floating point (`rebin` keeps
    integer counts and would truncate expected values)."""
    m = v.shape[0] // REBIN
    return np.asarray(v, dtype=np.float64).reshape(m, REBIN, m, REBIN).sum(axis=(1, 3))


def features(az_deg: float, m_max: int) -> np.ndarray:
    """Columns tau^(m+k) cos/sin(m psi), m = 1..m_max, k = 0..2, at the world direction
    of every fine detector bin for a detector at azimuth `az_deg`."""
    sx, sy, _ = detector_to_sky(tx, ty, Pose(0.0, 0.0, 0.0, az_deg))
    tau, psi = np.hypot(sx, sy), np.arctan2(sy, sx)
    cols = [
        tau ** (m + k) * f(m * psi)
        for m in range(1, m_max + 1)
        for k in range(3)
        for f in (np.cos, np.sin)
    ]
    return np.stack(cols, axis=-1)


def fit(m_max: int) -> dict:
    fa, fb = features(0.0, m_max), features(TURN, m_max)
    x = np.column_stack([np.ones(use.sum()), (fa - fb)[use]])
    k, n = n_a[use], (n_a + n_b)[use]
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(T["SKY_A"] / T["SKY_B"])
    for _ in range(50):
        p = expit(x @ beta)
        w = n * p * (1 - p)
        hess = x.T @ (x * w[:, None])
        step = np.linalg.solve(hess, x.T @ (k - n * p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise RuntimeError("binomial fit did not converge")
    cov = np.linalg.inv(hess)
    c = T["SKY_A"] / T["SKY_B"] * np.exp(-beta[0])
    return {"m_max": m_max, "beta": beta, "cov": cov, "c": c, "fa": fa, "fb": fb}


def checks(f: dict) -> dict:
    """Goodness of fit and detector-frame residual structure on view bins."""
    logit = f["beta"][0] + (f["fa"] - f["fb"]) @ f["beta"][1:]
    p = np.where(use, expit(logit), 0.0)
    n = np.where(use, n_a + n_b, 0.0)
    obs = np.where(use, n_a, 0.0)

    coarse = block_sum
    o, e, var = coarse(obs), coarse(n * p), coarse(n * p * (1 - p))
    ok = (coarse(n) >= cfg.view.min_sky_counts) & (var > 0)
    pull = np.where(ok, (o - e) / np.sqrt(np.where(ok, var, 1.0)), np.nan)
    dof = int(ok.sum()) - len(f["beta"])
    chi2 = float(np.nansum(pull**2))
    rows = {}
    for axis, name in ((1, "rows_along_tx"), (0, "rows_along_ty")):
        cnt = ok.sum(axis=axis)
        mean = np.nansum(pull, axis=axis) / np.maximum(cnt, 1)
        bad = (cnt > 0) & (np.abs(mean) > 3 / np.sqrt(np.maximum(cnt, 1)))
        rows[name] = {"rows": int((cnt > 0).sum()), "failing": int(bad.sum())}
    rows_pass = all(r["failing"] <= 0.05 * r["rows"] for r in rows.values())
    band = 3 * np.sqrt(2 / dof)
    return {
        "chi2_per_dof": chi2 / dof,
        "dof": dof,
        "chi2_pass_band": [1 - band, 1 + band],
        "chi2_pass": bool(abs(chi2 / dof - 1) <= band),
        "projection": rows,
        "projection_pass": bool(rows_pass),
        "pull_map": pull,
        "split": {"n": coarse(n), "observed_a": o, "expected_a": e, "used": ok},
    }


MOD_TAU = 0.5  # sky tangent of the circle on which the modulation is quoted
MOD_DRAWS = 2000


def modulation(f: dict, rng: np.random.Generator) -> dict:
    """Azimuthal modulation of the sky flux on the circle tau = MOD_TAU, in percent of its
    azimuthal mean (m = 0 is not identified): peak-to-peak of exp(s) and the amplitude of
    each harmonic, with errors from Gaussian draws of the fitted coefficients."""
    m_max = f["m_max"]
    psi = np.linspace(-np.pi, np.pi, 721)[:-1]
    cols = np.stack(
        [
            MOD_TAU ** (m + k) * fn(m * psi)
            for m in range(1, m_max + 1)
            for k in range(3)
            for fn in (np.cos, np.sin)
        ],
        axis=-1,
    )
    draws = rng.multivariate_normal(f["beta"][1:], f["cov"][1:, 1:], MOD_DRAWS)

    def stats(beta):
        s = cols @ beta
        e = np.exp(s - np.log(np.mean(np.exp(s))))
        amps = []
        for m in range(1, m_max + 1):
            a = np.mean(s * np.cos(m * psi)) * 2
            b = np.mean(s * np.sin(m * psi)) * 2
            amps.append(100 * np.hypot(a, b))
        return [100 * (e.max() - e.min()), *amps]

    nominal = stats(f["beta"][1:])
    spread = np.std([stats(b) for b in draws], axis=0, ddof=1)
    doc = {"tau": MOD_TAU, "peak_to_peak_pct": nominal[0], "peak_to_peak_pct_err": spread[0]}
    for m in range(1, m_max + 1):
        doc[f"m{m}_pct"], doc[f"m{m}_pct_err"] = nominal[m], spread[m]
    return doc


def reference(f: dict, phi: float) -> np.ndarray:
    m = f["m_max"]
    s = {k: features(a, m) @ f["beta"][1:] for k, a in (("a", 0.0), ("b", TURN), ("p", phi))}
    t_a, t_b = T["SKY_A"], T["SKY_B"]
    denom = t_a * np.exp(s["a"]) + f["c"] * t_b * np.exp(s["b"])
    return np.where(use, (n_a + n_b) * (t_a + t_b) * np.exp(s["p"]) / denom, 0.0)


def write_ingest(ref: np.ndarray, name: str) -> Path:
    """An ingest directory with the room positions' counts and `ref` as the sky."""
    d = out / f"ingest_{name}"
    d.mkdir(parents=True)
    full = src / "ingest_SKY"
    meta = json.loads((full / "meta.json").read_text())
    for pid in cfg.position_ids:
        shutil.copy(full / f"counts_{pid}.npz", d / f"counts_{pid}.npz")
    sky_meta = dict(meta["exposures"]["SKY"])
    sky_meta.update(
        live_time_s=T["SKY_A"] + T["SKY_B"],
        source=f"joint SKY_A+SKY_B reference, {name}",
        live_time_estimator="sum of SKY_A and SKY_B exact clock sums",
    )
    meta["exposures"]["SKY"] = sky_meta
    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    save_counts(AngularHist(ref, hist["SKY_A"].xedges, hist["SKY_A"].yedges), d, "SKY", sky_meta)
    return d


def coarse_mean(v: np.ndarray) -> np.ndarray:
    """Mean of a fine-bin map over each view bin's used fine bins (NaN where none)."""
    w, tot = block_sum(use), block_sum(np.where(use, v, 0.0))
    return np.where(w > 0, tot / np.where(w > 0, w, 1.0), np.nan)


def uniform_pull() -> np.ndarray:
    """Pulls of the roof-A share per view bin for a sky without azimuthal structure
    (one common share), on the view bins of `checks`."""
    o, n = block_sum(np.where(use, n_a, 0.0)), block_sum(np.where(use, n_a + n_b, 0.0))
    p0 = o.sum() / n.sum()
    ok = n >= cfg.view.min_sky_counts
    var = np.where(ok, n * p0 * (1 - p0), 1.0)
    return np.where(ok, (o - n * p0) / np.sqrt(var), np.nan)


out.mkdir(parents=True, exist_ok=False)
summary = {"turn_deg": TURN, "live_h": {s: t / 3600 for s, t in T.items()}, "fits": {}}
pulls, split = {}, {}
for m_max in (2, 3):
    f = fit(m_max)
    ch = checks(f)
    pulls[m_max] = ch.pop("pull_map")
    split[m_max] = ch.pop("split")
    err = np.sqrt(np.diag(f["cov"]))
    c_err = f["c"] * err[0]
    summary["fits"][f"m{m_max}"] = {
        "c": f["c"],
        "c_err": c_err,
        "c_flag": bool(abs(f["c"] - 1) > 0.02),
        "coefficients": f["beta"][1:].tolist(),
        "coefficient_errors": err[1:].tolist(),
        "covariance": f["cov"].tolist(),
        "modulation": modulation(
            f,
            np.random.default_rng(
                np.random.SeedSequence(cfg.uncertainty.seed, spawn_key=(4, m_max))
            ),
        ),
        **ch,
    }
    for name, phi in PHIS.items():
        ref = reference(f, phi)
        np.save(out / f"ref_m{m_max}_{name}.npy", ref)
        write_ingest(ref, f"m{m_max}_{name}")
np.savez_compressed(out / "pulls.npz", **{f"m{k}": v for k, v in pulls.items()})
# Per view bin, nominal fit: roof tracks, roof-A share observed and fitted, live times.
np.savez_compressed(
    out / "split.npz",
    edges=rebin(AngularHist(n_a, hist["SKY_A"].xedges, hist["SKY_A"].yedges), REBIN).xedges,
    t_a=T["SKY_A"],
    t_b=T["SKY_B"],
    pull=pulls[3],
    pull_uniform=uniform_pull(),
    sky_log=coarse_mean(features(0.0, 3) @ f["beta"][1:]),
    reference=block_sum(reference(f, 0.0)),
    **split[3],
)
(out / "joint_sky.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
