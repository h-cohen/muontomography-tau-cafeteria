"""Per-channel charge calibration from the single-hit spectra of the raw data.

A channel that fires alone in its ASIC saw a through-going muon crossing one bar, so its
high-gain charge, net of the channel's pedestal, follows a Landau distribution smeared by
the electronics. Each channel's spectrum is fitted with a Landau convolved with a Gaussian
by Poisson maximum likelihood over a window that starts at a fraction of the peak and stops
below the saturated top of the range. The channel's MPV is the mode of the fitted curve;
its error is the fit covariance propagated to that mode.

Fitting the days separately as well gives the drift of the MPV over the campaign. A channel
is flagged when it has too few single hits, its fit fails, the fitted MPV departs from the
empirical peak of the spectrum by more than `peak_tol_frac` (the model does not describe
it), or its MPV departs from the median of its ASIC by more than `outlier_frac`.
The model is not exact: triangular bars add a path-length spread, so the tail runs above
the fit and chi2 per degree of freedom is far above one; the empirical peak is the
model-independent check on the MPV. Shared hits are excluded: charge sharing
between neighbouring bars adds a flat low-charge pile that is not a Landau.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.stats import landau

from cafetomo.config import CalibrationSettings
from cafetomo.rawdata import CHARGE_EDGES, N_ASICS, N_CHANNELS, PEDESTAL_EDGES

_STEP = 10.0  # ADC step of the convolution grid
_MIN_SIGMA = 0.5 * _STEP
_MIN_WINDOW_BINS = 8
_PEAK_FLOOR_ADC = 1000.0  # a spectrum peaking below this net charge is not a muon peak


def langau(x: np.ndarray, loc: float, xi: float, sigma: float) -> np.ndarray:
    """Landau (location `loc`, scale `xi`) convolved with a Gaussian of width `sigma`, as a
    density in x. The standard Landau has its mode 0.4293 scales below its location."""
    x = np.asarray(x, dtype=np.float64)
    sigma = max(float(sigma), _MIN_SIGMA)
    pad = 8.0 * sigma
    grid = np.arange(x.min() - pad, x.max() + pad + _STEP, _STEP)
    kernel_x = np.arange(-pad, pad + _STEP, _STEP)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    smeared = fftconvolve(landau.pdf(grid, loc=loc, scale=xi), kernel, mode="same")
    return np.interp(x, grid, smeared)


def _mode(loc: float, xi: float, sigma: float) -> float:
    """Peak of the curve, from a parabola through the best grid point and its neighbours
    (a grid-quantized mode would add noise to its numerical derivatives)."""
    x = np.arange(loc - 3.0 * xi, loc + 3.0 * xi, 1.0)
    y = langau(x, loc, xi, sigma)
    i = int(np.clip(np.argmax(y), 1, x.size - 2))
    left, mid, right = y[i - 1], y[i], y[i + 1]
    return float(x[i] + 0.5 * (left - right) / (left - 2.0 * mid + right))


@dataclass(frozen=True)
class ChannelFit:
    ok: bool
    status: str  # ok | low_stats | fit_failed | inactive
    n: int  # single-hit counts
    mpv: float = np.nan  # net of pedestal, ADC
    mpv_err: float = np.nan
    peak: float = np.nan  # empirical peak: the mode of the 3-bin smoothed spectrum, net ADC
    above_window: float = np.nan  # fraction of single hits above the fit window (saturation)
    loc: float = np.nan
    xi: float = np.nan
    sigma: float = np.nan
    amp: float = np.nan
    chi2_ndf: float = np.nan
    lo: float = np.nan  # fit window, net ADC
    hi: float = np.nan


def _failed(status: str, n: int) -> ChannelFit:
    return ChannelFit(False, status, n)


def _deviance_residuals(n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.clip(mu, 1e-9, None)
    log_term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0)
    return np.sign(n - mu) * np.sqrt(np.clip(2.0 * (mu - n + log_term), 0.0, None))


def fit_channel(hist: np.ndarray, pedestal: float, settings: CalibrationSettings) -> ChannelFit:
    """Fit one channel's single-hit charge histogram (on `CHARGE_EDGES`)."""
    counts = np.asarray(hist, dtype=np.float64)
    n_total = int(counts.sum())
    if n_total < settings.min_solo_counts:
        return _failed("low_stats", n_total)
    width = float(CHARGE_EDGES[1] - CHARGE_EDGES[0])
    net = 0.5 * (CHARGE_EDGES[:-1] + CHARGE_EDGES[1:]) - pedestal
    in_range = (net > _PEAK_FLOOR_ADC) & (net < settings.fit_max_adc)
    peak = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    if not in_range.any() or counts[in_range].sum() == 0:
        return _failed("fit_failed", n_total)
    mode0 = float(net[in_range][np.argmax(peak[in_range])])
    window = (net >= settings.fit_min_frac * mode0) & (net <= settings.fit_max_adc)
    if (counts[window] > 0).sum() < _MIN_WINDOW_BINS:
        return _failed("fit_failed", n_total)
    x, y = net[window], counts[window]

    def model(theta):
        loc, xi, sigma, amp = theta
        return amp * width * langau(x, loc, xi, sigma)

    xi0 = 0.08 * mode0
    start = [mode0 + 0.4293 * xi0, xi0, 0.05 * mode0, y.sum()]
    lower = [0.0, 10.0, _MIN_SIGMA, 1.0]
    upper = [3.0 * settings.fit_max_adc, 5000.0, 3000.0, 20.0 * y.sum()]
    try:
        res = least_squares(
            lambda t: _deviance_residuals(y, model(t)), start, bounds=(lower, upper), x_scale="jac"
        )
    except ValueError:
        return _failed("fit_failed", n_total)
    loc, xi, sigma, amp = res.x
    # Fisher information from central differences of the model: the optimizer's own
    # finite-difference Jacobian is too noisy on the interpolated curve to invert.
    steps = np.array([5.0, 5.0, 5.0, 0.01 * amp])
    cols = []
    for i in range(4):
        up, dn = res.x.copy(), res.x.copy()
        up[i] += steps[i]
        dn[i] -= steps[i]
        cols.append((model(up) - model(dn)) / (2.0 * steps[i]))
    fisher = np.array(cols) / np.sqrt(np.clip(model(res.x), 1e-9, None))
    try:
        cov = np.linalg.inv(fisher @ fisher.T)
    except np.linalg.LinAlgError:
        return _failed("fit_failed", n_total)
    if not res.success or not np.all(np.isfinite(cov)):
        return _failed("fit_failed", n_total)

    def mpv_of(t):
        return _mode(t[0], t[1], t[2])

    grad = np.zeros(3)
    for i, step in enumerate((5.0, 5.0, 5.0)):
        up, dn = res.x[:3].copy(), res.x[:3].copy()
        up[i] += step
        dn[i] -= step
        grad[i] = (mpv_of(up) - mpv_of(dn)) / (2.0 * step)
    err = float(np.sqrt(max(grad @ cov[:3, :3] @ grad, 0.0)))
    mu = model(res.x)
    pearson = float((((y - mu) ** 2) / np.clip(mu, 1e-9, None)).sum() / (y.size - 4))
    above = float(counts[net > settings.fit_max_adc].sum() / n_total)
    return ChannelFit(
        True, "ok", n_total, mpv_of(res.x), err, mode0, above, loc, xi, sigma, amp, pearson,
        float(x[0]), float(x[-1]),
    )  # fmt: skip


def _pedestal(ped_hist: np.ndarray) -> float:
    """Median of one channel's unflagged charge, from its pedestal histogram."""
    cum = np.cumsum(ped_hist)
    if cum[-1] == 0:
        return float("nan")
    centres = 0.5 * (PEDESTAL_EDGES[:-1] + PEDESTAL_EDGES[1:])
    return float(centres[np.searchsorted(cum, 0.5 * cum[-1])])


def _pedestal_width(ped_hist: np.ndarray) -> float:
    cum = np.cumsum(ped_hist)
    if cum[-1] == 0:
        return float("nan")
    centres = 0.5 * (PEDESTAL_EDGES[:-1] + PEDESTAL_EDGES[1:])
    lo, hi = (centres[np.searchsorted(cum, q * cum[-1])] for q in (0.16, 0.84))
    return float(0.5 * (hi - lo))


def exposure_gains(files, channel_bars: tuple, settings: CalibrationSettings) -> np.ndarray:
    """`relative_gains` of the calibration fitted on the summed spectra of `files`."""
    doc = summarize(
        [sum(f.hit_hist for f in files)],
        [sum(f.ped_hist for f in files)],
        n_flagged=sum(f.n_flagged for f in files),
        n_valid_rows=sum(f.n_valid_rows for f in files),
        channel_bars=channel_bars,
        settings=settings,
    )
    return relative_gains(doc)


def relative_gains(doc: dict) -> np.ndarray:
    """Gain of every (ASIC, channel) relative to its ASIC: the MPV over the median MPV of
    the ASIC's good channels; 1 where a channel has no good calibration."""
    gains = np.ones((N_ASICS, N_CHANNELS))
    ok = [r for r in doc["channels"] if r["status"] == "ok"]
    for a in {r["asic"] for r in ok}:
        rows = [r for r in ok if r["asic"] == a]
        median = np.median([r["mpv_adc"] for r in rows])
        for r in rows:
            gains[a, r["channel"]] = r["mpv_adc"] / median
    return gains


def summarize(
    hit_hists: list[np.ndarray],
    ped_hists: list[np.ndarray],
    n_flagged: np.ndarray,
    n_valid_rows: int,
    channel_bars: tuple,
    settings: CalibrationSettings,
) -> dict:
    """Fit every channel on the summed spectra and on each file; return the record.

    `hit_hists` and `ped_hists` hold one (ASIC, channel, bin) array per file."""
    total_hit = sum(hit_hists)
    total_ped = sum(ped_hists)
    rows = []
    for a in range(N_ASICS):
        bar_of = {ch: b for b, ch in enumerate(channel_bars[a])}
        for c in range(N_CHANNELS):
            ped = _pedestal(total_ped[a, c])
            fit = (
                fit_channel(total_hit[a, c], ped, settings)
                if np.isfinite(ped)
                else _failed("low_stats", int(total_hit[a, c].sum()))
            )
            if fit.status == "low_stats" and bar_of.get(c) is None:
                fit = ChannelFit(False, "inactive", fit.n)  # an unmapped channel with no signal
            days = []
            for h, p in zip(hit_hists, ped_hists, strict=True):
                pd = _pedestal(p[a, c])
                days.append(
                    fit_channel(h[a, c], pd, settings).mpv if np.isfinite(pd) else float("nan")
                )
            finite = [d for d in days if np.isfinite(d)]
            rows.append(
                {
                    "asic": a,
                    "channel": c,
                    "bar": bar_of.get(c),
                    "status": fit.status,
                    "n_solo": fit.n,
                    "hit_fraction": float(n_flagged[a, c]) / max(n_valid_rows, 1),
                    "pedestal_adc": ped,
                    "pedestal_width_adc": _pedestal_width(total_ped[a, c]),
                    "mpv_adc": fit.mpv,
                    "mpv_err_adc": fit.mpv_err,
                    "peak_adc": fit.peak,
                    "above_window_fraction": fit.above_window,
                    "landau_scale_adc": fit.xi,
                    "gauss_sigma_adc": fit.sigma,
                    "chi2_ndf": fit.chi2_ndf,
                    "fit": {"loc": fit.loc, "xi": fit.xi, "sigma": fit.sigma, "amp": fit.amp},
                    "fit_window_adc": [fit.lo, fit.hi],
                    "day_mpv_adc": days,
                    "day_rms_adc": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                }
            )
    asics = []
    for a in range(N_ASICS):
        for r in rows:
            if (
                r["asic"] == a
                and r["status"] == "ok"
                and abs(r["mpv_adc"] / r["peak_adc"] - 1.0) > settings.peak_tol_frac
            ):
                r["status"] = "poor_fit"
        ok = [r for r in rows if r["asic"] == a and r["status"] == "ok"]
        median = float(np.median([r["mpv_adc"] for r in ok])) if ok else float("nan")
        for r in ok:
            if abs(r["mpv_adc"] / median - 1.0) > settings.outlier_frac:
                r["status"] = "mpv_outlier"
        kept = [r for r in rows if r["asic"] == a and r["status"] == "ok"]
        asics.append(
            {
                "asic": a,
                "median_mpv_adc": median,
                "mpv_rms_frac": float(np.std([r["mpv_adc"] for r in kept], ddof=1) / median)
                if len(kept) > 1
                else None,
                "n_ok": len(kept),
            }
        )
    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    n_ok = status_counts.get("ok", 0)
    n_inactive = status_counts.get("inactive", 0)
    return {
        "n_files": len(hit_hists),
        "n_valid_rows": int(n_valid_rows),
        "status_counts": status_counts,
        "n_ok": n_ok,
        "n_inactive": n_inactive,
        "n_flagged": len(rows) - n_ok - n_inactive,
        "asics": asics,
        "channels": rows,
    }


def render(
    doc: dict,
    hit_hists: list[np.ndarray],
    ped_hists: list[np.ndarray],
    out_dir: Path,
    name: str,
    settings: CalibrationSettings,
) -> None:
    """Write <name>_channelcal.json, one spectrum grid per ASIC and an MPV summary figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_channelcal.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True, default=float) + "\n"
    )
    total = sum(hit_hists)
    width = float(CHARGE_EDGES[1] - CHARGE_EDGES[0])
    centres = 0.5 * (CHARGE_EDGES[:-1] + CHARGE_EDGES[1:])
    by_key = {(r["asic"], r["channel"]): r for r in doc["channels"]}
    for a in range(N_ASICS):
        fig, axes = plt.subplots(4, 8, figsize=(20, 9), sharex=True)
        for c, ax in enumerate(axes.ravel()):
            r = by_key[(a, c)]
            net = centres - r["pedestal_adc"] if np.isfinite(r["pedestal_adc"]) else centres
            ax.step(net, total[a, c], where="mid", color="#52514e", lw=0.8)
            if r["fit"]["amp"] is not None and np.isfinite(r["fit"]["amp"]):
                f = r["fit"]
                lo, hi = r["fit_window_adc"]
                xs = net[(net >= lo) & (net <= hi)]
                ax.plot(
                    xs,
                    f["amp"] * width * langau(xs, f["loc"], f["xi"], f["sigma"]),
                    "#d1495b",
                    lw=1.2,
                )
                ax.axvline(r["mpv_adc"], color="#1b6ca8", lw=0.8)
            tag = "ok" if r["status"] == "ok" else r["status"]
            bar = "-" if r["bar"] is None else r["bar"]
            mpv = f"{r['mpv_adc']:.0f}" if np.isfinite(r["mpv_adc"]) else "n/a"
            ax.set_title(f"ch{c} bar{bar}  MPV {mpv}  {tag}", fontsize=7)
            ax.set_xlim(0, 12000)
            ax.tick_params(labelsize=6)
        fig.suptitle(
            f"{name} ASIC {a}: single-hit charge net of pedestal (ADC), Landau x Gauss fit",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out / f"{name}_channelcal_asic{a}.png", dpi=110)
        plt.close(fig)

    fig, axes = plt.subplots(1, N_ASICS, figsize=(18, 4.2), sharey=True)
    for a, ax in enumerate(axes):
        rows = [by_key[(a, c)] for c in range(N_CHANNELS)]
        ch = np.arange(N_CHANNELS)
        for r in rows:
            days = np.array([d if d is not None else np.nan for d in r["day_mpv_adc"]], float)
            ax.plot([r["channel"]] * days.size, days, ".", color="#c8c7c0", ms=3)
        mpv = np.array([r["mpv_adc"] for r in rows])
        err = np.array([r["mpv_err_adc"] for r in rows])
        mapped = np.array([r["bar"] is not None for r in rows])
        bad = np.array([r["status"] not in ("ok", "inactive") for r in rows])
        ax.errorbar(
            ch[mapped & ~bad],
            mpv[mapped & ~bad],
            err[mapped & ~bad],
            fmt="o",
            ms=3,
            color="#1b6ca8",
        )
        ax.errorbar(
            ch[~mapped & ~bad],
            mpv[~mapped & ~bad],
            err[~mapped & ~bad],
            fmt="s",
            ms=3,
            color="#898781",
        )
        lo, hi = settings.mpv_axis_adc
        # a flagged channel without an MPV sits on the bottom edge of the shared axis
        shown = np.clip(np.nan_to_num(mpv[bad], nan=lo), lo, hi)
        ax.plot(ch[bad], shown, "x", color="#d1495b", clip_on=False)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-1, N_CHANNELS)
        ax.axhline(doc["asics"][a]["median_mpv_adc"], color="#0b0b0b", lw=0.6, ls=":")
        ax.set_title(f"ASIC {a}", fontsize=9)
        ax.set_xlabel("channel")
    axes[0].set_ylabel("MPV net of pedestal (ADC)")
    fig.suptitle(
        f"{name}: MPV per channel (blue mapped, grey unmapped, x flagged; light dots one per day)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / f"{name}_channelcal_summary.png", dpi=130)
    plt.close(fig)
