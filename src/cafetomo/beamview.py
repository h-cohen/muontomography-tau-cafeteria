"""Two-dimensional angular view of the opacity: the open-sky-divided counts of one
position on the detector's (tan theta_x, tan theta_y) grid.

    lambda = -ln(max(n_pos / (t_pos / t_sky * n_sky), t_floor)),   sigma^2 = 1/n_pos + 1/n_sky

per view bin, the same transmission as `opacity` but kept at its native angular resolution
instead of being scattered onto the sky grid, so a beam reads as a stripe whose position
in tan theta is its lateral offset over its height. Smoothing and background removal are
display aids: they never feed an inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from cafetomo.angular import AngularHist, rebin
from cafetomo.config import ViewSettings


@dataclass(frozen=True)
class OpacityView:
    edges: np.ndarray
    lam: np.ndarray  # NaN where a count floor is not met
    sigma: np.ndarray
    weight: np.ndarray  # 1/sigma^2, zero where not measured
    counts_pos: np.ndarray
    counts_sky: np.ndarray
    live_ratio: float


def opacity_view(
    pos: AngularHist, sky: AngularHist, live_ratio: float, view: ViewSettings, t_floor: float
) -> OpacityView:
    """Opacity per view bin from a position's counts, the sky's and the live-time ratio."""
    if pos.values.shape != sky.values.shape or not (
        np.allclose(pos.xedges, sky.xedges) and np.allclose(pos.yedges, sky.yedges)
    ):
        raise ValueError("position and sky counts must share one binning")
    p = rebin(pos, view.rebin_factor)
    s = rebin(sky, view.rebin_factor)
    n_pos = p.values.astype(np.float64)
    n_sky = s.values.astype(np.float64)
    measured = (n_pos >= view.min_counts) & (n_sky >= view.min_sky_counts)
    safe_pos = np.where(measured, n_pos, 1.0)
    safe_sky = np.where(measured, n_sky, 1.0)
    transmission = safe_pos / (live_ratio * safe_sky)
    lam = np.where(measured, -np.log(np.clip(transmission, t_floor, None)), np.nan)
    sigma = np.where(measured, np.sqrt(1.0 / safe_pos + 1.0 / safe_sky), np.nan)
    weight = np.where(measured, 1.0 / np.where(measured, sigma, 1.0) ** 2, 0.0)
    return OpacityView(p.xedges, lam, sigma, weight, p.values, s.values, float(live_ratio))


def without_floors(view: ViewSettings) -> ViewSettings:
    """The same view with every bin drawn that has a count in both sources: the whole
    square before the count floors."""
    return replace(view, min_counts=1.0, min_sky_counts=1.0)


def smooth_weighted(x: np.ndarray, weight: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian smoothing weighted by `weight` (normalized convolution): unmeasured
    bins (zero weight) contribute nothing and stay NaN."""
    w = np.asarray(weight, dtype=np.float64)
    num = gaussian_filter(np.where(w > 0, x, 0.0) * w, sigma_bins, mode="constant")
    den = gaussian_filter(w, sigma_bins, mode="constant")
    return np.where(w > 0, num / np.where(den > 0, den, 1.0), np.nan)


def local_contrast(
    lam: np.ndarray, weight: np.ndarray, smooth_bins: float, background_bins: float
) -> np.ndarray:
    """Smoothed opacity minus its broad background: what stands out on the scale of a beam."""
    return smooth_weighted(lam, weight, smooth_bins) - smooth_weighted(lam, weight, background_bins)


def _panels(view: OpacityView, settings: ViewSettings) -> list[tuple]:
    smoothed = smooth_weighted(view.lam, view.weight, settings.smooth_bins)
    contrast = local_contrast(view.lam, view.weight, settings.smooth_bins, settings.background_bins)
    lo, hi = settings.lam_range
    c = settings.contrast_range
    return [
        ("opacity", view.lam, "magma_r", lo, hi, r"$\lambda$"),
        (
            f"opacity, smoothed ({settings.smooth_bins:g} bin)",
            smoothed,
            "magma_r",
            lo,
            hi,
            r"$\lambda$",
        ),
        ("local contrast", contrast, "RdBu_r", -c, c, r"$\Delta\lambda$"),
    ]


def render(
    view: OpacityView, unfiltered: OpacityView, settings: ViewSettings, out_dir: Path, name: str
) -> Path:
    """Write <name>_beams2d.{png,npz,json}: the figure, its arrays and a summary.

    The top row is `view` inside |tan| < t_display; the bottom row is `unfiltered`, the
    same counts with no count floor, over the whole histogram square."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_panels(view, settings), _panels(unfiltered, settings)]
    centres = 0.5 * (view.edges[:-1] + view.edges[1:])
    np.savez_compressed(
        out / f"{name}_beams2d.npz",
        edges=view.edges,
        lam=view.lam,
        sigma=view.sigma,
        smoothed=rows[0][1][1],
        contrast=rows[0][2][1],
        lam_unfiltered=unfiltered.lam,
        sigma_unfiltered=unfiltered.sigma,
        smoothed_unfiltered=rows[1][1][1],
        contrast_unfiltered=rows[1][2][1],
        counts_pos=view.counts_pos,
        counts_sky=view.counts_sky,
    )

    t = settings.t_display
    extent = [view.edges[0], view.edges[-1]] * 2
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.8))
    for r, (row, panels) in enumerate(zip(axes, rows, strict=True)):
        for ax, (title, img, cmap, vmin, vmax, label) in zip(row, panels, strict=True):
            cm = plt.get_cmap(cmap).with_extremes(bad="#efeeea")
            h = ax.imshow(img.T, origin="lower", extent=extent, cmap=cm, vmin=vmin, vmax=vmax)
            ax.set_title(title if r == 0 else f"{title}, no count floor", fontsize=10)
            ax.set_xlabel(r"$\tan\theta_x$")
            ax.set_ylabel(r"$\tan\theta_y$")
            lim = t if r == 0 else float(view.edges[-1])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            fig.colorbar(h, ax=ax, shrink=0.8, label=label)
    width = float(view.edges[1] - view.edges[0])
    fig.suptitle(
        f"{name}: {int(view.counts_pos.sum()):,} tracks divided by the open sky, "
        f"{width:g} tan bins; dark = more opaque. Top: count floors "
        f"({settings.min_counts:g} / {settings.min_sky_counts:g}); bottom: the whole square",
        fontsize=11,
    )
    fig.tight_layout()
    png = out / f"{name}_beams2d.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    measured = np.isfinite(view.lam)
    summary = {
        "name": name,
        "bin_width_tan": width,
        "measured_bins": int(measured.sum()),
        "unfiltered_bins": int(np.isfinite(unfiltered.lam).sum()),
        "median_sigma": float(np.nanmedian(view.sigma)),
        "median_sigma_unfiltered": float(np.nanmedian(unfiltered.sigma)),
        "tracks": int(view.counts_pos.sum()),
        "sky_tracks": int(view.counts_sky.sum()),
        "live_time_ratio": view.live_ratio,
        "tan_centre_range": [float(centres[0]), float(centres[-1])],
    }
    (out / f"{name}_beams2d.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return png


@dataclass(frozen=True)
class SkyDifference:
    """One position's opacity against two skies, the first a subset of the second's days."""

    first: OpacityView
    second: OpacityView
    delta: np.ndarray  # first - second; NaN unless both are measured
    noise: np.ndarray  # counting sigma of delta: sqrt(1/n_sky_first - 1/n_sky_second)


def sky_difference(first: OpacityView, second: OpacityView) -> SkyDifference:
    """The position's own counts cancel in the difference, and the shared days of a
    nested sky cancel too, leaving the noise of the days only `second` holds."""
    both = np.isfinite(first.lam) & np.isfinite(second.lam)
    n1 = np.where(both, first.counts_sky, 1).astype(np.float64)
    n2 = np.where(both, second.counts_sky, 1).astype(np.float64)
    noise = np.where(both, np.sqrt(np.clip(1.0 / n1 - 1.0 / n2, 0.0, None)), np.nan)
    return SkyDifference(first, second, np.where(both, first.lam - second.lam, np.nan), noise)


def render_sky_difference(
    diffs: dict[str, SkyDifference], settings: ViewSettings, path: Path, labels=("raw", "ROOT")
) -> Path:
    """One row per position: opacity against each sky (smoothed), their difference
    (smoothed) and the unsmoothed difference in units of its counting noise."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = settings.lam_range
    t = settings.t_display
    fig, axes = plt.subplots(len(diffs), 4, figsize=(21, 5.2 * len(diffs)), squeeze=False)
    for row, (name, d) in zip(axes, diffs.items(), strict=True):
        weight = np.where(np.isfinite(d.delta), d.second.weight, 0.0)
        smooth_delta = smooth_weighted(np.nan_to_num(d.delta), weight, settings.smooth_bins)
        pull = d.delta / d.noise
        rms = float(np.nanstd(d.delta))
        expected = float(np.sqrt(np.nanmean(d.noise**2)))
        panels = [
            (f"{name}: {labels[0]} sky", smooth_weighted(d.first.lam, d.first.weight,
             settings.smooth_bins), "magma_r", lo, hi, r"$\lambda$"),
            (f"{name}: {labels[1]} sky", smooth_weighted(d.second.lam, d.second.weight,
             settings.smooth_bins), "magma_r", lo, hi, r"$\lambda$"),
            (f"{labels[0]} - {labels[1]}, smoothed (mean {np.nanmean(d.delta):+.3f})",
             smooth_delta, "RdBu_r", -0.1, 0.1, r"$\Delta\lambda$"),
            (f"difference / counting noise (rms {rms:.3f}, noise {expected:.3f})",
             pull, "RdBu_r", -4.0, 4.0, r"$\Delta\lambda / \sigma$"),
        ]  # fmt: skip
        extent = [d.first.edges[0], d.first.edges[-1]] * 2
        for ax, (title, img, cmap, vmin, vmax, label) in zip(row, panels, strict=True):
            cm = plt.get_cmap(cmap).with_extremes(bad="#efeeea")
            h = ax.imshow(img.T, origin="lower", extent=extent, cmap=cm, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xlim(-t, t)
            ax.set_ylim(-t, t)
            ax.set_xlabel(r"$\tan\theta_x$")
            ax.set_ylabel(r"$\tan\theta_y$")
            fig.colorbar(h, ax=ax, shrink=0.8, label=label)
    fig.suptitle(
        f"Opacity against the {labels[0]} and the {labels[1]} sky reference "
        "(same position counts; detector frame, 0.025 tan bins)",
        fontsize=11,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
