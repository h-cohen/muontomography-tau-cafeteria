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
from dataclasses import dataclass
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


def render(view: OpacityView, settings: ViewSettings, out_dir: Path, name: str) -> Path:
    """Write <name>_beams2d.{png,npz,json}: the figure, its arrays and a summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    smoothed = smooth_weighted(view.lam, view.weight, settings.smooth_bins)
    contrast = local_contrast(view.lam, view.weight, settings.smooth_bins, settings.background_bins)
    centres = 0.5 * (view.edges[:-1] + view.edges[1:])
    np.savez_compressed(
        out / f"{name}_beams2d.npz",
        edges=view.edges,
        lam=view.lam,
        sigma=view.sigma,
        smoothed=smoothed,
        contrast=contrast,
        counts_pos=view.counts_pos,
        counts_sky=view.counts_sky,
    )

    lo, hi = settings.lam_range
    c = settings.contrast_range
    t = settings.t_display
    extent = [view.edges[0], view.edges[-1]] * 2
    panels = [
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
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True, sharey=True)
    for ax, (title, img, cmap, vmin, vmax, label) in zip(axes, panels, strict=True):
        cm = plt.get_cmap(cmap).with_extremes(bad="#efeeea")
        h = ax.imshow(img.T, origin="lower", extent=extent, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\tan\theta_x$")
        fig.colorbar(h, ax=ax, shrink=0.8, label=label)
    axes[0].set_ylabel(r"$\tan\theta_y$")
    axes[0].set_xlim(-t, t)
    axes[0].set_ylim(-t, t)
    width = float(view.edges[1] - view.edges[0])
    fig.suptitle(
        f"{name}: {int(view.counts_pos.sum()):,} tracks divided by the open sky, "
        f"{width:g} tan bins; dark = more opaque",
        fontsize=11,
    )
    png = out / f"{name}_beams2d.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    measured = np.isfinite(view.lam)
    summary = {
        "name": name,
        "bin_width_tan": width,
        "measured_bins": int(measured.sum()),
        "median_sigma": float(np.nanmedian(view.sigma)),
        "tracks": int(view.counts_pos.sum()),
        "sky_tracks": int(view.counts_sky.sum()),
        "live_time_ratio": view.live_ratio,
        "tan_centre_range": [float(centres[0]), float(centres[-1])],
    }
    (out / f"{name}_beams2d.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return png
