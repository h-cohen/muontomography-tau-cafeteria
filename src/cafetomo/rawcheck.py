"""Checks of the raw-event build against the ROOT histograms, and of the sky reference
and channel response across epochs, collected into one results record.

The raw files and the ROOT files are two readings of the same detector. Where they hold
the same clusters, the raw counts must reproduce the ROOT histogram bin for bin: their
chi2 per bin, with the raw counts scaled to the ROOT total, stays below one because the
clusters are shared. Where they hold different days, the shapes of their angular
distributions are compared on 0.05-tan bins against the Poisson scatter of the ratio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from cafetomo.angular import AngularHist, rebin
from cafetomo.config import RunSubset

_SHAPE_REBIN = 10  # 0.05 tan bins
_SHAPE_MIN_COUNTS = 400


def run_halves(subset: RunSubset) -> tuple[RunSubset, RunSubset]:
    """The earlier and later runs of one orientation; an odd run goes to the later half."""
    first, last = subset.runs
    if last <= first:
        raise ValueError(f"run halves need at least two runs, got {subset.runs}")
    mid = first + (last - first + 1) // 2
    return replace(subset, runs=(first, mid - 1)), replace(subset, runs=(mid, last))


def count_match(raw: AngularHist, root: AngularHist) -> tuple[float, float]:
    """(chi2 per occupied bin, raw total / ROOT total) with `root` scaled to `raw`."""
    a = raw.values.astype(np.float64)
    r = root.values.astype(np.float64)
    scale = a.sum() / r.sum()
    var = a + scale**2 * r
    used = var > 0
    chi2 = float(((a - scale * r) ** 2 / np.where(used, var, 1.0))[used].sum() / used.sum())
    return chi2, float(scale)


def epoch_scatter(
    first: Sequence[AngularHist], second: Sequence[AngularHist]
) -> tuple[float, float]:
    """(std, Poisson std) of the shape ratio of two sets of days on 0.05-tan bins."""
    a = sum(rebin(h, _SHAPE_REBIN).values for h in first).astype(np.float64)
    b = sum(rebin(h, _SHAPE_REBIN).values for h in second).astype(np.float64)
    used = (a >= _SHAPE_MIN_COUNTS) & (b >= _SHAPE_MIN_COUNTS)
    ratio = (b[used] / b.sum()) / (a[used] / a.sum())
    noise = np.sqrt(1.0 / a[used] + 1.0 / b[used])
    return float(np.std(ratio / ratio.mean())), float(np.sqrt(np.mean(noise**2)))


def lambda_shift(lam: np.ndarray, reference: np.ndarray) -> tuple[float, float, int]:
    """(mean, std, bins) of lam - reference over bins measured in both."""
    both = np.isfinite(lam) & np.isfinite(reference)
    d = (np.asarray(lam) - np.asarray(reference))[both]
    return float(d.mean()), float(d.std()), int(both.sum())
