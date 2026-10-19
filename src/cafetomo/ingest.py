"""Ingest of the DAQ's pre-binned ROOT histograms into counts_<id>.npz files.

Each file holds a `txty` TH2 of counts over (tan θx, tan θy), axis 0 = tx, the
same convention as `np.histogram2d(tan_x, tan_y)`, and a `dT` histogram of the
time between consecutive tracks. The counts are cropped to the configured
binning and written with the live time, which sets the absolute opacity gauge.

The ROOT bins must coincide with the configured binning after cropping: a
mismatch is an error, never a silent resample.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cafetomo.angular import AngularHist, save_counts
from cafetomo.config import Config

if TYPE_CHECKING:
    from cafetomo.rawdata import RawIngest


@dataclass(frozen=True)
class IngestResult:
    source_id: str
    path: Path
    total_in_file: int
    total_kept: int
    live_time_s: float


def _crop_index(file_edges: np.ndarray, target: np.ndarray, axis: str) -> int:
    i0 = int(np.argmin(np.abs(file_edges - target[0])))
    sl = file_edges[i0 : i0 + target.size]
    if sl.size != target.size or not np.allclose(sl, target, atol=1e-9):
        raise ValueError(
            f"ROOT {axis} bins ({file_edges.size - 1} over {file_edges[0]:g}..{file_edges[-1]:g}) "
            f"do not contain the configured binning ({target.size - 1} over "
            f"{target[0]:g}..{target[-1]:g}) on a common edge set"
        )
    return i0


def crop_counts(
    values: np.ndarray, xe: np.ndarray, ye: np.ndarray, name: str, edges: np.ndarray
) -> AngularHist:
    """Crop counts on bin edges (xe, ye) to `edges`, which must be a run of the same edges."""
    i0 = _crop_index(xe, edges, "x")
    j0 = _crop_index(ye, edges, "y")
    n = edges.size - 1
    kept = np.round(values[i0 : i0 + n, j0 : j0 + n]).astype(np.int64)
    return AngularHist(values=kept, xedges=edges.copy(), yedges=edges.copy(), name=name)


def read_root_counts(path: Path, hist: str, edges: np.ndarray) -> tuple[AngularHist, int]:
    """Load a ROOT TH2 and crop it to `edges`. Returns (hist, total counts in file)."""
    import uproot

    with uproot.open(path) as f:
        h = f[hist]
        values = np.asarray(h.values(), dtype=np.float64)
        xe = np.asarray(h.axes[0].edges(), dtype=np.float64)
        ye = np.asarray(h.axes[1].edges(), dtype=np.float64)

    if not np.allclose(values, np.round(values)) or (values < 0).any():
        raise ValueError(f"{Path(path).name}:{hist} is not a histogram of non-negative counts")
    return crop_counts(values, xe, ye, hist, edges), int(round(values.sum()))


def interpolated_live_time(counts: np.ndarray, edges: np.ndarray) -> float:
    """Sum of the intervals in a histogram, placing each at its mean within its bin.

    Bin centres overstate the time of a falling distribution (by ~1% for 0.05 s bins and
    a 0.14 s mean interval). Here the density inside a bin is exponential with the log
    slope of its neighbours, whose mean position sits below the centre."""
    n = np.asarray(counts, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    c = 0.5 * (e[:-1] + e[1:])
    w = np.diff(e)
    lo = np.concatenate([[0], np.arange(n.size - 1)])
    hi = np.concatenate([np.arange(1, n.size), [n.size - 1]])
    both = (n[lo] > 0) & (n[hi] > 0) & (lo != hi)
    slope = np.zeros_like(n)
    slope[both] = np.log(n[lo[both]] / n[hi[both]]) / (c[hi[both]] - c[lo[both]])
    x = np.clip(slope * w, -50.0, 50.0)
    small = np.abs(x) < 1e-6
    safe = np.where(small, 1.0, x)
    mean_offset = w * np.where(small, 0.5 - x / 12.0, 1.0 / safe - 1.0 / np.expm1(safe))
    return float((n * (e[:-1] + mean_offset)).sum())


def read_live_time(path: Path, hist: str = "dT", interpolate: bool = False) -> float:
    """Live time of a run: the sum of its inter-event intervals.

    `dT` is the DAQ's histogram of time between consecutive tracks (one entry
    per track; the files have no overflow). Summing bin-centre x count gives the
    time the detector was taking data, excluding gaps between runs. Cross-checked
    against the exponential slope of `dT` and the `rate` profile: rate ratios
    agree to 0.3%.

    A missing histogram raises KeyError (uproot's KeyInFileError); `ingest`
    converts that into a ValueError naming the file.
    """
    import uproot

    with uproot.open(path) as f:
        h = f[hist]
        v = np.asarray(h.values(), dtype=np.float64)
        e = np.asarray(h.axes[0].edges(), dtype=np.float64)
        overflow = float(h.values(flow=True)[-1])
    if overflow > 0:
        raise ValueError(
            f"{Path(path).name}:{hist} has {overflow:g} overflow intervals; "
            "their durations are unknown, so the live time is not measured"
        )
    if interpolate:
        return interpolated_live_time(v, e)
    return float((0.5 * (e[:-1] + e[1:]) * v).sum())


def ingest(
    cfg: Config, out_dir: Path, raw: dict[str, RawIngest] | None = None
) -> list[IngestResult]:
    """Write counts_<id>.npz for every exposure and the open-sky reference.

    Exposures (or the sky reference) named in `raw` take their counts and live time from
    the event-level files (`rawdata.ingest_raw`) instead of the ROOT histograms. Their live time is
    exact, so the ROOT sources then use the interpolated `dT` sum (the bin-centre sum
    is ~1% high) and every gauge ratio keeps one definition.

    Every source is read and validated before anything is written, so a failure
    never leaves a half-ingested directory.
    """
    out_dir = Path(out_dir)
    raw = raw or {}
    unknown = sorted(set(raw) - {*cfg.position_ids, cfg.sky_reference.id})
    if unknown:
        raise ValueError(f"raw counts for unknown exposure(s) {unknown}")
    edges = cfg.binning.edges()
    sources: list[tuple[str, str, str, dict]] = [
        (e.id, e.root_file, e.root_hist, {"pose": vars(e.pose)}) for e in cfg.exposures
    ]
    s = cfg.sky_reference
    sources.append((s.id, s.root_file, s.root_hist, {"role": "open_sky_reference"}))

    loaded = []
    for sid, fname, hist, meta in sources:
        if sid in raw:
            r = raw[sid]
            folder = Path(cfg.raw.exposures[sid])
            meta = dict(
                meta,
                raw_files=[f.name for f in r.files],
                raw_rows=sum(f.n_rows for f in r.files),
                raw_rows_without_track=sum(f.n_no_cluster for f in r.files),
                raw_clock_resets=sum(f.n_clock_resets for f in r.files),
                raw_clock_divisor=cfg.raw.clock_divisor,
            )
            loaded.append(
                (
                    sid,
                    folder.name,
                    "txty (raw clusters)",
                    meta,
                    r.hist,
                    r.total_in_files,
                    r.live_time_s,
                )
            )
            continue
        path = cfg.data_dir / fname
        h, total = read_root_counts(path, hist, edges)
        try:
            live = read_live_time(path, interpolate=bool(raw))
        except KeyError as exc:
            raise ValueError(
                f"{path.name}: no 'dT' histogram; the absolute opacity gauge "
                "needs a measured live time"
            ) from exc
        loaded.append((sid, fname, hist, meta, h, total, live))

    results = []
    for sid, fname, hist, meta, h, total, live in loaded:
        out = save_counts(
            h,
            out_dir,
            sid,
            dict(
                meta,
                exposure=sid,
                source=fname,
                root_hist=hist,
                total_in_file=total,
                live_time_s=live,
                live_time_estimator=(
                    "exact clock sum"
                    if sid in raw
                    else "interpolated dT sum"
                    if raw
                    else "dT bin-centre sum"
                ),
            ),
        )
        results.append(IngestResult(sid, out, total, h.total, live))
    return results


def live_times(ingest_dir: str | Path, ids: Sequence[str]) -> dict[str, float]:
    """Live time per source id, as recorded by `ingest` in meta.json."""
    doc = json.loads((Path(ingest_dir) / "meta.json").read_text())["exposures"]
    missing = [i for i in ids if i not in doc]
    if missing:
        raise ValueError(f"{ingest_dir}: no ingested source(s) {missing}")
    return {i: float(doc[i]["live_time_s"]) for i in ids}
