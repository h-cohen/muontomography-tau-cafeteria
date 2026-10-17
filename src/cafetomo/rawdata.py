"""Angular counts and live time of one exposure from the DAQ's event-level dumps.

A `.data` file is a semicolon-separated table with CRLF line ends and one row per
four-fold cluster: 4 cluster columns, then for each of 4 readout slots 9 bookkeeping
columns followed by 32 hit flags and 32 low- and 32 high-gain charges. The slot does
not identify the ASIC: each slot's `ASIC_k` column does, and the order rotates between
rows, so every per-channel column is re-indexed by the ASIC it reports.

Reconstruction follows the DAQ that produced the pre-binned ROOT histograms, so the
open-sky division stays like for like. The bars read out by one ASIC are triangular and
share light between neighbours. A layer's position is the charge centroid of its
strongest bar and the bars beside it that fired, with the pedestal removed; one bar
alone gives its centre. This reproduces the per-layer hit-position histograms of the
ROOT files. The DAQ's tan(theta_x) is the ASIC 1 - ASIC 0 pair and tan(theta_y) the
ASIC 2 - ASIC 3 pair, each as (upper - lower) / layer gap; the gaps (37.00 and 36.80 cm)
and the pairing are those whose tangent combs coincide bin for bin with the ROOT
marginals. Tangents are histogrammed on the ROOT binning.

The run clock counts `clock_divisor` run-seconds per second of data: a file of one
day's data spans 50 days of clock. Live time is the summed clock gaps between
consecutive clusters, as for the ROOT `dT` histogram; a backwards step is a new run and
contributes no gap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cafetomo.angular import AngularHist
from cafetomo.config import Config, RawSettings
from cafetomo.ingest import crop_counts

N_COLUMNS = 424
N_ASICS = 4
N_CHANNELS = 32
_HEAD = 4
_SLOT = 105
_FLAGS = 9  # bookkeeping columns before a slot's hit flags
_HEADER = b"ID_CLUSTER"
_FULL_EDGES = np.linspace(-2.0, 2.0, 801)  # the ROOT txty binning


@dataclass(frozen=True)
class RawBlock:
    timecode_ns: np.ndarray  # (n,) cluster run clock
    hit: np.ndarray  # (n, ASIC, channel) hit flag
    charge: np.ndarray  # (n, ASIC, channel) high-gain charge
    n_bad_asics: int  # rows whose four slots did not name the four ASICs once each


def _parse(lines: list[bytes], name: str) -> np.ndarray:
    text = b"".join(lines).replace(b"\r\n", b";").replace(b"\n", b";").replace(b";", b" ")
    values = np.fromstring(text, sep=" ")
    if values.size != len(lines) * N_COLUMNS:
        raise ValueError(
            f"{name}: {len(lines)} rows hold {values.size} numbers, not {N_COLUMNS} per row"
        )
    return values.reshape(-1, N_COLUMNS)


def _block(rows: np.ndarray) -> RawBlock:
    base = _HEAD + _SLOT * np.arange(N_ASICS)
    slot_asic = rows[:, base].astype(np.int64)
    slot_hit = np.stack([rows[:, b + _FLAGS : b + _FLAGS + N_CHANNELS] > 0 for b in base], axis=1)
    slot_hg = np.stack(
        [rows[:, b + _FLAGS + 2 * N_CHANNELS : b + _SLOT] for b in base], axis=1
    ).astype(np.float32)
    ok = (np.sort(slot_asic, axis=1) == np.arange(N_ASICS)).all(axis=1)
    order = np.argsort(np.where(ok[:, None], slot_asic, np.arange(N_ASICS)), axis=1)
    pick = order[:, :, None]
    hit = np.take_along_axis(slot_hit, pick, axis=1) & ok[:, None, None]
    charge = np.take_along_axis(slot_hg, pick, axis=1)
    return RawBlock(rows[:, 1].copy(), hit, charge, int((~ok).sum()))


def read_blocks(path: Path, rows: int = 50_000) -> Iterator[RawBlock]:
    """Stream a `.data` file in blocks of at most `rows` clusters, skipping header lines."""
    path = Path(path)
    pending: list[bytes] = []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(_HEADER) or not line.strip():
                continue
            pending.append(line)
            if len(pending) == rows:
                yield _block(_parse(pending, path.name))
                pending = []
    if pending:
        yield _block(_parse(pending, path.name))


def cluster_positions(
    hit: np.ndarray, charge: np.ndarray, bars: tuple, pedestal: float, pitch_cm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Position (cm) of one layer's cluster per row, and whether the row has one.

    `hit` and `charge` are (n, channel); `bars[b]` is the channel of bar b. Channels
    that no bar reads are ignored."""
    chan = np.asarray(bars)
    weight = np.where(hit[:, chan], np.clip(charge[:, chan] - pedestal, 0.0, None), 0.0)
    peak = weight.argmax(axis=1)
    bar = np.arange(chan.size)
    near = np.abs(bar[None, :] - peak[:, None]) <= 1
    w = np.where(near, weight, 0.0)
    total = w.sum(axis=1)
    ok = total > 0
    centre = pitch_cm * (bar + 1.0)
    return (w * centre).sum(axis=1) / np.where(ok, total, 1.0), ok


def tracks_from_positions(pos: np.ndarray, settings: RawSettings) -> tuple[np.ndarray, np.ndarray]:
    """Tangents from per-ASIC layer positions (n, ASIC)."""
    (xl, xu), (yl, yu) = settings.tx_asics, settings.ty_asics
    gx, gy = settings.gap_cm
    return (pos[:, xu] - pos[:, xl]) / gx, (pos[:, yu] - pos[:, yl]) / gy


@dataclass(frozen=True)
class FileCounts:
    name: str
    values: np.ndarray  # int64 counts on the ROOT binning, tan +-2
    n_rows: int
    n_tracks: int
    n_no_cluster: int  # a layer had no bar with charge above pedestal
    n_bad_asics: int
    n_clock_resets: int
    live_time_s: float


def count_file(
    path: Path, settings: RawSettings, edges: np.ndarray, block_rows: int = 50_000
) -> FileCounts:
    values = np.zeros((edges.size - 1,) * 2, dtype=np.int64)
    n_rows = n_tracks = n_no_cluster = n_bad = n_resets = 0
    clock_ns = 0.0
    last = None
    for blk in read_blocks(path, block_rows):
        n = blk.timecode_ns.size
        pos = np.empty((n, N_ASICS))
        ok = np.ones(n, bool)
        for a in range(N_ASICS):
            pos[:, a], good = cluster_positions(
                blk.hit[:, a],
                blk.charge[:, a],
                settings.channel_bars[a],
                settings.pedestal_hg[a],
                settings.bar_pitch_cm,
            )
            ok &= good
        tx, ty = tracks_from_positions(pos[ok], settings)
        values += np.histogram2d(tx, ty, [edges, edges])[0].astype(np.int64)

        t = blk.timecode_ns if last is None else np.concatenate([[last], blk.timecode_ns])
        gap = np.diff(t)
        n_resets += int((gap < 0).sum())
        clock_ns += float(gap[gap >= 0].sum())
        last = float(blk.timecode_ns[-1])
        n_rows += n
        n_tracks += int(ok.sum())
        n_no_cluster += int((~ok).sum())
        n_bad += blk.n_bad_asics
    return FileCounts(
        Path(path).name,
        values,
        n_rows,
        n_tracks,
        n_no_cluster,
        n_bad,
        n_resets,
        clock_ns / 1e9 / settings.clock_divisor,
    )


@dataclass(frozen=True)
class RawIngest:
    hist: AngularHist  # cropped to the configured binning
    total_in_files: int  # tracks on the full tan +-2 range
    live_time_s: float
    files: tuple[FileCounts, ...]
    cached: int  # files read back from the cache instead of recounted


def _digest(path: Path, settings: RawSettings) -> str:
    """Identity of one file's counts: its size and name plus every setting that shapes them."""
    spec = asdict(settings) | {"exposures": None, "bytes": path.stat().st_size}
    return hashlib.sha256(json.dumps([path.name, spec], sort_keys=True).encode()).hexdigest()


def _load(cache: Path, digest: str) -> FileCounts | None:
    if not cache.is_file():
        return None
    with np.load(cache) as d:
        if str(d["digest"]) != digest:
            return None
        meta = json.loads(str(d["meta"]))
        return FileCounts(values=d["values"], **meta)


def _store(cache: Path, digest: str, c: FileCounts) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in asdict(c).items() if k != "values"}
    np.savez_compressed(
        cache, digest=np.array(digest), meta=np.array(json.dumps(meta)), values=c.values
    )


def _count_job(args) -> FileCounts:
    path, settings = args
    return count_file(path, settings, _FULL_EDGES)


def ingest_raw(cfg: Config, exposure: str, cache_dir: Path, jobs: int = 1) -> RawIngest:
    """Angular counts and live time of `exposure` from every `.data` file of its directory.

    Files are counted independently (in `jobs` processes) and each result is cached under
    `cache_dir`, keyed by the file and the settings, so an interrupted or repeated run
    only recounts what changed."""
    raw = cfg.raw
    if raw is None or exposure not in raw.exposures:
        raise ValueError(f"config has no raw data for exposure {exposure!r}")
    folder = Path(raw.exposures[exposure])
    paths = sorted(folder.glob("*.data"))
    if not paths:
        raise ValueError(f"{folder}: no .data files")
    cache = Path(cache_dir)
    digests = [_digest(p, raw) for p in paths]
    keys = [cache / f"{exposure}_{d[:16]}.npz" for d in digests]
    found = [_load(k, d) for k, d in zip(keys, digests, strict=True)]
    todo = [i for i, c in enumerate(found) if c is None]
    work = [(paths[i], raw) for i in todo]
    if jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(jobs) as pool:
            done = list(pool.map(_count_job, work))
    else:
        done = [_count_job(w) for w in work]
    for i, c in zip(todo, done, strict=True):
        _store(keys[i], digests[i], c)
        found[i] = c
    files = tuple(found)
    full = sum(c.values for c in files)
    return RawIngest(
        hist=crop_counts(full, _FULL_EDGES, _FULL_EDGES, "txty", cfg.binning.edges()),
        total_in_files=int(full.sum()),
        live_time_s=float(sum(c.live_time_s for c in files)),
        files=files,
        cached=len(files) - len(todo),
    )
