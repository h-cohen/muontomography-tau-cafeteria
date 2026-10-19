"""Angular counts and live time of one exposure from the DAQ's event-level dumps.

A `.data` file comes in one of two exports. The table is semicolon-separated with CRLF
line ends and one row per four-fold cluster: 4 cluster columns, then for each of 4
readout slots 9 bookkeeping columns followed by 32 hit flags and 32 low- and 32
high-gain charges. The record export is comma-separated over 14 lines per cluster: a
`1234BBBB1234` marker, the 4 cluster fields, then per slot its 9 bookkeeping fields, 32
hit flags and 32 high-gain charges (no low gain); large timecodes may be written in E
notation. In both, the slot does not identify the ASIC: each slot's `ASIC_k` column
does, and the order rotates between rows, so every per-channel column is re-indexed by
the ASIC it reports.

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
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from cafetomo.angular import AngularHist
from cafetomo.config import Config, RawSettings
from cafetomo.ingest import crop_counts

N_COLUMNS = 424  # numbers per cluster in the table export
N_ASICS = 4
N_CHANNELS = 32
_HEAD = 4
_FLAGS = 9  # bookkeeping columns before a slot's hit flags
_MARKER = b"1234BBBB1234,"


@dataclass(frozen=True)
class _Layout:
    numbers: int  # per cluster
    slot: int  # numbers per readout slot
    hg: int  # offset of the high-gain charges within a slot


_TABLE = _Layout(N_COLUMNS, 105, _FLAGS + 2 * N_CHANNELS)
_RECORDS = _Layout(_HEAD + 4 * 73, 73, _FLAGS + N_CHANNELS)
_HEADER = b"ID_CLUSTER"
TAN_EDGES = np.linspace(-2.0, 2.0, 801)  # the ROOT txty binning
CHARGE_EDGES = np.arange(0.0, 16001.0, 50.0)  # high-gain ADC of a channel that fired alone
PEDESTAL_EDGES = np.arange(0.0, 4001.0, 10.0)  # high-gain ADC of a channel with no hit flag
_SCHEMA = 2  # bumped whenever the cached per-file counts gain or change a field


@dataclass(frozen=True)
class RawBlock:
    timecode_ns: np.ndarray  # (n,) cluster run clock
    hit: np.ndarray  # (n, ASIC, channel) hit flag
    charge: np.ndarray  # (n, ASIC, channel) high-gain charge
    asics_ok: np.ndarray  # (n,) the row's four slots named the four ASICs once each


def _parse(text: bytes, n: int, layout: _Layout, name: str) -> np.ndarray:
    """`n` clusters of numbers separated by any mix of `;`, `,` and whitespace."""
    values = np.fromstring(text.replace(b";", b" ").replace(b",", b" "), sep=" ")
    if values.size != n * layout.numbers:
        raise ValueError(
            f"{name}: {n} clusters hold {values.size} numbers, not {layout.numbers} each"
        )
    return values.reshape(n, layout.numbers)


def _block(rows: np.ndarray, layout: _Layout) -> RawBlock:
    base = _HEAD + layout.slot * np.arange(N_ASICS)
    slot_asic = rows[:, base].astype(np.int64)
    slot_hit = np.stack([rows[:, b + _FLAGS : b + _FLAGS + N_CHANNELS] > 0 for b in base], axis=1)
    slot_hg = np.stack(
        [rows[:, b + layout.hg : b + layout.hg + N_CHANNELS] for b in base], axis=1
    ).astype(np.float32)
    ok = (np.sort(slot_asic, axis=1) == np.arange(N_ASICS)).all(axis=1)
    order = np.argsort(np.where(ok[:, None], slot_asic, np.arange(N_ASICS)), axis=1)
    pick = order[:, :, None]
    hit = np.take_along_axis(slot_hit, pick, axis=1) & ok[:, None, None]
    charge = np.take_along_axis(slot_hg, pick, axis=1)
    return RawBlock(rows[:, 1].copy(), hit, charge, ok)


def _table_blocks(path: Path, rows: int) -> Iterator[RawBlock]:
    pending: list[bytes] = []
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(_HEADER) or not line.strip():
                continue
            pending.append(line)
            if len(pending) == rows:
                yield _block(_parse(b" ".join(pending), rows, _TABLE, path.name), _TABLE)
                pending = []
    if pending:
        yield _block(_parse(b" ".join(pending), len(pending), _TABLE, path.name), _TABLE)


def _record_blocks(path: Path, rows: int, chunk_bytes: int) -> Iterator[RawBlock]:
    """Split the stream on the record marker; the text after the last marker of a chunk
    is held back until the next chunk (or the end of the file) completes it."""
    done: list[bytes] = []
    tail = b""
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            parts = (tail + chunk).split(_MARKER)
            tail = parts.pop()
            done += [p for p in parts if p.strip()]
            while len(done) >= rows:
                head, done = done[:rows], done[rows:]
                yield _block(_parse(b" ".join(head), rows, _RECORDS, path.name), _RECORDS)
    if tail.strip():
        done.append(tail)
    if done:
        yield _block(_parse(b" ".join(done), len(done), _RECORDS, path.name), _RECORDS)


def read_blocks(path: Path, rows: int = 50_000, chunk_bytes: int = 1 << 25) -> Iterator[RawBlock]:
    """Stream a `.data` file of either export in blocks of at most `rows` clusters."""
    path = Path(path)
    with open(path, "rb") as f:
        start = f.read(len(_MARKER))
    if start == _MARKER:
        yield from _record_blocks(path, rows, chunk_bytes)
    else:
        yield from _table_blocks(path, rows)


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
    n_valid_rows: int  # rows whose slots named the four ASICs once each
    hit_hist: np.ndarray  # (ASIC, channel, charge bin): high-gain charge when it fired alone
    ped_hist: np.ndarray  # (ASIC, channel, pedestal bin): charge with no hit flag
    n_flagged: np.ndarray  # (ASIC, channel): rows with the hit flag, alone or not


def _charge_counts(mask: np.ndarray, charge: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per (ASIC, channel) histogram of `charge` where `mask`; overflow joins the last bin."""
    nbins = edges.size - 1
    bin_ = np.clip(((charge - edges[0]) // (edges[1] - edges[0])).astype(np.int64), 0, nbins - 1)
    _, asic, chan = np.nonzero(mask)
    flat = (asic * N_CHANNELS + chan) * nbins + bin_[mask]
    return np.bincount(flat, minlength=N_ASICS * N_CHANNELS * nbins).reshape(
        N_ASICS, N_CHANNELS, nbins
    )


def count_file(
    path: Path, settings: RawSettings, edges: np.ndarray, block_rows: int = 50_000
) -> FileCounts:
    values = np.zeros((edges.size - 1,) * 2, dtype=np.int64)
    n_rows = n_tracks = n_no_cluster = n_bad = n_resets = n_valid = 0
    hit_hist = np.zeros((N_ASICS, N_CHANNELS, CHARGE_EDGES.size - 1), dtype=np.int64)
    ped_hist = np.zeros((N_ASICS, N_CHANNELS, PEDESTAL_EDGES.size - 1), dtype=np.int64)
    n_flagged = np.zeros((N_ASICS, N_CHANNELS), dtype=np.int64)
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
        n_bad += int((~blk.asics_ok).sum())
        n_valid += int(blk.asics_ok.sum())
        alone = blk.hit & (blk.hit.sum(axis=2, keepdims=True) == 1)
        hit_hist += _charge_counts(alone, blk.charge, CHARGE_EDGES)
        ped_hist += _charge_counts(
            ~blk.hit & blk.asics_ok[:, None, None], blk.charge, PEDESTAL_EDGES
        )
        n_flagged += blk.hit.sum(axis=0)
    return FileCounts(
        Path(path).name,
        values,
        n_rows,
        n_tracks,
        n_no_cluster,
        n_bad,
        n_resets,
        clock_ns / 1e9 / settings.clock_divisor,
        n_valid,
        hit_hist,
        ped_hist,
        n_flagged,
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
    spec = asdict(settings) | {"exposures": None, "bytes": path.stat().st_size, "schema": _SCHEMA}
    return hashlib.sha256(json.dumps([path.name, spec], sort_keys=True).encode()).hexdigest()


_ARRAYS = ("values", "hit_hist", "ped_hist", "n_flagged")


def _load(cache: Path, digest: str) -> FileCounts | None:
    if not cache.is_file():
        return None
    with np.load(cache) as d:
        if str(d["digest"]) != digest:
            return None
        return FileCounts(**json.loads(str(d["meta"])), **{k: d[k] for k in _ARRAYS})


def _store(cache: Path, digest: str, c: FileCounts) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    meta = {f.name: getattr(c, f.name) for f in fields(c) if f.name not in _ARRAYS}
    np.savez_compressed(
        cache,
        digest=np.array(digest),
        meta=np.array(json.dumps(meta)),
        **{k: getattr(c, k) for k in _ARRAYS},
    )


def _count_job(args) -> FileCounts:
    path, settings = args
    return count_file(path, settings, TAN_EDGES)


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
        hist=crop_counts(full, TAN_EDGES, TAN_EDGES, "txty", cfg.binning.edges()),
        total_in_files=int(full.sum()),
        live_time_s=float(sum(c.live_time_s for c in files)),
        files=files,
        cached=len(files) - len(todo),
    )
