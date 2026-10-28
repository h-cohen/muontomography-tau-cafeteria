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
ROOT files. The DAQ's hit flag is no charge threshold, so a neighbour can carry a real
deposit without its flag: with `neighbour_sigmas` > 0, a layer with a single flagged bar
also takes a neighbour whose charge exceeds that channel's pedestal by as many pedestal
widths. The pedestals are Gaussian fits to each channel's unflagged charge, from a first
pass with the flag alone, and every charge is then net of its own channel's pedestal.

The DAQ's tan(theta_x) is the ASIC 1 - ASIC 0 pair and tan(theta_y) the
ASIC 2 - ASIC 3 pair, each as (upper - lower) / layer gap; the gaps (37.00 and 36.80 cm)
and the pairing are those whose tangent combs coincide bin for bin with the ROOT
marginals. Tangents are histogrammed on the ROOT binning.

The run clock counts `clock_divisor` run-seconds per second of data: a file of one
day's data spans 50 days of clock. Live time is the summed clock gaps between
consecutive clusters, as for the ROOT `dT` histogram; a backwards step is a new run and
contributes no gap. Runs held back to back in one file can thus be counted on their own.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from cafetomo.angular import AngularHist
from cafetomo.config import Config, RawSettings, RunSubset
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
_SCHEMA = 3  # bumped whenever the cached per-file counts gain or change a field


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


def _select(blk: RawBlock, keep: np.ndarray) -> RawBlock:
    return RawBlock(blk.timecode_ns[keep], blk.hit[keep], blk.charge[keep], blk.asics_ok[keep])


def fired_bars(
    hit: np.ndarray, charge: np.ndarray, bars: tuple, threshold: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bars of one layer taken into its cluster (n, bar), the rows with a single flagged
    bar, and the rows among those that took a neighbour above its `threshold` (per
    channel); without `threshold` only flagged bars are taken."""
    chan = np.asarray(bars)
    fired = hit[:, chan]
    lone = fired.sum(axis=1) == 1
    if threshold is None:
        return fired, lone, np.zeros_like(lone)
    bar = np.arange(chan.size)
    beside = np.abs(bar[None, :] - fired.argmax(axis=1)[:, None]) == 1
    extra = lone[:, None] & beside & (charge[:, chan] > np.asarray(threshold)[chan])
    return fired | extra, lone, extra.any(axis=1)


def cluster_positions(
    hit: np.ndarray,
    charge: np.ndarray,
    bars: tuple,
    pedestal: float | np.ndarray,
    pitch_cm: float,
    threshold: np.ndarray | None = None,
    gain: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Position (cm) of one layer's cluster per row, and whether the row has one.

    `hit` and `charge` are (n, channel); `bars[b]` is the channel of bar b. Channels
    that no bar reads are ignored. `pedestal` and `gain` (dividing the net charge) are
    one value or one per channel; `threshold` (per channel) admits neighbours of a lone
    flagged bar (`fired_bars`)."""
    chan = np.asarray(bars)
    ped, g = (np.asarray(v, dtype=float) for v in (pedestal, gain))
    ped = ped[chan] if ped.ndim else ped
    g = g[chan] if g.ndim else g
    fired, *_ = fired_bars(hit, charge, bars, threshold)
    weight = np.where(fired, np.clip(charge[:, chan] - ped, 0.0, None) / g, 0.0)
    peak = weight.argmax(axis=1)
    bar = np.arange(chan.size)
    near = np.abs(bar[None, :] - peak[:, None]) <= 1
    w = np.where(near, weight, 0.0)
    total = w.sum(axis=1)
    ok = total > 0
    centre = pitch_cm * (bar + 1.0)
    return (w * centre).sum(axis=1) / np.where(ok, total, 1.0), ok


def fit_pedestals(ped_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and width (ADC) of a Gaussian fitted to the core of each channel's unflagged
    charge (`ped_hist`, (ASIC, channel, bin)); NaN for a channel without entries.

    Deposits without a flag form a tail above the pedestal, so the fit takes the bins from
    two widths below the peak to one above it, the width first estimated from the half
    maximum on the low side; the fit is a parabola in log counts weighted by the counts.
    A peak narrower than three bins takes its bin centre and the bin's rms width."""
    centres = 0.5 * (PEDESTAL_EDGES[:-1] + PEDESTAL_EDGES[1:])
    step = PEDESTAL_EDGES[1] - PEDESTAL_EDGES[0]
    mean = np.full(ped_hist.shape[:2], np.nan)
    width = np.full(ped_hist.shape[:2], np.nan)
    for a, c in np.ndindex(*ped_hist.shape[:2]):
        h = ped_hist[a, c].astype(float)
        if h.sum() == 0:
            continue
        k = int(h.argmax())
        low = np.nonzero(h[:k] < 0.5 * h[k])[0]
        half = (k - low[-1] if low.size else 1) * step
        s0 = max(half / np.sqrt(2.0 * np.log(2.0)), step)
        win = (centres >= centres[k] - 2.0 * s0) & (centres <= centres[k] + s0) & (h > 0)
        if win.sum() < 3:
            mean[a, c], width[a, c] = centres[k], step / np.sqrt(12.0)
            continue
        x = centres[win] - centres[k]
        c2, c1, _ = np.polyfit(x, np.log(h[win]), 2, w=np.sqrt(h[win]))
        if c2 >= 0:
            raise ValueError(f"ASIC {a} channel {c}: pedestal has no Gaussian core")
        width[a, c] = np.sqrt(-0.5 / c2)
        mean[a, c] = centres[k] + c1 * width[a, c] ** 2
    return mean, width


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
    n_segments: int  # runs in the file, kept or not
    # layers with a single flagged bar, and those of them that took a neighbour
    # (counted with `neighbour_sigmas` > 0; zero otherwise)
    n_lone_layers: int = 0
    n_recovered_layers: int = 0


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
    path: Path,
    settings: RawSettings,
    edges: np.ndarray,
    block_rows: int = 50_000,
    segments: tuple[int, int] | None = None,
    pedestals: tuple[np.ndarray, np.ndarray] | None = None,
    gains: np.ndarray | None = None,
) -> FileCounts:
    """Counts of the clusters in runs `segments` = [first, stop) of the file, all by default;
    run k follows the k-th reset of the run clock. With `settings.neighbour_sigmas` > 0,
    `pedestals` (mean, width per (ASIC, channel), `fit_pedestals`) set the neighbour
    thresholds and the charge baselines. `gains` (per (ASIC, channel),
    `channelcal.relative_gains`) divide the net charges that weight the cluster centroid."""
    select = settings.neighbour_sigmas > 0
    if select and pedestals is None:
        raise ValueError("neighbour selection needs the channel pedestals")
    if select:
        for a, bars in enumerate(settings.channel_bars):
            bad = [b for b, c in enumerate(bars) if not np.isfinite(pedestals[0][a][c])]
            if bad:
                raise ValueError(f"ASIC {a} bar {bad[0]}: no pedestal measured")
    n_lone = n_recovered = 0
    values = np.zeros((edges.size - 1,) * 2, dtype=np.int64)
    n_rows = n_tracks = n_no_cluster = n_bad = n_resets = n_valid = 0
    hit_hist = np.zeros((N_ASICS, N_CHANNELS, CHARGE_EDGES.size - 1), dtype=np.int64)
    ped_hist = np.zeros((N_ASICS, N_CHANNELS, PEDESTAL_EDGES.size - 1), dtype=np.int64)
    n_flagged = np.zeros((N_ASICS, N_CHANNELS), dtype=np.int64)
    clock_ns = 0.0
    last = None
    run = 0
    for whole in read_blocks(path, block_rows):
        t = whole.timecode_ns if last is None else np.concatenate([[last], whole.timecode_ns])
        gap = np.diff(t)
        # run of every cluster in t; a gap belongs to the run of the cluster closing it
        runs = run + np.r_[0, np.cumsum(gap < 0)]
        run = int(runs[-1])
        last = float(t[-1])
        n_resets += int((gap < 0).sum())
        lo, hi = (0, np.inf) if segments is None else segments
        kept = (runs >= lo) & (runs < hi)
        clock_ns += float(gap[(gap >= 0) & kept[1:]].sum())
        keep = kept[t.size - whole.timecode_ns.size :]
        if not keep.any():
            continue
        blk = _select(whole, keep)
        n = blk.timecode_ns.size
        pos = np.empty((n, N_ASICS))
        ok = np.ones(n, bool)
        for a in range(N_ASICS):
            bars = settings.channel_bars[a]
            if select:
                mean, width = pedestals[0][a], pedestals[1][a]
                threshold = mean + settings.neighbour_sigmas * width
                _, lone, took = fired_bars(blk.hit[:, a], blk.charge[:, a], bars, threshold)
                n_lone += int(lone.sum())
                n_recovered += int(took.sum())
            else:
                mean, threshold = settings.pedestal_hg[a], None
            pos[:, a], good = cluster_positions(
                blk.hit[:, a],
                blk.charge[:, a],
                bars,
                mean,
                settings.bar_pitch_cm,
                threshold,
                1.0 if gains is None else gains[a],
            )
            ok &= good
        tx, ty = tracks_from_positions(pos[ok], settings)
        values += np.histogram2d(tx, ty, [edges, edges])[0].astype(np.int64)

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
        run + 1,
        n_lone,
        n_recovered,
    )


@dataclass(frozen=True)
class RawIngest:
    hist: AngularHist  # cropped to the configured binning
    total_in_files: int  # tracks on the full tan +-2 range
    live_time_s: float
    files: tuple[FileCounts, ...]
    cached: int  # files read back from the cache instead of recounted
    pedestals: tuple[np.ndarray, np.ndarray] | None = None  # (mean, width) with neighbours


def _digest(
    path: Path,
    settings: RawSettings,
    segments: tuple[int, int] | None,
    pedestals: tuple[np.ndarray, np.ndarray] | None = None,
    gains: np.ndarray | None = None,
) -> str:
    """Identity of one file's counts: its size and name, the runs kept and every setting
    that shapes them. The flag-only selection keeps the identity it had before neighbour
    selection existed; neighbour selection adds its pedestals, gain correction its gains."""
    spec = asdict(settings) | {
        "exposures": None,
        "first_run": None,
        "restarts": None,
        "subsets": None,
        "segments": segments,
        "bytes": path.stat().st_size,
        "schema": _SCHEMA,
    }
    del spec["gain_correction"]  # the gains themselves enter below
    if settings.neighbour_sigmas > 0:
        spec["pedestals"] = hashlib.sha256(np.asarray(pedestals).tobytes()).hexdigest()
    else:
        del spec["neighbour_sigmas"]
    if gains is not None:
        spec["gains"] = hashlib.sha256(np.asarray(gains, dtype=float).tobytes()).hexdigest()
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
    path, settings, segments, pedestals, gains = args
    return count_file(
        path, settings, TAN_EDGES, segments=segments, pedestals=pedestals, gains=gains
    )


def _run_segments(raw: RawSettings, subset: RunSubset) -> tuple[int, int]:
    """[first, stop) clock segments of a subset's runs; a run that restarted its clock
    (`raw.restarts`) spans one more segment per restart."""
    base = raw.first_run[subset.exposure]
    extra = np.asarray(raw.restarts.get(subset.exposure, ()), dtype=np.int64)
    first, last = subset.runs
    return (first - base + int((extra < first).sum()), last - base + 1 + int((extra <= last).sum()))


def ingest_raw(
    cfg: Config, exposure: str, cache_dir: Path, jobs: int = 1, gains: np.ndarray | None = None
) -> RawIngest:
    """Angular counts and live time of `exposure` from every `.data` file of its directory,
    or of a run subset (`raw.subsets`) from the single file of its exposure, with the
    cluster charges optionally divided by channel `gains` (`count_file`); with
    `raw.gain_correction` and no `gains`, by those of the exposure's own calibration.

    Files are counted independently (in `jobs` processes) and each result is cached under
    `cache_dir`, keyed by the file and the settings, so an interrupted or repeated run
    only recounts what changed."""
    raw = cfg.raw
    if raw is None or exposure not in {*raw.exposures, *raw.subsets}:
        raise ValueError(f"config has no raw data for exposure {exposure!r}")
    subset = raw.subsets.get(exposure)
    folder = Path(raw.exposures[subset.exposure if subset else exposure])
    paths = sorted(folder.glob("*.data"))
    if not paths:
        raise ValueError(f"{folder}: no .data files")
    segments = None
    if subset:
        if len(paths) != 1:
            raise ValueError(f"{folder}: a run subset needs one .data file, found {len(paths)}")
        segments = _run_segments(raw, subset)
    cache = Path(cache_dir)
    pedestals = None
    if raw.gain_correction and gains is None:
        from cafetomo.channelcal import exposure_gains

        plain = replace(cfg, raw=replace(raw, gain_correction=False))
        files = ingest_raw(plain, exposure, cache_dir, jobs).files
        gains = exposure_gains(files, raw.channel_bars, cfg.calibration)
    if raw.neighbour_sigmas > 0:
        flag = replace(cfg, raw=replace(raw, neighbour_sigmas=0.0, gain_correction=False))
        pedestals = fit_pedestals(
            sum(f.ped_hist for f in ingest_raw(flag, exposure, cache, jobs).files)
        )
    digests = [_digest(p, raw, segments, pedestals, gains) for p in paths]
    keys = [cache / f"{exposure}_{d[:16]}.npz" for d in digests]
    found = [_load(k, d) for k, d in zip(keys, digests, strict=True)]
    todo = [i for i, c in enumerate(found) if c is None]
    work = [(paths[i], raw, segments, pedestals, gains) for i in todo]
    if jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(jobs) as pool:
            done = list(pool.map(_count_job, work))
    else:
        done = [_count_job(w) for w in work]
    for i, c in zip(todo, done, strict=True):
        _store(keys[i], digests[i], c)
        found[i] = c
    files = tuple(found)
    if subset and files[0].n_segments < segments[1]:
        first = raw.first_run[subset.exposure]
        held = files[0].n_segments - len(raw.restarts.get(subset.exposure, ()))
        raise ValueError(
            f"{paths[0].name}: holds runs {first}-{first + held - 1}, "
            f"subset {exposure!r} asks for {subset.runs[0]}-{subset.runs[1]}"
        )
    full = sum(c.values for c in files)
    return RawIngest(
        hist=crop_counts(full, TAN_EDGES, TAN_EDGES, "txty", cfg.binning.edges()),
        total_in_files=int(full.sum()),
        live_time_s=float(sum(c.live_time_s for c in files)),
        files=files,
        cached=len(files) - len(todo),
        pedestals=pedestals,
    )
