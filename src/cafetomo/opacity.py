"""Opacity per sky direction, by division by the open-sky run.

The sky run of the same detector carries both the detector response and the
cosmic flux shape, and both cancel in the per-bin ratio

    T(d) = n_pos(d) / ((t_pos / t_sky) * n_sky(d)),   lambda = -ln(max(T, t_floor))

formed in the DETECTOR frame and scattered to the world-frame sky grid through
each position's pose. The live-time ratio is measured (`dT` histograms), so
lambda is absolute: its zero point is open sky, not a convention. The residual
systematic is a real flux difference between the runs (pressure, epoch), which
enters as a constant shift of lambda (`OpacityMaps.shifted`).
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cafetomo.angular import AnalysisGrid
from cafetomo.config import Config
from cafetomo.fitdata import FitData, RowIndex
from cafetomo.sky import SkyGrid, detector_to_sky, make_sky_grid


@dataclass(frozen=True)
class OpacityMaps:
    lam: dict[str, np.ndarray]   # per position, flat over the sky grid; NaN = not measured
    sky: SkyGrid

    def image(self, pid: str) -> np.ndarray:
        k = self.sky.n_bins
        return self.lam[pid].reshape(k, k)

    def shifted(self, delta: float) -> "OpacityMaps":
        return OpacityMaps(lam={p: v + delta for p, v in self.lam.items()}, sky=self.sky)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, sky_edges=self.sky.edges,
                            positions=np.array(json.dumps(sorted(self.lam))),
                            **{f"lam__{p}": v for p, v in self.lam.items()})

    @staticmethod
    def load(path: str | Path) -> "OpacityMaps":
        with np.load(path) as d:
            pids = json.loads(str(d["positions"]))
            return OpacityMaps(lam={p: d[f"lam__{p}"] for p in pids},
                               sky=SkyGrid(edges=d["sky_edges"]))


def _sky_grid(cfg: Config) -> SkyGrid:
    return make_sky_grid(cfg.opacity.sky_t_max, cfg.opacity.sky_n_bins)


def _accumulate(grid: AnalysisGrid, cfg: Config, sky: SkyGrid,
                live_time: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per position: (observed counts, expected open-sky counts) per sky bin."""
    sky_id = cfg.sky_reference.id
    if sky_id not in grid.counts:
        raise ValueError(f"analysis grid has no {sky_id!r} counts; ingest the sky reference")
    n_sky = grid.counts[sky_id].astype(np.float64)
    tx, ty = grid.tan_mesh()
    acc = cfg.detector.acceptance(tx, ty)
    t_sky = float(live_time[sky_id])
    out = {}
    for pid in cfg.position_ids:
        sx, sy, on_sky = detector_to_sky(tx, ty, cfg.exposure(pid).pose)
        flat, in_grid = sky.bin_index(sx, sy)
        live = (acc > 0) & on_sky & in_grid & (n_sky >= cfg.opacity.min_sky)
        obs = np.zeros(sky.flat_size)
        ref = np.zeros(sky.flat_size)
        np.add.at(obs, flat[live], grid.counts[pid][live].astype(np.float64))
        np.add.at(ref, flat[live], float(live_time[pid]) / t_sky * n_sky[live])
        out[pid] = (obs, ref)
    return out


def solve_opacity(grid: AnalysisGrid, cfg: Config,
                  live_time: dict[str, float]) -> OpacityMaps:
    """Absolute lambda per position and sky bin."""
    sky = _sky_grid(cfg)
    lam = {}
    for pid, (obs, ref) in _accumulate(grid, cfg, sky, live_time).items():
        out = np.full(sky.flat_size, np.nan)
        seen = ref > 0
        out[seen] = -np.log(np.clip(obs[seen] / ref[seen], cfg.opacity.t_floor, None))
        lam[pid] = out
    return OpacityMaps(lam=lam, sky=sky)


def poisson_sigma(grid: AnalysisGrid, cfg: Config,
                  live_time: dict[str, float]) -> dict[str, np.ndarray]:
    """Analytic sigma of lambda, sqrt(1/n_pos + 1/n_sky): cheap weights for
    objectives evaluated many times (the pose fit)."""
    sky = _sky_grid(cfg)
    out = {}
    for pid, (obs, ref) in _accumulate(grid, cfg, sky, live_time).items():
        s = np.full(sky.flat_size, np.nan)
        seen = ref > 0
        s[seen] = np.sqrt(1.0 / np.maximum(obs[seen], 1.0) + 1.0 / ref[seen])
        out[pid] = s
    return out


def opacity_sigma(grid: AnalysisGrid, cfg: Config, live_time: dict[str, float], *,
                  n_replicas: int, seed: int) -> dict[str, np.ndarray]:
    """Per-bin sigma of lambda from a Poisson bootstrap of every histogram,
    sky run included: the weights of the delivered fits."""
    rng = np.random.default_rng(seed)
    stacks: dict[str, list[np.ndarray]] = {}
    for _ in range(n_replicas):
        counts = {k: rng.poisson(v).astype(np.int64) for k, v in grid.counts.items()}
        maps = solve_opacity(AnalysisGrid(edges=grid.edges, counts=counts), cfg, live_time)
        for pid, lam in maps.lam.items():
            stacks.setdefault(pid, []).append(lam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # bins seen in no replica
        return {pid: np.nanstd(np.vstack(v), axis=0) for pid, v in stacks.items()}


def save_sigma(path: str | Path, sigma: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **sigma)


def load_sigma(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def build_fit_data(maps: OpacityMaps, cfg: Config, sigma: dict[str, np.ndarray], *,
                   rows: RowIndex | None = None) -> FitData:
    """Flatten opacity maps into (lambda, 1/sigma^2) over live rows.

    With `rows` the row set is PINNED (bootstrap replicas, systematics): every
    row keeps its place, so one cached system matrix and one voxel lattice
    serve all fits; a row whose lambda or sigma is not finite in this
    realisation gets weight 0 instead of disappearing.
    """
    if rows is None:
        rows = _live_rows(maps, cfg, sigma)
    lam = np.zeros(rows.n_rows)
    w = np.zeros(rows.n_rows)
    for k, pid in enumerate(rows.position_ids):
        sel = rows.pos_of_row == k
        flat = rows.sky_flat[sel]
        l_p = maps.lam[pid][flat]
        s_p = sigma[pid][flat]
        ok = np.isfinite(l_p) & np.isfinite(s_p) & (s_p > 0)
        lam[sel] = np.where(ok, l_p, 0.0)
        w[sel] = np.where(ok, 1.0 / np.where(ok, s_p, 1.0) ** 2, 0.0)
    return FitData(lam=lam, w=w, rows=rows)


def _live_rows(maps: OpacityMaps, cfg: Config, sigma: dict[str, np.ndarray]) -> RowIndex:
    n = maps.sky.n_bins
    c = maps.sky.centers
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    all_sx, all_sy = c[ii].ravel(), c[jj].ravel()
    in_reach = (np.abs(all_sx) <= cfg.opacity.max_tan) & (np.abs(all_sy) <= cfg.opacity.max_tan)
    pos, flat = [], []
    for k, pid in enumerate(cfg.position_ids):
        s = sigma[pid]
        live = np.isfinite(maps.lam[pid]) & np.isfinite(s) & (s > 0) & in_reach
        idx = np.nonzero(live)[0]
        pos.append(np.full(idx.size, k, dtype=np.int64))
        flat.append(idx.astype(np.int64))
    flat_all = np.concatenate(flat)
    if flat_all.size == 0:
        raise ValueError("no constrained sky directions: every lambda or sigma is non-finite")
    return RowIndex(position_ids=cfg.position_ids, pos_of_row=np.concatenate(pos),
                    sx=all_sx[flat_all], sy=all_sy[flat_all], sky_flat=flat_all)
