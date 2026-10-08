"""Statistical uncertainty by Poisson bootstrap over every histogram.

Each replica resamples the position AND sky counts and is pushed through the
whole chain -- opacity, autofocus, beams, beam depth, voxels -- with the
nominal row set, weights and lattice held fixed: the weights define the
estimator, they are not part of the noise.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cafetomo.angular import AnalysisGrid
from cafetomo.config import Config
from cafetomo.fitdata import RowIndex
from cafetomo.measure import measure
from cafetomo.opacity import build_fit_data, solve_opacity
from cafetomo.sky import SkyGrid
from cafetomo.voxels import VoxelGrid


def resample(grid: AnalysisGrid, rng: np.random.Generator) -> AnalysisGrid:
    return AnalysisGrid(
        edges=grid.edges,
        counts={k: rng.poisson(v).astype(np.int64) for k, v in grid.counts.items()},
    )


@dataclass(frozen=True)
class BootstrapResult:
    values: dict[str, np.ndarray]
    volume_mean: np.ndarray
    volume_sigma: np.ndarray

    def summary(self) -> dict[str, float | int]:
        """Mean and sigma over the finite replicas, with their count.

        A z-profile face can fall off the grid in some replicas (NaN by
        design); those replicas carry no value for that key, and `<k>_n`
        says how many did, so a thinned sigma is visible, not hidden."""
        out = {}
        for k, v in self.values.items():
            ok = v[np.isfinite(v)]
            out[f"{k}_mean"] = float(ok.mean()) if ok.size else float("nan")
            out[f"{k}_sigma"] = float(np.std(ok, ddof=1)) if ok.size > 1 else float("nan")
            out[f"{k}_n"] = int(ok.size)
        return out

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "values.npz", **self.values)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(self.volume_sigma > 0, self.volume_mean / self.volume_sigma, np.nan)
        np.savez_compressed(
            out / "volume_stats.npz",
            mean=self.volume_mean.astype(np.float32),
            sigma=self.volume_sigma.astype(np.float32),
            snr=snr.astype(np.float32),
        )


def run_bootstrap(
    grid: AnalysisGrid,
    cfg: Config,
    *,
    live_time: dict[str, float],
    sigma: dict[str, np.ndarray],
    rows: RowIndex,
    vgrid: VoxelGrid,
    sky: SkyGrid,
    cache_dir: str | Path | None = None,
) -> BootstrapResult:
    """A replica whose measurement raises stops the bootstrap: dropping it
    would bias the spread toward the replicas the chain happens to survive."""
    u = cfg.uncertainty
    # Weight estimation uses the root seed directly. A separate deterministic
    # stream prevents its draws from reappearing as measurement replicas.
    rng = np.random.default_rng(np.random.SeedSequence(u.seed, spawn_key=(1,)))
    values: dict[str, list[float]] = {}
    vols = []
    for replica in range(u.n_replicas):
        maps = solve_opacity(resample(grid, rng), cfg, live_time)
        m = measure(
            build_fit_data(maps, cfg, sigma, rows=rows), cfg, sky, vgrid=vgrid, cache_dir=cache_dir
        )
        for k, v in m.values.items():
            values.setdefault(k, []).append(v)
        vols.append(m.volume)
        print(f"bootstrap: completed {replica + 1}/{u.n_replicas}", file=sys.stderr, flush=True)
    arr = np.stack(vols)
    vsig = arr.std(axis=0, ddof=1) if len(vols) > 1 else np.full(arr.shape[1:], np.nan)
    return BootstrapResult(
        values={k: np.array(v) for k, v in values.items()},
        volume_mean=arr.mean(axis=0),
        volume_sigma=vsig,
    )
