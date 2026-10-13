"""Reproducible count thinning and retrospective angular holdouts.

Each source, including the shared roof reference, is split once. Angular masks
apply to room observations; they do not change acquisition live times.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from cafetomo.angular import AnalysisGrid


@dataclass(frozen=True)
class CountPartition:
    train: AnalysisGrid
    test: AnalysisGrid
    train_live: dict[str, float]
    test_live: dict[str, float]
    seed: int


def split_counts(
    grid: AnalysisGrid, live: dict[str, float], *, fraction: float, seed: int
) -> CountPartition:
    """Allocate integer events binomially, keeping an exact complement.

    Inputs are never mutated; source order does not alter the random stream.
    """
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("fraction must be finite and strictly between 0 and 1")
    if set(live) != set(grid.counts):
        raise ValueError("live times must match the count sources")
    for pid, seconds in live.items():
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"live time for {pid!r} must be finite and positive")
    integer_counts = {}
    for pid, values in grid.counts.items():
        values = np.asarray(values)
        if values.shape != (grid.n_bins, grid.n_bins):
            raise ValueError(f"counts for {pid!r} do not match the analysis grid shape")
        if (
            not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(values != np.floor(values))
            or np.any(values >= 2**63)
        ):
            raise ValueError(f"counts for {pid!r} must be nonnegative int64-compatible integers")
        integer_counts[pid] = values.astype(np.int64)
    rng = np.random.default_rng(seed)
    train = {
        pid: rng.binomial(integer_counts[pid], fraction).astype(np.int64)
        for pid in sorted(integer_counts)
    }
    test = {pid: integer_counts[pid] - train[pid] for pid in integer_counts}
    return CountPartition(
        train=AnalysisGrid(grid.edges.copy(), train),
        test=AnalysisGrid(grid.edges.copy(), test),
        train_live={pid: fraction * seconds for pid, seconds in live.items()},
        test_live={pid: (1 - fraction) * seconds for pid, seconds in live.items()},
        seed=seed,
    )


def angular_masks(
    shape: tuple[int, int], *, fold: int, folds: int = 5, tile: int = 5, guard: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return room training/test masks, with a square raw-bin guard.

    Tile assignment is (tile_x + 2*tile_y) % folds, including partial edges.
    Guard bins belong to neither mask; every bin is tested in exactly one fold.
    """
    settings = (*shape, fold, folds, tile, guard)
    if len(shape) != 2 or any(not isinstance(v, (int, np.integer)) for v in settings):
        raise ValueError("shape and fold settings must be integers")
    if min(shape) <= 0 or folds < 2 or tile < 1 or guard < 0 or not 0 <= fold < folds:
        raise ValueError("invalid shape, fold, folds, tile or guard")
    ii, jj = np.indices(shape)
    test = ((ii // tile + 2 * (jj // tile)) % folds) == fold
    excluded = test
    if guard:
        excluded = binary_dilation(test, structure=np.ones((3, 3), dtype=bool), iterations=guard)
    return ~excluded, test
