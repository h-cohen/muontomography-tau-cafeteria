"""The measurement vector every solver and analysis consumes.

One opacity λ and one weight per (position, sky direction) row.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RowIndex:
    """Which (position, sky direction) pairs are inverted, and in what order."""

    position_ids: tuple[str, ...]
    pos_of_row: np.ndarray  # [n_rows] index into position_ids
    sx: np.ndarray  # [n_rows] world-frame tangent, x
    sy: np.ndarray  # [n_rows] world-frame tangent, y
    sky_flat: np.ndarray  # [n_rows] flat index into the sky grid

    @property
    def n_rows(self) -> int:
        return int(self.pos_of_row.size)

    def mask_for(self, pid: str) -> np.ndarray:
        return self.pos_of_row == self.position_ids.index(pid)

    def t_reach(self) -> float:
        """Largest |tangent| carrying a measurement — what sizes the voxel grid."""
        return float(max(np.abs(self.sx).max(), np.abs(self.sy).max()))

    def directions(self) -> np.ndarray:
        """[n_rows, 3] unit world-frame direction of each row."""
        d = np.stack([self.sx, self.sy, np.ones_like(self.sx)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def key(self) -> str:
        h = hashlib.sha256()
        h.update(",".join(self.position_ids).encode())
        for arr in (self.pos_of_row, self.sx, self.sy, self.sky_flat):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]


@dataclass(frozen=True)
class FitData:
    lam: np.ndarray  # [n_rows] measured optical depth
    w: np.ndarray  # [n_rows] 1/sigma^2, zero on excluded rows
    rows: RowIndex

    def restricted(self, keep: np.ndarray) -> FitData:
        """Same rows, weights zeroed outside `keep`. The layout is shared so one
        cached system matrix serves the full fit and every holdout fit."""
        return FitData(lam=self.lam, w=np.where(keep, self.w, 0.0), rows=self.rows)
