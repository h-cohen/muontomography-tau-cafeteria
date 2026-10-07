"""Sky frame: the world frame, z up, with directions carried as tangents.

The detector response is fixed in the detector frame; the rock's opacity is fixed
in the sky frame. A pose rotates detector-frame directions into the sky frame,
so two positions with different azimuth see the same rock through different
detector bins.

Directions are tangents (tx, ty) meaning the unit vector along (tx, ty, 1).
Rotation is applied to the unit vector, not to the tangents, which is why the
map is carried through the unit sphere rather than done in tangent space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cafetomo.config import Pose

_HORIZON_EPS = 1e-6


def detector_to_sky(
    tx: np.ndarray, ty: np.ndarray, pose: Pose
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map detector-frame tangents to sky-frame tangents.

    Returns (sx, sy, valid). `valid` is False where the rotated ray points at or
    below the horizon, where a tangent representation is meaningless.
    """
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)

    vec = np.stack([tx, ty, np.ones_like(tx)], axis=-1)
    vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    rotated = vec @ pose.rotation().T
    z = rotated[..., 2]
    valid = z > _HORIZON_EPS

    safe_z = np.where(valid, z, 1.0)
    sx = np.where(valid, rotated[..., 0] / safe_z, np.nan)
    sy = np.where(valid, rotated[..., 1] / safe_z, np.nan)
    return sx, sy, valid


@dataclass(frozen=True)
class SkyGrid:
    edges: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def flat_size(self) -> int:
        return self.n_bins * self.n_bins

    def bin_index(self, sx: np.ndarray, sy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat bin index and an in-range mask. Out-of-range entries get index 0
        and mask False; callers must apply the mask, never trust the index."""
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        finite = np.isfinite(sx) & np.isfinite(sy)

        i = np.digitize(np.where(finite, sx, 0.0), self.edges) - 1
        j = np.digitize(np.where(finite, sy, 0.0), self.edges) - 1
        ok = finite & (i >= 0) & (i < self.n_bins) & (j >= 0) & (j < self.n_bins)

        flat = np.where(ok, i * self.n_bins + j, 0)
        return flat.astype(np.int64), ok


def make_sky_grid(t_max: float = 2.5, n_bins: int = 100) -> SkyGrid:
    """Sky-frame tangent grid at the same 0.05 resolution as the analysis grid.

    The +-2.5 span holds all counts of an untilted detector.
    """
    return SkyGrid(edges=np.linspace(-t_max, t_max, n_bins + 1))
