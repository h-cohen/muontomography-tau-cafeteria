"""Sparse system matrix: mean path length of each row's ray bundle per voxel.

A row is one (position, sky direction) pair from cafetomo.fitdata.RowIndex, and
its measurement is lambda = integral of opacity density along that ray. The
detector aperture spans one to two voxels, so a row is modelled as a bundle
of n_sub^2 parallel sub-rays across the aperture rather than as a pinhole.
Entries are path lengths AVERAGED over the bundle, so a bundle and a
pinhole through uniform material predict the same optical depth.

Directions already lie in the world frame and are not rotated again.
Detector azimuth rotates only the accepted footprint in its midpoint plane.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse

from cafetomo.fitdata import RowIndex
from cafetomo.voxels import VoxelGrid

# Bumped whenever ray-casting logic changes. It is part of the cache key, so a
# cache built before a fix is never served after it.
INVERSION_VERSION = 3

_SAMPLES_PER_VOXEL = 3  # sampling step along a ray = spacing / this
_ROW_BLOCK = 512  # rows processed per vectorised block


def coincidence_spans(
    directions: np.ndarray,
    aperture_m: float,
    layer_dz_m: float,
    az_deg: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Accepted midpoint widths in the two detector axes."""
    d = np.asarray(directions, dtype=np.float64)
    if aperture_m < 0 or layer_dz_m < 0:
        raise ValueError("invalid aperture geometry")
    angle = np.radians(np.broadcast_to(az_deg, (len(d),)))
    co, si = np.cos(angle), np.sin(angle)
    span = np.full((len(d), 2), aperture_m, dtype=np.float64)
    if layer_dz_m > 0:
        if not np.all(np.isfinite(d)) or np.any(d[:, 2] == 0):
            raise ValueError("direction has no finite coincidence support")
        slopes = np.column_stack([co * d[:, 0] + si * d[:, 1], -si * d[:, 0] + co * d[:, 1]])
        span -= layer_dz_m * np.abs(slopes / d[:, 2, None])
        if np.any(span <= 0):
            raise ValueError(
                "direction centre has no coincidence support; integrate its angular bin"
            )
    return span


def bundle_offsets(
    directions: np.ndarray,
    aperture_m: float,
    n_sub: int,
    *,
    layer_dz_m: float = 0.0,
    az_deg: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Uniform quadrature of the accepted detector-midpoint rectangle.

    Its widths are W-abs(t_local)*D and its axes rotate with the detector.
    D=0 is a generic single-plane aperture. Unsupported direction centres
    are rejected rather than replaced by artificial pinholes.
    """
    if n_sub < 1:
        raise ValueError("invalid aperture quadrature")
    d = np.asarray(directions, dtype=np.float64)
    span = coincidence_spans(d, aperture_m, layer_dz_m, az_deg)
    angle = np.radians(np.broadcast_to(az_deg, (len(d),)))
    co, si = np.cos(angle), np.sin(angle)
    g = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    gx, gy = np.meshgrid(g, g, indexing="ij")
    x, y = span[:, 0, None] * gx.ravel(), span[:, 1, None] * gy.ravel()
    offsets = np.zeros((len(d), n_sub**2, 3))
    offsets[:, :, 0] = co[:, None] * x - si[:, None] * y
    offsets[:, :, 1] = si[:, None] * x + co[:, None] * y
    return offsets


def build_system_matrix(
    rows: RowIndex,
    origins: dict[str, tuple[float, float, float]],
    grid: VoxelGrid,
    *,
    aperture_m: float,
    n_sub: int = 4,
    cache_dir: str | Path | None = None,
    layer_dz_m: float = 0.0,
    azimuths: dict[str, float] | None = None,
) -> sparse.csr_matrix:
    """A[row, voxel] in METRES, shape [rows.n_rows, grid.n_voxels]."""
    if cache_dir is not None:
        key = _cache_key(rows, origins, grid, aperture_m, n_sub, layer_dz_m, azimuths)
        cache = Path(cache_dir) / f"A_{key}.npz"
        if cache.exists():
            return sparse.load_npz(cache)

    dirs = rows.directions()  # [nr, 3]
    azimuths = {} if azimuths is None else azimuths
    az = np.array([azimuths.get(rows.position_ids[i], 0.0) for i in rows.pos_of_row])
    offs = bundle_offsets(dirs, aperture_m, n_sub, layer_dz_m=layer_dz_m, az_deg=az)
    starts = np.array([origins[rows.position_ids[i]] for i in rows.pos_of_row])
    starts = starts[:, None, :] + offs  # [nr, ns, 3]

    origin = np.asarray(grid.origin, dtype=np.float64)
    shape = np.asarray(grid.shape, dtype=np.int64)
    z0, z1 = grid.extent(2)

    # Entry and exit of the grid's z slab along each ray, measured from its own
    # start. dz > 0 always: a row's direction is normalize(sx, sy, 1). A
    # detector below the grid (the usual case) enters at z0; a detector already
    # inside the grid (embedded in the volume, looking up through it) starts
    # its path at t=0, not at z0 — using z0 unconditionally would count the
    # region behind the detector as if the ray had travelled through it.
    dz = dirs[:, 2, None]
    start_z = starts[:, :, 2]
    t_in = np.maximum((z0 - start_z) / dz, 0.0)
    t_out = (z1 - start_z) / dz
    length = np.maximum(t_out - t_in, 0.0)
    step = grid.spacing / _SAMPLES_PER_VOXEL
    # All sub-rays in a row share sampling fractions, but each traverses its
    # own segment. The longest segment controls the maximum sampling step.
    n_samp = np.maximum(np.ceil(length.max(axis=1) / step).astype(np.int64), 1)

    nr, ns = rows.n_rows, offs.shape[1]
    r_out, c_out, v_out = [], [], []
    for lo in range(0, nr, _ROW_BLOCK):
        hi = min(lo + _ROW_BLOCK, nr)
        cn = n_samp[lo:hi]
        row_rep = np.repeat(np.arange(lo, hi), cn)  # [Nt]
        # midpoint sampling fraction along each ray's in-slab segment
        base = np.repeat(np.concatenate([[0], np.cumsum(cn[:-1])]), cn)
        frac = (np.arange(int(cn.sum())) - base + 0.5) / np.repeat(cn, cn)
        t = t_in[row_rep] + frac[:, None] * length[row_rep]  # [Nt, ns]

        pts = starts[row_rep] + t[:, :, None] * dirs[row_rep, None, :]  # [Nt, ns, 3]
        idx = np.floor((pts - origin) / grid.spacing).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), axis=-1) & (length[row_rep] > 0)
        flat = (idx[..., 0] * grid.shape[1] + idx[..., 1]) * grid.shape[2] + idx[..., 2]
        # Divide by ns: the bundle AVERAGES path length, it does not accumulate.
        dl = length[row_rep] / n_samp[row_rep, None] / ns

        r_out.append(np.broadcast_to(row_rep[:, None], inside.shape)[inside])
        c_out.append(flat[inside])
        v_out.append(dl[inside])

    A = sparse.coo_matrix(
        (
            np.concatenate(v_out) if v_out else np.zeros(0),
            (
                np.concatenate(r_out) if r_out else np.zeros(0, dtype=np.int64),
                np.concatenate(c_out) if c_out else np.zeros(0, dtype=np.int64),
            ),
        ),
        shape=(nr, grid.n_voxels),
    ).tocsr()
    A.sum_duplicates()

    if cache_dir is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cache, A)
    return A


def _cache_key(
    rows: RowIndex,
    origins: dict,
    grid: VoxelGrid,
    aperture_m: float,
    n_sub: int,
    layer_dz_m: float = 0.0,
    azimuths: dict[str, float] | None = None,
) -> str:
    h = hashlib.sha256()
    h.update(f"v{INVERSION_VERSION}|".encode())
    h.update(rows.key().encode())
    azimuths = {} if azimuths is None else azimuths
    h.update(f"D:{layer_dz_m:.6f}|".encode())
    for pid in rows.position_ids:
        h.update(f"az:{azimuths.get(pid, 0.0):.6f}|".encode())
        x, y, z = origins[pid]
        h.update(f"{pid}:{x:.6f},{y:.6f},{z:.6f};".encode())
    h.update(f"{grid.key()}|{aperture_m:.6f}|{n_sub}|{_SAMPLES_PER_VOXEL}".encode())
    return h.hexdigest()[:16]
