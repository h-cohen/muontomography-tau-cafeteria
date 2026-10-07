"""Vertical depth of the ceiling beams.

Headline -- a parametric fit to the measured opacity. Each beam is a box along
y with bottom face z0, width w and depth h (shared) and its own opacity
density kappa_k, under a smooth per-position background (the slab and the
rest of the room). Depth is geometric: the shadow of a deep beam widens with
obliquity (apparent width ~ w + h*|tan|) and its lambda plateau is longest
near normal incidence; the two positions see each beam from different angles.
The fit uses the measured lambda directly, not the regularised volume, so it
is not subject to the inversion's smearing along depth. Linear parameters
(kappa, background) are projected out at each step (variable projection).

Cross-check -- the z extent of the beams in the reconstructed volume, read
against the analytic depth resolution. Regularisation and the depth null space
bias it; it is a consistency check, not a measurement.

An unmodelled sharp feature in the background (a slab shadow that ends inside
the band, say) is absorbed by the beam boxes and biases h, by ~3% on the test
phantom: a systematic to budget, not a fit failure.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from cafetomo.beams import find_beams
from cafetomo.config import Config
from cafetomo.fitdata import FitData, RowIndex
from cafetomo.raycast import bundle_offsets
from cafetomo.reconstruct import VoxelSolution
from cafetomo.resolution import depth_resolution, position_baselines
from cafetomo.sky import SkyGrid

# The off-beam reference column for a lone beam sits half the ceiling's
# 1.7 m beam pitch away: where the gap between neighbours would be centred.
_SINGLE_BEAM_OFF_M = 0.85

_GH = (np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)]), np.array([1 / 6, 2 / 3, 1 / 6]))


def box_path_lengths(starts: np.ndarray, dirs: np.ndarray, lo: np.ndarray,
                     hi: np.ndarray) -> np.ndarray:
    """Length of each ray (start, unit direction, forward only) inside the
    axis-aligned box [lo, hi]: the slab method."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (lo - starts) * inv
        t2 = (hi - starts) * inv
    parallel = dirs == 0
    inside = (starts >= lo) & (starts <= hi)
    t_near = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    t_far = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    enter = np.maximum(t_near.max(axis=-1), 0.0)
    leave = t_far.min(axis=-1)
    return np.clip(leave - enter, 0.0, None)


def beam_design(rows: RowIndex, origins: dict, *, aperture_m: float, n_sub: int, z0: float,
                w: float, h: float, xs, y_extent, angle_jitter: float = 0.0) -> np.ndarray:
    """Bundle-averaged path length of every row through every beam box,
    [n_rows, n_beams]. `angle_jitter` (tan units) smears directions along x
    by a 3-point Gauss-Hermite rule: the MCS systematic."""
    starts0 = np.array([origins[rows.position_ids[i]] for i in rows.pos_of_row])
    out = np.zeros((rows.n_rows, len(xs)))
    nodes, weights = _GH if angle_jitter > 0 else (np.zeros(1), np.ones(1))
    for node, wt in zip(nodes, weights, strict=True):
        d = np.stack([rows.sx + node * angle_jitter, rows.sy, np.ones(rows.n_rows)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        starts = starts0[:, None, :] + bundle_offsets(d, aperture_m, n_sub)
        dd = np.broadcast_to(d[:, None, :], starts.shape)
        for k, xk in enumerate(xs):
            lo = np.array([xk - w / 2, y_extent[0], z0])
            hi = np.array([xk + w / 2, y_extent[1], z0 + h])
            out[:, k] += wt * box_path_lengths(starts, dd, lo, hi).mean(axis=1)
    return out


def _background(rows: RowIndex, degree: int) -> np.ndarray:
    """Per-position polynomial in (sx, sy) up to total `degree`."""
    cols = []
    for k in range(len(rows.position_ids)):
        on = (rows.pos_of_row == k).astype(np.float64)
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                cols.append(on * rows.sx**i * rows.sy**j)
    return np.stack(cols, axis=-1)


@dataclass(frozen=True)
class BeamDepthFit:
    z0: float
    w: float
    h: float
    xs: tuple[float, ...]
    kappa: tuple[float, ...]
    chi2_per_dof: float
    at_bound: bool
    n_rows: int
    profiles: dict
    converged: bool
    n_eval: int

    def to_json(self) -> dict:
        return {"zbottom": self.z0, "w": self.w, "h": self.h, "ztop": self.z0 + self.h,
                "xs": list(self.xs), "kappa": list(self.kappa),
                "chisq_per_dof": self.chi2_per_dof, "at_bound": self.at_bound,
                "n_rows": self.n_rows, "profiles": self.profiles,
                "converged": self.converged, "n_eval": self.n_eval}


def fit_beam_depth(data: FitData, cfg: Config, *, xs_init, z0_init: float,
                   angle_jitter: float = 0.0) -> BeamDepthFit:
    """Fit the box model, seeded from the triangulated beams.

    `z0_init` is the triangulated beam height, which lies between the beams'
    bottom face and their centre; the box fit has local minima, and seeded at
    only one end it can settle tens of cm off in h at a worse chi^2 (seen on
    phantoms from either end). It is therefore fitted twice, with the bottom
    face seeded at z0_init and at z0_init - h_init/2 (the seed box centred
    there), and the lower chi^2 wins: both use the same rows, so the chi^2
    values compare directly. A non-finite chi^2 cannot be ranked and raises."""
    fits = [_fit_once(data, cfg, xs_init=xs_init, z0_init=z0, angle_jitter=angle_jitter)
            for z0 in (z0_init, z0_init - cfg.beamdepth.h_init_m / 2)]
    bad = [f.chi2_per_dof for f in fits if not np.isfinite(f.chi2_per_dof)]
    if bad:
        raise RuntimeError(f"beam-depth fit returned a non-finite chi^2/dof: {bad}")
    return min(fits, key=lambda f: f.chi2_per_dof)


def estimate_beam_depth(data: FitData, cfg: Config, sky: SkyGrid, *,
                        angle_jitter: float = 0.0) -> tuple[dict, BeamDepthFit]:
    """The reported beam-depth estimator: triangulate, then fit from it.

    The measurement and its phantom validation both call this, so the
    validation characterises exactly the estimator the paper quotes, seeds
    included. A failed triangulation raises rather than seeding with NaN."""
    beams = find_beams(data, cfg, sky)
    if not beams["ok"]:
        raise RuntimeError(f"beam triangulation failed: {beams}")
    fit = fit_beam_depth(data, cfg, xs_init=beams["beams_x"], z0_init=beams["z"],
                         angle_jitter=angle_jitter)
    return beams, fit


def _fit_once(data: FitData, cfg: Config, *, xs_init, z0_init: float,
              angle_jitter: float) -> BeamDepthFit:
    s = cfg.beamdepth
    keep = (data.w > 0) & (np.abs(data.rows.sy) <= s.band_sy)
    rows = RowIndex(data.rows.position_ids, data.rows.pos_of_row[keep], data.rows.sx[keep],
                    data.rows.sy[keep], data.rows.sky_flat[keep])
    lam, sw = data.lam[keep], np.sqrt(data.w[keep])
    bg = _background(rows, s.bg_degree)
    origins = cfg.origins()
    nb = len(xs_init)

    def design(theta):
        z0, w, h, *xs = theta
        return np.hstack([beam_design(rows, origins, aperture_m=cfg.detector.aperture_m,
                                      n_sub=s.n_sub, z0=z0, w=w, h=h, xs=xs,
                                      y_extent=s.y_extent_m, angle_jitter=angle_jitter), bg])

    def resid(theta):
        X = design(theta) * sw[:, None]
        coef, *_ = np.linalg.lstsq(X, lam * sw, rcond=None)
        return lam * sw - X @ coef

    x0 = np.array([z0_init, s.w_init_m, s.h_init_m, *xs_init], dtype=np.float64)
    lo = np.array([z0_init - 1.0, 0.05, 0.05, *(np.asarray(xs_init) - 0.3)])
    hi = np.array([z0_init + 1.0, 1.0, s.h_max_m, *(np.asarray(xs_init) + 0.3)])
    fit = optimize.least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), x_scale="jac")
    if fit.status <= 0:
        raise RuntimeError(f"beam-depth fit failed (status {fit.status}): {fit.message}")
    theta = fit.x
    X = design(theta)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], lam * sw, rcond=None)
    # The trust-region solver keeps iterates strictly inside the box and its
    # active_mask stays empty, so a parameter pinned by a bound stops a small
    # fraction of the span short of it: 1% of the span is "at the bound".
    span = hi - lo
    at_bound = bool(np.any(np.minimum(theta - lo, hi - theta) < 1e-2 * span))
    dof = max(rows.n_rows - (len(theta) + X.shape[1]), 1)
    model = X @ coef
    profiles = {}
    for k, pid in enumerate(rows.position_ids):
        sel = rows.pos_of_row == k
        sx = np.round(rows.sx[sel], 6)
        edges = np.unique(sx)
        prof_d = [float(np.mean(lam[sel][sx == e])) for e in edges]
        prof_m = [float(np.mean(model[sel][sx == e])) for e in edges]
        profiles[pid] = {"s": edges.tolist(), "data": prof_d, "model": prof_m}
    return BeamDepthFit(z0=float(theta[0]), w=float(theta[1]), h=float(theta[2]),
                        xs=tuple(float(v) for v in theta[3:]),
                        kappa=tuple(float(v) for v in coef[:nb]),
                        chi2_per_dof=float(np.sum(fit.fun**2) / dof), at_bound=at_bound,
                        n_rows=int(rows.n_rows), profiles=profiles,
                        converged=bool(fit.success), n_eval=int(fit.nfev))


def zprofile_depth(sol: VoxelSolution, cfg: Config, *, xs, w: float, z_ref: float) -> dict:
    """Beam-minus-between-beam opacity vs z in the volume, and its half-maximum extent.

    A face whose half-maximum crossing lies off the grid is NaN (not measured):
    clamping it to the grid edge would report the grid, not the beam."""
    g = sol.grid
    rho = sol.rho3()
    x = g.axis_centers(0)
    y = g.axis_centers(1)
    z = g.axis_centers(2)
    yb = (y >= cfg.beamdepth.y_band_m[0]) & (y <= cfg.beamdepth.y_band_m[1])
    xs = np.asarray(xs)
    on = np.zeros(x.size, dtype=bool)
    for xk in xs:
        on |= np.abs(x - xk) <= w / 2 + g.spacing / 2
    mids = 0.5 * (xs[:-1] + xs[1:]) if xs.size > 1 else xs + _SINGLE_BEAM_OFF_M
    off = np.zeros(x.size, dtype=bool)
    for m in mids:
        off |= np.abs(x - m) <= w / 2 + g.spacing / 2
    for name, sel in (("y band", yb), ("beam columns", on), ("between-beam columns", off)):
        if not sel.any():
            raise ValueError(f"z-profile: the {name} select no voxels of the grid")
    prof = rho[on][:, yb].mean(axis=(0, 1)) - rho[off][:, yb].mean(axis=(0, 1))
    i = int(np.argmax(prof))
    if prof[i] <= 0:
        raise ValueError("z-profile: no beam signal (beam minus between-beam contrast <= 0)")
    half = 0.5 * prof[i]
    lo_i = i
    while lo_i > 0 and prof[lo_i - 1] >= half:
        lo_i -= 1
    hi_i = i
    while hi_i < z.size - 1 and prof[hi_i + 1] >= half:
        hi_i += 1

    def cross(a: int, b: int) -> float:
        return float(np.interp(half, [prof[a], prof[b]], [z[a], z[b]]))

    bottom = cross(lo_i - 1, lo_i) if lo_i > 0 else float("nan")
    top = cross(hi_i + 1, hi_i) if hi_i < z.size - 1 else float("nan")
    baseline = max(position_baselines(cfg).values())
    sigma_t = cfg.opacity.sky_t_max * 2 / cfg.opacity.sky_n_bins / np.sqrt(12.0)
    return {"bottom": bottom, "top": top, "fwhm": top - bottom, "peak": float(z[i]),
            "resolution": depth_resolution(z_ref, baseline, sigma_t),
            "profile_z": z.tolist(), "profile": prof.tolist()}
