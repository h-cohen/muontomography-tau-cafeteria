"""Vertical depth of the ceiling beams.

Headline -- a parametric fit to the measured opacity. Each beam is a box along
y with bottom face z0, width w and depth h (shared) under a smooth
per-position background (the slab and the rest of the room). Depth is
geometric: the shadow of a deep beam widens with obliquity (apparent width
~ w + h*|tan|) and its lambda plateau is longest near normal incidence; the
two positions see each beam from different angles. The fit uses the measured
lambda directly, not the regularised volume, so it is not subject to the
inversion's smearing along depth. Linear parameters are projected out at each
step (variable projection).

The beams' opacity is pinned to concrete physics. A free opacity density per
beam trades off against h (contrast ~ kappa * h), which made the real-data
depth bimodal. The fit therefore runs in two passes: a free-kappa pass only
supplies the background lambda of every fitted row, which `muonphysics`
converts into the slant grammage X the ray already crosses; the second pass
models each beam crossing exactly, as lambda(X + 100 rho L) - lambda(X) for
total beam path L at concrete density rho, and refits only the geometry and
the background. The exact term is used because the linearised kappa(X) * L is
off by up to ~11% over the beam paths in question (tests/test_muonphysics.py).

Cross-check -- the z extent of the beams in the reconstructed volume, read
against the analytic depth resolution. Regularisation and the depth null space
bias it; it is a consistency check, not a measurement.

An unmodelled sharp feature in the background (a slab shadow that ends inside
the band, say) is absorbed by the beam boxes and biases h, by ~3% on the test
phantom: a systematic to budget, not a fit failure.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from cafetomo.beams import find_beams
from cafetomo.config import Config
from cafetomo.fitdata import FitData, RowIndex
from cafetomo.muonphysics import Transmission
from cafetomo.raycast import bundle_offsets
from cafetomo.reconstruct import VoxelSolution
from cafetomo.resolution import depth_resolution, position_baselines
from cafetomo.sky import SkyGrid

# The off-beam reference column for a lone beam sits half the ceiling's
# 1.7 m beam pitch away: where the gap between neighbours would be centred.
_SINGLE_BEAM_OFF_M = 0.85

_GH = (np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)]), np.array([1 / 6, 2 / 3, 1 / 6]))


def box_path_lengths(
    starts: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
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


def beam_design(
    rows: RowIndex,
    origins: dict,
    *,
    aperture_m: float,
    n_sub: int,
    z0: float,
    w: float,
    h: float,
    xs,
    y_extent,
    angle_jitter: float = 0.0,
) -> np.ndarray:
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
    """`kappa_mean` (1/m) and `overburden_mean` (slant g/cm^2) are the pinned
    opacity density and the grammage in front of the beams, averaged over the
    fitted rows weighted by each row's path through the beams; NaN for the
    free-kappa pass, which pins nothing. `background` is the fitted
    background lambda per fitted row, the input of the pinned pass."""

    z0: float
    w: float
    h: float
    xs: tuple[float, ...]
    chi2_per_dof: float
    at_bound: bool
    n_rows: int
    profiles: dict
    converged: bool
    n_eval: int
    kappa_mean: float
    overburden_mean: float
    density: float
    background: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict:
        return {
            "zbottom": self.z0,
            "w": self.w,
            "h": self.h,
            "ztop": self.z0 + self.h,
            "xs": list(self.xs),
            "kappa_mean": self.kappa_mean,
            "overburden_mean": self.overburden_mean,
            "density": self.density,
            "chisq_per_dof": self.chi2_per_dof,
            "at_bound": self.at_bound,
            "n_rows": self.n_rows,
            "profiles": self.profiles,
            "converged": self.converged,
            "n_eval": self.n_eval,
        }


@dataclass(frozen=True)
class _Band:
    """The rows the box fit uses: measured, within `band_sy` of the beams' axis."""

    rows: RowIndex
    lam: np.ndarray
    sw: np.ndarray

    def cos_theta(self) -> np.ndarray:
        return 1.0 / np.sqrt(1.0 + self.rows.sx**2 + self.rows.sy**2)


@dataclass(frozen=True)
class _Pinned:
    """Per fitted row: transmission model and the background grammage."""

    trans: Transmission
    x_bg: np.ndarray
    density: float

    def beam_lam(self, L: np.ndarray) -> np.ndarray:
        """Exact opacity added by beam path L [n_rows, n_beams] of concrete."""
        x = self.x_bg + 100.0 * self.density * L.sum(axis=1)
        return self.trans.lam(x) - self.trans.lam(self.x_bg)


def _band(data: FitData, cfg: Config) -> _Band:
    keep = (data.w > 0) & (np.abs(data.rows.sy) <= cfg.beamdepth.band_sy)
    rows = RowIndex(
        data.rows.position_ids,
        data.rows.pos_of_row[keep],
        data.rows.sx[keep],
        data.rows.sy[keep],
        data.rows.sky_flat[keep],
    )
    return _Band(rows=rows, lam=data.lam[keep], sw=np.sqrt(data.w[keep]))


def _best(fits: list[BeamDepthFit]) -> BeamDepthFit:
    bad = [f.chi2_per_dof for f in fits if not np.isfinite(f.chi2_per_dof)]
    if bad:
        raise RuntimeError(f"beam-depth fit returned a non-finite chi^2/dof: {bad}")
    return min(fits, key=lambda f: f.chi2_per_dof)


def fit_beam_depth(
    data: FitData,
    cfg: Config,
    *,
    xs_init,
    z0_init: float,
    angle_jitter: float = 0.0,
    density: float | None = None,
    flux_model: str | None = None,
) -> BeamDepthFit:
    """Fit the box model, seeded from the triangulated beams, with the beam
    opacity pinned to concrete of `density` (g/cm^3; None: the config's) under
    the muon spectrum `flux_model` (None: the config's).

    `z0_init` is the triangulated beam height, which lies between the beams'
    bottom face and their centre; the box fit has local minima, and seeded at
    only one end it can settle tens of cm off in h at a worse chi^2 (seen on
    phantoms from either end). Each pass is therefore started with the bottom
    face at z0_init and at z0_init - h0/2 (the seed box centred there), and the
    lower chi^2 wins: all starts use the same rows, so the chi^2 values compare
    directly. A non-finite chi^2 cannot be ranked and raises.

    The free pass (h0 = `h_init_m`) fixes the background grammage of the
    pinned pass (see the module docstring). The pinned chi^2 still has local
    minima in h, a few hundredths apart in chi^2/dof, so the pinned pass is
    started from every depth in `h_starts_m` and from the free pass's h. On
    h = 1.25 m phantoms, starts at 0.6 m and the free h alone missed the
    lowest-chi^2 depth by ~0.2 m in some noise draws (the free h itself can
    be bimodal, 0.45 vs 1.4 m); the spread of starts found it in every draw
    tried."""
    phys = cfg.physics
    band = _band(data, cfg)
    common = dict(xs_init=xs_init, angle_jitter=angle_jitter)

    def starts(h0: float, pinned: _Pinned | None) -> list[BeamDepthFit]:
        return [
            _fit_once(band, cfg, z0_init=z0, h_init=h0, pinned=pinned, **common)
            for z0 in (z0_init, z0_init - h0 / 2)
        ]

    h_init = cfg.beamdepth.h_init_m
    free = _best(starts(h_init, None))
    trans = Transmission(
        band.cos_theta(),
        threshold_gev=phys.detector_threshold_gev,
        model=phys.flux_model if flux_model is None else flux_model,
    )
    pinned = _Pinned(
        trans=trans,
        x_bg=trans.overburden(free.background),
        density=phys.concrete_density_gcm3 if density is None else float(density),
    )
    h_starts = (*cfg.beamdepth.h_starts_m, free.h)
    return _best([f for h0 in h_starts for f in starts(h0, pinned)])


def estimate_beam_depth(
    data: FitData, cfg: Config, sky: SkyGrid, *, angle_jitter: float = 0.0
) -> tuple[dict, BeamDepthFit]:
    """The reported beam-depth estimator: triangulate, then fit from it.

    The measurement and its phantom validation both call this, so the
    validation characterises exactly the estimator the paper quotes, seeds
    included. A failed triangulation raises rather than seeding with NaN."""
    beams = find_beams(data, cfg, sky)
    if not beams["ok"]:
        raise RuntimeError(f"beam triangulation failed: {beams}")
    fit = fit_beam_depth(
        data, cfg, xs_init=beams["beams_x"], z0_init=beams["z"], angle_jitter=angle_jitter
    )
    return beams, fit


def _fit_once(
    band: _Band,
    cfg: Config,
    *,
    xs_init,
    z0_init: float,
    h_init: float,
    angle_jitter: float,
    pinned: _Pinned | None,
) -> BeamDepthFit:
    """One seeded fit. Free (`pinned` None): every beam carries its own linear
    opacity density. Pinned: the beam term is fixed physics and only the
    background is linear."""
    s = cfg.beamdepth
    rows, lam, sw = band.rows, band.lam, band.sw
    bg = _background(rows, s.bg_degree)
    origins = cfg.origins()

    def paths(theta):
        z0, w, h, *xs = theta
        return beam_design(
            rows,
            origins,
            aperture_m=cfg.detector.aperture_m,
            n_sub=s.n_sub,
            z0=z0,
            w=w,
            h=h,
            xs=xs,
            y_extent=s.y_extent_m,
            angle_jitter=angle_jitter,
        )

    def split(theta):
        """(fixed part of the model, linear design matrix) at theta."""
        L = paths(theta)
        if pinned is None:
            return np.zeros(rows.n_rows), np.hstack([L, bg]), L
        return pinned.beam_lam(L), bg, L

    def resid(theta):
        fixed, X, _ = split(theta)
        y = (lam - fixed) * sw
        coef, *_ = np.linalg.lstsq(X * sw[:, None], y, rcond=None)
        return y - (X * sw[:, None]) @ coef

    x0 = np.array([z0_init, s.w_init_m, h_init, *xs_init], dtype=np.float64)
    lo = np.array([z0_init - 1.0, 0.05, 0.05, *(np.asarray(xs_init) - 0.3)])
    hi = np.array([z0_init + 1.0, 1.0, s.h_max_m, *(np.asarray(xs_init) + 0.3)])
    fit = optimize.least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), x_scale="jac")
    if fit.status <= 0:
        raise RuntimeError(f"beam-depth fit failed (status {fit.status}): {fit.message}")
    theta = fit.x
    fixed, X, L = split(theta)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], (lam - fixed) * sw, rcond=None)
    # The trust-region solver keeps iterates strictly inside the box and its
    # active_mask stays empty, so a parameter pinned by a bound stops a small
    # fraction of the span short of it: 1% of the span is "at the bound".
    span = hi - lo
    at_bound = bool(np.any(np.minimum(theta - lo, hi - theta) < 1e-2 * span))
    dof = max(rows.n_rows - (len(theta) + X.shape[1]), 1)
    model = fixed + X @ coef
    background = bg @ coef[-bg.shape[1] :]
    if pinned is None:
        kappa_mean = overburden_mean = density = float("nan")
    else:
        through = L.sum(axis=1)
        if not np.any(through > 0):
            raise RuntimeError("beam-depth fit: no fitted row crosses a beam")
        kappa = pinned.trans.kappa(pinned.x_bg, pinned.density)
        kappa_mean = float(np.average(kappa, weights=through))
        overburden_mean = float(np.average(pinned.x_bg, weights=through))
        density = pinned.density
    profiles = {}
    for k, pid in enumerate(rows.position_ids):
        sel = rows.pos_of_row == k
        sx = np.round(rows.sx[sel], 6)
        edges = np.unique(sx)
        prof_d = [float(np.mean(lam[sel][sx == e])) for e in edges]
        prof_m = [float(np.mean(model[sel][sx == e])) for e in edges]
        profiles[pid] = {"s": edges.tolist(), "data": prof_d, "model": prof_m}
    return BeamDepthFit(
        z0=float(theta[0]),
        w=float(theta[1]),
        h=float(theta[2]),
        xs=tuple(float(v) for v in theta[3:]),
        chi2_per_dof=float(np.sum(fit.fun**2) / dof),
        at_bound=at_bound,
        n_rows=int(rows.n_rows),
        profiles=profiles,
        converged=bool(fit.success),
        n_eval=int(fit.nfev),
        kappa_mean=kappa_mean,
        overburden_mean=overburden_mean,
        density=density,
        background=background,
    )


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
    return {
        "bottom": bottom,
        "top": top,
        "fwhm": top - bottom,
        "peak": float(z[i]),
        "resolution": depth_resolution(z_ref, baseline, sigma_t),
        "profile_z": z.tolist(),
        "profile": prof.tolist(),
    }
