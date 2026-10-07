"""Self-calibration of the second detector position.

Only pos0 is the frame origin; pos1's floor position was never surveyed. Its
pose is fitted by symmetric cross-position validation: reconstruct the room
from ONE position on a coarse lattice and score how well it predicts the
OTHER. Only at the true relative pose can one volume explain both views.

Angles alone fix the pose only up to scale (scaling the room and the baseline
together leaves every direction unchanged). With `selfcal.baseline_m` set, the
separation is the external length and only its bearing and the azimuth are
fitted; without it the separation is fitted too and the scale rides on the
prior (see beams.scale_closure).
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize

from cafetomo.angular import AnalysisGrid
from cafetomo.config import Config, Pose
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.inversion import solve
from cafetomo.opacity import build_fit_data, poisson_sigma, solve_opacity
from cafetomo.voxels import VoxelGrid, auto_grid


@dataclass(frozen=True)
class PoseFit:
    pose: Pose
    objective: float
    n_eval: int
    converged: bool


def cross_position_score(data: FitData, cfg: Config, grid: VoxelGrid) -> float:
    """Mean weighted residual of each position predicted from the other alone,
    with a free offset on the held-out view."""
    fwd = build_forward_model(data.rows, cfg, grid=grid)
    rc = replace(cfg.reconstruction, algorithm="sirt", n_iter=cfg.selfcal.n_iter)
    scores = []
    for train in data.rows.position_ids:
        x, _ = solve(fwd, data.restricted(data.rows.mask_for(train)), rc, fit_offsets=True)
        pred = fwd.predict(x)
        for test in data.rows.position_ids:
            if test == train:
                continue
            sel = data.rows.mask_for(test) & (data.w > 0)
            w = data.w[sel]
            r = data.lam[sel] - pred[sel]
            r = r - (w * r).sum() / w.sum()
            scores.append(float(np.mean(w * r**2)))
    return float(np.mean(scores))


def _search_space(cfg: Config) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Start point and bounds of the pose search, around the config's prior:
    (x, y, az) with the baseline free, (bearing, az) with it fixed."""
    s = cfg.selfcal
    prior = cfg.exposure(s.free_pose).pose
    az = (prior.az_deg - s.bounds_deg, prior.az_deg + s.bounds_deg)
    if s.baseline_m is None:
        return (
            np.array([prior.x, prior.y, prior.az_deg]),
            [
                (prior.x - s.bounds_m, prior.x + s.bounds_m),
                (prior.y - s.bounds_m, prior.y + s.bounds_m),
                az,
            ],
        )
    b0 = float(np.arctan2(prior.y, prior.x))
    db = s.bounds_m / s.baseline_m
    return np.array([b0, prior.az_deg]), [(b0 - db, b0 + db), az]


def _candidate_xy(cfg: Config) -> np.ndarray:
    """[k, 2] floor positions whose bounding box holds every candidate of the
    search: the corners of the box, or the ends and axis extremes of the arc."""
    s = cfg.selfcal
    _, bounds = _search_space(cfg)
    if s.baseline_m is None:
        (x0, x1), (y0, y1), _ = bounds
        return np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
    lo, hi = bounds[0]
    quarter = np.pi / 2
    axes = quarter * np.arange(np.ceil(lo / quarter), np.floor(hi / quarter) + 1)
    bearings = np.concatenate([[lo, hi], axes])
    return s.baseline_m * np.stack([np.cos(bearings), np.sin(bearings)], axis=-1)


def _objective_grids(cfg: Config, data: FitData) -> tuple[VoxelGrid, VoxelGrid]:
    """Coarse lattice holding the full ray footprint of every candidate pose,
    and its twin shifted by half a voxel in x and y.

    The footprint is taken from every fixed position plus the extremes of the
    free position's search region. Rotating the detector by up to `bounds_deg`
    carries its corner rows from tangent t to t (cos d + sin d), capped at
    `opacity.max_tan`, beyond which no row is kept. A lattice coarser than the
    features aliases them: the score then follows the phase of the lattice
    under the beams, not the pose. Averaging the two phases suppresses that."""
    s = cfg.selfcal
    z = cfg.exposure(s.free_pose).pose.z
    origins = {pid: o for pid, o in cfg.origins().items() if pid != s.free_pose}
    for k, (x, y) in enumerate(_candidate_xy(cfg)):
        origins[f"{s.free_pose}@{k}"] = (float(x), float(y), z)
    d = np.radians(min(s.bounds_deg, 45.0))
    reach = min(data.rows.t_reach() * (np.cos(d) + np.sin(d)), cfg.opacity.max_tan)
    g = auto_grid(
        replace(cfg.volume, spacing_m=s.spacing_m, xy_m=None),
        origins,
        reach,
        aperture_m=cfg.detector.aperture_m,
    )
    nx, ny, nz = g.shape
    shape = (nx + 1, ny + 1, nz)
    half = 0.5 * g.spacing
    return (
        VoxelGrid(origin=g.origin, spacing=g.spacing, shape=shape),
        VoxelGrid(
            origin=(g.origin[0] - half, g.origin[1] - half, g.origin[2]),
            spacing=g.spacing,
            shape=shape,
        ),
    )


def _pose_from(theta: np.ndarray, cfg: Config) -> Pose:
    s = cfg.selfcal
    z = cfg.exposure(s.free_pose).pose.z
    if s.baseline_m is None:
        x, y, az = theta
    else:
        bearing, az = theta
        x, y = s.baseline_m * np.cos(bearing), s.baseline_m * np.sin(bearing)
    return Pose(float(x), float(y), z, float(az))


def _scan_start(objective, cfg: Config) -> np.ndarray:
    """Start point of the pose search. With the separation fixed, the prior's
    bearing can be far from the true one (the prior pose is along +x), and
    Powell from there can stall in a local minimum:
    the bearing is first scanned over its bounds in `scan_deg` steps (ends
    included) at the prior azimuth, and the best one starts the search. The
    free-baseline search starts at the prior."""
    x0, bounds = _search_space(cfg)
    if cfg.selfcal.baseline_m is None:
        return x0
    lo, hi = bounds[0]
    step = np.radians(cfg.selfcal.scan_deg)
    if step <= 0:
        raise ValueError(f"selfcal.scan_deg must be positive, got {cfg.selfcal.scan_deg}")
    bearings = np.append(np.arange(lo, hi, step), hi)
    scores = [objective(np.array([b, x0[1]])) for b in bearings]
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"selfcal bearing scan hit a non-finite objective: {scores}")
    return np.array([bearings[int(np.argmin(scores))], x0[1]])


def fit_pose(grid: AnalysisGrid, cfg: Config, live_time: dict[str, float]) -> PoseFit:
    """Minimise the cross-position score over the free position's pose,
    bounded around the config's prior.

    Limits of the fitted pose:
    - The component along the beams (y) is pinned only by the beam ends and
      weakly: on a beam-ceiling phantom the free-baseline fit gave y = 0.54 m
      against a true 0.70 m while x was right to 0.02 m.
    - Azimuth steps below about 1 degree are invisible: every detector bin is
      scattered to the nearest sky bin, so a small rotation often leaves the
      row set unchanged and the score flat. Any az within such a flat stretch
      is as good as any other.
    - The score is not zero at the true pose (one view predicts the other
      only through a min-norm reconstruction), and that geometric part can
      pull the minimum; nothing here measures that pull."""
    s = cfg.selfcal
    nominal = build_fit_data(
        solve_opacity(grid, cfg, live_time), cfg, poisson_sigma(grid, cfg, live_time)
    )
    vgrids = _objective_grids(cfg, nominal)

    def objective(theta: np.ndarray) -> float:
        c = cfg.with_pose(s.free_pose, _pose_from(theta, cfg))
        data = build_fit_data(
            solve_opacity(grid, c, live_time), c, poisson_sigma(grid, c, live_time)
        )
        return float(np.mean([cross_position_score(data, c, g) for g in vgrids]))

    _, bounds = _search_space(cfg)
    res = optimize.minimize(
        objective,
        _scan_start(objective, cfg),
        method="Powell",
        bounds=bounds,
        options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 60},
    )
    if not res.success or not np.isfinite(res.fun) or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"pose fit failed: {res.message}; objective={res.fun}, pose={res.x}")
    return PoseFit(
        pose=_pose_from(res.x, cfg),
        objective=float(res.fun),
        n_eval=int(res.nfev),
        converged=bool(res.success),
    )


def pose_bootstrap(
    grid: AnalysisGrid, cfg: Config, live_time: dict[str, float], *, n: int, seed: int
) -> dict[str, float]:
    """Spread of the fitted pose over Poisson replicas of every histogram.

    This is the counting-statistics spread only. It inherits the limits listed
    in `fit_pose`: the y spread is wide along the beams, the az spread partly
    reflects the sky binning rather than statistics, and the geometric bias of
    the cross-validation score is identical in every replica, so it does not
    appear here."""
    if n < 2:
        raise ValueError(f"pose bootstrap needs at least 2 replicas for a spread, got {n}")
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        counts = {k: rng.poisson(v).astype(np.int64) for k, v in grid.counts.items()}
        poses.append(fit_pose(AnalysisGrid(edges=grid.edges, counts=counts), cfg, live_time).pose)
    arr = np.array([[p.x, p.y, p.az_deg] for p in poses])
    sd = arr.std(axis=0, ddof=1)
    return {"x": float(sd[0]), "y": float(sd[1]), "az_deg": float(sd[2])}


def pose_result(fit: PoseFit, sigma: dict[str, float], cfg: Config) -> dict:
    """Content of results/pose.json; `load_config(pose_file=...)` reads `free_pose` and `pose`."""
    p = fit.pose
    prior = cfg.exposure(cfg.selfcal.free_pose).pose
    return {
        "free_pose": cfg.selfcal.free_pose,
        "pose": {"x": p.x, "y": p.y, "z": p.z, "az_deg": p.az_deg},
        "x": p.x,
        "x_sigma": sigma["x"],
        "y": p.y,
        "y_sigma": sigma["y"],
        "az": p.az_deg,
        "az_sigma": sigma["az_deg"],
        "baseline": float(np.hypot(p.x, p.y)),
        "baseline_fixed": cfg.selfcal.baseline_m is not None,
        "prior_x": prior.x,
        "prior_y": prior.y,
        "objective": fit.objective,
        "n_eval": fit.n_eval,
        "converged": fit.converged,
    }
