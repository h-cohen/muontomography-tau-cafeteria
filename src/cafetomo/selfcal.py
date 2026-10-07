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


def _objective_grids(cfg: Config, data: FitData) -> tuple[VoxelGrid, VoxelGrid]:
    """Coarse lattice from the prior, padded so a pose anywhere inside the
    search bounds stays inside it, and its twin shifted by half a voxel in x
    and y. A lattice coarser than the features aliases them: the score then
    follows the phase of the lattice under the beams, not the pose. Averaging
    the two phases cancels that to first order."""
    s = cfg.selfcal
    g = auto_grid(replace(cfg.volume, spacing_m=s.spacing_m), cfg.origins(),
                  data.rows.t_reach(), aperture_m=cfg.detector.aperture_m)
    pad = int(np.ceil(s.bounds_m / g.spacing))
    nx, ny, nz = g.shape
    x0, y0 = g.origin[0] - pad * g.spacing, g.origin[1] - pad * g.spacing
    shape = (nx + 2 * pad + 1, ny + 2 * pad + 1, nz)
    half = 0.5 * g.spacing
    return (VoxelGrid(origin=(x0, y0, g.origin[2]), spacing=g.spacing, shape=shape),
            VoxelGrid(origin=(x0 - half, y0 - half, g.origin[2]), spacing=g.spacing,
                      shape=shape))


def _pose_from(theta: np.ndarray, cfg: Config) -> Pose:
    s = cfg.selfcal
    z = cfg.exposure(s.free_pose).pose.z
    if s.baseline_m is None:
        x, y, az = theta
    else:
        bearing, az = theta
        x, y = s.baseline_m * np.cos(bearing), s.baseline_m * np.sin(bearing)
    return Pose(float(x), float(y), z, float(az))


def fit_pose(grid: AnalysisGrid, cfg: Config, live_time: dict[str, float]) -> PoseFit:
    """Minimise the cross-position score over the free position's pose,
    bounded around the config's prior."""
    s = cfg.selfcal
    prior = cfg.exposure(s.free_pose).pose
    nominal = build_fit_data(solve_opacity(grid, cfg, live_time), cfg,
                             poisson_sigma(grid, cfg, live_time))
    vgrids = _objective_grids(cfg, nominal)

    def objective(theta: np.ndarray) -> float:
        c = cfg.with_pose(s.free_pose, _pose_from(theta, cfg))
        data = build_fit_data(solve_opacity(grid, c, live_time), c,
                              poisson_sigma(grid, c, live_time))
        return float(np.mean([cross_position_score(data, c, g) for g in vgrids]))

    if s.baseline_m is None:
        x0 = np.array([prior.x, prior.y, prior.az_deg])
        bounds = [(prior.x - s.bounds_m, prior.x + s.bounds_m),
                  (prior.y - s.bounds_m, prior.y + s.bounds_m),
                  (prior.az_deg - s.bounds_deg, prior.az_deg + s.bounds_deg)]
    else:
        b0 = float(np.arctan2(prior.y, prior.x))
        db = s.bounds_m / s.baseline_m
        x0 = np.array([b0, prior.az_deg])
        bounds = [(b0 - db, b0 + db), (prior.az_deg - s.bounds_deg, prior.az_deg + s.bounds_deg)]
    res = optimize.minimize(objective, x0, method="Powell", bounds=bounds,
                            options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 60})
    return PoseFit(pose=_pose_from(res.x, cfg), objective=float(res.fun),
                   n_eval=int(res.nfev), converged=bool(res.success))


def pose_bootstrap(grid: AnalysisGrid, cfg: Config, live_time: dict[str, float], *,
                   n: int, seed: int) -> dict[str, float]:
    """Spread of the fitted pose over Poisson replicas of every histogram."""
    if n < 2:
        raise ValueError(f"pose bootstrap needs at least 2 replicas for a spread, got {n}")
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        counts = {k: rng.poisson(v).astype(np.int64) for k, v in grid.counts.items()}
        poses.append(fit_pose(AnalysisGrid(edges=grid.edges, counts=counts), cfg,
                              live_time).pose)
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
        "x": p.x, "x_sigma": sigma["x"], "y": p.y, "y_sigma": sigma["y"],
        "az": p.az_deg, "az_sigma": sigma["az_deg"],
        "baseline": float(np.hypot(p.x, p.y)),
        "baseline_fixed": cfg.selfcal.baseline_m is not None,
        "prior_x": prior.x, "prior_y": prior.y,
        "objective": fit.objective, "n_eval": fit.n_eval, "converged": fit.converged,
    }
