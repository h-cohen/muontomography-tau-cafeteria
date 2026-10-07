from dataclasses import replace

import numpy as np
import pytest

from cafetomo.angular import AnalysisGrid
from cafetomo.config import Pose
from cafetomo.fitdata import FitData
from cafetomo.phantom import sky_rows
from cafetomo.selfcal import (
    PoseFit,
    _candidate_xy,
    _objective_grids,
    _pose_from,
    fit_pose,
    pose_bootstrap,
    pose_result,
)


def _campaign(cfg, true_pose: Pose):
    """Counts from a beam ceiling seen from pos0 and a pos1 at `true_pose`."""
    from cafetomo.forward import build_forward_model
    from cafetomo.phantom import beam_ceiling, sky_rows
    from cafetomo.voxels import VoxelGrid

    c = cfg.with_pose("pos1", true_pose)
    n_det, t = 50, 1.25
    edges = np.linspace(-t, t, n_det + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    tx, ty = np.meshgrid(cen, cen, indexing="ij")
    acc = c.detector.acceptance(tx, ty)
    sky_counts = np.round(4e5 * acc / acc.sum()).astype(np.int64) * 10
    g = VoxelGrid(origin=(-6.0, -6.0, 6.5), spacing=0.1, shape=(140, 120, 20))
    truth = beam_ceiling(g, xs=(-1.7, 0.0, 1.7, 3.4), z0=7.0, w=0.3, h=0.8,
                         kappa=(1.5,) * 4, y_extent=(-4.0, 4.0))
    counts = {"SKY": sky_counts}
    rng = np.random.default_rng(1)
    for pid in ("pos0", "pos1"):
        rows = sky_rows((pid,), t, n_det)
        rows = replace(rows, sx=tx.ravel(), sy=ty.ravel())
        fwd = build_forward_model(rows, c, grid=g)
        lam = fwd.predict(truth).reshape(n_det, n_det)
        counts[pid] = rng.poisson(0.5 * sky_counts * np.exp(-lam))
    return AnalysisGrid(edges=edges, counts=counts), {"SKY": 2.0, "pos0": 1.0, "pos1": 1.0}


def test_pose_from_free_baseline(cfg):
    p = _pose_from(np.array([1.6, -0.3, 2.5]), cfg)
    assert (p.x, p.y, p.az_deg) == (1.6, -0.3, 2.5)
    assert p.z == cfg.exposure(cfg.selfcal.free_pose).pose.z


def test_pose_from_fixed_baseline_keeps_the_separation(cfg):
    c = replace(cfg, selfcal=replace(cfg.selfcal, baseline_m=2.0))
    p = _pose_from(np.array([np.pi / 6, -4.0]), c)
    assert np.hypot(p.x, p.y) == pytest.approx(2.0)
    assert p.x == pytest.approx(np.sqrt(3.0))
    assert p.y == pytest.approx(1.0)
    assert p.az_deg == -4.0


@pytest.mark.parametrize("baseline", [None, 2.2])
def test_objective_grids_hold_every_candidate_footprint(cfg, baseline):
    c = replace(cfg, selfcal=replace(cfg.selfcal, baseline_m=baseline))
    rows = sky_rows(c.position_ids, 0.8, 16)
    data = FitData(lam=np.zeros(rows.n_rows), w=np.ones(rows.n_rows), rows=rows)
    t = rows.t_reach()
    corners = np.array([[sx, sy, 1.0] for sx in (-t, t) for sy in (-t, t)])
    half = 0.5 * c.detector.aperture_m
    z_top = c.volume.z_max_m
    s = c.selfcal
    prior = c.exposure(s.free_pose).pose
    for g in _objective_grids(c, data):
        (x0, x1), (y0, y1), (z0, z1) = (g.extent(a) for a in range(3))
        assert z0 <= c.volume.z_min_m and z1 >= z_top
        for x, y in _candidate_xy(c):
            for az in (prior.az_deg - s.bounds_deg, prior.az_deg + s.bounds_deg):
                d = corners @ Pose(x, y, prior.z, az).rotation().T
                sx = np.clip(d[:, 0] / d[:, 2], -c.opacity.max_tan, c.opacity.max_tan)
                sy = np.clip(d[:, 1] / d[:, 2], -c.opacity.max_tan, c.opacity.max_tan)
                rx, ry = x + sx * (z_top - prior.z), y + sy * (z_top - prior.z)
                assert x0 <= (rx - half).min() and (rx + half).max() <= x1
                assert y0 <= (ry - half).min() and (ry + half).max() <= y1


def test_candidate_xy_spans_the_baseline_arc(cfg):
    c = replace(cfg, selfcal=replace(cfg.selfcal, baseline_m=2.2, bounds_m=1.0))
    xy = _candidate_xy(c)
    assert np.hypot(xy[:, 0], xy[:, 1]) == pytest.approx(np.full(len(xy), 2.2))
    assert xy[:, 0].max() == pytest.approx(2.2)


def test_pose_result_keys(cfg):
    fit = PoseFit(pose=Pose(1.8, 0.6, 0.0, 1.5), objective=0.9, n_eval=40, converged=True)
    out = pose_result(fit, {"x": 0.02, "y": 0.03, "az_deg": 0.4}, cfg)
    assert set(out) == {"free_pose", "pose", "x", "x_sigma", "y", "y_sigma", "az",
                        "az_sigma", "baseline", "baseline_fixed", "prior_x", "prior_y",
                        "objective", "n_eval", "converged"}
    assert out["free_pose"] == "pos1"
    assert out["pose"] == {"x": 1.8, "y": 0.6, "z": 0.0, "az_deg": 1.5}
    assert (out["x_sigma"], out["y_sigma"], out["az_sigma"]) == (0.02, 0.03, 0.4)
    assert out["baseline"] == pytest.approx(np.hypot(1.8, 0.6))
    assert out["baseline_fixed"] is False
    assert (out["prior_x"], out["prior_y"]) == (1.75, 0.0)


def test_pose_bootstrap_needs_a_spread(cfg):
    with pytest.raises(ValueError, match="at least 2"):
        pose_bootstrap(None, cfg, {}, n=1, seed=0)


@pytest.mark.slow
def test_recovers_injected_offset(cfg):
    truth = Pose(1.80, 0.70, 0.0, 0.0)
    grid, live = _campaign(cfg, truth)
    prior = cfg.with_pose("pos1", Pose(1.75, 0.45, 0.0, 0.0))
    fixed = replace(prior.selfcal, baseline_m=float(np.hypot(truth.x, truth.y)))
    fit = fit_pose(grid, replace(prior, selfcal=fixed), live)
    assert fit.pose.x == pytest.approx(truth.x, abs=0.1)
    assert fit.pose.y == pytest.approx(truth.y, abs=0.1)
