import numpy as np
import pytest

from cafetomo.autofocus import (
    FocusScan,
    cv_height_scan,
    layer_cv_score,
    layer_grid,
    layer_grids,
    parabola_min,
)
from cafetomo.config import Pose
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.phantom import beam_ceiling, phantom_data, sky_rows
from cafetomo.voxels import VoxelGrid


def test_parabola_min_subgrid():
    zs = np.array([6.8, 7.0, 7.2])
    assert parabola_min(zs, (zs - 7.05) ** 2) == pytest.approx(7.05, abs=1e-9)


def test_parabola_min_edge_returns_grid_point():
    zs = np.array([6.8, 7.0, 7.2])
    assert parabola_min(zs, np.array([1.0, 2.0, 3.0])) == 6.8


def test_parabola_min_nonuniform_triple():
    zs = np.array([6.8, 7.0, 7.1])
    assert parabola_min(zs, (zs - 7.03) ** 2) == pytest.approx(7.03, abs=1e-9)


def test_parabola_min_flat_right_neighbour_splits_the_tie():
    zs = np.array([6.8, 7.0, 7.1])
    assert parabola_min(zs, np.array([3.0, 1.0, 1.0])) == pytest.approx(7.05, abs=1e-9)


def test_focus_scan_to_json():
    scan = FocusScan(zs=np.array([6.9, 7.0]), scores=np.array([2.0, 1.0]), z_best=7.0)
    assert scan.to_json() == {"z": 7.0, "scan_z": [6.9, 7.0], "scan_score": [2.0, 1.0]}


def test_layer_grids_are_half_voxel_twins(cfg):
    a, b = layer_grids(cfg, 0.9, 7.0)
    half = 0.5 * a.spacing
    assert a.spacing == b.spacing == cfg.autofocus.spacing_m
    assert a.shape == b.shape
    assert b.origin[0] == pytest.approx(a.origin[0] - half)
    assert b.origin[1] == pytest.approx(a.origin[1] - half)
    assert b.origin[2] == a.origin[2]
    assert a.origin[2] == pytest.approx(7.0 - cfg.autofocus.layer_thickness_m / 2)
    footprint = layer_grid(cfg, 0.9, 7.0)
    for g in (a, b):
        for axis in (0, 1, 2):
            lo, hi = g.extent(axis)
            assert lo <= footprint.extent(axis)[0] + 1e-9
            assert hi >= footprint.extent(axis)[1] - 1e-9


@pytest.mark.parametrize("empty", ["pos0", "pos1"])
def test_layer_cv_score_rejects_position_without_rows(cfg, empty):
    rows = sky_rows(cfg.position_ids, 0.9, 8)
    w = np.where(rows.mask_for(empty), 0.0, 1.0)
    data = FitData(lam=np.zeros(rows.n_rows), w=w, rows=rows)
    with pytest.raises(ValueError, match=empty):
        layer_cv_score(data, cfg, 7.0)


@pytest.mark.slow
def test_recovers_phantom_height(cfg):
    cfg = cfg.with_pose("pos1", Pose(1.78, 0.72, 0.0, 0.0))
    rows = sky_rows(cfg.position_ids, 0.9, 36)
    g = VoxelGrid(origin=(-6.0, -6.0, 6.0), spacing=0.1, shape=(140, 120, 26))
    truth = beam_ceiling(
        g, xs=(-1.7, 0.0, 1.7, 3.4), z0=7.0, w=0.3, h=0.4, kappa=(1.5,) * 4, y_extent=(-4.0, 4.0)
    )
    fwd = build_forward_model(rows, cfg, grid=g)
    like = FitData(lam=np.zeros(rows.n_rows), w=np.full(rows.n_rows, 1 / 0.03**2), rows=rows)
    data = phantom_data(fwd, truth, like, np.random.default_rng(0))
    scan = cv_height_scan(data, cfg)
    assert scan.z_best == pytest.approx(7.2, abs=0.25)  # layer centre: z0 + h/2
