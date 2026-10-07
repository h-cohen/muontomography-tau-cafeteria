from dataclasses import replace

import numpy as np
import pytest

from cafetomo.fitdata import FitData, RowIndex
from cafetomo.forward import build_forward_model
from cafetomo.inversion import solve
from cafetomo.raycast import INVERSION_VERSION
from cafetomo.reconstruct import VoxelSolution, solve_voxels

N_SIDE = 12


@pytest.fixture
def cfg2(cfg):
    """Real detector and two positions, a small 1-3 m slab so matrices stay tiny."""
    vol = replace(cfg.volume, z_min_m=1.0, z_max_m=3.0, spacing_m=0.5, n_aperture_sub=2)
    return replace(cfg, volume=vol, reconstruction=replace(
        cfg.reconstruction, n_iter=30, tv_alpha=0.01))


def _data(cfg, seed=0, sigma=None):
    """Both positions over one regular tangent grid, ~70% of rows measured."""
    rng = np.random.default_rng(seed)
    t = (np.arange(N_SIDE) + 0.5) / N_SIDE * 2.0 - 1.0
    sx, sy = (a.ravel() for a in np.meshgrid(t, t))
    n = sx.size
    n_pos = len(cfg.position_ids)
    rows = RowIndex(position_ids=cfg.position_ids,
                    pos_of_row=np.repeat(np.arange(n_pos), n),
                    sx=np.tile(sx, n_pos), sy=np.tile(sy, n_pos),
                    sky_flat=np.tile(np.arange(n), n_pos))
    lam = rng.uniform(0.0, 2.0, n * n_pos)
    live = rng.random(n * n_pos) < 0.7
    sig = np.ones(n * n_pos) if sigma is None else sigma
    return FitData(lam=lam, w=np.where(live, 1.0 / sig**2, 0.0), rows=rows)


def test_solve_produces_a_full_fit_and_one_holdout_per_position(cfg2):
    fits = solve_voxels(_data(cfg2), cfg2)
    assert set(fits) == {"full", "holdout_pos0", "holdout_pos1"}


def test_the_volume_is_nonnegative_and_correctly_shaped(cfg2):
    v = solve_voxels(_data(cfg2), cfg2)["full"]
    assert v.rho.shape == (v.grid.n_voxels,)
    assert v.rho3().shape == v.grid.shape
    assert v.rho.min() >= 0.0


def test_a_holdout_fit_uses_only_its_own_position(cfg2):
    """holdout_pos0 is TRAINED ON pos0 alone: the single-view fit whose
    disagreement with the other view is the cross-validation signal."""
    fits = solve_voxels(_data(cfg2), cfg2)
    assert fits["holdout_pos0"].info["n_rows_used"] < fits["full"].info["n_rows_used"]
    assert (fits["holdout_pos0"].info["n_rows_used"]
            + fits["holdout_pos1"].info["n_rows_used"]
            == fits["full"].info["n_rows_used"])


def test_holdouts_can_be_skipped(cfg2):
    assert set(solve_voxels(_data(cfg2), cfg2, holdouts=False)) == {"full"}


def test_offsets_are_held_at_zero_and_recorded_per_position(cfg2):
    fits = solve_voxels(_data(cfg2), cfg2)
    assert fits["full"].offsets == {"pos0": 0.0, "pos1": 0.0}


def test_an_explicit_grid_pins_the_lattice(cfg2):
    data = _data(cfg2)
    ref = solve_voxels(data, cfg2, holdouts=False)["full"]
    pinned = solve_voxels(data, cfg2, holdouts=False, grid=ref.grid)["full"]
    assert pinned.grid is ref.grid


def test_save_and_load_round_trip(cfg2, tmp_path):
    fits = solve_voxels(_data(cfg2), cfg2)
    p = tmp_path / "out" / "volume_full.npz"
    fits["full"].save(p)

    back = VoxelSolution.load(p)
    np.testing.assert_allclose(back.rho, fits["full"].rho, rtol=1e-6)
    assert back.grid == fits["full"].grid
    assert back.offsets == pytest.approx(fits["full"].offsets)
    assert back.position_ids == fits["full"].position_ids


def test_the_saved_volume_records_the_inversion_version(cfg2, tmp_path):
    p = tmp_path / "v.npz"
    solve_voxels(_data(cfg2), cfg2, holdouts=False)["full"].save(p)
    assert VoxelSolution.load(p).version == INVERSION_VERSION


def test_sigma_weights_are_threaded_through(cfg2):
    """A position whose sigma is enormous should barely move the fit."""
    n = N_SIDE * N_SIDE
    sigma = np.concatenate([np.full(n, 1e6), np.full(n, 0.1)])
    weighted = solve_voxels(_data(cfg2, sigma=sigma), cfg2, holdouts=False)["full"]
    uniform = solve_voxels(_data(cfg2), cfg2, holdouts=False)["full"]
    assert not np.allclose(weighted.rho, uniform.rho)


def test_the_volume_reproduces_its_own_measurements_better_than_a_zero_volume(cfg2):
    """The weakest honest fidelity claim, and the one that catches a matrix
    whose rays point the wrong way: the fit must beat doing nothing."""
    data = _data(cfg2)
    v = solve_voxels(data, cfg2, holdouts=False)["full"]
    fwd = build_forward_model(data.rows, cfg2, cache_dir=None)
    live = data.w > 0
    pred = fwd.predict(v.rho, v.offsets)
    flat = np.mean((data.lam[live] - np.mean(data.lam[live])) ** 2)
    assert np.mean((data.lam[live] - pred[live]) ** 2) < flat


def _flat_ceiling(cfg):
    """A flat slab at 6.6-7.4 m seen by both positions over a regular sky grid."""
    vol = replace(cfg.volume, spacing_m=0.5)
    cfg = replace(cfg, volume=vol, reconstruction=replace(
        cfg.reconstruction, n_iter=150, tv_alpha=0.01, tv_z_weight=0.5))
    n = 18
    t = (np.arange(n) + 0.5) / n * 1.8 - 0.9
    sx, sy = (a.ravel() for a in np.meshgrid(t, t, indexing="ij"))
    n_pos = len(cfg.position_ids)
    rows = RowIndex(position_ids=cfg.position_ids,
                    pos_of_row=np.repeat(np.arange(n_pos), sx.size),
                    sx=np.tile(sx, n_pos), sy=np.tile(sy, n_pos),
                    sky_flat=np.tile(np.arange(sx.size), n_pos))
    fwd = build_forward_model(rows, cfg, cache_dir=None)
    zc = fwd.grid.axis_centers(2)
    slab = (zc > 6.6) & (zc < 7.4)
    truth = np.zeros(fwd.grid.shape)
    truth[:, :, slab] = 0.125
    data = FitData(lam=fwd.A @ truth.ravel(), w=np.ones(rows.n_rows), rows=rows)
    covered = np.asarray(abs(fwd.A).sum(axis=0)).ravel().reshape(fwd.grid.shape)
    seen = covered[:, :, slab].sum(2) > 0
    column = float(truth[0, 0].sum() * fwd.grid.spacing)
    return cfg, fwd, data, seen, column


def _columns(fwd, x, seen):
    col = x.reshape(fwd.grid.shape).sum(2) * fwd.grid.spacing
    nx, ny = col.shape
    centre = col[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2].mean()
    return centre, np.percentile(col[seen], 95)


def test_measured_zero_point_recovers_a_flat_ceiling_flat(cfg):
    cfg, fwd, data, seen, column = _flat_ceiling(cfg)
    x, info = solve(fwd, data, cfg.reconstruction, fit_offsets=False)
    centre, rim = _columns(fwd, x, seen)
    assert info["offsets"] == {"pos0": 0.0, "pos1": 0.0}
    assert centre == pytest.approx(column, rel=0.1)
    assert rim == pytest.approx(centre, rel=0.15)


def test_free_offsets_hide_the_ceiling_and_brighten_the_rim(cfg):
    """The degeneracy a measured zero point removes: the free fit puts the
    slab's constant opacity into c_p and only its oblique excess reaches the
    voxels, in the outer shell."""
    cfg, fwd, data, seen, column = _flat_ceiling(cfg)
    x, info = solve(fwd, data, cfg.reconstruction, fit_offsets=True)
    centre, rim = _columns(fwd, x, seen)
    assert min(info["offsets"].values()) > 0.4 * column
    assert centre < 0.3 * column
    assert rim > 1.5 * centre
