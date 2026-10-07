from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from cafetomo.config import Reconstruction
from cafetomo.fitdata import FitData, RowIndex
from cafetomo.forward import ForwardModel, build_forward_model
from cafetomo.inversion import sirt, sirt_tv, solve
from cafetomo.phantom import sky_rows
from cafetomo.voxels import VoxelGrid


def _rc(**kw):
    """The unregularised baseline: damping off unless a test turns it on."""
    return Reconstruction(**{"coverage_damping": 0.0, **kw})


def _toy(n_rows=40, shape=(4, 4, 2), seed=0):
    """A small well-posed random system with a known nonnegative solution."""
    rng = np.random.default_rng(seed)
    n_vox = int(np.prod(shape))
    A = sparse.csr_matrix(rng.random((n_rows, n_vox)) * (rng.random((n_rows, n_vox)) < 0.5))
    truth = np.abs(rng.normal(size=n_vox))
    rows = RowIndex(
        position_ids=("pos0", "pos1"),
        pos_of_row=np.arange(n_rows) % 2,
        sx=np.zeros(n_rows),
        sy=np.zeros(n_rows),
        sky_flat=np.arange(n_rows),
    )
    fwd = ForwardModel(A=A, grid=VoxelGrid(origin=(0, 0, 0), spacing=1.0, shape=shape), rows=rows)
    lam = A @ truth
    return fwd, truth, FitData(lam=lam, w=np.ones(n_rows), rows=rows)


def test_sirt_recovers_a_noiseless_solution():
    fwd, truth, data = _toy()
    rc = _rc(algorithm="sirt", n_iter=400, chi2_target=1e-12)
    x, info = sirt(fwd, data, rc, fit_offsets=True)
    assert np.corrcoef(x, truth)[0, 1] > 0.95
    assert info["chi2_history"][-1] < info["chi2_history"][0]


def test_sirt_never_returns_a_negative_voxel():
    fwd, truth, data = _toy()
    rng = np.random.default_rng(1)
    noisy = FitData(
        lam=data.lam + rng.normal(scale=0.5, size=data.lam.size), w=data.w, rows=data.rows
    )
    x, _ = sirt(
        fwd,
        noisy,
        _rc(algorithm="sirt", n_iter=200, chi2_target=1e-12, nonneg=True),
        fit_offsets=True,
    )
    assert x.min() >= 0.0


def test_sirt_stops_early_at_the_discrepancy_target():
    fwd, truth, data = _toy()
    x, info = sirt(
        fwd, data, _rc(algorithm="sirt", n_iter=5000, chi2_target=1e-2), fit_offsets=True
    )
    assert info["n_iter_used"] < 5000


def test_zero_weight_rows_do_not_influence_the_fit():
    fwd, truth, data = _toy()
    poisoned = FitData(lam=data.lam.copy(), w=data.w.copy(), rows=data.rows)
    poisoned.lam[:5] = 1e6
    kept = np.ones(data.rows.n_rows, dtype=bool)
    kept[:5] = False

    rc = _rc(algorithm="sirt", n_iter=300, chi2_target=1e-12)
    a, _ = sirt(fwd, data.restricted(kept), rc, fit_offsets=True)
    b, _ = sirt(fwd, poisoned.restricted(kept), rc, fit_offsets=True)
    np.testing.assert_allclose(a, b)


def test_offsets_absorb_a_constant_shift_per_position():
    """With a per-position gauge, lambda has a free additive constant per
    position, so an overall level is not measured.

    The comparison is between the shifted and unshifted fits, not against zero.
    The unshifted fit already carries nonzero offsets absorbing the mean of
    A @ truth, so the absolute offsets say nothing; only their DIFFERENCE should
    track the injected shift.
    """
    fwd, truth, data = _toy()
    shift = np.where(data.rows.pos_of_row == 0, 0.7, -0.4)
    shifted = FitData(lam=data.lam + shift, w=data.w, rows=data.rows)

    rc = _rc(algorithm="sirt", n_iter=400, chi2_target=1e-12)
    x0, base = sirt(fwd, data, rc, fit_offsets=True)
    x1, moved = sirt(fwd, shifted, rc, fit_offsets=True)

    assert moved["offsets"]["pos0"] - base["offsets"]["pos0"] == pytest.approx(0.7, abs=0.15)
    assert moved["offsets"]["pos1"] - base["offsets"]["pos1"] == pytest.approx(-0.4, abs=0.15)
    assert np.corrcoef(x1, truth)[0, 1] > 0.9


def test_tv_denoises_a_piecewise_constant_volume_better_than_plain_sirt():
    rng = np.random.default_rng(3)
    shape = (8, 8, 4)
    n_vox = int(np.prod(shape))
    truth3 = np.zeros(shape)
    truth3[2:6, 2:6, 1:3] = 1.0
    truth = truth3.ravel()

    A = sparse.csr_matrix(rng.random((300, n_vox)) * (rng.random((300, n_vox)) < 0.3))
    rows = RowIndex(
        position_ids=("pos0",),
        pos_of_row=np.zeros(300, dtype=int),
        sx=np.zeros(300),
        sy=np.zeros(300),
        sky_flat=np.arange(300),
    )
    fwd = ForwardModel(A=A, grid=VoxelGrid(origin=(0, 0, 0), spacing=1.0, shape=shape), rows=rows)
    lam = A @ truth + rng.normal(scale=0.4, size=300)
    data = FitData(lam=lam, w=np.ones(300), rows=rows)

    plain, _ = sirt(
        fwd, data, _rc(algorithm="sirt", n_iter=300, chi2_target=1e-12), fit_offsets=True
    )
    tv, _ = sirt_tv(
        fwd,
        data,
        _rc(algorithm="tv", n_iter=300, tv_alpha=0.001, tv_z_weight=0.5),
        fit_offsets=True,
    )
    assert np.linalg.norm(tv - truth) < np.linalg.norm(plain - truth)


def test_tv_returns_the_best_iterate_not_the_last():
    fwd, truth, data = _toy()
    x, info = sirt_tv(fwd, data, _rc(algorithm="tv", n_iter=120, tv_alpha=0.01), fit_offsets=True)
    assert info["best_chi2"] <= min(info["chi2_history"])


def test_tv_with_zero_alpha_tracks_plain_sirt():
    """With tv_alpha = 0 the proximal step is just the nonnegativity clip, so the
    two solvers follow the same trajectory.

    They do NOT end on the same array: chi2 is evaluated before each update, so
    sirt_tv's best iterate is at most the one after n_iter - 1 updates while sirt
    returns the one after n_iter. The gap is one sweep, by construction.
    """
    fwd, truth, data = _toy()
    a, _ = sirt(fwd, data, _rc(algorithm="sirt", n_iter=50, chi2_target=-1.0), fit_offsets=True)
    b, _ = sirt_tv(fwd, data, _rc(algorithm="tv", n_iter=50, tv_alpha=0.0), fit_offsets=True)
    assert np.corrcoef(a, b)[0, 1] > 0.999
    assert np.linalg.norm(b - a) < 0.05 * np.linalg.norm(a)


def test_solve_dispatches_on_the_algorithm_name():
    fwd, truth, data = _toy()
    x, _ = solve(fwd, data, _rc(algorithm="tv", n_iter=20), fit_offsets=True)
    assert x.shape == (fwd.grid.n_voxels,)
    with pytest.raises(ValueError, match="unknown algorithm"):
        solve(fwd, data, _rc(algorithm="mlem", n_iter=5), fit_offsets=True)


def test_an_all_zero_weight_fit_returns_zeros_rather_than_dividing_by_zero():
    fwd, truth, data = _toy()
    dead = data.restricted(np.zeros(data.rows.n_rows, dtype=bool))
    x, info = sirt(fwd, dead, _rc(algorithm="sirt", n_iter=10, chi2_target=-1.0), fit_offsets=True)
    assert np.all(x == 0.0)
    assert np.all(np.isfinite(list(info["offsets"].values())))


def _noisy_ceiling(cfg, seed=0):
    """Flat slab 0.8 m thick at the top of the volume, noise growing toward the
    acceptance edge."""
    rows = sky_rows(cfg.position_ids, 0.9, 36)
    cfg = replace(cfg, volume=replace(cfg.volume, spacing_m=0.3))
    fwd = build_forward_model(rows, cfg, cache_dir=None)
    zc = fwd.grid.axis_centers(2)
    truth = np.zeros(fwd.grid.shape)
    truth[:, :, (zc > 6.6) & (zc < 7.4)] = 0.1
    t2 = rows.sx**2 + rows.sy**2
    sig = 0.007 * (1 + 8 * t2**2)
    lam = fwd.A @ truth.ravel() + np.random.default_rng(seed).normal(size=rows.n_rows) * sig
    rays = np.diff(fwd.A.tocsc().indptr)
    return fwd, FitData(lam=lam, w=1 / sig**2, rows=rows), rays, cfg.reconstruction


def test_coverage_damping_removes_the_noise_shell_without_losing_the_fit(cfg):
    """Voxels crossed by one or two rays are those rays' private unknowns; under
    non-negativity SIRT parks their noise there as a bright outer shell. The
    damping (strength proportional to 1/coverage) must remove it at the same
    fit quality -- and the undamped solve must show it, or the gate tests nothing."""
    fwd, data, rays, rc = _noisy_ceiling(cfg)
    low, cov = (rays >= 1) & (rays < 3), rays >= 30

    x0, info0 = sirt_tv(fwd, data, replace(rc, coverage_damping=0.0), fit_offsets=False)
    x1, info1 = sirt_tv(fwd, data, replace(rc, coverage_damping=0.05), fit_offsets=False)
    shell0 = x0[low].mean() / x0[cov].mean()
    shell1 = x1[low].mean() / x1[cov].mean()
    assert shell0 > 2.0  # the artifact exists without damping
    assert shell1 < 0.5  # and is gone with it
    assert info1["best_chi2"] <= info0["best_chi2"] + 0.15
