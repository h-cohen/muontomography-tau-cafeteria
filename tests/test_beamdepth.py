from dataclasses import replace

import numpy as np
import pytest
from scipy import optimize

from cafetomo import beamdepth
from cafetomo.beamdepth import (
    BeamDepthFit,
    beam_design,
    box_path_lengths,
    estimate_beam_depth,
    fit_beam_depth,
    zprofile_depth,
)
from cafetomo.fitdata import FitData
from cafetomo.phantom import beam_ceiling, sky_rows
from cafetomo.reconstruct import VoxelSolution
from cafetomo.voxels import VoxelGrid
from phantoms import beam_phantom, phantom_sky


def test_box_path_vertical_and_oblique():
    starts = np.zeros((2, 3))
    dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]) / np.array([[1.0], [np.sqrt(2)]])
    L = box_path_lengths(starts, dirs, np.array([-0.5, -1, 7.0]), np.array([0.5, 1, 8.0]))
    assert L[0] == pytest.approx(1.0)
    assert L[1] == pytest.approx(0.0)  # passes x = 7..8 at z = 7..8: misses the box


def test_box_path_clips_corner():
    d = np.array([[0.1, 0.0, 1.0]]) / np.linalg.norm([0.1, 0, 1])
    L = box_path_lengths(np.zeros((1, 3)), d, np.array([0.65, -1, 7.0]), np.array([0.95, 1, 9.0]))
    # enters at z = 7 (x = .70), exits through x = .95 at z = 9.5 -> clipped by z = 9
    assert L[0] == pytest.approx(2.0 * np.sqrt(1.01), rel=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("h_true", [0.6, 1.25])
def test_recovers_depth(cfg, h_true):
    c, data = beam_phantom(cfg, h_true, np.random.default_rng(3))
    fit = fit_beam_depth(data, c, xs_init=(-1.7, 0.0, 1.7, 3.4), z0_init=7.1)
    assert not fit.at_bound
    assert fit.h == pytest.approx(h_true, abs=0.15)
    assert fit.z0 == pytest.approx(7.0, abs=0.15)


def test_at_bound_flag(cfg):
    c, data = beam_phantom(cfg, 0.6, np.random.default_rng(4))
    tight = replace(c, beamdepth=replace(c.beamdepth, h_max_m=0.3, h_init_m=0.2))
    assert fit_beam_depth(data, tight, xs_init=(-1.7, 0.0, 1.7, 3.4), z0_init=7.1).at_bound


def test_zprofile_measures_column_extent(cfg):
    g = VoxelGrid(origin=(-3.0, -3.0, 5.0), spacing=0.1, shape=(60, 60, 50))
    rho = beam_ceiling(
        g, xs=(0.0,), z0=7.0, w=0.3, h=1.2, kappa=(1.0,), y_extent=(-3, 3), slab_thickness=0.0
    )
    sol = VoxelSolution(rho=rho.ravel(), grid=g, offsets={}, position_ids=("pos0", "pos1"))
    zp = zprofile_depth(sol, cfg, xs=(0.0,), w=0.3, z_ref=7.0)
    assert zp["bottom"] == pytest.approx(7.0, abs=0.1)
    assert zp["top"] == pytest.approx(8.2, abs=0.1)


def _analytic(cfg):
    """Noiseless lambda from the fit's own box model: a cheap stand-in for a phantom."""
    rows = sky_rows(cfg.position_ids, 0.9, 24)
    xs = (-1.7, 0.0, 1.7)
    L = beam_design(
        rows,
        cfg.origins(),
        aperture_m=cfg.detector.aperture_m,
        n_sub=2,
        z0=7.0,
        w=0.3,
        h=1.0,
        xs=xs,
        y_extent=(-5.0, 5.0),
    )
    lam = L @ np.full(len(xs), 1.2) + 0.1
    c = replace(cfg, beamdepth=replace(cfg.beamdepth, n_sub=2))
    return c, FitData(lam=lam, w=np.full(rows.n_rows, 1 / 0.02**2), rows=rows), xs


def test_fit_reports_convergence_and_json_keys(cfg):
    c, data, xs = _analytic(cfg)
    fit = fit_beam_depth(data, c, xs_init=xs, z0_init=7.1)
    assert fit.converged and fit.n_eval > 0
    js = fit.to_json()
    assert js["chisq_per_dof"] == fit.chi2_per_dof
    assert js["converged"] is True and js["n_eval"] == fit.n_eval
    assert not any(ch.isdigit() for key in js for ch in key)


def test_failed_solve_raises(cfg, monkeypatch):
    c, data, xs = _analytic(cfg)
    real = optimize.least_squares
    monkeypatch.setattr(optimize, "least_squares", lambda *a, **k: real(*a, **{**k, "max_nfev": 1}))
    with pytest.raises(RuntimeError, match="beam-depth fit failed"):
        fit_beam_depth(data, c, xs_init=xs, z0_init=7.5)


def _column(xs, h, *, z0=7.0):
    g = VoxelGrid(origin=(-3.0, -3.0, 5.0), spacing=0.1, shape=(60, 60, 50))
    rho = beam_ceiling(
        g, xs=xs, z0=z0, w=0.3, h=h, kappa=(1.0,) * len(xs), y_extent=(-3, 3), slab_thickness=0.0
    )
    return VoxelSolution(rho=rho.ravel(), grid=g, offsets={}, position_ids=("pos0", "pos1"))


def test_zprofile_face_off_grid_is_nan(cfg):
    zp = zprofile_depth(_column((0.0,), 3.5), cfg, xs=(0.0,), w=0.3, z_ref=7.0)
    assert zp["bottom"] == pytest.approx(7.0, abs=0.1)
    assert np.isnan(zp["top"]) and np.isnan(zp["fwhm"])


def test_zprofile_without_signal_raises(cfg):
    sol = _column((0.0,), 1.2)
    empty = replace(sol, rho=np.zeros_like(sol.rho))
    with pytest.raises(ValueError, match="no beam signal"):
        zprofile_depth(empty, cfg, xs=(0.0,), w=0.3, z_ref=7.0)


def test_zprofile_empty_selections_raise(cfg):
    with pytest.raises(ValueError, match="between-beam columns"):
        zprofile_depth(_column((2.8,), 1.2), cfg, xs=(2.8,), w=0.3, z_ref=7.0)
    far = replace(cfg, beamdepth=replace(cfg.beamdepth, y_band_m=(10.0, 11.0)))
    with pytest.raises(ValueError, match="y band"):
        zprofile_depth(_column((0.0,), 1.2), far, xs=(0.0,), w=0.3, z_ref=7.0)


def _stub_fit(chi2):
    return BeamDepthFit(
        z0=7.0,
        w=0.3,
        h=1.0,
        xs=(0.0,),
        kappa=(1.0,),
        chi2_per_dof=chi2,
        at_bound=False,
        n_rows=1,
        profiles={},
        converged=True,
        n_eval=1,
    )


@pytest.mark.parametrize("winner", [0, 1])
def test_fit_seeds_bottom_and_centre_and_keeps_lower_chisq(cfg, monkeypatch, winner):
    seeds = []
    chis = (1.0, 2.0) if winner == 0 else (2.0, 1.0)

    def once(data, c, *, xs_init, z0_init, angle_jitter):
        seeds.append(z0_init)
        return _stub_fit(chis[len(seeds) - 1])

    monkeypatch.setattr(beamdepth, "_fit_once", once)
    fit = fit_beam_depth(None, cfg, xs_init=(0.0,), z0_init=7.6)
    assert seeds == pytest.approx([7.6, 7.6 - cfg.beamdepth.h_init_m / 2])
    assert fit.chi2_per_dof == 1.0


def test_non_finite_chisq_raises(cfg, monkeypatch):
    monkeypatch.setattr(beamdepth, "_fit_once", lambda *a, **k: _stub_fit(np.nan))
    with pytest.raises(RuntimeError, match="non-finite"):
        fit_beam_depth(None, cfg, xs_init=(0.0,), z0_init=7.6)


@pytest.mark.slow
def test_estimate_escapes_the_one_seed_local_minimum(cfg):
    """Noise draw 5 of the h = 1.25 phantom: seeded only at the triangulated
    height, the fit settled at h = 1.10 with a worse chi^2."""
    c, data = beam_phantom(cfg, 1.25, np.random.default_rng(5))
    beams, fit = estimate_beam_depth(data, c, phantom_sky())
    assert beams["ok"] and fit.z0 < beams["z"]
    assert fit.h == pytest.approx(1.25, abs=0.05)
    assert fit.z0 == pytest.approx(7.0, abs=0.05)


def test_estimate_raises_when_triangulation_fails(cfg, monkeypatch):
    monkeypatch.setattr(beamdepth, "find_beams", lambda *a: {"ok": False})
    with pytest.raises(RuntimeError, match="triangulation failed"):
        estimate_beam_depth(None, cfg, None)
