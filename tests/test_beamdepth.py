from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import optimize

from cafetomo import beamdepth
from cafetomo.beamdepth import (
    beam_design,
    box_path_lengths,
    estimate_beam_depth,
    fit_beam_depth,
    zprofile_depth,
)
from cafetomo.fitdata import FitData
from cafetomo.muonphysics import Transmission, kappa_concrete, overburden_from_lambda
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


def test_variable_projection_bounds_beam_opacity_but_not_background():
    design = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    coef = beamdepth._linear_coefficients(design, np.array([-1.0, 2.0, 0.0]), 2)
    assert coef == pytest.approx([0.0, 2.5, -0.5])


@pytest.mark.slow
@pytest.mark.parametrize("model", ["geometry", "concrete"])
@pytest.mark.parametrize("h_true", [0.6, 1.25])
def test_noisy_depth_recovery_has_small_ensemble_bias(cfg, model, h_true):
    """A single noisy estimate has statistical scatter; test mean bias over
    all five predeclared draws, not a noise seed selected for accuracy."""
    cfg = replace(
        cfg,
        beamdepth=replace(cfg.beamdepth, model=model),
        volume=replace(cfg.volume, n_aperture_sub=8),
    )
    zero = SimpleNamespace(normal=lambda size: np.zeros(size))
    c, clean = beam_phantom(cfg, h_true, zero)
    rng = np.random.default_rng(3)
    fitted = []
    for _ in range(5):
        data = replace(clean, lam=clean.lam + rng.normal(size=clean.lam.size) / np.sqrt(clean.w))
        fit = fit_beam_depth(data, c, xs_init=(-1.7, 0.0, 1.7, 3.4), z0_init=7.1)
        assert fit.converged and not fit.at_bound
        fitted.append((fit.h, fit.z0))
    mean = np.mean(fitted, axis=0)
    assert mean[0] == pytest.approx(h_true, abs=0.15)
    assert mean[1] == pytest.approx(7.0, abs=0.15)


def test_at_bound_flag(cfg):
    cfg = replace(cfg, beamdepth=replace(cfg.beamdepth, model="concrete"))
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
    """Noiseless lambda from the pinned fit's own physics: concrete beams of
    depth 1.0 m behind a uniform overburden of lambda 0.07. A cheap stand-in
    for a phantom."""
    rows = sky_rows(cfg.position_ids, 0.9, 24)
    xs = (-1.7, 0.0, 1.7)
    L = beam_design(
        rows,
        cfg.origins(),
        aperture_m=cfg.detector.aperture_m,
        layer_dz_m=cfg.detector.layer_dz_cm / 100.0,
        azimuths={e.id: e.pose.az_deg for e in cfg.exposures},
        n_sub=2,
        z0=7.0,
        w=0.3,
        h=1.0,
        xs=xs,
        y_extent=(-5.0, 5.0),
    )
    p = cfg.physics
    cos = 1.0 / np.sqrt(1.0 + rows.sx**2 + rows.sy**2)
    trans = Transmission(cos, threshold_gev=p.detector_threshold_gev, model=p.flux_model)
    x_bg = trans.overburden(np.full(rows.n_rows, 0.07))
    lam = trans.lam(x_bg + 100.0 * p.concrete_density_gcm3 * L.sum(axis=1))
    c = replace(cfg, beamdepth=replace(cfg.beamdepth, n_sub=2, model="concrete"))
    return c, FitData(lam=lam, w=np.full(rows.n_rows, 1 / 0.02**2), rows=rows), xs


def test_fit_reports_convergence_and_json_keys(cfg):
    c, data, xs = _analytic(cfg)
    fit = fit_beam_depth(data, c, xs_init=xs, z0_init=7.1)
    assert fit.converged and fit.n_eval > 0
    js = fit.to_json()
    assert js["chisq_per_dof"] == fit.chi2_per_dof
    assert js["converged"] is True and js["n_eval"] == fit.n_eval
    assert "kappa" not in js
    assert (js["kappa_mean"], js["overburden_mean"]) == (fit.kappa_mean, fit.overburden_mean)
    assert js["density"] == c.physics.concrete_density_gcm3
    assert not any(ch.isdigit() for key in js for ch in key)


def test_pinned_fit_recovers_noiseless_physics(cfg):
    """The beams' opacity is not free: the noiseless model data come back at
    their depth, with the pinned kappa of concrete behind lambda 0.07."""
    c, data, xs = _analytic(cfg)
    fit = fit_beam_depth(data, c, xs_init=xs, z0_init=7.1)
    assert fit.h == pytest.approx(1.0, abs=0.05)
    assert fit.z0 == pytest.approx(7.0, abs=0.05)
    p = c.physics
    k_vertical = kappa_concrete(
        overburden_from_lambda(
            0.07, 1.0, threshold_gev=p.detector_threshold_gev, model=p.flux_model
        ),
        1.0,
        p.concrete_density_gcm3,
        threshold_gev=p.detector_threshold_gev,
        model=p.flux_model,
    )
    assert fit.kappa_mean == pytest.approx(float(k_vertical), rel=0.25)


def _geometry_analytic(cfg):
    c, _, xs = _analytic(cfg)
    c = replace(c, beamdepth=replace(c.beamdepth, model="geometry"))
    rows = sky_rows(c.position_ids, 0.9, 24)
    paths = beam_design(
        rows,
        c.origins(),
        aperture_m=c.detector.aperture_m,
        layer_dz_m=c.detector.layer_dz_cm / 100.0,
        azimuths={e.id: e.pose.az_deg for e in c.exposures},
        n_sub=2,
        z0=7.0,
        w=0.3,
        h=1.2,
        xs=xs,
        y_extent=(-5.0, 5.0),
    )
    data = FitData(
        lam=paths @ np.array([0.12, 0.18, 0.15]) + 0.07,
        w=np.full(rows.n_rows, 1 / 0.02**2),
        rows=rows,
    )
    return c, data, xs


def test_geometric_fit_recovers_depth_without_a_density_constraint(cfg):
    c, data, xs = _geometry_analytic(cfg)
    fit = fit_beam_depth(data, c, xs_init=xs, z0_init=7.4)
    assert fit.h == pytest.approx(1.2, abs=0.05)
    assert fit.z0 == pytest.approx(7.0, abs=0.05)
    assert fit.kappa == pytest.approx([0.12, 0.18, 0.15], abs=0.01)
    assert np.isnan(fit.density)
    altered = replace(c, physics=replace(c.physics, concrete_density_gcm3=3.0))
    other = fit_beam_depth(data, altered, xs_init=xs, z0_init=7.4)
    assert other.h == pytest.approx(fit.h, abs=1e-8)


def test_depth_profile_refits_nuisance_parameters_and_recovers_injected_minimum(cfg):
    c, data, xs = _geometry_analytic(cfg)
    profile = beamdepth.profile_beam_depth(
        data, c, xs_init=xs, z0_init=7.4, heights=np.array([0.6, 1.2, 1.8])
    )
    assert profile["h"][np.argmin(profile["chi2"])] == 1.2
    assert min(profile["chi2"]) < 1e-6
    assert not profile["minimum_at_edge"]


def test_density_and_flux_model_keywords_override_the_config(cfg):
    c, data, xs = _analytic(cfg)
    nominal = fit_beam_depth(data, c, xs_init=xs, z0_init=7.1)
    denser = fit_beam_depth(data, c, xs_init=xs, z0_init=7.1, density=2.6)
    assert denser.density == 2.6 and denser.kappa_mean > nominal.kappa_mean
    assert denser.h < nominal.h
    alt = fit_beam_depth(data, c, xs_init=xs, z0_init=7.1, flux_model=c.physics.flux_model_alt)
    assert alt.kappa_mean != nominal.kappa_mean


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


def test_profile_at_nominal_depth_cannot_worsen_known_solution(cfg):
    c, data, xs = _geometry_analytic(cfg)
    data = replace(data, lam=data.lam + np.random.default_rng(7).normal(0, 0.02, data.lam.size))
    nominal = fit_beam_depth(data, c, xs_init=xs, z0_init=7.1)
    profile = beamdepth.profile_beam_depth(
        data, c, xs_init=xs, z0_init=7.1, heights=[nominal.h], nominal=nominal
    )
    band = beamdepth._band(data, c)
    n_parameters = (
        3 + 2 * len(xs) + beamdepth._background(band.rows, c.beamdepth.bg_degree).shape[1]
    )
    nominal_chi2 = nominal.chi2_per_dof * (band.rows.n_rows - n_parameters)
    assert profile["chi2"][0] <= nominal_chi2 + 1e-5


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


def test_fit_starts_free_then_pinned_and_keeps_lowest_chisq(cfg, monkeypatch):
    c, data, xs = _analytic(cfg)
    calls = []
    real = beamdepth._fit_once

    def spy(band, cf, *, z0_init, h_init, pinned, **kw):
        fit = real(band, cf, z0_init=z0_init, h_init=h_init, pinned=pinned, **kw)
        calls.append((z0_init, h_init, pinned is None, fit))
        return fit

    monkeypatch.setattr(beamdepth, "_fit_once", spy)
    fit = fit_beam_depth(data, c, xs_init=xs, z0_init=7.6)
    h0 = c.beamdepth.h_init_m
    free = [x for x in calls if x[2]]
    pinned = [x for x in calls if not x[2]]
    assert [(z, h) for z, h, *_ in free] == pytest.approx([(7.6, h0), (7.6 - h0 / 2, h0)])
    h_free = min((x[3] for x in free), key=lambda f: f.chi2_per_dof).h
    hs = (*c.beamdepth.h_starts_m, h_free)
    expect = [start for h in hs for start in ((7.6, h), (7.6 - h / 2, h))]
    assert [(z, h) for z, h, *_ in pinned] == pytest.approx(expect)
    assert fit.chi2_per_dof == min(x[3].chi2_per_dof for x in pinned)
    assert np.isfinite(fit.kappa_mean) and all(x[3].kappa for x in free)


def test_non_finite_chisq_raises(cfg, monkeypatch):
    c, data, xs = _analytic(cfg)
    real = beamdepth._fit_once

    def nan_fit(*a, **k):
        return replace(real(*a, **k), chi2_per_dof=np.nan)

    monkeypatch.setattr(beamdepth, "_fit_once", nan_fit)
    with pytest.raises(RuntimeError, match="non-finite"):
        fit_beam_depth(data, c, xs_init=xs, z0_init=7.6)


@pytest.mark.slow
def test_concrete_multistarts_agree_and_select_the_lowest_cost(cfg, monkeypatch):
    """Optional material-model starts agree and select the lowest cost;
    physical recovery accuracy is checked by the ensemble test."""
    cfg = replace(cfg, beamdepth=replace(cfg.beamdepth, model="concrete"))
    c, data = beam_phantom(cfg, 1.25, np.random.default_rng(3))
    calls = []
    real = beamdepth._fit_once

    def spy(band, cf, *, h_init, pinned, **kw):
        fit = real(band, cf, h_init=h_init, pinned=pinned, **kw)
        calls.append((h_init, pinned is None, fit))
        return fit

    monkeypatch.setattr(beamdepth, "_fit_once", spy)
    fit = fit_beam_depth(data, c, xs_init=(-1.7, 0.0, 1.7, 3.4), z0_init=7.1)
    free = min((f for _, is_free, f in calls if is_free), key=lambda f: f.chi2_per_dof)
    from_free = [f.h for h0, is_free, f in calls if not is_free and h0 == free.h]
    assert len(from_free) == 2
    assert abs(from_free[0] - from_free[1]) < 0.05
    pinned = [f for _, is_free, f in calls if not is_free]
    assert fit.chi2_per_dof == min(f.chi2_per_dof for f in pinned)


@pytest.mark.slow
def test_reported_estimator_recovers_noisy_concrete_geometry(cfg):
    """The reported triangulation-seeded estimator meets the campaign's
    0.15 m recovery tolerance on an independent concrete noise draw."""
    cfg = replace(cfg, beamdepth=replace(cfg.beamdepth, model="concrete"))
    c, data = beam_phantom(cfg, 1.25, np.random.default_rng(5))
    beams, fit = estimate_beam_depth(data, c, phantom_sky())
    assert beams["ok"] and fit.z0 < beams["z"]
    assert fit.h == pytest.approx(1.25, abs=0.15)
    assert fit.z0 == pytest.approx(7.0, abs=0.15)


def test_estimate_raises_when_triangulation_fails(cfg, monkeypatch):
    monkeypatch.setattr(beamdepth, "find_beams", lambda *a: {"ok": False})
    with pytest.raises(RuntimeError, match="triangulation failed"):
        estimate_beam_depth(None, cfg, None)


def test_scattering_paths_keep_the_observed_detector_footprint(cfg):
    rows = sky_rows(cfg.position_ids, 0.8, 1)
    rows = replace(rows, sx=np.full(rows.n_rows, 0.8), sy=np.zeros(rows.n_rows))
    kwargs = dict(
        aperture_m=cfg.detector.aperture_m,
        layer_dz_m=cfg.detector.layer_dz_cm / 100.0,
        n_sub=4,
        z0=7.0,
        w=0.3,
        h=1.0,
        xs=(5.8,),
        y_extent=(-5.0, 5.0),
    )
    plain = beam_design(rows, cfg.origins(), **kwargs)
    # Side nodes miss the box; only the central GH node (weight 2/3) crosses.
    # Deflection before reaching the detector does not shrink its entry area.
    blurred = beam_design(rows, cfg.origins(), angle_jitter=0.1, **kwargs)
    np.testing.assert_allclose(blurred, (2 / 3) * plain)


def test_aperture_average_resolves_partial_vertical_beam_exactly(cfg):
    rows = sky_rows(("pos0",), 0.1, 1)
    paths = beam_design(
        rows,
        cfg.origins(),
        aperture_m=0.35,
        layer_dz_m=0.389,
        n_sub=4,
        z0=7.0,
        w=0.14,
        h=1.0,
        xs=(0.0,),
        y_extent=(-5.0, 5.0),
    )
    assert paths[0, 0] == pytest.approx(0.14 / 0.35, abs=1e-12)


def test_rotated_aperture_average_has_the_triangular_projection(cfg):
    rows = sky_rows(("pos0",), 0.1, 1)
    paths = beam_design(
        rows,
        cfg.origins(),
        aperture_m=0.35,
        layer_dz_m=0.389,
        azimuths={"pos0": 45.0},
        n_sub=4,
        z0=7.0,
        w=0.14,
        h=1.0,
        xs=(0.0,),
        y_extent=(-5.0, 5.0),
    )
    side = 0.35 / np.sqrt(2)
    expected = 0.14 / side - 0.14**2 / (4 * side**2)
    assert paths[0, 0] == pytest.approx(expected, abs=1e-12)


def test_oblique_aperture_average_matches_independent_overlap_integral(cfg):
    from scipy.integrate import quad

    rows = sky_rows(("pos0",), 0.1, 1)
    rows = replace(rows, sx=np.array([0.2]))
    paths = beam_design(
        rows,
        cfg.origins(),
        aperture_m=0.35,
        layer_dz_m=0.389,
        n_sub=4,
        z0=7.0,
        w=0.3,
        h=1.0,
        xs=(1.5,),
        y_extent=(-5.0, 5.0),
    )
    span = 0.35 - 0.2 * 0.389

    def overlap(z):
        lo = max(-span / 2, 1.35 - 0.2 * z)
        hi = min(span / 2, 1.65 - 0.2 * z)
        return max(hi - lo, 0.0) / span

    expected = quad(overlap, 7.0, 8.0, epsabs=1e-12)[0] * np.sqrt(1 + 0.2**2)
    assert paths[0, 0] == pytest.approx(expected, abs=1e-8)


def test_bound_fit_retains_parameters_but_reports_unmeasured_dimensions(cfg):
    c, data, xs = _geometry_analytic(cfg)
    fit = fit_beam_depth(data, c, xs_init=xs, z0_init=7.1)
    report = replace(fit, at_bound=True).to_json()
    assert report["h"] == fit.h
    assert report["zbottom"] == fit.z0
    assert not report["depth_resolved"]
    assert np.isnan(report["h_measurement"])
    assert np.isnan(report["zbottom_measurement"])


def test_exact_aperture_paths_do_not_count_material_behind_tracker(cfg):
    rows = sky_rows(("pos0",), 0.1, 1)
    kwargs = dict(
        aperture_m=0.35,
        layer_dz_m=0.389,
        n_sub=4,
        z0=7.0,
        w=0.5,
        h=2.0,
        xs=(0.0,),
        y_extent=(-5.0, 5.0),
    )
    inside = beam_design(rows, {"pos0": (0.0, 0.0, 8.0)}, **kwargs)
    above = beam_design(rows, {"pos0": (0.0, 0.0, 10.0)}, **kwargs)
    assert inside[0, 0] == pytest.approx(1.0)
    assert above[0, 0] == 0.0


def test_finite_beam_end_retains_partial_aperture_paths(cfg):
    rows = sky_rows(("pos0",), 0.1, 1)
    paths = beam_design(
        rows,
        cfg.origins(),
        aperture_m=0.35,
        layer_dz_m=0.389,
        n_sub=4,
        z0=7.0,
        w=0.5,
        h=1.0,
        xs=(0.0,),
        y_extent=(-5.0, 0.0),
    )
    assert paths[0, 0] == pytest.approx(0.5)
