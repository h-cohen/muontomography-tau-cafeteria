from dataclasses import replace

import numpy as np
import pytest

from cafetomo import validation
from cafetomo.beamdepth import estimate_beam_depth
from cafetomo.validation import validate_autofocus, validate_depth
from phantoms import SLAB_M, XS, Z0, W, beam_phantom, phantom_sky

H_TRUE = 1.25
RHO = 2.4
NOMINAL = {
    "xs": list(XS),
    "w": W,
    "h": H_TRUE,
    "density": RHO,
    "overburden_mean": 100.0 * RHO * SLAB_M,
    "zbottom": Z0,
}


@pytest.fixture(scope="module")
def phantom(cfg):
    cfg = replace(cfg, beamdepth=replace(cfg.beamdepth, model="concrete"))
    c, data = beam_phantom(cfg, H_TRUE, np.random.default_rng(6))
    return c.fast(), data


@pytest.mark.slow
def test_validate_depth_runs_the_reported_estimator(phantom, monkeypatch):
    """Recovery goes through estimate_beam_depth, seeded by triangulating each
    realisation: validation has no route to the fit that could take truth seeds."""
    c, data = phantom
    calls = []

    def spy(d, cf, sky, **kw):
        calls.append(d)
        return estimate_beam_depth(d, cf, sky, **kw)

    monkeypatch.setattr(validation, "estimate_beam_depth", spy)
    assert not hasattr(validation, "fit_beam_depth")
    out = validate_depth(data, c, NOMINAL, sky=phantom_sky())
    n = len(c.validation.depth_h_true_m)
    assert n == 1 and len(calls) == n * c.validation.n_realizations
    assert all(len(out[k]) == n for k in ("depth_true", "depth_mean", "depth_spread"))
    assert np.isfinite(out["depth_max_bias"])
    assert out["depth_max_bias"] == pytest.approx(abs(out["depth_mean"][0] - out["depth_true"][0]))
    assert not any(ch.isdigit() for k in out for ch in k)


@pytest.mark.slow
def test_validate_autofocus_reports_signed_bias(phantom):
    c, data = phantom
    out = validate_autofocus(data, c, NOMINAL)
    err = np.array(out["focus_recovered"]) - np.array(out["focus_injected"])
    assert out["focus_injected"] == [z + H_TRUE / 2 for z in c.validation.focus_heights_m]
    assert out["focus_bias"] == pytest.approx(float(err.mean()))
    assert out["focus_max_error"] == pytest.approx(float(np.abs(err).max()))
    assert np.isfinite(out["focus_bias"])
    assert not any(ch.isdigit() for k in out for ch in k)


def test_validation_needs_a_finite_overburden(cfg):
    cfg = replace(cfg, beamdepth=replace(cfg.beamdepth, model="concrete"))
    from cafetomo.voxels import VoxelGrid

    g = VoxelGrid(origin=(-3.0, -3.0, 6.0), spacing=0.1, shape=(10, 10, 10))
    with pytest.raises(ValueError, match="finite fitted density and overburden"):
        validation._truth(g, cfg, NOMINAL | {"overburden_mean": float("nan")}, z0=7.0, h=0.5)


def test_validation_truth_contains_high_beams_and_the_complete_slab(cfg):
    cfg = replace(cfg, beamdepth=replace(cfg.beamdepth, model="concrete"))
    from cafetomo.fitdata import FitData
    from cafetomo.phantom import sky_rows

    rows = sky_rows(cfg.position_ids, 0.1, 2)
    data = FitData(lam=np.zeros(rows.n_rows), w=np.ones(rows.n_rows), rows=rows)
    nominal = NOMINAL | {"zbottom": 8.9}
    grid = validation._truth_grid(cfg, data, nominal)
    # 8.9 m bottom + 2 m beam + 0.3 m slab: no material may fall off the grid.
    assert grid.extent(2)[1] >= 11.2
    truth = validation._truth(grid, cfg, nominal, z0=8.9, h=2.0).reshape(grid.shape)
    occupied_z = grid.axis_centers(2)[np.any(truth > 0, axis=(0, 1))]
    assert occupied_z.max() == pytest.approx(11.175, abs=1e-6)


def test_geometry_validation_uses_fitted_opacity_without_a_density(cfg):
    from cafetomo.voxels import VoxelGrid

    g = VoxelGrid(origin=(-3.0, -3.0, 6.0), spacing=0.05, shape=(120, 120, 60))
    nominal = NOMINAL | {
        "density": float("nan"),
        "overburden_mean": float("nan"),
        "kappa": [0.12] * len(XS),
        "background_mean": 0.07,
    }
    truth = validation._truth(g, cfg, nominal, z0=7.0, h=1.2)
    assert np.isfinite(truth).all()
    assert np.max(truth) == pytest.approx(0.07 / 0.3)


def test_geometry_depth_phantoms_preserve_observed_column_opacity(cfg):
    from cafetomo.forward import build_forward_model
    from cafetomo.phantom import sky_rows
    from cafetomo.voxels import VoxelGrid

    nominal = {
        "xs": [0.0],
        "w": 0.3,
        "h": 0.6,
        "zbottom": 7.0,
        "kappa": [0.18],
        "background_mean": 0.07,
    }
    grid = VoxelGrid(origin=(-0.5, -0.5, 6.5), spacing=0.1, shape=(10, 10, 25))
    truth = validation._truth(grid, cfg, nominal, z0=7.0, h=1.2)
    rows = sky_rows(("pos0",), 0.1, 1)
    c = replace(cfg, volume=replace(cfg.volume, n_aperture_sub=1))
    fwd = build_forward_model(rows, c, grid=grid)
    predicted = fwd.predict(truth)
    assert predicted[0] == pytest.approx(0.18 * 0.6 + 0.07, abs=1e-10)
