import numpy as np
import pytest

from cafetomo.angular import AnalysisGrid
from cafetomo.opacity import (
    OpacityMaps,
    build_fit_data,
    load_sigma,
    opacity_sigma,
    poisson_sigma,
    save_sigma,
    solve_opacity,
)
from cafetomo.sky import detector_to_sky


def _grid(cfg, absorb=0.0, n=50, t=1.25, sky_rate=2000.0):
    edges = np.linspace(-t, t, n + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    tx, ty = np.meshgrid(c, c, indexing="ij")
    acc = cfg.detector.acceptance(tx, ty)
    sky = np.round(sky_rate * acc / acc.max()).astype(np.int64)
    pos = np.round(0.5 * sky * np.exp(-absorb)).astype(np.int64)
    return AnalysisGrid(edges=edges, counts={"SKY": sky, "pos0": pos, "pos1": pos})


LIVE = {"SKY": 2.0, "pos0": 1.0, "pos1": 1.0}


def test_absolute_lambda_recovers_uniform_absorption(cfg):
    maps = solve_opacity(_grid(cfg, absorb=0.4), cfg, LIVE)
    lam = maps.lam["pos0"]
    seen = np.isfinite(lam)
    assert seen.sum() > 100
    assert np.nanmedian(lam) == pytest.approx(0.4, abs=0.03)


def test_missing_live_time_raises(cfg):
    with pytest.raises(ValueError, match="SKY"):
        solve_opacity(_grid(cfg), cfg, {"pos0": 1.0, "pos1": 1.0})
    with pytest.raises(ValueError, match="pos1"):
        solve_opacity(_grid(cfg), cfg, {"SKY": 2.0, "pos0": 1.0})


@pytest.mark.parametrize("bad", ["SKY", "pos0"])
def test_non_positive_live_time_raises(cfg, bad):
    with pytest.raises(ValueError, match=bad):
        solve_opacity(_grid(cfg), cfg, {**LIVE, bad: 0.0})


def test_bootstrap_sigma_close_to_poisson(cfg):
    g = _grid(cfg, absorb=0.2)
    a = poisson_sigma(g, cfg, LIVE)["pos0"]
    b = opacity_sigma(g, cfg, LIVE, n_replicas=40, seed=1)["pos0"]
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    assert np.median(b[ok] / a[ok]) == pytest.approx(1.0, abs=0.25)


def test_fit_data_rows_and_weights(cfg):
    g = _grid(cfg, absorb=0.2)
    maps = solve_opacity(g, cfg, LIVE)
    data = build_fit_data(maps, cfg, poisson_sigma(g, cfg, LIVE))
    assert data.rows.position_ids == ("pos0", "pos1")
    assert np.all(data.w > 0) and np.all(np.isfinite(data.lam))
    assert data.rows.t_reach() <= cfg.opacity.max_tan + 1e-9


def test_pinned_rows_nan_gets_zero_weight(cfg):
    g = _grid(cfg, absorb=0.2)
    sig = poisson_sigma(g, cfg, LIVE)
    nominal = build_fit_data(solve_opacity(g, cfg, LIVE), cfg, sig)
    maps = solve_opacity(g, cfg, LIVE)
    k = nominal.rows.sky_flat[0]
    lam = {p: v.copy() for p, v in maps.lam.items()}
    lam["pos0"][k] = np.nan
    pinned = build_fit_data(OpacityMaps(lam=lam, sky=maps.sky), cfg, sig, rows=nominal.rows)
    assert pinned.rows is nominal.rows
    assert pinned.w[0] == 0.0 and np.isfinite(pinned.lam).all()
    np.testing.assert_array_equal(pinned.w[1:], nominal.w[1:])
    np.testing.assert_array_equal(pinned.lam[1:], nominal.lam[1:])


def test_save_load_roundtrip(cfg, tmp_path):
    maps = solve_opacity(_grid(cfg), cfg, LIVE)
    maps.save(tmp_path / "maps.npz")
    back = OpacityMaps.load(tmp_path / "maps.npz")
    np.testing.assert_array_equal(back.lam["pos1"], maps.lam["pos1"])
    assert back.sky.n_bins == maps.sky.n_bins


def test_shifted_adds_constant(cfg):
    maps = solve_opacity(_grid(cfg), cfg, LIVE)
    s = maps.shifted(0.03)
    assert np.nanmax(np.abs(s.lam["pos0"] - maps.lam["pos0"] - 0.03)) < 1e-12


def _structured_scene(cfg, scale=0.3, level=1e5, seed=0):
    """Sky run = response x flux x level; position = scale x sky x exp(-lambda).

    The response carries strong per-bin structure, so a solver that ignored
    the sky run would print it into the opacity.
    """
    edges = np.linspace(-1.25, 1.25, 51)
    c = 0.5 * (edges[:-1] + edges[1:])
    tx, ty = np.meshgrid(c, c, indexing="ij")
    rng = np.random.default_rng(seed)
    response = cfg.detector.acceptance(tx, ty) * rng.uniform(0.3, 1.0, tx.shape)
    n_sky = response / (1.0 + tx**2 + ty**2) * level
    lam_true = 0.8 * np.exp(-((tx - 0.2) ** 2 + (ty + 0.1) ** 2) / 0.1)
    counts = {"SKY": np.round(n_sky).astype(np.int64)}
    for pid in cfg.position_ids:
        counts[pid] = np.round(scale * n_sky * np.exp(-lam_true)).astype(np.int64)
    return AnalysisGrid(edges=edges, counts=counts), lam_true, (tx, ty)


def _detector_bins(cfg, maps, tx, ty, pid="pos0"):
    sx, sy, _ = detector_to_sky(tx, ty, cfg.exposure(pid).pose)
    return maps.sky.bin_index(sx, sy)


def test_response_structure_cancels_and_no_constant_is_left(cfg):
    g, lam_true, (tx, ty) = _structured_scene(cfg, scale=1.0)
    live = {"SKY": 1000.0, "pos0": 1000.0, "pos1": 1000.0}
    maps = solve_opacity(g, cfg, live)
    flat, ok = _detector_bins(cfg, maps, tx, ty)
    lam = maps.lam["pos0"][flat[ok]]
    seen = np.isfinite(lam)
    diff = lam[seen] - lam_true[ok][seen]
    assert seen.sum() > 500
    assert np.std(diff) < 0.01
    assert abs(np.mean(diff)) < 0.005
    assert np.corrcoef(lam[seen], lam_true[ok][seen])[0, 1] > 0.999


def test_live_time_ratio_sets_the_scale(cfg):
    g, _, _ = _structured_scene(cfg, scale=0.25)
    ratio = solve_opacity(g, cfg, {"SKY": 1000.0, "pos0": 250.0, "pos1": 250.0})
    assert np.nanmin(ratio.lam["pos0"]) == pytest.approx(0.0, abs=0.01)
    wrong = solve_opacity(g, cfg, {"SKY": 1000.0, "pos0": 1000.0, "pos1": 1000.0})
    assert np.nanmin(wrong.lam["pos0"]) == pytest.approx(np.log(4.0), abs=0.01)


def test_sparse_sky_bins_are_not_constrained(cfg):
    g, _, (tx, ty) = _structured_scene(cfg, level=30.0)
    maps = solve_opacity(g, cfg, {"SKY": 1.0, "pos0": 1.0, "pos1": 1.0})
    flat, ok = _detector_bins(cfg, maps, tx, ty)
    sparse = ok & (g.counts["SKY"] < cfg.opacity.min_sky)
    assert sparse.any()
    assert np.isnan(maps.lam["pos0"][flat[sparse]]).all()


def test_poisson_sigma_is_the_ratio_error(cfg):
    g, _, (tx, ty) = _structured_scene(cfg, level=400.0)
    live = {"SKY": 1.0, "pos0": 1.0, "pos1": 1.0}
    sig = poisson_sigma(g, cfg, live)["pos0"]
    maps = solve_opacity(g, cfg, live)
    flat, ok = _detector_bins(cfg, maps, tx, ty)
    # a sky bin fed by exactly one detector bin: the analytic form is exact there
    hits = np.bincount(flat[ok], minlength=maps.sky.flat_size)
    one = np.nonzero(
        ok & (hits[np.where(ok, flat, 0)] == 1) & (g.counts["SKY"] >= cfg.opacity.min_sky)
    )
    i, j = one[0][0], one[1][0]
    n_p, n_s = g.counts["pos0"][i, j], g.counts["SKY"][i, j]
    assert sig[flat[i, j]] == pytest.approx(np.sqrt(1 / n_p + 1 / n_s))


def test_missing_sky_reference_counts_raises(cfg):
    g, _, _ = _structured_scene(cfg)
    no_sky = AnalysisGrid(
        edges=g.edges, counts={k: v for k, v in g.counts.items() if k != cfg.sky_reference.id}
    )
    with pytest.raises(ValueError, match="SKY"):
        solve_opacity(no_sky, cfg, LIVE)


UNEQUAL = {"SKY": 4.0, "pos0": 1.0, "pos1": 1.0}


def test_poisson_sigma_uses_raw_sky_counts_under_unequal_live_times(cfg):
    g, _, (tx, ty) = _structured_scene(cfg, level=400.0)
    sig = poisson_sigma(g, cfg, UNEQUAL)["pos0"]
    maps = solve_opacity(g, cfg, UNEQUAL)
    flat, ok = _detector_bins(cfg, maps, tx, ty)
    hits = np.bincount(flat[ok], minlength=maps.sky.flat_size)
    one = np.nonzero(
        ok & (hits[np.where(ok, flat, 0)] == 1) & (g.counts["SKY"] >= cfg.opacity.min_sky)
    )
    i, j = one[0][0], one[1][0]
    n_p, n_s = g.counts["pos0"][i, j], g.counts["SKY"][i, j]
    assert sig[flat[i, j]] == pytest.approx(np.sqrt(1 / n_p + 1 / n_s))


def test_bootstrap_sigma_matches_poisson_under_unequal_live_times(cfg):
    g = _grid(cfg, absorb=0.2)
    a = poisson_sigma(g, cfg, UNEQUAL)["pos0"]
    b = opacity_sigma(g, cfg, UNEQUAL, n_replicas=60, seed=2)["pos0"]
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    assert np.median(b[ok] / a[ok]) == pytest.approx(1.0, abs=0.1)


def test_save_sigma_creates_parent_directory(cfg, tmp_path):
    g = _grid(cfg)
    sig = poisson_sigma(g, cfg, LIVE)
    save_sigma(tmp_path / "new" / "sigma.npz", sig)
    np.testing.assert_array_equal(load_sigma(tmp_path / "new" / "sigma.npz")["pos0"], sig["pos0"])
