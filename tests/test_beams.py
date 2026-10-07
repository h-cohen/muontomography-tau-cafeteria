from dataclasses import replace

import numpy as np
import pytest

from cafetomo.beams import (
    beam_peaks_subbin,
    find_beams,
    match_peaks,
    sky_images,
    triangulate,
    verify_gate,
)
from cafetomo.config import BeamSettings
from cafetomo.fitdata import FitData
from cafetomo.forward import build_forward_model
from cafetomo.phantom import beam_ceiling, phantom_data, sky_rows
from cafetomo.reconstruct import VoxelSolution
from cafetomo.sky import make_sky_grid
from cafetomo.voxels import VoxelGrid


def test_subbin_peak_refines():
    g = np.linspace(-1, 1, 41)
    p = np.exp(-((g - 0.013) ** 2) / 0.01)
    assert beam_peaks_subbin(g, p, 0.5)[0] == pytest.approx(0.013, abs=0.005)


@pytest.mark.parametrize("offset", np.linspace(-0.45, 0.45, 19))
def test_subbin_peak_of_a_beam_narrower_than_a_bin(offset):
    """A box 0.8 bin wide, integrated into unit bins: the refinement must not
    lock onto the bin centre."""
    k = np.arange(-10, 11, dtype=float)
    lo = np.maximum(k - 0.5, offset - 0.4)
    hi = np.minimum(k + 0.5, offset + 0.4)
    prof = np.clip(hi - lo, 0.0, None)
    assert beam_peaks_subbin(k, prof, 0.5)[0] == pytest.approx(offset, abs=0.15)


def test_match_peaks_groups_by_world_position():
    origins = {"pos0": (0.0, 0.0, 0.0), "pos1": (1.0, 0.0, 0.0)}
    peaks = {"pos0": np.array([0.0, 0.5]), "pos1": np.array([-1 / 7, 0.5 - 1 / 7])}
    feats = match_peaks(peaks, origins, "x", 7.0, 0.3)
    assert len(feats) == 2 and all(len(f["obs"]) == 2 for f in feats)


def test_sky_images_leave_unweighted_rows_unmeasured():
    sky = make_sky_grid(1.0, 4)
    rows = sky_rows(("pos0", "pos1"), 1.0, 4)
    lam = np.arange(rows.n_rows, dtype=float)
    w = np.ones(rows.n_rows)
    w[0] = 0.0
    imgs = sky_images(FitData(lam=lam, w=w, rows=rows), sky)
    assert np.isnan(imgs["pos0"][0, 0])
    assert imgs["pos0"][1, 2] == 6.0 and imgs["pos1"][1, 2] == 22.0


def test_triangulate_needs_two_positions():
    img = np.zeros((10, 10))
    with pytest.raises(ValueError, match="two positions"):
        triangulate({"pos0": img}, {"pos0": (0.0, 0.0, 0.0)}, np.linspace(-1, 1, 10),
                    7.0, 0.01, BeamSettings())


def _comb_images(origins, centers, xs, z):
    """Sky images of thin beams along y at world x = xs, height z: a Gaussian
    bump at each beam's tangent from each origin."""
    out = {}
    for pid, (ox, _, oz) in origins.items():
        tb = (np.asarray(xs) - ox) / (z - oz)
        prof = np.exp(-((centers[:, None] - tb[None, :]) ** 2) / (2 * 0.03**2)).sum(axis=1)
        out[pid] = np.repeat(prof[:, None], len(centers), axis=1)
    return out


def test_verify_gate_passes_matching_reconstruction():
    centers = make_sky_grid(1.0, 200).centers
    origins = {"pos0": (0.0, 0.0, 0.0), "pos1": (1.78, 0.0, 0.0)}
    xs = (-3.4, -1.7, 0.0, 1.7, 3.4)
    images = _comb_images(origins, centers, xs, 7.0)
    g = VoxelGrid(origin=(-6.0, -4.0, 6.0), spacing=0.1, shape=(120, 80, 20))
    truth = beam_ceiling(g, xs=xs, z0=6.9, w=0.2, h=0.2, kappa=(1.0,) * 5,
                         y_extent=(-4.0, 4.0), slab_kappa=0.0)
    sol = VoxelSolution(rho=truth.ravel(), grid=g, offsets={}, position_ids=tuple(origins))
    gate = verify_gate(images, origins, centers, 7.0, sol, BeamSettings())
    assert gate["passed"] and gate["n_beams_data"] >= 3
    assert gate["recon_layer_z"] == pytest.approx(7.0, abs=0.11)


def _grid_data(images, sky, pids):
    rows = sky_rows(pids, float(sky.edges[-1]), sky.n_bins)
    lam = np.concatenate([images[pid].ravel()[rows.sky_flat[rows.pos_of_row == k]]
                          for k, pid in enumerate(pids)])
    return FitData(lam=lam, w=np.ones(rows.n_rows), rows=rows)


@pytest.mark.parametrize(("min_y_corr", "used"), [(-1.0, True), (1.01, False)])
def test_y_features_enter_the_joint_fit_only_above_min_y_corr(cfg, min_y_corr, used):
    cfg = replace(cfg.with_pose("pos1", cfg.exposure("pos1").pose.__class__(1.78, 0.72, 0.0, 0.0)),
                  beams=replace(cfg.beams, min_y_corr=min_y_corr))
    sky = make_sky_grid(cfg.opacity.sky_t_max, cfg.opacity.sky_n_bins)
    origins = cfg.origins()
    xs_img = _comb_images(origins, sky.centers, (-3.4, -1.7, 0.0, 1.7, 3.4), 7.0)
    swapped = {pid: (oy, ox, oz) for pid, (ox, oy, oz) in origins.items()}
    ys_img = _comb_images(swapped, sky.centers, (-2.0, 1.0), 7.0)
    images = {pid: xs_img[pid] + ys_img[pid].T for pid in origins}
    res = find_beams(_grid_data(images, sky, cfg.position_ids), cfg, sky)
    assert res["ok"] and res["y_used"] is used
    assert (res["n_features_y"] > 0) is used
    assert res["z_x"] == pytest.approx(7.0, abs=0.1) and np.isfinite(res["z_x_sigma"])
    if not used:
        assert res["z"] == res["z_x"]


@pytest.mark.slow
def test_triangulates_phantom_ceiling(cfg):
    cfg = cfg.with_pose("pos1", cfg.exposure("pos1").pose.__class__(1.78, 0.72, 0.0, 0.0))
    sky = make_sky_grid(cfg.opacity.sky_t_max, cfg.opacity.sky_n_bins)
    rows = sky_rows(cfg.position_ids, cfg.opacity.sky_t_max, cfg.opacity.sky_n_bins)
    keep = (np.abs(rows.sx) <= 0.9) & (np.abs(rows.sy) <= 0.9)
    rows = rows.__class__(rows.position_ids, rows.pos_of_row[keep], rows.sx[keep],
                          rows.sy[keep], rows.sky_flat[keep])
    g = VoxelGrid(origin=(-7.0, -7.0, 6.5), spacing=0.1, shape=(160, 140, 15))
    truth = beam_ceiling(g, xs=(-3.4, -1.7, 0.0, 1.7, 3.4), z0=7.0, w=0.3, h=0.3,
                         kappa=(2.0,) * 5, y_extent=(-5.0, 5.0))
    fwd = build_forward_model(rows, cfg, grid=g)
    like = FitData(lam=np.zeros(rows.n_rows), w=np.full(rows.n_rows, 1 / 0.02**2), rows=rows)
    res = find_beams(phantom_data(fwd, truth, like, np.random.default_rng(0)), cfg, sky)
    assert res["ok"]
    assert res["z"] == pytest.approx(7.15, abs=0.2)
    assert res["pitch"] == pytest.approx(1.7, abs=0.15)
