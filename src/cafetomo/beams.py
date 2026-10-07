"""Beam-level, model-free verification of a reconstruction.

The ceiling structure is a set of beams, and the beams are visible directly in
each position's opacity image -- so the reconstruction can be verified at the
level of the physics claim, without trusting the solver:

1. Opacity profiles: collapse each position's lambda image along the beam
   direction (vertical beams -> profile vs tan_x; horizontal -> profile vs tan_y).
2. Parallax height: project both positions' profiles onto a candidate plane z
   and correlate them. Real structure at height z* makes the profiles align at
   z* (the two positions are separated by a known baseline, so the alignment
   height is a triangulation, independent of any reconstruction). The x and y
   baselines give two independent estimates.
3. Beam match: peak positions in the data profiles (at the parallax height)
   vs peak positions in the reconstruction's layer slice, beam by beam.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from cafetomo.config import BeamSettings, Config
from cafetomo.fitdata import FitData
from cafetomo.reconstruct import VoxelSolution
from cafetomo.sky import SkyGrid

_GATE_Y_BAND_M = 2.0       # |y| range of the reconstruction slice profiled by the gate
_GATE_SMOOTH_M = 0.12      # Gaussian sigma on that profile, so voxel noise is not a beam


def _axis_index(axis: str) -> int:
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return 0 if axis == "x" else 1


def _require_two(pids, what: str) -> None:
    if len(pids) < 2:
        raise ValueError(f"{what} needs at least two positions, got {list(pids)}")


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmean without the all-NaN RuntimeWarning (all-NaN rows -> NaN)."""
    n = np.isfinite(a).sum(axis=axis)
    s = np.nansum(a, axis=axis)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def sky_images(data: FitData, sky: SkyGrid) -> dict[str, np.ndarray]:
    """Each position's measured lambda on the world sky grid; NaN where no
    weighted row exists."""
    out = {}
    for k, pid in enumerate(data.rows.position_ids):
        img = np.full(sky.flat_size, np.nan)
        sel = (data.rows.pos_of_row == k) & (data.w > 0)
        img[data.rows.sky_flat[sel]] = data.lam[sel]
        out[pid] = img.reshape(sky.n_bins, sky.n_bins)
    return out


def profile(img: np.ndarray, centers: np.ndarray, axis: str = "x",
            band: float = 0.32) -> tuple[np.ndarray, np.ndarray]:
    """Opacity profile vs tan(theta_axis), averaged over |t_other| < band.

    Images are indexed [i_x, j_y], so the x profile averages over axis 1.
    """
    sel = np.abs(centers) < band
    if _axis_index(axis) == 0:
        return centers, _nanmean(img[:, sel], axis=1)
    return centers, _nanmean(img[sel, :], axis=0)


def world_profile(t: np.ndarray, prof: np.ndarray, origin: tuple, axis: str, z: float,
                  grid: np.ndarray) -> np.ndarray:
    """A tangent profile resampled onto world coordinates of the plane at height z;
    NaN outside what the position sees."""
    w = origin[_axis_index(axis)] + t * (z - origin[2])
    ok = np.isfinite(prof)
    return np.interp(grid, w[ok], prof[ok], left=np.nan, right=np.nan)


def parallax_scan(images: dict, origins: dict, centers: np.ndarray, axis: str,
                  grid: np.ndarray, zs: np.ndarray,
                  band: float = 0.32) -> tuple[np.ndarray, float, float]:
    """Correlation of the first two positions' world-projected profiles vs height.

    Returns (correlations, best_z, best_corr); correlation is NaN at heights where
    the two projections overlap on fewer than 10 grid points.
    """
    pids = list(images)
    _require_two(pids, "parallax_scan")
    profs = {pid: profile(images[pid], centers, axis, band) for pid in pids}
    corrs = np.full(len(zs), np.nan)
    for k, z in enumerate(zs):
        ws = [world_profile(*profs[pid], origins[pid], axis, float(z), grid) for pid in pids]
        ok = np.all([np.isfinite(w) for w in ws], axis=0)
        if ok.sum() < 10:
            continue
        a, b = ws[0][ok], ws[1][ok]
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a**2).sum() * (b**2).sum())
        if denom > 0:
            corrs[k] = float((a * b).sum() / denom)
    if not np.isfinite(corrs).any():
        raise ValueError(f"parallax_scan along {axis}: the positions never overlap on the grid")
    i = int(np.nanargmax(corrs))
    return corrs, float(zs[i]), float(corrs[i])


def beam_peaks(grid: np.ndarray, prof: np.ndarray, prom_sigmas: float = 0.5) -> np.ndarray:
    """Grid positions of profile maxima more prominent than prom_sigmas x its scatter."""
    ok = np.isfinite(prof)
    if ok.sum() < 5:
        return np.array([])
    prom = prom_sigmas * float(np.nanstd(prof))
    idx, _ = find_peaks(prof[ok], prominence=prom)
    return grid[ok][idx]


def beam_peaks_subbin(grid: np.ndarray, prof: np.ndarray, prom_sigmas: float = 0.5) -> np.ndarray:
    """`beam_peaks` refined to sub-bin precision by a parabola through each maximum.

    Triangulation is acutely sensitive to peak position: with baseline ~1.8 m and a
    0.05 tan-unit bin, one bin of quantization in (t_0 - t_1) moves the closed-form
    height by ~1 m. Bin-centre peaks are therefore not good enough, and the parabolic
    refinement removes most of that error.
    """
    ok = np.isfinite(prof)
    if ok.sum() < 5:
        return np.array([])
    g, p = grid[ok], prof[ok]
    idx, _ = find_peaks(p, prominence=prom_sigmas * float(np.nanstd(prof)))
    out = []
    for i in idx:
        if 0 < i < len(p) - 1:
            denom = p[i - 1] - 2 * p[i] + p[i + 1]
            # denom < 0 at a maximum; guard against a flat or pathological triple
            delta = 0.5 * (p[i - 1] - p[i + 1]) / denom if denom < 0 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            step = g[i + 1] - g[i]
            out.append(g[i] + delta * step)
        else:
            out.append(g[i])
    return np.asarray(out)


def match_peaks(peaks: dict, origins: dict, axis: str, z0: float, tol_m: float) -> list[dict]:
    """Group per-position tan-space peaks into features by their world position at z0.

    Each position sees the same beam at a different angle; projected to the plane
    z0 they land on the same world coordinate (that is what makes z0 the right
    height). So project, then match nearest-neighbour across positions within
    `tol_m`. Returns one entry per feature seen by >= 2 positions:
    {"world0": x, "obs": {pid: t}}.
    """
    ax = _axis_index(axis)
    pids = list(peaks)
    ref = pids[0]
    o0 = origins[ref]
    base = o0[ax] + np.asarray(peaks[ref]) * (z0 - o0[2])
    feats = [{"world0": float(w), "obs": {ref: float(t)}}
             for w, t in zip(base, peaks[ref], strict=True)]
    for pid in pids[1:]:
        o = origins[pid]
        for t in peaks[pid]:
            w = o[ax] + t * (z0 - o[2])
            if not feats:
                continue
            d = [abs(w - f["world0"]) for f in feats]
            i = int(np.argmin(d))
            if d[i] < tol_m and pid not in feats[i]["obs"]:
                feats[i]["obs"][pid] = float(t)
    return [f for f in feats if len(f["obs"]) >= 2]


def triangulate(images: dict, origins: dict, centers: np.ndarray, z0: float,
                sigma_t: float, s: BeamSettings) -> dict:
    """Joint least-squares ray intersection over every matched beam, both axes.

    Each (position, beam) peak fixes a ray; a beam at height z and world position X
    is seen by position i at tan angle t = (X - p_i)/(z - z_i). Imposing that ALL
    beams share one ceiling height turns the per-pair closed form
    z = dx/(t_0 - t_1) into an overdetermined fit for (z, {X_k}, {Y_m}), which is
    both more precise and testable: the residual scatter says whether a single
    plane actually explains the data.

    Independent of the reconstruction AND of the cross-validation autofocus -- it
    uses only peak positions in the opacity images -- so it is a genuine
    cross-check rather than a restatement. `sigma_t` is the per-peak angular
    uncertainty (tan units). An underdetermined fit returns ok=False with the fit
    quantities NaN (not measured).
    """
    _require_two(list(images), "triangulate")
    feats = {}
    for axis, band in (("x", s.band_x), ("y", s.band_y)):
        peaks = {pid: beam_peaks_subbin(*profile(img, centers, axis, band), s.prominence_sigmas)
                 for pid, img in images.items()}
        feats[axis] = match_peaks(peaks, origins, axis, z0, s.match_tol_m)
    nx, ny = len(feats["x"]), len(feats["y"])
    n_obs = sum(len(f["obs"]) for a in feats for f in feats[a])
    n_par = 1 + nx + ny
    counts = {"n_features_x": nx, "n_features_y": ny, "n_observations": n_obs,
              "dof": n_obs - n_par, "sigma_t_assumed": float(sigma_t)}
    if nx + ny < 1 or n_obs <= n_par:
        nan = float("nan")
        return {"ok": False, "z": nan, "z_sigma": nan, **counts, "chi_per_dof": nan,
                "resid_rms_tan": nan, "beams_x": [], "beams_y": []}

    def unpack(p):
        return p[0], p[1: 1 + nx], p[1 + nx:]

    def resid(p):
        z, xs, ys = unpack(p)
        out = []
        for axis, vals in (("x", xs), ("y", ys)):
            ax = _axis_index(axis)
            for f, w in zip(feats[axis], vals, strict=True):
                for pid, t in f["obs"].items():
                    o = origins[pid]
                    out.append(((w - o[ax]) / (z - o[2]) - t) / sigma_t)
        return np.asarray(out)

    p0 = np.concatenate([[z0], [f["world0"] for f in feats["x"]],
                         [f["world0"] for f in feats["y"]]])
    fit = optimize.least_squares(resid, p0, method="lm")
    z, xs, ys = unpack(fit.x)

    dof = n_obs - n_par
    ssr = float(np.sum(fit.fun**2))
    # Scale the covariance by the achieved residual variance rather than trusting
    # sigma_t: it absorbs a mis-set sigma_t and reports the fit's own consistency.
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * (ssr / dof)
        sigma_z = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        sigma_z = float("nan")
    return {"ok": True, "z": float(z), "z_sigma": sigma_z, **counts,
            "chi_per_dof": ssr / dof,
            "resid_rms_tan": float(np.sqrt(ssr / n_obs) * sigma_t),
            "beams_x": [float(v) for v in xs], "beams_y": [float(v) for v in ys]}


def angular_beam_period(img: np.ndarray, centers: np.ndarray, axis: str = "x",
                        t_window: float = 0.6) -> float:
    """Dominant angular period of the beam pattern in one position, in tan-units.

    Measured by the FFT peak of the detrended opacity profile with parabolic sub-bin
    refinement. This is a per-position quantity: it needs no baseline, no pose and
    no height, which is what makes it useful for closing the scale (see
    scale_closure). NaN when fewer than 16 measured bins fall in the window.
    """
    t, p = profile(img, centers, axis)
    sel = np.isfinite(p) & (np.abs(t) < t_window)
    tt, pp = t[sel], p[sel]
    if len(tt) < 16:
        return float("nan")
    pp = pp - np.polyval(np.polyfit(tt, pp, 3), tt)
    freq = np.fft.rfftfreq(len(tt), tt[1] - tt[0])
    amp = np.abs(np.fft.rfft(pp * np.hanning(len(pp))))
    k = int(np.argmax(amp[2:])) + 2
    if k + 1 < len(amp):
        denom = amp[k - 1] - 2 * amp[k] + amp[k + 1]
        dk = 0.5 * (amp[k - 1] - amp[k + 1]) / denom if denom < 0 else 0.0
        f_pk = freq[k] + float(np.clip(dk, -0.5, 0.5)) * (freq[1] - freq[0])
    else:
        f_pk = freq[k]
    return float(1.0 / f_pk) if f_pk > 0 else float("nan")


def scale_closure(images: dict, origins: dict, centers: np.ndarray, z_m: float) -> dict:
    """Close the (baseline, ceiling height, beam pitch) scale triangle.

    Triangulation fixes only a RATIO: z = d / (t_1 - t_2), so the height scales with
    the assumed baseline d and cannot be got from the angles alone. But each
    position also measures the beam pattern's angular period on its own -- no
    baseline, pose or height involved -- and the physical pitch is
    pitch = z * period. The three quantities are therefore locked together: fixing
    ANY ONE of baseline, height or pitch by an independent measurement (a tape
    measure on the detector separation, or on the ceiling beam spacing) determines
    the other two. Reporting the triple makes the assumption visible instead of
    leaving it buried in the pose config. Baseline-derived values are NaN with
    fewer than two positions.
    """
    pids = list(images)
    if len(pids) >= 2:
        a, b = origins[pids[0]], origins[pids[1]]
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    else:
        d = float("nan")
    per = {pid: angular_beam_period(img, centers) for pid, img in images.items()}
    vals = [v for v in per.values() if np.isfinite(v)]
    period = float(np.mean(vals)) if vals else float("nan")
    pitch = period * z_m
    # z and pitch both scale linearly with the assumed baseline; these let a reader
    # rescale to any surveyed d without re-running anything.
    has_d = np.isfinite(d) and d > 0
    return {"baseline": d, "period": period, "period_per_position": per, "z": float(z_m),
            "pitch": pitch,
            "z_per_baseline": z_m / d if has_d else float("nan"),
            "pitch_per_baseline": pitch / d if has_d else float("nan")}


def verify_gate(images: dict, origins: dict, centers: np.ndarray, z_m: float,
                sol: VoxelSolution, s: BeamSettings) -> dict:
    """Do the reconstruction's beams sit where the raw data put them?

    Data beams are peaks of the positions' mean x-profile projected to the plane
    z_m; reconstruction beams are peaks of its maximum-mass layer, profiled across
    x over |y| < 2 m. Each data beam's offset is to the nearest reconstruction
    beam; the gate passes when their mean |offset| is within s.gate_max_offset_m.
    Reconstruction peaks are sought only where the data profile is measured, so
    side-wall artefacts outside the shared view cannot claim a match.
    """
    g = sol.grid
    xs_r = g.axis_centers(0)
    ys_r = g.axis_centers(1)
    xgrid = np.linspace(xs_r[0], xs_r[-1], 400)

    world = [world_profile(*profile(img, centers, "x", s.band_x), origins[pid], "x", z_m, xgrid)
             for pid, img in images.items()]
    data_prof = _nanmean(np.stack(world), axis=0)
    pk_data = beam_peaks(xgrid, data_prof, s.prominence_sigmas)

    pos = np.maximum(sol.rho3(), 0.0)
    iz = int(np.argmax(pos.sum(axis=(0, 1))))
    sl = pos[:, :, iz]
    yband = np.abs(ys_r) < _GATE_Y_BAND_M
    prof_r = gaussian_filter1d(sl[:, yband].mean(axis=1), _GATE_SMOOTH_M / g.spacing)
    recon = np.where(np.isfinite(data_prof), np.interp(xgrid, xs_r, prof_r), np.nan)
    pk_recon = beam_peaks(xgrid, recon, s.prominence_sigmas)

    offsets = ([float(pk_recon[np.argmin(np.abs(pk_recon - p))] - p) for p in pk_data]
               if len(pk_recon) else [])
    mean_abs = float(np.mean(np.abs(offsets))) if offsets else float("nan")
    return {"offsets": offsets, "mean_abs_offset": mean_abs,
            "passed": bool(np.isfinite(mean_abs) and mean_abs <= s.gate_max_offset_m),
            "n_beams_data": int(len(pk_data)), "n_beams_recon": int(len(pk_recon)),
            "recon_layer_z": float(g.axis_centers(2)[iz])}


def find_beams(data: FitData, cfg: Config, sky: SkyGrid) -> dict:
    """Model-free beam geometry: parallax heights on both axes, the joint
    triangulated height and beam positions, and the scale triple."""
    s = cfg.beams
    images = sky_images(data, sky)
    origins = cfg.origins()
    c = sky.centers
    (x0, x1), (y0, y1) = cfg.volume.viewer_crop_xy_m
    xgrid = np.linspace(x0 + 1.0, x1 - 1.0, 400)
    ygrid = np.linspace(y0 + 0.6, y1 - 0.6, 400)
    zs = np.linspace(s.z_scan_m[0], s.z_scan_m[1], s.n_z)
    cx, zx, rx = parallax_scan(images, origins, c, "x", xgrid, zs, s.band_x)
    cy, zy, ry = parallax_scan(images, origins, c, "y", ygrid, zs, s.band_y)
    sigma_t = float(c[1] - c[0]) / np.sqrt(12.0)
    tri = triangulate(images, origins, c, zx, sigma_t, s)
    z = tri["z"] if tri["ok"] else zx
    closure = scale_closure(images, origins, c, z)
    return {"ok": tri["ok"],
            "parallax_x_z": zx, "parallax_x_corr": rx, "parallax_y_z": zy, "parallax_y_corr": ry,
            "scan_z": zs.tolist(), "scan_corr_x": cx.tolist(), "scan_corr_y": cy.tolist(),
            **{k: v for k, v in tri.items() if k != "ok"},
            **{k: v for k, v in closure.items() if k != "z"},
            "n_beams_x": len(tri["beams_x"]), "n_beams_y": len(tri["beams_y"])}
