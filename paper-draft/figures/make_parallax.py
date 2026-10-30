"""Ceiling-plane height: parallax correlation, projected shadows, angular spectra and the
corrected data-driven autofocus (cross-position prediction score versus layer height)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from cafetomo.beams import angular_spectrum, profile, sky_images, world_profile
from cafetomo.cli import _fit_data


def projected_profiles(t, profiles: dict, origins: dict, *, z: float, n: int = 400):
    """Each position's x-profile on one world grid of the plane at height z."""
    lo = min(o[0] + t.min() * (z - o[2]) for o in origins.values())
    hi = max(o[0] + t.max() * (z - o[2]) for o in origins.values())
    grid = np.linspace(lo, hi, n)
    return grid, {
        pid: world_profile(t, p, origins[pid], "x", z, grid) for pid, p in profiles.items()
    }


def relative_score(score) -> np.ndarray:
    """Score excess over its minimum, as a fraction of the minimum."""
    s = np.asarray(score, dtype=float)
    return s / s.min() - 1.0


def main(argv=None) -> None:
    a, cfg = frame.setup(argv)
    beams = style.load_result(a, "beams")
    maps, _, data = _fit_data(cfg, Path(a.runs) / "opacity")
    images = sky_images(data, maps.sky)
    centers = maps.sky.centers
    band = cfg.beams.band_x
    origins = cfg.origins()

    fig, ((ax, bx), (cx, dx)) = plt.subplots(
        2, 2, figsize=(style.FULL_IN, 4.6), layout="constrained"
    )
    ax.plot(
        beams["scan_z"], beams["scan_corr_x"], color=style.INK, label=rf"$\rho_{frame.ax('x')}$"
    )
    ax.plot(
        beams["scan_z"], beams["scan_corr_y"], color=style.MUTED, label=rf"$\rho_{frame.ax('y')}$"
    )
    ax.axvline(beams["z_x"], color=style.INK, ls="--", lw=0.8, label=rf"$z_{frame.ax('x')}$")
    ax.axvline(beams["z"], color=style.INK, ls=":", lw=0.8, label=r"joint $z$")
    ax.set(xlabel="trial ceiling height (m)", ylabel="parallax correlation")
    ax.legend(loc="lower center", ncols=2)

    profiles = {pid: profile(images[pid], centers, "x", band) for pid in data.rows.position_ids}
    t = next(iter(profiles.values()))[0]
    grid, proj = projected_profiles(
        t, {pid: p for pid, (_, p) in profiles.items()}, origins, z=beams["z"]
    )
    for pid, p in proj.items():
        bx.plot(grid, p, color=style.POSITION_COLORS[pid], label=pid)
    for x in beams["beams_x"]:
        bx.axvline(x, color=style.GRID, lw=0.8, zorder=0)
    bx.set(
        xlabel=rf"${frame.ax('x')}$ at joint $z$ (m)", ylabel=r"opacity $\lambda$ (dimensionless)"
    )
    bx.legend(loc="upper right")

    for pid in data.rows.position_ids:
        freq, amp = angular_spectrum(images[pid], centers, band=band)
        cx.plot(
            freq, amp / amp[2:].max(), "o-", ms=2.5, color=style.POSITION_COLORS[pid], label=pid
        )
        cx.axvline(1 / beams["period_per_position"][pid], color=style.POSITION_COLORS[pid], lw=0.8)
    cx.set(xlabel=r"angular frequency (1/tangent unit)", ylabel="normalized FFT amplitude")
    cx.legend(loc="upper right")

    af = json.loads((Path(a.results) / "autofocuslayer.json").read_text())
    dx.axvspan(
        af["z"] - af["z_total"], af["z"] + af["z_total"], color=style.GRID, label="evaluated total"
    )
    dx.plot(af["scan_z"], relative_score(af["scan_score"]), "o-", ms=3, color=style.INK)
    dx.axvline(af["z"], color=style.INK, ls="--", lw=0.8, label="autofocus")
    dx.axvline(
        beams["z_x"],
        color=style.POSITION_COLORS["pos1"],
        ls=":",
        lw=0.8,
        label=rf"triangulated $z_{frame.ax('x')}$",
    )
    dx.set(xlabel="trial layer height (m)", ylabel=r"$\mathcal{C}/\mathcal{C}_{\min}-1$")
    dx.legend(loc="upper center")
    style.label_panels((ax, bx, cx, dx))
    style.save(fig, a)


if __name__ == "__main__":
    main()
