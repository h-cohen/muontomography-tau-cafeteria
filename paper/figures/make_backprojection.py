"""Figure 3: model-free backprojection of the measured opacity onto the beam plane."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from cafetomo.backproject import backproject_plane, plane_axes
from cafetomo.opacity import OpacityMaps, build_fit_data, load_sigma

RES_M = 0.05


def main(argv=None) -> None:
    a, cfg = style.setup(argv)
    beams = style.load_result(a, "beams")
    opacity = Path(a.runs) / "opacity"
    data = build_fit_data(
        OpacityMaps.load(opacity / "maps.npz"), cfg, load_sigma(opacity / "sigma.npz")
    )
    z = beams["z"]
    xs, ys = plane_axes(cfg, z, cfg.opacity.max_tan, RES_M)
    _, mean = backproject_plane(data, cfg, z, xs, ys)

    fig, ax = plt.subplots(figsize=(style.COLUMN_IN, 2.9))
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    seen = np.isfinite(mean)
    if not seen.any():
        raise ValueError("backprojection figure has no measured ray intersections")
    low, high = np.percentile(mean[seen], [2, 98])
    # Show sampled intersections as visible markers, without interpolating
    # unmeasured pixels. All points remain; extreme colours are saturated.
    h = ax.scatter(
        xx[seen],
        yy[seen],
        c=mean[seen],
        s=6,
        edgecolors="none",
        cmap=style.SEQUENTIAL,
        vmin=low,
        vmax=high,
    )
    for k, xb in enumerate(beams["beams_x"]):
        ax.axvline(
            xb, color=style.INK, lw=0.5, ls="--", label="triangulated beams" if k == 0 else None
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"plane z = {z:.2f} m", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), handlelength=1.5)
    fig.colorbar(
        h,
        ax=ax,
        shrink=0.8,
        pad=0.03,
        extend="both",
        label=r"mean opacity $\lambda$ (dimensionless)",
    )
    style.save(fig, a)


if __name__ == "__main__":
    main()
