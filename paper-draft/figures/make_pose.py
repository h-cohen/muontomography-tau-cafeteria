"""Pose objective along each free coordinate through the fitted pose of pos1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import frame
import style
from cafetomo.cli import _grid
from cafetomo.config import load_config
from cafetomo.selfcal import pose_objective


def label(name: str) -> str:
    if name == "az":
        return r"$\varphi_1$ (deg)"
    return rf"${frame.ax(name)}_1$ (m)"


def scan_axes(pose: dict, n: int = 11) -> dict:
    return {
        "x": np.linspace(pose["x"] - 0.15, pose["x"] + 0.15, n),
        "y": np.linspace(pose["y"] - 0.30, pose["y"] + 0.30, n),
        "az": np.linspace(pose["az"] - 3.0, pose["az"] + 3.0, n),
    }


def main(argv=None) -> None:
    a, _ = frame.setup(argv)
    pose = style.load_result(a, "pose")
    prior = load_config(a.config)
    grid, live = _grid(prior, Path(a.runs) / "ingest")
    objective = pose_objective(grid, prior, live)
    best = {k: pose[k] for k in ("x", "y", "az")}
    order = ("x", "y", "az")

    fig, axes = plt.subplots(1, 3, figsize=(style.FULL_IN, 2.3), sharey=True, layout="constrained")
    for ax, (name, values) in zip(axes, scan_axes(best).items(), strict=True):
        scores = []
        for v in values:
            theta = np.array([v if k == name else best[k] for k in order])
            scores.append(objective(theta))
            print(f"pose profile {name}={v:.3f}: {scores[-1]:.4f}", flush=True)
        sigma = pose[f"{name}_sigma"]
        ax.axvspan(best[name] - sigma, best[name] + sigma, color=style.GRID, label=r"$\pm\sigma$")
        ax.plot(values, scores, "o-", ms=3, color=style.INK)
        ax.axvline(best[name], color=style.INK_SECONDARY, lw=0.8, ls="--", label="fitted")
        ax.set(xlabel=label(name))
    axes[0].set_ylabel(r"objective $\mathcal{J}$")
    axes[0].legend(loc="upper center")
    style.label_panels(axes)
    style.save(fig, a)


if __name__ == "__main__":
    main()
