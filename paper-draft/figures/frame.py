"""Axis naming of the production paper's figures.

The analysis frame has the beams along y; the production paper names the axes
as the detector does (beams along x), so its figures swap x and y. Scripts take
an optional `--frame detector`; the draft paper's figures keep the analysis frame.
"""

import sys

import numpy as np

import style

FRAMES = ("analysis", "detector")
_frame = {"name": "analysis"}


def set_frame(name: str) -> None:
    if name not in FRAMES:
        raise ValueError(f"unknown frame {name!r}")
    _frame["name"] = name


def ax(name: str) -> str:
    """Axis name shown for an analysis-frame axis."""
    if _frame["name"] == "detector" and name in ("x", "y"):
        return {"x": "y", "y": "x"}[name]
    return name


def point(x: float, y: float) -> tuple[float, float]:
    """Plot coordinates (horizontal, vertical) of an analysis-frame (x, y) point."""
    return (y, x) if _frame["name"] == "detector" else (x, y)


def rect(x0: float, y0: float, width: float, height: float) -> tuple:
    """Rectangle arguments ((corner), width, height) of an analysis-frame box."""
    if _frame["name"] == "detector":
        return (y0, x0), height, width
    return (x0, y0), width, height


def top_view(img, xlim, ylim):
    """An [i_x, j_y] image and its imshow extent, with the shown x axis horizontal."""
    img = np.asarray(img)
    if _frame["name"] == "detector":
        return img, [ylim[0], ylim[1], xlim[0], xlim[1]]
    return img.T, [xlim[0], xlim[1], ylim[0], ylim[1]]


def setup(argv=None):
    """`style.setup` after removing and applying an optional `--frame NAME`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    name = "analysis"
    if "--frame" in argv:
        i = argv.index("--frame")
        name = argv[i + 1]
        del argv[i : i + 2]
    set_frame(name)
    return style.setup(argv)
