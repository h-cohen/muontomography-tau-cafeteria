"""Angular count histograms over (tan θx, tan θy), axis 0 = tan θx, and the coarser
analysis grid the opacity is solved on."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AngularHist:
    """Raw int64 counts: the Poisson bootstrap and MLEM both need un-normalised
    integers, so nothing here may scale them."""

    values: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    name: str = "txty"

    @property
    def total(self) -> int:
        return int(self.values.sum())


def save_counts(hist: AngularHist, out_dir: Path, exposure_id: str, meta: dict) -> Path:
    """Write counts_<id>.npz and merge `meta` into the directory's meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"counts_{exposure_id}.npz"
    np.savez_compressed(
        path, values=hist.values, xedges=hist.xedges, yedges=hist.yedges, name=np.array(hist.name)
    )

    meta_path = out_dir / "meta.json"
    doc = json.loads(meta_path.read_text()) if meta_path.exists() else {"exposures": {}}
    doc.setdefault("exposures", {})[exposure_id] = dict(meta, total_counts=hist.total)
    meta_path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path


def load_counts(path: Path) -> AngularHist:
    with np.load(path) as d:
        return AngularHist(
            values=d["values"], xedges=d["xedges"], yedges=d["yedges"], name=str(d["name"])
        )


def rebin(hist: AngularHist, factor: int) -> AngularHist:
    """Sum `factor` x `factor` blocks of bins. Counts are conserved exactly.

    The fine 500x500 histograms average only a few counts per bin; a Poisson fit
    needs counts, not noise, so the analysis coarsens them first.
    """
    n = hist.values.shape[0]
    if n % factor:
        raise ValueError(f"factor {factor} must divide the bin count {n}")
    m = n // factor
    values = hist.values.reshape(m, factor, m, factor).sum(axis=(1, 3))
    edges = hist.xedges[::factor]
    return AngularHist(values=values.astype(np.int64), xedges=edges, yedges=edges, name=hist.name)


@dataclass(frozen=True)
class AnalysisGrid:
    edges: np.ndarray
    counts: dict[str, np.ndarray]

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def tan_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """tx, ty as [m, m] arrays. tx varies along axis 0, matching the
        histogram convention `np.histogram2d(tan_x, tan_y)`."""
        c = self.centers
        return np.meshgrid(c, c, indexing="ij")


def load_analysis_grid(
    ingest_dir: str | Path, ids: Sequence[str], factor: int = 10
) -> AnalysisGrid:
    ingest_dir = Path(ingest_dir)
    edges = None
    counts: dict[str, np.ndarray] = {}
    for eid in ids:
        h = rebin(load_counts(ingest_dir / f"counts_{eid}.npz"), factor)
        if edges is None:
            edges = h.xedges
        elif edges.shape != h.xedges.shape or not np.allclose(edges, h.xedges):
            raise ValueError(f"exposure {eid!r} has different binning from the others")
        counts[eid] = h.values
    if edges is None:
        raise ValueError("no exposures given")
    return AnalysisGrid(edges=edges, counts=counts)
