"""Shared fixtures for the viewer's Playwright tests.

`run_fixture` writes a tiny synthetic exported run (not campaign data), so
tests are fast and self-contained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cafetomo.viewerbuild import build

VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer"

# Two beams inside the default fixture volume (x 0..3, y 0..2.5, z 1..3 m),
# shaped like meta.json's `beams` block from cafetomo.export.
BEAMS = {
    "boxes": [{"x": x, "w": 0.3, "zbottom": 1.6, "ztop": 2.8, "y_extent": [0.2, 2.3]}
              for x in (0.9, 2.1)],
    "h": 1.2,
    "h_sigma": 0.15,
}


@pytest.fixture(scope="session")
def dist_path(tmp_path_factory) -> Path:
    return build(VIEWER_DIR, out_path=tmp_path_factory.mktemp("dist") / "index.html")


@pytest.fixture
def run_fixture(tmp_path):
    """Write a synthetic run dir; returns its Path. shape/layers/beams overridable."""
    def _make(shape=(6, 5, 4), layers=("volume",), spacing_m=0.5,
              origin_m=(0.0, 0.0, 1.0), depth_resolved=False, beams=None):
        run = tmp_path / "run"
        run.mkdir()
        rng = np.random.default_rng(0)
        vol = rng.random(shape, dtype=np.float32)
        np.save(run / "volume.npy", vol)
        for name in layers:
            if name == "volume":
                continue
            np.save(run / f"{name}.npy", rng.random(shape, dtype=np.float32))
        meta = {
            "shape": list(shape),
            "axis_order": "xyz",
            "origin_m": list(origin_m),
            "spacing_m": spacing_m,
            "units": "opacity density [1/m]",
            "value_range": [float(vol.min()), float(vol.max())],
            "suggested_iso": [float(vol.max()) * 0.3, float(vol.max()) * 0.6],
            "run": "run",
            "layers": list(layers),
            "detectors": [{"id": "pos0", "x": 0.0, "y": 0.0, "z": 0.0, "az_deg": 0.0},
                          {"id": "pos1", "x": 1.75, "y": 0.0, "z": 0.0, "az_deg": 0.0}],
            "resolution": {
                "max_baseline_m": 1.75,
                "n_positions": 2,
                "depth_resolved": depth_resolved,
                "verdict": "depth NOT resolved: synthetic fixture verdict"
                if not depth_resolved else "depth resolved: synthetic fixture verdict",
            },
        }
        if beams is not None:
            meta["beams"] = beams
        (run / "meta.json").write_text(json.dumps(meta))
        return run
    return _make


def canvas_data(page, selector="#gl-canvas") -> str:
    return page.evaluate(f"() => document.querySelector('{selector}').toDataURL()")


def canvas_is_blank(page, selector) -> bool:
    data = page.evaluate(f"""
        () => {{
            const src = document.querySelector('{selector}');
            const c = document.createElement('canvas');
            c.width = src.width; c.height = src.height;
            return [src.toDataURL(), c.toDataURL()];
        }}
    """)
    return data[0] == data[1]


def assert_run_loaded(page):
    """What every successful load must show: no load error, a drawn volume,
    gizmo and colour bar, and the resolution verdict unedited."""
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready",
                           timeout=15000)
    assert page.evaluate("() => window.__viewerError || null") is None
    assert not canvas_is_blank(page, "#gl-canvas"), "the volume must render"
    assert not canvas_is_blank(page, "#gizmo-canvas"), "axis gizmo must render"
    assert not canvas_is_blank(page, "#legend-canvas"), "colorbar legend must render"
    assert "depth NOT resolved" in page.locator("#resolution-banner").text_content()
    assert page.locator("#overlay-onboarding").is_hidden()


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Headless Chromium renders WebGL on SwiftShader (CPU), where the raymarch
    costs microseconds per pixel; tests exercise logic, not resolution, so a
    small viewport keeps every frame cheap. The shipped viewer is unaffected."""
    return {**browser_context_args, "viewport": {"width": 760, "height": 520}}
