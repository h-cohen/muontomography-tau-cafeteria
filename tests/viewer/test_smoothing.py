"""Display smoothing of voxel cubes: 2x / 4x trilinear sub-cubes and an
isosurface. Display-only: Off is the original per-voxel render, hover keeps
reporting the solved voxel, and the legend (on screen and in the PNG) says
the view is interpolated."""

import json

import numpy as np
import pytest

from .conftest import assert_run_loaded, canvas_data

pytestmark = pytest.mark.browser

SHAPE = (8, 7, 6)


def _smooth_run(run_fixture):
    """A blob whose trilinear interpolant differs visibly from its voxels."""
    run = run_fixture(shape=SHAPE)
    x, y, z = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij")
    vol = np.exp(-((x - 3.5) ** 2 + (y - 3) ** 2 + (z - 2.5) ** 2) / 6.0).astype(np.float32)
    np.save(run / "volume.npy", vol)
    meta = json.loads((run / "meta.json").read_text())
    meta["value_range"] = [0.0, 1.0]
    meta["suggested_iso"] = [0.4, 0.8]
    (run / "meta.json").write_text(json.dumps(meta))
    return run


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    page.locator("#load-run-input").set_input_files(str(run))
    assert_run_loaded(page)


def _render(page):
    page.evaluate("() => window.__viewerState.idleNow()")
    return canvas_data(page)


def test_smoothing_is_offered_only_for_per_voxel_cubes(page, dist_path, run_fixture):
    _load(page, dist_path, _smooth_run(run_fixture))
    sel = page.locator("#smoothing")
    assert sel.input_value() == "off"
    assert sel.is_disabled(), "fog mode has nothing to smooth"
    page.locator("#render-mode").select_option("cubes")
    page.evaluate("() => window.__viewerState.idleNow()")
    assert sel.is_enabled()
    assert "adds no information" in sel.get_attribute("title")
    assert "solved 0.5 m" in sel.get_attribute("title")
    labels = sel.locator("option").all_inner_texts()
    assert labels == ["Off", "2× (0.25 m)", "4× (0.125 m)", "8× (0.0625 m)", "Isosurface"]
    page.locator("#cube-size").select_option("2")
    page.evaluate("() => window.__viewerState.idleNow()")
    assert sel.is_disabled(), "merged cubes are not smoothed"


def test_each_mode_changes_pixels_and_off_restores_the_original(page, dist_path, run_fixture):
    _load(page, dist_path, _smooth_run(run_fixture))
    page.locator("#render-mode").select_option("cubes")
    off = _render(page)
    seen = {"off": off}
    for mode in ["2", "4", "8", "iso"]:
        page.locator("#smoothing").select_option(mode)
        seen[mode] = _render(page)
    assert len(set(seen.values())) == 5, "every mode must draw differently"
    page.locator("#smoothing").select_option("off")
    assert _render(page) == off, "Off must be pixel-identical to the unsmoothed cubes"


def test_legend_labels_the_view_on_screen_and_in_the_png(page, dist_path, run_fixture):
    _load(page, dist_path, _smooth_run(run_fixture))
    legend = page.locator("#display-legend")
    page.locator("#render-mode").select_option("cubes")
    assert legend.is_hidden()
    page.locator("#smoothing").select_option("4")
    page.evaluate("() => window.__viewerState.idleNow()")
    assert legend.inner_text() == "display: trilinear 4× (solved 0.5 m)"
    exported = page.evaluate("() => window.__viewerState.exportImage()")
    assert "display: trilinear 4× (solved 0.5 m)" in exported["lines"]
    assert exported["dataURL"] != canvas_data(page), "the PNG draws the legend panel"
    page.locator("#smoothing").select_option("iso")
    page.evaluate("() => window.__viewerState.idleNow()")
    assert legend.inner_text() == "display: trilinear isosurface at 0.4 1/m (solved 0.5 m)"
    page.locator("#cube-size").select_option("2")
    page.evaluate("() => window.__viewerState.idleNow()")
    assert legend.is_hidden(), "no smoothing is drawn at cube size 2"
    lines = page.evaluate("() => window.__viewerState.exportImage().lines")
    assert not any(line.startswith("display:") for line in lines)


def test_fast_preview_drops_to_off_then_restores(page, dist_path, run_fixture):
    _load(page, dist_path, _smooth_run(run_fixture))
    page.locator("#render-mode").select_option("cubes")
    page.evaluate("() => { const s = window.__viewerState; s.interacting = true; s.render(); }")
    off_preview = canvas_data(page)
    page.locator("#smoothing").select_option("4")
    page.evaluate("() => { const s = window.__viewerState; s.interacting = true; s.render(); }")
    assert canvas_data(page) == off_preview, "preview renders unsmoothed cubes"
    assert page.locator("#display-legend").inner_text().startswith("display: trilinear 4×")
    smoothed = _render(page)
    assert smoothed != off_preview, "the idle render restores the smoothing"


def test_hover_reports_the_solved_voxel_value(page, dist_path, run_fixture):
    _load(page, dist_path, _smooth_run(run_fixture))
    page.locator("#render-mode").select_option("cubes")
    page.locator("#camera-preset-top").click()
    for mode in ["2", "4", "8", "iso"]:
        page.locator("#smoothing").select_option(mode)
        hit = page.evaluate("""() => {
            const s = window.__viewerState;
            s.idleNow();
            const r = document.querySelector('#gl-canvas').getBoundingClientRect();
            const h = s.pick(r.left + r.width / 2, r.top + r.height / 2);
            if (!h) return null;
            const [, ny, nz] = s.meta.shape;
            return { ...h, solved: s.layerData.get('volume')[h.i * ny * nz + h.j * nz + h.k] };
        }""")
        assert hit is not None, mode
        assert hit["value"] == hit["solved"], mode
