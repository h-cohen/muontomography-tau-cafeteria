"""Detector fields of view: the acceptance frustum of each position, drawn up
to the top of the volume. Off by default; offered only when the run carries
the tracker geometry."""

import pytest

from .conftest import DETECTOR_GEOMETRY, assert_run_loaded, canvas_data

pytestmark = pytest.mark.browser


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    page.locator("#load-run-input").set_input_files(str(run))
    assert_run_loaded(page)


def _set(page, selector, value):
    page.locator(selector).evaluate(
        f"el => {{ el.value = '{value}'; el.dispatchEvent(new Event('input')); }}"
    )
    page.wait_for_timeout(50)


def test_fov_is_offered_off_by_default_and_draws_when_checked(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(detector_geometry=DETECTOR_GEOMETRY))
    toggle = page.locator("#toggle-fov")
    assert page.locator("#fov-control").is_visible()
    assert not toggle.is_checked()
    assert page.locator("#fov-legend").is_hidden()
    # 2 detectors x 8 triangles x 3 vertices
    assert page.evaluate("() => window.__viewerState.fovFaceVertexCount") == 48

    without = canvas_data(page)
    toggle.check()
    page.wait_for_timeout(50)
    with_fov = canvas_data(page)
    assert with_fov != without, "the field-of-view frusta must change the rendered pixels"
    legend = page.locator("#fov-legend").inner_text()
    assert legend == "field of view: |tan θ| ≤ 0.91 per axis (W/D)"
    toggle.uncheck()
    page.wait_for_timeout(50)
    assert canvas_data(page) == without


def test_fov_opacity_and_color_change_only_the_overlay(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(detector_geometry=DETECTOR_GEOMETRY))
    without = canvas_data(page)
    page.locator("#toggle-fov").check()
    page.wait_for_timeout(50)
    full = canvas_data(page)
    _set(page, "#fov-opacity", "0.5")
    assert page.locator("#fov-opacity-readout").inner_text() == "50%"
    half = canvas_data(page)
    assert half not in (full, without)
    _set(page, "#fov-opacity", "0")
    assert canvas_data(page) == without, "0% opacity hides walls and edges"
    _set(page, "#fov-opacity", "1")
    _set(page, "#fov-color", "#00ff00")
    assert canvas_data(page) not in (full, without)
    assert page.evaluate("() => window.__viewerState.fovColor") == "#00ff00"


def test_run_without_tracker_geometry_hides_the_control(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture())
    assert page.locator("#fov-control").is_hidden()
    assert page.evaluate("() => window.__viewerState.showFov") is False
