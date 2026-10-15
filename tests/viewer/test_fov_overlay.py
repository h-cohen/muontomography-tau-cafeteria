"""Detector fields of view: the acceptance frustum of each position, drawn up
to the top of the volume. Off by default; offered only when the run carries
the tracker geometry."""

import json

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
    assert page.locator("#fov-opacity-readout").inner_text() == "60%"
    # 2 detectors x 8 triangles x 3 vertices
    assert page.evaluate("() => window.__viewerState.fovFaceVertexCount") == 48

    without = canvas_data(page)
    toggle.check()
    page.wait_for_timeout(50)
    with_fov = canvas_data(page)
    assert with_fov != without, "the field-of-view frusta must change the rendered pixels"
    legend = page.locator("#fov-legend").inner_text()
    assert legend == "field of view: |tan θ| ≤ 0.91 per axis (W/D), shaded by acceptance"
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


def test_flat_shading_draws_walls_instead_of_the_acceptance_field(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(detector_geometry=DETECTOR_GEOMETRY))
    page.locator("#toggle-fov").check()
    page.wait_for_timeout(50)
    shaded = canvas_data(page)
    page.locator("#fov-weighted").uncheck()
    page.wait_for_timeout(50)
    assert canvas_data(page) != shaded
    legend = page.locator("#fov-legend").inner_text()
    assert legend == "field of view: |tan θ| ≤ 0.91 per axis (W/D)"


def test_intersection_mode_draws_only_the_shared_region(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(detector_geometry=DETECTOR_GEOMETRY))
    without = canvas_data(page)
    page.locator("#toggle-fov").check()
    page.wait_for_timeout(50)
    separate = canvas_data(page)
    assert page.evaluate("() => window.__viewerState.fovRingVertexCount") > 0
    page.locator("#fov-mode").select_option("intersection")
    page.wait_for_timeout(50)
    shared = canvas_data(page)
    assert shared not in (separate, without)
    assert page.locator("#fov-legend").inner_text().startswith("shared field of view:")
    page.locator("#fov-weighted").uncheck()
    page.wait_for_timeout(50)
    assert canvas_data(page) not in (shared, without)
    _set(page, "#fov-opacity", "0")
    assert canvas_data(page) == without, "0% opacity hides the field and its rings"


def test_a_single_view_cannot_offer_an_intersection(page, dist_path, run_fixture):
    run = run_fixture(detector_geometry=DETECTOR_GEOMETRY)
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["detectors"] = meta["detectors"][:1]
    meta_path.write_text(json.dumps(meta))
    _load(page, dist_path, run)
    assert page.locator("#fov-mode option[value=intersection]").is_disabled()
