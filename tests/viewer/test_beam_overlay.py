"""The fitted beam boxes drawn over the volume, with their depth in the legend.

Shown by default when the run has a beams block, absent (control and legend
hidden) when it does not.
"""

import pytest

from .conftest import BEAMS, assert_run_loaded, canvas_data

pytestmark = pytest.mark.browser


def _load(page, dist_path, run):
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    page.locator("#load-run-input").set_input_files(str(run))
    assert_run_loaded(page)


def test_beam_boxes_draw_by_default_and_toggle_off(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(beams=BEAMS))
    toggle = page.locator("#toggle-beams")
    assert page.locator("#beams-control").is_visible()
    assert toggle.is_checked()
    assert page.locator("#beam-legend").inner_text() == "beam depth h = 1.20 ± 0.15 m"
    # 2 boxes x 12 edges x 2 endpoints
    assert page.evaluate("() => window.__viewerState.beamVertexCount") == 48

    with_boxes = canvas_data(page)
    toggle.uncheck()
    page.wait_for_timeout(50)
    without_boxes = canvas_data(page)
    assert without_boxes != with_boxes, "the beam boxes must change the rendered pixels"
    toggle.check()
    page.wait_for_timeout(50)
    assert canvas_data(page) == with_boxes, "re-enabling must draw the same boxes again"


def test_legend_without_an_error_budget_shows_the_bare_depth(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(beams={**BEAMS, "h_sigma": None}))
    assert page.locator("#beam-legend").inner_text() == "beam depth h = 1.20 m"


def test_no_beams_block_hides_the_control_and_legend(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture())
    assert page.locator("#beams-control").is_hidden()
    assert page.locator("#beam-legend").is_hidden()
    assert page.evaluate("() => window.__viewerState.showBeams") is False
