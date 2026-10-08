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


def test_beam_opacity_and_color_change_only_the_overlay(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(beams=BEAMS))
    original = canvas_data(page)
    page.locator("#beam-opacity").evaluate(
        "el => { el.value = '0.5'; el.dispatchEvent(new Event('input')); }"
    )
    page.wait_for_timeout(50)
    half = canvas_data(page)
    assert half != original
    assert page.locator("#beam-opacity-readout").inner_text() == "50%"
    page.locator("#beam-opacity").evaluate(
        "el => { el.value = '0'; el.dispatchEvent(new Event('input')); }"
    )
    page.wait_for_timeout(50)
    transparent = canvas_data(page)
    page.locator("#toggle-beams").uncheck()
    page.wait_for_timeout(50)
    assert canvas_data(page) == transparent
    assert transparent != original
    assert transparent != half
    page.locator("#toggle-beams").check()
    page.locator("#beam-opacity").evaluate(
        "el => { el.value = '1'; el.dispatchEvent(new Event('input')); }"
    )
    page.locator("#beam-color").evaluate(
        "el => { el.value = '#00ff00'; el.dispatchEvent(new Event('input')); }"
    )
    page.wait_for_timeout(50)
    green = canvas_data(page)
    assert green != original
    assert green != transparent
    assert page.locator("#beam-opacity-readout").inner_text() == "100%"
    assert page.evaluate("() => window.__viewerState.opacity") == 1


def test_faded_beam_png_preserves_half_alpha_on_transparent_volume(page, dist_path, run_fixture):
    _load(page, dist_path, run_fixture(beams=BEAMS))
    page.locator("#opacity").evaluate(
        "el => { el.value = '0'; el.dispatchEvent(new Event('input')); }"
    )
    page.locator("#beam-opacity").evaluate(
        "el => { el.value = '0.5'; el.dispatchEvent(new Event('input')); }"
    )
    page.wait_for_timeout(50)
    alphas = page.evaluate("""async () => {
      const image = new Image();
      image.src = document.querySelector('#gl-canvas').toDataURL('image/png');
      await image.decode();
      const c = document.createElement('canvas'); c.width=image.width; c.height=image.height;
      const ctx=c.getContext('2d'); ctx.drawImage(image,0,0);
      const data=ctx.getImageData(0,0,c.width,c.height).data;
      const values=new Set(); for(let i=3;i<data.length;i+=4) values.add(data[i]);
      return [...values];
    }""")
    assert any(126 <= value <= 129 for value in alphas), alphas


def test_conditional_fit_defaults_to_vertical_depth_and_can_switch_baseline(
    page, dist_path, run_fixture
):
    import json

    run = run_fixture(beams=BEAMS)
    path = run / "meta.json"
    meta = json.loads(path.read_text())
    conditional = {
        **BEAMS,
        "conditional": True,
        "h": 1.45,
        "h_sigma": 0.11,
        "boxes": [{**box, "ztop": box["zbottom"] + 1.45} for box in BEAMS["boxes"]],
    }
    meta["beam_models"] = {"conditional": conditional, "matched": BEAMS}
    meta["beam_model_default"] = "conditional"
    path.write_text(json.dumps(meta))
    _load(page, dist_path, run)
    assert page.locator("#beam-model").input_value() == "conditional"
    assert "conditional z depth h = 1.45" in page.locator("#beam-legend").inner_text()
    assert page.evaluate("() => window.__viewerState.beamFaceVertexCount") == 72
    page.locator("#beam-model").select_option("matched")
    assert page.locator("#beam-legend").inner_text() == "beam depth h = 1.20 ± 0.15 m"
