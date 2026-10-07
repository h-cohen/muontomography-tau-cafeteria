"""Every control exercised in one page session, with zero console errors.

The run carries every layer the exporter can write and a beams block, so no
control is left disabled for lack of data.
"""
import pytest

from .conftest import BEAMS, assert_run_loaded, canvas_data

pytestmark = pytest.mark.browser


def test_every_control_is_operable_without_console_errors(page, dist_path, run_fixture):
    run = run_fixture(layers=("volume", "sigma", "snr", "views", "rays"), beams=BEAMS)
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")
    page.locator("#load-run-input").set_input_files(str(run))
    assert_run_loaded(page)
    assert page.title() == "TAU cafeteria — muon tomography volume"

    page.locator("#colormap-select").select_option("inferno")

    clip_header = page.locator('[data-section="sec-clip"] .section-header')
    clip_header.click()
    assert page.locator('[data-section="sec-clip"]').get_attribute("data-open") == "false"
    clip_header.click()
    assert page.locator('[data-section="sec-clip"]').get_attribute("data-open") == "true"

    for key in ("volume", "sigma", "snr", "views", "rays"):
        assert page.locator(f"#layer-{key}").count() == 1, f"missing layer control for {key}"
        page.locator(f"#layer-{key}").check()
    page.locator("#layer-volume").check()

    page.locator("#camera-preset-top").click()
    page.locator("#camera-preset-iso").click()

    box = page.locator("#xfer-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.5)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.5)
    page.mouse.up()

    hist = page.locator("#histogram-canvas").bounding_box()
    page.mouse.move(hist["x"] + hist["width"] * 0.15, hist["y"] + hist["height"] / 2)
    page.mouse.down()
    page.mouse.move(hist["x"] + hist["width"] * 0.35, hist["y"] + hist["height"] / 2)
    page.mouse.up()
    assert page.locator("#window-readout").text_content().strip()

    page.locator("#clip-x-max").fill("0.6")
    page.locator("#clip-x-max").dispatch_event("input")
    page.locator("#clip-plane-enabled").check()
    page.locator("#slice-axis").select_option("z")
    page.locator("#slice-pos").fill("0.5")
    page.locator("#slice-pos").dispatch_event("input")
    page.locator("#sigma-gate-enabled").check()
    page.locator("#sigma-gate-value").fill("0.5")
    page.locator("#sigma-gate-value").dispatch_event("input")
    # The fixture's random snr (< 1) and rays (< 1) layers would hide every
    # voxel under the default gates; exercise the thresholds, then turn both
    # off so later checks see a populated volume.
    page.locator("#snr-gate-value").fill("0.5")
    page.locator("#snr-gate-value").dispatch_event("input")
    page.locator("#coverage-gate-value").fill("1")
    page.locator("#coverage-gate-value").dispatch_event("input")
    page.locator("#snr-gate-enabled").uncheck()
    page.locator("#coverage-gate-enabled").uncheck()
    page.locator("#render-mode").select_option("cubes")
    page.locator("#cube-size").select_option("2")
    page.locator("#render-mode").select_option("fog")

    canvas_box = page.locator("#gl-canvas").bounding_box()
    page.mouse.move(canvas_box["x"] + canvas_box["width"] / 2,
                    canvas_box["y"] + canvas_box["height"] / 2)
    page.wait_for_timeout(100)

    # Shortcut keys are ignored while a form control has focus.
    page.locator("#gl-canvas").click()
    page.keyboard.press("Shift+Slash")
    assert page.locator("#overlay-shortcuts").is_visible()
    page.keyboard.press("Escape")
    assert page.locator("#overlay-shortcuts").is_hidden()
    page.keyboard.press("2")

    page.locator("#view-name").fill("smoke-test-view")
    page.locator("#save-view-btn").click()
    saved = page.locator("#saved-views li", has_text="smoke-test-view")
    assert saved.count() == 1
    saved.get_by_role("button", name="Apply").click()

    with page.expect_download() as dl:
        page.locator("#export-png-btn").click()
    assert dl.value.suggested_filename == "cafetomo-volume-view.png"

    page.locator("#toggle-detectors").check()
    page.locator("#toggle-beams").uncheck()
    page.locator("#toggle-beams").check()

    before = canvas_data(page)
    page.locator("#camera-preset-top").click()
    page.locator("#camera-preset-observation").click()
    page.wait_for_timeout(50)
    assert canvas_data(page) != before
    cam = page.evaluate("() => window.__viewerState.camera")
    assert abs(cam["yaw"] - 0.6) < 1e-6 and abs(cam["pitch"] - 0.30) < 1e-6

    def clear_color():
        return page.evaluate(
            "() => { const gl = document.querySelector('#gl-canvas').getContext('webgl2'); "
            "return Array.from(gl.getParameter(gl.COLOR_CLEAR_VALUE)); }")
    theme = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
    before_clear = clear_color()
    page.locator("#theme-toggle").click()
    assert page.evaluate("() => document.documentElement.getAttribute('data-theme')") != theme
    assert clear_color() != before_clear
    page.locator("#theme-toggle").click()

    # Undo the slice, clip and gate first: on the thin z slice, shading and
    # smoothing have almost nothing to act on.
    page.locator("#slice-axis").select_option("none")
    page.locator("#clip-plane-enabled").uncheck()
    page.locator("#sigma-gate-enabled").uncheck()
    page.locator("#clip-x-max").fill("1")
    page.locator("#clip-x-max").dispatch_event("input")
    page.evaluate("() => window.__viewerState.idleNow()")
    for selector in ("#toggle-shading", "#toggle-smooth"):
        before = canvas_data(page)
        page.locator(selector).uncheck()
        page.wait_for_timeout(50)
        assert canvas_data(page) != before, selector
        page.locator(selector).check()

    assert console_errors == [], f"JS console errors during interaction: {console_errors}"
