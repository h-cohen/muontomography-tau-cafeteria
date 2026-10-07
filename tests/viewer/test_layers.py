import pytest

pytestmark = pytest.mark.browser


def test_layer_panel_lists_every_layer_and_switching_changes_render(page, dist_path, run_fixture):
    run_dir = run_fixture(layers=("volume", "sigma", "views"))
    page.goto(dist_path.resolve().as_uri())
    page.wait_for_selector("#gl-canvas")

    # A webkitdirectory input takes the run directory itself.
    page.locator("#load-run-input").set_input_files(str(run_dir))
    page.wait_for_function("() => window.__viewerState && window.__viewerState.ready")

    for key in ("volume", "sigma", "views"):
        assert page.locator(f"#layer-{key}").count() == 1

    before = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    page.locator("#layer-views").check()
    after = page.evaluate("() => document.querySelector('#gl-canvas').toDataURL()")
    assert before != after
