"""A build with the run embedded opens straight into the volume: the file
picker is never touched, and the result is the same as loading the run."""

import pytest

from cafetomo.viewerbuild import build

from .conftest import BEAMS, VIEWER_DIR, assert_run_loaded

pytestmark = pytest.mark.browser


def test_embedded_run_renders_without_the_file_picker(page, run_fixture, tmp_path):
    run = run_fixture(layers=("volume", "sigma", "snr"), beams=BEAMS)
    html = build(VIEWER_DIR, out_path=tmp_path / "dist" / "index.html", embed_dir=run)
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    page.goto(html.resolve().as_uri())
    assert_run_loaded(page)
    for key in ("volume", "sigma", "snr"):
        assert page.locator(f"#layer-{key}").count() == 1
    assert page.locator("#run-name").text_content() == "run"
    assert page.locator("#beam-legend").is_visible()
    assert console_errors == []
