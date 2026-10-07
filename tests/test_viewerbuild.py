import base64
import json
from pathlib import Path

import pytest

from cafetomo.viewerbuild import build

VIEWER_DIR = Path(__file__).resolve().parents[1] / "viewer"


def test_build_produces_single_html_with_no_export_or_import(tmp_path):
    out = build(VIEWER_DIR, out_path=tmp_path / "index.html")
    assert out == tmp_path / "index.html"
    text = out.read_text()
    assert "<!DOCTYPE html>" in text or "<!doctype html>" in text
    assert "export function" not in text
    assert "export const" not in text
    assert "import {" not in text
    assert "parseNpy" in text
    assert "beamBoxVertices" in text
    assert "initViewer(document.body)" in text
    assert '<script id="embedded-run"' not in text


def test_embed_inlines_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    meta = json.dumps({"shape": [1, 1, 1]}).encode()
    (run / "meta.json").write_bytes(meta)
    (run / "volume.npy").write_bytes(b"\x93NUMPY")
    (run / "notes.txt").write_text("not part of the run")
    out = build(VIEWER_DIR, out_path=tmp_path / "index.html", embed_dir=run)
    text = out.read_text()
    assert '<script id="embedded-run" type="application/json">' in text
    assert base64.b64encode(meta).decode() in text
    start = text.index('<script id="embedded-run" type="application/json">')
    payload = text[start:].split(">", 1)[1].split("</script>", 1)[0]
    assert sorted(json.loads(payload)["files"]) == ["meta.json", "volume.npy"]
    # The bundle runs as soon as it is parsed, so the run must precede it.
    assert text.index('<script id="embedded-run"') < text.index("initViewer(document.body)")


def test_embed_of_a_directory_without_meta_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="meta.json"):
        build(VIEWER_DIR, out_path=tmp_path / "index.html", embed_dir=tmp_path)


def test_a_closing_script_tag_in_the_sources_fails_loudly(tmp_path):
    viewer = tmp_path / "viewer"
    (viewer / "src").mkdir(parents=True)
    (viewer / "shell.html").write_text((VIEWER_DIR / "shell.html").read_text())
    for src in (VIEWER_DIR / "src").iterdir():
        (viewer / "src" / src.name).write_text(src.read_text())
    (viewer / "src" / "npy.mjs").write_text("// see </script> here\n")
    with pytest.raises(ValueError, match="</script"):
        build(viewer, out_path=tmp_path / "index.html")
