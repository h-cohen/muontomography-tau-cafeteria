"""Concatenates viewer/src/*.mjs into the single-file HTML deliverable.

Node's `node:test` imports these same .mjs files directly (unit tests run
against real ES module semantics); this module strips the export/import
syntax those files need for that and inlines them as one classic <script>
so the shipped page has zero network requests and works from file://.

With `embed_dir`, the run itself is inlined too, so the page is the whole
supplementary material: one file a reader opens, with nothing to pick.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

# Dependency order: a module may only import from a module earlier in this list.
_MODULE_ORDER = [
    "npy.mjs",
    "mat4.mjs",
    "grid.mjs",
    "transfer.mjs",
    "histogram.mjs",
    "clip.mjs",
    "camera.mjs",
    "layers.mjs",
    "dock.mjs",
    "colormap.mjs",
    "views.mjs",
    "shortcuts.mjs",
    "markers.mjs",
    "beams.mjs",
    "gates.mjs",
    "picker.mjs",
    "runload.mjs",
    "model.mjs",
    "app.mjs",
]

_EXPORT_RE = re.compile(r"^export (default )?")
_IMPORT_RE = re.compile(r"^\s*import\s*\{[^}]*\}\s*from\s*['\"][^'\"]+['\"];?\s*$")


def _strip_module_syntax(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if _IMPORT_RE.match(line):
            continue
        out_lines.append(_EXPORT_RE.sub("", line))
    return "\n".join(out_lines)


def _embedded_run_tag(embed_dir: Path) -> str:
    """The run's arrays and metadata as base64 inside a JSON <script> block,
    which the browser never executes; runload.mjs's embeddedFiles decodes it.
    A directory without meta.json is not a run, and the page would open
    empty, so it raises."""
    if not (embed_dir / "meta.json").is_file():
        raise FileNotFoundError(f"{embed_dir} has no meta.json: not an exported run")
    files = {p.name: base64.b64encode(p.read_bytes()).decode()
             for p in sorted(embed_dir.iterdir()) if p.suffix in {".npy", ".json"}}
    return ('<script id="embedded-run" type="application/json">'
            + json.dumps({"files": files}) + "</script>")


def build(viewer_dir: Path = Path("viewer"), *, out_path: Path,
          embed_dir: Path | None = None) -> Path:
    viewer_dir = Path(viewer_dir)
    src_dir = viewer_dir / "src"
    shell = (viewer_dir / "shell.html").read_text()

    chunks = []
    for name in _MODULE_ORDER:
        chunks.append(f"// ---- {name} ----")
        chunks.append(_strip_module_syntax((src_dir / name).read_text()))
    chunks.append("initViewer(document.body);")
    bundle = "\n".join(chunks)
    # The bundle is inlined into a classic <script>; a closing tag anywhere in
    # it, even in a comment, would end the script there and break the page.
    if re.search(r"</script", bundle, re.I):
        raise ValueError("viewer sources contain '</script': it would end the inline bundle")
    page = shell.replace("/* __VIEWER_BUNDLE__ */", bundle)

    if embed_dir is not None:
        # Right after <body>, ahead of the bundle's <script>: the bundle runs
        # as soon as it is parsed and must find the run already in the DOM.
        page = page.replace("<body>", "<body>\n" + _embedded_run_tag(Path(embed_dir)), 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)
    return out_path
