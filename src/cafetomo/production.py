"""The production paper (Overleaf) reads numbers and figures generated here.

`generated/` in the production repo is written only by the sync; its
MANIFEST.json names the source commit and hashes every result it was made
from and every file it holds, so an edit in Overleaf or a later change to
results/ is caught instead of silently publishing stale numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

MANIFEST = "MANIFEST.json"

# Arguments that hold labels, keys, paths or lengths, never claims.
_NON_CLAIM_ARGS = re.compile(
    r"\\(?:ref|eqref|label|cite\w*|includegraphics|input|include|usepackage|documentclass"
    r"|graphicspath|bibliography\w*|section\*?|subsection\*?|hspace|vspace)\s*(?:\[[^\]]*\])?\{[^}]*\}"
)
_OPTIONS = re.compile(r"\[[^\]]*\]")
_NUMBER = re.compile(r"(?<![\\\w.])\d+(?:\.\d+)?")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _results(results: Path) -> dict[str, str]:
    return {p.name: _sha(p) for p in sorted(results.glob("*.json"))}


def _files(generated: Path) -> dict[str, str]:
    return {
        p.relative_to(generated).as_posix(): _sha(p)
        for p in sorted(generated.rglob("*"))
        if p.is_file() and p.name != MANIFEST
    }


def write_manifest(generated: Path, *, results: Path, source_commit: str) -> dict:
    doc = {
        "source_commit": source_commit,
        "results": _results(results),
        "files": _files(generated),
    }
    (generated / MANIFEST).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def check(generated: Path, *, results: Path) -> list[str]:
    path = generated / MANIFEST
    if not path.is_file():
        return [f"generated/{MANIFEST} is missing: run make production-sync"]
    doc = json.loads(path.read_text())
    problems = []
    now = _files(generated)
    for rel, sha in doc["files"].items():
        if rel not in now:
            problems.append(f"generated/{rel} is missing")
        elif now[rel] != sha:
            problems.append(f"generated/{rel} differs from {MANIFEST}")
    problems += [f"generated/{rel} is not in {MANIFEST}" for rel in now if rel not in doc["files"]]
    current = _results(results)
    for name in sorted(set(current) | set(doc["results"])):
        if current.get(name) != doc["results"].get(name):
            problems.append(f"results/{name} changed since the last sync")
    return problems


def literal_numbers(tex_root: Path, files: list[str]) -> list[str]:
    hits = []
    for rel in files:
        path = tex_root / rel
        for n, line in enumerate(path.read_text().splitlines(), 1):
            text = re.sub(r"(?<!\\)%.*", "", line)
            text = _OPTIONS.sub("", _NON_CLAIM_ARGS.sub("", text))
            hits += [f"{rel}:{n}: {m.group(0)}" for m in _NUMBER.finditer(text)]
    return hits
