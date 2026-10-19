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
import shutil
import subprocess
import tempfile
from pathlib import Path

MANIFEST = "MANIFEST.json"

# Arguments that hold labels, keys, paths or lengths, never claims.
_NON_CLAIM_ARGS = re.compile(
    r"\\(?:ref|eqref|label|cite\w*|includegraphics|input|include|usepackage|documentclass"
    r"|graphicspath|bibliography\w*|section\*?|subsection\*?|subsubsection\*?|paragraph\*?|hspace|vspace)\s*(?:\[[^\]]*\])?\{[^}]*\}"
)
_OPTIONS = re.compile(r"\[[^\]]*\]")
# A number scaling a page dimension is layout, not a claim.
_LAYOUT = r"(?!\\(?:text|line|column|paper)(?:width|height))"
_NUMBER = re.compile(r"(?<![\\\w.^])\d+(?:\.\d+)?(?![A-Za-z.\d])" + _LAYOUT)


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


_MATH_ENVS = r"(?:equation|align|gather|multline|eqnarray|algorithmic|displaymath|math)\*?"
_MATH_TOKEN = re.compile(
    rf"(\\\[|\\\]|(?<!\\)\$|\\begin\{{{_MATH_ENVS}\}}|\\end\{{{_MATH_ENVS}\}})"
)
# In math: decimals and multi-digit integers; single digits and braced arguments
# (sqrt{12}, x^{10}) are structural.
_MATH_NUMBER = re.compile(r"(?<![\\\w.^_{])(?:\d+\.\d+|\d{2,})(?![A-Za-z.\d}])")


def _claims(line: str, state: tuple[bool, bool]) -> tuple[list[str], tuple[bool, bool]]:
    """Numbers in one line that look like typed claims, and the (environment, inline)
    math state after it.

    Prose: every number. Math: decimals and multi-digit integers, since formulas
    carry structural single digits (1/2, 4(2+z^2)) and braced arguments (sqrt{12})."""
    env, inline = state
    hits = []
    for part in _MATH_TOKEN.split(line):
        token = _MATH_TOKEN.fullmatch(part)
        if token and (part == "\\[" or part.startswith("\\begin")):
            env = True
        elif token and (part == "\\]" or part.startswith("\\end")):
            env = False
        elif token:
            inline = not inline
        elif env or inline:
            hits += [m.group(0) for m in _MATH_NUMBER.finditer(part)]
        else:
            text = _OPTIONS.sub("", _NON_CLAIM_ARGS.sub("", part))
            hits += [m.group(0) for m in _NUMBER.finditer(text)]
    return hits, (env, inline)


def literal_numbers(tex_root: Path, files: list[str]) -> list[str]:
    hits = []
    for rel in files:
        path = tex_root / rel
        if not path.is_file():
            hits.append(f"{rel}: file not found")
            continue
        state = (False, False)
        for n, line in enumerate(path.read_text().splitlines(), 1):
            found, state = _claims(re.sub(r"(?<!\\)%.*", "", line), state)
            hits += [f"{rel}:{n}: {x}" for x in found]
    return hits


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)} in {repo} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def _require_clean(repo: Path, *exclude: str) -> None:
    pathspec = ["--", ".", *(f":(exclude){e}" for e in exclude)]
    if _git(repo, "status", "--porcelain", *pathspec):
        raise RuntimeError(f"uncommitted changes in {repo}: commit or stash them first")


def _fast_forward(production: Path) -> None:
    """Bring in Overleaf edits; never overwrite them."""
    if not _git(production, "branch", "--show-current"):
        raise RuntimeError(
            f"{production} is not on a branch: run git -C {production} checkout main"
        )
    _git(production, "fetch", "-q", "origin")
    upstream = _git(production, "rev-parse", "--abbrev-ref", "@{upstream}")
    behind, ahead = (
        int(n)
        for n in _git(
            production, "rev-list", "--left-right", "--count", f"{upstream}...HEAD"
        ).split()
    )
    if behind and ahead:
        raise RuntimeError(
            f"{production} and {upstream} have diverged ({ahead} local, {behind} remote commits): "
            "push or rebase the local commits first"
        )
    if behind:
        _git(production, "merge", "-q", "--ff-only", upstream)


def sync(production: Path, *, staged: Path, results: Path, source_root: Path) -> str:
    if not (staged / "numbers.tex").is_file():
        raise FileNotFoundError(f"{staged} has no numbers.tex: run make production-stage")
    # The production clone is usually a submodule of the source repository:
    # its pointer moves with every sync and is bumped on request, not here.
    root = source_root.resolve()
    inside = production.resolve().is_relative_to(root)
    _require_clean(
        source_root, *([production.resolve().relative_to(root).as_posix()] if inside else [])
    )
    _require_clean(production)
    _fast_forward(production)
    commit = _git(source_root, "rev-parse", "HEAD")
    generated = production / "generated"
    with tempfile.TemporaryDirectory(dir=production) as tmp:
        fresh = Path(tmp) / "generated"
        shutil.copytree(staged, fresh)
        write_manifest(fresh, results=results, source_commit=commit)
        if generated.exists():
            shutil.rmtree(generated)
        fresh.rename(generated)
    _git(production, "add", "-A", "generated")
    if not _git(production, "status", "--porcelain"):
        return _git(production, "rev-parse", "HEAD")
    _git(
        production,
        "-c",
        "user.name=Hadar Cohen",
        "-c",
        "user.email=hal.nls@gmail.com",
        "commit",
        "-q",
        "-m",
        f"chore: sync generated from cafetomo {commit[:7]}",
    )
    return _git(production, "rev-parse", "HEAD")
