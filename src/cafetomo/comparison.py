"""How a reproduction's results differ from the reference results.

Every JSON and macro file under the reference root is compared with the same
path under the candidate root: byte-identical, equal as data (NaN equals NaN,
key order ignored), or different, with each differing number's reference
value, candidate value and delta. A scalar `k` with a sibling `k_total` or
`k_sigma` gets that uncertainty as its scale, so a difference can be read
against the quoted precision. Nothing is tolerated or rounded away.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}")


def _is_number(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _same(a, b) -> bool:
    if _is_number(a) and _is_number(b):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _walk(a, b, path: str, numeric: list, other: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                other.append(p)
            else:
                _walk(a[k], b[k], p, numeric, other)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _walk(x, y, f"{path}[{i}]", numeric, other)
    elif _is_number(a) and _is_number(b):
        if not _same(a, b):
            numeric.append({"path": path, "reference": a, "candidate": b, "delta": b - a})
    elif not _same(a, b):
        other.append(path)


def _scale(doc: dict, path: str) -> float | None:
    if "." in path or "[" in path:
        return None
    for suffix in ("_total", "_sigma"):
        v = doc.get(path + suffix)
        if _is_number(v) and math.isfinite(v) and v > 0:
            return float(v)
    return None


def _compare_json(ref: Path, new: Path) -> dict:
    a, b = json.loads(ref.read_text()), json.loads(new.read_text())
    numeric: list = []
    other: list = []
    _walk(a, b, "", numeric, other)
    if not numeric and not other:
        return {"status": "equal"}
    for d in numeric:
        s = _scale(a, d["path"]) if isinstance(a, dict) else None
        d["scale"] = s
        d["delta_over_scale"] = d["delta"] / s if s else None
    return {"status": "different", "numeric": numeric, "other": other}


def _macros(path: Path) -> dict[str, str]:
    return dict(_MACRO.findall(path.read_text()))


def _compare_tex(ref: Path, new: Path) -> dict:
    a, b = _macros(ref), _macros(new)
    changed = [
        {"name": k, "reference": a[k], "candidate": b[k]}
        for k in sorted(a)
        if k in b and a[k] != b[k]
    ]
    added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    if not (changed or added or removed):
        return {"status": "equal"}
    return {"status": "different", "macros": changed, "added": added, "removed": removed}


def _files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".json", ".tex")
    }


def compare_trees(reference: Path, candidate: Path) -> dict:
    reference, candidate = Path(reference), Path(candidate)
    ref_files, new_files = _files(reference), _files(candidate)
    files = {}
    for rel in sorted(ref_files):
        a, b = reference / rel, candidate / rel
        if rel not in new_files:
            files[rel] = {"status": "missing"}
        elif a.read_bytes() == b.read_bytes():
            files[rel] = {"status": "identical"}
        elif a.suffix == ".json":
            files[rel] = _compare_json(a, b)
        else:
            files[rel] = _compare_tex(a, b)
    return {"files": files, "extra": sorted(new_files - ref_files)}


def summary_lines(report: dict) -> list[str]:
    lines = []
    for rel, f in report["files"].items():
        line = f"{f['status']:>10}  {rel}"
        if f["status"] == "different" and "numeric" in f:
            worst = max(
                (abs(d["delta_over_scale"]) for d in f["numeric"] if d["delta_over_scale"]),
                default=None,
            )
            line += f"  ({len(f['numeric'])} numbers, {len(f['other'])} other"
            line += f"; largest |delta|/uncertainty {worst:.2g})" if worst else ")"
        elif f["status"] == "different":
            n = (len(f["macros"]), len(f["added"]), len(f["removed"]))
            line += f"  ({n[0]} changed, {n[1]} added, {n[2]} removed)"
        lines.append(line)
    lines += [f"{'extra':>10}  {rel}" for rel in report["extra"]]
    return lines
