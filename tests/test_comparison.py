"""compare-results: how a reproduction differs from the reference results."""

import json

import pytest

from cafetomo.comparison import compare_trees


def _write(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_identical_equal_different_and_missing(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"
    _write(ref, "a.json", {"x": 1.0})
    _write(new, "a.json", {"x": 1.0})
    _write(ref, "b.json", {"x": float("nan"), "y": [1, 2]})
    _write(new, "b.json", {"y": [1, 2], "x": float("nan")})
    _write(ref, "c.json", {"z": 7.0, "z_total": 0.5, "label": "k", "deep": {"v": 1.0}})
    _write(new, "c.json", {"z": 7.2, "z_total": 0.5, "label": "m", "deep": {"v": 1.5}})
    _write(ref, "only-ref.json", {})
    _write(new, "production/only-new.json", {})
    r = compare_trees(ref, new)
    assert r["files"]["a.json"]["status"] == "identical"
    assert r["files"]["b.json"]["status"] == "equal"
    c = r["files"]["c.json"]
    assert c["status"] == "different"
    z = next(d for d in c["numeric"] if d["path"] == "z")
    assert z["delta"] == pytest.approx(0.2) and z["scale"] == pytest.approx(0.5)
    assert z["delta_over_scale"] == pytest.approx(0.4)
    assert {d["path"] for d in c["numeric"]} == {"z", "deep.v"}
    assert c["other"] == ["label"]
    assert r["files"]["only-ref.json"]["status"] == "missing"
    assert r["extra"] == ["production/only-new.json"]


def test_macro_files_compare_by_value(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"
    _write(ref, "numbers.tex", "\\newcommand{\\A}{1.5}\n\\newcommand{\\B}{2}\n")
    _write(
        new, "numbers.tex", "\\newcommand{\\A}{1.6}\n\\newcommand{\\B}{2}\n\\newcommand{\\C}{3}\n"
    )
    m = compare_trees(ref, new)["files"]["numbers.tex"]
    assert m["status"] == "different"
    assert m["macros"] == [{"name": "A", "reference": "1.5", "candidate": "1.6"}]
    assert m["added"] == ["C"] and m["removed"] == []
