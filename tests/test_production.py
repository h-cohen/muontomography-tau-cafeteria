"""Production paper sync: manifest, consistency check and typed-number lint."""

import json
from pathlib import Path

from cafetomo.production import MANIFEST, check, literal_numbers, write_manifest


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    results = tmp_path / "results"
    results.mkdir()
    (results / "beams.json").write_text('{"z": 6.9}')
    gen = tmp_path / "generated"
    (gen / "figures").mkdir(parents=True)
    (gen / "numbers.tex").write_text("\\newcommand{\\BeamsZ}{6.9}\n")
    (gen / "figures" / "height.pdf").write_bytes(b"%PDF-fake")
    return gen, results


def test_manifest_records_commit_results_and_files(tmp_path):
    gen, results = _tree(tmp_path)
    doc = write_manifest(gen, results=results, source_commit="abc123")
    assert doc == json.loads((gen / MANIFEST).read_text())
    assert doc["source_commit"] == "abc123"
    assert set(doc["results"]) == {"beams.json"}
    assert set(doc["files"]) == {"numbers.tex", "figures/height.pdf"}


def test_check_is_clean_right_after_writing(tmp_path):
    gen, results = _tree(tmp_path)
    write_manifest(gen, results=results, source_commit="abc123")
    assert check(gen, results=results) == []


def test_check_reports_edited_generated_file(tmp_path):
    gen, results = _tree(tmp_path)
    write_manifest(gen, results=results, source_commit="abc123")
    (gen / "numbers.tex").write_text("\\newcommand{\\BeamsZ}{7.3}\n")
    assert check(gen, results=results) == ["generated/numbers.tex differs from MANIFEST.json"]


def test_check_reports_added_and_missing_files(tmp_path):
    gen, results = _tree(tmp_path)
    write_manifest(gen, results=results, source_commit="abc123")
    (gen / "figures" / "height.pdf").unlink()
    (gen / "extra.tex").write_text("x")
    assert sorted(check(gen, results=results)) == [
        "generated/extra.tex is not in MANIFEST.json",
        "generated/figures/height.pdf is missing",
    ]


def test_check_reports_stale_results(tmp_path):
    gen, results = _tree(tmp_path)
    write_manifest(gen, results=results, source_commit="abc123")
    (results / "beams.json").write_text('{"z": 7.0}')
    (results / "pose.json").write_text("{}")
    assert sorted(check(gen, results=results)) == [
        "results/beams.json changed since the last sync",
        "results/pose.json changed since the last sync",
    ]


def test_check_without_manifest_fails(tmp_path):
    gen, results = _tree(tmp_path)
    assert check(gen, results=results) == [
        "generated/MANIFEST.json is missing: run make production-sync"
    ]


def test_literal_numbers_flags_typed_claims_only(tmp_path):
    sec = tmp_path / "sections"
    sec.mkdir()
    (sec / "a.tex").write_text(
        "\\section{Height}\\label{sec:h2}\n"
        "The ceiling is at 7.3 m and the depth $h=\\ArraydepthH$ m.\n"
        "% the old value was 6.5\n"
        "See Fig.~\\ref{fig:3} and \\cite{pdg2024}; width "
        "\\includegraphics[width=0.9\\columnwidth]{f1.pdf}.\n"
        "Two positions, \\num{\\PoseX} and 50\\% of bins.\n"
    )
    (sec / "hardware.tex").write_text("23 bars at 59 V\n")
    assert literal_numbers(tmp_path, ["sections/a.tex"]) == [
        "sections/a.tex:2: 7.3",
        "sections/a.tex:5: 50",
    ]
