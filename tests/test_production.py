"""Production paper sync: manifest, consistency check and typed-number lint."""

import json
import subprocess
from pathlib import Path

import pytest

from cafetomo.production import MANIFEST, check, literal_numbers, sync, write_manifest


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


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """source repo (clean, one commit), bare 'overleaf' remote, production clone, staged dir."""
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "README").write_text("x")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "init")
    remote = tmp_path / "overleaf.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(remote), str(seed))
    (seed / "main.tex").write_text("\\input{generated/numbers}\n")
    _git(seed, "add", ".")
    _git(seed, "commit", "-qm", "template")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    prod = tmp_path / "prod"
    _git(tmp_path, "clone", "-q", str(remote), str(prod))
    staged = tmp_path / "stage"
    (staged / "figures").mkdir(parents=True)
    (staged / "numbers.tex").write_text("\\newcommand{\\BeamsZ}{6.9}\n")
    (staged / "figures" / "height.pdf").write_bytes(b"%PDF")
    results = src / "results"
    results.mkdir()
    (results / "beams.json").write_text("{}")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "results")
    return {"src": src, "seed": seed, "prod": prod, "staged": staged, "results": results}


def _sync(r):
    return sync(r["prod"], staged=r["staged"], results=r["results"], source_root=r["src"])


def test_sync_writes_generated_and_commits(repos):
    sha = _sync(repos)
    prod = repos["prod"]
    assert (prod / "generated" / "numbers.tex").read_text().startswith("\\newcommand")
    assert check(prod / "generated", results=repos["results"]) == []
    assert _git(prod, "rev-parse", "HEAD") == sha
    assert "chore: sync generated from cafetomo" in _git(prod, "log", "-1", "--format=%s")
    assert _git(prod, "status", "--porcelain") == ""


def test_sync_is_a_no_op_when_nothing_changed(repos):
    first = _sync(repos)
    assert _sync(repos) == first


def test_sync_fast_forwards_remote_edits(repos):
    seed = repos["seed"]
    (seed / "intro.tex").write_text("coauthor text")
    _git(seed, "add", ".")
    _git(seed, "commit", "-qm", "coauthor")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    _sync(repos)
    assert (repos["prod"] / "intro.tex").read_text() == "coauthor text"


def test_sync_refuses_diverged_history(repos):
    seed, prod = repos["seed"], repos["prod"]
    (seed / "a.tex").write_text("remote")
    _git(seed, "add", ".")
    _git(seed, "commit", "-qm", "remote")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    (prod / "b.tex").write_text("local")
    _git(prod, "add", ".")
    _git(prod, "commit", "-qm", "local")
    with pytest.raises(RuntimeError, match="diverged"):
        _sync(repos)


def test_sync_refuses_dirty_production(repos):
    (repos["prod"] / "main.tex").write_text("edited, not committed")
    with pytest.raises(RuntimeError, match="uncommitted changes in"):
        _sync(repos)


def test_sync_refuses_dirty_source(repos):
    (repos["src"] / "README").write_text("edited")
    with pytest.raises(RuntimeError, match="uncommitted changes in"):
        _sync(repos)


def test_sync_refuses_missing_stage(repos, tmp_path):
    with pytest.raises(FileNotFoundError):
        sync(
            repos["prod"],
            staged=tmp_path / "nope",
            results=repos["results"],
            source_root=repos["src"],
        )


def test_check_cli_fails_on_problems_and_only_warns_on_typed_numbers(tmp_path, capsys):
    from cafetomo.cli import main

    gen, results = _tree(tmp_path)
    (tmp_path / "3d.tex").write_text("The ceiling is at 7.3 m.\n")
    write_manifest(gen, results=results, source_commit="abc123")
    argv = ["production-check", "--production", str(tmp_path), "--results", str(results)]
    main([*argv, "--lint", "3d.tex"])
    assert "warning: typed number 3d.tex:1: 7.3" in capsys.readouterr().out
    (gen / "numbers.tex").write_text("edited")
    with pytest.raises(SystemExit) as exc:
        main([*argv, "--lint", "3d.tex"])
    assert exc.value.code == 1
    assert "error: generated/numbers.tex differs" in capsys.readouterr().out


def test_sync_ignores_its_own_submodule_pointer(repos, tmp_path):
    """Inside the analysis repo the production clone is a submodule; its moved
    pointer after a sync must not count as an uncommitted source change."""
    src = repos["src"]
    remote = tmp_path / "overleaf.git"
    _git(src, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(remote), "prod")
    _git(src, "commit", "-qm", "submodule")
    prod = src / "prod"
    _git(prod, "-c", "user.name=T", "-c", "user.email=t@t", "checkout", "-q", "main")
    first = sync(prod, staged=repos["staged"], results=repos["results"], source_root=src)
    (repos["staged"] / "numbers.tex").write_text("\\newcommand{\\BeamsZ}{7.0}\n")
    second = sync(prod, staged=repos["staged"], results=repos["results"], source_root=src)
    assert second != first


def test_sync_on_detached_head_says_how_to_fix_it(repos):
    """`git submodule update --init` leaves the clone detached."""
    prod = repos["prod"]
    _git(prod, "checkout", "-q", "--detach")
    with pytest.raises(RuntimeError, match="not on a branch: run git -C .* checkout main"):
        _sync(repos)


def test_git_failures_carry_the_git_message(repos):
    _git(repos["prod"], "remote", "set-url", "origin", str(repos["prod"].parent / "gone.git"))
    with pytest.raises(RuntimeError, match="git fetch .* failed: .*gone.git"):
        _sync(repos)


def test_literal_numbers_reports_a_missing_lint_file(tmp_path):
    assert literal_numbers(tmp_path, ["3d.tex"]) == ["3d.tex: file not found"]
