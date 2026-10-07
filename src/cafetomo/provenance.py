"""Provenance record written next to every stage output."""

import hashlib
import json
import subprocess
from pathlib import Path

from cafetomo import __version__


def _git_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    return commit + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def write_meta(out_dir: str | Path, *, config_path: str | Path, stage: str,
               extra: dict | None = None) -> Path:
    """meta.json: which code and which config produced this directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "stage": stage,
        "cafetomo_version": __version__,
        "git_commit": _git_commit(),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:16],
        **(extra or {}),
    }
    path = out / "meta.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path
