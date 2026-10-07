"""The repository is self-contained: no trace of predecessor projects."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"\b(megid\w*|muontomo|cafeteria_3d\w*)\b", re.I)


def test_no_predecessor_names():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    hits = []
    for rel in tracked:
        if rel == "tests/test_repo_hygiene.py":
            continue
        path = ROOT / rel
        if path.suffix in {".root", ".png", ".pdf"} or not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        hits += [f"{rel}: {m.group(0)}" for m in FORBIDDEN.finditer(text)]
    assert not hits, "\n".join(hits)
