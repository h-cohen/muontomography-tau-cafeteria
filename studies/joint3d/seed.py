"""Seed a 3D baseline from a joint open-sky ingest: seed.py CONFIG JOINT_INGEST ROOT.

ROOT/baseline/ingest is a copy of the joint ingest (positions and reference counts as the
3D chain reads them) and ROOT/results/data.json its track counts and live hours, so that
`make RUNS=ROOT/baseline RESULTS=ROOT/results ...` starts at the pose fit.
"""

import json
import os
import shutil
import sys
from pathlib import Path

from cafetomo.angular import load_counts
from cafetomo.cli import _source_names
from cafetomo.config import load_config

config, src, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
dst, results = root / "baseline" / "ingest", root / "results"
if root.exists():
    raise SystemExit(f"{root} exists")
shutil.copytree(src, dst)
meta = json.loads((dst / "meta.json").read_text())
names = _source_names(load_config(config))
doc = {}
for sid, e in meta["exposures"].items():
    doc[f"{names[sid]}_tracks"] = int(load_counts(dst / f"counts_{sid}.npz").values.sum())
    doc[f"{names[sid]}_hours"] = e["live_time_s"] / 3600.0
results.mkdir(parents=True)
(results / "data.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
(root / "baseline" / "SEEDED_FROM").write_text(f"{src.resolve()}\n")
# newer than the config and code, so make takes the seeded ingest as built
for p in root.rglob("*"):
    os.utime(p)
print(doc)
