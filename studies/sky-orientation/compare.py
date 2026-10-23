"""Sky-swap comparison: lambda per position against SKY_A, SKY_B and the whole sky run.

The position counts are the same in every map, so the difference of two maps carries
only the sky-run noise: per sky bin var(dlam) ~ 1/n_A + 1/n_B (raw sky counts summed into
the bin). The pull dlam / sqrt(1/n_A + 1/n_B) is unit-width if the two orientations see
the same angular distribution.
"""

import json
import sys
from pathlib import Path

import numpy as np

from cafetomo.cli import _grid
from cafetomo.config import load_config
from cafetomo.opacity import OpacityMaps, _accumulate, _sky_grid

root = Path(sys.argv[1])
cfg = load_config(root / "config.yaml", pose_file="results/pose.json")
maps, nsky, live = {}, {}, {}
for s in ("SKY_A", "SKY_B", "SKY"):
    maps[s] = OpacityMaps.load(root / f"opacity_{s}" / "maps.npz")
    grid, lt = _grid(cfg, root / f"ingest_{s}")
    live[s] = lt
    acc = _accumulate(grid, cfg, _sky_grid(cfg), lt)
    nsky[s] = {p: v[2] for p, v in acc.items()}

out = {"live_h": {s: {k: v / 3600 for k, v in lt.items()} for s, lt in live.items()}}
for p in cfg.position_ids:
    a, b, w = (maps[s].lam[p] for s in ("SKY_A", "SKY_B", "SKY"))
    na, nb = nsky["SKY_A"][p], nsky["SKY_B"][p]
    ok = np.isfinite(a) & np.isfinite(b) & (na > 0) & (nb > 0)
    d = (b - a)[ok]
    pull = d / np.sqrt(1 / na[ok] + 1 / nb[ok])
    wt = 1 / (1 / na[ok] + 1 / nb[ok])
    out[p] = {
        "bins": int(ok.sum()),
        "mean_lam_full": float(np.nanmean(w[ok])),
        "dlam_B_minus_A_mean": float(d.mean()),
        "dlam_B_minus_A_wmean": float((d * wt).sum() / wt.sum()),
        "dlam_B_minus_A_rms": float(d.std()),
        "pull_mean": float(pull.mean()),
        "pull_std": float(pull.std()),
        "excess_rms_per_bin": float(np.sqrt(max(d.var() - np.mean(1 / na[ok] + 1 / nb[ok]), 0))),
        "full_minus_A_mean": float((w - a)[ok].mean()),
        "full_minus_B_mean": float((w - b)[ok].mean()),
    }
print(json.dumps(out, indent=1))
(root / "skyswap.json").write_text(json.dumps(out, indent=1))
