"""Where on the sky lambda(SKY_B) - lambda(SKY_A) of pos0 sits: by tangent radius, and by
sign of each world tangent for |t| > 0.5. Pulls use the sky-run noise only."""

from pathlib import Path

import numpy as np

from cafetomo.cli import _grid
from cafetomo.config import load_config
from cafetomo.opacity import OpacityMaps, _accumulate, _sky_grid

r = Path("runs/skyswap")
cfg = load_config(r / "config.yaml", pose_file="results/pose.json")
L, N = {}, {}
for s in ("SKY_A", "SKY_B"):
    L[s] = OpacityMaps.load(r / f"opacity_{s}" / "maps.npz")
    g, lt = _grid(cfg, r / f"ingest_{s}")
    N[s] = _accumulate(g, cfg, _sky_grid(cfg), lt)["pos0"][2]
e = L["SKY_A"].sky.edges
c = (e[1:] + e[:-1]) / 2
X, Y = np.meshgrid(c, c, indexing="ij")
X, Y = X.ravel(), Y.ravel()
a, b = L["SKY_A"].lam["pos0"], L["SKY_B"].lam["pos0"]
na, nb = N["SKY_A"], N["SKY_B"]
ok = np.isfinite(a) & np.isfinite(b) & (na > 0) & (nb > 0)
p = np.where(ok, (b - a) / np.sqrt(1 / np.where(ok, na, 1) + 1 / np.where(ok, nb, 1)), np.nan)
t = np.hypot(X, Y)
for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]:
    m = ok & (t >= lo) & (t < hi)
    d = (b - a)[m]
    print(f"|t| {lo}-{hi}: bins {m.sum():4d}", end=" ")
    print(f"pull mean {p[m].mean():+.2f} std {p[m].std():.2f} dlam mean {d.mean():+.4f}")
for nm, m in [("sx>0", X > 0), ("sx<0", X < 0), ("sy>0", Y > 0), ("sy<0", Y < 0)]:
    m = ok & m & (t > 0.5)
    print(f"{nm} |t|>0.5: dlam mean {(b - a)[m].mean():+.4f} pull mean {p[m].mean():+.2f}")
