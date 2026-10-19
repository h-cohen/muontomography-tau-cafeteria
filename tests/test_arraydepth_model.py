"""array_model evaluates the conditional array fit exactly as the fit scored it."""

import numpy as np
import pytest

from cafetomo.arraydepth import array_model, fit_array_depth
from cafetomo.config import load_config
from cafetomo.fitdata import FitData, RowIndex


def _data(cfg):
    tx, ty = np.meshgrid(np.linspace(-0.7, 0.7, 15), np.linspace(-0.15, 0.15, 3))
    sx, sy = tx.ravel(), ty.ravel()
    rows = RowIndex(
        tuple(cfg.position_ids),
        np.repeat([0, 1], len(sx)),
        np.tile(sx, 2),
        np.tile(sy, 2),
        np.arange(2 * len(sx)),
    )
    lam = 0.1 + 0.05 * np.cos(2 * np.pi * rows.sx / 0.23) + 0.02 * rows.sy**2
    return FitData(lam, np.full(len(lam), 2500.0), rows)


def test_array_model_reproduces_fit_chi2():
    cfg = load_config("configs/cafeteria.yaml")
    data = _data(cfg)
    centres = np.array([-1.6, 0.0, 1.6, 3.2])
    rec = fit_array_depth(data, cfg, centres=centres, height_seed=7.0)
    m = array_model(data, cfg, rec, height_seed=7.0)
    assert m["chi2"] == pytest.approx(rec["chi2"], rel=1e-9)
    assert len(m["lam"]) == len(m["beam"]) == len(m["background"]) == len(m["pos"])
    assert set(m["pos"]) == set(cfg.position_ids)
