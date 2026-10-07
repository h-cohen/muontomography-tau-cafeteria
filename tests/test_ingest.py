from dataclasses import replace

import numpy as np
import pytest
import uproot

from cafetomo.angular import load_counts
from cafetomo.config import Binning
from cafetomo.ingest import ingest, live_times, read_live_time, read_root_counts


def _write_th2(path, values, edges, with_dt=False):
    with uproot.recreate(path) as f:
        f["txty"] = (values.astype(np.float32), edges, edges)
        if with_dt:
            f["dT"] = (np.array([4.0, 2.0]), np.array([0.0, 1.0, 2.0]))


def test_crop_keeps_each_count_at_its_own_tangent(tmp_path):
    edges = np.linspace(-2, 2, 801)
    v = np.zeros((800, 800))
    v[150, 649] = 7          # tx bin [-1.25, -1.245), ty bin [1.245, 1.25)
    v[0, 0] = 99             # outside +-1.25: dropped
    _write_th2(tmp_path / "a.root", v, edges)
    target = Binning(t_max=1.25, n_bins=500).edges()
    h, total = read_root_counts(tmp_path / "a.root", "txty", target)
    assert h.values.shape == (500, 500)
    assert h.values[0, 499] == 7
    assert h.total == 7 and total == 106


def test_incompatible_binning_is_an_error_not_a_resample(tmp_path):
    edges = np.linspace(-2, 2, 801)
    _write_th2(tmp_path / "a.root", np.ones((800, 800)), edges)
    with pytest.raises(ValueError, match="common edge set"):
        read_root_counts(tmp_path / "a.root", "txty", np.linspace(-1.25, 1.25, 301))


def test_non_integer_counts_are_an_error(tmp_path):
    edges = np.linspace(-2, 2, 801)
    _write_th2(tmp_path / "a.root", np.full((800, 800), 0.5), edges)
    with pytest.raises(ValueError, match="non-negative counts"):
        read_root_counts(tmp_path / "a.root", "txty", Binning().edges())


def test_synthetic_ingest_writes_exposures_sky_and_live_times(cfg, tmp_path):
    edges = np.linspace(-2, 2, 801)
    data = tmp_path / "data"
    data.mkdir()
    names = [e.root_file for e in cfg.exposures] + [cfg.sky_reference.root_file]
    for i, name in enumerate(names):
        _write_th2(data / name, np.full((800, 800), i + 1.0), edges, with_dt=True)
    res = ingest(replace(cfg, data_dir=data), tmp_path / "out")
    assert [r.source_id for r in res] == ["pos0", "pos1", "SKY"]
    sky = load_counts(tmp_path / "out" / "counts_SKY.npz")
    assert sky.values.shape == (500, 500) and (sky.values == 3).all()
    assert live_times(tmp_path / "out", ["pos0"])["pos0"] == pytest.approx(4 * 0.5 + 2 * 1.5)


def test_ingest_without_dt_names_the_file(cfg, tmp_path):
    edges = np.linspace(-2, 2, 801)
    data = tmp_path / "data"
    data.mkdir()
    for e in cfg.exposures:
        _write_th2(data / e.root_file, np.ones((800, 800)), edges)
    with pytest.raises(ValueError, match="no 'dT' histogram"):
        ingest(replace(cfg, data_dir=data), tmp_path / "out")


def test_live_times_unknown_id_is_an_error(cfg, tmp_path):
    ingest(cfg, tmp_path)
    with pytest.raises(ValueError, match="nope"):
        live_times(tmp_path, ["pos0", "nope"])


def test_real_ingest_writes_counts_and_live_times(cfg, tmp_path):
    res = ingest(cfg, tmp_path)
    assert {r.source_id for r in res} == {"pos0", "pos1", "SKY"}
    lt = live_times(tmp_path, ["pos0", "pos1", "SKY"])
    assert all(v > 0 for v in lt.values())
    with np.load(tmp_path / "counts_SKY.npz") as d:
        assert d["values"].shape == (500, 500)


def test_missing_dt_raises(tmp_path):
    f = tmp_path / "nodt.root"
    with uproot.recreate(f) as out:
        out["txty"] = (np.ones((800, 800)), np.linspace(-2, 2, 801), np.linspace(-2, 2, 801))
    with pytest.raises((ValueError, KeyError), match="dT"):
        read_live_time(f)
