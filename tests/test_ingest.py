import json
from dataclasses import replace

import numpy as np
import pytest
import uproot

from cafetomo.angular import load_counts
from cafetomo.config import Binning
from cafetomo.ingest import (
    ingest,
    interpolated_live_time,
    live_times,
    read_live_time,
    read_root_counts,
)


def _write_th2(path, values, edges, with_dt=False):
    with uproot.recreate(path) as f:
        f["txty"] = (values.astype(np.float32), edges, edges)
        if with_dt:
            f["dT"] = (np.array([4.0, 2.0]), np.array([0.0, 1.0, 2.0]))


def test_crop_keeps_each_count_at_its_own_tangent(tmp_path):
    edges = np.linspace(-2, 2, 801)
    v = np.zeros((800, 800))
    v[150, 649] = 7  # tx bin [-1.25, -1.245), ty bin [1.245, 1.25)
    v[0, 0] = 99  # outside +-1.25: dropped
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


def test_failure_on_a_later_source_writes_nothing(cfg, tmp_path):
    edges = np.linspace(-2, 2, 801)
    data = tmp_path / "data"
    data.mkdir()
    for e in cfg.exposures:
        _write_th2(data / e.root_file, np.ones((800, 800)), edges, with_dt=True)
    _write_th2(data / cfg.sky_reference.root_file, np.ones((800, 800)), edges)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no 'dT' histogram"):
        ingest(replace(cfg, data_dir=data), out)
    assert not out.exists() or not list(out.iterdir())


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


def test_interpolated_live_time_removes_the_bin_centre_bias():
    rng = np.random.default_rng(1)
    dt = rng.exponential(0.14, 400_000)
    edges = np.arange(0.0, 20.0001, 0.05)
    counts, _ = np.histogram(dt, edges)
    centre = float((0.5 * (edges[:-1] + edges[1:]) * counts).sum())
    assert centre / dt.sum() - 1 > 0.008  # bin centres overstate the time by ~1%
    assert interpolated_live_time(counts, edges) == pytest.approx(dt.sum(), rel=1e-3)


def test_interpolated_live_time_is_exact_for_one_bin_of_flat_density():
    assert interpolated_live_time(np.array([10.0]), np.array([0.0, 2.0])) == pytest.approx(10.0)


def test_ingest_with_raw_counts_replaces_only_those_exposures(cfg, tmp_path):
    from cafetomo.angular import AngularHist
    from cafetomo.rawdata import FileCounts, RawIngest

    edges = np.linspace(-2, 2, 801)
    data = tmp_path / "data"
    data.mkdir()
    for i, name in enumerate([e.root_file for e in cfg.exposures] + [cfg.sky_reference.root_file]):
        _write_th2(data / name, np.full((800, 800), i + 1.0), edges, with_dt=True)
    cut = cfg.binning.edges()
    hist = AngularHist(np.full((500, 500), 5, dtype=np.int64), cut, cut, "txty")
    z = np.zeros((4, 32, 1), np.int64)
    file = FileCounts(
        "f.data", np.zeros((800, 800), np.int64), 10, 9, 1, 0, 0, 12.5, 10, z, z, z[..., 0], 1
    )
    raw = {"pos0": RawIngest(hist, 9 * 500 * 500, 12.5, (file,), 0)}
    c = replace(
        cfg, data_dir=data, raw=replace(cfg.raw, exposures={"pos0": str(tmp_path)}, subsets={})
    )
    res = ingest(c, tmp_path / "out", raw)
    assert (load_counts(tmp_path / "out" / "counts_pos0.npz").values == 5).all()
    assert (load_counts(tmp_path / "out" / "counts_pos1.npz").values == 2).all()
    live = live_times(tmp_path / "out", ["pos0", "pos1"])
    assert live["pos0"] == 12.5 and live["pos1"] < 2 * 0.5 + 4 * 1.5  # below the bin-centre sum
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())["exposures"]
    assert meta["pos0"]["live_time_estimator"] == "exact clock sum"
    assert meta["SKY"]["live_time_estimator"] == "interpolated dT sum"
    assert [r.source_id for r in res] == ["pos0", "pos1", "SKY"]
    with pytest.raises(ValueError, match="unknown exposure"):
        ingest(c, tmp_path / "out2", {"nope": raw["pos0"]})


def test_raw_counts_may_replace_the_sky_reference_too(cfg, tmp_path):
    from cafetomo.angular import AngularHist
    from cafetomo.rawdata import FileCounts, RawIngest

    edges = np.linspace(-2, 2, 801)
    data = tmp_path / "data"
    data.mkdir()
    for i, name in enumerate([e.root_file for e in cfg.exposures] + [cfg.sky_reference.root_file]):
        _write_th2(data / name, np.full((800, 800), i + 1.0), edges, with_dt=True)
    cut = cfg.binning.edges()
    z = np.zeros((4, 32, 1), np.int64)
    file = FileCounts(
        "s.data", np.zeros((800, 800), np.int64), 10, 9, 1, 0, 0, 99.0, 10, z, z, z[..., 0], 1
    )
    sky = RawIngest(
        AngularHist(np.full((500, 500), 7, dtype=np.int64), cut, cut, "txty"), 1, 99.0, (file,), 0
    )
    c = replace(cfg, data_dir=data, raw=replace(cfg.raw, exposures={"SKY": str(tmp_path)}))
    ingest(c, tmp_path / "out", {"SKY": sky})
    assert (load_counts(tmp_path / "out" / "counts_SKY.npz").values == 7).all()
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())["exposures"]
    assert meta["SKY"]["live_time_s"] == 99.0 and meta["SKY"]["role"] == "open_sky_reference"
    assert meta["pos0"]["live_time_estimator"] == "interpolated dT sum"
