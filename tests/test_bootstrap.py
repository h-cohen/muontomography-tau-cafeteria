import numpy as np
import pytest

from cafetomo.angular import AnalysisGrid
from cafetomo.bootstrap import BootstrapResult, resample


def test_resample_preserves_shapes_and_means():
    edges = np.linspace(-1.0, 1.0, 6)
    mu = np.arange(1, 26, dtype=np.int64).reshape(5, 5) * 4
    grid = AnalysisGrid(edges=edges, counts={"pos0": mu, "sky": mu[::-1]})
    rng = np.random.default_rng(1)
    draws = [resample(grid, rng) for _ in range(200)]
    assert all(d.edges is edges for d in draws)
    for k, v in grid.counts.items():
        stack = np.stack([d.counts[k] for d in draws])
        assert stack.shape == (200, *v.shape) and stack.dtype == np.int64
        assert np.all(np.abs(stack.mean(axis=0) - v) <= 3 * np.sqrt(v / 200))


def _result(values):
    vol = np.zeros((2, 2, 2))
    return BootstrapResult(values={k: np.asarray(v, dtype=float) for k, v in values.items()},
                           volume_mean=vol, volume_sigma=vol)


def test_summary_keys_and_single_replica_sigma():
    s = _result({"depth_h": [1.2]}).summary()
    assert set(s) == {"depth_h_mean", "depth_h_sigma", "depth_h_n"}
    assert s["depth_h_mean"] == 1.2 and np.isnan(s["depth_h_sigma"]) and s["depth_h_n"] == 1


def test_summary_ignores_non_finite_replicas():
    s = _result({"zprofile_top": [8.0, np.nan, 8.2, np.inf]}).summary()
    assert s["zprofile_top_n"] == 2
    assert s["zprofile_top_mean"] == pytest.approx(8.1)
    assert s["zprofile_top_sigma"] == pytest.approx(np.std([8.0, 8.2], ddof=1))


def test_summary_with_no_finite_replica_is_nan_without_warning():
    with np.errstate(all="raise"):
        s = _result({"zprofile_top": [np.nan, np.nan]}).summary()
    assert s["zprofile_top_n"] == 0
    assert np.isnan(s["zprofile_top_mean"]) and np.isnan(s["zprofile_top_sigma"])


def test_summary_keys_are_digit_free():
    s = _result({"beams_z": [7.0, 7.1], "beams_zx": [7.0, 7.2]}).summary()
    assert not any(ch.isdigit() for k in s for ch in k)


def test_save_writes_values_and_volume_stats(tmp_path):
    r = BootstrapResult(values={"depth_h": np.array([1.0, 1.2])},
                        volume_mean=np.array([[[2.0, 0.0]]]), volume_sigma=np.array([[[0.5, 0.0]]]))
    r.save(tmp_path / "boot")
    with np.load(tmp_path / "boot" / "values.npz") as d:
        assert d["depth_h"].tolist() == [1.0, 1.2]
    with np.load(tmp_path / "boot" / "volume_stats.npz") as d:
        assert d["snr"][0, 0, 0] == 4.0 and np.isnan(d["snr"][0, 0, 1])
