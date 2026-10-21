"""Autofocus-only rerun: same estimator, same replica stream, autofocus budget."""

import numpy as np
import pytest

from cafetomo import autofocusstudy, measure, systematics
from cafetomo.autofocus import FocusScan


def _scan(z):
    return FocusScan(zs=np.array([z - 0.1, z, z + 0.1]), scores=np.array([2.0, 1.0, 2.0]), z_best=z)


def test_measure_autofocus_only_skips_the_other_estimators(monkeypatch):
    monkeypatch.setattr(measure, "cv_height_scan", lambda *a, **k: _scan(7.0))

    def boom(*a, **k):
        raise AssertionError("beam depth must not run")

    monkeypatch.setattr(measure, "estimate_beam_depth", boom)
    m = measure.measure(None, None, None, autofocus_only=True)
    assert m.values == {"autofocus_z": 7.0}
    assert m.volume is None


def test_shift_functions_forward_autofocus_only(monkeypatch):
    seen = []

    def fake(*a, **k):
        seen.append(k.get("autofocus_only"))
        return measure.Measurement(values={"autofocus_z": 7.1}, volume=None)

    monkeypatch.setattr(systematics, "measure", fake)

    class Maps:
        def shifted(self, d):
            return self

    monkeypatch.setattr(systematics, "build_fit_data", lambda *a, **k: None)
    nominal = measure.Measurement(values={"autofocus_z": 7.0}, volume=None)
    cfg = type("C", (), {"uncertainty": type("U", (), {"flux_scale_frac": 0.03})()})()
    d = systematics.flux_scale_shift(
        Maps(), cfg, None, None, nominal, sky=None, autofocus_only=True
    )
    assert seen == [True]
    assert d == {"autofocus_z": pytest.approx(0.1)}


def test_budget_combines_stat_and_shifts_in_quadrature():
    doc = autofocusstudy.summarize(
        nominal=_scan(7.0),
        replicas=[6.9, 7.1, 7.0],
        shifts={"flux": 0.0, "mcs": 0.0, "pose": 0.2, "bg": 0.0},
        validation={"focus_bias": 0.05, "focus_max_error": 0.1},
        layer_thickness=0.3,
    )
    assert doc["z"] == 7.0 and doc["z_n"] == 3
    assert doc["z_stat"] == pytest.approx(0.1)
    assert doc["z_total"] == pytest.approx(np.hypot(0.1, 0.2))
    assert doc["layer_thickness"] == 0.3
    assert doc["scan_z"] == [6.9, 7.0, 7.1]
