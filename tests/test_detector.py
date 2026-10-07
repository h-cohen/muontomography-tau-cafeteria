import numpy as np
import pytest

from cafetomo.detector import Detector


def test_aperture_and_edge():
    d = Detector(active_width_cm=35.375, layer_dz_cm=38.9)
    assert d.aperture_m == pytest.approx(0.35375)
    assert d.max_tan == pytest.approx(35.375 / 38.9)


def test_acceptance_vanishes_at_edge_and_peaks_at_zenith():
    d = Detector(active_width_cm=35.375, layer_dz_cm=38.9)
    t = np.array([0.0, 0.5, d.max_tan, 1.2])
    acc = d.acceptance(t, np.zeros_like(t))
    assert acc[0] == acc.max()
    assert acc[2] == pytest.approx(0.0, abs=1e-9)
    assert acc[3] == 0.0
