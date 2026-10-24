import numpy as np
import pytest

from cafetomo.angular import AngularHist
from cafetomo.rawcheck import count_match, epoch_scatter, lambda_shift

EDGES = np.linspace(-1.25, 1.25, 501)
RNG = np.random.default_rng(5)


def _hist(values):
    return AngularHist(np.asarray(values, dtype=np.int64), EDGES, EDGES, "txty")


def _smooth_field(scale=200.0):
    t = 0.5 * (EDGES[:-1] + EDGES[1:])
    return scale * np.exp(-(t[:, None] ** 2 + t[None, :] ** 2) / 0.5)


def test_identical_counts_match_exactly():
    h = _hist(RNG.poisson(_smooth_field()))
    chi2, scale = count_match(h, h)
    assert chi2 == 0.0 and scale == 1.0


def test_counts_from_one_shape_match_at_about_one_and_a_flip_does_not():
    mu = _smooth_field()
    mu[:250] *= 1.3  # break the mirror symmetry
    a, b = _hist(RNG.poisson(mu)), _hist(RNG.poisson(0.5 * mu))
    chi2, scale = count_match(a, b)
    assert chi2 == pytest.approx(1.0, abs=0.1) and scale == pytest.approx(2.0, rel=0.01)
    assert count_match(_hist(a.values[::-1]), b)[0] > 2.0


def test_epoch_scatter_is_noise_for_one_shape_and_more_for_a_gradient():
    mu = _smooth_field(400.0)
    days = [_hist(RNG.poisson(mu)) for _ in range(6)]
    std, noise = epoch_scatter(days[:3], days[3:])
    assert std == pytest.approx(noise, rel=0.15)
    t = 0.5 * (EDGES[:-1] + EDGES[1:])
    tilted = [_hist(RNG.poisson(mu * (1 + 0.15 * t[None, :]))) for _ in range(3)]
    std, noise = epoch_scatter(days[:3], tilted)
    assert std > 1.5 * noise


def test_lambda_shift_over_bins_measured_in_both():
    a = np.array([0.1, 0.2, np.nan, 0.4])
    b = np.array([0.12, 0.18, 0.3, np.nan])
    mean, std, n = lambda_shift(a, b)
    assert n == 2 and mean == pytest.approx(0.0) and std == pytest.approx(0.02)


def test_run_halves_split_one_orientation_and_keep_its_turn():
    from cafetomo.config import RunSubset
    from cafetomo.rawcheck import run_halves

    early, late = run_halves(RunSubset("SKY", (58, 77), -90.0))
    assert early == RunSubset("SKY", (58, 67), -90.0)
    assert late == RunSubset("SKY", (68, 77), -90.0)
    early, late = run_halves(RunSubset("SKY", (37, 57), 0.0))
    assert (early.runs, late.runs) == ((37, 46), (47, 57))
    with pytest.raises(ValueError, match="two runs"):
        run_halves(RunSubset("SKY", (40, 40), 0.0))
