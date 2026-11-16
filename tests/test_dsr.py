import math

import numpy as np
import pytest

from dsr import deflated_sharpe, expected_max_sharpe, psr, sharpe_stats


def test_reference_cases_from_the_plan():
    sr0 = expected_max_sharpe(0.001, 100)
    assert sr0 == pytest.approx(0.080025, abs=1e-6)
    assert psr(0.1, sr0, 1000) == pytest.approx(0.7356, abs=1e-4)
    assert psr(0.1, sr0, 1000, skew=-0.5, kurt=5) == pytest.approx(0.7301, abs=1e-4)
    # two years of hourly bars, annualised Sharpe 2.0, the best of 1000 trials under the null
    T, per_year = 17520, 8760
    sr0 = expected_max_sharpe(1 / T, 1000)
    assert sr0 * math.sqrt(per_year) == pytest.approx(2.30, abs=0.01)
    assert psr(2.0 / math.sqrt(per_year), sr0, T) == pytest.approx(0.335, abs=1e-3)


def test_one_trial_is_plain_psr_against_zero():
    assert expected_max_sharpe(0.001, 1) == 0.0
    assert psr(0.1, 0.0, 1000) == pytest.approx(0.99919, abs=1e-5)
    assert psr(0.05, 0.05, 500) == 0.5  # observed equals benchmark: a coin flip


def test_bar_rises_with_trials_and_confidence_rises_with_data():
    bars = [expected_max_sharpe(0.001, n) for n in (2, 10, 100, 1000, 10000)]
    assert bars == sorted(bars) and bars[0] > 0
    assert expected_max_sharpe(0.004, 100) == pytest.approx(2 * expected_max_sharpe(0.001, 100))
    confidence = [psr(0.1, 0.08, t) for t in (100, 1000, 10000)]
    assert confidence == sorted(confidence)
    assert psr(0.1, 0.08, 1000, skew=-1) < psr(0.1, 0.08, 1000) < psr(0.1, 0.08, 1000, skew=1)
    assert psr(0.1, 0.08, 1000, kurt=9) < psr(0.1, 0.08, 1000)


def test_sharpe_stats_on_known_shapes():
    x = np.random.default_rng(0).normal(0.001, 0.01, 200_000)
    sr, n, skew, kurt = sharpe_stats(x)
    assert n == 200_000 and sr == pytest.approx(0.1, abs=0.01)
    assert skew == pytest.approx(0, abs=0.03) and kurt == pytest.approx(3, abs=0.06)  # raw kurtosis, not excess
    assert sharpe_stats(np.random.default_rng(1).exponential(1, 200_000))[2] == pytest.approx(2, abs=0.1)


def test_formula_matches_simulation_and_deflation_removes_lucky_winners():
    """Pure-noise strategies: pick the best of N, as a naive search would."""
    rng = np.random.default_rng(0)
    n_trials, n_obs, reps = 50, 250, 400
    best, naive, deflated = [], 0, 0
    for _ in range(reps):
        r = rng.normal(0, 1, (n_trials, n_obs))
        sharpes = r.mean(axis=1) / r.std(axis=1, ddof=1)
        winner = r[sharpes.argmax()]
        best.append(sharpes.max())
        naive += psr(sharpes.max(), 0.0, n_obs) > 0.95
        deflated += deflated_sharpe(winner, sharpes.var(ddof=1), n_trials) > 0.95
    assert np.mean(best) == pytest.approx(expected_max_sharpe(1 / n_obs, n_trials), rel=0.05)
    assert naive / reps > 0.85      # without correction the lucky winner nearly always looks significant
    assert deflated / reps < 0.10   # with it, almost never
