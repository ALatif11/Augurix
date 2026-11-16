"""Probabilistic and deflated Sharpe ratio.

Bailey and Lopez de Prado: "The Sharpe Ratio Efficient Frontier" (2012) for PSR,
"The Deflated Sharpe Ratio" (2014) for the correction for many trials.

Every Sharpe ratio in this file is PER BAR, not annualised. Mixing the two is the
classic way to get a wrong answer here.
"""
import math
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def sharpe_stats(returns):
    """(per-bar Sharpe, number of observations, skewness, kurtosis) of a return series.

    Kurtosis is the raw kind: 3 for a normal distribution, not 0.
    """
    x = np.asarray(returns, dtype=float)
    z = (x - x.mean()) / x.std()
    return float(x.mean() / x.std(ddof=1)), len(x), float((z ** 3).mean()), float((z ** 4).mean())


def psr(sr, benchmark, n_obs, skew=0.0, kurt=3.0):
    """Probability that the true Sharpe is above `benchmark`, given the observed one.

    The denominator is the Sharpe estimate's standard error with the correction for
    non-normal returns: negative skew and fat tails make an estimate less certain.
    """
    variance_term = max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12)
    return _NORMAL.cdf((sr - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(variance_term))


def expected_max_sharpe(trial_var, n_trials):
    """The Sharpe the best of `n_trials` strategies with no skill is expected to show.

    trial_var: variance of the per-bar Sharpe across the trials.
    """
    if n_trials <= 1:
        return 0.0  # one trial is no selection; the formula's inverse CDF would be minus infinity
    return math.sqrt(trial_var) * ((1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
                                   + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns, trial_var, n_trials):
    """Probability that this strategy's true Sharpe beats what luck alone gives the best of n_trials."""
    sr, n_obs, skew, kurt = sharpe_stats(returns)
    return psr(sr, expected_max_sharpe(trial_var, n_trials), n_obs, skew, kurt)
