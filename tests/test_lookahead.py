"""The detector must catch every planted cheat and pass every honest strategy."""
import numpy as np
import pandas as pd
import pytest

import baselines
from lookahead import check_lookahead, cut_points


def bars(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    close[int(n * 0.3)] = close.max() * 1.5  # the all-time high is early, so only early cuts can expose full-sample cheats
    return pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
                         "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": rng.uniform(1, 2, n)})


def sma(df):
    return (df.close.rolling(24).mean() > df.close.rolling(168).mean()).astype(float)


def next_bar_up(df):
    return (df.close.shift(-1) > df.close).astype(float)


def fitted_on_everything(df):
    r = df.close.pct_change().fillna(0).to_numpy()
    slope, intercept = np.polyfit(r[:-1], r[1:], 1)  # coefficients learned from all rows, future included
    return pd.Series(np.clip(0.5 + 50 * (slope * r + intercept), 0, 1), index=df.index)


def only_on_long_data(df):
    return next_bar_up(df) if len(df) > 3500 else sma(df)


def only_in_last_tenth(df):
    late = np.arange(len(df)) > 0.9 * len(df)
    return sma(df).where(~late, next_bar_up(df))


CHEATS = {
    "shift(-1)": next_bar_up,
    "smoothed then shifted back": lambda df: (df.close.rolling(5).mean().shift(-1) > df.close).astype(float),
    "centred rolling window": lambda df: (df.close.rolling(25, center=True).mean() > df.close).astype(float),
    "z-score with full-sample mean and std": lambda df: ((df.close - df.close.mean()) / df.close.std() < 0).astype(float),
    "scaled by full-sample min and max": lambda df: (df.close - df.close.min()) / (df.close.max() - df.close.min()),
    "threshold at a full-sample quantile": lambda df: (df.close < df.close.quantile(0.3)).astype(float),
    "long while below the dataset's all-time high": lambda df: (df.close < df.close.max()).astype(float),
    "rank over the full series": lambda df: (df.close.rank(pct=True) < 0.5).astype(float),
    "bfill": lambda df: (df.close.where(df.ts.dt.hour == 0).bfill() > df.close).astype(float),
    "model fitted on the whole dataset": fitted_on_everything,
    "compares with the last row": lambda df: (df.close < df.close.iloc[-1]).astype(float),
    "exponential average of the reversed series": lambda df: (df.close[::-1].ewm(span=20).mean()[::-1] > df.close).astype(float),
    "cheats only when the data is long": only_on_long_data,
    "cheats only in the last tenth of bars": only_in_last_tenth,
}


def rsi(df, n=14):
    d = df.close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / down) < 40).astype(float)


def stateful_loop(df):
    """Enter on a 48-bar high, leave on a 24-bar low, carried bar by bar."""
    hi, lo = df.close.rolling(48).max().to_numpy(), df.close.rolling(24).min().to_numpy()
    close, out, held = df.close.to_numpy(), np.zeros(len(df)), 0.0
    for i in range(len(df)):
        if close[i] >= hi[i]:
            held = 1.0
        elif close[i] <= lo[i]:
            held = 0.0
        out[i] = held
    return pd.Series(out, index=df.index)


HONEST = {
    "moving-average crossover": sma,
    "RSI": rsi,
    "expanding z-score": lambda df: ((df.close - df.close.expanding().mean()) / df.close.expanding().std() < 0).astype(float),
    "rolling volatility filter": lambda df: (df.close.pct_change().rolling(24).std() < df.close.pct_change().rolling(168).std()).astype(float),
    "hour-of-day rule": lambda df: df.ts.dt.hour.isin([0, 1, 2, 3]).astype(float),
    "moving average by convolution": lambda df: pd.Series(np.convolve(df.close, np.ones(24) / 24)[:len(df)] > df.close, index=df.index).astype(float),
    "stateful loop": stateful_loop,
    "fractional positions with warm-up NaNs": lambda df: (df.close.rolling(500).rank(pct=True)),
    **{f.__name__: f for f in baselines.BASELINES},
}


@pytest.mark.parametrize("name", CHEATS)
def test_every_planted_cheat_is_caught(name):
    ok, reason = check_lookahead(CHEATS[name], bars())
    assert not ok, f"missed: {name}"
    assert "later bars are removed" in reason


@pytest.mark.parametrize("name", HONEST)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_false_alarm_on_honest_strategies(name, seed):
    assert check_lookahead(HONEST[name], bars(seed)) == (True, "ok")


def rare_next_bar(df):
    """Long for one bar when the NEXT bar rises a lot: a leak that reaches one bar ahead and is active under 1% of the time."""
    return (df.close.shift(-1) / df.close - 1 > 0.025).astype(float)


def rare_next_bar_behind_filters(df):
    nxt = df.close.shift(-1) / df.close - 1
    return ((nxt > 0.02) & (df.volume > df.volume.rolling(24).mean()) & (df.close > df.close.rolling(48).mean())).astype(float)


@pytest.mark.parametrize("cheat", [rare_next_bar, rare_next_bar_behind_filters])
def test_a_rare_one_bar_leak_is_caught_whatever_the_seed(cheat):
    df = bars(n=20000)
    active = cheat(df).mean()
    assert 0 < active < 0.01  # rare enough that cuts placed blind nearly always miss it
    assert all(not check_lookahead(cheat, df, seed)[0] for seed in range(10))
    # the blind cuts alone: the last bar before a cut is almost never one the leak acted on
    full = cheat(df).to_numpy()
    missed = sum(all(np.allclose(cheat(df.iloc[:c]).to_numpy(), full[:c]) for c in cut_points(len(df), seed)) for seed in range(10))
    assert missed >= 5


def test_change_cuts_end_on_bars_where_the_position_moved():
    from lookahead import change_cuts
    full = np.array([0, 0, 1, 1, 0, 0, 0, 1.0])
    assert change_cuts(full) == {3, 5}  # prefixes ending on bars 2 and 4; bar 7 is the last bar, so its prefix is the full run
    assert change_cuts(np.zeros(50)) == set()


def test_randomness_is_reported_as_nondeterministic():
    ok, reason = check_lookahead(lambda df: pd.Series(np.random.rand(len(df)), index=df.index), bars())
    assert not ok and reason.startswith("nondeterministic")


def test_crash_on_a_prefix_and_wrong_length_are_rejected_with_a_reason():
    ok, reason = check_lookahead(lambda df: (df.close > df.close.iloc[3990]).astype(float), bars())
    assert not ok and "crashed when given only the first" in reason
    ok, reason = check_lookahead(lambda df: df.close.iloc[10:], bars())
    assert not ok and "shape" in reason


def test_strategy_cannot_corrupt_the_data_between_runs():
    def vandal(df):
        pos = sma(df)
        df["close"] = 0.0  # edits its own copy only
        return pos
    assert check_lookahead(vandal, bars())[0]


def test_cuts_cover_the_early_part_and_the_last_bar():
    cuts = cut_points(35000)
    assert min(cuts) <= 0.02 * 35000 and max(cuts) == 34999 and len(cuts) >= 18
    assert sum(c < 0.3 * 35000 for c in cuts) >= 3
    assert cuts == cut_points(35000)  # seeded, so a rejection can be reproduced
