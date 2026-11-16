"""Three hand-written baselines plus buy-and-hold, one per strategy family.

Parameters are the textbook ones from the cited sources and were never tuned on
this data. These are pilots: they are not counted in the trial count and are
never sent to validation or the holdout.

    python baselines.py     # prints discovery-period metrics
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtest


def _daily(x, ts):
    """Hold the value seen at 00:00 UTC for the rest of the day."""
    return x.where(ts.dt.hour == 0).ffill()


def buy_and_hold(df):
    return pd.Series(1.0, index=df.index)


def sma_trend(df):
    """Trend: long while price is above its 20-day average, decided once a day.

    Detzel, Liu, Strauss, Zhou and Zhu (2021), Financial Management: ratios of
    price to its moving average forecast daily Bitcoin returns.
    """
    above = (df.close > df.close.rolling(480).mean()).astype(float)
    return _daily(above, df.ts)


def rsi_reversion(df, n=14):
    """Mean reversion: buy when hourly RSI(14) falls below 30, sell above 50.

    Wilder's standard settings. Short-horizon reversal in crypto is documented
    gross of costs; whether it survives fees is the point of testing it.
    """
    d = df.close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / down)
    return pd.Series(np.where(rsi < 30, 1.0, np.where(rsi > 50, 0.0, np.nan)), index=df.index).ffill()


def vol_managed(df):
    """Volatility: cut exposure when 30-day variance is above its own history.

    Moreira and Muir (2017), Journal of Finance, "Volatility-Managed Portfolios":
    exposure proportional to the inverse of recent realised variance. Here the
    scale is the expanding median, so it uses no future data, and it is capped at 1.
    """
    var = df.close.pct_change().rolling(720).var()
    return _daily((var.expanding().median() / var).clip(upper=1), df.ts)


BASELINES = [buy_and_hold, sma_trend, rsi_reversion, vol_managed]

if __name__ == "__main__":
    dfs = [pd.read_parquet(p) for p in sorted(Path("data/splits/discovery").glob("*.parquet"))]
    rows = {f.__name__: backtest.evaluate([d.close for d in dfs], [f(d) for d in dfs]) for f in BASELINES}
    print(pd.DataFrame(rows).T.round(3).to_string())
