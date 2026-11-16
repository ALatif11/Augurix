"""Three hand-written baselines plus buy-and-hold, one per strategy family.

Parameters are the textbook ones from the cited sources and were never tuned on
this data. These are pilots: they are not counted in the trial count and are
never sent to validation or the holdout.

    python baselines.py     # prints discovery-period metrics
"""
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

FAMILY = {"buy_and_hold": "other", "sma_trend": "trend", "rsi_reversion": "mean_reversion", "vol_managed": "volatility"}

if __name__ == "__main__":
    import inspect

    import lookahead
    import pipeline
    import store

    frames = pipeline.load_discovery()
    closes = [d.close.to_numpy() for d in frames.values()]
    log = store.count("run_id = 'baselines'") == 0  # put them in the attempt log once, as a reference run
    rows = {}
    for f in BASELINES:
        # Trusted hand-written code, so it runs on the host; generated code never does.
        pos = [backtest.clean_positions(f(d)) for d in frames.values()]
        rows[f.__name__] = m = backtest.evaluate(closes, pos)
        if log:
            store.add({**m, "run_id": "baselines", "pilot": True, "tries": 0, "family": FAMILY[f.__name__],
                       "hypothesis": (inspect.getdoc(f) or "Always fully long.").split("\n\n")[0].replace("\n", " "),
                       "code": inspect.getsource(f), "seconds": 0.0,
                       "lookahead": "pass" if all(lookahead.check_lookahead(f, d)[0] for d in frames.values()) else "fail",
                       "status": "ok" if m["trades_per_symbol"] >= pipeline.MIN_TRADES else "too_few_trades",
                       "equity": backtest.equity_curve(closes, pos)[::24].tolist()})
    print(pd.DataFrame(rows).T.round(3).to_string())
