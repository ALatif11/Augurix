import numpy as np
import pandas as pd

import baselines


def test_baselines_return_one_position_per_bar_within_range():
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC"),
                       "close": 100 * np.cumprod(1 + rng.normal(0, 0.01, n))})
    for f in baselines.BASELINES:
        pos = f(df)
        assert len(pos) == n and pos.index.equals(df.index)
        assert pos.dropna().between(0, 1).all()
        assert pos.dropna().nunique() > 1 or f is baselines.buy_and_hold  # it actually trades
