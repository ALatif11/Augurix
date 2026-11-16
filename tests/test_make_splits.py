import pandas as pd
import pytest

import make_splits as ms


def bars(start, periods):
    return pd.DataFrame({"ts": pd.date_range(start, periods=periods, freq="1h", tz="UTC"), "close": 1.0})


def test_discovery_stops_before_validation_starts():
    df = bars("2022-12-31 22:00", 4)  # 22:00, 23:00, then two validation bars
    part = ms.discovery(df)
    assert len(part) == 2
    assert part.ts.max() == pd.Timestamp("2022-12-31 23:00", tz="UTC")


def test_holdout_bars_in_source_are_refused():
    with pytest.raises(ValueError):
        ms.discovery(bars("2024-12-31 23:00", 2))  # second bar is 2025-01-01 00:00
