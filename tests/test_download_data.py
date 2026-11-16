import io
import zipfile

import pandas as pd
import pytest

import download_data as dd


def make_zip(rows, header=False):
    lines = [",".join(dd.COLS)] if header else []
    lines += [",".join(map(str, r)) for r in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", "\n".join(lines))
    return buf.getvalue()


def row(open_time):
    return [open_time, 1, 2, 0.5, 1.5, 10, open_time + 1, 15, 3, 5, 7, 0]


def test_ms_and_us_timestamps_parse_to_same_instant():
    ms = 1_704_067_200_000  # 2024-01-01T00:00:00Z
    a = dd.parse_klines(make_zip([row(ms)]))
    b = dd.parse_klines(make_zip([row(ms * 1000)], header=True))
    assert a.ts[0] == b.ts[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert list(a.columns) == dd.KEEP
    assert a.close[0] == b.close[0] == 1.5


def test_locked_months_refused_without_flag():
    with pytest.raises(SystemExit):
        dd.check_unlocked(["2024-12", "2025-01"], holdout=False)
    dd.check_unlocked(["2024-12"], holdout=False)
    dd.check_unlocked(["2025-01"], holdout=True)
