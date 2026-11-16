import pytest

from static_check import check

GOOD = """
import numpy as np
import pandas as pd

def strategy(df):
    fast = df.close.rolling(24).mean()
    slow = df.close.rolling(24 * 7).mean()
    return (fast > slow).astype(float)
"""


def test_plain_strategy_is_accepted():
    assert check(GOOD) is None


@pytest.mark.parametrize("code, why", [
    ("def strategy(df):\n    return df.close >", "syntax"),
    ("def helper(df):\n    return df.close", "strategy"),
    ("import os\n" + GOOD, "import"),
    ("from urllib import request\n" + GOOD, "import"),
    (GOOD + "\nopen('/etc/passwd')", "open"),
    (GOOD + "\nx = ().__class__", "__class__"),
    (GOOD + "\ny = pd.read_parquet('/data/x.parquet')", "read_parquet"),
    (GOOD + "\nz = np.load('x.npy')", "load"),
    (GOOD.replace("24 * 7", "2016"), "years"),          # a window that happens to look like a year
    (GOOD + "\n# works well after 2020-03-13", "years"),  # a date from the model's memory
])
def test_rejections(code, why):
    assert why in check(code)


def test_ordinary_numbers_are_not_mistaken_for_years():
    assert check(GOOD.replace("24 * 7", "200")) is None
    assert check(GOOD + "\nx = 0.2021\ny = 12020\n") is None
