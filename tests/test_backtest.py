import numpy as np
import pytest

import backtest as bt


def random_case(rng, n=200):
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return close, rng.uniform(0, 1, n).round(1)


def loop_net(close, pos, fee):
    """Independent bar-by-bar reference, written without numpy vector ops."""
    out, prev = [], 0.0
    for t in range(1, len(close)):
        p = pos[t - 1]
        out.append(p * (close[t] / close[t - 1] - 1) - fee * abs(p - prev))
        prev = p
    return out


def test_toy_example_matches_hand_calculation():
    close, pos = [100, 110, 99, 108.9], [1, 0, 1, 0]
    assert bt.bar_returns(close, pos, 0.001)[3] == pytest.approx([0.099, -0.001, 0.099], abs=1e-12)
    m = bt.evaluate([close], [pos], fee=0.001)
    assert m["total_return"] == pytest.approx(0.206593199, abs=1e-9)
    assert m["max_drawdown"] == pytest.approx(-0.001, abs=1e-9)
    assert m["trades_per_symbol"] == 3


def test_always_long_without_fees_is_buy_and_hold_with_zero_timing():
    close, _ = random_case(np.random.default_rng(0))
    m = bt.evaluate([close], [np.ones(len(close))], fee=0)
    assert m["total_return"] == pytest.approx(close[-1] / close[0] - 1)
    assert m["timing_sharpe"] == 0.0


def test_flat_earns_nothing():
    close, _ = random_case(np.random.default_rng(1))
    m = bt.evaluate([close], [np.zeros(len(close))])
    assert m["total_return"] == 0 and m["net_sharpe"] == 0 and m["timing_sharpe"] == 0


def test_matches_loop_reference_and_fees_only_hurt():
    rng = np.random.default_rng(2)
    for _ in range(50):
        close, pos = random_case(rng)
        assert bt.bar_returns(close, pos, 0.001)[3] == pytest.approx(loop_net(close, pos, 0.001), abs=1e-12)
        cheap, dear = (bt.evaluate([close], [pos], fee=f)["total_return"] for f in (0.001, 0.002))
        assert dear <= cheap


def test_later_positions_never_change_earlier_returns():
    rng = np.random.default_rng(3)
    for _ in range(50):
        close, pos = random_case(rng)
        t = int(rng.integers(1, len(close) - 1))
        changed = pos.copy()
        changed[t:] = rng.uniform(0, 1, len(close) - t)
        # net[:t] covers bars 1..t, which may only depend on pos[0..t-1]
        assert np.array_equal(bt.bar_returns(close, pos)[3][:t], bt.bar_returns(close, changed)[3][:t])


def test_positions_are_cleaned():
    assert list(bt.clean_positions([np.nan, -1, 0.5, 7])) == [0, 0, 0.5, 1]


def test_start_scores_only_later_bars():
    close, pos = random_case(np.random.default_rng(4))
    m = bt.evaluate([close], [pos], start=120)
    scored = np.array(loop_net(close, pos, bt.FEE))[119:]  # returns of bars 120..n-1
    assert m["total_return"] == pytest.approx(np.prod(1 + scored) - 1)
