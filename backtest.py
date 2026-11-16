"""The judge: turns positions into net returns and metrics. No LLM, no strategy code.

Conventions (PLAN.md, "Conventions the judge uses"):
  - pos[t] is decided at the close of bar t and earns the return of bar t+1.
  - net[t] = pos[t-1] * r[t] - fee * |pos[t-1] - pos[t-2]|, flat before the first bar.
  - Positions are clipped to [0, 1]; NaN counts as 0.
"""
import numpy as np

FEE = 0.001           # per unit of position change (Binance spot taker)
BARS_PER_YEAR = 8760  # hourly bars, 24/7 market


def clean_positions(pos):
    return np.clip(np.nan_to_num(np.asarray(pos, dtype=float), nan=0.0), 0.0, 1.0)


def bar_returns(close, pos, fee=FEE):
    """Return (r, held, change, net), each of length n-1 and aligned to bars 1..n-1.

    held[i] is the position carried through bar i+1, which is pos[i]. The last
    position is never used because there is no later bar for it to earn.
    """
    close = np.asarray(close, dtype=float)
    held = clean_positions(pos)[:-1]
    r = close[1:] / close[:-1] - 1
    change = np.abs(np.diff(held, prepend=0.0))
    return r, held, change, held * r - fee * change


def sharpe(x, periods=BARS_PER_YEAR):
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return 0.0 if sd < 1e-12 else float(x.mean() / sd * np.sqrt(periods))


def portfolio_returns(closes, positions, fee=FEE, start=0):
    """Per-bar (net, timing) returns of the equal-weight portfolio over the scored bars.

    closes, positions: one array per symbol, all the same length.
    start: index of the first scored bar. Earlier bars are indicator warm-up
    (used when scoring validation or the holdout on top of older history).
    """
    k = max(start - 1, 0)  # net[i] belongs to bar i+1
    nets, timings = [], []
    for close, pos in zip(closes, positions):
        r, held, _, net = (a[k:] for a in bar_returns(close, pos, fee))
        nets.append(net)
        # Subtract what a constant position of the same average size earned, so
        # the score rewards only *when* the strategy was in the market.
        timings.append(net - held.mean() * r)
    return np.mean(nets, axis=0), np.mean(timings, axis=0)


def equity_curve(closes, positions, fee=FEE, start=0):
    """Compounded value of 1 unit in the equal-weight portfolio, one point per scored bar plus the start."""
    return np.concatenate([[1.0], np.cumprod(1 + portfolio_returns(closes, positions, fee, start)[0])])


def evaluate(closes, positions, fee=FEE, start=0):
    """Score one strategy on several symbols as an equal-weight portfolio. Arguments as portfolio_returns."""
    net, timing = portfolio_returns(closes, positions, fee, start)
    k = max(start - 1, 0)
    turnover, trades, mean_pos = [], [], []
    for close, pos in zip(closes, positions):
        _, held, change, _ = (a[k:] for a in bar_returns(close, pos, fee))
        turnover.append(change.sum())
        trades.append(np.count_nonzero(change > 1e-12))
        mean_pos.append(held.mean())
    equity = np.concatenate([[1.0], np.cumprod(1 + net)])
    years = len(net) / BARS_PER_YEAR
    return {
        "timing_sharpe": sharpe(timing),
        "net_sharpe": sharpe(net),
        "total_return": float(equity[-1] - 1),
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "turnover_per_year": float(np.mean(turnover) / years),
        "trades_per_symbol": float(np.mean(trades)),
        "mean_position": float(np.mean(mean_pos)),
    }
