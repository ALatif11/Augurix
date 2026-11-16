"""Look-ahead detector. Runs inside the sandbox container, so it imports only numpy.

The idea: a strategy that uses no future data gives the same positions for bars
0..c-1 whether or not the bars from c onward exist. So run it on all the data, run
it again on only the first c bars, and compare the part they share. Any difference
means some position depended on bars that came after it.

What this cannot catch:
  - A strategy that hard-codes dates from the model's memory. That uses no future
    *data*, so nothing changes when data is removed. static_check.py rejects year
    literals for this reason.
  - A leak confined to a stretch of bars that no cut happens to land in. Cuts are a
    sample, not every bar.  # ponytail: 20 cuts; raise N_RANDOM_CUTS if a real miss shows up
"""
import numpy as np

# Early cuts are not optional. A statistic over the whole dataset (its maximum, say)
# stops changing once the cut is past the bar where that maximum occurs, so a detector
# with only late cuts would pass a strategy that uses it.
FIXED_CUTS = (0.02, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85)
N_RANDOM_CUTS = 12
TOL = 1e-9  # honest rolling/expanding/ewm code reproduces a prefix exactly; this only absorbs float noise


def cut_points(n, seed=0):
    """Where to truncate: spread over the whole range, plus n-1, plus seeded random ones."""
    rng = np.random.default_rng(seed)
    cuts = {int(n * f) for f in FIXED_CUTS} | {n - 1} | set(rng.integers(n // 50, n, N_RANDOM_CUTS).tolist())
    return sorted(c for c in cuts if 2 <= c < n)


def _positions(strategy, df):
    # A copy each time, so a strategy that edits its input cannot affect the other runs.
    return np.asarray(strategy(df.copy()), dtype=float)


def _differs(a, b):
    return ~np.isclose(a, b, rtol=0.0, atol=TOL, equal_nan=True)


def check_lookahead(strategy, df, seed=0):
    """Return (ok, reason). `reason` is written for the coder agent to act on."""
    n = len(df)
    full = _positions(strategy, df)
    if full.shape != (n,):
        return False, f"returned positions of shape {full.shape} for {n} bars"
    if _differs(full, _positions(strategy, df)).any():
        return False, "nondeterministic: two runs on identical data gave different positions"
    for c in cut_points(n, seed):
        try:
            part = _positions(strategy, df.iloc[:c])
        except Exception as e:
            return False, f"crashed when given only the first {c} bars ({type(e).__name__}: {e})"
        if part.shape != (c,):
            return False, f"returned positions of shape {part.shape} when given the first {c} bars"
        bad = _differs(part, full[:c])  # the whole shared prefix, not just the last bar
        if bad.any():
            return False, (f"{int(bad.sum())} of the first {c} positions change when later bars are removed "
                           f"(first at bar {int(bad.argmax())}): the strategy uses data from after the bar it trades on")
    return True, "ok"
