"""One attempt from code to logged result: static check -> sandbox -> judge -> log.

Deterministic. No LLM in here, and only discovery data.
"""
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import sandbox
import static_check
import store

MIN_TRADES = 30  # position changes per symbol over the discovery period
DISCOVERY_YEARS = 4  # 2019-01-01 to 2022-12-31 (make_splits.py)
TIMEOUT = 20  # seconds on discovery. Validation runs the same code on 1.5 times the bars under sandbox.py's
              # 60s default, so anything that passes here has room there. An honest strategy takes about 1s.


def load_discovery():
    """{symbol: bars} from the discovery split, the only data generation may touch."""
    files = sorted(Path("data/splits/discovery").glob("*.parquet"))
    if not files:
        raise SystemExit("no discovery data: run download_data.py, then make_splits.py")
    return {p.stem.split("_")[0]: pd.read_parquet(p) for p in files}


def judge(code, frames, packed):
    """Run the stages and return the result fields for the log. Does not write anything."""
    reason = static_check.check(code)
    if reason:
        return {"status": "static_reject", "error": reason}
    out = sandbox.run(code, packed, timeout=TIMEOUT)
    if "error" in out:
        hint = (" rolling().apply() with a Python function or a loop over rows is the usual cause; use the built-in "
                "rolling methods instead.") if out["error"] == "timeout" else ""
        return {"status": out["error"], "error": out["detail"] + hint}
    symbols = sorted(frames)
    closes = [frames[s].close.to_numpy() for s in symbols]
    positions = [backtest.clean_positions(np.array(out["positions"][s], dtype=float)) for s in symbols]
    row = backtest.evaluate(closes, positions)
    row["lookahead"] = out["lookahead"] or "skipped"
    if row["trades_per_symbol"] == 0:
        # Not a quiet strategy: a strategy that never moves is nearly always a coding bug, so the coder gets it back.
        return {"status": "no_trades", "lookahead": row["lookahead"],
                "error": "the strategy never changed position on any symbol: every bar had the same position. "
                         "The entry condition is never true, or never reaches the positions. The usual cause is a breakout "
                         "compared with a rolling maximum that includes the current bar (close > high.rolling(N).max() can never "
                         "be true): take the previous N bars with .rolling(N).max().shift(1). Otherwise check that the "
                         "conditions are turned into positions."}
    row["equity"] = backtest.equity_curve(closes, positions)[::24].tolist()  # one point a day, for the dashboard
    row["pos_hash"] = hashlib.sha1((np.round(np.concatenate(positions), 6) + 0.0).tobytes()).hexdigest()
    row["status"] = "too_few_trades" if row["trades_per_symbol"] < MIN_TRADES else "ok"
    return row


def attempt(code, frames, packed, **meta):
    """Judge the code, log the attempt with its metadata (run_id, pilot, hypothesis, ...), return the row."""
    started = time.time()
    row = judge(code, frames, packed)
    if row["status"] == "ok" and store.count("run_id = ? AND pos_hash = ?", [meta["run_id"], row["pos_hash"]]):
        row["status"] = "duplicate"
    row.update(meta, code=code, seconds=time.time() - started)
    row["id"] = store.add(row)
    return row
