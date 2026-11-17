"""Runs INSIDE the sandbox container. Never import this on the host.

stdin:  one Parquet file of bars with a `symbol` column.
stdout: {"positions": {symbol: [...]}, "lookahead": "pass" | null}
        or {"error": "crash" | "bad_output" | "lookahead", "detail": text}

If /lookahead.py is mounted it must expose check_lookahead(strategy, df) -> (ok, reason).
"""
import importlib.util
import io
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run():
    data = pd.read_parquet(io.BytesIO(sys.stdin.buffer.read()))
    load("/job/strategy.py", "strategy_mod").strategy  # fail here, with the real error, if the file does not load

    def strategy(df):
        # A fresh copy of the module for every call. A strategy that kept the first data it was given
        # (a module-level list, a default argument) could otherwise replay the full data when the
        # look-ahead test hands it a prefix, and pass. # ponytail: state hidden on numpy or pandas
        # themselves would survive this; run each call in its own process if that ever shows up.
        return load("/job/strategy.py", "strategy_mod").strategy(df)

    detector = load("/lookahead.py", "lookahead").check_lookahead if os.path.exists("/lookahead.py") else None
    out = {"positions": {}, "lookahead": "pass" if detector else None}
    for symbol, df in data.groupby("symbol", sort=True):
        df = df.drop(columns="symbol").reset_index(drop=True)
        pos = np.asarray(strategy(df.copy()), dtype=float)
        if pos.shape != (len(df),):
            return {"error": "bad_output", "detail": f"expected {len(df)} positions, got shape {pos.shape}"}
        out["positions"][symbol] = pos.tolist()
        if detector:
            ok, reason = detector(strategy, df)
            if not ok:
                return {"error": "lookahead", "detail": f"{symbol}: {reason}"}
    return out


if __name__ == "__main__":
    real_stdout, sys.stdout = sys.stdout, sys.stderr  # strategy prints must not corrupt the result
    try:
        result = run()
    except Exception:
        result = {"error": "crash", "detail": traceback.format_exc()[-1500:]}
    json.dump(result, real_stdout)
