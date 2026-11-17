"""Week 5, last step: open the holdout. This runs exactly once.

    python open_holdout.py --run gen-v1

It refuses unless PREREGISTRATION.md is committed and tagged prereg-v1, the judge code
is unchanged since that tag, and the shortlist still matches the registered hashes. It
writes results/HOLDOUT_OPENED.json BEFORE computing anything, so a second run is refused
even if the first one crashes halfway. Whatever it prints is the project's result.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import backtest
import dsr
import make_splits
import run_validation
import sandbox
import store

ALPHA = run_validation.ALPHA
PERMUTATIONS = 1000
TAG = "prereg-v1"
FROZEN = ["PREREGISTRATION.md", "backtest.py", "dsr.py", "lookahead.py", "harness.py", "sandbox.py",
          "static_check.py", "run_validation.py", "open_holdout.py", "make_splits.py", "Dockerfile"]


def shortlist(run_id):
    """The strategies registered for the holdout: the latest validation pass's shortlist, best first."""
    try:
        return store.query(
            "SELECT v.attempt_id, v.code_sha, a.code FROM validation v JOIN attempts a ON a.id = v.attempt_id "
            "WHERE v.run_id = ? AND v.shortlisted AND v.pass_no = (SELECT max(pass_no) FROM validation WHERE run_id = ?) "
            "ORDER BY v.timing_sharpe DESC", [run_id, run_id])
    except duckdb.CatalogException:  # validation has never been run
        return []


def registration_problems(picks, cwd="."):
    """Everything that makes opening the holdout illegitimate right now. Empty list means go."""
    problems = []
    prereg = Path(cwd, "PREREGISTRATION.md")
    text = prereg.read_text(encoding="utf-8") if prereg.exists() else ""
    if not text:
        problems.append("PREREGISTRATION.md does not exist; run run_validation.py first")

    def git(*args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)

    if not git("tag", "--list", TAG).stdout.strip():
        problems.append(f"git tag {TAG} does not exist; commit PREREGISTRATION.md and tag it before downloading the holdout")
    else:
        changed = git("diff", "--name-only", TAG, "--", *FROZEN).stdout.split()
        if changed:
            problems.append(f"changed since {TAG}: {', '.join(changed)}. The test must run exactly as registered")
    for s in picks:
        if hashlib.sha256(s["code"].encode()).hexdigest() != s["code_sha"]:
            problems.append(f"attempt {s['attempt_id']}: its code no longer matches the hash recorded at validation")
        elif s["code_sha"] not in text:
            problems.append(f"attempt {s['attempt_id']} is not in PREREGISTRATION.md")
    return problems


def history_problems(frames, history):
    """The holdout run must start from the same first bar as the data validation used.

    The strategies run on everything they are given and only the bars from March 2025 are scored.
    Given only the holdout months, a long window or an expanding statistic would start empty and the
    strategy tested would not be the one registered. download_data.py --holdout fetches from 2019 by
    default; this catches a --start that cut the history short.
    """
    return [f"{s}: the holdout data starts {frames[s].ts.iloc[0]}, not {history[s].ts.iloc[0]} where the earlier data starts; "
            "download it again with the default --start" if s in history else f"{s}: not in data/klines"
            for s in frames if s not in history or frames[s].ts.iloc[0] != history[s].ts.iloc[0]]


def open_once(run_id, frames, start, results=Path("results"), seed=0):
    """Score the shortlist on the bars from `start` on. Refuses if the holdout was ever opened."""
    marker = Path(results, "HOLDOUT_OPENED.json")
    if marker.exists():
        raise SystemExit(f"The holdout was already opened: {marker.read_text()[:300]}\nIt cannot be opened again.")
    picks = shortlist(run_id)
    if not picks:
        raise SystemExit("The shortlist is empty: nothing reached the holdout. The result is zero survivors, "
                         "and the holdout stays closed.")
    Path(results).mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "run_id": run_id, "attempt_ids": [s["attempt_id"] for s in picks]}))

    symbols = sorted(frames)
    closes = [frames[s].close.to_numpy() for s in symbols]
    packed = sandbox.pack(frames)
    k = len(picks)
    flat = [np.zeros(len(c)) for c in closes]
    positions, strategies = [], []
    for s in picks:
        out = sandbox.run(s["code"], packed)
        if "error" in out:  # registered rule: contributes zero returns, is not a survivor
            positions.append(flat)
            strategies.append({"attempt_id": s["attempt_id"], "status": out["error"], "error": out["detail"], "survived": False})
            continue
        pos = [backtest.clean_positions(np.array(out["positions"][sym], dtype=float)) for sym in symbols]
        positions.append(pos)
        sc = run_validation.score(closes, pos, start)
        strategies.append({"attempt_id": s["attempt_id"], "status": "ok", **sc, "survived": bool(sc["psr0"] >= 1 - ALPHA / k)})

    def portfolio(all_positions, fee=backtest.FEE):
        # The mean of the strategies' timing series is the timing series of holding them in equal weight.
        timing = np.mean([backtest.portfolio_returns(closes, p, fee, start)[1] for p in all_positions], axis=0)
        if timing.std() < 1e-12:
            return 0.0, 0.5
        sr, n, skew, kurt = dsr.sharpe_stats(timing)
        return backtest.sharpe(timing), dsr.psr(sr, 0.0, n, skew, kurt)

    sharpe, psr0 = portfolio(positions)
    rng = np.random.default_rng(seed)
    null = [portfolio([run_validation.shifted(p, rng) for p in positions])[0] for _ in range(PERMUTATIONS)]
    ts = next(iter(frames.values())).ts
    result = {
        "run_id": run_id, "window": [str(ts.iloc[start]), str(ts.iloc[-1])], "years": (len(ts) - start) / backtest.BARS_PER_YEAR,
        "shortlisted": k,
        "primary": {"timing_sharpe": sharpe, "psr0": psr0, "passed": bool(psr0 >= 1 - ALPHA),
                    "permutation_p": (1 + sum(x >= sharpe for x in null)) / (PERMUTATIONS + 1),
                    "timing_sharpe_at_double_fee": portfolio(positions, 2 * backtest.FEE)[0]},
        "survivors": sum(s["survived"] for s in strategies), "secondary_threshold": 1 - ALPHA / k,
        "strategies": strategies,
    }
    Path(results, f"holdout_{run_id}.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    args = p.parse_args(argv)
    if Path("results/HOLDOUT_OPENED.json").exists():
        raise SystemExit("The holdout was already opened. It cannot be opened again.")
    files = sorted(Path("data/holdout").glob("*.parquet"))
    problems = registration_problems(shortlist(args.run))
    frames = {f.stem.split("_")[0]: pd.read_parquet(f) for f in files}
    if not files:
        problems.append("no holdout data: run download_data.py --holdout only after the tag exists")
    else:
        problems += history_problems(frames, {p.stem.split("_")[0]: pd.read_parquet(p, columns=["ts"])
                                              for p in Path("data/klines").glob("*.parquet")})
    if problems:
        raise SystemExit("Refusing to open the holdout:\n- " + "\n- ".join(problems))
    r = open_once(args.run, frames, run_validation.first_bar(frames, make_splits.HOLDOUT_START))
    pr = r["primary"]
    print(f"Holdout {r['window'][0]} .. {r['window'][1]} ({r['years']:.2f} years), {r['shortlisted']} strategies")
    print(f"PRIMARY: portfolio timing Sharpe {pr['timing_sharpe']:+.2f}, PSR vs 0 = {pr['psr0']:.3f} -> "
          f"{'PASSED' if pr['passed'] else 'FAILED'} (permutation p = {pr['permutation_p']:.3f}, "
          f"at double fee {pr['timing_sharpe_at_double_fee']:+.2f})")
    print(f"SECONDARY: {r['survivors']} of {r['shortlisted']} individually significant at PSR >= {r['secondary_threshold']:.4f}")
    for s in r["strategies"]:
        print(f"  #{s['attempt_id']}: " + (f"timing {s['timing_sharpe']:+.2f}, PSR {s['psr0']:.3f}" if s["status"] == "ok" else s["status"])
              + ("  SURVIVED" if s["survived"] else ""))
    print(f"Written to results/holdout_{args.run}.json. This is final.")


if __name__ == "__main__":
    main()
