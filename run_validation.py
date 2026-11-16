"""Week 5: score the official run on the validation period, ONCE, and write the pre-registration.

    python run_validation.py --run gen-v1

Run it only after generation is finished and frozen. Nothing it produces may reach an
agent prompt, and nothing may be tuned after looking at it. Results go to the
`validation` table and results/, never to the `attempts` table the agents read.
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
import sandbox
import store

ALPHA = 0.05
MAX_SHORTLIST = 10
NULL_DRAWS = 100
BACKTESTED = "status IN ('ok', 'too_few_trades', 'duplicate')"
TABLE = """CREATE TABLE IF NOT EXISTS validation (
    run_id VARCHAR, pass_no INTEGER, attempt_id BIGINT, status VARCHAR, error VARCHAR, code_sha VARCHAR,
    timing_sharpe DOUBLE, net_sharpe DOUBLE, sr_bar DOUBLE, n_obs INTEGER, skew DOUBLE, kurt DOUBLE,
    psr0 DOUBLE, dsr DOUBLE, shortlisted BOOLEAN)"""
COLUMNS = ["run_id", "pass_no", "attempt_id", "status", "error", "code_sha", "timing_sharpe", "net_sharpe",
           "sr_bar", "n_obs", "skew", "kurt", "psr0", "dsr", "shortlisted"]


def score(closes, positions, start, fee=backtest.FEE):
    """Statistics of one strategy's timing returns over the bars from `start` on."""
    net, timing = backtest.portfolio_returns(closes, positions, fee, start)
    out = {"net_sharpe": backtest.sharpe(net), "n_obs": len(timing)}
    if timing.std() < 1e-12:  # never trades in the window: no evidence either way
        return {**out, "timing_sharpe": 0.0, "sr_bar": 0.0, "skew": 0.0, "kurt": 3.0, "psr0": 0.5}
    sr, n, skew, kurt = dsr.sharpe_stats(timing)
    return {**out, "timing_sharpe": backtest.sharpe(timing), "sr_bar": sr, "skew": skew, "kurt": kurt,
            "psr0": dsr.psr(sr, 0.0, n, skew, kurt)}


def shifted(positions, rng):
    """A null copy of a strategy: the same positions rotated in time by a random amount.

    Turnover, time in the market and the shape of the trades are all kept. Only the
    alignment with prices is destroyed, so whatever a shifted copy scores is luck.

    Known limit: a strategy that repeats on the calendar (an hour-of-day rule) is
    unchanged by a shift of whole days (1 copy in 24) and keeps most of its edge an hour
    either side of that. For such strategies this null is conservative: it understates
    real skill and never overstates it. That is why it is reported next to the PSR tests
    and is not itself a gate.
    """
    n = len(positions[0])
    offset = int(rng.integers(n // 10, n - n // 10))
    return [np.roll(p, offset) for p in positions]


def pick_shortlist(scored, n_trials, option="B"):
    """Add `dsr` and `shortlisted` to each scored strategy. Returns (trial variance, expected best Sharpe).

    Option B (default): shortlist on individual significance, PSR against 0 of at least 0.95.
    Option A: shortlist on the deflated Sharpe instead. See PLAN.md, "Decision for Adam".
    """
    trial_var = float(np.var([s["sr_bar"] for s in scored], ddof=1)) if len(scored) > 1 else 0.0
    sr0 = dsr.expected_max_sharpe(trial_var, n_trials)
    for s in scored:
        s["dsr"] = dsr.psr(s["sr_bar"], sr0, s["n_obs"], s["skew"], s["kurt"])
        s["shortlisted"] = False
    gate = "dsr" if option == "A" else "psr0"
    passed = sorted((s for s in scored if s[gate] >= 1 - ALPHA), key=lambda s: -s["timing_sharpe"])
    for s in passed[:MAX_SHORTLIST]:
        s["shortlisted"] = True
    return trial_var, sr0


def last_pass(run_id):
    try:
        return store.query("SELECT max(pass_no) AS n FROM validation WHERE run_id = ?", [run_id])[0]["n"] or 0
    except (duckdb.CatalogException, IndexError):  # no validation table yet, or no log at all
        return 0


def validate(run_id, frames, start, option="B", seed=0):
    """Score every discovery-gate survivor of an official run on the bars from `start` on.

    frames: {symbol: bars} covering discovery AND validation; earlier bars are warm-up.
    The sandbox re-runs the look-ahead detector on this longer data, and a strategy that
    fails there is rejected even though it passed on discovery alone.
    """
    candidates = store.query(
        "SELECT id, code FROM attempts WHERE run_id = ? AND NOT pilot AND status = 'ok' "
        "AND lookahead = 'pass' AND timing_sharpe > 0 ORDER BY id", [run_id])
    n_trials = store.count(f"run_id = ? AND NOT pilot AND {BACKTESTED}", [run_id])
    symbols = sorted(frames)
    closes = [frames[s].close.to_numpy() for s in symbols]
    packed = sandbox.pack(frames)
    rng = np.random.default_rng(seed)
    rows, scored, null_passes = [], [], []
    for c in candidates:
        row = {"attempt_id": c["id"], "code_sha": hashlib.sha256(c["code"].encode()).hexdigest()}
        out = sandbox.run(c["code"], packed)
        if "error" in out:
            rows.append({**row, "status": out["error"], "error": out["detail"]})
            continue
        positions = [backtest.clean_positions(np.array(out["positions"][s], dtype=float)) for s in symbols]
        row.update(score(closes, positions, start), status="ok")
        null_passes += [score(closes, shifted(positions, rng), start)["psr0"] >= 1 - ALPHA for _ in range(NULL_DRAWS)]
        scored.append(row)
        rows.append(row)
    trial_var, sr0 = pick_shortlist(scored, n_trials, option)

    pass_no = last_pass(run_id) + 1
    with store.connect() as con:
        con.execute(TABLE)
        if rows:
            con.executemany(f"INSERT INTO validation VALUES ({', '.join('?' * len(COLUMNS))})",
                            [[{**r, "run_id": run_id, "pass_no": pass_no}.get(c) for c in COLUMNS] for r in rows])
    ts = next(iter(frames.values())).ts
    return {
        "run_id": run_id, "pass_no": pass_no, "option": option, "seed": seed,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": [str(ts.iloc[start]), str(ts.iloc[-1])],
        "n_trials": n_trials, "candidates": len(candidates), "validated": len(scored),
        "rejected_on_longer_data": len(rows) - len(scored),
        "trial_variance_per_bar": trial_var,
        "expected_best_sharpe_from_luck": sr0 * np.sqrt(backtest.BARS_PER_YEAR),
        "real_pass_rate": float(np.mean([s["psr0"] >= 1 - ALPHA for s in scored])) if scored else None,
        "null_pass_rate": float(np.mean(null_passes)) if null_passes else None,
        "shortlist": [{k: s[k] for k in ("attempt_id", "code_sha", "timing_sharpe", "psr0", "dsr")}
                      for s in sorted(scored, key=lambda s: -s["timing_sharpe"]) if s["shortlisted"]],
    }


def preregistration(summary, commit):
    """The text of PREREGISTRATION.md: everything about the holdout test, fixed before the data exists here."""
    k = len(summary["shortlist"])
    lines = [
        "# Pre-registration of the holdout test", "",
        f"Written {summary['ran_at']} from validation pass {summary['pass_no']} of run `{summary['run_id']}`, "
        f"at commit `{commit}`. Committed and tagged `prereg-v1` before any holdout data is downloaded. "
        "Nothing below, and no code that computes it, changes after that tag.", "",
        "## What happened before this point", "",
        f"- Official attempts that produced a backtest (the trial count N): {summary['n_trials']}",
        f"- Passed the discovery gate and were run on validation: {summary['candidates']}",
        f"- Rejected on the longer data (look-ahead or crash): {summary['rejected_on_longer_data']}",
        f"- Best annualised Sharpe expected from luck alone at this N: {summary['expected_best_sharpe_from_luck']:.2f}",
        f"- Share individually significant on validation: {_pct(summary['real_pass_rate'])} of real strategies, "
        f"{_pct(summary['null_pass_rate'])} of time-shifted copies with no timing information",
        f"- Shortlist rule: option {summary['option']}, at most {MAX_SHORTLIST}, ranked by validation timing Sharpe", "",
    ]
    if not k:
        return "\n".join(lines + [
            "## Result", "",
            "No strategy reached the shortlist. The holdout will not be opened. The final result is zero survivors.", ""])
    lines += ["## Shortlist", "", "| Attempt | SHA-256 of the strategy code | Validation timing Sharpe | PSR vs 0 | Deflated Sharpe |",
              "|---|---|---|---|---|"]
    lines += [f"| {s['attempt_id']} | `{s['code_sha']}` | {s['timing_sharpe']:.2f} | {s['psr0']:.3f} | {s['dsr']:.3f} |"
              for s in summary["shortlist"]]
    lines += [
        "", "## The test", "",
        f"- Data: the same symbols and bar size as discovery, from Binance bulk dumps. Scored bars start "
        f"{make_splits.HOLDOUT_START.date()} and run through the last complete month available when the data is downloaded. "
        "Earlier bars are indicator warm-up only.",
        f"- Scoring: `backtest.py` as of the commit above. Fee {backtest.FEE} per unit of position change. "
        "The score is the timing Sharpe.",
        f"- **Primary test:** the equal-weight portfolio of the {k} shortlisted strategies has a probabilistic Sharpe ratio "
        f"against zero of at least {1 - ALPHA} on the holdout. This is the project's one headline claim.",
        f"- **Secondary:** a strategy counts as a survivor if its own probabilistic Sharpe ratio against zero is at least "
        f"{1 - ALPHA / k:.4f} (Bonferroni, {ALPHA} / {k}).",
        "- A strategy that crashes or fails the look-ahead test on the holdout data contributes zero returns to the "
        "portfolio and is not a survivor.",
        "- Reported but not tests: results at double the fee, and a permutation p-value from time-shifted copies.",
        "- The holdout is opened once, by `open_holdout.py`. Whatever it prints is the result.", "",
        "## Next commands", "", "```",
        "git add PREREGISTRATION.md results/",
        'git commit -m "Pre-register the holdout test"',
        "git tag prereg-v1",
        "python download_data.py --holdout --end <last complete month, e.g. 2026-11>",
        f"python open_holdout.py --run {summary['run_id']}",
        "```", ""]
    return "\n".join(lines)


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def load_through_validation():
    frames = {p.stem.split("_")[0]: pd.read_parquet(p) for p in sorted(Path("data/klines").glob("*.parquet"))}
    if not frames:
        raise SystemExit("no data in data/klines")
    for df in frames.values():
        make_splits.discovery(df)  # raises if anything from 2025 on is present
    return frames


def first_bar(frames, when):
    """Index of the first bar at or after `when`; all symbols must share one timeline."""
    starts = {int((df.ts >= when).to_numpy().argmax()) for df in frames.values()}
    if len(starts) != 1 or len({len(df) for df in frames.values()}) != 1:
        raise SystemExit("symbols do not share the same bars; cannot score them as one portfolio")
    return starts.pop()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--option", choices=["A", "B"], default="B", help="shortlist rule; see PLAN.md")
    p.add_argument("--rerun", action="store_true", help="run validation again; the pass number is recorded and must be reported")
    args = p.parse_args(argv)
    if not store.count("run_id = ?", [args.run]) or store.count("run_id = ? AND pilot", [args.run]):
        raise SystemExit(f"'{args.run}' is not an official run. Validation is never run on pilot attempts: "
                         "looking at their validation results would leak into the prompts.")
    if last_pass(args.run) and not args.rerun:
        raise SystemExit("Validation has already been run for this run. Running it again turns validation into a second "
                         "training set. If you must, pass --rerun; the pass number is recorded and has to be reported.")
    frames = load_through_validation()
    summary = validate(args.run, frames, first_bar(frames, make_splits.DISCOVERY_END), args.option)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/validation_{args.run}.json").write_text(json.dumps(summary, indent=1))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    Path("PREREGISTRATION.md").write_text(preregistration(summary, commit), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "shortlist"}, indent=1))
    print(f"shortlist: {[s['attempt_id'] for s in summary['shortlist']]}")
    print("Wrote PREREGISTRATION.md. Read it, then follow its 'Next commands'.")


if __name__ == "__main__":
    main()
