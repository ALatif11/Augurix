"""Bake-off table: how well each model drives the pipeline, from discovery-only pilot runs.

    python bakeoff.py bake-qwen bake-gemma

No validation or holdout data is touched, and no score is compared: a best in-sample Sharpe from
100 attempts says nothing about a model. The measures are about code and search quality only.
Valid code is a share of all attempts; look-ahead rejects and never traded are shares of the attempts whose
code ran; duplicates are a share of the attempts that were backtested.
"""
import sys

import store

RAN = "status NOT IN ('static_reject', 'crash', 'bad_output', 'timeout')"  # the code passed the check and produced positions


def row(run_id):
    with store.connect(read_only=True) as con:
        a = con.execute(f"""
            SELECT any_value(model), count(*), count(*) FILTER (WHERE tries = 0),
                   count(*) FILTER (WHERE {RAN}),
                   count(*) FILTER (WHERE status = 'lookahead'),
                   count(*) FILTER (WHERE status = 'no_trades'),
                   count(*) FILTER (WHERE status IN ('ok', 'too_few_trades', 'duplicate')),
                   count(*) FILTER (WHERE status = 'duplicate'),
                   min(created_at), max(created_at)
            FROM attempts WHERE run_id = ?""", [run_id]).fetchone()
        model, n, ideas, ran, look, never, backtested, dup, start, end = a
        if not n:
            raise SystemExit(f"no attempts logged for run {run_id}")
        # ponytail: llm_calls has no run_id, so calls are matched by model and time window (from 15 minutes before the
        # first attempt). Exact only when nothing else used the same model at the same time, as in the bake-off.
        out, secs = con.execute("""SELECT sum(tokens_out), sum(seconds) FROM llm_calls WHERE model = ?
                                   AND created_at BETWEEN ? - INTERVAL 15 MINUTE AND ?""", [model, start, end]).fetchone()
    hours = max((end - start).total_seconds() / 3600, 1e-9)
    pct = lambda x, d: f"{100 * x / d:.0f}%" if d else "-"
    return [run_id, model, n, ideas, pct(ran, n), pct(look, ran), pct(never, ran), pct(dup, backtested),
            f"{n / hours:.0f}", f"{out / secs:.0f}" if secs else "-"]


HEAD = ["Run", "Model", "Attempts", "Ideas", "Valid code", "Look-ahead rejects", "Never traded", "Duplicates",
        "Attempts per hour", "Tokens out per second"]


def table(runs):
    lines = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    return "\n".join(lines + ["| " + " | ".join(map(str, row(r))) + " |" for r in runs])


if __name__ == "__main__":
    print(table(sys.argv[1:]) if sys.argv[1:] else __doc__)
