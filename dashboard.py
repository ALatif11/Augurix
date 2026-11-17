"""Read-only live dashboard over the attempt log.

    python dashboard.py            # then open http://127.0.0.1:8000

It shows discovery-period results only. Validation and holdout stay locked until their
scripts have run (PLAN.md). It never fetches live prices: those are holdout data.
"""
import functools
import json
import math
import os
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb
import numpy as np

import backtest
import dsr
import pipeline
import store

ROOT = Path(__file__).parent
LIST_COLS = ("id, created_at, run_id, pilot, tries, parent_id, family, left(hypothesis, 300) AS hypothesis, status, lookahead, "
             "timing_sharpe, net_sharpe, trades_per_symbol, mean_position, "
             # the earlier attempts an idea says it builds on, for the lineage view
             "CAST(regexp_extract_all(hypothesis, '#([0-9]{1,9})', 1) AS BIGINT[]) AS refs")
BACKTESTED = "status IN ('ok', 'too_few_trades', 'duplicate')"


def clean(x):
    """Make a query result JSON-safe: no NaN or infinity, timestamps as text."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x.isoformat() if hasattr(x, "isoformat") else x


def state(run=None):
    where, params = ("run_id = ?", [run]) if run else ("true", [])
    funnel = store.query(f"""
        SELECT count(*) AS generated,
               count(*) FILTER (status <> 'static_reject') AS static_ok,
               count(*) FILTER (status IN ('lookahead', 'no_trades') OR {BACKTESTED}) AS ran,
               count(*) FILTER ((status = 'no_trades' OR {BACKTESTED}) AND lookahead = 'pass') AS lookahead_ok,
               count(*) FILTER (status = 'ok' AND timing_sharpe > 0 AND lookahead = 'pass') AS gate_checked,
               count(*) FILTER (status = 'ok' AND timing_sharpe > 0) AS gate_all,
               count(*) FILTER (created_at > (now() AT TIME ZONE 'UTC') - INTERVAL 1 HOUR) AS last_hour,
               median(seconds) AS median_seconds,
               max(timing_sharpe) FILTER (status = 'ok') AS best
        FROM attempts WHERE {where}""", params)
    return clean({
        "runs": store.query("SELECT run_id, count(*) AS n, bool_or(pilot) AS pilot FROM attempts "
                            "GROUP BY run_id ORDER BY max(id) DESC"),
        "funnel": funnel[0] if funnel else None,
        "scores": [r["timing_sharpe"] for r in store.query(
            f"SELECT timing_sharpe FROM attempts WHERE {where} AND {BACKTESTED}", params)],
        "attempts": store.query(f"SELECT {LIST_COLS} FROM attempts WHERE {where} ORDER BY id DESC LIMIT 1000", params),
        "tele": telemetry(where, params),
        "locks": {"detector": (ROOT / "lookahead.py").exists(),
                  "holdout_opened": (ROOT / "results" / "HOLDOUT_OPENED.json").exists()},
        "week5": week5(run),
        "lessons": lessons(run),
    })


def telemetry(where, params):
    """Run-level aggregates for the side panels. Every number is computed here from the log; none is model-written."""
    span = store.query(f"""
        SELECT min(created_at) AS first, max(created_at) AS last, count(*) FILTER (tries = 0) AS ideas,
               count(*) FILTER ({BACKTESTED} AND NOT pilot) AS counted, count(*) FILTER (status = 'ok') AS scored,
               count(*) FILTER (status = 'ok' AND trades_per_symbol > 1000) AS heavy,
               median(timing_sharpe) FILTER (status = 'ok') AS median, max(model) AS model, max(prompt_version) AS prompts
        FROM attempts WHERE {where}""", params)
    if not span or span[0]["first"] is None:
        return None
    span = span[0]
    window = [span["first"] - timedelta(minutes=10), span["last"]]  # the first model call precedes the first attempt
    return {
        "span": span,
        # what the best of this many no-skill strategies is expected to score (same rule as the run report)
        "luck": dsr.expected_max_sharpe(1 / pipeline.DISCOVERY_YEARS, span["scored"]),
        "statuses": store.query(f"SELECT status, count(*) AS n FROM attempts WHERE {where} GROUP BY status ORDER BY n DESC", params),
        "families": store.query(f"""
            SELECT coalesce(family, 'none') AS family, count(*) AS n, count(*) FILTER (status = 'ok') AS scored,
                   median(timing_sharpe) FILTER (status = 'ok') AS median, max(timing_sharpe) FILTER (status = 'ok') AS best
            FROM attempts WHERE {where} GROUP BY 1 ORDER BY 1""", params),
        "pace": store.query(f"SELECT time_bucket(INTERVAL 5 MINUTE, created_at) AS t, count(*) AS n FROM attempts "
                            f"WHERE {where} GROUP BY 1 ORDER BY 1", params),
        "llm": store.query("SELECT role, count(*) AS calls, sum(seconds) AS seconds, sum(tokens_out) AS tokens FROM llm_calls "
                           "WHERE created_at BETWEEN ? AND ? GROUP BY role ORDER BY seconds DESC", window),
        "flow": store.query("SELECT time_bucket(INTERVAL 1 MINUTE, created_at) AS t, sum(tokens_out) AS n FROM llm_calls "
                            "WHERE created_at BETWEEN ? AND ? GROUP BY 1 ORDER BY 1", window),
    }


def lessons(run=None):
    """The reviewer agent's latest notes: for the chosen run, or for whichever run was reviewed last."""
    try:
        run = run or store.query("SELECT run_id FROM lessons ORDER BY created_at DESC LIMIT 1")[0]["run_id"]
    except (duckdb.CatalogException, IndexError):
        return []
    return [{"run_id": run, **note} for note in store.lessons(run, 3)][::-1]


def week5(run=None):
    """Shortlist size and holdout outcome. Both stay None until run_validation.py and open_holdout.py have run."""
    out = {"shortlisted": None, "holdout": None}
    try:
        rows = store.query("SELECT count(*) FILTER (shortlisted) AS n, count(*) AS total FROM validation "
                           "WHERE (run_id = ? OR ? IS NULL) AND pass_no = (SELECT max(pass_no) FROM validation)", [run, run])
        if rows and rows[0]["total"]:
            out["shortlisted"] = rows[0]["n"]
    except duckdb.CatalogException:  # validation has never been run
        pass
    files = sorted((ROOT / "results").glob(f"holdout_{run or '*'}.json"))
    if files:
        r = json.loads(files[-1].read_text())
        out["holdout"] = {"survivors": r["survivors"], "primary_passed": r["primary"]["passed"]}
    return out


@functools.cache
def benchmark():
    """Buy-and-hold on the discovery period, one point a day, to draw behind a strategy's equity."""
    try:
        frames = pipeline.load_discovery()
    except SystemExit:
        return {"dates": [], "equity": []}
    closes = [f.close.to_numpy() for f in frames.values()]
    ts = next(iter(frames.values())).ts
    return {"dates": [t.strftime("%Y-%m-%d") for t in ts[::24]],
            "equity": backtest.equity_curve(closes, [np.ones(len(c)) for c in closes], fee=0)[::24].tolist()}


def curves(run=None):
    """Thinned equity curves of the latest backtested attempts, worst timing Sharpe first, for the 3D terrain."""
    where, params = ("run_id = ?", [run]) if run else ("true", [])
    rows = store.query(f"SELECT id, family, timing_sharpe, equity FROM attempts WHERE {where} AND status = 'ok' "
                       "AND equity IS NOT NULL ORDER BY id DESC LIMIT 200", params)
    b = benchmark()
    return clean({"dates": b["dates"][::12], "benchmark": b["equity"][::12],
                  "curves": [{"id": r["id"], "family": r["family"], "timing": r["timing_sharpe"], "eq": r["equity"][::12]}
                             for r in sorted(rows, key=lambda r: r["timing_sharpe"])]})


def attempt(attempt_id):
    rows = store.query("SELECT * FROM attempts WHERE id = ?", [attempt_id])
    return clean({"attempt": rows[0], "benchmark": benchmark()}) if rows else None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self.reply((ROOT / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self.reply_json(state(parse_qs(url.query).get("run", [None])[0]))
        if url.path == "/api/equity":
            return self.reply_json(curves(parse_qs(url.query).get("run", [None])[0]))
        if url.path.startswith("/api/attempt/") and url.path.rsplit("/", 1)[1].isdigit():
            found = attempt(int(url.path.rsplit("/", 1)[1]))
            if found:
                return self.reply_json(found)
        self.send_error(404)

    def reply_json(self, obj):
        self.reply(json.dumps(obj).encode(), "application/json")

    def reply(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Dashboard on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()  # localhost only: it serves generated code
