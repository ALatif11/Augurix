"""Read-only live dashboard over the attempt log.

    python dashboard.py            # then open http://127.0.0.1:8000

It shows discovery-period results only. Validation and holdout stay locked until their
scripts have run (PLAN.md). It never fetches live prices: those are holdout data.
"""
import functools
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import backtest
import pipeline
import store

ROOT = Path(__file__).parent
LIST_COLS = ("id, created_at, run_id, pilot, tries, family, hypothesis, status, lookahead, "
             "timing_sharpe, net_sharpe, trades_per_symbol")
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
               count(*) FILTER (status = 'lookahead' OR {BACKTESTED}) AS ran,
               count(*) FILTER ({BACKTESTED} AND lookahead = 'pass') AS lookahead_ok,
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
        "attempts": store.query(f"SELECT {LIST_COLS} FROM attempts WHERE {where} ORDER BY id DESC LIMIT 300", params),
        "locks": {"detector": (ROOT / "lookahead.py").exists(),
                  "holdout_opened": (ROOT / "results" / "HOLDOUT_OPENED.json").exists()},
    })


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
