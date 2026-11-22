"""Read-only live dashboard over the attempt log.

    python dashboard.py            # then open http://127.0.0.1:8000

It shows discovery-period results only. Validation and holdout stay locked until their
scripts have run (PLAN.md). It never fetches live prices: those are holdout data.
"""
import functools
import json
import math
import os
import re
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb
import numpy as np

import agents
import backtest
import dsr
import pipeline
import sandbox
import static_check
import store

ROOT = Path(__file__).parent
LIST_COLS = ("id, created_at, run_id, pilot, tries, parent_id, family, left(hypothesis, 300) AS hypothesis, status, lookahead, "
             "timing_sharpe, net_sharpe, trades_per_symbol, mean_position, split_part(prompt_version, '/', 2) AS island, "
             # average hours per holding: time in the market over the number of entries (two position changes
             # make a round trip). Approximate when positions are fractional; for sorting and filtering by style.
             f"CASE WHEN trades_per_symbol > 0 THEN mean_position * {pipeline.DISCOVERY_YEARS * 8760} "
             "/ greatest(trades_per_symbol / 2, 1) END AS hold_hours, "
             # the earlier attempts an idea says it builds on, for the lineage view
             "CAST(regexp_extract_all(hypothesis, '#([0-9]{1,9})', 1) AS BIGINT[]) AS refs")
BACKTESTED = "status IN ('ok', 'too_few_trades', 'duplicate')"
POSITIONS = ROOT / "runs" / "positions"  # replay cache: one file per attempt
REPLAYABLE = ("ok", "too_few_trades", "duplicate", "no_trades")  # statuses whose code ran and produced positions
_sandbox_lock = threading.Lock()  # one replay in the sandbox at a time
OFFICIAL_N = 1000  # PLAN.md: the official run stops at 1,000 attempts


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
        "attempts": with_groups(run, store.query(f"SELECT {LIST_COLS} FROM attempts WHERE {where} ORDER BY id DESC "
                                                 f"LIMIT {5000 if run else 1000}", params)),
        "tele": telemetry(where, params),
        "locks": {"detector": (ROOT / "lookahead.py").exists(),
                  "holdout_opened": (ROOT / "results" / "HOLDOUT_OPENED.json").exists()},
        "week5": week5(run),
        "fee": backtest.FEE,  # the replay recomputes one symbol's equity bar by bar with the judge's formula
        "lessons": lessons(run),
    })


def with_groups(run, rows):
    """Add each attempt's near-copy group. Groups only mean something inside one run."""
    found = groups(run, store.count("run_id = ?", [run])) if run else {}
    for r in rows:
        r["group"] = found.get(r["id"], r["id"])
    return rows


@functools.lru_cache(maxsize=8)
def groups(run, _count):
    """{attempt id: group id} for one run. _count is the run's size, so the cache refreshes as it grows.

    Ideas that share an exit and an entry condition (agents._skeleton, the same test the researcher's
    proposals must pass) are one rule with a filter swapped, and fall in one group, also through a
    chain. A retry belongs to its idea's group. The group id is the group's earliest attempt.
    """
    parent, idea, seen = {}, None, []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in store.query("SELECT id, tries, hypothesis FROM attempts WHERE run_id = ? ORDER BY id", [run]):
        parent[r["id"]] = r["id"]
        if r["tries"] and idea is not None:
            parent[r["id"]] = idea
            continue
        idea = r["id"]
        entry, exit_ = agents._skeleton(r["hypothesis"])
        if exit_:
            for j, e, x in seen:
                if x == exit_ and e & entry and find(j) != find(idea):
                    parent[max(find(j), find(idea))] = min(find(j), find(idea))  # the earlier root wins
        seen.append((idea, entry, exit_))
    return {i: find(i) for i in parent}


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
    # Windows globs ignore case, so "holdout_*" also matches HOLDOUT_OPENED.json, the lock file; keep only result files.
    files = [f for f in sorted((ROOT / "results").glob(f"holdout_{run or '*'}.json")) if f.name.startswith("holdout_")]
    if files:
        r = json.loads(files[-1].read_text())
        out["holdout"] = {"survivors": r["survivors"], "primary_passed": r["primary"]["passed"]}
    return out


REFUSALS = (("numbers changed", "Only the numbers changed"), ("same exit", "A filter swapped on a crowded rule"),
            ("builds on", "Built on a used-up line"), ("JSON", "Not valid JSON"))


def live(run):
    """What the search is doing now, for the Live tab: progress, pace and the researcher's latest proposals.

    A proposal counts as accepted when an attempt in this run carries its idea. Why one was turned away is
    in the next ask's prompt (agents.propose adds the reason). Skipped turns are counted from the run's log.
    """
    span = store.query("SELECT min(created_at) AS a, max(created_at) AS b, bool_or(pilot) AS pilot, max(id) AS last, "
                       "count(*) FILTER (created_at > (now() AT TIME ZONE 'UTC') - INTERVAL 1 HOUR) AS last_hour "
                       "FROM attempts WHERE run_id = ?", [run])
    if not span or span[0]["a"] is None:
        return None
    span = span[0]
    newest = span["last"] == store.query("SELECT max(id) AS m FROM attempts")[0]["m"]  # still running, or the latest run
    upto = (span["b"] + timedelta(days=1)) if newest else span["b"] + timedelta(minutes=10)
    calls = store.query("SELECT created_at, prompt, response FROM llm_calls WHERE role = 'researcher' AND created_at BETWEEN ? AND ? "
                        "ORDER BY created_at DESC", [span["a"] - timedelta(minutes=10), upto])
    ideas = {r["hypothesis"]: r["id"] for r in store.query("SELECT id, hypothesis FROM attempts WHERE run_id = ? AND tries = 0", [run])}
    reasons, recent = {}, []
    for k, c in enumerate(calls):
        why = (c["prompt"] or "").partition("not accepted: ")[2]
        if why:
            label = next((name for key, name in REFUSALS if key in why), "Other")
            reasons[label] = reasons.get(label, 0) + 1
        if len(recent) < 8:
            try:
                h = json.loads(re.search(r"\{.*\}", c["response"] or "", re.DOTALL).group(0))
            except (AttributeError, ValueError):
                continue
            recent.append({"t": c["created_at"], "family": h.get("family"), "hypothesis": h.get("hypothesis"),
                           "accepted": ideas.get(h.get("hypothesis")), "latest": k == 0})
    log = ROOT / "runs" / f"{run}.log"
    skipped = sum(1 for line in log.open(encoding="utf-8", errors="replace") if line.startswith("- turn skipped")) if log.exists() else 0
    return clean({"target": None if span["pilot"] else OFFICIAL_N, "last_hour": span["last_hour"], "asks": len(calls),
                  "refused": sum(reasons.values()), "reasons": reasons, "recent": recent, "skipped": skipped})


@functools.cache
def bars():
    """Discovery-period hourly candles for the replay chart: {symbol: {t, o, h, l, c}}, t in Unix seconds.

    Discovery only. Validation and holdout bars must never reach this page (PLAN.md, "Splits").
    """
    try:
        frames = pipeline.load_discovery()
    except SystemExit:
        return {}
    return {s: {"t": df.ts.values.astype("datetime64[s]").astype("int64").tolist(),
                **{k: df[col].round(8).tolist() for k, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"))}}
            for s, df in frames.items()}


@functools.cache
def _packed():
    return sandbox.pack(pipeline.load_discovery())


def change_points(pos):
    """[[bar index, new position], ...] for every bar where the position differs from the bar before (flat before bar 0)."""
    p = backtest.clean_positions(pos)
    return [[int(i), round(float(p[i]), 6)] for i in np.flatnonzero(np.diff(p, prepend=0.0))]


def positions(attempt_id):
    """An attempt's positions on the discovery bars, as change points per symbol, or {"error": text}.

    Positions are not stored in the log, so the strategy is run again in the Docker sandbox, the same
    code on the same data that produced its logged score, and the result is cached in runs/positions/.
    """
    path = POSITIONS / f"{attempt_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = store.query("SELECT code, status FROM attempts WHERE id = ?", [attempt_id])
    if not rows:
        return None
    if rows[0]["status"] not in REPLAYABLE:
        return {"error": "This attempt never produced positions, so there is nothing to replay."}
    if static_check.check(rows[0]["code"] or ""):
        return {"error": "This attempt's code does not pass the static check, so it is not run again."}
    with _sandbox_lock:
        if path.exists():  # another request finished it while this one waited
            return json.loads(path.read_text())
        try:
            out = sandbox.run(rows[0]["code"], _packed(), timeout=60, detector=False)
        except (RuntimeError, OSError) as e:
            return {"error": f"The sandbox could not start. Is Docker Desktop running? ({str(e)[:160]})"}
        except SystemExit as e:
            return {"error": str(e)}
    if "error" in out:
        return {"error": f"Running it again failed ({out['error']}): {out['detail'][-300:]}"}
    result = {"changes": {s: change_points(np.array(out["positions"][s], dtype=float)) for s in sorted(out["positions"])}}
    POSITIONS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def attempt(attempt_id):
    rows = store.query("SELECT * FROM attempts WHERE id = ?", [attempt_id])
    return clean({"attempt": rows[0]}) if rows else None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self.reply((ROOT / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self.reply_json(state(parse_qs(url.query).get("run", [None])[0]))
        if url.path == "/api/live":
            return self.reply_json(live(parse_qs(url.query).get("run", [""])[0]))
        if url.path == "/api/bars":
            return self.reply_json(bars())
        if url.path.startswith("/api/positions/") and url.path.rsplit("/", 1)[1].isdigit():
            found = positions(int(url.path.rsplit("/", 1)[1]))
            if found:
                return self.reply_json(found)
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
