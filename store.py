"""The attempt log: every strategy attempt, including failures, in one DuckDB file.

The `attempts` table only ever holds discovery-period metrics. Validation and holdout
results go in separate tables written by their own scripts (PLAN.md).

DuckDB allows one writing process or several reading ones, not both at once. Every
call here opens the file, does one thing and closes it, and retries briefly if another
process (the generator or the dashboard) has it open.
"""
import os
import time
from contextlib import contextmanager
from pathlib import Path

import duckdb

DB = Path(os.environ.get("FINPROJ_DB", "runs/attempts.duckdb"))

SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS attempt_id;
CREATE TABLE IF NOT EXISTS attempts (
    id BIGINT PRIMARY KEY DEFAULT nextval('attempt_id'),
    created_at TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC'),  -- UTC; plain TIMESTAMP so reading needs no pytz
    run_id VARCHAR, pilot BOOLEAN, model VARCHAR, prompt_version VARCHAR,
    parent_id BIGINT, tries INTEGER,
    family VARCHAR, hypothesis VARCHAR, code VARCHAR,
    status VARCHAR, error VARCHAR, lookahead VARCHAR, pos_hash VARCHAR,
    timing_sharpe DOUBLE, net_sharpe DOUBLE, total_return DOUBLE, max_drawdown DOUBLE,
    turnover_per_year DOUBLE, trades_per_symbol DOUBLE, mean_position DOUBLE,
    equity DOUBLE[], seconds DOUBLE
);
CREATE TABLE IF NOT EXISTS llm_calls (
    created_at TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC'), role VARCHAR, model VARCHAR,
    prompt VARCHAR, response VARCHAR, tokens_in INTEGER, tokens_out INTEGER, seconds DOUBLE
);
CREATE TABLE IF NOT EXISTS lessons (
    created_at TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC'), run_id VARCHAR, after_attempt BIGINT, text VARCHAR
);
"""
COLUMNS = ["run_id", "pilot", "model", "prompt_version", "parent_id", "tries", "family", "hypothesis",
           "code", "status", "error", "lookahead", "pos_hash", "timing_sharpe", "net_sharpe",
           "total_return", "max_drawdown", "turnover_per_year", "trades_per_symbol", "mean_position",
           "equity", "seconds"]


@contextmanager
def connect(read_only=False):
    DB.parent.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        try:
            con = duckdb.connect(str(DB), read_only=read_only)
            break
        except duckdb.IOException:  # another process has the file open
            if i == 99:
                raise
            time.sleep(0.1)
    try:
        if not read_only:
            con.execute(SCHEMA)
        yield con
    finally:
        con.close()


def add(row):
    """Insert one attempt (a dict; unknown keys are ignored) and return its id."""
    cols = [c for c in COLUMNS if c in row]
    sql = f"INSERT INTO attempts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) RETURNING id"
    with connect() as con:
        return con.execute(sql, [row[c] for c in cols]).fetchone()[0]


def log_llm(**call):
    with connect() as con:
        con.execute(f"INSERT INTO llm_calls ({', '.join(call)}) VALUES ({', '.join('?' * len(call))})",
                    list(call.values()))


def query(sql, params=()):
    """Read-only query; returns a list of dicts ([] if nothing has been logged yet)."""
    if not DB.exists():
        return []
    with connect(read_only=True) as con:
        cur = con.execute(sql, list(params))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]


def count(where="true", params=()):
    rows = query(f"SELECT count(*) AS n FROM attempts WHERE {where}", params)
    return rows[0]["n"] if rows else 0


def add_lesson(run_id, after_attempt, text):
    """Store the reviewer's notes covering attempts up to `after_attempt`."""
    with connect() as con:
        con.execute("INSERT INTO lessons (run_id, after_attempt, text) VALUES (?, ?, ?)", [run_id, after_attempt, text])


def lessons(run_id, limit=5):
    """The latest reviewer notes for a run, oldest first ([] if there are none yet)."""
    try:
        return query("SELECT created_at, after_attempt, text FROM lessons WHERE run_id = ? "
                     "ORDER BY after_attempt DESC LIMIT ?", [run_id, limit])[::-1]
    except duckdb.CatalogException:  # a log created before the lessons table existed
        return []
