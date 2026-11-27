"""The committed record of the official run agrees with itself: the funnel, N, and the pre-registered code hashes."""
import hashlib
import json
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = (ROOT / "results" / "gen-v1_attempts.parquet").as_posix()
VALIDATION = (ROOT / "results" / "gen-v1_validation.parquet").as_posix()


def test_shortlisted_code_is_the_preregistered_code():
    registered = dict(re.findall(r"^\| (\d+) \| `([0-9a-f]{64})`", (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8"), re.M))
    rows = duckdb.sql(f"SELECT a.id, a.code FROM '{ATTEMPTS}' a JOIN '{VALIDATION}' v ON v.attempt_id = a.id "
                      "WHERE v.shortlisted").fetchall()
    assert len(registered) == 10 and {str(i): hashlib.sha256(c.encode()).hexdigest() for i, c in rows} == registered


def test_funnel_and_n_match_the_published_numbers():
    count = lambda where: duckdb.sql(f"SELECT count(*) FROM '{ATTEMPTS}' WHERE {where}").fetchall()[0][0]
    assert count("true") == 661
    assert count("status <> 'static_reject'") == 631
    assert count("status NOT IN ('static_reject', 'crash', 'bad_output', 'timeout')") == 603
    assert count("status IN ('ok', 'too_few_trades', 'duplicate', 'no_trades')") == 594
    n = count("status IN ('ok', 'too_few_trades', 'duplicate')")
    v = json.loads((ROOT / "results" / "validation_gen-v1.json").read_text(encoding="utf-8"))
    assert n == v["n_trials"] == 534
    assert duckdb.sql(f"SELECT count(*), count(*) FILTER (WHERE shortlisted) FROM '{VALIDATION}'").fetchall() == [(266, 10)]
    h = json.loads((ROOT / "results" / "holdout_gen-v1.json").read_text(encoding="utf-8"))
    assert h["survivors"] == 1 and not h["primary"]["passed"]
