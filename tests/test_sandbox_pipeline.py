"""End-to-end tests through the real Docker sandbox. Skipped when the image is not available."""
import subprocess

import numpy as np
import pandas as pd
import pytest

import backtest
import generate
import pipeline
import sandbox
import store


def image_ready():
    try:
        return subprocess.run(["docker", "image", "inspect", sandbox.IMAGE], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not image_ready(), reason="needs Docker and `docker build -t finproj-sandbox .`")

SMA = """
import pandas as pd

def strategy(df):
    return (df.close > df.close.rolling(50).mean()).astype(float)
"""


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(0)
    n = 3000
    ts = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    return {s: pd.DataFrame({"ts": ts, "close": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)), "volume": 1.0})
            for s in ("AAA", "BBB")}


@pytest.fixture(scope="module")
def packed(frames):
    return sandbox.pack(frames)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")


def test_sandbox_result_matches_running_the_same_logic_on_the_host(frames, packed):
    row = pipeline.attempt(SMA, frames, packed, run_id="t", pilot=True)
    assert row["status"] == "ok" and row["lookahead"] == "pass"  # the detector ran inside the container
    host = [(f.close > f.close.rolling(50).mean()).astype(float) for f in frames.values()]
    expected = backtest.evaluate([f.close for f in frames.values()], host)
    assert row["timing_sharpe"] == pytest.approx(expected["timing_sharpe"])
    assert row["total_return"] == pytest.approx(expected["total_return"])
    assert len(row["equity"]) == 125  # one point a day
    assert store.count("run_id = 't'") == 1


def test_same_positions_twice_is_a_duplicate(frames, packed):
    pipeline.attempt(SMA, frames, packed, run_id="t", pilot=True)
    again = pipeline.attempt(SMA + "\n# reworded\n", frames, packed, run_id="t", pilot=True)
    assert again["status"] == "duplicate"


@pytest.mark.parametrize("body, status", [
    ("    return df.close.rolling(5).mean()[:10]", "bad_output"),
    ("    return df.no_such_column", "crash"),
    ("    return (df.close * 0).astype(float)", "no_trades"),          # never moves: a bug, sent back to the coder
    ("    return (df.ts > df.ts.iloc[0]).astype(float)", "too_few_trades"),  # enters once and holds
])
def test_failures_get_the_right_status(frames, packed, body, status):
    row = pipeline.attempt("def strategy(df):\n" + body + "\n", frames, packed, run_id="t", pilot=True)
    assert row["status"] == status


def test_lookahead_is_caught_inside_the_container_and_never_scored(frames, packed):
    cheat = "def strategy(df):\n    return (df.close.shift(-1) > df.close).astype(float)\n"
    row = pipeline.attempt(cheat, frames, packed, run_id="t", pilot=True)
    assert row["status"] == "lookahead" and "later bars are removed" in row["error"]
    assert row.get("timing_sharpe") is None  # a perfect-foresight Sharpe must never reach the log


def test_a_strategy_cannot_remember_the_full_data_between_calls(frames, packed):
    # It keeps the first data it is given and answers every later call from that. Without a fresh module
    # per call it replays the full data when the look-ahead test hands it a prefix, and passes.
    cheat = """
import numpy as np
import pandas as pd
SEEN = []

def strategy(df, also_seen=[]):
    for memory in (SEEN, also_seen):
        if not memory:
            memory.append(df.close.to_numpy())
    full = max(SEEN[0], also_seen[0], key=len)
    nxt = np.append(full[1:] / full[:-1] - 1, 0.0)
    return pd.Series((nxt > 0)[:len(df)].astype(float), index=df.index)
"""
    row = pipeline.attempt(cheat, frames, packed, run_id="t", pilot=True)
    assert row["status"] == "lookahead" and row.get("timing_sharpe") is None


def test_static_reject_never_reaches_docker(frames, packed, monkeypatch):
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: pytest.fail("ran rejected code"))
    row = pipeline.attempt("import os\ndef strategy(df):\n    return df.close\n", frames, packed, run_id="t", pilot=True)
    assert row["status"] == "static_reject"


def test_container_is_locked_down(packed):
    """These bypass the static check on purpose: the container itself must hold."""
    runaway = "def strategy(df):\n    while True:\n        pass\n"
    assert sandbox.run(runaway, packed, timeout=3)["error"] == "timeout"
    network = "import urllib.request\ndef strategy(df):\n    urllib.request.urlopen('http://example.com', timeout=5)\n"
    assert sandbox.run(network, packed)["error"] == "crash"
    write = "def strategy(df):\n    open('/job/x', 'w')\n"
    assert sandbox.run(write, packed)["error"] == "crash"
    listing = "import os\ndef strategy(df):\n    raise RuntimeError(str(os.listdir('/job')) + str(os.path.exists('/data')))\n"
    assert "['strategy.py']False" in sandbox.run(listing, packed)["detail"]


def test_graph_feeds_the_error_back_and_retries(frames, packed):
    replies = iter(["```python\ndef strategy(df):\n    return df.nope\n```", "```python" + SMA + "```"])
    prompts = []

    def llm(role, system, user, schema=None, temperature=0.8):
        prompts.append(user)
        return next(replies)

    out = generate.build(frames, packed, llm).invoke(
        {"run_id": "g", "pilot": True, "tries": 0, "hypothesis": "long above the 50h average"})
    assert out["tries"] == 2 and out["row"]["status"] == "ok"
    assert "It was rejected" in prompts[1] and "nope" in prompts[1]
    rows = store.query("SELECT status, tries, parent_id FROM attempts WHERE run_id = 'g' ORDER BY id")
    assert [r["status"] for r in rows] == ["crash", "ok"] and rows[1]["parent_id"] is not None
