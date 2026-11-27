from datetime import datetime

import bakeoff
import store


def test_rates_use_the_right_denominators(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")
    for tries, status in [(0, "static_reject"), (1, "lookahead"), (2, "ok"), (0, "duplicate"), (0, "no_trades")]:
        store.add({"run_id": "b", "model": "m", "tries": tries, "status": status})
    store.log_llm(role="coder", model="m", prompt="p", response="r", tokens_in=1, tokens_out=300, seconds=10.0)
    store.log_llm(role="coder", model="other", prompt="p", response="r", tokens_in=1, tokens_out=999, seconds=1.0)
    with store.connect() as con:  # spread the attempts over half an hour
        con.execute("UPDATE attempts SET created_at = ? WHERE id = 1", [datetime(2026, 1, 1, 0, 0)])
        con.execute("UPDATE attempts SET created_at = ? WHERE id > 1", [datetime(2026, 1, 1, 0, 30)])
        con.execute("UPDATE llm_calls SET created_at = ?", [datetime(2026, 1, 1, 0, 10)])
    r = bakeoff.row("b")
    # 5 attempts, 3 ideas; 4 ran; 1 of 4 look-ahead, 1 of 4 never traded; 1 of 2 backtested is a duplicate;
    # 10 attempts an hour; 30 tokens a second from model m only
    assert r == ["b", "m", 5, 3, "80%", "25%", "25%", "50%", "10", "30"]
