import json

import dashboard
import store


def test_state_and_attempt_are_json_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")
    assert dashboard.state()["funnel"] is None  # nothing logged yet
    store.log_llm(role="coder", model="m", prompt="p", response="r", tokens_in=10, tokens_out=40, seconds=2.0)  # the model call comes before the attempt it produces
    store.add({"run_id": "r", "pilot": True, "status": "ok", "lookahead": "skipped", "hypothesis": "h", "code": "c",
               "timing_sharpe": 0.3, "net_sharpe": float("nan"), "equity": [1.0, 1.1], "seconds": 2.0})
    store.add({"run_id": "r", "pilot": True, "status": "static_reject", "error": "import not allowed",
               "family": "trend", "hypothesis": "Refine #1 and #999 with a volume filter", "parent_id": 1})
    st = json.loads(json.dumps(dashboard.state("r"), allow_nan=False))
    f = st["funnel"]
    assert (f["generated"], f["static_ok"], f["ran"], f["lookahead_ok"], f["gate_all"], f["last_hour"]) == (2, 1, 1, 0, 1, 2)
    assert st["scores"] == [0.3] and st["attempts"][0]["status"] == "static_reject"
    assert st["attempts"][0]["refs"] == [1, 999] and st["attempts"][0]["parent_id"] == 1
    t = st["tele"]
    assert t["span"]["scored"] == 1 and t["span"]["counted"] == 0 and t["luck"] == 0  # one pilot score: no luck bar, nothing in N
    assert {s["status"]: s["n"] for s in t["statuses"]} == {"ok": 1, "static_reject": 1}
    assert t["llm"][0]["role"] == "coder" and t["flow"][0]["n"] == 40 and t["pace"][0]["n"] == 2
    assert st["week5"]["shortlisted"] is None  # stays locked until run_validation.py has run
    assert st["attempts"][1]["net_sharpe"] is None  # NaN must not reach the browser
    one = json.loads(json.dumps(dashboard.attempt(1), allow_nan=False))
    assert one["attempt"]["equity"] == [1.0, 1.1] and dashboard.attempt(99) is None
