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
    assert t["llm"][0]["role"] == "coder" and t["llm"][0]["tokens"] == 40 and t["pace"][0]["n"] == 2
    assert st["week5"]["shortlisted"] is None  # stays locked until run_validation.py has run
    assert st["attempts"][1]["net_sharpe"] is None  # NaN must not reach the browser
    one = json.loads(json.dumps(dashboard.attempt(1), allow_nan=False))
    assert one["attempt"]["equity"] == [1.0, 1.1] and dashboard.attempt(99) is None


def test_near_copies_share_a_group_and_retries_follow_their_idea(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")
    rule = "Enter when the close price is above the {w}-hour high and {f}. Exit when the close falls below the 168-hour rolling mean."
    store.add({"run_id": "r", "tries": 0, "hypothesis": rule.format(w=672, f="volume rises above its average"), "status": "ok"})  # 1
    store.add({"run_id": "r", "tries": 1, "hypothesis": rule.format(w=672, f="volume rises above its average"), "status": "ok"})  # 2: retry of 1
    store.add({"run_id": "r", "tries": 0, "hypothesis": "Hold while the 24-hour mean of volume is above its 168-hour mean. Exit when it is not.",
               "status": "ok"})                                                                                          # 3: unrelated
    store.add({"run_id": "r", "tries": 0, "hypothesis": rule.format(w=720, f="the 24-hour range is shrinking"), "status": "ok"})  # 4: copy of 1
    g = dashboard.groups("r", 4)
    assert g == {1: 1, 2: 1, 3: 3, 4: 1}
    assert {a["id"]: a["group"] for a in dashboard.state("r")["attempts"]} == g


def test_replay_reruns_once_and_returns_change_points(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")
    monkeypatch.setattr(dashboard, "POSITIONS", tmp_path / "positions")
    monkeypatch.setattr(dashboard, "_packed", lambda: b"")
    code = "def strategy(df):\n    return df.close * 0\n"
    store.add({"run_id": "r", "status": "ok", "code": code})                      # 1
    store.add({"run_id": "r", "status": "crash", "code": code})                   # 2
    store.add({"run_id": "r", "status": "ok", "code": "import os\n" + code})      # 3
    calls = []

    def fake_run(c, packed, timeout=60, detector=True):
        calls.append(detector)
        return {"positions": {"BTC": [0, 0, 1, 1, 0.5, float("nan")], "ETH": [1, 1, 1, 1, 1, 1]}}

    monkeypatch.setattr(dashboard.sandbox, "run", fake_run)
    out = dashboard.positions(1)
    assert out == {"changes": {"BTC": [[2, 1.0], [4, 0.5], [5, 0.0]], "ETH": [[0, 1.0]]}}  # NaN counts as flat, as in the judge
    assert dashboard.positions(1) == out and calls == [False]  # cached, and the detector is not rerun
    assert "never produced positions" in dashboard.positions(2)["error"]
    assert "static check" in dashboard.positions(3)["error"] and dashboard.positions(99) is None

    def docker_down(*a, **k):
        raise RuntimeError("docker could not start the sandbox")

    monkeypatch.setattr(dashboard.sandbox, "run", docker_down)
    store.add({"run_id": "r", "status": "ok", "code": code})                      # 4
    assert "Docker Desktop" in dashboard.positions(4)["error"]
