"""Validation, null control, pre-registration and holdout opening. Synthetic data only:
these tests must never read the real validation or holdout bars."""
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

import agents
import backtest
import open_holdout
import pipeline
import run_validation as rv
import sandbox
import store
from tests.test_sandbox_pipeline import image_ready

N, START = 6000, 3000


def market(seed=0):
    rng = np.random.default_rng(seed)
    return [100 * np.cumprod(1 + rng.normal(0, 0.01, N)) for _ in range(2)]


def noise_positions(rng):
    """Random day-long holds: plenty of trades, no information."""
    return [np.repeat(rng.integers(0, 2, N // 24 + 1), 24)[:N].astype(float) for _ in range(2)]


def skilled_positions(closes, rng, noise=0.03):
    """A fixture with real foresight: a noisy peek at the next bar's return."""
    return [(np.append(np.diff(c) / c[:-1], 0) + rng.normal(0, noise, N) > 0).astype(float) for c in closes]


def test_portfolio_returns_agree_with_evaluate():
    closes, rng = market(), np.random.default_rng(1)
    for start in (0, 1, START):
        pos = noise_positions(rng)
        net, timing = backtest.portfolio_returns(closes, pos, start=start)
        m = backtest.evaluate(closes, pos, start=start)
        assert backtest.sharpe(timing) == m["timing_sharpe"] and backtest.sharpe(net) == m["net_sharpe"]
        assert len(net) == N - max(start, 1)


def test_score_separates_skill_from_no_skill():
    closes, rng = market(), np.random.default_rng(2)
    assert rv.score(closes, [np.ones(N)] * 2, START, fee=0)["psr0"] == 0.5  # always long: no timing at all
    assert rv.score(closes, skilled_positions(closes, rng), START)["psr0"] > 0.99
    s = rv.score(closes, noise_positions(rng), START)
    assert s["n_obs"] == N - START and s["timing_sharpe"] == pytest.approx(s["sr_bar"] * np.sqrt(8760))


def test_shifted_copies_keep_the_trading_and_lose_the_edge():
    closes, rng = market(), np.random.default_rng(3)
    pos = skilled_positions(closes, rng)
    null = [rv.shifted(pos, rng) for _ in range(200)]
    assert all(n[0].mean() == pos[0].mean() for n in null)
    assert all(abs(np.abs(np.diff(n[0])).sum() - np.abs(np.diff(pos[0])).sum()) <= 1 for n in null)
    rate = np.mean([rv.score(closes, n, START)["psr0"] >= 0.95 for n in null])
    assert rate < 0.15  # close to the 5% a test at this level lets through by luck


def test_shortlist_rules():
    closes, rng = market(), np.random.default_rng(4)
    scored = [dict(rv.score(closes, noise_positions(rng), START), name=f"noise{i}") for i in range(60)]
    scored.append(dict(rv.score(closes, skilled_positions(closes, rng), START), name="skilled"))
    var, sr0 = rv.pick_shortlist(scored, n_trials=500)
    picked = [s["name"] for s in scored if s["shortlisted"]]
    assert "skilled" in picked and len(picked) <= rv.MAX_SHORTLIST
    assert sr0 > 0 and all(s["dsr"] <= s["psr0"] for s in scored)  # deflating never raises confidence
    assert len(picked) - 1 <= 8  # about 5% of 60 noise strategies slip through option B by luck
    rv.pick_shortlist(scored, n_trials=500, option="A")
    assert [s["name"] for s in scored if s["shortlisted"]] == ["skilled"]  # option A lets luck through far less

    many = [dict(rv.score(closes, skilled_positions(closes, rng), START)) for _ in range(15)]
    rv.pick_shortlist(many, n_trials=15)
    kept = [s for s in many if s["shortlisted"]]
    assert len(kept) == rv.MAX_SHORTLIST
    assert min(s["timing_sharpe"] for s in kept) >= max(s["timing_sharpe"] for s in many if not s["shortlisted"])


def test_preregistration_text():
    summary = {"run_id": "r", "pass_no": 1, "option": "B", "ran_at": "t", "n_trials": 500, "candidates": 40,
               "rejected_on_longer_data": 2, "expected_best_sharpe_from_luck": 2.1, "real_pass_rate": 0.1,
               "null_pass_rate": 0.05, "shortlist": [{"attempt_id": 7, "code_sha": "ab" * 32, "timing_sharpe": 1.5, "psr0": 0.99, "dsr": 0.2}]}
    text = rv.preregistration(summary, "abc1234")
    assert "ab" * 32 in text and "Primary test" in text and "abc1234" in text and "0.9500" in text
    empty = rv.preregistration({**summary, "shortlist": []}, "abc1234")
    assert "zero survivors" in empty and "Primary test" not in empty


def test_registration_guard(tmp_path):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True, capture_output=True)

    code = "def strategy(df):\n    return df.close * 0\n"
    import hashlib
    sha = hashlib.sha256(code.encode()).hexdigest()
    picks = [{"attempt_id": 1, "code": code, "code_sha": sha}]
    git("init", "-q")
    assert any("does not exist" in p for p in open_holdout.registration_problems(picks, tmp_path))
    (tmp_path / "PREREGISTRATION.md").write_text(f"shortlist: {sha}")
    (tmp_path / "backtest.py").write_text("FEE = 0.001\n")
    git("add", "-A"), git("commit", "-qm", "prereg")
    assert any("tag" in p for p in open_holdout.registration_problems(picks, tmp_path))
    git("tag", "prereg-v1")
    assert open_holdout.registration_problems(picks, tmp_path) == []

    (tmp_path / "backtest.py").write_text("FEE = 0.0\n")  # the judge edited after registration
    assert any("backtest.py" in p for p in open_holdout.registration_problems(picks, tmp_path))
    git("checkout", "-q", "backtest.py")
    assert any("no longer matches" in p for p in open_holdout.registration_problems([{**picks[0], "code": code + "#"}], tmp_path))
    assert any("not in PREREGISTRATION" in p for p in open_holdout.registration_problems(
        [{"attempt_id": 2, "code": "x", "code_sha": hashlib.sha256(b"x").hexdigest()}], tmp_path))


def test_shift_null_is_conservative_for_calendar_strategies():
    """Documents the limit stated in rv.shifted: whole-day shifts leave an hour-of-day rule intact."""
    hours = np.arange(N) % 24
    pos = [(hours < 6).astype(float)] * 2
    assert np.array_equal(np.roll(pos[0], 24 * 50), pos[0])
    rng = np.random.default_rng(5)
    unchanged = np.mean([np.array_equal(rv.shifted(pos, rng)[0], pos[0]) for _ in range(2000)])
    assert unchanged == pytest.approx(1 / 24, abs=0.015)


# --- end to end through the real sandbox, on a synthetic market with one genuine, honest edge ---

MOMENTUM = "def strategy(df):\n    return (df.close.pct_change() > 0).astype(float)\n"
SMA = "def strategy(df):\n    return (df.close.rolling({a}).mean() > df.close.rolling({b}).mean()).astype(float)\n"
CHEAT_ON_LONGER_DATA = ("def strategy(df):\n    if len(df) <= 5000:\n        return (df.close.pct_change(2) > 0).astype(float)\n"
                        "    return (df.close.shift(-1) > df.close).astype(float)\n")


@pytest.mark.skipif(not image_ready(), reason="needs Docker and the sandbox image")
def test_validation_then_holdout_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")
    monkeypatch.setattr(rv, "NULL_DRAWS", 20)
    monkeypatch.setattr(open_holdout, "PERMUTATIONS", 100)
    rng = np.random.default_rng(0)
    n, disc_end, val_end = 9000, 5000, 7000
    ts = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    frames = {}
    for s in ("AAA", "BBB"):
        r = rng.normal(0, 0.01, n)
        for t in range(1, n):
            r[t] += 0.3 * r[t - 1]  # returns carry over: an up bar makes the next one likelier to be up
        frames[s] = pd.DataFrame({"ts": ts, "close": 100 * np.cumprod(1 + r), "volume": 1.0})
    disc = {s: f.iloc[:disc_end] for s, f in frames.items()}
    through_val = {s: f.iloc[:val_end] for s, f in frames.items()}

    codes = [MOMENTUM, SMA.format(a=24, b=96), SMA.format(a=12, b=48), CHEAT_ON_LONGER_DATA]
    ids = [pipeline.attempt(c, disc, sandbox.pack(disc), run_id="v", pilot=False, tries=0, hypothesis="h")["id"] for c in codes]
    pipeline.attempt(MOMENTUM.replace("()", "(3)"), disc, sandbox.pack(disc), run_id="p", pilot=True, tries=0, hypothesis="pilot")

    summary = rv.validate("v", through_val, disc_end)
    assert summary["n_trials"] == 4 and summary["pass_no"] == 1
    assert summary["rejected_on_longer_data"] == 1  # honest on discovery, caught once more bars exist
    assert [s["attempt_id"] for s in summary["shortlist"]][0] == ids[0]
    rows = {r["attempt_id"]: r for r in store.query("SELECT * FROM validation")}
    assert rows[ids[3]]["status"] == "lookahead" and rows[ids[0]]["shortlisted"] and len(rows) == summary["candidates"]
    assert "validation" not in agents.summary("v").lower() and "PSR" not in agents.summary("v")
    assert rv.validate("v", through_val, disc_end)["pass_no"] == 2  # a second pass is numbered, never silent
    json.dumps(summary)

    picks = open_holdout.shortlist("v")
    assert ids[0] in [p["attempt_id"] for p in picks] and all(len(p["code_sha"]) == 64 for p in picks)
    result = open_holdout.open_once("v", frames, val_end, results=tmp_path / "results")
    assert result["primary"]["passed"] and result["primary"]["permutation_p"] < 0.05
    assert result["strategies"][0]["attempt_id"] == ids[0] and result["strategies"][0]["survived"]
    assert json.loads((tmp_path / "results" / "HOLDOUT_OPENED.json").read_text())["run_id"] == "v"
    with pytest.raises(SystemExit, match="already opened"):
        open_holdout.open_once("v", frames, val_end, results=tmp_path / "results")


def test_empty_shortlist_never_opens_the_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")
    store.add({"run_id": "v", "pilot": False, "status": "ok"})
    with pytest.raises(SystemExit, match="zero survivors"):
        open_holdout.open_once("v", {}, 0, results=tmp_path / "results")
    assert not (tmp_path / "results" / "HOLDOUT_OPENED.json").exists()
