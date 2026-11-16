import pytest

import agents
import store


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")


def test_extract_code_takes_the_fenced_block():
    reply = "Here you go:\n```python\ndef strategy(df):\n    return df.close * 0\n```\nHope it helps."
    assert agents.extract_code(reply) == "def strategy(df):\n    return df.close * 0\n"
    assert agents.extract_code("def strategy(df):\n    return 1") == "def strategy(df):\n    return 1\n"


def test_summary_shows_scores_and_failures_but_never_code():
    store.add({"run_id": "r", "family": "trend", "hypothesis": "long above the 200h average", "code": "SECRET",
               "status": "ok", "timing_sharpe": 0.4, "trades_per_symbol": 80})
    store.add({"run_id": "r", "family": "volume", "hypothesis": "buy volume spikes", "code": "SECRET",
               "status": "crash", "error": "KeyError: 'vol'"})
    text = agents.summary("r")
    assert "Attempts so far: 2" in text and "timing Sharpe 0.40" in text and "KeyError" in text
    assert "SECRET" not in text
    assert agents.summary("other") == "Nothing has been tried yet."


def test_propose_validates_json_and_retries_once():
    replies = iter(['{"family": "astrology", "hypothesis": "x"}',
                    '```json\n{"family": "trend", "hypothesis": "Long when the 24h mean is above the 168h mean.", "parent_id": null}\n```'])
    seen = []

    def llm(role, system, user, schema=None, temperature=0.8):
        seen.append(user)
        return next(replies)

    h = agents.propose("r", llm)
    assert h.family == "trend" and h.parent_id is None and len(seen) == 2
    assert "least-tried family" in seen[0]
