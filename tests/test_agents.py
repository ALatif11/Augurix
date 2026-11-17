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


def _ok(score, trades, hypothesis="long above the 200h average", **extra):
    store.add({"run_id": "r", "tries": 0, "family": "trend", "hypothesis": hypothesis, "status": "ok",
               "timing_sharpe": score, "trades_per_symbol": trades, **extra})


def test_summary_shows_scores_by_turnover_and_forgets_earlier_rounds(monkeypatch):
    _ok(0.4, 80), _ok(-3.0, 2000), _ok(-5.0, 4000)
    store.add_lesson("r", 3, "  Next: try OLD idea")
    assert "under 300: +0.40 over 1; 300 to 1000: none yet; over 1000: -4.00 over 2" in agents.summary("r")
    assert "OLD idea" in agents.summary("r")
    monkeypatch.setattr(agents, "MEMORY", 3)  # three attempts fill a round, so the next one starts with nothing
    assert agents.summary("r") == "Nothing has been tried yet."
    _ok(0.1, 50, "buy the fourth idea of the run")
    text = agents.summary("r")
    assert "Attempts so far: 1." in text and "fourth idea" in text and "200h average" not in text and "OLD idea" not in text


def test_propose_names_attempts_that_have_been_built_on_enough():
    _ok(1.0, 100)                                      # id 1
    _ok(0.5, 100, "Refine #1 with a volume filter")    # refine turn (turn 1)
    _ok(0.5, 100, "Combine #1 and a range filter")
    _ok(0.5, 100, "a third variation", parent_id=1)
    _ok(0.5, 100, "something unrelated to the others")
    seen = []

    def llm(role, system, user, schema=None, temperature=0.8):
        seen.append(user)
        return '{"family": "trend", "hypothesis": "Long when the 24h mean is above the 168h mean.", "parent_id": null}'

    agents.propose("r", llm)  # turn 5: a refine turn
    assert "Do not use them again: #1." in seen[0]


def test_review_passes_on_only_the_previous_next_steps():
    _ok(0.4, 80)
    store.add_lesson("r", 0, "  Worked: OLDWORKED\n  Failed: OLDFAILED\n  Next: try the OLDNEXT idea")
    seen = []

    def llm(role, system, user, schema=None, temperature=0.8):
        seen.append(user)
        return '{"worked": [], "failed": ["#1 lost"], "next_steps": ["something new"]}'

    assert "Next: something new" in agents.review("r", 0, llm)
    assert "OLDNEXT" in seen[0] and "OLDWORKED" not in seen[0] and "Do not repeat" in seen[0]


def test_hypothesis_length_is_capped():
    with pytest.raises(agents.ValidationError):
        agents.Hypothesis(family="trend", hypothesis="x" * 451)


def test_extract_code_accepts_a_reply_cut_off_before_the_closing_fence():
    assert agents.extract_code("```python\ndef strategy(df):\n    return df.close\n") == "def strategy(df):\n    return df.close\n"
