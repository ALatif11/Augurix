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


def _researcher(*ideas):
    """A fake model that proposes the given ideas in turn, and the prompts it was sent."""
    replies, seen = iter(ideas), []

    def llm(role, system, user, schema=None, temperature=0.8):
        seen.append(user)
        idea, parent = next(replies)
        return agents.Hypothesis(family="trend", hypothesis=idea, parent_id=parent).model_dump_json()

    return llm, seen


def test_a_line_of_ideas_is_used_up_however_the_refinements_are_chained():
    _ok(1.0, 100)                                           # 1: the original
    _ok(0.5, 100, "Refine #1 with a volume filter")         # 2 builds on 1
    _ok(0.5, 100, "swap the exit for a slower one", parent_id=2)  # 3 builds on 2, not on 1
    _ok(0.5, 100, "something unrelated to the others")      # 4: its own line
    rows, _ = agents._round("r")
    assert agents._spent(rows) == set()                     # two refinements: the line is still open
    _ok(0.5, 100, "Combine #3 with a range filter")         # 5: the third refinement of the line that starts at 1
    _ok(0.5, 100, "Refine #5 again")                        # 6: one too many
    rows, _ = agents._round("r")
    assert agents._spent(rows) == {1, 2, 3, 5, 6}           # the whole line, and not the unrelated idea
    assert "#1 (used up)" in agents.summary("r") and "#4 (used up)" not in agents.summary("r")

    llm, seen = _researcher(("Take #6 and add a volatility filter to it", None), ("a new idea, built on nothing", 5),
                            ("Buy when the 480-hour high is broken, sell below the 480-hour mean", None))
    h = agents.propose("r", llm)
    assert h.hypothesis.startswith("Buy when") and len(seen) == 3
    assert "not accepted: it builds on #6" in seen[1] and "not accepted: it builds on #5" in seen[2]  # by mention, then by parent_id


def test_an_idea_that_only_changes_the_numbers_is_turned_away():
    _ok(1.2, 58, "Enter when the close exceeds the 480-hour high and volume is 1.3 times its 24-hour mean.")
    llm, seen = _researcher(("Enter when the close exceeds the 360-hour high and volume is 1.5 times its 48-hour mean.", None))
    assert agents.propose("r", llm).hypothesis.startswith("Enter when the close exceeds the 360")  # one re-tuning is allowed
    _ok(1.1, 79, "Enter when the close exceeds the 360-hour high and volume is 1.5 times its 48-hour mean.")
    llm, seen = _researcher(("Enter when the close exceeds the 240-hour high and volume is 1.4 times its 24-hour mean.", None),
                            ("Enter when the 24-hour mean of volume crosses above its 168-hour mean.", None))
    assert agents.propose("r", llm).hypothesis.startswith("Enter when the 24-hour mean")
    assert "only the numbers changed" in seen[1]

    stubborn = ("Enter when the close exceeds the 120-hour high and volume is 2.0 times its 12-hour mean.", None)
    llm, seen = _researcher(stubborn, stubborn, stubborn)
    assert agents.propose("r", llm).hypothesis == stubborn[0] and len(seen) == 3  # after three asks the run moves on


def test_explore_turns_show_what_the_family_already_scored():
    _ok(-3.5, 2251, "Enter between hours 8 and 12 when the 1-hour return is positive")
    others = [f for f in agents.FAMILIES if f != "trend"]
    for i in range(15):  # every other family has been tried more, and 16 ideas make this an explore turn
        store.add({"run_id": "r", "tries": 0, "family": others[i % 6], "hypothesis": f"filler idea {i}", "status": "crash", "error": "boom"})
    llm, seen = _researcher(("Hold while the 168-hour mean of close is above the 720-hour mean", None))
    agents.propose("r", llm)
    assert "least-tried family: trend" in seen[0] and "timing Sharpe -3.50, 2251 trades" in seen[0].split("already tried:")[1]


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
