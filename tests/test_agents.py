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
    llm, seen = _researcher(*[stubborn] * 5)
    assert agents.propose("r", llm) is None and len(seen) == 5  # after five asks the turn is skipped, never accepted
    assert "least-tried family" in seen[3] and "not accepted" not in seen[3]  # the fourth ask explores elsewhere, afresh


def test_after_three_refusals_the_turn_explores_another_family():
    _ok(1.2, 58, "Enter when the close exceeds the 480-hour high and volume is 1.3 times its 24-hour mean.")
    _ok(1.1, 79, "Enter when the close exceeds the 360-hour high and volume is 1.5 times its 48-hour mean.")
    again = ("Enter when the close exceeds the 240-hour high and volume is 1.4 times its 24-hour mean.", None)
    fresh = ("Enter when the 24-hour rolling mean of quote_volume crosses above its 168-hour mean. Exit when it falls below.", None)
    llm, seen = _researcher(again, again, again, fresh)
    assert agents.propose("r", llm).hypothesis == fresh[0] and len(seen) == 4
    family = seen[3].split("least-tried family: ")[1].split(".")[0]
    assert family != "trend"  # never the family it was just stuck in


def test_style_sets_the_holding_paragraph_and_both_alternates_two_at_a_time():
    systems = []

    def llm(role, system, user, schema=None, temperature=0.8):
        systems.append(system)
        return agents.Hypothesis(family="volume", hypothesis=f"an idea numbered {len(systems)} about volume").model_dump_json()

    agents.propose("r", llm, "day")
    assert "closed within 24 hours" in systems[-1] and "{holding}" not in systems[-1]
    agents.propose("r", llm)
    assert "holding for days, not hours" in systems[-1]
    kinds = []
    for k in range(6):
        kinds.append(agents.style_for("r", "both"))
        _ok(0.1, 50, f"filler idea {k}")
    assert kinds == ["long", "long", "day", "day", "long", "long"]


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


BREAKOUT = "Enter when the close price is above the maximum price of the previous {w}-hour window and {f}. Exit when the close price falls below the 168-hour rolling mean."


def test_one_rule_with_its_filter_swapped_is_turned_away_after_three():
    # pilot5's top ten: the same breakout entry and exit, a different second filter each time
    for w, f in [(672, "the 24-hour rolling standard deviation of close prices is lower than the 336-hour one"),
                 (720, "the 168-hour rolling mean of close prices is increasing"),
                 (672, "the 168-hour rolling mean of quote_volume is greater than the 336-hour rolling mean of quote_volume")]:
        _ok(1.3, 89, BREAKOUT.format(w=w, f=f))
    fourth = (BREAKOUT.format(w=720, f="the 24-hour rolling mean of (high - low) is increasing"), None)
    different = ("Enter when the 24-hour rolling mean of volume is above its 168-hour rolling mean and the close is above the "
                 "48-hour rolling mean. Exit when the 24-hour rolling mean of volume falls below its 168-hour rolling mean.", None)
    llm, seen = _researcher(fourth, different)
    assert agents.propose("r", llm).hypothesis == different[0]
    assert "3 earlier ideas already have the same exit" in seen[1] and "maximum price of the previous hour window" in seen[1]


def test_the_skeleton_check_leaves_unrelated_ideas_alone():
    a = agents._skeleton(BREAKOUT.format(w=672, f="the 168-hour rolling mean of close prices is increasing"))
    b = agents._skeleton("Enter when the close price is above the maximum price of the previous 336-hour window and volume rises. "
                         "Exit when the close price falls below the 24-hour rolling minimum.")
    assert a[0] & b[0] and a[1] != b[1]  # same entry, different exit: a different rule
    assert agents._skeleton("buy volume spikes")[1] == ""  # no exit stated, so it can never match anything
    h = agents.Hypothesis(family="trend", hypothesis=BREAKOUT.format(w=720, f="the 24-hour rolling mean of (high - low) is increasing"))
    assert agents._objection(h, agents.Counter(), set(), [a, b, b]) is None  # only one earlier idea shares its exit


def test_in_a_both_run_each_style_is_an_island():
    store.add({"run_id": "r", "tries": 0, "family": "trend", "prompt_version": "v8-pilot/long", "hypothesis": "LONGIDEA hold above the 480-hour mean",
               "status": "ok", "timing_sharpe": 1.3, "trades_per_symbol": 40})
    store.add({"run_id": "r", "tries": 0, "family": "trend", "prompt_version": "v8-pilot/long", "hypothesis": "LONGIDEA two", "status": "ok",
               "timing_sharpe": 1.2, "trades_per_symbol": 40})
    store.add({"run_id": "r", "tries": 0, "family": "time_of_day", "prompt_version": "v8-pilot/day", "hypothesis": "DAYIDEA buy at hour 14",
               "status": "ok", "timing_sharpe": -1.0, "trades_per_symbol": 900})
    store.add_lesson("r/long", 2, "  Next: LONGNOTE")
    assert "LONGIDEA" not in agents.summary("r", "day") and "DAYIDEA" in agents.summary("r", "day")
    assert "LONGNOTE" in agents.summary("r", "long") and "LONGNOTE" not in agents.summary("r", "day")
    assert [l["text"] for l in store.lessons("r")] == ["  Next: LONGNOTE"]  # the run's notes include its islands' for the report
    systems, users = [], []

    def llm(role, system, user, schema=None, temperature=0.8):
        systems.append(system), users.append(user)
        return agents.Hypothesis(family="volume", hypothesis="Hold for 6 hours after volume doubles its 24-hour mean.").model_dump_json()

    agents.propose("r", llm, "both")  # three ideas so far: the fourth turn of a both run is a day idea
    assert "closed within 24 hours" in systems[0] and "LONGIDEA" not in users[0] and "DAYIDEA" in users[0]


def test_the_hold_for_n_hours_idiom_in_the_coder_prompt_holds_n_hours_from_each_entry():
    import pandas as pd
    assert "start = cond & ~cond.shift(1, fill_value=False)" in agents.CODER
    cond = pd.Series([False, True, True, True, True, True, False, False, True, False, False, False])
    start = cond & ~cond.shift(1, fill_value=False)
    positions = start.astype(float).rolling(3, min_periods=1).max()
    # entries at bars 1 and 8: in for exactly 3 bars after each, even though the first condition lasted 5 bars
    assert positions.tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
