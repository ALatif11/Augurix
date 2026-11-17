"""The unattended loop: time limit, survival of failures, reviews, and the final report. No Docker, no model."""
import subprocess

import pytest

import agents
import generate
import sandbox
import store

BUILD = generate.build  # the real graph; the loop fixture replaces generate.build


class FakeGraph:
    """Stands in for the LangGraph pipeline: logs one attempt per call, failing on the calls listed."""

    def __init__(self, fail_on=()):
        self.calls, self.fail_on = 0, set(fail_on)

    def invoke(self, start):
        self.calls += 1
        if self.calls in self.fail_on:
            raise OSError("model server unreachable")
        score = 0.3 * self.calls - 0.6
        row = {"run_id": start["run_id"], "pilot": start["pilot"], "tries": 0, "family": "trend", "status": "ok",
               "hypothesis": f"idea {self.calls}", "timing_sharpe": score, "trades_per_symbol": 100.0,
               "mean_position": 0.5, "lookahead": "pass", "model": "fake", "prompt_version": "t"}
        row["id"] = store.add(row)
        return {"row": row, "tries": 1, "hypothesis": row["hypothesis"]}


@pytest.fixture
def loop(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")
    monkeypatch.setattr(generate.pipeline, "load_discovery", lambda: {})
    monkeypatch.setattr(sandbox, "pack", lambda frames: b"")
    monkeypatch.setattr(generate, "keep_awake", lambda: None)
    monkeypatch.setattr(generate.time, "sleep", lambda s: None)
    monkeypatch.setattr(agents, "review", lambda run_id, after, island=None: f"  Worked: nothing | reviewed after {after}")

    def run(graph, *args):
        monkeypatch.setattr(generate, "build", lambda frames, packed, style="long": graph)
        generate.main(["--run", "t", *args])
        return graph
    return run


def test_failures_are_survived_and_reviews_happen_on_schedule(loop, tmp_path):
    graph = loop(FakeGraph(fail_on={2, 3}), "--n", "5", "--review-every", "2")
    assert store.count("run_id = 't'") == 5 and graph.calls == 7  # two failed calls logged nothing and were retried
    notes = store.lessons("t")
    assert len(notes) == 2 and notes[0]["after_attempt"] < notes[1]["after_attempt"]
    assert "reviewed after 0" in notes[0]["text"]
    report = (tmp_path / "t_report.md").read_text(encoding="utf-8")
    assert "5 attempts" in report and "in-sample" in report and "pilot" in report.lower()
    assert "| 5 | trend | +1.50 |" in report and "What the reviewer agent recorded" in report
    assert "about +0.60 by luck alone" in report  # best of 5 no-skill annual Sharpes over 4 years: sd 0.5 times 1.19
    assert "backtested strategies made over 1,000 trades per symbol" in report


def test_persistent_failure_stops_the_run_instead_of_spinning(loop):
    graph = loop(FakeGraph(fail_on=set(range(1, 100))), "--n", "5")
    assert graph.calls == generate.MAX_FAILURES and store.count() == 0


def test_a_skipped_turn_logs_nothing_and_endless_skipping_stops_the_run(loop, monkeypatch):
    turns = []
    monkeypatch.setattr(agents, "propose", lambda run_id, llm, style="long", kind=None: turns.append(1))  # every proposal turned away
    loop(BUILD({}, b"", llm=None), "--n", "5")
    assert store.count() == 0 and len(turns) == generate.MAX_FAILURES  # nothing logged, and it stopped instead of spinning


def test_a_stuck_style_rests_while_the_other_one_works(loop, monkeypatch):
    # gen-v1 stopped at 291 attempts: every skipped long-hold turn handed the next turn to long holds again
    asked = []

    def propose(run_id, llm, style="long", kind=None):
        asked.append(kind)
        return None if kind == "long" else agents.Hypothesis(family="trend", hypothesis=f"day trading idea number {len(asked)}")

    def attempt(code, frames, packed, **row):
        row = {**row, "status": "ok", "timing_sharpe": 0.1, "trades_per_symbol": 50.0, "mean_position": 0.5, "lookahead": "pass"}
        row["id"] = store.add(row)
        return row

    monkeypatch.setattr(agents, "propose", propose)
    monkeypatch.setattr(agents, "write_code", lambda *a, **k: "code")
    monkeypatch.setattr(generate.pipeline, "attempt", attempt)
    loop(BUILD({}, b"", llm=None, style="both"), "--n", "25", "--style", "both")
    assert store.count() == 25  # the run went on: day trades took the turns long holds could not use
    assert asked[:2] == ["long", "day"] and asked.count("long") >= 2  # long holds rested, then were asked again


def test_time_limit_ends_the_run_and_still_writes_a_report(loop, tmp_path):
    graph = loop(FakeGraph(), "--hours", "0.0000001")
    assert graph.calls <= 1 and (tmp_path / "t_report.md").exists()


def test_a_run_resumes_where_it_stopped(loop):
    loop(FakeGraph(), "--n", "3")
    graph = loop(FakeGraph(), "--n", "5")
    assert graph.calls == 2 and store.count("run_id = 't'") == 5


def test_review_text_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.duckdb")
    store.add({"run_id": "r", "tries": 0, "family": "trend", "hypothesis": "long above the 200h average", "status": "ok",
               "timing_sharpe": 0.8, "trades_per_symbol": 90})
    store.add({"run_id": "r", "tries": 0, "family": "volume", "hypothesis": "buy volume spikes", "status": "crash", "error": "KeyError"})
    seen = {}

    def llm(role, system, user, schema=None, temperature=0.8):
        seen[role] = user
        return '{"worked": ["#1 slow trend filters"], "failed": ["#2 crashed on a missing column"], "next_steps": ["try volume with the right column name"]}'

    text = agents.review("r", 0, llm)
    assert "Worked: #1 slow trend filters" in text and "Next: try volume" in text
    assert "timing Sharpe 0.80" in seen["reviewer"] and "KeyError" in seen["reviewer"]
    store.add_lesson("r", 2, text)
    assert "Lessons from earlier reviews" in agents.summary("r") and "#1 slow trend filters" in agents.summary("r")
    assert store.lessons("other") == []


@pytest.mark.parametrize("code", [125, 3221225794])  # engine down; Windows could not start the docker client (0xC0000142)
def test_docker_failure_is_not_logged_as_a_strategy_failure(monkeypatch, code):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"Traceback: boom"))
    assert sandbox.run("def strategy(df):\n    return df.close\n", b"")["error"] == "crash"  # an ordinary failure is still the strategy's
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, code, b"", b"failed to connect to the docker API"))
    with pytest.raises(RuntimeError, match="docker could not start"):
        sandbox.run("def strategy(df):\n    return df.close\n", b"")
