"""Generate and test strategies. Every attempt is logged; restarting continues where it stopped.

    python generate.py --ideas ideas.txt                    # coder agent on hand-written ideas
    python generate.py --run night1 --hours 5               # unattended: research for five hours, then report
    python generate.py --run pilot2 --n 50                  # stop at 50 attempts instead of a time
    python generate.py --run gen-v1 --n 1000 --official     # after the prompt freeze (PLAN.md, Week 4)
    python generate.py --run night1 --report                # only rewrite the report for a run

Per idea the graph is:  propose -> write_code -> test -> (rejected for a fixable reason? write_code again) -> end
Every --review-every attempts a reviewer agent turns the latest results into lessons, which the
researcher reads from then on. When the run ends it writes runs/<run>_report.md.
"""
import argparse
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import agents
import dsr
import pipeline
import sandbox
import store

MAX_TRIES = 3
MAX_FAILURES = 10  # infrastructure failures in a row before an unattended run gives up
FIXABLE = {"static_reject", "crash", "bad_output", "timeout", "lookahead", "no_trades"}
BACKTESTED = ("ok", "too_few_trades", "duplicate")


class State(TypedDict, total=False):
    run_id: str
    pilot: bool
    hypothesis: str
    family: str | None
    parent_id: int | None
    code: str
    row: dict
    tries: int


def build(frames, packed, llm=agents.chat):
    def propose(s):
        h = agents.propose(s["run_id"], llm)
        return {"hypothesis": h.hypothesis, "family": h.family, "parent_id": h.parent_id}

    def write_code(s):
        prev = s.get("row")
        return {"code": agents.write_code(s["hypothesis"], llm, prev and prev["code"], prev and prev["error"])}

    def test(s):
        prev = s.get("row")
        row = pipeline.attempt(
            s["code"], frames, packed, run_id=s["run_id"], pilot=s["pilot"], model=agents.MODEL,
            prompt_version=agents.PROMPT_VERSION, family=s.get("family"), hypothesis=s["hypothesis"],
            parent_id=prev["id"] if prev else s.get("parent_id"), tries=s["tries"])
        return {"row": row, "tries": s["tries"] + 1}

    g = StateGraph(State)
    g.add_node("propose", propose)
    g.add_node("write_code", write_code)
    g.add_node("test", test)
    g.add_conditional_edges(START, lambda s: "write_code" if s.get("hypothesis") else "propose",
                            ["write_code", "propose"])
    g.add_edge("propose", "write_code")
    g.add_edge("write_code", "test")
    g.add_conditional_edges(
        "test", lambda s: "write_code" if s["row"]["status"] in FIXABLE and s["tries"] < MAX_TRIES else END,
        ["write_code", END])
    return g.compile()


def keep_awake():
    """Stop Windows from sleeping while this process runs; an overnight run is useless if the machine dozes off."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def review_if_due(run_id, every):
    """Every `every` attempts, have the reviewer agent write lessons from the results since its last review."""
    last = store.lessons(run_id, 1)
    after = last[0]["after_attempt"] if last else 0
    if store.count("run_id = ? AND id > ?", [run_id, after]) < every:
        return
    newest = store.query("SELECT max(id) AS m FROM attempts WHERE run_id = ?", [run_id])[0]["m"]
    try:
        text = agents.review(run_id, after)
    except Exception as e:  # a bad review must never stop the search; it is retried after the next attempt
        print(f"! review skipped: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return
    store.add_lesson(run_id, newest, text)
    print(f"~ lessons after #{newest}:\n{text}", flush=True)


def write_report(run_id):
    """Write runs/<run>_report.md: what the run did and found, all from the log. Returns the path."""
    rows = store.query("SELECT id, tries, pilot, model, prompt_version, family, hypothesis, status, error, lookahead, "
                       "timing_sharpe, trades_per_symbol, mean_position, created_at FROM attempts "
                       "WHERE run_id = ? ORDER BY id", [run_id])
    path = store.DB.parent / f"{run_id}_report.md"
    if not rows:
        path.write_text(f"# Run {run_id}\n\nNo attempts were logged.\n", encoding="utf-8")
        return path
    scored = [r for r in rows if r["status"] == "ok"]
    scores = [r["timing_sharpe"] for r in scored]
    hours = max((rows[-1]["created_at"] - rows[0]["created_at"]).total_seconds() / 3600, 1e-9)
    count = lambda pred: sum(1 for r in rows if pred(r))
    idea = lambda r: " ".join((r["hypothesis"] or "").split())[:200]
    out = [
        f"# Run {run_id}", "",
        f"{'Pilot' if rows[0]['pilot'] else 'Official'} run, {rows[0]['model'] or 'hand-written'}, prompts {rows[-1]['prompt_version']}. "
        f"{rows[0]['created_at']:%Y-%m-%d %H:%M} to {rows[-1]['created_at']:%H:%M} UTC: {len(rows)} attempts on "
        f"{count(lambda r: r['tries'] == 0)} ideas, about {len(rows) / hours:.0f} attempts an hour.", "",
        "**Every score below is in-sample (discovery period, 2019–2022). None of it is evidence a strategy works.**"
        + (" This was a pilot: its attempts do not count toward the trial count and must never be validated." if rows[0]["pilot"] else ""), "",
        "## Funnel", "", "| Stage | Attempts |", "|---|---|",
        f"| Generated | {len(rows)} |",
        f"| Passed static check | {count(lambda r: r['status'] != 'static_reject')} |",
        f"| Ran in the sandbox | {count(lambda r: r['status'] in BACKTESTED + ('lookahead', 'no_trades'))} |",
        f"| Passed look-ahead test | {count(lambda r: r['lookahead'] == 'pass')} |",
        f"| Backtested with enough trades, not a duplicate | {len(scored)} |",
        f"| Timing Sharpe above 0 (discovery gate) | {count(lambda r: r['status'] == 'ok' and r['timing_sharpe'] > 0)} |", "",
        "## Outcomes", "", "| Outcome | Attempts |", "|---|---|",
    ]
    statuses = sorted({r["status"] for r in rows}, key=lambda s: -count(lambda r: r["status"] == s))
    out += [f"| {s} | {count(lambda r: r['status'] == s)} |" for s in statuses]
    if len(scores) > 1:
        # An annual Sharpe measured over Y years has sampling variance of about 1/Y. The spread of the
        # scores themselves is the wrong input here: it is mostly fee losses from overtrading, not noise
        # (night1: variance 8.6, which put the "luck" bar at an absurd +8.75).
        luck = dsr.expected_max_sharpe(1 / pipeline.DISCOVERY_YEARS, len(scores))
        out += ["", "## Best by discovery timing Sharpe", "",
                f"Median score {statistics.median(scores):+.2f}. The best of {len(scores)} strategies with no skill at all is expected to "
                f"reach about {luck:+.2f} by luck alone, so a top score near that number means nothing.", "",
                "| # | Family | Timing Sharpe | Trades | In market | Idea |", "|---|---|---|---|---|---|"]
        out += [f"| {r['id']} | {r['family'] or '-'} | {r['timing_sharpe']:+.2f} | {r['trades_per_symbol']:.0f} | {r['mean_position']:.0%} | {idea(r)} |"
                for r in sorted(scored, key=lambda r: -r["timing_sharpe"])[:10]]
        out += ["", "## By family", "", "| Family | Attempts | Backtested | Median | Best |", "|---|---|---|---|---|"]
        for fam in sorted({r["family"] or "-" for r in rows}):
            fs = [r["timing_sharpe"] for r in scored if (r["family"] or "-") == fam]
            out.append(f"| {fam} | {count(lambda r: (r['family'] or '-') == fam)} | {len(fs)} | "
                       + (f"{statistics.median(fs):+.2f} | {max(fs):+.2f} |" if fs else "- | - |"))
        heavy = [r["timing_sharpe"] for r in scored if r["trades_per_symbol"] > 1000]
        light = [r["timing_sharpe"] for r in scored if r["trades_per_symbol"] <= 1000]
        out += ["", "## Trading fees", "",
                f"{len(heavy)} of {len(scored)} backtested strategies made over 1,000 trades per symbol"
                + (f". Their median score is {statistics.median(heavy):+.2f}" if heavy else "")
                + (f", against {statistics.median(light):+.2f} for the rest." if heavy and light else ".")]
    failed = [r for r in rows if r["error"]]
    if failed:
        reasons = {}
        for r in failed:
            key = f"{r['status']}: " + " ".join(r["error"].split())[-110:]
            reasons[key] = reasons.get(key, 0) + 1
        out += ["", "## Most common failures", ""] + [f"- {n}× {k}" for k, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]]
    notes = store.lessons(run_id, 50)
    if notes:
        out += ["", "## What the reviewer agent recorded", "",
                "Written by the model from discovery results only. Read them as its working notes, not as findings."]
        for n in notes:
            out += ["", f"After attempt #{n['after_attempt']}:", "", "```", n["text"], "```"]
    window = [rows[0]["created_at"] - timedelta(minutes=10), rows[-1]["created_at"]]  # the first call precedes the first attempt
    calls = store.query("SELECT role, count(*) AS n, median(seconds) AS s, median(tokens_out) AS t FROM llm_calls "
                        "WHERE created_at >= ? AND created_at <= ? GROUP BY role ORDER BY role", window)
    if calls:
        out += ["", "## Model calls during the run", "", "| Role | Calls | Median seconds | Median tokens out |", "|---|---|---|---|"]
        out += [f"| {c['role']} | {c['n']} | {c['s']:.1f} | {c['t'] or 0:.0f} |" for c in calls]
        # agents.propose re-asks with the reason in the prompt, so the log of prompts is the record of refusals
        back = store.query("SELECT count(*) FILTER (prompt LIKE '%not accepted: it is an earlier idea%') AS retuned, "
                           "count(*) FILTER (prompt LIKE '%not accepted: it builds on%') AS spent, "
                           "count(*) FILTER (prompt LIKE '%earlier ideas already have the same exit%') AS swapped FROM llm_calls "
                           "WHERE role = 'researcher' AND created_at >= ? AND created_at <= ?", window)[0]
        out += ["", f"Proposals the code sent back to the researcher: {back['retuned']} for only changing the numbers of an earlier "
                    f"idea, {back['spent']} for building on a line of ideas that was already used up, {back['swapped']} for keeping an earlier "
                    "rule's exit and entry and swapping only a filter."]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pilot", help="run id; attempts are grouped and resumed by it")
    p.add_argument("--n", type=int, help="stop when the run has this many attempts (default 20 unless --hours is given)")
    p.add_argument("--hours", type=float, help="keep going for this long, then stop after the idea in progress")
    p.add_argument("--ideas", help="text file with one plain-English idea per line; skips the researcher agent")
    p.add_argument("--review-every", type=int, default=20, help="attempts between reviewer passes")
    p.add_argument("--official", action="store_true", help="count this run towards the trial count N")
    p.add_argument("--report", action="store_true", help="only write the report for --run and exit")
    args = p.parse_args(argv)
    if args.report:
        print(write_report(args.run))
        return
    if args.official and not Path("lookahead.py").exists():
        raise SystemExit("An official run needs the look-ahead detector (lookahead.py). See PLAN.md, Week 3.")

    limit = args.n if args.n is not None else (None if args.hours else 20)
    deadline = time.time() + args.hours * 3600 if args.hours else None
    ideas = [s.strip() for s in Path(args.ideas).read_text(encoding="utf-8").splitlines() if s.strip()] if args.ideas else None
    keep_awake()
    frames = pipeline.load_discovery()
    graph = build(frames, sandbox.pack(frames))
    failures = 0
    while not (deadline and time.time() >= deadline):
        start = {"run_id": args.run, "pilot": not args.official, "tries": 0}
        if ideas:
            done = store.count("run_id = ? AND tries = 0", [args.run])  # ideas already started
            if done >= len(ideas):
                break
            start["hypothesis"] = ideas[done]
        elif limit is not None and store.count("run_id = ?", [args.run]) >= limit:
            break
        try:
            out = graph.invoke(start)
        except Exception as e:
            # Model server restarting, Docker hiccup, a reply that would not parse twice: wait and carry on.
            # Nothing was logged for this idea, so it is simply tried again.
            failures += 1
            print(f"! {type(e).__name__}: {str(e)[:200]} ({failures} in a row)", flush=True)
            if failures >= MAX_FAILURES:
                print(f"stopping: {MAX_FAILURES} failures in a row", flush=True)
                break
            time.sleep(min(30 * failures, 300))
            continue
        failures = 0
        r = out["row"]
        score = f"{r['timing_sharpe']:+.2f}" if r.get("timing_sharpe") is not None else "  -  "
        print(f"#{r['id']:<5} {r['status']:<15} tries={out['tries']} timing={score}  {out['hypothesis'][:80]}", flush=True)
        review_if_due(args.run, args.review_every)
    print(f"report: {write_report(args.run)}", flush=True)


if __name__ == "__main__":
    main()
