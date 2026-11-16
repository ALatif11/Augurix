"""Generate and test strategies. Every attempt is logged; restarting continues where it stopped.

    python generate.py --ideas ideas.txt                    # Week 2: coder agent on hand-written ideas
    python generate.py --run pilot2 --n 50                  # the researcher agent proposes the ideas
    python generate.py --run gen-v1 --n 1000 --official     # after the prompt freeze (PLAN.md, Week 4)

Per idea the graph is:  propose -> write_code -> test -> (rejected for a fixable reason? write_code again) -> end
"""
import argparse
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import agents
import pipeline
import sandbox
import store

MAX_TRIES = 3
FIXABLE = {"static_reject", "crash", "bad_output", "timeout", "lookahead"}


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pilot", help="run id; attempts are grouped and resumed by it")
    p.add_argument("--n", type=int, default=20, help="stop when the run has this many attempts")
    p.add_argument("--ideas", help="text file with one plain-English idea per line; skips the researcher agent")
    p.add_argument("--official", action="store_true", help="count this run towards the trial count N")
    args = p.parse_args(argv)
    if args.official and not Path("lookahead.py").exists():
        raise SystemExit("An official run needs the look-ahead detector (lookahead.py). See PLAN.md, Week 3.")

    ideas = [s.strip() for s in Path(args.ideas).read_text(encoding="utf-8").splitlines() if s.strip()] if args.ideas else None
    frames = pipeline.load_discovery()
    graph = build(frames, sandbox.pack(frames))
    while True:
        start = {"run_id": args.run, "pilot": not args.official, "tries": 0}
        if ideas:
            done = store.count("run_id = ? AND tries = 0", [args.run])  # ideas already started
            if done >= len(ideas):
                break
            start["hypothesis"] = ideas[done]
        elif store.count("run_id = ?", [args.run]) >= args.n:
            break
        out = graph.invoke(start)
        r = out["row"]
        score = f"{r['timing_sharpe']:+.2f}" if r.get("timing_sharpe") is not None else "  -  "
        print(f"#{r['id']:<5} {r['status']:<15} tries={out['tries']} timing={score}  {out['hypothesis'][:80]}", flush=True)


if __name__ == "__main__":
    main()
