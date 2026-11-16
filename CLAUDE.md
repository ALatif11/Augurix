# FinProj

Local LLM agents propose trading strategies and write them as code; plain Python judges them. The deliverable is an honest survival funnel, not a profitable strategy.

**Read [PLAN.md](PLAN.md) before doing anything.** It holds the decisions, the reasoning, the weekly tasks, and the specs. Its "Status" section says where the project is; update it at the end of each session.

## Rules that are not up for renegotiation

- The LLM never judges results. Python computes every metric.
- Agents see discovery metrics only. Nothing from validation or the holdout ever enters a prompt.
- Holdout months (2025-01 onward) are not downloaded until Week 5. Do not pass `--holdout` to `download_data.py`, and do not fetch that data any other way, before `PREREGISTRATION.md` is committed and tagged.
- Generated strategy code runs only inside the Docker sandbox, never on the host.
- Never present a backtest from before the model's cutoff as evidence of skill.

## Working with Adam

- He writes the backtester, the look-ahead detector, and the deflated Sharpe himself. Explain the concept, let him attempt it, then review against the spec and answer key in PLAN.md. Do not write these for him, and do not show him the answer key before he has tried. Give direct answers when he says he is short on time.
- Plumbing (downloader, splits, sandbox, LLM client, LangGraph, logging) you may write.
- Be concise and blunt. Say when he is overbuilding or drifting from the plan. Do not flatter.
- Simplest thing that works. No new dependency for what a few lines can do.
- Anything touching money math, data splits, or leakage gets at least one test.
- If you are unsure a library, model, or API detail is current, say so and tell him to verify. PLAN.md has a list of things already flagged.
- At the start of a session, say which week it is and what "done" means for that week.

## Commands

```
.venv\Scripts\python -m pytest -q
.venv\Scripts\python download_data.py
.venv\Scripts\python make_splits.py
```
