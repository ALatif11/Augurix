# FinProj

Local LLM agents propose trading strategies and write them as code; plain Python judges them. The deliverable is an honest survival funnel, not a profitable strategy.

**Read [PLAN.md](PLAN.md) before doing anything.** It holds the decisions, the reasoning, the weekly tasks, and the specs. Its "Status" section says where the project is; update it at the end of each session.

## Rules that are not up for renegotiation

- The LLM never judges results. Python computes every metric.
- Agents see discovery metrics only. Nothing from validation or the holdout ever enters a prompt.
- Holdout months (2025-01 onward) are not downloaded until Week 5. Do not pass `--holdout` to `download_data.py`, and do not fetch that data any other way, before `PREREGISTRATION.md` is committed and tagged.
- `run_validation.py` runs once, on the official run, after generation is frozen. Never run it on a pilot run or to "see how things look".
- `open_holdout.py` runs once, ever. Never delete `results/HOLDOUT_OPENED.json`, move the `prereg-v1` tag, or edit a judge file after that tag to get it to run.
- Generated strategy code runs only inside the Docker sandbox, never on the host.
- Never present a backtest from before the model's cutoff as evidence of skill.

## Working with Adam

- He must be able to defend every line in an interview. The backtester (`backtest.py`), look-ahead detector (`lookahead.py`) and deflated Sharpe (`dsr.py`) were written by the assistant at his request on 2026-11-16, so he has not yet learned them by writing them. Quiz him on them with the questions in PLAN.md's specs, Socratically, until he can explain each one. Give direct answers when he says he is short on time.
- Do not change those three files casually. They are the judge; any change needs a test and a reason recorded in PLAN.md.
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
.venv\Scripts\python baselines.py
docker build -t finproj-sandbox .                      (Docker Desktop must be running)
wsl -d Ubuntu-24.04 -- bash -lc "~/llama.cpp/build/bin/llama-server -m ~/models/gemma-4-12B-it-Q4_K_M.gguf -ngl 99 -c 16384 -np 1 --reasoning-budget 0 --host 127.0.0.1 --port 8080"
                                                       (the model server; leave it running in its own terminal)
.venv\Scripts\python generate.py --ideas ideas.txt     (needs Docker and the model server running)
.venv\Scripts\python generate.py --run NAME --hours 5  (unattended pilot: research, review, report in runs/NAME_report.md)
overnight.cmd NAME HOURS [long|day|both]                (unattended: starts the model server, runs, stops it; Docker must be running)
.venv\Scripts\python dashboard.py                      (http://127.0.0.1:8000)
```

Week 5 commands (`run_validation.py`, `download_data.py --holdout`, `open_holdout.py`) are deliberately not listed here. Their order and their locks are in PLAN.md under "Week 5". Do not run any of them to test something; the tests use synthetic data.

PLAN.md has a "Code map" table saying what each file does and where it runs. `harness.py` runs only inside the container. `lookahead.py` runs there on generated code; on the host it is only ever called on trusted hand-written code (tests, baselines).
