# Augurix

A local AI proposes trading strategies and writes them as code. Plain Python decides whether any of them work.

Most "AI trading" projects show a good backtest. This one is built to find out whether a good backtest means anything. A language model running on one consumer GPU generates hundreds of strategies for Bitcoin and Ethereum; each one is sandboxed, checked for cheating, scored after fees, and corrected for the number of ideas tried. The deliverable is a **survival funnel**: how many strategies were generated, and how many survived each check, ending with one test on market data the model could not have seen. Zero survivors is an acceptable result.

> **Status:** the pipeline, judge and statistics are built and tested. Pilot runs are tuning the agents. The official run, validation and the one-shot holdout test have not happened yet, so there is no headline result. Every number below is from pilots, in-sample, and is not evidence that any strategy works.

## How it works

```
researcher ─► coder ─► static check ─► Docker sandbox ─► judge ─► log ─► reviewer
    ▲                                  (strategy + look-ahead test)          │
    └──────────────── summary of what was tried, discovery data only ◄──────┘
```

1. **Researcher agent** proposes one idea in plain English ("buy when the 168-hour average crosses above the 504-hour average; sell when it falls back"). Code turns away re-tunings of earlier ideas and near-copies, and alternates exploring new kinds of strategy with refining good ones. Day trades and longer holds can be searched as separate islands so one lucky result cannot pull the whole search toward it.
2. **Coder agent** writes the idea as a Python function that returns a position (0 to 100% long) for every hourly bar.
3. **Static check** rejects imports, file access and hard-coded calendar years before anything runs.
4. **Docker sandbox** runs the code with no network, no files and capped memory and CPU. Data goes in on stdin and positions come out on stdout.
5. **Look-ahead test** reruns the strategy on truncated copies of the data. If a position changes when later bars are removed, the code used the future and is rejected.
6. **Judge** (`backtest.py`, plain NumPy) scores the positions after a 0.1% fee per trade. No model ever scores anything.
7. **Reviewer agent** reads each batch of results and writes notes the researcher works from. Failures go back to the coder with the error, up to three tries.

All three agents are one local model (Gemma 4 12B, 4-bit, served by llama.cpp) with different prompts. Nothing calls a paid API.

## Why the results can be trusted

| Risk | What blocks it |
|---|---|
| The model remembers what the market did | Three time splits: discovery 2019–2022, validation 2023–2024, holdout from March 2025. The model's training ends January 2025, so only the holdout is truly unseen. |
| Code that peeks at future prices | The truncation test above, plus a static check for hard-coded dates. Tested against planted cheats. |
| The holdout leaks into development | Holdout months are not even downloaded until the test is pre-registered and tagged in git. The downloader refuses without the tag. |
| Trying thousands of ideas until one looks good by luck | Every attempt is logged and counted. The score that matters is the **timing Sharpe** (what a strategy earns beyond holding the same average position), judged with the probabilistic and deflated Sharpe ratio (Bailey and López de Prado). Shuffled-in-time copies of each strategy act as a null control. |
| Moving the goalposts after seeing results | `PREREGISTRATION.md` fixes the shortlist, the test and the thresholds, and is committed and tagged before the holdout data exists. `open_holdout.py` runs once, refuses if any judge file changed since the tag, and records that it ran before computing anything. |
| Generated code harming the machine | It runs only in the locked-down container. |

## What the pilots found so far

In-sample, on discovery data only:

- **Fees decide most results.** Strategies that trade more than 1,000 times over four years have a median timing Sharpe near −2.7. Day-trade strategies (held under a day) lose to the 0.2% round-trip cost almost every time.
- **The model converges.** Left alone, Gemma refines its single best result dozens of times ("hill-climbing on noise"). One run put 81 ideas on top of one lucky strategy; another produced ten top results that were one rule with a different filter each. Code-level checks for re-tunings, near-copies and over-refined lines, an exploration fallback, and per-style islands were added in response.
- **Nothing beats luck yet.** The best pilot scores sit at the level the best of that many no-skill strategies is expected to reach (about +1.3 to +1.5 annualised).
- **The look-ahead test catches real cheats.** About 3% of generated strategies used future data and were rejected.

## The dashboard

`python dashboard.py`, then http://127.0.0.1:8000. A read-only workbench over the log: filters on the left, strategies in the middle, the selected strategy on the right. Every score is drawn against zero and the luck benchmark on the same small ruler, so "is this better than luck?" is answered wherever a score appears.

- **Strategies:** linked filters (drag across histograms of score, trades and time in the market, or box-select a scatter), near-copies collapsed into one row per rule, and a replay of any strategy's trades on the historical candles with each buy and sell marked as it happens.
- **Funnel:** how many attempts survived each check, where the rest stopped, and results by family.
- **Compare runs:** up to three runs side by side.
- **Live:** a running search as it happens: progress toward the official run's 1,000 attempts, each style's scores arriving in order against the luck line, and the researcher's latest proposals with the ones code turned away.
- **Reviewer notes:** the reviewer agent's working notes.

It shows discovery data only and never fetches live prices, because current prices are holdout data.

## Running it

Requirements: Windows with WSL (Ubuntu), an NVIDIA GPU with about 10 GB free, Docker Desktop, Python 3.12 or later, and llama.cpp built in WSL with `gemma-4-12B-it-Q4_K_M.gguf`.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python download_data.py        (Binance bulk data, 2019-2024, checksums verified)
.venv\Scripts\python make_splits.py
docker build -t finproj-sandbox .
.venv\Scripts\python -m pytest -q            (134 tests; 12 need Docker running)
overnight.cmd pilot1 5 both                  (starts the model server, searches for 5 hours, stops it)
.venv\Scripts\python dashboard.py
```

The Week 5 scripts (`run_validation.py`, `open_holdout.py`) run once, in a fixed order with locks described in `PLAN.md`. Do not run them to try things out.

## Files

| File | What it does |
|---|---|
| `backtest.py` | The judge: positions to net returns and metrics |
| `lookahead.py` | Look-ahead detector (truncation test) |
| `dsr.py` | Probabilistic and deflated Sharpe ratio |
| `agents.py` | Model client and the researcher, coder and reviewer prompts and checks |
| `generate.py` | The agent loop (LangGraph), unattended runs, run reports |
| `pipeline.py`, `sandbox.py`, `harness.py`, `static_check.py` | One attempt: check, sandbox, score, log |
| `store.py` | DuckDB attempt log |
| `baselines.py` | Hand-written reference strategies |
| `download_data.py`, `make_splits.py` | Data and the time splits |
| `run_validation.py`, `open_holdout.py` | Validation, pre-registration and the one-shot holdout test |
| `dashboard.py`, `dashboard.html` | The dashboard |
| `PLAN.md` | Every decision and the reasoning behind it, the schedule and the specs |

## Limits

- Long or flat only, on two symbols, with hourly bars and a flat 0.1% fee. No shorting, leverage, slippage model or funding costs.
- Strategies can see absolute dates and price levels. That could help a rule keyed to something the model remembers, but only on data from before its cutoff, so it cannot help on the holdout.
- One 12B model at 4-bit. A stronger model might propose better ideas; whether it would find real edge is exactly what this setup is built to test.
- The holdout is about 21 months. A Sharpe ratio measured over that long has a standard error of roughly 0.75, so only a large effect can pass.
