"""The LLM side: one local model, two roles (hypothesis, coder), told apart only by prompt.

The model never sees a metric it could game: it gets discovery-period results from the
log and nothing else. Prompts must stay free of market hints, dates and "what works"
(PLAN.md, "Leakage channels"). Bump PROMPT_VERSION whenever a prompt changes.
"""
import json
import os
import re
import time
import urllib.request
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

import store

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")  # any OpenAI-compatible server
MODEL = os.environ.get("LLM_MODEL", "gemma4:12b")
PROMPT_VERSION = "v0-pilot"
FAMILIES = ["trend", "mean_reversion", "breakout", "volatility", "volume", "time_of_day", "other"]


class Hypothesis(BaseModel):
    family: Literal["trend", "mean_reversion", "breakout", "volatility", "volume", "time_of_day", "other"]
    hypothesis: str = Field(min_length=20)
    parent_id: int | None = None


def chat(role, system, user, schema=None, temperature=0.8):
    """One chat completion against the local server. Logs the call and returns the reply text."""
    body = {"model": MODEL, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": role, "schema": schema, "strict": True}}
    req = urllib.request.Request(BASE_URL + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    started = time.time()
    for last in (False, True):  # one retry; a hung or restarting server is the common overnight failure
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                reply = json.load(r)
            break
        except OSError:
            if last:
                raise
            time.sleep(5)
    text = reply["choices"][0]["message"]["content"] or ""
    usage = reply.get("usage") or {}
    store.log_llm(role=role, model=MODEL, prompt=system + "\n\n" + user, response=text,
                  tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"),
                  seconds=time.time() - started)
    return text


CODER = """You write one Python function for a trading-strategy research system.

Interface:
    def strategy(df):
        ...
        return positions

df is a pandas DataFrame of hourly bars for ONE symbol, oldest row first, with columns:
ts (UTC timestamp), open, high, low, close, volume, quote_volume, trades.
positions is a pandas Series with the same length and index as df. Each value is between
0 (out of the market) and 1 (fully long). There is no shorting.

Rules:
- The position on a row may use only that row and earlier rows. Use rolling(), expanding(),
  ewm() and shift(n) with n >= 1. Never use shift(-n), centered windows, bfill, or any
  statistic taken over the whole DataFrame (a plain .mean(), .std(), .max(), .min(),
  .quantile() or .rank() on a full column). Code that uses later rows is detected and rejected.
- Import only numpy, pandas and math. No file access.
- No calendar years or dates anywhere, including comments. Hour of day and day of week
  taken from df.ts are fine.
- Vectorised pandas and numpy only. It must finish in a few seconds on 35,000 rows.
  The environment has pandas 2.2 and numpy 1.26.
- Deterministic: no randomness.
- Every position change costs 0.1%, and a strategy that barely trades is discarded.

Reply with one ```python code block holding the complete function and its imports. Nothing else."""

RESEARCHER = """You are a quantitative researcher proposing ideas for systematic long/flat strategies
on hourly bars of liquid cryptocurrencies.

An idea must be computable from one symbol's own history of open, high, low, close, volume,
quote_volume and trades, using only past bars. Positions range from 0 (out) to 1 (fully long).

Ideas are scored by timing Sharpe: the Sharpe ratio after 0.1% trading costs, minus what
holding a constant position of the same average size would have earned. Simply being long
scores nothing; only being in the market at better times than average scores.

You will be shown what has been tried so far. Propose ONE new idea, stated precisely enough to
implement: the indicators, the window lengths in hours, and the entry and exit rules. Do not
mention specific dates, events or price levels.

Reply with JSON only:
{"family": one of """ + json.dumps(FAMILIES) + """, "hypothesis": "<the idea>", "parent_id": <id of the attempt you are refining, or null>}"""


def extract_code(text):
    """The Python inside the reply's last fenced block, or the whole reply if it has no fence."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip() + "\n"


def write_code(hypothesis, llm=chat, previous_code=None, error=None):
    user = f"Idea: {hypothesis}"
    if previous_code:
        user += f"\n\nYour previous attempt:\n```python\n{previous_code}```\n\nIt was rejected: {error}\n\nWrite a corrected version."
    return extract_code(llm("coder", CODER, user, temperature=0.4))


def summary(run_id):
    """What the researcher is shown: counts, best, worst and latest attempts. Built by code, never by the model."""
    rows = store.query("SELECT id, family, hypothesis, status, error, timing_sharpe, trades_per_symbol "
                       "FROM attempts WHERE run_id = ? ORDER BY id", [run_id])
    if not rows:
        return "Nothing has been tried yet."

    def tally(key):
        counts = {}
        for r in rows:
            counts[r[key] or "none"] = counts.get(r[key] or "none", 0) + 1
        return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))

    def line(r):
        result = (f"timing Sharpe {r['timing_sharpe']:.2f}, {r['trades_per_symbol']:.0f} trades"
                  if r["timing_sharpe"] is not None else f"{r['status']}: {(r['error'] or '')[:100]}")
        return f"  #{r['id']} [{r['family'] or '-'}] {r['hypothesis'][:220]} -> {result}"

    scored = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: -r["timing_sharpe"])
    parts = [f"Attempts so far: {len(rows)}.", f"By family: {tally('family')}.", f"By outcome: {tally('status')}."]
    if scored:
        parts += ["Best:"] + [line(r) for r in scored[:5]]
        if scored[5:]:
            parts += ["Worst:"] + [line(r) for r in scored[5:][-5:]]
    parts += ["Most recent:"] + [line(r) for r in rows[-5:]]
    return "\n".join(parts)


def propose(run_id, llm=chat):
    """Ask the researcher for the next idea, alternating between exploring and refining."""
    n = store.count("run_id = ?", [run_id])
    tried = {r["family"]: r["n"] for r in store.query(
        "SELECT family, count(*) AS n FROM attempts WHERE run_id = ? GROUP BY family", [run_id])}
    if n % 2 == 0:
        ask = f"Propose an idea in the least-tried family: {min(FAMILIES, key=lambda f: tried.get(f, 0))}."
    else:
        ask = "Refine or vary one of the better attempts, or take a failed idea and fix what was wrong with it."
    user = summary(run_id) + "\n\n" + ask
    for last in (False, True):
        text = llm("researcher", RESEARCHER, user, schema=Hypothesis.model_json_schema())
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate a fence or stray words around the JSON
            return Hypothesis.model_validate_json(match.group(0) if match else text)
        except ValidationError:
            if last:
                raise
