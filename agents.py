"""The LLM side: one local model, two roles (hypothesis, coder), told apart only by prompt.

The model never sees a metric it could game: it gets discovery-period results from the
log and nothing else. Prompts must stay free of market hints, dates and "what works"
(PLAN.md, "Leakage channels"). Bump PROMPT_VERSION whenever a prompt changes.
"""
import json
import os
import re
import statistics
import time
import urllib.request
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

import store

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")  # llama-server in WSL; any OpenAI-compatible server works
MODEL = os.environ.get("LLM_MODEL", "gemma-4-12B-it-Q4_K_M")  # a label for the log; llama-server serves whatever it loaded
PROMPT_VERSION = "v6-pilot"
MAX_TOKENS = 1500  # a strategy is ~150 tokens; without a cap the model can ramble until the call times out
MEMORY = 200  # attempts before the researcher starts again with an empty memory. In night1 the ideas only got
              # worse after the first 200, as it kept refining its own refinements.
MAX_REFINES = 3  # refinements allowed in one line of ideas (an idea and everything built on it, directly or through
                 # another refinement). night1 built on its top score 81 times; pilot4 got round a per-attempt cap by chaining.
MAX_SAME_SKELETON = 3  # ideas allowed to share an exit condition and an entry condition, numbers removed: an idea and
                       # two variants that swap the other filter. pilot5's top ten were one breakout rule with a
                       # different second filter each time; a check on the whole wording could not see it, because
                       # the short plain-language ideas v5 asks for all share one sentence template.
MAX_SAME_RULE = 2  # ideas allowed to share one wording once the numbers are removed: an idea and one re-tuning of it.
                   # pilot4 proposed one breakout rule with volume multipliers of 1.3, 1.5 and 1.4.
FAMILIES = ["trend", "mean_reversion", "breakout", "volatility", "volume", "time_of_day", "other"]


class Hypothesis(BaseModel):
    family: Literal["trend", "mean_reversion", "breakout", "volatility", "volume", "time_of_day", "other"]
    hypothesis: str = Field(min_length=20, max_length=450)  # the server's grammar enforces this, cutting a longer idea off
    parent_id: int | None = None


def chat(role, system, user, schema=None, temperature=0.8):
    """One chat completion against the local server. Logs the call and returns the reply text."""
    body = {"model": MODEL, "temperature": temperature, "max_tokens": MAX_TOKENS,
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
  rolling(), expanding() and ewm() already stop at the current row, so do not add shift(1) to them.
- For a rule with separate entry and exit conditions, build a Series that is 1.0 where the entry
  condition is true, 0.0 where the exit condition is true and NaN elsewhere (numpy.where on the
  boolean conditions), then .ffill().fillna(0). Index with boolean masks, never with 0/1 integers.
- Import only numpy, pandas and math. No file access.
- No calendar years or dates anywhere, including comments. Hour of day and day of week
  taken from df.ts are fine.
- Vectorised pandas and numpy only. It must finish in a few seconds on 35,000 rows and is cut
  off after 20. Never call .apply() with a Python function on a rolling window or a column, and
  never loop over rows: use the built-in rolling methods (mean, std, sum, min, max, quantile,
  corr, skew, kurt). The environment has pandas 2.2 and numpy 1.26.
- Deterministic: no randomness.
- Every position change costs 0.1%, and a strategy that barely trades is discarded.

Reply in exactly two parts:
PLAN: at most 5 short lines saying how you will compute the positions. Do your thinking here.
Then one ```python code block holding the complete function and its imports. Inside the code
block write no comment lines and no docstring; it should rarely need more than 30 lines."""

RESEARCHER = """You are a quantitative researcher proposing ideas for systematic long/flat strategies
on hourly bars of liquid cryptocurrencies.

An idea must be computable from one symbol's own history of open, high, low, close, volume,
quote_volume and trades, using only past bars. Positions range from 0 (out) to 1 (fully long).

Ideas are scored by timing Sharpe: the Sharpe ratio after 0.1% trading costs, minus what
holding a constant position of the same average size would have earned. Simply being long
scores nothing; only being in the market at better times than average scores.

Every position change costs 0.1%, and the history is about 35,000 hourly bars. Aim for fewer than
300 position changes over it, which means holding for days, not hours. A condition on a single bar
(a 1-hour return, one bar's volume against its average) flips constantly, so never use one as the
exit, and use one for entry only together with a slower condition. Base the exit on a rolling
window of 24 hours or more.

You will be shown what has been tried so far. Propose ONE new idea, stated precisely enough to
implement: the indicators, the window lengths in hours, and the entry and exit rules. Do not
mention specific dates, events or price levels.

Keep it simple: at most two indicators, at most two entry conditions and one exit condition, in
under 60 words. Describe each indicator by how it is computed ("the 24-hour rolling mean of
volume"), never by an invented name. Adding one more filter to an idea that already has two is
not a new idea, and neither is an earlier idea with different window lengths or thresholds. Keeping
an earlier idea's exit and one of its entry conditions and swapping only the other condition is
not a new idea either.

Reply with JSON only:
{"family": one of """ + json.dumps(FAMILIES) + """, "hypothesis": "<the idea>", "parent_id": <id of the attempt you are refining, or null>}"""


def extract_code(text):
    """The Python inside the reply's last fenced block, or the whole reply if it has no fence.

    A reply cut off by the token cap has an opening fence and no closing one; that still counts as a block.
    """
    blocks = [b for b in re.findall(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, re.DOTALL) if b.strip()]
    return (blocks[-1] if blocks else text).strip() + "\n"


def write_code(hypothesis, llm=chat, previous_code=None, error=None):
    user = f"Idea: {hypothesis}"
    if previous_code:
        user += (f"\n\nYour previous attempt:\n```python\n{previous_code}```\n\nIt was rejected: {error}\n\n"
                 "Write a corrected version. If the code above stops in the middle, your reply was too long.")
    user += "\n\nRemember: a short PLAN first, then the code block with no comment lines in it."
    return extract_code(llm("coder", CODER, user, temperature=0.4))


def _rule(text):
    """An idea with its numbers and punctuation removed, so re-tunings of one rule compare equal."""
    return " ".join(re.findall(r"[a-z]+", (text or "").lower()))


def _refs(r):
    return {int(x) for x in re.findall(r"#(\d+)", r["hypothesis"] or "")} | {r["parent_id"]}


def _spent(rows):
    """Ids of attempts whose line of ideas has had MAX_REFINES refinements already.

    A line is an idea plus everything built on it, directly or through another refinement, so
    refining the latest refinement counts against the same line as refining the original.
    """
    root, known, idea = {}, {r["id"] for r in rows}, None

    def find(i):
        while root.setdefault(i, i) != i:
            i = root[i]
        return i

    for r in rows:
        if not r["tries"]:
            idea = r["id"]
        for other in ([idea] if r["tries"] else _refs(r)):  # a retry belongs to its idea
            if other in known and other < r["id"]:
                root[find(r["id"])] = find(other)
    size = Counter(find(r["id"]) for r in rows if not r["tries"])
    return {r["id"] for r in rows if size[find(r["id"])] > MAX_REFINES}


def _skeleton(text):
    """(entry conditions, exit condition) of an idea, each with numbers and punctuation removed.

    Ideas are written "Enter when A and B. Exit when C." Two ideas with the same C and a shared A or B
    are one rule with a filter swapped. Text that does not follow the pattern gets an empty exit and
    never matches anything.
    """
    entry, _, exit_ = (text or "").lower().partition(" exit ")
    entry = re.sub(r"^\W*(enter|go long|buy)\w*\s+(when|if)\s+", "", entry.strip())
    conditions = {_rule(c) for c in re.split(r"\band\b|,|;", entry)}
    return frozenset(c for c in conditions if len(c.split()) >= 3), _rule(re.sub(r"^(when|if)\s+", "", exit_.strip()).split(".")[0])


def _objection(h, rules, spent, skeletons=()):
    """Why code turns a proposal away before any of it is written or run, or None if it stands.

    rules: Counter of earlier ideas' wording without numbers. skeletons: earlier ideas' _skeleton values.
    """
    if rules[_rule(h.hypothesis)] >= MAX_SAME_RULE:
        return "it is an earlier idea with only the numbers changed. Change the mechanism: a different indicator, condition or exit."
    entry, exit_ = _skeleton(h.hypothesis)
    shared = [e & entry for e, x in skeletons if exit_ and x == exit_ and e & entry]
    if len(shared) >= MAX_SAME_SKELETON:
        return (f"{len(shared)} earlier ideas already have the same exit and the entry condition '{sorted(shared[-1])[0]}', "
                "with only the other condition swapped. Use a different main entry condition and a different exit.")
    used = sorted(i for i in _refs({"hypothesis": h.hypothesis, "parent_id": h.parent_id}) if i in spent)
    if used:
        return (f"it builds on #{used[0]}, and that line of ideas has already been refined {MAX_REFINES} times. "
                "Propose something that does not build on an earlier attempt.")


def _line(r, spent=()):
    result = (f"timing Sharpe {r['timing_sharpe']:.2f}, {r['trades_per_symbol']:.0f} trades"
              if r["timing_sharpe"] is not None else f"{r['status']}: {(r['error'] or '')[:100]}")
    return f"  #{r['id']}{' (used up)' if r['id'] in spent else ''} [{r['family'] or '-'}] {(r['hypothesis'] or '')[:220]} -> {result}"


def _round(run_id):
    """The attempts of the current round of MEMORY attempts, and the reviewer's lessons written during it."""
    rows = store.query("SELECT id, tries, parent_id, family, hypothesis, status, error, timing_sharpe, trades_per_symbol "
                       "FROM attempts WHERE run_id = ? ORDER BY id", [run_id])
    rows = rows[len(rows) // MEMORY * MEMORY:]
    return rows, [l for l in store.lessons(run_id, 2) if rows and l["after_attempt"] >= rows[0]["id"]]


def summary(run_id):
    """What the researcher is shown about the current round: counts, best, worst and latest attempts, and lessons.

    The numbers are assembled by code. Only the lessons are model-written, and they
    were written from these same discovery-period numbers.
    """
    rows, lessons = _round(run_id)
    if not rows:
        return "Nothing has been tried yet."

    def tally(key):
        counts = {}
        for r in rows:
            counts[r[key] or "none"] = counts.get(r[key] or "none", 0) + 1
        return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))

    spent = _spent(rows)
    scored = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: -r["timing_sharpe"])
    parts = [f"Attempts so far: {len(rows)}.", f"By family: {tally('family')}.", f"By outcome: {tally('status')}."]
    if scored:
        parts += ["Best:"] + [_line(r, spent) for r in scored[:5]]
        if scored[5:]:
            parts += ["Worst:"] + [_line(r, spent) for r in scored[5:][-5:]]

        def median(lo, hi):
            xs = [r["timing_sharpe"] for r in scored if lo <= r["trades_per_symbol"] < hi]
            return f"{statistics.median(xs):+.2f} over {len(xs)}" if xs else "none yet"
        parts.append(f"Median score by number of trades: under 300: {median(0, 300)}; 300 to 1000: {median(300, 1000)}; "
                     f"over 1000: {median(1000, float('inf'))}.")
    parts += ["Most recent:"] + [_line(r, spent) for r in rows[-5:]]
    if spent:
        parts.append(f'Attempts marked "used up" belong to a line of ideas already refined {MAX_REFINES} times. Do not build on them.')
    if lessons:
        parts += ["Lessons from earlier reviews of this search (oldest first):"] + [l["text"] for l in lessons]
    return "\n".join(parts)


def propose(run_id, llm=chat):
    """Ask the researcher for the next idea, alternating exploration with refining or following the reviewer.

    Code turns away a proposal that only re-tunes an earlier rule or extends a used-up line, and asks again.
    """
    turn = store.count("run_id = ? AND tries = 0", [run_id])
    tried = {r["family"]: r["n"] for r in store.query(
        "SELECT family, count(*) AS n FROM attempts WHERE run_id = ? GROUP BY family", [run_id])}
    rows, lessons = _round(run_id)
    # Half the turns explore. With one turn in three, the first night run put 21 of 30 ideas into
    # variations of its single best trend strategy, which is hill-climbing on noise.
    if turn % 2 == 0:
        family = sorted(FAMILIES, key=lambda f: tried.get(f, 0))[(turn // 2) % 2]  # least-tried, then second least-tried
        seen = store.query("SELECT id, family, hypothesis, status, error, timing_sharpe, trades_per_symbol FROM attempts "
                           "WHERE run_id = ? AND family = ? AND tries = 0 ORDER BY id DESC LIMIT 12", [run_id, family])
        ask = f"Propose an idea in the least-tried family: {family}."
        if seen:  # shown with their results, so it can see what heavy trading did to this family's earlier ideas
            ask += (" It must differ in mechanism, not only in window lengths, from these already tried:\n"
                    + "\n".join(_line(r) for r in seen))
    elif turn % 4 == 1 or not lessons:
        ask = ("Take one of the better attempts, or a failed idea, and change its mechanism: a different indicator, "
               "condition or exit. Changing only window lengths or thresholds is not accepted.")
    else:
        ask = "Follow one of the next steps suggested in the most recent lessons, stated as a precise new idea."
    earlier = [r["hypothesis"] for r in store.query("SELECT hypothesis FROM attempts WHERE run_id = ? AND tries = 0", [run_id])]
    rules, skeletons = Counter(map(_rule, earlier)), [_skeleton(t) for t in earlier]
    spent, user, h, reason = _spent(rows), summary(run_id) + "\n\n" + ask, None, None
    for _ in range(3):
        text = llm("researcher", RESEARCHER, user + (f"\n\nYour last proposal was not accepted: {reason}" if reason else ""),
                   schema=Hypothesis.model_json_schema())
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate a fence or stray words around the JSON
            h = Hypothesis.model_validate_json(match.group(0) if match else text)
        except ValidationError as e:
            error, reason = e, "it was not valid JSON in the required form."
            continue
        reason = _objection(h, rules, spent, skeletons)
        if not reason:
            return h
    if h is None:
        raise error
    return h  # ponytail: after three asks the last valid proposal is taken as it is; it still counts as a trial


class Review(BaseModel):
    worked: list[str]
    failed: list[str]
    next_steps: list[str]


REVIEWER = """You are reviewing a batch of results from an automated search for long/flat strategies on
hourly cryptocurrency bars. A later researcher will read your notes instead of the raw results.

Scores are timing Sharpe on one fixed four-year history: the Sharpe ratio after 0.1% trading costs,
minus what a constant position of the same average size would have earned. Around zero means no
skill. Anything between -0.5 and +0.5 cannot be told apart from luck, so do not call a small positive
number a success. Many strategies were tried, so even the best score is partly luck.

Write three short lists, at most 5 items each, one sentence per item, referring to attempts by #id:
- worked: what the ideas scoring clearly above +0.5 have in common. If none did, say so.
- failed: what scored clearly below zero, was rejected or crashed, and the likely reason. Use the
  error text. Note when heavy trading (many trades) is what sank an idea.
- next_steps: specific directions not yet tried, or a concrete fix for a promising failure. Never
  repeat a next step you gave before, and never suggest adding another filter to an idea that
  later attempts already refined without improving it. A next step must change the mechanism, not
  only a window length or threshold. Simpler ideas that trade less often are preferred.

Do not mention dates, events or price levels.
Reply with JSON only: {"worked": [...], "failed": [...], "next_steps": [...]}"""


def review(run_id, after_id, llm=chat):
    """Have the model turn the attempts after `after_id` into lessons. Returns the text to store."""
    rows = store.query("SELECT id, family, hypothesis, status, error, timing_sharpe, trades_per_symbol FROM attempts "
                       "WHERE run_id = ? AND id > ? ORDER BY id DESC LIMIT 40", [run_id, after_id])[::-1]
    earlier = store.lessons(run_id, 1)
    user = "Results in this batch:\n" + "\n".join(_line(r) for r in rows)
    if earlier:
        user += ("\n\nNext steps you gave after the previous batch. Do not repeat any of them; if this batch shows one "
                 "was tried, say in worked or failed what came of it:\n" + earlier[0]["text"].split("Next:", 1)[-1].strip())
    for last in (False, True):
        text = llm("reviewer", REVIEWER, user, schema=Review.model_json_schema(), temperature=0.3)
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            r = Review.model_validate_json(match.group(0) if match else text)
            return "\n".join(f"  {label}: " + (" | ".join(s.strip() for s in items[:5]) or "nothing")
                             for label, items in (("Worked", r.worked), ("Failed", r.failed), ("Next", r.next_steps)))
        except ValidationError:
            if last:
                raise
