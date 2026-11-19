# Augurix plan

Local LLM agents propose trading strategies and write them as code. Plain Python judges them. The deliverable is an honest funnel: how many strategies were generated and how many survived each check, ending with a holdout the model could not have seen. Zero survivors is an acceptable result.

Written 2026-11-16 by Claude Fable 5.2 as a handoff. It records decisions and the reasoning behind them so a later session, with any model, can continue without re-deriving them. If you change a decision, change it here and say why.

## Status

Update this section at the end of every working session.

- [ ] Week 1 (Nov 16–22): data, splits, backtester, 3 hand-written strategies
- [ ] Week 2 (Nov 23–29): model served, coder agent, sandbox
- [x] Week 3 (Nov 30–Dec 6): look-ahead detector and planted-cheat suite (done 2026-11-16; Adam still has to learn it)
- [ ] Week 4 (Dec 7–13): hypothesis agent, log, full loop, overnight run
- [ ] Week 5 (Dec 14–20): validation, deflated Sharpe, open the holdout once
- [ ] Week 6 (Dec 21–27): README, write-up, bake-off table

Week 1 is functionally complete as of 2026-11-16: `download_data.py`, `make_splits.py`, `backtest.py`, `baselines.py`, 12 tests passing. Data downloaded (see below). Discovery split written: 35,006 bars per symbol in `data/splits/discovery/`. One thing remains before ticking the box: Adam works the toy example by hand and reads `backtest.py` until he can explain every line.

Also built on 2026-11-16, ahead of schedule (96 tests passing, 9 of them through the real Docker sandbox):

- Week 3: `lookahead.py` catches all 14 planted cheats and passes all 12 honest strategies, on synthetic data and on the real discovery data. Wired into the sandbox.
- Week 5 math: `dsr.py` (probabilistic and deflated Sharpe), checked against the answer key and a simulation.

- Week 2 plumbing: static check, sandbox image and runner, harness, attempt log, judging pipeline, LLM client, coder agent, 20 ideas in `ideas.txt`.
- Week 4 plumbing: researcher (hypothesis) agent, log summary, LangGraph loop with retry, resumable `generate.py`.
- Live dashboard (`dashboard.py`, `dashboard.html`).

Verified: the sandbox gives the same result as the host for the trend baseline (timing Sharpe 0.6926, 153 trades) in about 1.6 seconds per attempt; the container blocks network, file writes and runaway loops; the graph feeds an error back to the coder and stops after 3 tries.

**Pilot runs so far (2026-11-16), none of which count toward `N` or may be validated:**

- `pilot`: the 20 hand-written ideas. The first pass stalled at idea 8 when one coder reply ran to 16,000 tokens; replies are now capped at 1,500. Finished on the second pass: 28 attempts, all 20 ideas reached a result. This is Week 2's "done".
- `night1`: the first unattended researcher run, 07:25 to 11:54 EST, 42 attempts under `v2-pilot` and 727 under `v3-pilot`. Report: `runs/night1_report.md`. 769 attempts on 487 ideas, 400 backtested, no infrastructure failures, 171 attempts an hour.
  - **Result: nothing distinguishable from luck.** Best discovery timing Sharpe +1.45; the best of 400 no-skill strategies over four years is expected to reach about +1.49. 39 of 400 beat the hand-written 20-day trend baseline (0.69) in-sample. The top one (#410) is in the market 0.7% of the time on 43 trades, and its code does not compute what its description says.
  - **The search got worse as it ran.** Attempts 1–200: 35 scores above +0.5, median about 0. Attempts 400–769: 6 above +0.5, median about −2. `no_trades` rose from 9% to about 25–36% of attempts.
  - **Why:** the researcher stacks jargon. 407 of 487 ideas contain "Volume-Weighted", 195 start with "Refine", median idea length is 700 characters, and entries end up as four conditions ANDed together that never fire. The reviewer's notes repeat themselves: "refine the breakout combinations from #410" appears in 15 consecutive reviews. Breakout got 293 of 769 attempts because every non-explore turn went there.
  - **Fees decide most scores.** 190 of 400 backtested strategies made over 1,000 trades per symbol; their median is −2.7. Under 1,000 trades the median is about 0.
  - **Timeouts:** all 40 used `rolling().apply()` with a Python function. At 60 seconds each they cost about 40 minutes of the run.
  - **Fixes made the same day as prompts `v4-pilot`** (`agents.py`, `pipeline.py`; each has a test):
    - Ideas are capped at 450 characters by the JSON schema, which llama.cpp's grammar enforces by cutting the string off. The researcher prompt asks for at most two indicators, two entry conditions and one exit, under 60 words, described by how they are computed and not by an invented name.
    - The researcher prompt states the fee arithmetic, and the summary it reads now has a code-computed line: median score for under 300, 300 to 1,000 and over 1,000 trades.
    - The coder prompt bans `.apply()` with a Python function and row loops. Discovery timeout is 20 seconds (`pipeline.TIMEOUT`), and a timeout goes back to the coder with that hint. `sandbox.run` keeps its 60-second default, which validation uses on 1.5 times the bars. Cost: 20 of night1's 400 backtested strategies took over 20 seconds and would now be sent back.
    - The reviewer sees only its previous "Next" list, with an instruction not to repeat it. Before, it was given all its earlier notes and copied them forward.
    - An attempt that three ideas have already built on (`agents.MAX_REFINES`, counted from `parent_id` and from `#id` mentions) is named to the researcher as used up.
    - Memory resets every 200 attempts (`agents.MEMORY`): the summary and lessons cover only the current round. Family counts and the duplicate check still cover the whole run. This is one run id, so an official run still has one `N`.
  - The report's "luck" line was wrong on the first write (+8.75): it used the spread of the scores, which is mostly fee losses. `generate.py` now uses the sampling variance of an annual Sharpe, `1 / years`. The same caution applies to Option A below, whose bar comes from the variance across validated candidates.

Prompt history: `v0` produced comment monologues of 1,200 tokens on average. `v1` ("no comments") was ignored, because with thinking off the model reasons in comments. `v2` gives it a short PLAN section before the code block, and most replies are now 17 to 26 lines with no comments. `v3` changes only the researcher's turn order. In the first ten minutes of `night1` (42 attempts under `v2`, log in `runs/night1_part1.log`), one turn in three explored and 21 of 30 ideas were variations of the single best trend strategy. Under `v3` every other turn explores the least-tried or second least-tried family. `night1` was restarted at 07:37 EST with `v3`, so it mixes both; the `prompt_version` column says which attempt used which. `v4` and `v5` are described under the pilot runs above.

**Model server (decided 2026-11-16): llama.cpp, not Ollama.** Ollama is not installed. `llama-server` was already built in WSL for the Shadow-Fleet project, with `gemma-4-12B-it-Q4_K_M.gguf` (7.7 GB) and `Qwen3-14B-Q4_K_M.gguf` in `~/models`. Start it with:

```
wsl -d Ubuntu-24.04 -- bash -lc "~/llama.cpp/build/bin/llama-server -m ~/models/gemma-4-12B-it-Q4_K_M.gguf -ngl 99 -c 16384 -np 1 --reasoning-budget 0 --host 127.0.0.1 --port 8080"
```

- `-c 16384 -np 1`: one 16k context. Without `-np 1` the server may split the context across several slots.
- `--reasoning-budget 0`: turns the model's thinking off. This build has no `--reasoning off` flag, whatever Shadow-Fleet's notes say.
- With it loaded the GPU shows 10.3 of 12.2 GB used, about 2 GB of that being the desktop.
- One GPU tenant at a time: stop it before running Shadow-Fleet's server, which uses the same port.
- The quant is Q4_K_M, not the Q5 or Q6 first planned. It is what is on disk; record it in the write-up.

**First contact with the real model (one call per agent, 2026-11-16):** the coder returned a valid strategy that passed the static check (246 tokens in 22 seconds, the first call after loading); the researcher returned schema-valid JSON in 2.5 seconds at about 45 tokens a second. The coder lagged its moving average by one extra bar that the idea did not ask for. **Still not verified:** the 20-idea run, retry behaviour with a real model, and anything about throughput beyond those two calls.

**Judge audit (2026-11-17). Findings 1, 2 and 4 were fixed the same day, each with a test; Adam delegated 3 and 5, settled below.** `backtest.py`, `lookahead.py`, `dsr.py`, `run_validation.py`, `open_holdout.py`, `harness.py` and `pipeline.py` were read against the specs and probed on synthetic data. The money math, the PSR and expected-maximum formulas, the scoring-window slice, the Bonferroni threshold and the one-shot locks are correct. Findings, worst first:

1. **The look-ahead test misses a leak that is rarely active and reaches only a bar or two ahead.** A prefix differs from the full run only in the last few bars before a cut, so with 20 cuts a rule that looks one bar ahead on 0.4% of bars (timing Sharpe 5.6, 282 trades, so it passes every other gate) was caught in 2 of 20 seeds. The same cheat behind three ANDed filters, active 2.5% of the time, was caught in 7 of 20. Fix: add cuts placed right after a sample of the bars where the full run's position changes. A leak that causes a trade is then tested at that trade. In the experiment this caught all three cheats and passed an honest rule. It roughly doubles the detector's run time. **Fixed:** `lookahead.change_cuts` adds up to 20 such cuts, none shorter than 2% of the data so honest code is not failed for crashing on a handful of rows. Reason for changing a judge file: a false survivor is the one result that would invalidate the project. Test: two rare one-bar cheats are caught for every seed, and the old cuts alone miss them in at least half the seeds.
2. **State kept between calls defeats the test.** A strategy that stores the first data it is given in a module-level list passes the static check and the detector with a timing Sharpe of 77, because the harness calls it on the full data first. The model is unlikely to write this by accident. **Fixed:** `harness.py` loads a fresh copy of the strategy module for every call, which also resets default arguments. State stored on numpy or pandas themselves would survive; running each call in its own process is the upgrade if that ever appears. Test (needs Docker): a strategy that remembers its first input is rejected as look-ahead.
3. **Strategies can see absolute time and price levels.** `ts` holds real dates, and the year-literal ban does not stop an epoch number or "the third year of the data". A rule keyed to the model's memory could pass discovery and validation, both of which are before its cutoff. The holdout is after the cutoff, so the headline result is still protected. Options: accept and document, or shift `ts` by a whole number of weeks in the harness (keeps hour and weekday). Adam decides. **Settled (accepted, 2026-11-17):** not fixed. The cost is at most a wasted shortlist slot on a rule keyed to remembered dates, which then fails on the holdout, because the holdout is after the model's cutoff. Shifting `ts` would hide dates but not price levels, so it buys little. State this limit in the write-up.
4. **`run_validation.first_bar` returns 0 when no bar is at or after the date**, which would score the warm-up bars too. It needs a refusal. Related: the holdout run gets its history only because `download_data.py --holdout` defaults to `--start 2019-01`; with a later `--start` the warm-up silently shrinks to the embargo months. **Fixed:** `first_bar` refuses when nothing is at or after the date, and `open_holdout.history_problems` refuses holdout data whose first bar is not the first bar of `data/klines`.
5. **The "best Sharpe expected from luck" in the pre-registration uses the variance across the validated candidates**, not across all N trials and not the sampling variance this file's own table in "Decision for Adam" was computed with. Under option B it is only a reported number; under option A it sets the gate. Using `1 / n_obs` would match the table and the run report. **Fixed (2026-11-17):** `pick_shortlist` keeps the spread across candidates, as Bailey and Lopez de Prado prescribe, but never lets it fall below the no-skill variance `1 / bars`. With only a few candidates the spread can come out near zero, and so would the luck bar. Test: two identical candidates fall back to `1 / bars`.

Assumptions that are choices, not bugs, and that Adam should be able to state: a position is entered at the same close that produced its signal; the last position never pays an exit fee; a gap in the data is treated as one bar; the equal-weight portfolio is rebalanced every bar at no cost.

Next, in order:

1. `pilot4` (prompts `v4-pilot`) was stopped by Adam at 139 of 250 attempts on 2026-11-16. Resume with `python generate.py --run pilot4 --n 250`; it continues from attempt 140 and writes `runs/pilot4_report.md` when it ends. Seen at 50 attempts, against the same stretch of `night1`: 304 attempts an hour (214), 0 crashes (6), 0 timeouts, median idea 220 characters (494). Still wrong: 18 of 38 backtested strategies made over 1,000 trades, so the fee line has not changed what gets proposed; refine turns have become parameter sweeps (#849, #851, #853 are one breakout rule with volume multipliers 1.3, 1.5, 1.4), which the three-per-parent cap does not stop because each refines the one before; the reviewer still repeated one of its next steps. The memory reset at 200 attempts has not been exercised yet. Both were addressed the same day as prompts `v5-pilot` (below); neither fix has run against the model yet.
   - **Sweeps and chains (enforced in code, `agents.propose`):** a proposal is turned away and the researcher asked again, up to three times, when (a) its wording with the numbers removed matches two earlier ideas of the run (`MAX_SAME_RULE`), or (b) it builds on an attempt whose line of ideas has already had three refinements (`MAX_REFINES`). A line is an idea plus everything built on it directly or through another refinement, so chaining no longer escapes the cap. After three asks the last valid proposal is taken. Replayed over the logged ideas, the checks would have turned away 60 of `pilot4`'s 123 (54 as re-tunings) and 141 of `night1`'s 487.
   - **Overtrading (prompt only, so unverified):** the researcher is told to aim for under 300 position changes, never to use a single-bar condition as the exit, and to base exits on a window of 24 hours or more. Explore turns now show each earlier idea of the family with its score and trade count. If a pilot still shows about half the backtests over 1,000 trades, the next lever is a required "typical holding hours" field in the proposal, checked by code.
   - **`pilot5` (prompts `v5-pilot`, 250 attempts) was started on 2026-11-17.** It writes `runs/pilot5_report.md` when it ends and then stops the model server by itself. If it was interrupted, `python generate.py --run pilot5 --n 250` resumes it. **How to read the report, without needing an assistant:**
     - "Trading fees" section: the share of backtested strategies over 1,000 trades. `pilot4` had 46 of 89 (52%), median score -3.21 against +0.51 for the rest. If `pilot5` is under about 30%, the overtrading prompt worked. If it is still near half, add the required "typical holding hours" field described above before anything else.
     - Last line, "Proposals the code sent back": how often the two new checks fired. Zero on a 250-attempt run would mean they are not wired in; `pilot4`'s ideas would have triggered about 60.
     - "By family": no family should hold more than about a third of the attempts.
     - "Outcomes": `no_trades` and `timeout` should each stay under about 10% of attempts.
     - **Seen at 50 attempts (2026-11-17), against the same stretch of `pilot4`:** 0 of 43 backtests over 1,000 trades (was 18 of 38), median 117 trades (was 616), 1 timeout-free run at 241 attempts an hour, and the code sent back 8 proposals (1 re-tuning, 7 on used-up lines). The overtrading fix works. **New problem:** 22 of 47 ideas are filed under trend, and ideas filed under other families are the same base rule ("168-hour average above the 336-hour average") with one different filter added. 20 of 43 backtests score above +0.5, but they are near-copies of one in-sample trend effect, not 20 pieces of evidence. The two checks do not catch this, because the wording differs and the ideas do not cite an earlier attempt. Next lever, not yet built: turn away a proposal whose words overlap heavily (say 70% or more) with several earlier ideas of the run, and count the family label only if the rule does not contain another family's base condition. Check the finished report's "By family" table and its top ten before freezing: if the top ten are one rule with different filters, fix this first.
     - **Finished at 221 of 250 attempts (2026-11-17).** The model server went down with the assistant's session and the run stopped itself after 10 failed calls; the last five attempts (#1161 to #1165) are Windows failing to start the docker client, logged as crashes before `sandbox.py` treated that as an infrastructure failure (fixed the same day, with a test). Against the four checks: **fees pass** (15 of 147 backtests over 1,000 trades, 10%, was 52%); **the checks fire** (62 proposals sent back as re-tunings, 21 for used-up lines); **variety fails** (breakout 83 of 221 attempts, 38%, and 27 duplicates, attempts whose positions were identical to an earlier one); **no_trades fails** (35, 16%). The top ten are all one rule: close above the 672- or 720-hour high, exit below the 168-hour mean, with a different filter each time. The best, +1.40, sits at the luck benchmark of +1.33 for 147 strategies, and since the ten are near-copies the run tested far fewer distinct ideas than 147. **Verdict: not ready to freeze.** Build the word-overlap check described in the line above, rerun a pilot, then freeze.
   - **Prompts `v6-pilot` (2026-11-17): the skeleton check.** Word overlap of the whole idea was tried first and rejected: `v5`'s short plain-language ideas share one sentence template, so even an 80% threshold would have turned away half of `pilot5`. Instead `agents._skeleton` reduces an idea to its entry conditions and its exit condition, numbers removed, and `propose` turns a proposal away when three earlier ideas of the run (`MAX_SAME_SKELETON`) share its exit and one of its entry conditions. Replayed over the log it turns away 54% of `pilot5`'s ideas, all ten of its top ten, and 0 of `night1`'s. Ideas that do not follow the "Enter when … Exit when …" pattern are never matched by it.
   - **`pilot6` (`v6-pilot`, 6 hours) was started by `overnight.cmd pilot6 6` on 2026-11-17.** `overnight.cmd` starts the model server, runs `generate.py` for the given hours, then stops the server. Launched through Windows (WMI), it does not stop when the assistant's session ends, which is what killed `pilot5`. Read `runs/pilot6_report.md` with the four checks above, plus: the top ten should no longer be one rule, and the last line should show refusals "for keeping an earlier rule's exit and entry". Six hours also tests the memory reset every 200 attempts, which no pilot has reached yet.
   - **Seen at 128 attempts of `pilot6`:** the skeleton check fired 66 times and the re-tuning check 77, yet 56 of 113 ideas share one exit ("the short rolling mean of close falls below the long one", a moving-average crossover). The cause is the escape hatch in `agents.propose`: after three refusals the last valid proposal is accepted so a run cannot stall, and Gemma keeps returning to the crossover. **Fixed the same day (prompts `v7-pilot`):** after three refusals the turn becomes an exploration of the least-tried family other than the one just proposed, with two more asks; only then is the last valid proposal taken. Islands (one memory per family) remain the stronger fix if a pilot still converges.
   - **Holding styles (`v7-pilot`, 2026-11-17, at Adam's request):** `generate.py --style long|day|both` (and `overnight.cmd NAME HOURS STYLE`). `long` is the previous behaviour and the default. `day` asks for positions opened and closed within 24 hours, flat otherwise; the coder prompt now shows how to hold for at most N hours (`entry.rolling(N, min_periods=1).max()`). `both` alternates the two styles two ideas at a time, so each style gets explore and refine turns. The style of each attempt is logged as a suffix on `prompt_version` (`v7-pilot/day`). The style is a request to the model, not a check: the dashboard measures what each strategy actually did.
   - **Expect day trading to lose to fees.** A round trip costs 0.2%, and one per day is about 1,460 a year per symbol. In `pilot5` the 15 backtested strategies that held under a day had median timing Sharpe -1.38 and best +0.24, and over all pilots strategies with over 1,000 trades have a median near -2.7. A day-trading search that finds nothing is a valid result for the funnel, not a bug.
   - **Decide before the freeze:** the official run is either one style or `both`. Two separate official runs would both count toward the trial count, so `N` would be their total.
   - **Islands (`v8-pilot`, 2026-11-17, at Adam's request):** in a `both` run each holding style is searched as its own island. `agents.scope(run, island)` filters every query the researcher and reviewer make to that style's attempts (the `prompt_version` suffix), so each island has its own summary, explore and refine turns, refinement lines, duplicate checks and reviewer notes (stored under `run/long` and `run/day`; `store.lessons(run)` returns both for the report and dashboard). Runs with one style behave as before. Test: `test_in_a_both_run_each_style_is_an_island`.
   - **`pilot6` was stopped at about 220 attempts on 2026-11-17** to free the GPU: it was running `v6`, whose escape hatch was already found and fixed, so its last five hours would have re-measured a known problem. Its report was written with `--report`.
   - **`pilot7` (`v8-pilot`, `both`, 6 hours) was started by `overnight.cmd pilot7 6 both` on 2026-11-17.** It is the first run of the explore fallback, the day style and the islands. Read `runs/pilot7_report.md` with the checks above, then in the dashboard compare it with `pilot5` and filter by holding style: expect the day-trade island to score worse (fees), and neither island's top ten to be one rule.
   - **Never-traded attempts (`v9-pilot`, 2026-11-17):** 64 of the 104 `no_trades` attempts in `pilot5` to `pilot7` (62%) compared the close with a rolling maximum that included the current bar (`close > high.rolling(N).max()`), which can never be true. The coder prompt told it never to add `shift(1)` to a rolling window; that rule now makes an exception for breakouts over the previous N bars, and the `no_trades` message sent back to the coder names this cause and its fix. Before, retries resubmitted identical code. Sandbox test: `test_a_breakout_on_a_window_that_includes_its_own_bar_is_sent_back_with_the_fix`. Not yet run against the model: `pilot7` started on `v8`.
   - **`pilot7` at 123 attempts (0.6 h):** the islands work. Both styles spread over all seven families, and the day island holds for hours (median 14 h, 24 of 30 backtests under a day). Day trades lose to fees as expected: median timing Sharpe -1.01 against +0.25 for long holds, 19 of 30 over 1,000 trades. Convergence is lower but not gone: long has 42 ideas in 10 rules (largest group 13), day 40 in 13 (largest 9); the code sent 90 proposals back. Two failures, both fixed in `v9` the same day: 17 of the long island's 20 never-traded attempts are the rolling-maximum bug above, and 11 bad outputs in the day island came from the `v8` hint for holding N hours, which was wrong (it held while the condition stayed true, plus N) and set the model arguing with itself in comments until the length cap cut the code off. The hint now marks only the first bar of each entry; a test checks the idiom. `v9` was corrected in place because no run had used it.
     - If all four look right, the prompts are ready to freeze: decide option A or B, commit, `git tag gen-v1`, then `python generate.py --run gen-v1 --n 1000 --official --style both`.
   - **`pilot7` finished (`v8-pilot`, 6.0 hours, 1,158 attempts, 0 errors):** do not freeze on it. 511 attempts (44%) never traded, and 430 of those are the unshifted rolling-maximum bug that `v9` fixes, so `v8`'s output is mostly wasted retries. Long island: 537 attempts, 330 ideas, 192 backtested, median +0.47, best +1.31, median hold 118 h. Day island: 621 attempts, 330 ideas, 181 backtested, median -0.47, best +1.11, median hold 16 h, 69 over 1,000 trades, 62 bad outputs (the `v8` hold-N hint). Neither best beats the luck benchmark for 373 backtests (+1.48). Families stay spread in both islands, but convergence is back at the level of positions: 131 of the long island's 330 ideas fall in one near-copy group (a 24 h over 168 h mean with a volatility filter, first at #1437 and repeated to the end) although the skeleton check sent 930 proposals back. The day island is less concentrated (90 rules, largest 47). `pilot8` runs `v9` with the same settings to measure the two fixes; if `no_trades` falls well below 44% and bad outputs below 5%, the remaining question before freezing is whether near-copy groups need a cap of their own.
   - **`pilot8` (`v9-pilot`, `both`) was stopped after 1.2 hours and 198 attempts on 2026-11-17, once both `v9` fixes had shown they work:** never-traded fell from 44% to 10% (20 of 198) and bad outputs to 4.5% (9). Convergence did not improve: the day island had already put 48 of its 81 ideas in one near-copy group. Its report was written with `--report`.
   - **Why the near-copy checks did not hold (`v10-pilot`, 2026-11-17):** the checks caught the copies, but `propose` took the last proposal after five refusals. Replaying the checks over `pilot7`'s accepted ideas, 168 of the long island's 330 would have been refused, 116 of them in its largest group. A turn whose five proposals are all refused is now skipped: nothing is run or logged, so it is not a trial, and ten skipped turns in a row stop the run. Tests: `test_an_idea_that_only_changes_the_numbers_is_turned_away` and `test_a_skipped_turn_logs_nothing_and_endless_skipping_stops_the_run`. Cost: in `pilot7`'s long island about half the turns would have been skipped, so fewer ideas an hour.
   - **`pilot9` (`v10-pilot`, `both`, 6 hours) was started by `overnight.cmd pilot9 6 both` on 2026-11-17.** Check the largest near-copy group per island (target: a handful, not dozens), the number of skipped turns (`grep -c "turn skipped" runs/pilot9.log`) and ideas per hour against `pilot8`.
   - **`pilot9` ended after 1.5 hours and 227 attempts** when its model server went away: no out-of-memory error and no WSL restart, but someone came back to the machine 25 minutes before, so its minimized window was most likely closed. `overnight.cmd` now starts the server with no window. 227 attempts were enough for the question: **`v10` fixed convergence.** The largest near-copy group is 16 of 86 ideas for long holds and 9 of 84 for day trades, against 131 of 330 and 47 of 330 in `pilot7` and 48 of 81 for day trades in `pilot8`. Distinct rules: 27 and 43. 59 turns were skipped, and ideas came at about 116 an hour against 140 in `pilot8`, an acceptable cost. Never-traded attempts stay low (29, 13%) and bad outputs at 5% (12). The look-ahead test caught 11 strategies, 9 of them day trades. Long holds: 77 backtested, median +0.19, best +1.22. Day trades: 69 backtested, median -0.34, best +0.70. Nothing reaches the luck benchmark for 146 backtests (+1.33). **The prompts are ready to freeze** once Adam settles option A or B (findings 3 and 5 of the judge audit were already settled).
2. Adam reads `backtest.py`, `lookahead.py` and `dsr.py` against their specs until he can explain them.
3. **Done (2026-11-17):** Option B; the official run is `both` (one run, both islands, every attempt in either counts toward one N). The prompts (`v10`) are frozen and tagged `gen-v1`. Nothing in `agents.py`, `generate.py`, `pipeline.py` or the judge files changes from here; a change means a new tag and a new official run, and the write-up says so. Start the official run with `overnight.cmd gen-v1 12 both --n 1000 --official` (12 hours is only a cap; a run stopped short resumes with the same command).
   - **The official run `gen-v1` was started on 2026-11-17** with that command, from the tagged commit. Its first attempt is #2991 (`v10/long`, not a pilot). When it has 1,000 attempts, read `runs/gen-v1_report.md`; then Week 5 starts, in the order and with the locks under "Week 5".
   - **`gen-v1` stopped itself at 291 attempts after 1.95 hours (2026-11-17), not at 1,000.** The `v10` safety stop fired: ten turns in a row whose five proposals were all turned away. All ten were long-hold turns: the researcher kept proposing the 168-hour breakout with a volume filter (the family of the top scores), and the skeleton check refused each one. A skipped turn logs nothing, so `style_for` handed the turn to the long island again and the day island, still productive, never got it. 49 turns were skipped in all; no errors. The stop is a rule fixed before the run, not a choice made from scores, so the 291 attempts stand as official attempts. Funnel: 291 generated, 273 passed the static check, 259 ran, 257 passed the look-ahead test, 232 backtested, 108 above zero. Long holds: 142 attempts, 127 ideas, 46 rules (largest group 24), 115 backtested, median +0.19, best +1.31 (#3159). Day trades: 149 attempts, 126 ideas, 59 rules (largest 9), 117 backtested, median -0.37, best +1.14. Luck benchmark for 232 backtests: +1.41. Nothing beats it. Week 5 has not started. **Adam decides:** resume `gen-v1` unchanged, accept it at 291, or fix the stall and run `gen-v2` (with `gen-v1`'s attempts still counted in N).
   - **`v11` (2026-11-17, Adam: "add safeguards that allow the model to terminate the idea and move onto the next" instead of stopping the run).** Three changes. (1) A style whose turn is skipped rests for 10 ideas while the other style takes the turns (`generate.build`, `REST`); the run stops only when both are stuck for ten turns in a row. Test: `test_a_stuck_style_rests_while_the_other_one_works`. (2) The researcher's summary marks a rule "closed" once it has `MAX_SAME_SKELETON` variants, and says any further variant is turned away (`agents._closed`). Test: `test_wording_does_not_hide_a_near_copy_and_full_rules_are_marked_closed`. (3) `_rule` drops filler words ("the", "current", "rolling", "hour's") and maps synonyms (price to close, average to mean), and "Enter long when" is parsed like "Enter when": #3040 had slipped past the check by writing "the current close". Replayed over `gen-v1`, `v11` would have refused 33 of the long island's 127 accepted ideas and 25 of the day island's 126; regrouped, the long island's largest rule group is 49 (was 24) and the day island has 40 rules (was 59), largest 34 (was 9). **So `gen-v1` was more repetitive than `v10`'s grouping reported; the write-up must use the `v11` grouping.** N is counted per run id (`run_validation.py`), so the plan is to continue the run as `gen-v1` on `v11` under a new tag `gen-v1.1`, not to start a `gen-v2`: every official attempt then counts toward one N, and each attempt's `prompt_version` records which prompts produced it. Before that, `pilot10` starts from a pilot copy of `gen-v1`'s 291 attempts and runs `v11` for 90 minutes, to check that it no longer stalls in exactly that state.
   - **`pilot10` result (2026-11-17):** the safeguards worked and the model still ran dry. Starting from `gen-v1`'s 291 attempts, `v11` added 90 attempts on 67 ideas in 38 minutes, then stopped with both styles stuck (10 skipped turns in a row, alternating long and day; 44 skipped turns in all, 0 errors). Long holds came back after resting: 20 new ideas, 6 of them new rules. Day trades: 47 ideas, 16 new rules. Never-traded rose to 21 of 90 (23%, against 3% in `gen-v1`): pushed off its favourite rules, the model writes rarer conditions that never fire. Reading: under the novelty checks, this 12B model has roughly 320 distinct-enough ideas across both styles, and `gen-v1` had used most of them. Continuing `gen-v1` on `v11` would add about 90 attempts. **This is the finding for the write-up, and the reason to accept `gen-v1` at 291 rather than chase 1,000.**
   - **`v12` (2026-11-17, Adam: seed the researcher with strategy families from economic papers and traders' reports).** `seeds.txt` lists 26 families (13 long, 11 day, 2 both), each with its mechanism and sources: time-series momentum, moving-average rules, channel breakouts, nearness to the high, volatility-managed exposure, volume-conditioned momentum, attention, lottery spikes, realised skewness, illiquidity shocks, trend strength, on-balance volume, squeezes, volatility trailing stops, intraday momentum, trading sessions, day of week, short-term reversal, the volume-return relation, overreaction, pullbacks in an uptrend, opening-range and volatility breakouts, volatility clustering, VWAP reversion. Rules fixed with the list: every source was published before 2023-01-01, so none can carry information about validation or holdout years; every mechanism is computable from one symbol's own bars; cross-sectional findings are marked as adapted; families come from the literature, not from `gen-v1`'s scores, and several whose sources found nothing after costs stay in. Each explore turn (half the turns) now offers one seed at random for its holding style, in place of "the least-tried family"; refine and reviewer turns are unchanged, and so is every near-copy check. Random, not least-used, so a seed the model cannot get past the checks does not come back every turn. After three refusals the turn moves to a different seed. The seed id is stored in a new `seed` column. Tests: `test_explore_turns_offer_a_seed_and_show_what_it_already_scored`, `test_after_three_refusals_the_turn_explores_another_seed`, `test_seeds_are_complete_and_cover_both_styles`. What this changes in the claim: the funnel now tests whether a small model can turn published, pre-2023 anomalies into rules that survive, which ties the write-up to McLean & Pontiff 2016 (JF: published anomalies lose about a third to half of their returns out of sample). The researcher's own ideas are still half the turns. **`pilot11` (2026-11-18 01:10–02:41 UTC, both styles, from a pilot copy of `gen-v1`'s 291 attempts, new rows #3954 on):** ran the full 1.5-hour cap without stalling, against `pilot10`'s 38 minutes before both styles were stuck. 223 new attempts on 183 distinct ideas (pilot10: 90 on 66), about 150 an hour. Never traded 6% (13), pilot10 23%. 34 skipped turns, never two in a row. Day trades took 174 attempts, long holds 49: long holds still rest often. Median timing Sharpe +0.11 day, +0.42 long. Best +1.78 (#4106, a first-four-hours breakout reached by refining the `opening_range` seed), against a luck bar of about +1.49 for the 394 backtested strategies in the pilot; in-sample, no evidence of skill. Refine turns hold most of the top scores (median +0.62 over 65); of the seeds, `pullback` was worst (median −2.01, heavy trading). The top ten are mostly variations on one breakout, so distinct ideas overstate distinct rules. Next: Adam decides whether `gen-v1` continues on `v12` (tag `gen-v1.1`, same run id so N stays one count); at pilot11's rate the remaining 709 attempts take about five hours if it does not stall later. **Adam said continue (2026-11-18):** tag `gen-v1.1` on 874b47c; `gen-v1` resumed official on `v12` with `overnight.cmd gen-v1 12 both --n 1000 --official`, first new attempt #4177. Attempts #2991–#3281 are `v10`, #4177 on are `v12`. **Stopped again (2026-11-18) at 661 attempts** (370 on `v12`, #4177–#4546, about three hours) by the same rule, 10 skipped turns in a row: both styles were turned away on every turn. Skipped turns per 50 attempts rose from 10 to 9, 30, 37, 51 and 93 before the stop, so this is the model running dry again, not a crash; seeding raised the ceiling from about 320 to about 660 ideas. 299 distinct hypotheses; long holds 140 (median +0.49, best +1.43, 4% never traded), day trades 230 (median +0.08, best +1.27, 20% never traded). **Adam decided (2026-11-19): one more change, `v13`, then generation is final.** Prior work: LLM ideation plateaus as duplicates pile up (Si, Yang & Hashimoto 2024: 200 unique of 4,000 seed ideas), which this run reproduces. `v13` takes two published, training-free fixes and changes only the researcher's explore turns; the judge is untouched. (1) Avoid overused pieces, after Frequent Subtree Avoidance (Shi, Duan & Li, AAAI 2026): the explore ask lists this style's most used conditions (entry conditions and exits, numbers removed, counted over all its accepted ideas whatever they scored) and tells the model not to use them; candidates that use one anyway are tried last. (2) Verbalized sampling (Zhang et al., ICML 2026): an explore turn asks for five different ideas with the probability the model would have proposed each, and the near-copy check tries them least likely first. Rejected: evolution of strategies by score (FunSearch, AlphaEvolve, CogAlpha) and score-driven scheduling (RD-Agent(Q)), because they hill-climb discovery scores (Allen & Karjalainen 1999: rules found by a genetic search did not beat buy-and-hold after costs out of sample); a bigger model, because most current ones were trained on data from after the holdout began; an operator grammar (WorldQuant's 101 alphas), because it removes the LLM. **Test, fixed before running:** `pilot12` from a pilot copy of `gen-v1`'s 661 attempts, both styles, 1.5 hours. It passes if all three hold: at least 100 new attempts without the stop rule firing; at most 10% never trade; the number of distinct conditions across accepted ideas grows in both halves of the pilot. If it passes, `gen-v1` continues on `v13` under tag `gen-v1.2` (same run id, one N). If it fails, `gen-v1` is frozen at N = 661. **Either way `v13` is the last change to generation:** whenever `gen-v1` next stops, by the stop rule or at 1,000, generation is frozen at that N.
  - **`pilot12` failed its pre-set test (2026-11-19), so `gen-v1` is frozen at N = 661 and generation is over.** 1.5 hours on `v13` from a pilot copy of `gen-v1`'s 661 attempts, new rows #5208–#5417. Test 1 passed: 210 new attempts on 157 ideas, one skipped turn in the whole run (pilot11: 34), never near the stop rule. Test 3 passed: distinct conditions (`agents._pieces` over accepted ideas, numbers removed) rose from 343 to 391 in the first half and 429 in the second. **Test 2 failed: 22 of 210 never traded, 10.5%, against a limit of 10%**, one attempt over. All 22 are day trades (22 of 125, 18%); long holds had none. The limit was fixed before the run and is not moved after it: an exception by one attempt is the kind of after-the-fact choice the luck bar exists to price. So `v13` fixed what it was built for (stalling and variety) and still failed the bar as written. In-sample, pilot rows, outside N: long holds 85 attempts, median +0.18, best +1.50; day trades 125, median −0.41, best +1.17. For the write-up: the ideation ceiling under the near-copy checks was about 320 ideas, about 660 with seeds, and `v13` removed the stall in a 1.5-hour test; the official funnel is the 661 attempts of `gen-v1` (`v10` and `v12`). Next is Week 5, which starts with `run_validation.py --run gen-v1` once; Adam must first confirm option A or B for the gate.
4. Week 5 scripts are written and tested on synthetic data only (134 tests in total, 12 of which need Docker running). They have never touched the real validation bars and must not until the official run is frozen. See "Week 5" for the exact sequence.

Baselines on discovery (2019–2022, BTC and ETH equal weight, 10 bp fee). In-sample numbers from a period everyone knows; not evidence of anything except that the judge behaves sensibly.

| Strategy | Timing Sharpe | Net Sharpe | Total return | Max drawdown | Trades per symbol | Mean position |
|---|---|---|---|---|---|---|
| Buy and hold | −0.50 (artifact, see "Timing Sharpe") | 1.02 | +623% | −78% | 1 | 1.00 |
| Price above 20-day average | 0.69 | 1.34 | +812% | −52% | 153 | 0.52 |
| Hourly RSI reversion | −0.48 | −0.09 | −39% | −64% | 489 | 0.17 |
| Volatility-managed | −0.45 | 0.91 | +311% | −78% | 841 | 0.78 |

The trend baseline's 0.69 is about 1.4 standard errors from zero over four years, so it is not significant even before any correction. The baselines are pilots: not counted in `N`, never sent to validation or the holdout.

Data as downloaded: BTCUSDT and ETHUSDT, 1h, 2019-01-01 to 2024-12-31, 52,549 bars each (52,608 expected). 59 bars are missing in 20 gaps, at identical times in both symbols, so they are exchange outages: 28 in 2019, 17 in 2020, 13 in 2021, 1 in 2023. The longest is 10 hours, ending 2019-05-15 13:00 UTC. That puts 58 missing bars in discovery and 1 in validation. No nulls, no duplicate timestamps, OHLC consistent, 4 zero-volume bars per symbol. All 144 checksums verified; hashes and download times are in `data/manifest.json`.

Week 5 is probably Penn State finals week (check the calendar). The plan accounts for it: the long generation run starts at the end of Week 4 and runs unattended.

## Who writes what

Adam must be able to defend every line in an interview.

| Piece | Who writes it |
|---|---|
| Backtester, baseline strategies, look-ahead detector, deflated Sharpe | Written by the assistant on 2026-11-16 at Adam's request ("You can write the backtester", then "You write both"), because he was about to lose access to the model that wrote this plan. He still has to be able to explain each one line by line. Quiz him with the questions in the specs below until he can; that is now the open work on these pieces. |
| Downloader, splits, sandbox wiring, LLM client, LangGraph glue, logging | Assistant may write. Anything touching splits or leakage still gets a test. |

## Decisions that are settled

| Decision | Reason |
|---|---|
| The LLM never judges. It proposes ideas and writes code; Python computes every metric. | An LLM has memorised market history, so its opinion of a backtest is contaminated. |
| Crypto only: BTCUSDT and ETHUSDT, hourly, Binance bulk dumps. | One clean pipeline, no splits or dividends, no delisting problem for these two. Picking more coins that still exist today adds survivorship bias. Add coins only if ahead of schedule, and pick them by a rule that uses pre-2019 information. |
| Primary model: Gemma 4 12B. Its model card states a training cutoff of January 2025. | The cutoff fixes where the holdout can start. Checked on Google's model card on 2026-11-16. |
| Only the primary model's strategies go to the holdout. | A bake-off model with a later cutoff would force a shorter holdout for everyone. The bake-off compares pipeline metrics on discovery data only. |
| The holdout months are not downloaded until the strategies are frozen. | Data that is not on the machine cannot leak. `download_data.py` refuses months from 2025-01 without `--holdout`, and `data/manifest.json` timestamps prove when each file arrived. |
| Validation runs once, after generation is finished. | If validation numbers exist while agents are running, they will leak into prompts. |
| Stocks, phone alerts, live trading: not in the six weeks. | See "After Week 6". |

## Splits

| Split | Range (UTC) | Length | Who sees it |
|---|---|---|---|
| Discovery | 2019-01-01 to 2022-12-31 | 4 years, about 35,000 bars | Strategy code runs on it; agents see its metrics |
| Validation | 2023-01-01 to 2024-12-31 | 2 years, about 17,500 bars | Code runs on it once at the end; agents never see anything from it |
| Embargo | 2025-01-01 to 2025-02-28 | 2 months | Never scored. Covers uncertainty in the model's stated cutoff |
| Holdout | 2025-03-01 to the last complete month when opened | about 21 months | Not on disk until Week 5; opened once |

Indicators need history to warm up. To score validation, run the strategy on discovery plus validation bars and score only the validation bars. Same for the holdout.

Discovery and validation are both before the model's cutoff. The model knows roughly what happened in them, so nothing measured there is evidence of skill. Only the holdout counts.

If the primary model changes, `LOCKED_FROM` in `download_data.py` and this table must move to the new model's cutoff month.

## Conventions the judge uses

These are fixed. Every test and every agent prompt assumes them.

- **Strategy interface:** `strategy(df) -> pandas Series` of positions, same length and index as `df`. `df` holds one symbol with columns `ts, open, high, low, close, volume, quote_volume, trades`.
- **Position:** a number in [0, 1]. 0 is flat, 1 is fully long. No shorting, because spot data cannot support an honest short. Values outside the range are clipped; NaN counts as 0.
- **Timing:** the position at bar `t` may use data up to and including bar `t`'s close. It is entered at that close and earns the return of bar `t+1`.
- **Returns:** `r[t] = close[t] / close[t-1] - 1`.
- **Net return:** `net[t] = pos[t-1] * r[t] - fee * |pos[t-1] - pos[t-2]|`, with the position before the first bar equal to 0.
- **Fee:** 0.001 (10 basis points) per unit of position change. This is Binance's standard spot taker fee; verify it is still current. Survivors are re-run at 0.002 as a sensitivity check.
- **Annualisation:** 8,760 bars per year. Not 252.
- **Portfolio:** run the strategy on each symbol separately and average the per-bar net returns.
- **Gaps:** Binance has outage gaps (59 bars in this data). Leave them; do not forward-fill. The return across a gap is simply the return between the two bars on either side of it.

### Timing Sharpe (the score that gates everything)

A strategy that is always long earns a positive Sharpe whenever crypto goes up. That is market exposure, not skill. The gating score removes it:

`timing[t] = net[t] - mean_pos * r[t]`

where `mean_pos` is the strategy's average position over the scored window. The timing Sharpe is the annualised Sharpe of `timing`. It rewards only *when* the strategy is in the market. An always-long strategy scores exactly zero before fees. If the series has zero variance, the Sharpe is 0.

Known artifact: with fees, buy-and-hold's timing series is one entry fee followed by zeros, and the Sharpe formula turns that into −0.50 over four years (it is `−sqrt(8760 / bars)`). The number is meaningless. It does not affect the gate, which only asks for a value above 0.

Report the plain net Sharpe and buy-and-hold next to it, but gate on timing Sharpe.

## The funnel

| Stage | Passes when | Checked by |
|---|---|---|
| 0. Generated | The model returned a hypothesis and parseable Python | `ast.parse` |
| 1. Static check | Imports only `numpy`, `pandas`, `math`; no `open`, `eval`, `exec`, `__import__`, dunder access; no hard-coded year literal such as `2021` | AST walk plus one regex |
| 2. Sandbox | Ran inside Docker within the time limit and returned a valid position series | Sandbox runner |
| 3. Look-ahead | Truncation test passes and two runs give identical output | Adam's detector |
| 4. Discovery gate | At least 30 position changes and timing Sharpe above 0 on discovery | Backtester |
| 5. Validation shortlist | Validation timing Sharpe is individually significant (PSR against 0 of at least 0.95); keep at most the top 10 | `run_validation.py` |
| 6. Holdout | See "Holdout test" | `open_holdout.py` |

Every attempt is logged with the stage where it stopped. The headline result is the count at each stage.

The hard-coded-year check exists because the truncation test cannot catch a strategy that says "go long on 2020-03-13". That strategy uses no future *data*; it uses the model's memory.

### Trial count

`N` for the deflated Sharpe is the number of official-run attempts that produced a backtest (stages 4 and later, including failures at stage 4). Pilot runs before the prompt freeze are flagged `pilot` and are excluded from validation and from `N`. Strategies are correlated, so the true number of independent trials is lower; using the raw count is conservative. Say so in the write-up.

## Decision for Adam: how strict the validation gate is

The old prompt says "correct with the deflated Sharpe ratio using the total trial count". Taken as a hard gate, that almost certainly leaves nothing to test on the holdout. Under the null, with a 2-year validation window, the deflated Sharpe reaches 0.95 only at these annualised Sharpe levels:

| Trials | Best Sharpe expected from pure luck | Sharpe needed to pass |
|---|---|---|
| 10 | 1.1 | 2.3 |
| 100 | 1.8 | 3.0 |
| 1,000 | 2.3 | 3.5 |

A net timing Sharpe of 3.5 on hourly BTC and ETH after fees is not a realistic expectation.

- **Option A (literal old rule):** gate validation on deflated Sharpe at 0.95. Expect zero survivors and a holdout that is never opened.
- **Option B (recommended, and what the funnel table above assumes):** validation only builds a shortlist of at most 10. The deflated Sharpe is computed and reported for every shortlisted strategy, which shows how much of the in-sample result is selection luck. The multiple-testing control that the final claim rests on happens at the holdout, which is independent of everything before it.

Option B is statistically valid because the holdout played no part in choosing the shortlist. It also gives the holdout a job. **Decided (Adam, 2026-11-17): Option B.** `run_validation.py` runs once with its default; never with `--option A`, and never both ways.

### Holdout test (pre-registered before opening)

- **Primary:** the equal-weight portfolio of the shortlist has a holdout timing Sharpe with PSR against 0 of at least 0.95. One test, so no correction. With about 21 months this needs an annualised Sharpe of roughly 1.2 to 1.3.
- **Secondary:** each shortlisted strategy alone, at `0.05 / K` (Bonferroni). With K = 10 that needs roughly 1.9 to 2.0; with K = 5, about 1.75.
- **Null control:** take a strategy's position series and rotate it in time by a random amount. This keeps its turnover and time in the market and destroys its timing. As built:
  - At validation, every validated strategy gets 100 shifted copies. The summary reports the share of real strategies that are individually significant next to the share of shifted copies that are (about 5% by construction). If the two are close, the agents added nothing.
  - At the holdout, the shortlist portfolio gets 1,000 shifted versions, giving a permutation p-value for the primary result.
  - Both are reported, not gates. Limit: a strategy that repeats on the calendar, such as an hour-of-day rule, is unchanged by a shift of whole days, so 1 copy in 24 keeps its full edge and neighbours keep most of it. For those strategies the null is conservative. In a test with a real planted hour-of-day edge the PSR test passed while the permutation p-value was only 0.10.

Sharpe precision depends on the calendar span of the data, not the number of bars. The standard error of an annualised Sharpe is roughly `1 / sqrt(years)`: about 0.71 for validation and 0.76 for the holdout. Hourly bars do not tighten it. This is why the holdout length matters and why a model with a late cutoff is expensive.

## Leakage channels and what blocks each

| Channel | Blocked by |
|---|---|
| Strategy code reads future bars | Look-ahead detector |
| Judge misaligns positions and returns | Toy example and the alignment property test |
| Picking the best of many trials | Deflated Sharpe reported; holdout test controls the final claim |
| Model remembers market history | Holdout after the cutoff; hard-coded-year check |
| Validation results reach the agents | Validation runs only after generation ends, in a separate script |
| Adam or a cloud assistant knows the 2025–26 market | Agent prompts stay generic with no market hints; prompts are committed before the official run; no thresholds change after validation is seen; nothing changes after the holdout |
| Peeking at the holdout | Not on disk; opened by one script that refuses to run twice |
| Re-running generation after seeing validation | Every official attempt counts in `N`; if validation is ever run twice, the write-up says so |
| Coin selection | BTC and ETH only; stated as a limitation |
| Optimistic costs | Re-run survivors at double fees |

## Architecture

```
hypothesis agent ─► coder agent ─► static check ─► Docker sandbox ─► judge (host) ─► critic agent
        ▲                                          (strategy + look-ahead test)          │
        └──────────────────── summary of the attempt log (DuckDB) ◄──────────────────────┘
```

- **One model, three prompts.** Served by llama.cpp's `llama-server`; the code talks to an OpenAI-compatible endpoint, so any other server is a URL change.
- **Hypothesis and critic return JSON** validated by Pydantic. **The coder returns a fenced Python block**, not JSON. Small models break JSON escaping when code is inside a string.
- **The sandbox returns positions only.** The host computes all metrics from them. Generated code never touches a metric.
- **Sandbox command** (in `sandbox.py`): `docker run --rm -i --name <id> --network none --cpus 1 --memory 1g --pids-limit 128 --read-only --tmpfs /tmp`, with only the strategy file and `harness.py` mounted read-only. Killing the `docker` client on timeout does not stop the container, so `timeout 60 python /harness.py` runs inside it and `docker kill <id>` is the backstop.
- **No data files exist inside the container.** Bars go in on stdin and positions come back on stdout. A strategy can only see the DataFrame it is handed, which closes the "read the full file from disk to beat the truncation test" loophole. The static check also bans `open` and `read_*`.
- **Only discovery data is ever sent during generation.**
- **The image pins pandas 2.2.3 and numpy 1.26.4** on Python 3.12. The model's training ends January 2025, so it writes code for those versions; the same code breaks on pandas 3 and numpy 2. The host runs pandas 3; results match exactly on the baseline.
- **Docker Desktop must be running** before `generate.py` or the container tests. Its engine can take several minutes to start.
- **Write each node as a plain function first** and test it without LangGraph. Wire the graph in Week 4.
- **The DuckDB log is the checkpoint.** On restart the loop reads the log and continues. One writer process only.

### What the hypothesis agent is shown

It cannot read a thousand attempts in a 16k context. Build the summary in code, under about 3,000 tokens:

- attempt counts per family and per failure stage
- the five best and five worst by discovery timing Sharpe: hypothesis text and metrics, no code
- the last five attempts with critic notes

Each hypothesis carries a `family` from a fixed list: trend, mean reversion, breakout, volatility, volume, time of day, other. Alternate between "refine attempt X" and "propose something from the least-tried family". Without this the model proposes moving-average crossovers forever. Reject exact duplicates by hashing the position series. Add a correlation check only if near-duplicates are visibly eating the run.

### Attempt log (one DuckDB table, `attempts`)

As built in `store.py`: `id, created_at, run_id, pilot, model, prompt_version, parent_id, tries, family, hypothesis, code, status, error, lookahead, pos_hash, timing_sharpe, net_sharpe, total_return, max_drawdown, turnover_per_year, trades_per_symbol, mean_position, equity, seconds`

- Every metric in this table is from the discovery period. Nothing else may ever be written to it.
- `status` is one of: `static_reject, crash, timeout, bad_output, lookahead, no_trades, too_few_trades, duplicate, ok`. Unparseable model output lands in `static_reject`.
- `no_trades` means the position never changed on any symbol. That is nearly always a coding bug, so it goes back to the coder and does not count toward `N`. `too_few_trades` (1 to 29 changes) is a real but too-quiet strategy and does count.
- `lookahead` is `pass`, or `skipped` while the detector does not exist.
- `tries` is 0 for the first attempt at an idea; a retry's `parent_id` points at the attempt it fixes.
- `equity` is one point a day, for the dashboard.
- Every model call (prompt, reply, tokens, seconds) is in a second table, `llm_calls`.
- The file is `runs/attempts.duckdb`, git-ignored. Export the official run to Parquet and commit that at the end.
- DuckDB allows one writer or several readers at a time, not both. `store.py` opens the file per call and retries, which is what lets the dashboard read while the generator writes. Ceiling: fine at one write a minute; if runs ever write many times a second, switch the log to SQLite in WAL mode.

### Code map

| File | Role | Runs where |
|---|---|---|
| `download_data.py`, `make_splits.py` | Data in, discovery split out | Host, by hand |
| `backtest.py` | The judge | Host |
| `baselines.py` | Hand-written reference strategies | Host |
| `static_check.py` | AST and year-literal filter | Host |
| `sandbox.py` | Starts the container, pipes data | Host |
| `harness.py` | Loads and runs the strategy, calls the look-ahead detector | Container only |
| `lookahead.py` | **Adam's.** `check_lookahead(strategy, df) -> (ok, reason)` | Container only |
| `pipeline.py` | static check, sandbox, judge, log; one call per attempt | Host |
| `store.py` | Attempt log | Host |
| `agents.py` | LLM client, coder and researcher prompts, log summary | Host |
| `generate.py` | LangGraph loop and command line | Host |
| `dashboard.py`, `dashboard.html` | Live read-only view of the log | Host, localhost only |
| `dsr.py` | Probabilistic and deflated Sharpe | Host |
| `run_validation.py` | Validation scoring, null control, shortlist, writes `PREREGISTRATION.md` | Host, once |
| `open_holdout.py` | Registration checks, then the holdout test | Host, once ever |

Validation results live in a separate `validation` table and in `results/`. `agents.py` reads only `attempts`, so they cannot reach a prompt; a test checks this.

**Reviewer agent (added 2026-11-16 at Adam's request).** Every 20 attempts (`--review-every`) the model is shown the latest batch of discovery results and writes three short lists: what worked, what failed and why, and what to try next. The text is stored in the `lessons` table and the researcher reads the two most recent notes from then on. This is the project's memory across a long run.

- It does not judge anything. Every gate is still Python. The reviewer only turns numbers the researcher could already see into notes.
- It sees discovery results only, like the researcher.
- Its prompt tells it that scores between −0.5 and +0.5 are luck. It will still over-read noise sometimes; that is what validation is for.
- The researcher cycles through three instructions: explore the least-tried family (shown what that family already tried), refine a better attempt, follow one of the reviewer's next steps.

**Unattended runs.** `python generate.py --run <name> --hours 5` researches until the time is up, then writes `runs/<name>_report.md`. It keeps Windows awake while it runs, waits and retries if the model server or Docker fails (giving up after 10 failures in a row), and resumes from the log if restarted with the same run name. `--report` rewrites the report for a finished run.

Validation and holdout results go in separate tables written by `run_validation.py` and `open_holdout.py`. The agent loop never opens them.

## Week by week

### Week 1 (Nov 16–22). Done when the backtester matches a manual calculation on the toy example.

1. `git init`, first commit. Commits are the audit trail for the prompt freeze and the pre-registration.
2. Run `python download_data.py`. Expect about 52,608 bars per symbol for 2019-01 to 2024-12, minus a few outage gaps. Record the gap count.
3. `make_splits.py`: writes `data/splits/discovery/` (bars before 2023-01-01) and refuses to run if `data/klines` holds any bar from 2025 on. There is no separate validation file: `run_validation.py` reads `data/klines`, which ends 2024-12-31, and scores bars from `DISCOVERY_END` on. Import the boundary constants from `make_splits.py`; do not retype the dates anywhere.
4. Backtester: done, in `backtest.py`. `evaluate()` scores a strategy across symbols and takes a `start` index so validation and the holdout can be scored on top of warm-up history.
5. Baselines: done, in `baselines.py`. One per family, each with textbook parameters from its source and no tuning: price above its 20-day average (Detzel et al. 2021), hourly RSI(14) reversion (Wilder's 30/50 levels), volatility-managed exposure (Moreira and Muir 2017). Do not adjust their parameters after seeing results.
6. **Adam:** work the toy example by hand, then read `backtest.py` against it.

### Week 2 (Nov 23–29, Thanksgiving). Done when 20 generated strategies run without crashing the pipeline.

A strategy crashing is fine and gets logged. The pipeline crashing is not.

Steps 2 to 4 are built. What is left is step 1, then running `python generate.py --ideas ideas.txt` (step 3's test) and step 5's measurements. Model settings come from the environment: `LLM_BASE_URL` (default `http://127.0.0.1:8080/v1`) and `LLM_MODEL` (default `gemma-4-12B-it-Q4_K_M`, a label for the log only; the server answers with whatever model it loaded).

How to work through this test:

- Run `python generate.py --ideas ideas.txt` with the dashboard open. Each idea gets up to three tries.
- Read the failures in the dashboard's attempts table. The error text is what the coder was shown on its retry.
- If one kind of failure repeats, change the `CODER` prompt in `agents.py`, bump `PROMPT_VERSION`, and rerun under a new run id (`--run pilot2`). An ideas run resumes by run id, so reusing `pilot` would skip everything.
- When most ideas end in a backtest, move on to the researcher loop: `python generate.py --run pilot3 --n 30`.
- Everything here is a pilot. None of it counts toward `N`, and none of it may be validated.

1. Done: `llama-server` with Gemma 4 12B Q4_K_M, 16k context, thinking off. See "Status" for the command.
2. LLM client: one function, with a timeout and a retry. Log every call (prompt, response, tokens, seconds) to a DuckDB table `llm_calls`.
3. Coder agent, fed 20 hand-written plain-English ideas. The hypothesis agent comes in Week 4.
4. Static check, Docker image (Python 3.12, numpy, pandas, pyarrow), sandbox runner, harness.
5. Measure: share of attempts that produce valid code, seconds per strategy. Estimate: about one strategy a minute, so 1,000 take roughly 17 hours. Replace the estimate with the measurement.

### Week 3 (Nov 30–Dec 6). Done when the detector catches 100% of planted cheats with no false alarms.

1. **Adam:** look-ahead detector and the planted-cheat suite, per the spec below.
2. Add the hard-coded-year check to the static stage.
3. Run the detector inside the sandbox harness.
4. If ahead: **Adam** writes the deflated Sharpe now. It is pure math with no dependencies on the rest, and doing it early protects finals week.

### Week 4 (Dec 7–13). Done when the loop runs unattended overnight and resumes after being killed.

Steps 1 and 3 are built and tested against a scripted model only: `python generate.py --run pilot2 --n 50`. Step 2 was cut (see "Code map"). The real work left this week is prompt tuning on pilot runs, then steps 4 to 6. Bump `PROMPT_VERSION` in `agents.py` on every prompt change.

1. Hypothesis agent and the log summary.
2. Critic agent: cut.
3. LangGraph wiring.
4. Kill the process mid-run and restart it. It must continue from the log with no duplicate or lost attempts.
5. Overnight hazards on Windows: sleep and automatic update restarts, a hung model call (needs the timeout), leftover containers.
6. Freeze: commit the prompts and tag `gen-v1`. Everything before the tag is pilot. Start the official run to 1,000 attempts at the end of the week so it runs through finals.

### Week 5 (Dec 14–20). Done when the funnel numbers are final.

The scripts are built. The whole week is this sequence, run once, in order, after `generate.py --run gen-v1 --official` has finished and nothing more will be generated:

```
python run_validation.py --run gen-v1        # scores validation, runs the null, writes PREREGISTRATION.md
(read PREREGISTRATION.md)
git add PREREGISTRATION.md results/ && git commit -m "Pre-register the holdout test" && git tag prereg-v1
python download_data.py --holdout --end <last complete month>
python open_holdout.py --run gen-v1          # once, ever
```

Locks that enforce the order:

- `run_validation.py` refuses a pilot run, and refuses a second pass unless `--rerun` is given; passes are numbered and a second one has to be reported.
- `download_data.py --holdout` refuses until the tag `prereg-v1` exists.
- `open_holdout.py` refuses unless `PREREGISTRATION.md` and the judge files (`backtest.py`, `dsr.py`, `lookahead.py`, `harness.py`, `sandbox.py`, `static_check.py`, `run_validation.py`, `open_holdout.py`, `make_splits.py`, `Dockerfile`) are unchanged since the tag, and every shortlisted strategy's code still matches its registered SHA-256.
- It writes `results/HOLDOUT_OPENED.json` before computing anything and refuses forever after.
- An empty shortlist means the holdout is never opened and the result is zero survivors.

`--option A` on `run_validation.py` selects the strict deflated-Sharpe gate; the default is B. Adam still has to confirm the choice before running it. Do not run it both ways and pick.

The original step list, for reference:

1. **Adam:** deflated Sharpe, per the spec below, if not done.
2. `run_validation.py`: one run over all stage-4 survivors. Shortlist, with deflated Sharpe reported (`dsr.py`; convert annualised numbers to per-bar first). The sandbox re-runs the look-ahead detector on the longer discovery-plus-validation data. A strategy that fails there is rejected even though it passed on discovery: a full-sample statistic can look honest on one dataset and leak on a longer one.
3. Null control.
4. Pre-registration: commit `PREREGISTRATION.md` listing the shortlist (code hashes), the holdout window, the primary and secondary tests, and the fee. Tag `prereg-v1`.
5. `python download_data.py --holdout --end <last complete month>`.
6. `open_holdout.py`: refuses to run if `results/HOLDOUT_OPENED.json` exists, writes that file first, then scores. One run. Whatever it says is the result.

### Week 6 (Dec 21–27). Done when someone else can rerun it from the README.

1. README: the funnel table first, then method, limitations, and the five commands to reproduce.
2. Bake-off: about 100 discovery-only attempts per extra model. Compare valid-code rate, look-ahead rejection rate, duplicate rate, speed. No validation, no holdout.
3. Results page (cut first; a table in the README is enough).

### If short on time, cut in this order

Langfuse, results page, critic agent, vectorbt cross-check, bake-off down to one extra model, target of 1,000 down to 300. Never cut the look-ahead detector, the holdout, the toy test, or the pre-registration.

## Stack notes

- **Python 3.14 is what is installed.** pandas and pyarrow work on it. numba, which vectorbt needs, often lags new Python versions. If vectorbt will not install, cross-check the vectorised backtester against a plain loop version Adam writes himself. That is a better independent check anyway, and it adds no dependency.
- **Langfuse** self-hosted now needs Postgres, ClickHouse, Redis and object storage. The `llm_calls` table gives tracing from day one. Add Langfuse in Week 4 only if the loop already works, with a two-hour limit.
- **Docker Desktop on WSL2** reads Windows-mounted folders slowly. The files here are small enough not to matter; if sandbox start-up dominates the run time, move the data into a Docker volume.
- **Hypothesis (the testing library)** was planned for the backtester's properties; seeded random cases do that job, so it is not installed.
- **Dashboard layout (third version, 2026-11-17, built after asking Adam what he uses it for):** judge a finished run, compare runs, find and inspect strategies. Order follows "overview first, zoom and filter, then details on demand" (Shneiderman 1996). Sections: the verdict against the luck benchmark and the funnel; **Compare runs** (pick up to three: a table of key measures per run, and the share of backtested attempts at or below each score, one line per run); **Narrow down** (drag across histograms of timing Sharpe, trades and time in market, or box-select on a score-against-trades scatter; every chart and the table follow the same filters); **Strategies** (near-copies collapsed into one row per rule, with the group's size, expandable); **Inspect** (metrics, a replay of the strategy's trades, where the idea came from and what was built on it, equity against buy-and-hold, code). The 3D views were removed: position on a flat common scale is read far more accurately than depth (Cleveland and McGill 1984). Near-copy groups come from `dashboard.groups`, which applies the researcher's skeleton test (`agents._skeleton`) transitively within a run.
- **Replay:** `/api/positions/<id>` reruns an attempt's code in the Docker sandbox on the discovery bars, without the look-ahead test, and caches its position changes in `runs/positions/`. Docker must be running; the first view of a strategy takes one to three seconds. `/api/bars` serves discovery candles only. The page draws them with Lightweight Charts 5.0.9 (TradingView, Apache 2.0, from jsDelivr), the only library on the page: candles, buy and sell markers, the position underneath, and a play mode that reveals the bars at a chosen speed with a short animation on each trade (none when the system asks for reduced motion). Its running equity is one symbol's, recomputed in the browser with the judge's formula for display; the scores everywhere else come from `backtest.py`.
- **Holding time in the dashboard:** `hold_hours` is time in the market divided by the number of entries (two position changes per round trip), computed in SQL from logged columns, so it covers every run, old ones included. It is exact for all-or-nothing positions and approximate for partial ones. The page filters by Day trades (under 24 hours), Multi-day (1 day to 2 weeks) and Long (2 weeks or more), sorts by holding time, and shows the share of day trades when comparing runs.
- **Dashboard:** `python dashboard.py`, then http://127.0.0.1:8000. Standard-library HTTP server and one HTML file, no build step, polls every 5 seconds. It shows discovery results only, labels them in-sample, and keeps validation, holdout and the forward test locked. It must never fetch current prices before the holdout is opened: those are holdout data. Move to FastAPI and React only if it needs more than three read-only routes.
- **No OpenAI client library and no Pydantic-only dependency:** the LLM call is one `urllib` POST; Pydantic arrives with LangGraph and validates the researcher's JSON.

## Specs for the three pieces Adam writes

All three are now implemented by the assistant (see "Who writes what"). These specs are the review checklists and the source of quiz questions. Use the questions to probe, not to hint. Answer keys are at the bottom of this file.

### Backtester

Implemented in `backtest.py` by the assistant; this spec is now the review checklist and the source of quiz questions. All six tests below exist in `tests/test_backtest.py`, plus a comparison against an independent bar-by-bar loop, which replaces the vectorbt cross-check. The property tests use seeded random cases, not the Hypothesis library; add it only if a real need appears.

Input: close prices, positions, fee. Output: the net return series, then metrics (net Sharpe, timing Sharpe, max drawdown, turnover, number of position changes). Follows "Conventions the judge uses" exactly.

Tests it must pass:

1. The toy example in the answer key, to 9 decimal places.
2. Fee 0 and position always 1 reproduces buy-and-hold exactly.
3. Position always 0 gives all-zero returns.
4. Raising the fee never raises final equity.
5. Alignment property: changing positions from bar `t` onward never changes net returns up to and including bar `t`.
6. With fee 0, always-long has a timing Sharpe of exactly 0 (the zero-variance rule).

Mistakes to look for:

- `pos[t] * r[t]`: the position earns the return of the bar it was computed from. This is look-ahead inside the judge, and it makes every strategy look brilliant.
- Fee charged on the position instead of the change in position.
- No fee on the first entry.
- Annualising with 252.
- Max drawdown computed on returns instead of on the equity curve.
- Division by zero when returns have no variance.

Questions: Why can't the position at bar `t` earn bar `t`'s return? What does a strategy that flips every bar pay per year in fees? Why is always-long not skill?

### Look-ahead detector

Idea: a strategy that uses no future data gives the same positions for bars up to `c` whether or not bars after `c` exist. Run it on the full data and on data cut at `c`, and compare.

Implemented in `lookahead.py`; tests in `tests/test_lookahead.py`. It uses about 40 cuts: seven fixed fractions from 2% to 85%, the last bar, 12 seeded random ones, and up to 20 placed right after bars where the position changed (added 2026-11-17; see "Judge audit"). It costs about 0.04 seconds on 35,000 bars for a vectorised strategy.

Measured detection: on the real BTC and ETH discovery data, 14 of 14 cheats caught and 12 of 12 honest strategies passed. On 30 random datasets per cheat, 13 cheats were caught 30 of 30 times. The exception is "long while below the dataset's all-time high", caught 24 of 30. The six misses are datasets where the high falls inside the first 2% of bars, before the earliest cut. In those cases the leak changes at most the positions before that high, so its effect on the score is negligible, but the write-up should state the limit: **a leak confined to bars before the first cut, or to a stretch no cut lands in, is not seen.**

Interface the rest of the system expects: a file `lookahead.py` in the project root with `check_lookahead(strategy, df) -> (ok, reason)`, where `strategy` is the function, `df` is one symbol's bars, `ok` is a bool and `reason` is a short string for the log. It may import only numpy and pandas, because it runs inside the container. Once the file exists, `sandbox.py` mounts it and `harness.py` calls it for every symbol; a failure is logged with status `lookahead` and the reason is fed back to the coder.

Requirements:

- Compare the whole prefix up to the cut, not only the last bar.
- Several cuts spread over the entire range, early ones included, some random with a logged seed.
- Run the full data twice; different outputs mean the strategy is non-deterministic, which is its own rejection.
- The strategy receives a copy, and the original must be unchanged afterwards.
- NaN equals NaN. Use a tight tolerance; honest rolling and expanding calculations reproduce exactly on a prefix.

Planted cheats it must catch (all of them):

`shift(-1)`; centred rolling window; z-score using the full-sample mean and standard deviation; scaling by the full-sample min and max; threshold at a full-sample quantile; "long while close is below the all-time max of the dataset"; `rank(pct=True)` over the full series; `bfill`; a model fitted on the whole dataset; anything using `df.iloc[-1]`; an exponential average run on the reversed series; a cheat that activates only when `len(df)` exceeds a threshold; a cheat that activates only in the last 10% of bars.

Honest controls it must pass: moving-average crossover, RSI, expanding z-score, rolling volatility filter, hour-of-day rule.

Mistakes to look for:

- Checking only the final bar of each cut.
- All cuts late in the data. "Long while close is below the dataset max" looks honest for every cut after the max occurs.
- A loose tolerance that hides small leaks.

Questions: Which cheat does a single late cut miss, and why? What can this test never detect? (A strategy that hard-codes dates from the model's memory.)

### Deflated Sharpe ratio

Source: Bailey and López de Prado, "The Deflated Sharpe Ratio" (2014). The probabilistic Sharpe ratio (PSR) is from their 2012 paper "The Sharpe Ratio Efficient Frontier".

Implemented in `dsr.py` (`psr`, `expected_max_sharpe`, `deflated_sharpe`, `sharpe_stats`); tests in `tests/test_dsr.py`. In the simulation test, the best of 50 pure-noise strategies looks significant under a plain test more than 85% of the time and under the deflated Sharpe less than 10% of the time.

All Sharpe values here are per bar, not annualised.

- `PSR(SR*) = Φ( (SR − SR*) · sqrt(T − 1) / sqrt(1 − skew · SR + (kurt − 1) / 4 · SR²) )`
- `SR0 = sqrt(V) · ( (1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e)) )`
- `DSR = PSR(SR0)`

`T` is the number of return observations. `skew` and `kurt` are of the strategy's returns, with `kurt` equal to 3 for a normal distribution. `V` is the variance of the per-bar Sharpe across trials. `N` is the trial count. `γ` is 0.5772156649 (Euler–Mascheroni). `Φ` is the standard normal CDF.

Tests it must pass:

1. The reference cases in the answer key.
2. Decreasing in `N`, increasing in `T`.
3. `N = 1` returns PSR against 0.
4. Simulation: generate `N` strategies of pure noise, take the best, compute its DSR. It should exceed 0.95 in only a small share of repetitions (allow up to about 10%).

Mistakes to look for:

- Feeding annualised Sharpe into a per-bar formula.
- Excess kurtosis (normal = 0) where the formula wants raw kurtosis (normal = 3). `scipy.stats.kurtosis` returns excess by default.
- `V` in different units from `SR`.
- `N = 1` makes `Φ⁻¹(0)` minus infinity. Handle it explicitly.

Limitation to state in the write-up: the formula assumes independent returns. Volatility clustering in crypto makes the true uncertainty somewhat larger.

Questions: Why does the bar rise with `N`? Why does it rise with `V`? What would the expected best Sharpe be from 1,000 coin-flip strategies over two years?

## After Week 6

In this order, each only if the one before it happened.

1. **Forward test.** Once a month, download the new month and score the frozen shortlist. This is the "live test" from the original idea, and it is the cleanest evidence available because it is after everyone's knowledge, including Adam's.
2. **Stocks as a robustness check.** Run frozen survivors unchanged on stock data the agents never saw.
3. **Phone alerts.** Only if the primary holdout test passed. A scheduled script that posts to a push service (ntfy or similar) when a frozen strategy changes position is a few lines. A holdout pass at 5% is weak evidence and not a reason to risk money.

## Things to verify before relying on them

- Resolved 2026-11-16: Gemma 4 12B Q4_K_M with a 16k context fits in 12 GB (10.3 GB shown in use).
- Training cutoffs of Devstral Small 2 and Qwen3-Coder-30B-A3B (not found in a search on 2026-11-16; check their Hugging Face model cards).
- Resolved 2026-11-16: llama-server accepts `response_format` with a JSON schema; the researcher call returned valid JSON.
- The "Profit Mirage" numbers. The old prompt says returns fell 50 to 72%. The abstract gives no numbers, and a secondary summary reports Sharpe decay of about 51 to 62%. Read the paper body before quoting a figure in the README or an interview.
- Binance's current spot taker fee.
- Penn State's finals dates.
- Whether Binance spot kline files ever carry a header row. The parser handles both.

## Answer keys

These are the numbers the tests check against. Working the toy backtest by hand is still the fastest way to understand the backtester.

### Toy backtest

Close: 100, 110, 99, 108.9. Positions at bars 0, 1, 2: 1, 0, 1. Fee 0.001.

| Bar | Return | Position held | Position change | Net return | Equity |
|---|---|---|---|---|---|
| 1 | +0.10 | 1 | 1 | 0.099 | 1.099 |
| 2 | −0.10 | 0 | 1 | −0.001 | 1.097901 |
| 3 | +0.10 | 1 | 1 | 0.099 | 1.206593199 |

Total turnover 3. Max drawdown −0.1%. Buy-and-hold ends at 1.089.

### Deflated Sharpe reference cases

| Case | Inputs | Result |
|---|---|---|
| 1 | SR 0.1 per bar, T 1000, skew 0, kurt 3, N 100, V 0.001 | SR0 = 0.080025, DSR = 0.7356 |
| 2 | As case 1 with skew −0.5, kurt 5 | DSR = 0.7301 |
| 3 | SR 0.1, T 1000, skew 0, kurt 3, N 1 | DSR = PSR(0) = 0.99919 |
| 4 | Annualised SR 2.0 on 2 years hourly (T 17,520), N 1000, V = 1/T | SR0 annualised = 2.30, DSR = 0.335 |

A simulation of 100 noise strategies with 500 observations gave an average best Sharpe of 0.1119 per bar against 0.1132 from the SR0 formula.

## Sources

- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Binance public data format: https://github.com/binance/binance-public-data/
- Profit Mirage: https://arxiv.org/abs/2510.07920
