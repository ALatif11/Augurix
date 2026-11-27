# Can a small local LLM find trading rules that survive an honest test?

Augurix, November 2026. Code, data checksums and every number below are in this repository; [README.md](README.md) says how to rerun it.

## The question

A language model can write a trading strategy in seconds, and almost any strategy looks good on some backtest. The question here is narrower than "can AI trade": if a small model running on one consumer GPU proposes hundreds of rules for Bitcoin and Ethereum, and every rule is judged by code that cannot be argued with, how many survive a test on data the model could not have seen? The answer is reported as a survival funnel, and zero was an acceptable outcome.

## Setup

**Data.** Hourly Binance spot bars for BTCUSDT and ETHUSDT from the public bulk dumps, with SHA-256 checksums recorded on download. Three splits by time:

| Split | Months | Who sees it |
|---|---|---|
| Discovery | 2019-01 to 2022-12 | The agents, as scores only |
| Validation | 2023-01 to 2024-12 | Python, once |
| Holdout | 2025-03 to 2026-10 (Jan and Feb 2025 are indicator warm-up) | Python, once, after pre-registration |

The model's training data ends in January 2025. Discovery and validation are both before that cutoff, so a rule could in principle be built from remembered market history. Only the holdout is clean, which is why it carries the headline test.

**Model.** Gemma 4 12B instruct, 4-bit (Q4_K_M), served locally by llama.cpp. The plan was a 5- or 6-bit quant; Q4_K_M is what was already on disk, and it runs with a 16k context on a 12 GB card. One model plays three roles with different prompts: a researcher that proposes an idea in plain English, a coder that writes it as a Python function returning a position between 0 and 1 for every bar, and a reviewer that reads batches of results and writes notes for the researcher.

**Judge.** Plain Python, no model involved:

1. A static check rejects imports, file access and hard-coded calendar years.
2. The code runs in a Docker container with no network and capped memory and CPU.
3. A look-ahead test reruns the strategy on about 40 truncated copies of the data. If any position up to the cut changes when later bars are removed, the code used the future.
4. A vectorised backtester scores the positions with a 0.1% fee on every change in position. A position decided at the close of bar `t` earns bar `t+1`'s return.

**The score.** Most strategies on a rising asset make money by being long. The score that matters is the timing Sharpe: the annualised Sharpe ratio of the strategy's net return minus what holding its own average position would have earned. Always-long has a timing Sharpe of zero before fees.

**Counting tries.** Every attempt in the official run is logged. The trial count N for the deflated Sharpe ratio (Bailey and López de Prado, 2014) is the number of official attempts that produced a backtest. With many tries, the best score from luck alone rises, and the deflated Sharpe asks whether a result beats that bar.

## What happened during generation

The official run, `gen-v1`, was meant to reach 1,000 attempts. It stopped twice on its own, both times by a stop rule fixed before it started (ten turns in a row where every proposal was turned away as a near-copy of an earlier idea).

- **First stop, 291 attempts (prompts `v10`).** The model kept proposing the same 168-hour breakout with a different filter, and the near-copy check kept refusing it. A pilot with better rest rules between the two holding styles (`v11`) ran dry after 90 more attempts. Under the near-copy checks this model had roughly 320 distinct-enough ideas in it.
- **Seeding (prompts `v12`).** Half of the researcher's "explore" turns were seeded with one of 26 strategy families from research published before 2023 (time-series momentum, channel breakouts, day-of-week effects, short-term reversal and others). The run continued under the same run id, so every attempt counts toward one N, and each row records which prompt version produced it: #2991 to #3281 are `v10`, #4177 to #4546 are `v12`. It stopped again at **661 attempts**. Seeding roughly doubled the ceiling. It also changes what the funnel tests: whether a small model can turn published anomalies into rules that survive, when McLean and Pontiff (2016) found that published anomalies lose about a third to half of their returns after publication.
- **A last fix that failed its own test (`v13`).** Two published fixes for repetitive LLM ideation (asking for several ideas with their probabilities, and steering away from overused conditions) were tried in a pilot with three pass conditions set in advance. It fixed the stalling and grew the variety of conditions, but 10.5% of its strategies never traded, against a limit of 10%. The limit was not moved. Generation was frozen at 661.

This plateau matches what Si, Yang and Hashimoto (2024) found for LLM research ideas: duplicates pile up quickly. It is a finding in its own right. A 12B model asked for distinct trading rules runs out of them in the low hundreds.

## The funnel

| Stage | Left |
|---|---|
| Generated | 661 |
| Passed the static check | 631 |
| Ran in the sandbox | 603 |
| Passed the look-ahead test | 594 |
| Timing Sharpe above 0 on discovery | 266 |
| Individually significant on validation | 13 |
| Shortlisted (top 10) | 10 |
| **Survived the holdout** | **1** |

Between "passed the look-ahead test" and the discovery gate, 60 strategies never traded at all, leaving N = 534 backtested.

## Validation

All 266 strategies with a positive discovery score were run once on 2023 to 2024. None failed the look-ahead test on the longer data.

- 13 of 266 (4.9%) had a probabilistic Sharpe ratio against zero of at least 0.95. That is about what chance alone produces at a 5% level. As a null control, the same strategies were run on copies with their positions shifted in time, which keeps how often they trade and drops any timing information; 1.3% of those passed.
- The luck bar at N = 534 was an annualised Sharpe of 2.17. The best validation score was 1.82. **No strategy had a deflated Sharpe ratio near significance (the best was 0.28),** so the strict rule (option A) would have shortlisted nothing.
- The pre-chosen rule (option B, picked before validation ran) shortlisted the ten strategies with the highest validation timing Sharpe among those 13.

The shortlist turned out to be close to two ideas. All ten are day trades in the market 1% to 4% of the time. Seven are one "Monday breakout" (Monday's close above the 168-hour high, with a volume or volatility filter, out after 8 to 12 hours), and three are a short volatility breakout. That was reported and not fixed: changing the shortlist rule after seeing it would have been tuning on validation.

## Pre-registration and the holdout

Before any holdout data existed on disk, `PREREGISTRATION.md` fixed the ten strategies (by SHA-256 of their code), the window, the fee and both tests. It was committed, tagged `prereg-v1` and pushed to GitHub, which gives an outside timestamp. The downloader refuses holdout months without that tag. `open_holdout.py` refuses to run if any judge file changed since the tag, writes a lock file before computing anything, and refuses forever after. It ran once, on 2026-11-22.

- **Primary test, failed.** The equal-weight portfolio of the ten had a holdout timing Sharpe of +0.47 and a probabilistic Sharpe ratio of 0.73, against the 0.95 required. At double the fee it was −0.18. Reported but not part of the test: it beat its own time-shifted copies with a permutation p-value of 0.011.
- **Secondary test, one survivor.** Each strategy alone needed a probabilistic Sharpe ratio of at least 0.995 (Bonferroni for ten tests). #4323 passed with 0.9964 and a timing Sharpe of +1.85. Its rule: buy when the hour's price change is more than three times the past week's typical hourly change and the close is above the past week's high, then sell 12 hours later.

The holdout test is not deflated for the 534 tries. Those tries were used to choose the shortlist on discovery and validation data, and the holdout is new data that played no part in the choice. The only multiple testing left is the ten strategies tested on it, and that is what the Bonferroni bar covers.

## What the result means

Weak evidence, and no reason to trade anything.

- The one survivor cleared its bar by 0.0014. It is in the market about 4% of the time, and its hourly returns have a skew of 13 and a kurtosis of 536, so a handful of large moves carry the result. It was not tested at double the fee on its own.
- Its two near-copies in the shortlist also made money on the holdout (+1.75 and +1.49), and all seven Monday breakouts lost or went flat (−1.78 to +0.03). That split was noticed after the fact. Three near-copies agreeing is one idea working once, not three confirmations, and a short volatility breakout is a known family (it came in through the "overreaction" seed).
- The portfolio test, the one claim the project committed to, failed.

The process mattered more than the result. Without the luck bar, the 4.9% validation pass rate would have looked like a discovery. Without pre-registration, the obvious move after seeing the shortlist would have been to drop the Monday rules, which is exactly the after-the-fact selection that makes backtests lie.

## Bake-off

Is Gemma the right model to drive this? Qwen3-14B (also 4-bit Q4_K_M) ran the same code for 100 discovery-only pilot attempts, and so did Gemma, back to back on 2026-11-22. Both used the same server settings (8k context with an 8-bit cache, needed for Qwen to fit in 12 GB) and the current prompts (`v13`), so these Gemma numbers are a fresh run and not the official one. No validation, no holdout, and no scores compared: a best in-sample Sharpe from 100 tries says nothing about a model.

| Model | Attempts | Ideas | Valid code | Look-ahead rejects | Never traded | Duplicates | Attempts per hour | Output tokens per second |
|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | 100 | 76 | 80% | 0% | 11% | 1% | 290 | 50 |
| Gemma 4 12B | 100 | 90 | 94% | 0% | 6% | 0% | 184 | 48 |

Valid code is the share of attempts whose code passed the static check and ran. The other rates are shares of the attempts that ran, or (duplicates) of those backtested. A share measured on 100 attempts has a standard error of 2 to 4 points, so the 14-point gap in valid code is about three standard errors and real; the 5-point gap in never-traded is within noise.

- **Gemma writes working code more often.** 19 of Qwen's 100 attempts crashed, against 1 of Gemma's, so Qwen spent more of its attempts on retries (76 ideas against 90).
- **Qwen is faster per attempt, not per token.** Both generate about 50 tokens a second; Qwen's answers are shorter (about 50,000 output tokens in all against 83,000), so it finished its 100 attempts in 21 minutes against Gemma's 33.
- **Neither cheated here.** No attempt from either model failed the look-ahead test (the official run had 9 in 603), so in these runs the coder's mistakes were crashes and rules that never fire, not leaks.

Gemma stays the better fit for this pipeline. The bake-off does not say which model has better ideas; finding that out would take a full official run and a holdout of its own per model.

## Limitations

- **Long or flat only,** two symbols, hourly bars, a flat 0.1% fee. No shorting, leverage, slippage or funding costs.
- **The prompts changed during the official run** (`v10` for the first 291 attempts, `v12` after). Both are in one N, which is the conservative choice, but it is one run made with two prompt versions.
- **N counts correlated strategies as independent.** Many attempts are near-copies, so the true number of independent tries is lower and the luck bar is set too high. That makes the deflated Sharpe stricter than it needs to be.
- **The deflated and probabilistic Sharpe ratios assume independent returns.** Volatility clustering in crypto makes the real uncertainty somewhat larger, which cuts the other way.
- **The look-ahead test has a blind spot.** A leak confined to bars before the first cut, or to a stretch no cut lands in, is not seen. On planted cheats it caught 14 of 14 on real data; one cheat ("long while below the all-time high of the dataset") was missed in 6 of 30 random datasets, all with the high in the first 2% of bars.
- **Strategies can see real dates and price levels.** A rule keyed to remembered history could pass discovery and validation. It cannot help on the holdout, which is after the model's cutoff.
- **The time-shift null is harsh on calendar rules.** Shifting positions breaks a Monday rule's alignment with Mondays, so the null pass rate may understate how often a real but uninformative calendar rule passes.
- **The holdout is short.** Over 1.67 years a Sharpe ratio has a standard error of roughly 0.75, so only a large effect could pass.

## Next

A forward test: once a month, download the new month and score the frozen shortlist, unchanged. It is the only test that is after everyone's knowledge, mine included. Survivor #4323 is the strategy to watch, at both the normal and double fee.

## References

- Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*.
- Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*.
- McLean, R. D. and Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*.
- Si, C., Yang, D. and Hashimoto, T. (2024). Can LLMs generate novel research ideas?
