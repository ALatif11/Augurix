# Pre-registration of the holdout test

Written 2026-11-20T09:32:40+00:00 from validation pass 1 of run `gen-v1`, at commit `7908973`. Committed and tagged `prereg-v1` before any holdout data is downloaded. Nothing below, and no code that computes it, changes after that tag.

## What happened before this point

- Official attempts that produced a backtest (the trial count N): 534
- Passed the discovery gate and were run on validation: 266
- Rejected on the longer data (look-ahead or crash): 0
- Best annualised Sharpe expected from luck alone at this N: 2.17
- Share individually significant on validation: 4.9% of real strategies, 1.3% of time-shifted copies with no timing information
- Shortlist rule: option B, at most 10, ranked by validation timing Sharpe

## Shortlist

| Attempt | SHA-256 of the strategy code | Validation timing Sharpe | PSR vs 0 | Deflated Sharpe |
|---|---|---|---|---|
| 4461 | `6af35a8199be0c3f431bbc8571cfdc75d6468d86d90f95db9a6d7db2af366ddb` | 1.82 | 0.998 | 0.283 |
| 4453 | `1889eec26db46f9e7f51f5820b060b12814356a9f5f126aa74c8dc559d348ee6` | 1.64 | 0.995 | 0.199 |
| 4377 | `bdcffda541d517a5083a35f5fd43e806472d749705f0b26766c2427718ade4ff` | 1.55 | 0.994 | 0.155 |
| 4311 | `6dd96991a7df7fe3b652820f4bb700f0efcec1d0329abbb2494158541503b08e` | 1.51 | 0.992 | 0.146 |
| 4351 | `64670e3a2b26508c262e35fea0293e45684261ed6ead09461972846c46fae923` | 1.42 | 0.986 | 0.125 |
| 4260 | `56b6743420e45c3492c3c0f40130da1521a7dbcb09e88698d26cc4120b20b8b5` | 1.32 | 0.969 | 0.116 |
| 4504 | `66ac856421e6491220aca935b366f49d2f486c7c7e3e4d3d4eac6b8a8805a65d` | 1.32 | 0.971 | 0.111 |
| 4350 | `3905f7c45926eab2d6ee96f39f5db3aebca56021f35304ed32ad9b9ca26cb728` | 1.31 | 0.969 | 0.110 |
| 4341 | `44c25dc20c0be534c1334609d31ea72e458bc968db5073fae93b2fe7baa864cd` | 1.30 | 0.975 | 0.094 |
| 4323 | `7d17289979ca3ddc76b0727201bf35e3a346398f519b6137adf0f2ad81be4fd4` | 1.25 | 0.964 | 0.093 |

## The test

- Data: the same symbols and bar size as discovery, from Binance bulk dumps. Scored bars start 2025-03-01 and run through the last complete month available when the data is downloaded. Earlier bars are indicator warm-up only.
- Scoring: `backtest.py` as of the commit above. Fee 0.001 per unit of position change. The score is the timing Sharpe.
- **Primary test:** the equal-weight portfolio of the 10 shortlisted strategies has a probabilistic Sharpe ratio against zero of at least 0.95 on the holdout. This is the project's one headline claim.
- **Secondary:** a strategy counts as a survivor if its own probabilistic Sharpe ratio against zero is at least 0.9950 (Bonferroni, 0.05 / 10).
- A strategy that crashes or fails the look-ahead test on the holdout data contributes zero returns to the portfolio and is not a survivor.
- Reported but not tests: results at double the fee, and a permutation p-value from time-shifted copies.
- The holdout is opened once, by `open_holdout.py`. Whatever it prints is the result.

## Next commands

```
git add PREREGISTRATION.md results/
git commit -m "Pre-register the holdout test"
git tag prereg-v1
python download_data.py --holdout --end <last complete month, e.g. 2026-11>
python open_holdout.py --run gen-v1
```
