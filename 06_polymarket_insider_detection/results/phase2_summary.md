# Phase 2c summary: market-level detector (independent news times, within-event test)

Markets scored (full universe): 14403; extra out-of-universe cases: 10. Markets with an independent news time: 4249 (override 1, scheduled_end 4248). Level D: 4070; D_last24: 4070; max_jump: 12891; resolving jump detected: 2916; flow-before-move score (new window): 1649 (old window: 1523); within-event score: 1684; combined score (>= 2 components): 5163.

All percentiles, ranks and flag cut-offs below use the universe pool only; extra cases get the percentile / rank / flag they would have had against that pool.

Within-event status: within-event component present for 1684 of 4249 independent markets; sibling data fetched for 1201 of 1201 independent-market events.

## Run performance (this run)

Wall time per stage (s): inputs 1, series 7, baseline 122, score 1, signals 57, jumps 19, flow 82, combine 1, write_scores 0, initiators 8, plots 19, total 317. Peak working set: 6.6 GB. Flagged markets: 78; figures are drawn for at most 60 of them plus the labelled cases (never one per market).

## Level D (independent-news-time markets only)

Baseline scheme for every score below: **timesplit**. Independent markets in early windows without enough earlier training markets are UNSCORABLE (D null, reason `no_prior_training_window`), see the next section.

| quantile | D |
|---|---|
| 0.00 | 0.0000 |
| 0.25 | 0.0017 |
| 0.50 | 0.0050 |
| 0.75 | 0.0224 |
| 0.90 | 0.0739 |
| 0.95 | 0.1327 |
| 0.99 | 0.3349 |
| 1.00 | 0.8209 |

Share with D == 0: 0.3%. New top-1% threshold (99th percentile of D over 4070 independent markets): 0.3349; markets at or above it: 41.

| info_type | n | median_D | max_D |
|---|---|---|---|
| scheduled_reveal | 4070 | 0.0050 | 0.8209 |

Histogram: `figures/D_histogram.png`

- late-window D (last 24 h): n=4070, median 0.0136, p90 0.1279, p99 0.5181, max 0.8302.
- max_jump: n=12891, median 0.0788, p90 0.3746, p99 0.7802, max 0.9970.

## Baseline scheme: time split vs cross-fit

Time split (default): expanding calendar-month windows over the independent news time; window W is scored by quantile models fitted on all independent markets whose news time is strictly before W starts, so no market and no later information sets its own baseline. A window needs >= 100 earlier training markets and >= 500 thinned training rows, otherwise its markets are unscorable (reported, not scored in-sample). Windows, minimums and the model spec were fixed before any labelled-case rank was inspected. Cross-fit (5 folds by market, a market's model may be fitted on LATER markets) is kept for comparison as `D_crossfit`.

Independent markets: 4249. With a time-split D: 4070. With a cross-fit D: 4221. Cross-fit D but no time-split D (unscorable early window): 151.

| window | markets | train_markets | train_rows | scorable |
|---|---|---|---|---|
| 2025-01 | 155 | 0 | 0 | NO (unscorable) |
| 2025-02 | 187 | 155 | 6490 | yes |
| 2025-03 | 192 | 342 | 15356 | yes |
| 2025-04 | 83 | 534 | 24818 | yes |
| 2025-05 | 254 | 617 | 29318 | yes |
| 2025-06 | 3 | 871 | 42509 | yes |
| 2025-07 | 91 | 874 | 42677 | yes |
| 2025-08 | 56 | 965 | 47549 | yes |
| 2025-09 | 209 | 1021 | 50302 | yes |
| 2025-10 | 186 | 1230 | 61177 | yes |
| 2025-11 | 158 | 1416 | 71490 | yes |
| 2026-01 | 258 | 1574 | 80189 | yes |
| 2026-02 | 296 | 1832 | 94637 | yes |
| 2026-03 | 177 | 2128 | 111125 | yes |
| 2026-04 | 151 | 2305 | 121037 | yes |
| 2026-05 | 393 | 2456 | 129493 | yes |
| 2026-06 | 129 | 2849 | 151463 | yes |
| 2026-07 | 104 | 2978 | 158687 | yes |
| 2026-08 | 247 | 3082 | 164486 | yes |
| 2026-09 | 250 | 3329 | 178243 | yes |
| 2026-10 | 324 | 3579 | 192243 | yes |
| 2026-11 | 343 | 3903 | 210387 | yes |

Spearman rank correlation of D_timesplit vs D_crossfit over the 4070 markets with both: 0.974.
Top-1% D thresholds (99th percentile of the scheme's own pool): time split 0.3349 (n=4070), cross-fit 0.3347 (n=4221).
Markets at or above the own top-1% D threshold: time split 41, cross-fit 43, in both 38.

Drop out of the level-D top 1% under the time split (5); D_timesplit is NaN where the market is unscorable:

| question | info_type | D_timesplit | D_crossfit |
|---|---|---|---|
| Will NVIDIA be the largest company in the world by market cap on January 31 | scheduled_reveal | NaN | 0.5313 |
| Will Michelle Obama attend presidential inauguration? | scheduled_reveal | NaN | 0.5233 |
| Will Apple be the largest company in the world by market cap on January 31? | scheduled_reveal | NaN | 0.4887 |
| Will Nancy Pelosi attend presidential inauguration? | scheduled_reveal | NaN | 0.3975 |
| Will Jair Bolsonaro attend presidential inauguration? | scheduled_reveal | NaN | 0.3632 |

Enter the level-D top 1% under the time split (3):

| question | info_type | D_timesplit | D_crossfit |
|---|---|---|---|
| Will "Jurassic World" Opening Weekend Box Office be between $139-150m? | scheduled_reveal | 0.3478 | 0.3263 |
| Will 'Fight or Flight' gross between $1.2-1.5m opening weekend? | scheduled_reveal | 0.3468 | 0.3324 |
| Will "Good Fortune" Opening Weekend Box Office be less than 8m? | scheduled_reveal | 0.3370 | 0.3297 |

Held-out check on the labelled cases (rank 1 = highest D; the scheme was fixed before these ranks were looked at; extras are ranked against the universe pool):

| case | market | type | indep | D_timesplit | rank_timesplit | D_crossfit | rank_crossfit | reason |
|---|---|---|---|---|---|---|---|---|
| maduro_removal_jan_2026 | Maduro out by January 31, 2026? | extra | True | 0.0000 | 4028/4070 | 0.0000 | 4138/4221 |  |
| nobel_peace_2025 | Will María Corina Machado win the Nobel Peace Prize in  | universe | True | 0.0274 | 894/4070 | 0.0242 | 994/4221 |  |

## Resolving-jump detection and flow before the move

Jump detection over universe markets: no_jump (4042), jump_at_start (3418), clear jump (2916), reverted (2143), pre_level_high (1294), short_history (399), gradual_or_multi_step (191).
Jump size: median 0.475; hours from jump start to the market's best news time: median 9.2, p10 -13.1, p90 1542.2 (independent news times only: 7.4).
Flow-before-move among markets with a detected jump: computed (1148), short_history (994), wallet_coverage_low (331), thin_window (213), fresh_short_history (111), hhi_short_history; fresh_short_history (49), ofi_short_history; hhi_short_history; fresh_short_history (49), hhi_short_history; wallet_coverage_low (10), ofi_short_history; hhi_short_history; wallet_coverage_low (8), ofi_short_history; wallet_coverage_low (3).
fresh-wallet share available (>= 50% winner-side USD with known first_seen) for 1148 of 1856 markets with a window.
Phase 2c window: [jump_start - 24 h, min(half-jump point, jump_start + 2 h)]; the part after the jump start lasts median 1.38 h (p90 2.00 h) over 2916 markets, so it contains the trades that caused the first half of the jump (the 2b window ended at jump_start and missed them).

The earlier history is unchanged in kind (24 h blocks before the window start). Flow and volume measured here therefore include the jump-initiating trades; no threshold was tuned on a labelled case.

## Flagged markets (top 1% of the Fisher combination, universe pool)

Rule: ceil(1% of 5163 scored markets) = 52 highest `fisher_score` (ties included); a market is also listed if it is in the rank-mean top 1%. IsolationForest is a cross-check only.

| rank | question | info_type | indep | D | D_last24 | max_jump | flow | within | we_diff24 | jump_start | fisher_p | iso | driver |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Will 'The Seed of the Sacred Fig' win Best Foreign Language Film in th | scheduled_reveal | True | 0.8209 | 0.8267 | 0.910 | NaN | NaN | NaN | none | 8.7e-08 | False | D |
| 2 | Will Xiaomi have the best Chinese AI model at the end of September 202 | scheduled_reveal | True | 0.5528 | 0.8302 | 0.785 | 0.37 | 0.84 | 1.54 | 2026-09-26 01:32 | 2.3e-06 | True | D_last24 |
| 3 | Will Google be the third-best Code Arena / WebDev AI lab at the end of | scheduled_reveal | True | 0.3369 | 0.7874 | 0.818 | 0.80 | 0.79 | 0.70 | 2026-10-28 17:06 | 2.5e-06 | True | D_last24 |
| 4 | Will Pfizer (PFE) beat quarterly earnings? | scheduled_reveal | True | 0.6269 | 0.6613 | 0.290 | NaN | 0.98 | 4.30 | none | 3.6e-06 | False | D |
| 5 | Will Alibaba have the best Chinese AI model at the end of September 20 | scheduled_reveal | True | 0.5158 | 0.7588 | 0.830 | 0.47 | NaN | NaN | 2026-09-26 01:32 | 1.1e-05 | False | D |
| 6 | Will Xiaomi be the third-best AI lab at the end of September 2026? | scheduled_reveal | True | 0.4637 | 0.7518 | 0.713 | NaN | 0.75 | 3.11 | 2026-09-26 01:32 | 1.2e-05 | False | D_last24 |
| 7 | Will claude-opus-4-6-thinking be the best AI model on August 1, 2026? | scheduled_reveal | True | 0.5684 | 0.6402 | 0.645 | 0.59 | NaN | NaN | 2026-07-27 04:29 | 1.5e-05 | False | D |
| 8 | Will Meta be the third-best Code Arena / WebDev AI lab at the end of O | scheduled_reveal | True | 0.2717 | 0.6100 | 0.879 | 0.78 | NaN | NaN | 2026-10-28 17:06 | 1.8e-05 | True | max_jump |
| 9 | Will MSC Industrial Direct (MSM) beat quarterly earnings? | scheduled_reveal | True | 0.1386 | 0.1055 | 0.890 | 0.73 | 1.00 | 5.60 | 2026-04-01 10:30 | 1.8e-05 | True | within_event |
| 10 | Will any other model be the best AI model on August 1, 2026? | scheduled_reveal | True | 0.5670 | 0.6415 | 0.701 | 0.68 | 0.42 | 0.10 | 2026-07-27 04:29 | 2.5e-05 | True | D |
| 11 | Will Alibaba be the third-best Math AI lab at the end of October 2026? | scheduled_reveal | True | 0.3604 | 0.7535 | 0.465 | 0.72 | 0.84 | 5.60 | 2026-10-28 16:38 | 2.5e-05 | True | D_last24 |
| 12 | Will OpenAI be the second-best AI Agent Lab at the end of October 2026 | scheduled_reveal | True | 0.3321 | 0.7738 | 0.815 | 0.59 | NaN | NaN | 2026-10-28 17:06 | 2.9e-05 | False | D_last24 |
| 13 | Will Google be the second-best AI Agent Lab at the end of October 2026 | scheduled_reveal | True | 0.3340 | 0.7863 | 0.788 | NaN | 0.10 | -0.19 | 2026-10-28 17:06 | 5.9e-05 | True | D_last24 |
| 14 | Will 'Shadow Force' gross less than $1.2m opening weekend? | scheduled_reveal | True | 0.3898 | 0.4499 | 0.780 | NaN | NaN | NaN | none | 7.3e-05 | False | D |
| 15 | Will 'Minecraft' gross more than 120m on opening weekend? | scheduled_reveal | True | 0.5062 | 0.7755 | 0.277 | NaN | 0.30 | 0.00 | none | 1.4e-04 | False | D_last24 |
| 16 | "The Naked Gun" Rotten Tomatoes score is 90% or higher? | scheduled_reveal | True | 0.2881 | 0.6569 | 0.615 | NaN | 0.73 | 3.15 | none | 1.4e-04 | False | D_last24 |
| 17 | Will Meta be the third-best AI lab at the end of September 2026? | scheduled_reveal | True | 0.4074 | 0.6732 | 0.547 | 0.45 | NaN | NaN | 2026-09-26 01:32 | 1.9e-04 | False | D_last24 |
| 18 | Will OpenAI be the third-best AI Agent Lab at the end of October 2026? | scheduled_reveal | True | 0.2914 | 0.7105 | 0.460 | 0.72 | 0.56 | 0.14 | 2026-10-28 17:06 | 2.0e-04 | True | D_last24 |
| 19 | Will OpenAI be the second-best Code Arena / WebDev AI lab at the end o | scheduled_reveal | True | 0.4960 | 0.5125 | 0.458 | 0.43 | 0.85 | 1.29 | 2026-09-23 17:06 | 2.1e-04 | False | D |
| 20 | Will CAVA say "Hyphen" during earnings call? | scheduled_reveal | True | 0.1935 | 0.4002 | 0.977 | NaN | 0.46 | 0.04 | none | 2.2e-04 | False | max_jump |
| 21 | Will Anthropic be the second-best Code Arena / WebDev AI lab at the en | scheduled_reveal | True | 0.4633 | 0.4710 | 0.546 | 0.51 | NaN | NaN | 2026-09-24 02:27 | 2.3e-04 | False | D |
| 22 | Will "The Naked Gun" Opening Weekend Box Office be between $23m and $2 | scheduled_reveal | True | 0.4302 | 0.4648 | 0.370 | NaN | NaN | NaN | none | 2.6e-04 | False | D |
| 23 | Will Hims say "Lab Testing" during earnings call? | scheduled_reveal | True | 0.2719 | 0.3588 | 0.428 | NaN | 0.95 | 1.97 | none | 2.7e-04 | False | D |
| 24 | Will 'The Monkey' gross between 12-15m on its opening weekend? | scheduled_reveal | True | 0.4583 | 0.6752 | 0.230 | NaN | 0.69 | 3.31 | 2025-02-21 06:19 | 2.9e-04 | False | D |
| 25 | "Nobody 2" Rotten Tomatoes score is between 80–89%? | scheduled_reveal | True | 0.1878 | 0.3701 | 0.878 | NaN | NaN | NaN | none | 3.0e-04 | False | max_jump |
| 26 | Will "The Odyssey" score at least 95 on the Rotten Tomatoes Tomatomete | scheduled_reveal | True | 0.3120 | 0.5452 | 0.290 | NaN | 0.92 | 1.30 | none | 3.1e-04 | False | D_last24 |
| 27 | Will 'Fight or Flight' gross less than $0.9m opening weekend? | scheduled_reveal | True | 0.4096 | 0.4504 | 0.390 | NaN | NaN | NaN | none | 3.4e-04 | False | D |
| 28 | Will "The Naked Gun" Opening Weekend Box Office be more than $26m? | scheduled_reveal | True | 0.4767 | 0.4785 | 0.235 | NaN | NaN | NaN | none | 3.5e-04 | False | D |
| 29 | Will Google be the third-best AI Agent Lab at the end of October 2026? | scheduled_reveal | True | 0.2363 | 0.5584 | 0.611 | 0.61 | NaN | NaN | 2026-10-28 17:06 | 3.7e-04 | False | D_last24 |
| 30 | Will the Associated Press first call the 2025 New Jersey governor elec | scheduled_reveal | True | 0.2544 | 0.3464 | 0.380 | NaN | 0.96 | 2.02 | none | 3.8e-04 | False | within_event |
| 31 | Will Trump’s approval rating be <42.5% on October 3? | scheduled_reveal | True | 0.3822 | 0.4159 | 0.454 | NaN | NaN | NaN | none | 3.9e-04 | False | D |
| 32 | Will "Jurassic World" Opening Weekend Box Office be less than $117m? | scheduled_reveal | True | 0.4159 | 0.4289 | 0.363 | NaN | NaN | NaN | none | 4.2e-04 | False | D |
| 33 | Will Trump’s approval rating be 45.0–45.4% on September 5? | scheduled_reveal | True | 0.4185 | 0.4206 | 0.330 | NaN | NaN | NaN | none | 4.5e-04 | False | D |
| 34 | Will "The Bad Guys 2" Opening Weekend Box Office be less than $16m? | scheduled_reveal | True | 0.4112 | 0.4150 | 0.365 | NaN | NaN | NaN | none | 4.7e-04 | False | D |
| 35 | Will Trump’s approval rating be ≥46.5% on July 4? | scheduled_reveal | True | 0.4100 | 0.4386 | 0.350 | NaN | NaN | NaN | none | 4.7e-04 | False | D |
| 36 | Will "Melania" score at least 25 on the Rotten Tomatoes Tomatometer? | scheduled_reveal | True | 0.1500 | 0.3881 | 0.431 | NaN | 0.96 | 2.61 | none | 4.7e-04 | False | within_event |
| 37 | Will Harley-Davidson (HOG) beat quarterly earnings? | scheduled_reveal | True | 0.0035 | 0.0248 | 0.833 | 0.59 | 1.00 | 5.94 | 2025-11-04 11:51 | 4.9e-04 | False | within_event |
| 38 | Will 'The Monkey' gross more than 15m on its opening weekend? | scheduled_reveal | True | 0.4048 | 0.5480 | 0.239 | NaN | NaN | NaN | none | 5.4e-04 | False | D |
| 39 | Will the Ornn H100 Index be between $2.50 and $2.75 on September 30, 2 | scheduled_reveal | True | 0.0768 | 0.2185 | 0.908 | NaN | 0.89 | 7.80 | none | 5.7e-04 | False | max_jump |
| 40 | Will Trump’s approval rating be ≥44.5% on October 3? | scheduled_reveal | True | 0.3471 | 0.4239 | 0.425 | NaN | NaN | NaN | 2025-09-27 16:36 | 5.7e-04 | False | D |
| 41 | Will 'Dune: Part Two' win Best Cinematography in the 2025 Critics' Cho | scheduled_reveal | True | 0.3719 | 0.3725 | 0.440 | NaN | NaN | NaN | none | 5.8e-04 | False | D |
| 42 | Will DraftKings say "Retention" during earnings call? | scheduled_reveal | True | 0.0348 | 0.1741 | 0.485 | NaN | 0.99 | 4.82 | 2025-11-07 10:19 | 6.0e-04 | False | within_event |
| 43 | CZ mindshare all time high by March 31? | deadline | False | NaN | NaN | 0.997 | 0.63 | NaN | NaN | 2026-04-04 02:16 | 6.2e-04 | False | max_jump |
| 44 | Will "The Devil Wears Prada 2" Opening Weekend Box Office be between 7 | scheduled_reveal | True | 0.3733 | 0.7456 | 0.201 | 0.54 | 0.74 | 2.54 | 2026-04-30 05:00 | 6.7e-04 | False | D_last24 |
| 45 | Will PayPal say "Google" during earnings call? | scheduled_reveal | True | 0.0662 | 0.3225 | 0.620 | 0.84 | 0.69 | 6.07 | 2026-02-03 13:58 | 6.8e-04 | True | max_jump |
| 46 | Will the Associated Press first call the 2025 Virginia governor electi | scheduled_reveal | True | 0.3092 | 0.3886 | 0.475 | NaN | NaN | NaN | none | 6.8e-04 | False | D |
| 47 | Will 'Last Breath' between 3-6m on its opening weekend? | scheduled_reveal | True | 0.3029 | 0.5681 | 0.338 | NaN | NaN | NaN | 2025-02-27 22:05 | 6.9e-04 | False | D_last24 |
| 48 | Will "The Bad Guys 2" Opening Weekend Box Office be more than $25m? | scheduled_reveal | True | 0.3938 | 0.4859 | 0.245 | NaN | NaN | NaN | none | 7.0e-04 | False | D |
| 49 | Will "A Minecraft Movie" Rotten Tomatoes score be between 40 and 50? | scheduled_reveal | True | 0.2824 | 0.5837 | 0.310 | 0.61 | 0.80 | 3.71 | 2025-04-04 15:14 | 8.5e-04 | False | D_last24 |
| 50 | Will "PAW Patrol: The Dino Movie" Opening Weekend Box Office be betwee | scheduled_reveal | True | 0.2607 | 0.5415 | 0.366 | NaN | 0.76 | 1.32 | none | 9.5e-04 | False | D_last24 |
| 51 | Fear & Greed Index says "Greed" on Friday? | scheduled_reveal | True | 0.2529 | 0.3748 | 0.320 | NaN | 0.91 | 2.06 | none | 9.8e-04 | False | D |
| 52 | Will Trump’s approval rating be between 42.5% and 42.9% on October 3? | scheduled_reveal | True | 0.3187 | 0.3764 | 0.410 | NaN | NaN | NaN | 2025-09-27 23:47 | 9.9e-04 | False | D |
| 53 | Will Jeffrey Randall Allen (Player 831) win the Beast Games? | scheduled_reveal | True | 0.1829 | 0.2636 | 0.410 | NaN | 0.94 | 3.91 | none | 1.0e-03 | False | within_event |
| 64 | Will Trump say "IQ" during the America Business Forum on November 5? | unscheduled_event | False | NaN | NaN | 0.610 | 0.96 | NaN | NaN | 2025-11-05 19:10 | 1.4e-03 | False | flow_score |
| 74 | Will Trump’s approval rating be 44.5–44.9% on September 5? | scheduled_reveal | True | 0.2671 | 0.2929 | 0.440 | NaN | NaN | NaN | none | 1.6e-03 | False | D |
| 78 | Will "Hamilton" Opening Weekend Box Office be less than $8.8m? | scheduled_reveal | True | 0.2585 | 0.4113 | 0.371 | NaN | NaN | NaN | none | 1.7e-03 | False | D |
| 80 | Will Rachel Maddow be named in newly released Epstein files by March 3 | deadline | False | NaN | NaN | 0.792 | 0.91 | NaN | NaN | 2026-01-30 16:21 | 1.7e-03 | False | flow_score |
| 81 | Will "The Naked Gun" Opening Weekend Box Office be between $20m and $2 | scheduled_reveal | True | 0.3232 | 0.3965 | 0.285 | NaN | NaN | NaN | none | 1.8e-03 | False | D |
| 82 | Will "Hamilton" Opening Weekend Box Office be between $8.8m and $9.5m? | scheduled_reveal | True | 0.2306 | 0.3946 | 0.404 | NaN | NaN | NaN | none | 1.8e-03 | False | D_last24 |
| 92 | Will "Hamilton" Opening Weekend Box Office be more than $11m? | scheduled_reveal | True | 0.2529 | 0.3975 | 0.335 | NaN | NaN | NaN | none | 2.2e-03 | False | D |
| 99 | Will Trump’s approval rating be <44.5% on July 4? | scheduled_reveal | True | 0.3043 | 0.2627 | 0.360 | NaN | NaN | NaN | none | 2.5e-03 | False | D |
| 102 | Will the Indian Union Muslim League (IUML) win 22 or more seats in the | scheduled_reveal | False | NaN | NaN | 0.717 | 0.80 | NaN | NaN | 2026-05-04 05:10 | 2.6e-03 | False | max_jump |
| 108 | Will "Mission: Impossible – The Final Reckoning" Rotten Tomatoes score | scheduled_reveal | True | 0.1545 | 0.4458 | 0.353 | NaN | NaN | NaN | none | 3.0e-03 | False | D_last24 |
| 111 | Will Quentin Tarantino be named in newly released Epstein files by Mar | deadline | False | NaN | NaN | 0.855 | 0.79 | NaN | NaN | 2026-01-30 16:23 | 3.2e-03 | False | max_jump |
| 128 | Will the Bank of Korea hold at the August 2026 meeting? | scheduled_reveal | True | 0.2492 | 0.2207 | 0.380 | NaN | NaN | NaN | none | 3.9e-03 | False | D |
| 137 | Will Abdul El-Sayed win the Michigan Senate Democratic primary by less | scheduled_reveal | False | NaN | NaN | 0.416 | 0.87 | NaN | NaN | 2026-08-05 01:11 | 4.8e-03 | False | flow_score |
| 145 | Will Trump say "Coal" during the America Business Forum on November 5? | unscheduled_event | False | NaN | NaN | 0.448 | 0.94 | NaN | NaN | 2025-11-05 18:33 | 5.4e-03 | False | flow_score |
| 146 | Israel enacts military exemption for Yeshiva students by Sep 30? | deadline | False | NaN | NaN | 0.616 | 0.89 | NaN | NaN | 2026-07-14 18:47 | 5.4e-03 | False | flow_score |
| 155 | Will JP Morgan (JPM) Q2 investment banking fees be above $3.15B? | unscheduled_event | False | NaN | NaN | 0.485 | 0.84 | NaN | NaN | 2026-07-14 10:13 | 6.2e-03 | False | flow_score |
| 176 | Will Al Gore be named in newly released Epstein files by March 31? | deadline | False | NaN | NaN | 0.670 | 0.80 | NaN | NaN | 2026-01-30 16:23 | 7.8e-03 | False | max_jump |
| 180 | ChatGPT Outage by April 3? | deadline | False | NaN | NaN | 0.800 | 0.74 | NaN | NaN | 2026-04-02 14:20 | 8.0e-03 | False | max_jump |
| 183 | Pooh Shiesty charged by May 31? | deadline | False | NaN | NaN | 0.700 | 0.79 | NaN | NaN | 2026-05-16 17:05 | 8.3e-03 | False | max_jump |
| 193 | Will the People's Party for Freedom and Democracy win the third most s | scheduled_reveal | False | NaN | NaN | 0.631 | 0.70 | NaN | NaN | 2025-10-29 17:49 | 9.3e-03 | False | max_jump |
| 198 | Will Trump say "Kamala" this week? | deadline | False | NaN | NaN | 0.590 | 0.83 | NaN | NaN | 2025-05-07 20:27 | 9.6e-03 | False | flow_score |
| 235 | Will John Williams advance from the AK-AL primary election? | scheduled_reveal | False | NaN | NaN | 0.395 | 0.79 | NaN | NaN | 2026-09-01 01:05 | 1.3e-02 | False | flow_score |
| 240 | Will Trump say "mandate" 3+ times during his inauguration speech? | scheduled_reveal | False | NaN | NaN | 0.500 | 0.72 | NaN | NaN | 2025-01-20 18:06 | 1.3e-02 | False | max_jump |
| 263 | Will Trump meet with Emmanuel Macron in October? | deadline | False | NaN | NaN | 0.621 | 0.76 | NaN | NaN | 2025-10-11 12:00 | 1.6e-02 | False | max_jump |
| 281 | Will the Green Party win at least 600 council seat elections in the 20 | scheduled_reveal | False | NaN | NaN | 0.415 | 0.74 | NaN | NaN | 2026-05-08 01:32 | 1.8e-02 | False | max_jump |

## Labelled cases: rank on every score

Labelled markets in the table: 75. Rank 1 = most suspicious; denominators are the number of universe markets with that score. Extra (out-of-universe) cases are ranked against the universe pool, never inside it. NaN = measure not available (see the reason columns in market_scores.csv).

Context / candidate cases are capped at 10 markets per case here (39 more are in `market_scores.csv`).

| case | label | market | type | indep | rank_D | rank_D24 | rank_maxjump | rank_flow_old | rank_flow_new | rank_within | rank_combined | jump |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| maduro_removal_jan_2026 | insider_suspected | Maduro out by January 31, 2026? | extra | True | 4028/4070 | 3770/4070 | 9689/12891 | 34/1523 | 163/1649 | NaN | 1572/5163 | yes |
| nobel_peace_2025 | insider_suspected | Will María Corina Machado win the Nobel Peace Prize  | universe | True | 894/4070 | 253/4070 | 929/12891 | 775/1523 | 954/1649 | 9/1684 | 105/5163 | yes |
| axiom_expose_feb_2026 | case_candidate, case_context | Will ZachXBT release his investigation on February 2 | universe | False | NaN | NaN | 33/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_candidate, case_context | ZachXBT investigation post over 14M views on day one | universe | False | NaN | NaN | 2140/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_candidate, case_context | ZachXBT investigation post over 8M views on day one? | universe | False | NaN | NaN | 3468/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_candidate, case_context | ZachXBT investigation post over 25M views on day one | universe | False | NaN | NaN | 10839/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_candidate, case_context | ZachXBT investigation post over 10M views on day one | universe | False | NaN | NaN | 10839/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_candidate, case_context | ZachXBT investigation post over 30M views on day one | extra | False | NaN | NaN | 10959/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | No release by February 27 | universe | False | NaN | NaN | 762/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | Will ZachXBT release his investigation on February 2 | universe | False | NaN | NaN | 1596/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | ZachXBT investigation post over 12M views on day one | universe | False | NaN | NaN | 1635/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | ZachXBT investigation post over 6M views on day one? | universe | False | NaN | NaN | 5226/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | ZachXBT investigation post over 4M views on day one? | universe | False | NaN | NaN | 10124/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | ZachXBT investigation post over 18M views on day one | universe | False | NaN | NaN | 12279/12891 | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | Will ZachXBT release his investigation on February 2 | universe | False | NaN | NaN | NaN | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | Will ZachXBT release his investigation on February 2 | universe | False | NaN | NaN | NaN | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | Will ZachXBT release his investigation on February 2 | universe | False | NaN | NaN | NaN | NaN | NaN | NaN | NaN | short_history |
| axiom_expose_feb_2026 | case_context | Will ZachXBT release his investigation on February 2 | universe | False | NaN | NaN | NaN | NaN | NaN | NaN | NaN | short_history |
| bubblemaps_military_cluster | case_candidate | Iran strike on US military by February 28? | extra | False | NaN | NaN | 887/12891 | 628/1523 | 885/1649 | NaN | 1262/5163 | yes |
| bubblemaps_military_cluster | case_candidate | US strikes Iran by February 28, 2026? | extra | False | NaN | NaN | 5515/12891 | 249/1523 | 348/1649 | NaN | 1775/5163 | yes |
| bubblemaps_military_cluster | case_candidate | Iran Strike on Israel by February 28? | extra | False | NaN | NaN | 52/12891 | NaN | NaN | NaN | NaN | reverted |
| bubblemaps_military_cluster | case_candidate | Israel strikes Iran by February 28, 2026? | extra | False | NaN | NaN | 5997/12891 | NaN | NaN | NaN | NaN | reverted |
| bubblemaps_military_cluster | case_candidate, case_context | Will US or Israel strike Iran on March 1, 2026? | extra | False | NaN | NaN | 3247/12891 | NaN | NaN | NaN | NaN | short_history |
| bubblemaps_military_cluster | case_candidate, case_context | Will US or Israel strike Iran on March 2, 2026? | extra | False | NaN | NaN | 3410/12891 | NaN | NaN | NaN | NaN | no_jump |
| bubblemaps_military_cluster | case_candidate, case_context | Will US or Israel strike Iran on March 3, 2026? | extra | False | NaN | NaN | 4314/12891 | NaN | NaN | NaN | NaN | no_jump |
| bubblemaps_military_cluster | case_candidate, case_context | Will US or Israel strike Iran on March 4, 2026? | extra | False | NaN | NaN | 5541/12891 | NaN | NaN | NaN | NaN | no_jump |
| google_year_in_search_2025 | case_context | Will Bianca Censori rank in Google’s Top 5 Most Sear | universe | False | NaN | NaN | 288/12891 | 1060/1523 | 1380/1649 | NaN | 557/5163 | yes |
| google_year_in_search_2025 | case_context | Will Luigi Mangione rank in Google’s Top 5 Most Sear | universe | False | NaN | NaN | 7229/12891 | 724/1523 | 923/1649 | NaN | 2584/5163 | yes |
| google_year_in_search_2025 | case_context | Will d4vd rank in Google’s Top 5 Most Searched Peopl | universe | False | NaN | NaN | 6502/12891 | 1024/1523 | 1183/1649 | NaN | 2769/5163 | yes |
| google_year_in_search_2025 | case_context | Will Kendrick Lamar rank in Google’s Top 5 Most Sear | universe | False | NaN | NaN | 7733/12891 | 863/1523 | 1032/1649 | NaN | 2880/5163 | yes |
| google_year_in_search_2025 | case_context | Will Tyler Robinson rank in Google’s Top 5 Most Sear | universe | False | NaN | NaN | 8370/12891 | 1445/1523 | 1572/1649 | NaN | 3856/5163 | yes |
| google_year_in_search_2025 | case_context | Will Sam Altman rank in Google’s Top 5 Most Searched | universe | False | NaN | NaN | 4146/12891 | NaN | NaN | NaN | NaN | no_jump |
| google_year_in_search_2025 | case_context | Will Justin Trudeau rank in Google’s Top 5 Most Sear | universe | False | NaN | NaN | 4337/12891 | NaN | NaN | NaN | NaN | no_jump |
| google_year_in_search_2025 | case_context | Will María Corina Machado rank in Google’s Top 5 Mos | universe | False | NaN | NaN | 5818/12891 | NaN | NaN | NaN | NaN | pre_level_high |
| google_year_in_search_2025 | case_context | Will Katy Perry rank in Google’s Top 5 Most Searched | universe | False | NaN | NaN | 6645/12891 | NaN | NaN | NaN | NaN | no_jump |
| google_year_in_search_2025 | case_context | Will Taylor Swift rank in Google’s Top 5 Most Search | universe | False | NaN | NaN | 8370/12891 | NaN | NaN | NaN | NaN | jump_at_start |

### Nobel Peace Prize 2025 (Machado)

- News time 2025-10-10 09:00:00 (override), independent=True. Level D 0.0274 (rank 894/4070), D_last24 0.1915 (rank 253/4070), max_jump 0.428 at 9 h before news (rank 929/12891), flow_score 0.48 (new window rank 954/1649; old window rank 775/1523), within-event 0.99 (rank 9/1684), combined rank 105/5163, main driver within_event.
- Peer band: share of the 168 hourly points with q above the peer m90: 0.0% (last 24 h: 0.0%); largest q - m90 = -0.272. Detected jump: start 2025-10-09 21:20:47, size 0.968, pre-level 0.008.
- Chart: `figures/case_NOBEL_will-mar-a-corina-machado-win-the-nobel-_0x14a3df.png`; the visual verdict (does q(t) clear the band?) must be read from the image together with these shares.

## Extra labelled cases (out of universe)

| case | mapping | market | taker_rows | tape_start | news_ts | news_source | jump_start | D | D_last24 | max_jump | flow | flow_old | within | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| maduro_removal_jan_2026 | unambiguous | Maduro out by January 31, 2026? | 7910 | 2025-12-22 12:29 | 2026-01-03 06:00 | independent: first public reports: blasts over Caracas ~06:00 UTC 2026-01-03 (AFP;  | 2026-01-03 03:03:53 | 0.000 | 0.000 | 0.024 | 0.70 | 0.80 | NaN | 1572 |
| axiom_expose_feb_2026 | candidate | Will ZachXBT release his investigation on February 26,  | 104 | 2026-02-26 01:03 | 2026-02-26 13:41 | price-based (no documented UTC minute for the ZachXBT post) | short_history | NaN | NaN | 0.930 | NaN | NaN | NaN | NaN |
| axiom_expose_feb_2026 | candidate | ZachXBT investigation post over 30M views on day one?- | 74 | 2026-02-26 01:02 | 2026-02-26 19:55 | price-based (no documented UTC minute for the ZachXBT post) | short_history | NaN | NaN | 0.010 | NaN | NaN | NaN | NaN |
| axiom_expose_feb_2026 | candidate | ZachXBT investigation post over 14M views on day one? | 116 | 2026-02-25 23:08 | 2026-02-26 19:44 | price-based (no documented UTC minute for the ZachXBT post) | short_history | NaN | NaN | 0.282 | NaN | NaN | NaN | NaN |
| axiom_expose_feb_2026 | candidate | ZachXBT investigation post over 10M views on day one? | 164 | 2026-02-25 23:11 | 2026-02-26 19:51 | price-based (no documented UTC minute for the ZachXBT post) | short_history | NaN | NaN | 0.010 | NaN | NaN | NaN | NaN |
| axiom_expose_feb_2026 | candidate | ZachXBT investigation post over 8M views on day one? | 118 | 2026-02-25 23:31 | 2026-02-26 20:02 | price-based (no documented UTC minute for the ZachXBT post) | short_history | NaN | NaN | 0.190 | NaN | NaN | NaN | NaN |
| axiom_expose_feb_2026 | candidate | ZachXBT investigation post over 25M views on day one? | 9 | 2026-02-26 01:48 | 2026-02-26 20:19 | price-based (no documented UTC minute for the ZachXBT post) | short_history | NaN | NaN | 0.010 | NaN | NaN | NaN | NaN |
| bubblemaps_military_cluster | candidate | Israel strikes Iran by February 28, 2026? | 8285 | 2026-02-16 09:20 | 2026-02-28 06:21 | price-based (no documented UTC minute for the 2026-02-28 strikes) | reverted | NaN | NaN | 0.090 | NaN | NaN | NaN | NaN |
| bubblemaps_military_cluster | candidate | Iran Strike on Israel by February 28? | 8918 | 2026-02-16 23:53 | 2026-02-28 20:41 | price-based (no documented UTC minute for the 2026-02-28 strikes) | reverted | NaN | NaN | 0.908 | NaN | NaN | NaN | NaN |
| bubblemaps_military_cluster | candidate | Iran strike on US military by February 28? | 1936 | 2026-02-17 11:12 | 2026-03-01 03:31 | price-based (no documented UTC minute for the 2026-02-28 strikes) | 2026-02-28 06:17:53 | NaN | NaN | 0.433 | 0.49 | 0.55 | NaN | 1262 |
| bubblemaps_military_cluster | candidate | Will US or Israel strike Iran on March 1, 2026? | 2927 | 2026-02-28 14:00 | 2026-03-01 00:07 | price-based (no documented UTC minute for the 2026-02-28 strikes) | short_history | NaN | NaN | 0.204 | NaN | NaN | NaN | NaN |
| bubblemaps_military_cluster | candidate | Will US or Israel strike Iran on March 2, 2026? | 3193 | 2026-02-28 14:11 | 2026-03-01 04:24 | price-based (no documented UTC minute for the 2026-02-28 strikes) | no_jump | NaN | NaN | 0.193 | NaN | NaN | NaN | NaN |
| bubblemaps_military_cluster | candidate | Will US or Israel strike Iran on March 3, 2026? | 3085 | 2026-02-28 14:16 | 2026-03-01 11:00 | price-based (no documented UTC minute for the 2026-02-28 strikes) | no_jump | NaN | NaN | 0.150 | NaN | NaN | NaN | NaN |
| bubblemaps_military_cluster | candidate | Will US or Israel strike Iran on March 4, 2026? | 2544 | 2026-02-28 14:54 | 2026-03-01 20:11 | price-based (no documented UTC minute for the 2026-02-28 strikes) | no_jump | NaN | NaN | 0.100 | NaN | NaN | NaN | NaN |
| bubblemaps_military_cluster | candidate | US strikes Iran by February 28, 2026? | 58625 | 2026-02-16 09:31 | 2026-02-28 06:52 | price-based (no documented UTC minute for the 2026-02-28 strikes) | 2026-02-28 04:48:21 | NaN | NaN | 0.100 | 0.62 | 0.65 | NaN | 1775 |

`candidate` mapping: the press names accounts but not the market, so every candidate is scored and none is singled out. Tapes are capped to the 12 days before resolution (`/v2/trades` has no time filter for the market feed). Short-lived markets (created < 36 h before resolution) cannot get a resolving-jump or flow-before-move reading.

## Within-event pre-reveal test and placebo

within-event component present for 1684 of 4249 independent markets; sibling data fetched for 1201 of 1201 independent-market events.

Markets with a within-event statistic: 1670 (binary: 793, multi: 877).
Pooled placebo over all multi-outcome events with a comparable-loser set (`within_event.run()`; winner drawn at random within the event for the null):
- Final 24h: n_events = 809; obs_mean_diff = 0.8932; obs_median_diff = 0.2697; obs_share_above_q95 = 0.529; null_mean_diff = -0.01016; null_median_diff = 0; null_share_above_q95 = 0.165; p_mean_diff = 9.999e-05; p_median_diff = 9.999e-05; p_share_above_q95 = 9.999e-05; median_winner_pct = 0.75
- Final 12h: n_events = 809; obs_mean_diff = 0.7126; obs_median_diff = 0.1589; obs_share_above_q95 = 0.4845; null_mean_diff = -0.01368; null_median_diff = 0; null_share_above_q95 = 0.1611; p_mean_diff = 9.999e-05; p_median_diff = 9.999e-05; p_share_above_q95 = 9.999e-05; median_winner_pct = 0.75

Highest level-D markets and what the within-event test says about them (a market whose level D is high but whose winner is not an outlier among its event's outcomes is explained by event structure; NaN = no sibling outcomes, not a scheduled reveal or no price data):
| market | info_type | D | rank_D | group | diff_24h | pct_24h | within | rank_within |
|---|---|---|---|---|---|---|---|---|
| Will 'The Seed of the Sacred Fig' win Best Foreign Lang | scheduled_reveal | 0.8209 | 1 | multi | NaN | NaN | NaN | NaN |
| Will Pfizer (PFE) beat quarterly earnings? | scheduled_reveal | 0.6269 | 2 | binary | 4.30 | 0.97 | 0.98 | 19 |
| Will claude-opus-4-6-thinking be the best AI model on A | scheduled_reveal | 0.5684 | 3 | multi | NaN | NaN | NaN | NaN |
| Will any other model be the best AI model on August 1,  | scheduled_reveal | 0.5670 | 4 | multi | 0.10 | 0.88 | 0.42 | 1021 |
| Will Xiaomi have the best Chinese AI model at the end o | scheduled_reveal | 0.5528 | 5 | multi | 1.54 | 0.95 | 0.84 | 169 |
| Will Alibaba have the best Chinese AI model at the end  | scheduled_reveal | 0.5158 | 6 | multi | NaN | NaN | NaN | NaN |
| Will 'Minecraft' gross more than 120m on opening weeken | scheduled_reveal | 0.5062 | 7 | multi | 0.00 | 0.50 | 0.30 | 1275 |
| Will OpenAI be the second-best Code Arena / WebDev AI l | scheduled_reveal | 0.4960 | 8 | multi | 1.29 | 0.97 | 0.85 | 159 |
| Will "The Naked Gun" Opening Weekend Box Office be more | scheduled_reveal | 0.4767 | 9 | multi | NaN | NaN | NaN | NaN |
| Will Xiaomi be the third-best AI lab at the end of Sept | scheduled_reveal | 0.4637 | 10 | multi | 3.11 | 0.88 | 0.75 | 294 |
| Will Anthropic be the second-best Code Arena / WebDev A | scheduled_reveal | 0.4633 | 11 | multi | NaN | NaN | NaN | NaN |
| Will 'The Monkey' gross between 12-15m on its opening w | scheduled_reveal | 0.4583 | 12 | multi | 3.31 | 0.88 | 0.69 | 450 |

Box-office markets with level D: 97, of which 26 have a within-event reading (the rest have no usable sibling outcome path). Among the 2 in the level-D top 10, Will 'Minecraft' gross more th (D rank 7, within-event 0.30, rank 1275/1684), Will "The Naked Gun" Opening W (D rank 9, within-event NaN, rank NaN/1684). A within-event percentile near 1 would mean the winner stood out among its siblings; values around 0.5 mean it did not, i.e. the level-D flag is not confirmed by the within-event test.

## Jump-initiating trades (`data/private/jump_initiators.csv`, not in results/ because it holds wallet addresses)

Top winner-side taker wallets inside the flow window (24 h before the jump start to the half-jump point, at most 2 h in) for 2630 markets with a jump; labelled cases are listed with up to 5 wallets, others 3. `first_seen_src` shows coverage (wallets.parquet, else extra_wallets.parquet, else none); `only_mkt` = the wallet traded only this market in the taker store (a subset of Polymarket, so a lower bound on breadth).
Rank-1 initiator has a known first_seen for 2057 of 2630 markets; 47 of them traded only that market in the store.
Per-wallet rows (addresses, USD, first_seen) are in `data/private/jump_initiators.csv`; only aggregates are kept in this summary.
## Agreement between rankings

Over 5163 universe markets with all three scores. Spearman:

- fisher_score~rank_mean: 0.972
- fisher_score~iso_score: 0.263
- rank_mean~iso_score: 0.185

Flag counts: {'fisher': 52, 'rank_mean': 52, 'iso': 52}; overlap: {'fisher&rank_mean': 26, 'fisher&iso': 9, 'rank_mean&iso': 4}.

## Coverage and missing data

Level D is NaN for price_based_news_time (8801), no_price_data (1339), no_prior_training_window (155), too_few_hours (21), no_news_ts (17).
- D: available for 4070 of 14403
- D_last24: available for 4070 of 14403
- max_jump: available for 12891 of 14403
- flow_score: available for 1649 of 14403
- within_event: available for 1684 of 14403
- fb_ofi_z: available for 1649 of 14403
- fb_fresh_z: available for 1148 of 14403
- fb_hhi_z: available for 1588 of 14403
- fb_vol_z: available for 1843 of 14403
Components available per market: 0: 9240, 2: 1055, 3: 2317, 4: 1384, 5: 407.
Wallet first_seen: 77978 wallet rows with first_seen (backfill may be incomplete).

## Caveats

- Only 4249 of 14403 markets have an independent news time (4070 with level D); every other market is scored on max_jump / flow only. A 1% flag is 52 market(s).
- Independent times come from the override CSV and the `scheduled_end` rule (hand-checked sample in the README); end_ts is often date-level.
- Markets with a price-based news time only have max_jump (cut off by construction before the 0.9 crossing) and, if a clear jump with enough history exists, flow-before-move; most get no combined score.
- Jump detection needs >= 36 h of series and >= 12 h before the jump; many markets are short-lived or start at the jump, so only a minority have a flow reading. Thresholds are round numbers, not tuned on any labelled case.
- Flow z-scores use 24 h blocks of the market's own earlier history (needs >= 3); fresh share needs >= 50% wallet coverage. Extra markets use first_seen fetched for the top winner-side wallets only (extra_wallets.parquet).
- Signals use taker rows only. Fisher p-values are rank-based pseudo p-values, signals are correlated. The level baseline uses a time split (see the baseline-scheme section): early independent markets are unscorable by design, and each scored market sees only earlier peers, so early windows have a smaller and more non-stationary training set than the cross-fit's.
- Figures: 120 charts in `figures/` (flagged_*, case_*, jump_start_ts dotted).

## Direction-free burst news times (phase 3c Part B)

Method and the acceptance criterion (fixed 2026-11-13T18:15Z, before any result) are in `markets/newstime_burst.py`. Validation set = scheduled reveals with an independent time (n = 4249; one exact override, the rest date-level `scheduled_end`).

- Burst found for 2686 of 4249 (63.2 %); error = burst - independent time (h): median -106.0, quartiles -1186.0 / 9.0.
- Median |error| over all (undetected = infinite): 1460.0 h (criterion <= 6 h: NOT met); within 6 h: 6.8 %, within 24 h: 26.0 % (criterion >= 70 %: NOT met).
- YES-winner median |error| 145.0 h vs NO-winner 2764.0 h (criterion YES <= 2 x NO: met).
- **burst_validated = False**. deadline / unscheduled_event markets stay screen-only (one-shot screen); the wallet calibration is unchanged.

Markets per group (burst found): deadline 2458/3811, scheduled_no_indep 1682/3267, scheduled_validation 2686/4249, unscheduled_event 1897/3076.
