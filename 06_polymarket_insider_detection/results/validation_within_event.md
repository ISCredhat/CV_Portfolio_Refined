# Within-event test: synthetic-insider power and placebos (phase 4A / 4C)

Aggregate-only output of `pmdetect validate within-event` (no wallet addresses, no market or event identifiers). Thresholds are the PLAN.md acceptance thresholds written before any run; they were not changed.

## What this is, and what it is not

A **simulation**: a buyer of the eventual winner is planted into real, otherwise untouched price paths with an empirical, linear-in-USD (log-odds) price impact estimated from each market's own taker tape, and the real pipeline statistic is recomputed. It is **not proof of real-world detectability**. It does not model buyers who split orders over time or across wallets, impact that is concave in size, or leaks that are already public news (then every outcome moves and there is nothing to single out). Planted buys are one order with a permanent price effect; buys earlier than the window start are invisible to a 'change in the last day' statistic by construction.

## Method

* Population: real scheduled-reveal multi-outcome events from `within_event.parquet` with a pipeline statistic and >= 3 comparable outcomes. Used: **809**. Excluded: {'multi_events_total': 1008, 'no_pipeline_statistic': 18, 'statistic_n_comparable_lt3': 181, 'pipeline_parity_or_missing_series': 0, 'used': 809}.
* Plantable ('unremarkable', fixed in advance): real winner percentile < 0.75 in the window being tested (24 h: 344 events; 12 h: 381).
* Impact: per-market Huber slope of 1-h log-odds change on signed net taker USD ($1k), tape before the window end; own slope if >= 30 hour-pairs, positive and t >= 2.0, else the pooled median by price bucket. Own-market impact used for **322** of 809 events, pooled fallback for 487. Median slope used: 0.179 log-odds per $1k (pooled median 0.193; 1248 of 4239 fitted markets valid).
* Planting: winner log-odds +beta*stake/1000 from reveal - k h onward (clipped to [0.005, 0.995]), no decay; siblings scaled by (S - p_w')/(S - p_w) so the comparable YES prices keep their sum.
* Statistic: exactly `within_event.py` (24 h and 12 h windows ending 1 h before the reveal, forward-filled last price, same clip, same stale rule, mid-rank winner percentile).
* Detection: planted winner percentile >= 0.95. FPR: share of untouched real events (same population) with real winner percentile >= 0.95. Criterion: detection >= 50% at FPR <= 10%. Bootstrap: 1000 event resamples, 95% percentile intervals; seed 20261113.

## Power table, 24 h window (detection rate, 95% CI)

FPR on untouched events: **11.7%** (95% CI 9.8% to 14.0%, n = 809); on all multi events incl. 2-outcome ones (sensitivity): 9.6%.

| stake \ lead (h) | 2 | 6 | 12 | 24 | 48 | 72 |
|---|---|---|---|---|---|---|
| $500 | 2% [1%, 4%] | 2% [1%, 4%] | 2% [1%, 4%] | 2% [1%, 4%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $2,000 | 5% [3%, 7%] | 5% [3%, 7%] | 5% [3%, 7%] | 5% [3%, 7%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $5,000 | 12% [8%, 15%] | 12% [8%, 15%] | 12% [8%, 15%] | 12% [8%, 15%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $20,000 | 17% [13%, 21%] | 17% [13%, 21%] | 17% [13%, 21%] | 17% [13%, 21%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $50,000 | 17% [13%, 21%] | 17% [13%, 21%] | 17% [13%, 21%] | 17% [13%, 21%] | 0% [0%, 0%] | 0% [0%, 0%] |

Cells meeting the criterion: **none** (0 of 30).
With the alternative FPR denominator (all multi events): none either.

## Power table, 12 h window (detection rate, 95% CI)

FPR on untouched events: **11.1%** (95% CI 9.1% to 13.3%, n = 809); on all multi events incl. 2-outcome ones (sensitivity): 9.1%.

| stake \ lead (h) | 2 | 6 | 12 | 24 | 48 | 72 |
|---|---|---|---|---|---|---|
| $500 | 3% [1%, 5%] | 3% [1%, 5%] | 3% [1%, 5%] | 0% [0%, 0%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $2,000 | 8% [5%, 10%] | 8% [5%, 10%] | 8% [5%, 10%] | 0% [0%, 0%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $5,000 | 13% [9%, 16%] | 13% [9%, 16%] | 13% [9%, 16%] | 0% [0%, 0%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $20,000 | 14% [11%, 18%] | 14% [11%, 18%] | 14% [11%, 18%] | 0% [0%, 0%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $50,000 | 15% [11%, 18%] | 15% [11%, 18%] | 15% [11%, 18%] | 0% [0%, 0%] | 0% [0%, 0%] | 0% [0%, 0%] |

Cells meeting the criterion: **none** (0 of 30).
With the alternative FPR denominator (all multi events): none either.

## Supplementary: events with >= 10 comparable outcomes only (24 h)

Only these can reach a mid-rank percentile of 0.95. FPR among them is **53.7%** (95% CI 46.3% to 61.0%, n = 177), so real winners of large events are routinely already at the top of the pre-reveal ranking.

| stake \ lead (h) | 2 | 6 | 12 | 24 | 48 | 72 |
|---|---|---|---|---|---|---|
| $500 | 12% [5%, 19%] | 12% [5%, 19%] | 12% [5%, 19%] | 12% [5%, 19%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $2,000 | 25% [15%, 36%] | 25% [15%, 36%] | 25% [15%, 36%] | 25% [15%, 36%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $5,000 | 60% [48%, 72%] | 60% [48%, 72%] | 60% [48%, 72%] | 60% [48%, 72%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $20,000 | 88% [80%, 96%] | 88% [80%, 96%] | 88% [80%, 96%] | 88% [80%, 96%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $50,000 | 88% [80%, 96%] | 88% [80%, 96%] | 88% [80%, 96%] | 88% [80%, 96%] | 0% [0%, 0%] | 0% [0%, 0%] |

## Sensitivity: half the estimated impact (24 h)

| stake \ lead (h) | 2 | 6 | 12 | 24 | 48 | 72 |
|---|---|---|---|---|---|---|
| $500 | 1% [0%, 2%] | 1% [0%, 2%] | 1% [0%, 2%] | 1% [0%, 2%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $2,000 | 4% [2%, 6%] | 4% [2%, 6%] | 4% [2%, 6%] | 4% [2%, 6%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $5,000 | 7% [4%, 9%] | 7% [4%, 9%] | 7% [4%, 9%] | 7% [4%, 9%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $20,000 | 16% [12%, 20%] | 16% [12%, 20%] | 16% [12%, 20%] | 16% [12%, 20%] | 0% [0%, 0%] | 0% [0%, 0%] |
| $50,000 | 17% [13%, 21%] | 17% [13%, 21%] | 17% [13%, 21%] | 17% [13%, 21%] | 0% [0%, 0%] | 0% [0%, 0%] |

## Verdict for the README

**No cell with stake <= $20k meets the PLAN criterion** (the false-positive rate on untouched events (11.7%) already exceeds 10%). Per the PLAN, the within-event test is reported as a **descriptive result, not a detector**.

## Placebos (4C)

Pass rules (PLAN): (a) window ending 48 h before the reveal passes if its permutation p > 0.01 or its excess share above the 95th percentile of losers (observed minus null) is at most half that of the real window; (b) 100 shuffled winners pass if the real share sits inside the central 95% interval of the shuffle distribution.

| placebo | window | set | n | observed | null | excess | p | real-window excess | pass |
|---|---|---|---|---|---|---|---|---|---|
| (a) 48 h-earlier window | 24 h | intersection | 794 | 0.416 | 0.171 | +0.245 | 0.0005 | +0.365 | **FAIL** |
| (a) 48 h-earlier window | 24 h | own | 794 | 0.416 | 0.171 | +0.245 | 0.0005 | +0.364 | **FAIL** |
| (b) shuffled winners: share_above_q95_of_losers | 24 h | real window | 809 | 0.529 | 0.165 | +0.364 | central 95%: [0.143, 0.187] | real at pct 1.00 of shuffles | **FAIL** |
| (b) shuffled winners: share_percentile_ge_0.95 | 24 h | real window | 809 | 0.117 | 0.016 | +0.102 | central 95%: [0.008, 0.027] | real at pct 1.00 of shuffles | **FAIL** |
| (a) 48 h-earlier window | 12 h | intersection | 796 | 0.371 | 0.165 | +0.206 | 0.0005 | +0.325 | **FAIL** |
| (a) 48 h-earlier window | 12 h | own | 796 | 0.371 | 0.165 | +0.206 | 0.0005 | +0.323 | **FAIL** |
| (b) shuffled winners: share_above_q95_of_losers | 12 h | real window | 809 | 0.485 | 0.162 | +0.323 | central 95%: [0.135, 0.190] | real at pct 1.00 of shuffles | **FAIL** |
| (b) shuffled winners: share_percentile_ge_0.95 | 12 h | real window | 809 | 0.111 | 0.016 | +0.096 | central 95%: [0.007, 0.023] | real at pct 1.00 of shuffles | **FAIL** |

Reading: a real winner that is already high in the pre-reveal ranking is not by itself evidence of an informed buyer, because markets converge on the eventual winner as the reveal nears (predictable convergence, public news). The placebo (a) shows how much of the excess exists two days earlier, when a window-limited informed buyer is not expected to dominate.

## Caveats

* Simulation with an empirical, linear-ish impact; not real-world detectability.
* The 24 h / 12 h statistic cannot see buys placed before the window start (about 25 h / 13 h before the reveal); long leads show only the false-positive floor.
* The statistic uses only the two window end points, so every lead inside the window (24 h window: k = 2 to 24 h) gives the same rate; identical columns are expected, not a bug.
* Percentile detection needs >= 10 comparable outcomes; most events have fewer.
* Hourly last-price impact estimates carry bid-ask bounce (attenuation) and endogeneity; fits are per market from thin tapes; fallback markets use a pooled median.
* Splitting, wallet fragmentation and public-news leaks are not modelled.
* Wall time 42 s.
