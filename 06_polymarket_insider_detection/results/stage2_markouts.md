# Stage 2, task B: taker-trade markouts and wallet skill v2

Aggregate results only (no wallet addresses). Pre-registered pass rule: docs/STAGE2_PLAN.md section 9. Periods by market `closed_ts`: fit <= 2025-12-31, tune 2026-01-01 to 2026-06-30; the sealed period was never loaded. Wallets are described as 'anomalous' or 'skilled', never as having acted on non-public information.

## Verdict: PASS (skill persists)

- fit: 3,303,663 trades with a 24 h markout, 375,541 wallets (41,647 with >= 10), 6,460 markets, 2,047 events.
- tune: 2,450,967 trades with a 24 h markout, 235,718 wallets (21,023 with >= 10), 4,409 markets, 998 events.

## Pre-registered pass rule (24 h markout per dollar, tune period)

| check | value | result |
| --- | --- | --- |
| fit-top-decile tune mean, lower 95% bound > 0 | +0.07131 [+0.02354, +0.12527] | pass |
| above fit-bottom decile | top +0.07131 vs bottom +0.00180 (diff +0.06951 [+0.01825, +0.12742]) | pass |
| Spearman fit-shrunk vs tune skill, lower 95% bound > 0 | +0.1451 [+0.0782, +0.1795] | pass |
| shuffled-wallet baseline 95th pct below observed top decile | baseline p95 +0.04645 (mean +0.01613, 200 shuffles) vs observed +0.07131 | pass |

Eligible wallets (>= 20 trades in each period after the 7-day embargo): 743; 2,132 markets overlapping the boundary dropped; trades used: fit 3,280,907, tune 1,192,673. The bootstrap resamples tune-period events (1,000 resamples, seed 20261113); the fit classification is held fixed.

Concentration check of the top decile (tune period): largest wallet 13.7% of its dollars, top five wallets 40.9%; equal-weighted wallet mean +0.05308, median +0.03115 (bottom decile median -0.00769); share of wallets with positive tune skill 69% in the top vs 47% in the bottom decile.

### Tune-period 24 h markout per dollar by fit-period decile (gross, then spread-adjusted)

| fit decile | wallets | gross | 95% interval | spread-adjusted | 95% interval |
| --- | --- | --- | --- | --- | --- |
| 1 | 75 | +0.00180 | [-0.01680, +0.01537] | -0.01810 | [-0.03770, -0.00488] |
| 2 | 75 | +0.00439 | [-0.01857, +0.02928] | -0.01821 | [-0.04636, +0.00922] |
| 3 | 75 | +0.02458 | [-0.03496, +0.08599] | +0.00688 | [-0.05333, +0.06484] |
| 4 | 74 | +0.00825 | [-0.02372, +0.03178] | -0.00508 | [-0.03676, +0.01949] |
| 5 | 74 | -0.00358 | [-0.00929, +0.00152] | -0.00993 | [-0.01638, -0.00493] |
| 6 | 74 | +0.00138 | [-0.00191, +0.00479] | -0.00314 | [-0.00667, +0.00009] |
| 7 | 74 | +0.01527 | [-0.00924, +0.04093] | +0.00377 | [-0.02093, +0.02758] |
| 8 | 74 | +0.01777 | [-0.01405, +0.04699] | +0.00324 | [-0.02898, +0.03215] |
| 9 | 74 | +0.03649 | [+0.01971, +0.05213] | +0.01649 | [-0.00057, +0.03139] |
| 10 | 74 | +0.07131 | [+0.02354, +0.12527] | +0.05249 | [+0.00673, +0.10349] |
| all eligible | 743 | +0.00980 | [+0.00382, +0.01596] | -0.00072 | [-0.00668, +0.00477] |

## Share of wallets classed skilled (descriptive; definitions differ from the literature)

Skilled = empirical-Bayes shrunk 24 h skill > 0 AND the one-sided sign-randomisation p, adjusted by Benjamini-Hochberg across wallets with >= 10 trades, <= 5%. Published figures for comparison: 3.14% of 1.72M wallets 'skilled winners', 44% of those retained out of sample (different test, different outcome measure, different universe).

Two versions of the p-value: (i) the pre-registered Monte-Carlo sign-randomisation p with 2,000 draws, whose smallest possible value is 1/2001, so Benjamini-Hochberg over tens of thousands of wallets cannot reject anyone unless several hundred wallets sit at that floor (a resolution limit, not evidence of no skill); (ii) a normal p from the exact mean and variance of the same sign-shuffle null (no resolution floor), used for the retention figures and for the forecasting features.

| period | wallets with >= 10 trades | skilled (MC p, 2,000 draws) | share | skilled (analytic p) | share |
| --- | --- | --- | --- | --- | --- |
| fit | 41,647 | 0 | 0.00% | 42 | 0.10% |
| tune | 21,023 | 0 | 0.00% | 0 | 0.00% |

Retention (analytic classification): of 18 fit-skilled wallets with >= 10 tune trades, 72.2% have a positive tune skill and 0.0% are again classed skilled in the tune period (for reference 0.00% of all tune wallets with >= 10 trades are classed skilled).

## Markouts by horizon (pooled, size-weighted, per dollar of notional)

Gross is side-signed in the traded token's price; spread-adjusted subtracts the half-spread proxy per trade. 'zero' = share of markouts that are exactly zero (no later observation before t+h). 5 min and 1 h are the pre-registered short horizons, 6 h the extra sign check, 24 h the headline, 'res' the final payoff.

| horizon | period | trades | kept | gross [95%] | spread-adjusted [95%] | zero | wallets > 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | fit | 3,928,182 | 99.8% | +0.00006 [-0.00056, +0.00068] | -0.01304 [-0.01428, -0.01190] | 65.2% | 29.2% |
| 5m | tune | 2,728,152 | 99.8% | -0.00116 [-0.00255, +0.00008] | -0.01543 [-0.01824, -0.01308] | 65.6% | 30.7% |
| 1h | fit | 3,913,092 | 99.5% | -0.00173 [-0.00303, -0.00049] | -0.01479 [-0.01661, -0.01310] | 34.6% | 28.1% |
| 1h | tune | 2,716,055 | 99.4% | -0.00112 [-0.00299, +0.00061] | -0.01535 [-0.01852, -0.01265] | 40.2% | 30.6% |
| 6h | fit | 3,731,736 | 94.9% | -0.00386 [-0.00534, -0.00228] | -0.01624 [-0.01810, -0.01431] | 14.6% | 30.3% |
| 6h | tune | 2,613,022 | 95.6% | -0.00489 [-0.00726, -0.00238] | -0.01876 [-0.02238, -0.01509] | 18.6% | 29.6% |
| 24h | fit | 3,303,663 | 84.0% | -0.00391 [-0.00641, -0.00138] | -0.01555 [-0.01854, -0.01281] | 7.5% | 36.1% |
| 24h | tune | 2,450,967 | 89.7% | -0.00507 [-0.00783, -0.00237] | -0.01873 [-0.02302, -0.01481] | 8.9% | 31.9% |
| res | fit | 3,934,349 | 100.0% | -0.00279 [-0.00984, +0.00387] | -0.01592 [-0.02287, -0.00916] | 0.0% | 52.1% |
| res | tune | 2,733,135 | 100.0% | +0.00469 [-0.00140, +0.01083] | -0.00962 [-0.01607, -0.00287] | 0.0% | 42.9% |

## Construction and caveats

- Price at t+h: last observation at or before t+h of the YES-equivalent series (all taker fills plus the stored winner-token price history converted to YES terms, observations after closed_ts removed). Horizon dropped (never padded) if t+h is after min(last observation, closed_ts) or the last observation is older than 6 h (h <= 1 h), 24 h (6 h) or 48 h (24 h).
- Because the last observation can be the trade itself, short-horizon markouts contain many exact zeros; this dilutes them toward zero but does not bias the sign.
- The sign-randomisation null conditions on which markets, prices and sign counts a wallet faced; the null mean is not zero (a market that drifts up rewards buyers in the null as well). Skill itself is the raw pre-registered statistic; the p-value is against the shuffled null.
- A taker's own entry price already contains its half spread, so subtracting a further half spread (pre-registered) is conservative.
- The persistence test uses the embargoed sets; the skilled share and the retention figures use all fit and tune trades without embargo.
- Markets with no resolved YES/NO outcome are excluded; fills after a market's closed_ts are dropped.
