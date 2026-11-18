# Integrity report

- Markets with a completed trade tape: **14403**
- Dropped by the at-least-20-trades rule: **6** (in_universe set to false)
- Markets kept: **14397**; trades kept: **21,296,624**; distinct wallet-market pairs: 3,805,977
- Reconciling with Gamma volume (|tape notional / Gamma volume - 1| <= 15%), among kept markets with a Gamma volume: **14379 of 14397 (99.9%)**
- Markets with a long interior gap in the tape (> 7 days and > 50% of the span): 148
- Markets with row-level issues (bad size/price/wallet): 0

## Status of kept markets

| status | markets |
| --- | ---: |
| ok | 14379 |
| tape_below_gamma | 14 |
| tape_above_gamma | 4 |

## Tape notional / Gamma volume (kept markets)

- min 0.53, p10 1.00, median 1.00, p90 1.00, max 1.78
- Gamma `volumeNum` is the *share-notional* volume: the sum of `size` (shares, each worth $1 at resolution) over every tape row divided by 2, because `taker_only=false` lists each fill twice (taker row and maker row). So `notional_usd = SUM(size) / 2` reconciles with Gamma, for binary and neg-risk markets alike. The cash volume `SUM(size * price)` (`trade_usd`, `cash_ratio`) is not what Gamma reports and is not used for reconciliation; a large gap between the two just reflects the typical price of the traded side.

## Dropped by the 20-trade rule

- `zachxbt-investigation-post-over-25m-views-on-day-one`: 19 trades
- `will-trump-nominate-another-person-for-energy-secretary`: 15 trades
- `will-potus-tweet-less-than-5-times-feb-7-14`: 6 trades
- `zachxbt-investigation-post-over-18m-views-on-day-one`: 15 trades
- `will-trump-say-russia-during-fox-news-oval-office-interview`: 18 trades
- `elon-musk-of-tweets-october-2025-440-459`: 18 trades
