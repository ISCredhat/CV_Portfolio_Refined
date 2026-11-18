# Can public trade data flag anomalous trading on Polymarket?

A reproducible data pipeline over 14,402 resolved Polymarket markets, a pre-registered validation framework
that plants synthetic informed traders into real data, and, mostly, **negative results** about what public
trade data can detect. Neither of the two detectors I built meets the success criterion I wrote down before
running the validation, and a second, equally pre-registered round found nothing in the same data that is
worth trading: tradable convergence fails, a forecast model adds nothing beyond the market price, and the
one narrow positive (some wallets show persistent gross markout skill) is not evidence that copying pays. The
most useful outputs are the pipeline, the validation method and an honest account of why the detectors and
the trading ideas fail. This is research, not financial advice.

Flags in this project are called **anomalous**, never "insider". They are statistical statements about
trade patterns, not findings of wrongdoing, and nothing here tries to identify a person. No wallet address
appears anywhere in this repository (see [Ethics](#limitations-and-ethics)).

## The question

Polymarket is a prediction market on a public blockchain: every trade, with its wallet, is visible. Some
markets resolve on information that a few people may know early (an award announcement, a military
operation, a product release). Can price and trade data alone single out the markets, or the wallets, where
someone traded on that information before it was public?

## Results at a glance

| | Result |
|---|---|
| Data | 14,402 markets, 21.3 M trades; the trade tape reconciles with Polymarket's own volume figure for 14,379 of 14,397 markets (99.9%) |
| Detector 1: within-event pre-reveal test | False-positive rate on untouched events **11.7%** (limit 10%). **0 of 30** stake/lead cells meet the 50%-detection criterion. Both placebos fail: winners already rank high two days before a scheduled reveal |
| Detector 2: one-shot bet screen | **0 of 30** cells reach 50% detection (14% to 24%). Label-flip placebo passes at 0.74% (limit 2%), but flipped losers enter the top tier more than twice as often as real winners |
| Repeat-wallet test (calibration + FDR) | **0** of 14,850 eligible wallets pass the false-discovery-rate threshold |
| Level discrepancy score D | Circular for price-based news times; usable for only 4,249 markets with an independent event time |
| Direction-free burst news times | Failed validation: median error 1,460 h against a 6 h criterion |
| Stage two A: tradable convergence | **Fails** (0 of 6 versions): the price-selected leader is not underpriced; net returns after costs are negative or about zero. See [Stage two](#stage-two-can-any-of-this-be-traded) |
| Stage two B: wallet skill from markouts | Passes the frozen rule **narrowly**: 743 eligible wallets, top fit decile +0.071 per dollar (24 h, tune period), +0.052 after the spread proxy, top 5 wallets hold 41% of that decile's dollars; not evidence that copying pays |
| Stage two C: forecast benchmark (stop gate) | **Fails**: flow, skill and impact features add nothing beyond the market price (the full model is worse); the strategy track stopped by the pre-registered rule. Sealed period (from 2026-07-01) never opened |
| Descriptive finding | Within 809 scheduled-reveal events, the eventual winner moves far more than the losers in the last day (p about 0.0001 against shuffled winners). This is convergence on the answer, not evidence of leaks |

## Method

The starting point is Paulden (2016), "Smashing the racket", which looks for insider betting by comparing a
market's price path with what comparable markets usually do (a discrepancy score against a peer baseline)
and by reading the price evolution within a single event as the market's changing view of who will win.
What I took from it, and what I did not:

* **Adapted: discrepancy scores against peer baselines.** For each market with an independent event time,
  the price of the eventual winner is compared with the conditional 10/50/90% quantiles of peer markets
  (fitted on markets that resolved earlier, so no market sets its own baseline). `D` is the average amount
  by which the winner's price sits above the peer median before the event.
* **Adapted: the within-event view.** In a multi-outcome event (a Nobel prize with 20 candidates) every
  outcome is a rival. The statistic is the winner's 24 h and 12 h log-odds move before the reveal compared with
  all the other outcomes of the same event, giving a percentile inside the event.
* **Not adapted:** the paper's data (betting exchange, with bookmaker-quality timing), its specific
  model, and any claim about a particular fraud. I use my own quantile baseline, my own windows and
  thresholds, and a different market structure. This is an adaptation of the idea, not a replication.
* **New here:** a one-shot bet screen (a fresh wallet, a large position in a low-priced winner, shortly before
  the news) and a validation framework: synthetic informed buyers planted into real price paths and real
  score distributions, with placebo tests. Acceptance thresholds were written down in `PLAN.md` before
  anything ran and were not changed afterwards.

## Data

Collected from Polymarket's free, keyless APIs (Gamma for markets, Data API v2 for trades, prices and wallet
activity), stored as raw JSON cache, DuckDB and Parquet; none of that is committed (`data/` is about 18 GB).

* Universe: 14,402 resolved markets, $5k to $500k volume, resolved cleanly to YES/NO, excluding crypto
  up/down, sports games and weather. Closed between May 2024 and November 2026 (198 of the 14,396 universe markets closed before 2025; the collection windows start in January 2025 by end date).
* Trades: 21,296,624 rows (taker and maker side) and an 8.5 M-row taker-only tape. Six markets with fewer than
  20 trades were dropped; 148 have a long interior gap in the tape.
* **Integrity result:** `SUM(size) / 2` over the all-rows tape reproduces Gamma's reported volume for 14,379 of
  14,397 markets (99.9%; median ratio 1.00). Cash volume `SUM(size * price)` does not, and is not the metric.
* Information type: each market is tagged `scheduled_reveal`, `unscheduled_event` or `deadline`. Only 4,249
  scheduled-reveal markets have a **news time that does not depend on the price** (an override or a scheduled
  end time checked against trade timing). That limitation drives most of the negative results below.

API quirks found along the way (full list in [docs/REPRODUCE.md](docs/REPRODUCE.md)):

* Gamma `/markets` stops paging near offset 5,000 (HTTP 422); the `/events/keyset` cursor endpoint is needed.
  Its `tags` are mostly `["all"]`; the event object carries the real tags.
* `outcomes`, `outcomePrices` and `clobTokenIds` arrive as JSON-encoded strings.
* Data API v2 uses cursor paging only (an `offset` gives HTTP 400), at most 20 condition ids per `/trades`
  call, no trade id and no USD field (`usd = size * price`).
* With `taker_only=false` **every fill is listed twice** (a taker row and a maker row, same transaction
  hash), and there is no taker flag. The maker row sits on the **complement token at the complement
  price**. Anything directional needs the taker-only tape.
* **Gamma volume = `SUM(size) / 2`** over that all-rows tape, exactly (share notional, $1 per share).
* `/prices-history` rejects fine buckets for old markets; only the 3 h and 12 h grains survive, so price
  resolution depends on market age.
* Rate limits are per IP (HTTP 429 with `Retry-After`); the collectors run at about 5 requests/s.

## What works

* **The pipeline and its integrity checks.** The data layer is reproducible, resumable and reconciles with
  an independent figure (above). 188 offline tests; wallet-free fixtures.
* **A time-split baseline is stable.** Fitting the peer baseline on earlier markets and scoring later ones gives
  the same ranking as a cross-fitted baseline (Spearman 0.974 over 4,070 markets).
* **A descriptive convergence effect.** Across 809 scheduled-reveal events with at least three comparable
  outcomes, the eventual winner's 24 h move exceeds the 95th percentile of the losers' moves in 52.9% of
  events, against 16.5% when the winner label is shuffled within events (100 shuffles: 14.3% to 18.7%;
  permutation p about 0.0001). Markets move toward the answer before a scheduled reveal. In the Nobel Peace
  Prize 2025 event the winner is the largest mover of 21 comparable outcomes
  ([figure](results/figures/nobel_event_moves.png)), but that is what convergence predicts.
* **The validation framework itself.** Planting synthetic buyers into real data is what exposed that neither
  detector separates an informed buyer from ordinary convergence.

## What does not work, and why

* **Within-event test: not a detector.** False-positive rate on untouched events is 11.7% (95% interval 9.8% to
  14.0%) against a 10% limit, so no cell can pass. The placebos fail too: a window ending 48 hours *before* the
  reveal already shows 67% of the real window's excess (share above the 95th percentile of the losers 0.416
  against a 0.171 null). Winners rank high two days early, which is convergence or public news, not a
  window-limited informed buyer ([figure](results/figures/within_event_placebo.png)).
* **One-shot screen: ranks the idealised pattern, but not separably.** An injected fresh-wallet winning bet
  reaches the top 1% tier in only 14% to 24% of cells; its median rank is about the 93rd percentile of 249,203
  real winning positions. The binding component is improbability (it limits 98% of the misses): many injected
  winners are favourites bought at a high price. The tier is rare among real winners (875 positions, 0.35%),
  but real *losers* scored as if they had won enter it 0.74% of the time, more than twice as often. Tier
  membership alone is not evidence of anything.
* **Level D is circular for price-based times.** 398 of 399 pilot news times were "first time the winner's
  price stays above 0.9", which is defined by the very price path D measures. Only the 4,249 scheduled
  reveals have an independent time, so D cannot be applied to deadline or unscheduled markets, which are
  most of the universe.
* **No independent time for deadline and unscheduled markets.** A Gamma end date is a deadline, not the event
  time, for most of them. I tried a direction-free price-and-volume burst detector as a replacement and
  validated it on the 4,249 markets with known times: it found a burst for 63% but the median absolute error
  was 1,460 h against a 6 h criterion fixed in advance (6.8% within 6 h, 26% within 24 h). Those markets stay
  in the one-shot screen only.
* **Repeat-wallet test: nothing passes.** 14,850 wallets have at least five positions with an independent
  cutoff; none passes Benjamini-Hochberg control. Most wallets have one or two positions, so the test has
  little power.
* **Clusters and funding links.** Co-trading clusters were computed (8,294) but are not validated, and
  funding-chain linkage is a stub.

## Validation: power tables

Simulations, not proof of real-world detectability. A synthetic buyer of the eventual winner is planted into
real, otherwise untouched events (stake in USD; entry `k` hours before the cutoff) and the real statistic is
recomputed. Cells that meet the pre-registered criterion (detection of at least 50% while the false-positive
rate is at most 10%): **none, for either detector.** Heatmaps:
[within-event](results/figures/validation_within_event_power.png),
[one-shot](results/figures/validation_oneshot_power.png).

**Within-event test, 24 h window: detection rate** (809 events, 344 plantable; false-positive rate 11.7%).
The statistic uses only the window end points, so every lead of 24 h or less gives the same rate; leads of 48 h
and 72 h fall outside the window.

| stake \ lead (h) | 2 | 6 | 12 | 24 | 48 | 72 |
|---|---|---|---|---|---|---|
| $500 | 2% | 2% | 2% | 2% | 0% | 0% |
| $2,000 | 5% | 5% | 5% | 5% | 0% | 0% |
| $5,000 | 12% | 12% | 12% | 12% | 0% | 0% |
| $20,000 | 17% | 17% | 17% | 17% | 0% | 0% |
| $50,000 | 17% | 17% | 17% | 17% | 0% | 0% |

Restricted to the 177 events with at least 10 comparable outcomes, detection reaches 60% at $5,000 and 88% at
$20,000 and above, but the false-positive rate there is 53.7%: real winners of large events are routinely
already at the top of the ranking.

**One-shot screen: detection rate** of the injected fresh-wallet winning position (tier = real 99th
percentile score and USD at least 100; 4,128 to 4,224 markets per cell):

| stake \ lead (h) | 2 | 6 | 12 | 24 | 48 | 72 |
|---|---|---|---|---|---|---|
| $500 | 14% | 16% | 18% | 20% | 21% | 22% |
| $2,000 | 15% | 17% | 19% | 20% | 22% | 24% |
| $5,000 | 15% | 18% | 19% | 21% | 22% | 24% |
| $20,000 | 15% | 18% | 19% | 21% | 22% | 24% |
| $50,000 | 15% | 18% | 19% | 21% | 22% | 24% |

Median rank of the injected position among the 249,203 real winning positions (percentile, 100 = above all):
88 to 95. Not modelled: price impact (large stakes in thin markets are optimistic), split orders, several
wallets, leaks that are already public news. Full tables with 95% intervals, the 12 h window and a
half-impact sensitivity run are in `results/validation_within_event.md` and `results/validation_oneshot.md`.

## Labelled cases (held out)

Five documented cases (the Nobel Peace Prize 2025, a Venezuela-related market, the Axiom investigation
markets, a military-strike cluster, and Google Year in Search markets) were used only after every rule and
threshold was fixed; none set a parameter. Of 75 labelled markets:

| detector | flags | misses | cannot test |
|---|---|---|---|
| Within-event | 1 | 0 | 74 |
| One-shot | 2 | 3 | 70 |
| Price measures (descriptive) | 0 | 31 | 44 |

The flagged within-event market is the Nobel Peace Prize 2025 winner (24 h percentile 0.976, rank 21 of
1,670), which is what the validation predicts for a winner already ranked high before the reveal (11.7% of
untouched events do the same). The one-shot tier contains the Venezuela-related market and the Nobel market,
and two wallets named in public reporting (referred to here by role only; ranks 13 and 251 of 249,203). A
fresh wallet buying a low-priced winner shortly before the news is exactly what the score rewards. **This is
not evidence of leaks:** two suspected cases cannot calibrate or validate anything, the context cases have no
confirmed informed trader, most labelled markets cannot be tested (no independent time or no valid cutoff), and
the placebo shows the tier does not separate informed from uninformed positions by itself. A miss on a context
case carries no information either.

## Hand audit of the extremes

Thirty highest within-event extremes and thirty highest one-shot positions (outside the one Venezuela-related
market that dominates the top ranks) were classified by a fixed rule set written in advance. There is no
ground truth and no news source was read offline, so `unclear` is the default.

| class | within-event | one-shot |
|---|---|---|
| plausible leak | 0 (0%) | 2 (7%) |
| predictable convergence | 15 (50%) | 1 (3%) |
| public-news reaction | 8 (27%) | 3 (10%) |
| unclear | 7 (23%) | 24 (80%) |

"Plausible leak" means only that offline data could not explain the move by a public trigger. Of the one-shot
rows, 24 of 30 use a jump-start cutoff, which is chosen from the price move itself and is therefore
outcome-selected. Details and the exact rules: `results/validation_hand_audit.md`.

<!-- STAGE2:BEGIN -->
## Stage two: can any of this be traded?

Stage one produced no validated detector, so the natural next question was whether any signal in this data
is worth trading. I wrote the pass marks down **before running anything** (frozen on 2026-11-14 at 12:50 UTC in
[docs/STAGE2_PLAN.md](docs/STAGE2_PLAN.md), section 9) and ran three tests, all offline on the data already
collected. The short answer is **no**: the pre-registered rule stopped the strategy track. This is research,
**not financial advice**, and nothing here places or recommends a trade.

**Design.** Markets are assigned to periods by their close time: *fit* up to 2025-12-31, *tune* 2026-01-01 to
2026-06-30, and a *sealed* period from 2026-07-01 onward that was **never opened** (the code drops it at load
and a test enforces it; no hyperparameters were searched, so "tune" was used purely as an evaluation period).
Costs: entry at the next traded price after a 5-minute delay (1 and 30 minutes are also in the CSV), plus a
taker fee of rate x p x (1 - p) per share, with the rate set by category (0% geopolitics and world, 3% sports,
4% finance, politics, tech and mentions, 5% economics, culture, weather and other, 7% crypto). That fee schedule
comes from a secondary source and still needs confirming with the venue. A strategy passes only if the lower
95% bound of its net return per dollar (bootstrap resampling whole events) is above zero, it beats a placebo,
and it stays positive at doubled costs.

| Test | Verdict | In one line |
|---|---|---|
| A. Tradable convergence (buy the price-selected leader of a multi-outcome event days before the reveal) | **Fails**: 0 of 6 versions pass | The leader is not underpriced (win rate within about 0.05 of its entry price); net returns are negative or about zero |
| B. Wallet skill from markouts (do wallets whose trades moved the price their way keep doing so?) | **Passes** the frozen rule, narrowly | Top fit-period decile earns +0.071 per dollar over 24 h on the tune period (+0.052 after the spread proxy); it is concentrated and is not evidence that copying pays |
| C. Forecast benchmark (do flow, wallet-skill and impact features beat the market price?) | **Fails**: no gain in either half, at any horizon | The price is already well calibrated; the full model is worse than the price |

### A. Tradable convergence

At 7 days, 72 hours and 48 hours before the reveal of a multi-outcome scheduled event, buy the outcome that
looks like the leader **from prices alone** (largest log-odds rise over the prior window, or highest price; never
the eventual winner) and hold to resolution. Fit and tune periods only, 5-minute entry delay.

| Leader chosen by | Bought at T minus | Events | Win rate | Mean entry price | Win rate minus entry | Net return per dollar [95%] | Doubled-cost stress | Frozen rule |
|---|---|---|---|---|---|---|---|---|
| largest log-odds rise | 7 days | 128 | 0.594 | 0.564 | +0.030 | -0.176 [-0.369, +0.005] | -0.147 | FAIL |
| largest log-odds rise | 72 h | 204 | 0.613 | 0.570 | +0.043 | -0.091 [-0.227, +0.058] | -0.089 | FAIL |
| largest log-odds rise | 48 h | 257 | 0.553 | 0.566 | -0.013 | -0.177 [-0.294, -0.054] | -0.188 | FAIL |
| highest price | 7 days | 170 | 0.735 | 0.727 | +0.009 | +0.041 [-0.083, +0.162] | +0.055 | FAIL |
| highest price | 72 h | 240 | 0.746 | 0.754 | -0.008 | -0.005 [-0.117, +0.121] | -0.012 | FAIL |
| highest price | 48 h | 301 | 0.744 | 0.754 | -0.010 | -0.000 [-0.116, +0.106] | -0.012 | FAIL |

None of the six versions has a lower bound above zero, so none passes (and none would after correcting for six
cells). The best-looking number is the highest-price leader at 7 days: its tune-period-only return is
positive but its fit-period return is not, and its pooled interval spans zero. A placebo that buys a random
non-leader matched on price exists for only part of the trades (the leader is often much pricier than every
sibling), so "beats the placebo" is weak evidence either way.

### B. Wallet skill from markouts

A **markout** is how far the price of the traded token moved in the trade's favour afterwards, per dollar, so it
needs neither the final outcome nor a news time. Wallets were classified on the fit period and followed into the
tune period (743 wallets had at least 20 trades in each period).

| Tune-period 24 h markout per dollar | Gross [95%] | After spread proxy [95%] |
|---|---|---|
| Wallets in the top fit-period skill decile | +0.071 [+0.024, +0.125] | +0.052 [+0.007, +0.103] |
| Wallets in the bottom fit-period skill decile | +0.002 [-0.017, +0.015] | negative (see CSV) |
| All eligible wallets | +0.010 [+0.004, +0.016] | -0.001 [-0.007, +0.005] |

* Rank correlation of fit-period and tune-period skill: +0.145 [+0.078, +0.179].
  The shuffled-wallet baseline's 95th percentile for the top-decile mean is +0.046, below the observed
  +0.071. All four frozen checks pass.
* The result is narrow: only 743 wallets are eligible, about 74 per decile; the **top 5 wallets hold
  41% of the top decile's dollars**; 69% of the top decile's wallets have positive
  tune skill against 47% of the bottom decile's; and across all eligible wallets the markout is about zero
  after the spread proxy.
* Taker trades lose on average. Pooled over all taker trades in the tune period:

| Horizon | Gross, all taker trades [95%] | After spread proxy [95%] |
|---|---|---|
| 5 minutes | -0.0012 [-0.0025, +0.0001] | -0.0154 [-0.0182, -0.0131] |
| 1 hour | -0.0011 [-0.0030, +0.0006] | -0.0153 [-0.0185, -0.0126] |
| 6 hours | -0.0049 [-0.0073, -0.0024] | -0.0188 [-0.0224, -0.0151] |
| 24 hours | -0.0051 [-0.0078, -0.0024] | -0.0187 [-0.0230, -0.0148] |
| to resolution | +0.0047 [-0.0014, +0.0108] | -0.0096 [-0.0161, -0.0029] |

* **Share of wallets classed skilled.** With the pre-registered Monte-Carlo test (2,000 draws) and
  Benjamini-Hochberg correction, 0.00% of 41,647 fit-period wallets qualify; that zero is a
  resolution limit of the test (its smallest possible p-value is 1/2001), not proof of no skill. An analytic version
  without that limit classes 0.10% (42 wallets) as skilled. For comparison, a published
  study reports 3.14% of accounts as skilled with 44% staying skilled out of sample; the definitions differ, so this
  is descriptive only. Of the 18 fit-skilled wallets with at least 10 tune trades,
  72% have positive tune skill, but none is classed skilled again.

Reading: some active wallets show persistent gross markout skill over this period. That is a statement about the
past, not evidence that copying them is profitable (the spread proxy absorbs most of it, the group is small and
concentrated, and a copier would trade later than the wallet did).

### C. Forecast benchmark (the stop gate)

At 7 days, 3 days and 1 day before a market's scheduled end, for open markets with at least 20 earlier taker
trades, predict the outcome using only data before that time. Models: a logistic recalibration of the price and
shallow gradient-boosted trees (fixed hyperparameters, no tuning) with price dynamics, order-flow imbalance, volume
surge, buyer concentration, fresh-wallet share, net flow of wallets classed skilled on the fit period, impact per
dollar, and market descriptors. Purged walk-forward validation (train on months before the test month with a 7-day
embargo). Positive improvement means the model beats the market price on log-loss.

| Forecast time before end | Model | Test rows / events | Price log-loss | Improvement over price, Jan-Mar 2026 [95%] | Improvement, Apr-Jun 2026 [95%] |
|---|---|---|---|---|---|
| 7 days | logistic recalibration of the price | 2936 / 647 | 0.177 | -0.0009 [-0.0025, +0.0010] | -0.0000 [-0.0014, +0.0013] |
| 7 days | gradient-boosted trees, all features | 2936 / 647 | 0.177 | -0.0126 [-0.0212, -0.0039] | -0.0104 [-0.0161, -0.0045] |
| 3 days | logistic recalibration of the price | 2944 / 659 | 0.162 | -0.0010 [-0.0031, +0.0012] | +0.0002 [-0.0012, +0.0018] |
| 3 days | gradient-boosted trees, all features | 2944 / 659 | 0.162 | -0.0055 [-0.0125, +0.0015] | -0.0064 [-0.0136, -0.0002] |
| 1 day | logistic recalibration of the price | 2908 / 662 | 0.157 | -0.0006 [-0.0024, +0.0013] | +0.0001 [-0.0011, +0.0013] |
| 1 day | gradient-boosted trees, all features | 2908 / 662 | 0.157 | -0.0071 [-0.0143, -0.0005] | -0.0067 [-0.0123, -0.0014] |

No model beats the price in both halves at any horizon, so the frozen rule fails and **the strategy track stops**
(the remaining candidate strategies, such as following informed flow or skilled wallets, were not run). The
all-features model is significantly worse than the price at 7 days. The price is well calibrated; feature-group
ablation and a stress test that drops the largest 10% of markets by volume agree
([results/stage2_forecast.md](results/stage2_forecast.md)).

### Caveats and what was not done

* The fee schedule is from a secondary source. There is no historical order book, so spreads and impact are
  inferred from the taker tape.
* About half of the leader trades in test A were skipped because no price traded within 6 hours, and the volume cap
  (1% of trailing 24-hour volume) shrank 91% of the stakes, so the backtest says little about size.
* The wallet-skill feature in the forecast is nearly empty (only 42 wallets qualify under the analytic rule), so
  test C is a weak test of that feature in particular.
* The builders added a few definitions beyond the frozen text, listed here for transparency: a 6-hour markout
  horizon with a 24-hour staleness limit; a spread proxy equal to the gap to the preceding opposite-side fill within
  10 minutes (else the market median, else the global median); a purge rule for training rows (closed at least 7 days
  before the test month); test rows must have forecast time on or after 2026-01-01; and "beats the placebo" read as
  exceeding the upper 95% limit of 200 price-matched placebo draws.
* `scripts/orderbook_recorder.py` (a polite live order-book recorder for a forward paper test) exists and is tested
  offline, but it was **never run**. Funding-chain data and the Hugging Face on-chain dataset were not used; the
  steps are written in [docs/STAGE2_PLAN.md](docs/STAGE2_PLAN.md) section 10.
* Reproduce: `pmdetect stage2 convergence`, `pmdetect stage2 markouts` and `pmdetect stage2 forecast` (they need the
  local data store); this section is generated from the result CSVs by `scripts/make_readme_stage2.py`.
<!-- STAGE2:END -->

## Limitations and ethics

* **Simulation is not detection.** The power tables describe a best-case idealised buyer in real data. They
  say nothing about how real informed traders behave.
* **Few labels.** Two suspected markets and two wallet classes; no confirmed ground truth anywhere.
* **Timing.** Independent event times exist for 4,249 markets, all scheduled reveals. Price history is 3-hourly
  for most older markets. Wallet `first_seen` is known for only part of the wallets.
* **Convergence is not insider trading.** A winner that rises before a scheduled reveal is the normal
  behaviour of a market absorbing public information.
* **Ethics.** All data are public blockchain and API data. The words "insider" and "leak" are not used for
  flags; "anomalous" means statistically unusual and nothing more. The project does not try to identify
  people. Wallet-level outputs are written to `data/private/`, which is gitignored; **no wallet address
  appears in the repository** (`python scripts/privacy_scan.py` checks every file outside `data/`, and a
  test runs it). Test fixtures and the demo data use fake ids.

## Reproduce

Small sample, offline, seconds, no data download (50 fixture markets with wallet ids anonymised):

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]          # Python 3.11+; Windows paths shown
.venv\Scripts\pmdetect demo run              # about 3 s; writes demo_output/ (gitignored)
.venv\Scripts\python -m pytest               # about 40 s, all offline
.venv\Scripts\ruff check src tests scripts
.venv\Scripts\python scripts\privacy_scan.py
.venv\Scripts\python scripts\make_figures.py # rebuilds the curated figures from results/ (offline)
```

The validation commands need the local data store built by the full pipeline (not committed):

```
pmdetect validate within-event    # about 45 s; power grid + placebos -> results/validation_within_event*
pmdetect validate oneshot         # a few seconds to about a minute; power grid + label flip -> results/validation_oneshot*
pmdetect validate cases           # labelled cases, held out
pmdetect validate audit           # hand audit lists
pmdetect stage2 convergence       # stage two A, a few seconds
pmdetect stage2 markouts          # stage two B, builds markouts over about 6.7 M trades (about 7 minutes)
pmdetect stage2 forecast          # stage two C
python scripts/make_readme_stage2.py   # regenerates the "Stage two" section of this README from results/stage2_*.csv
```

The **full pipeline** (collect markets, trades and prices, snapshots, news times, market and wallet scoring)
takes about 1.5 days of rate-limited API time on one PC, is resumable, and each step is listed with its command,
duration and disk use in [docs/REPRODUCE.md](docs/REPRODUCE.md). `data/` is about 18 GB. The scoring steps run
offline from Parquet in minutes (market scoring 267 s with 6.5 GB peak memory; wallet scoring under a minute).

## Next steps

* **Independent event times** for deadline and unscheduled markets (news archives or source feeds). This is the
  main obstacle: without them the level score, the repeat test and the burst method cannot be validated.
* **Funding-chain data** (who funded a fresh wallet) to link wallets that a co-trading test cannot.
* **A forward-in-time test:** freeze the rules now and score new markets as they resolve, so no result can
  influence the thresholds.
* A within-event statistic that is robust to convergence (for example, one conditioned on public news or on
  the market's own earlier drift) before calling it a detector.
* The trading track is **closed** by the pre-registered stage-two rule. Reopening it would need a new idea and
  new pre-registered marks, not a retuned version of these tests; the sealed period from 2026-07-01 is still
  unopened and would be the place for any single, pre-declared final check.

## Layout

```
src/pmdetect/        collect/ (APIs), store/, markets/ (news times, baseline, scores, within-event),
                     wallets/ (positions, calibration, FDR, filters, one-shot, clusters), validate/,
                     stage2/ (convergence backtest, markouts, forecast benchmark), report/, demo.py
scripts/             bulk fetch and load scripts, make_figures.py, make_readme_stage2.py, privacy_scan.py,
                     orderbook_recorder.py (tested offline, never run)
tests/               offline tests; fixtures/ (recorded API pages, anonymised demo Parquet)
docs/REPRODUCE.md    reproduce guide and API quirks
results/             aggregate tables (*.md, *.csv) and curated figures/; no per-wallet rows
PLAN.md              phase plans and the pre-registered acceptance thresholds
```
