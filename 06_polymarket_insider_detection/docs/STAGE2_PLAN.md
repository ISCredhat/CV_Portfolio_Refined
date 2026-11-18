# Stage 2 plan: from anomaly detection to testable trading signals

Status: **CLOSED (2026-11-14).** Round 1 was run against the frozen marks in section 9 (not edited) and the
strategy track stopped. Outcome: A tradable convergence FAILED (0 of 6 versions; the price-selected leader is not
underpriced); C forecast benchmark FAILED (flow, skill and impact features add nothing beyond the market price,
the full model is worse), which under the pre-registered rule stops the track, so strategies S2, S3 and S5 were
not run; B wallet-skill persistence PASSED the frozen rule narrowly (743 eligible wallets, top fit decile +0.071
per dollar 24 h markout on the tune period, +0.052 after the spread proxy, top 5 wallets 41% of that decile's
dollars) and is not evidence that copying pays. The sealed period (from 2026-07-01) was never opened. Results and
caveats: README section "Stage two: can any of this be traded?" and `results/stage2_*.md`. Section 10 (data steps
for Isaac) remains available if the work is ever reopened; `scripts/orderbook_recorder.py` was never run.
Original status line: DRAFT for Isaac's approval, written 2026-11-14 before anything was run.

This is a research and backtesting plan, not financial advice. It does not recommend any trade, and no part
of it places orders: live trading is a separate decision that stays with Isaac. Everything below is to be
tested offline or on paper first, with the success criteria written down before any run (as in phase 4).

## 1. Where stage 1 leaves us (honest starting point)

- Data and tooling are solid: 14,402 resolved markets, 21.3M trades, tape reconciles with Gamma volume for
  99.9% of markets, a pre-registered validation framework with planted insiders, 189 offline tests.
- Neither detector passed its pre-registered test. Within-event: 11.7% false-positive rate, 0 of 30 power
  cells, both placebos fail (winners already rank high 48 h before a reveal = convergence). One-shot screen:
  0 of 30 cells at 50% detection, flipped losers enter the tier more than twice as often as real winners.
  Zero wallets pass FDR.
- Therefore: **there is currently no demonstrated edge to trade.** A strategy built on the stage-1 flags
  would be building on a result that failed its own test. Stage 2 must first find out whether ANY signal in
  this data predicts price moves or outcomes beyond the market price, net of costs.
- Two look-ahead traps already found, to avoid in any strategy: (1) conditioning on the eventual winner (our
  within-event statistic is ex post and not tradable; a tradable version must select on the observed move,
  not the outcome); (2) news times defined by the price itself.

## 2. What the recent literature says (preprints; mostly not yet peer reviewed, verify before relying)

- **Calibration is conditional, not a fixed longshot rule.** A 2026 preprint on 353M trades over 429,000
  binary contracts (Kalshi and Polymarket) finds calibration depends on domain, time to resolution and trade
  size; political markets show underconfidence (prices compressed toward 50%) that replicates on Polymarket;
  about half of raw slope variation looks like estimation noise; explained variance falls from 87.3% in
  sample to 71.5% out of sample. The abstract does not address fees or exploitability. [Le 2026]
- **Three distinct layers of informed-trading detection**, argued to be complementary: an account-level
  sign-randomization skill test (3.14% of 1.72M Polymarket accounts "skilled winners"; 44% of those
  retained the label out of sample, vs about 10% for skilled mutual funds; a separate lifecycle insider flag
  with unknown precision), a wallet-market composite screen, and a per-market Information Leakage Score
  timed against an article-derived public-event timestamp. Documented cases are reported as systematically
  deadline-resolved. [Nechepurenko 2026; Gomez-Cram et al. 2026; Mitts and Ofir 2026; second-hand details]
- **Persistence of skill exists but copying is not shown to pay.** Coverage of on-chain studies reports a
  small skilled minority; evidence that following wallets is profitable is thin, and bots dominate
  latency-sensitive arbitrage. (Secondary sources only.)
- **Paulden-style methods**: price-vs-model discrepancy (a fair price from ELO or similar models, large
  deviations flag fixes) and, in a 2026 preprint, a state-space model of expected in-play volume with large
  positive unexplained deviations as the flag. I could verify only Paulden's abstract, not the full method.
- **Microstructure measures (VPIN, Kyle's lambda)** are standard tools for order-flow toxicity and price
  impact; VPIN's validity as a leading indicator is contested. I found no prediction-market application, so
  using them here is an extrapolation to test, not a known result.

## 3. Ways to improve the model (mapped to the failures we found)

| Failure in stage 1 | Method | Needs |
| --- | --- | --- |
| News time is price-based, so level D is circular and deadline/unscheduled markets have no independent time | **External event timestamps**: UMA resolution-proposal time (on-chain, independent of price), news-article timestamps (ILS-style), Gamma `eventStartTime` where present | New data: Polygon/UMA events (public RPC or subgraph, possibly an API key: a manual step), a news feed or GDELT |
| Wallet test on final outcomes has no power and mixes luck with skill | **Markouts**: a wallet's trade is informed if the price moves its way afterwards (1 h, 24 h, 7 d), size-weighted, vs a matched-market null. Independent of final outcome and of any news time | Existing data |
| Poisson-binomial win test is fragile | **Event-level sign-randomization** (permutation test conditional on which markets and prices a wallet chose), plus **empirical-Bayes shrinkage** of wallet skill (beta-binomial / hierarchical) and **out-of-sample persistence** (train period vs test period) as the headline | Existing data |
| Flags dominated by predictable convergence | **Ex-ante** features only: tradable versions of within-event ranks (rank of observed move, not of the winner), time-to-resolution, category, liquidity | Existing data |
| Single hand-built scores, no forecasting benchmark | **Forecasting benchmark**: does flow/skill information improve Brier score or log-loss over the market price alone? Gradient-boosted model, purged walk-forward CV with embargo, calibrated outputs, conformal intervals | Existing data |
| Level baseline needed independent times | **Microstructure features** (Kyle lambda per market from the taker tape, volume-bucket imbalance adapted to bounded prices, trade-size tails) used as model inputs, not detectors | Existing data |
| Co-trading clusters are weak | **Funding-chain graph** (who funded whom) to link wallets | Polygon transfer data, likely a key or paid API: manual step |
| No model-implied fair price | **Paulden-style fair-price models** for sports/election/data-release markets (Elo, polls, consensus forecasts) | External data; narrower scope |

Statistical hygiene for everything: cluster bootstrap by event, time-series splits only (never random),
pre-registered thresholds, multiple-testing control across every candidate signal and strategy tested.

## 4. Candidate strategies (hypotheses to test, none assumed to work)

Each is stated as the question a backtest must answer, with the costs included.

- **S1 Conditional calibration overlay.** Does the calibration curve conditional on (domain, horizon,
  trade size), fitted on earlier periods, beat the quoted price net of spread and fees? (Literature: effect
  is domain-specific and about half noise; after fees it may vanish.)
- **S2 Informed-flow following.** On markets where taker flow is imbalanced and toxic (Kyle lambda or flow
  imbalance high), does a position in the flow direction earn positive markout after the spread, at holding
  periods of hours to days? (Bots dominate sub-second horizons, so only slower horizons are plausible.)
- **S3 Persistent-skill signal.** Do net positions of wallets with out-of-sample-persistent skill predict
  forward returns once an observation delay (1 min, 1 h, 6 h) is imposed? Copying has no evidence of paying
  in the literature I found; the delay test is the honest one.
- **S4 Tradable convergence.** Our ex-post finding that winners rise before reveals is not tradable. The
  tradable form: buy the outcome with the largest observed rise over the prior 24 to 48 h and hold to
  resolution or exit at the reveal; is the forward return above the market's own drift, net of costs?
- **S5 Forecast-improvement trading.** If the benchmark in section 3 beats the price in Brier score
  out of sample, trade only where the model-implied edge exceeds a cost buffer, with fractional-Kelly
  sizing capped at a small share of observed liquidity.

Prior expectation, stated up front: prediction markets are competitive and our own validation was negative,
so the most likely outcome is that most or all of these show no net edge. A clear "no edge after costs"
result is a legitimate and publishable finding.

## 5. Backtest and decision protocol (fixed before running)

1. **Walk-forward only**: fit on windows before time T, trade after T, expanding or rolling, with an
   embargo of at least the market's resolution horizon. No random splits.
2. **Costs and frictions**: spread and impact estimated from the taker tape (we have no historical order
   book: spread is inferred from consecutive opposite-side taker fills; impact from the phase 4 model),
   platform fees from the CURRENT fee schedule (to be checked, not assumed), a position cap relative to
   observed volume, resolution lag and capital lockup.
3. **Metrics**: mean net return per dollar staked, hit rate, drawdown, cluster-bootstrap 95% intervals by
   event, results per category and per half-year.
4. **Pass criteria** (draft, to be frozen with Isaac): a strategy passes only if the lower 95% bound of net
   return per dollar is above zero in at least two disjoint time periods, after a multiple-testing
   correction across all strategies and parameters tried (e.g. White's reality check or SPA test, deflated
   Sharpe), and it survives a stress test with costs doubled and impact doubled. Anything else is reported as
   "no edge".
5. **Forward paper test**: a collector that records live prices and order-book snapshots (new, network, run
   by Isaac) and logs the signals' hypothetical trades without money for a fixed period (for example 8
   weeks) is the last gate before anything is considered for real use.
6. **Kill rules**: stop and report if costs consume the edge, if edge appears in one period only, or if
   capacity is below a stated minimum.

## 6. Roadmap

| Step | Content | Data | Effort | Gate |
| --- | --- | --- | --- | --- |
| 2a | Markout analysis and wallet skill v2 (sign-randomization, shrinkage, out-of-sample persistence) | Existing | 3 to 5 days | Does persistence beat a placebo out of sample? |
| 2b | Forecasting benchmark: flow, skill and microstructure features vs the price (Brier, log-loss), purged walk-forward CV | Existing | 3 to 5 days | Any out-of-sample gain over the price? If no, stop the strategy track |
| 2c | Independent event times: UMA proposal timestamps, news timestamps; redo level D, calibration and repeat test for deadline and unscheduled markets | New (network, maybe a key) | 1 to 2 weeks | Validation criterion fixed in advance (as with the burst method) |
| 2d | Strategy backtests S1 to S5 with the cost model, pre-registered pass criteria | Existing plus 2c | 1 to 2 weeks | Section 5 pass criteria |
| 2e | Forward paper-trading collector and a fixed paper period | New (network) | Collector 2 to 3 days, then weeks of waiting | No live capital unless Isaac decides after a pass |
| 2f | Optional: funding-chain graph for wallet linking | New (Polygon, key or paid API) | 1 to 2 weeks | Only if 2a shows wallet signal |

Steps 2a and 2b come first because they are cheap, use existing data, and decide whether the rest is worth
doing. If 2b shows no gain over price, the honest conclusion is that this data does not support a strategy.

## 7. What needs Isaac

- Approval of the pass criteria in section 5 (they must be frozen before any backtest).
- Any step needing keys, accounts or a paid API (2c, 2f) or a long-running fetch (2e): manual steps,
  never run by the agent without his go.
- Whether this stays a research and portfolio project (recommended) or is ever taken toward real money.
  If real money is ever considered: check eligibility and rules for his jurisdiction and the venue, do not
  trade on non-public information, and treat any result here as research, not advice. I cannot place trades.

## 8. Sources

- Le, N. A. (2026), calibration of 353M trades on Kalshi and Polymarket, arXiv:2602.19520.
- Nechepurenko (2026), comparison of three informed-trading detection layers, arXiv:2605.02287; ForesightFlow
  (SSRN 6687441). Gomez-Cram, Guo, Jensen and Kung (2026) and Mitts and Ofir (2026) as described there and in
  secondary coverage (full texts not checked).
- Paulden, T. (2016), "Smashing the Racket", Significance 13(3), 16-21 (abstract only); the 2014 Journal of
  Prediction Markets tennis fixing study; Winkelmann et al. (2026), arXiv:2605.30209.
- Easley, Lopez de Prado and O'Hara on VPIN and Kyle's lambda (background; contested validity).
- Press and data-blog coverage of Polymarket profitability and copy trading (secondary, conflicting figures).

---

## 9. FROZEN pre-registered pass marks for round 1 (2026-11-14T12:50:00Z, written BEFORE any stage-2 run; do not change)

These replace the 2a/2b criteria in the 'Stage 2: steps 2a and 2b' section of PLAN.md (written earlier the same day, nothing had been run). Approved by Isaac ("go for it") via the coordinator.

**Periods** (by a market's `closed_ts`, UTC): fit = closed on or before 2025-12-31; tune = 2026-01-01 to 2026-06-30; **sealed = 2026-07-01 onward, NOT opened this round** (code drops it at load and a test enforces it). Hyperparameters are fixed in advance (no search); "tune" is therefore an evaluation period this round.

**Costs** (all backtests): entry at the next traded price after a 5-minute delay (headline; 1 and 30 minutes also reported), taker fee per share = rate x p x (1 - p) with rate by category: geopolitics/world 0%, sports 3%, finance/politics/tech/mentions 4%, economics/culture/weather/other 5%, crypto 7% (SECONDARY SOURCE, to confirm with the venue; conservative max over a market's matching tags, unmatched = 5%). Position capped at a small share of recent market volume (cap and share of skipped trades reported).

**Strategy pass rule:** net return per dollar has a lower 95% bootstrap bound above zero when resampling whole EVENTS (1,000 resamples, seed 20261113), AND it beats its placebo, AND it stays positive at doubled costs (fees and delay cost doubled: fee x2 and the 30-minute entry). Otherwise "no edge".

**A. Tradable convergence (strategy 2):** multi-outcome scheduled-reveal events; at T-7d, T-72h, T-48h (T = independent news time) pick the leader from prices only: (i) largest log-odds rise over the prior window, and separately (ii) highest price; never the eventual winner. Buy and hold to resolution. Report win rate vs entry price (is the leader underpriced?), net return per dollar after costs, by horizon and event size (outcome count), placebo = random non-leader matched on price. Fit and tune periods only.

**B. Markouts and wallet skill v2:** for every taker trade the side-signed markout in the traded token's price at 5 min, 1 h, 24 h and to resolution (trades whose horizon falls after the last observation or resolution are dropped for that horizon). Wallet skill by event-level sign randomisation (2,000 draws) with empirical-Bayes shrinkage; classify on the fit period, report persistence on the tune period against a shuffled-wallet baseline; report the share of wallets classed skilled next to the published 3.14% and 44% persistence figures (definitions differ, so comparison is descriptive). PASS (skill persists): fit-top-decile tune-period mean 24 h markout per dollar > 0 with lower 95% bound above zero, above the fit-bottom decile, rank correlation fit vs tune skill positive with lower 95% bound above zero, and the shuffled baseline's 95th percentile below the observed top-decile value. Also report spread-adjusted markouts. Wallet-level output only under data/private.

**C. Forecasting benchmark (the stop gate):** at t = end_ts - H, H in {7d, 3d, 1d}, markets still open at t with at least 20 taker trades before t: add skilled-wallet net flow (skill from the fit period only), order-flow imbalance and impact per dollar (plus the pre-registered feature list in the earlier section) to the market price; compare out-of-sample Brier score and log-loss with the price alone using purged walk-forward validation (train on months before M with a 7-day embargo, test on M, tune period only). PASS: model log-loss below the price benchmark with the lower 95% event-cluster bootstrap bound of the improvement above zero in BOTH halves of the tune period (Jan-Mar and Apr-Jun 2026) for at least one horizon. If it fails, the strategy track STOPS (S2/S3/S5 not run).

**Decision rule:** a failed A, B or C is reported as a negative result in the README; nothing is retuned and the sealed period stays closed until a separate decision by Isaac.

---

## 10. Next data steps that need Isaac (NOT run; written 2026-11-14)

Nothing below has been run. Facts come from the public pages as read on 2026-11-14 (dataset card, docs); verify before use.

### 10.1 Hugging Face on-chain dataset (TimeSeventeen/Polymarket-v1, CC-BY-4.0, no login mentioned)
- What it is: Polymarket v1 on-chain logs (arXiv 2606.04217): `OrderFilled/` (raw maker-taker tape incl. relayer records), `daily_aligned/` (cleaned standard binary markets, 602M rows, 24 columns), `daily_aligned_multi/` (neg-risk multi-outcome, 144M rows, adds `neg_risk_market_id`), `CTF/` (preparations, splits, merges, resolutions, redemptions; 839M rows, about 8.5 GB). About 49 GB of Parquet in the main layers (52.7 GB listed in total), about 2.79B rows, 2022-11-21 to 2026-04-28, partitioned by day. Columns include `maker`, `taker`, `block_timestamp`, `price`, `usdc_amount`, `fee_usdc`, `taker_direction`, `condition_id`, `category`.
- Why it helps: (i) explicit `fee_usdc` lets us CHECK the fee schedule we only have from a secondary source; (ii) `CTF/resolutions` gives on-chain resolution timestamps, a candidate independent time for deadline and unscheduled markets (not the UMA proposal time, so confirm what it measures); (iii) maker and taker wallets are both labelled, which would complete the maker-side picture; (iv) coverage back to 2022 for more events.
- **Do not download it all**: it would take about 50 GB of the roughly 69 GB free on C: (our data folder is 19 GB). Filter plan, in order, each step a go/no-go:
  1. List sizes without downloading (about a minute): in a terminal from the project folder, `.venv\Scripts\python.exe -m pip install huggingface_hub`, then run a short snippet that walks `HfApi().list_repo_tree("TimeSeventeen/Polymarket-v1", repo_type="dataset", recursive=True)` and sums file sizes per top-level folder and per month. Share the totals.
  2. `CTF/resolutions` only (expected small, hundreds of MB at most; confirm in step 1): download with `snapshot_download(..., allow_patterns=["CTF/resolutions/*"])` or read it remotely with DuckDB `hf://datasets/TimeSeventeen/Polymarket-v1/...`.
  3. A one-week sample of `daily_aligned` (and `daily_aligned_multi`) inside our fit period, to compare against our tape (row counts per market, maker/taker rows, fee_usdc by category): expect one to a few GB.
  4. Only then decide on a filtered pull: select needed columns, keep rows whose `condition_id` is in our 14,402-market list, write to our own Parquet (about a few GB), delete the raw download. Read with DuckDB or lazy Polars, never load whole layers.
- Overlap warning: coverage ends 2026-04-28, so it cannot extend the tune period past April 2026 and it cannot touch the sealed period (2026-07-01 onward) at all; do not use it to peek.

### 10.2 Polygonscan key for funding links (needed only if step 2a shows a wallet signal: step 2f)
- Free keys exist and the free tier is documented at 5 calls per second and 100,000 calls per day (Standard 10/s, Advanced 20/s, Professional 30/s). Isaac creates the account and key himself (an account and credentials: never typed or seen by the agent) at the Polygonscan API-keys page; check whether the newer multi-chain Etherscan key now applies instead (unverified).
- Store it as an environment variable in his own shell (`POLYGONSCAN_API_KEY`), not in any repo file.
- Cost estimate (to verify): one to three token-transfer calls per candidate wallet for the first inbound USDC funding transfer; for the roughly 8,000 candidates that is 8,000 to 24,000 calls, which fits one day's free quota (about 30 to 90 minutes at 5/s). I have not verified the exact endpoint names or parameters for token transfers: confirm in the Polygonscan docs before writing the collector.
- The collector (restartable script, same pattern as `first_seen_candidates.py`) is not written yet; it should be written and tested offline before he runs anything.

### 10.3 Live order-book recorder (forward paper test, step 2e)
- Script written and tested offline (`scripts/orderbook_recorder.py`, `tests/test_orderbook_recorder.py`), NOT run. It polls the public batch endpoint `POST https://clob.polymarket.com/books` (500 tokens per request, no key needed per the docs I read; the page gives no rate limit, so the script waits at least 1 second between requests and honours Retry-After) every N seconds and appends top-of-book rows (best bid and ask, sizes, 5-cent depth, last trade price, tick size, hash) to hourly Parquet parts under `data/orderbook/` (gitignored). It places no orders and needs no credentials.
- Before starting it Isaac needs a tokens file: one token id per line for the OPEN markets worth watching (for example the YES and NO tokens of the open markets in the categories we backtest; a list of 200 to 500 tokens is enough to start). Deriving that list needs a Gamma call for open markets; we have only closed markets stored, so the list has to be produced by a small separate step (to be written and tested offline) or typed in by hand.
- Command to start in a terminal, from `06_polymarket_insider_detection`, once the tokens file exists: `.venv\Scripts\python.exe scripts\orderbook_recorder.py --tokens-file data\manual\watch_tokens.txt --interval 60 --hours 168`. About 500 tokens at one poll per minute is roughly 1 request per minute, which is far below any plausible limit; disk is roughly 20 to 50 MB per day compressed (an estimate). The PC must stay on and awake for the whole period; Ctrl+C is safe and the script can be restarted at any time.
