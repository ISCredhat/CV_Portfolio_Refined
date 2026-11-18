# Hand audit of the detectors' extremes (phase 4E)

Aggregate-only: counts, shares and market questions; no wallet addresses and no event or wallet identifiers. Row-level reasons are in `data/private/audit_within_event.csv` and `data/private/audit_oneshot.csv` (the second contains wallet addresses and stays in `data/private/`).

**This is a descriptive audit with no ground truth.** The classes are produced by a fixed rule set over offline data (see the criteria), written before looking at the results. A human reading the news or the source feed would classify some rows differently. A class says what the offline price, trade and wallet data look like; it is not evidence about any person or wallet, and 'plausible leak' means only that the offline data cannot exclude one.

## The two lists

* **Within-event, top 30**: real multi-outcome events with >= 3 comparable outcomes and a computed statistic (809 events), ordered by the winner's 24 h within-event percentile (descending), then the winner-minus-median-loser log-odds difference `diff_24h`, then the number of comparable outcomes (all descending), then the winner's condition id ascending (only to make the order deterministic). The percentile is a mid-rank, so large events tie near 0.97-0.98 and the tie-break decides the order. Percentile values in the list: 0.964 to 0.980.
* **One-shot, top 30**: real winning positions with a valid cutoff of non-excluded wallets, USD >= 100 (the watchlist's size rule), outside the Maduro market (which holds 214 of the 249203 valid positions and the top ranks), ordered by the one-shot score (descending), then USD (descending), then wallet (ascending, deterministic only). Tier threshold (99th percentile of valid-cutoff scores): 0.7721. Cutoff method of the list: independent_news 6, jump_start 24. A `jump_start` cutoff is the start of the price move itself, so those rows are outcome-selected and the move follows the cutoff by construction.

## Mix of classes

| class | within-event n | within-event share | one-shot n | one-shot share |
|---|---|---|---|---|
| plausible_leak | 0 | 0% | 2 | 7% |
| predictable_convergence | 15 | 50% | 1 | 3% |
| public_news_reaction | 8 | 27% | 3 | 10% |
| unclear | 7 | 23% | 24 | 80% |
| total | 30 | 100% | 30 | 100% |

## Criteria (fixed in advance, same for every row)

Price path: the winner's forward-filled price on an hourly grid, built from the Gamma price history (3-hourly for most events) plus the winner's YES-token taker trades where they exist. A path with observations more than 6 h apart cannot show a step and is `unclear`. A **step** is the largest rise over a short lag (>= 2 h, or the observation spacing); it is **sharp** when it is >= 0.10 probability points and >= 60% of the rise over the whole window.

* `predictable_convergence`: the winner rose by less than 0.02 while already a favourite (>= 0.5); or the rise is gradual (no sharp step), including a favourite firming up; or it was already rising strongly in the 48 h before the window. One-shot: a gradual rise after the entry, or a favourite bought with no visible move.
* `public_news_reaction`: a sharp step with at least one visible public trigger: a related outcome of the same event moved (either direction) before the winner's step, other events with the same reveal time stepped in the same hours (>= 3 peers; for jump-start cutoffs, >= 3 other markets repricing within 2 h), a supplied known release time, or the step within 12 h of the market opening. One-shot: the entry came after a rise of >= 0.10 in the preceding 24 h.
* `plausible_leak`: a sharp step that ends at least 3 h before the end of the window, no public trigger above, not a polls/results/leaderboard/earnings-type market, AND tape evidence of one-sided buying (>= 3 trades, net buy ratio >= 0.8) that is concentrated (top-3 wallets >= 50% of buy USD) or from fresh wallets (>= 50% of known-age buy USD from wallets first seen <= 14 days earlier). One-shot: a fresh (<= 14 days) or single-market wallet bought before a sharp step, with the wallet at least 25% of winner-side buy USD within 3 h of its entry, and no trigger above. Without tape the row stays `unclear`.
* `unclear`: everything else, including sparse or missing price data, steps in the last 3 h before the reveal (cannot be separated from the first public results), sharp steps in markets of a type with frequent public intermediate data, one-shot entries that preceded a step also made by other markets (not separable from early public information), and rows where the winner did not visibly move at all (the percentile or score then comes from the siblings, the entry price, the wallet's freshness or the size).

## Examples (market questions only)

### Within-event

* **plausible_leak**: (none)
* **predictable_convergence**: "Will PPSO win the most seats in the 2026 Costa Rican Legislative Assembly election?"; "Will any other company have the best AI model on LiveBench (Mathematics) at the end of October 2026?"; "Will Anthropic have the best AI model at the end of September 2026?"; "Will Marcelo Castro win the most votes in the 2026 Piauí Senate election?"
* **public_news_reaction**: "Will Julio Cesar win the second-most votes in the 2026 Piauí Senate election?"; "Will Augusto Cury finish in third place in the first round of the 2026 Brazilian presidential election?"; "Will Marília Campos win the second-most votes in the 2026 Minas Gerais Senate election?"; "Will Luiz Inácio Lula da Silva finish in second place in the first round of the 2026 Brazilian presidential election?"
* **unclear**: "Will María Corina Machado win the Nobel Peace Prize in 2025?"; "Will Rodrigo Paz win the 2025 Bolivia presidential election?"; "Will Pau Cubarsí win the Young Player Award at the 2026 FIFA World Cup?"; "Will Laura Virginia Fernández Delgado win the 2026 Costa Rican presidential election?"

### One-shot

* **plausible_leak**: "Will the Bank of Israel decrease the Bank of Israel Interest Rate after the September decision?"; "Pump.fun airdrop in Q3 2025?"
* **predictable_convergence**: "US strike on Syria by December 31?"
* **public_news_reaction**: "Will Trump admin release any more Epstein related files by January 31?"; "Will Bulgaria be in the top 5 at Eurovision 2026?"; "US strikes Iran by February 28, 2026?"
* **unclear**: "Will Finland be in the top 5 at Eurovision 2026?"; "Will Trump’s approval rating be 44.5–44.9% on July 4?"; "US strikes Iran by February 28, 2026?"; "US strike on Syria by December 31?"

## Notes on the mix

* Within-event: 8 of the 8 public-news rows are events that share a reveal time with >= 3 other events whose winners stepped in the same hours (election markets that all reprice together on the same count or poll), not an individual market's signal.
* Within-event `unclear`: 4 rows have price observations 12 h apart (no step can be seen), 1 are sharp steps in poll/results/leaderboard-type markets, 1 are sharp steps in the last 3 h, 1 are sharp steps whose tape is balanced or spread over many wallets, 0 show no visible rise (the percentile comes from siblings).
* Within-event `predictable_convergence`: 6 are favourites already high with no visible move, 6 are gradual or favourite-firming rises, 3 were already rising before the window.
* One-shot: 24 of 30 rows have a jump-start (outcome-selected) cutoff; 22 entered before a sharp step that other markets or related outcomes also made (consistent with public news; whether it was public at the entry cannot be judged); 2 show no price move before the window end.
* No row's class rests on a news or social-media source: a row labelled `plausible_leak` is one the offline rules could not explain by public triggers, nothing more.

## What the offline data cannot decide

* Whether a step followed public information: no news, social-media or source feed was read; the only public-trigger evidence is another market moving first, a market-wide repricing at the same hour, a supplied release time (none exists offline) and the market opening time.
* Whether the buyer knew anything: a one-sided, fresh-wallet buy before a step is also what an early, correct but public-information buyer looks like. Trade-time resolution is the tape's; the Gamma price history is 3-hourly for most events, so 'sharp' means 'within about 3 h' there.
* Wallet history is partial: first_seen is known for only part of the wallets (store backfill and top-up); unknown ages are not treated as fresh.
* Events with many siblings put most of the percentile on the siblings: with >= 10 comparable outcomes and stale low-probability siblings (a zero median loser move), a winner that merely moves a little ranks at the top. The within-event validation found that 53.7% of such real winners already exceed percentile 0.95; the placebo shows the same ranking two days earlier.
* Jump-start cutoffs (part of the one-shot list) are chosen from the price path itself, so the 'before any visible public trigger' test is weakest exactly there.

## Limitations

* No ground truth. A human reading news would classify differently; the mix is a description of what these lists look like offline, not an estimate of how many rows are real leaks.
* The classes are not evidence about any person, wallet or market operator. 'Anomalous' is a statistical statement.
* Rule thresholds are fixed, simple and coarse; different thresholds move rows between classes (the default for an undecidable row is `unclear`, which is therefore expected to be large).
* Tiny lists (30 each): shares carry large sampling noise and say nothing about the rest of either ranking.
* The within-event detector failed its own criterion (false-positive rate 11.7% > 10%, both placebos fail) and the one-shot screen reaches no cell with >= 50% detection; this audit does not rescue either.
