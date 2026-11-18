# Labelled cases on every score (phase 4D): HELD OUT, never tuned on

Aggregate/market-level only: no wallet addresses; the two labelled wallets are referred to as `maduro_labelled_wallet` and `machado_jump_wallet` (addresses stay in `data/private/`). Full numeric table: `results/validation_labelled_cases.csv`. **Every number below is HELD OUT: none of these cases set a threshold, a baseline or a rule; they are reported after the fact.**

## How to read it

* Rank 1 is the most extreme. 'Percentile' is 100 x (N - rank + 1) / N within the pool (100 = most extreme).
* Within-event: pool = the universe markets that have the statistic in market_scores.csv (event winners, multi and binary groups; wider than the 809-event validation population). Flags if the 24 h winner percentile is >= 0.95 (the validation's detection rule). The validation found a 11.7% false-positive rate for exactly this rule on untouched events, and both placebos fail.
* One-shot: best position of the market among 249203 real valid-cutoff winning positions of non-excluded wallets; tier = score >= 0.7721 (99th percentile) and USD >= 100. Flags only when a tier position has a scheduled (independent) cutoff; a jump-start cutoff is chosen from the price move itself (outcome-selected), so such a tier membership is shown but not interpreted.
* Price measures (level D, late-window D24, max_jump, flow before the move, Fisher-combined): rank/N in the universe pool; `*` = top 1% of that pool. They are documented negative or untested results, not packaged detectors, so 'flags' there is descriptive.
* CANNOT_TEST means the measure does not exist for that case or cannot be interpreted; the reason is in the CSV (`why_*` columns) and summarised below.

## Verdict counts over the 75 labelled markets

* Within-event: FLAGS 1, MISSES 0, CANNOT_TEST 74
* One-shot: FLAGS 2, MISSES 3, CANNOT_TEST 70
* Price measures (descriptive): FLAGS 0, MISSES 31, CANNOT_TEST 44

## Summary (read with the validation results)

* **Nothing here is evidence of a leak.** The labelled insider-suspected cases are two markets and two wallet classes; the context cases (Google Year in Search, Axiom/ZachXBT, Bubblemaps military cluster) have no confirmed informed trader at all, so a flag or a miss on them carries no information about the detectors beyond their base rates.
* **Nobel Peace Prize 2025 (insider-suspected market)**: within-event 24 h winner percentile 0.976 (rank 21/1670) -> FLAGS. That is what the validation says to expect of a winner that was already high in the pre-reveal ranking (convergence): 11.7% of untouched events do the same, and the 48-hour-earlier placebo fails. The level-D and combined scores do not put it in the top 1% (D 894/4070, combined 105/5163). The hand audit classifies the move as unclear (the buying was balanced across many wallets).
* **Maduro (insider-suspected market)**: one-shot FLAGS (best position rank 1/249203); within-event and the price measures cannot be tested (out-of-universe, no sibling data; price measures uninformative, see the CSV). The Maduro market holds 27 of the 875 tier positions and 8 of the 10 highest scores. The label-flip placebo shows flipped losers entering the tier 0.74% against 0.35% for real winners (more than twice as often) while passing the 2% rule as written, so tier membership by itself does not separate informed from uninformed positions.
* **maduro_labelled_wallet**: FLAGS (in the tier (rank 13 of 249203, percentile 100.00; cutoff independent_news); watchlist tier: one_shot_anomalous). **machado_jump_wallet**: FLAGS (in the tier (rank 251 of 249203, percentile 99.90; cutoff independent_news); watchlist tier: one_shot_anomalous). A fresh wallet buying a low-priced winner shortly before the news is exactly what the score rewards. Both are in the tier with a scheduled cutoff, but the tier holds 875 real positions and flipped-label losers enter it more than twice as often as real winners, so this is a consistency check on two publicly reported wallets, not a validation, and it cannot show that the screen separates informed from lucky buyers.
* **Context cases**: Axiom and Bubblemaps candidates are untestable (short history, uncertain mapping, no independent time); the Google Year in Search markets have no independent time and mostly no computable measure. Where a number is shown for them it is informational. The one-shot tier does contain positions in the Iran-strike markets, but with jump-start cutoffs only, which is outcome-selected.
* **Compatibility**: among the informative labelled rows the within-event test flags the Nobel winner market (a winner that already ranked at the top before the reveal, which the placebos show is common) and the one-shot tier contains the Maduro market, the Nobel market and the two named wallets. All of this is compatible with convergence plus the score rewarding low-priced, fresh, late winners; it is not evidence of leaks. Two insider-suspected markets cannot establish detection power, and the validation already shows that neither detector meets its pre-registered criterion.

## Per-case table (HELD OUT)

Cells: value (rank/N; percentile). Within-event columns are the winner's percentile in its event and the log-odds difference to the median loser. Order: insider-suspected first, then candidates, then context.

| case | label | row | market | WE pct24 | WE pct12 | WE diff24 | one-shot best | D | D24 | max_jump | flow | combined | WE | one-shot | price |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| maduro_removal_jan_2026 | insider_suspected | market | Maduro out by January 31, 2026? | NA | NA | NA | 1/249203; 100.00%; tier yes | 0.000 (4028/4070) | 0.000 (3770/4070) | 0.024 (9689/12891) | 0.695 (163/1649) | 0.518 (1572/5163) | CANNOT_TEST | FLAGS | CANNOT_TEST |
| maduro_removal_jan_2026 | insider_suspected | maduro_labelled_wallet | Maduro out by January 31, 2026? | NA | NA | NA | 13/249203; 100.00%; tier yes | NA | NA | NA | NA | NA | CANNOT_TEST | FLAGS | CANNOT_TEST |
| nobel_peace_2025 | insider_suspected | market | Will María Corina Machado win the Nobel Peace Prize in 2025? | 0.976 (21/1670; 99%) | 0.976 (21/1684; 99%) | 6.65 (22/1670; 99%) | 98/249203; 99.96%; tier yes | 0.027 (894/4070) | 0.192 (253/4070) | 0.428 (929/12891) | 0.480 (954/1649) | 2.539 (105/5163) | FLAGS | FLAGS | MISSES |
| nobel_peace_2025 | insider_suspected | machado_jump_wallet | Will María Corina Machado win the Nobel Peace Prize in 2025? | NA | NA | NA | 251/249203; 99.90%; tier yes | NA | NA | NA | NA | NA | CANNOT_TEST | FLAGS | CANNOT_TEST |
| axiom_expose_feb_2026 | case_candidate, case_context | market | Will ZachXBT release his investigation on February 26, 8-9am? | NA | NA | NA | NA | NA | NA | 0.930 (33/12891) * | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_candidate, case_context | market | ZachXBT investigation post over 10M views on day one? | NA | NA | NA | NA | NA | NA | 0.010 (10839/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_candidate, case_context | market | ZachXBT investigation post over 14M views on day one? | NA | NA | NA | NA | NA | NA | 0.282 (2140/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_candidate, case_context | market | ZachXBT investigation post over 25M views on day one? | NA | NA | NA | NA | NA | NA | 0.010 (10839/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_candidate, case_context | market | ZachXBT investigation post over 30M views on day one?- | NA | NA | NA | NA | NA | NA | 0.010 (10959/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_candidate, case_context | market | ZachXBT investigation post over 8M views on day one? | NA | NA | NA | NA | NA | NA | 0.190 (3468/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| bubblemaps_military_cluster | case_candidate | market | Iran Strike on Israel by February 28? | NA | NA | NA | NA | NA | NA | 0.908 (52/12891) * | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| bubblemaps_military_cluster | case_candidate | market | Iran strike on US military by February 28? | NA | NA | NA | 1538/249203; 99.38%; tier no | NA | NA | 0.433 (887/12891) | 0.493 (885/1649) | 0.660 (1262/5163) | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| bubblemaps_military_cluster | case_candidate | market | Israel strikes Iran by February 28, 2026? | NA | NA | NA | NA | NA | NA | 0.090 (5997/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| bubblemaps_military_cluster | case_candidate | market | US strikes Iran by February 28, 2026? | NA | NA | NA | 17/249203; 99.99%; tier yes | NA | NA | 0.100 (5515/12891) | 0.616 (348/1649) | 0.444 (1775/5163) | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| bubblemaps_military_cluster | case_candidate, case_context | market | Will US or Israel strike Iran on March 1, 2026? | NA | NA | NA | NA | NA | NA | 0.204 (3247/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| bubblemaps_military_cluster | case_candidate, case_context | market | Will US or Israel strike Iran on March 2, 2026? | NA | NA | NA | NA | NA | NA | 0.193 (3410/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| bubblemaps_military_cluster | case_candidate, case_context | market | Will US or Israel strike Iran on March 3, 2026? | NA | NA | NA | NA | NA | NA | 0.150 (4314/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| bubblemaps_military_cluster | case_candidate, case_context | market | Will US or Israel strike Iran on March 4, 2026? | NA | NA | NA | NA | NA | NA | 0.100 (5541/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | No release by February 27 | NA | NA | NA | NA | NA | NA | 0.455 (762/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | Will ZachXBT release his investigation on February 26, 2-3pm? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | Will ZachXBT release his investigation on February 26, 3-4pm? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | Will ZachXBT release his investigation on February 26, 4-5am? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | Will ZachXBT release his investigation on February 26, 6-7am? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | Will ZachXBT release his investigation on February 26, 7-8am? | NA | NA | NA | NA | NA | NA | 0.335 (1596/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | Will ZachXBT release his investigation on February 27, 2-3pm? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | Will ZachXBT release his investigation on February 27, 9-10pm? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | ZachXBT investigation post over 12M views on day one? | NA | NA | NA | NA | NA | NA | 0.330 (1635/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | ZachXBT investigation post over 18M views on day one? | NA | NA | NA | NA | NA | NA | 0.000 (12279/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | ZachXBT investigation post over 2M views on day one? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | ZachXBT investigation post over 4M views on day one? | NA | NA | NA | NA | NA | NA | 0.019 (10124/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| axiom_expose_feb_2026 | case_context | market | ZachXBT investigation post over 6M views on day one? | NA | NA | NA | NA | NA | NA | 0.110 (5226/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Andrew Tate rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Andy Byron rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.035 (8760/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Bad Bunny rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.025 (9521/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Benjamin Netanyahu rank in Google’s Top 5 Most Searched People of | NA | NA | NA | NA | NA | NA | 0.019 (10176/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Beyoncé rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.004 (11638/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Bianca Censori rank in Google’s Top 5 Most Searched People of 202 | NA | NA | NA | 1795/249203; 99.28%; tier yes | NA | NA | 0.649 (288/12891) | 0.371 (1380/1649) | 1.221 (557/5163) | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Billie Eilish rank in Google’s Top 5 Most Searched People of 2025 | NA | NA | NA | NA | NA | NA | 0.000 (12279/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Catherine Princess of Wales rank in Google’s Top 5 Most Searched  | NA | NA | NA | NA | NA | NA | 0.003 (11766/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Charlie Kirk be ranked #1 globally in Google’s Year in Search 202 | NA | NA | NA | NA | NA | NA | 0.030 (9065/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Dharmendra be ranked #1 globally in Google’s Year in Search 2025  | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Diane Keaton be ranked #1 globally in Google’s Year in Search 202 | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Diddy rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.020 (9887/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Diogo Jota be ranked #1 globally in Google’s Year in Search 2025  | NA | NA | NA | NA | NA | NA | 0.000 (12279/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will D’Angelo be ranked #1 globally in Google’s Year in Search 2025 Pa | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Elon Musk rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.030 (9062/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will George Foreman be ranked #1 globally in Google’s Year in Search 2 | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Giorgio Armani be ranked #1 globally in Google’s Year in Search 2 | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Greta Thunberg rank in Google’s Top 5 Most Searched People of 202 | NA | NA | NA | NA | NA | NA | 0.034 (8872/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Imane Khelif rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Isabella Ladera rank in Google’s Top 5 Most Searched People of 20 | NA | NA | NA | NA | NA | NA | 0.030 (9065/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Jair Bolsonaro rank in Google’s Top 5 Most Searched People of 202 | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Jimmy Carter rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Joe Biden rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Justin Trudeau rank in Google’s Top 5 Most Searched People of 202 | NA | NA | NA | NA | NA | NA | 0.148 (4337/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Kamala Harris rank in Google’s Top 5 Most Searched People of 2025 | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Kanye West / Ye rank in Google’s Top 5 Most Searched People of 20 | NA | NA | NA | NA | NA | NA | 0.025 (9521/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Katy Perry rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.072 (6645/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Kendrick Lamar rank in Google’s Top 5 Most Searched People of 202 | NA | NA | NA | 11983/249203; 95.19%; tier no | NA | NA | 0.050 (7733/12891) | 0.459 (1032/1649) | 0.174 (2880/5163) | CANNOT_TEST | MISSES | MISSES |
| google_year_in_search_2025 | case_context | market | Will Kylian Mbappé rank in Google’s Top 5 Most Searched People of 2025 | NA | NA | NA | NA | NA | NA | 0.000 (12279/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Lionel Messi rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.014 (10521/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Luigi Mangione rank in Google’s Top 5 Most Searched People of 202 | NA | NA | NA | 37071/249203; 85.12%; tier no | NA | NA | 0.060 (7229/12891) | 0.484 (923/1649) | 0.226 (2584/5163) | CANNOT_TEST | MISSES | MISSES |
| google_year_in_search_2025 | case_context | market | Will Mark Carney rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will María Corina Machado rank in Google’s Top 5 Most Searched People  | NA | NA | NA | NA | NA | NA | 0.094 (5818/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Mohammed Sinwar be ranked #1 globally in Google’s Year in Search  | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Pope Francis be ranked #1 globally in Google’s Year in Search 202 | NA | NA | NA | NA | NA | NA | 0.030 (9302/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Pope Francis rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Robert Redford be ranked #1 globally in Google’s Year in Search 2 | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Sam Altman rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.155 (4146/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Taylor Swift rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.040 (8370/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Timothée Chalamet rank in Google’s Top 5 Most Searched People of  | NA | NA | NA | NA | NA | NA | 0.000 (12279/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Travis Kelce rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | NA | NA | NA | 0.000 (12279/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Tyler Robinson rank in Google’s Top 5 Most Searched People of 202 | NA | NA | NA | 1582/249203; 99.37%; tier yes | NA | NA | 0.040 (8370/12891) | 0.274 (1572/1649) | 0.066 (3856/5163) | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Val Kilmer be ranked #1 globally in Google’s Year in Search 2025  | NA | NA | NA | NA | NA | NA | NA | NA | NA | CANNOT_TEST | CANNOT_TEST | CANNOT_TEST |
| google_year_in_search_2025 | case_context | market | Will Vladimir Putin rank in Google’s Top 5 Most Searched People of 202 | NA | NA | NA | NA | NA | NA | 0.000 (12279/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will Volodymyr Zelenskyy rank in Google’s Top 5 Most Searched People o | NA | NA | NA | NA | NA | NA | 0.040 (8565/12891) | NA | NA | CANNOT_TEST | CANNOT_TEST | MISSES |
| google_year_in_search_2025 | case_context | market | Will d4vd rank in Google’s Top 5 Most Searched People of 2025? | NA | NA | NA | 2711/249203; 98.91%; tier no | NA | NA | 0.076 (6502/12891) | 0.426 (1183/1649) | 0.191 (2769/5163) | CANNOT_TEST | MISSES | MISSES |

## Reasons, per case

### maduro_removal_jan_2026

* Within-event: out-of-universe market: no sibling-outcome price data | wallet-level class: the within-event test scores markets, not wallets
* One-shot: 27 position(s) in the tier with a scheduled (independent) cutoff | in the tier (rank 13 of 249203, percentile 100.00; cutoff independent_news); watchlist tier: one_shot_anomalous
* Price measures: out-of-universe market ranked against the universe pool; D = 0 and max_jump 0.024 are not informative: the news time is a documented press time (about 06:00 UTC), the recorded price move starts at 03:03 UTC and the tape starts 2025-12-22, so the 'pre-news' path is ill-defined | wallet-level class: price measures score markets, not wallets

### nobel_peace_2025

* Within-event: 24 h winner percentile 0.976 >= 0.95 (the test's false-positive rate on untouched events is 11.7%) | wallet-level class: the within-event test scores markets, not wallets
* One-shot: 31 position(s) in the tier with a scheduled (independent) cutoff | in the tier (rank 251 of 249203, percentile 99.90; cutoff independent_news); watchlist tier: one_shot_anomalous
* Price measures: not in the top 1% on any computable price measure (not packaged detectors) | wallet-level class: price measures score markets, not wallets

### axiom_expose_feb_2026

* Within-event: no independent reveal time (the pre-reveal window needs one); Axiom / ZachXBT candidates (x5) | out-of-universe market: no sibling-outcome price data; Axiom / ZachXBT candidates
* One-shot: Axiom / ZachXBT candidates (uninterpretable here) (x6)
* Price measures: values shown for information only; Axiom / ZachXBT candidates (x6)

### bubblemaps_military_cluster

* Within-event: out-of-universe market: no sibling-outcome price data; Bubblemaps military-cluster candidates (x8)
* One-shot: Bubblemaps military-cluster candidates (uninterpretable here) (x8)
* Price measures: values shown for information only; Bubblemaps military-cluster candidates (x8)

### axiom_expose_feb_2026

* Within-event: no independent reveal time (the pre-reveal window needs one); Axiom / ZachXBT candidates (x13)
* One-shot: Axiom / ZachXBT candidates (uninterpretable here) (x13)
* Price measures: no independent time (level D needs one); short history (x7) | values shown for information only; Axiom / ZachXBT candidates (x6)

### google_year_in_search_2025

* Within-event: no independent reveal time (the pre-reveal window needs one) (x46)
* One-shot: no valid-cutoff winning position (no independent news time and no price jump start) (x41) | best position not in the tier (cutoff jump_start) (x3) | 1 tier position(s) exist only with a jump-start cutoff (chosen from the price move itself: outcome-selected), so the tier is not interpretable | 2 tier position(s) exist only with a jump-start cutoff (chosen from the price move itself: outcome-selected), so the tier is not interpretable
* Price measures: not in the top 1% on any computable price measure (not packaged detectors) (x30) | no independent time (level D needs one) (x16)

## Caveats

* Two insider-suspected markets and two wallet classes cannot calibrate or validate anything; nothing was tuned on them.
* Maduro and the context cases rely on press-documented times (hour precision) rather than a feed.
* Ranks of out-of-universe markets are computed against the universe pool, not inside it.
* A `NA` is a missing measure, not a zero; CANNOT_TEST is not a miss.
