# Phase 3 summary: anomalous wallet detector (aggregate only)

These are statistical flags of anomalous trading, not findings of wrongdoing. No wallet addresses or per-wallet rows appear here; wallet-level output is in data/private/.

## Coverage
- Positions built: 1048214
- Wallets tested: 200778
- Wallets passing FDR (before non-directional exclusion): 0

## Non-directional exclusions (rules, thresholds, counts)
- Wallets screened: 652383; excluded (any rule): 102032
- `contract`: 0 wallets. Rule: known exchange/CTF/adapter/USDC/zero addresses; first_seen null or < 2020-01-01; n_markets_global >= 3000
- `two_sided`: 98110 wallets. Rule: >= 3 taker markets and >= 30% of them with both outcome tokens bought or a buy/sell round trip on one token
- `maker`: 5400 wallets. Rule: >= 50 maker-side fills in >= 5 markets and maker fills >= 50% of all fills (activity), AND two-sided / inventory-neutral over its maker markets: gross-weighted |net|/gross <= 0.3 and >= 50% of them with minority-side volume share >= 25%
- `broad_no_edge`: 1500 wallets. Rule: >= 30 positions in store or n_markets_global >= 500, with mean(won - entry price) < 0.02

## Win-test p-value histogram (all tested wallets, 10 equal bins on [0,1])
- counts: 1472, 1957, 2625, 3197, 3717, 5871, 6880, 9022, 10910, 155127
- p < 0.05: 853; p < 0.01: 108 of 200778

## One-shot scoring (winning pre-cutoff positions)
- Scored winning positions: 661270
- Score distribution: n=661270 min=0 q25=0 median=0.0391 q75=0.373 q90=0.519 q99=0.731 max=0.954
- Positions with >=1 imputed component: 205011
- Winning positions with no valid cutoff (no proximity credit, screened only): 283665
- Winning positions with wallet first_seen missing: 353657 of 661270
- Proximity = 1 for entries <= 72 h before the cutoff, exp(-(h-72)/72) beyond; freshness = wallet age at entry (see oneshot.py).

## Watchlist tier sizes
- repeat_anomalous: 0
- one_shot_anomalous: 860
- watch: 17781

## Co-trading clusters
- Clusters: 8294; wallets in clusters: 35708
- Size distribution: n=8294 min=2 q25=2 median=2 q75=4 q90=7 q99=32 max=591
- Largest cluster size: 591; oversized (> cap, not calibrated): 14
- Pooled-cluster calibration tests: 5412; p < 0.05: 42

## Caveats
- Power is limited by the number of calibration-eligible positions (independent news time only); most wallets have one or two positions, so the repeat test covers few wallets.
- Positions without a valid cutoff and jump-start positions are screened by the one-shot score only (the jump start is outcome-selected).
- Funding-chain linkage is a stub (`link_by_funding`); clusters use co-trading only.
- Scale run: positions by cutoff method: independent_news=445148, jump_start=120260, none=482806; markets with an independent news time: 4250; markets with a jump start: 2919.
- Repeat test and calibration: 200778 wallets with >= 1 independent-news position, 14850 eligible (>= 5), BH passes 0 (exact tail) / 0 (bootstrap) / 0 (both).
- Maker rule coverage: the all-rows (maker+taker) tape covers 14403 of 14413 taker-tape markets (99.9%); INCOMPLETE: maker fills are only visible in covered markets, so the maker exclusion under-counts and is not final until the all-rows tape is loaded.
- Stage timings (s): load inputs=0.4, positions (build)=3.9, calibration (isotonic, cross-fitted)=2.4, wallet tests (Poisson-binomial + bootstrap) + FDR=4.8, filters (two-sided, maker, contract, broad)=29.8, one-shot scoring=0.9, co-trading clusters=4.2, cluster calibration=0.5, watchlist=0.7
- Maker rule redesign (two-sided, inventory-neutral; see filters.py): activity-only rule flagged 9453 wallets, the new rule 5400 (overlap 5400); wallets excluded overall 104971 under the old rule vs 102032 now; 2939 wallets move from excluded to included.
- Net-to-gross histogram (gross-weighted |net|/gross over maker markets, wallets passing the maker activity screen; threshold fixed a priori at <= 0.3): 0.0-0.1: 3464, 0.1-0.2: 1197, 0.2-0.3: 944, 0.3-0.4: 801, 0.4-0.5: 622, 0.5-0.6: 507, 0.6-0.7: 495, 0.7-0.8: 413, 0.8-0.9: 353, 0.9-1.0: 657
- Cluster link level (shared markets needed; oversized components are re-linked upward from 2): 2: 6720, 3: 952, 4: 353, 5: 90, 6: 52, 7: 56, 8: 71
- first_seen missing for 353657 of 661270 winning positions (store backfill covers only part of the wallets; the first_seen top-up fills the candidate set).
