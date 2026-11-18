# Polymarket insider-trading detector: working plan

This file holds the parts that phase 1 needs locally: scope, the work split between the two builders, and the shared interfaces both of them code against.

## Scope reminder

- Python package `pmdetect` that collects resolved, thin Polymarket event markets and their trade tapes into a local DuckDB store.
- Later phases will score markets (a discrepancy score against a peer-market baseline, adapted from Paulden 2016, "Smashing the racket") and wallets (calibration test, one-shot bet score, clusters). **Phase 1 is data only.**
- Free, keyless APIs only:
  - Gamma: `https://gamma-api.polymarket.com/markets` (limit/offset paging; filters `closed`, `volume_num_min/max`, `end_date_min/max`, `tag_id`, `condition_ids`; fields include `conditionId`, `clobTokenIds`, `outcomes`, `outcomePrices`, `volumeNum`, `startDate`, `endDate`, `closedTime`, `umaResolutionStatus`, `tags`, `events`).
  - Data API v2: `https://data-api.polymarket.com/v2/...` (`/trades`, `/prices-history`, `/resolutions`, `/activity`, `/tokens`; cursor paging through `pagination.next_cursor`; **no `offset`**, which returns 400; snake_case fields such as `proxy_wallet` and `condition_id`; at most 20 condition IDs per `/trades` call; IP rate limits with `429` + `Retry-After`; OpenAPI spec at `/v2/openapi.json`).
- Check the field names against the live OpenAPI spec and real responses before relying on them. The notes above come from the docs as of 2026-11-11.

## Where code can run

- The device sandbox shell and the cloud container both get a proxy `403` for polymarket.com **and** pypi.org (checked 2026-11-11). Anything that installs packages or calls the APIs must run on Isaac's PC through a Remote Control session in this folder.
- Isaac's Python is 3.14 on Windows. Code must work on Windows: use `pathlib`, avoid shell-specific scripts, and use a local `.venv` inside this project folder.

## Repo conventions (sibling projects 03 and 05)

`README.md`, `PLAN.md`, `pyproject.toml`, `requirements.txt`, `src/`, `tests/`, `scripts/`, `data/` (gitignored except `data/README.md`), `results/` (small curated outputs allowed), `notebooks/`. The root `.gitignore` already ignores `data/*`, `*.parquet`, `*.duckdb`, `*_cache/` and `.venv/`. The parent folder is a git repo with a GitHub remote, so **do not commit or push unless Isaac asks.**

## Work split

### Builder A: foundation and market universe (plan 1.1, 1.2, 1.5)

Owns: `pyproject.toml`, `requirements.txt`, `README.md` stub, `data/README.md`, ruff and pytest config, `src/pmdetect/__init__.py`, `src/pmdetect/cli.py`, `src/pmdetect/collect/http.py`, `src/pmdetect/collect/gamma.py`, `src/pmdetect/store/schema.sql`, `src/pmdetect/store/db.py`, `src/pmdetect/markets/universe.py`, `src/pmdetect/markets/info_type.py`, `data/manual/info_type_labels.csv`, `data/manual/known_cases.csv`, and tests for these.

1. **First, so Builder B is unblocked:** scaffold the package, the venv and dependencies, `http.py`, `schema.sql` and `db.py` exactly as in the interface section below. Then message Builder B that the foundation is in.
2. Gamma collector for closed markets (resumable paging, raw responses cached).
3. Universe filter: $5k to $500k volume; resolved cleanly to a binary outcome (not disputed or cancelled); exclude crypto "up or down" price markets, sports games and weather markets (by tags and slug or keyword rules, kept in one config dict). The "at least 20 trades" rule is applied after Builder B loads trades.
4. Information type tagging (`scheduled_reveal`, `unscheduled_event`, `deadline`) from tags and keywords, checked against about 30 hand-labelled markets in `info_type_labels.csv`. Report agreement.
5. Seed `labels` from `known_cases.csv` with the 5 validation cases in the plan doc (Nobel Peace Prize 2025, Google Year in Search 2025, Axiom exposé Feb 2026, Maduro removal Jan 2026, the Bubblemaps military cluster). Find their markets' slugs and condition IDs through Gamma. Store wallets only where published reporting names them.
6. CLI: `pmdetect collect markets` and `pmdetect universe`.

### Builder B: trades, prices, wallets, news time and integrity (plan 1.3, 1.4, 1.5)

Owns: `src/pmdetect/collect/data_api.py`, `src/pmdetect/markets/newstime.py`, `src/pmdetect/quality/integrity.py`, `data/manual/news_time_overrides.csv`, and tests for these. It codes against Builder A's `http.py` and `db.py` interfaces, and can start before they land.

1. `/trades` collector: batches of up to 20 condition IDs, follows cursors to the end, idempotent upsert on `trade_id` (or `tx_hash` + log index, whichever the API gives as unique).
2. `/prices-history` per token at the finest fidelity the API offers, and `/resolutions` for the final outcome (fall back to final prices of 1 and 0).
3. `/activity` for every wallet that traded a universe market, deriving `first_seen`, `n_markets` and `total_usd` into `wallets`.
4. News time (1.3): use the override CSV first (scheduled reveals and validation cases). Otherwise take the first time the price of the eventual winner crosses 0.9 and stays above it until resolution. Record which method was used.
5. Integrity report (1.5): per market, compare summed trade USD with Gamma volume, find gaps in the tape, and apply the at-least-20-trades rule. Write `results/integrity_report.csv` plus a short markdown summary.
6. CLI functions (Builder A wires them into `cli.py`): see the interface section.

### Phase 1 is done when (milestone M1)

A pilot of about 200 universe markets plus the Nobel 2025 market is in the store, `pmdetect integrity` shows trades reconciling with Gamma volume for at least 95% of markets (or explains why not), and `pytest` passes with no network access. Then stop and report. Do not start phase 2.

## Shared interfaces (both builders code against these)

```python
# src/pmdetect/collect/http.py  (Builder A)
def get_json(url: str, params: dict | None = None, *, use_cache: bool = True) -> Any: ...
    # throttled (default about 5 req/s), tenacity retries on 5xx and timeouts,
    # sleeps for Retry-After on 429, caches the raw JSON under data/raw_cache/
    # keyed by a hash of url + sorted params
def paginate_cursor(url: str, params: dict, *, data_key: str = "data",
                    cursor_param: str = "cursor") -> Iterator[dict]: ...
    # yields rows across pages until pagination.next_cursor is null

# src/pmdetect/store/db.py  (Builder A)
DEFAULT_DB: Path  # data/pmdetect.duckdb
def connect(path: Path = DEFAULT_DB) -> duckdb.DuckDBPyConnection: ...  # applies schema.sql
def upsert(con, table: str, df: "polars.DataFrame", keys: list[str]) -> int: ...

# src/pmdetect/collect/data_api.py etc.  (Builder B; Builder A wires these into the CLI)
def collect_trades(condition_ids: list[str]) -> int: ...           # pmdetect collect trades
def collect_prices(token_ids: list[str]) -> int: ...               # pmdetect collect prices
def collect_resolutions(condition_ids: list[str]) -> int: ...      # part of collect trades
def collect_wallet_activity(wallets: list[str]) -> int: ...        # pmdetect collect wallets
def estimate_news_times() -> int: ...                              # pmdetect newstime
def run_integrity_report() -> Path: ...                            # pmdetect integrity
```

Tables (in `schema.sql`; timestamps are UTC `TIMESTAMP`, money in USD `DOUBLE`):

| Table | Key | Columns |
| --- | --- | --- |
| `markets` | condition_id | event_id, slug, question, category, tags (VARCHAR[]), info_type, volume_usd, liquidity_usd, start_ts, end_ts, closed_ts, uma_status, outcome (YES/NO/NULL), news_ts, news_ts_method, in_universe (BOOL), fetched_at |
| `tokens` | token_id | condition_id, outcome_label |
| `trades` | trade_id | tx_hash, condition_id, token_id, wallet, side, outcome_label, size, price, usd, ts |
| `prices` | (token_id, ts) | price |
| `wallets` | wallet | first_seen, n_markets, total_usd, funding_source (NULL until phase 3), fetched_at |
| `labels` | label_id | condition_id, wallet, case_name, label, source_url, notes |
| `fetch_log` | (endpoint, key) | last_cursor, completed (BOOL), fetched_at |

Dependencies (Builder A installs them all up front): httpx, tenacity, pydantic, duckdb, polars, pyarrow, typer, pytest, respx, ruff.

Tests use small recorded API responses in `tests/fixtures/` and never touch the network.

---

# Phase 2: market-level detector (pilot of 399 markets)

Full spec: plan doc section "Phase 2". Goal: a discrepancy score D per market (how far the winner's price ran toward 1 before the news, against a peer baseline), supporting signals, a combined rank, top-1% flags and paper-style charts. Stop at milestone M2; do not start phase 3. No commits, pushes or deletes.

## Rules for everyone

- **Never take the DuckDB write lock** (the wallet backfill is still writing). Work from Parquet in `data/snapshots/` (`scripts/make_snapshot.py` regenerates it; markets, tokens, trades, prices, wallets, labels) or open DuckDB with `read_only=True` plus retry.
- **`/trades` lists every fill twice** (taker row and maker row, same `transaction_hash`), and the store has no taker flag. Any directional, volume or per-wallet measure must use taker rows only. Builder B writes `data/snapshots/trades_taker.parquet` first (re-fetch with `taker_only=true` through `get_json`, cached, no DB writes) and exposes `load_taker_trades()`; Builder A needs only prices and markets.
- Pre-news window: 7 days to 1 hour before `news_ts` (1 h buffer because news_ts is imprecise). Markets without `news_ts` or without a YES/NO outcome are scored as NaN and listed in the summary, not dropped silently.
- Scoring universe: markets with `in_universe` and a completed tape (the 399 pilot markets). Tests are offline with synthetic data.

## Builder A: series, baseline, score

Owns `src/pmdetect/markets/series.py`, `baseline.py`, `score.py`, `tests/test_phase2_a.py`.

```python
# series.py
def load_inputs(snapshot_dir: Path = SNAPSHOTS) -> tuple[pl.DataFrame, pl.DataFrame]: ...  # markets, prices(+tokens)
def winner_series(markets, prices, trades=None) -> pl.DataFrame: ...
    # columns: condition_id, t (hourly UTC grid), tau_h (hours to news_ts, > 0), q (winner price,
    #          last price forward-filled; YES price if YES won else 1 - YES price), covers pre-news window
def q0_table(series) -> pl.DataFrame: ...  # condition_id, q0 (price 7d before news; else first price), q0_flag ("young_market")
# baseline.py
def fit_predict_cross(series, q0, info_type, n_splits=5, seed=0) -> pl.DataFrame: ...
    # pooled conditional quantiles (10/50/90) of q ~ spline(log tau) + q0 + info_type (deadline separate/interaction);
    # statsmodels QuantReg (or sklearn quantile GBM), parsimonious; 5-fold cross-fitting by market, so
    # a market never sets its own baseline. columns: condition_id, t, tau_h, m10, m50, m90
# score.py
def score_markets(...) -> pl.DataFrame: ...
    # one row per market: D = mean over hourly grid (7d .. 1h before news) of max(0, q - m50);
    # D_signed (mean of q - m50), share_above_m90, q0, q0_flag, n_hours, info_type, news_ts_method
```

Document in the module that a time split (fit early, score later) replaces cross-fitting once the full backfill exists.

## Builder B: signals, combine, plots, CLI

Owns `src/pmdetect/markets/signals.py`, `combine.py`, `src/pmdetect/report/plots.py`, `tests/test_phase2_b.py`, and the `score-markets` command in `cli.py` (edit only that command).

```python
# signals.py (taker rows only, pre-news window)
def load_taker_trades(refresh: bool = False) -> pl.DataFrame: ...
def compute_signals(markets, taker_trades, wallets) -> pl.DataFrame: ...
    # per market: ofi_24h (net winner USD bought / total USD, last 24h), vol_surge_z (median/MAD z of hourly
    # volume vs the market's own earlier history), max_jump (largest single-trade price move toward the answer),
    # fresh_share (winner-side USD from wallets with first_seen < 7d before the trade, or only-market wallets;
    # fall back to first seen in the store while the backfill is incomplete; column fresh_method says which),
    # hhi (Herfindahl of winner-side USD across wallets)
# combine.py
def percentiles_by_type(df, cols) -> pl.DataFrame: ...   # percentile within info_type
def combine(scores: pl.DataFrame, signals: pl.DataFrame) -> pl.DataFrame: ...
    # Fisher and mean-of-ranks combinations, flag top 1% for each, isolation forest cross-check (not headline),
    # main_driver column; writes nothing itself
# plots.py
def plot_market(row, series, baseline, taker_trades, out_dir) -> Path: ...  # paper Figure 3 style
def plot_d_histogram(scores, out_path) -> Path: ...                          # paper Figure 2 style
# cli: `pmdetect score-markets` runs series -> baseline -> score -> signals -> combine -> plots
#      and writes results/market_scores.csv, results/figures/, results/phase2_summary.md
```

## Shared tests

A synthetic market with a planted early drift toward the winner ranks first; D = 0 for a market that stays on the median; taker-only deduplication is checked (a fill listed twice counts once).

## M2 exit report

D distribution and top-1% threshold; flagged markets (question, D, main driver); rank and percentile of every labelled case, with the Nobel Peace Prize 2025 (Machado) chart called out (clearly above the peer band before the announcement?); agreement of Fisher vs rank-mean vs isolation forest; caveats (pilot size, news_ts accuracy); wallet backfill progress; pytest and ruff status.

Environment note: statsmodels, scikit-learn, scipy and matplotlib are installed in `.venv`. The first import after an install can fail once with "DLL load failed ... Application Control policy" (a transient first-load block that clears on retry, also the likely cause of the earlier pyarrow ImportError). Retry once; do not work around it, and report if it persists.

---

# Phase 2b: baseline rework (news-time circularity)

Problem: 398 of 399 pilot `news_ts` values are the first 0.9 crossing of the winner price, so the level baseline and D are circular (Machado, the only independent time, scores D = 0.0115 although its price jumps 0.03 -> 0.7 about 10 h before the announcement). Rules as before: no git, no commits, read-only DuckDB (never the write lock) or Parquet snapshots, offline tests, stop at a report. Deleting your own temp files inside this folder is fine. Do not tune any threshold on the Nobel case; fix the method first, then report every labelled case as a held-out check.

Shared output (Builder A writes it, Builder B reads it): `data/snapshots/news_independent.parquet` with `condition_id, news_ts_indep, indep_method, independent (bool)`; `indep_method` in {`override`, `scheduled_end`, `none`}.

## Builder A: independent news times, level baseline, late-window measures

Owns `markets/newstime_indep.py`, edits to `markets/series.py`, `baseline.py`, `score.py` (phase 2 files), `tests/test_phase2b_a.py`.
1. Independent news time: the manual override CSV, or the scheduled end time (`end_ts` / `closed_ts`) only where it is a real event time (award announcements, scheduled releases, elections, data releases). Hand-check a sample, document the rule in the module docstring and README (e.g. restrict to `info_type == scheduled_reveal` with end_ts within a tolerance of the first price jump or of the final trade, and say how many were verified by hand). Never take a 0.9-crossing time as independent.
2. Fit the m10/m50/m90 level baseline ONLY on markets with an independent news time (cross-fitted as before), keep the same parsimonious spec. Score level-D only for independent markets; other markets get level-D null with reason `price_based_news_time`. Report how many markets are independent.
3. For every market (independent or not) add late-window measures against that market's own best available news time: D over the last 24 h only (using the same baseline) for independent markets, and `max_jump` / `max_jump_hours_before` (largest pre-news single-step move of q toward the answer, with its timing) for all markets.
4. Output `data/snapshots/news_independent.parquet` first so Builder B can use it, then update `market_D.parquet` with the new columns.

## Builder B: flow-before-the-move, extra labelled cases, combine, rerun

Owns `markets/changepoint.py`, `markets/flow.py`, `markets/cases_extra.py`, edits to `markets/signals.py`, `combine.py`, `report/phase2.py`, `report/plots.py`, the `score-markets` command, `tests/test_phase2b_b.py`.
1. Resolving-jump detection on q(t) (CUSUM or a simple changepoint): jump start = the time q first leaves its pre-jump level. Then, in the 24 h before the jump start, taker winner-side order-flow imbalance, fresh-wallet share, Herfindahl concentration and volume surge, each against the market's own earlier history (robust z / percentile within info_type). This answers "was there informed flow before the price moved" without relying on a price-defined news time. Record `jump_start_ts`, `jump_size`, and null with reasons where no clear single jump exists.
2. More positives, fetched WITHOUT writing to DuckDB (a wallet backfill process holds the write lock intermittently): load the Maduro market (in `data/manual/pilot_markets.csv`, trades missing from the store; its tape and prices) and the Axiom exposé and Bubblemaps-cluster candidate markets, as out-of-universe labelled cases, into `data/snapshots/extra_trades_taker.parquet` / `extra_prices.parquet` / `extra_markets.parquet` using the same collectors' logic (cached `get_json`, taker_only=true for the flow measures; prices-history for q(t)). Candidates come from the `labels` table / `known_cases.csv`; if the mapping is uncertain, list candidates and say so, do not guess a single market. Give each extra market a news time only from an independent source documented in `news_time_overrides.csv`-style reporting you can cite; otherwise mark it price-based. Make `score-markets` score them alongside the pilot (flagged as `out_of_universe`, excluded from the baseline fit and from the percentile pools).
3. Update combine: percentiles within info_type of level-D (independent only), late-window D, max_jump, flow-before-move measures; Fisher and rank-mean over whatever is available (document missing-data handling), isolation forest still only a cross-check; flag top 1%.
4. Re-run end to end once Builder A's `news_independent.parquet` and new `market_D.parquet` exist (poll every ~30 s), regenerate results/market_scores.csv, figures (include every labelled case incl. extras; mark jump_start_ts) and results/phase2_summary.md (data-driven): new D threshold, flagged markets with drivers, every labelled case's rank on each score (level-D, late-window D, max jump, flow-before-move, combined), whether the Machado chart clears the band, number of independent-news-time markets, caveats.

---

# Phase 2c: within-event pre-reveal test and flow-window fix

Same rules as 2/2b (no git, never the DuckDB write lock, Parquet snapshots only for anything fetched, offline tests, no tuning on labelled cases, stop at a report, keep the wallet backfill running).

## Builder A: `markets/within_event.py`, `tests/test_phase2c_a.py` -> `data/snapshots/within_event.parquet`
For every pilot market with an independent news time (independent = scheduled_reveal; use `news_independent.parquet`), get its Gamma event (`events` in the market / event_id), fetch ALL sibling outcome markets of that event (Gamma event endpoint, cached `get_json`) and the price history of each sibling's YES token (`/v2/prices-history`, same ladder as `collect_prices`, plus taker/any trade prices where available) into `data/snapshots/event_prices.parquet` and `event_markets.parquet` (never DuckDB). Statistic per outcome: change in log-odds (clip price to [0.005, 0.995], last price at or before the grid time, forward-filled) over the final 24 h and final 12 h before the independent reveal (use the same end buffer as elsewhere, 1 h before). Per event: winner's move minus the median losing outcome's move, plus the winner's move as a percentile among all outcomes of the event; flag events with a single outcome (binary) separately. Pool across events: placebo distribution of winner moves vs loser moves (share of events whose winner move exceeds the 95th percentile of the losers' moves, with a permutation null where winners are chosen at random within event). Binary version: log-odds move of the eventual winner over the same windows for single scheduled-reveal markets, compared against the same window in other binary scheduled reveals (percentile). Report Machado's position (Nobel event: all candidates' moves) and whether the box-office flagged markets (Fight or Flight, Bad Guys 2) fall away. Output columns: condition_id, event_id, n_outcomes, win_move_24h, win_move_12h, med_lose_move_24h, med_lose_move_12h, diff_24h, diff_12h, winner_pct_24h, winner_pct_12h, group (multi|binary), reason.

## Builder B: `markets/flow.py`, `markets/combine.py`, `report/phase2.py`, `tests/test_phase2c_b.py`
(1) Fix the flow window: from 24 h before `jump_start_ts` up to the point where the price has covered half of the jump or the first 2 h of the jump, whichever is earlier; measures versus the market's own earlier history as before; thresholds not tuned on labelled cases. (2) Report jump-initiating trades per market (and for every labelled case): the top wallets by winner-side taker USD in that window, with first_seen and whether this was their only market (use data/snapshots wallets + extra_wallets; note coverage), write `results/jump_initiators.csv`. (3) Add Builder A's within-event columns (diff_24h, diff_12h, winner_pct) into combine as new components when present and re-run `pmdetect score-markets` after A's `within_event.parquet` exists (poll every ~30 s); regenerate results/market_scores.csv, figures and results/phase2_summary.md with each labelled case's rank on the new scores beside the old ones, the placebo distribution, and a note on whether the box-office false positives vanish on the within-event test.

---

# Phase 3: wallet-level detector (milestone M4)

Plan doc section "Phase 3". Same rules as before: no git, never the DuckDB write lock (read Parquet in `data/snapshots/`, refresh with `scripts/make_snapshot.py`, or DuckDB `read_only=True` with retry), offline tests, no tuning on labelled cases, stop at M4, keep the wallet backfill running. Use the word "anomalous" in every output and doc, never "insider": these are statistical flags, not findings of wrongdoing.

**Privacy rule:** `results/*.csv` is allowed into git by the root `.gitignore`, and the repo has a GitHub remote. No file containing wallet addresses may live in `results/` (or anywhere outside `data/`). Every wallet-level output goes to `data/private/` (gitignored with the rest of `data/*`). Only aggregate statistics (counts, distributions, tier sizes, p-value histograms; no addresses, no per-wallet rows) go into `results/`. The coordinator's M4 report may quote addresses.

Data caveat: only the 399 pilot markets plus the extra out-of-universe labelled markets are in the store, and the `trades` tape lists every fill twice (taker row + maker row, the maker row is recorded at the complement price). The taker-only tape is `data/snapshots/trades_taker.parquet` (+ `extra_trades_taker.parquet`). The store's `wallets` table is only partly backfilled (`wallets.parquet` plus `extra_wallets.parquet`).

## Builder A: positions, calibration, FDR

Owns `src/pmdetect/wallets/__init__.py`, `positions.py`, `calibration.py`, `fdr.py`, `tests/test_phase3_a.py`.

```python
# positions.py
def build_positions(taker_trades, markets, news_indep, jumps, event_markets=None) -> pl.DataFrame: ...
    # one row per (wallet, event): net position per outcome token from the wallet's own rows (side, token, price right;
    # taker rows only, so no complement-price confusion), volume-weighted entry price of the net position, usd, size,
    # won (bool or null), first_ts, cutoff_ts, cutoff_method (independent_news | jump_start | news_minus_buffer),
    # info_type, hours_before_cutoff. Multi-outcome events collapse to ONE position per wallet per event (dependent
    # outcomes not counted twice; document how the entry price and result of the collapsed position are defined).
    # Only positions opened before the cutoff count. Writes data/private/positions.parquet.
def crosscheck_positions(positions, wallets: list[str], n: int = 10) -> pl.DataFrame: ...
    # compares ~10 sampled wallets against /v2/positions or /v2/user-pnl (cached get_json); report mismatches.
# calibration.py
def fit_isotonic(positions, n_splits=5, seed=0) -> ...   # outcome on entry price, cross-fitted, by info_type
def wallet_tests(positions) -> pl.DataFrame: ...
    # per wallet: n_pos, wins, exact Poisson-binomial upper tail (DP) of wins vs recalibrated p~,
    # Z = (wins - sum p~)/sqrt(sum p~(1-p~)), USD-weighted excess PnL with a bootstrap p-value. Writes data/private/wallet_tests.parquet.
def poisson_binomial_sf(k, p) -> float: ...
# fdr.py
def benjamini_hochberg(p, alpha=0.05) -> ...; def apply_fdr(tests, min_positions=5, alpha=0.05) -> pl.DataFrame
```

## Builder B: filters, one-shot, clusters, watchlist

Owns `src/pmdetect/wallets/filters.py`, `oneshot.py`, `clusters.py`, `src/pmdetect/watchlist.py`, `tests/test_phase3_b.py`. Reads Builder A's `data/private/positions.parquet` and `wallet_tests.parquet` once they exist (poll every ~30 s; write against the interface until then).

```python
# filters.py
def flag_non_directional(taker_trades, positions, wallets) -> pl.DataFrame: ...
    # market makers/bots (both sides of the same market often; very many markets with near-zero edge), contract or exchange addresses
    # (document each rule and the counts excluded)
# oneshot.py
def score_oneshot(positions, markets, wallets, market_scores, within_event) -> pl.DataFrame: ...
    # winning pre-cutoff positions scored as the product of within-market percentiles of improbability (-log entry price),
    # size (USD vs the market's median position), earliness (hours before the cutoff), freshness (first_seen age, only-market in store, n_markets_global when known),
    # and market context (within-event extreme, or flow and late-window ranks)
# clusters.py
def cotrading_clusters(taker_trades, min_markets=2, window_minutes=10) -> pl.DataFrame: ...
    # link wallets that bought the same outcome in the same market within 10 minutes in >= 2 markets; connected components;
    # rerun the calibration on each pooled cluster. Stub interface link_by_funding(wallets) for Polygon funding links later.
# watchlist.py
def build_watchlist(...) -> Path: ...   # data/private/watchlist.csv; tiers: repeat_anomalous (after FDR), one_shot_anomalous, watch; each row with its evidence
```

CLI: `pmdetect score-wallets` runs everything and writes `data/private/*` plus the aggregate-only `results/phase3_summary.md`.

## Shared tests
A planted wallet that always wins long shots passes FDR; a calibrated random wallet does not; the Poisson-binomial tail matches brute force on small cases; event-collapse logic (no double counting of dependent outcomes); taker-row position netting; privacy: `results/` outputs contain no wallet addresses.

## M4 report
Wallets tested and excluded; how many pass FDR; top one-shot positions with market and evidence; clusters found; whether the Maduro wallet named in reporting and the Machado jump wallet (referred to by role only, never by address) land in the one-shot tier and at what rank; the position cross-check result; a power caveat (only the 399 pilot markets plus extras); test and ruff status; wallet backfill status.

---

# Phase 3b: spec fixes on the pilot (no full-universe backfill)

Same rules (no git, never the DuckDB write lock, privacy rule: wallet-level outputs only in `data/private/`, "anomalous" not "insider", no tuning on the named wallets). Do NOT start the full-universe backfill (the user decides); the running wallet backfill may finish.

## Builder A: cutoff circularity (positions.py, calibration.py, fdr.py, tests)
Positions whose cutoff is `news_minus_buffer` with a price-based news time (22,807) include buying during the resolving jump itself and inflate long-shot win rates. New rule: cutoff = independent news time if one exists, else the jump start, else NO valid cutoff. Positions with no valid cutoff (`cutoff_method = none`) stay in `positions.parquet` (marked) and in the one-shot screen but are excluded from the calibration fit and from the repeat test. Refit the calibration; write aggregate-only `results/phase3b_calibration.csv` with observed win rate and mean p~ by entry-price decile before and after. Report FDR passes and eligible wallet counts after the fix.

## Builder B: one-shot definitions (oneshot.py, watchlist.py, tests) + first_seen top-up
- Earliness becomes proximity: highest for entries within about 72 h before the cutoff (or inside the jump window), decaying for earlier entries; fix the shape in advance, document it in the docstring, apply the same function to every position (no tuning on cases). Positions with no valid cutoff are marked `no_valid_cutoff` and have no proximity credit.
- Freshness becomes wallet age at entry (entry time minus first_seen), not first_seen alone; keep the only-market sub-score; missing first_seen still counted.
- First_seen top-up: for every winning wallet in the one-shot candidate set (define the set explicitly, e.g. the top few thousand winning positions by score before the freshness component) that lacks first_seen, fetch it via `/v2/activity?user=..&type=TRADE&start=1&sort_direction=ASC&limit=1` (cached get_json, ~5 req/s, background script with a log, resumable) into `data/private/first_seen_topup.parquet` (never DuckDB; prefer wallets.parquet/extra_wallets.parquet entries first; snapshot again at the end since the backfill keeps adding rows).
- Estimate the cost of the full-universe backfill (all 14,402 universe markets' trades and prices, plus first_seen for their wallets) from the pilot's measured throughput (request counts and wall time in logs/ and the cache sizes under data/raw_cache): requests, hours at the current ~5 req/s, disk, and note DuckDB and Parquet sizes; give ranges and the assumptions. Do not run it.

---

# Light full-universe backfill (all 14,402 universe markets)

Done by `scripts/bulk_fetch.py` (run by the user in a terminal; stages prices, taker, trades; resumable; parts under `data/tapes/`), loaded by `scripts/load_tapes.py` (`prices` and `taker` steps done; `trades` after the all-rows stage completes), then:
- `data/snapshots/trades_taker.parquet` is now the FULL taker tape (8.48M rows, 14,403 markets); the pilot-only copy is `trades_taker_pilot.parquet`. DuckDB `prices` holds the winner-token price history of every universe market (4.0M rows; only the WINNING token's series was fetched for non-pilot markets, since q(t) needs only that, so the losing token has no series). `extra_*` snapshots (Maduro, Axiom/Bubblemaps candidates) are unchanged.
- DuckDB `trades` (all rows, taker + maker) is loaded after the all-rows stage; `fetch_log` rows endpoint 'trades' mark loaded markets.
- Rules unchanged: no git, never run a network fetch inside an agent (write restartable scripts, do NOT run them, the coordinator decides), wallet-level outputs in `data/private/` only, "anomalous" not "insider", nothing tuned on the labelled cases.

## Builder S1: market pipeline at full scale
Owns `markets/newstime.py` (add a vectorised parquet-based bulk estimator with a parity test against the existing per-market function on the 399 pilot markets; apply to DuckDB `markets.news_ts` in one batched UPDATE), `markets/newstime_indep.py`, `markets/within_event.py` (+ `scripts/event_fetch_full.py`, restartable, per-event part files, NOT run; request-count estimate), `markets/series.py`, `baseline.py`, `score.py`, `report/phase2.py` and `signals.py`/`flow.py` as needed for scale, plus tests. Run everything local at full scale (non-network): bulk news times, independent news times, then `pmdetect score-markets` on the full tape with timings; remove pilot-only assumptions; keep the extra labelled cases working.

## Builder S2: wallet pipeline at full scale
Owns `wallets/*`, `watchlist.py`, `scripts/run_phase3_a.py`, `scripts/first_seen_candidates.py` (new, restartable, NOT run), tests. Scale positions (8.5M taker rows), calibration (independent-news positions only), filters (maker detection needs the all-rows tape: loaded into DuckDB/`trades.parquet` after the trades stage; code must work with whatever is there and say so), clusters (avoid O(n^2) co-trading joins), watchlist. Candidate set for first_seen: wallets behind the top 5,000 one-shot positions by score-without-freshness, every jump initiator, every wallet with at least 5 calibration-eligible positions, excluding wallets that already have first_seen. Candidate-fetch script writes `data/private/first_seen_topup.parquet` incrementally (resumable). Needs S1's `news_independent.parquet` and jump data for the final run: poll for them (they are re-written by S1) and re-run `score-wallets` at the end.

---

# Phase 3c: non-directional filter redefined, independent times for deadline / unscheduled markets, time-split baseline

Standing rules: no git, no network (fetch scripts are written restartable and NOT run; the coordinator gives Isaac the command), never the DuckDB write lock, wallet addresses only in `data/private/`, "anomalous" not "insider", nothing tuned on the labelled cases (decide thresholds and acceptance criteria BEFORE looking at them).

## Part A: maker / non-directional filter (`wallets/filters.py`)
Market making means two-sided behaviour, not order type. Build positions from both taker and maker fills with the correct side and price for maker rows (the maker row is recorded at the complement price), so limit-order buys count as buys. A wallet is a market maker only if it is two-sided within markets (buys and sells the same token, or buys both outcomes, in a large share of its markets, with a low net-to-gross ratio, across enough markets). A wallet whose fills in a market are overwhelmingly one direction is directional whatever its maker share. Thresholds fixed in advance from the distribution (histogram of net-to-gross documented). Report wallets moved from excluded to included. Positions cross-check against `/v2/positions` from cached data only if it exists, else a script for Isaac; say whether maker fills close the earlier gap (API held more than the taker tape).

## Part B: independent event times for deadline and unscheduled markets (`markets/newstime.py` and a new changepoint module)
Offline from data on disk. (1) Gamma fields in `data/raw_cache` (umaEndDate, resolution/closed timestamps, event-level fields, description text): is there a field giving the real event or resolution-proposal time; relate it to the price jump on a sample; hand-check 30. (2) A volume-and-price changepoint method that does not condition on outcome direction: the largest abrupt repricing in either direction with a burst in trade count is the news time; flow is measured in the window before the burst begins. Validate on the 4,249 scheduled reveals with known times and report the error distribution. State the acceptance criterion (e.g. median absolute error and share within X hours) BEFORE looking at results; include these markets in calibration and the repeat test only if validation error meets it, else keep them in the one-shot screen only.

## Part C: time-split baseline (`markets/baseline.py`)
Fit the level baseline on earlier independent-news markets and score later ones (expanding windows, minimum training size, early windows reported unscorable); keep cross-fit as a flag for comparison.

Final report items: wallets reclassified; positions cross-check; validation error of the new news-time method and how many markets gain an independent time; eligible wallets and FDR passes; new top 10 one-shot positions; named wallets' ranks and tiers; labelled-case ranks; time-split vs cross-fit comparison.

---

# Phase 4 (validation) and Phase 5 (packaging)

Decision (2026-11-13): validate and package the two detectors that hold up, the **within-event pre-reveal test** and the **one-shot bet screen**. Level D, repeat-wallet FDR, burst times and clusters are documented as negative or untested results, not packaged as detectors. All offline: no network, no git commits or pushes, never the DuckDB write lock, wallet addresses only in `data/private/` (never in `results/`, `README.md`, `PLAN.md` or any file outside `data/`; this section never lists addresses), nothing tuned on labelled cases, long runs as resumable logged scripts under `logs/`, "anomalous" not "insider". Seeds are fixed (seed 20261113); bootstrap intervals use 1,000 resamples (95% percentile intervals).

## Acceptance thresholds (written 2026-11-13 18:50 UTC, BEFORE running anything in phase 4)

1. **Within-event synthetic insider (4A).** Detection = the planted winner's within-event percentile (among all outcomes of its event, 24 h window ending 1 h before the reveal) is at or above **0.95**. False-positive rate = share of untouched real events whose winner percentile is at or above 0.95. A cell (s, k) counts as *detected* if detection rate >= **50%** at FPR <= **10%**. The detector is called "working" for a stake/lead only in cells that meet this; the README states the smallest (s, k) cells that do and the cells that do not. If no cell with s <= $20k meets it, the README says the within-event test is a descriptive result, not a detector.
2. **One-shot synthetic insider (4B).** Tier threshold = the existing `one_shot_anomalous` rule (score at or above the 99th percentile of real valid-cutoff scores, USD >= 100). Detection = injected position at or above it; FPR = share of real positions in the tier (about 1% by construction). A cell is *detected* at detection rate >= **50%**. Also report the rank percentile of the injected position. Same "working" language as above.
3. **Placebos (4C).** (a) Within-event window ending 48 h before the reveal: pass if its permutation p-value is > 0.01 or its excess share above the 95th percentile (observed minus null) is at most half that of the real window. (b) Shuffled winners (100 shuffles): pass if the real-null share above q95 sits inside the shuffle distribution's central 95% interval. (c) One-shot scores for losing positions with the label flipped: pass if no more than **2%** land in the tier (twice the nominal 1%).
4. **Labelled cases (4D)** are held out: report only, never used to set any threshold above.
5. **Hand audit (4E)**: top 30 within-event extremes and top 30 one-shot positions outside the Maduro market; classes plausible leak / predictable convergence / public-news reaction / unclear; no threshold, report the mix; be conservative and say what cannot be judged offline.

## Work split
- **V1** (`src/pmdetect/validate/within_event_sim.py`): 4A and the within-event parts of 4C, plus the power heatmap.
- **V2** (`src/pmdetect/validate/oneshot_sim.py`): 4B and the one-shot part of 4C, plus the power heatmap.
- **V3** (`src/pmdetect/validate/cases.py`, `audit.py`): 4D and 4E, after V1 and V2 outputs exist.
- **P1** (packaging that does not depend on validation): small-sample reproduce path (about 50 markets, a few minutes, offline fixtures), `.gitignore`, tidy-up, pyproject metadata, LICENSE decision.
- **P2** (README and results): after V1 to V3 are done.
- CLI `pmdetect validate` (one owner: V1 creates it; V2/V3 add subcommands in separate files and register them with lazy imports).

Final check (P2): tests and ruff pass, word-bounded 40-hex scan outside `data/` finds no address, read-only `git status --porcelain` for this folder listed for Isaac. Do not commit.

---

# Stage 2: steps 2a (markouts, wallet skill v2) and 2b (forecasting benchmark)

Approved by Isaac 2026-11-14 ("go for it", after docs/STAGE2_PLAN.md). Only the offline steps on existing data start now. NOT started and needing his separate go: 2c (new data: UMA proposal times, news timestamps, possibly an API key), 2e (forward paper collector, network), 2f (funding-chain data, keys or paid API); no live trading, ever, by the agent. Standing rules: no git commits or pushes, no network, never the DuckDB write lock, wallet addresses only in `data/private/`, aggregate-only `results/`, "anomalous" not "insider", seed 20261113, 1,000 cluster-bootstrap resamples (by event), time-series splits only, nothing tuned on labelled cases. Wallet addresses and market ids must not appear in `results/`, README or docs (the privacy scan runs as a test).

## Pre-registered criteria (written 2026-11-14, BEFORE running anything in stage 2)

### 2a. Markouts and wallet skill v2
- **Markout** of a taker trade at horizon h in {1 h, 6 h, 24 h}: side-signed change in the TRADED TOKEN's price from the trade time to t+h (last price at or before t+h from the price series; trades whose t+h falls after the market's last observation, or after resolution, are dropped for that h, never padded), per dollar. Uses no final outcome and no news time. Wallet skill = size-weighted mean markout per dollar over the wallet's taker trades in a period.
- **Null:** event-level sign randomization: within each market, shuffle the trade signs (cluster = market/event), 2,000 draws; z and p per wallet; plus empirical-Bayes shrinkage (normal-normal) of wallet skill using wallets with at least 10 trades in the period.
- **Out-of-sample persistence (the headline):** split at the median market close date of the sample (train before, test after, with an embargo of 7 days: trades in markets that overlap the boundary are dropped). Wallets with at least 20 trades in EACH period. Statistics: (i) Spearman rank correlation of shrunk train skill with test skill, (ii) mean 24 h markout per dollar in the test period for the train top decile vs bottom decile vs all, with cluster-bootstrap 95% intervals.
- **PASS (skill persists):** the train-top-decile test-period mean 24 h markout per dollar is positive with the lower 95% bound above zero AND above the train-bottom-decile mean AND the rank correlation is positive with the lower 95% bound above zero. Placebo (must hold): the same statistics after shuffling wallet identities within the train period show no such effect (observed top-decile value above the placebo 95th percentile). Otherwise report "no persistent markout skill".
- Report ALSO the spread-adjusted version (markout minus half the estimated round-trip spread per trade, spread proxied from consecutive opposite-side taker fills in the same market within 10 minutes; fall back to the market's median), and the sign of results at 1 h and 6 h. Gross-only success does not count as tradable.

### 2b. Forecasting benchmark vs the market price
- **Task:** at snapshot time t = end_ts - H for H in {7 d, 3 d, 1 d} (end_ts is the scheduled end known at creation), for markets still open at t (`closed_ts` > t is knowable at t), with at least 20 taker trades before t, predict the YES-token outcome (market.outcome == YES). Features use only data strictly before t: logit price, price change over 1 h / 24 h / 7 d, taker flow imbalance and volume surge (24 h), Herfindahl of buying wallets, fresh-wallet share, prior-period wallet skill of the buyers (from 2a, train period only), info_type, log volume to date, hours to end, price spread proxy.
- **Benchmark:** the market price itself (log-loss and Brier of p). Candidate models: (a) logistic recalibration of logit price, (b) gradient-boosted trees (shallow, fixed hyperparameters set in advance: depth 3, 200 trees, learning rate 0.05, no tuning on the test period). Purged walk-forward by month of end_ts: train on months before M (embargo 7 days), test on month M; pooled test period, split into two disjoint halves.
- **PASS (information beyond price):** the model's pooled out-of-sample log-loss is below the price benchmark's in BOTH test halves, with the lower 95% cluster-bootstrap bound (by event) of the improvement above zero in each half, for at least one horizon H. Feature-group ablation reports which group (price dynamics, flow, wallet skill, other) carries the gain. Otherwise report "no measurable information beyond price" and STOP the strategy track (steps S1 to S5 of docs/STAGE2_PLAN.md are not run).
- Also report calibration curves of the benchmark, per category, and whether any gain survives a stress test dropping the largest 10% of markets by volume.

### Decision rule after 2a and 2b
If neither passes, the strategy track is closed: write the negative result into the README. If either passes, report effect sizes net of the spread proxy and propose (not run) backtests S2/S3/S5 with these pre-registered pass criteria from docs/STAGE2_PLAN.md section 5.

## Work split
- **T1** `src/pmdetect/stage2/markouts.py` + `tests/test_stage2_markouts.py`, CLI `pmdetect stage2 markouts`.
- **T2** `src/pmdetect/stage2/forecast.py` + `tests/test_stage2_forecast.py`, CLI `pmdetect stage2 forecast`.
- Shared CLI group in `src/pmdetect/stage2/cli.py` (T1 creates; T2 registers its command with a minimal edit; re-read before editing). Outputs: aggregate `results/stage2_*.csv/.md`, figures under `results/figures/stage2_*.png`; wallet-level rows only in `data/private/`.
