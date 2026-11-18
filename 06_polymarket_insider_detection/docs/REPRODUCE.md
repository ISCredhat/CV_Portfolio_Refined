# Reproducing the results

Two paths: the offline demo on 50 fixture markets (seconds, no data download), and the full pipeline
(about 1.5 days of API time in total, all resumable). Wallet-level outputs always go to `data/private/`
(gitignored); nothing under `results/` or this document names a wallet. Flags are statistical
("anomalous"), not findings of wrongdoing.

## 1. Demo on fixtures (offline)

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]          # Python 3.11+; Windows paths shown, POSIX works too
.venv\Scripts\pmdetect demo run              # writes demo_output/ (gitignored); a few seconds
.venv\Scripts\pmdetect demo fetch-plan       # prints (no network) how to collect a fresh ~50-market sample
.venv\Scripts\python -m pytest tests\test_demo.py
```

What `demo run` does, using the real pipeline code on `tests/fixtures/demo/` (1.4 MB of Parquet; 50
markets chosen with seed 20261113: the Nobel Peace Prize 2025 winner market, 22 multi-outcome scheduled
reveals with their 241 sibling outcomes, 12 single scheduled reveals, 5 deadline, 5 unscheduled and 5
scheduled markets without an independent time; 35,576 taker fills; wallet ids anonymised to
`wallet-NNNNN`, trade ids to `trade-NNNNNNN`, transaction hashes dropped):

1. price-based news times (first stable 0.9 crossing of the winner price, plus the manual override);
2. independent news times (override or scheduled end time);
3. resolving-jump start times (cutoff fallback);
4. within-event pre-reveal test: winner's 24 h / 12 h log-odds move against all outcomes of its event,
   pooled permutation placebo;
5. one-shot bet screen: positions, calibration, filters, one-shot scores, watchlist tiers;
6. `demo_output/demo_summary.md` (aggregates only) and a 40-hex scan of every output file.

It never touches `data/`, `results/` or DuckDB, and refuses `--out` inside those folders. The sample is far
too small for inference; it shows the code path and the shape of the outputs. Rebuild the fixtures from
the full snapshots (maintainers only): `python scripts/build_demo_fixtures.py`.

## 2. The full pipeline

Everything below runs from the project folder. Network steps share one ~5 requests/s budget
(`collect/http.py`); never run two at once. Durations are from `logs/` on one Windows PC.

| Step | Command | Time, disk (measured) |
|---|---|---|
| Closed Gamma markets (events keyset) | `pmdetect collect markets` | minutes; raw JSON cached in `data/raw_cache/` (6.3 GB total cache) |
| Universe filter, info types | `pmdetect universe`, `pmdetect infotype` | seconds; 14,402 markets in the universe |
| Pilot of 399 markets, trades, prices | `pmdetect pilot`, `pmdetect collect trades`, `pmdetect collect prices` | about 12 min paging for trades, 25 min for prices |
| Parquet snapshots | `python scripts/make_snapshot.py` | minutes; `data/snapshots/` is 3.8 GB |
| Wallet first_seen backfill (pilot wallets) | `python scripts/collect_wallets_first_seen.py` | 68,773 wallets in 3.8 h (`logs/wallets.log`) |
| Full-universe prices (winner token) | `python scripts/bulk_fetch.py prices` then `python scripts/load_tapes.py prices` | hours; resumable part files in `data/tapes/` |
| Full-universe taker tape | `python scripts/bulk_fetch.py taker` then `python scripts/load_tapes.py taker` | about 8.5 M rows, 14,403 markets; estimate 5.6 h (`results/phase3b_backfill_estimate.md`) |
| All-rows tape (taker + maker, optional) | `python scripts/bulk_fetch.py trades` then `python scripts/load_tapes.py trades` | needs the DuckDB write lock; about 21 M rows |
| Sibling prices for the within-event test + wallet first_seen top-up | `scripts\run_network_steps.bat` (= `event_fetch_full.py`, then `first_seen_candidates.py fetch --threads 3`) | step 1: 1,093 events, 17,286 siblings, 16,399 price calls, about 1.5 h; step 2: 7,962 wallets, about 25 min at 4.9 req/s. `data/tapes/` is 1.8 GB |
| News times | `pmdetect newstime`, `python -m pmdetect.markets.newstime_indep` | seconds to minutes |
| Market scoring (level D, signals, flow, within-event, figures) | `pmdetect score-markets` | 267 s, 6.5 GB peak memory (`logs/rescore_full.log`) |
| Wallet pipeline (positions, calibration, FDR, filters, one-shot, clusters, watchlist) | `pmdetect score-wallets` | 27 to 48 s, about 5.2 GB peak memory (`logs/score_wallets_burst.log`) |
| Validation simulations | `pmdetect validate ...` (see the README) | see `results/validation_*` |
| Tests, lint | `python -m pytest`, `ruff check src tests scripts` | all offline |

Disk: DuckDB 5.9 GB, snapshots 3.8 GB (the all-rows `trades.parquet` is 3.2 GB of it), raw cache 6.3 GB,
tapes 1.8 GB, private outputs 189 MB. `data/` is about 18 GB in total; none of it is committed.

Rules that apply throughout: read Parquet snapshots (or DuckDB `read_only=True`), never hold the write
lock during analysis; `--refresh-taker` is disabled because the taker tape is the full-universe file.

## 3. API quirks found

- Polymarket Data API v2 pages with cursors (`pagination.next_cursor`); an `offset` parameter returns
  HTTP 400. At most 20 condition ids per `/trades` call; page limit 1,000.
- `/v2/trades` has no trade id and no USD field: `usd = size * price`, and `trade_id` is a content hash.
- With `taker_only=false` every fill appears twice (a taker row and a maker row with the same transaction
  hash, the maker row on the complement token at the complement price) and there is no taker flag in the
  data. Any directional, volume or per-wallet measure must use the `taker_only=true` tape (one row per fill).
- `/v2/prices-history` rejects a bucket that is too fine for old markets (HTTP 400); old markets keep only
  the 3 h and 12 h grains. The collector walks a ladder (600, 1800, 10800, 43200 s) and keeps the first
  non-empty one, so price resolution varies by market age. Only the winning token's series was fetched for
  non-pilot markets.
- Gamma `/markets` stops paging at offset about 5,000 (HTTP 422, use keyset) and its `tags` are mostly just
  `["all"]`; the event object (`/events/keyset`) carries the real topical tags and nests its markets.
- `/v2/activity` gives a wallet's first trade with `start=1&sort_direction=ASC&limit=1` (one cheap call per
  wallet, throttle-bound at 5 req/s); the full history is far heavier (3,900 wallets cost 4.6 GB of cache).
- Rate limits are per IP: `429` with `Retry-After` (handled with a shared back-off across threads). Large
  trade pages (about 0.9 MB) are latency-bound at 0.6 to 1.1 pages/s, well below the throttle.
- `/v2/resolutions` returns `payouts` in micro-USDC per outcome slot; `/v2/tokens` gives `final_price`.
- Field names are snake_case (`proxy_wallet`, `condition_id`); check them against `/v2/openapi.json`, since
  the docs and the live responses differed in places.
- Scheduled `end_ts` in Gamma is a deadline, not the event time, for most markets; only scheduled reveals
  whose last trade and closing time sit close to `end_ts` get an independent news time (4,249 of 14,403).
- Windows note: scipy / scikit-learn can fail their first import with an "Application Control policy" DLL
  error; it clears on a retry.
