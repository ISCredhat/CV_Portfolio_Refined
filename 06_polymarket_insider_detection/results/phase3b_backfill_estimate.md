# Phase 3b: full-universe backfill cost estimate (aggregate only, NOT run)

Scope: all 14,402 in-universe markets (3,875 events): trades (`taker_only=false` as in phase 1, plus the
taker-only tape if needed), prices (both tokens), and `first_seen` for every wallet seen on those tapes.
Measured from the 399-market pilot (398 in-universe + 1 case market) via `logs/wallets.log`, `fetch_log`
timestamps (DuckDB opened `read_only`), `data/raw_cache` file mtimes/sizes and the Parquet snapshots.
No wallet addresses appear in this file.

## 1. Pilot measurements

| Item | Pilot measurement |
|---|---|
| Phase-1 trade tape (`taker_only=false`) | 989 cached `/trades` pages (2.5 per market), 893 MB cache (2.24 MB/market), 643,347 rows (maker + taker), ~12 + 2 min of active paging = ~1.14 pages/s |
| Taker-only re-fetch (`taker_only=true`) | 376 pages (incl. extra labelled markets), 328 MB, 10.3 min = ~0.6 pages/s, 258,741 taker rows for the 399 markets |
| Prices (`/prices-history`, finest-bucket ladder) | 800 tokens (2 per binary market), 1,621 cached requests (2.0 per token), 31 MB, 25 min = ~1.1 req/s, 245,550 price rows |
| first_seen (`/activity?limit=1`) | 68,773 wallets (72,673 total, 3,900 had the earlier full-activity pass), 3.8 h at 100 wallets per 20 s = 5.0 req/s (throttle-bound, `MIN_INTERVAL` = 0.2 s), ~1.2 KB per response |
| Full-activity pass (not needed for first_seen) | 3,900 wallets, 6,270 pages, 4.6 GB cache |
| Unique wallets on the pilot tapes | 72,673 (66,573 taker-side = 91.6%) |
| Store sizes | DuckDB 208 MB (643k trades, 245k prices, 65k+ wallets, 41k market rows); Parquet: trades 82 MB, trades_taker 21 MB, prices 0.9 MB, wallets 2.5 MB; raw_cache 6.0 GB, 82k files |

Large pages (about 0.9 MB, 944 rows on average) are latency-bound at ~0.6 to 1.1 pages/s, far below the
5 req/s throttle; the tiny first_seen calls are throttle-bound at 5 req/s. The throttle is per process
(shared lock in `collect/http.py`): the pilot ran trades and prices concurrently in two processes.

## 2. Scaling from the pilot (the pilot is a random sample, so scale by volume and trade count)

The pilot is one random market per event with a fixed seed. Its volume distribution matches the universe
(median $34.9k vs $32.8k; mean $87k vs $80k), but the heavy tail drives trade counts, so the estimate is
stratified by Gamma volume band rather than by market count (14,402 / 399 = 36.1x; by volume 33.4x).

| Volume band | Universe markets | Pilot markets | Pilot taker rows / market | Pilot all rows / market | Est. taker rows | Est. all rows |
|---|---|---|---|---|---|---|
| $5k to $20k | 5,230 | 134 | 185 | 457 | 0.97 M | 2.39 M |
| $20k to $100k | 5,543 | 154 | 439 | 1,097 | 2.44 M | 6.08 M |
| $100k to $500k | 3,629 | 110 | 1,456 | 3,614 | 5.28 M | 13.11 M |
| Total | 14,402 | 398 | | | 8.7 M | 21.6 M |

Range for the trade-row estimate: 8.0 to 9.5 M taker rows, 19.5 to 24 M all rows (the top band is heavy
tailed: 110 pilot markets, max 6,143 taker rows).

Unique wallets: rarefaction on the pilot (random subsets of 25, 50, 100, 200, 300, 399 markets gave 7.6k,
11.4k, 21.8k, 41.7k, 59.6k, 72.7k wallets) follows a power law with exponent 0.85 (0.80 on the last
segment). Extrapolating to 14,402 markets gives 1.3 M (local exponent) to 1.5 M (global fit); with
saturation (exponent 0.75) 1.07 M. Planning range 1.0 to 1.6 M wallets, central 1.35 M (about 91% of them
taker-side).

## 3. Requests, wall time and disk

| Component | Requests (central; range) | Hours at the measured rate (range) | Hours at an ideal 5 req/s | Raw cache |
|---|---|---|---|---|
| Trades, `taker_only=false` | 33 k (23 to 36 k) | 8.1 h at 1.14 pages/s (5.6 to 8.8 h) | 1.8 h | ~30 GB (21 to 32 GB) |
| Taker-only tape (optional) | 12 k (9 to 13 k) | 5.6 h at 0.6 pages/s (4.5 to 6.3 h) | 0.7 h | ~11 GB (9 to 12 GB) |
| Prices (2 tokens per market) | 58 k (2.0 per token x 28.8 k tokens) | 14.6 h at 1.1 req/s (13 to 16 h); about 7 h if only YES tokens | 3.2 h | ~1.1 GB |
| Tokens and resolutions | about 2 k | under 0.5 h | 0.1 h | under 0.1 GB |
| first_seen for ~1.35 M wallets | 1.35 M (1.0 to 1.6 M) | 75 h at 5 req/s (56 to 89 h) | 75 h | 1.6 GB of payload but 1.35 M small files: ~5.4 GB of NTFS clusters (see below) |
| Total (central) | ~1.45 M requests | ~104 h run serially; ~75 h wall if trades/prices and first_seen run as separate processes (first_seen dominates) | ~81 h | ~43 GB (with the taker pass) |

Reading: first_seen for every wallet is about 93% of all requests and 70% of the serial wall time.
Without it the market data (trades, taker tape, prices) are about 105 k requests and 28 h serially (10 to
12 h if trades, taker tape and prices run in three processes, an assumption that the IP rate limit
tolerates ~3 to 4 req/s total on large pages; the collector already retries 429 with Retry-After).

A targeted alternative is far cheaper: the pilot's candidate set (top 3,000 winning positions by the
one-shot score without freshness) lacked first_seen for only 362 wallets after the existing backfill, so
fetching first_seen only for candidate wallets after the market tapes are loaded costs a few thousand
requests (minutes), not 75 hours.

## 4. Disk and store sizes

| Store | Pilot | Full universe (central; range) |
|---|---|---|
| `data/raw_cache` (JSON) | 6.0 GB (82k files; 4.6 GB of it is the 3,900-wallet full-activity pass) | ~43 GB with the taker pass (~32 GB without), +~5 GB of cluster slack if the 1.35 M first_seen responses are cached; use a no-cache write for first_seen (as `use_cache=False` still writes a file today) or skip it |
| DuckDB `pmdetect.duckdb` | 208 MB | ~3.5 GB (3 to 5 GB; the ~120 B/row Parquet ratio suggests 2.6 to 2.9 GB for 21.6 M trade rows, plus wallets ~0.1 GB, prices ~0.1 GB, and transient WAL growth during bulk upserts) |
| Parquet `trades.parquet` (all rows) | 82 MB | ~2.7 GB (2.4 to 3.0 GB) |
| Parquet `trades_taker.parquet` | 21 MB | ~0.75 GB (0.65 to 0.8 GB) |
| Parquet `prices.parquet` | 0.9 MB | ~35 MB |
| Parquet `wallets.parquet` | 2.5 MB (41 k wallets) | ~80 MB |
| Memory for analysis | | `pl.read_parquet` of 21.6 M all-side rows needs roughly 4 to 8 GB RAM: read the taker tape only, or scan lazily by market |

## 5. Assumptions and caveats

- Pilot throughput holds for the full run (same API, same host, no new IP throttling). A 429 storm or a
  server-side change would lengthen the first_seen pass, which is the throttle-bound part.
- Row counts scale with Gamma volume band (stratified); wallets scale with the pilot's rarefaction exponent.
  Extrapolating a power law by 36x is a long way: the real wallet count could fall outside 1.0 to 1.6 M.
- The pilot has one market per event; the universe has 3,875 events, about 3.7 markets per event, so
  sibling markets share wallets more than random pilot markets do. This pushes the true wallet count
  toward the low end of the range.
- The 989 phase-1 pages for 643 k rows is ~650 rows per page while a sample of cached pages averages 944;
  both rates are carried in the ranges above (23 k pages from 944 rows/page, 36 k pages from the empirical
  page-per-row ratio).
- Wall time with parallel processes assumes the API tolerates it; the DuckDB write lock allows only one
  writer at a time, so parallel collectors must write Parquet parts and a single process must upsert.
- Not included: re-running positions, calibration and clustering on the larger tape (minutes to an hour),
  and news-time and Gamma-event fetches for the new markets.
- Pilot first_seen top-up for the one-shot candidate set (Phase 3b) is separate and already done.
