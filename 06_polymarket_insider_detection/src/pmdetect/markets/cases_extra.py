"""Extra labelled cases (Maduro, Axiom expose candidates, Bubblemaps cluster candidates).

These markets are *out of universe*: they are fetched (read-only HTTP, cached/uncached
``get_json``) into ``data/snapshots/extra_{trades_taker,prices,markets}.parquet`` and never into
DuckDB. ``score-markets`` scores them next to the pilot but they are excluded from the baseline
fit and from the percentile pools (``out_of_universe = True``).

Market mapping
--------------
* Maduro: unambiguous. ``Maduro out by January 31, 2026?`` (in ``pilot_markets.csv``).
* Axiom expose (ZachXBT, 2026-02-26) and Bubblemaps military cluster: the press names accounts only
  by username and never names the exact market, so the mapping is a *candidate set*
  (``mapping == "candidate"``). We score every candidate and never pick one.

News times (independent only if documented):
* Maduro: first public evidence 2026-01-03 06:00 UTC (AFP: blasts over Caracas about 2 am local);
  Trump announced the capture on Truth Social about 09:30 UTC (time zone conversion approximate).
  The earlier time is used, which keeps the pre-news window conservative.
* Axiom / Iran markets: no documented UTC minute, so ``news_ts_indep`` is null (price-based).

Tape window: the ``/v2/trades`` market feed has no time filter (start/end only apply to the
``user`` shape), pages are newest-first, so we page until the cutoff
(``closed_ts - TAPE_DAYS`` days) and stop. ``MAX_PAGES`` caps a huge market; if hit, the tape
starts later than the cutoff (``tape_complete`` False).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from pmdetect.store.db import PROJECT_ROOT

SNAPSHOTS = PROJECT_ROOT / "data" / "snapshots"
EXTRA_TRADES = SNAPSHOTS / "extra_trades_taker.parquet"
EXTRA_PRICES = SNAPSHOTS / "extra_prices.parquet"
EXTRA_MARKETS = SNAPSHOTS / "extra_markets.parquet"
EXTRA_WALLETS = SNAPSHOTS / "extra_wallets.parquet"
PARTS = SNAPSHOTS / "_extra_parts"
TRADES_URL = "https://data-api.polymarket.com/v2/trades"

TAPE_DAYS = 12
MAX_PAGES = 1500

MADURO = "0x580adc1327de9bf7c179ef5aaffa3377bb5cb252b7d6390b027172d43fd6f993"

# (case_name, condition_id prefix or full, mapping, news_ts_indep, indep_method, source)
_MADURO_NEWS = datetime(2026, 1, 3, 6, 0)
_MADURO_SRC = (
    "first public reports: blasts over Caracas ~06:00 UTC 2026-01-03 (AFP; capture announced by Trump "
    "on Truth Social ~09:30 UTC); "
    "e.g. https://www.aljazeera.com/news/2026/1/3/explosions-heard-over-venezuelan-capital-caracas-amid-us-tensions"
)
CASES: list[dict] = [
    {"case_name": "maduro_removal_jan_2026", "cid": MADURO, "mapping": "unambiguous",
     "news_ts_indep": _MADURO_NEWS, "indep_method": "override", "news_source": _MADURO_SRC},
]
_AXIOM = "axiom_expose_feb_2026"
for _cid in (
    "0xaa6667a28cdcb85bfa9af9ad00e79941c3b9162a28b4fdc8eba6b84dbbd32546",  # release 8-9am YES
    "0x6021471fe6d2a6405c37e147a2cf774514411410d7994ae40659e6c95fab24f5",  # >30M views
    "0x9c950f312ad1abef3040259558ae1cad2744c44358c772cec16acf6192a84bf9",  # >14M
    "0xec4851c0969216a93ed3e3910f62a57769c7c566f427021e01f690a8571d2a2b",  # >10M
    "0x9e6f70d70ff42d2f598483821f92cd8519ebbffc5887af51f53f3056ac3d6360",  # >8M
    "0x241b9abc285e4c44502da433a989af9016011a5f7e2273ae684d49d660af651b",  # >25M
):
    CASES.append({"case_name": _AXIOM, "cid": _cid, "mapping": "candidate", "news_ts_indep": None,
                  "indep_method": "none",
                  "news_source": "price-based (no documented UTC minute for the ZachXBT post)"})
_BUB = "bubblemaps_military_cluster"
for _cid in (
    "0x291f0f6023efa933d36cc80fd55f9d176309a92b61b9567fd4f044a8b21873fb",  # Israel strikes Iran by Feb 28
    "0xb3ebf217cf2f393a66030c072b04b893268506923e01b23f1bcf3504c3d319c2",  # Iran strike on Israel
    "0x41e5b3c5f0f57504ea391cf56fc7d4cdb43f951baa8780a9896b008b37fc4d8a",  # Iran strike on US mil
    "0xaffffdcf1ac14c2b3d6888f5efc76c0ea1aa5db4891588cc6e99f6e37e938acb",  # strike on Mar 1
    "0xba580e15cb600f9fb9a70c7a04a239e4df64a1d5c13780a5192525e4d1c71eec",  # Mar 2
    "0xd0f7f38de53e4ed1f1a33198491c848f90ac2f41e9086ffef58c64ff3eb63592",  # Mar 3
    "0x1dbe370933194e449ebec13b96fd1f5d2fde064c880d6b8db100f2730c50357c",  # Mar 4
    "0x3488f31e6449f9803f99a8b5dd232c7ad883637f1c86e6953305a2ef19c77f20",  # US strikes Iran by Feb 28 ($90M)
):
    CASES.append({"case_name": _BUB, "cid": _cid, "mapping": "candidate", "news_ts_indep": None,
                  "indep_method": "none",
                  "news_source": "price-based (no documented UTC minute for the 2026-02-28 strikes)"})


def load_extra(snapshot_dir: Path = SNAPSHOTS):
    """(markets, trades, prices) of the extra cases, or None when not fetched yet."""
    m, t, p = (snapshot_dir / n for n in
               ("extra_markets.parquet", "extra_trades_taker.parquet", "extra_prices.parquet"))
    if not (m.exists() and t.exists() and p.exists()):
        return None
    return pl.read_parquet(m), pl.read_parquet(t), pl.read_parquet(p)


def _page_trades(cid: str, cutoff: datetime, log) -> tuple[list[dict], bool]:
    """Page newest-first until ts < cutoff. Returns (rows, complete)."""
    from pmdetect.collect.http import get_json

    rows: list[dict] = []
    cursor = None
    cut = cutoff.replace(tzinfo=UTC).timestamp()
    for page_no in range(MAX_PAGES):
        p = {"condition": cid, "limit": 1000, "taker_only": "true", "filter_amount": 0}
        if cursor:
            p["cursor"] = cursor
        page = get_json(TRADES_URL, p, use_cache=False)
        data = page.get("data") or []
        rows.extend(r for r in data if (r.get("timestamp") or 0) >= cut)
        if page_no % 50 == 0:
            log(f"  {cid[:10]} page {page_no} rows {len(rows)}")
        if data and min(r.get("timestamp") or 0 for r in data) < cut:
            return rows, True
        cursor = (page.get("pagination") or {}).get("next_cursor")
        if not cursor:
            return rows, True
    return rows, False


def fetch_extra_cases(log=print) -> None:
    """Fetch tapes, prices and metadata of CASES into the extra_*.parquet snapshots."""
    from pmdetect.collect.data_api import fetch_price_series, trades_to_frame

    PARTS.mkdir(parents=True, exist_ok=True)
    mk = pl.read_parquet(SNAPSHOTS / "markets.parquet")
    tk = pl.read_parquet(SNAPSHOTS / "tokens.parquet")
    meta_rows, price_frames = [], []
    for case in CASES:
        cid = case["cid"]
        m = mk.filter(pl.col("condition_id") == cid)
        if m.height == 0:
            log(f"skip {cid}: not in markets snapshot")
            continue
        r = m.to_dicts()[0]
        toks = tk.filter(pl.col("condition_id") == cid)
        tid = {str(x["outcome_label"]).lower(): x["token_id"] for x in toks.to_dicts()}
        ref = r["closed_ts"] or r["end_ts"]
        cutoff = ref - timedelta(days=TAPE_DAYS)
        part = PARTS / f"{cid}.parquet"
        if part.exists():
            tr = pl.read_parquet(part)
            complete = True
            log(f"{cid[:10]} cached part {tr.height}")
        else:
            log(f"{cid[:10]} {r['question'][:60]} cutoff {cutoff}")
            rows, complete = _page_trades(cid, cutoff, log)
            tr = trades_to_frame(rows).unique(subset=["trade_id"]).sort("ts")
            tr.write_parquet(part)
            log(f"  -> {tr.height} taker rows complete={complete}")
        for token in tid.values():
            pts, _bucket = fetch_price_series(token)
            if pts:
                price_frames.append(pl.DataFrame({
                    "token_id": [token] * len(pts),
                    "ts": [datetime.fromtimestamp(int(p["timestamp"]), UTC).replace(tzinfo=None) for p in pts],
                    "price": [float(p["price"]) for p in pts],
                }, schema={"token_id": pl.Utf8, "ts": pl.Datetime("us"), "price": pl.Float64}))
        meta_rows.append({
            **{k: r[k] for k in ("condition_id", "slug", "question", "info_type", "volume_usd",
                                 "start_ts", "end_ts", "closed_ts", "outcome")},
            "case_name": case["case_name"], "mapping": case["mapping"],
            "yes_token_id": tid.get("yes"), "no_token_id": tid.get("no"),
            "tape_start": tr["ts"].min() if tr.height else None,
            "tape_rows": tr.height, "tape_complete": complete,
            "news_ts_indep": case["news_ts_indep"], "indep_method": case["indep_method"],
            "news_source": case["news_source"], "out_of_universe": True,
        })
    trades = pl.concat([pl.read_parquet(p) for p in sorted(PARTS.glob("*.parquet"))])
    trades.write_parquet(EXTRA_TRADES)
    pl.concat(price_frames).unique(subset=["token_id", "ts"]).write_parquet(EXTRA_PRICES)
    pl.DataFrame(meta_rows).write_parquet(EXTRA_MARKETS)
    log(f"done: {len(meta_rows)} markets, {trades.height} taker rows")


def extra_prices_labelled(markets: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Extra prices with condition_id and outcome_label ('Yes'/'No') via the token ids."""
    lab = pl.concat([
        markets.select(pl.col("yes_token_id").alias("token_id"), pl.lit("Yes").alias("outcome_label"),
                       "condition_id"),
        markets.select(pl.col("no_token_id").alias("token_id"), pl.lit("No").alias("outcome_label"),
                       "condition_id"),
    ])
    return prices.join(lab, on="token_id")


def fetch_extra_wallets(max_wallets: int = 6000, lookback_days: int = 7, log=print) -> None:
    """First-seen timestamps (``/v2/activity``, limit=1 ascending) for the winner-side wallets that
    traded in the ``lookback_days`` before each extra market's detected jump start.

    Written to ``extra_wallets.parquet`` (wallet, first_seen, n_markets = markets in the extra tape).
    Wallets already in ``wallets.parquet`` are skipped (flow uses both). Parquet only.
    """
    from pmdetect.collect.data_api import ACTIVITY_URL
    from pmdetect.collect.http import get_json
    from pmdetect.markets.changepoint import detect_jumps

    m, t, p = load_extra()
    jumps = detect_jumps(m, extra_prices_labelled(m, p), t).filter(pl.col("jump_start_ts").is_not_null())
    known = set(pl.read_parquet(SNAPSHOTS / "wallets.parquet", columns=["wallet"])["wallet"])
    done: dict[str, dict] = {}
    if EXTRA_WALLETS.exists():
        done = {r["wallet"]: r for r in pl.read_parquet(EXTRA_WALLETS).iter_rows(named=True)}
    cand: dict[str, float] = {}
    nmk = t.group_by("wallet").agg(pl.col("condition_id").n_unique().alias("n"))
    nmk = dict(nmk.iter_rows())
    for cid, js, outcome in jumps.join(m.select("condition_id", "outcome"), on="condition_id").select(
        "condition_id", "jump_start_ts", "outcome"
    ).iter_rows():
        win = outcome.lower()
        x = t.filter((pl.col("condition_id") == cid) & (pl.col("ts") < js)
                     & (pl.col("ts") >= js - timedelta(days=lookback_days)))
        x = x.filter(((pl.col("outcome_label").str.to_lowercase() == win) & (pl.col("side") == "BUY"))
                     | ((pl.col("outcome_label").str.to_lowercase() != win) & (pl.col("side") == "SELL")))
        for w, u in x.group_by("wallet").agg(pl.col("usd").sum()).iter_rows():
            cand[w] = cand.get(w, 0.0) + u
    todo = [w for w, _ in sorted(cand.items(), key=lambda kv: -kv[1])
            if w not in known and w not in done][:max_wallets]
    log(f"candidate wallets {len(cand)}; to fetch {len(todo)}")
    rows = list(done.values())
    for i, w in enumerate(todo):
        page = get_json(ACTIVITY_URL, {"user": w, "type": "TRADE", "start": 1,
                                       "sort_direction": "ASC", "limit": 1}, use_cache=False)
        d = page.get("data") or []
        if d:
            rows.append({"wallet": w,
                         "first_seen": datetime.fromtimestamp(int(d[0]["timestamp"]), UTC).replace(tzinfo=None),
                         "n_markets": int(nmk.get(w, 1))})
        if i % 250 == 249 or i == len(todo) - 1:
            pl.DataFrame(rows, schema={"wallet": pl.Utf8, "first_seen": pl.Datetime("us"),
                                       "n_markets": pl.Int64}).write_parquet(EXTRA_WALLETS)
            log(f"  wallets {i + 1}/{len(todo)}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    if len(sys.argv) > 1 and sys.argv[1] == "wallets":
        fetch_extra_wallets()
    else:
        fetch_extra_cases()
