"""Supporting market-level signals, computed on taker rows only in the pre-news window.

``/v2/trades`` lists every fill twice (a taker row and a maker row with the same
``transaction_hash``) and the store has no taker flag, so volume and direction measures must not
use the all-rows tape. ``fetch_taker_trades`` re-fetches the tape with ``taker_only=true`` (cached,
read-only, never touches DuckDB) and ``load_taker_trades`` exposes it from
``data/snapshots/trades_taker.parquet``.

Pre-news window: ``[news_ts - 7 d, news_ts - 1 h)``; the 1 h buffer absorbs news_ts imprecision.
A trade moves "toward the winner" when it buys the winning token or sells the losing token.
All signals are oriented so that a larger value is more suspicious. Empty or too-thin windows give
NaN (never an error) and a ``nan_reason``.
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.store.db import PROJECT_ROOT

SNAPSHOTS = PROJECT_ROOT / "data" / "snapshots"
TAKER_PARQUET = SNAPSHOTS / "trades_taker.parquet"
TRADES_URL = "https://data-api.polymarket.com/v2/trades"

WINDOW_START = timedelta(days=7)
WINDOW_END = timedelta(hours=1)
LAST_24H = timedelta(hours=24)
FRESH_DAYS = timedelta(days=7)
MIN_HIST_HOURS = 24
SIGNAL_COLS = ["ofi_24h", "vol_surge_z", "max_trade_jump", "fresh_share", "hhi"]
MAD_SCALE = 1.4826


# --------------------------------------------------------------------------- taker tape


def _tape_condition_ids() -> list[str]:
    """Pilot markets = every condition with trades in the all-rows snapshot."""
    t = pl.read_parquet(SNAPSHOTS / "trades.parquet", columns=["condition_id"])
    return sorted(t["condition_id"].unique().to_list())


def fetch_taker_trades(condition_ids: list[str] | None = None, out: Path = TAKER_PARQUET) -> Path:
    """Re-fetch trades with ``taker_only=true`` (batches of 20, cursor-paged, cached on disk).

    Writes parquet only; never writes into DuckDB.
    """
    from pmdetect.collect.data_api import MAX_CONDITIONS_PER_TRADES_CALL, trades_to_frame
    from pmdetect.collect.http import paginate_cursor

    ids = condition_ids or _tape_condition_ids()
    frames = []
    for i in range(0, len(ids), MAX_CONDITIONS_PER_TRADES_CALL):
        batch = ids[i : i + MAX_CONDITIONS_PER_TRADES_CALL]
        params = {
            "condition": ",".join(batch),
            "limit": 1000,
            "taker_only": "true",
            "filter_amount": 0,
        }
        rows = list(paginate_cursor(TRADES_URL, params))
        if rows:
            frames.append(trades_to_frame(rows))
        print(f"batch {i // 20 + 1}/{math.ceil(len(ids) / 20)}: {len(rows)} taker rows", flush=True)
    df = pl.concat(frames) if frames else trades_to_frame([])
    df = df.unique(subset=["trade_id"], keep="first").sort(["condition_id", "ts"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out


def load_taker_trades(refresh: bool = False, columns: list[str] | None = None) -> pl.DataFrame:
    """Taker-only trades (one row per fill). Fetches the tape if the parquet is missing.

    ``columns`` limits what is read (the full tape is 8.5M rows; the long string columns
    ``tx_hash`` / ``trade_id`` / ``token_id`` dominate memory and the market pipeline does not
    need them)."""
    if refresh or not TAKER_PARQUET.exists():
        fetch_taker_trades()
    return pl.read_parquet(TAKER_PARQUET, columns=columns)


def taker_vs_all_check(all_trades: pl.DataFrame, taker: pl.DataFrame) -> pl.DataFrame:
    """Per market: taker rows vs unique tx hashes of the all-rows tape, and USD/size checks."""
    a = all_trades.group_by("condition_id").agg(
        pl.len().alias("n_all"),
        pl.col("tx_hash").n_unique().alias("n_tx_all"),
        (pl.col("size").sum() / 2).alias("size_all_half"),
        (pl.col("usd").sum() / 2).alias("usd_all_half"),
    )
    t = taker.group_by("condition_id").agg(
        pl.len().alias("n_taker"),
        pl.col("tx_hash").n_unique().alias("n_tx_taker"),
        pl.col("size").sum().alias("size_taker"),
        pl.col("usd").sum().alias("usd_taker"),
    )
    j = a.join(t, on="condition_id", how="full", coalesce=True).fill_null(0)
    return j.with_columns(
        (pl.col("n_taker") == pl.col("n_tx_all")).alias("rows_eq_unique_tx"),
        (pl.col("size_taker") / pl.col("size_all_half").replace(0, None)).alias("size_ratio"),
        (pl.col("usd_taker") / pl.col("usd_all_half").replace(0, None)).alias("usd_ratio"),
    )


# --------------------------------------------------------------------------- signals


def _winner_label(outcome: str | None) -> str | None:
    return {"YES": "yes", "NO": "no"}.get(outcome or "")


def _first_seen_lookup(wallets: pl.DataFrame, taker: pl.DataFrame) -> pl.DataFrame:
    """wallet -> first_seen (wallets table), store_first (first trade in the store), n_markets."""
    store_first = taker.group_by("wallet").agg(pl.col("ts").min().alias("store_first"))
    w = wallets.select(
        "wallet",
        pl.col("first_seen"),
        pl.col("n_markets") if "n_markets" in wallets.columns else pl.lit(None).alias("n_markets"),
    ).unique(subset=["wallet"])
    return store_first.join(w, on="wallet", how="left")


def _nan_row(cid: str, reason: str) -> dict:
    return {
        "condition_id": cid,
        **{c: float("nan") for c in SIGNAL_COLS},
        "fresh_method": None,
        "fresh_cov": float("nan"),
        "n_trades_window": 0,
        "window_usd": 0.0,
        "nan_reason": reason,
    }


def _market_signals(
    cid: str, news_ts, outcome: str, tr: pl.DataFrame, fs: pl.DataFrame | None
) -> dict:
    """Signals of one market; ``fs`` (wallet first-seen lookup) is joined here unless ``tr``
    already carries ``first_seen`` / ``store_first`` / ``n_markets`` (``compute_signals`` joins
    once for the whole tape, so each market does not rebuild a hash of every wallet)."""
    win = _winner_label(outcome)
    start, end = news_ts - WINDOW_START, news_ts - WINDOW_END
    tr = tr.with_columns(pl.col("outcome_label").str.to_lowercase().alias("_lab"))
    tr = tr.filter(pl.col("_lab").is_in(["yes", "no"]) & (pl.col("usd") > 0))
    pre = tr.filter((pl.col("ts") >= start) & (pl.col("ts") < end)).sort("ts")
    if pre.height == 0:
        return _nan_row(cid, "no taker trades in pre-news window")
    is_win_tok = pl.col("_lab") == win
    is_buy = pl.col("side").str.to_uppercase() == "BUY"
    toward = (is_win_tok & is_buy) | (~is_win_tok & ~is_buy)
    pre = pre.with_columns(
        toward.alias("toward"),
        pl.when(toward).then(pl.col("usd")).otherwise(-pl.col("usd")).alias("signed_usd"),
        pl.when(is_win_tok).then(pl.col("price")).otherwise(1 - pl.col("price")).alias("pw"),
    )
    out = _nan_row(cid, "")
    out["n_trades_window"] = pre.height
    out["window_usd"] = float(pre["usd"].sum())
    reasons = []

    # ofi_24h
    last = pre.filter(pl.col("ts") >= end - LAST_24H)
    tot = float(last["usd"].sum()) if last.height else 0.0
    if tot > 0:
        out["ofi_24h"] = float(last["signed_usd"].sum()) / tot
    else:
        reasons.append("no trades in last 24h")

    # vol_surge_z: mean hourly volume of last 24h vs robust z against earlier hourly history
    full = tr.filter(pl.col("ts") < end)
    first_hour = full["ts"].min().replace(minute=0, second=0, microsecond=0)
    cut = end - LAST_24H
    n_hist = int((cut - first_hour).total_seconds() // 3600)
    if n_hist >= MIN_HIST_HOURS:
        hv = (
            full.with_columns(
                ((pl.col("ts") - first_hour).dt.total_seconds() // 3600).cast(pl.Int64).alias("h")
            )
            .group_by("h")
            .agg(pl.col("usd").sum())
        )
        hist = np.zeros(n_hist)
        for h, v in hv.iter_rows():
            if 0 <= h < n_hist:
                hist[h] = v
        recent = float(last["usd"].sum()) / 24.0
        med = float(np.median(hist))
        scale = MAD_SCALE * float(np.median(np.abs(hist - med)))
        if scale <= 0:
            scale = float(np.std(hist))  # zero-inflated history: MAD is 0, fall back to the std
        if scale > 0:
            out["vol_surge_z"] = (recent - med) / scale
        else:
            reasons.append("flat volume history")
    else:
        reasons.append(f"history < {MIN_HIST_HOURS}h for vol_surge_z")

    # max_trade_jump: largest single-trade increase of the winner price within the window
    if pre.height >= 2:
        out["max_trade_jump"] = float(max(0.0, pre["pw"].diff().drop_nulls().max()))
    else:
        reasons.append("<2 trades for max_trade_jump")

    # winner-side USD: fresh_share and hhi
    ws = pre.filter(pl.col("toward"))
    wtot = float(ws["usd"].sum()) if ws.height else 0.0
    if wtot > 0:
        per = ws.group_by("wallet").agg(pl.col("usd").sum())
        out["hhi"] = float(((per["usd"] / wtot) ** 2).sum())
        j = (ws if fs is None else ws.join(fs, on="wallet", how="left")).with_columns(
            pl.col("ts").sub(FRESH_DAYS).alias("_cut")
        )
        known = pl.col("first_seen").is_not_null()
        first = pl.when(known).then(pl.col("first_seen")).otherwise(pl.col("store_first"))
        fresh = (first > pl.col("_cut")) | (known & (pl.col("n_markets").fill_null(0) == 1))
        cov = float(j.filter(known)["usd"].sum()) / wtot
        out["fresh_share"] = float(j.filter(fresh)["usd"].sum()) / wtot
        out["fresh_cov"] = cov
        out["fresh_method"] = (
            "wallets_first_seen" if cov >= 0.95 else "store_fallback" if cov < 0.05 else "mixed"
        )
    else:
        reasons.append("no winner-side USD")
    out["nan_reason"] = "; ".join(reasons) if reasons else None
    return out


def compute_signals(markets: pl.DataFrame, taker_trades: pl.DataFrame, wallets: pl.DataFrame) -> pl.DataFrame:
    """One row per market in ``markets`` (needs condition_id, news_ts, outcome)."""
    fs = _first_seen_lookup(wallets, taker_trades)
    joined = taker_trades.join(fs, on="wallet", how="left")  # one join for the whole tape
    parts = {k[0]: v for k, v in joined.partition_by("condition_id", as_dict=True).items()}
    del joined
    rows = []
    for cid, news_ts, outcome in markets.select("condition_id", "news_ts", "outcome").iter_rows():
        if news_ts is None:
            rows.append(_nan_row(cid, "no news_ts"))
        elif _winner_label(outcome) is None:
            rows.append(_nan_row(cid, "no YES/NO outcome"))
        elif cid not in parts:
            rows.append(_nan_row(cid, "no taker trades"))
        else:
            rows.append(_market_signals(cid, news_ts, outcome, parts[cid], None))
    schema = {
        "condition_id": pl.Utf8,
        **{c: pl.Float64 for c in SIGNAL_COLS},
        "fresh_method": pl.Utf8,
        "fresh_cov": pl.Float64,
        "n_trades_window": pl.Int64,
        "window_usd": pl.Float64,
        "nan_reason": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)
