"""Hourly winner-price series q(t) on the pre-news window (phase 2).

q(t) is the price of the eventual winning outcome token (YES price if YES won, else the NO price,
i.e. 1 - YES). Observations come from two sources merged on time:

* ``prices`` (``/prices-history``): coarse and uneven (10 min for recent short-lived markets,
  3 h or 12 h buckets for older ones);
* the taker tape when supplied (one row per fill; a fill on the losing token counts at
  ``1 - price``, so every fill contributes a winner price; only the price level is used, not
  direction or volume).

The grid is hourly and anchored on ``news_ts``: ``tau_h`` = 1, 2, ..., 168 hours before news
(window = 7 days to 1 hour before). q at a grid point is the last observation at or before it
(forward fill, never backward fill, so nothing from the future leaks in). Grid points before a
market's first observation are omitted (no invented history), and ``stale_h`` records the age of
the observation that was carried forward, so coarse 12 h buckets are visible downstream.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl

SNAPSHOTS = Path(__file__).resolve().parents[3] / "data" / "snapshots"
WINDOW_H = 168  # 7 days
BUFFER_H = 1  # news_ts is imprecise: stop one hour before it

SERIES_SCHEMA = {
    "condition_id": pl.String,
    "t": pl.Datetime("us"),
    "tau_h": pl.Float64,
    "q": pl.Float64,
    "stale_h": pl.Float64,
}


def tape_condition_ids(snapshot_dir: Path = SNAPSHOTS) -> pl.Series | None:
    """Distinct condition_ids of the full taker tape (None if the file is missing)."""
    path = snapshot_dir / "trades_taker.parquet"
    if not path.exists():
        return None
    return pl.scan_parquet(path).select("condition_id").unique().collect()["condition_id"]


def load_inputs(snapshot_dir: Path = SNAPSHOTS) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (markets, prices) from the Parquet snapshot, full universe.

    markets = every ``in_universe`` market plus any market that has rows in the taker tape (the
    Nobel validation market sits outside the universe filter but inside the tape), with the stored
    ``in_universe`` column. prices is joined with tokens (``condition_id``, ``outcome_label``) and
    restricted to those markets (non-pilot markets only have the winning token's history). Never
    opens DuckDB. Without a taker snapshot it falls back to ``in_universe`` only.
    """
    markets = pl.read_parquet(snapshot_dir / "markets.parquet")
    tokens = pl.read_parquet(snapshot_dir / "tokens.parquet")
    prices = pl.read_parquet(snapshot_dir / "prices.parquet")
    tape = tape_condition_ids(snapshot_dir)
    keep = pl.col("in_universe")
    if tape is not None:
        keep = keep | pl.col("condition_id").is_in(tape.implode())
    markets = markets.filter(keep)
    prices = prices.join(tokens, on="token_id", how="inner").filter(
        pl.col("condition_id").is_in(markets["condition_id"].implode())
    )
    return markets, prices


def load_winner_trades(
    snapshot_dir: Path = SNAPSHOTS, markets: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Taker-tape price points (condition_id, token_id, outcome_label, ts, price), full tape.

    One row per fill (``trades_taker.parquet``); ``winner_series`` turns a fill on the losing token
    into a winner price of ``1 - price``. Falls back to the all-rows ``trades.parquet`` if no
    taker snapshot exists.
    """
    cols = ["condition_id", "token_id", "outcome_label", "ts", "price"]
    path = snapshot_dir / "trades_taker.parquet"
    if not path.exists():
        path = snapshot_dir / "trades.parquet"
    lf = pl.scan_parquet(path).select(cols)
    if markets is not None:
        lf = lf.filter(pl.col("condition_id").is_in(markets["condition_id"].implode()))
    return lf.collect()


def _winner_label(outcome: str | None) -> str | None:
    o = (outcome or "").strip().upper()
    return o if o in ("YES", "NO") else None


def winner_series(
    markets: pl.DataFrame,
    prices: pl.DataFrame,
    trades: pl.DataFrame | None = None,
    window_h: int = WINDOW_H,
    buffer_h: int = BUFFER_H,
    news_col: str = "news_ts",
) -> pl.DataFrame:
    """Hourly winner-price grid per market: condition_id, t, tau_h, q, stale_h.

    Vectorised (one as-of join over all markets, no per-market loop).

    ``news_col`` names the column of ``markets`` that anchors the grid (phase 2b passes
    ``news_ts_best`` = independent news time where there is one, else the price-based ``news_ts``).

    ``prices`` needs token_id/condition_id/outcome_label/ts/price (see ``load_inputs``);
    ``trades`` (optional) the same columns, one row per fill (taker tape): a fill on the winning
    token counts at its price, a fill on the losing token at ``1 - price`` (the same winner price
    ``changepoint.winner_points`` uses), and trades win ties with a price observation at the same
    instant. Markets without news_ts, a YES/NO outcome, or any observation before news are simply
    absent here; ``score_markets`` reports why.
    """
    mk = (
        markets.select(
            "condition_id",
            pl.col(news_col).alias("_news"),
            pl.col("outcome").str.strip_chars().str.to_uppercase().alias("_win"),
        )
        .filter(pl.col("_news").is_not_null() & pl.col("_win").is_in(["YES", "NO"]))
    )
    if mk.height == 0:
        return pl.DataFrame(schema=SERIES_SCHEMA)
    lab = pl.col("outcome_label").str.strip_chars().str.to_uppercase()
    parts = []
    if prices.height:
        parts.append(
            prices.select("condition_id", "ts", "price", lab.alias("_lab"), pl.lit(0).alias("_src"))
            .join(mk.select("condition_id", "_win"), on="condition_id", how="inner")
            .filter(pl.col("_lab") == pl.col("_win"))
            .select("condition_id", "ts", pl.col("price").alias("q"), "_src")
        )
    if trades is not None and trades.height:
        parts.append(
            trades.select("condition_id", "ts", "price", lab.alias("_lab"), pl.lit(1).alias("_src"))
            .join(mk.select("condition_id", "_win"), on="condition_id", how="inner")
            .filter(pl.col("_lab").is_in(["YES", "NO"]))
            .select(
                "condition_id",
                "ts",
                pl.when(pl.col("_lab") == pl.col("_win"))
                .then(pl.col("price"))
                .otherwise(1.0 - pl.col("price"))
                .alias("q"),
                "_src",
            )
        )
    if not parts:
        return pl.DataFrame(schema=SERIES_SCHEMA)
    obs = (
        pl.concat(parts)
        .with_columns(pl.col("ts").cast(pl.Datetime("us")))
        .sort(["condition_id", "ts", "_src"], maintain_order=True)
        .unique(subset=["condition_id", "ts"], keep="last", maintain_order=True)
        .select("condition_id", pl.col("ts").alias("_ots"), "q")
        .sort("_ots")
    )
    taus = pl.DataFrame({"tau_h": np.arange(buffer_h, window_h + 1, dtype=float)})
    grid = (
        mk.select("condition_id", pl.col("_news").cast(pl.Datetime("us")))
        .join(taus, how="cross")
        .with_columns((pl.col("_news") - pl.duration(hours=pl.col("tau_h"))).alias("t"))
        .sort("t")
    )
    with warnings.catch_warnings():  # obs/grid are globally sorted on the as-of key (by-groups unchecked)
        warnings.simplefilter("ignore", UserWarning)
        j = grid.join_asof(
            obs, left_on="t", right_on="_ots", by="condition_id", strategy="backward"
        )
    return (
        j.filter(pl.col("_ots").is_not_null())
        .select(
            "condition_id",
            "t",
            "tau_h",
            pl.col("q").clip(0.0, 1.0),
            ((pl.col("t") - pl.col("_ots")).dt.total_microseconds() / 3.6e9).alias("stale_h"),
        )
        .sort(["condition_id", "tau_h"], descending=[False, True])
        .cast(SERIES_SCHEMA)
    )


def q0_table(series: pl.DataFrame, window_h: int = WINDOW_H) -> pl.DataFrame:
    """q0 per market: price 7 days before news, else the first available price ('young_market')."""
    if series.height == 0:
        return pl.DataFrame(
            schema={"condition_id": pl.String, "q0": pl.Float64, "q0_flag": pl.String}
        )
    s = series.sort(["condition_id", "tau_h"], descending=[False, True])
    first = s.group_by("condition_id", maintain_order=True).first()
    return first.select(
        "condition_id",
        pl.col("q").alias("q0"),
        pl.when(pl.col("tau_h") >= window_h)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.lit("young_market"))
        .alias("q0_flag"),
    )
