"""Price-independent news times (phase 2b).

Why: 398 of 399 pilot ``news_ts`` values are the first time the winner price crosses 0.9, so a
baseline of "price level before news" built on them is circular (the news time is defined by the
price having already arrived). A news time is *independent* here only if it is defined without
looking at the price path:

1. ``override``: a row in ``data/manual/news_time_overrides.csv`` (documented, cited announcement
   time; currently the Nobel Peace Prize 2025 market only);
2. ``scheduled_end``: ``markets.end_ts`` of a ``info_type == 'scheduled_reveal'`` market, but only
   where end_ts demonstrably is the event/resolution time and not a loose deadline. Rule, using
   trade timing (never the price level) as the evidence::

       scheduled_reveal
       AND  -12 h <= last_trade_ts - end_ts <= +60 h     (the market traded until the event and
                                                           stopped within ~a day of it; a deadline
                                                           that passed long after, or long before,
                                                           the tape ends is rejected)
       AND  -12 h <= closed_ts   - end_ts <= +96 h       (resolved shortly after end_ts)

   ``news_ts_indep = end_ts``. end_ts is usually date-level (00:00 / 12:00 UTC), typically a few
   hours to a day *before* the real reveal (Grammys, Oscars, Powell press conferences, data
   releases), which only makes the pre-news window conservative (it stops earlier). The opposite
   error (end_ts hours *after* the reveal, e.g. some election nights) puts a few post-news hours in
   the window; this is counted in the hand check below.

Everything else (``deadline``, ``unscheduled_event``, scheduled markets whose end_ts is a placeholder
far from the tape) gets ``indep_method == 'none'``, ``independent == False``. A 0.9 crossing is never
used here.

Hand check (see README.md, "Independent news times"): 30 accepted ``scheduled_end`` markets
sampled at random (seed 11) were inspected one by one (question, end_ts, closed_ts, winner price
path from 72 h before to 60 h after end_ts, real event time known from the question): 27 clean
(end_ts equals or precedes the real reveal), 3 contaminated (end_ts hours *after* the reveal: Lula
first round and the Vermont governor race (election-night result 5-8 h before end_ts, clear), and
AfD Sachsen-Anhalt (small, 0.08 step 8 h before end_ts)). Miss rate: 2/30 clear (6.7 %), 3/30
including the marginal one (10 %). A random sample of 12 of the 83 rejected scheduled_reveal markets
showed 7 correct rejections (placeholder end dates far from the tape) and 4-5 conservative false
rejections (a real election/primary/weekend whose tape ran >60 h past end_ts or sits on the -12 h
boundary): the rule errs on the side of exclusion.

Run: ``python -m pmdetect.markets.newstime_indep`` writes data/snapshots/news_independent.parquet
(every universe market; the last trade time comes from the full taker tape).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from pmdetect.markets.newstime import load_overrides, parse_ts
from pmdetect.markets.series import SNAPSHOTS

LAST_TRADE_MIN_H = -12.0
LAST_TRADE_MAX_H = 60.0
CLOSED_MIN_H = -12.0
CLOSED_MAX_H = 96.0

OUT_SCHEMA = {
    "condition_id": pl.String,
    "news_ts_indep": pl.Datetime("us"),
    "indep_method": pl.String,
    "independent": pl.Boolean,
}


def independent_news_times(
    markets: pl.DataFrame,
    last_trade: pl.DataFrame,
    overrides: tuple[dict, dict] | None = None,
) -> pl.DataFrame:
    """Return condition_id, news_ts_indep, indep_method, independent (one row per market).

    ``markets``: condition_id, slug (optional), info_type, end_ts, closed_ts.
    ``last_trade``: condition_id, last_trade_ts (any observation of the tape; use the last trade).
    ``overrides``: (by_condition_id, by_slug) as returned by ``newstime.load_overrides``.
    """
    by_cond, by_slug = overrides if overrides is not None else ({}, {})
    ov: dict[str, object] = {}
    for r in markets.iter_rows(named=True):
        row = by_cond.get(r["condition_id"].lower()) or by_slug.get(r.get("slug") or "")
        if row is not None:
            ov[r["condition_id"]] = parse_ts(row["news_ts"])
    df = markets.join(last_trade, on="condition_id", how="left")
    d_last = (pl.col("last_trade_ts") - pl.col("end_ts")).dt.total_minutes() / 60
    d_close = (pl.col("closed_ts") - pl.col("end_ts")).dt.total_minutes() / 60
    sched = (
        (pl.col("info_type") == "scheduled_reveal")
        & pl.col("end_ts").is_not_null()
        & d_last.is_between(LAST_TRADE_MIN_H, LAST_TRADE_MAX_H)
        & d_close.is_between(CLOSED_MIN_H, CLOSED_MAX_H)
    ).fill_null(False)
    df = df.with_columns(
        pl.col("condition_id")
        .map_elements(lambda c: ov.get(c), return_dtype=pl.Datetime("us"))
        .alias("_ov"),
    ).with_columns(
        pl.when(pl.col("_ov").is_not_null())
        .then(pl.lit("override"))
        .when(sched)
        .then(pl.lit("scheduled_end"))
        .otherwise(pl.lit("none"))
        .alias("indep_method")
    )
    return df.select(
        "condition_id",
        pl.when(pl.col("indep_method") == "override")
        .then(pl.col("_ov"))
        .when(pl.col("indep_method") == "scheduled_end")
        .then(pl.col("end_ts"))
        .otherwise(pl.lit(None, dtype=pl.Datetime("us")))
        .cast(pl.Datetime("us"))
        .alias("news_ts_indep"),
        "indep_method",
        (pl.col("indep_method") != "none").alias("independent"),
    )


def last_trade_times(snapshot_dir: Path = SNAPSHOTS) -> pl.DataFrame:
    """condition_id, last_trade_ts from the FULL taker tape (``trades_taker.parquet``).

    Falls back to the all-rows ``trades.parquet`` if no taker snapshot exists (pilot-only).
    """
    path = snapshot_dir / "trades_taker.parquet"
    if not path.exists():
        path = snapshot_dir / "trades.parquet"
    return (
        pl.scan_parquet(path)
        .group_by("condition_id")
        .agg(pl.col("ts").max().alias("last_trade_ts"))
        .collect()
    )


def build(snapshot_dir: Path = SNAPSHOTS, write: bool = True) -> pl.DataFrame:
    """Compute for every universe market (+ taker-tape markets) and write the parquet.

    Same rule as documented above; the last trade time now comes from the full taker tape.
    """
    from pmdetect.markets.series import load_inputs

    markets, _ = load_inputs(snapshot_dir)
    out = independent_news_times(markets, last_trade_times(snapshot_dir), load_overrides())
    if write:
        out.write_parquet(snapshot_dir / "news_independent.parquet")
    return out


def summarize(res: pl.DataFrame, markets: pl.DataFrame) -> pl.DataFrame:
    """Number of markets per info_type and indep_method (independent = method != 'none')."""
    return (
        res.join(markets.select("condition_id", "info_type"), on="condition_id", how="left")
        .group_by("info_type", "indep_method")
        .agg(pl.len().alias("n"))
        .sort("info_type", "indep_method")
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from pmdetect.markets.series import load_inputs

    res = build()
    m, _ = load_inputs()
    print(summarize(res, m))
    print("independent:", int(res["independent"].sum()), "of", res.height)
