"""Discrepancy score D per market (phase 2), adapted from Paulden (2016).

D = mean over the hourly grid (7 days .. 1 hour before news_ts) of max(0, q - m50), where q is the
winner's price and m50 the peer median of the chosen baseline scheme (time split by default, or
cross-fit; see baseline.py). D_signed is the mean of
q - m50, share_above_m90 the fraction of hours with q above the peer 90th percentile.

Phase 2b: the level baseline is fitted on markets with a price-independent news time only
(``newstime_indep``), and level-D (D, D_signed, share_above_m90) is scored only for those; every
other market gets D = null with reason ``price_based_news_time``. Late-window measures use each
market's best available news time (``news_ts_best``: independent time if any, else the price-based
``news_ts``):

* ``D_last24``: the same D restricted to the last 24 h of the window (tau_h <= 24), independent
  markets only, same baseline (needs >= 12 grid hours);
* ``max_jump`` / ``max_jump_hours_before``: the largest one-hour step of q toward the answer
  (q_t - q_{t-1h}, hourly grid, 7 d .. 1 h before news) and its end, in hours before news. For
  markets with a price-based news time the window ends 1 h before the 0.9 crossing, so the climb
  to 0.9 is partly cut off by construction; read those values with that in mind.

Time split: independent markets in an early window with too little earlier training data are
UNSCORABLE: D = null with reason ``no_prior_training_window`` (never scored in-sample). The
``baseline_scheme`` column records which scheme produced D.

Markets without news_ts / YES-NO outcome / enough price hours get NaN scores and a ``reason``;
none are dropped.

Run: ``python -m pmdetect.markets.score`` (reads data/snapshots, writes
data/snapshots/market_D.parquet, prints the distribution of D).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from pmdetect.markets.baseline import fit_predict_cross, fit_predict_timesplit
from pmdetect.markets.series import (
    SNAPSHOTS,
    load_inputs,
    load_winner_trades,
    q0_table,
    winner_series,
)

MIN_HOURS = 24
OUT_COLS = [
    "condition_id", "question", "info_type", "news_ts", "news_ts_method", "outcome", "q0",
    "q0_flag", "n_hours", "D", "D_signed", "share_above_m90", "reason",
    "news_ts_best", "news_ts_indep", "indep_method", "independent",
    "D_last24", "n_hours_last24", "max_jump", "max_jump_hours_before", "baseline_scheme",
]  # fmt: skip
MIN_HOURS_LATE = 12
LATE_H = 24


def score_markets(
    markets: pl.DataFrame,
    series: pl.DataFrame,
    baseline: pl.DataFrame,
    q0: pl.DataFrame,
    min_hours: int = MIN_HOURS,
    news_indep: pl.DataFrame | None = None,
    scheme: str | None = None,
    unscorable: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per market in ``markets`` (columns: see OUT_COLS).

    ``news_indep`` (condition_id, news_ts_indep, indep_method, independent) gates level-D and
    D_last24 to independent markets; None means every market counts as independent.
    ``scheme`` labels the baseline (``baseline_scheme``); ``unscorable`` (condition_id) lists
    markets whose time-split window had no usable earlier training set.
    """
    j = series.join(
        baseline.select("condition_id", "t", "m10", "m50", "m90"), on=["condition_id", "t"]
    )
    j = j.with_columns(
        (pl.col("q") - pl.col("m50")).alias("diff"),
    )
    late = (
        j.filter(pl.col("tau_h") <= LATE_H)
        .group_by("condition_id")
        .agg(
            pl.len().alias("n_hours_last24"),
            pl.col("diff").clip(lower_bound=0).mean().alias("D_last24"),
        )
    )
    step = (
        series.sort(["condition_id", "tau_h"], descending=[False, True])
        .with_columns((pl.col("q") - pl.col("q").shift(1).over("condition_id")).alias("step"))
        .filter(pl.col("step").is_not_null())
    )
    jumps = step.group_by("condition_id").agg(
        pl.col("step").max().alias("max_jump"),
        pl.col("tau_h").gather(pl.col("step").arg_max()).first().alias("max_jump_hours_before"),
    )
    agg = j.group_by("condition_id").agg(
        pl.len().alias("n_hours"),
        pl.col("diff").clip(lower_bound=0).mean().alias("D"),
        pl.col("diff").mean().alias("D_signed"),
        (pl.col("q") > pl.col("m90")).mean().alias("share_above_m90"),
    )
    if news_indep is None:
        ni = markets.select(
            "condition_id",
            pl.lit(None, dtype=pl.Datetime("us")).alias("news_ts_indep"),
            pl.lit(None, dtype=pl.String).alias("indep_method"),
            pl.lit(True).alias("independent"),
        )
    else:
        ni = news_indep.select("condition_id", "news_ts_indep", "indep_method", "independent")
    base = markets.select(
        "condition_id",
        *[
            c
            for c in ("question", "info_type", "news_ts", "news_ts_method", "outcome")
            if c in markets.columns
        ],
    )
    out = (
        base.join(ni, on="condition_id", how="left")
        .with_columns(
            pl.col("independent").fill_null(False),
            pl.coalesce("news_ts_indep", "news_ts").alias("news_ts_best"),
        )
        .join(q0, on="condition_id", how="left")
        .join(agg, on="condition_id", how="left")
        .join(late, on="condition_id", how="left")
        .join(jumps, on="condition_id", how="left")
        .join(
            series.group_by("condition_id").agg(pl.len().alias("_n_series")),
            on="condition_id",
            how="left",
        )
        .with_columns(pl.col("n_hours").fill_null(0), pl.col("n_hours_last24").fill_null(0))
    )
    unscor_ids = [] if unscorable is None else unscorable["condition_id"].to_list()
    out = out.with_columns(
        pl.when(pl.col("news_ts_best").is_null())
        .then(pl.lit("no_news_ts"))
        .when(~pl.col("outcome").is_in(["YES", "NO"]) | pl.col("outcome").is_null())
        .then(pl.lit("no_outcome"))
        .when(pl.col("_n_series").is_null())
        .then(pl.lit("no_price_data"))
        .when(~pl.col("independent"))
        .then(pl.lit("price_based_news_time"))
        .when(pl.col("info_type").is_null())
        .then(pl.lit("no_info_type"))
        .when(pl.col("condition_id").is_in(unscor_ids))
        .then(pl.lit("no_prior_training_window"))
        .when(pl.col("n_hours") < min_hours)
        .then(pl.lit("too_few_hours"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("reason")
    )
    bad = pl.col("reason").is_not_null()
    out = out.with_columns(
        [
            pl.when(bad).then(None).otherwise(pl.col(c)).alias(c)
            for c in ("D", "D_signed", "share_above_m90")
        ]
    )
    bad_late = bad | (pl.col("n_hours_last24") < MIN_HOURS_LATE)
    out = out.with_columns(
        pl.when(bad_late).then(None).otherwise(pl.col("D_last24")).alias("D_last24")
    )
    out = out.with_columns(
        pl.when(pl.col("D").is_not_null()).then(pl.lit(scheme)).otherwise(None).cast(pl.String)
        .alias("baseline_scheme")
    )
    return out.select([c for c in OUT_COLS if c in out.columns])


def run(
    snapshot_dir: Path = SNAPSHOTS,
    out_path: Path | None = None,
    seed: int = 0,
    scheme: str = "timesplit",
) -> pl.DataFrame:
    from pmdetect.markets.newstime_indep import build

    markets, prices = load_inputs(snapshot_dir)
    trades = load_winner_trades(snapshot_dir, markets)
    ni = build(snapshot_dir)  # also (re)writes news_independent.parquet
    markets = markets.join(ni, on="condition_id", how="left").with_columns(
        pl.coalesce("news_ts_indep", "news_ts").alias("news_ts_best")
    )
    series = winner_series(markets, prices, trades, news_col="news_ts_best")
    q0 = q0_table(series)
    ind_ids = ni.filter(pl.col("independent"))["condition_id"].to_list()
    it = markets.select("condition_id", "info_type")
    unscor = None
    if scheme == "timesplit":
        baseline, plan = fit_predict_timesplit(
            series, q0, it, markets.select("condition_id", pl.col("news_ts_best").alias("news_ts")),
            score_ids=ind_ids,
        )
        unscor = plan.filter(~pl.col("scorable"))
    else:
        baseline = fit_predict_cross(series, q0, it, seed=seed, independent_ids=ind_ids)
    scores = score_markets(
        markets, series, baseline, q0, news_indep=ni, scheme=scheme, unscorable=unscor
    )
    out_path = out_path or (snapshot_dir / "market_D.parquet")
    scores.write_parquet(out_path)
    return scores


def describe(scores: pl.DataFrame) -> str:
    ok = scores.filter(pl.col("D").is_not_null())
    lines = [f"markets: {scores.height}  scored: {ok.height}  NaN: {scores.height - ok.height}"]
    for r, n in scores.filter(pl.col("reason").is_not_null())["reason"].value_counts().iter_rows():
        lines.append(f"  NaN reason {r}: {n}")
    if ok.height:
        qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        lines.append("D quantiles: " + ", ".join(f"{q:.2f}={ok['D'].quantile(q):.4f}" for q in qs))
        lines.append(
            f"share D==0: {(ok['D'] == 0).mean():.3f}; young_market: {ok['q0_flag'].is_not_null().sum()}"
        )
        for it, g in ok.group_by("info_type"):
            lines.append(
                f"  {it[0]}: n={g.height} D median={g['D'].median():.4f} p90={g['D'].quantile(0.9):.4f}"
            )
        late = scores.filter(pl.col("D_last24").is_not_null())["D_last24"]
        if late.len():
            lines.append(
                "D_last24 quantiles (independent): "
                + ", ".join(f"{q:.2f}={late.quantile(q):.4f}" for q in qs)
            )
        mj = scores.filter(pl.col("max_jump").is_not_null())
        for flag, g in mj.group_by("independent"):
            lines.append(
                f"max_jump independent={flag[0]} n={g.height}: "
                + ", ".join(f"{q:.2f}={g['max_jump'].quantile(q):.3f}" for q in qs)
            )
        top = ok.sort("D", descending=True).head(10)
        lines.append("top 10:")
        for r in top.iter_rows(named=True):
            lines.append(
                f"  D={r['D']:.4f} q0={r['q0']:.2f} {r['info_type']} {str(r.get('question'))[:80]}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(describe(run()))
