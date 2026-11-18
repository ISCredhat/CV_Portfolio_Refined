"""One-shot anomalous scoring of winning pre-cutoff positions (Phase 3b definitions).

Each winning position is scored as the geometric mean of five components in [0, 1]:

* ``c_improb``  : -log(entry price) (price clipped to [0.01, 0.99]); rarer win = higher. Min of the
                  within-market and the global percentile (absolute guard).
* ``c_size``    : USD relative to the median position USD of the same market; min of the within-market
                  percentile of that ratio and the global percentile of USD.
* ``c_early``   : PROXIMITY to the cutoff (name kept for the column). A FIXED function of h = hours before
                  the valid cutoff (independent news time, else jump start); not a percentile and not tuned::

                      proximity(h) = 1                    for h <= 72   (the last 3 days; entries inside
                                                                         the jump window are before the
                                                                         jump start by construction, so
                                                                         they are covered by this branch)
                      proximity(h) = exp(-(h - 72) / 72)  for h > 72    (e-folding time 72 h: 0.37 at
                                                                         6 days, 0.14 at 9 days, 0.05 at 12)

                  so early speculation scores low and a late, informed-looking entry scores high. Positions
                  with no valid cutoff (``valid_cutoff`` false / ``cutoff_method`` none) are marked
                  ``no_valid_cutoff``, get NO proximity credit (``c_early`` = 0, hence ``score`` = 0 and a
                  rank below every valid position) and are still screened: ``score_screen`` is the geometric
                  mean of the four other components and is what the watch tier uses for them.
* ``c_fresh``   : freshness = WALLET AGE AT ENTRY. Mean of the available sub-scores: youth at entry
                  (age = first_ts of the position minus the wallet's first_seen, clipped at 0; younger =
                  higher, global percentile of the age), fewer markets globally (``n_markets_global``, global
                  percentile, when known) and an only-market-in-store indicator (1.0 if the wallet has a
                  single event in the store, else 0.5). first_seen comes from wallets.parquet,
                  extra_wallets.parquet and the top-up file (:func:`merge_first_seen`).
* ``c_context`` : market context percentile: mean of the available among within-event winner percentile
                  (within_event.parquet), within_event_pct, flow_score_pct and D_last24_pct
                  (market_scores.csv); max over the markets of the event.

Percentiles of improbability and size are computed within market (event) over ALL pre-cutoff positions of
that market; markets with fewer than ``MIN_GROUP`` positions fall back to the global percentile. Missing
components are imputed with the neutral value 0.5 and counted in ``n_missing`` (context missing);
unavailable freshness sub-scores are counted in ``n_fresh_missing`` (c_fresh averages those that exist; the
only-market sub-score always exists). ``score_product`` is the raw product; ``score`` = product ** (1/5),
the geometric mean, which is what ranks. ``score_nofresh`` is the geometric mean of the four non-freshness
components (proximity credit 0 for no-valid-cutoff rows) and ``score_cand`` ranks the first_seen top-up
candidate set (see :func:`topup_candidates`).
"""

from __future__ import annotations

import polars as pl

PROXIMITY_FLAT_H = 72.0
PROXIMITY_TAU_H = 72.0
N_CAND_VALID = 2000
N_CAND_NOVALID = 1000
MIN_GROUP = 5
NEUTRAL = 0.5
COMPONENTS = ("c_improb", "c_size", "c_early", "c_fresh", "c_context")


def proximity_expr(h: pl.Expr) -> pl.Expr:
    """Fixed proximity function of hours before the cutoff (see the module docstring)."""
    return (
        pl.when(h <= PROXIMITY_FLAT_H)
        .then(1.0)
        .otherwise(((h - PROXIMITY_FLAT_H) / -PROXIMITY_TAU_H).exp())
    )


def proximity(h: float) -> float:
    """Scalar proximity(h): 1 for h <= 72, exp(-(h-72)/72) beyond."""
    import math

    return 1.0 if h <= PROXIMITY_FLAT_H else math.exp(-(h - PROXIMITY_FLAT_H) / PROXIMITY_TAU_H)


def merge_first_seen(
    wallets: pl.DataFrame | None, topup: pl.DataFrame | None
) -> pl.DataFrame | None:
    """Combine store first_seen with the top-up file: store values win, the top-up fills the gaps.

    Output columns: wallet, first_seen, n_markets (null for top-up-only wallets).
    """
    parts = []
    if wallets is not None and wallets.height:
        parts.append(
            wallets.select(
                "wallet", "first_seen", pl.col("n_markets").cast(pl.Int64), pl.lit(0).alias("_pri")
            )
        )
    if topup is not None and topup.height:
        parts.append(
            topup.select(
                "wallet",
                "first_seen",
                pl.lit(None, pl.Int64).alias("n_markets"),
                pl.lit(1).alias("_pri"),
            ).with_columns(pl.col("first_seen").cast(pl.Datetime("us")))
        )
    if not parts:
        return wallets
    both = pl.concat(parts).with_columns(pl.col("first_seen").cast(pl.Datetime("us")))
    return (
        both.sort(["wallet", pl.col("first_seen").is_null(), "_pri"])
        .unique("wallet", keep="first", maintain_order=True)
        .drop("_pri")
    )


def _pct(col: str, key: str, out: str, df: pl.DataFrame) -> pl.DataFrame:
    """Within-key average-rank percentile (rank/n), global fallback for small groups."""
    return (
        df.with_columns(
            (pl.col(col).rank("average").over(key) / pl.len().over(key)).alias("_w"),
            (pl.col(col).rank("average") / pl.len()).alias("_g"),
            pl.len().over(key).alias("_n"),
        )
        .with_columns(
            pl.when(pl.col("_n") >= MIN_GROUP).then(pl.col("_w")).otherwise(pl.col("_g")).alias(out)
        )
        .drop("_w", "_g", "_n")
    )


def _context(
    markets: pl.DataFrame | None,
    market_scores: pl.DataFrame | None,
    within_event: pl.DataFrame | None,
) -> pl.DataFrame:
    """Context percentile keyed by market key (event_id and condition_id both), column ``c_context``."""
    parts = []
    emap = (
        markets.select("condition_id", "event_id").drop_nulls().unique("condition_id")
        if markets is not None and {"condition_id", "event_id"} <= set(markets.columns)
        else pl.DataFrame(schema={"condition_id": pl.String, "event_id": pl.String})
    )
    if market_scores is not None and "condition_id" in market_scores.columns:
        cols = [
            c
            for c in ("within_event_pct", "flow_score_pct", "D_last24_pct")
            if c in market_scores.columns
        ]
        if cols:
            ms = market_scores.select("condition_id", *cols).with_columns(
                pl.mean_horizontal(*[pl.col(c) for c in cols]).alias("ctx")
            )
            parts.append(ms.select("condition_id", "ctx"))
    if within_event is not None and "winner_pct_24h" in within_event.columns:
        we = within_event.select("condition_id", pl.col("winner_pct_24h").alias("ctx")).drop_nulls()
        parts.append(we)
    if not parts:
        return pl.DataFrame(schema={"key": pl.String, "c_context": pl.Float64})
    cond = (
        pl.concat(parts)
        .drop_nulls("ctx")
        .group_by("condition_id")
        .agg(pl.col("ctx").mean().alias("c"))
    )
    by_cond = cond.select(pl.col("condition_id").alias("key"), pl.col("c").alias("c_context"))
    by_event = (
        cond.join(emap, on="condition_id")
        .group_by("event_id")
        .agg(pl.col("c").max().alias("c_context"))
        .select(pl.col("event_id").alias("key"), "c_context")
    )
    return pl.concat([by_event, by_cond]).unique("key")


def score_oneshot(
    positions: pl.DataFrame,
    markets: pl.DataFrame | None,
    wallets: pl.DataFrame | None,
    market_scores: pl.DataFrame | None,
    within_event: pl.DataFrame | None,
) -> pl.DataFrame:
    """Score winning pre-cutoff positions; returns one row per winning position, best first."""
    p = positions
    if "event_id" not in p.columns:
        p = p.with_columns(pl.col("condition_id").alias("event_id"))
    if "condition_id" not in p.columns:
        p = p.with_columns(pl.col("event_id").alias("condition_id"))
    p = p.with_columns(pl.col("event_id").fill_null(pl.col("condition_id")).alias("mkey"))
    if "valid_cutoff" in p.columns:
        valid = pl.col("valid_cutoff").fill_null(False)
    elif "cutoff_method" in p.columns:
        valid = pl.col("cutoff_method").is_in(["independent_news", "independent_burst", "jump_start"])
    else:
        valid = pl.lit(True)
    p = p.with_columns((valid & pl.col("hours_before_cutoff").is_not_null()).alias("_valid"))
    # pre-cutoff only (defensive); rows without any cutoff keep a null hours_before_cutoff
    p = p.filter(pl.col("hours_before_cutoff").is_null() | (pl.col("hours_before_cutoff") > 0))
    p = p.with_columns((~pl.col("_valid")).alias("no_valid_cutoff"))
    p = p.with_columns(
        (-pl.col("entry_price").clip(0.01, 0.99).log()).alias("_improb"),
        (pl.col("usd") / pl.col("usd").median().over("mkey")).alias("_size"),
    )
    p = _pct("_improb", "mkey", "c_improb", p)
    p = _pct("_size", "mkey", "c_size", p)
    # absolute guard: a within-market extreme must also be extreme globally (a $5 win at 0.986 in a
    # market where everyone trades at 0.997 is not improbable or large), so take the min of the two
    for src, dst in (("_improb", "c_improb"), ("usd", "c_size")):
        p = p.with_columns(
            pl.min_horizontal(pl.col(dst), pl.col(src).rank("average") / pl.len()).alias(dst)
        )
    p = p.with_columns(
        pl.when(pl.col("_valid"))
        .then(proximity_expr(pl.col("hours_before_cutoff")))
        .otherwise(0.0)
        .alias("c_early")
    )

    # wallet-level freshness inputs
    nev = p.group_by("wallet").agg(pl.col("mkey").n_unique().alias("n_events_store"))
    p = p.join(nev, on="wallet", how="left")
    if wallets is not None and wallets.height:
        w = wallets.select(
            "wallet", "first_seen", pl.col("n_markets").cast(pl.Float64).alias("n_markets_global")
        ).unique("wallet")
        p = p.join(w, on="wallet", how="left")
    else:
        p = p.with_columns(
            pl.lit(None, pl.Datetime("us")).alias("first_seen"),
            pl.lit(None, pl.Float64).alias("n_markets_global"),
        )
    p = p.with_columns(
        ((pl.col("first_ts") - pl.col("first_seen")).dt.total_seconds() / 86400.0)
        .clip(lower_bound=0)
        .alias("_age_days"),
        (pl.col("n_events_store") == 1).alias("only_market_store"),
    )
    p = p.with_columns(
        (1 - pl.col("_age_days").rank("average") / pl.col("_age_days").count()).alias("_s_age"),
        (1 - pl.col("n_markets_global").rank("average") / pl.col("n_markets_global").count()).alias(
            "_s_nm"
        ),
        pl.when(pl.col("only_market_store")).then(1.0).otherwise(0.5).alias("_s_only"),
    ).with_columns(pl.mean_horizontal("_s_age", "_s_nm", "_s_only").alias("c_fresh"))

    p = p.with_columns(
        pl.sum_horizontal(pl.col("_s_age").is_null(), pl.col("_s_nm").is_null())
        .cast(pl.Int32)
        .alias("n_fresh_missing")
    )
    ctx = _context(markets, market_scores, within_event)
    p = (
        p.join(ctx.rename({"key": "mkey"}), on="mkey", how="left")
        if ctx.height
        else p.with_columns(pl.lit(None, pl.Float64).alias("c_context"))
    )
    if ctx.height:  # fall back to the position's condition_id key
        c2 = ctx.rename({"key": "condition_id", "c_context": "_c2"})
        p = (
            p.join(c2, on="condition_id", how="left")
            .with_columns(pl.col("c_context").fill_null(pl.col("_c2")))
            .drop("_c2")
        )

    p = p.with_columns(
        pl.sum_horizontal(*[pl.col(c).is_null().cast(pl.Int32) for c in COMPONENTS]).alias(
            "n_missing"
        )
    ).with_columns(*[pl.col(c).fill_null(NEUTRAL).fill_nan(NEUTRAL) for c in COMPONENTS])
    p = p.with_columns(
        pl.reduce(lambda a, b: a * b, [pl.col(c) for c in COMPONENTS]).alias("score_product")
    ).with_columns((pl.col("score_product") ** (1 / len(COMPONENTS))).alias("score"))
    others = [c for c in COMPONENTS if c != "c_early"]
    nofresh = [c for c in COMPONENTS if c != "c_fresh"]
    three = [c for c in nofresh if c != "c_early"]
    geo = lambda cols: pl.reduce(lambda a, b: a * b, [pl.col(c) for c in cols]) ** (1 / len(cols))  # noqa: E731
    p = p.with_columns(
        geo(others).alias("score_screen"),
        geo(nofresh).alias("score_nofresh"),
        # candidate ranking for the top-up: valid rows by the 4-component score, no-valid rows by the
        # 3-component screen (no proximity, no freshness)
        pl.when(pl.col("no_valid_cutoff"))
        .then(geo(three))
        .otherwise(geo(nofresh))
        .alias("score_cand"),
        ((pl.col("first_ts") - pl.col("first_seen")).dt.total_seconds() / 86400.0)
        .clip(lower_bound=0)
        .alias("wallet_age_days_at_entry"),
    )

    win = p.filter(pl.col("won") == True)  # noqa: E712
    keep = [
        c
        for c in (
            "wallet",
            "event_id",
            "condition_id",
            "usd",
            "entry_price",
            "hours_before_cutoff",
            "first_ts",
            "first_seen",
            "wallet_age_days_at_entry",
            "no_valid_cutoff",
            "cutoff_method",
            "only_market_store",
            "n_markets_global",
            "info_type",
            *COMPONENTS,
            "n_missing",
            "n_fresh_missing",
            "score_product",
            "score",
            "score_screen",
            "score_nofresh",
            "score_cand",
        )
        if c in win.columns
    ]
    out = win.select(keep).sort(["score", "score_screen"], descending=True)
    if markets is not None and "question" in markets.columns:
        q = markets.select("condition_id", "question").unique("condition_id")
        out = out.join(q, on="condition_id", how="left")
    return out.with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))


def topup_candidates(
    positions: pl.DataFrame,
    markets: pl.DataFrame | None,
    market_scores: pl.DataFrame | None,
    within_event: pl.DataFrame | None,
    known_first_seen: pl.DataFrame | None,
    n_valid: int = N_CAND_VALID,
    n_novalid: int = N_CAND_NOVALID,
) -> pl.DataFrame:
    """Wallets of the one-shot candidate set that still lack a first_seen.

    Candidate set (explicit): winning positions scored WITHOUT the freshness component (first_seen unknown
    to the scorer), the top ``n_valid`` positions with a valid cutoff by ``score_nofresh`` (improbability,
    size, proximity, context) plus the top ``n_novalid`` positions without a valid cutoff by the
    3-component screen. Returns wallet, rank (best position first), no_valid_cutoff, score_cand for wallets
    whose first_seen is not in ``known_first_seen`` (non-null first_seen counts as known).
    """
    sc = score_oneshot(positions, markets, None, market_scores, within_event)
    cand = pl.concat(
        [
            sc.filter(~pl.col("no_valid_cutoff")).sort("score_cand", descending=True).head(n_valid),
            sc.filter(pl.col("no_valid_cutoff"))
            .sort("score_cand", descending=True)
            .head(n_novalid),
        ]
    ).sort("score_cand", descending=True)
    cand = cand.unique("wallet", keep="first", maintain_order=True).select(
        "wallet", "no_valid_cutoff", "score_cand"
    )
    if known_first_seen is not None and known_first_seen.height:
        known = known_first_seen.filter(pl.col("first_seen").is_not_null())["wallet"]
        cand = cand.filter(~pl.col("wallet").is_in(known.implode()))
    return cand.with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
