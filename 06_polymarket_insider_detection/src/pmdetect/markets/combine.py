"""Combine the market-level measures into one ranking (phase 2b).

Direction: every component is oriented so that a LARGER value is MORE suspicious.

Components (``COMPONENTS``):

* ``D``        level discrepancy D, independent-news-time markets only (null otherwise);
* ``D_last24`` the same restricted to the last 24 h before the independent news time;
* ``max_jump`` largest hourly step of q toward the answer before the market's best news time;
* ``flow_score`` flow-before-the-move: mean of the percentiles of the available robust z of
  ``fb_ofi_z, fb_fresh_z, fb_hhi_z, fb_vol_z`` (needs >= ``MIN_FLOW`` of them), i.e. the 24 h
  before the detected jump start versus the market's own earlier history. One aggregate so four
  correlated flow measures do not count as four independent tests in Fisher. Since phase 2c the
  window runs from 24 h before the jump start to the half-jump point (at most 2 h into the jump).
* ``within_event`` (phase 2c, needs ``data/snapshots/within_event.parquet`` joined as ``we_*``
  columns): mean of the within-info_type/group percentiles of the available
  ``we_diff_24h, we_diff_12h, we_winner_pct_24h, we_winner_pct_12h``. LARGER = more suspicious:
  ``diff`` = the winner's pre-reveal log-odds move minus the median losing sibling's move;
  ``winner_pct`` = the winner's move as a percentile among all outcomes of its event. Percentiles
  are taken within (info_type, group) with group in {multi, binary}, so single-outcome markets are
  only compared with each other. Missing (not a scheduled-reveal market, no siblings, no price data)
  means null: the component is simply not available (needs >= 1 of the four). One aggregate so the
  four correlated numbers count once in Fisher.

The older news-time-based signals (``ofi_24h`` ... ``hhi``) stay in the table for information but
are NOT components: their window is anchored on a price-defined news time.

Percentiles are computed within ``info_type`` among the *pool* (``in pool`` = not
``out_of_universe``): ``pct = average_rank_ascending / n_valid`` in (0, 1]; ties share the average
rank. Out-of-universe markets (extra labelled cases) are never part of a pool; they receive the
percentile they would have had, ``(n_pool_below + 0.5 * n_pool_equal) / n_pool``, and likewise
"would-be" ranks and flags.

Missing data: a market simply has fewer components (NaN percentile = not available). Both
combinations need at least ``MIN_COMPONENTS`` (2) available components, otherwise the combined
score is null (``n_components`` says why). Consequently markets with a price-based news time (no
level D, no D_last24) and no flow measures have only ``max_jump`` and get no combined score. The
Fisher statistic has 2k degrees of freedom for the k available components, so markets with
different k stay comparable on the p scale, but a market with more available components can reach a
smaller p; this is a heuristic ordering.

Combinations:

* ``fisher``: per component p = 1 - pct + 1/n_valid (largest value in a group gets p = 1/n, never
  0). X = -2 * sum(ln p_i), combined p from chi-square(2k); ``fisher_score = -log10(p)``. CAVEAT:
  the components are correlated and p is a rank-based pseudo p-value, not a probability.
* ``rank_mean``: mean of the available percentiles.

Top 1% flags: the ``ceil(0.01 * n_scored_pool)`` highest scores among pool markets (ties at the
cut-off all flagged), for Fisher and rank-mean. An out-of-universe market is flagged
("would be flagged") if it reaches the pool's cut-off. The IsolationForest on the percentile matrix
(NaN -> 0.5, fixed seed, fitted on the pool, applied to all) is a cross-check only.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import IsolationForest

FLOW_Z_COLS = ["fb_ofi_z", "fb_fresh_z", "fb_hhi_z", "fb_vol_z"]
WE_COLS = ["we_diff_24h", "we_diff_12h", "we_winner_pct_24h", "we_winner_pct_12h"]
COMPONENTS = ["D", "D_last24", "max_jump", "flow_score", "within_event"]
MIN_WE = 1
MIN_COMPONENTS = 2
MIN_FLOW = 2
TOP_FRAC = 0.01
SEED = 0


def percentiles_by_type(
    df: pl.DataFrame, cols: list[str], type_col: str = "info_type", pool_col: str | None = None
) -> pl.DataFrame:
    """Add ``{col}_pct`` (average-rank percentile within ``type_col``; NaN/null stays null).

    With ``pool_col`` (boolean, True = in pool) the percentile of pool rows is computed among the
    pool only; other rows get the percentile they would have against the pool.
    """
    out = df
    types = df[type_col].to_list()
    pool = (
        np.ones(df.height, dtype=bool)
        if pool_col is None
        else df[pool_col].fill_null(True).to_numpy().astype(bool)
    )
    for c in cols:
        x = df[c].cast(pl.Float64).fill_nan(None).to_numpy(allow_copy=True)
        x = np.where(np.isnan(x.astype(float)), np.nan, x).astype(float)
        pct = np.full(df.height, np.nan)
        nn = np.zeros(df.height)
        for t in set(types):
            idx = np.array([i for i, tt in enumerate(types) if tt == t])
            ok = idx[~np.isnan(x[idx])]
            pool_idx = ok[pool[ok]]
            n = len(pool_idx)
            nn[idx] = n
            if n == 0:
                continue
            pv = np.sort(x[pool_idx])
            for i in ok:
                less = np.searchsorted(pv, x[i], side="left")
                eq = np.searchsorted(pv, x[i], side="right") - less
                if pool[i]:
                    pct[i] = (less + (eq + 1) / 2.0) / n  # average rank incl. itself
                else:
                    pct[i] = (less + 0.5 * eq) / n
        out = out.with_columns(
            pl.Series(f"{c}_pct", pct, dtype=pl.Float64, nan_to_null=True),
            pl.Series(f"{c}_n", nn, dtype=pl.Float64),
        )
    return out


def _top_k(n_scored: int) -> int:
    return max(1, math.ceil(TOP_FRAC * n_scored))


def _flag_top(score: np.ndarray, k: int, pool: np.ndarray) -> tuple[np.ndarray, float]:
    """Flag scores >= the k-th highest pool score (ties included); returns (flags, cut-off)."""
    valid = ~np.isnan(score) & pool
    flag = np.zeros(len(score), dtype=bool)
    if valid.sum() == 0:
        return flag, float("nan")
    thr = float(np.sort(score[valid])[::-1][min(k, int(valid.sum())) - 1])
    flag[~np.isnan(score)] = score[~np.isnan(score)] >= thr
    return flag, thr


def _would_be_rank(score: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Rank (1 = highest) among pool scores; non-pool rows get 1 + number of pool scores above."""
    ps = score[pool & ~np.isnan(score)]
    out = np.full(len(score), np.nan)
    for i, v in enumerate(score):
        if not np.isnan(v):
            out[i] = 1 + int((ps > v).sum())
    return out


def add_flow_score(
    df: pl.DataFrame, prefix: str = "", out: str = "flow_score"
) -> pl.DataFrame:
    """``flow_score`` = mean percentile of the available flow z columns (>= MIN_FLOW), else null.

    ``prefix``/``out`` let the phase-2b window (``old_`` columns) be scored for comparison.
    """
    zc = [prefix + c for c in FLOW_Z_COLS if prefix + c in df.columns]
    if not zc:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(out))
    df = percentiles_by_type(df, zc, pool_col="out_of_universe_pool")
    pcts = pl.concat_list([pl.col(f"{c}_pct") for c in zc])
    return df.with_columns(
        pl.when(pcts.list.drop_nulls().list.len() >= MIN_FLOW)
        .then(pcts.list.drop_nulls().list.mean())
        .otherwise(None)
        .cast(pl.Float64)
        .alias(out)
    )


def add_within_event(df: pl.DataFrame) -> pl.DataFrame:
    """``within_event`` = mean percentile (within info_type and group) of the available ``we_*``."""
    wc = [c for c in WE_COLS if c in df.columns]
    if not wc:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("within_event"))
    grp = pl.col("we_group").fill_null("none") if "we_group" in df.columns else pl.lit("none")
    df = df.with_columns((pl.col("info_type").cast(pl.Utf8).fill_null("none") + "|" + grp).alias("_we_key"))
    df = percentiles_by_type(df, wc, type_col="_we_key", pool_col="out_of_universe_pool")
    pcts = pl.concat_list([pl.col(f"{c}_pct") for c in wc])
    df = df.with_columns(
        pl.when(pcts.list.drop_nulls().list.len() >= MIN_WE)
        .then(pcts.list.drop_nulls().list.mean())
        .otherwise(None)
        .cast(pl.Float64)
        .alias("within_event")
    )
    return df.drop("_we_key")


def combine(scores: pl.DataFrame, signals: pl.DataFrame) -> pl.DataFrame:
    """Join scores (needs condition_id, info_type) with signals and add rankings and flags.

    ``out_of_universe`` (bool, optional column of ``scores`` or ``signals``) marks extra labelled
    cases that are scored but excluded from every pool and flag cut-off.
    """
    sig_cols = [c for c in signals.columns if c not in scores.columns or c == "condition_id"]
    df = scores.join(signals.select(sig_cols), on="condition_id", how="left")
    if "out_of_universe" not in df.columns:
        df = df.with_columns(pl.lit(False).alias("out_of_universe"))
    df = df.with_columns(pl.col("out_of_universe").fill_null(False))
    df = df.with_columns((~pl.col("out_of_universe")).alias("out_of_universe_pool"))
    df = add_flow_score(df)
    df = add_flow_score(df, prefix="old_", out="flow_score_old")  # comparison only, not a component
    df = add_within_event(df)
    comps = [c for c in COMPONENTS if c in df.columns]
    # flow_score is already a percentile mean; rank it again within info_type like the others
    df = percentiles_by_type(df, comps, pool_col="out_of_universe_pool")
    pool = df["out_of_universe_pool"].to_numpy().astype(bool)

    pct = np.column_stack([df[f"{c}_pct"].to_numpy() for c in comps]).astype(float)
    n_valid = np.column_stack([df[f"{c}_n"].to_numpy() for c in comps]).astype(float)
    avail = ~np.isnan(pct)
    k = avail.sum(axis=1)
    ok = k >= MIN_COMPONENTS

    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.clip(1.0 - pct + 1.0 / np.maximum(n_valid, 1.0), 1e-12, 1.0)
        stat = np.where(avail, -2.0 * np.log(p), 0.0).sum(axis=1)
        p_f = stats.chi2.sf(stat, 2 * np.maximum(k, 1))
        fisher_score = -np.log10(np.clip(p_f, 1e-300, 1.0))
        rank_mean = np.where(avail, pct, 0.0).sum(axis=1) / np.maximum(k, 1)
    fisher_score = np.where(ok, fisher_score, np.nan)
    p_f = np.where(ok, p_f, np.nan)
    rank_mean = np.where(ok, rank_mean, np.nan)

    # isolation forest cross-check on the percentile matrix (NaN -> 0.5), fitted on the pool
    iso = np.full(len(df), np.nan)
    fit_rows = ok & pool
    if fit_rows.sum() >= 10:
        X = np.where(avail, pct, 0.5)
        model = IsolationForest(n_estimators=300, random_state=SEED).fit(X[fit_rows])
        iso[ok] = -model.score_samples(X[ok])

    n_scored = int((ok & pool).sum())
    kk = _top_k(n_scored)
    flag_f, thr_f = _flag_top(fisher_score, kk, pool)
    flag_r, thr_r = _flag_top(rank_mean, kk, pool)
    flag_i, _ = _flag_top(iso, kk, pool)

    best = np.where(avail, pct, -1.0).argmax(axis=1)
    driver = [comps[b] if o else None for b, o in zip(best, ok, strict=True)]

    df = df.with_columns(
        pl.Series("n_components", np.where(ok, k, 0), dtype=pl.Int64),
        pl.Series("fisher_p", p_f, dtype=pl.Float64, nan_to_null=True),
        pl.Series("fisher_score", fisher_score, dtype=pl.Float64, nan_to_null=True),
        pl.Series("rank_mean", rank_mean, dtype=pl.Float64, nan_to_null=True),
        pl.Series("iso_score", iso, dtype=pl.Float64, nan_to_null=True),
        pl.Series("flag_fisher", flag_f),
        pl.Series("flag_rank_mean", flag_r),
        pl.Series("flag_iso", flag_i),
        pl.Series("main_driver", driver, dtype=pl.Utf8),
    )
    if "flow_score_old" in df.columns:  # rank among the pool for the old-vs-new table
        df = percentiles_by_type(df, ["flow_score_old"], pool_col="out_of_universe_pool")
    for col in (*comps, "flow_score_old", "fisher_score", "rank_mean", "iso_score"):
        if col not in df.columns:
            continue
        v = df[col].cast(pl.Float64).fill_nan(None).to_numpy(allow_copy=True).astype(float)
        df = df.with_columns(
            pl.Series(f"rank_{col}", _would_be_rank(v, pool), dtype=pl.Float64).cast(pl.Int64, strict=False)
        )
    df = df.with_columns(pl.col("flag_fisher").alias("flagged"))
    return df.with_columns(
        pl.lit(thr_f).alias("fisher_cutoff"), pl.lit(thr_r).alias("rank_mean_cutoff")
    )


def rank_agreement(df: pl.DataFrame) -> dict:
    """Spearman correlations between the three rankings, and top-k overlap of the flags."""
    cols = ["fisher_score", "rank_mean", "iso_score"]
    sub = df.select(cols).drop_nulls()
    corr = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            corr[f"{a}~{b}"] = (
                float(stats.spearmanr(sub[a].to_numpy(), sub[b].to_numpy())[0])
                if sub.height > 2
                else float("nan")
            )
    sets = {
        n: set(df.filter(pl.col(f"flag_{n}"))["condition_id"].to_list())
        for n in ("fisher", "rank_mean", "iso")
    }
    return {
        "n": sub.height,
        "spearman": corr,
        "flag_overlap": {
            "fisher&rank_mean": len(sets["fisher"] & sets["rank_mean"]),
            "fisher&iso": len(sets["fisher"] & sets["iso"]),
            "rank_mean&iso": len(sets["rank_mean"] & sets["iso"]),
        },
        "flag_counts": {k: len(v) for k, v in sets.items()},
    }
