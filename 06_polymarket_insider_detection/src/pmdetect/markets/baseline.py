"""Peer-market baseline: pooled conditional quantiles of the winner price (phase 2).

Model (one linear quantile regression per tau in 10/50/90, ``statsmodels.QuantReg``)::

    q ~ B(log tau_h)            cubic B-spline basis, 4 knots (6 columns, acts as the intercept)
        + q0 + q0 * log tau_h   starting level and how its influence fades toward the news
        + is_deadline + is_deadline * log tau_h + is_deadline * q0   # deadline markets separate
        + is_unscheduled        # scheduled_reveal is the reference level

Time split (default scheme, ``fit_predict_timesplit``): the baseline is fitted only on
independent-news markets that resolved EARLIER and scores only LATER markets, so no market, and no
information from after a market's own window, sets its baseline. Scheme (fixed in advance, not
tuned on any labelled case):

* every market is placed in a calendar-month window of its news time (``news_ts_best``; for
  independent markets this is the independent time, so the window never depends on the price path);
* expanding window: for window W starting at T_k the training set is ALL training markets whose news
  time is strictly before T_k; the markets whose news time falls in [T_k, T_k + 1 month) are scored.
  Peers inside the same window are never in the training set. Every independent market is scored
  exactly once, by a model that saw only the past;
* a window needs at least ``MIN_TRAIN_MARKETS`` (100) training markets and ``TS_MIN_TRAIN_ROWS``
  (500) thinned training rows; otherwise its markets are UNSCORABLE (no baseline rows, D null with
  reason ``no_prior_training_window``), never scored in-sample or with a model that saw them;
* ``train_ids`` (default: the scored ids) lets out-of-universe markets such as the extra Maduro market
  be scored by the same expanding models without ever entering a fit.

Cross-fitting (``fit_predict_cross``, kept for comparison via ``--baseline crossfit``): markets are
assigned to ``n_splits`` folds (seeded shuffle); the quantile models for fold k are fitted on the
markets of the other folds only, then predict fold k. A market never sets its own baseline, but it
is scored by a model fitted partly on LATER markets (look-ahead between peers), which the time split
removes. Rows are thinned (every ``thin``-th hour) when fitting because neighbouring hours of one
market are almost identical; prediction uses every hour.

Predicted quantiles are sorted per row (no crossing) and clipped to [0, 1].

Phase 2b: ``independent_ids`` restricts both fitting and prediction to markets whose news time is
independent of the price path (see ``newstime_indep``); price-defined news times make the level
baseline circular, so those markets never enter the fit and get no baseline rows. With only
scheduled-reveal markets in the fit the deadline/unscheduled columns are constant zero; such
columns are dropped per fit (``_fit_fold`` records the kept-column mask), so the spec shrinks
automatically instead of fitting degenerate terms.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from sklearn.preprocessing import SplineTransformer

QUANTILES = (0.1, 0.5, 0.9)
MIN_TRAIN_ROWS = 50  # cross-fit fallback
MIN_TRAIN_MARKETS = 100  # time split: fixed in advance, not tuned
TS_MIN_TRAIN_ROWS = 500
TS_WINDOW = "1mo"
LOG_TAU_RANGE = (0.0, float(np.log(168.0)))
_SPLINE = SplineTransformer(
    n_knots=4, degree=3, knots=np.linspace(*LOG_TAU_RANGE, 4).reshape(-1, 1), include_bias=True
)
_SPLINE.fit(np.linspace(*LOG_TAU_RANGE, 20).reshape(-1, 1))


def design(df: pl.DataFrame) -> np.ndarray:
    """Design matrix from columns tau_h, q0, info_type."""
    lt = np.log(np.clip(df["tau_h"].to_numpy().astype(float), 1.0, None))
    q0 = df["q0"].to_numpy().astype(float)
    it = df["info_type"].to_numpy()
    dl = (it == "deadline").astype(float)
    un = (it == "unscheduled_event").astype(float)
    basis = _SPLINE.transform(np.clip(lt, *LOG_TAU_RANGE).reshape(-1, 1))
    extra = np.column_stack([q0, q0 * lt, dl, dl * lt, dl * q0, un])
    return np.column_stack([basis, extra])


def _fit_fold(train: pl.DataFrame, quantiles=QUANTILES, thin: int = 3):
    """Fit one QuantReg per quantile on ``train`` (columns tau_h, q, q0, info_type)."""
    from statsmodels.regression.quantile_regression import QuantReg

    tr = train.filter((pl.col("tau_h").cast(pl.Int64) % thin) == 0) if thin > 1 else train
    X = design(tr)
    keep = np.abs(X).sum(axis=0) > 0  # drop columns that are identically zero in the training set
    X = X[:, keep]
    y = tr["q"].to_numpy().astype(float)
    models = []
    for qt in quantiles:
        with warnings.catch_warnings():  # covariance warnings on degenerate synthetic data
            warnings.simplefilter("ignore")
            res = QuantReg(y, X).fit(q=qt, max_iter=2000, p_tol=1e-6)
        models.append(res.params)
    return models, keep


def _predict(fitted, df: pl.DataFrame) -> np.ndarray:
    models, keep = fitted
    X = design(df)[:, keep]
    pred = np.column_stack([X @ b for b in models])
    return np.clip(np.sort(pred, axis=1), 0.0, 1.0)


def assign_folds(condition_ids: list[str], n_splits: int, seed: int) -> dict[str, int]:
    ids = sorted(set(condition_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    return {ids[i]: int(k % n_splits) for k, i in enumerate(order)}


def fit_predict_cross(
    series: pl.DataFrame,
    q0: pl.DataFrame,
    info_type: pl.DataFrame,
    n_splits: int = 5,
    seed: int = 0,
    thin: int = 3,
    independent_ids: set[str] | list[str] | None = None,
) -> pl.DataFrame:
    """Cross-fitted peer quantiles: condition_id, t, tau_h, m10, m50, m90, fold.

    ``independent_ids``: if given, only these markets are used (fit and prediction); every other
    market is ignored, so a market with a price-based news time cannot influence any baseline.

    ``info_type`` is a frame (condition_id, info_type), e.g. ``markets.select(...)``. Markets
    lacking q0 or info_type get no baseline rows (score_markets reports them as NaN with a reason).
    """
    if independent_ids is not None:
        series = series.filter(pl.col("condition_id").is_in(list(independent_ids)))
    df = (
        series.join(q0.select("condition_id", "q0"), on="condition_id", how="inner")
        .join(info_type.select("condition_id", "info_type"), on="condition_id", how="inner")
        .filter(pl.col("q0").is_not_null() & pl.col("info_type").is_not_null())
    )
    out_schema = {
        "condition_id": pl.String,
        "t": pl.Datetime("us"),
        "tau_h": pl.Float64,
        "m10": pl.Float64,
        "m50": pl.Float64,
        "m90": pl.Float64,
        "fold": pl.Int64,
    }
    if df.height == 0:
        return pl.DataFrame(schema=out_schema)
    cids = df["condition_id"].unique().to_list()
    k = max(2, min(n_splits, len(cids)))
    folds = assign_folds(cids, k, seed)
    df = df.with_columns(
        pl.col("condition_id").replace_strict(folds, return_dtype=pl.Int64).alias("fold")
    )
    parts = []
    for f in range(k):
        train = df.filter(pl.col("fold") != f)
        test = df.filter(pl.col("fold") == f)
        if test.height == 0:
            continue
        if train.height < MIN_TRAIN_ROWS:
            continue
        pred = _predict(_fit_fold(train, thin=thin), test)
        parts.append(
            test.select("condition_id", "t", "tau_h", "fold").with_columns(
                pl.Series("m10", pred[:, 0]),
                pl.Series("m50", pred[:, 1]),
                pl.Series("m90", pred[:, 2]),
            )
        )
    if not parts:
        return pl.DataFrame(schema=out_schema)
    return pl.concat(parts).select(list(out_schema))


TS_PLAN_SCHEMA = {
    "condition_id": pl.String,
    "window": pl.Datetime("us"),
    "n_train_markets": pl.Int64,
    "n_train_rows": pl.Int64,
    "scorable": pl.Boolean,
}


def fit_predict_timesplit(
    series: pl.DataFrame,
    q0: pl.DataFrame,
    info_type: pl.DataFrame,
    news_ts: pl.DataFrame,
    score_ids: set[str] | list[str],
    train_ids: set[str] | list[str] | None = None,
    window: str = TS_WINDOW,
    min_train_markets: int = MIN_TRAIN_MARKETS,
    min_train_rows: int = TS_MIN_TRAIN_ROWS,
    thin: int = 3,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Expanding-window time-split peer quantiles (see module docstring).

    ``news_ts`` is a frame (condition_id, news_ts) holding each market's (independent) news time.
    Returns ``(baseline, plan)``: ``baseline`` has the same columns as ``fit_predict_cross`` (``fold``
    is the window index); ``plan`` has one row per scored market with its window start, the number of
    training markets / thinned rows available and ``scorable``. Markets in unscorable windows have no
    baseline rows.
    """
    score_ids = set(score_ids)
    train_ids = score_ids if train_ids is None else set(train_ids)
    out_schema = {
        "condition_id": pl.String, "t": pl.Datetime("us"), "tau_h": pl.Float64,
        "m10": pl.Float64, "m50": pl.Float64, "m90": pl.Float64, "fold": pl.Int64,
    }  # fmt: skip
    df = (
        series.filter(pl.col("condition_id").is_in(list(score_ids | train_ids)))
        .join(q0.select("condition_id", "q0"), on="condition_id", how="inner")
        .join(info_type.select("condition_id", "info_type"), on="condition_id", how="inner")
        .filter(pl.col("q0").is_not_null() & pl.col("info_type").is_not_null())
        .join(
            news_ts.select("condition_id", pl.col("news_ts").alias("_news")).drop_nulls(),
            on="condition_id",
            how="inner",
        )
        .with_columns(pl.col("_news").dt.truncate(window).alias("_win"))
    )
    empty_plan = pl.DataFrame(schema=TS_PLAN_SCHEMA)
    if df.height == 0:
        return pl.DataFrame(schema=out_schema), empty_plan
    pool = df.filter(pl.col("condition_id").is_in(list(train_ids)))
    sc = df.filter(pl.col("condition_id").is_in(list(score_ids)))
    wins = sorted(sc["_win"].unique().to_list())
    parts, plans = [], []
    for k, w in enumerate(wins):
        train = pool.filter(pl.col("_news") < w)
        n_m = train["condition_id"].n_unique() if train.height else 0
        n_rows = (
            train.filter((pl.col("tau_h").cast(pl.Int64) % thin) == 0).height if thin > 1 else train.height
        )
        test = sc.filter(pl.col("_win") == w)
        ok = n_m >= min_train_markets and n_rows >= min_train_rows
        plans.append(
            test.group_by("condition_id").agg(pl.len().alias("_n"))
            .select(
                "condition_id",
                pl.lit(w, dtype=pl.Datetime("us")).alias("window"),
                pl.lit(n_m, dtype=pl.Int64).alias("n_train_markets"),
                pl.lit(n_rows, dtype=pl.Int64).alias("n_train_rows"),
                pl.lit(ok).alias("scorable"),
            )
        )
        if not ok:
            continue
        pred = _predict(_fit_fold(train, thin=thin), test)
        parts.append(
            test.select("condition_id", "t", "tau_h", pl.lit(k, dtype=pl.Int64).alias("fold"))
            .with_columns(
                pl.Series("m10", pred[:, 0]),
                pl.Series("m50", pred[:, 1]),
                pl.Series("m90", pred[:, 2]),
            )
        )
    plan = pl.concat(plans).select(list(TS_PLAN_SCHEMA)) if plans else empty_plan
    base = pl.concat(parts).select(list(out_schema)) if parts else pl.DataFrame(schema=out_schema)
    return base, plan
