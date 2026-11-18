"""Calibration and per-wallet tests (phase 3A).

* ``fit_isotonic``: isotonic regression of the win indicator on entry price, **cross-fitted** (events,
  hence all outcome tokens of a market and of its siblings, are assigned to one of ``n_splits`` folds
  at random; a position's recalibrated probability p~ comes from a model that never saw its event) and
  fitted **by info_type**, falling back to the pooled curve when the type has fewer than
  ``MIN_TYPE_N`` resolved positions (in the training folds) or a single class. p~ is clipped to
  [P_MIN, P_MAX] so a flat-zero isotonic step cannot create infinite evidence.
* Under H0 (a wallet's entries are as informative as everyone's at that price and info type) the wins
  of a wallet are independent Bernoulli(p~_i). ``poisson_binomial_sf`` gives the exact upper tail
  P(W >= wins) by dynamic programming; ``z`` is the normal-approximation statistic; ``excess_pnl`` is
  sum size_i * (won_i - p~_i) = realised minus expected profit in USD (each share bought at price e
  pays 1 if it wins: profit = size*(won - e), expected = size*(p~ - e)), with a parametric-bootstrap
  p-value (``N_BOOT`` Bernoulli(p~) draws, +1 smoothed), computed only for wallets with >= 5 positions.
* Only resolved event-positions (won not null) enter, and (phase 3b) only ``calib_eligible``
  positions (independent news cutoff; falls back to ``valid_cutoff`` if that column is absent, and to
  all positions if both are). Jump-start and no-cutoff positions are excluded from the fit and the
  repeat test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

PRIVATE = Path(__file__).resolve().parents[3] / "data" / "private"
MIN_TYPE_N = 500
P_MIN, P_MAX = 0.01, 0.99
N_BOOT = 5000
MIN_BOOT_POS = 5


def poisson_binomial_sf(k: int, p) -> float:
    """Exact P(W >= k) for W = sum of independent Bernoulli(p_i), by dynamic programming."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    pmf = np.zeros(n + 1)
    pmf[0] = 1.0
    for i, pi in enumerate(p):
        pmf[1 : i + 2] = pmf[1 : i + 2] * (1 - pi) + pmf[0 : i + 1] * pi
        pmf[0] *= 1 - pi
    return float(min(1.0, max(0.0, pmf[k:].sum())))


def _valid_resolved(positions: pl.DataFrame) -> pl.DataFrame:
    d = positions.filter(pl.col("won").is_not_null())
    col = "calib_eligible" if "calib_eligible" in d.columns else "valid_cutoff"
    if col in d.columns:
        d = d.filter(pl.col(col).fill_null(False))
    return d


def decile_table(d: pl.DataFrame, n_bins: int = 10, p_col: str | None = "p_tilde") -> pl.DataFrame:
    """Aggregate calibration by entry-price quantile bin: n, mean entry, observed win rate, mean p~.
    ``d`` needs entry_price, won (and ``p_col`` if not None). No wallet identifiers in the output."""
    d = d.filter(pl.col("won").is_not_null())
    x = d["entry_price"].to_numpy()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    aggs = [pl.len().alias("n"), pl.col("entry_price").min().alias("price_lo"),
            pl.col("entry_price").max().alias("price_hi"), pl.col("entry_price").mean().alias("mean_entry"),
            pl.col("won").cast(pl.Float64).mean().alias("win_rate")]  # fmt: skip
    aggs.append(pl.col(p_col).mean().alias("mean_p_tilde") if p_col else pl.lit(None, dtype=pl.Float64).alias("mean_p_tilde"))  # fmt: skip
    return d.with_columns(pl.Series("decile", b + 1)).group_by("decile").agg(aggs).sort("decile")


def _iso(x: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)


def _predict(df_train: pl.DataFrame, df_test: pl.DataFrame) -> np.ndarray:
    """p~ for df_test rows from models fitted on df_train (per info_type, pooled fallback)."""
    pooled = _iso(df_train["entry_price"].to_numpy(), df_train["y"].to_numpy().astype(float))
    out = pooled.predict(df_test["entry_price"].to_numpy())
    it_te = df_test["info_type"].to_numpy()
    for it in np.unique(it_te[it_te != None]):  # noqa: E711
        tr = df_train.filter(pl.col("info_type") == it)
        if tr.height < MIN_TYPE_N or tr["y"].n_unique() < 2:
            continue
        m = _iso(tr["entry_price"].to_numpy(), tr["y"].to_numpy().astype(float))
        sel = it_te == it
        out[sel] = m.predict(df_test["entry_price"].to_numpy()[sel])
    return np.clip(out, P_MIN, P_MAX)


def fit_isotonic(positions: pl.DataFrame, n_splits: int = 5, seed: int = 0):
    """Cross-fitted recalibration. Returns (positions_resolved + ``p_tilde``, curve summary table).

    The curve table (aggregate, no wallets) is the full-sample fit evaluated on a price grid, per
    info_type used (``pooled`` or the type), with observed win rate by price bin for reference.
    """
    d = _valid_resolved(positions).with_columns(pl.col("won").cast(pl.Int8).alias("y"))
    d = d.with_row_index("_i")
    events = d["event_id"].unique().sort()
    rng = np.random.default_rng(seed)
    fmap = pl.DataFrame(
        {"event_id": events, "_fold": rng.permutation(len(events)) % n_splits}
    ).with_columns(pl.col("_fold").cast(pl.Int64))
    fold = d.select("event_id").join(fmap, on="event_id", how="left")["_fold"].to_numpy()
    p = np.zeros(d.height)
    for f in range(n_splits):
        te = fold == f
        if not te.any():
            continue
        tr_df, te_df = d.filter(pl.Series(~te)), d.filter(pl.Series(te))
        p[te] = _predict(tr_df, te_df)
    d = d.with_columns(pl.Series("p_tilde", p)).drop("_i")
    # curve summary
    grid = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    rows = []
    pooled = _iso(d["entry_price"].to_numpy(), d["y"].to_numpy().astype(float))
    for g, pv in zip(grid, np.clip(pooled.predict(grid), P_MIN, P_MAX), strict=True):
        rows.append({"curve": "pooled", "entry_price": float(g), "p_tilde": float(pv)})
    for it in sorted(x for x in d["info_type"].unique().to_list() if x is not None):
        sub = d.filter(pl.col("info_type") == it)
        if sub.height < MIN_TYPE_N or sub["y"].n_unique() < 2:
            continue
        m = _iso(sub["entry_price"].to_numpy(), sub["y"].to_numpy().astype(float))
        for g, pv in zip(grid, np.clip(m.predict(grid), P_MIN, P_MAX), strict=True):
            rows.append({"curve": it, "entry_price": float(g), "p_tilde": float(pv)})
    curve = pl.DataFrame(rows)
    bins = np.array([0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0001])
    b = np.digitize(d["entry_price"].to_numpy(), bins) - 1
    obs = (
        d.with_columns(pl.Series("bin", b))
        .group_by("bin")
        .agg(pl.len().alias("n"), pl.col("entry_price").mean().alias("mean_entry"),
             pl.col("y").mean().alias("win_rate"), pl.col("p_tilde").mean().alias("mean_p_tilde"))
        .sort("bin")
    )  # fmt: skip
    return d, curve, obs


def _boot_p(p: np.ndarray, size: np.ndarray, won: np.ndarray, rng, n_boot: int = N_BOOT) -> float:
    obs = float((size * (won - p)).sum())
    draws = rng.random((n_boot, p.size)) < p
    null = (draws - p) @ size
    return float((1 + (null >= obs - 1e-9).sum()) / (n_boot + 1))


_PB_MAX_CELLS = 20_000_000


def poisson_binomial_sf_batch(p: np.ndarray, offsets: np.ndarray, k: np.ndarray) -> np.ndarray:
    """P(W >= k_w) for many wallets at once (same exact DP as :func:`poisson_binomial_sf`).

    ``p`` is the concatenation of every wallet's probabilities, ``offsets`` (length W + 1) the run
    boundaries and ``k`` the win counts. Wallets are bucketed by size (powers of two) and padded with
    p = 0, which leaves the pmf unchanged, so a bucket is one vectorised DP over (wallets x n+1)."""
    n = np.diff(offsets)
    out = np.ones(n.size)
    live = np.where(k > 0)[0]
    bucket = np.where(n[live] > 0, np.ceil(np.log2(np.maximum(n[live], 1))).astype(int), 0)
    for b in np.unique(bucket):
        idx = live[bucket == b]
        width = int(n[idx].max())
        step = max(1, _PB_MAX_CELLS // (width + 1))
        for a in range(0, idx.size, step):
            sub = idx[a : a + step]
            mat = np.zeros((sub.size, width))
            lens = n[sub]
            rr = np.repeat(np.arange(sub.size), lens)
            cc = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
            mat[rr, cc] = p[np.repeat(offsets[sub], lens) + cc]
            pmf = np.zeros((sub.size, width + 1))
            pmf[:, 0] = 1.0
            for i in range(width):
                pi = mat[:, i : i + 1]
                pmf[:, 1 : i + 2] = pmf[:, 1 : i + 2] * (1 - pi) + pmf[:, 0 : i + 1] * pi
                pmf[:, 0] *= 1 - pi[:, 0]
            tail = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
            out[sub] = tail[np.arange(sub.size), k[sub]]
    return np.clip(out, 0.0, 1.0)


def wallet_tests(positions: pl.DataFrame, write: bool = True, seed: int = 0, calibrated: pl.DataFrame | None = None) -> pl.DataFrame:  # fmt: skip
    """One row per wallet: n_pos, wins, expected, z, p_pb, excess_pnl, p_boot, n_usd, ...

    ``positions`` is the output of build_positions; it is recalibrated here (fit_isotonic) unless
    ``calibrated`` (fit_isotonic's first output) is given or ``positions`` already has ``p_tilde``.
    Scale: per-wallet sums are polars aggregations, the exact tail is a bucketed vectorised DP and only
    wallets with >= MIN_BOOT_POS positions run the (per-wallet, seeded, sorted-wallet-order) bootstrap.
    """
    if calibrated is None:
        if "p_tilde" in positions.columns:
            calibrated = _valid_resolved(positions).with_columns(
                pl.col("won").cast(pl.Int8).alias("y")
            )
        else:
            calibrated, _, _ = fit_isotonic(positions, seed=seed)
    d = calibrated.select("wallet", "p_tilde", "y", "size", "entry_price").sort("wallet")
    p_ = pl.col("p_tilde")
    y_ = pl.col("y").cast(pl.Float64)
    sz, e_ = pl.col("size"), pl.col("entry_price")
    agg = d.group_by("wallet", maintain_order=True).agg(
        pl.len().cast(pl.Int64).alias("n_pos"),
        pl.col("y").cast(pl.Int64).sum().alias("wins"),
        p_.sum().alias("expected"),
        (p_ * (1 - p_)).sum().alias("_var"),
        (sz * (y_ - p_)).sum().alias("excess_pnl"),
        (sz * (y_ - e_)).sum().alias("realised_pnl"),
        (sz * (p_ - e_)).sum().alias("expected_pnl"),
        (sz * e_).sum().alias("n_usd"),
        e_.mean().alias("mean_entry"),
        ((e_ < 0.2) & (pl.col("y") == 1)).sum().cast(pl.Int64).alias("n_longshot_wins"),
    )
    n = agg["n_pos"].to_numpy()
    offsets = np.concatenate([[0], np.cumsum(n)])
    p_arr = d["p_tilde"].to_numpy()
    p_pb = poisson_binomial_sf_batch(p_arr, offsets, agg["wins"].to_numpy())
    p_boot: list[float | None] = [None] * agg.height
    y_arr = d["y"].to_numpy().astype(float)
    sz_arr = d["size"].to_numpy()
    rng = np.random.default_rng(seed)
    for w in np.where(n >= MIN_BOOT_POS)[0]:  # sorted-wallet order = the old per-wallet loop order
        a, b = offsets[w], offsets[w + 1]
        p_boot[w] = _boot_p(p_arr[a:b], sz_arr[a:b], y_arr[a:b], rng)
    out = agg.with_columns(
        pl.when(pl.col("_var") > 0)
        .then((pl.col("wins") - pl.col("expected")) / pl.col("_var").sqrt())
        .otherwise(None)
        .alias("z"),
        pl.Series("p_pb", p_pb),
        pl.Series("p_boot", p_boot, dtype=pl.Float64),
        pl.col("n_pos").alias("n_events"),
    ).select(
        "wallet", "n_pos", "wins", "expected", "z", "p_pb", "excess_pnl", "realised_pnl",
        "expected_pnl", "p_boot", "n_usd", "mean_entry", "n_longshot_wins", "n_events",
    )  # fmt: skip
    if write:
        PRIVATE.mkdir(parents=True, exist_ok=True)
        pth = PRIVATE / "wallet_tests.parquet"
        out.write_parquet(pth.with_suffix(".tmp"))
        pth.with_suffix(".tmp").replace(pth)
    return out
