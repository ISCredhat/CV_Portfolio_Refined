"""News-time estimation (plan 1.3).

Order of preference per market:
1. ``data/manual/news_time_overrides.csv`` (scheduled reveals, validation cases);
2. ``price_cross_0.9``: first point of the eventual winner's price series that is above
   ``THRESHOLD`` and stays above until the end of the series (resolution);
3. ``trade_cross_0.9``: the same rule on the winner token's trade prices, used when there is no
   price series or it is coarser than ``COARSE_SECONDS`` (old markets only keep 3 h buckets).

Markets without a clean outcome or without a stable crossing keep news_ts NULL with the method
``no_outcome`` / ``no_cross`` / ``no_data`` so the reason is recorded.

Two implementations of the same rule:

* ``estimate_news_times`` + ``estimate_for_market``: the original per-market code on DuckDB (needs
  a ``fetch_log`` 'trades' entry or an override, uses ALL fill rows of the winner token as the
  trade fallback); kept as the reference for the parity test;
* ``bulk_news_times`` / ``estimate_news_times_bulk``: vectorised polars over the Parquet snapshots
  (winner-token price series + the taker tape mapped to winner prices, one row per fill) for every
  market at once, no fetch_log dependency, one batched UPDATE into DuckDB. Differences to the
  reference can only come from the trade fallback (taker rows vs all rows; see ``parity``).
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pmdetect.collect.data_api import connect
from pmdetect.store.db import PROJECT_ROOT

THRESHOLD = 0.9
COARSE_SECONDS = 3600
OVERRIDES_PATH = PROJECT_ROOT / "data" / "manual" / "news_time_overrides.csv"


def parse_ts(text: str) -> datetime:
    """Parse an ISO-8601 timestamp into a naive UTC datetime."""
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def load_overrides(path: Path = OVERRIDES_PATH) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return (by_condition_id, by_slug) override rows."""
    by_cond: dict[str, dict] = {}
    by_slug: dict[str, dict] = {}
    if not path.exists():
        return by_cond, by_slug
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("news_ts") or "").strip():
                continue
            if (row.get("condition_id") or "").strip():
                by_cond[row["condition_id"].strip().lower()] = row
            if (row.get("slug") or "").strip():
                by_slug[row["slug"].strip()] = row
    return by_cond, by_slug


def stable_cross(points: list[tuple[datetime, float]], threshold: float = THRESHOLD):
    """Timestamp of the first point after which the price stays >= threshold to the end.

    points must be sorted by time. Returns (ts, status) with status 'ok', 'always_above'
    (every point is above: news precedes the data) or 'no_cross' (series ends below threshold).
    """
    if not points:
        return None, "no_data"
    last_below = None
    for i, (_, p) in enumerate(points):
        if p < threshold:
            last_below = i
    if last_below is None:
        return points[0][0], "always_above"
    if last_below == len(points) - 1:
        return None, "no_cross"
    return points[last_below + 1][0], "ok"


def _median_spacing(points: list[tuple[datetime, float]]) -> float | None:
    if len(points) < 2:
        return None
    gaps = sorted((b[0] - a[0]).total_seconds() for a, b in zip(points, points[1:], strict=False))
    return gaps[len(gaps) // 2]


def estimate_for_market(
    price_points: list[tuple[datetime, float]], trade_points: list[tuple[datetime, float]]
) -> tuple[datetime | None, str]:
    """Apply the crossing rule, preferring the finer of price series / trade tape."""
    spacing = _median_spacing(price_points)
    use_prices = bool(price_points) and (spacing is None or spacing <= COARSE_SECONDS)
    if use_prices or (price_points and not trade_points):
        ts, status = stable_cross(price_points)
        method = "price_cross_0.9"
    else:
        ts, status = stable_cross(trade_points)
        method = "trade_cross_0.9"
    if status == "no_data":
        return None, "no_data"
    if status == "no_cross":
        return None, "no_cross"
    if status == "always_above":
        method += "_always_above"
    return ts, method


def estimate_news_times(overrides_path: Path = OVERRIDES_PATH) -> int:
    """Fill markets.news_ts / news_ts_method for markets with a loaded tape and override markets.

    Returns the number of markets that received a news_ts.
    """
    by_cond, by_slug = load_overrides(overrides_path)
    con = connect()
    try:
        rows = con.execute(
            """
            SELECT condition_id, slug, outcome FROM markets
            WHERE condition_id IN (SELECT UNNEST(?::VARCHAR[]))
               OR slug IN (SELECT UNNEST(?::VARCHAR[]))
               OR condition_id IN (
                   SELECT key FROM fetch_log WHERE endpoint = 'trades' AND completed)
            """,
            [list(by_cond), list(by_slug)],
        ).fetchall()
        n_set = 0
        for cid, slug, outcome in rows:
            ov = by_cond.get(cid) or by_slug.get(slug)
            if ov:
                ts, method = parse_ts(ov["news_ts"]), (ov.get("method") or "override").strip()
                if not method.startswith("override"):
                    method = f"override_{method}"
            elif outcome not in ("YES", "NO"):
                ts, method = None, "no_outcome"
            else:
                ts, method = _estimate_from_data(con, cid, outcome)
            con.execute(
                "UPDATE markets SET news_ts = ?, news_ts_method = ? WHERE condition_id = ?",
                [ts, method, cid],
            )
            n_set += ts is not None
        return n_set
    finally:
        con.close()


def _winner_token(con, cid: str, outcome: str) -> str | None:
    labels = con.execute(
        "SELECT token_id, outcome_label FROM tokens WHERE condition_id = ?", [cid]
    ).fetchall()
    for tid, label in labels:
        if (label or "").strip().upper() == outcome:
            return tid
    return None


def _estimate_from_data(con, cid: str, outcome: str) -> tuple[datetime | None, str]:
    tid = _winner_token(con, cid, outcome)
    if tid is None:
        return None, "no_data"
    prices = con.execute("SELECT ts, price FROM prices WHERE token_id = ? ORDER BY ts", [tid])
    price_points = [(t, float(p)) for t, p in prices.fetchall()]
    trades = con.execute(
        "SELECT ts, price FROM trades WHERE token_id = ? ORDER BY ts, trade_id", [tid]
    )
    trade_points = [(t, float(p)) for t, p in trades.fetchall()]
    return estimate_for_market(price_points, trade_points)


# --------------------------------------------------------------------------- bulk (polars)


def _override_frame(markets: pl.DataFrame, overrides: tuple[dict, dict] | None) -> pl.DataFrame:
    """condition_id, ov_ts, ov_method for markets matched by condition_id or slug."""
    by_cond, by_slug = overrides if overrides is not None else ({}, {})
    slugs = markets["slug"].to_list() if "slug" in markets.columns else [None] * markets.height
    rows = []
    for cid, slug in zip(markets["condition_id"].to_list(), slugs, strict=True):
        ov = by_cond.get((cid or "").lower()) or by_slug.get(slug or "")
        if ov:
            method = (ov.get("method") or "override").strip()
            if not method.startswith("override"):
                method = f"override_{method}"
            rows.append((cid, parse_ts(ov["news_ts"]), method))
    return pl.DataFrame(
        rows,
        schema={"condition_id": pl.String, "ov_ts": pl.Datetime("us"), "ov_method": pl.String},
        orient="row",
    )


def _cross_table(pts: pl.DataFrame, tag: str, threshold: float) -> pl.DataFrame:
    """Vectorised ``stable_cross`` per market.

    ``pts``: condition_id, ts, price (+ optional trade_id for a deterministic tie order); returns
    condition_id, {tag}_n, {tag}_ts, {tag}_status with status ok / always_above / no_cross.
    """
    sort_cols = ["condition_id", "ts"] + (["trade_id"] if "trade_id" in pts.columns else [])
    p = pts.sort(sort_cols).with_columns(pl.int_range(pl.len()).over("condition_id").alias("_i"))
    agg = p.group_by("condition_id").agg(
        pl.len().alias(f"{tag}_n"),
        pl.col("_i").filter(pl.col("price") < threshold).max().alias("_lb"),
    )
    agg = agg.with_columns(
        pl.when(pl.col("_lb").is_null())
        .then(pl.lit("always_above"))
        .when(pl.col("_lb") == pl.col(f"{tag}_n") - 1)
        .then(pl.lit("no_cross"))
        .otherwise(pl.lit("ok"))
        .alias(f"{tag}_status"),
        pl.when(pl.col("_lb").is_null()).then(0).otherwise(pl.col("_lb") + 1).alias("_target"),
    )
    hit = p.join(agg.select("condition_id", "_target"), on="condition_id").filter(
        pl.col("_i") == pl.col("_target")
    )
    out = agg.join(
        hit.select("condition_id", pl.col("ts").alias(f"{tag}_ts")), on="condition_id", how="left"
    )
    return out.select("condition_id", f"{tag}_n", f"{tag}_ts", f"{tag}_status")


def _median_spacing_us(prices: pl.DataFrame) -> pl.DataFrame:
    """condition_id, spacing_us: ``gaps[len // 2]`` of the sorted consecutive gaps (as the
    reference ``_median_spacing``); null for fewer than 2 points."""
    g = (
        prices.sort("condition_id", "ts")
        .with_columns(pl.col("ts").diff().over("condition_id").dt.total_microseconds().alias("gap"))
        .filter(pl.col("gap").is_not_null())
        .sort("condition_id", "gap")
        .with_columns(
            pl.int_range(pl.len()).over("condition_id").alias("_i"),
            pl.len().over("condition_id").alias("_n"),
        )
        .filter(pl.col("_i") == pl.col("_n") // 2)
    )
    return g.select("condition_id", pl.col("gap").alias("spacing_us"))


def bulk_news_times(
    markets: pl.DataFrame,
    tokens: pl.DataFrame,
    prices: pl.DataFrame,
    trades: pl.DataFrame,
    overrides: tuple[dict, dict] | None = None,
    threshold: float = THRESHOLD,
    coarse_seconds: int = COARSE_SECONDS,
) -> pl.DataFrame:
    """Vectorised news-time estimate for every market: condition_id, news_ts, news_ts_method.

    ``markets``: condition_id, outcome, slug (optional). ``tokens``: token_id, condition_id,
    outcome_label. ``prices``: token_id, ts, price (any tokens; only the winner token is used).
    ``trades``: condition_id, token_id, ts, price (+ trade_id for tie order), ideally the taker
    tape (one row per fill); each fill becomes a winner-token price (1 - price when the fill was
    on the losing token). Same rule and status strings as
    ``estimate_for_market``: override first; prices when their median spacing is at most
    ``coarse_seconds`` (or there are no trades), else trades; ``no_outcome`` / ``no_data`` /
    ``no_cross``; ``*_always_above`` suffix when every point is above the threshold.
    """
    base = markets.select("condition_id", "outcome")
    win = (
        base.filter(pl.col("outcome").is_in(["YES", "NO"]))
        .join(
            tokens.select(
                "token_id",
                "condition_id",
                pl.col("outcome_label").str.strip_chars().str.to_uppercase().alias("_lab"),
            ),
            left_on=["condition_id", "outcome"],
            right_on=["condition_id", "_lab"],
            how="inner",
        )
        .unique(subset="condition_id", keep="first", maintain_order=True)
        .select("condition_id", "token_id")
    )
    p = prices.join(win, on="token_id", how="inner").select("condition_id", "ts", "price")
    # Trade points as WINNER-token prices: a fill on the winner token counts at its price, a fill
    # on the losing token at 1 - price. The all-rows tape lists every fill twice (taker row on one
    # token, maker row on the complement token at the complement price), so the winner token's
    # rows of the all-rows tape are exactly these values; taker rows alone, filtered to the winner
    # token, would miss every fill whose taker traded the loser token.
    tcols = ["condition_id", "ts", "price"] + (["trade_id"] if "trade_id" in trades.columns else [])
    t = (
        trades.select([c for c in ("trade_id", "condition_id", "token_id", "ts", "price") if c in trades.columns])
        .join(win.rename({"token_id": "_win_tok"}), on="condition_id", how="inner")
        .with_columns(
            pl.when(pl.col("token_id") == pl.col("_win_tok"))
            .then(pl.col("price"))
            .otherwise((1.0 - pl.col("price")).round(6))
            .alias("price")
        )
        .select(tcols)
    )
    pc = _cross_table(p, "price", threshold)
    tc = _cross_table(t, "trade", threshold)
    sp = _median_spacing_us(p)
    df = (
        base.join(pc, on="condition_id", how="left")
        .join(tc, on="condition_id", how="left")
        .join(sp, on="condition_id", how="left")
        .join(_override_frame(markets, overrides), on="condition_id", how="left")
        .with_columns(
            pl.col("price_n").fill_null(0),
            pl.col("trade_n").fill_null(0),
            pl.col("outcome").is_in(["YES", "NO"]).fill_null(False).alias("_clean"),
        )
    )
    has_p = pl.col("price_n") > 0
    has_t = pl.col("trade_n") > 0
    fine = pl.col("spacing_us").is_null() | (pl.col("spacing_us") <= coarse_seconds * 1_000_000)
    use_prices = (has_p & fine) | (has_p & ~has_t)
    df = df.with_columns(
        pl.when(use_prices).then(pl.col("price_status")).otherwise(pl.col("trade_status")).alias("_st"),
        pl.when(use_prices).then(pl.col("price_ts")).otherwise(pl.col("trade_ts")).alias("_ts"),
        pl.when(use_prices)
        .then(pl.lit("price_cross_0.9"))
        .otherwise(pl.lit("trade_cross_0.9"))
        .alias("_meth"),
    )
    out = df.with_columns(
        pl.when(pl.col("ov_ts").is_not_null())
        .then(pl.col("ov_ts"))
        .when(~pl.col("_clean") | pl.col("_st").is_null() | (pl.col("_st") == "no_cross"))
        .then(pl.lit(None, dtype=pl.Datetime("us")))
        .otherwise(pl.col("_ts"))
        .alias("news_ts"),
        pl.when(pl.col("ov_ts").is_not_null())
        .then(pl.col("ov_method"))
        .when(~pl.col("_clean"))
        .then(pl.lit("no_outcome"))
        .when(pl.col("_st").is_null())
        .then(pl.lit("no_data"))
        .when(pl.col("_st") == "no_cross")
        .then(pl.lit("no_cross"))
        .when(pl.col("_st") == "always_above")
        .then(pl.col("_meth") + "_always_above")
        .otherwise(pl.col("_meth"))
        .alias("news_ts_method"),
    )
    return out.select("condition_id", "news_ts", "news_ts_method")


def load_bulk_inputs(snapshot_dir: Path | None = None) -> dict[str, pl.DataFrame]:
    """Parquet inputs for ``bulk_news_times``: markets in the universe or with a taker tape."""
    snap = snapshot_dir or (PROJECT_ROOT / "data" / "snapshots")
    markets = pl.read_parquet(
        snap / "markets.parquet", columns=["condition_id", "slug", "outcome", "in_universe"]
    )
    tokens = pl.read_parquet(snap / "tokens.parquet")
    prices = pl.read_parquet(snap / "prices.parquet")
    trades = pl.read_parquet(
        snap / "trades_taker.parquet",
        columns=["trade_id", "condition_id", "token_id", "ts", "price"],
    )
    tape = trades["condition_id"].unique()
    markets = markets.filter(pl.col("in_universe") | pl.col("condition_id").is_in(tape.implode()))
    return {"markets": markets, "tokens": tokens, "prices": prices, "trades": trades}


def estimate_news_times_bulk(
    overrides_path: Path = OVERRIDES_PATH, snapshot_dir: Path | None = None, write: bool = True
) -> pl.DataFrame:
    """Bulk news times for every universe market (+ markets with a taker tape or an override).

    Writes ``markets.news_ts`` / ``news_ts_method`` in ONE batched UPDATE through a registered
    Arrow table (the DuckDB write lock is held only for that statement). ``write=False`` only
    returns the frame. Needs no ``fetch_log`` entry.
    """
    inp = load_bulk_inputs(snapshot_dir)
    res = bulk_news_times(
        inp["markets"], inp["tokens"], inp["prices"], inp["trades"], load_overrides(overrides_path)
    )
    if write:
        con = connect()
        try:
            con.execute("CREATE OR REPLACE TEMP TABLE _nt AS SELECT * FROM res")
            con.execute(
                "UPDATE markets SET news_ts = _nt.news_ts, news_ts_method = _nt.news_ts_method "
                "FROM _nt WHERE markets.condition_id = _nt.condition_id"
            )
            con.execute("DROP TABLE _nt")
        finally:
            con.close()
    return res


def _reference_per_market(mk, tokens, prices, all_tr, ov) -> pl.DataFrame:
    """The original ``estimate_for_market`` applied market by market on the same inputs."""
    by_c, by_s = ov
    py = []
    slugs = mk["slug"].to_list() if "slug" in mk.columns else [None] * mk.height
    for (cid, outcome), slug in zip(mk.select("condition_id", "outcome").iter_rows(), slugs, strict=True):
        o = by_c.get(cid) or by_s.get(slug or "")
        if o:
            m = (o.get("method") or "override").strip()
            m = m if m.startswith("override") else f"override_{m}"
            py.append((cid, parse_ts(o["news_ts"]), m))
        elif outcome not in ("YES", "NO"):
            py.append((cid, None, "no_outcome"))
        else:
            wt = tokens.filter(
                (pl.col("condition_id") == cid)
                & (pl.col("outcome_label").str.strip_chars().str.to_uppercase() == outcome)
            )["token_id"]
            if wt.len() == 0:
                py.append((cid, None, "no_data"))
                continue
            pp = prices.filter(pl.col("token_id") == wt[0]).sort("ts")
            tt = all_tr.filter(pl.col("token_id") == wt[0]).sort("ts", "trade_id")
            ts, meth = estimate_for_market(
                list(zip(pp["ts"].to_list(), pp["price"].to_list(), strict=True)),
                list(zip(tt["ts"].to_list(), tt["price"].to_list(), strict=True)),
            )
            py.append((cid, ts, meth))
    return pl.DataFrame(
        py,
        schema={"condition_id": pl.String, "py_ts": pl.Datetime("us"), "py_method": pl.String},
        orient="row",
    )


def parity(snapshot_dir: Path | None = None, overrides_path: Path = OVERRIDES_PATH) -> dict:
    """Bulk estimator vs the per-market function on the pilot markets (those with a snapshot
    ``trades.parquet`` all-rows tape).

    Three results per market: ``bulk`` (taker tape, the production path), ``bulk_all`` (same code
    fed the all-rows tape: isolates implementation equality from the taker-vs-all-rows data
    difference) and ``py`` (the original function, all rows).
    """
    snap = snapshot_dir or (PROJECT_ROOT / "data" / "snapshots")
    inp = load_bulk_inputs(snap)
    all_tr = pl.read_parquet(
        snap / "trades.parquet", columns=["trade_id", "token_id", "condition_id", "ts", "price"]
    )
    mk = inp["markets"].filter(pl.col("condition_id").is_in(all_tr["condition_id"].unique().implode()))
    ov = load_overrides(overrides_path)
    bulk = bulk_news_times(mk, inp["tokens"], inp["prices"], inp["trades"], ov)
    bulk_all = bulk_news_times(mk, inp["tokens"], inp["prices"], all_tr, ov)
    py = _reference_per_market(mk, inp["tokens"], inp["prices"], all_tr, ov)
    j = (
        bulk.rename({"news_ts": "bulk_ts", "news_ts_method": "bulk_method"})
        .join(bulk_all.rename({"news_ts": "all_ts", "news_ts_method": "all_method"}), on="condition_id")
        .join(py, on="condition_id")
    )

    def same(a: str, b: str, am: str, bm: str) -> pl.Expr:
        return ((pl.col(a) == pl.col(b)) | (pl.col(a).is_null() & pl.col(b).is_null())) & (
            pl.col(am) == pl.col(bm)
        )

    j = j.with_columns(
        same("bulk_ts", "py_ts", "bulk_method", "py_method").alias("bulk_eq_py"),
        same("all_ts", "py_ts", "all_method", "py_method").alias("bulk_all_eq_py"),
    )
    return {
        "n": j.height,
        "bulk_taker_equals_reference": int(j["bulk_eq_py"].sum()),
        "bulk_allrows_equals_reference": int(j["bulk_all_eq_py"].sum()),
        "differences": j.filter(~pl.col("bulk_eq_py")),
    }
