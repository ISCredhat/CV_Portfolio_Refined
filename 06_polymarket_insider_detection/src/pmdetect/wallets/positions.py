"""Wallet positions from the taker tape (phase 3A).

Definitions (all from the wallet's own *taker* rows, so every price is the price that wallet paid or
received for the token it traded; the maker row of a fill sits at the complement price and is not used):

* Token position: per (wallet, condition, token) only rows with ``ts < cutoff_ts`` of the market count.
  ``buy_size``/``buy_usd`` from BUY rows, ``sell_size`` from SELL rows. Net size = ``buy_size - sell_size``
  (a SELL reduces inventory); tokens with net size <= 0 (sold out, or sells of inventory acquired off the
  tape) are not positions. Entry price = volume-weighted BUY price (average cost, so sells do not change
  it); ``size`` = net size, ``usd`` = size * entry price.
* Event collapse: one row per (wallet, event). Among a wallet's token positions in the same event
  (other outcomes of a multi-outcome event, or both Yes and No of one market) the one with the largest
  ``usd`` is kept; its entry price, token and result define the collapsed position, so dependent outcomes
  are never counted twice. ``n_tokens`` / ``usd_all`` record what was folded away. Only markets present in
  the tape (pilot + extra markets) can contribute, so an event position is built from those alone.
* ``won``: the kept token's label (Yes/No) equals the market outcome (YES/NO); null if unresolved.
* Cutoff (phase 3b rule): independent news time if the market has one, else the changepoint
  ``jump_start_ts``, else NO valid cutoff (``cutoff_method`` = independent_news | jump_start | none;
  ``cutoff_ts`` null, ``valid_cutoff`` False). A price-based news time is not a valid cutoff because
  buying during the resolving jump itself would count as "early". ``hours_before_cutoff`` is measured
  from first_ts and is null without a valid cutoff.
* Legacy columns (phase 3a semantics, kept so consumers still work): ``legacy_cutoff_ts`` /
  ``legacy_cutoff_method`` = independent news > jump start > ``news_ts - 1 h`` (news_minus_buffer), and
  ``hours_before_legacy_cutoff``. Entries are ALWAYS bounded by ``ts < legacy_cutoff_ts`` (which equals
  ``cutoff_ts`` whenever the cutoff is valid, so no entry after a valid cutoff can enter a position);
  ``calib_eligible`` = (cutoff_method == independent_news): only these enter the calibration fit and
  the repeat test (jump_start cutoffs are outcome-selected: a jump toward the winner must exist; they
  stay valid for the one-shot screen). Positions without a valid cutoff are kept for the one-shot screen only and are excluded from the
  calibration fit and the repeat test (``valid_cutoff`` False). Markets with no news time at all and
  no independent/jump time yield no positions.
* Burst cutoff (phase 3c Part B, ``markets/newstime_burst.py``): ``news_burst.parquet`` is consumed ONLY
  where ``burst_validated`` is true (the pre-registered validation criterion on the scheduled reveals was
  met) and the row has ``independent_burst`` true (deadline / unscheduled_event markets). Order of
  precedence: independent_news > independent_burst > jump_start > none. ``cutoff_method`` =
  ``independent_burst`` is calibration-eligible (``calib_eligible``) exactly like ``independent_news``.
  If the file is absent or ``burst_validated`` is false, ``build_positions`` is identical to the phase 3b
  behaviour (the burst markets keep their jump-start / no-cutoff status, screen-only).
  Pipeline wiring: ``pipeline.load_news_burst()`` -> ``build_positions(..., news_burst=...)``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIVATE = PROJECT_ROOT / "data" / "private"
BUFFER = timedelta(hours=1)

POSITION_COLS = [
    "wallet", "event_id", "condition_id", "token_id", "outcome_label", "entry_price", "usd", "size",
    "won", "first_ts", "cutoff_ts", "cutoff_method", "info_type", "hours_before_cutoff",
    "n_tokens", "usd_all", "valid_cutoff", "legacy_cutoff_ts", "legacy_cutoff_method",
    "hours_before_legacy_cutoff", "calib_eligible",
]  # fmt: skip


def usable_burst(news_burst: pl.DataFrame | None) -> pl.DataFrame | None:
    """condition_id, news_ts_burst of the markets whose burst time may be used as a cutoff: only rows
    with ``burst_validated`` AND ``independent_burst`` true (None if there are none or no file)."""
    if news_burst is None or not news_burst.height:
        return None
    need = {"condition_id", "news_ts_burst", "burst_validated", "independent_burst"}
    if not need <= set(news_burst.columns):
        return None
    ok = news_burst.filter(
        pl.col("burst_validated").fill_null(False)
        & pl.col("independent_burst").fill_null(False)
        & pl.col("news_ts_burst").is_not_null()
    )
    if not ok.height:
        return None
    return ok.select("condition_id", pl.col("news_ts_burst").cast(pl.Datetime("us"))).unique("condition_id")


def market_cutoffs(
    markets: pl.DataFrame,
    news_indep: pl.DataFrame | None,
    jumps: pl.DataFrame | None,
    news_burst: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """condition_id, cutoff_ts/cutoff_method (independent news > validated burst > jump start > none)
    and the legacy columns legacy_cutoff_ts/legacy_cutoff_method (... > news - 1 h) used to bound entries.
    ``news_burst`` (news_burst.parquet) counts only through ``usable_burst`` (validated rows only)."""
    ts = pl.Datetime("us")
    m = markets.select("condition_id", pl.col("news_ts").cast(ts))
    if news_indep is not None and news_indep.height:
        ni = news_indep.select(
            "condition_id",
            pl.when(pl.col("independent").fill_null(False)).then(pl.col("news_ts_indep").cast(ts))
            .alias("news_ts_indep"),
        ).unique("condition_id")  # fmt: skip
        m = m.join(ni, on="condition_id", how="left")
    else:
        m = m.with_columns(pl.lit(None, dtype=ts).alias("news_ts_indep"))
    if jumps is not None and jumps.height:
        j = jumps.select("condition_id", pl.col("jump_start_ts").cast(ts)).unique("condition_id")
        m = m.join(j, on="condition_id", how="left")
    else:
        m = m.with_columns(pl.lit(None, dtype=ts).alias("jump_start_ts"))
    ub = usable_burst(news_burst)
    if ub is not None:
        m = m.join(ub, on="condition_id", how="left")
    else:
        m = m.with_columns(pl.lit(None, dtype=ts).alias("news_ts_burst"))
    valid = pl.coalesce(pl.col("news_ts_indep"), pl.col("news_ts_burst"), pl.col("jump_start_ts"))
    m = m.select(
        "condition_id",
        valid.alias("cutoff_ts"),
        pl.when(pl.col("news_ts_indep").is_not_null()).then(pl.lit("independent_news"))
        .when(pl.col("news_ts_burst").is_not_null()).then(pl.lit("independent_burst"))
        .when(pl.col("jump_start_ts").is_not_null()).then(pl.lit("jump_start"))
        .otherwise(pl.lit("none")).alias("cutoff_method"),
        pl.coalesce(valid, pl.col("news_ts") - BUFFER).alias("legacy_cutoff_ts"),
        pl.when(pl.col("news_ts_indep").is_not_null()).then(pl.lit("independent_news"))
        .when(pl.col("news_ts_burst").is_not_null()).then(pl.lit("independent_burst"))
        .when(pl.col("jump_start_ts").is_not_null()).then(pl.lit("jump_start"))
        .when(pl.col("news_ts").is_not_null()).then(pl.lit("news_minus_buffer"))
        .alias("legacy_cutoff_method"),
    )  # fmt: skip
    return m


def token_positions(taker_trades: pl.DataFrame | pl.LazyFrame, cutoffs: pl.DataFrame):
    """Net position per (wallet, condition, token) from taker rows strictly before the market's
    legacy cutoff (= the valid cutoff when there is one, else news - 1 h).

    Works on a DataFrame (returns a DataFrame) or a LazyFrame (returns a LazyFrame; scan the parquet
    with only wallet/condition_id/token_id/outcome_label/side/size/usd/ts so 8M+ rows stream)."""
    cutoffs = cutoffs.filter(pl.col("legacy_cutoff_ts").is_not_null())
    if isinstance(taker_trades, pl.LazyFrame):
        cutoffs = cutoffs.lazy()
    t = taker_trades.join(cutoffs, on="condition_id", how="inner").filter(
        pl.col("legacy_cutoff_ts").is_not_null() & (pl.col("ts") < pl.col("legacy_cutoff_ts"))
    )
    buy = pl.col("side") == "BUY"
    sell = pl.col("side") == "SELL"
    g = t.group_by("wallet", "condition_id", "token_id").agg(
        pl.col("outcome_label").first(),
        pl.col("cutoff_ts").first(),
        pl.col("cutoff_method").first(),
        pl.col("legacy_cutoff_ts").first(),
        pl.col("legacy_cutoff_method").first(),
        pl.col("size").filter(buy).sum().alias("buy_size"),
        pl.col("usd").filter(buy).sum().alias("buy_usd"),
        pl.col("size").filter(sell).sum().alias("sell_size"),
        pl.col("ts").filter(buy).min().alias("first_ts"),
    )
    g = g.with_columns(
        pl.col("buy_size").fill_null(0.0), pl.col("buy_usd").fill_null(0.0),
        pl.col("sell_size").fill_null(0.0),
    )  # fmt: skip
    g = g.filter(pl.col("buy_size") > 0).with_columns(
        (pl.col("buy_size") - pl.col("sell_size")).alias("size"),
        (pl.col("buy_usd") / pl.col("buy_size")).alias("entry_price"),
    )
    return g.filter(pl.col("size") > 1e-9).with_columns(
        (pl.col("size") * pl.col("entry_price")).alias("usd")
    )


def build_positions(
    taker_trades: pl.DataFrame | pl.LazyFrame,
    markets: pl.DataFrame,
    news_indep: pl.DataFrame | None,
    jumps: pl.DataFrame | None,
    event_markets: pl.DataFrame | None = None,
    write: bool = False,
    out: Path | None = None,
    n_parts: int = 6,
    news_burst: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per (wallet, event): see the module docstring. ``markets`` needs condition_id, outcome,
    info_type, news_ts and event_id (or give ``event_markets``); a market without an event is its own."""
    cut = market_cutoffs(markets, news_indep, jumps, news_burst)
    if isinstance(taker_trades, pl.LazyFrame):
        # market-hash partitions are independent (a position never spans markets): same result, 1/n_parts
        # of the peak memory on an 8M+ row tape
        parts = [
            token_positions(
                taker_trades.filter((pl.col("condition_id").hash() % n_parts) == k), cut
            ).collect(engine="streaming")
            for k in range(n_parts)
        ]
        tp = pl.concat(parts)
        del parts
    else:
        tp = token_positions(taker_trades, cut)
    mk = markets.select("condition_id", "outcome", "info_type", *(["event_id"] if "event_id" in markets.columns else []))  # fmt: skip
    if "event_id" not in mk.columns:
        if event_markets is None:
            mk = mk.with_columns(pl.col("condition_id").alias("event_id"))
        else:
            em = event_markets.select("condition_id", "event_id").unique("condition_id")
            mk = mk.join(em, on="condition_id", how="left")
    mk = mk.unique("condition_id")
    tp = tp.join(mk, on="condition_id", how="left")
    tp = tp.with_columns(pl.coalesce("event_id", "condition_id").alias("event_id"))
    tp = tp.with_columns(
        pl.when(pl.col("outcome").is_null()).then(None)
        .otherwise(pl.col("outcome_label").str.to_uppercase() == pl.col("outcome")).alias("won")
    )  # fmt: skip
    tp = tp.sort(["wallet", "event_id", "usd", "token_id"], descending=[False, False, True, False])
    pos = tp.group_by("wallet", "event_id", maintain_order=True).agg(
        pl.all().first(),
        pl.len().alias("n_tokens"),
        pl.col("usd").sum().alias("usd_all"),
    )
    pos = pos.with_columns(
        ((pl.col("cutoff_ts") - pl.col("first_ts")).dt.total_seconds() / 3600.0).alias(
            "hours_before_cutoff"
        ),
        ((pl.col("legacy_cutoff_ts") - pl.col("first_ts")).dt.total_seconds() / 3600.0).alias(
            "hours_before_legacy_cutoff"
        ),
        pl.col("cutoff_ts").is_not_null().alias("valid_cutoff"),
        pl.col("cutoff_method").is_in(["independent_news", "independent_burst"]).alias("calib_eligible"),
    ).select(POSITION_COLS)
    if write:
        PRIVATE.mkdir(parents=True, exist_ok=True)
        p = Path(out) if out else PRIVATE / "positions.parquet"
        pos.write_parquet(p.with_suffix(".tmp"))
        p.with_suffix(".tmp").replace(p)
    return pos


# --------------------------------------------------------------------------- cross-check

API = "https://data-api.polymarket.com/v2/positions"


def _api_positions(wallet: str, conditions: list[str]) -> list[dict]:
    from pmdetect.collect.http import get_json

    rows: list[dict] = []
    for i in range(0, len(conditions), 20):
        for status in ("OPEN", "CLOSED", "REDEEMABLE", "REDEEMABLE_LOST"):
            cur = None
            for _ in range(20):
                params = {"user": wallet, "condition": ",".join(conditions[i : i + 20]),
                          "status": status, "limit": 1000, "cursor": cur}  # fmt: skip
                page = get_json(API, params)
                rows += [dict(r, _status=status) for r in page.get("data", [])]
                cur = (page.get("pagination") or {}).get("next_cursor")
                if not cur:
                    break
    return rows


def crosscheck_positions(
    positions: pl.DataFrame,
    wallets: list[str],
    n: int = 10,
    taker_trades: pl.DataFrame | None = None,
    tol_price: float = 0.02,
    tol_rel: float = 0.15,
) -> pl.DataFrame:
    """Compare token-level entry prices with /v2/positions for sampled wallets (store markets only).

    Per (wallet, token) on the store's conditions: our full-tape BUY VWAP and bought size (all taker
    rows, ``taker_trades`` required) against the API's ``avg_price`` and ``total_size``. The API also
    sees maker fills, merges and transfers, so size agreement is only expected for purely-taker
    wallets; entry-price agreement is the meaningful check. One row per compared token.
    """
    if taker_trades is None:
        raise ValueError("taker_trades required")
    out = []
    for w in wallets[:n]:
        tw = taker_trades.filter(pl.col("wallet") == w)
        conds = tw["condition_id"].unique().to_list()
        ours = (
            tw.filter(pl.col("side") == "BUY")
            .group_by("token_id", "condition_id")
            .agg(pl.col("size").sum().alias("buy_size"), pl.col("usd").sum().alias("buy_usd"))
            .with_columns((pl.col("buy_usd") / pl.col("buy_size")).alias("our_entry"))
        )
        try:
            api = _api_positions(w, conds)
        except Exception as e:  # noqa: BLE001
            out.append({"wallet": w, "token_id": None, "status": f"api_error:{type(e).__name__}"})
            continue
        by_tok = {r["token_id"]: r for r in api}
        for r in ours.iter_rows(named=True):
            a = by_tok.get(r["token_id"])
            if a is None:
                out.append({"wallet": w, "token_id": r["token_id"], "condition_id": r["condition_id"],
                            "our_entry": r["our_entry"], "our_buy_size": r["buy_size"],
                            "status": "missing_in_api"})  # fmt: skip
                continue
            ap, asz = float(a["avg_price"]), float(a["total_size"])
            out.append({
                "wallet": w, "token_id": r["token_id"], "condition_id": r["condition_id"],
                "our_entry": r["our_entry"], "api_avg_price": ap, "our_buy_size": r["buy_size"],
                "api_total_size": asz, "status": a["_status"],
                "price_ok": abs(ap - r["our_entry"]) <= tol_price,
                "size_ok": abs(asz - r["buy_size"]) <= tol_rel * max(asz, r["buy_size"]),
            })  # fmt: skip
    return pl.DataFrame(out, infer_schema_length=None)
