"""Exclude non-directional wallets (market makers, bots, contracts) before anomaly scoring.

Rules (thresholds are module constants, set a priori and NOT tuned on labelled cases):

1. ``contract``      : address is a known Polymarket exchange / CTF / adapter / USDC / zero address, OR
                       ``first_seen`` is null or before 2020-01-01 (zero-timestamp quirk), OR
                       ``n_markets_global >= CONTRACT_N_MARKETS`` (hub-like, thousands of markets).
2. ``two_sided``     : among its taker markets (>= TWO_SIDED_MIN_MARKETS) at least TWO_SIDED_SHARE of them
                       are markets where the wallet taker-BOUGHT both outcome tokens (YES and NO), or
                       bought and sold the same token (round trip).
3. ``maker``         : market maker = two-sided, inventory-neutral quoting. Needs ALL of
                       (a) activity: >= MAKER_MIN_ROWS maker-side fills (all-rows tape rows not in the
                           taker tape) in >= MAKER_MIN_MARKETS markets, maker fills >= MAKER_SHARE of all fills;
                       (b) inventory-neutral: gross-weighted net-to-gross ratio over its maker markets
                           ``sum|L-S| / sum(L+S) <= MAKER_MAX_NTG`` (L/S = long/short share volume in a
                           YES-equivalent direction, over ALL its maker+taker rows in the market);
                       (c) two-sided: >= MAKER_MIN_TWO_SIDED_MKTS of its maker markets have a minority-side
                           volume share >= TWO_SIDED_MINORITY.
                       A directional wallet that merely enters by resting limit orders (net long one side,
                       minority share ~ 0) fails (b) and (c) and is NOT excluded. Order type is not the
                       criterion; behaviour is. ``maker_old`` keeps the previous activity-only rule (a)
                       for comparison and is not part of ``excluded``. Thresholds are fixed a priori, from
                       the pooled distribution of the activity-screened wallets only (net-to-gross median
                       0.21, minority-share is bimodal at flat and one-sided; histogram in
                       results/phase3_summary.md): ntg <= 0.30 means the net position is at most ~30% of
                       gross, i.e. a minority side of >= 35% of volume; not tuned on any labelled wallet.
                       Without token/side/size columns on the all-rows tape rule (b)/(c) cannot be
                       evaluated and the rule falls back to (a) alone (reported as such).
4. ``broad_no_edge`` : active in many markets (>= BROAD_MIN_POSITIONS positions in the store or
                       n_markets_global >= BROAD_N_MARKETS_GLOBAL) with near-zero edge (mean of
                       won - entry price below BROAD_MAX_EDGE, computed over the wallet's positions;
                       skipped if the wallet has fewer than BROAD_MIN_POSITIONS positions in the store
                       and no n_markets_global).
"""

from __future__ import annotations

import polars as pl

# PUBLIC CONTRACT ADDRESSES, NOT WALLETS OF PERSONS: Polymarket exchange / conditional-token / adapter
# contracts, USDC.e and the zero / burn addresses. They are the only 40-hex strings allowed outside
# data/ (scripts/privacy_scan.py allow-lists exactly this set; tests/test_privacy_scan.py checks they agree).
KNOWN_CONTRACTS = frozenset(
    a.lower()
    for a in (
        "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",  # CTF Exchange
        "0xC5d563A36AE78145C45a50134d48A1215220f80a",  # Neg Risk CTF Exchange
        "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",  # Conditional Tokens
        "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",  # Neg Risk Adapter
        "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # USDC.e
        "0x3A3BD7bb9528E159577F7C2e685CC81A765002E2",  # Neg Risk fee module
        "0x0000000000000000000000000000000000000000",
        "0x000000000000000000000000000000000000dead",
    )
)
EPOCH_CUTOFF = "2020-01-01"
CONTRACT_N_MARKETS = 3000
TWO_SIDED_MIN_MARKETS = 3
TWO_SIDED_SHARE = 0.30
MAKER_MIN_ROWS = 50
MAKER_MIN_MARKETS = 5
MAKER_SHARE = 0.50
TWO_SIDED_MINORITY = 0.25
MAKER_MAX_NTG = 0.30
MAKER_MIN_TWO_SIDED_MKTS = 0.50
BROAD_MIN_POSITIONS = 30
BROAD_N_MARKETS_GLOBAL = 500
BROAD_MAX_EDGE = 0.02

RULES = ("contract", "two_sided", "maker", "broad_no_edge")


def _lazy(x: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return x if isinstance(x, pl.LazyFrame) else x.lazy()


def _two_sided(taker: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Per wallet: n_taker_markets and the share of those markets that are two-sided (both outcome
    tokens taker-bought, or a buy/sell round trip on one token). One streaming pass over the tape."""
    tok = (
        _lazy(taker)
        .group_by("wallet", "condition_id", "token_id")
        .agg(
            (pl.col("side") == "BUY").any().alias("_b"),
            (pl.col("side") == "SELL").any().alias("_s"),
        )
    )
    mkt = tok.group_by("wallet", "condition_id").agg(
        (pl.col("_b") & pl.col("_s")).any().alias("_round"),
        pl.col("_b").sum().alias("_nbuy_tokens"),
    )
    return (
        mkt.with_columns((pl.col("_round") | (pl.col("_nbuy_tokens") >= 2)).alias("two_sided_mkt"))
        .group_by("wallet")
        .agg(
            pl.len().alias("n_taker_markets"),
            pl.col("two_sided_mkt").mean().alias("two_sided_share"),
        )
    )


def _two_sided_partitioned(taker: pl.LazyFrame, n_parts: int = 6) -> pl.DataFrame:
    """:func:`_two_sided` over wallet-hash partitions (a wallet is wholly in one partition, so the
    concatenation is exact) to bound peak memory on the full tape."""
    parts = [
        _two_sided(taker.filter((pl.col("wallet").hash() % n_parts) == k)).collect(engine="streaming")
        for k in range(n_parts)
    ]
    return pl.concat(parts)


def maker_coverage(
    all_trades: pl.DataFrame | pl.LazyFrame | None, taker: pl.DataFrame | pl.LazyFrame
) -> dict:
    """How much of the taker tape's market set the all-rows (maker+taker) tape covers.

    The maker rule can only see maker fills in markets present in ``all_trades``; the caller reports
    this so a partial all-rows tape is never mistaken for a complete one."""
    t_m = _lazy(taker).select("condition_id").unique().collect()["condition_id"]
    if all_trades is None:
        return {"taker_markets": t_m.len(), "all_rows_markets": 0, "covered": 0, "share": 0.0,
                "complete": False}  # fmt: skip
    a_m = _lazy(all_trades).select("condition_id").unique().collect()["condition_id"]
    cov = int(t_m.is_in(a_m.implode()).sum())
    return {
        "taker_markets": t_m.len(), "all_rows_markets": a_m.len(), "covered": cov,
        "share": cov / max(t_m.len(), 1), "complete": cov >= t_m.len(),
    }  # fmt: skip


def _maker(
    all_rows: pl.DataFrame | pl.LazyFrame, taker: pl.DataFrame | pl.LazyFrame
) -> pl.LazyFrame:
    """Maker-side rows = all-rows tape minus taker rows, matched on a 64-bit hash of trade_id (anti
    join on integers: far lighter than on 66-character strings), restricted to the markets that the
    all-rows tape contains so the maker share compares like with like."""
    a, t = _lazy(all_rows), _lazy(taker)
    covered = a.select("condition_id").unique()
    t_cov = t.join(covered, on="condition_id", how="semi")
    t_ids = t_cov.select(pl.col("trade_id").hash().alias("_h"))
    maker = (
        a.with_columns(pl.col("trade_id").hash().alias("_h"))
        .join(t_ids, on="_h", how="anti")
        .filter(pl.col("wallet").is_not_null())
    )
    m = maker.group_by("wallet").agg(
        pl.len().alias("n_maker_rows"), pl.col("condition_id").n_unique().alias("n_maker_markets")
    )
    tk = t_cov.group_by("wallet").agg(pl.len().alias("n_taker_rows_cov"))
    return m.join(tk, on="wallet", how="left").with_columns(
        pl.col("n_taker_rows_cov").fill_null(0),
        (
            pl.col("n_maker_rows") / (pl.col("n_maker_rows") + pl.col("n_taker_rows_cov").fill_null(0))
        ).alias("maker_share"),
    ).drop("n_taker_rows_cov")  # fmt: skip


def _maker_flow(
    all_rows: pl.DataFrame | pl.LazyFrame, taker: pl.DataFrame | pl.LazyFrame, n_parts: int = 8
) -> pl.DataFrame:
    """Inventory-neutrality of each wallet in the markets where it posts maker fills.

    Uses ALL of the wallet's rows (maker + taker) in those markets. Every row of the all-rows tape is the
    wallet's OWN fill with its own token, side and price (a maker row on the complement token at the
    complement price is a real fill of the complement token, e.g. the maker BUYs NO at 0.913 against a
    taker BUY of YES at 0.087), so no price or side transformation is needed; rows are only mapped to a
    common YES-equivalent direction per market: with ``tok0`` the lexicographically smallest token id seen
    for the condition, a fill is *long* when it is BUY of tok0 or SELL of another token, *short* otherwise.
    Per (wallet, market): ``L``/``S`` = long/short share volume, ``gross = L + S``,
    ``ntg = |L - S| / gross`` (0 = flat, two-sided; 1 = purely directional). Wallet aggregates over its
    maker markets: gross-weighted ``ntg_w = sum|L-S| / sum gross`` and the share of maker markets that are
    two-sided (``minority share = min(L,S)/gross >= TWO_SIDED_MINORITY``). Needs token_id, side, size.
    Wallet-hash partitions (a wallet lives in exactly one) bound memory."""
    a, t = _lazy(all_rows), _lazy(taker)
    covered = a.select("condition_id").unique()
    t_ids = (
        t.join(covered, on="condition_id", how="semi")
        .select(pl.col("trade_id").hash().alias("_h"))
        .with_columns(pl.lit(True).alias("_tk"))
        .unique("_h")
    )
    tok0 = a.group_by("condition_id").agg(pl.col("token_id").min().alias("_tok0")).collect()
    parts = []
    for k in range(n_parts):
        rows = (
            a.filter(pl.col("wallet").is_not_null() & ((pl.col("wallet").hash() % n_parts) == k))
            .with_columns(pl.col("trade_id").hash().alias("_h"))
            .join(t_ids, on="_h", how="left")
            .join(tok0.lazy(), on="condition_id", how="left")
            .with_columns(
                (
                    ((pl.col("side") == "BUY") == (pl.col("token_id") == pl.col("_tok0")))
                ).alias("_long"),
                pl.col("_tk").is_null().alias("_mk"),
            )
        )
        pm = rows.group_by("wallet", "condition_id").agg(
            pl.col("size").filter(pl.col("_long")).sum().alias("L"),
            pl.col("size").filter(~pl.col("_long")).sum().alias("S"),
            pl.col("_mk").any().alias("_has_maker"),
        )
        pm = pm.filter(pl.col("_has_maker")).with_columns((pl.col("L") + pl.col("S")).alias("gross"))
        parts.append(
            pm.group_by("wallet")
            .agg(
                pl.len().alias("n_flow_markets"),
                ((pl.col("L") - pl.col("S")).abs().sum() / pl.col("gross").sum()).alias("ntg_w"),
                (
                    (pl.min_horizontal("L", "S") / pl.col("gross")) >= TWO_SIDED_MINORITY
                ).mean().alias("two_sided_mkt_share"),
            )
            .collect(engine="streaming")
        )
    return pl.concat(parts)


def ntg_histogram(flags: pl.DataFrame, bins: int = 10) -> pl.DataFrame:
    """Aggregate counts of the gross-weighted net-to-gross ratio over wallets that pass the maker ACTIVITY
    screen (no addresses): used to document where the a-priori threshold sits."""
    d = flags.filter(pl.col("maker_active") & pl.col("ntg_w").is_not_null())
    edges = [i / bins for i in range(bins + 1)]
    lab = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(bins)]
    h = d.select(
        pl.col("ntg_w").cut(edges[1:-1], labels=lab, left_closed=True).alias("ntg_bin")
    ).group_by("ntg_bin").len().rename({"len": "wallets"})
    return pl.DataFrame({"ntg_bin": lab}).join(h.with_columns(pl.col("ntg_bin").cast(pl.String)), on="ntg_bin", how="left").with_columns(pl.col("wallets").fill_null(0))


def flag_non_directional(
    taker_trades: pl.DataFrame | pl.LazyFrame,
    positions: pl.DataFrame,
    wallets: pl.DataFrame,
    all_trades: pl.DataFrame | pl.LazyFrame | None = None,
) -> pl.DataFrame:
    """One row per wallet seen in ``taker_trades`` (plus positions/wallets): rule flags and ``excluded``.

    ``wallets`` needs wallet, first_seen, n_markets (global); ``positions`` needs wallet, entry_price, won.
    ``all_trades`` is the maker+taker tape (trades.parquet); without it the maker rule is skipped. With a
    LazyFrame ``taker_trades`` (needs wallet, condition_id, token_id, side; plus trade_id when the maker
    rule runs) the 8M-row tape streams and is never held in memory whole.
    """
    taker = _lazy(taker_trades)
    base = (
        pl.concat([taker.select("wallet").unique().collect(), positions.select("wallet")])
        .unique()
        .drop_nulls()
    )
    wal = wallets.select(
        "wallet",
        pl.col("first_seen"),
        pl.col("n_markets").cast(pl.Int64).alias("n_markets_global"),
    ).unique("wallet")
    out = base.join(wal, on="wallet", how="left").join(
        _two_sided_partitioned(taker), on="wallet", how="left"
    )
    has_flow = False
    if all_trades is not None and _lazy(all_trades).select(pl.len()).collect().item():
        out = out.join(_maker(all_trades, taker).collect(engine="streaming"), on="wallet", how="left")
        has_flow = {"token_id", "side", "size"} <= set(_lazy(all_trades).collect_schema().names())
        if has_flow:
            out = out.join(_maker_flow(all_trades, taker), on="wallet", how="left")
    else:
        out = out.with_columns(
            pl.lit(None, pl.Int64).alias("n_maker_rows"),
            pl.lit(None, pl.Int64).alias("n_maker_markets"),
            pl.lit(None, pl.Float64).alias("maker_share"),
        )
        has_flow = False
    if not has_flow:
        out = out.with_columns(
            pl.lit(None, pl.Int64).alias("n_flow_markets"),
            pl.lit(None, pl.Float64).alias("ntg_w"),
            pl.lit(None, pl.Float64).alias("two_sided_mkt_share"),
        )
    pe = (
        positions.filter(pl.col("won").is_not_null())
        .group_by("wallet")
        .agg(
            pl.len().alias("n_pos"),
            (pl.col("won").cast(pl.Float64) - pl.col("entry_price")).mean().alias("mean_edge"),
        )
    )
    out = out.join(pe, on="wallet", how="left")
    epoch = pl.lit(EPOCH_CUTOFF).str.to_datetime()
    active = (
        (pl.col("n_maker_rows").fill_null(0) >= MAKER_MIN_ROWS)
        & (pl.col("n_maker_markets").fill_null(0) >= MAKER_MIN_MARKETS)
        & (pl.col("maker_share").fill_null(0) >= MAKER_SHARE)
    )
    out = out.with_columns(
        (
            pl.col("wallet").str.to_lowercase().is_in(list(KNOWN_CONTRACTS))
            | (pl.col("first_seen").is_not_null() & (pl.col("first_seen") < epoch))
            | (pl.col("n_markets_global").fill_null(0) >= CONTRACT_N_MARKETS)
        ).alias("contract"),
        (
            (pl.col("n_taker_markets").fill_null(0) >= TWO_SIDED_MIN_MARKETS)
            & (pl.col("two_sided_share").fill_null(0) >= TWO_SIDED_SHARE)
        ).alias("two_sided"),
        active.alias("maker_active"),
        active.alias("maker_old"),
        (
            active
            & (
                (pl.col("ntg_w").fill_null(1.0) <= MAKER_MAX_NTG)
                & (pl.col("two_sided_mkt_share").fill_null(0.0) >= MAKER_MIN_TWO_SIDED_MKTS)
                if has_flow
                else pl.lit(True)
            )
        ).alias("maker"),
        (
            (
                (pl.col("n_pos").fill_null(0) >= BROAD_MIN_POSITIONS)
                | (pl.col("n_markets_global").fill_null(0) >= BROAD_N_MARKETS_GLOBAL)
            )
            & pl.col("mean_edge").is_not_null()
            & (pl.col("mean_edge") < BROAD_MAX_EDGE)
        ).alias("broad_no_edge"),
    )
    out = out.with_columns(pl.any_horizontal(*RULES).alias("excluded"))
    old_rules = [pl.col("maker_old" if r == "maker" else r) for r in RULES]
    out = out.with_columns(pl.any_horizontal(*old_rules).alias("excluded_old_maker_rule"))
    return out.with_columns(
        pl.concat_str(
            [pl.when(pl.col(r)).then(pl.lit(r)).otherwise(None) for r in RULES],
            separator="+",
            ignore_nulls=True,
        ).alias("reasons")
    )


def exclusion_counts(flags: pl.DataFrame) -> dict[str, int]:
    d = {"wallets": flags.height, "excluded": int(flags["excluded"].sum())}
    for r in RULES:
        d[r] = int(flags[r].sum())
    if "maker_old" in flags.columns:
        d["maker_old"] = int(flags["maker_old"].sum())
        d["maker_both"] = int((flags["maker_old"] & flags["maker"]).sum())
        d["excluded_old"] = int(flags["excluded_old_maker_rule"].sum())
        d["moved_to_included"] = int((flags["excluded_old_maker_rule"] & ~flags["excluded"]).sum())
    return d


RULE_DOC = {
    "contract": (
        f"known exchange/CTF/adapter/USDC/zero addresses; first_seen null or < {EPOCH_CUTOFF}; "
        f"n_markets_global >= {CONTRACT_N_MARKETS}"
    ),
    "two_sided": (
        f">= {TWO_SIDED_MIN_MARKETS} taker markets and >= {TWO_SIDED_SHARE:.0%} of them with both "
        "outcome tokens bought or a buy/sell round trip on one token"
    ),
    "maker": (
        f">= {MAKER_MIN_ROWS} maker-side fills in >= {MAKER_MIN_MARKETS} markets and maker fills "
        f">= {MAKER_SHARE:.0%} of all fills (activity), AND two-sided / inventory-neutral over its maker "
        f"markets: gross-weighted |net|/gross <= {MAKER_MAX_NTG} and >= {MAKER_MIN_TWO_SIDED_MKTS:.0%} of "
        f"them with minority-side volume share >= {TWO_SIDED_MINORITY:.0%}"
    ),
    "broad_no_edge": (
        f">= {BROAD_MIN_POSITIONS} positions in store or n_markets_global >= {BROAD_N_MARKETS_GLOBAL}, "
        f"with mean(won - entry price) < {BROAD_MAX_EDGE}"
    ),
}
