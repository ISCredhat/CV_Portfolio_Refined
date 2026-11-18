"""Co-trading wallet clusters (no funding-chain data yet).

Link two wallets if they taker-BOUGHT the same outcome token in the same market within
``window_minutes`` of each other in at least ``min_markets`` distinct markets. Connected components of
that graph are clusters. Guards against giant components: callers must drop market makers/bots first
(``filters.flag_non_directional``); anchors with more than ``MAX_NEIGHBOURS`` other buyers inside the
window are skipped (crowd moments carry no pair information); components larger than ``MAX_CLUSTER`` are
re-linked at a higher number of shared markets (``link_components``; at full-universe scale chance
co-buys chain ~19k wallets together at 2 markets) and, if still too large, reported but not calibrated.
Implementation is vectorised for 8M-row tapes (see ``_pairs``).
"""

from __future__ import annotations

import numpy as np
import polars as pl

MAX_NEIGHBOURS = 25
MAX_CLUSTER = 50


def link_by_funding(wallets: list[str]) -> pl.DataFrame:
    """Stub for Polygon funding-chain links (wallet_a, wallet_b, source). Returns an empty frame for now."""
    return pl.DataFrame(schema={"wallet_a": pl.String, "wallet_b": pl.String, "source": pl.String})


_CHUNK_PAIRS = 4_000_000


def _pairs(
    taker_trades: pl.DataFrame | pl.LazyFrame, window_minutes: int, max_neighbours: int
) -> pl.DataFrame:
    """Wallet pairs that bought the same token within the window, counted over distinct markets.

    Vectorised and bounded: BUY rows are dictionary-coded to integers and sorted by one composite key
    (token code, timestamp), so every anchor's window is one ``searchsorted`` over the whole tape (no
    per-market Python loop and no join). Pairs are emitted anchor-by-anchor in chunks of at most
    ``_CHUNK_PAIRS`` pairs and de-duplicated per chunk, so memory is bounded; the per-anchor cap
    (``max_neighbours``) also bounds the total work at ``max_neighbours`` x BUY rows even in a market
    with hundreds of thousands of buys (a crowd moment carries no pair information)."""
    buys = (
        (taker_trades if isinstance(taker_trades, pl.LazyFrame) else taker_trades.lazy())
        .filter((pl.col("side") == "BUY") & pl.col("wallet").is_not_null())
        .select("condition_id", "token_id", "wallet", "ts")
        .collect()
    )
    empty = pl.DataFrame(
        schema={"wallet_a": pl.String, "wallet_b": pl.String, "n_markets": pl.Int64}
    )
    if buys.height < 2:
        return empty
    wallets_sorted = buys["wallet"].unique().sort()  # dense rank = index in this list
    wid = buys["wallet"].rank("dense").to_numpy().astype(np.int64) - 1
    tid = buys["token_id"].rank("dense").to_numpy().astype(np.int64) - 1
    cid = buys["condition_id"].rank("dense").to_numpy().astype(np.int64) - 1
    ts = buys["ts"].cast(pl.Int64).to_numpy()  # microseconds
    del buys
    win = int(window_minutes) * 60_000_000
    unit = 1
    if (int(ts.max() - ts.min()) + win).bit_length() + int(tid.max()).bit_length() > 62:
        unit = 1_000_000  # extreme range: fall back to second resolution
    rel = (ts - ts.min()) // unit
    win //= unit
    shift = (int(rel.max()) + win).bit_length() + 1
    key = (tid << shift) | rel
    order = np.argsort(key, kind="stable")
    key, wid, cid = key[order], wid[order], cid[order]
    hi = np.searchsorted(key, key + win, side="right")
    cnt = hi - np.arange(key.size) - 1
    anchors = np.where((cnt >= 1) & (cnt <= max_neighbours))[0]
    n_c = int(cid.max()) + 1
    n_w = len(wallets_sorted)
    parts: list[np.ndarray] = []
    start = 0
    csum = np.cumsum(cnt[anchors])
    while start < anchors.size:
        base = csum[start - 1] if start else 0
        end = int(np.searchsorted(csum, base + _CHUNK_PAIRS, side="right"))
        end = max(end, start + 1)
        a_idx = anchors[start:end]
        c = cnt[a_idx]
        i_rep = np.repeat(a_idx, c)
        j = i_rep + 1 + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
        wa, wb = wid[i_rep], wid[j]
        keep = wa != wb
        lo_w, hi_w = np.minimum(wa, wb)[keep], np.maximum(wa, wb)[keep]
        parts.append(np.unique((lo_w * n_w + hi_w) * n_c + cid[i_rep][keep]))
        start = end
    if not parts:
        return empty
    triples = np.unique(np.concatenate(parts))
    pair_key, n_mk = np.unique(triples // n_c, return_counts=True)
    return pl.DataFrame(
        {
            "wallet_a": wallets_sorted.gather(pair_key // n_w),
            "wallet_b": wallets_sorted.gather(pair_key % n_w),
            "n_markets": n_mk.astype(np.int64),
        }
    )


MAX_LINK_LEVEL = 8


def _label_components(a: pl.Series, b: pl.Series) -> pl.DataFrame:
    """(wallet, lab) for the connected components of the edge list (a[i], b[i])."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    names = pl.concat([a, b]).unique().sort()
    idx = names.to_frame("w").with_row_index("i")
    ia = a.to_frame("w").join(idx, on="w", how="left")["i"].to_numpy()
    ib = b.to_frame("w").join(idx, on="w", how="left")["i"].to_numpy()
    n = names.len()
    g = coo_matrix((np.ones(ia.size), (ia, ib)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return pl.DataFrame({"wallet": names, "lab": lab})


def link_components(
    pairs: pl.DataFrame, min_markets: int, max_cluster: int = MAX_CLUSTER, refine: bool = True
) -> pl.DataFrame:
    """Connected components of the pairs linked in >= ``min_markets`` markets: (wallet, lab, link_level).

    With ``refine`` a component larger than ``max_cluster`` (chance co-buys in popular markets chain
    thousands of wallets together at the base threshold) is re-linked using only ITS edges with one more
    market in common, repeatedly, up to ``MAX_LINK_LEVEL``; wallets whose links all drop out leave the
    cluster. A component still above the cap at the last level is returned as is (marked oversized by the
    caller). ``link_level`` is the min-markets threshold at which a wallet's final component formed."""
    out: list[pl.DataFrame] = []
    edges, level, base = pairs.filter(pl.col("n_markets") >= min_markets), min_markets, 0
    while edges.height:
        comp = _label_components(edges["wallet_a"], edges["wallet_b"])
        comp = comp.with_columns(pl.len().over("lab").alias("_n"), (pl.col("lab") + base).alias("lab"))
        big = comp.filter(pl.col("_n") > max_cluster) if refine else comp.head(0)
        done = comp.filter(pl.col("_n") <= max_cluster) if refine else comp
        out.append(done.drop("_n").with_columns(pl.lit(level).cast(pl.Int64).alias("link_level")))
        base += int(comp["lab"].max()) + 1
        if big.height == 0:
            break
        if level >= MAX_LINK_LEVEL:  # cannot split further: keep the oversized component
            out.append(big.drop("_n").with_columns(pl.lit(level).cast(pl.Int64).alias("link_level")))
            break
        bw = big["wallet"]
        edges = edges.filter(
            pl.col("wallet_a").is_in(bw.implode()) & pl.col("wallet_b").is_in(bw.implode())
            & (pl.col("n_markets") >= level + 1)
        )  # fmt: skip
        level += 1
    if not out:
        return pl.DataFrame(schema={"wallet": pl.String, "lab": pl.Int64, "link_level": pl.Int64})
    return pl.concat(out).with_columns(pl.col("lab").cast(pl.Int64))


def cotrading_clusters(
    taker_trades: pl.DataFrame | pl.LazyFrame,
    min_markets: int = 2,
    window_minutes: int = 10,
    exclude_wallets: set[str] | None = None,
    max_neighbours: int = MAX_NEIGHBOURS,
    refine: bool = True,
) -> pl.DataFrame:
    """Return (wallet, cluster_id, cluster_size, oversized, link_level). cluster_id is 1..K, largest first.

    ``link_level`` equals ``min_markets`` unless an oversized component had to be re-linked at a higher
    number of shared markets (``refine``, see :func:`link_components`)."""
    t = taker_trades
    if exclude_wallets:
        ex = pl.Series("wallet", sorted(exclude_wallets), dtype=pl.String)
        t = t.filter(~pl.col("wallet").is_in(ex.implode()))
    pairs = _pairs(t, window_minutes, max_neighbours).filter(pl.col("n_markets") >= min_markets)
    schema = {
        "wallet": pl.String,
        "cluster_id": pl.Int64,
        "cluster_size": pl.Int64,
        "oversized": pl.Boolean,
        "link_level": pl.Int64,
    }
    if pairs.height == 0:
        return pl.DataFrame(schema=schema)
    comp = link_components(pairs, min_markets, MAX_CLUSTER, refine)
    order = (
        comp.group_by("lab")
        .agg(pl.len().alias("n"), pl.col("wallet").min().alias("m"))
        .sort(["n", "m"], descending=[True, False])
        .with_row_index("cluster_id", offset=1)
    )
    df = comp.join(order.select("lab", "cluster_id", pl.col("n").alias("cluster_size")), on="lab")
    return (
        df.with_columns(
            pl.col("cluster_id").cast(pl.Int64), pl.col("cluster_size").cast(pl.Int64),
            (pl.col("cluster_size") > MAX_CLUSTER).alias("oversized"),
        )
        .select(list(schema))
        .sort(["cluster_id", "wallet"])
    )


def pool_cluster_positions(positions: pl.DataFrame, clusters: pl.DataFrame) -> pl.DataFrame:
    """One pooled pseudo-wallet per cluster: per (cluster, event) usd-weighted entry, summed usd, any win.

    The pseudo-wallet id is ``cluster_<id>`` so it can go straight into ``wallet_tests``.
    """
    cl = clusters.filter(~pl.col("oversized")).select("wallet", "cluster_id")
    key = "event_id" if "event_id" in positions.columns else "condition_id"
    j = positions.join(cl, on="wallet", how="inner")
    if j.height == 0:
        return j.head(0)
    aggs = [
        ((pl.col("entry_price") * pl.col("usd")).sum() / pl.col("usd").sum()).alias("entry_price"),
        pl.col("usd").sum().alias("usd"),
        pl.col("won").max().alias("won"),
        pl.col("first_ts").min().alias("first_ts"),
        pl.col("hours_before_cutoff").max().alias("hours_before_cutoff"),
    ]
    for c in ("cutoff_ts", "cutoff_method", "info_type", "condition_id"):
        if c in j.columns and c != key:
            aggs.append(pl.col(c).first().alias(c))
    if "size" in j.columns:
        aggs.append(pl.col("size").sum().alias("size"))
    if "p_tilde" in j.columns:  # globally cross-fitted recalibrated probability, usd-weighted
        aggs.append(
            ((pl.col("p_tilde") * pl.col("usd")).sum() / pl.col("usd").sum()).alias("p_tilde")
        )
    g = j.group_by(["cluster_id", key]).agg(aggs)
    return g.with_columns(pl.format("cluster_{}", pl.col("cluster_id")).alias("wallet")).drop(
        "cluster_id"
    )


def calibrate_clusters(calibrated: pl.DataFrame, clusters: pl.DataFrame) -> pl.DataFrame:
    """Rerun Builder A's ``wallet_tests`` on every pooled (non-oversized) cluster.

    ``calibrated`` is the globally cross-fitted positions frame (``fit_isotonic`` first output, with
    ``p_tilde``), so the pooled cluster is tested against the global null, not a tiny local fit.
    Nothing is written (``write=False``): ``wallet_tests.parquet`` stays the per-wallet file.
    """
    pooled = pool_cluster_positions(calibrated, clusters)
    if pooled.height == 0:
        return pl.DataFrame(schema={"cluster_id": pl.Int64, "n_pos": pl.Int64, "p_pb": pl.Float64})
    from pmdetect.wallets.calibration import wallet_tests

    res = wallet_tests(pooled, write=False)
    return res.rename({"wallet": "cluster_wallet"}).with_columns(
        pl.col("cluster_wallet").str.replace("cluster_", "").cast(pl.Int64).alias("cluster_id")
    )
