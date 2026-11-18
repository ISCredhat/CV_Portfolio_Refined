"""Phase 3 Builder B tests: filters, one-shot scoring, clusters, watchlist, privacy (offline, synthetic)."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from pmdetect import watchlist as wl
from pmdetect.wallets import clusters, filters, oneshot

T0 = datetime(2025, 6, 1, 12, 0)


def addr(i: int) -> str:
    return "0x" + f"{i:040x}"


def _trade(tid, wallet, cond, tok, side, price, minutes=0, usd=100.0):
    return {
        "trade_id": tid, "tx_hash": tid, "condition_id": cond, "token_id": tok, "wallet": wallet,
        "side": side, "outcome_label": "Yes" if tok.endswith("y") else "No",
        "size": usd / price, "price": price, "usd": usd, "ts": T0 + timedelta(minutes=minutes),
    }  # fmt: skip


def _wallets(ws, n_markets=None):
    return pl.DataFrame(
        {
            "wallet": ws,
            "first_seen": [datetime(2025, 1, 1)] * len(ws),
            "n_markets": n_markets or [5] * len(ws),
        }
    ).with_columns(pl.col("n_markets").cast(pl.Int64))


def _positions(rows):
    return pl.DataFrame(rows, infer_schema_length=None)


def test_market_maker_and_contract_filter():
    mm, directional, lp = addr(1), addr(2), addr(3)
    contract = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # public CTF Exchange contract, not a wallet
    rows, n = [], 0
    for m in range(5):
        c = f"c{m}"
        for tok in (f"{c}y", f"{c}n"):  # market maker buys both outcomes
            rows.append(_trade(f"t{n}", mm, c, tok, "BUY", 0.5))
            n += 1
        rows.append(_trade(f"t{n}", directional, c, f"{c}y", "BUY", 0.3))
        n += 1
        rows.append(_trade(f"t{n}", contract, c, f"{c}y", "BUY", 0.5))
        n += 1
    taker = pl.DataFrame(rows)
    # maker rows (not in the taker tape) for the passive liquidity provider
    # (two-sided: alternating BUY / SELL of the same token in every market, ends flat)
    maker_rows = [
        _trade(f"m{i}", lp, f"c{i % 6}", f"c{i % 6}y", "SELL" if (i // 6) % 2 else "BUY", 0.5) for i in range(60)
    ]
    all_trades = pl.concat([taker, pl.DataFrame(maker_rows)])
    positions = _positions(
        [{"wallet": directional, "entry_price": 0.3, "won": True} for _ in range(3)]
        + [{"wallet": lp, "entry_price": 0.5, "won": True}]
    )
    wal = _wallets([mm, directional, lp, contract])
    f = filters.flag_non_directional(taker, positions, wal, all_trades).to_dicts()
    by = {r["wallet"]: r for r in f}
    assert by[mm]["two_sided"] and by[mm]["excluded"]
    assert by[contract]["contract"] and by[contract]["excluded"]
    assert by[lp]["maker"] and by[lp]["excluded"]
    assert not by[directional]["excluded"]
    counts = filters.exclusion_counts(pl.DataFrame(f))
    assert counts["excluded"] == 3 and counts["contract"] == 1


def test_broad_no_edge_and_epoch_quirk():
    broad, quirk = addr(10), addr(11)
    pos = _positions([{"wallet": broad, "entry_price": 0.5, "won": i % 2 == 0} for i in range(40)])
    taker = pl.DataFrame([_trade("a", broad, "c0", "c0y", "BUY", 0.5), _trade("b", quirk, "c0", "c0y", "BUY", 0.5)])
    wal = pl.DataFrame(
        {"wallet": [broad, quirk], "first_seen": [datetime(2025, 1, 1), datetime(1970, 1, 1)], "n_markets": [40, 3]}
    ).with_columns(pl.col("n_markets").cast(pl.Int64))
    by = {r["wallet"]: r for r in filters.flag_non_directional(taker, pos, wal).to_dicts()}
    assert by[broad]["broad_no_edge"] and by[quirk]["contract"]


def _os_positions():
    rows = []
    for i in range(10):
        rows.append(
            {
                "wallet": addr(100 + i), "event_id": "e1", "condition_id": "c1", "entry_price": 0.30 + 0.05 * i,
                "usd": 100.0 + 10 * i, "won": True, "first_ts": T0, "hours_before_cutoff": 10.0 + i,
                "info_type": "deadline",
            }
        )
    return _positions(rows)


def test_oneshot_monotonic_in_each_component():
    base = _os_positions()
    wal = pl.DataFrame(
        {"wallet": base["wallet"], "first_seen": [datetime(2025, 5, 1)] * 10, "n_markets": [3] * 10}
    ).with_columns(pl.col("n_markets").cast(pl.Int64))
    target = base["wallet"][5]
    ref = oneshot.score_oneshot(base, None, wal, None, None).filter(pl.col("wallet") == target)["score"][0]
    for col, delta in (("entry_price", -0.25), ("usd", 5000.0)):
        b = base.with_columns(
            pl.when(pl.col("wallet") == target).then(pl.col(col) + delta).otherwise(pl.col(col)).alias(col)
        )
        s = oneshot.score_oneshot(b, None, wal, None, None).filter(pl.col("wallet") == target)["score"][0]
        assert s > ref, col
    # proximity (Phase 3b): moving the entry much earlier lowers the score
    b = base.with_columns(
        pl.when(pl.col("wallet") == target).then(500.0).otherwise(pl.col("hours_before_cutoff"))
        .alias("hours_before_cutoff")
    )
    s = oneshot.score_oneshot(b, None, wal, None, None).filter(pl.col("wallet") == target)["score"][0]
    assert s < ref
    # a very fresh wallet (first_seen just before entry) scores higher
    w2 = wal.with_columns(
        pl.when(pl.col("wallet") == target).then(datetime(2025, 6, 1, 11, 0)).otherwise(pl.col("first_seen"))
        .alias("first_seen")
    )
    s = oneshot.score_oneshot(base, None, w2, None, None).filter(pl.col("wallet") == target)["score"][0]
    assert s > ref


def test_oneshot_missing_components_explicit_and_losers_excluded():
    base = _os_positions().with_columns(pl.Series("won", [True] * 9 + [False]))
    out = oneshot.score_oneshot(base, None, None, None, None)
    assert out.height == 9  # losing position is not scored
    assert (out["n_missing"] >= 1).all()  # context always missing here
    assert out["score"].is_between(0, 1).all()
    assert out["rank"].to_list() == list(range(1, 10))


def test_cotrading_clusters_planted():
    w1, w2, w3, w4, w5 = (addr(i) for i in range(21, 26))
    rows, n = [], 0
    for m in range(2):  # w1, w2 co-trade in two markets within 5 minutes
        c = f"c{m}"
        rows += [_trade(f"t{n}", w1, c, f"{c}y", "BUY", 0.2, 0), _trade(f"t{n+1}", w2, c, f"{c}y", "BUY", 0.2, 5)]
        n += 2
    # w3, w4 only co-trade in one market; w5 trades the same token far outside the window
    rows += [_trade("x1", w3, "c5", "c5y", "BUY", 0.2, 0), _trade("x2", w4, "c5", "c5y", "BUY", 0.2, 3)]
    rows += [_trade("y1", w5, "c0", "c0y", "BUY", 0.2, 600)]
    # opposite outcome within the window must not link
    rows += [_trade("z1", w5, "c1", "c1n", "BUY", 0.8, 1)]
    cl = clusters.cotrading_clusters(pl.DataFrame(rows))
    assert cl.height == 2 and set(cl["wallet"]) == {w1, w2}
    assert cl["cluster_size"].to_list() == [2, 2] and not cl["oversized"].any()
    # excluding a filtered wallet removes the link
    assert clusters.cotrading_clusters(pl.DataFrame(rows), exclude_wallets={w2}).height == 0
    assert clusters.link_by_funding([w1, w2]).height == 0


def test_cluster_pooling_one_pseudo_wallet():
    w1, w2 = addr(31), addr(32)
    cl = pl.DataFrame(
        {"wallet": [w1, w2], "cluster_id": [1, 1], "cluster_size": [2, 2], "oversized": [False, False]}
    )
    pos = _positions(
        [
            {"wallet": w, "event_id": e, "entry_price": 0.2, "usd": 50.0, "size": 250.0, "won": True,
             "first_ts": T0, "hours_before_cutoff": 5.0, "p_tilde": 0.2}
            for w in (w1, w2)
            for e in ("e1", "e2")
        ]
    )
    pooled = clusters.pool_cluster_positions(pos, cl)
    assert pooled.height == 2 and set(pooled["wallet"]) == {"cluster_1"}
    assert pooled["usd"].to_list() == [100.0, 100.0]


def test_watchlist_tiers_and_privacy(tmp_path):
    ws = [addr(40 + i) for i in range(6)]  # 0 repeat, 1 one-shot, 2 watch(nominal), 3 excluded repeat, 4 low, 5 low
    tests = pl.DataFrame(
        {
            "wallet": ws, "n_pos": [8, 1, 6, 9, 1, 1], "p_pb": [1e-6, 0.4, 0.004, 1e-7, 0.9, 0.8],
            "q_pb": [1e-5, 1.0, 0.1, 1e-6, 1.0, 1.0], "flag": [True, False, False, True, False, False],
        }
    )
    scored = pl.DataFrame(
        {
            "wallet": [ws[1], ws[0], ws[4], ws[5], ws[3]] + [addr(90 + i) for i in range(100)],
            "score": [0.95, 0.5, 0.04, 0.03, 0.99] + [0.01 * i / 100 + 0.05 for i in range(100)],
            "usd": [500.0] * 105, "entry_price": [0.1] * 105, "hours_before_cutoff": [20.0] * 105,
            "condition_id": ["c1"] * 105,
        }
    )
    flags = pl.DataFrame({"wallet": ws, "excluded": [False, False, False, True, False, False]})
    cl = pl.DataFrame({"wallet": [], "cluster_id": [], "cluster_size": [], "oversized": []}, schema={
        "wallet": pl.String, "cluster_id": pl.Int64, "cluster_size": pl.Int64, "oversized": pl.Boolean})
    pos = _positions([{"wallet": w, "condition_id": "c1", "event_id": "e1", "entry_price": 0.2, "usd": 10.0,
                       "won": True} for w in ws])
    out = tmp_path / "private" / "watchlist.csv"
    wl.build_watchlist(tests, scored, flags, cl, None, pos, out_path=out)
    t = {r["wallet"]: r["tier"] for r in pl.read_csv(out).iter_rows(named=True)}
    assert t[ws[0]] == "repeat_anomalous"
    assert t[ws[1]] == "one_shot_anomalous"
    assert t[ws[2]] == "watch"
    assert ws[3] not in t  # excluded wallets never listed
    assert ws[4] not in t and ws[5] not in t

    # aggregate-only summary: no addresses, and the guard raises if one sneaks in
    summ = tmp_path / "results" / "phase3_summary.md"
    wl.write_phase3_summary(
        summ, n_positions=6, n_wallets_tested=6, exclusion={"wallets": 6, "excluded": 1, "contract": 1},
        rule_doc={"contract": "x"}, tests=tests, n_fdr_pass=2, oneshot=scored.with_columns(pl.lit(0).alias("n_missing")),
        watchlist=pl.read_csv(out), clusters=cl, cluster_tests=None,
    )  # fmt: skip
    text = summ.read_text(encoding="utf-8")
    assert "insider" not in text.lower() and "anomalous" in text
    assert wl.assert_results_clean(tmp_path / "results") == 1
    with pytest.raises(AssertionError):
        wl.assert_no_addresses(f"leak {ws[0]}")
    (tmp_path / "results" / "bad.csv").write_text(f"wallet\n{ws[0]}\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        wl.assert_results_clean(tmp_path / "results")
    # 64-hex condition ids are not wallet addresses
    wl.assert_no_addresses("0x" + "ab" * 32)


def test_real_results_dir_has_no_wallet_addresses():
    from pmdetect.store.db import PROJECT_ROOT

    d = PROJECT_ROOT / "results"
    if d.exists():
        wl.assert_results_clean(d)


def test_maker_rule_two_sided_quoter_excluded_directional_limit_buyer_not():
    """Market making is two-sided, inventory-neutral behaviour, not order type."""
    quoter, buyer, buy_both = addr(20), addr(21), addr(22)
    taker_rows, maker_rows = [], []
    n = 0
    for m in range(6):
        c = f"k{m}"
        for j in range(12):  # 72 maker rows each
            # quoter: rests BUY and SELL of YES alternately (ends flat)
            maker_rows.append(_trade(f"q{n}", quoter, c, f"{c}y", "BUY" if j % 2 == 0 else "SELL", 0.5))
            # directional limit-order buyer: only ever rests BUYs of YES
            maker_rows.append(_trade(f"b{n}", buyer, c, f"{c}y", "BUY", 0.4))
            # buys YES and NO by resting orders (hedged = flat in YES-equivalent terms)
            maker_rows.append(_trade(f"x{n}", buy_both, c, f"{c}y" if j % 2 == 0 else f"{c}n", "BUY", 0.5))
            n += 1
        taker_rows.append(_trade(f"tk{m}", addr(30), c, f"{c}y", "BUY", 0.5))  # unrelated taker
    taker = pl.DataFrame(taker_rows)
    all_rows = pl.concat([taker, pl.DataFrame(maker_rows)])
    positions = _positions([{"wallet": w, "entry_price": 0.4, "won": True} for w in (quoter, buyer, buy_both)])
    wal = _wallets([quoter, buyer, buy_both, addr(30)])
    fl = filters.flag_non_directional(taker, positions, wal, all_rows)
    by = {r["wallet"]: r for r in fl.to_dicts()}
    for w in (quoter, buyer, buy_both):  # all pass the old, activity-only screen
        assert by[w]["maker_old"] and by[w]["maker_active"]
    assert by[quoter]["maker"] and by[quoter]["excluded"] and by[quoter]["ntg_w"] < 0.05
    assert by[buy_both]["maker"] and by[buy_both]["ntg_w"] < 0.05  # YES+NO buys net to flat
    assert not by[buyer]["maker"] and not by[buyer]["excluded"] and by[buyer]["ntg_w"] == pytest.approx(1.0)
    ec = filters.exclusion_counts(fl)
    assert ec["maker_old"] == 3 and ec["maker"] == 2 and ec["moved_to_included"] == 1
    h = filters.ntg_histogram(fl)
    assert int(h["wallets"].sum()) == 3


def test_maker_row_is_own_fill_on_complement_token():
    """A maker row on the complement token (BUY NO at 0.9 vs a taker BUY YES at 0.1) is the maker's own
    BUY of NO; in YES-equivalent terms it is SHORT YES, so a wallet that also buys YES by resting orders
    is flat."""
    w = addr(40)
    rows = []
    for m in range(5):
        c = f"z{m}"
        for j in range(12):
            tok, px = (f"{c}y", 0.1) if j % 2 == 0 else (f"{c}n", 0.9)
            rows.append(_trade(f"r{m}_{j}", w, c, tok, "BUY", px, usd=100.0 * px))  # 100 shares each
    taker = pl.DataFrame([_trade(f"t{m}", addr(41), f"z{m}", f"z{m}y", "BUY", 0.1) for m in range(5)])
    fl = filters._maker_flow(pl.concat([taker, pl.DataFrame(rows)]), taker, n_parts=2)
    r = fl.filter(pl.col("wallet") == w).row(0, named=True)
    assert r["n_flow_markets"] == 5 and r["ntg_w"] < 0.2


def test_results_dir_has_no_wallet_addresses():
    """Regression guard: the committed results/ outputs never contain a wallet address."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "results"
    pat = re.compile(r"(?<![0-9a-zA-Z])0x[0-9a-fA-F]{40}(?![0-9a-zA-Z])")  # 64-hex condition ids do not match
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".md", ".csv", ".txt", ".json", ".html"}:
            assert not pat.search(f.read_text(encoding="utf-8", errors="ignore")), f.name
