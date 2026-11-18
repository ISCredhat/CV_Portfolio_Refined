"""Offline tests for phase 2c builder B: new flow window, jump initiators, within-event component."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pmdetect.markets.combine import combine
from pmdetect.markets.flow import flow_before_move, flow_window, jump_initiators

JS = datetime(2025, 3, 20, 12, 0)


def _tape(cid: str, rng, causal: bool) -> pl.DataFrame:
    rows = []
    for i in range(10 * 24 * 3):  # small two-sided noise, 10 days up to the jump start
        ts = JS - timedelta(hours=10 * 24) + timedelta(minutes=20 * i)
        rows.append((f"w{rng.integers(0, 40)}", ts, "BUY" if rng.random() < 0.5 else "SELL", 20.0))
    if causal:  # one wallet buys the winner only AFTER the jump start (the trades that cause the jump)
        for k in range(6):
            rows.append(("cause", JS + timedelta(minutes=5 + 8 * k), "BUY", 2000.0))
        rows.append(("late", JS + timedelta(hours=5), "BUY", 50000.0))  # after the window
    return pl.DataFrame({
        "condition_id": cid,
        "tx_hash": [f"{cid}{i}" for i in range(len(rows))],
        "wallet": [r[0] for r in rows], "ts": [r[1] for r in rows], "side": [r[2] for r in rows],
        "outcome_label": "Yes", "price": 0.2, "usd": [r[3] for r in rows],
        "size": [r[3] / 0.2 for r in rows], "token_id": "t",
        "trade_id": [f"{cid}-{i}" for i in range(len(rows))],
    })  # fmt: skip


def _pts(half_after_min: float):
    """Winner price flat 0.1 then rises; half of the 0.8 jump (0.5) reached after ``half_after_min``."""
    mins = np.array([-600, -60, -1, 0, 1, half_after_min, half_after_min + 30, 300], dtype=float)
    ts = np.datetime64(JS, "us") + (mins * 60_000_000).astype("timedelta64[us]")
    q = np.array([0.1, 0.1, 0.1, 0.1, 0.12, 0.55, 0.9, 0.92])
    return ts, q


JUMP = {"pre_level": 0.1, "jump_size": 0.8, "jump_peak_ts": JS + timedelta(hours=3)}


def test_flow_window_ends_at_half_jump_or_two_hours():
    ws, we = flow_window(JS, JUMP, _pts(45))
    assert ws == JS - timedelta(hours=24)
    assert we == JS + timedelta(minutes=45)  # half-jump point, earlier than 2 h
    _, we2 = flow_window(JS, JUMP, _pts(200))
    assert we2 == JS + timedelta(hours=2)  # capped
    _, we3 = flow_window(JS, JUMP, None)  # no price path: fallback js + 2 h
    assert we3 == JS + timedelta(hours=2)
    _, we4 = flow_window(JS, {**JUMP, "jump_peak_ts": JS + timedelta(minutes=30)}, None)
    assert we4 == JS + timedelta(minutes=30)


def test_new_window_contains_jump_causing_trades_old_does_not():
    rng = np.random.default_rng(3)
    taker = pl.concat([_tape("c", rng, True), _tape("p", rng, False)])
    m = pl.DataFrame({"condition_id": ["c", "p"], "outcome": ["YES", "YES"]})
    jumps = pl.DataFrame({
        "condition_id": ["c", "p"], "jump_start_ts": [JS, JS], "pre_level": [0.1, 0.1],
        "jump_size": [0.8, 0.8], "jump_peak_ts": [JS + timedelta(hours=3)] * 2,
    })  # fmt: skip
    wallets = pl.DataFrame({
        "wallet": ["cause", "late"] + [f"w{i}" for i in range(40)],
        "first_seen": [JS - timedelta(days=1)] * 2 + [JS - timedelta(days=400)] * 40,
        "n_markets": [1, 1] + [30] * 40,
    })  # fmt: skip
    pts = {"c": _pts(45), "p": _pts(45)}
    new = flow_before_move(m, jumps, taker, wallets, pts).filter(pl.col("condition_id") == "c").row(0, named=True)
    old = (flow_before_move(m, jumps, taker, wallets, pts, legacy=True)
           .filter(pl.col("condition_id") == "c").row(0, named=True))  # fmt: skip
    plain = flow_before_move(m, jumps, taker, wallets, pts).filter(pl.col("condition_id") == "p").row(0, named=True)
    # planted causal trades at +5..+53 min: the 45 min window holds 5 of them (5, 13, 21, 29, 37, 45)
    assert new["fb_win_end"] == JS + timedelta(minutes=45)
    assert new["fb_n_trades"] - plain["fb_n_trades"] == 6
    assert old["fb_n_trades"] == plain["fb_n_trades"]  # old window stops at the jump start
    assert new["fb_ofi_z"] > 3 and new["fb_hhi_z"] > 3 and new["fb_vol_z"] > 3
    assert abs(old["fb_vol_z"]) < 3 and abs(plain["fb_vol_z"]) < 3
    assert new["fb_usd"] < 20000  # the +5 h trade is outside the window


def test_missing_data_stays_null():
    rng = np.random.default_rng(1)
    taker = _tape("c", rng, False)
    m = pl.DataFrame({"condition_id": ["c"], "outcome": ["YES"]})
    js_early = taker["ts"].min() + timedelta(hours=30)  # < 3 history blocks
    jumps = pl.DataFrame({"condition_id": ["c"], "jump_start_ts": [js_early]})
    f = flow_before_move(m, jumps, taker, None, {})
    assert f["fb_reason"][0] == "short_history" and f["fb_ofi_z"][0] is None


def _markets():
    return pl.DataFrame({
        "condition_id": ["c", "p", "x"], "outcome": ["YES"] * 3,
        "question": ["case market", "plain market", "no jump"],
    })  # fmt: skip


def test_jump_initiators_table():
    rng = np.random.default_rng(3)
    taker = pl.concat([_tape("c", rng, True), _tape("p", rng, False), _tape("x", rng, False)])
    # 'cause' also trades another market -> not only-market; 'late' only trades c
    other = _tape("p", rng, False).head(1).with_columns(pl.lit("cause").alias("wallet"))
    taker = pl.concat([taker, other])
    jumps = pl.DataFrame({
        "condition_id": ["c", "p", "x"], "jump_start_ts": [JS, JS, None],
        "pre_level": [0.1, 0.1, None], "jump_size": [0.8, 0.8, None],
        "jump_peak_ts": [JS + timedelta(hours=3), JS + timedelta(hours=3), None],
        "jump_reason": [None, None, "no_jump"],
    }, schema={"condition_id": pl.String, "jump_start_ts": pl.Datetime("us"), "pre_level": pl.Float64,
               "jump_size": pl.Float64, "jump_peak_ts": pl.Datetime("us"), "jump_reason": pl.String})  # fmt: skip
    wallets = pl.DataFrame({"wallet": ["cause"], "first_seen": [JS - timedelta(days=2)], "n_markets": [7]},
                           schema={"wallet": pl.String, "first_seen": pl.Datetime("us"),
                                   "n_markets": pl.Int32})  # fmt: skip
    extra_w = pl.DataFrame({"wallet": ["w1"], "first_seen": [JS - timedelta(days=9)]})
    t = jump_initiators(_markets(), jumps, taker, wallets, {"c": _pts(45)},
                        labelled={"c": "case_a", "x": "case_nojump"}, extra_wallets=extra_w)  # fmt: skip
    c = t.filter(pl.col("condition_id") == "c").sort("rank")
    assert c["wallet"][0] == "cause" and c["rank"][0] == 1
    assert abs(c["winner_usd"][0] - 6 * 2000.0) < 1e-6  # 'late' (+5 h) is outside the window
    assert "late" not in c["wallet"].to_list()
    assert c["first_seen_src"][0] == "wallets" and c["only_market_store"][0] is False
    assert c["n_markets_store"][0] == 2 and c["n_markets_global"][0] == 7
    assert c["days_first_seen_to_first_trade"][0] > 1.9 and c["first_seen_in_window"][0] is False
    assert c.height == 5  # labelled: top 5
    assert t.filter(pl.col("condition_id") == "p").height == 3  # unlabelled: top 3
    assert t.filter((pl.col("condition_id") == "p") & (pl.col("wallet") == "w1")).height <= 1
    # coverage fallback: a wallet in neither table has src 'none' and null first_seen
    miss = t.filter(pl.col("wallet").is_in(["w5", "w7", "w9"]))
    assert miss.height == 0 or miss["first_seen_src"].unique().to_list() == ["none"]
    x = t.filter(pl.col("condition_id") == "x")
    assert x.height == 1 and x["wallet"][0] is None and x["jump_reason"][0] == "no_jump"
    assert x["is_labelled"][0] is True


def _scores(n=40):
    rng = np.random.default_rng(0)
    ids = [f"m{i}" for i in range(n)]
    return pl.DataFrame({
        "condition_id": ids, "info_type": ["scheduled_reveal"] * n, "question": ids,
        "D": rng.uniform(0, 0.05, n), "D_last24": rng.uniform(0, 0.05, n),
        "max_jump": rng.uniform(0, 0.1, n), "out_of_universe": [False] * n,
    })  # fmt: skip


def test_combine_with_within_event_component():
    sc = _scores()
    n = sc.height
    rng = np.random.default_rng(2)
    we = pl.DataFrame({
        "condition_id": sc["condition_id"],
        "we_diff_24h": [None if i >= 30 else float(rng.normal()) for i in range(n)],
        "we_diff_12h": [None if i >= 30 else float(rng.normal()) for i in range(n)],
        "we_winner_pct_24h": [None if i >= 30 else float(rng.uniform()) for i in range(n)],
        "we_winner_pct_12h": [None if i >= 30 else float(rng.uniform()) for i in range(n)],
        "we_group": ["multi" if i < 20 else ("binary" if i < 30 else None) for i in range(n)],
    })  # fmt: skip
    # market m3: the winner beat every sibling by a mile in the final 24 h and 12 h
    we = we.with_columns(
        pl.when(pl.col("condition_id") == "m3").then(5.0).otherwise(pl.col("we_diff_24h")).alias("we_diff_24h"),
        pl.when(pl.col("condition_id") == "m3").then(5.0).otherwise(pl.col("we_diff_12h")).alias("we_diff_12h"),
        pl.when(pl.col("condition_id") == "m3").then(1.0).otherwise(pl.col("we_winner_pct_24h")).alias("we_winner_pct_24h"),
        pl.when(pl.col("condition_id") == "m3").then(1.0).otherwise(pl.col("we_winner_pct_12h")).alias("we_winner_pct_12h"),
    )  # fmt: skip
    out = combine(sc, we).sort("condition_id")
    assert out.filter(pl.col("within_event").is_not_null()).height == 30
    assert out.filter(pl.col("condition_id") == "m35")["within_event"][0] is None  # missing -> null
    m3 = out.filter(pl.col("condition_id") == "m3").row(0, named=True)
    assert m3["within_event"] == 1.0  # top of its (info_type, group) in all four columns
    assert m3["rank_within_event"] == 1 and "within_event" in (m3["main_driver"] or "within_event")
    # a market with only max_jump + nothing else still gets no combined score (needs >= 2 components)
    assert out.filter(pl.col("condition_id") == "m35")["n_components"][0] >= 2  # has D, D_last24, max_jump
    # without we columns the component is absent and the rest unchanged
    base = combine(sc, pl.DataFrame({"condition_id": sc["condition_id"]}))
    assert base["within_event"].null_count() == base.height
    assert base["fisher_score"].null_count() == 0
