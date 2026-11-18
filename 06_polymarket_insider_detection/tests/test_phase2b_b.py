"""Offline tests for phase 2b builder B: jump detection, flow before the move, extras vs pools."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pmdetect.markets import series as S
from pmdetect.markets.changepoint import detect_jump, detect_jumps, price_news_ts
from pmdetect.markets.combine import combine
from pmdetect.markets.flow import flow_before_move
from pmdetect.report.phase2 import _baseline_with_extras

T0 = datetime(2025, 3, 1)


def _ts(hours: np.ndarray) -> np.ndarray:
    return np.datetime64(T0, "us") + (hours * 3_600_000_000).astype("timedelta64[us]")


def test_jump_detected_on_planted_step_with_start_time():
    h = np.arange(0, 200, 0.5)
    q = np.full_like(h, 0.10) + np.random.default_rng(0).normal(0, 0.003, h.size)
    ramp = (h >= 120) & (h < 123)
    q[ramp] = 0.10 + 0.8 * (h[ramp] - 120) / 3
    q[h >= 123] = 0.92
    r = detect_jump(_ts(h), np.clip(q, 0, 1))
    assert r["jump_reason"] is None
    assert 0.7 < r["jump_size"] <= 0.85
    start_h = (r["jump_start_ts"] - np.datetime64(T0, "us")) / np.timedelta64(1, "h")
    assert 119.5 <= start_h <= 121.5  # q first leaves 0.10 at h = 120


def test_jump_null_with_reasons():
    h = np.arange(0, 200, 1.0)
    rng = np.random.default_rng(1)
    flat = np.clip(0.3 + rng.normal(0, 0.004, h.size), 0, 1)
    assert detect_jump(_ts(h), flat)["jump_reason"] == "no_jump"
    slow = np.linspace(0.1, 0.95, h.size)  # drift over 200 h, no single jump
    assert detect_jump(_ts(h), slow)["jump_reason"] in ("no_jump", "gradual_or_multi_step")
    assert detect_jump(_ts(h[:20]), flat[:20])["jump_reason"] == "short_history"
    spike = flat.copy()
    spike[100:103] = 0.8  # spike that reverts
    assert detect_jump(_ts(h), spike)["jump_reason"] == "reverted"
    early = np.where(h < 5, 0.1, 0.9)  # jump right at the start of the tape
    assert detect_jump(_ts(h), early)["jump_reason"] == "jump_at_start"


def test_detect_jumps_on_frames_and_price_news_ts():
    h = np.arange(0, 150, 1.0)
    q = np.where(h < 100, 0.1, 0.95)
    ts = [T0 + timedelta(hours=float(x)) for x in h]
    prices = pl.DataFrame({"condition_id": "m", "outcome_label": "Yes", "ts": ts, "price": q})
    m = pl.DataFrame(
        {
            "condition_id": ["m", "x"],
            "outcome": ["YES", None],
            "closed_ts": [T0 + timedelta(hours=150), None],
        }
    )
    j = detect_jumps(m, prices, None).sort("condition_id")
    assert j.filter(pl.col("condition_id") == "m")["jump_reason"][0] is None
    assert j.filter(pl.col("condition_id") == "x")["jump_reason"][0] == "no_outcome"
    t = price_news_ts(_ts(h), q)
    assert (t - np.datetime64(T0, "us")) / np.timedelta64(1, "h") == 100


def _tape(cid: str, informed: bool, js: datetime, rng) -> pl.DataFrame:
    rows = []
    for i in range(10 * 24 * 3):  # 3 small two-sided trades per hour for 10 days before js
        ts = js - timedelta(hours=10 * 24) + timedelta(minutes=20 * i)
        rows.append((f"w{rng.integers(0, 40)}", ts, "BUY" if rng.random() < 0.5 else "SELL", 20.0))
    if informed:
        for i in range(24):  # a fresh wallet buys the winner every hour in the 24 h before js
            rows.append(("whale", js - timedelta(hours=24) + timedelta(hours=i, minutes=5), "BUY", 900.0))
    return pl.DataFrame(
        {
            "condition_id": cid,
            "tx_hash": [f"{cid}{i}" for i in range(len(rows))],
            "wallet": [r[0] for r in rows],
            "ts": [r[1] for r in rows],
            "side": [r[2] for r in rows],
            "outcome_label": "Yes",
            "price": 0.2,
            "usd": [r[3] for r in rows],
            "size": [r[3] / 0.2 for r in rows],
            "token_id": "t",
            "trade_id": [f"{cid}-{i}" for i in range(len(rows))],
        }
    )


def test_flow_before_move_picks_planted_informed_flow():
    rng = np.random.default_rng(3)
    js = datetime(2025, 3, 20, 12, 0)
    taker = pl.concat([_tape("inf", True, js, rng), _tape("plain", False, js, rng)])
    m = pl.DataFrame({"condition_id": ["inf", "plain"], "outcome": ["YES", "YES"]})
    jumps = pl.DataFrame({"condition_id": ["inf", "plain"], "jump_start_ts": [js, js]})
    wallets = pl.DataFrame(
        {
            "wallet": ["whale"] + [f"w{i}" for i in range(40)],
            "first_seen": [js - timedelta(days=1)] + [js - timedelta(days=400)] * 40,
            "n_markets": [1] + [30] * 40,
        }
    )
    f = flow_before_move(m, jumps, taker, wallets)
    inf = f.filter(pl.col("condition_id") == "inf").row(0, named=True)
    pla = f.filter(pl.col("condition_id") == "plain").row(0, named=True)
    assert inf["fb_reason"] is None and inf["fb_hist_blocks"] >= 3
    assert inf["fb_ofi_z"] > 3 and inf["fb_vol_z"] > 3 and inf["fb_hhi_z"] > 3
    assert inf["fb_fresh_z"] > 3 and inf["fb_fresh"] > 0.8
    assert abs(pla["fb_ofi_z"]) < 3 and abs(pla["fb_vol_z"]) < 3
    nj = pl.DataFrame(
        {"condition_id": ["inf"], "jump_start_ts": [None]},
        schema={"condition_id": pl.String, "jump_start_ts": pl.Datetime("us")},
    )
    out = flow_before_move(m, nj, taker, wallets)
    assert out.filter(pl.col("condition_id") == "inf")["fb_reason"][0] == "no_jump"


def _scores(n=30):
    rng = np.random.default_rng(0)
    ids = [f"m{i}" for i in range(n)]
    return pl.DataFrame(
        {
            "condition_id": ids,
            "info_type": ["scheduled_reveal"] * n,
            "question": ids,
            "D": rng.uniform(0, 0.05, n),
            "D_last24": rng.uniform(0, 0.05, n),
            "max_jump": rng.uniform(0, 0.1, n),
            "out_of_universe": [False] * n,
        }
    )


def test_extra_cases_do_not_enter_pools_or_flags():
    sc = _scores()
    base = combine(sc, pl.DataFrame({"condition_id": sc["condition_id"]}))
    extra = pl.DataFrame(
        {
            "condition_id": ["x"],
            "info_type": ["scheduled_reveal"],
            "question": ["x"],
            "D": [9.0],
            "D_last24": [9.0],
            "max_jump": [0.99],
            "out_of_universe": [True],
        }
    )
    both = combine(
        pl.concat([sc, extra], how="vertical_relaxed"),
        pl.DataFrame({"condition_id": [*sc["condition_id"], "x"]}),
    )
    a = base.sort("condition_id").filter(pl.col("condition_id") != "x")
    b = both.sort("condition_id").filter(pl.col("condition_id") != "x")
    for c in ("D_pct", "D_last24_pct", "max_jump_pct", "fisher_score", "rank_mean", "rank_D"):
        assert a[c].to_list() == b[c].to_list(), c  # pool markets are untouched by the extra
    x = both.filter(pl.col("condition_id") == "x").row(0, named=True)
    assert x["D_pct"] == 1.0 and x["rank_D"] == 1 and x["flag_fisher"]  # would-be flag
    n_pool_flags = both.filter(pl.col("flag_fisher") & ~pl.col("out_of_universe")).height
    assert n_pool_flags == base.filter(pl.col("flag_fisher")).height


def test_extra_independent_market_does_not_change_baseline_fit():
    rng = np.random.default_rng(5)
    news = datetime(2025, 5, 1)
    taus = np.arange(1, 169, dtype=float)
    ids = [f"p{i}" for i in range(12)]
    rows = []
    for cid in [*ids, "x"]:
        q = np.clip(0.2 + 0.001 * (168 - taus) + rng.normal(0, 0.01, 168), 0, 1)
        if cid == "x":
            q = np.clip(q + 0.5, 0, 1)  # extreme extra market
        rows.append(
            pl.DataFrame(
                {
                    "condition_id": cid,
                    "t": [news - timedelta(hours=float(t)) for t in taus],
                    "tau_h": taus,
                    "q": q,
                    "stale_h": 0.0,
                }
            )
        )
    series = pl.concat(rows)
    q0 = S.q0_table(series)
    markets = pl.DataFrame(
        {
            "condition_id": [*ids, "x"],
            "info_type": ["scheduled_reveal"] * 13,
            "independent": [True] * 13,
            "out_of_universe": [False] * 12 + [True],
        }
    )
    with_x = _baseline_with_extras(series, q0, markets, ids, scheme="crossfit")
    without = _baseline_with_extras(
        series.filter(pl.col("condition_id") != "x"), q0, markets.filter(pl.col("condition_id") != "x"), ids,
        scheme="crossfit",
    )
    key = ["condition_id", "t"]
    a = with_x.filter(pl.col("condition_id") != "x").sort(key)
    b = without.sort(key)
    assert np.allclose(a["m50"].to_numpy(), b["m50"].to_numpy())  # the extra never set a baseline
    assert with_x.filter(pl.col("condition_id") == "x").height == 168  # but it gets one
