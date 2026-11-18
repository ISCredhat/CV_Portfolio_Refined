"""Phase 3A tests (offline, synthetic): positions, Poisson-binomial, calibration, FDR."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from pmdetect.wallets.calibration import fit_isotonic, poisson_binomial_sf, wallet_tests
from pmdetect.wallets.fdr import apply_fdr, benjamini_hochberg
from pmdetect.wallets.positions import build_positions, market_cutoffs

T0 = datetime(2025, 1, 1, 12, 0)


def _trades(rows):
    return pl.DataFrame(
        [
            {"trade_id": str(i), "tx_hash": str(i), "condition_id": c, "token_id": f"{c}-{lab}",
             "wallet": w, "side": s, "outcome_label": lab, "size": sz, "price": px, "usd": sz * px,
             "ts": T0 + timedelta(hours=h)}
            for i, (w, c, lab, s, sz, px, h) in enumerate(rows)
        ],
        schema_overrides={"ts": pl.Datetime("us")},
    )  # fmt: skip


def _markets():
    return pl.DataFrame(
        {
            "condition_id": ["A", "B", "C"],
            "event_id": ["E1", "E1", "E2"],
            "outcome": ["NO", "YES", "YES"],
            "info_type": ["deadline", "deadline", "scheduled_reveal"],
            "news_ts": [T0 + timedelta(hours=10)] * 3,
        },
        schema_overrides={"news_ts": pl.Datetime("us")},
    )


def test_poisson_binomial_matches_brute_force():
    rng = np.random.default_rng(1)
    for n in (1, 3, 6, 9):
        p = rng.uniform(0.02, 0.98, n)
        for k in range(0, n + 2):
            brute = 0.0
            for bits in itertools.product([0, 1], repeat=n):
                if sum(bits) >= k:
                    pr = 1.0
                    for b, pi in zip(bits, p, strict=True):
                        pr *= pi if b else 1 - pi
                    brute += pr
            assert poisson_binomial_sf(k, p) == pytest.approx(brute, abs=1e-12)


def test_cutoff_priority():
    m = _markets()
    ni = pl.DataFrame({"condition_id": ["A", "B", "C"],
                       "news_ts_indep": [T0 + timedelta(hours=5), T0 + timedelta(hours=6), None],
                       "independent": [True, False, False]},
                      schema_overrides={"news_ts_indep": pl.Datetime("us")})  # fmt: skip
    jumps = pl.DataFrame({"condition_id": ["B", "C"], "jump_start_ts": [T0 + timedelta(hours=7), None]},
                         schema_overrides={"jump_start_ts": pl.Datetime("us")})  # fmt: skip
    c = market_cutoffs(m, ni, jumps).sort("condition_id")
    assert c["cutoff_method"].to_list() == ["independent_news", "jump_start", "none"]
    assert c["cutoff_ts"].to_list() == [T0 + timedelta(hours=5), T0 + timedelta(hours=7), None]
    assert c["legacy_cutoff_method"].to_list() == ["independent_news", "jump_start", "news_minus_buffer"]
    assert c["legacy_cutoff_ts"][2] == T0 + timedelta(hours=9)


def test_taker_netting_and_cutoff():
    # market C (cutoff = news - 1h = T0+9h): buys 100@0.2 and 100@0.4, sells 50, late buy ignored
    tr = _trades([
        ("w1", "C", "Yes", "BUY", 100, 0.2, 1),
        ("w1", "C", "Yes", "BUY", 100, 0.4, 2),
        ("w1", "C", "Yes", "SELL", 50, 0.9, 3),
        ("w1", "C", "Yes", "BUY", 1000, 0.95, 9.5),  # after cutoff
        ("w2", "C", "No", "SELL", 30, 0.5, 1),  # sell with no inventory -> no position
    ])  # fmt: skip
    pos = build_positions(tr, _markets(), None, None)
    assert pos.height == 1
    r = pos.row(0, named=True)
    assert r["wallet"] == "w1" and r["token_id"] == "C-Yes"
    assert r["size"] == pytest.approx(150)
    assert r["entry_price"] == pytest.approx((20 + 40) / 200)  # VWAP of buys only
    assert r["usd"] == pytest.approx(150 * 0.3)
    assert r["won"] is True and r["cutoff_method"] == "none"
    assert r["valid_cutoff"] is False and r["cutoff_ts"] is None and r["hours_before_cutoff"] is None
    assert r["legacy_cutoff_method"] == "news_minus_buffer"
    assert r["hours_before_legacy_cutoff"] == pytest.approx(8.0)


def test_event_collapse_keeps_largest_token_once():
    # E1 = markets A (outcome NO) and B (outcome YES): wallet buys Yes in A ($10) and Yes in B ($50)
    tr = _trades([
        ("w1", "A", "Yes", "BUY", 100, 0.1, 1),   # $10, loses (A outcome NO)
        ("w1", "B", "Yes", "BUY", 100, 0.5, 2),   # $50, wins
        ("w1", "C", "Yes", "BUY", 10, 0.5, 2),    # other event
    ])  # fmt: skip
    pos = build_positions(tr, _markets(), None, None).sort("event_id")
    assert pos["event_id"].to_list() == ["E1", "E2"]  # one row per event
    e1 = pos.row(0, named=True)
    assert e1["condition_id"] == "B" and e1["won"] is True and e1["n_tokens"] == 2
    assert e1["usd_all"] == pytest.approx(60.0) and e1["entry_price"] == pytest.approx(0.5)


def test_unresolved_market_has_null_won():
    m = _markets().with_columns(pl.when(pl.col("condition_id") == "C").then(None).otherwise(pl.col("outcome")).alias("outcome"))  # fmt: skip
    pos = build_positions(_trades([("w1", "C", "Yes", "BUY", 10, 0.5, 1)]), m, None, None)
    assert pos["won"].to_list() == [None]


def _synthetic_positions(seed=0, n_random=300, per=6):
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for w in range(n_random):
        for _ in range(per):
            e = float(rng.uniform(0.05, 0.95))
            rows.append({"wallet": f"r{w}", "event_id": f"ev{k}", "entry_price": e, "size": 100.0,
                         "won": bool(rng.random() < e), "info_type": "deadline" if k % 2 else "unscheduled_event"})  # fmt: skip
            k += 1
    for _j in range(12):  # planted wallet: always wins long shots (price 0.1)
        rows.append({"wallet": "planted", "event_id": f"ev{k}", "entry_price": 0.1, "size": 100.0,
                     "won": True, "info_type": "deadline"})  # fmt: skip
        k += 1
    # background of ordinary 0.1 entries so the 0.1 price bin is calibrated (about 10% win)
    for _ in range(600):
        rows.append({"wallet": f"bg{k}", "event_id": f"ev{k}", "entry_price": 0.1, "size": 10.0,
                     "won": bool(rng.random() < 0.1), "info_type": "deadline"})  # fmt: skip
        k += 1
    return pl.DataFrame(rows)


def test_planted_wallet_passes_fdr_random_does_not():
    pos = _synthetic_positions()
    tests = wallet_tests(pos, write=False)
    res = apply_fdr(tests, min_positions=5, alpha=0.05)
    planted = res.filter(pl.col("wallet") == "planted").row(0, named=True)
    assert planted["n_pos"] == 12 and planted["wins"] == 12
    assert planted["flag"] is True and planted["p_pb"] < 1e-6
    assert planted["excess_pnl"] > 0
    flagged_random = res.filter(pl.col("flag") & pl.col("wallet").str.starts_with("r"))
    assert flagged_random.height == 0
    # background wallets with <5 positions are never eligible
    assert not res.filter(pl.col("wallet").str.starts_with("bg"))["eligible"].any()


def test_cross_fit_uses_other_folds_and_clips():
    pos = _synthetic_positions()
    d, curve, obs = fit_isotonic(pos, n_splits=5, seed=0)
    assert d["p_tilde"].min() >= 0.01 and d["p_tilde"].max() <= 0.99
    assert d.height == pos.height and curve.height > 0
    # calibrated: mean p~ close to the win rate
    assert abs(d["p_tilde"].mean() - d["y"].mean()) < 0.03


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    rej = benjamini_hochberg(p, 0.05)
    assert rej.tolist() == [True, True, False, False, False, False, False, False, False]
    assert not benjamini_hochberg(np.array([0.2, 0.5, 0.9]), 0.05).any()
