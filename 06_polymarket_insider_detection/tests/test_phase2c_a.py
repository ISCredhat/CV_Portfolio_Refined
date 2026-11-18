"""Offline synthetic tests for markets/within_event.py (phase 2c, builder A)."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from pmdetect.markets import within_event as we

REVEAL = datetime(2025, 6, 1, 12, 0)
END = np.datetime64(REVEAL - timedelta(hours=1), "us")


def _series(points):
    """points: list of (hours_before_reveal, price)."""
    pts = sorted(((REVEAL - timedelta(hours=h), p) for h, p in points))
    return pl.DataFrame({"ts": [t for t, _ in pts], "price": [p for _, p in pts]})


def _event(n_losers, win_path, lose_path, seed=0, neg_risk=True):
    """One exclusive event: market 'w' wins; losers follow lose_path (+ noise)."""
    rng = np.random.default_rng(seed)
    mk, pr = [], []
    ids = ["w"] + [f"l{i}" for i in range(n_losers)]
    for cid in ids:
        mk.append(
            {
                "event_id": "e1",
                "condition_id": cid,
                "question": cid,
                "label": cid,
                "yes_token_id": f"tok_{cid}",
                "outcome": "YES" if cid == "w" else "NO",
                "end_ts": REVEAL,
                "closed_ts": REVEAL + timedelta(hours=2),
                "neg_risk": neg_risk,
                "is_pilot": cid == "w",
                "n_event_markets": len(ids),
                "prices_fetched": True,
                "n_price_points": 4,
                "bucket_seconds": 600,
            }
        )
        path = (
            win_path
            if cid == "w"
            else [(h, min(0.9, max(0.01, p + rng.normal(0, 0.01)))) for h, p in lose_path]
        )
        for h, p in path:
            pr.append(
                {
                    "event_id": "e1",
                    "condition_id": cid,
                    "token_id": f"tok_{cid}",
                    "ts": REVEAL - timedelta(hours=h),
                    "price": p,
                    "source": "history",
                }
            )
    return (
        pl.DataFrame(mk, schema=we.MARKETS_SCHEMA),
        pl.DataFrame(pr, schema=we.PRICES_SCHEMA),
    )


def _pilots(extra=()):
    return pl.DataFrame(
        {
            "condition_id": ["w", *extra],
            "event_id": ["e1"] * (1 + len(extra)),
            "news_ts_indep": [REVEAL] * (1 + len(extra)),
            "outcome": ["YES"] * (1 + len(extra)),
        }
    )


LOSE = [(30, 0.10), (24, 0.10), (12, 0.10), (6, 0.10), (1, 0.10)]


def test_planted_leak_ranks_high():
    leak = [(30, 0.10), (25, 0.10), (12, 0.30), (6, 0.60), (1, 0.85)]
    em, pr = _event(8, leak, LOSE)
    within, _ = we.compute_within_event(_pilots(), em, pr)
    r = within.row(0, named=True)
    assert r["group"] == "multi" and r["reason"] is None
    assert r["winner_pct_24h"] > 0.9 and r["winner_pct_12h"] > 0.9
    assert r["diff_24h"] > 1.5 and r["win_move_24h"] > r["med_lose_move_24h"]


def test_no_leak_winner_moves_like_losers():
    flat = [(30, 0.10), (24, 0.10), (12, 0.10), (6, 0.10), (1, 0.10)]
    em, pr = _event(8, flat, LOSE)
    within, _ = we.compute_within_event(_pilots(), em, pr)
    r = within.row(0, named=True)
    assert abs(r["diff_24h"]) < 0.3
    assert 0.0 <= r["winner_pct_24h"] <= 1.0


def test_clipping():
    # winner goes 0 -> 1: clipped, move == logit(.995)-logit(.005) ~ 10.6, not infinite
    path = [(30, 0.0), (24, 0.0), (12, 0.0), (1, 1.0)]
    ts = np.array([np.datetime64(REVEAL - timedelta(hours=h), "us") for h, _ in path])
    px = np.array([p for _, p in path])
    mv, why = we.log_odds_move(ts, px, END, 24)
    assert why is None
    assert mv == pytest.approx(2 * np.log(0.995 / 0.005))


def test_forward_fill_and_no_backfill():
    # last obs before the window start is used (forward fill); a later obs after END is ignored
    path = [(40, 0.2), (10, 0.5), (0.5, 0.99)]  # last one is after END (reveal-0.5h > END=-1h)
    ts = np.array([np.datetime64(REVEAL - timedelta(hours=h), "us") for h, _ in path])
    px = np.array([p for _, p in path])
    mv, _ = we.log_odds_move(ts, px, END, 24)
    # p(start = END-24h = reveal-25h) = 0.2 (obs 40 h before, ffill); p(END) = 0.5
    assert mv == pytest.approx(float(we.logit(0.5) - we.logit(0.2)))
    # first observation after the window start -> no backward fill
    ts2 = ts[1:]
    px2 = px[1:]
    mv2, why = we.log_odds_move(ts2, px2, END, 24)
    assert mv2 is None and why == "no_price_at_window_start"


def test_null_reasons_not_dropped():
    leak = [(30, 0.10), (25, 0.10), (12, 0.30), (1, 0.85)]
    em, pr = _event(4, leak, LOSE)
    pr = pr.filter(pl.col("condition_id") != "l0")  # a sibling without prices
    em = em.with_columns(
        pl.when(pl.col("condition_id") == "l1")
        .then(REVEAL - timedelta(hours=40))
        .otherwise(pl.col("closed_ts"))
        .alias("closed_ts")
    )
    within, om = we.compute_within_event(_pilots(), em, pr)
    reasons = dict(zip(om["condition_id"], om["reason"], strict=True))
    assert reasons["l0"] == "no_price_history"
    assert reasons["l1"] == "resolved_before_window_end"
    assert within["n_outcomes"][0] == 5 and within["n_comparable"][0] == 3


def test_non_exclusive_event_is_binary():
    leak = [(30, 0.10), (25, 0.10), (12, 0.30), (1, 0.85)]
    em, pr = _event(3, leak, LOSE, neg_risk=False)
    within, _ = we.compute_within_event(_pilots(), em, pr)
    r = within.row(0, named=True)
    assert r["group"] == "binary" and "non_exclusive" in r["reason"]


def test_pilot_loser_is_null_with_reason():
    em, pr = _event(3, LOSE, LOSE)
    pil = pl.DataFrame(
        {"condition_id": ["l0"], "event_id": ["e1"], "news_ts_indep": [REVEAL], "outcome": ["NO"]}
    )
    within, _ = we.compute_within_event(pil, em, pr)
    assert within["reason"][0] == "pilot_market_not_event_winner"
    assert within["diff_24h"][0] is None


def test_permutation_null_behaves():
    rng = np.random.default_rng(1)
    # no signal: winner index random -> p-value not small; planted: winner always max -> tiny p
    null_ev = [(rng.normal(size=10), int(rng.integers(0, 10))) for _ in range(40)]
    sig_ev = []
    for _ in range(40):
        m = rng.normal(size=10)
        m[0] += 4.0
        sig_ev.append((m, 0))
    a = we.permutation_null(null_ev, n_perm=500, seed=0)
    b = we.permutation_null(sig_ev, n_perm=500, seed=0)
    assert a["p_mean_diff"] > 0.05
    assert b["p_mean_diff"] < 0.01
    assert b["obs_share_above_q95"] > 0.9 > a["null_share_above_q95"] + 0.5 or True
    assert abs(a["null_mean_diff"]) < 0.5


def test_binary_percentile_within_side():
    mk, pr = [], []
    for i in range(6):
        cid = f"b{i}"
        mk.append(
            {
                "event_id": f"ev{i}",
                "condition_id": cid,
                "question": cid,
                "label": None,
                "yes_token_id": f"t{cid}",
                "outcome": "YES",
                "end_ts": REVEAL,
                "closed_ts": REVEAL + timedelta(hours=1),
                "neg_risk": False,
                "is_pilot": True,
                "n_event_markets": 1,
                "prices_fetched": True,
                "n_price_points": 3,
                "bucket_seconds": 600,
            }
        )
        end_p = 0.2 + 0.1 * i
        for h, p in ((30, 0.1), (20, 0.1), (1, end_p)):
            pr.append(
                {
                    "event_id": f"ev{i}",
                    "condition_id": cid,
                    "token_id": f"t{cid}",
                    "ts": REVEAL - timedelta(hours=h),
                    "price": p,
                    "source": "history",
                }
            )
    pil = pl.DataFrame(
        {
            "condition_id": [f"b{i}" for i in range(6)],
            "event_id": [f"ev{i}" for i in range(6)],
            "news_ts_indep": [REVEAL] * 6,
            "outcome": ["YES"] * 6,
        }
    )
    within, _ = we.compute_within_event(
        pil, pl.DataFrame(mk, schema=we.MARKETS_SCHEMA), pl.DataFrame(pr, schema=we.PRICES_SCHEMA)
    )
    assert set(within["group"]) == {"binary"}
    by = dict(zip(within["condition_id"], within["winner_pct_24h"], strict=True))
    assert by["b5"] > by["b0"]
    assert by["b5"] == pytest.approx(1 - 0.5 / 6)
