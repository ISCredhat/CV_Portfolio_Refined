"""Offline synthetic tests for the stage-2 tradable convergence backtest."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from pmdetect.stage2 import common
from pmdetect.stage2 import convergence as cv

T0 = datetime(2025, 6, 10)


def dt(x):
    return np.datetime64(x, "us")


def make_event(wins, n=4, event_id="E1", rates=0.0):
    ms = [
        {"cid": f"{event_id}c{i}", "yes_token": f"tok{i}", "won": wins[i],
         "closed_ts": dt(T0 + timedelta(days=2)), "rate": rates}
        for i in range(n)
    ]
    return {"event_id": event_id, "T": dt(T0), "closed_ts": T0, "period": "fit",
            "n_outcomes": n, "markets": ms}


def make_series(ev, final):
    """Hourly price paths over the week before T; outcome 1 rises, outcome 0 is highest."""
    base = [0.50, 0.20, 0.15, 0.10]
    ts = np.array([dt(T0 - timedelta(hours=h)) for h in range(24 * 8, -1, -1)])
    out = {}
    for i, m in enumerate(ev["markets"]):
        frac = np.linspace(0, 1, len(ts))
        px = base[i] + frac * (final[i] - base[i])
        out[m["cid"]] = (ts, px)
    return out


def make_tapes(ev, px_level, every_min=30):
    tapes = {}
    start = T0 - timedelta(days=9)
    ts = [start + timedelta(minutes=every_min * k) for k in range(int(9 * 24 * 60 / every_min))]
    for i, m in enumerate(ev["markets"]):
        tapes[m["cid"]] = cv.make_tape(ts, [px_level[i]] * len(ts), [100.0] * len(ts))
    return tapes


def test_leader_choice_never_uses_winner():
    for wins in ([True, False, False, False], [False, False, False, True]):
        ev = make_event(wins)
        series = make_series(ev, final=[0.5, 0.45, 0.2, 0.1])
        tapes = make_tapes(ev, [0.5, 0.45, 0.2, 0.1])
        trades, _ = cv.build_trades([ev], tapes, series, delays_min=(5,))
        by = {(t["rule"], t["horizon"]): t["px"] for t in trades}
        # highest price -> outcome 0; log-odds rise -> outcome 1; identical whatever the winner
        assert by[("highest_price", "48h")] == pytest.approx(0.5)
        assert by[("logodds_rise", "48h")] == pytest.approx(0.45)
    c = [{"cid": "a", "move": 1.0, "price": 0.3}, {"cid": "b", "move": 0.2, "price": 0.6}]
    assert cv.pick_leader(c, "logodds_rise") == "a" and cv.pick_leader(c, "highest_price") == "b"
    assert "won" not in c[0]


def test_sealed_period_guard():
    em = pl.DataFrame({
        "event_id": ["a"] * 3 + ["b"] * 3,
        "condition_id": [f"a{i}" for i in range(3)] + [f"b{i}" for i in range(3)],
        "yes_token_id": [f"t{i}" for i in range(6)], "outcome": ["YES", "NO", "NO"] * 2,
        "neg_risk": [True] * 6,
        "closed_ts": [datetime(2025, 5, 1)] * 3 + [datetime(2026, 7, 2)] * 3,
    })
    ni = pl.DataFrame({"condition_id": em["condition_id"], "news_ts_indep": [datetime(2025, 5, 1)] * 6,
                       "independent": [True] * 6})
    mk = pl.DataFrame({"condition_id": em["condition_id"], "tags": [["politics"]] * 6})
    events, ex = cv.build_events(em, ni, mk)
    assert [e["event_id"] for e in events] == ["a"] and ex["sealed_period_dropped"] == 1
    with pytest.raises(AssertionError):
        common.assert_not_sealed(em)


def test_entry_price_delay_and_skip():
    ts = [T0 + timedelta(minutes=m) for m in (0, 3, 7, 40, 900)]
    tape = cv.make_tape(ts, [0.30, 0.31, 0.35, 0.40, 0.90], [10, 10, 10, 10, 10])
    far = dt(T0 + timedelta(days=5))
    e5 = cv.find_entry(tape, None, dt(T0 + timedelta(minutes=5)), far)
    assert e5["px"] == pytest.approx(0.35)  # first trade at or after the delayed time
    assert e5["vol24"] == pytest.approx(20.0)  # the two earlier trades only
    assert cv.find_entry(tape, None, dt(T0 + timedelta(minutes=30)), far)["px"] == pytest.approx(0.40)
    # nothing within 6 h after entry time -> skipped (the 900-minute trade is 14.6 h later)
    assert cv.find_entry(tape, None, dt(T0 + timedelta(minutes=45)), far) == "no_price_within_6h"
    assert cv.find_entry(tape, None, dt(T0 + timedelta(minutes=5)), dt(T0 + timedelta(minutes=6))) == "closed_before_entry"
    # snapshot fallback only when there is no tape
    ser = (np.array([dt(T0 + timedelta(hours=1))]), np.array([0.5]))
    assert cv.find_entry(None, ser, dt(T0), far)["source"] == "snapshot_fallback"
    assert cv.find_entry(None, None, dt(T0), far) == "no_price_data"
    # volume cap: $100 stake, at most 1% of trailing volume, below $1 skipped
    assert cv.stake_for({"vol24": 1e6}) == 100.0
    assert cv.stake_for({"vol24": 5000.0}) == pytest.approx(50.0)
    assert cv.stake_for({"vol24": 50.0}) is None


def test_fee_arithmetic_via_common():
    ev = make_event([True, False, False, False], rates=0.04)
    series = make_series(ev, final=[0.5, 0.2, 0.15, 0.1])
    tapes = make_tapes(ev, [0.5, 0.2, 0.15, 0.1])
    trades, _ = cv.build_trades([ev], tapes, series, delays_min=(5,))
    t = next(t for t in trades if t["rule"] == "highest_price" and t["horizon"] == "48h")
    exp = common.net_return_per_dollar(0.5, True, 0.04)
    assert cv._rets([t], 1.0)[0] == pytest.approx(exp)
    assert exp == pytest.approx((1 - 0.51) / 0.51)
    assert cv._rets([t], 2.0)[0] == pytest.approx((1 - 0.52) / 0.52)  # fee x2
    assert 0 < t["stake"] <= 100.0


def test_bootstrap_resamples_whole_events():
    rows = []
    for e, (won, n) in enumerate([(True, 3), (False, 3), (True, 3), (False, 3)]):
        for _ in range(n):
            rows.append({"event_id": f"e{e}", "px": 0.5, "won": won, "rate": 0.0, "stake": 100.0,
                         "capped": False, "cands": np.zeros((0, 2))})
    a = cv.event_aggregates(rows)
    assert len(a["n"]) == 4 and list(a["n"]) == [3, 3, 3, 3]  # clusters are events, not trades
    pt, draws = cv.bootstrap_ratio(a["pnl"], a["stake"], n_boot=500)
    # per-trade returns are +1 or -1 within an event, so each draw is a mean of whole-event values
    assert set(np.round(draws * 4).astype(int)) <= {-4, -2, 0, 2, 4}
    assert pt == pytest.approx(0.0)
    g = cv.group_stats(rows, with_placebo=False)
    assert g["n_events"] == 4 and g["ret_lo"] < 0 < g["ret_hi"]


def test_placebo_matched_on_price():
    cands = np.array([[100.0, 0.2], [100.0, -0.5]])
    row = {"event_id": "e", "px": 0.5, "won": True, "rate": 0.0, "stake": 100.0, "capped": False, "cands": cands}
    far = dict(row, event_id="f", cands=np.zeros((0, 2)))
    p = cv.placebo_draws([row, far], n_draws=50)
    assert p["n_matched"] == 1  # a trade with no in-band sibling has no placebo
    assert p["placebo_lo"] >= -0.5 and p["placebo_hi"] <= 0.2
    # candidates in build_trades are within 0.05 of the leader's entry price
    ev = make_event([True, False, False, False])
    series = make_series(ev, final=[0.5, 0.47, 0.2, 0.1])
    tapes = make_tapes(ev, [0.5, 0.47, 0.2, 0.1])
    trades, _ = cv.build_trades([ev], tapes, series, delays_min=(5,))
    t = next(t for t in trades if t["rule"] == "highest_price" and t["horizon"] == "48h")
    assert len(t["cands"]) == 1  # only the 0.47 sibling is within 0.05 of 0.5
