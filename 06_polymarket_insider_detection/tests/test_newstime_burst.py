"""Phase 3c Part B tests (offline, synthetic): direction-free burst detector, validation helper,
acceptance criterion, and the wallet pipeline's respect for ``burst_validated``."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from tests.test_phase3_a import T0, _markets, _trades

from pmdetect.markets import newstime_burst as nb
from pmdetect.wallets.positions import build_positions, market_cutoffs, usable_burst

H0 = datetime(2025, 3, 1, 0, 0)


def _path(up: bool, step_bin: int = 60, n: int = 120, lo: float = 0.2, hi: float = 0.8):
    """YES price flat at lo (or hi), stepping to hi (lo) at bin ``step_bin``; trades burst there."""
    a, b = (lo, hi) if up else (hi, lo)
    P = np.where(np.arange(n) < step_bin, a, b).astype(float)
    C = np.ones(n)
    C[step_bin : step_bin + 3] = 40
    return P, C


def _rows(cid: str, P, C, label_flip: bool = False):
    """Hourly rows (taker-fill tape + one price point) for a market; label_flip puts the fills on the
    No token at 1 - price (same YES-price path seen through the other token)."""
    pts = []
    for k, (p, c) in enumerate(zip(P, C, strict=True)):
        for i in range(int(c)):
            ts = H0 + timedelta(hours=k, minutes=i % 50)
            pts.append({"condition_id": cid, "outcome_label": "No" if label_flip else "Yes",
                        "price": (1 - p) if label_flip else p, "ts": ts})  # fmt: skip
    return pl.DataFrame(pts, schema_overrides={"ts": pl.Datetime("us")})


def test_planted_burst_up_and_down_same_time():
    for up in (True, False):
        P, C = _path(up, step_bin=60)
        r = nb.detect_burst(P, C)
        assert r["start_bin"] == 60, r
        assert (r["delta"] > 0) == up and abs(abs(r["delta"]) - 0.6) < 1e-9
        assert r["z"] >= nb.MIN_Z


def test_direction_free_twins_same_time_via_pipeline():
    """A YES-winner market and its NO-winner mirror image (YES price falls, fills seen on the No token)
    give the same burst time: nothing depends on which side won."""
    P, C = _path(True)
    tape = pl.concat([_rows("yes_wins", P, C), _rows("no_wins", 1 - P, C, label_flip=True)])
    prices = tape.head(0)
    pts = nb.yes_points(prices, tape)
    out = nb.detect_bursts(nb.hourly_bins(pts)).sort("condition_id")
    t = dict(zip(out["condition_id"], out["news_ts_burst"], strict=True))
    assert t["yes_wins"] == t["no_wins"] == H0 + timedelta(hours=60)
    s = dict(zip(out["condition_id"], out["burst_size"], strict=True))
    assert s["yes_wins"] > 0 > s["no_wins"]  # YES-price direction, not winner direction


def test_start_is_where_price_leaves_level_not_peak():
    P = np.full(100, 0.3)
    P[50:] = 0.35  # small first step (< 20 % of the final move)
    P[53:] = 0.9  # the real repricing
    C = np.ones(100)
    C[50:56] = 30
    r = nb.detect_burst(P, C)
    assert r["start_bin"] == 53  # first bin covering >= 20 % of the 0.6 move


def test_no_burst_when_price_moves_without_trades_or_trades_without_move():
    P, _ = _path(True)
    quiet = np.ones(len(P))  # a jump with no trade burst
    assert nb.detect_burst(P, quiet)["reason"] == "no_burst"
    P2 = np.full(120, 0.5)
    C2 = np.ones(120)
    C2[60:63] = 40  # a trade burst with no repricing
    assert nb.detect_burst(P2, C2)["reason"] == "no_burst"
    assert nb.detect_burst(np.full(5, 0.5), np.ones(5))["reason"] == "short_history"


def test_largest_move_wins_in_either_direction():
    P = np.full(200, 0.5)
    P[50:] = 0.65  # +0.15 burst
    P[150:] = 0.05  # -0.60 burst (largest, downward)
    C = np.ones(200)
    C[50:53] = 40
    C[150:153] = 40
    r = nb.detect_burst(P, C)
    assert r["start_bin"] == 150 and r["delta"] < 0


def test_validation_errors_and_stats():
    t0 = datetime(2025, 1, 1)
    burst = pl.DataFrame({"condition_id": ["a", "b", "c", "d"],
                          "news_ts_burst": [t0 + timedelta(hours=2), t0 - timedelta(hours=30), None, t0]},
                         schema_overrides={"news_ts_burst": pl.Datetime("us")})  # fmt: skip
    indep = pl.DataFrame({"condition_id": ["a", "b", "c", "d", "e"], "news_ts_indep": [t0] * 5,
                          "independent": [True, True, True, False, True],
                          "indep_method": ["scheduled_end"] * 5},
                         schema_overrides={"news_ts_indep": pl.Datetime("us")})  # fmt: skip
    mk = pl.DataFrame({"condition_id": ["a", "b", "c", "d", "e"], "outcome": ["YES", "NO", "NO", "YES", "OTHER"]})
    v = nb.validation_errors(burst, indep, mk).sort("condition_id")
    assert v["condition_id"].to_list() == ["a", "b", "c"]  # d not independent, e has no YES/NO outcome
    e = v["err_h"].to_list()
    assert e[0] == 2.0 and e[1] == -30.0 and np.isnan(e[2])
    st = nb.error_stats(np.array(e))
    assert st["n"] == 3 and st["n_detected"] == 2
    assert st["share_within_24h"] == 1 / 3 and st["share_within_6h"] == 1 / 3  # nan counts as a miss
    assert st["median_signed_h"] == -14.0


def test_criterion_function():
    n = 100
    rng = np.random.default_rng(0)
    outcome = np.array(["YES", "NO"] * (n // 2))
    good = rng.normal(0, 2, n)  # all within 24 h, median abs about 1.4 h
    r = nb.criterion(good, outcome)
    assert r["burst_validated"] and r["crit_median_ok"] and r["crit_share24_ok"] and r["crit_symmetry_ok"]
    # median too large
    r = nb.criterion(np.full(n, 12.0), outcome)
    assert not r["crit_median_ok"] and not r["burst_validated"]
    # median fine but fewer than 70 % within 24 h (undetected counts as a miss)
    bad = np.where(np.arange(n) < 40, np.nan, 1.0)
    r = nb.criterion(bad, outcome)
    assert not r["crit_share24_ok"] and not r["burst_validated"]
    # asymmetry: YES-winner errors 3x the NO-winner errors (above the 2x allowance)
    err = np.where(outcome == "YES", 6.0, 1.5)
    r = nb.criterion(err, outcome)
    assert r["crit_median_ok"] and r["crit_share24_ok"] and not r["crit_symmetry_ok"]
    assert not r["burst_validated"]
    # NO-winner worse than YES-winner is not what the criterion tests
    err2 = np.where(outcome == "YES", 1.5, 6.0)
    assert nb.criterion(err2, outcome)["crit_symmetry_ok"]


def test_assemble_independent_only_when_validated():
    burst = pl.DataFrame({"condition_id": ["d1", "u1", "s1", "d2"],
                          "news_ts_burst": [H0, H0, H0, None], "burst_size": [0.3] * 4,
                          "burst_trade_z": [5.0] * 4, "burst_n_trades": [20.0] * 4,
                          "burst_reason": [None, None, None, "no_burst"]},
                         schema_overrides={"news_ts_burst": pl.Datetime("us")})  # fmt: skip
    mk = pl.DataFrame({"condition_id": ["d1", "u1", "s1", "d2"],
                       "info_type": ["deadline", "unscheduled_event", "scheduled_reveal", "deadline"]})
    ni = pl.DataFrame({"condition_id": ["d1", "u1", "s1", "d2"], "independent": [False, False, True, False]})
    off = nb.assemble(burst, mk, ni, False)
    assert not off["independent_burst"].any() and not off["burst_validated"].any()
    on = nb.assemble(burst, mk, ni, True).sort("condition_id")
    # only deadline / unscheduled markets WITH a burst; scheduled markets and no-burst markets never
    assert dict(zip(on["condition_id"], on["independent_burst"], strict=True)) == {
        "d1": True, "d2": False, "s1": False, "u1": True}  # fmt: skip
    assert on.filter(pl.col("independent_burst"))["indep_method_burst"].unique().to_list() == ["burst"]
    assert set(on["validation_group"]) == {"deadline", "unscheduled_event", "scheduled_validation"}


# ------------------------------------------------------------------------- wallet pipeline


def _burst_df(validated: bool):
    return pl.DataFrame({"condition_id": ["A", "C"],
                         "news_ts_burst": [T0 + timedelta(hours=4), T0 + timedelta(hours=5)],
                         "burst_validated": [validated] * 2, "independent_burst": [True, True]},
                        schema_overrides={"news_ts_burst": pl.Datetime("us")})  # fmt: skip


def test_pipeline_ignores_unvalidated_burst():
    m = _markets()
    assert usable_burst(_burst_df(False)) is None
    base = market_cutoffs(m, None, None).sort("condition_id")
    unval = market_cutoffs(m, None, None, news_burst=_burst_df(False)).sort("condition_id")
    assert base.equals(unval)
    assert set(unval["cutoff_method"]) == {"none"}


def test_pipeline_uses_validated_burst_with_precedence_and_calibration_flag():
    m = _markets()
    ni = pl.DataFrame({"condition_id": ["A"], "news_ts_indep": [T0 + timedelta(hours=2)], "independent": [True]},
                      schema_overrides={"news_ts_indep": pl.Datetime("us")})  # fmt: skip
    jumps = pl.DataFrame({"condition_id": ["C"], "jump_start_ts": [T0 + timedelta(hours=7)]},
                         schema_overrides={"jump_start_ts": pl.Datetime("us")})  # fmt: skip
    c = market_cutoffs(m, ni, jumps, news_burst=_burst_df(True)).sort("condition_id")
    # A: independent news beats the burst; C: validated burst beats the jump start; B: nothing
    assert c["cutoff_method"].to_list() == ["independent_news", "none", "independent_burst"]
    assert c["cutoff_ts"][2] == T0 + timedelta(hours=5)

    tr = _trades([
        ("w1", "C", "Yes", "BUY", 100, 0.3, 1),
        ("w1", "C", "Yes", "BUY", 900, 0.9, 6),  # after the burst start: never enters
        ("w2", "B", "Yes", "BUY", 100, 0.3, 1),  # no cutoff
    ])  # fmt: skip
    for validated in (False, True):
        pos = build_positions(tr, m, None, None, news_burst=_burst_df(validated)).sort("wallet")
        w1 = pos.filter(pl.col("wallet") == "w1").row(0, named=True)
        if validated:
            assert w1["cutoff_method"] == "independent_burst" and w1["calib_eligible"] is True
            assert w1["entry_price"] == 0.3 and w1["size"] == 100
        else:
            assert w1["cutoff_method"] == "none" and w1["calib_eligible"] is False
    w2 = pos.filter(pl.col("wallet") == "w2").row(0, named=True)
    assert w2["cutoff_method"] == "none" and w2["calib_eligible"] is False
