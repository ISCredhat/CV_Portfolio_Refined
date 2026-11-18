"""Offline synthetic tests for stage 2 task C (forecast benchmark): no look-ahead, embargo, bootstrap, guards."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pmdetect.stage2 import forecast as F

T = datetime(2026, 2, 10, 12, 0)


def _tr(n=60, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    ts = [T - timedelta(hours=float(h)) for h in sorted(rng.uniform(0.1, 200, n), reverse=True)]
    df = pl.DataFrame(
        {
            "mcode": [0] * n,
            "wcode": rng.integers(0, 8, n).tolist(),
            "ts": ts,
            "q": rng.choice([-1.0, 1.0], n).tolist(),
            "usd": rng.uniform(5, 100, n).tolist(),
            "y": np.clip(0.5 + np.cumsum(rng.normal(0, 0.01, n)), 0.05, 0.95).tolist(),
            "first_ts": [T - timedelta(days=30)] * n,
            "sk": pl.Series(rng.integers(0, 2, n).tolist(), dtype=pl.Int8),
        }
    )
    if extra is not None:
        df = pl.concat([df, extra]).sort("ts")
    return df


def _hist():
    return pl.DataFrame({"mcode": [0, 0], "ots": [T - timedelta(days=9), T - timedelta(days=3)], "y": [0.4, 0.45]})


SNAP = pl.DataFrame({"mcode": [0], "H": ["1d"], "t": [T]})


def test_features_use_nothing_at_or_after_t():
    base = F.snapshot_features(_tr(), _hist(), SNAP)
    # add extreme fills at t and after t, and price history at/after t: nothing may change
    late = pl.DataFrame(
        {
            "mcode": [0, 0, 0],
            "wcode": [1, 2, 3],
            "ts": [T, T + timedelta(minutes=1), T + timedelta(days=2)],
            "q": [1.0, 1.0, -1.0],
            "usd": [1e6, 1e6, 1e6],
            "y": [0.99, 0.01, 0.99],
            "first_ts": [T - timedelta(days=30)] * 3,
            "sk": pl.Series([1, 1, 1], dtype=pl.Int8),
        }
    )
    hist2 = pl.concat([_hist(), pl.DataFrame({"mcode": [0, 0], "ots": [T, T + timedelta(hours=1)], "y": [0.99, 0.01]})])
    pert = F.snapshot_features(_tr(extra=late), hist2, SNAP)
    assert base.height == 1 and pert.height == 1
    for c in base.columns:
        a, b = base[c][0], pert[c][0]
        if isinstance(a, float):
            assert (a != a and b != b) or a == pytest.approx(b), c
        else:
            assert a == b, c
    # and the feature set responds to data strictly before t
    earlier = F.snapshot_features(_tr(seed=1), _hist(), SNAP)
    assert earlier["price"][0] != base["price"][0]


def test_features_expected_values_on_a_tiny_example():
    tr = pl.DataFrame(
        {
            "mcode": [0] * 4,
            "wcode": [1, 2, 1, 3],
            "ts": [T - timedelta(hours=30), T - timedelta(hours=5), T - timedelta(hours=4), T - timedelta(hours=1)],
            "q": [1.0, 1.0, -1.0, 1.0],
            "usd": [100.0, 50.0, 50.0, 100.0],
            "y": [0.5, 0.52, 0.5, 0.6],
            "first_ts": [T - timedelta(days=30), T - timedelta(days=2), T - timedelta(days=30), T - timedelta(days=30)],
            "sk": pl.Series([1, 0, 1, 0], dtype=pl.Int8),
        }
    )
    r = F.snapshot_features(tr, pl.DataFrame(schema={"mcode": pl.Int64, "ots": pl.Datetime("us"), "y": pl.Float64}), SNAP)
    assert r["price"][0] == pytest.approx(0.6)
    # 24 h window: trades 2-4: signed usd = 50 - 50 + 100 = 100 over 200
    assert r["ofi_24h"][0] == pytest.approx(0.5)
    assert r["fresh_share"][0] == pytest.approx(50 / 200)  # wallet 2 is 2 days old at its trade
    assert r["skill_flow_all"][0] == pytest.approx((100 - 50) / 300)
    assert r["skill_flow_24h"][0] == pytest.approx(-50 / 200)
    # wallets pushing YES up: w1 100, w2 50, w3 100 -> shares .4,.2,.4 -> HHI .36
    assert r["hhi_yes"][0] == pytest.approx(0.36)
    assert r["dp_24h"][0] == pytest.approx(F.logit(0.6) - F.logit(0.5))
    assert r["log_vol"][0] == pytest.approx(np.log(301))


def test_walk_forward_embargo_and_month_masks():
    feat = pl.DataFrame(
        {
            "t": [datetime(2025, 12, 1), datetime(2026, 1, 20), datetime(2026, 1, 25), datetime(2026, 2, 3), datetime(2026, 2, 12)],
            "end_ts": [datetime(2025, 12, 8), datetime(2026, 1, 27), datetime(2026, 2, 1), datetime(2026, 2, 10), datetime(2026, 2, 19)],
            "closed_ts": [datetime(2025, 12, 5), datetime(2026, 1, 26), datetime(2026, 2, 1, 6), datetime(2026, 2, 8), datetime(2026, 2, 19)],
        }
    )
    tr, te = F.split_masks(feat, datetime(2026, 2, 1), datetime(2026, 3, 1))
    # closed 2026-01-26 is within 7 days of Feb 1: purged; Dec 5 is fine
    assert tr.tolist() == [True, False, False, False, False]
    assert te.tolist() == [False, False, True, True, True]
    assert not (tr & te).any()
    # test rows need t >= 2026-01-01 and a tune-period close
    f2 = feat.with_columns(pl.Series("t", [datetime(2025, 12, 28)] * 5), pl.Series("end_ts", [datetime(2026, 2, 3)] * 5))
    assert not F.split_masks(f2, datetime(2026, 2, 1), datetime(2026, 3, 1))[1].any()
    assert F.train_cutoff(datetime(2026, 2, 1)) == datetime(2026, 1, 25)


def test_improvement_bootstrap_by_event():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n).astype(float)
    p_price = np.full(n, 0.5)
    good = np.where(y == 1, 0.7, 0.3)
    ev = np.arange(n)
    r = F.improvement_ci(y, p_price, good, ev, n_boot=300, seed=1)
    assert r["dll"] > 0 and r["dll_lo"] > 0 and r["dbr"] > 0
    assert r["logloss_model"] < r["logloss_price"]
    same = F.improvement_ci(y, p_price, p_price.copy(), ev, n_boot=200, seed=1)
    assert same["dll"] == pytest.approx(0) and same["dll_lo"] == pytest.approx(0) and same["dll_hi"] == pytest.approx(0)
    # clustering widens the interval: 40 events of 50 identical outcomes
    yc = np.repeat(rng.integers(0, 2, 40), 50).astype(float)
    pc = np.full(yc.size, 0.5)
    mc = np.repeat(np.where(rng.random(40) < 0.5, 0.6, 0.4), 50)  # model right or wrong per event
    w_cl = F.improvement_ci(yc, pc, mc, np.repeat(np.arange(40), 50), n_boot=400, seed=2)
    w_iid = F.improvement_ci(yc, pc, mc, np.arange(yc.size), n_boot=400, seed=2)
    assert (w_cl["dll_hi"] - w_cl["dll_lo"]) > (w_iid["dll_hi"] - w_iid["dll_lo"])


def test_losses_are_clipped_and_correct():
    ll, br = F.losses(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert np.isfinite(ll).all() and ll.max() < 0.01
    ll, br = F.losses(np.array([1.0]), np.array([0.25]))
    assert ll[0] == pytest.approx(-np.log(0.25)) and br[0] == pytest.approx(0.5625)


def test_sealed_markets_cannot_enter_candidates():
    mk = pl.DataFrame(
        {"mcode": [0], "end_ts": [datetime(2026, 8, 1)], "closed_ts": [datetime(2026, 7, 2)], "start_ts": [datetime(2026, 1, 1)]}
    )
    with pytest.raises(AssertionError):
        F.candidate_snapshots(mk)
    ok = mk.with_columns(closed_ts=pl.lit(datetime(2026, 5, 18)), end_ts=pl.lit(datetime(2026, 5, 20)))
    s = F.candidate_snapshots(ok)
    assert set(s["H"].to_list()) == {"7d", "3d"}  # closed_ts must exceed t = end - H: 1 d snapshot is after the close
    with pytest.raises(AssertionError):
        F.assert_not_sealed(pl.DataFrame({"closed_ts": [datetime(2026, 7, 1)]}))


def test_event_fold_is_deterministic_and_balanced():
    e = np.arange(5000)
    f = F.event_fold(e)
    assert set(np.unique(f)) == set(range(F.K_FOLDS))
    assert np.array_equal(f, F.event_fold(e))
    assert np.bincount(f.astype(int)).min() > 700


def test_classify_skilled_finds_planted_wallets():
    rng = np.random.default_rng(1)
    n_m, per, n_w = 600, 30, 80
    m = np.repeat(np.arange(n_m), per)
    w = rng.integers(0, n_w, m.size)
    q = rng.choice([-1.0, 1.0], m.size)
    d = rng.normal(0, 0.1, m.size) + np.where(w < 8, 0.06 * q, 0.0)
    size = rng.uniform(5, 50, m.size)
    d_ = {"w": w, "m": m, "e": m, "q": q, "a": size * d, "usd": size * 0.5, "hs_usd": size * 0.004}
    sk = F.classify_skilled(d_, n_w)
    assert sk[:8].mean() >= 0.75
    assert sk[8:].mean() < 0.1


def test_fit_predict_models_run_and_recalibrate():
    rng = np.random.default_rng(2)
    lp = rng.normal(0, 1.5, 3000)
    p_true = 1 / (1 + np.exp(-(0.6 * lp)))  # prices over-confident: slope 0.6
    y = (rng.random(3000) < p_true).astype(int)
    x = np.column_stack([lp, rng.normal(size=3000)])
    pr = F.fit_predict("recal", ["logit_p"], x[:2000], y[:2000], x[2000:])
    base, _ = F.losses(y[2000:].astype(float), 1 / (1 + np.exp(-lp[2000:])))
    rec, _ = F.losses(y[2000:].astype(float), pr)
    assert rec.mean() < base.mean()
    ph = F.fit_predict("hgb", ["logit_p", "noise"], x[:2000], y[:2000], x[2000:])
    assert ph.shape == (1000,) and ((ph > 0) & (ph < 1)).all()
    # one-class training set falls back to the price itself
    pr1 = F.fit_predict("hgb", ["logit_p"], x[:50, :1], np.ones(50, int), x[50:60, :1])
    assert np.allclose(pr1, 1 / (1 + np.exp(-x[50:60, 0])))


def test_pass_rule_needs_both_halves_with_lower_bound_above_zero():
    def row(h, mod, s, dll, lo):
        return {"H": h, "model": mod, "sample": s, "dll": dll, "dll_lo": lo}

    rows = [row("7d", "hgb_all", "JanMar", 0.01, 0.002), row("7d", "hgb_all", "AprJun", 0.01, -0.001)]
    rows += [row("3d", "hgb_all", "JanMar", 0.01, 0.002), row("3d", "hgb_all", "AprJun", 0.02, 0.004)]
    out = F.apply_pass_rule(rows, ["7d", "3d", "1d"])
    assert out[("7d", "hgb_all")] is False and out[("3d", "hgb_all")] is True and out[("1d", "hgb_all")] is False


def test_privacy_no_40_hex_in_stage2_forecast_results():
    pat = re.compile(r"(?<![0-9A-Za-z])(?:0x)?[0-9a-fA-F]{40}(?![0-9A-Za-z])")
    root = Path(__file__).resolve().parents[1] / "results"
    for f in root.glob("stage2_forecast*"):
        assert not pat.search(f.read_text(encoding="utf-8")), f.name
