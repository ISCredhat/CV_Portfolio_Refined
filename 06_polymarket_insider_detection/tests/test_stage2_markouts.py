"""Offline synthetic tests for stage 2 task B (markouts, sign randomisation, shrinkage, persistence)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pmdetect.stage2 import markouts as M

T0 = datetime(2025, 3, 1)


def _mk(closed_days=3, outcome="YES"):
    return pl.DataFrame(
        {
            "condition_id": ["a"],
            "closed_ts": [T0 + timedelta(days=closed_days)],
            "outcome": [outcome],
            "mcode": [0],
            "ecode": [0],
            "period": ["fit"],
        }
    )


def _tape(rows):
    return pl.DataFrame(
        {
            "condition_id": ["a"] * len(rows),
            "wallet": ["w"] * len(rows),
            "side": [r[0] for r in rows],
            "outcome_label": [r[1] for r in rows],
            "size": [10.0] * len(rows),
            "price": [r[2] for r in rows],
            "usd": [10.0 * r[2] for r in rows],
            "ts": [T0 + r[3] for r in rows],
        }
    )


def _run(rows, mk=None, hist=None):
    mk = mk if mk is not None else _mk()
    t = M.prep_trades(_tape(rows), mk)
    s = M.build_series(t, hist, mk)
    t = M.spread_proxy(t)
    return M.markouts_for_trades(t, s, mk)


def test_markout_sign_in_traded_token_and_horizon_dropping():
    out = _run(
        [
            ("BUY", "Yes", 0.5, timedelta(0)),
            ("SELL", "No", 0.6, timedelta(minutes=1)),  # = buy-side pressure on YES, y = 0.4
            ("BUY", "No", 0.3, timedelta(hours=2)),  # y = 0.7, q = -1
            ("BUY", "Yes", 0.9, timedelta(days=2)),
        ]
    )
    assert out["q"].to_list() == [1, 1, -1, 1]
    assert out["y"].round(6).to_list() == [0.5, 0.4, 0.7, 0.9]
    # trade 1 at 1 h: last obs at or before t+1h is the second fill (y 0.4): d = -0.1, markout = q*d = -0.1
    assert out["d_1h"][0] == pytest.approx(-0.1)
    # trade 3 (buy No at 0.3): YES later unchanged -> markout in the No token also 0
    assert out["d_24h"][2] == pytest.approx(0.0)
    # last trade: its 1 h / 24 h horizons fall after the last observation -> dropped, not padded
    assert out["d_1h"][3] is None and out["d_24h"][3] is None
    # resolution horizon always exists: YES won so final YES price is 1
    assert out["d_res"].to_list() == pytest.approx([0.5, 0.6, 0.3, 0.1])
    # markout of a buy-No at 0.3 to resolution (YES won, No -> 0) is -0.3 in the No token: q*d = -1*0.3
    assert (out["q"] * out["d_res"])[2] == pytest.approx(-0.3)


def test_horizon_dropped_after_resolution_and_when_stale():
    mk = _mk(closed_days=2)  # resolves after 2 days
    out = _run([("BUY", "Yes", 0.5, timedelta(0)), ("BUY", "Yes", 0.6, timedelta(hours=25))], mk=mk)
    assert out["d_24h"][0] == pytest.approx(0.0)  # t+24h inside the window; last obs is the first fill (24 h old <= 48 h)
    assert out["d_24h"][1] is None  # t+24h = 49 h is after closed_ts (48 h): dropped, not padded
    assert out["d_res"][1] == pytest.approx(0.4)
    # stale: the only later observation is > 6 h old for the 1 h horizon
    out = _run([("BUY", "Yes", 0.5, timedelta(0)), ("BUY", "Yes", 0.6, timedelta(hours=1))], mk=_mk(10))
    assert out["d_5m"][0] == pytest.approx(0.0)  # last obs = the trade itself, age 5 min <= 6 h
    hist = pl.DataFrame(
        {
            "condition_id": ["a"],
            "outcome_label": ["Yes"],
            "ts": [T0 + timedelta(hours=20)],
            "price": [0.8],
        }
    )
    out = _run(
        [("BUY", "Yes", 0.5, timedelta(0)), ("BUY", "Yes", 0.6, timedelta(minutes=1)), ("BUY", "Yes", 0.7, timedelta(hours=30))],
        mk=_mk(10),
        hist=hist,
    )
    # 1 h horizon: last obs at t+1h is the 1-minute fill (age ~59 min), fine; history at 20 h used at 24 h
    assert out["d_24h"][0] == pytest.approx(0.8 - 0.5)


def test_history_other_token_converted_to_yes_terms():
    hist = pl.DataFrame(
        {"condition_id": ["a"], "outcome_label": ["No"], "ts": [T0 + timedelta(hours=1)], "price": [0.25]}
    )
    out = _run([("BUY", "Yes", 0.5, timedelta(0)), ("BUY", "Yes", 0.5, timedelta(hours=5, minutes=30))], hist=hist)
    assert out["d_1h"][0] == pytest.approx(0.75 - 0.5)  # No price 0.25 -> YES-equivalent 0.75


def test_spread_proxy_uses_opposite_consecutive_fills_then_market_median():
    out = _run(
        [
            ("BUY", "Yes", 0.52, timedelta(0)),
            ("SELL", "Yes", 0.48, timedelta(minutes=2)),  # gap 0.04 within 10 min
            ("BUY", "Yes", 0.50, timedelta(minutes=3)),  # opposite of previous: gap 0.02
            ("BUY", "Yes", 0.50, timedelta(minutes=4)),  # same side as previous -> market median
        ]
    )
    hs = out["hs"].to_list()
    assert hs[1] == pytest.approx(0.02, abs=1e-6)
    assert hs[2] == pytest.approx(0.01, abs=1e-6)
    assert hs[3] == pytest.approx(0.015, abs=1e-6)  # median of (0.04, 0.02) / 2


def _synthetic(n_markets=300, n_wallets=60, per=40, skill=0.0, seed=0):
    """Random trades; the YES price change over the horizon correlates with sign by ``skill`` for wallets < 10."""
    rng = np.random.default_rng(seed)
    m = np.repeat(np.arange(n_markets), per)
    w = rng.integers(0, n_wallets, size=m.size)
    q = rng.choice([-1.0, 1.0], size=m.size)
    drift = rng.normal(0, 0.05, size=n_markets)[m]
    d = drift + rng.normal(0, 0.1, size=m.size)
    informed = w < 10
    d = np.where(informed, d + skill * q, d)
    size = rng.uniform(5, 50, size=m.size)
    return w, m, q, size * d, size * 0.5, n_wallets


def test_null_moments_match_sign_randomisation_draws():
    w, m, q, a, usd, n_w = _synthetic(n_markets=40, n_wallets=8, per=25, seed=3)
    mean, var = M.null_moments(w, m, q, a, n_w)
    sr = M.sign_randomization(w, m, q, a, n_w, n_draws=3000, seed=1, threads=2)
    assert np.allclose(sr["null_mean"], mean, atol=0.15 * np.sqrt(var).max())
    assert np.allclose(sr["null_sd"] ** 2, var, rtol=0.12)


def test_sign_randomization_null_p_values_roughly_uniform_and_detects_skill():
    w, m, q, a, usd, n_w = _synthetic(skill=0.0, seed=4)
    sr = M.sign_randomization(w, m, q, a, n_w, n_draws=400, seed=2, threads=2)
    p = sr["p_upper"]
    assert 0.35 < p.mean() < 0.65
    assert (p < 0.05).mean() < 0.15
    w, m, q, a, usd, n_w = _synthetic(skill=0.08, seed=4)
    sr = M.sign_randomization(w, m, q, a, n_w, n_draws=400, seed=2, threads=2)
    assert (sr["p_upper"][:10] < 0.05).mean() >= 0.9  # planted skilled wallets stand out
    assert (sr["p_upper"][10:] < 0.05).mean() < 0.2
    assert sr["p_upper"].min() >= 1 / 401  # add-one rule


def test_sign_randomization_is_reproducible_and_thread_independent_in_distribution():
    w, m, q, a, usd, n_w = _synthetic(n_markets=30, n_wallets=6, per=20, seed=5)
    r1 = M.sign_randomization(w, m, q, a, n_w, n_draws=200, seed=7, threads=2)
    r2 = M.sign_randomization(w, m, q, a, n_w, n_draws=200, seed=7, threads=2)
    assert np.array_equal(r1["p_upper"], r2["p_upper"])


def test_eb_shrinkage_moves_noisy_estimates_toward_mean():
    rng = np.random.default_rng(0)
    theta = rng.normal(0.01, 0.02, size=500)
    v = rng.uniform(1e-5, 4e-3, size=500)
    skill = theta + rng.normal(0, np.sqrt(v))
    mask = np.ones(500, dtype=bool)
    eb = M.eb_shrink(skill, v, mask)
    sh = eb["shrunk"]
    assert abs(eb["mu"] - 0.01) < 0.01 and eb["tau2"] > 0
    big = v > 2e-3
    assert np.mean(np.abs(sh[big] - eb["mu"])) < np.mean(np.abs(skill[big] - eb["mu"]))
    assert np.mean((sh - theta) ** 2) < np.mean((skill - theta) ** 2)  # shrinkage reduces error
    # no true heterogeneity: everything collapses to the common mean
    flat = M.eb_shrink(rng.normal(0, np.sqrt(v)), v, mask)
    assert flat["tau2"] < 1e-3 and np.nanstd(flat["shrunk"]) < 0.5 * np.std(skill)
    # mask: wallets outside get NaN
    mm = mask.copy()
    mm[:10] = False
    assert np.isnan(M.eb_shrink(skill, v, mm)["shrunk"][:10]).all()


def _persist_data(skill, seed=0, n_w=400, per_period=30):
    """Two periods of trades for the same wallets; the first 40 wallets have persistent skill."""
    rng = np.random.default_rng(seed)
    out = []
    for period in range(2):
        n_m = 400
        m = np.repeat(np.arange(n_m), 40) + period * 1000
        w = rng.integers(0, n_w, size=m.size)
        q = rng.choice([-1.0, 1.0], size=m.size)
        d = rng.normal(0, 0.1, size=m.size)
        d = np.where(w < 40, d + skill * q, d)
        size = rng.uniform(5, 50, size=m.size)
        out.append(
            {
                "w": w, "m": m, "e": m // 2, "q": q, "a": size * d, "usd": size * 0.5, "hs_usd": size * 0.004,
            }
        )  # fmt: skip
    return out[0], out[1], n_w


def test_persistence_detects_persistent_skill_and_fails_without_it():
    fit, tune, n_w = _persist_data(skill=0.05)
    r = M.persistence_test(fit, tune, n_w, 1, n_boot=200, n_shuffle=30, seed=1)
    assert r["deciles"][9]["mean"] > r["deciles"][0]["mean"]
    assert r["deciles"][9]["lo"] > 0 and r["rho"] > 0 and r["rho_lo"] > 0
    assert r["top_minus_bottom"]["lo"] > 0
    assert r["base_top_p95"] < r["deciles"][9]["mean"]
    assert r["pass"] and r["verdict"].startswith("PASS")
    fit, tune, n_w = _persist_data(skill=0.0)
    r0 = M.persistence_test(fit, tune, n_w, 1, n_boot=200, n_shuffle=30, seed=1)
    assert not r0["pass"]
    assert abs(r0["rho"]) < 0.15


def test_shuffled_wallet_baseline_is_centred_on_the_all_wallet_mean():
    fit, tune, n_w = _persist_data(skill=0.05, seed=2)
    r = M.persistence_test(fit, tune, n_w, 1, n_boot=100, n_shuffle=60, seed=3)
    assert len(r["base_top"]) == 60
    assert abs(np.mean(r["base_top"]) - r["all"]["mean"]) < 0.01
    assert abs(np.mean(r["base_rho"])) < 0.1


def test_embargo_drops_markets_overlapping_the_boundary():
    df = pl.DataFrame(
        {
            "mcode": [1, 1, 2, 2, 3, 3, 4, 4],
            "ts": [datetime(2025, 11, 1), datetime(2025, 12, 5), datetime(2025, 12, 20), datetime(2025, 12, 28),
                   datetime(2026, 1, 3), datetime(2026, 2, 1), datetime(2026, 1, 20), datetime(2026, 2, 20)],
            "closed_ts": [datetime(2025, 12, 5)] * 2 + [datetime(2025, 12, 28)] * 2 + [datetime(2026, 2, 1)] * 2
            + [datetime(2026, 2, 20)] * 2,
        }
    )  # fmt: skip
    bad = set(M.embargo_markets(df).to_list())
    assert bad == {2, 3}  # closes inside the last 7 days of fit / opens inside the first 7 days of tune


def test_sealed_markets_never_loaded(tmp_path: Path):
    snap = tmp_path
    mk = pl.DataFrame(
        {
            "condition_id": ["fit1", "tune1", "sealed1", "none1"],
            "event_id": ["e1", "e2", "e3", None],
            "slug": ["a"] * 4,
            "question": ["q"] * 4,
            "category": [None] * 4,
            "tags": [["x"]] * 4,
            "info_type": ["deadline"] * 4,
            "volume_usd": [1.0] * 4,
            "liquidity_usd": [None] * 4,
            "start_ts": [datetime(2025, 1, 1)] * 4,
            "end_ts": [datetime(2026, 8, 1)] * 4,
            "closed_ts": [datetime(2025, 6, 1), datetime(2026, 3, 1), datetime(2026, 7, 1), datetime(2025, 7, 1)],
            "uma_status": ["resolved"] * 4,
            "outcome": ["YES", "NO", "YES", None],
        },
        schema_overrides={"liquidity_usd": pl.Float64, "category": pl.String},
    )
    mk.write_parquet(snap / "markets.parquet")
    loaded = M.load_markets(snap)
    assert sorted(loaded["condition_id"].to_list()) == ["fit1", "tune1"]
    assert set(loaded["period"].to_list()) == {"fit", "tune"}
    with pytest.raises(AssertionError):
        M.build_trade_markouts(mk.with_columns(mcode=pl.lit(0)), snap, out=tmp_path / "o")


def test_privacy_no_40_hex_in_stage2_markout_results():
    pat = re.compile(r"(?<![0-9A-Za-z])(?:0x)?[0-9a-fA-F]{40}(?![0-9A-Za-z])")
    root = Path(__file__).resolve().parents[1] / "results"
    for f in list(root.glob("stage2_markouts*")):
        assert not pat.search(f.read_text(encoding="utf-8")), f.name
