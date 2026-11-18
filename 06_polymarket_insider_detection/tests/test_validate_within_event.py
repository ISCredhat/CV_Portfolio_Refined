"""Offline tests for the within-event synthetic-insider simulation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pmdetect.markets.within_event import BUFFER_H
from pmdetect.validate import within_event_sim as sim


def _event(n=8, seed=0, eid="e"):
    rng = np.random.default_rng(seed)
    end = np.datetime64("2026-01-10T00:00:00", "us")
    ts = end - np.arange(100, -1, -1) * np.timedelta64(1, "h")
    base = np.full(n, 1.0 / n)
    series = []
    for i in range(n):
        px = np.clip(base[i] * np.exp(np.cumsum(rng.normal(0, 0.02, len(ts)))), 0.01, 0.9)
        series.append((ts, px))
    cids = [f"c{i}" for i in range(n)]
    evs = [sim.make_sim_event(eid, c, cids, series, end) for c in cids]
    return min(evs, key=lambda e: e.raw_pct[24])  # unremarkable winner (lowest percentile)


def test_huber_recovers_known_impact():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, 400)
    y = 0.15 * x + rng.normal(0, 0.1, 400)
    y[:10] += 5.0  # outliers
    b, se = sim.huber_slope(x, y)
    assert abs(b - 0.15) < 0.02 and se > 0


def test_estimate_impacts_from_tape():
    rng = np.random.default_rng(2)
    t0 = datetime(2026, 1, 1)
    rows = []
    lp = -1.0
    for h in range(300):
        flow = rng.normal(0, 1.5)  # $k
        lp += 0.2 * flow + rng.normal(0, 0.02)
        p = 1 / (1 + np.exp(-lp))
        rows.append(
            dict(condition_id="m", outcome_label="Yes", side="BUY" if flow > 0 else "SELL",
                 price=float(p), usd=abs(flow) * 1000, ts=t0 + timedelta(hours=h, minutes=5))
        )
    tape = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us")))
    imp = sim.estimate_impacts(tape, {"m": datetime(2027, 1, 1)})
    r = imp.row(0, named=True)
    assert r["valid"] and abs(r["beta"] - 0.2) < 0.03
    fb = sim.pooled_fallback(imp, min_n=1)
    assert abs(fb["all"] - r["beta"]) < 1e-12
    assert sim.impact_for("zzz", 0.5, {}, fb)[1] == "fallback"


def test_plant_monotone_in_stake_and_lead_and_blind_before_window():
    ev = _event()
    pct = [sim.planted_percentile(ev, 24, d, 6) for d in (0.0, 0.3, 1.0, 3.0)]
    assert pct == sorted(pct) and pct[-1] > pct[0]
    pl_ = [sim.planted_percentile(ev, 24, 2.0, k) for k in (72, 48, 30, 24, 12, 2)]
    assert pl_[0] == pl_[1] == pl_[2] or pl_[2] <= pl_[3]  # earlier than window start: no help
    assert pl_[3] >= pl_[2] and pl_[4] >= pl_[3] - 1e-12
    assert sim.planted_percentile(ev, 24, 0.0, 6) == ev.raw_pct[24]


def test_siblings_renormalised_and_clip_respected():
    ev = _event()
    p = ev.pre[24]
    lo, hi = sim.CLIP
    t0 = ev.end + np.timedelta64(BUFFER_H * 3600, "s") - np.timedelta64(6 * 3600, "s")
    assert p["end"] >= t0
    # reproduce the repricing used for the end point
    price = p["pe"].copy()
    pw = price[ev.win_idx]
    pwm = min(max(1 / (1 + np.exp(-(np.log(pw / (1 - pw)) + 1.0))), lo), hi)
    s = price.sum()
    f = (s - pwm) / (s - pw)
    new = price * f
    new[ev.win_idx] = pwm
    assert abs(new.sum() - s) < 1e-9 and np.all(np.delete(new, ev.win_idx) < np.delete(price, ev.win_idx))
    raw = sim.plant_moves(ev, 24, 0.0, 6)
    mv = sim.plant_moves(ev, 24, 1.0, 6)
    assert mv[ev.win_idx] > raw[ev.win_idx] - 1e-9  # winner up, siblings relatively down
    # a huge shock stays within the clip bounds in the statistic (finite moves)
    mv = sim.plant_moves(ev, 24, 50.0, 6)
    assert np.all(np.isfinite(mv))


def test_fpr_uses_untouched_events_only():
    evs = [_event(seed=i, eid=str(i)) for i in range(12)]
    raw = np.array([e.raw_pct[24] >= 0.95 for e in evs])
    betas = np.full(len(evs), 0.2)
    t_small = sim.power_table(evs, betas, 24, "all", 1.0, None, 50, 1, stakes=(500,), leads=(2,))
    t_big = sim.power_table(evs, betas, 24, "all", 1.0, None, 50, 1, stakes=(50000,), leads=(6,))
    assert t_small["fpr_untouched"][0] == t_big["fpr_untouched"][0] == raw.mean()


def test_bootstrap_interval_sanity():
    rng = np.random.default_rng(0)
    n = 400
    det = (rng.random((n, 1)) < 0.6)
    elig = np.ones(n, bool)
    fp = rng.random(n) < 0.05
    b = sim.bootstrap_grid(det, elig, fp, n_boot=500, seed=3)
    rate = det.mean()
    assert b["det_lo"][0] < rate < b["det_hi"][0]
    assert 0.04 < b["det_hi"][0] - b["det_lo"][0] < 0.16
    assert b["fpr_lo"] <= fp.mean() <= b["fpr_hi"]
    assert 0.0 <= b["p_meets"][0] <= 1.0
    # more events -> narrower interval
    n2 = 4000
    d2 = rng.random((n2, 1)) < 0.6
    b2 = sim.bootstrap_grid(d2, np.ones(n2, bool), rng.random(n2) < 0.05, n_boot=500, seed=3)
    assert (b2["det_hi"] - b2["det_lo"])[0] < (b["det_hi"] - b["det_lo"])[0]
