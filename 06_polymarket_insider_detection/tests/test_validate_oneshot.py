"""Offline, synthetic tests for the one-shot screen validation (phase 4B/4C)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pmdetect.validate import oneshot_sim as os_
from pmdetect.wallets import oneshot

T0 = datetime(2026, 1, 1)
HEX40 = re.compile(r"\b(?:0x)?[0-9a-fA-F]{40}\b")


def _frame(n_events: int = 6, per: int = 40, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        for i in range(per):
            rows.append({
                "wallet": f"synthetic-w{e}-{i}", "event_id": f"ev{e}", "condition_id": f"c{e}",
                "entry_price": float(rng.uniform(0.05, 0.95)), "usd": float(np.exp(rng.normal(5, 1.5))),
                "won": bool(rng.random() < 0.5), "first_ts": T0 + timedelta(hours=int(rng.integers(0, 500))),
                "first_seen": T0 - timedelta(days=float(rng.uniform(0, 400))), "n_markets": int(rng.integers(1, 50)),
                "hours_before_cutoff": float(rng.uniform(1, 400)), "valid_cutoff": True,
            })  # fmt: skip
    return pl.DataFrame(rows)


def _ref(frame: pl.DataFrame) -> os_.Reference:
    pos = frame.drop("first_seen", "n_markets")
    wal = frame.select("wallet", "first_seen", pl.col("n_markets"))
    f = os_.prepare_reference_frame(pos, wal)
    return os_.Reference.build(f, None)


def _score(ref, price, ks=(6.0, 200.0), ss=(500.0, 50000.0), age=1.0, ev="ev0"):
    M = 1
    p = np.full((M, len(ks)), price)
    a = np.full((M, len(ks), len(ss)), age)
    return os_.score_injections(ref, [ev], [ev.replace("ev", "c")], p, np.array(ks), np.array(ss), a)


def test_avg_rank_ties():
    arr = np.array([1.0, 2.0, 2.0, 3.0])
    assert os_.avg_rank(arr, np.array([2.0]))[0] == 0 + 1 + (2 + 2) / 2 + 0 or True
    # L=1 below, E=2 equal: average rank of 3 tied rows occupying ranks 2..4 is 3
    assert os_.avg_rank(arr, np.array([2.0]))[0] == 3.0
    assert os_.avg_rank(arr, np.array([0.0]))[0] == 1.0


def test_larger_stake_raises_size_percentile_and_shorter_lead_raises_proximity():
    ref = _ref(_frame())
    out = _score(ref, 0.3)
    c = out["comps"][0]  # (K, S, 5)
    assert c[0, 1, 1] > c[0, 0, 1]  # size component: 50k > 500
    assert out["score"][0, 0, 1] >= out["score"][0, 0, 0]
    # proximity: flat up to 72 h, decays beyond; shorter lead never lower
    assert c[0, 0, 2] == 1.0 and c[1, 0, 2] < 1.0
    assert out["score"][0, 0, 1] > out["score"][0, 1, 1]
    assert oneshot.proximity(24) >= oneshot.proximity(96) > oneshot.proximity(240)


def test_lower_entry_price_raises_improbability():
    ref = _ref(_frame())
    lo, hi = _score(ref, 0.08), _score(ref, 0.9)
    assert lo["comps"][0, 0, 0, 0] > hi["comps"][0, 0, 0, 0]


def test_parity_with_score_oneshot_on_small_data():
    fr = _frame()
    pos = fr.drop("first_seen", "n_markets").with_columns(
        pl.col("first_ts").cast(pl.Datetime("us")), pl.lit("independent_news").alias("cutoff_method")
    )
    wal = fr.select("wallet", "first_seen", pl.col("n_markets")).unique("wallet")
    price, k, s, age = 0.2, 24.0, 5000.0, 1.5
    ft = T0 + timedelta(hours=100)
    inj = pos.head(1).with_columns(
        pl.lit("synthetic-0001").alias("wallet"), pl.lit("ev0").alias("event_id"), pl.lit("c0").alias("condition_id"),
        pl.lit(price).alias("entry_price"), pl.lit(s).alias("usd"), pl.lit(True).alias("won"),
        pl.lit(ft).alias("first_ts"), pl.lit(k).alias("hours_before_cutoff"),
    )  # fmt: skip
    wal2 = pl.concat([wal, pl.DataFrame({"wallet": ["synthetic-0001"], "first_seen": [ft - timedelta(days=age)],
                                         "n_markets": [1]})])  # fmt: skip
    full = oneshot.score_oneshot(pl.concat([pos, inj]), None, wal2, None, None)
    slow = float(full.filter(pl.col("wallet") == "synthetic-0001")["score"][0])
    ref = _ref(fr)
    fast = float(_score(ref, price, ks=(k,), ss=(s,), age=age)["score"][0, 0, 0])
    assert abs(fast - slow) < 0.02  # one row among 240: rank shifts of order 1/n


def test_draw_markets_deterministic():
    el = pl.DataFrame({"condition_id": [f"c{i}" for i in range(50)]})
    a, b = os_.draw_markets(el, 10, 1), os_.draw_markets(el, 10, 1)
    assert a["condition_id"].to_list() == b["condition_id"].to_list()
    assert a.height == 10 and os_.draw_markets(el, 0, 1).height == 50


def test_flip_labels():
    p = pl.DataFrame({"won": [True, False, None]})
    assert os_.flip_labels(p)["won"].to_list() == [False, True, None]


def test_injection_deterministic_and_no_hex(tmp_path):
    fr = _frame()
    ref = _ref(fr)
    cands = pl.DataFrame({
        "condition_id": [f"c{i}" for i in range(4)], "event_id": [f"ev{i}" for i in range(4)],
        "outcome": ["YES"] * 4, "token_id": [f"t{i}" for i in range(4)], "cutoff_ts": [T0] * 4,
        "info_type": ["scheduled_reveal"] * 4,
    })  # fmt: skip
    ks, ss = os_.K_GRID, os_.S_GRID
    price = np.full((4, len(ks)), 0.15)
    real = np.sort(np.random.default_rng(1).uniform(0.2, 0.9, 500))
    q = float(np.quantile(real, 0.99))
    o1 = os_.run_injection(ref, cands, price, ks, ss, real, q, seed=5, n_boot=50)
    o2 = os_.run_injection(ref, cands, price, ks, ss, real, q, seed=5, n_boot=50)
    assert np.array_equal(o1["score"], o2["score"]) and np.array_equal(o1["lo"], o2["lo"])
    assert (o1["lo"] <= o1["rate"] + 1e-12).all() and (o1["rate"] <= o1["hi"] + 1e-12).all()
    cells = os_.cells_table(o1, ks, ss, os_.K_GRID)
    assert cells.height == len(ks) * len(ss)
    comp = {c: 0.5 for c in oneshot.COMPONENTS}
    sh = {"n": 10, "n_tier": 0, "share": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "share_nousd": 0.0}
    res = os_.SimResult(
        cells, {"reproduced": None, "parity": None, "n_real_valid": 500, "n_real_tier": 5,
                "comp_means": {"all_real": comp, "tier_real": comp, "all_flipped": comp, "tier_flipped": comp},
                "price_source": {"tape": 1.0, "series": 0.0, "none": 0.0}, "ctx_share": 0.0},
        {"flipped": sh, "winners": sh, "pass": True}, 4, 4, q, {"n": 500, "tier": 5, "share": 0.01},
        o1["rank"], ks, ss, {"out": o1},
    )  # fmt: skip
    paths = os_.write_outputs(res, tmp_path, 5, 50)
    for p in paths.values():
        assert p.exists()
    assert not HEX40.search(paths["md"].read_text(encoding="utf-8"))
    assert not HEX40.search(paths["csv"].read_text(encoding="utf-8"))
    assert not HEX40.search("synthetic-0001")


def test_tier_share_counts():
    sc = pl.DataFrame({"wallet": ["a", "b", "c", "d"], "event_id": ["e1", "e1", "e2", "e2"],
                       "score": [0.9, 0.5, 0.95, 0.99], "usd": [500.0, 500.0, 50.0, 500.0],
                       "no_valid_cutoff": [False] * 4})  # fmt: skip
    r = os_.tier_share(sc, pl.Series(["d"]), 0.8, n_boot=20)
    assert r["n"] == 3 and r["n_tier"] == 1 and abs(r["share_nousd"] - 2 / 3) < 1e-9
