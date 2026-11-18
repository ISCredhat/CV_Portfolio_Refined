"""Phase 3b Builder B tests: proximity, wallet age at entry, no_valid_cutoff, first_seen top-up, privacy."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl

from pmdetect import watchlist as wl
from pmdetect.wallets import oneshot

T0 = datetime(2025, 6, 1, 12, 0)


def addr(i: int) -> str:
    return "0x" + f"{i:040x}"


def _pos(n=12, **over):
    rows = []
    for i in range(n):
        r = {
            "wallet": addr(200 + i), "event_id": "e1", "condition_id": "c1", "entry_price": 0.2 + 0.02 * i,
            "usd": 100.0 + 10 * i, "won": True, "first_ts": T0, "hours_before_cutoff": 10.0 + i,
            "cutoff_method": "independent_news", "valid_cutoff": True, "info_type": "deadline",
        }  # fmt: skip
        r.update(over)
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None)


def _wal(ws, first_seen=datetime(2025, 1, 1)):
    return pl.DataFrame(
        {"wallet": ws, "first_seen": [first_seen] * len(ws), "n_markets": [3] * len(ws)}
    ).with_columns(pl.col("n_markets").cast(pl.Int64))


def _score_h(h: float) -> float:
    base = _pos()
    tgt = base["wallet"][5]
    b = base.with_columns(
        pl.when(pl.col("wallet") == tgt).then(h).otherwise(pl.col("hours_before_cutoff")).alias("hours_before_cutoff")
    )
    out = oneshot.score_oneshot(b, None, _wal(base["wallet"].to_list()), None, None)
    return out.filter(pl.col("wallet") == tgt)["c_early"][0]


def test_proximity_flat_then_monotone_decay():
    assert oneshot.proximity(0.5) == 1.0 and oneshot.proximity(72.0) == 1.0
    assert math.isclose(oneshot.proximity(144.0), math.exp(-1.0))
    hs = [1, 24, 72, 73, 100, 200, 500, 2000]
    vals = [oneshot.proximity(h) for h in hs]
    assert vals[0] == vals[1] == vals[2] == 1.0
    assert all(a > b for a, b in zip(vals[2:], vals[3:], strict=False))
    # same function in the scorer, independent of the other positions of the market
    assert _score_h(30.0) == 1.0 and _score_h(72.0) == 1.0
    assert math.isclose(_score_h(144.0), math.exp(-1.0))
    assert _score_h(300.0) < _score_h(144.0) < _score_h(30.0)


def test_wallet_age_at_entry_older_scores_lower():
    base = _pos()
    ws = base["wallet"].to_list()
    tgt = ws[5]
    young = _wal(ws).with_columns(
        pl.when(pl.col("wallet") == tgt).then(T0 - timedelta(hours=2)).otherwise(pl.col("first_seen")).alias("first_seen")
    )
    old = _wal(ws).with_columns(
        pl.when(pl.col("wallet") == tgt).then(datetime(2020, 1, 1)).otherwise(pl.col("first_seen")).alias("first_seen")
    )
    ry = oneshot.score_oneshot(base, None, young, None, None).filter(pl.col("wallet") == tgt).row(0, named=True)
    ro = oneshot.score_oneshot(base, None, old, None, None).filter(pl.col("wallet") == tgt).row(0, named=True)
    assert ry["c_fresh"] > ro["c_fresh"] and ry["score"] > ro["score"]
    assert math.isclose(ry["wallet_age_days_at_entry"], 2 / 24, rel_tol=1e-6)
    # age is measured at entry: a first_seen after the entry clips to age 0
    fut = _wal(ws).with_columns(pl.lit(T0 + timedelta(days=3)).alias("first_seen"))
    assert (oneshot.score_oneshot(base, None, fut, None, None)["wallet_age_days_at_entry"] == 0).all()


def test_missing_first_seen_counted():
    base = _pos()
    ws = base["wallet"].to_list()
    out = oneshot.score_oneshot(base, None, _wal(ws[:6]), None, None)
    assert (out.filter(pl.col("wallet").is_in(ws[6:]))["n_fresh_missing"] >= 1).all()
    assert (out.filter(pl.col("wallet").is_in(ws[:6]))["n_fresh_missing"] == 0).all()


def test_no_valid_cutoff_marked_no_proximity_credit():
    base = _pos()
    nv = base["wallet"][3]
    p = base.with_columns(
        pl.when(pl.col("wallet") == nv).then(False).otherwise(pl.col("valid_cutoff")).alias("valid_cutoff"),
        pl.when(pl.col("wallet") == nv).then(pl.lit("none")).otherwise(pl.col("cutoff_method")).alias("cutoff_method"),
        pl.when(pl.col("wallet") == nv).then(None).otherwise(pl.col("hours_before_cutoff")).alias("hours_before_cutoff"),
    )
    out = oneshot.score_oneshot(p, None, _wal(base["wallet"].to_list()), None, None)
    r = out.filter(pl.col("wallet") == nv).row(0, named=True)
    assert r["no_valid_cutoff"] and r["c_early"] == 0.0 and r["score"] == 0.0 and r["score_screen"] > 0
    assert out.filter(pl.col("wallet") != nv)["no_valid_cutoff"].sum() == 0
    assert out["wallet"][-1] == nv  # ranked below every valid position, still present (screened)
    # legacy frame without valid_cutoff: cutoff_method decides (news_minus_buffer is not valid)
    legacy = base.drop("valid_cutoff").with_columns(
        pl.when(pl.col("wallet") == nv).then(pl.lit("news_minus_buffer")).otherwise(pl.col("cutoff_method"))
        .alias("cutoff_method")
    )
    o2 = oneshot.score_oneshot(legacy, None, None, None, None)
    assert o2.filter(pl.col("wallet") == nv)["no_valid_cutoff"][0]


def test_watchlist_no_valid_cutoff_never_one_shot(tmp_path):
    ws = [addr(300 + i) for i in range(120)]
    scored = pl.DataFrame(
        {
            "wallet": ws, "score": [0.0] * 5 + [0.01 + 0.001 * i for i in range(115)],
            "score_screen": [0.99 - 0.001 * i for i in range(120)],
            "no_valid_cutoff": [True] * 5 + [False] * 115, "usd": [500.0] * 120,
            "entry_price": [0.1] * 120, "hours_before_cutoff": [20.0] * 120, "condition_id": ["c"] * 120,
        }
    )  # fmt: skip
    tests = pl.DataFrame({"wallet": ws, "p_pb": [0.9] * 120, "flag": [False] * 120})
    flags = pl.DataFrame({"wallet": ws, "excluded": [False] * 120})
    cl = pl.DataFrame(
        schema={"wallet": pl.String, "cluster_id": pl.Int64, "cluster_size": pl.Int64, "oversized": pl.Boolean}
    )
    pos = pl.DataFrame(
        {"wallet": ws, "condition_id": ["c"] * 120, "event_id": ["e"] * 120, "entry_price": [0.1] * 120,
         "usd": [5.0] * 120, "won": [True] * 120}
    )  # fmt: skip
    out = wl.build_watchlist(tests, scored, flags, cl, None, pos, out_path=tmp_path / "private" / "w.csv")
    d = pl.read_csv(out)
    nv_rows = d.filter(pl.col("wallet").is_in(ws[:5]))
    assert nv_rows.height >= 1 and (nv_rows["tier"] == "watch").all() and nv_rows["no_valid_cutoff"].all()
    one = d.filter(pl.col("tier") == "one_shot_anomalous")
    assert one.height >= 1 and not one["no_valid_cutoff"].any()


def test_merge_first_seen_and_topup_candidates():
    w = pl.DataFrame({"wallet": ["a", "b"], "first_seen": [datetime(2025, 1, 1), None], "n_markets": [4, 2]})
    t = pl.DataFrame(
        {"wallet": ["b", "c", "a", "d"],
         "first_seen": [datetime(2025, 2, 1), datetime(2025, 3, 1), datetime(2024, 1, 1), None]}
    )  # fmt: skip
    m = oneshot.merge_first_seen(w, t).sort("wallet")
    got = dict(zip(m["wallet"], m["first_seen"], strict=True))
    assert got["a"] == datetime(2025, 1, 1)  # store value wins
    assert got["b"] == datetime(2025, 2, 1) and got["c"] == datetime(2025, 3, 1)  # top-up fills the gap
    assert got["d"] is None
    assert oneshot.merge_first_seen(None, None) is None

    pos = _pos(12)
    ws = pos["wallet"].to_list()
    known = pl.DataFrame({"wallet": ws[:4], "first_seen": [datetime(2025, 1, 1)] * 4})
    c = oneshot.topup_candidates(pos, None, None, None, known, n_valid=8, n_novalid=2)
    assert not set(c["wallet"]) & set(ws[:4]) and c.height > 0
    assert c["rank"].to_list() == list(range(1, c.height + 1))
    c2 = oneshot.topup_candidates(pos, None, None, None, None, n_valid=12, n_novalid=0)
    assert c2.height == 12 and c2["score_cand"].is_sorted(descending=True)
    nvp = pos.with_columns(pl.lit(False).alias("valid_cutoff"))
    c3 = oneshot.topup_candidates(nvp, None, None, None, None, n_valid=5, n_novalid=3)
    assert c3.height == 3 and c3["no_valid_cutoff"].all()


def test_privacy_phase3b_summary_has_no_addresses(tmp_path):
    base = _pos()
    sc = oneshot.score_oneshot(base, None, _wal(base["wallet"].to_list()), None, None)
    tests = pl.DataFrame({"wallet": base["wallet"], "p_pb": [0.5] * base.height, "flag": [False] * base.height})
    summ = tmp_path / "results" / "s.md"
    wl.write_phase3_summary(
        summ, n_positions=12, n_wallets_tested=12, exclusion={"wallets": 12, "excluded": 0}, rule_doc={},
        tests=tests, n_fdr_pass=0, oneshot=sc, watchlist=pl.DataFrame(schema={"tier": pl.String}),
        clusters=pl.DataFrame(), cluster_tests=None,
    )  # fmt: skip
    text = summ.read_text(encoding="utf-8")
    assert "no valid cutoff" in text and "insider" not in text.lower()
    assert wl.assert_results_clean(tmp_path / "results") == 1
