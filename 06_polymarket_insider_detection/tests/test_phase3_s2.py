"""Full-universe scale logic for the wallet pipeline (offline, synthetic): vectorised tests,
lazy/partitioned positions and filters, co-trading pairs and refinement, candidate selection, the
restartable first_seen fetch (get_json mocked) and an end-to-end tiny score-wallets run."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from pmdetect.wallets import calibration, clusters, filters, firstseen, pipeline
from pmdetect.wallets.positions import build_positions

T0 = datetime(2025, 1, 1, 12, 0)


def addr(i: int) -> str:
    return "0x" + f"{i:040x}"


def cid(i: int) -> str:
    return "0x" + f"{i:064x}"


# ------------------------------------------------------------------------------------ calibration


def test_batch_poisson_binomial_matches_scalar():
    rng = np.random.default_rng(3)
    ns = rng.integers(1, 30, 200)
    off = np.concatenate([[0], np.cumsum(ns)])
    p = rng.uniform(0.01, 0.99, off[-1])
    k = np.array([rng.integers(0, n + 1) for n in ns])
    got = calibration.poisson_binomial_sf_batch(p, off, k)
    ref = [calibration.poisson_binomial_sf(int(k[i]), p[off[i] : off[i + 1]]) for i in range(ns.size)]
    assert np.allclose(got, ref, atol=1e-12)


def _calibrated(n_wallets=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wallets):
        for e in range(int(rng.integers(1, 9))):
            pr = float(rng.uniform(0.05, 0.9))
            rows.append({"wallet": addr(w), "event_id": f"e{w}_{e}", "entry_price": pr,
                         "size": float(rng.uniform(10, 500)), "p_tilde": min(0.99, max(0.01, pr + 0.02)),
                         "y": int(rng.random() < pr)})  # fmt: skip
    return pl.DataFrame(rows).with_columns(pl.col("y").cast(pl.Int8))


def test_wallet_tests_vectorised_matches_reference_loop():
    d = _calibrated()
    got = calibration.wallet_tests(d, write=False, calibrated=d)
    rng = np.random.default_rng(0)
    for (w,), g in d.sort("wallet").group_by("wallet", maintain_order=True):
        p, y = g["p_tilde"].to_numpy(), g["y"].to_numpy().astype(float)
        sz, e = g["size"].to_numpy(), g["entry_price"].to_numpy()
        r = got.filter(pl.col("wallet") == w).row(0, named=True)
        assert r["n_pos"] == p.size and r["wins"] == int(y.sum())
        assert r["p_pb"] == pytest.approx(calibration.poisson_binomial_sf(int(y.sum()), p), abs=1e-12)
        assert r["excess_pnl"] == pytest.approx(float((sz * (y - p)).sum()))
        assert r["realised_pnl"] == pytest.approx(float((sz * (y - e)).sum()))
        assert r["n_longshot_wins"] == int(((e < 0.2) & (y == 1)).sum())
        if p.size >= calibration.MIN_BOOT_POS:
            assert r["p_boot"] == pytest.approx(calibration._boot_p(p, sz, y, rng))
        else:
            assert r["p_boot"] is None


def test_fit_isotonic_folds_never_see_own_event():
    pos = pl.DataFrame(
        {"wallet": [addr(i % 7) for i in range(300)], "event_id": [f"e{i // 3}" for i in range(300)],
         "entry_price": [0.1 + 0.8 * ((i * 37) % 100) / 100 for i in range(300)],
         "won": [(i * 13) % 5 == 0 for i in range(300)], "info_type": ["deadline"] * 300,
         "size": [10.0] * 300, "calib_eligible": [True] * 300}
    )  # fmt: skip
    cal, _curve, _obs = calibration.fit_isotonic(pos, n_splits=5)
    assert cal.height == 300 and cal["p_tilde"].is_between(0.01, 0.99).all()


# ----------------------------------------------------------------------------------------- positions


def _tape():
    rows = []
    for i in range(60):
        c = cid(i % 4)
        rows.append({"trade_id": f"t{i}", "wallet": addr(i % 9), "condition_id": c, "token_id": c[:10] + "Y",
                     "outcome_label": "Yes", "side": "BUY" if i % 5 else "SELL", "size": 10.0 + i,
                     "usd": (10.0 + i) * 0.3, "ts": T0 + timedelta(hours=i)})  # fmt: skip
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us")))


def _mk():
    return pl.DataFrame(
        {"condition_id": [cid(i) for i in range(4)], "event_id": ["e0", "e0", "e1", "e2"],
         "outcome": ["YES", "NO", "YES", "YES"], "info_type": ["deadline"] * 4,
         "news_ts": [T0 + timedelta(hours=80)] * 4}
    ).with_columns(pl.col("news_ts").cast(pl.Datetime("us")))


def test_positions_lazy_partitioned_equals_eager():
    t, m = _tape(), _mk()
    eager = build_positions(t, m, None, None).sort(["wallet", "event_id"])
    lazy = build_positions(t.lazy(), m, None, None, n_parts=3).sort(["wallet", "event_id"])
    assert eager.height > 0
    assert eager.equals(lazy)


# ------------------------------------------------------------------------------------------ filters


def test_filters_lazy_equals_eager_and_maker_coverage():
    t = _tape()
    pos = build_positions(t, _mk(), None, None)
    wal = pl.DataFrame({"wallet": [addr(0)], "first_seen": [T0], "n_markets": [3]}).with_columns(
        pl.col("first_seen").cast(pl.Datetime("us")), pl.col("n_markets").cast(pl.Int64)
    )
    a = filters.flag_non_directional(t, pos, wal).sort("wallet")
    b = filters.flag_non_directional(t.lazy(), pos, wal).sort("wallet")
    assert a.equals(b)
    # all-rows tape = taker rows + maker rows for market 0 only (partial coverage)
    maker = pl.DataFrame(
        [{"trade_id": f"m{i}", "wallet": addr(50), "condition_id": cid(0)} for i in range(60)]
    )
    allrows = pl.concat([t.select("trade_id", "wallet", "condition_id"), maker])
    cov = filters.maker_coverage(allrows, t)
    assert cov["taker_markets"] == 4 and cov["covered"] == 4 and cov["complete"]
    part = allrows.filter(pl.col("condition_id") == cid(0))
    cov2 = filters.maker_coverage(part, t)
    assert cov2["covered"] == 1 and not cov2["complete"] and cov2["share"] == pytest.approx(0.25)
    assert filters.maker_coverage(None, t)["covered"] == 0
    # maker wallet needs >= 5 maker markets: put it in all markets of a bigger synthetic tape
    big = pl.concat(
        [t.select("trade_id", "wallet", "condition_id")]
        + [pl.DataFrame([{"trade_id": f"q{c}_{i}", "wallet": addr(50), "condition_id": cid(c)} for i in range(20)])
           for c in range(4)]  # fmt: skip
    )
    fl = filters._maker(big, t).collect()
    assert fl.filter(pl.col("wallet") == addr(50))["n_maker_rows"][0] == 80
    assert fl.filter(pl.col("wallet") == addr(50))["maker_share"][0] == 1.0
    # taker rows are not maker rows (anti-join on trade id)
    assert fl.filter(pl.col("wallet") == addr(1)).height == 0


# ------------------------------------------------------------------------------------------ clusters


def _buy_rows(spec):
    return pl.DataFrame(
        [{"condition_id": c, "token_id": c + "y", "wallet": w, "side": "BUY",
          "ts": T0 + timedelta(minutes=m)} for w, c, m in spec]  # fmt: skip
    ).with_columns(pl.col("ts").cast(pl.Datetime("us")))


def _brute_pairs(df, window_min, cap):
    rows = df.sort(["token_id", "ts"]).to_dicts()
    out = {}
    for i, a in enumerate(rows):
        nb = [r for r in rows[i + 1 :] if r["token_id"] == a["token_id"] and (r["ts"] - a["ts"]).total_seconds() <= window_min * 60]
        if not nb or len(nb) > cap:
            continue
        for b in nb:
            if a["wallet"] != b["wallet"]:
                k = tuple(sorted((a["wallet"], b["wallet"])))
                out.setdefault(k, set()).add(a["condition_id"])
    return {k: len(v) for k, v in out.items()}


def test_pairs_vectorised_matches_brute_force():
    rng = np.random.default_rng(5)
    spec = [(addr(int(rng.integers(0, 12))), cid(int(rng.integers(0, 5))), int(rng.integers(0, 400))) for _ in range(300)]
    df = _buy_rows(spec)
    got = clusters._pairs(df, 10, 6)
    mine = {(r["wallet_a"], r["wallet_b"]): r["n_markets"] for r in got.iter_rows(named=True)}
    assert mine == _brute_pairs(df, 10, 6)
    assert clusters._pairs(df.lazy(), 10, 6).height == got.height


def test_cluster_refinement_splits_chance_chain():
    # a strong group (4 wallets, 6 shared markets) chained to a big weak crowd (1 shared market per link)
    strong = [addr(i) for i in range(4)]
    pairs = [(a, b, 6) for i, a in enumerate(strong) for b in strong[i + 1 :]]
    weak = [addr(100 + i) for i in range(70)]
    pairs += [(weak[i], weak[i + 1], 2) for i in range(69)] + [(strong[0], weak[0], 2)]
    pf = pl.DataFrame(pairs, schema=["wallet_a", "wallet_b", "n_markets"], orient="row")
    flat = clusters.link_components(pf, 2, max_cluster=50, refine=False)
    assert flat["lab"].n_unique() == 1 and flat.height == 74
    ref = clusters.link_components(pf, 2, max_cluster=50, refine=True)
    sizes = ref.group_by("lab").len()["len"].to_list()
    assert sizes == [4] and set(ref["wallet"]) == set(strong)  # weak chain dropped, strong core kept
    assert ref["link_level"].min() >= 3


def test_cotrading_clusters_schema_and_oversize_flag():
    spec = []
    for m in range(3):  # 3 wallets always buying together in 3 markets
        for k, w in enumerate((addr(1), addr(2), addr(3))):
            spec.append((w, cid(m), m * 1000 + k))
    cl = clusters.cotrading_clusters(_buy_rows(spec), min_markets=2)
    assert cl.columns == ["wallet", "cluster_id", "cluster_size", "oversized", "link_level"]
    assert cl.height == 3 and cl["cluster_size"].to_list() == [3, 3, 3] and cl["link_level"].to_list() == [2] * 3


# ------------------------------------------------------------------------------- first_seen candidates


def _scored(n=12):
    rows = []
    for i in range(n):
        rows.append({"wallet": addr(i), "no_valid_cutoff": i >= 8, "score_nofresh": 1 - i / 100,
                     "score_cand": 0.9 - i / 100})  # fmt: skip
    return pl.DataFrame(rows)


def test_candidate_table_components_priority_and_exclusion():
    pos = pl.DataFrame(
        {"wallet": [addr(20)] * 6 + [addr(21)] * 4, "calib_eligible": [True] * 10, "won": [True, False] * 5}
    )
    cand = firstseen.candidate_table(pos, _scored(), pl.Series("w", [addr(30), addr(1)]), n_top=3, n_novalid=1,
                                     exclude=pl.Series("w", [addr(2)]))  # fmt: skip
    ws = cand["wallet"].to_list()
    assert addr(2) not in ws  # excluded (non-directional) even though top-ranked
    assert set(ws) == {addr(0), addr(1), addr(3), addr(8), addr(20), addr(30)}
    c = {r["wallet"]: r for r in cand.iter_rows(named=True)}
    assert c[addr(20)]["calib_ge5"] and not c[addr(20)]["top_oneshot"]
    assert c[addr(8)]["novalid_screen"] and c[addr(1)]["jump_initiator"] and c[addr(1)]["top_oneshot"]
    # priority: jump initiators first, then top one-shot, then calib_ge5, then the no-valid screen
    assert ws[:2] == [addr(1), addr(30)] and ws[-1] == addr(8)
    assert cand["priority"].to_list() == list(range(1, cand.height + 1))
    assert firstseen.hours_estimate(18000) == pytest.approx(1.0)


def test_fetch_candidates_is_restartable_and_atomic(tmp_path, monkeypatch):
    private = tmp_path
    ws = [addr(i) for i in range(7)]
    pl.DataFrame({"wallet": ws, "priority": list(range(1, 8))}).write_parquet(private / "first_seen_candidates.parquet")
    calls: list[str] = []

    def fake_get_json(url, params=None, *, use_cache=True):
        assert use_cache is False and params["sort_direction"] == "ASC" and params["limit"] == 1
        calls.append(params["user"])
        if params["user"] == ws[3]:
            raise RuntimeError("boom")  # skipped, retried next run
        if params["user"] == ws[4]:
            return {"data": []}  # no trades -> null first_seen row
        return {"data": [{"timestamp": 1_700_000_000}]}

    monkeypatch.setattr("pmdetect.collect.http.get_json", fake_get_json)
    r1 = firstseen.fetch_candidates(max_n=5, threads=2, private=private, log=lambda m: None, save_every=2)
    assert r1["ok"] == 4 and r1["errors"] == 1
    top = pl.read_parquet(private / "first_seen_topup.parquet")
    assert top.height == 4 and top.filter(pl.col("wallet") == ws[4])["first_seen"][0] is None
    assert not list(private.glob("*.tmp"))
    n1 = len(calls)
    r2 = firstseen.fetch_candidates(threads=2, private=private, log=lambda m: None, save_every=2)
    assert len(calls) - n1 == 3 and r2["saved"] == 6  # ws[3] retried, ws[5], ws[6] new; ws[3] errors again
    assert pl.read_parquet(private / "first_seen_topup.parquet")["wallet"].n_unique() == 6


# ----------------------------------------------------------------------------- end-to-end tiny run


def test_score_wallets_end_to_end_tiny(tmp_path):
    snap, private, results = tmp_path / "snap", tmp_path / "private", tmp_path / "results"
    for d in (snap, private, results):
        d.mkdir()
    n_m = 30
    mk = pl.DataFrame(
        {"condition_id": [cid(i) for i in range(n_m)], "event_id": [f"ev{i}" for i in range(n_m)],
         "question": [f"q{i}" for i in range(n_m)], "outcome": ["YES" if i % 2 else "NO" for i in range(n_m)],
         "info_type": ["scheduled_reveal"] * n_m,
         "news_ts": [T0 + timedelta(days=10)] * n_m}
    ).with_columns(pl.col("news_ts").cast(pl.Datetime("us")))
    mk.write_parquet(snap / "markets.parquet")
    pl.DataFrame(
        {"condition_id": mk["condition_id"], "news_ts_indep": mk["news_ts"], "indep_method": ["scheduled_end"] * n_m,
         "independent": [True] * n_m}
    ).write_parquet(snap / "news_independent.parquet")
    rng = np.random.default_rng(2)
    rows = []
    for i in range(1500):
        m = int(rng.integers(0, n_m))
        lab = "Yes" if rng.random() < 0.5 else "No"
        px = float(rng.uniform(0.05, 0.95))
        rows.append({"trade_id": f"t{i}", "tx_hash": f"h{i}", "condition_id": cid(m), "token_id": f"{m}{lab}",
                     "wallet": addr(int(rng.integers(0, 60))), "side": "BUY", "outcome_label": lab, "size": 20.0,
                     "price": px, "usd": 20.0 * px, "ts": T0 + timedelta(hours=float(rng.uniform(0, 200)))})  # fmt: skip
    taker = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us")))
    taker.write_parquet(snap / "trades_taker.parquet")
    taker.select("trade_id", "wallet", "condition_id").filter(pl.col("condition_id").is_in([cid(i) for i in range(10)])).write_parquet(snap / "trades.parquet")  # partial all-rows
    pl.DataFrame({"wallet": [addr(1)], "first_seen": [T0 - timedelta(days=30)], "n_markets": [5], "total_usd": [1.0]}).with_columns(
        pl.col("first_seen").cast(pl.Datetime("us")), pl.col("n_markets").cast(pl.Int32)
    ).write_parquet(snap / "wallets.parquet")
    r = pipeline.run_score_wallets(snap, private, results, rebuild_positions=None, log=lambda m: None)
    assert r["positions"] > 0 and r["tested"] > 0
    assert not r["maker_coverage"]["complete"]  # partial all-rows tape is reported as incomplete
    text = (results / "phase3_summary.md").read_text(encoding="utf-8")
    assert "INCOMPLETE" in text and "0x" not in text  # privacy: no addresses in results/
    for f in ("positions.parquet", "wallet_tests_fdr.parquet", "oneshot_scores.parquet", "watchlist.csv"):
        assert (private / f).exists()
    # second run without input changes reuses the positions file
    r2 = pipeline.run_score_wallets(snap, private, results, rebuild_positions=None, log=lambda m: None)
    assert any("load" in t["stage"] for t in r2["timings"])
    # candidate selection runs locally from the outputs
    sel = firstseen.select_candidates(snap, private, results, n_top=5, n_novalid=2, log=lambda m: None)
    assert sel["to_fetch"] >= 0 and (private / "first_seen_candidates.parquet").exists()
