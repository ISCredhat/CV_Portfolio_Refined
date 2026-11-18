"""Full-universe backfill, builder S1: bulk news times, independent times from the taker tape,
vectorised winner series, incremental within-event fetch, end-to-end pipeline. Offline/synthetic."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import polars as pl
import pytest

from pmdetect.markets import newstime, newstime_indep, within_event
from pmdetect.markets import series as S
from pmdetect.store import db

T0 = datetime(2025, 3, 1)
NS = pl.Datetime("us")


# --------------------------------------------------------------------------- bulk news times


def _synthetic_world(seed: int = 0, n: int = 60):
    """Markets with random price series (fine or coarse), taker fills on both tokens, edge cases."""
    rng = np.random.default_rng(seed)
    mk, tok, pr, tr = [], [], [], []
    for i in range(n):
        cid = f"c{i}"
        outcome = "YES" if i % 3 else "NO"
        if i == 5:
            outcome = None
        if i == 6:
            outcome = "INVALID"
        yes, no = f"{cid}-y", f"{cid}-n"
        mk.append((cid, f"slug{i}", outcome))
        tok += [(yes, cid, "Yes"), (no, cid, "No")]
        win_tok = yes if outcome == "YES" else no
        kind = i % 6
        step = 600 if kind in (0, 1, 2) else 10800  # fine vs coarse price history
        m = int(rng.integers(0, 40)) if kind != 3 else 0
        level = rng.uniform(0.2, 1.0, size=m)
        if kind == 1:
            level = np.clip(level + 0.5, 0, 1)  # often above the threshold
        if kind == 4:
            level = np.full(m, 0.97)  # always above
        if kind == 5 and m:
            level[-1] = 0.3  # ends below: no_cross
        for j, v in enumerate(level):
            pr.append((win_tok, T0 + timedelta(seconds=step * j), float(round(v, 3))))
        # taker fills: each fill on a random token, price = that token's price
        k = int(rng.integers(0, 40)) if i not in (3, 7) else 0
        q = np.clip(rng.uniform(0.3, 1.0, size=k) if kind != 5 else rng.uniform(0.3, 0.8, size=k), 0, 1)
        for j in range(k):
            on_winner = bool(rng.integers(0, 2))
            p = float(round(q[j] if on_winner else 1 - q[j], 3))
            tr.append(
                (f"{cid}-t{j}", cid, win_tok if on_winner else (no if win_tok == yes else yes),
                 T0 + timedelta(minutes=17 * j), p)
            )  # fmt: skip
    markets = pl.DataFrame(mk, schema=["condition_id", "slug", "outcome"], orient="row")
    tokens = pl.DataFrame(tok, schema=["token_id", "condition_id", "outcome_label"], orient="row")
    prices = pl.DataFrame(pr, schema={"token_id": pl.String, "ts": NS, "price": pl.Float64}, orient="row")
    trades = pl.DataFrame(
        tr,
        schema={"trade_id": pl.String, "condition_id": pl.String, "token_id": pl.String, "ts": NS, "price": pl.Float64},
        orient="row",
    )
    return markets, tokens, prices, trades


def test_bulk_news_times_matches_per_market_function():
    markets, tokens, prices, trades = _synthetic_world()
    ov = ({}, {"slug9": {"news_ts": "2025-04-01T12:00:00Z", "method": "scheduled_reveal"}})
    bulk = newstime.bulk_news_times(markets, tokens, prices, trades, ov).sort("condition_id")
    assert bulk.height == markets.height
    expect = []
    for cid, slug, outcome in markets.iter_rows():
        if slug == "slug9":
            expect.append((cid, newstime.parse_ts("2025-04-01T12:00:00Z"), "override_scheduled_reveal"))
            continue
        if outcome not in ("YES", "NO"):
            expect.append((cid, None, "no_outcome"))
            continue
        wt = f"{cid}-y" if outcome == "YES" else f"{cid}-n"
        pp = prices.filter(pl.col("token_id") == wt).sort("ts")
        t = trades.filter(pl.col("condition_id") == cid).sort("ts", "trade_id")
        tp = [
            (r["ts"], r["price"] if r["token_id"] == wt else round(1 - r["price"], 6))
            for r in t.iter_rows(named=True)
        ]
        ts, meth = newstime.estimate_for_market(list(zip(pp["ts"], pp["price"], strict=True)), tp)
        expect.append((cid, ts, meth))
    exp = pl.DataFrame(expect, schema={"condition_id": pl.String, "news_ts": NS, "news_ts_method": pl.String}, orient="row").sort("condition_id")
    assert bulk["news_ts"].to_list() == exp["news_ts"].to_list()
    assert bulk["news_ts_method"].to_list() == exp["news_ts_method"].to_list()
    methods = set(bulk["news_ts_method"])  # the synthetic world exercises every status string
    for needed in ("price_cross_0.9", "trade_cross_0.9", "no_outcome", "no_cross", "override_scheduled_reveal"):
        assert needed in methods
    assert any(m.endswith("_always_above") for m in methods)
    assert "no_data" in methods


def test_bulk_news_times_updates_duckdb_without_fetch_log(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    markets, tokens, prices, trades = _synthetic_world(seed=1, n=12)
    trades = trades.with_columns(pl.lit("w").alias("wallet"))
    markets.with_columns(
        pl.lit(True).alias("in_universe"),
    ).write_parquet(snap / "markets.parquet")
    tokens.write_parquet(snap / "tokens.parquet")
    prices.write_parquet(snap / "prices.parquet")
    trades.write_parquet(snap / "trades_taker.parquet")
    path = tmp_path / "t.duckdb"
    monkeypatch.setattr(newstime, "connect", partial(db.connect, path))
    con = db.connect(path)
    for cid, slug, outcome in markets.iter_rows():
        con.execute(
            "INSERT INTO markets (condition_id, slug, outcome, in_universe) VALUES (?,?,?,TRUE)",
            [cid, slug, outcome],
        )
    con.close()  # note: no fetch_log rows at all
    res = newstime.estimate_news_times_bulk(tmp_path / "none.csv", snapshot_dir=snap)
    con = db.connect(path)
    got = dict(
        (r[0], (r[1], r[2]))
        for r in con.execute("SELECT condition_id, news_ts, news_ts_method FROM markets").fetchall()
    )
    con.close()
    assert len(got) == markets.height
    for r in res.iter_rows(named=True):
        assert got[r["condition_id"]] == (r["news_ts"], r["news_ts_method"])
    assert sum(v[0] is not None for v in got.values()) == res["news_ts"].is_not_null().sum() > 0


# --------------------------------------------------------------------------- independent news times


def test_independent_news_times_use_the_full_taker_tape(tmp_path):
    snap = tmp_path
    end = datetime(2025, 5, 1, 12)
    mk = pl.DataFrame(
        {
            "condition_id": ["a", "b", "c", "d"],
            "slug": ["a", "b", "c", "d"],
            "info_type": ["scheduled_reveal"] * 3 + ["deadline"],
            "end_ts": [end] * 4,
            "closed_ts": [end + timedelta(hours=5)] * 4,
            "in_universe": [True] * 4,
            "outcome": ["YES"] * 4,
        }
    )
    mk.write_parquet(snap / "markets.parquet")
    pl.DataFrame({"token_id": ["t"], "condition_id": ["a"], "outcome_label": ["Yes"]}).write_parquet(snap / "tokens.parquet")
    pl.DataFrame(schema={"token_id": pl.String, "ts": NS, "price": pl.Float64}).write_parquet(snap / "prices.parquet")
    rows = [  # last trade: a = end+10h (ok), b = end-30h (too early), c = end+200h (too late), d any
        ("a", end + timedelta(hours=10)), ("a", end - timedelta(hours=3)),
        ("b", end - timedelta(hours=30)), ("c", end + timedelta(hours=200)), ("d", end),
    ]  # fmt: skip
    pl.DataFrame(rows, schema={"condition_id": pl.String, "ts": NS}, orient="row").write_parquet(snap / "trades_taker.parquet")
    out = newstime_indep.build(snap, write=True).sort("condition_id")
    assert out["independent"].to_list() == [True, False, False, False]
    assert out["indep_method"].to_list() == ["scheduled_end", "none", "none", "none"]
    assert (snap / "news_independent.parquet").exists()
    summ = newstime_indep.summarize(out, mk)
    assert summ.filter(pl.col("indep_method") == "scheduled_end")["n"].sum() == 1


# --------------------------------------------------------------------------- series at scale


def _naive_series(markets, prices, trades, news_col="news_ts", window_h=168, buffer_h=1):
    """The earlier per-market numpy implementation (reference), with fills on the losing token
    mapped to 1 - price."""
    out = {}
    for cid, news, outcome in markets.select("condition_id", news_col, "outcome").iter_rows():
        if news is None or outcome not in ("YES", "NO"):
            continue
        obs = [(r["ts"], r["price"], 0) for r in prices.filter((pl.col("condition_id") == cid) & (pl.col("outcome_label").str.to_uppercase() == outcome)).iter_rows(named=True)]
        if trades is not None:
            for r in trades.filter(pl.col("condition_id") == cid).iter_rows(named=True):
                lab = r["outcome_label"].upper()
                obs.append((r["ts"], r["price"] if lab == outcome else 1 - r["price"], 1))
        obs.sort(key=lambda o: (o[0], o[2]))
        for tau in range(buffer_h, window_h + 1):
            t = news - timedelta(hours=tau)
            last = None
            for o in obs:
                if o[0] <= t:
                    last = o
            if last is not None:
                out[(cid, tau)] = (min(1.0, max(0.0, last[1])), (t - last[0]).total_seconds() / 3600)
    return out


def test_vectorised_winner_series_matches_reference():
    rng = np.random.default_rng(3)
    news = datetime(2025, 6, 10, 9)
    mk, pr, tr = [], [], []
    for i in range(8):
        outcome = ["YES", "NO"][i % 2] if i != 6 else None
        mk.append((f"m{i}", news + timedelta(hours=i) if i != 7 else None, outcome))
        for _j in range(int(rng.integers(0, 12))):
            pr.append((f"m{i}-{'y' if outcome != 'NO' else 'n'}", f"m{i}", "Yes" if outcome != "NO" else "No",
                       news - timedelta(hours=float(rng.uniform(0, 300))), float(rng.uniform(0, 1))))  # fmt: skip
        for _j in range(int(rng.integers(0, 20))):
            lab = ["Yes", "No"][int(rng.integers(0, 2))]
            tr.append((f"m{i}", f"m{i}-{lab}", lab, news - timedelta(hours=float(rng.uniform(0, 300))), float(rng.uniform(0, 1))))  # fmt: skip
    markets = pl.DataFrame(mk, schema={"condition_id": pl.String, "news_ts": NS, "outcome": pl.String}, orient="row")
    prices = pl.DataFrame(pr, schema=["token_id", "condition_id", "outcome_label", "ts", "price"], orient="row").with_columns(pl.col("ts").cast(NS))
    trades = pl.DataFrame(tr, schema=["condition_id", "token_id", "outcome_label", "ts", "price"], orient="row").with_columns(pl.col("ts").cast(NS))
    got = S.winner_series(markets, prices, trades)
    ref = _naive_series(markets, prices, trades)
    assert got.height == len(ref) > 0
    for r in got.iter_rows(named=True):
        q, stale = ref[(r["condition_id"], int(r["tau_h"]))]
        assert r["q"] == pytest.approx(q) and r["stale_h"] == pytest.approx(stale, abs=1e-9)
    assert got["t"].dtype == NS


def test_load_inputs_covers_the_whole_universe_and_tape_markets(tmp_path):
    snap = tmp_path
    base = {"slug": None, "question": None, "info_type": None, "news_ts": None}
    rows = [
        {"condition_id": "u1", "in_universe": True, "outcome": "YES", **base},
        {"condition_id": "u2", "in_universe": True, "outcome": "NO", **base},
        {"condition_id": "tapeonly", "in_universe": False, "outcome": "YES", **base},
        {"condition_id": "other", "in_universe": False, "outcome": "YES", **base},
    ]
    pl.DataFrame(rows).write_parquet(snap / "markets.parquet")
    tok = [(f"{c}-y", c, "Yes") for c in ("u1", "u2", "tapeonly", "other")]
    pl.DataFrame(tok, schema=["token_id", "condition_id", "outcome_label"], orient="row").write_parquet(snap / "tokens.parquet")
    pl.DataFrame(
        [(f"{c}-y", T0, 0.5) for c in ("u1", "u2", "tapeonly", "other")],
        schema={"token_id": pl.String, "ts": NS, "price": pl.Float64}, orient="row",
    ).write_parquet(snap / "prices.parquet")  # fmt: skip
    pl.DataFrame(
        [("tapeonly", T0, 0.4), ("u1", T0, 0.4)], schema={"condition_id": pl.String, "ts": NS, "price": pl.Float64}, orient="row",
    ).write_parquet(snap / "trades_taker.parquet")  # fmt: skip
    markets, prices = S.load_inputs(snap)
    assert sorted(markets["condition_id"]) == ["tapeonly", "u1", "u2"]  # not restricted to the tape
    assert sorted(prices["condition_id"].unique()) == ["tapeonly", "u1", "u2"]


# --------------------------------------------------------------------------- within-event, incremental


def _gamma_event(eid, n=3, winner=0, base_ts="2025-05-01T00:00:00Z"):
    ms = []
    for k in range(n):
        yes_price = "1" if k == winner else "0"
        ms.append(
            {
                "conditionId": f"0x{eid}{k}",
                "question": f"q{k}",
                "groupItemTitle": f"opt{k}",
                "clobTokenIds": json.dumps([f"tok{eid}{k}y", f"tok{eid}{k}n"]),
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps([yes_price, str(1 - int(yes_price))]),
                "endDate": base_ts,
                "closedTime": "2025-05-01 01:00:00+00",
            }
        )
    return {"id": eid, "negRisk": True, "markets": ms}


def _series_points(token, ladder=None, n=40):
    base = int(datetime(2025, 4, 28).timestamp())
    h = abs(hash(token)) % 7 / 10
    return [{"timestamp": base + 3600 * 3 * i, "price": min(0.99, 0.2 + h * i / n)} for i in range(n)], 10800


@pytest.fixture
def we_snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    end = datetime(2025, 5, 1, 0)
    eids = ["11", "12", "13"]
    mk = [(f"0x{e}0", e, f"q{e}", "YES") for e in eids]
    pl.DataFrame(mk, schema=["condition_id", "event_id", "question", "outcome"], orient="row").write_parquet(snap / "markets.parquet")
    pl.DataFrame(
        {"condition_id": [m[0] for m in mk], "news_ts_indep": [end] * 3, "indep_method": ["scheduled_end"] * 3, "independent": [True] * 3}
    ).write_parquet(snap / "news_independent.parquet")  # fmt: skip
    return snap


def test_incremental_fetch_is_resumable_atomic_and_consolidates(we_snapshot, tmp_path, monkeypatch):
    calls = {"get": [], "series": []}

    def fake_get(url, params=None, **kw):
        calls["get"].append(url)
        eid = url.rsplit("/", 1)[1]
        if eid == "12" and not calls.get("failed"):
            calls["failed"] = True
            raise RuntimeError("boom")
        return _gamma_event(eid)

    def fake_series(tok, ladder=None):
        calls["series"].append(tok)
        return _series_points(tok)

    monkeypatch.setattr(within_event, "get_json", fake_get)
    monkeypatch.setattr(within_event, "fetch_price_series", fake_series)
    parts = tmp_path / "events"
    log = []
    st = within_event.fetch_event_data(we_snapshot, parts, log=log.append)
    assert st["events"] == 2 and len(st["failed"]) == 1 and st["failed"][0][0] == "12"
    assert within_event.done_events(parts) == {"11", "13"}
    assert not list(parts.rglob("*.tmp"))  # atomic: no leftovers
    n_series = len(calls["series"])
    # restart: only the failed event is fetched, finished events cost nothing
    calls["get"].clear()
    st2 = within_event.fetch_event_data(we_snapshot, parts, log=log.append)
    assert st2["events"] == 1 and st2["already_done"] == 2
    assert calls["get"] == [f"{within_event.GAMMA}/events/12"]
    assert len(calls["series"]) == n_series + 3
    calls["get"].clear()
    st3 = within_event.fetch_event_data(we_snapshot, parts, log=log.append)
    assert st3["events"] == 0 and calls["get"] == []
    # consolidation = concatenated parts, written atomically to the snapshot
    out = within_event.consolidate(we_snapshot, parts, log=lambda s: None)
    em = pl.read_parquet(we_snapshot / "event_markets.parquet")
    ep = pl.read_parquet(we_snapshot / "event_prices.parquet")
    assert out["events"] == 3 and em.height == 9 and sorted(em["event_id"].unique()) == ["11", "12", "13"]
    assert ep.height == 9 * 40 and set(ep["source"]) == {"history"}
    assert em["is_pilot"].sum() == 3  # the three independent markets
    # the whole thing then runs offline
    plc = within_event.run(we_snapshot, n_perm=50, log=lambda s: None, parts_dir=parts)
    assert "24h" in plc and (we_snapshot / "within_event.parquet").exists()
    w = pl.read_parquet(we_snapshot / "within_event.parquet")
    assert set(w["group"]) == {"multi"}


def test_pilot_snapshot_is_imported_not_refetched(we_snapshot, tmp_path, monkeypatch):
    monkeypatch.setattr(within_event, "get_json", lambda *a, **k: _gamma_event(a[0].rsplit("/", 1)[1]))
    monkeypatch.setattr(within_event, "fetch_price_series", _series_points)
    first = tmp_path / "e1"
    within_event.fetch_event_data(we_snapshot, first, log=lambda s: None)
    within_event.consolidate(we_snapshot, first, log=lambda s: None)  # writes event_markets/prices snapshots
    monkeypatch.setattr(within_event, "get_json", lambda *a, **k: pytest.fail("network call"))
    second = tmp_path / "e2"
    st = within_event.fetch_event_data(we_snapshot, second, log=lambda s: None)
    assert st["imported"] == 3 and st["events"] == 0
    a = within_event.consolidate(we_snapshot, second, log=lambda s: None)
    assert a["markets"] == 9


def test_trade_price_rows_use_all_rows_tape_first_then_taker_mapped(we_snapshot):
    mk = pl.DataFrame(
        {"event_id": ["e", "e"], "condition_id": ["A", "B"], "yes_token_id": ["ay", "by"]}
    )
    snap = we_snapshot
    pl.DataFrame(
        [("A", "ay", T0, 0.7), ("A", "an", T0, 0.3)],
        schema={"condition_id": pl.String, "token_id": pl.String, "ts": NS, "price": pl.Float64}, orient="row",
    ).write_parquet(snap / "trades.parquet")  # fmt: skip
    pl.DataFrame(
        [("A", "ay", T0, 0.9), ("B", "by", T0, 0.6), ("B", "bn", T0 + timedelta(hours=1), 0.25)],
        schema={"condition_id": pl.String, "token_id": pl.String, "ts": NS, "price": pl.Float64}, orient="row",
    ).write_parquet(snap / "trades_taker.parquet")  # fmt: skip
    t = within_event.trade_price_rows(snap, mk).sort("condition_id", "ts")
    # A comes from the all-rows tape (YES-token rows only), B from the taker tape mapped to YES
    assert t.filter(pl.col("condition_id") == "A")["price"].to_list() == [0.7]
    assert t.filter(pl.col("condition_id") == "B")["price"].to_list() == [0.6, 0.75]
    assert set(t["source"]) == {"trade"}


def test_cost_estimate_counts_events_siblings_and_series(we_snapshot, tmp_path):
    snap = we_snapshot
    rows = []
    for e in ("11", "12", "13"):
        for k in range(4):
            closed = datetime(2025, 4, 1) if k == 3 else datetime(2025, 5, 2)  # k=3 closed early
            rows.append((f"0x{e}{k}", e, closed))
    pl.DataFrame(rows, schema={"condition_id": pl.String, "event_id": pl.String, "closed_ts": NS}, orient="row").write_parquet(snap / "markets_all.parquet")
    base = pl.read_parquet(snap / "markets.parquet")
    full = pl.DataFrame(rows, schema={"condition_id": pl.String, "event_id": pl.String, "closed_ts": NS}, orient="row").join(
        base.select("condition_id", "question", "outcome"), on="condition_id", how="left"
    )
    full.write_parquet(snap / "markets.parquet")
    est = within_event.estimate_fetch_cost(snap, tmp_path / "none")
    assert est["events_total"] == 3 and est["events_todo"] == 3
    # per event: independent market (k=0) + 2 not-early siblings = 3 series; early sibling skipped
    assert est["series_calls_todo"] == 9
    assert est["requests_todo"] == pytest.approx(3 + 9 * within_event.REQ_PER_SERIES_COARSE)
    assert est["hours_at_5rps"] == pytest.approx(est["requests_todo"] / 5 / 3600)


def test_binary_percentile_vectorisation_matches_brute_force():
    rng = np.random.default_rng(5)
    n = 25
    pilots = pl.DataFrame(
        {
            "condition_id": [f"b{i}" for i in range(n)],
            "event_id": [f"e{i}" for i in range(n)],
            "news_ts_indep": [datetime(2025, 5, 1)] * n,
            "outcome": ["YES" if i % 2 else "NO" for i in range(n)],
        }
    )
    em, pr = [], []
    for i in range(n):
        em.append((f"e{i}", f"b{i}", "q", "l", f"b{i}y", pilots["outcome"][i], datetime(2025, 5, 1), datetime(2025, 5, 1, 2), False, True, 1, True, 10, 3600))  # fmt: skip
        for h in range(0, 60):
            pr.append((f"e{i}", f"b{i}", f"b{i}y", datetime(2025, 4, 28) + timedelta(hours=h), float(np.clip(0.5 + rng.normal(0, 0.1), 0.01, 0.99)), "history"))  # fmt: skip
    mk = pl.DataFrame(em, schema=list(within_event.MARKETS_SCHEMA), orient="row").cast(within_event.MARKETS_SCHEMA)
    prices = pl.DataFrame(pr, schema=list(within_event.PRICES_SCHEMA), orient="row").cast(within_event.PRICES_SCHEMA)
    w, _ = within_event.compute_within_event(pilots, mk, prices)
    for side in ("YES", "NO"):
        ids = [c for c, o in zip(pilots["condition_id"], pilots["outcome"], strict=True) if o == side]
        sub = w.filter(pl.col("condition_id").is_in(ids) & pl.col("win_move_24h").is_not_null())
        vals = {r["condition_id"]: r["win_move_24h"] for r in sub.iter_rows(named=True)}
        for cid, v in vals.items():
            others = np.array([x for c, x in vals.items() if c != cid])
            exp_pct = within_event.mid_percentile(v, np.append(others, v))
            got = sub.filter(pl.col("condition_id") == cid).row(0, named=True)
            assert got["winner_pct_24h"] == pytest.approx(exp_pct)
            assert got["diff_24h"] == pytest.approx(v - float(np.median(others)))


def test_permutation_null_matches_the_loop_implementation():
    rng = np.random.default_rng(9)
    events = [(rng.normal(size=int(rng.integers(2, 9))), 0) for _ in range(15)]

    def loop_version(events, n_perm, seed):  # the earlier implementation, kept as the reference
        r = np.random.default_rng(seed)
        ev = [(np.asarray(m, float), w) for m, w in events if len(m) >= 2]

        def stat(pick):
            d, s = [], []
            for (m, _), k in zip(ev, pick, strict=True):
                st = within_event.event_stats(m, int(k))
                d.append(st["diff"])
                s.append(st["win"] > st["q95_lose"])
            return float(np.mean(d)), float(np.median(d)), float(np.mean(s))

        obs = stat([w for _, w in ev])
        sizes = [len(m) for m, _ in ev]
        null = np.empty((n_perm, 3))
        for b in range(n_perm):
            null[b] = stat([r.integers(0, n) for n in sizes])
        return obs, null

    obs, null = loop_version(events, 200, 4)
    got = within_event.permutation_null(events, n_perm=200, seed=4)
    assert got["obs_mean_diff"] == pytest.approx(obs[0])
    assert got["null_mean_diff"] == pytest.approx(null[:, 0].mean())
    assert got["p_share_above_q95"] == pytest.approx((1 + (null[:, 2] >= obs[2]).sum()) / 201)


# --------------------------------------------------------------------------- pipeline at scale


def _world_snapshot(snap, n=45, seed=2):
    """A synthetic full-universe snapshot big enough for the baseline/combine, with a planted
    early-drift market and a labelled case."""
    rng = np.random.default_rng(seed)
    end0 = datetime(2025, 7, 1, 12)
    mk, tok, pr, tr, ni = [], [], [], [], []
    w = [f"w{k}" for k in range(60)]
    for i in range(n):
        cid, yes, no = f"c{i}", f"c{i}y", f"c{i}n"
        end = end0 + timedelta(days=i)
        it = ["scheduled_reveal", "unscheduled_event", "deadline"][i % 3] if i % 5 else "scheduled_reveal"
        mk.append(dict(condition_id=cid, event_id=f"e{i}", slug=cid, question=f"Q{i}?", category=None, tags=None, info_type=it,  # noqa: C408
                       volume_usd=1e4, liquidity_usd=1e3, start_ts=end - timedelta(days=20), end_ts=end,
                       closed_ts=end + timedelta(hours=2), uma_status="resolved", outcome="YES",
                       news_ts=None, news_ts_method=None, in_universe=True, fetched_at=end))  # fmt: skip
        tok += [(yes, cid, "Yes"), (no, cid, "No")]
        level = 0.15 + 0.01 * (i % 20)
        for h in range(24 * 10):
            t = end - timedelta(hours=24 * 10 - h)
            q = level + 0.02 * np.sin(h / 9)
            if i == 7:  # planted drift: climbs to 0.9 starting 5 days before the end
                q = level + (0.9 - level) * min(1.0, max(0.0, (h - 24 * 5) / 72))
            q = float(np.clip(q + rng.normal(0, 0.003), 0.01, 0.99))
            pr.append((yes, t, q))
            if h % 6 == 0:
                tr.append((f"t{i}-{h}", f"x{i}-{h}", cid, yes, w[int(rng.integers(0, 60))], "BUY", "Yes", 10.0, q, 10 * q, t + timedelta(minutes=5)))  # fmt: skip
        ni.append((cid, end, "scheduled_end" if it == "scheduled_reveal" else "none", it == "scheduled_reveal"))
    pl.DataFrame(mk).write_parquet(snap / "markets.parquet")
    pl.DataFrame(tok, schema=["token_id", "condition_id", "outcome_label"], orient="row").write_parquet(snap / "tokens.parquet")
    pl.DataFrame(pr, schema={"token_id": pl.String, "ts": NS, "price": pl.Float64}, orient="row").write_parquet(snap / "prices.parquet")
    pl.DataFrame(
        tr,
        schema={"trade_id": pl.String, "tx_hash": pl.String, "condition_id": pl.String, "token_id": pl.String, "wallet": pl.String,
                "side": pl.String, "outcome_label": pl.String, "size": pl.Float64, "price": pl.Float64, "usd": pl.Float64, "ts": NS},
        orient="row",
    ).write_parquet(snap / "trades_taker.parquet")  # fmt: skip
    pl.DataFrame(
        {"wallet": w, "first_seen": [datetime(2025, 1, 1)] * 60, "n_markets": [3] * 60, "total_usd": [100.0] * 60}
    ).write_parquet(snap / "wallets.parquet")
    pl.DataFrame(
        {"label_id": ["l1"], "condition_id": ["c7"], "wallet": [None], "case_name": ["planted"], "label": ["insider_suspected"],
         "source_url": [None], "notes": [None]},
        schema={"label_id": pl.String, "condition_id": pl.String, "wallet": pl.String, "case_name": pl.String, "label": pl.String,
                "source_url": pl.String, "notes": pl.String},
    ).write_parquet(snap / "labels.parquet")  # fmt: skip
    # news times as the bulk estimator would write them (price based for the non-independent)
    inp = newstime.load_bulk_inputs(snap)
    nt = newstime.bulk_news_times(inp["markets"], inp["tokens"], inp["prices"], inp["trades"], ({}, {}))
    m = pl.read_parquet(snap / "markets.parquet").drop("news_ts", "news_ts_method").join(nt, on="condition_id")
    m.write_parquet(snap / "markets.parquet")
    pl.DataFrame(ni, schema={"condition_id": pl.String, "news_ts_indep": NS, "indep_method": pl.String, "independent": pl.Boolean}, orient="row").with_columns(
        pl.when(pl.col("independent")).then(pl.col("news_ts_indep")).otherwise(None).alias("news_ts_indep")
    ).write_parquet(snap / "news_independent.parquet")  # fmt: skip


def test_run_phase2_on_a_synthetic_universe(tmp_path):
    from pmdetect.report.phase2 import run_phase2

    snap = tmp_path / "snap"
    snap.mkdir()
    _world_snapshot(snap)
    res = run_phase2(
        results_dir=tmp_path / "results", snapshot_dir=snap, max_flag_plots=3, private_dir=tmp_path / "private",
        baseline_scheme="crossfit",  # the 45 synthetic markets span <2 months: no time-split history
    )
    df = res["scores"]
    assert df.height == 45  # every universe market scored/listed, none dropped
    assert res["n_scored_D"] > 5 and res["d_threshold"] >= 0
    assert set(res["timings"]) >= {"series", "baseline", "signals", "jumps", "flow", "plots", "total"}
    assert res["peak_mb"] > 0 or np.isnan(res["peak_mb"])
    # figures only for flagged / labelled markets, never one per market
    figs = list((tmp_path / "results" / "figures").glob("*.png"))
    assert 1 <= len(figs) <= 3 + 1 + 1  # <= flagged cap + labelled case + histogram
    summary = (tmp_path / "results" / "phase2_summary.md").read_text(encoding="utf-8")
    assert "within-event component ABSENT" in summary and "Run performance" in summary
    assert "pilot" not in summary.split("## Caveats")[0].lower().replace("pilot markets and", "")
    assert not (tmp_path / "private" / "jump_initiators.csv").read_text(encoding="utf-8").startswith("0x")


def test_run_phase2_timesplit_reports_unscorable_early_markets(tmp_path):
    from pmdetect.report.phase2 import run_phase2

    snap = tmp_path / "snap"
    snap.mkdir()
    _world_snapshot(snap)
    res = run_phase2(
        results_dir=tmp_path / "results", snapshot_dir=snap, max_flag_plots=1, private_dir=tmp_path / "private"
    )
    df = res["scores"]
    assert res["n_scored_D"] == 0  # all synthetic windows lack >= 100 earlier training markets
    assert df.filter(pl.col("reason") == "no_prior_training_window").height > 0
    assert df["D_crossfit"].is_not_null().sum() > 5  # the cross-fit comparison column is still filled
    summary = (tmp_path / "results" / "phase2_summary.md").read_text(encoding="utf-8")
    assert "Baseline scheme: time split vs cross-fit" in summary and "unscorable" in summary
