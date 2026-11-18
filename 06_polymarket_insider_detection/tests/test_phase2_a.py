"""Phase 2 builder A tests: series, baseline, score. Offline, synthetic data only."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pmdetect.markets import baseline as bl
from pmdetect.markets.baseline import fit_predict_cross
from pmdetect.markets.score import score_markets
from pmdetect.markets.series import q0_table, winner_series

NEWS = datetime(2026, 1, 20, 12, 0, 0)
TYPES = ["scheduled_reveal", "unscheduled_event", "deadline"]


def _market(i, outcome="YES", news=NEWS, info_type=None):
    return {
        "condition_id": f"c{i}",
        "question": f"m{i}",
        "info_type": info_type or TYPES[i % 3],
        "news_ts": news,
        "news_ts_method": "synthetic",
        "outcome": outcome,
    }


def _prices(cid, outcome, path, news=NEWS, step_h=1, n_days=10):
    """Winner-token (and mirrored loser-token) price rows; path(tau_h) -> winner price."""
    rows = []
    start = news - timedelta(days=n_days)
    t = start
    while t <= news + timedelta(hours=2):
        tau = (news - t).total_seconds() / 3600
        w = float(path(tau))
        win, lose = ("YES", "NO") if outcome == "YES" else ("NO", "YES")
        rows.append((f"{cid}-{win}", cid, win.title(), t, w))
        rows.append((f"{cid}-{lose}", cid, lose.title(), t, 1 - w))
        t += timedelta(hours=step_h)
    return rows


def _build(paths, outcomes=None):
    mk, rows = [], []
    for i, path in enumerate(paths):
        o = (outcomes or {}).get(i, "YES")
        mk.append(_market(i, o))
        rows += _prices(f"c{i}", o, path)
    markets = pl.DataFrame(mk).with_columns(pl.col("news_ts").cast(pl.Datetime("us")))
    prices = pl.DataFrame(
        rows, schema=["token_id", "condition_id", "outcome_label", "ts", "price"], orient="row"
    ).with_columns(pl.col("ts").cast(pl.Datetime("us")))
    return markets, prices


def _pipeline(markets, prices, **kw):
    series = winner_series(markets, prices)
    q0 = q0_table(series)
    base = fit_predict_cross(series, q0, markets.select("condition_id", "info_type"), **kw)
    return series, q0, base, score_markets(markets, series, base, q0)


def test_winner_series_uses_winner_token_and_grid():
    markets, prices = _build([lambda tau: 0.3, lambda tau: 0.2], outcomes={1: "NO"})
    s = winner_series(markets, prices)
    for cid, expected in (("c0", 0.3), ("c1", 0.2)):  # NO won in c1: its own (NO) price is q
        g = s.filter(pl.col("condition_id") == cid)
        assert g["tau_h"].min() == 1 and g["tau_h"].max() == 168 and g.height == 168
        assert np.allclose(g["q"].to_numpy(), expected)
        assert (g["t"] < NEWS).all()


def test_forward_fill_without_lookahead_and_young_flag():
    markets = pl.DataFrame([_market(0), _market(1)]).with_columns(
        pl.col("news_ts").cast(pl.Datetime("us"))
    )
    # c0: one 12h-bucket price 3 days before news (0.4) then 0.8 only 2 hours before news
    # c1: first price only 2 days before news (young market)
    rows = [
        ("c0-YES", "c0", "Yes", NEWS - timedelta(days=9), 0.1),
        ("c0-YES", "c0", "Yes", NEWS - timedelta(days=3), 0.4),
        ("c0-YES", "c0", "Yes", NEWS - timedelta(hours=2), 0.8),
        ("c1-YES", "c1", "Yes", NEWS - timedelta(days=2), 0.5),
    ]
    prices = pl.DataFrame(
        rows, schema=["token_id", "condition_id", "outcome_label", "ts", "price"], orient="row"
    ).with_columns(pl.col("ts").cast(pl.Datetime("us")))
    s = winner_series(markets, prices)
    g = s.filter(pl.col("condition_id") == "c0").sort("tau_h")
    assert (
        g.filter(pl.col("tau_h") == 1)["q"][0] == 0.8
        or g.filter(pl.col("tau_h") == 1)["q"][0] == 0.4
    )
    assert g.filter(pl.col("tau_h") == 1)["q"][0] == 0.8  # at tau=1 the 2h-old price is the latest
    assert g.filter(pl.col("tau_h") == 3)["q"][0] == 0.4  # price at tau=2 not yet visible
    assert g.filter(pl.col("tau_h") == 3)["stale_h"][0] == 69.0
    c1 = s.filter(pl.col("condition_id") == "c1")
    assert c1["tau_h"].max() == 48  # no invented history before the first observation
    q0 = q0_table(s).sort("condition_id")
    assert q0["q0"].to_list() == [0.1, 0.5]
    assert q0["q0_flag"].to_list() == [None, "young_market"]


def test_trades_supplement_prices():
    markets = pl.DataFrame([_market(0)]).with_columns(pl.col("news_ts").cast(pl.Datetime("us")))
    cols = ["token_id", "condition_id", "outcome_label", "ts", "price"]
    prices = pl.DataFrame(
        [("c0-YES", "c0", "Yes", NEWS - timedelta(days=5), 0.2)], schema=cols, orient="row"
    )
    trades = pl.DataFrame(
        [("c0-YES", "c0", "Yes", NEWS - timedelta(hours=5, minutes=30), 0.6)],
        schema=cols,
        orient="row",
    )
    s = winner_series(markets, prices, trades).sort("tau_h")
    assert s.filter(pl.col("tau_h") == 6)["q"][0] == 0.2
    assert s.filter(pl.col("tau_h") == 5)["q"][0] == 0.6


def test_median_market_has_zero_D_and_planted_drift_ranks_first():
    n = 30
    levels = np.linspace(0.1, 0.5, n)
    paths = [(lambda tau, lv=lv: lv) for lv in levels]
    drift_i = 7
    lv = levels[drift_i]
    # early drift: moves from its start level to 0.9 over the first 3 days of the window
    paths[drift_i] = lambda tau, lv=lv: lv + (0.9 - lv) * min(1.0, max(0.0, (168 - tau) / 72))
    markets, prices = _build(paths)
    _, _, base, sc = _pipeline(markets, prices)
    assert base.height > 0
    ok = sc.filter(pl.col("reason").is_null())
    assert ok.height == n
    top = ok.sort("D", descending=True)
    assert top["condition_id"][0] == f"c{drift_i}"
    flat = ok.filter(pl.col("condition_id") != f"c{drift_i}")
    assert flat["D"].max() < 0.05 * top["D"][0]
    assert top["D"][0] > 0.2 and top["share_above_m90"][0] > 0.3


def test_stationary_peers_have_D_zero():
    markets, prices = _build([lambda tau: 0.25] * 20)
    _, _, _, sc = _pipeline(markets, prices)
    assert sc["D"].max() < 1e-6 and sc["D"].min() >= 0


def test_crossfit_never_trains_on_own_market(monkeypatch):
    rng = np.random.default_rng(1)
    levels = rng.uniform(0.1, 0.5, 23)
    paths = [(lambda tau, a=a: a) for a in levels]
    markets, prices = _build(paths)
    seen = []
    orig = bl._fit_fold

    def spy(train, *a, **k):
        seen.append(set(train["condition_id"].unique().to_list()))
        return orig(train, *a, **k)

    monkeypatch.setattr(bl, "_fit_fold", spy)
    series = winner_series(markets, prices)
    base = fit_predict_cross(
        series, q0_table(series), markets.select("condition_id", "info_type"), n_splits=5
    )
    assert len(seen) == 5
    by_fold = base.group_by("fold").agg(pl.col("condition_id").unique().alias("cids"))
    all_cids = set(markets["condition_id"])
    covered = set()
    for _fold, cids in by_fold.iter_rows():
        test = set(cids)
        assert test and all(test.isdisjoint(tr) for tr in seen if tr == all_cids - test)
        assert not any(test & tr for tr in seen if tr == all_cids - test)
        covered |= test
    for tr in seen:  # each training set excludes a whole fold of markets
        assert len(tr) < len(all_cids)
    assert covered == all_cids  # every market predicted exactly once, by a model that excluded it
    assert base.group_by(["condition_id", "t"]).len()["len"].max() == 1


def test_quantiles_ordered_and_clipped():
    rng = np.random.default_rng(3)
    ab = rng.uniform(0, 1, (30, 2))
    paths = [(lambda tau, a=a, b=b: a + (b - a) * (1 - tau / 168)) for a, b in ab]
    markets, prices = _build(paths)
    _, _, base, _ = _pipeline(markets, prices)
    assert (base["m10"] <= base["m50"]).all() and (base["m50"] <= base["m90"]).all()
    assert base["m10"].min() >= 0 and base["m90"].max() <= 1


def test_unscorable_markets_are_nan_with_reason():
    paths = [lambda tau: 0.3] * 12
    markets, prices = _build(paths)
    extra = pl.DataFrame(
        [
            {**_market(100), "news_ts": None},
            {**_market(101), "outcome": None},
            _market(102),  # no prices at all
        ]
    ).with_columns(pl.col("news_ts").cast(pl.Datetime("us")))
    markets = pl.concat([markets, extra], how="diagonal_relaxed")
    # a market with only ~10 hours of history before news
    short = pl.DataFrame(
        _prices("c103", "YES", lambda tau: 0.3, n_days=10 / 24),
        schema=["token_id", "condition_id", "outcome_label", "ts", "price"],
        orient="row",
    )
    short = short.with_columns(pl.col("ts").cast(pl.Datetime("us")))
    prices = pl.concat([prices, short])
    markets = pl.concat(
        [
            markets,
            pl.DataFrame([_market(103)]).with_columns(pl.col("news_ts").cast(pl.Datetime("us"))),
        ]
    )
    _, _, _, sc = _pipeline(markets, prices)
    assert sc.height == markets.height  # nothing dropped
    reasons = dict(zip(sc["condition_id"], sc["reason"], strict=True))
    assert reasons["c100"] == "no_news_ts" and reasons["c101"] == "no_outcome"
    assert reasons["c102"] == "no_price_data" and reasons["c103"] == "too_few_hours"
    assert (
        sc.filter(pl.col("condition_id").is_in(["c100", "c101", "c102", "c103"]))["D"].null_count()
        == 4
    )
    assert sc.filter(pl.col("reason").is_null()).height == 12
