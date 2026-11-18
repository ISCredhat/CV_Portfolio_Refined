"""Time-split baseline tests: offline, synthetic."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pmdetect.markets import baseline as bl
from pmdetect.markets.baseline import fit_predict_timesplit
from pmdetect.markets.score import score_markets
from pmdetect.markets.series import q0_table

NS = pl.Datetime("us")


def _world(per_month=60, months=(1, 2, 3, 4), drift_id="drift"):
    rng = np.random.default_rng(11)
    taus = np.arange(1, 169, dtype=float)
    rows, meta = [], []
    for m in months:
        for i in range(per_month):
            cid = f"m{m}_{i}"
            news = datetime(2025, m, 1 + i % 27, 12)
            lvl = float(rng.uniform(0.25, 0.35))
            q = lvl + rng.normal(0, 0.01, 168)
            meta.append((cid, news))
            rows.append((cid, news, q))
    news = datetime(2025, months[-1], 28, 12)  # planted late market: climbs to 0.95 over 3 days
    q = np.full(168, 0.3) + np.where(taus <= 72, (0.95 - 0.3) * (72 - taus) / 72 + 0.0, 0.0)
    meta.append((drift_id, news))
    rows.append((drift_id, news, q))
    series = pl.concat(
        [
            pl.DataFrame(
                {
                    "condition_id": cid,
                    "t": [news - timedelta(hours=float(t)) for t in taus],
                    "tau_h": taus,
                    "q": np.clip(q, 0, 1),
                    "stale_h": 0.0,
                }
            )
            for cid, news, q in rows
        ]
    ).with_columns(pl.col("t").cast(NS))
    markets = pl.DataFrame(
        {
            "condition_id": [c for c, _ in meta],
            "question": [c for c, _ in meta],
            "info_type": "scheduled_reveal",
            "news_ts": [n for _, n in meta],
            "news_ts_method": "scheduled_end",
            "outcome": "YES",
        }
    ).with_columns(pl.col("news_ts").cast(NS))
    return series, markets


def _run(series, markets, **kw):
    q0 = q0_table(series)
    it = markets.select("condition_id", "info_type")
    news = markets.select("condition_id", "news_ts")
    ids = markets["condition_id"].to_list()
    return q0, fit_predict_timesplit(series, q0, it, news, score_ids=ids, **kw)


def test_no_training_market_is_later_than_the_markets_it_scores(monkeypatch):
    series, markets = _world()
    seen = []
    real_fit, real_pred = bl._fit_fold, bl._predict

    def spy_fit(train, *a, **k):
        seen.append({"train_max": train["_news"].max()})
        return real_fit(train, *a, **k)

    def spy_pred(fitted, test):
        seen[-1]["test_min"] = test["_news"].min()
        seen[-1]["test_win"] = test["_win"].min()
        return real_pred(fitted, test)

    monkeypatch.setattr(bl, "_fit_fold", spy_fit)
    monkeypatch.setattr(bl, "_predict", spy_pred)
    _, (base, plan) = _run(series, markets)
    assert seen, "at least one window must be scorable"
    for s in seen:
        assert s["train_max"] < s["test_win"] <= s["test_min"]  # strictly earlier than the window
    scored = set(base["condition_id"])
    assert scored and all(c.startswith(("m3_", "m4_")) or c == "drift" for c in scored)
    assert plan.filter(pl.col("scorable"))["n_train_markets"].min() >= bl.MIN_TRAIN_MARKETS


def test_early_windows_are_unscorable_with_a_reason():
    series, markets = _world()
    q0, (base, plan) = _run(series, markets)
    unsc = plan.filter(~pl.col("scorable"))
    assert {c.split("_")[0] for c in unsc["condition_id"]} == {"m1", "m2"}
    assert not set(unsc["condition_id"]) & set(base["condition_id"])  # no in-sample baseline rows
    sc = score_markets(markets, series, base, q0, scheme="timesplit", unscorable=unsc)
    early = sc.filter(pl.col("condition_id").str.starts_with("m1_") | pl.col("condition_id").str.starts_with("m2_"))
    assert early["D"].null_count() == early.height
    assert set(early["reason"]) == {"no_prior_training_window"}
    late = sc.filter(pl.col("condition_id").str.starts_with("m3_"))
    assert late["D"].null_count() == 0 and set(late["baseline_scheme"]) == {"timesplit"}


def test_planted_drift_in_a_late_market_scores_high():
    series, markets = _world()
    q0, (base, plan) = _run(series, markets)
    unsc = plan.filter(~pl.col("scorable"))
    sc = score_markets(markets, series, base, q0, scheme="timesplit", unscorable=unsc)
    ok = sc.filter(pl.col("D").is_not_null())
    d = ok.filter(pl.col("condition_id") == "drift")["D"][0]
    others = ok.filter(pl.col("condition_id") != "drift")["D"]
    assert d > 0.1 and d > others.max() * 5
    assert ok.sort("D", descending=True)["condition_id"][0] == "drift"


def test_train_ids_keep_extra_markets_out_of_the_fit():
    series, markets = _world()
    q0 = q0_table(series)
    it = markets.select("condition_id", "info_type")
    news = markets.select("condition_id", "news_ts")
    ids = [c for c in markets["condition_id"] if c != "drift"]
    a, _ = fit_predict_timesplit(series, q0, it, news, score_ids=[*ids, "drift"], train_ids=ids)
    b, _ = fit_predict_timesplit(
        series.filter(pl.col("condition_id") != "drift"), q0, it, news, score_ids=ids, train_ids=ids
    )
    key = ["condition_id", "t"]
    x = a.filter(pl.col("condition_id") != "drift").sort(key)["m50"].to_numpy()
    assert np.allclose(x, b.sort(key)["m50"].to_numpy())
    assert a.filter(pl.col("condition_id") == "drift").height == 168
