"""Phase 2b builder A tests: independent news times, independent-only baseline, late-window."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pmdetect.markets.baseline import fit_predict_cross
from pmdetect.markets.newstime_indep import independent_news_times
from pmdetect.markets.score import score_markets
from pmdetect.markets.series import q0_table, winner_series

END = datetime(2026, 3, 1, 12, 0, 0)


def _mk(cid, info_type="scheduled_reveal", end=END, closed=None, news=None):
    return {
        "condition_id": cid,
        "slug": cid,
        "question": cid,
        "info_type": info_type,
        "end_ts": end,
        "closed_ts": closed or end + timedelta(hours=10),
        "news_ts": news or end,
        "news_ts_method": "trade_cross_0.9",
        "outcome": "YES",
    }


def _frame(rows):
    return pl.DataFrame(rows).with_columns(
        pl.col("end_ts").cast(pl.Datetime("us")),
        pl.col("closed_ts").cast(pl.Datetime("us")),
        pl.col("news_ts").cast(pl.Datetime("us")),
    )


def _prices(cid, path, news, n_days=9):
    rows, t = [], news - timedelta(days=n_days)
    while t <= news + timedelta(hours=2):
        w = float(path((news - t).total_seconds() / 3600))
        rows.append((f"{cid}-Y", cid, "Yes", t, w))
        rows.append((f"{cid}-N", cid, "No", t, 1 - w))
        t += timedelta(hours=1)
    return pl.DataFrame(
        rows, schema=["token_id", "condition_id", "outcome_label", "ts", "price"], orient="row"
    ).with_columns(pl.col("ts").cast(pl.Datetime("us")))


def _flat(cid, lvl):
    return _prices(cid, lambda tau: lvl, END)


def _ni(ids, bad_ids=()):
    return pl.DataFrame(
        {
            "condition_id": list(ids) + list(bad_ids),
            "news_ts_indep": [END] * len(ids) + [None] * len(bad_ids),
            "indep_method": ["scheduled_end"] * len(ids) + ["none"] * len(bad_ids),
            "independent": [True] * len(ids) + [False] * len(bad_ids),
        },
        schema_overrides={"news_ts_indep": pl.Datetime("us")},
    )


def test_independence_rule_never_uses_price_crossings():
    far = END + timedelta(days=90)
    rows = [
        _mk("sched_ok"),
        _mk("deadline_type", info_type="deadline"),
        _mk("unsched", info_type="unscheduled_event"),
        _mk("placeholder_end", end=far, closed=far + timedelta(hours=5)),
        _mk("resolved_long_before", closed=END - timedelta(days=30)),
        _mk("override_me", info_type="unscheduled_event"),
    ]
    last = pl.DataFrame(
        {
            "condition_id": [r["condition_id"] for r in rows],
            "last_trade_ts": [END + timedelta(hours=5)] * 4 + [END - timedelta(days=30), END],
        }
    ).with_columns(pl.col("last_trade_ts").cast(pl.Datetime("us")))
    ov = ({}, {"override_me": {"news_ts": "2026-02-28T09:00:00Z"}})
    out = independent_news_times(_frame(rows), last, ov).to_dicts()
    by = {r["condition_id"]: r for r in out}
    assert by["sched_ok"]["indep_method"] == "scheduled_end"
    assert by["sched_ok"]["news_ts_indep"] == END
    assert by["override_me"]["indep_method"] == "override"
    assert by["override_me"]["news_ts_indep"] == datetime(2026, 2, 28, 9, 0)
    for cid in ("deadline_type", "unsched", "placeholder_end", "resolved_long_before"):
        assert by[cid]["indep_method"] == "none" and not by[cid]["independent"]
        assert by[cid]["news_ts_indep"] is None


def test_non_independent_markets_never_enter_the_fit():
    rng = np.random.default_rng(1)
    rows, prices, ids = [], [], []
    for i in range(24):
        rows.append(_mk(f"i{i}"))
        prices.append(_flat(f"i{i}", float(rng.uniform(0.2, 0.5))))
        ids.append(f"i{i}")
    for i in range(8):  # price-based markets sitting at 0.99 for the whole week
        rows.append(_mk(f"b{i}"))
        prices.append(_flat(f"b{i}", 0.99))
    markets, prices = _frame(rows), pl.concat(prices)
    series = winner_series(markets, prices)
    q0 = q0_table(series)
    it = markets.select("condition_id", "info_type")
    with_bad = fit_predict_cross(series, q0, it, independent_ids=ids)
    only_good = fit_predict_cross(series.filter(pl.col("condition_id").is_in(ids)), q0, it)
    assert set(with_bad["condition_id"]) == set(ids)  # no baseline rows for the others
    a = with_bad.sort(["condition_id", "t"])
    b = only_good.sort(["condition_id", "t"])
    assert np.allclose(a["m50"].to_numpy(), b["m50"].to_numpy())  # the bad ones had no say
    assert a["m50"].max() < 0.7  # not dragged toward 0.99


def test_independent_drift_scores_high_and_others_are_null():
    rng = np.random.default_rng(3)
    rows, prices, ids = [], [], []
    for i in range(30):
        rows.append(_mk(f"i{i}"))
        prices.append(_flat(f"i{i}", float(rng.uniform(0.25, 0.35))))
        ids.append(f"i{i}")
    rows.append(_mk("drift"))  # same start, climbs to 0.95 over the last 3 days
    prices.append(
        _prices("drift", lambda tau: 0.3 if tau > 72 else 0.3 + 0.65 * (72 - tau) / 72, END)
    )
    ids.append("drift")
    rows.append(_mk("pricebased"))
    prices.append(_flat("pricebased", 0.99))
    markets, prices = _frame(rows), pl.concat(prices)
    series = winner_series(markets, prices)
    q0 = q0_table(series)
    it = markets.select("condition_id", "info_type")
    base = fit_predict_cross(series, q0, it, independent_ids=ids)
    sc = score_markets(markets, series, base, q0, news_indep=_ni(ids, ["pricebased"]))
    sc = sc.sort("D", descending=True, nulls_last=True)
    assert sc["condition_id"][0] == "drift"
    assert sc["D"][0] > 5 * sc["D"][1]
    pb = sc.filter(pl.col("condition_id") == "pricebased").row(0, named=True)
    assert pb["D"] is None and pb["D_last24"] is None and pb["reason"] == "price_based_news_time"
    assert pb["max_jump"] is not None  # late-window size still reported for every market


def test_late_window_d_and_max_jump_on_planted_data():
    # flat ~0.3 markets; "late" jumps 0.3 -> 0.9 ten hours before news; "early" 100 h before
    rng = np.random.default_rng(5)
    rows, prices = [], []
    for i in range(25):
        rows.append(_mk(f"f{i}"))
        prices.append(_flat(f"f{i}", float(rng.uniform(0.25, 0.35))))
    rows += [_mk("late"), _mk("early")]
    prices.append(_prices("late", lambda tau: 0.3 if tau > 10 else 0.9, END))
    prices.append(_prices("early", lambda tau: 0.3 if tau > 100 else 0.9, END))
    markets, prices = _frame(rows), pl.concat(prices)
    series = winner_series(markets, prices)
    q0 = q0_table(series)
    ids = markets["condition_id"].to_list()
    it = markets.select("condition_id", "info_type")
    base = fit_predict_cross(series, q0, it, independent_ids=ids)
    sc = score_markets(markets, series, base, q0, news_indep=_ni(ids))
    r = {x["condition_id"]: x for x in sc.to_dicts()}
    assert abs(r["late"]["max_jump"] - 0.6) < 1e-6
    assert r["late"]["max_jump_hours_before"] == 10.0
    assert abs(r["early"]["max_jump"] - 0.6) < 1e-6
    assert r["early"]["max_jump_hours_before"] == 100.0
    assert r["late"]["D_last24"] > 0.2  # 10 of 24 hours sit ~0.6 above the median
    assert r["late"]["D_last24"] > 3 * r["f0"]["D_last24"]
    assert r["f0"]["max_jump"] < 1e-9 and r["f0"]["D_last24"] < 0.05
