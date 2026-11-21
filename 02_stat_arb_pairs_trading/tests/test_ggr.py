"""ggr.py: hand-computed toy examples, SSD selection, overlap averaging, point in time, Do & Faff tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import ggr as G  # noqa: E402
from pairs import stats as ST  # noqa: E402

PI = np.array([1.0, 1.12, 1.15, 1.05, 0.98, 1.0, 1.14, 1.20])[:, None]
PJ = np.ones((8, 1))
SD = np.array([0.05])                                 # trigger at |Pi - Pj| > 0.10


def test_no_wait_hand_computed_round_trips():
    pnl, tr = G.simulate_pairs(PI, PJ, SD, 2.0, wait=0)
    assert [(t["open"], t["close"], t["reason"]) for t in tr] == [(1, 4, "crossing"), (6, 7, "end")]
    assert np.isclose(tr[0]["ret"], -(0.98 / 1.12 - 1)) and np.isclose(tr[1]["ret"], -(1.20 / 1.14 - 1))
    assert tr[0]["side"] == 1 and tr[0]["days"] == 3
    assert np.isclose(pnl[:, 0].sum(), tr[0]["ret"] + tr[1]["ret"])              # daily marks add up to the trades
    assert np.isclose(pnl[2, 0], -(1.15 - 1.12) / 1.12) and pnl[1, 0] == 0 and pnl[5, 0] == 0


def test_one_day_wait_opens_and_closes_a_day_later():
    _, tr = G.simulate_pairs(PI, PJ, SD, 2.0, wait=1, wait_close=True)
    assert [(t["open"], t["close"]) for t in tr] == [(2, 5)]                      # the day-7 signal cannot open
    assert np.isclose(tr[0]["ret"], -(1.0 / 1.15 - 1))
    _, tr2 = G.simulate_pairs(PI, PJ, SD, 2.0, wait=1, wait_close=False)
    assert np.isclose(tr2[0]["ret"], -(0.98 / 1.15 - 1)) and tr2[0]["close"] == 4


def test_long_leg_and_short_leg_pnl_sign():
    pi = np.array([1.0, 0.88, 0.95, 1.0, 1.0])[:, None]                          # i cheap: long i, short j
    pnl, tr = G.simulate_pairs(pi, np.ones((5, 1)), SD, 2.0, wait=0)
    assert tr[0]["side"] == -1 and tr[0]["close"] == 3 and np.isclose(tr[0]["ret"], 1.0 / 0.88 - 1)


def test_delisting_closes_the_pair_and_nan_while_open_raises():
    pj = np.array([1.0, 1.0, 1.0, 0.9, np.nan, np.nan, np.nan, np.nan])[:, None]
    dead = np.zeros((8, 1), bool)
    dead[3] = True
    pnl, tr = G.simulate_pairs(PI, pj, SD, 2.0, wait=0, dead_j=dead)
    assert len(tr) == 1 and tr[0]["reason"] == "delisting" and tr[0]["close"] == 3
    assert np.isclose(tr[0]["ret"], -(1.05 / 1.12 - 1) + (0.9 / 1.0 - 1))
    assert np.isfinite(pnl).all() and np.isclose(pnl.sum(), tr[0]["ret"])
    try:
        G.simulate_pairs(PI, pj, SD, 2.0, wait=0)                                   # no dead flag: NaN while open
        raise AssertionError
    except AssertionError as e:
        assert "NaN" in str(e)


def test_ssd_selection_and_share_class_exclusion():
    rng = np.random.default_rng(0)
    base = np.cumprod(1 + rng.standard_normal((250, 5)) * 0.01, 0)
    Pn = base.copy()
    Pn[:, 1] = Pn[:, 0] * (1 + 1e-4 * rng.standard_normal(250))                   # columns 0, 1 nearly identical
    Pn /= Pn[0]
    i, j, d = G.select_pairs(Pn, (1, 1))
    assert (i[0], j[0]) == (0, 1)
    assert np.isclose(d[0], ((Pn[:, 0] - Pn[:, 1]) ** 2).sum())
    i2, j2, _ = G.select_pairs(Pn, (1, 1), permco=np.array([7, 7, 1, 2, 3]))      # same firm: dropped
    assert (i2[0], j2[0]) != (0, 1)
    ia, ja, dd = G.select_pairs(Pn, (1, None))
    assert len(ia) == 10 and (np.diff(dd) >= 0).all()
    assert len(G.select_pairs(Pn, (3, 5))[0]) == 3
    assert len(G.select_pairs(Pn, (1, None), group=np.array([0, 0, 0, 1, 1]))[0]) == 3 + 1


def test_overlapping_portfolios_are_equal_weight_averaged():
    idx = lambda a, n: pd.period_range(a, periods=n, freq="M")
    p1 = pd.DataFrame({"committed": [0.01, 0.02, 0.03], "employed": [0.02, 0.04, 0.06]}, index=idx("2020-01", 3))
    p2 = pd.DataFrame({"committed": [0.03, 0.05], "employed": [0.06, 0.10]}, index=idx("2020-02", 2))
    out = G.aggregate({pd.Period("2020-01"): p1, pd.Period("2020-02"): p2})
    assert np.allclose(out["committed"], [0.01, 0.025, 0.04]) and np.allclose(out["employed"], [0.02, 0.05, 0.08])
    assert out["n_portfolios"].tolist() == [1, 2, 2]


def _market(seed=0, n=8, months=("2018-01", "2020-12")):
    days = pd.bdate_range(months[0] + "-01", months[1] + "-28")
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((len(days), 3)) * 0.01
    r = np.column_stack([common[:, k % 3] + rng.standard_normal(len(days)) * 0.01 for k in range(n)])
    return pd.DataFrame(100 * np.cumprod(1 + r, 0), index=days, columns=1000 + np.arange(n))


def test_run_end_to_end_committed_vs_employed_and_stats():
    tr = _market()
    cfg = G.GGRConfig(selection=(1, 6), wait=0)
    res = G.run(tr, "2019-01", "2020-06", cfg)
    assert res.monthly["n_portfolios"].max() == 6 and res.monthly.index.is_monotonic_increasing
    assert res.stats["portfolios"] == 18 and res.stats["round_trips"] == len(res.trades)
    assert (res.portfolios["n_traded"] <= res.portfolios["n_selected"]).all()
    assert len(res.pairs) == res.portfolios["n_selected"].sum()
    # employed = committed x selected / traded inside each portfolio, so |employed| >= |committed| overall
    assert res.monthly["employed"].abs().sum() >= res.monthly["committed"].abs().sum() - 1e-12
    tot = res.trades.groupby("start")["ret"].sum()
    ports = res.portfolios.set_index("start")
    by_port_committed = tot / ports["n_selected"]
    assert np.isfinite(by_port_committed).all()
    w1 = G.run(tr, "2019-01", "2019-03", G.GGRConfig(selection=(1, 6), wait=1))
    assert w1.trades["days"].min() >= 0


def test_point_in_time_selection_and_traded_mask():
    tr = _market(1)
    X = tr.to_numpy()
    start = pd.Period("2019-07", "M")
    cfg = G.GGRConfig(selection=(1, 5), wait=0)
    base = G.run_portfolio(X, tr.index, start, cfg)
    pert = X.copy()
    cut = np.nonzero(tr.index >= start.start_time)[0][3]                          # after the first trading days
    pert[cut:] *= np.random.default_rng(0).uniform(0.9, 1.1, pert[cut:].shape)
    p2 = G.run_portfolio(pert, tr.index, start, cfg)
    assert (base["ia"] == p2["ia"]).all() and (base["ja"] == p2["ja"]).all() and np.allclose(base["ssd"], p2["ssd"])
    # a no-trade day in the formation window removes the stock from the universe
    traded = np.ones_like(X, bool)
    traded[int(np.nonzero(tr.index >= "2018-09-03")[0][0]), base["ia"][0]] = False
    p3 = G.run_portfolio(X, tr.index, start, cfg, traded=traded)
    assert base["ia"][0] not in set(p3["ia"]) | set(p3["ja"])
    # a perturbation after the end of the trading window changes nothing
    late = X.copy()
    late[np.nonzero(tr.index >= (start + 6).start_time)[0]] *= 2
    p4 = G.run_portfolio(late, tr.index, start, cfg)
    assert np.allclose(base["pnl"], p4["pnl"])


def test_dofaff_table1_and_table2():
    rng = np.random.default_rng(3)
    idx = pd.period_range("1989-01", "2002-12", freq="M")
    e = 0.004 + 0.02 * rng.standard_normal(len(idx))
    monthly = pd.DataFrame({"employed": e, "committed": e * 0.9, "n_portfolios": 6}, index=idx)
    t1 = G.dofaff_table1(monthly, {"a": ("1989-01", "1995-12"), "full": ("1989-01", "2002-12")})
    sub = e[:84]
    ols = ST.newey_west_ols(sub, np.ones((84, 1)), 6)
    assert np.isclose(t1.loc["a", "mean_employed"], sub.mean()) and np.isclose(t1.loc["a", "t_nw"], ols["t"][0])
    assert np.isclose(t1.loc["a", "median_employed"], np.median(sub))
    assert np.isclose(t1.loc["a", "sharpe_monthly"], sub.mean() / sub.std(ddof=1))           # D&F basis
    assert np.isclose(t1.loc["a", "sharpe_annual"], sub.mean() / sub.std(ddof=1) * np.sqrt(12))
    assert np.isclose(t1.loc["a", "share_neg"], (sub < 0).mean()) and t1.loc["full", "months"] == len(idx)
    assert np.isclose(t1.loc["full", "mean_committed"], (e * 0.9).mean())
    pairs = pd.DataFrame({"start": pd.PeriodIndex(["2000-01"] * 10, freq="M"), "y": 1, "x": 2,
                          "opened": [False, True, True, True, True, True, True, True, True, True],
                          "round_trips": [0, 0, 0, 1, 1, 1, 2, 3, 4, 1],
                          "payoff": [0, -.06, -.12, .03, .03, .03, .12, .12, .12, .06],
                          "rt_days": [np.nan, np.nan, np.nan, 30, 40, 50, 20, 20, 20, 30]})
    t2 = G.dofaff_table2(pairs, {"p": ("1999-01", "2001-12")})
    sh = t2["share"].to_numpy()
    assert np.allclose(sh, [0.1, 0.2, 0.4, 0.3]) and np.isclose(t2.loc[("p", 2), "avg_monthly_payoff"], -0.015)
    assert np.isclose(t2.loc[("p", 3), "avg_rt_days"], 37.5) and np.isclose(t2.loc[("p", 4), "avg_rt_days"], 20.0)


def test_run_equals_assembled_parts_of_two_blocks():
    tr = _market(2)
    cfg = G.GGRConfig(selection=(1, 6), wait=1)
    one = G.run(tr, "2019-01", "2020-06", cfg)
    two = G.assemble([G.run_parts(tr, "2019-01", "2019-09", cfg), G.run_parts(tr, "2019-10", "2020-06", cfg)])
    pd.testing.assert_frame_equal(one.monthly, two.monthly)
    pd.testing.assert_frame_equal(one.pairs, two.pairs)
    pd.testing.assert_frame_equal(one.trades, two.trades)
    assert one.stats == two.stats
    with pytest.raises(ValueError):
        G.assemble([G.run_parts(tr, "2019-01", "2019-09", cfg), G.run_parts(tr, "2019-09", "2019-10", cfg)])
