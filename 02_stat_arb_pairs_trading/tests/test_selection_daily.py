"""selection_daily: equivalence with the per-pair loop, filters, point-in-time perturbation, choose()."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import contracts as CT  # noqa: E402
from pairs import selection as S_OLD  # noqa: E402
from pairs import selection_daily as SD  # noqa: E402

DAYS = pd.bdate_range("2022-10-03", periods=300)
MONTH = pd.Period("2023-07", "M")
CUT = MONTH.start_time


def _ou(rng, n, phi=0.8, sd=0.01):
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = phi * u[i - 1] + rng.standard_normal() * sd
    return u


def make_logp(seed=0):
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    base = lambda: np.log(40) + np.cumsum(rng.standard_normal(n)) * 0.015
    cols = [base() for _ in range(12)]
    cols[1] = 1.3 * cols[0] + _ou(rng, n)            # (0,1) cointegrated, group 10
    cols[3] = 0.9 * cols[2] + _ou(rng, n)            # (2,3) cointegrated and same PERMCO
    cols[7] = 1.1 * cols[6] + _ou(rng, n)            # (6,7) cointegrated, group 20
    cols[11] = 1.0 * cols[10] + _ou(rng, n)          # (10,11) cointegrated but FF48 "Other"
    return pd.DataFrame(np.column_stack(cols), index=DAYS)


def make_universe():
    permno = 1001 + np.arange(12)
    permco = permno.copy()
    permco[3] = permco[2]
    return pd.DataFrame({"permno": permno, "permco": permco, "col": np.arange(12), "mcap": 1e9, "price": 40.0,
                         "exch": "N", "siccd": 1000, "ff48": [10] * 6 + [20] * 4 + [48] * 2,
                         "sic2": 10, "gics_ind": None, "traded_share": 1.0})


def _cands(logp=None, uni=None, **kw):
    return SD.test_month(make_logp() if logp is None else logp, make_universe() if uni is None else uni, MONTH, **kw)


def test_schema_groups_and_flags():
    df = _cands()
    for c in CT.CANDIDATE_COLUMNS:
        assert c in df.columns, c
    pairs = set(zip(df["y"], df["x"])) | set(zip(df["x"], df["y"]))
    assert (1011, 1012) not in pairs                                  # FF48 "Other" never pairs
    assert (1001, 1007) not in pairs                                  # different industries never pair
    assert len(df) == 15 + 6                                          # C(6,2) in group 10, C(4,2) in group 20
    assert df["month"].eq("2023-07").all() and df["n_universe"].eq(12).all()
    row = lambda a, b: df[((df.y == a) & (df.x == b)) | ((df.y == b) & (df.x == a))].iloc[0]
    assert row(1001, 1002).p_eg < 0.05 and row(1001, 1002).eligible
    sp = row(1003, 1004)
    assert sp.same_permco and not sp.eligible and sp.p_eg < 0.05      # flagged, excluded from eligible
    assert df["same_permco"].sum() == 1
    assert row(1007, 1008).eligible
    assert not df.loc[df.p_eg > 0.05, "eligible"].any()
    assert df["johansen"].notna().sum() == (df["p_eg"] <= 0.05).sum()
    assert not _cands(cfg=SD.SelectionCfg(exclude_same_permco=False)).query("same_permco")["eligible"].eq(False).all()


def test_matches_per_pair_loop():
    logp, uni = make_logp(1), make_universe()
    df = SD.test_month(logp, uni, MONTH)
    rows = SD.formation_window(logp.index, MONTH, 6)
    W = logp.to_numpy()[rows]
    for _, r in df.sample(10, random_state=0).iterrows():
        i, j = sorted((r.iy, r.ix))
        old = S_OLD.test_pair(W[:, i], W[:, j], bar_hours=1.0, maxlag=12)
        assert abs(old["p_eg"] - r.p_eg) < 1e-8 and abs(old["eg_stat"] - r.eg_stat) < 1e-8
        assert abs(old["beta"] - r.beta) < 1e-8 and abs(old["sigma"] - r.sigma) < 1e-10
        assert abs(old["var_beta"] - r.var_beta) < 1e-10 and abs(old["var_alpha"] - r.var_alpha) < 1e-12
        assert abs(old["hl_hours"] - r.hl_days) < 1e-8 and abs(old["ssd"] - r.ssd) < 1e-8
        assert abs(old["mean_ly"] - r.mean_ly) < 1e-10 and abs(old["sd_lx"] - r.sd_lx) < 1e-10
        assert (r.iy == (j if old["flip"] else i))
        if not np.isnan(r.hurst):
            assert abs(old["hurst"] - r.hurst) < 1e-10


def test_min_obs_drops_and_counts_pairs():
    logp = make_logp()
    rows = SD.formation_window(logp.index, MONTH, 6)
    logp.iloc[rows[:40], 5] = np.nan                                   # stock 5 has 40 of ~126 days missing
    logp.iloc[rows[::2], 9] = np.nan                                   # stock 9 has half missing
    df = _cands(logp=logp)
    n_form = len(rows)
    assert df.attrs["min_obs"] == int(np.ceil(0.8 * n_form))
    assert (df["n_obs"] >= df.attrs["min_obs"]).all()
    assert df.attrs["n_dropped_min_obs"] == 3 + 5                      # stock 9 loses 3 pairs, stock 5 (40 gone: 86 < 101) 5
    assert not ((df.y == 1010) | (df.x == 1010)).any()
    assert df.attrs["n_pairs"] == 21


def test_perturbation_after_formation_end_changes_nothing():
    logp = make_logp()
    base = _cands(logp=logp)
    pert = logp.copy()
    pert.loc[pert.index >= CUT] += np.random.default_rng(5).standard_normal((int((pert.index >= CUT).sum()), 12)) * 0.5
    pd.testing.assert_frame_equal(base, _cands(logp=pert))
    # a change on the last formation day does matter (the test is not vacuous)
    pert2 = logp.copy()
    pert2.iloc[int(np.nonzero(logp.index < CUT)[0][-1]), 0] += 0.3
    assert not base["p_eg"].equals(_cands(logp=pert2)["p_eg"])
    # run_months gives the same tables, serially and in parallel, with and without the future
    months = [MONTH, MONTH - 1]
    unis = {str(m): make_universe() for m in months}
    r1 = SD.run_months(logp, unis, months, n_jobs=1)
    r2 = SD.run_months(pert, unis, months, n_jobs=2)
    for m in months:
        pd.testing.assert_frame_equal(r1[str(m)], r2[str(m)])
    pd.testing.assert_frame_equal(r1[str(MONTH)], base)


def test_choose_rules():
    df = _cands()
    c = SD.choose(df, "coint")
    assert c["eligible"].all() and (np.diff(c["p_eg"].to_numpy()) >= 0).all()
    assert not c["same_permco"].any()
    # per-stock cap: build a table where stock 1 is in 3 eligible pairs
    t = df.iloc[:5].copy()
    t[["eligible", "plausible", "same_permco"]] = [True, True, False]
    t["y"], t["x"] = 1, [2, 3, 4, 5, 6]
    t["p_eg"] = [0.001, 0.002, 0.003, 0.004, 0.005]
    assert len(SD.choose(t, "coint", n=5)) == 2
    t["y"], t["x"] = [1, 1, 2, 3, 4], [2, 3, 3, 4, 5]
    assert SD.choose(t, "coint", n=5).index.tolist() == [t.index[i] for i in (0, 1, 2, 4)]   # (3,4) skipped: stock 3 full
    d = SD.choose(df, "distance", n=3)
    allowed = df[df.plausible & ~df.same_permco].sort_values("ssd")
    assert d["ssd"].iloc[0] == allowed["ssd"].iloc[0] and len(d) == 3
    r1, r2 = SD.choose(df, "random", n=4, seed=3), SD.choose(df, "random", n=4, seed=3)
    assert r1.index.equals(r2.index) and not SD.choose(df, "random", n=4, seed=4).index.equals(r1.index)
    try:
        SD.choose(df, "bogus")
        raise AssertionError
    except ValueError:
        pass


def test_alternative_industry_columns_and_no_industry():
    uni = make_universe()
    n_none = len(_cands(uni=uni, group_col="none"))
    assert n_none == 66                                                # C(12, 2): no constraint
    uni["sic2"] = [1] * 4 + [2] * 8
    assert len(_cands(uni=uni, group_col="sic2")) == 6 + 28
    assert len(_cands(uni=uni, group_col="gics_ind")) == 0             # all None: nobody may pair


def test_smoke_with_synthetic_crsp_universe_selection_and_backtest():
    """Workstream A's synthetic data through universe -> selection_daily -> backtest.run with its cost function."""
    from pairs import backtest as B
    from pairs import costs_daily as CD
    from pairs import crsp as R
    from pairs import synthetic_crsp as SC
    from pairs import universe as U
    d = SC.make_crsp(1, "2000-01-03", "2002-06-28", 100)
    panel = R.build_panel(d["dsf_v2"], d["stocknames_v2"], d.get("dsedelist"))
    bars = R.build_daily_bars(panel, "t1_close")
    CT.validate_bars(bars)
    months = pd.period_range("2001-01", "2001-12", freq="M")
    logp = pd.DataFrame(np.log(bars.close), index=bars.days)
    unis = {str(m): U.select(panel, d["stocknames_v2"], m, top_n=80) for m in months}
    cands = SD.run_months(logp, unis, months)
    sel = {k: SD.choose(v) for k, v in cands.items()}
    assert sum(len(v) for v in sel.values()) > 0
    out = B.run(bars, sel, CD.half_spread_table(panel, months), ["rolling"], [2.0], [0.5], months=months,
                cost_fn=CD.make_cost_fn(bars))
    assert len(out["trades"]) > 0 and np.isfinite(out["net"].to_numpy()).all()
    assert set(out["trades"]["reason"]) <= set(B.REASONS.values())
