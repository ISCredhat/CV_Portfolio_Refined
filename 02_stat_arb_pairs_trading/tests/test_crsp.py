"""crsp: panel, TR index, missing-return policy, delisting folding/imputation, bars, short ban, B2 cross-check."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import contracts as K  # noqa: E402
from pairs import crsp  # noqa: E402
from pairs.synthetic_crsp import make_crsp  # noqa: E402


@lru_cache(maxsize=None)
def data():
    return make_crsp(1)


@lru_cache(maxsize=None)
def panel():
    d = data()
    return crsp.build_panel(d["dsf_v2"], d["stocknames_v2"], d["dsedelist"])


def test_panel_shape_sorted_and_tr_is_cumprod():
    p = panel()
    assert np.all(np.diff(p.permnos) > 0) and p.days.is_monotonic_increasing
    j = 5
    col = p.permnos[j]
    s = data()["dsf_v2"].query("permno == @col").sort_values("dlycaldt")
    r = np.r_[0.0, s["dlyret"].to_numpy()[1:]]
    r = np.nan_to_num(r)
    got = p.tr[p.days.get_indexer(s["dlycaldt"]), j]
    np.testing.assert_allclose(got, np.cumprod(1 + r), rtol=1e-10)


def test_split_keeps_tr_continuous_while_raw_price_halves():
    p = panel()
    sh = data()["dsf_v2"].groupby("permno")["shrout"].agg(lambda s: s.max() / s.min())
    j = np.searchsorted(p.permnos, sh[sh > 1.9].index[0])
    ratio = p.prc[:, j] / p.tr[:, j]
    assert np.nanmax(ratio) / np.nanmin(ratio) > 1.9
    assert np.nanmax(np.abs(np.diff(np.log(p.tr[:, j])))) < 0.4                  # no jump in TR


def test_missing_return_policy_counts_and_gap_rule():
    d = data()["dsf_v2"].copy()
    pn = d["permno"].iloc[1000]
    rows = d.index[d["permno"] == pn]
    k = rows[50]
    d.loc[k, "dlyret"] = np.nan
    d.loc[k, "dlyretmissflg"] = "XX"
    p = crsp.build_panel(d)
    j = np.searchsorted(p.permnos, pn)
    i = p.days.get_loc(d.loc[k, "dlycaldt"])
    assert np.isnan(p.tr[i, j]) and p.diag["missing_ret_rows"] == 1
    assert p.diag["missing_ret_lost"] == 1                                        # next row is a single-day return
    nxt = d.loc[rows[51], "dlyret"]
    prev = p.tr[i - 1, j]
    np.testing.assert_allclose(p.tr[i + 1, j], prev * (1 + nxt), rtol=1e-10)


def test_required_columns_fail_loudly():
    d = data()["dsf_v2"].drop(columns=["dlyret"])
    with pytest.raises(KeyError, match="dlyret"):
        crsp.normalise_dsf(d)
    with pytest.raises(KeyError):
        crsp.name_spells(data()["stocknames_v2"].drop(columns=["namedt"]))


def test_delisting_events_imputation_and_dead_mask():
    d, p = data(), panel()
    ev = crsp.delisting_events(p)
    assert set(K.DELIST_COLUMNS) <= set(ev.columns)
    truth = d["truth_events"].set_index("permno")
    assert len(ev) == len(truth) and set(ev["permno"]) == set(truth.index)
    miss_perf = truth[(truth["dlstcd"] // 100 == 5) & truth["dlret_true"].isna()].index
    e = ev.set_index("permno")
    assert len(miss_perf) > 0
    for pn in miss_perf:
        assert e.at[pn, "dlret_imputed"] and np.isclose(e.at[pn, "dlret"], CC.DELIST_IMPUTE[e.at[pn, "exch"]])
    present = truth[truth["dlret_true"].notna()].index
    assert not e.loc[present, "dlret_imputed"].any()
    assert e["is_merger"].sum() == (truth["dlstcd"].between(*CC.MERGER_CODES)).sum()
    assert p.diag["dlret_imputed"] == len(miss_perf)
    # dead = last row of each delisted security, nothing else
    assert p.dead.sum() == len(ev)
    for r in ev.itertuples():
        j = np.searchsorted(p.permnos, r.permno)
        assert p.dead[p.days.get_loc(r.last_date), j]
        assert np.isfinite(p.ret[p.days.get_loc(r.last_date), j])                 # imputed / folded: never NaN
    alive = np.setdiff1d(p.permnos, ev["permno"])
    assert not p.dead[:, np.searchsorted(p.permnos, alive)].any()


def test_ciz_delisting_row_return_counts_without_a_price():
    """CIZ puts the delisting on its own row (trading day after the last trade) with price DP, 0 (DA) or missing
    (DM). Its return must reach the TR index whatever the price: TR(dead) / TR(last trade) - 1 = delisting return."""
    d, p = data(), panel()
    x = d["dsf_v2"]
    y = x[x["dlydelflg"] == "Y"].set_index("permno")
    assert {"DA", "DM", "DP"} <= set(y["dlyprcflg"])
    ev = crsp.delisting_events(p).set_index("permno")
    seen_zero = False
    for pn, row in y.iterrows():
        j = np.searchsorted(p.permnos, pn)
        t = p.days.get_loc(row["dlycaldt"])
        assert p.dead[t, j] and np.isfinite(p.tr[t, j])
        prev = t - 1
        while not np.isfinite(p.tr[prev, j]):
            prev -= 1
        assert np.isclose(p.tr[t, j] / p.tr[prev, j] - 1, ev.at[pn, "dlret"])
        seen_zero |= p.tr[t, j] == 0
    assert seen_zero                                                           # a worthless delisting is in the data


def test_imputation_uses_exchange_and_bounds():
    d = data()
    dsf = d["dsf_v2"].copy()
    ev = d["truth_events"]
    pn = ev[(ev["dlstcd"] // 100 == 5) & ev["dlret_true"].isna()]["permno"].iloc[0]
    dl = d["dsedelist"]
    for ex, want in (("Q", -0.55), ("A", -0.30)):
        x = dsf.copy()
        x.loc[x["permno"] == pn, "primaryexch"] = ex
        p = crsp.build_panel(x, delist=dl)
        assert np.isclose(crsp.delisting_events(p).set_index("permno").at[pn, "dlret"], want)


def test_build_daily_bars_modes():
    p = panel()
    bt = {m: crsp.build_daily_bars(p, m) for m in K.EXEC_MODES}
    c = bt["t1_close"]
    np.testing.assert_array_equal(c.exec[:-1], c.close[1:])
    np.testing.assert_array_equal(c.exec[-1], c.close[-1])
    assert (c.fill_day[:-1] == np.arange(len(c.days) - 1) + 1).all()
    np.testing.assert_array_equal(bt["t_close"].exec, bt["t_close"].close)
    assert (bt["t_close"].fill_day == np.arange(len(c.days))).all()
    o = bt["t1_open"]
    assert o.open_fill.any() and not o.open_fill[:-1][p.days[1:] < pd.Timestamp(CC.OPEN_FIRST_MONTH + "-01")].any()
    t, j = np.argwhere(o.open_fill)[500]
    np.testing.assert_allclose(o.exec[t, j], p.tr[t + 1, j] * p.open[t + 1, j] / p.prc[t + 1, j])
    nf = np.argwhere(~o.open_fill[:-1] & np.isfinite(o.close[1:]))[:50]
    for t, j in nf:
        assert o.exec[t, j] == o.close[t + 1, j]
    live = np.isfinite(p.tr)                       # stale (carried) cells are tested in test_ffill_within_life
    live &= ~(p.dead & ~np.isfinite(p.prc))         # unpriced delisting rows: test_unpriced_delisting_row_...
    np.testing.assert_allclose(c.factor[live], (p.prc / p.tr)[live])
    assert c.exch.dtype == object and c.fresh.dtype == bool and c.dt_days == 1.0 and c.bars_per_hour is None


def test_tr_open_on_dividend_day_is_not_raw_ratio_of_prices():
    p = panel()
    o = crsp.build_daily_bars(panel(), "t1_open")
    t, j = np.argwhere(o.open_fill)[100]
    prev_close_tr = o.close[t, j]
    assert np.isclose(o.exec[t, j] / o.close[t + 1, j], p.open[t + 1, j] / p.prc[t + 1, j])
    assert prev_close_tr > 0


def test_short_ban_proxy():
    d = make_crsp(2, start="2008-06-02", end="2008-12-31", n_permno=60)
    p = crsp.build_panel(d["dsf_v2"], d["stocknames_v2"], d["dsedelist"])
    with pytest.raises(ValueError, match="ban"):
        crsp.build_daily_bars(p, "t1_close")
    ff = pd.Series(np.where(np.arange(len(p.permnos)) % 2 == 0, 44, 13), index=p.permnos)
    b = crsp.build_daily_bars(p, "t1_close", ff48=ff)
    ban = (p.days >= "2008-09-19") & (p.days <= "2008-10-08")
    banned = (ff == 44).to_numpy()
    assert not b.short_ok[np.ix_(ban, banned)].any() and b.short_ok[np.ix_(ban, ~banned)].all()
    assert b.short_ok[~ban].all()


def test_delisting_crosscheck_b2_and_detects_errors():
    d = data()
    t = crsp.delisting_crosscheck(d["dsf_v2"], d["dsedelist"], d["dsf_legacy"])
    s = crsp.crosscheck_summary(t)
    assert s["passes_99pct"] and s["n_events"] == len(d["dsedelist"]) and s["match_rate"] == 1.0
    assert s["missing_dlret_by_code_decade"]
    bad = d["dsedelist"].copy()
    bad.loc[bad["dlret"].notna().idxmax(), "dlret"] += 0.2
    assert (~crsp.delisting_crosscheck(d["dsf_v2"], bad, d["dsf_legacy"]).query("not dlret_missing")["match"]).sum() == 1
    weak = crsp.delisting_crosscheck(d["dsf_v2"], d["dsedelist"])                  # no legacy dsf: weaker mode
    assert (weak["mode"] == "dlret_only").all()


def test_perturbation_nothing_before_cutoff_depends_on_later_data():
    d = data()["dsf_v2"]
    cut = pd.Timestamp("1993-06-30")
    x = d.copy()
    late = x["dlycaldt"] > cut
    rng = np.random.default_rng(0)
    x.loc[late, "dlyret"] = rng.normal(0, 0.05, late.sum())
    x.loc[late, "dlyprc"] *= 1.7
    p0 = crsp.build_panel(d)
    p1 = crsp.build_panel(x)
    n = int((p0.days <= cut).sum())
    np.testing.assert_array_equal(p0.tr[:n], p1.tr[:n])
    np.testing.assert_array_equal(p0.prc[:n], p1.prc[:n])
    np.testing.assert_array_equal(p0.fresh[:n], p1.fresh[:n])


def test_chunk_loading_roundtrip(tmp_path):
    d = data()["dsf_v2"]
    (tmp_path / "dsf_v2").mkdir()
    for y, g in d.groupby(d["dlycaldt"].dt.year):
        g.to_parquet(tmp_path / "dsf_v2" / f"y{y}.parquet")
    keep = d["permno"].unique()[:10]
    got = crsp.load_dsf(tmp_path, "1992-03-01", "1993-02-28", permnos=keep)
    assert set(got["permno"]) <= set(keep) and got["dlycaldt"].min() >= pd.Timestamp("1992-03-01")
    with pytest.raises(FileNotFoundError):
        crsp.load_dsf(tmp_path, "1989-01-01", "1990-12-31")


def test_chunk_loading_without_parquet_filters(tmp_path, monkeypatch):
    """Same rows when pyarrow.dataset is unavailable (seen on Windows with Smart App Control blocking a DLL)."""
    d = data()["dsf_v2"]
    (tmp_path / "dsf_v2").mkdir()
    for y, g in d.groupby(d["dlycaldt"].dt.year):
        g.to_parquet(tmp_path / "dsf_v2" / f"y{y}.parquet")
    keep = d["permno"].unique()[:10]
    want = crsp.load_dsf(tmp_path, "1992-03-01", "1993-02-28", permnos=keep)
    cols = ["dlycaldt", "dlyret"]
    want_cols = crsp.read_chunks(tmp_path, "dsf_v2", [1992], columns=cols, permnos=keep)
    real = pd.read_parquet

    def no_filters(path, *a, **k):
        if k.get("filters") is not None:
            raise ValueError("the 'filters' keyword is not supported when the pyarrow.dataset module is not available")
        return real(path, *a, **k)

    monkeypatch.setattr(crsp.pd, "read_parquet", no_filters)
    got = crsp.load_dsf(tmp_path, "1992-03-01", "1993-02-28", permnos=keep)
    pd.testing.assert_frame_equal(got, want)
    got_cols = crsp.read_chunks(tmp_path, "dsf_v2", [1992], columns=cols, permnos=keep)
    assert list(got_cols.columns) == cols
    pd.testing.assert_frame_equal(got_cols.reset_index(drop=True), want_cols.reset_index(drop=True))


def test_ffill_within_life_never_fills_before_listing_or_after_death():
    from pairs import crsp as R2
    a = np.array([[np.nan, 1.0], [1.0, np.nan], [np.nan, 2.0], [2.0, np.nan], [np.nan, np.nan]])
    f = R2.ffill_within_life(a)
    assert np.isnan(f[0, 0]) and f[2, 0] == 1.0 and f[3, 0] == 2.0 and np.isnan(f[4, 0])   # col 0 dies at row 3
    assert f[1, 1] == 1.0 and np.isnan(f[3, 1]) and np.isnan(f[4, 1])                       # col 1 dies at row 2


def test_unpriced_delisting_row_keeps_the_last_share_factor():
    """CIZ 'DA' (price 0, return present) and 'DM' rows are exits at the TR close; their share factor is the last
    priced row's, so the raw value for ticks and commissions is last price x (1 + return), never NaN (a NaN there
    made the cost of a merger exit non-finite)."""
    from pairs.synthetic_crsp import make_crsp
    d = make_crsp(5, "1990-01-02", "1997-12-31", 120)
    p = crsp.build_panel(d["dsf_v2"], d["stocknames_v2"], d["dsedelist"])
    y = d["dsf_v2"][d["dsf_v2"]["dlydelflg"] == "Y"]
    unpriced = y[~(pd.to_numeric(y["dlyprc"], errors="coerce") > 0)]
    assert set(unpriced["dlyprcflg"]) >= {"DA", "DM"}
    for mode in ("t1_close", "t_close"):
        b = crsp.build_daily_bars(p, mode, ff48={})
        for e in unpriced.itertuples():
            t, j = p.days.get_loc(pd.Timestamp(e.dlycaldt)), int(np.searchsorted(p.permnos, e.permno))
            assert p.dead[t, j] and np.isfinite(b.close[t, j]) and np.isfinite(b.factor[t, j])
            assert b.factor[t, j] == b.factor[t - 1, j]
            assert np.isclose(b.close[t, j] * b.factor[t, j], p.prc[t - 1, j] * b.close[t, j] / b.close[t - 1, j])
