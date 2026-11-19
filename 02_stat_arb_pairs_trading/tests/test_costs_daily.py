"""costs_daily: ticks, commissions, spread table (sources, fallback, point in time), cost function."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import costs_daily as CD  # noqa: E402
from pairs import crsp  # noqa: E402
from pairs.synthetic_crsp import make_crsp  # noqa: E402

MONTHS = pd.period_range("1990-02", "1995-12", freq="M")


@lru_cache(maxsize=None)
def data():
    return make_crsp(1)


def build(d):
    return crsp.build_panel(d["dsf_v2"], d["stocknames_v2"], d["dsedelist"])


@lru_cache(maxsize=None)
def base():
    p = build(data())
    cal = CD.calibrate_spread_map(p)
    return p, cal, CD.half_spread_table(p, MONTHS, cal)


def test_tick_table_by_exchange_and_date():
    assert CD.tick_size("N", "1996-01-02") == 0.125
    assert CD.tick_size("N", "1997-06-24") == 0.0625 and CD.tick_size("N", "1997-06-23") == 0.125
    assert CD.tick_size("A", "1997-05-07") == 0.0625 and CD.tick_size("Q", "1997-05-07") == 0.125
    assert CD.tick_size("Q", "2001-04-08") == 0.0625 and CD.tick_size("Q", "2001-04-09") == 0.01
    assert CD.tick_size("N", "2001-01-29") == 0.01
    t = CD.tick_size(np.array(["N", "Q", "A"], dtype=object), np.array(["1999-01-04"] * 3, dtype="datetime64[ns]"))
    assert t.tolist() == [0.0625] * 3


def test_commission_schedule_do_faff_2012():
    got = CD.commission_bps(np.array([1950, 1963, 1974, 1990, 1997, 2004, 2007, 2009, 2020]))
    assert got.tolist() == [70, 70, 72, 20, 10, 10, 7, 9, 9]


def test_table_never_nan_and_counts():
    p, cal, h = base()
    assert h.shape == (len(MONTHS), len(p.permnos)) and np.isfinite(h.values).all() and (h.values > 0).all()
    s = h.attrs["sources"]
    assert (s.sum(axis=1) == len(p.permnos)).all()
    assert s.loc["1990-02":"1992-12", "calibrated"].sum() > 0 and s.loc["1993-02":, "calibrated"].sum() == 0
    assert s["quoted"].sum() > 0 and s["fallback"].sum() > 0
    assert cal["n"] > 1000 and cal["r2"] > 0.05


def test_quoted_value_and_fallback_rule():
    p, cal, h = base()
    m = pd.Period("1994-06", freq="M")
    prev = (p.days.to_period("M") == m - 1)
    j = int(np.argmax(np.isfinite(p.bid[prev]).sum(0)))
    q = (p.ask - p.bid) / (0.5 * (p.ask + p.bid))
    want = 0.5 * np.nanmedian(q[prev, j]) * 1e4
    assert np.isclose(h.loc[m].iloc[j], want)
    # a stock with no data in the previous month gets 1.25 x group median
    no = ~np.isfinite(p.prc[prev]).any(axis=0)
    assert no.any()
    vals = h.loc[m].to_numpy()[no]
    assert (vals > np.nanmin(h.loc[m].to_numpy()[~no]) * 0.5).all()
    ex = pd.Series(p.exch[prev][-1])
    mult = CC.SPREAD_FALLBACK_MULT
    ok = ~no
    for jj in np.nonzero(no)[0][:5]:
        assert vals[0] >= 0 and mult == 1.25 and np.isfinite(h.loc[m].iloc[jj])


def test_spread_table_point_in_time_perturbation():
    p, cal, h = base()
    d = data()
    x = {k: v.copy() for k, v in d.items()}
    cut = pd.Timestamp("1994-06-01")
    late = x["dsf_v2"]["dlycaldt"] >= cut                   # month m = 1994-06 uses only May and earlier
    rng = np.random.default_rng(3)
    for c in ("dlybid", "dlyask", "dlyhigh", "dlylow", "dlyprc"):
        x["dsf_v2"].loc[late, c] *= rng.uniform(0.5, 2.0, late.sum())
    h2 = CD.half_spread_table(build(x), MONTHS, cal)
    m = pd.Period("1994-06", freq="M")
    pd.testing.assert_frame_equal(h.loc[:m], h2.loc[:m], check_exact=False, rtol=1e-12)
    assert not np.allclose(h.loc[m + 1:].values, h2.loc[m + 1:].values)


def test_estimators_positive_and_track_true_spread():
    p, cal, h = base()
    terms = CD._daily_terms(p)
    rows = np.nonzero(p.days.to_period("M") == pd.Period("1994-03", freq="M"))[0]
    st = CD._stats(terms, rows)
    assert np.isfinite(st["ar"]).sum() > 100 and (st["ar"][np.isfinite(st["ar"])] >= 0).all()
    assert (st["cs"][np.isfinite(st["cs"])] >= 0).all()


def test_calibration_needs_enough_data():
    d = make_crsp(5, start="1991-01-02", end="1992-06-30", n_permno=30)
    with pytest.raises(ValueError, match="calibrate"):
        CD.calibrate_spread_map(build(d))


def _bars(px=5.0, day="1997-01-06", exch="N", open_fill=False):
    from pairs import contracts as K
    d0 = pd.Timestamp(day)
    days = pd.DatetimeIndex([d0, d0 + pd.Timedelta(days=1), d0 + pd.Timedelta(days=2)])
    n = np.ones((3, 2))
    return K.DailyBars(permnos=np.array([1, 2]), days=days, close=n, exec=n, fill_day=np.array([1, 2, 2]),
                       fresh=np.ones((3, 2), bool), factor=n * px, dead=np.zeros((3, 2), bool),
                       short_ok=np.ones((3, 2), bool), exch=np.full((3, 2), exch, dtype=object),
                       open_fill=np.full((3, 2), open_fill))


def test_cost_fn_floor_no_cap_commission_impact_and_open_multiplier():
    f = CD.make_cost_fn(_bars())
    px = np.full((1, 2), 5.0)
    c = f(np.array([1.0, 200.0]), px, np.array([0]), np.array([0, 1]), 1.0)
    comm = CD.commission_bps(np.array([1997]))[0]                                  # 1997: 10 bps (Do & Faff 2012)
    half_tick = 0.5 * 0.125 / 5.0 * 1e4
    assert np.isclose(c[0, 0], half_tick + comm + CC.IMPACT_BPS)                  # floor binds, 125 bp > 50 bp cap
    assert np.isclose(c[0, 1], 200.0 + comm + CC.IMPACT_BPS)                       # 200 bps not capped
    assert c[0, 0] > 50
    assert np.allclose(f(np.array([1.0, 200.0]), px, np.array([0]), np.array([0, 1]), 2.0), 2 * c)
    fo = CD.make_cost_fn(_bars(open_fill=True))
    co = fo(np.array([1.0, 200.0]), px, np.array([0]), np.array([0, 1]), 1.0)
    assert np.isclose(co[0, 1] - c[0, 1], 0.5 * 200.0)                             # spread x1.5, rest unchanged
    flat = CD.make_cost_fn(_bars(), flat_commission=CC.COMMISSION_FLAT_SENSITIVITY)
    cf = flat(np.array([1.0, 200.0]), px, np.array([0]), np.array([0, 1]), 1.0)
    assert np.isclose(cf[0, 1], 200.0 + 0.0035 / 5.0 * 1e4 + 1.0)
    with pytest.raises(ValueError, match="NaN"):
        f(np.array([np.nan, 1.0]), px, np.array([0]), np.array([0, 1]), 1.0)


def test_cost_fn_tick_changes_with_exchange_and_date():
    px = np.full((1, 2), 20.0)
    hs = np.array([0.1, 0.1])
    old = CD.make_cost_fn(_bars(20.0, "1996-01-02", "N"))(hs, px, np.array([0]), np.array([0, 1]), 1.0)
    new = CD.make_cost_fn(_bars(20.0, "2005-01-04", "N"))(hs, px, np.array([0]), np.array([0, 1]), 1.0)
    assert old[0, 0] > new[0, 0]
