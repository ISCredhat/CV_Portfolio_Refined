"""universe: filters at formation end, top N, schema, point-in-time perturbation, ever_top_n."""
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
from pairs import crsp, universe  # noqa: E402
from pairs.synthetic_crsp import make_crsp  # noqa: E402

MONTH = "1993-01"


@lru_cache(maxsize=None)
def data():
    return make_crsp(1)


def build(d):
    return crsp.build_panel(d["dsf_v2"], d["stocknames_v2"], d["dsedelist"])


def test_schema_filters_and_ranking():
    d = data()
    p = build(d)
    u = universe.select(p, d["stocknames_v2"], MONTH, top_n=50)
    assert len(u) == 50 and set(K.UNIVERSE_COLUMNS) <= set(u.columns)
    for c, t in K.UNIVERSE_COLUMNS.items():
        if t != "object":
            assert str(u[c].dtype) == t, (c, u[c].dtype)
    assert u["mcap"].is_monotonic_decreasing
    assert (u["price"] >= CC.MIN_PRICE_LAST_DAY).all() and (u["traded_share"] >= CC.MIN_TRADED_SHARE).all()
    assert (p.permnos[u["col"]] == u["permno"]).all()
    assert set(u["exch"]) <= {"N", "A", "Q"}
    sh = d["stocknames_v2"]
    adr = set(sh.loc[sh["usincflg"] == "N", "permno"])
    assert not adr & set(u["permno"])
    f = u.attrs["funnel"]
    assert f["in_panel"] >= f["share_filter"] >= f["price"] >= f["traded_share"] >= f["selected"]
    assert f["traded_share"] < f["share_filter"]                                  # late listings / thin stocks drop
    assert np.isin(u["ff48"], [13, 36, 44, 31, 30, 43, 23, 34]).all()


def test_share_filter_uses_spell_valid_at_formation_end():
    d = data()
    p = build(d)
    u0 = universe.select(p, d["stocknames_v2"], MONTH, top_n=500)
    pn = int(u0["permno"].iloc[0])
    nm = d["stocknames_v2"].copy()
    end = u0.attrs["formation_end"]
    nm.loc[nm["permno"] == pn, "usincflg"] = "N"                                   # flagged after nothing: valid at end
    u1 = universe.select(p, nm, MONTH, top_n=500)
    assert pn not in set(u1["permno"])
    nm2 = d["stocknames_v2"].copy()
    nm2.loc[nm2["permno"] == pn, "nameenddt"] = end - pd.Timedelta(days=3)         # no spell valid at the end
    assert pn not in set(universe.select(p, nm2, MONTH, 500)["permno"])


def test_missing_filter_column_raises():
    d = data()
    with pytest.raises(KeyError, match="issuertype"):
        universe.select(build(d), d["stocknames_v2"].drop(columns=["issuertype"]), MONTH, 50)


def test_formation_window_must_be_covered():
    d = data()
    with pytest.raises(ValueError):
        universe.select(build(d), d["stocknames_v2"], "1990-03", 50)


def test_perturbation_after_formation_end_changes_nothing():
    d = data()
    base = universe.select(build(d), d["stocknames_v2"], MONTH, top_n=80)
    cut = base.attrs["formation_end"]
    x = {k: v.copy() for k, v in d.items()}
    m = x["dsf_v2"]["dlycaldt"] > cut
    rng = np.random.default_rng(1)
    x["dsf_v2"].loc[m, "dlyprc"] *= rng.uniform(0.2, 5, m.sum())
    x["dsf_v2"].loc[m, "dlycap"] *= rng.uniform(0.1, 9, m.sum())
    x["dsf_v2"].loc[m, "dlyvol"] = 0.0
    x["dsf_v2"].loc[m, "primaryexch"] = "A"
    nm = x["stocknames_v2"]
    later = nm["namedt"] > cut
    nm.loc[later, "usincflg"] = "N"
    nm.loc[later, "siccd"] = 9999
    got = universe.select(build(x), nm, MONTH, top_n=80)
    pd.testing.assert_frame_equal(base.drop(columns="col"), got.drop(columns="col"))


def test_ever_top_n_covers_selected_months():
    d = data()
    months = ["1993-01", "1994-01", "1995-01"]
    ever = universe.ever_top_n(d["dsf_v2"], d["stocknames_v2"], months, top_n=40)
    p = build(d)
    for m in months:
        u = universe.select(p, d["stocknames_v2"], m, top_n=40)
        assert set(u["permno"]) <= set(ever)
    assert len(ever) < len(p.permnos)
