"""synthetic_crsp: determinism, CIZ columns and the injected features."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import coint as K  # noqa: E402
from pairs import crsp  # noqa: E402
from pairs.synthetic_crsp import make_crsp  # noqa: E402


@lru_cache(maxsize=None)
def data(seed=1):
    return make_crsp(seed)


def test_deterministic_and_seed_changes_output():
    a, b, c = make_crsp(3, n_permno=40), make_crsp(3, n_permno=40), make_crsp(4, n_permno=40)
    pd.testing.assert_frame_equal(a["dsf_v2"], b["dsf_v2"])
    assert not a["dsf_v2"]["dlyprc"].equals(c["dsf_v2"]["dlyprc"])


def test_columns_and_scale():
    d = data()
    assert d["dsf_v2"]["permno"].nunique() == 300
    for c in crsp.DSF_REQUIRED + ("dlyopen", "dlybid", "dlyask", "dlyhigh", "dlylow", "dlydelflg", "dlyprcflg",
                                  "dlyretmissflg", "dlyretdurflg", "primaryexch", "siccd", "shrout", "dlycap"):
        assert c in d["dsf_v2"].columns, c
    assert {"permno", "namedt", "nameenddt", "sharetype", "usincflg", "issuertype", "conditionaltype",
            "tradingstatusflg", "securitysubtype", "primaryexch", "siccd"} <= set(d["stocknames_v2"].columns)
    assert {"permno", "dlstdt", "dlstcd", "dlret"} <= set(d["dsedelist"].columns)
    assert {"gvkey", "indfrom", "indthru", "gind"} <= set(d["co_hgic"].columns)
    trade_rows = d["dsf_v2"]["dlydelflg"] != "Y"
    assert (d["dsf_v2"].loc[trade_rows, "dlyprc"] > 0).all()
    y = d["dsf_v2"][~trade_rows]                              # CIZ-style delisting rows: DP / DA (price 0) / DM
    assert set(y["dlyprcflg"]) == {"DP", "DA", "DM"} and y["dlyvol"].isna().all()
    assert (y.loc[y["dlyprcflg"] == "DA", "dlyprc"] == 0).all() and y.loc[y["dlyprcflg"] == "DM", "dlyprc"].isna().all()


def test_features_present():
    d = data()
    x = d["dsf_v2"]
    assert (x["dlyprcflg"] == "BA").any() and (x.loc[x["dlyprcflg"] == "BA", "dlyvol"] == 0).all()
    assert x.groupby("permno")["primaryexch"].nunique().gt(1).any()               # exchange changes
    assert x.groupby("permno")["siccd"].nunique().gt(1).any()                     # SIC changes
    assert d["stocknames_v2"].groupby("permno").size().gt(1).any()
    assert x.groupby("permco")["permno"].nunique().eq(2).sum() == 8               # share classes
    pre = x[(x["dlycaldt"] < "1993-01-01") & (x["primaryexch"] != "Q")]
    assert pre["dlybid"].isna().all()
    assert x.loc[x["dlycaldt"] < "1992-07-01", "dlyopen"].isna().all() and x["dlyopen"].notna().any()
    ev = d["truth_events"]
    assert (ev["dlstcd"].between(500, 599)).any() and (ev["dlstcd"].between(200, 399)).any()
    assert ev["dlret_true"].isna().any() and ev["dlret_true"].notna().any()
    assert (x["dlyretdurflg"] == "MD").any()


def test_tr_differs_from_price_with_splits_and_dividends():
    p = crsp.build_panel(data()["dsf_v2"])
    ratio = p.prc / p.tr
    moved = np.nanmax(ratio, axis=0) / np.nanmin(ratio, axis=0)
    assert (moved > 1.01).sum() > 50                                              # dividends or splits


def test_injected_pairs_are_cointegrated():
    d = data()
    p = crsp.build_panel(d["dsf_v2"])
    pv = []
    for r in d["truth_pairs"].itertuples():
        iy, ix = np.searchsorted(p.permnos, [r.y, r.x])
        pv.append(K.engle_granger(np.log(p.tr[:, iy]), np.log(p.tr[:, ix]), maxlag=6)["pvalue"])
    assert np.median(pv) < 0.05 and max(pv) < 0.2, pv
