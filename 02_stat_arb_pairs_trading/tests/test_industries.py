"""industries: FF48 map, SIC2, GICS as of a date."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import industries as I  # noqa: E402
from pairs.synthetic_crsp import make_crsp  # noqa: E402


def test_file_hash_and_structure():
    assert I.sha256_file(I.SICCODES48_PATH) == I.SICCODES48_SHA256
    table, names = I.load_ff48()
    assert len(names) == 48 and table.shape == (10000,)
    with pytest.raises(ValueError, match="sha256"):
        I.load_ff48.__wrapped__(str(I.SICCODES48_PATH), "0" * 64)


def test_known_sic_codes_and_other():
    got = I.ff48_from_sic(np.array([2834, 3674, 6021, 4911, 1311, 7372, 3711, 5812, 0, 9999, 4955]))
    assert got.tolist() == [13, 36, 44, 31, 30, 34, 23, 43, 48, 48, 48]   # 7372 software is FF48 34 BusSv in the official file
    assert I.ff48_from_sic(np.array([np.nan, -5, 123456])).tolist() == [48, 48, 48]
    assert I.sic2(np.array([2834, 3674.0, np.nan])).tolist() == [28, 36, 0]


def test_synthetic_sics_map_to_declared_industries():
    from pairs.synthetic_crsp import INDUSTRIES
    for sic, ff, _ in INDUSTRIES:
        assert I.ff48_from_sic(np.array([sic]))[0] == ff


def test_gics_asof_rules():
    d = make_crsp(1, n_permno=80)
    hg, ccm = d["co_hgic"], d["ccmxpf_lnkhist"]
    pm = ccm["lpermno"].unique()
    early = I.gics_asof(pm, "1991-06-01", hg, ccm)
    assert early.isna().all()                                                      # before the history starts
    mid = I.gics_asof(pm, "1992-06-01", hg, ccm)
    assert mid.notna().sum() >= 70
    late = I.gics_asof(pm, "1994-06-01", hg, ccm)
    switch = hg.groupby("gvkey").size().loc[lambda s: s > 1].index
    assert len(switch) > 0
    changed = (mid != late) & mid.notna() & late.notna()
    assert changed.sum() >= 1
    # non-qualifying links ('J'/'LD') are ignored: removing every qualifying link for a permno gives None
    q = ccm[~((ccm["lpermno"] == pm[0]) & ccm["linkprim"].isin(["P", "C"]))]
    assert I.gics_asof([pm[0]], "1994-06-01", hg, q).isna().all()
    # a link that has ended is not valid
    e = ccm.copy()
    e.loc[e["lpermno"] == pm[1], "linkenddt"] = pd.Timestamp("1992-01-01")
    assert I.gics_asof([pm[1]], "1994-06-01", hg, e).isna().all()
