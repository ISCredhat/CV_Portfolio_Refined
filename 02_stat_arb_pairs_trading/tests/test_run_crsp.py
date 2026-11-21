"""scripts/run_crsp.py and the memory-saving loaders it uses: the frozen gate, the year-at-a-time top-N PERMNOs,
column-limited loading and the block-wise GGR replication (same answers as one panel over the whole sample)."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import crsp as R  # noqa: E402
from pairs import ggr as G  # noqa: E402
from pairs import pipeline_crsp as P  # noqa: E402
from pairs import prereg  # noqa: E402
from pairs import universe as U  # noqa: E402
from pairs.synthetic_crsp import make_crsp  # noqa: E402


@lru_cache(maxsize=None)
def data():
    return make_crsp(5, "1990-01-02", "1997-12-31", 120)


def with_a_delisting(d: dict, cut: str = "1994-09-15") -> dict:
    """A copy where one ordinary share that trades every day delists mid-sample (CIZ 'Y' row, -30%), so the
    GGR comparison reaches a delisting close."""
    d = dict(d)
    dsf = d["dsf_v2"].copy()
    dsf["dlycaldt"] = pd.to_datetime(dsf["dlycaldt"])
    w = dsf[(dsf["dlycaldt"] >= pd.Timestamp(cut) - pd.DateOffset(months=18)) & (dsf["dlycaldt"] <= pd.Timestamp(cut))]
    traded = ((w["dlyprcflg"] == "TR") & (w["dlyvol"] > 0)).groupby(w["permno"]).agg(["all", "size"])
    ns = set(d["stocknames_v2"].loc[d["stocknames_v2"]["sharetype"] == "NS", "permno"])
    ok = traded[traded["all"] & (traded["size"] == traded["size"].max())].index
    p = int([q for q in ok if q in ns][0])
    dsf = dsf[~((dsf["permno"] == p) & (dsf["dlycaldt"] > pd.Timestamp(cut)))]
    last = dsf.index[dsf["permno"] == p][-1]
    dsf.loc[last, ["dlydelflg", "dlyret"]] = ["Y", -0.3]
    d["dsf_v2"] = dsf.reset_index(drop=True)
    return d


def with_a_halt(d: dict, start: str = "1994-03-15", end: str = "1996-02-15") -> dict:
    """A copy where another ordinary share that trades every day before `start` keeps CIZ rows without prices or
    returns (dlyprcflg 'NT', as after leaving its exchange) from `start` to `end`, then trades again: the no-price
    span crosses several block ends."""
    d = dict(d)
    dsf = d["dsf_v2"].copy()
    dsf["dlycaldt"] = pd.to_datetime(dsf["dlycaldt"])
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    w = dsf[(dsf["dlycaldt"] >= a - pd.DateOffset(months=18)) & (dsf["dlycaldt"] < a)]
    traded = ((w["dlyprcflg"] == "TR") & (w["dlyvol"] > 0)).groupby(w["permno"]).agg(["all", "size"])
    ns = set(d["stocknames_v2"].loc[d["stocknames_v2"]["sharetype"] == "NS", "permno"])
    late = set(dsf.loc[dsf["dlycaldt"] >= pd.Timestamp("1997-06-30"), "permno"])
    ok = traded[traded["all"] & (traded["size"] == traded["size"].max())].index
    p = int([q for q in ok if q in ns and q in late][-1])
    m = (dsf["permno"] == p) & (dsf["dlycaldt"] >= a) & (dsf["dlycaldt"] <= b)
    dsf.loc[m, ["dlyprc", "dlyret", "dlyvol"]] = np.nan
    dsf.loc[m, "dlyprcflg"] = "NT"
    d["dsf_v2"] = dsf.reset_index(drop=True)
    d["halted"] = p
    return d


def write_chunks(raw: Path, d: dict) -> Path:
    """The download layout: dsf_v2 one parquet per calendar year, other keys all.parquet."""
    (raw / "dsf_v2").mkdir(parents=True)
    dsf = d["dsf_v2"]
    for y, g in dsf.groupby(pd.to_datetime(dsf["dlycaldt"]).dt.year):
        g.to_parquet(raw / "dsf_v2" / f"y{y}.parquet", index=False)
    for key in ("stocknames_v2", "dsedelist"):
        (raw / key).mkdir()
        d[key].to_parquet(raw / key / "all.parquet", index=False)
    return raw


def load_script():
    spec = importlib.util.spec_from_file_location("run_crsp_script", ROOT / "scripts" / "run_crsp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frozen_cfg(frozen: bool):
    cfg = copy.deepcopy(prereg.load(verify=False))
    cfg["meta"]["frozen"] = frozen
    return lambda verify=True: cfg


def test_panel_columns_reads_only_what_the_panel_uses(tmp_path):
    raw = write_chunks(tmp_path, data())
    cols = R.panel_columns(raw)
    assert set(cols) <= set(R.PANEL_COLUMNS) and set(R.DSF_REQUIRED) <= set(cols)
    assert "securitytype" not in cols and "dlyretx" not in cols


def test_ever_top_n_chunked_equals_the_one_frame_version(tmp_path):
    d = data()
    raw = write_chunks(tmp_path, d)
    months = pd.period_range("1991-01", "1997-12", freq="M")
    full = U.ever_top_n(d["dsf_v2"], d["stocknames_v2"], months, top_n=30)
    got = U.ever_top_n_chunked(raw, d["stocknames_v2"], months, top_n=30)
    assert len(full) > 30 and np.array_equal(full, got)


def test_ggr_blocks_equal_one_panel_over_the_whole_sample(tmp_path):
    d = with_a_halt(with_a_delisting(data()))
    raw = write_chunks(tmp_path, d)
    names, delist = d["stocknames_v2"], d["dsedelist"]
    first, last = "1991-03", "1997-06"
    per = {"a": ("1991-04", "1994-06"), "b": ("1994-07", "1997-12")}
    g0 = str((pd.Period(first, "M") - 12).start_time.date())
    g1 = str(((pd.Period(last, "M") + 6).start_time - pd.Timedelta(days=1)).date())
    gp = R.build_panel(R.load_dsf(raw, g0, g1), names, delist)
    tr = pd.DataFrame(gp.tr, index=gp.days, columns=gp.permnos)
    uni = P.ggr_share_universe(gp.days, gp.permnos, names)
    cfgs = {f"s{a}_{b}_w{w}": G.GGRConfig(wait=w, selection=(a, b)) for (a, b) in ((1, 5), (6, 10)) for w in (1, 0)}
    cfgs["all_w1"] = G.GGRConfig(wait=1, selection=(1, None))       # every pair: reaches stocks that delist
    blocks = P.run_ggr_blocks(raw, names, delist, cfgs, first, last, periods=per, block_months=7,
                              lead_in_months=1, lead_out_months=2)
    n_delist = 0
    for k, c in cfgs.items():
        one = P.run_ggr(tr, first, last, c.wait, traded=gp.fresh, dead=gp.dead, permco=gp.permco, periods=per,
                        universe=uni, selection=c.selection)
        b = blocks[k]
        assert len(one["monthly"]) > 60
        pd.testing.assert_frame_equal(b["monthly"], one["monthly"], rtol=1e-12, atol=1e-14)
        pd.testing.assert_frame_equal(b["table1"], one["table1"], rtol=1e-10, atol=1e-14)
        pd.testing.assert_frame_equal(b["table2"], one["table2"], rtol=1e-10, atol=1e-14)
        assert b["stats"] == pytest.approx(one["stats"], rel=1e-12, nan_ok=True)
        n_delist += one["stats"]["delisting_exits"]
        if k == "all_w1":                                 # the halted stock is held across a block end
            tt = G.run(pd.DataFrame(R.ffill_within_life(gp.tr), index=gp.days, columns=gp.permnos), first, last,
                       c, dead=gp.dead, universe=uni, traded=gp.fresh, permco=gp.permco).trades
            h = tt[(tt["y"] == d["halted"]) | (tt["x"] == d["halted"])]
            assert ((h["open"] < pd.Timestamp("1994-03-15")) & (h["close"] > pd.Timestamp("1994-03-15"))).any()
    assert n_delist > 0                                   # delisting exits are exercised


def test_ggr_universe_keeps_ordinary_common_shares_only():
    d = data()
    gp = R.build_panel(d["dsf_v2"], d["stocknames_v2"], d["dsedelist"])
    u = P.ggr_share_universe(gp.days, gp.permnos, d["stocknames_v2"])("1993-01")
    spells = R.names_asof(R.name_spells(d["stocknames_v2"]), pd.Timestamp("1992-12-31"), gp.permnos)
    adr = set(spells.loc[spells["sharetype"] != "NS", "permno"])
    assert adr and not (set(gp.permnos[u]) & adr) and u.sum() > 50


def test_run_crsp_runs_a_frozen_registration_and_labels_dry_runs(tmp_path, monkeypatch):
    raw = write_chunks(tmp_path / "raw", data())
    res = tmp_path / "results"
    monkeypatch.setattr(CC, "WRDS_RAW", raw)
    monkeypatch.setattr(CC, "RESULTS_CRSP", res)
    script = load_script()
    # unfrozen and no --allow-unfrozen: refused before anything is read or written
    monkeypatch.setattr(script.prereg, "load", frozen_cfg(False))
    assert script.main(["--top-n", "60"]) == 2 and not res.exists()
    # unfrozen dry run is labelled as such
    monkeypatch.setattr(script.P, "run_ggr_blocks", lambda *a, **k: {})
    assert script.main(["--top-n", "60", "--ggr-only", "--allow-unfrozen"]) == 0
    assert json.loads((res / "top60_DRYRUN_unfrozen" / "run_info_ggr.json").read_text())["prereg_frozen"] is False
    # frozen (meta.frozen in the YAML): the primary run goes ahead and writes to top{N}
    monkeypatch.setattr(script.prereg, "load", frozen_cfg(True))
    assert script.main(["--top-n", "60", "--first", "1994-07", "--last", "1997-12"]) == 0
    out = res / "top60"
    assert json.loads((out / "run_info.json").read_text())["prereg_frozen"] is True
    h = json.loads((out / "hypotheses.json").read_text())
    assert set(h) == {"H1", "H2", "H3", "H4"}
    assert (out / "performance.csv").exists()
