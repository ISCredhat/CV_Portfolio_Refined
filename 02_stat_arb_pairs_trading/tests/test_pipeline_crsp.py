"""End to end on synthetic CRSP: build inputs -> universes -> selection -> backtests -> H1-H4 -> GGR tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import pipeline_crsp as P  # noqa: E402
from pairs import synthetic_crsp as SC  # noqa: E402


def test_end_to_end_synthetic(tmp_path):
    d = SC.make_crsp(3, "1991-01-02", "1997-12-31", 140)
    inp = P.build_inputs(d["dsf_v2"], d["stocknames_v2"], d.get("dsedelist"))
    months = pd.period_range("1996-01", "1997-12", freq="M")
    out = P.run_main(inp, months, out_dir=tmp_path, top_n=100)
    h = out["hypotheses"]
    assert set(h) == {"H1", "H2", "H3", "H4"}
    for k in h:
        assert 0.0 <= h[k]["p_holm"] <= 1.0 and h[k]["p_holm"] >= h[k]["p"] - 1e-12
    fz = out["frozen"]
    assert (fz["method"], fz["entry"], fz["exit"]) == (CC.FROZEN["method"], CC.FROZEN["entry"], CC.FROZEN["exit"])
    perf = out["performance"]
    assert {f"frozen_{m}_n{CC.N_PAIRS}" for m in ("t1_close", "t1_open", "t_close")} <= set(perf["series"])
    assert out["funnel"]["n_selected"].sum() > 0
    # the variants grid is the 18 registered combinations
    assert out["results"]["variants"]["net"].shape[1] == 18
    for f in ["hypotheses.json", "performance.csv", "selection_funnel.csv", "h2_monthly_excess.csv"]:
        assert (tmp_path / f).exists()
    # injected cointegrated pairs should pass the screen more often than the time-shifted null
    assert out["h2_monthly"]["observed"].mean() >= out["h2_monthly"]["null"].mean()


def test_ggr_tables_on_synthetic():
    d = SC.make_crsp(4, "1990-01-02", "1994-12-30", 80)
    inp = P.build_inputs(d["dsf_v2"], d["stocknames_v2"], d.get("dsedelist"), exec_modes=("t1_close",))
    tr = pd.DataFrame(inp.panel.tr, index=inp.panel.days, columns=inp.panel.permnos)
    per = {"a": ("1991-02", "1992-12"), "b": ("1993-01", "1994-06")}
    r = P.run_ggr(tr, "1991-01", "1993-12", wait=1, dead=inp.panel.dead, traded=inp.panel.fresh, permco=inp.panel.permco, periods=per)
    t1 = r["table1"]
    assert len(t1) == 2 and np.isfinite(t1.select_dtypes("number").to_numpy()).any()
