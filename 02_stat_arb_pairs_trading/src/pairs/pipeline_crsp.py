"""Part A pipeline: CRSP daily data -> point-in-time universe -> daily pair selection -> frozen backtest ->
pre-registered hypotheses H1-H4 -> GGR / Do & Faff replication. Writes tables to ``results/crsp/``.

Stages (each is a plain function so the end-to-end test can run them on synthetic CRSP data):
    inputs      panel (only PERMNOs ever in the top N), names, log TR prices, DailyBars per execution mode
    universes   universe.select for every trading month (decided at formation end)
    candidates  selection_daily.run_months (within FF48, daily bars, joblib over months)
    selections  coint (strategy), distance (H3 benchmark), N_PAIRS and N_PAIRS_SECONDARY
    backtests   frozen configuration under each execution mode, the 18 variants under the primary mode,
                distance selection under the primary mode
    hypotheses  H1 (stationary bootstrap), H2 (time-shifted null, NW), H3 (Ledoit-Wolf), H4 (gross trend), Holm
    ggr         Do & Faff (2010) Table 1 / Table 2 statistics on the GGR rules

The frozen configuration is read from ``config/preregistration.yaml`` and asserted; nothing here re-selects it.
Robustness runs (universe sizes, industry definitions, placebo, capacity, negative controls, bridge tests) are
listed in the plan and are added after the primary run, logged as such.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as B
from . import config_crsp as CC
from . import contracts as K
from . import costs_daily as CD
from . import crsp as R
from . import ggr as G
from . import industries as IND
from . import nulls as N
from . import selection_daily as SD
from . import stats as ST
from . import universe as U


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


@dataclass
class Inputs:
    panel: R.Panel
    names: pd.DataFrame
    logp: pd.DataFrame
    bars: dict[str, K.DailyBars] = field(default_factory=dict)


def frozen(cfg: dict | None = None) -> dict:
    """The registered configuration. With no YAML (tests), the config_crsp copy is used; prereg.load() asserts
    the two agree whenever the YAML exists."""
    if cfg is not None:
        return cfg
    try:
        from . import prereg
        return prereg.frozen_configuration()
    except FileNotFoundError:
        return dict(CC.FROZEN)


def ff48_asof(names: pd.DataFrame, date) -> pd.Series:
    """PERMNO -> FF48 from CRSP historical SIC valid on `date` (used for the 2008 short-ban proxy)."""
    a = R.names_asof(R.name_spells(names), pd.Timestamp(date))
    return pd.Series(IND.ff48_from_sic(a["siccd"].to_numpy()), index=a["permno"].to_numpy())


def build_inputs(dsf: pd.DataFrame, names: pd.DataFrame, delist: pd.DataFrame | None,
                 exec_modes=(CC.EXEC_PRIMARY, CC.EXEC_CO_REPORTED, CC.EXEC_NO_DELAY)) -> Inputs:
    panel = R.build_panel(dsf, names, delist)
    ban = ff48_asof(names, CC.SHORT_BAN[0]) if (panel.days >= pd.Timestamp(CC.SHORT_BAN[0])).any() else None
    bars = {m: R.build_daily_bars(panel, m, ban) for m in exec_modes}
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = pd.DataFrame(np.log(panel.tr), index=panel.days)
    return Inputs(panel, names, logp, bars)


def universes(inp: Inputs, months, top_n: int = CC.UNIVERSE_TOP_N) -> dict[str, pd.DataFrame]:
    return {str(m): U.select(inp.panel, inp.names, m, top_n=top_n) for m in months}


def select(cands: dict[str, pd.DataFrame], method: str, n_pairs: int, seed: int = 0) -> dict[str, pd.DataFrame]:
    return {k: SD.choose(v, method, n=n_pairs, seed=seed) for k, v in cands.items()}


def backtest(bars: K.DailyBars, sel: dict, half_spread: pd.DataFrame, months, n_pairs: int,
             methods, entries, exits, cost_mult: float = 1.0) -> dict:
    return B.run(bars, sel, half_spread, methods, entries, exits, cost_mult=cost_mult, months=months,
                 cost_fn=CD.make_cost_fn(bars), n_pairs=n_pairs)


def _col(frame: pd.DataFrame, fz: dict) -> pd.Series:
    return frame[(fz["method"], fz["entry"], fz["exit"])]


def _window(s: pd.Series, period: tuple[str, str]) -> pd.Series:
    a, b = pd.Period(period[0], "M").start_time, pd.Period(period[1], "M").end_time
    return s[(s.index >= a) & (s.index <= b)]


def hypotheses(net: pd.Series, gross: pd.Series, net_distance: pd.Series, h2_excess: pd.Series,
               periods: dict = CC.PERIODS, seed: int = CC.SEED) -> dict:
    """H1-H4 exactly as registered, plus the Holm correction across them."""
    bo = periods["backward_oos"]
    x1 = _window(net, bo)
    h1 = ST.sharpe_stationary_bootstrap(x1.to_numpy(), reps=CC.BOOT_REPS, seed=seed)
    h1["p"] = h1["p_le_0"]
    h2 = N.h2_test(h2_excess)
    h3 = ST.ledoit_wolf_sharpe_diff(x1.to_numpy(), _window(net_distance, bo).to_numpy())
    h3["p"] = h3["p_one_sided"]
    g_m = gross.groupby(gross.index.to_period("M")).sum()
    h4 = ST.trend_test_with_break(g_m, CC.TREND_BREAK, CC.NW_LAGS_TREND)
    h4["p"] = h4["slope_p_neg"]
    p = [h1["p"], h2["p"], h3["p"], h4["p"]]
    adj, rej = ST.holm(np.nan_to_num(p, nan=1.0))
    out = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    for k, a, r in zip(out, adj, rej):
        out[k]["p_holm"], out[k]["reject_holm"] = float(a), bool(r)
    out["H2"]["supported"] = bool(out["H2"].get("supported", False) and rej[1])
    return out


def run_main(inp: Inputs, months, out_dir: Path | None = None, top_n: int = CC.UNIVERSE_TOP_N,
             n_jobs: int = 1, fz: dict | None = None, null_kind: str = "shifted") -> dict:
    """Primary run. Returns every table; writes them under `out_dir` when given."""
    fz = frozen(fz)
    months = pd.PeriodIndex(months, freq="M")
    log(f"universes ({len(months)} months, top {top_n})")
    unis = universes(inp, months, top_n)
    log("candidates")
    cands = SD.run_months(inp.logp, unis, months, n_jobs=n_jobs)
    log("spreads")
    try:
        calib = CD.calibrate_spread_map(inp.panel)
    except ValueError as e:                 # panel without 1993-96 NYSE/AMEX quotes: pre-1993 uses the fallback
        log(f"  no spread calibration ({e}); pre-1993 NYSE/AMEX spreads use the fallback rule")
        calib = None
    hs = CD.half_spread_table(inp.panel, months, calib)
    sel = {n: select(cands, "coint", n) for n in (CC.N_PAIRS, CC.N_PAIRS_SECONDARY)}
    sel_dist = select(cands, "distance", CC.N_PAIRS)
    one = ([fz["method"]], [fz["entry"]], [fz["exit"]])
    res: dict[str, dict] = {}
    log("backtests")
    for mode, bars in inp.bars.items():
        res[f"frozen_{mode}_n{CC.N_PAIRS}"] = backtest(bars, sel[CC.N_PAIRS], hs, months, CC.N_PAIRS, *one)
    prim = inp.bars[CC.EXEC_PRIMARY]
    res[f"frozen_{CC.EXEC_PRIMARY}_n{CC.N_PAIRS_SECONDARY}"] = backtest(prim, sel[CC.N_PAIRS_SECONDARY], hs, months,
                                                                       CC.N_PAIRS_SECONDARY, *one)
    res["variants"] = backtest(prim, sel[CC.N_PAIRS], hs, months, CC.N_PAIRS, CC.VARIANTS["method"],
                               CC.VARIANTS["entry"], CC.VARIANTS["exit"])
    res["distance"] = backtest(prim, sel_dist, hs, months, CC.N_PAIRS, *one)
    log(f"H2 null ({null_kind})")
    exc = N.monthly_excess(inp.logp, cands, kind=null_kind)
    key = f"frozen_{CC.EXEC_PRIMARY}_n{CC.N_PAIRS}"
    hyp = hypotheses(_col(res[key]["net"], fz), _col(res[key]["gross"], fz), _col(res["distance"]["net"], fz),
                     exc["excess_pp"])
    perf = performance_table(res, fz)
    funnel = pd.DataFrame([{"month": k, "n_universe": len(unis[k]), "n_candidates": len(v),
                            "n_p5": int((v["p_eg"] <= CC.ALPHA).sum()) if len(v) else 0,
                            "n_bh": int(v["bh_pass"].sum()) if len(v) else 0,
                            "n_eligible": int(v["eligible"].sum()) if len(v) else 0,
                            "n_selected": len(sel[CC.N_PAIRS][k])} for k, v in cands.items()])
    out = {"hypotheses": hyp, "performance": perf, "funnel": funnel, "h2_monthly": exc, "results": res,
           "frozen": fz}
    if out_dir is not None:
        write(out, Path(out_dir))
    return out


def performance_table(res: dict, fz: dict) -> pd.DataFrame:
    """Net and gross Sharpe, return and trade counts per series and period."""
    rows = []
    for name, r in res.items():
        if name == "variants":
            continue
        for per, rng in {"all": (None, None), **CC.PERIODS}.items():
            n, g = _col(r["net"], fz), _col(r["gross"], fz)
            if rng[0]:
                n, g = _window(n, rng), _window(g, rng)
            if len(n) < 2:
                continue
            sd = n.std(ddof=1)
            rows.append({"series": name, "period": per, "days": len(n),
                         "net_ann_return": float(n.mean() * CC.TRADING_DAYS),
                         "net_sharpe": float(n.mean() / sd * np.sqrt(CC.TRADING_DAYS)) if sd > 0 else np.nan,
                         "gross_ann_return": float(g.mean() * CC.TRADING_DAYS),
                         "n_trades": int(len(r["trades"])) if per == "all" else np.nan})
    return pd.DataFrame(rows)


def run_ggr(tr: pd.DataFrame, first_start: str, last_start: str, wait: int, traded=None, dead=None,
            permco=None, periods: dict | None = None) -> dict:
    """GGR rules on `tr`; returns the Do & Faff Table 1 and Table 2 statistics for `periods`. Prices are carried
    over no-price days inside a stock's life (as in the main backtest); `traded` (e.g. ``panel.fresh``) applies
    the GGR/Do & Faff formation filter that drops any stock with a day without a trade or valid return."""
    tr = pd.DataFrame(R.ffill_within_life(tr.to_numpy(float)), index=tr.index, columns=tr.columns)
    res = G.run(tr, first_start, last_start, G.GGRConfig(wait=wait), dead=dead, traded=traded, permco=permco)
    periods = periods or {"1962-1988": (CC.GGR_FIRST_TRADE_MONTH, "1988-12"), "1989-2002": ("1989-01", "2002-12"),
                          "2003-2009": ("2003-01", CC.GGR_LAST_TRADE_MONTH),
                          "full": (CC.GGR_FIRST_TRADE_MONTH, CC.GGR_LAST_TRADE_MONTH)}
    return {"table1": G.dofaff_table1(res.monthly, periods), "table2": G.dofaff_table2(res.pairs, periods),
            "monthly": res.monthly}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def write(out: dict, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "hypotheses.json").write_text(json.dumps(_jsonable(out["hypotheses"]), indent=2, default=str))
    (d / "frozen_config.json").write_text(json.dumps(out["frozen"], indent=2))
    out["performance"].to_csv(d / "performance.csv", index=False)
    out["funnel"].to_csv(d / "selection_funnel.csv", index=False)
    out["h2_monthly"].to_csv(d / "h2_monthly_excess.csv")
    for name, r in out["results"].items():
        r["net"].to_csv(d / f"daily_net_{name}.csv.gz")
        if len(r["trades"]):
            r["trades"].to_csv(d / f"trades_{name}.csv.gz", index=False)
