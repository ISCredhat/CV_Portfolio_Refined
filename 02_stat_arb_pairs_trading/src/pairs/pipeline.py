"""Compute stages (slow parts, cached in data/processed/):

    select  : monthly candidate tables (within-industry and unconstrained)
    costs   : monthly spread estimates per ticker
    grid    : walk-forward backtests of every (bar size, hedge method, entry z, exit z)
    extras  : experiments on the configuration chosen on the validation period
              (selection-method placebo, latency, persistence re-tests)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import backtest as B
from . import config as C
from . import costs as TC
from . import data as D
from . import selection as S


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Context:
    """Lazily loaded shared inputs."""

    def __init__(self):
        self.panel = D.load_panel()
        self._b30 = None
        self._daily = None

    @property
    def bars_sel(self):
        if self._b30 is None:
            self._b30 = self.panel.bars(C.SELECTION_BAR)
        return self._b30

    @property
    def daily(self):
        if self._daily is None:
            self._daily = self.panel.daily()
        return self._daily

    @property
    def raw_last(self):
        return self.daily["last"] * self.panel.split_factor_day

    @property
    def bar_hours(self) -> float:
        return pd.Timedelta(C.SELECTION_BAR).total_seconds() / 3600


def _p(name: str):
    return C.DATA_PROC / name


# ------------------------------------------------------------------------------------ select
def stage_select(ctx: Context, unconstrained: bool = True) -> None:
    for constrained in [True] + ([False] if unconstrained else []):
        name = "candidates.pkl" if constrained else "candidates_unconstrained.pkl"
        log(f"Testing candidate pairs ({'within industry' if constrained else 'all pairs'}) ...")
        out = {}
        for m in S.trade_months():
            out[str(m)] = S.test_month(m, ctx.bars_sel, ctx.daily, ctx.raw_last, ctx.panel.tickers,
                                       ctx.panel.cal, ctx.bar_hours, constrained=constrained)
        pd.to_pickle(out, _p(name))
        n = sum(len(v) for v in out.values())
        log(f"  {n:,} pair-months tested -> {name}")


def load_candidates(unconstrained: bool = False) -> dict[str, pd.DataFrame]:
    return pd.read_pickle(_p("candidates_unconstrained.pkl" if unconstrained else "candidates.pkl"))


def selections(cands: dict, method: str = "coint", seed: int = 0) -> dict[str, pd.DataFrame]:
    return {m: S.choose(c, method, seed=seed) for m, c in cands.items()}


# ------------------------------------------------------------------------------------ costs
def stage_costs(ctx: Context) -> None:
    months = pd.period_range(ctx.panel.cal["days"][0], ctx.panel.cal["days"][-1], freq="M")
    full = TC.monthly_spread_bps(ctx.panel, months)
    full.to_pickle(_p("spread_full_bps.pkl"))
    half = full.shift(1) / 2                        # applied in the *next* month (point in time)
    half.to_pickle(_p("half_spread_prev_month.pkl"))
    log(f"Spread estimates: {full.shape[0]} months x {full.shape[1]} tickers")


def load_half_spread() -> pd.DataFrame:
    return pd.read_pickle(_p("half_spread_prev_month.pkl"))


# ------------------------------------------------------------------------------------ grid
def stage_grid(ctx: Context, bars=None) -> None:
    sel = selections(load_candidates(), "coint")
    half = load_half_spread()
    for bar in bars or C.BAR_SIZES:
        t = time.time()
        bd = B.BarData(ctx.panel, bar)
        res = B.run(bd, sel, half)
        pd.to_pickle(res, _p(f"grid_{bar}.pkl"))
        log(f"  grid {bar}: {len(res['trades']):,} trades, {time.time() - t:.0f}s")


def load_grid(bar: str) -> dict:
    return pd.read_pickle(_p(f"grid_{bar}.pkl"))


# ------------------------------------------------------------------------------------ helpers
def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(C.TRADING_DAYS)) if r.std() > 0 else np.nan


def split_periods(r: pd.Series | pd.DataFrame) -> dict:
    val_end = pd.Period(C.VALIDATION_END, freq="M").end_time
    return {"validation": r[r.index <= val_end], "oos": r[r.index > val_end], "full": r}


def grid_table() -> pd.DataFrame:
    rows = []
    for bar in C.BAR_SIZES:
        g = load_grid(bar)
        for col in g["net"].columns:
            row = {"bar": bar, "method": col[0], "entry": col[1], "exit": col[2]}
            for per, r in split_periods(g["net"][col]).items():
                row[f"{per}_sharpe"] = sharpe(r)
                row[f"{per}_ann_return"] = r.mean() * C.TRADING_DAYS
            for per, r in split_periods(g["gross"][col]).items():
                row[f"{per}_gross_sharpe"] = sharpe(r)
            tr = g["trades"]
            tr = tr[(tr["method"] == col[0]) & (tr["entry"] == col[1]) & (tr["exit"] == col[2])]
            row["n_trades"] = len(tr)
            row["ann_cost"] = g["cost"][col].mean() * C.TRADING_DAYS
            rows.append(row)
    return pd.DataFrame(rows)


def best_config(table: pd.DataFrame) -> dict:
    b = table.sort_values("validation_sharpe", ascending=False).iloc[0]
    return {"bar": b["bar"], "method": b["method"], "entry": float(b["entry"]), "exit": float(b["exit"])}


# ------------------------------------------------------------------------------------ extras
def stage_extras(ctx: Context, n_placebo: int = 100) -> None:
    table = grid_table()
    table.to_pickle(_p("grid_table.pkl"))
    best = best_config(table)
    (C.DATA_PROC / "best_config.json").write_text(json.dumps(best))
    log(f"Configuration chosen on validation: {best}")
    half = load_half_spread()
    cands = load_candidates()
    one = dict(methods=[best["method"]], entries=[best["entry"]], exits=[best["exit"]])
    bd = B.BarData(ctx.panel, best["bar"])

    # 1) selection rules with identical trading rules
    out = {}
    out["distance"] = B.run(bd, selections(cands, "distance"), half, **one)
    out["unconstrained"] = B.run(bd, selections(load_candidates(unconstrained=True), "coint"), half, **one)
    # random same-industry pairs: many placebo portfolios in one pass (slot label = seed)
    placebo = {}
    for m, c in cands.items():
        parts = []
        for s in range(n_placebo):
            ch = S.choose(c, "random", seed=s)
            if len(ch):
                parts.append(ch.assign(slot=s))
        placebo[m] = pd.concat(parts) if parts else pd.DataFrame()
    out["random"] = B.run(bd, placebo, half, keep_slot=True, **one)
    pd.to_pickle(out, _p("selection_methods.pkl"))
    log("  selection-method runs done")

    # 2) execution latency (signal on bar close; fill k minutes later, or on the next bar's close)
    sel = selections(cands, "coint")
    lat = {}
    for k in [0, 1, 5, 15]:
        lat[f"{k} min"] = B.run(B.BarData(ctx.panel, best["bar"], latency_min=k), sel, half, **one)
    lat["next bar close"] = B.run(B.BarData(ctx.panel, best["bar"], latency_bars=1), sel, half, **one)
    pd.to_pickle(lat, _p("latency.pkl"))
    log("  latency runs done")

    # 3) does formation-window cointegration persist? re-test pairs on the *next* 6 months
    rows = []
    months = S.trade_months()
    b = ctx.bars_sel
    logp = np.log(b["close"])
    rng = np.random.default_rng(0)
    for m in months[: len(months) - C.FORMATION_MONTHS]:
        c = cands[str(m)]
        if c.empty:
            continue
        nxt = m + C.FORMATION_MONTHS                  # window m .. m+5 is the "formation" of m+6
        days = S.formation_days(ctx.panel.cal, nxt)
        idx = np.nonzero(np.isin(b["day"], days))[0]
        sample = c[c["eligible"]].index.tolist()
        others = c.index[~c["eligible"]].tolist()
        sample += list(rng.choice(others, size=min(len(others), 20), replace=False)) if others else []
        for k in sample:
            y, x = logp[idx, c.at[k, "iy"]], logp[idx, c.at[k, "ix"]]
            ok = np.isfinite(y) & np.isfinite(x)
            if ok.sum() < 200:
                continue
            r = S.test_pair(y[ok], x[ok], ctx.bar_hours)
            rows.append({"month": str(m), "y": c.at[k, "y"], "x": c.at[k, "x"], "eligible": bool(c.at[k, "eligible"]),
                         "selected_rank_ok": bool(c.at[k, "eligible"]), "p_form": c.at[k, "p_eg"],
                         "p_next": r["p_eg"], "hl_form": c.at[k, "hl_hours"], "hl_next": r["hl_hours"],
                         "beta_form": c.at[k, "beta"], "beta_next": r["beta"]})
    pd.DataFrame(rows).to_pickle(_p("persistence.pkl"))
    log(f"  persistence re-tests: {len(rows)} pairs")
