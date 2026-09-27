"""Calendar-time portfolio backtest with liquidity-based transaction costs.

Each selected event opens a position at the open of the session after its filing
date and holds it for `hold` sessions. On every day the portfolio is equally
weighted across open positions (the standard calendar-time portfolio of Lyon,
Barber & Tsai 1999), so returns are a genuine daily P&L series. (Averaging overlapping
3-month event returns instead would overstate the Sharpe ratio.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .returns import event_daily_returns


def one_way_cost_bps(adv: pd.Series) -> pd.Series:
    bounds = [b for b, _ in C.COST_SCHEDULE]
    costs = [c for _, c in C.COST_SCHEDULE]
    idx = np.searchsorted(bounds, adv.fillna(0).to_numpy(), side="right")
    idx = np.clip(idx, 0, len(costs) - 1)
    return pd.Series(np.asarray(costs)[idx], index=adv.index)


def calendar_portfolio(sel: pd.DataFrame, adj_close: np.ndarray, adj_open: np.ndarray,
                       dates: pd.DatetimeIndex, hold: int = C.LABEL_HORIZON,
                       cost_mult: float = 1.0, rf: pd.Series | None = None,
                       weighting: str = "drift") -> pd.DataFrame:
    """Daily gross/net returns and number of open positions for selected events.

    weighting="drift": every new position is bought with one unit of capital and then
        left to drift with its price (buy-and-hold, no hidden rebalancing trades).
    weighting="equal": positions re-weighted to 1/N every day (academic calendar-time
        portfolio; implies daily rebalancing whose costs are not charged).
    Costs: one-way cost (bps, from 20d ADV) charged on entry and again on exit, in
    proportion to the position's weight on that day. Days with no open position earn
    the risk-free rate.
    """
    T = len(dates)
    n_ev = len(sel)
    entry = sel["entry_row"].to_numpy()
    er = event_daily_returns(entry, sel["col"].to_numpy(), adj_close, adj_open, hold)
    rows = entry[:, None] + np.arange(hold)[None, :]
    valid = np.isfinite(er) & (rows < T)
    if weighting == "drift":
        growth = np.where(valid, 1.0 + er, 1.0).cumprod(axis=1)
        w = np.concatenate([np.ones((n_ev, 1)), growth[:, :-1]], axis=1)   # value at start of day
    else:
        w = np.ones_like(er)
    w = np.where(valid, w, 0.0)
    r_w = np.where(valid, er, 0.0) * w
    r_sum = np.bincount(rows[valid], weights=r_w[valid], minlength=T)[:T]
    w_sum = np.bincount(rows[valid], weights=w[valid], minlength=T)[:T]
    n_open = np.bincount(rows[valid], minlength=T)[:T].astype(float)
    c = one_way_cost_bps(sel["adv"]).to_numpy() / 1e4 * cost_mult
    cost_sum = np.bincount(entry, weights=c * w[:, 0], minlength=T)[:T]
    last_k = np.where(valid, np.arange(hold)[None, :], -1).max(axis=1)
    exited = (last_k >= 0) & (entry + last_k < T - 1)            # still open at data end: no exit yet
    exit_rows = (entry + last_k)[exited]
    exit_w = (w[np.arange(n_ev), np.clip(last_k, 0, None)] * (1.0 + np.where(valid, er, 0.0)[np.arange(n_ev), np.clip(last_k, 0, None)]))[exited]
    cost_sum += np.bincount(exit_rows, weights=c[exited] * exit_w, minlength=T)[:T]
    with np.errstate(invalid="ignore", divide="ignore"):
        gross = np.where(w_sum > 0, r_sum / w_sum, np.nan)
        net = np.where(w_sum > 0, (r_sum - cost_sum) / w_sum, np.nan)
    out = pd.DataFrame({"gross": gross, "net": net, "n_open": n_open}, index=dates)
    rfd = rf.reindex(dates).fillna(0.0) if rf is not None else pd.Series(0.0, index=dates)
    out["gross"] = out["gross"].fillna(rfd)
    out["net"] = out["net"].fillna(rfd)
    first = int(np.argmax(n_open > 0)) if (n_open > 0).any() else T
    return out.iloc[first:]
