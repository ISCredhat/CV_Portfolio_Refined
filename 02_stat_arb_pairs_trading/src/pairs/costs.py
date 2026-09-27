"""Transaction-cost model.

One-way cost per leg, in basis points of the traded value:

    half-spread  = max(Abdi-Ranaldo estimate / 2,  half a tick / price)
    commission   = $0.0035 per share / price
    impact       = 1 bp

The effective spread is estimated each month from 1-minute high/low/close bars with the
Abdi & Ranaldo (2017) close-high-low estimator and applied to the *following* month (point in
time). US stocks quote in whole cents, so the spread can never be below one tick: for a $5 stock
that floor alone is 10 bps per round trip, which is why low-priced names are expensive to trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def monthly_spread_bps(panel, months: pd.PeriodIndex) -> pd.DataFrame:
    """Abdi-Ranaldo (2017) effective spread (full spread, bps) per ticker and calendar month.

    s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})], with c the log close and eta the log mid-range
    of 1-minute bars; pairs of bars must be consecutive minutes of the same session."""
    ts_month = pd.PeriodIndex(panel.cal["ts"], freq="M")
    day = panel.cal["day_of_min"]
    out = {}
    for m in months:
        idx = np.nonzero(ts_month == m)[0]
        if len(idx) < 2:
            continue
        c = np.log(np.asarray(panel.close[idx], dtype="float64"))
        eta = 0.5 * (np.log(np.asarray(panel.high[idx], dtype="float64"))
                     + np.log(np.asarray(panel.low[idx], dtype="float64")))
        same = (day[idx][1:] == day[idx][:-1])[:, None]
        prod = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
        prod = np.where(same & np.isfinite(prod), prod, np.nan)
        s2 = 4 * np.nanmean(prod, axis=0)
        out[m] = np.sqrt(np.clip(s2, 0, None)) * 1e4
    return pd.DataFrame(out, index=panel.tickers).T


def leg_cost_bps(half_spread_bps: np.ndarray, raw_price: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
    """One-way cost per leg in bps given the (previous month's) half-spread and the as-traded price."""
    raw_price = np.asarray(raw_price, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        tick_half = 0.5 * C.TICK / raw_price * 1e4
        commission = C.COMMISSION_PER_SHARE / raw_price * 1e4
    hs = np.minimum(np.maximum(np.nan_to_num(half_spread_bps, nan=0.0), tick_half), C.MAX_HALF_SPREAD_BPS)
    return multiplier * (hs + commission + C.IMPACT_BPS)
