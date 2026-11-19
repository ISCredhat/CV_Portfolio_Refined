"""Interfaces shared by the CRSP data layer (workstream A) and the strategy layer (workstream B).

Neither workstream may change this file. If a change is unavoidable, stop and report it to the integrator.

Indexing convention
-------------------
A *panel* has rows = trading days (``days``) and columns = PERMNOs (``permnos``), both sorted ascending.
Every 2-D array below is ``(n_days, n_permnos)`` and float64 unless stated. Column ``j`` always means
``permnos[j]``. Prices are on the **total-return (TR) scale**: ``tr[t, j] = prod_{s<=t}(1 + ret[s, j])`` with
the delisting return folded into the last row. NaN means "no valid price that day".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

EXEC_MODES = ("t1_close", "t1_open", "t_close")


@dataclass
class DailyBars:
    """Drop-in replacement for ``backtest.BarData`` on daily CRSP data (built by ``crsp.build_daily_bars``).

    Attributes used by ``backtest.run_month`` (same names and meaning as BarData):
      close     TR close.
      exec      TR fill price of an order generated at the close of row t (depends on ``exec_mode``):
                t1_close -> close[t+1]; t1_open -> TR open of t+1 (= close[t+1] * open_raw/close_raw), falling
                back to close[t+1] when the open is missing; t_close -> close[t]. The last row's exec = close.
      fill_day  int (n_days,): row index of the day each row's fill happens (t+1, or t for t_close).
      day       int (n_days,) = arange(n_days).
      days      DatetimeIndex of trading days.
      ts        = days (one bar per day).
      month     PeriodIndex(days, freq="M").
      fresh     bool: the stock traded that day (trade-price flag and volume > 0). No entry unless both legs fresh.
      factor    raw price / TR close (so raw price = TR * factor). Used for share counts, ticks and commissions.
      dt_days = 1.0, bars_per_day = 1, bars_per_hour = None (daily bars carry half-lives in days).

    Additional CRSP-only attributes:
      dead      bool: True on a security's last CRSP row (delisting or end of data inside the panel). Any open
                pair with a dead leg is force-closed on that row: dead leg at close (delisting return included),
                live leg at its close that day. Exit reason "delisting".
      short_ok  bool: False where a new short leg is forbidden (2008 short-sale ban proxy).
      borrow_fee (n_permnos,) annual borrow fee per stock, or None for config default.
      exch      object (n_permnos,) primary exchange code at the time ('N', 'A', 'Q') for each column, per row
                when it changes: stored as a 2-D object array (n_days, n_permnos) to stay point in time.
      exec_mode one of EXEC_MODES.
      open_fill bool: True where the t1_open fill actually used an open price (for cost multipliers and counts).
    """

    permnos: np.ndarray
    days: pd.DatetimeIndex
    close: np.ndarray
    exec: np.ndarray
    fill_day: np.ndarray
    fresh: np.ndarray
    factor: np.ndarray
    dead: np.ndarray
    short_ok: np.ndarray
    exch: np.ndarray
    exec_mode: str = "t1_close"
    open_fill: Optional[np.ndarray] = None
    borrow_fee: Optional[np.ndarray] = None
    dt_days: float = 1.0
    bars_per_day: int = 1
    bars_per_hour: Optional[float] = None
    day: np.ndarray = field(init=False)
    ts: pd.DatetimeIndex = field(init=False)
    month: pd.PeriodIndex = field(init=False)

    def __post_init__(self):
        self.day = np.arange(len(self.days))
        self.ts = self.days
        self.month = pd.PeriodIndex(self.days, freq="M")


def validate_bars(bd: DailyBars) -> None:
    """Raise AssertionError if `bd` breaks the contract (shapes, dtypes, exec mode, fill days)."""
    n, k = len(bd.days), len(bd.permnos)
    for name in ["close", "exec", "fresh", "factor", "dead", "short_ok", "exch"]:
        a = getattr(bd, name)
        assert a.shape == (n, k), f"{name}: {a.shape} != {(n, k)}"
    assert bd.exec_mode in EXEC_MODES, bd.exec_mode
    assert bd.fill_day.shape == (n,)
    assert np.all(np.diff(bd.permnos) > 0) and np.all(np.diff(bd.days.values).astype("int64") > 0)
    assert bd.fresh.dtype == bool and bd.dead.dtype == bool and bd.short_ok.dtype == bool


# ---------------------------------------------------------------------------------------------------------
# Cost function passed to backtest.run_month(..., cost_fn=...). Built by costs_daily.make_cost_fn(bars, ...).
#   cost_fn(hs_bps, raw_price, rows, cols, mult) -> one-way cost per leg in bps, shape (len(rows), len(cols))
#     hs_bps    (len(cols),) half-spread in bps for this month (previous-month estimate; never NaN)
#     raw_price (len(rows), len(cols)) as-traded raw price
#     rows      int row indices into the panel (days); cols int column indices (permnos)
#     mult      cost multiplier (sensitivity)
CostFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


# ---------------------------------------------------------------------------------------------------------
# DataFrame schemas (column name -> dtype). Producers must emit exactly these columns (extra columns allowed).

# universe.select(...) -> one row per stock in the month's universe, decided at formation end.
UNIVERSE_COLUMNS = {
    "permno": "int64", "permco": "int64", "col": "int64",       # col = column index in the panel
    "mcap": "float64", "price": "float64", "exch": "object",
    "siccd": "int64", "ff48": "int64", "sic2": "int64", "gics_ind": "object",   # gics_ind may be None pre-1999
    "traded_share": "float64",
}

# selection_daily.test_month(...) -> one row per candidate pair (B produces; integration consumes).
CANDIDATE_COLUMNS = {
    "month": "object", "y": "int64", "x": "int64", "iy": "int64", "ix": "int64",   # y, x are PERMNOs
    "group": "object", "p_eg": "float64", "eg_stat": "float64", "beta": "float64", "sigma": "float64",
    "mean_ly": "float64", "mean_lx": "float64", "sd_lx": "float64", "n_obs": "int64",
    "var_beta": "float64", "var_alpha": "float64", "hl_days": "float64", "hurst": "float64",
    "ssd": "float64", "johansen": "float64", "bh_pass": "bool", "johansen_pass": "bool",
    "plausible": "bool", "eligible": "bool", "n_universe": "int64", "same_permco": "bool",
}

# costs_daily.half_spread_table(...) -> DataFrame(index=PeriodIndex months, columns=permnos), bps, no NaN.
# The row for month m is estimated from month m-1 only.

# crsp.delisting_events(...) -> one row per delisting event.
DELIST_COLUMNS = {
    "permno": "int64", "last_date": "datetime64[ns]", "dlstcd": "int64", "exch": "object",
    "dlret": "float64", "dlret_imputed": "bool", "is_merger": "bool",
}
