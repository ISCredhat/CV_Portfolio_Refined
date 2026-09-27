"""Event-window return paths, buy-and-hold abnormal returns (BHAR) and daily returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def adjusted_open(px: dict[str, pd.DataFrame]) -> np.ndarray:
    """Open price on the same (split + dividend) adjusted basis as adj_close."""
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = px["adj_close"].to_numpy() / px["close"].to_numpy()
    return px["open"].to_numpy() * factor


def value_paths(entry_row: np.ndarray, col: np.ndarray, adj_close: np.ndarray, adj_open: np.ndarray,
                horizon: int) -> np.ndarray:
    """Growth of $1 bought at the entry open, marked at each close for `horizon` sessions.

    Returns array (n_events, horizon): column k = value at the close of session k
    (k=0 is the entry day). If the stock stops trading (delisting, halt) the last
    price is carried forward (position treated as exited at the last price). Sessions
    beyond the end of the data are NaN (outcome not yet realised).
    """
    T = adj_close.shape[0]
    rows = entry_row[:, None] + np.arange(horizon)[None, :]
    beyond = rows >= T
    rows_c = np.clip(rows, 0, T - 1)
    vals = adj_close[rows_c, col[:, None]].astype("float64")
    vals = pd.DataFrame(vals).ffill(axis=1).to_numpy()
    entry_px = adj_open[entry_row, col].astype("float64")
    paths = vals / entry_px[:, None]
    paths[beyond] = np.nan
    return paths


def benchmark_paths(entry_row: np.ndarray, bench_close: np.ndarray, bench_open: np.ndarray,
                    horizon: int) -> np.ndarray:
    T = len(bench_close)
    rows = entry_row[:, None] + np.arange(horizon)[None, :]
    beyond = rows >= T
    vals = bench_close[np.clip(rows, 0, T - 1)] / bench_open[entry_row][:, None]
    vals[beyond] = np.nan
    return vals


def bhar_table(paths: np.ndarray, bpaths: np.ndarray, horizons: list[int], prefix: str) -> pd.DataFrame:
    """Raw and abnormal (vs benchmark) buy-and-hold returns at each horizon (1-indexed)."""
    out = {}
    for h in horizons:
        raw = paths[:, h - 1] - 1.0
        out[f"ret_{h}"] = raw
        out[f"{prefix}_{h}"] = raw - (bpaths[:, h - 1] - 1.0)
    return pd.DataFrame(out)


def daily_returns(adj_close: np.ndarray) -> np.ndarray:
    """Close-to-close simple returns; NaN where either close is missing."""
    r = np.full_like(adj_close, np.nan, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = adj_close[1:] / adj_close[:-1] - 1.0
    return r


def event_daily_returns(entry_row: np.ndarray, col: np.ndarray, adj_close: np.ndarray,
                        adj_open: np.ndarray, hold: int) -> np.ndarray:
    """Per-event daily return path (n_events, hold): day 0 = open->close of entry day.

    Missing prices (halts/delisting) are treated as 0% return (cash) for that day;
    sessions beyond the data end are NaN.
    """
    paths = value_paths(entry_row, col, adj_close, adj_open, hold)
    prev = np.concatenate([np.ones((len(paths), 1)), paths[:, :-1]], axis=1)
    r = paths / prev - 1.0
    beyond = np.isnan(paths)
    r = np.where(np.isfinite(r), r, 0.0)
    r[beyond] = np.nan
    return r
