"""Data layer: Databento 1-minute CSVs -> clean, split-adjusted, regular-hours minute panels,
and resampling to any bar size with a realistic execution price.

Conventions
-----------
* Databento `ts_event` is the bar *start* in UTC. We convert to New York time (DST-aware) and keep
  regular trading hours only: bars starting 09:30 ... 15:59 ET (12:59 on NYSE early-close days).
* The vendor files were cut at 20:00 UTC, which is 16:00 ET in summer but 15:00 ET in winter, so on
  standard-time days the last trading hour is missing for every ticker. The session end of each day
  is therefore detected from the data (the last minute in which most large caps traded).
* Every trading day gets the full grid of RTH minutes, so all tickers share one time axis.
  A minute with no trade is NaN (never filled with an invented price).
* Prices are back-adjusted for splits and spin-offs so that the latest prices are as traded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

FIELDS = ["close", "high", "low", "volume"]


# ------------------------------------------------------------------------------ reading
def read_minute_csv(path) -> pd.DataFrame:
    """One ticker's CSV -> RTH rows with columns date, minute (of day, ET), high, low, close, volume."""
    df = pd.read_csv(path, usecols=["ts_event", "high", "low", "close", "volume"],
                     dtype={"high": "float64", "low": "float64", "close": "float64", "volume": "float64"})
    ts = pd.to_datetime(df["ts_event"], format="%Y-%m-%d %H:%M:%S%z", utc=True).dt.tz_convert(C.TZ)
    minute = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    date = ts.dt.tz_localize(None).dt.normalize()
    keep = (minute >= C.RTH_START_MIN) & (minute < C.RTH_END_MIN) & (df["close"].to_numpy() > 0)
    out = pd.DataFrame({"date": date[keep].to_numpy(), "minute": minute[keep].astype("int16")})
    for f in ["high", "low", "close", "volume"]:
        out[f] = df[f].to_numpy()[keep]
    return out.drop_duplicates(["date", "minute"], keep="last").reset_index(drop=True)


# ------------------------------------------------------------------------------ calendar
def session_ends(core: list[pd.DataFrame], min_share: float = 0.5) -> pd.Series:
    """Per-day session end (minute of day, exclusive): one minute after the last minute in which
    at least `min_share` of the `core` (very liquid) tickers traded. NYSE early closes are capped at 13:00."""
    days = pd.DatetimeIndex(sorted(set().union(*[set(df["date"].unique()) for df in core])))
    grid = np.zeros((len(days), C.RTH_END_MIN - C.RTH_START_MIN))
    for df in core:
        np.add.at(grid, (days.get_indexer(df["date"]), df["minute"].to_numpy() - C.RTH_START_MIN), 1)
    share = grid / len(core)
    last = np.array([np.nonzero(r >= min_share)[0].max() if (r >= min_share).any() else -1 for r in share])
    end = pd.Series(C.RTH_START_MIN + last + 1, index=days)
    early = days.isin(pd.to_datetime(C.EARLY_CLOSES))
    end[early] = np.minimum(end[early], C.HALF_DAY_END_MIN)
    return end


def make_calendar(day_counts: pd.Series, session_end: pd.Series, min_tickers: int = 20) -> dict:
    """Trading days = days on which at least `min_tickers` tickers printed in regular hours.

    Returns the minute grid: for every trading day, one slot per session minute."""
    days = pd.DatetimeIndex(sorted(day_counts[day_counts >= min_tickers].index))
    n_per_day = session_end.reindex(days).fillna(C.RTH_END_MIN).to_numpy().astype(int) - C.RTH_START_MIN
    day_start = np.concatenate([[0], np.cumsum(n_per_day)[:-1]])
    n = int(n_per_day.sum())
    day_of_min = np.repeat(np.arange(len(days)), n_per_day)
    minute_of_day = (np.arange(n) - day_start[day_of_min] + C.RTH_START_MIN).astype("int16")
    ts = days[day_of_min] + pd.to_timedelta(minute_of_day.astype("int64"), unit="m")
    return {"days": days, "day_start": day_start, "n_per_day": n_per_day, "day_of_min": day_of_min,
            "minute_of_day": minute_of_day, "ts": ts, "session_end": n_per_day + C.RTH_START_MIN}


def grid_positions(cal: dict, date: np.ndarray, minute: np.ndarray) -> np.ndarray:
    """Row index in the minute grid for each (date, minute); -1 if not a grid slot."""
    di = cal["days"].get_indexer(pd.DatetimeIndex(date))
    ok = di >= 0
    pos = np.full(len(date), -1, dtype="int64")
    off = minute.astype("int64") - C.RTH_START_MIN
    ok &= off < cal["n_per_day"][np.clip(di, 0, None)]
    pos[ok] = cal["day_start"][di[ok]] + off[ok]
    return pos


# ------------------------------------------------------------------------------ splits
def split_factors(splits: pd.DataFrame, tickers: list[str], days: pd.DatetimeIndex) -> np.ndarray:
    """Per-day price divisor: product of split ratios with ex-date *after* that day.

    `splits` has columns ticker, date (ex-date), ratio (new shares per old share; a 1-for-10
    reverse split is 0.1, a spin-off is recorded as the price-adjustment ratio)."""
    f = np.ones((len(days), len(tickers)))
    col = {t: i for i, t in enumerate(tickers)}
    for r in splits.itertuples(index=False):
        if r.ticker in col and days[0] < pd.Timestamp(r.date) <= days[-1]:
            f[days < pd.Timestamp(r.date), col[r.ticker]] *= float(r.ratio)
    return f


def overnight_jumps(daily_last: pd.DataFrame, daily_first: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
    """Close-to-next-open moves larger than `threshold` in log terms: split candidates.

    Every candidate is matched to the nearest 'round' split ratio so it can be reviewed by hand."""
    prev = daily_last.shift(1)
    r = np.log(prev / daily_first)
    rows = []
    ratios = np.array(sorted({float(a) / b for a in range(1, 51) for b in range(1, 51)
                              if (a == 1 or b == 1) and a != b}))
    for (d, t), v in r.stack().items():
        if abs(v) > threshold:
            raw = float(np.exp(v))
            near = ratios[np.argmin(np.abs(np.log(ratios) - v))]
            rows.append({"ticker": t, "date": d, "prev_close": float(prev.loc[d, t]),
                         "open": float(daily_first.loc[d, t]), "implied_ratio": raw, "nearest_ratio": near,
                         "ratio_error": abs(np.log(raw / near))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ array helpers
def ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down axis 0 (vectorised)."""
    a = np.asarray(a)
    n = a.shape[0]
    ar = np.arange(n).reshape((n,) + (1,) * (a.ndim - 1))
    idx = np.where(np.isnan(a), 0, ar)
    np.maximum.accumulate(idx, axis=0, out=idx)
    # slots before the first valid value point at row 0, which is NaN there: they stay NaN
    return np.take_along_axis(a, idx, axis=0)


def bfill(a: np.ndarray) -> np.ndarray:
    """Backward-fill NaNs up axis 0."""
    return ffill(a[::-1])[::-1]


# ------------------------------------------------------------------------------ bars
def bar_ends(cal: dict, bar: str) -> np.ndarray:
    """Index (in the minute grid) of the last minute of each bar. Bars are anchored at 09:30
    and never span two sessions; the last bar of a day may be shorter (e.g. 15:30-16:00 for 1h)."""
    if bar in ("1D", "1d", "D"):
        return cal["day_start"] + cal["n_per_day"] - 1
    k = int(pd.Timedelta(bar).total_seconds() // 60)
    ends = []
    for s, n in zip(cal["day_start"], cal["n_per_day"]):
        e = np.arange(s + k - 1, s + n, k)
        if len(e) == 0 or e[-1] != s + n - 1:
            e = np.append(e, s + n - 1)
        ends.append(e)
    return np.concatenate(ends)


def bars_per_day(bar: str) -> float:
    if bar in ("1D", "1d", "D"):
        return 1.0
    k = pd.Timedelta(bar).total_seconds() / 60
    return float(np.ceil(390 / k))


class MinutePanel:
    """Split-adjusted RTH minute panel (minutes x tickers) plus the derived price series
    needed by the backtester: last traded price and next traded price."""

    def __init__(self, cal: dict, tickers: list[str], close: np.ndarray, high: np.ndarray,
                 low: np.ndarray, volume: np.ndarray):
        self.cal, self.tickers = cal, list(tickers)
        self.col = {t: i for i, t in enumerate(self.tickers)}
        self.close, self.high, self.low, self.volume = close, high, low, volume
        self._last = None
        self._next = None

    @property
    def last(self) -> np.ndarray:
        """Last traded price at or before each minute (stale prices carried forward)."""
        if self._last is None:
            self._last = ffill(self.close)
        return self._last

    @property
    def next(self) -> np.ndarray:
        """Close of the first minute bar at or after each minute that has a trade:
        the price a marketable order sent at the start of that minute would (roughly) get."""
        if self._next is None:
            self._next = bfill(self.close)
        return self._next

    def bars(self, bar: str, latency_min: int = C.LATENCY_MIN) -> dict:
        """Resample to `bar`: close = last trade in the bar (or earlier, if none),
        exec = price of an order sent at the bar close and filled `latency_min` minutes later."""
        ends = bar_ends(self.cal, bar)
        n = self.close.shape[0]
        close = self.last[ends]
        if latency_min == 0:
            exe = close.copy()
        else:
            # an order sent at the end of minute `e` is filled at the close of minute e + latency
            nxt = np.minimum(ends + latency_min, n - 1)
            exe = self.next[nxt]
            exe = np.where(np.isnan(exe), close, exe)
        return {"ends": ends, "ts": self.cal["ts"][ends], "day": self.cal["day_of_min"][ends],
                "close": close, "exec": exe}

    def daily(self) -> dict:
        """Daily summaries from the minute panel (per ticker)."""
        d = self.cal["day_of_min"]
        nd = len(self.cal["days"])
        last = self.last[self.cal["day_start"] + self.cal["n_per_day"] - 1]
        traded = ~np.isnan(self.close)
        n_trades = np.zeros((nd, len(self.tickers)))
        np.add.at(n_trades, d, traded)
        dv = np.nan_to_num(self.close * self.volume)
        dollar_vol = np.zeros((nd, len(self.tickers)))
        np.add.at(dollar_vol, d, dv)
        first = self.next[self.cal["day_start"]]
        return {"last": last, "first": first, "minutes_traded": n_trades, "dollar_volume": dollar_vol}


def load_panel(proc_dir=None, mmap: bool = False) -> MinutePanel:
    """Load the processed minute panel written by scripts/build_data.py."""
    import json
    import pickle
    from pathlib import Path
    proc = Path(proc_dir or C.DATA_PROC)
    with open(proc / "calendar.pkl", "rb") as fh:
        cal = pickle.load(fh)
    tickers = json.loads((proc / "tickers.json").read_text())
    arr = {f: np.load(proc / f"minute_{f}.npy", mmap_mode="r" if mmap else None) for f in FIELDS}
    p = MinutePanel(cal, tickers, arr["close"], arr["high"], arr["low"], arr["volume"])
    p.split_factor_day = np.load(proc / "split_factor_day.npy")   # raw (as-traded) = adjusted * factor
    return p
