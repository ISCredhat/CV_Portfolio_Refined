"""Monthly, point-in-time pair selection.

For every trading month m:
  1. Formation window = the 6 calendar months before m (30-minute bars, split-adjusted log prices).
  2. Universe = tickers that traded on >= 95% of formation days, median price >= $3 and median
     daily Nasdaq dollar volume >= $5M (all measured inside the window).
  3. Candidates = every pair of universe tickers in the same industry group.
  4. Engle-Granger test in both directions; the pair's p-value is 2 x the smaller one
     (Bonferroni for trying two orderings).
  5. Eligible pairs have p <= 5%, pass the Johansen trace test, have an OU half-life of
     2 h - 10 days, a Hurst exponent < 0.5 and a plausible hedge ratio.
  6. Rank by p-value; keep up to N_PAIRS pairs with each ticker in at most 2 of them.

Benjamini-Hochberg is also computed across all candidates (FDR 5%) and reported: it shows how
many of the selected pairs are statistically credible discoveries rather than luck. Alternative
selection rules (distance / SSD, random same-industry pairs, no industry constraint) reuse the
same candidate table so that only the ranking differs.

Nothing from month m (or later) is used to choose the pairs traded in month m.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import coint as K
from . import config as C


def trade_months() -> pd.PeriodIndex:
    return pd.period_range(C.FIRST_TRADE_MONTH, C.LAST_TRADE_MONTH, freq="M")


def formation_days(cal: dict, month: pd.Period) -> np.ndarray:
    """Indices of trading days in the formation window of `month`."""
    start = (month - C.FORMATION_MONTHS).start_time
    end = month.start_time
    d = cal["days"]
    return np.nonzero((d >= start) & (d < end))[0]


def universe(daily: dict, raw_last: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Boolean mask over tickers that pass the liquidity filters inside `days`."""
    traded = daily["minutes_traded"][days] > 0
    cover = traded.mean(0)
    with np.errstate(all="ignore"):
        px = np.nanmedian(np.where(traded, raw_last[days], np.nan), 0)
        dv = np.nanmedian(np.where(traded, daily["dollar_volume"][days], np.nan), 0)
    return (cover >= C.MIN_DAYS_COVERAGE) & (px >= C.MIN_PRICE) & (dv >= C.MIN_DOLLAR_VOLUME)


def candidate_pairs(tickers: list[str], ok: np.ndarray, constrained: bool = True) -> list[tuple[int, int]]:
    idx = np.nonzero(ok)[0]
    out = []
    for i, j in itertools.combinations(idx, 2):
        gi, gj = C.GROUP_OF.get(tickers[i]), C.GROUP_OF.get(tickers[j])
        if not constrained or (gi is not None and gi == gj):
            out.append((int(i), int(j)))
    return out


def test_pair(la: np.ndarray, lb: np.ndarray, bar_hours: float, maxlag: int = 12) -> dict:
    """All formation statistics for one candidate pair (log prices, NaN-free, same length)."""
    r1 = K.engle_granger(la, lb, maxlag=maxlag)
    r2 = K.engle_granger(lb, la, maxlag=maxlag)
    if r1["pvalue"] <= r2["pvalue"]:
        r, flip = r1, False
    else:
        r, flip = r2, True
    ly, lx = (lb, la) if flip else (la, lb)
    my, mx = ly.mean(), lx.mean()
    u = r["resid"]
    return {"flip": flip, "p_eg": min(1.0, 2 * min(r1["pvalue"], r2["pvalue"])), "eg_stat": r["stat"],
            "beta": r["beta"], "sigma": float(u.std(ddof=2)), "mean_ly": float(my), "mean_lx": float(mx),
            "sd_lx": float(lx.std()), "n_obs": len(u), "var_beta": float(u.var(ddof=2) / ((lx - mx) ** 2).sum()),
            "var_alpha": float(u.var(ddof=2) / len(u)),
            "hl_hours": K.half_life(u) * bar_hours, "hurst": K.hurst(u),
            "ssd": float(np.sum((np.exp(la - la[0]) - np.exp(lb - lb[0])) ** 2)),
            "_ly": ly, "_lx": lx}


def test_month(month: pd.Period, bars: dict, daily: dict, raw_last: np.ndarray, tickers: list[str],
               cal: dict, bar_hours: float, constrained: bool = True) -> pd.DataFrame:
    """Formation statistics for every candidate pair of `month` (one row per pair)."""
    days = formation_days(cal, month)
    ok = universe(daily, raw_last, days)
    rows = np.nonzero(np.isin(bars["day"], days))[0]
    logp = np.log(bars["close"][rows])
    out = []
    for i, j in candidate_pairs(tickers, ok, constrained):
        a, b = logp[:, i], logp[:, j]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            continue
        r = test_pair(a[m], b[m], bar_hours)
        y, x = (j, i) if r["flip"] else (i, j)
        gi, gj = C.GROUP_OF.get(tickers[i]), C.GROUP_OF.get(tickers[j])
        row = {"month": str(month), "y": tickers[y], "x": tickers[x], "iy": y, "ix": x,
               "group": gi if (gi is not None and gi == gj) else "cross-industry",
               **{k: v for k, v in r.items() if not k.startswith("_") and k != "flip"}}
        row["johansen"] = np.nan
        if row["p_eg"] <= C.ALPHA:
            row["johansen"] = K.johansen_trace(np.column_stack([r["_ly"], r["_lx"]]))["trace"]
        out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["bh_pass"] = K.benjamini_hochberg(df["p_eg"].to_numpy(), C.FDR)
    df["johansen_pass"] = df["johansen"] > K.JOHANSEN_CV_R0[1]
    lo, hi = C.HALF_LIFE_HOURS
    blo, bhi = C.HEDGE_RATIO_RANGE
    df["plausible"] = df["beta"].between(blo, bhi)
    df["eligible"] = ((df["p_eg"] <= C.ALPHA) & df["johansen_pass"] & df["hl_hours"].between(lo, hi)
                      & (df["hurst"] < C.MAX_HURST) & df["plausible"])
    df["n_universe"] = int(ok.sum())
    return df


def choose(cands: pd.DataFrame, method: str = "coint", n: int = C.N_PAIRS, seed: int = 0) -> pd.DataFrame:
    """Pick up to `n` pairs for one month from its candidate table.

    coint    : eligible pairs ranked by Engle-Granger p-value (the strategy)
    distance : lowest sum of squared differences of normalised prices (Gatev et al. 2006)
    random   : random pairs from the same candidate set (placebo)"""
    if cands.empty:
        return cands
    if method == "coint":
        pool = cands[cands["eligible"]].sort_values(["p_eg", "hl_hours"])
    elif method == "distance":
        pool = cands[cands["plausible"]].sort_values("ssd")
    elif method == "random":
        pool = cands[cands["plausible"]].sample(frac=1.0, random_state=seed)
    else:
        raise ValueError(method)
    used: dict[str, int] = {}
    keep = []
    for k in pool.index:
        y, x = cands.at[k, "y"], cands.at[k, "x"]
        if used.get(y, 0) >= C.MAX_PAIRS_PER_TICKER or used.get(x, 0) >= C.MAX_PAIRS_PER_TICKER:
            continue
        keep.append(k)
        used[y] = used.get(y, 0) + 1
        used[x] = used.get(x, 0) + 1
        if len(keep) >= n:
            break
    return cands.loc[keep]
