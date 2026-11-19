"""Daily-data transaction costs: spread estimates, ticks, commissions and the ``contracts.CostFn``.

Half-spread for month m is estimated from month m-1 only (point in time):
  quoted      half of the median closing quoted spread (ask-bid)/mid over valid days (>= MIN_QUOTE_DAYS);
  Abdi-Ranaldo (2017) close-high-low estimator, s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})];
  Corwin-Schultz (2012) high-low estimator (negative daily values set to 0, then averaged).
Before QUOTES_RELIABLE_FROM (1993-01) NYSE/AMEX quotes are not used: a calibrated mapping, fitted on 1993-96
quoted spreads (``calibrate_spread_map``, plain least squares on log quoted half-spread ~ log AR, log CS, log
price, log size, Duan smearing; no statsmodels), predicts the half-spread from the estimators (review M6).
CAVEAT: those coefficients are fitted on data dated after the early months they are applied to. They are a
cost-level constant, not a signal; the pre-registration should say so. Pass a fitted ``calib`` explicitly.
Missing estimate -> ``SPREAD_FALLBACK_MULT`` x median of the same exchange x price decile of that month's
available estimates (then the exchange median, then the all-stock median; raise if there is nothing at all).
There is no spread cap (review M5). The table is never NaN; ``table.attrs["sources"]`` counts per month how
many stocks used each source (quoted / calibrated / fallback).
Verify on WRDS: ``dlybid``/``dlyask`` are closing quotes (not zero-filled), dlyhigh/dlylow are raw (unadjusted).
Commissions: Do & Faff (2012) Table 2 institutional series in bps (config_crsp.COMMISSION_BPS); tick dates from config_crsp.TICKS.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config_crsp as CC
from . import contracts as K

MIN_QUOTE_DAYS = 5
MIN_EST_DAYS = 10
CALIB_PERIOD = ("1993-01", "1996-12")
_BPS = 1e4


# ------------------------------------------------------------------------------------ ticks, commissions
def tick_size(exch, date) -> np.ndarray:
    """Minimum price increment ($) by exchange code and date (config_crsp.TICKS; 1/8 before the first date).
    `exch` and `date` broadcast. Unknown exchange codes use the NYSE table."""
    ex = np.asarray(exch, dtype=object)
    d = np.asarray(pd.to_datetime(np.asarray(date)).values if not np.isscalar(date) else pd.Timestamp(date).to_datetime64())
    ex, d = np.broadcast_arrays(ex, d)
    out = np.full(ex.shape, 0.125)
    for code in ("N", "A", "Q"):
        sel = (ex == code) if code != "N" else ~np.isin(ex, ["A", "Q"])
        for first, tick in CC.TICKS[code]:
            out = np.where(sel & (d >= np.datetime64(first)), tick, out)
    return out


def commission_bps(year) -> np.ndarray:
    """One-way commission in bps of traded value by calendar year (``COMMISSION_BPS``: Do & Faff 2012 Table 2,
    institutional). Before the first year: the first value; after the last year: the last value."""
    y = np.asarray(year)
    out = np.full(y.shape, float(CC.COMMISSION_BPS[0][1]))
    for first, c in CC.COMMISSION_BPS:
        out = np.where(y >= first, float(c), out)
    return out


# ------------------------------------------------------------------------------------ daily estimators
def _daily_terms(panel) -> dict[str, np.ndarray]:
    """Per day x stock: relative quoted spread, Abdi-Ranaldo product term (t, t+1), Corwin-Schultz S (t, t+1)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = 0.5 * (panel.bid + panel.ask)
        q = np.where((panel.bid > 0) & (panel.ask > panel.bid), (panel.ask - panel.bid) / mid, np.nan)
        c = np.log(panel.prc)
        hl_ok = (panel.high >= panel.low) & (panel.high > 0) & (panel.low > 0)
        h = np.where(hl_ok, np.log(panel.high), np.nan)
        l = np.where(hl_ok, np.log(panel.low), np.nan)
        eta = 0.5 * (h + l)
        ar = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])                      # row t uses t and t+1
        # a split between t and t+1 breaks raw H/L comparisons: mask
        lf = np.log(panel.prc / panel.tr)
        split = np.abs(lf[1:] - lf[:-1]) > np.log(1.25)
        ar = np.where(split, np.nan, ar)
        beta = (h[:-1] - l[:-1]) ** 2 + (h[1:] - l[1:]) ** 2
        gamma = (np.maximum(h[:-1], h[1:]) - np.minimum(l[:-1], l[1:])) ** 2
        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        cs = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        cs = np.where(split, np.nan, np.maximum(cs, 0.0))
    pad = np.full((1, q.shape[1]), np.nan)
    return {"q": q, "ar": np.vstack([ar, pad]), "cs": np.vstack([cs, pad])}


def _stats(terms, mask_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Monthly per-stock half-spread estimates in bps (NaN when too few valid days) for the rows in mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = terms["q"][mask_rows]
        nq = np.isfinite(q).sum(0)
        quoted = np.where(nq >= MIN_QUOTE_DAYS, 0.5 * np.nanmedian(q, axis=0) * _BPS, np.nan)
        a = terms["ar"][mask_rows]
        na = np.isfinite(a).sum(0)
        ar = np.where(na >= MIN_EST_DAYS, 0.5 * np.sqrt(np.clip(4 * np.nanmean(a, axis=0), 0, None)) * _BPS, np.nan)
        c = terms["cs"][mask_rows]
        nc = np.isfinite(c).sum(0)
        cs = np.where(nc >= MIN_EST_DAYS, 0.5 * np.nanmean(c, axis=0) * _BPS, np.nan)
    return {"quoted": quoted, "ar": ar, "cs": cs}


def _design(ar, cs, price, cap) -> np.ndarray:
    return np.column_stack([np.ones(len(ar)), np.log1p(ar), np.log1p(cs), np.log(price), np.log(cap)])


def calibrate_spread_map(panel, period=CALIB_PERIOD, min_obs: int = 200) -> dict:
    """Fit log(quoted half-spread bps) on log(1+AR), log(1+CS), log price, log mcap over NYSE/AMEX stock-months in
    `period` (default 1993-96), each from the same month. Returns {"coef", "smear", "n", "r2"}. Raises when
    there are fewer than `min_obs` stock-months (the panel must contain the calibration years)."""
    terms = _daily_terms(panel)
    mon = panel.days.to_period("M")
    X, y = [], []
    for m in pd.period_range(*period, freq="M"):
        rows = np.nonzero(mon == m)[0]
        if len(rows) == 0:
            continue
        st = _stats(terms, rows)
        last = rows[-1]
        ex = panel.exch[last]
        ok = np.isfinite(st["quoted"]) & (st["quoted"] > 0) & np.isfinite(st["ar"]) & np.isfinite(st["cs"]) \
            & np.isin(ex, ["N", "A"]) & (panel.prc[last] > 0) & (panel.cap[last] > 0)
        if ok.any():
            X.append(_design(st["ar"][ok], st["cs"][ok], panel.prc[last][ok], panel.cap[last][ok]))
            y.append(np.log(st["quoted"][ok]))
    if not X or sum(len(v) for v in y) < min_obs:
        raise ValueError(f"calibrate_spread_map: fewer than {min_obs} NYSE/AMEX stock-months in {period}")
    X, y = np.vstack(X), np.concatenate(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    return {"coef": coef, "smear": float(np.mean(np.exp(res))), "n": int(len(y)),
            "r2": float(1 - res.var() / y.var()), "period": tuple(period)}


# ------------------------------------------------------------------------------------ monthly table
def half_spread_table(panel, months, calib: dict | None = None) -> pd.DataFrame:
    """Half-spread in bps, index = `months` (PeriodIndex), columns = panel PERMNOs, never NaN. Row m is
    estimated from month m-1 only. Needs the previous month in the panel for the quoted/estimator sources;
    a stock with nothing there gets the fallback. Pre-1993 NYSE/AMEX stocks need `calib` (else fallback)."""
    months = pd.PeriodIndex(months, freq="M")
    terms = _daily_terms(panel)
    mon = panel.days.to_period("M")
    k = len(panel.permnos)
    out = np.full((len(months), k), np.nan)
    src = np.zeros((len(months), 3), dtype=int)
    reliable = pd.Period(CC.QUOTES_RELIABLE_FROM, freq="M")
    for i, m in enumerate(months):
        prev = m - 1
        rows = np.nonzero(mon == prev)[0]
        est = np.full(k, np.nan)
        kind = np.zeros(k, int)            # 1 quoted, 2 calibrated, 0 none
        ex = np.full(k, "", dtype=object)
        px = np.full(k, np.nan)
        if len(rows):
            st = _stats(terms, rows)
            last = rows[-1]
            ex, px = panel.exch[last].astype(object), panel.prc[last]
            use_q = np.isfinite(st["quoted"]) & ((prev >= reliable) | (ex == "Q"))
            est = np.where(use_q, st["quoted"], np.nan)
            kind[use_q] = 1
            if calib is not None:
                need = ~use_q & np.isfinite(st["ar"]) & np.isfinite(st["cs"]) & (px > 0) & (panel.cap[last] > 0) \
                    & (prev < reliable) & np.isin(ex, ["N", "A"])
                if need.any():
                    Xd = _design(st["ar"][need], st["cs"][need], px[need], panel.cap[last][need])
                    est[need] = np.exp(Xd @ calib["coef"]) * calib["smear"]
                    kind[need] = 2
        have = np.isfinite(est)
        if (~have).any():
            fb = _fallback(est, ex, px, have)
            est = np.where(have, est, fb)
        out[i] = est
        src[i] = [(kind == 1).sum(), (kind == 2).sum(), (~have).sum()]
    if not np.isfinite(out).all():
        raise ValueError("half_spread_table: NaN left after the fallback")
    t = pd.DataFrame(out, index=months, columns=panel.permnos)
    t.attrs["sources"] = pd.DataFrame(src, index=months, columns=["quoted", "calibrated", "fallback"])
    return t


def _fallback(est, ex, px, have) -> np.ndarray:
    """SPREAD_FALLBACK_MULT x median of (exchange, price decile), else exchange, else all; raise if empty."""
    if not have.any():
        raise ValueError("half_spread_table: no spread estimate at all in a month; cannot form the fallback")
    dec = np.full(len(est), -1)
    ok = have & np.isfinite(px)
    if ok.sum() >= 10:
        edges = np.nanquantile(px[ok], np.linspace(0, 1, 11)[1:-1])
        dec = np.where(np.isfinite(px), np.searchsorted(edges, px), -1)
    fb = np.full(len(est), np.nan)
    allmed = np.median(est[have])
    for j in np.nonzero(~have)[0]:
        g = have & (ex == ex[j]) & (dec == dec[j]) & (dec[j] >= 0)
        if not g.any():
            g = have & (ex == ex[j]) & (ex[j] != "")
        fb[j] = np.median(est[g]) if g.any() else allmed
    return CC.SPREAD_FALLBACK_MULT * fb


# ------------------------------------------------------------------------------------ cost function
def make_cost_fn(bars: K.DailyBars, impact_bps: float = CC.IMPACT_BPS, open_mult: float = CC.OPEN_SPREAD_MULT,
                 flat_commission: float | None = None, tick: bool = True) -> K.CostFn:
    """contracts.CostFn: one-way bps per leg = mult x (spread + commission + impact), where
    spread = max(half-spread, half a tick / price) (no cap; x `open_mult` on t+1-open fills), commission =
    the Do & Faff (2012) bps schedule by fill year (or `flat_commission` $/share / price), impact = `impact_bps`. `rows` are order rows;
    the date used for ticks and commissions is that of the fill day."""
    def cost_fn(hs_bps, raw_price, rows, cols, mult=1.0):
        rows, cols = np.asarray(rows), np.asarray(cols)
        px = np.asarray(raw_price, dtype=float)
        if not np.isfinite(hs_bps).all():
            raise ValueError("cost_fn: NaN half-spread")
        day = bars.days.values[bars.fill_day[rows]][:, None]
        ex = bars.exch[np.ix_(bars.fill_day[rows], cols)]
        with np.errstate(divide="ignore", invalid="ignore"):
            floor = 0.5 * tick_size(ex, day) / px * _BPS if tick else 0.0
            hs = np.maximum(np.asarray(hs_bps, dtype=float)[None, :], floor)
            if bars.open_fill is not None:
                hs = np.where(bars.open_fill[np.ix_(rows, cols)], open_mult * hs, hs)
            yr = bars.days.year.values[bars.fill_day[rows]][:, None]
            comm = (flat_commission / px * _BPS) if flat_commission is not None else commission_bps(yr)
        return mult * (hs + comm + impact_bps)
    return cost_fn
