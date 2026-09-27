"""Hedge ratios and z-scores for a set of pairs over one trading month (vectorised across pairs).

All methods work on *centred* log prices  y~ = log P_y - mean_formation,  x~ likewise, and return
arrays of shape (n_bars, n_pairs):

    static  : hedge ratio and spread std fixed at their formation-window OLS values
              z = (y~ - b x~ - a) / sigma_formation                        (Gatev et al. style)
    rolling : OLS of y~ on x~ over the previous ROLLING_WINDOW_DAYS of bars (at least 20 bars)
              z = out-of-sample residual_t / std(residuals in the window)   (Bollinger style)
    kalman  : state [b, a] follows a random walk; y~_t = b_t x~_t + a_t + e_t
              z = innovation_t / std(innovations over the trailing window)  (Chan 2013 style,
              with the innovations standardised empirically so z is on the same scale as above)

Every quantity at bar t uses prices up to and including bar t only; orders are filled one minute
*after* the bar closes (see backtest.py), so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def static(y: np.ndarray, x: np.ndarray, f: dict) -> tuple[np.ndarray, np.ndarray]:
    beta = np.broadcast_to(f["beta"], y.shape).copy()
    z = (y - f["beta"] * x) / f["sigma"]           # centred data: OLS intercept is ~0
    return z, beta


def rolling(y: np.ndarray, x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling OLS fitted on the `window` bars *before* bar t (t-window .. t-1); bar t is then scored
    out of sample: z_t = (y~_t - a - b x~_t) / residual std of the window fit. Keeping bar t out of
    its own fit matters: an in-sample residual is mechanically bounded, so large |z| (and stop-losses)
    could never occur. `y`, `x` include `window` warm-up bars before the month."""
    t = np.arange(y.shape[0])
    lo = np.maximum(t - window, 0)

    def rsum(a):
        c = np.vstack([np.zeros((1,) + a.shape[1:]), np.cumsum(np.nan_to_num(a), axis=0)])
        return c[t] - c[lo]

    ok = np.isfinite(y) & np.isfinite(x)
    yy, xx = np.where(ok, y, 0.0), np.where(ok, x, 0.0)
    n = rsum(ok.astype(float))
    sx, sy, sxx, sxy, syy = rsum(xx), rsum(yy), rsum(xx * xx), rsum(xx * yy), rsum(yy * yy)
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = sxx - sx * sx / n
        b = (sxy - sx * sy / n) / vx
        a = (sy - b * sx) / n
        rss = syy - a * sy - b * sxy                 # residual sum of squares of the window fit
        sd = np.sqrt(np.clip(rss, 1e-18, None) / (n - 2))
        z = (y - a - b * x) / sd
    bad = (n < max(20, window // 2)) | ~ok
    z[bad] = np.nan
    b[bad] = np.nan
    return z, b


def kalman(y: np.ndarray, x: np.ndarray, f: dict, dt_days: float, window: int,
           memory_days: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Kalman filter, vectorised over pairs. `y`, `x` include `window` warm-up bars.

    Observation noise R = formation residual variance. The state noise per bar is
    Q = R * (dt / memory)^2 (for a, and scaled by 1/var(x~) for b), which makes the filter behave
    like an exponential average with a ~`memory_days` memory at every bar size
    (the steady-state gain of a local-level filter is ~sqrt(Q/R) per step).
    The innovation e_t = y~_t - (b_{t|t-1} x~_t + a_{t|t-1}) is standardised by the standard deviation
    of the previous `window` innovations."""
    memory_days = C.KALMAN_MEMORY_DAYS if memory_days is None else memory_days
    n, k = y.shape
    R = f["sigma"] ** 2
    step = (dt_days / memory_days) ** 2
    qa = R * step
    qb = R / np.maximum(f["sd_lx"] ** 2, 1e-8) * step
    b = f["beta"].astype(float).copy()
    a = np.zeros(k)
    p11, p12, p22 = f["var_beta"].astype(float).copy(), np.zeros(k), f["var_alpha"].astype(float).copy()
    innov = np.full((n, k), np.nan)
    beta = np.full((n, k), np.nan)
    for t in range(n):
        yt, xt = y[t], x[t]
        ok = np.isfinite(yt) & np.isfinite(xt)
        xt0 = np.where(ok, xt, 0.0)
        p11 = p11 + qb
        p22 = p22 + qa
        e = np.where(ok, yt, 0.0) - (b * xt0 + a)
        hp1 = p11 * xt0 + p12
        hp2 = p12 * xt0 + p22
        S = hp1 * xt0 + hp2 + R
        innov[t] = np.where(ok, e, np.nan)
        beta[t] = b
        k1 = np.where(ok, hp1 / S, 0.0)
        k2 = np.where(ok, hp2 / S, 0.0)
        b = b + k1 * e
        a = a + k2 * e
        p11 = p11 - k1 * hp1
        p12 = p12 - k1 * hp2
        p22 = p22 - k2 * hp2
    ok = np.isfinite(innov)
    e0 = np.where(ok, innov, 0.0)
    c1 = np.cumsum(np.vstack([np.zeros((1, k)), e0]), axis=0)
    c2 = np.cumsum(np.vstack([np.zeros((1, k)), e0 * e0]), axis=0)
    cn = np.cumsum(np.vstack([np.zeros((1, k)), ok.astype(float)]), axis=0)
    lo = np.maximum(np.arange(n) - window, 0)
    hi = np.arange(n)                                  # previous `window` bars, excluding t
    m = cn[hi] - cn[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = (c1[hi] - c1[lo]) / m
        sd = np.sqrt(np.clip((c2[hi] - c2[lo]) / m - mu * mu, 1e-18, None) * m / np.maximum(m - 1, 1))
        z = innov / sd
    z[(m < max(20, window // 2)) | ~ok] = np.nan
    return z, beta
