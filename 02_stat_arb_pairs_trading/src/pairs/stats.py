"""Statistics helpers: Newey-West regressions, clustered t-stats, Sharpe inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

TRADING_DAYS = 252


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = 5) -> dict:
    """OLS with Newey-West (Bartlett) HAC standard errors. X should include a constant."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {"beta": beta, "se": se, "t": beta / se, "resid": u,
            "r2": 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))}


def factor_regression(port: pd.Series, factors: pd.DataFrame, cols: list[str], lags: int = 5) -> dict:
    """Regress daily excess returns on factors; alpha annualised (x252)."""
    d = pd.concat([port.rename("p"), factors], axis=1, join="inner").dropna()
    y = d["p"] - d["rf"]
    X = np.column_stack([np.ones(len(d))] + [d[c] for c in cols])
    r = newey_west_ols(y.to_numpy(), X, lags)
    out = {"alpha_ann": r["beta"][0] * TRADING_DAYS, "alpha_t": r["t"][0], "r2": r["r2"], "n_days": len(d)}
    for i, c in enumerate(cols, start=1):
        out[f"beta_{c}"] = r["beta"][i]
        out[f"t_{c}"] = r["t"][i]
    return out


def clustered_mean(x: pd.Series, clusters: pd.Series) -> tuple[float, float, float]:
    """Mean with a cluster-robust standard error (events clustered by calendar month to
    allow for cross-correlation of returns of events filed at the same time)."""
    d = pd.DataFrame({"x": x, "c": clusters}).dropna()
    n = len(d)
    if n < 3:
        return np.nan, np.nan, np.nan
    m = d["x"].mean()
    g = (d["x"] - m).groupby(d["c"]).sum()
    G = len(g)
    var = (g ** 2).sum() / n ** 2 * (G / max(G - 1, 1))
    se = np.sqrt(var)
    return m, se, m / se if se > 0 else np.nan


def perf_stats(r: pd.Series, rf: pd.Series | None = None) -> dict:
    """Standard performance summary of a daily simple-return series."""
    r = r.dropna()
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = ex.mean() / ex.std() * np.sqrt(TRADING_DAYS) if ex.std() > 0 else np.nan
    downside = ex[ex < 0].std() * np.sqrt(TRADING_DAYS)
    dd = eq / eq.cummax() - 1
    return {
        "start": r.index[0].date(), "end": r.index[-1].date(), "years": round(years, 2),
        "cagr": cagr, "vol": vol, "sharpe": sharpe,
        "sortino": ex.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        "max_dd": dd.min(), "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "total_return": eq.iloc[-1] - 1, "hit_rate_daily": (r > 0).mean(),
        "skew": st.skew(r), "kurt": st.kurtosis(r, fisher=False),
    }


def sharpe_bootstrap_ci(ex: pd.Series, n_boot: int = 2000, block: int = 21, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap CI (fixed block length) for the annualised Sharpe ratio."""
    x = ex.dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    srs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        s = x[idx]
        srs[b] = s.mean() / s.std() * np.sqrt(TRADING_DAYS)
    return tuple(np.quantile(srs, [alpha / 2, 1 - alpha / 2]))


def deflated_sharpe(sr_ann: float, n_obs: int, skew: float, kurt: float, n_trials: int,
                    sr_trials_ann: list[float] | None = None) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio: probability that the true
    Sharpe is > 0 after accounting for the number of strategy variants tried."""
    sr = sr_ann / np.sqrt(TRADING_DAYS)                      # per-period Sharpe
    if sr_trials_ann is not None and len(sr_trials_ann) > 1:
        var_sr = np.var(np.asarray(sr_trials_ann) / np.sqrt(TRADING_DAYS), ddof=1)
    else:
        var_sr = 1.0 / n_obs
    gamma = 0.5772156649
    if n_trials > 1:
        sr0 = np.sqrt(var_sr) * ((1 - gamma) * st.norm.ppf(1 - 1 / n_trials)
                                 + gamma * st.norm.ppf(1 - 1 / (n_trials * np.e)))
    else:
        sr0 = 0.0
    z = (sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(st.norm.cdf(z))


# =====================================================================================================
# Inference for the CRSP study (pre-registered tests). numpy / scipy only: nothing here imports statsmodels.
# =====================================================================================================
def _hac_cov(U: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel long-run covariance of the columns of U (T, k), U already centred."""
    T = len(U)
    S = U.T @ U / T
    for L in range(1, min(lags, T - 1) + 1):
        G = U[L:].T @ U[:-L] / T
        S += (1 - L / (lags + 1)) * (G + G.T)
    return S


def nw_lags(n: int) -> int:
    """Newey-West (1994) rule of thumb for the Bartlett lag count: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_mean_test(x, lags: int = 6, side: str = "greater") -> dict:
    """t-test of mean(x) = 0 with a Newey-West (Bartlett) standard error (Newey & West 1987).
    side = "greater" tests mean > 0, "less" mean < 0, "two" two-sided. p-values use Student t with n - 1 df."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("newey_west_mean_test: NaN in input")
    r = newey_west_ols(x, np.ones((len(x), 1)), lags)
    t, n = float(r["t"][0]), len(x)
    p = {"greater": st.t.sf(t, n - 1), "less": st.t.cdf(t, n - 1), "two": 2 * st.t.sf(abs(t), n - 1)}[side]
    return {"mean": float(r["beta"][0]), "se": float(r["se"][0]), "t": t, "p": float(p), "n": n, "lags": lags}


def holm(p, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm (1979) step-down correction. Returns (adjusted p-values, reject mask at `alpha`)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= alpha


def politis_white_block(x, max_block: int | None = None) -> float:
    """Optimal mean block length for the *stationary* bootstrap: Politis & White (2004) with the
    Patton, Politis & White (2009) correction. Flat-top lag window, bandwidth chosen by the autocorrelation rule."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - x.mean()
    kmax = int(min(n - 2, np.ceil(np.sqrt(n)) + 5))
    gam = np.array([xc[k:] @ xc[:n - k] / n for k in range(kmax + 1)])
    rho = gam / gam[0] if gam[0] > 0 else np.zeros_like(gam)
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    crit = 2 * np.sqrt(np.log10(n) / n)
    mhat = kmax
    for m in range(0, kmax - kn + 1):                       # first m after which kn autocorrelations are all small
        if np.all(np.abs(rho[m + 1:m + kn + 1]) < crit):
            mhat = m
            break
    M = min(2 * max(mhat, 1), kmax)
    k = np.arange(-M, M + 1)
    tt = np.abs(k) / M
    lam = np.where(tt <= 0.5, 1.0, 2 * (1 - tt))
    g = gam[np.abs(k)]
    G = np.sum(lam * np.abs(k) * g)
    g0 = np.sum(lam * g)
    d_sb = 2 * g0 ** 2
    b = (2 * G ** 2 / d_sb) ** (1 / 3) * n ** (1 / 3) if d_sb > 0 else 1.0
    cap = max_block if max_block is not None else int(np.ceil(min(3 * np.sqrt(n), n / 3)))
    return float(np.clip(b, 1.0, max(cap, 1)))


def stationary_bootstrap_indices(n: int, reps: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(reps, n) resampling indices of the Politis-Romano (1994) stationary bootstrap: blocks start at a uniform
    position and have geometric length with the given mean; the series is treated as circular."""
    p = 1.0 / max(mean_block, 1.0)
    new = rng.random((reps, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, (reps, n))
    t = np.arange(n)[None, :]
    begin = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    return (np.take_along_axis(starts, begin, axis=1) + (t - begin)) % n


def sharpe_stationary_bootstrap(ex, reps: int = 2000, mean_block: float | None = None, alpha: float = 0.05,
                                seed: int = 0, periods: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe ratio of `ex` with a Politis-Romano stationary-bootstrap percentile CI. The mean block
    is Politis-White unless given. Returns the point estimate, the two-sided (alpha) CI and the one-sided lower
    bound at level alpha (H1 is supported when that bound is above 0)."""
    x = np.asarray(ex, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("sharpe_stationary_bootstrap: NaN in input")
    n = len(x)
    block = politis_white_block(x) if mean_block is None else float(mean_block)
    idx = stationary_bootstrap_indices(n, reps, block, np.random.default_rng(seed))
    s = x[idx]
    sd = s.std(axis=1, ddof=1)
    srs = np.where(sd > 0, s.mean(axis=1) / np.where(sd > 0, sd, 1), np.nan) * np.sqrt(periods)
    lo, hi = np.nanquantile(srs, [alpha / 2, 1 - alpha / 2])
    return {"sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(periods)), "lo": float(lo), "hi": float(hi),
            "lower_one_sided": float(np.nanquantile(srs, alpha)), "p_le_0": float(np.nanmean(srs <= 0)),
            "block": block, "reps": reps, "n": n}


def ledoit_wolf_sharpe_diff(r1, r2, lags: int | None = None, periods: int = TRADING_DAYS) -> dict:
    """Ledoit & Wolf (2008) test of H0: Sharpe(r1) = Sharpe(r2), HAC version (delta method, Bartlett kernel).

    Uses the moments (mu1, mu2, g1, g2) = (E r1, E r2, E r1^2, E r2^2): SR = mu / sqrt(g - mu^2). The one-sided
    p-value is for SR1 > SR2. Returns annualised Sharpe ratios and their difference."""
    a, b = np.asarray(r1, float), np.asarray(r2, float)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("ledoit_wolf_sharpe_diff: need two equal-length NaN-free series")
    T = len(a)
    lags = nw_lags(T) if lags is None else lags
    m1, m2, g1, g2 = a.mean(), b.mean(), (a * a).mean(), (b * b).mean()
    U = np.column_stack([a - m1, b - m2, a * a - g1, b * b - g2])
    Psi = _hac_cov(U, lags)
    v1, v2 = g1 - m1 ** 2, g2 - m2 ** 2
    grad = np.array([g1 / v1 ** 1.5, -g2 / v2 ** 1.5, -0.5 * m1 / v1 ** 1.5, 0.5 * m2 / v2 ** 1.5])
    se = float(np.sqrt(grad @ Psi @ grad / T))
    d = m1 / np.sqrt(v1) - m2 / np.sqrt(v2)
    z = d / se
    k = np.sqrt(periods)
    return {"sr1": float(m1 / np.sqrt(v1) * k), "sr2": float(m2 / np.sqrt(v2) * k), "diff": float(d * k),
            "se": se * k, "z": float(z), "p_one_sided": float(st.norm.sf(z)), "p_two_sided": float(2 * st.norm.sf(abs(z))),
            "lags": lags, "n": T}


def mde_sharpe(years: float, corr: float | None = None, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable annual Sharpe (one-sided test, given power) from SE(SR) ~ 1 / sqrt(years): a single
    Sharpe if `corr` is None, otherwise the difference of two Sharpes with that correlation (review power note).
    About 0.45 for 30.8 years and 1.4 for 3 years; 0.53 for the difference at corr = 0.3."""
    se = np.sqrt((1.0 if corr is None else 2 * (1 - corr)) / years)
    return float((st.norm.ppf(1 - alpha) + st.norm.ppf(power)) * se)


def mde_trend_change(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Total change over the sample, in units of the monthly SD, that a linear trend on `n` points detects with the
    given power (one-sided, iid errors): about 0.41 SD for n = 432 (H4 is low-powered by design)."""
    se_slope = 1.0 / np.sqrt(n * (n ** 2 - 1) / 12.0)
    return float((st.norm.ppf(1 - alpha) + st.norm.ppf(power)) * se_slope * n)


def trend_test_with_break(y: pd.Series, break_at: str = "2001-04", lags: int = 12) -> dict:
    """Trend of a monthly series with a pre-specified break (H4: gross monthly returns, NW(12), one-sided).

    Model A: y = a + b * years.                      b < 0 is the H4 alternative.
    Model B: y = a + b1 * years + c * post + d * post * (years - break); post = 1 from `break_at`.
             Reports the pre-break slope b1, the post-break slope b1 + d, the level shift c and their NW t-stats.
    Slopes are per year. One-sided p-values (normal) for slope < 0."""
    y = y.dropna()
    if not isinstance(y.index, pd.PeriodIndex):
        y = y.copy()
        y.index = y.index.to_period("M")
    t = (y.index.year - y.index.year[0] + (y.index.month - 1) / 12.0).to_numpy(float)
    yrs = t - t[0]
    post = (y.index >= pd.Period(break_at, "M")).astype(float)
    tb = yrs[post.astype(bool)][0] if post.any() else np.nan
    v = y.to_numpy(float)
    A = newey_west_ols(v, np.column_stack([np.ones(len(v)), yrs]), lags)
    out = {"n": len(v), "slope": float(A["beta"][1]), "slope_t": float(A["t"][1]),
           "slope_p_neg": float(st.norm.cdf(A["t"][1])), "lags": lags}
    if post.any() and (post == 0).any():
        XB = np.column_stack([np.ones(len(v)), yrs, post, post * (yrs - tb)])
        Bm = newey_west_ols(v, XB, lags)
        V = np.linalg.inv(XB.T @ XB)                          # HAC covariance for the post-break slope b1 + d
        Xu = XB * Bm["resid"][:, None]
        S = Xu.T @ Xu
        for L in range(1, lags + 1):
            G = Xu[L:].T @ Xu[:-L]
            S += (1 - L / (lags + 1)) * (G + G.T)
        cov = V @ S @ V
        w = np.array([0, 1, 0, 1.0])
        se_post = float(np.sqrt(w @ cov @ w))
        sp = float(Bm["beta"][1] + Bm["beta"][3])
        out.update({"slope_pre": float(Bm["beta"][1]), "slope_pre_t": float(Bm["t"][1]),
                    "slope_post": sp, "slope_post_t": sp / se_post, "level_shift": float(Bm["beta"][2]),
                    "level_shift_t": float(Bm["t"][2]), "n_pre": int((post == 0).sum()), "n_post": int(post.sum())})
    return out
