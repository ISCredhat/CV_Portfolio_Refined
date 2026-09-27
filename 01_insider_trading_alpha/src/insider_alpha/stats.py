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
    """Stationary (moving) block bootstrap CI for the annualised Sharpe ratio."""
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
