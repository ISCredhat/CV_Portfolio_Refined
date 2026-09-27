"""Time-series tests used for pair selection, implemented from scratch in numpy.

* Augmented Dickey-Fuller test with BIC lag selection and MacKinnon (1994) p-values
* Engle-Granger two-step cointegration test (MacKinnon p-values for 2 variables)
* Johansen trace test (2 variables, constant restricted to the cointegrating relation)
* Ornstein-Uhlenbeck half-life, Hurst exponent, Benjamini-Hochberg FDR control

The implementations follow `statsmodels.tsa.stattools.adfuller / coint` and
`statsmodels.tsa.vector_ar.vecm.coint_johansen`; `tests/test_pairs.py` checks them against
statsmodels when it is installed, and against Monte Carlo size/power otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------- MacKinnon p-values
# MacKinnon (1994) response-surface coefficients, constant-only case ('c'), rows = N variables.
_TAU_MAX_C = [2.74, 0.92, 0.55, 0.61, 0.79, 1.0]
_TAU_MIN_C = [-18.83, -18.86, -23.48, -28.07, -25.96, -23.27]
_TAU_STAR_C = [-1.61, -2.62, -3.13, -3.47, -3.78, -3.93]
_TAU_C_SMALLP = np.array([[2.1659, 1.4412, 3.8269e-2], [2.92, 1.5012, 3.9796e-2],
                          [3.4699, 1.4856, 3.164e-2], [3.9673, 1.4777, 2.6315e-2],
                          [4.5509, 1.5338, 2.9545e-2], [5.1399, 1.6036, 3.4445e-2]])
_TAU_C_LARGEP = np.array([[1.7339, 9.3202, -1.2745, -1.0368], [2.1945, 6.4695, -2.9198, -4.2377],
                          [2.5893, 4.5168, -3.6529, -5.0074], [3.0387, 4.5452, -3.3666, -4.1921],
                          [3.5049, 5.2098, -2.9158, -3.3468], [3.9489, 5.8933, -2.5359, -2.721]]) \
    * np.array([1, 1e-1, 1e-1, 1e-2])

# Johansen trace critical values (90%, 95%, 99%) for H0: r = 0 with 2 variables and the constant
# restricted to the cointegrating relation (MacKinnon, Haug & Michelis 1999, case 2). Our own
# simulation (tests/test_pairs.py) reproduces them: 18.00 / 20.25 / 25.10.
JOHANSEN_CV_R0 = (17.98, 20.26, 24.99)


def mackinnon_p(stat: float, n_vars: int = 1) -> float:
    """Asymptotic p-value of a Dickey-Fuller t-statistic (constant, no trend)."""
    i = n_vars - 1
    if stat > _TAU_MAX_C[i]:
        return 1.0
    if stat < _TAU_MIN_C[i]:
        return 0.0
    coef = _TAU_C_SMALLP[i] if stat <= _TAU_STAR_C[i] else _TAU_C_LARGEP[i]
    return float(norm.cdf(np.polyval(coef[::-1], stat)))


# ---------------------------------------------------------------------- ADF
def _ols(y: np.ndarray, X: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def adf(x: np.ndarray, maxlag: int | None = None, constant: bool = True, autolag: bool = True) -> dict:
    """Augmented Dickey-Fuller test of a unit root in `x`.

    Regression: dx_t = [c] + g * x_{t-1} + sum_{i=1..p} a_i dx_{t-i} + e_t, t-stat on g.
    The lag order p <= maxlag minimises BIC on a common sample (as statsmodels' autolag="BIC").
    `constant=False` is used for Engle-Granger residuals (already mean zero)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    ntrend = 1 if constant else 0
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    maxlag = max(0, min(maxlag, n // 2 - ntrend - 1))
    dx = np.diff(x)

    def design(p: int, start: int):
        # rows t = start..n-2 of dx (dx[t] is the dependent variable)
        rows = np.arange(start, len(dx))
        cols = [x[rows]]                             # x_{t-1} in dx_t = x_{t+1} - x_t indexing
        cols += [dx[rows - i] for i in range(1, p + 1)]
        if constant:
            cols.append(np.ones(len(rows)))
        return dx[rows], np.column_stack(cols)

    p = maxlag
    if autolag and maxlag > 0:
        best = (np.inf, 0)
        for lag in range(0, maxlag + 1):
            y, X = design(lag, maxlag)               # common sample for every candidate lag
            _, e = _ols(y, X)
            nobs = len(y)
            llf = -nobs / 2 * (np.log(2 * np.pi) + np.log(e @ e / nobs) + 1)
            bic = -2 * llf + X.shape[1] * np.log(nobs)
            if bic < best[0]:
                best = (bic, lag)
        p = best[1]
    y, X = design(p, p)
    beta, e = _ols(y, X)
    nobs, k = X.shape
    s2 = e @ e / (nobs - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    stat = beta[0] / np.sqrt(cov[0, 0])
    # p-value only for the constant case (the no-constant variant is used inside Engle-Granger,
    # which applies its own 2-variable response surface)
    pval = mackinnon_p(stat, 1) if constant else np.nan
    return {"stat": float(stat), "pvalue": pval, "lags": p, "nobs": nobs}


# ---------------------------------------------------------------------- Engle-Granger
def engle_granger(y: np.ndarray, x: np.ndarray, maxlag: int | None = None) -> dict:
    """Engle-Granger test: OLS y = a + b x + u, then ADF (no constant) on u.
    p-value from MacKinnon's 2-variable response surface (as statsmodels.coint)."""
    X = np.column_stack([np.ones(len(x)), x])
    (a, b), u = _ols(y, X)
    r = adf(u, maxlag=maxlag, constant=False)
    return {"stat": r["stat"], "pvalue": mackinnon_p(r["stat"], 2), "alpha": float(a), "beta": float(b),
            "resid": u, "lags": r["lags"]}


# ---------------------------------------------------------------------- Johansen
def johansen_trace(Y: np.ndarray, k_ar_diff: int = 1) -> dict:
    """Johansen trace test of H0: no cointegration (r = 0) for two log-price series.

    VECM: dx_t = a (b' x_{t-1} + c) + G dx_{t-1} + e_t, i.e. the constant sits *inside* the
    cointegrating relation (a pair spread has a non-zero mean but no drift). Note: statsmodels'
    `coint_johansen(det_order=0)` uses the unrestricted-constant critical values, which
    over-reject for driftless prices (about 11% at a nominal 5% in our simulation)."""
    x = np.asarray(Y, dtype=float)
    dx = np.diff(x, axis=0)
    n, m = dx.shape
    k = k_ar_diff
    Z0 = dx[k:]
    Z1 = np.column_stack([x[k:-1], np.ones(n - k)])
    Z2 = np.column_stack([dx[k - i:n - i] for i in range(1, k + 1)])

    def resid(a, b):
        return a - b @ np.linalg.lstsq(b, a, rcond=None)[0]

    R0, R1 = resid(Z0, Z2), resid(Z1, Z2)
    T = R0.shape[0]
    S00, S11, S01 = R0.T @ R0 / T, R1.T @ R1 / T, R0.T @ R1 / T
    M = np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))
    eig = np.clip(np.sort(np.real(np.linalg.eigvals(M)))[::-1][:m], 0, 1 - 1e-12)
    trace0 = float(-T * np.sum(np.log(1 - eig)))
    return {"trace": trace0, "cv": JOHANSEN_CV_R0, "eig": eig}


# ---------------------------------------------------------------------- spread diagnostics
def half_life(s: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life (in bars) from ds_t = a + b s_{t-1} + e: -ln 2 / ln(1 + b)."""
    s = np.asarray(s, dtype=float)
    X = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    (_, b), _ = _ols(np.diff(s), X)
    if b >= 0 or b <= -1:
        return np.inf if b >= 0 else 0.0
    return float(-np.log(2) / np.log(1 + b))


def hurst(s: np.ndarray, max_lag: int = 100) -> float:
    """Hurst exponent from the scaling of lagged differences: std(s_{t+k} - s_t) ~ k^H.
    H < 0.5 mean-reverting, H = 0.5 random walk, H > 0.5 trending."""
    s = np.asarray(s, dtype=float)
    lags = np.unique(np.logspace(np.log10(2), np.log10(min(max_lag, len(s) // 4)), 20).astype(int))
    tau = np.array([np.std(s[k:] - s[:-k]) for k in lags])
    ok = tau > 0
    return float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of hypotheses rejected at false discovery rate q (BH step-up)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    below = p[order] <= thresh
    reject = np.zeros(m, bool)
    if below.any():
        k = np.max(np.nonzero(below)[0])
        reject[order[: k + 1]] = True
    return reject
