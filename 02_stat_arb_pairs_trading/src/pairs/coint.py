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


# ---------------------------------------------------------------------- batched Engle-Granger
def mackinnon_p_vec(stat: np.ndarray, n_vars: int = 1) -> np.ndarray:
    """Vectorised `mackinnon_p` (same response surface, same clipping at the tails)."""
    s = np.asarray(stat, dtype=float)
    i = n_vars - 1
    small = norm.cdf(np.polyval(_TAU_C_SMALLP[i][::-1], s))
    large = norm.cdf(np.polyval(_TAU_C_LARGEP[i][::-1], s))
    p = np.where(s <= _TAU_STAR_C[i], small, large)
    p = np.where(s > _TAU_MAX_C[i], 1.0, np.where(s < _TAU_MIN_C[i], 0.0, p))
    return np.where(np.isfinite(s), p, np.nan)


def _adf_nc_batch(U: np.ndarray, maxlag: int | None) -> tuple[np.ndarray, np.ndarray]:
    """ADF without constant on every column of U (n, P): BIC lag on a common sample, then the final
    regression on the sample implied by the chosen lag (exactly what `adf(constant=False)` does).
    Normal equations are solved for all pairs at once. Returns (t-stat, lag) per column."""
    n, P = U.shape
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    maxlag = max(0, min(maxlag, n // 2 - 1))
    dx = np.diff(U, axis=0)

    def design(Ucols, dcols, p, start):
        rows = np.arange(start, n - 1)
        X = np.stack([Ucols[rows]] + [dcols[rows - i] for i in range(1, p + 1)], axis=2).transpose(1, 0, 2)
        return dcols[rows].T, X                              # (P, m), (P, m, 1 + p)

    best_lag = np.zeros(P, dtype=int)
    if maxlag > 0:
        y, X = design(U, dx, maxlag, maxlag)
        m = y.shape[1]
        G = np.einsum("pmi,pmj->pij", X, X)
        Zy = np.einsum("pmi,pm->pi", X, y)
        bic = np.empty((maxlag + 1, P))
        for lag in range(maxlag + 1):
            k = lag + 1
            b = np.linalg.solve(G[:, :k, :k], Zy[:, :k, None])[:, :, 0]
            e = y - np.einsum("pmi,pi->pm", X[:, :, :k], b)
            rss = np.maximum((e * e).sum(1), 1e-300)
            bic[lag] = m * (np.log(2 * np.pi) + np.log(rss / m) + 1) + k * np.log(m)
        best_lag = np.argmin(bic, axis=0)                    # first minimum, as the strict < in `adf`
    stat = np.empty(P)
    for p in np.unique(best_lag):
        idx = np.nonzero(best_lag == p)[0]
        y, X = design(U[:, idx], dx[:, idx], int(p), int(p))
        m, k = y.shape[1], int(p) + 1
        G = np.einsum("pmi,pmj->pij", X, X)
        Ginv = np.linalg.inv(G)
        b = np.einsum("pij,pj->pi", Ginv, np.einsum("pmi,pm->pi", X, y))
        e = y - np.einsum("pmi,pi->pm", X, b)
        s2 = (e * e).sum(1) / (m - k)
        stat[idx] = b[:, 0] / np.sqrt(s2 * Ginv[:, 0, 0])
    return stat, best_lag


def engle_granger_batch(Y: np.ndarray, X: np.ndarray, maxlag: int | None = None) -> dict:
    """Engle-Granger test for many pairs at once: column j of Y is regressed on column j of X.

    Same statistic, lag choice and MacKinnon p-value as `engle_granger` applied column by column (the test
    suite checks 1e-8), but the OLS and all the ADF regressions are batched, which is what makes the
    ~3-7k within-industry pairs per month affordable (review M8). Y and X are (n, P) and NaN-free; pairs
    with different numbers of observations must be passed in separate calls. A pair whose residual is
    numerically zero (identical series) gets NaN stat and p-value so that the caller can count it."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if Y.shape != X.shape or Y.ndim != 2:
        raise ValueError("Y and X must both be (n, P)")
    if not (np.isfinite(Y).all() and np.isfinite(X).all()):
        raise ValueError("engle_granger_batch needs NaN-free input")
    n, P = Y.shape
    mx, my = X.mean(0), Y.mean(0)
    xc, yc = X - mx, Y - my
    sxx = (xc * xc).sum(0)
    beta = (xc * yc).sum(0) / sxx
    alpha = my - beta * mx
    U = yc - beta * xc
    scale = np.maximum((yc * yc).sum(0), 1e-300)
    bad = ~np.isfinite(beta) | ((U * U).sum(0) <= 1e-20 * scale)
    stat = np.full(P, np.nan)
    lags = np.zeros(P, dtype=int)
    ok = np.nonzero(~bad)[0]
    if len(ok):
        stat[ok], lags[ok] = _adf_nc_batch(U[:, ok], maxlag)
    return {"stat": stat, "pvalue": mackinnon_p_vec(stat, 2), "alpha": alpha, "beta": beta, "resid": U,
            "lags": lags, "degenerate": bad}


def half_life_batch(S: np.ndarray) -> np.ndarray:
    """`half_life` for every column of S (n, P): OLS of ds_t on [1, s_{t-1}], -ln 2 / ln(1 + b)."""
    s0, ds = S[:-1], np.diff(S, axis=0)
    xc = s0 - s0.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (xc * (ds - ds.mean(0))).sum(0) / (xc * xc).sum(0)
        hl = -np.log(2) / np.log1p(b)
    return np.where(b >= 0, np.inf, np.where(b <= -1, 0.0, hl))


# ---------------------------------------------------------------------- simulated Johansen critical values
def johansen_cv_sim(T: int, reps: int = 20000, seed: int = 20261118) -> tuple[float, float, float]:
    """Finite-sample (90%, 95%, 99%) critical values of the trace test (r = 0, restricted constant, one lagged
    difference) for T observations of two independent driftless random walks with Gaussian steps.
    The asymptotic 20.26 under-rejects less at T = 126 than it should (review m7), so selection uses these."""
    rng = np.random.default_rng(seed)
    tr = np.empty(reps)
    for i in range(reps):
        tr[i] = johansen_trace(np.cumsum(rng.standard_normal((T, 2)), axis=0))["trace"]
    return tuple(float(q) for q in np.quantile(tr, [0.90, 0.95, 0.99]))


# Simulated with johansen_cv_sim(T, reps=40000, seed=JOHANSEN_SIM_SEED); regenerate with `python -m pairs.coint`.
JOHANSEN_SIM_SEED = 20261118
JOHANSEN_CV_SIM = {126: (18.414167527790283, 20.724192050999967, 25.67979997897743),
                   252: (18.207305744427106, 20.511770755852098, 25.567718807326045)}


def johansen_cv(T: int) -> tuple[float, float, float]:
    """Cached simulated critical values for the cached T nearest to `T` (126 = 6 months, 252 = 12 months)."""
    key = min(JOHANSEN_CV_SIM, key=lambda k: abs(k - T))
    return JOHANSEN_CV_SIM[key]


if __name__ == "__main__":                                 # pragma: no cover
    for T_ in (126, 252):
        print(T_, johansen_cv_sim(T_, 40000, JOHANSEN_SIM_SEED))
