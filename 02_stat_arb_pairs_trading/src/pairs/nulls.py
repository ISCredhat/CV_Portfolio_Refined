"""Null distribution for H2: how often does the Engle-Granger screen pass pairs that are not cointegrated?

The old iid-Gaussian null is wrong for real returns (review B6): volatility clustering and fat tails push the
5% screen to 6-8% at n = 126. The pre-registered null therefore resamples **each leg's own formation-window
returns independently** with the Politis-Romano stationary bootstrap (mean block 21 days; 5 and 63 as
sensitivities). That keeps every marginal feature of the data and removes any link between the two legs, so the
pass rate of the same screen on the resampled pairs is the rate under "no cointegration".

Statistic (H2): per month, excess = observed pass rate - null pass rate (percentage points), both computed on the
same candidate pairs with the same sample sizes. The monthly excess series is tested with a one-sided
Newey-West(6) t-test (months overlap by 5/6 of their formation days, so a binomial test would be invalid);
"supported" additionally needs a mean excess of at least `H2_MIN_EXCESS_PP` (review M10).

Calibration note (found while testing): on pure Gaussian random walks at n = 126 the iid resampling null passes
5.0% of pairs, but the stationary bootstrap passes 6.7% (mean block 5), 7.9% (block 21) and 7.2% (block 63): a
block bootstrap of one short sample is not size-correct for the cointegration screen. The H2 null is therefore
conservative (it sets the bar higher than the true null would). Both are reported; the iid Gaussian simulation
(`iid_gaussian_null_rate`) is the reference for the nominal level.

Primary null (integrator, 2026-11-18): because of that calibration problem, the pre-registered H2 null is the
**time-shifted partner** null (`kind="shifted"`): for a candidate pair (y, x) in month m, leg y keeps its own
formation window and leg x is replaced by **its own** formation window from month m - 36 (then m - 48, m - 60,
m + 36, m + 48, m + 60 if that window has too few observations). Both legs are real price paths with all their
fat tails and volatility clustering, but from different calendar periods, so they cannot be cointegrated with each
other. No resampling is involved, so on independent random walks the null is size-correct by construction (tested).
The shifted window can lie after m: that is allowed because the null rate is a diagnostic that never feeds a
trading decision. The stationary-bootstrap and iid nulls are reported as sensitivities.

Caches are keyed by (n_obs or formation days, null type, block, seed, month) so a change in any of them can
never return a stale result (review B7).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import coint as K
from . import config_crsp as CC
from . import selection_daily as SD
from . import stats as ST

H2_MIN_EXCESS_PP = CC.H2_MIN_EXCESS_PP
NW_LAGS = CC.NW_LAGS_MONTHLY


# ------------------------------------------------------------------------------------------------ cache
def cache_key(n_obs: int, kind: str, block: float, seed: int, tag: str = "") -> str:
    return f"null_{kind}_n{int(n_obs)}_b{block:g}_s{int(seed)}" + (f"_{tag}" if tag else "")


def cached(cache_dir: str | Path | None, key: str, fn):
    """Return the JSON dict stored under `key`, computing it with `fn()` on a miss. cache_dir=None disables caching."""
    if cache_dir is None:
        return fn()
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------------------------------------ screens
def _pass_flags(a: np.ndarray, b: np.ndarray, alpha: float, maxlag: int) -> np.ndarray:
    """p_eg = 2 x min p of the two directions, as selection_daily; True where p_eg <= alpha (degenerate -> False)."""
    p1 = K.engle_granger_batch(a, b, maxlag)["pvalue"]
    p2 = K.engle_granger_batch(b, a, maxlag)["pvalue"]
    p = np.minimum(1.0, 2 * np.minimum(np.nan_to_num(p1, nan=1.0), np.nan_to_num(p2, nan=1.0)))
    return p <= alpha


def iid_gaussian_null_rate(n_obs: int, reps: int = 4000, seed: int = CC.SEED, alpha: float = CC.ALPHA,
                           maxlag: int = CC.EG_MAXLAG, cache_dir=None) -> dict:
    """Reference only: pass rate of the screen on independent Gaussian random walks of length n_obs."""
    def run():
        rng = np.random.default_rng(seed)
        a = np.cumsum(rng.standard_normal((n_obs, reps)), 0)
        b = np.cumsum(rng.standard_normal((n_obs, reps)), 0)
        return {"rate": float(_pass_flags(a, b, alpha, maxlag).mean()), "n": reps, "n_obs": n_obs}
    return cached(cache_dir, cache_key(n_obs, "gauss", 1, seed, f"r{reps}"), run)


def resampled_null_flags(W: np.ndarray, ia: np.ndarray, ib: np.ndarray, kind: str = "stationary",
                         block: float = CC.BOOT_MEAN_BLOCK, seed: int = CC.SEED, reps_per_pair: int = CC.NULL_REPS_PER_PAIR,
                         alpha: float = CC.ALPHA, maxlag: int = CC.EG_MAXLAG) -> np.ndarray:
    """Screen outcome (bool) for every resampled pair: shape (len(ia) * reps_per_pair,).

    W (n_days, n_cols) holds the formation-window log prices; ia/ib are the columns of the pair legs. For each pair
    the joint observations are compressed in time order (the same sample the real test used), each leg's own
    log returns are resampled independently (`kind="stationary"`: stationary bootstrap with mean `block`;
    `kind="iid"`: iid resampling with replacement), and the price paths are rebuilt from the leg's first price."""
    if kind not in ("stationary", "iid"):
        raise ValueError(kind)
    mb = block if kind == "stationary" else 1.0
    rng = np.random.default_rng(seed)
    A, B = W[:, ia], W[:, ib]
    M = np.isfinite(A) & np.isfinite(B)
    nobs = M.sum(0)
    out = np.zeros((reps_per_pair, len(ia)), dtype=bool)
    for n in np.unique(nobs):
        n = int(n)
        idx = np.nonzero(nobs == n)[0]
        if n < 8:
            raise ValueError(f"pair with {n} joint observations reached the null")
        order = np.argsort(~M[:, idx], axis=0, kind="stable")[:n]
        a, b = np.take_along_axis(A[:, idx], order, 0), np.take_along_axis(B[:, idx], order, 0)
        ra, rb = np.diff(a, axis=0), np.diff(b, axis=0)                 # each leg's own returns, (n-1, P)
        for r in range(reps_per_pair):
            ia_ = ST.stationary_bootstrap_indices(n - 1, len(idx), mb, rng)    # (P, n-1)
            ib_ = ST.stationary_bootstrap_indices(n - 1, len(idx), mb, rng)
            sa = np.take_along_axis(ra, ia_.T, 0)
            sb = np.take_along_axis(rb, ib_.T, 0)
            pa = np.vstack([a[:1], a[:1] + np.cumsum(sa, 0)])
            pb = np.vstack([b[:1], b[:1] + np.cumsum(sb, 0)])
            out[r, idx] = _pass_flags(pa, pb, alpha, maxlag)
    return out.ravel()


SHIFT_MONTHS = (-36, -48, -60, 36, 48, 60)


def shifted_null_flags(logp: pd.DataFrame, cands: pd.DataFrame, month: pd.Period, formation_months: int,
                       min_obs: int, alpha: float = CC.ALPHA, maxlag: int = CC.EG_MAXLAG,
                       shifts=SHIFT_MONTHS) -> np.ndarray:
    """Screen outcome for every candidate pair with leg x replaced by its own window from a shifted month.

    Pairs for which no shift gives `min_obs` joint observations are dropped (returned flags cover the rest)."""
    P = logp.to_numpy(dtype=float)
    rows_y = SD.formation_window(logp.index, month, formation_months)
    iy, ix = cands["iy"].to_numpy(), cands["ix"].to_numpy()
    out = np.full(len(cands), -1, dtype=np.int8)
    todo = np.arange(len(cands))
    for sh in shifts:
        if len(todo) == 0:
            break
        rows_x = SD.formation_window(logp.index, month + sh, formation_months)
        if len(rows_x) < min_obs:
            continue
        Wy, Wx = P[rows_y][:, iy[todo]], P[rows_x][:, ix[todo]]
        ny, nx = np.isfinite(Wy).sum(0), np.isfinite(Wx).sum(0)
        n = np.minimum(ny, nx)
        ok = n >= min_obs
        for nn in np.unique(n[ok]):
            sel = np.nonzero(ok & (n == nn))[0]
            # compress each leg's finite observations in time order and keep the first nn of each
            oy = np.argsort(~np.isfinite(Wy[:, sel]), axis=0, kind="stable")[:nn]
            ox = np.argsort(~np.isfinite(Wx[:, sel]), axis=0, kind="stable")[:nn]
            a = np.take_along_axis(Wy[:, sel], oy, 0)
            b = np.take_along_axis(Wx[:, sel], ox, 0)
            out[todo[sel]] = _pass_flags(a, b, alpha, maxlag)
        todo = todo[~ok]
    return out[out >= 0].astype(bool)


# ------------------------------------------------------------------------------------------------ H2
def month_null_rate(logp: pd.DataFrame, cands: pd.DataFrame, month: pd.Period, kind: str = "shifted",
                    block: float = CC.BOOT_MEAN_BLOCK, seed: int = CC.SEED, formation_months: int = CC.FORMATION_MONTHS,
                    alpha: float = CC.ALPHA, maxlag: int = CC.EG_MAXLAG, cache_dir=None,
                    reps_per_pair: int = CC.NULL_REPS_PER_PAIR) -> dict:
    """Null pass rate of one month's candidate pairs. Reads only formation-window rows of `logp`."""
    rows = SD.formation_window(logp.index, month, formation_months)
    if len(rows) == 0:
        raise ValueError(f"no formation days before {month}")

    def run():
        if cands.empty:
            return {"rate": float("nan"), "n": 0}
        if kind == "shifted":
            f = shifted_null_flags(logp, cands, month, formation_months,
                                   int(np.ceil(CC.MIN_OBS_SHARE * len(rows))), alpha, maxlag)
            return {"rate": float(f.mean()) if len(f) else float("nan"), "n": int(len(f))}
        W = logp.to_numpy(dtype=float)[rows]
        ss = np.random.SeedSequence([int(seed), int(month.ordinal)])
        f = resampled_null_flags(W, cands["iy"].to_numpy(), cands["ix"].to_numpy(), kind, block,
                                 int(ss.generate_state(1)[0]), reps_per_pair, alpha, maxlag)
        return {"rate": float(f.mean()), "n": int(len(f))}
    return cached(cache_dir, cache_key(len(rows), kind, block, seed, f"{month}_p{len(cands)}_r{reps_per_pair}"), run)


def monthly_excess(logp: pd.DataFrame, cands_by_month: dict[str, pd.DataFrame], kind: str = "shifted",
                   block: float = CC.BOOT_MEAN_BLOCK, seed: int = CC.SEED, alpha: float = CC.ALPHA,
                   formation_months: int = CC.FORMATION_MONTHS, cache_dir=None) -> pd.DataFrame:
    """Per month: observed pass rate (share of candidate pairs with p_eg <= alpha), null pass rate, excess in pp."""
    rows = []
    for key in sorted(cands_by_month):
        c = cands_by_month[key]
        if c.empty:
            continue
        m = pd.Period(key, "M")
        null = month_null_rate(logp, c, m, kind, block, seed, formation_months, alpha, cache_dir=cache_dir)
        obs = float((c["p_eg"] <= alpha).mean())
        rows.append({"month": m, "n_pairs": len(c), "observed": obs, "null": null["rate"],
                     "excess_pp": 100 * (obs - null["rate"])})
    return pd.DataFrame(rows).set_index("month")


def h2_test(excess_pp, lags: int = NW_LAGS, min_excess_pp: float = H2_MIN_EXCESS_PP, alpha: float = CC.ALPHA) -> dict:
    """One-sided Newey-West(6) test that the mean monthly excess pass rate is > 0, plus the >= 1 pp rule."""
    x = np.asarray(pd.Series(excess_pp).dropna(), dtype=float)
    r = ST.newey_west_mean_test(x, lags, "greater")
    r["supported"] = bool(r["p"] < alpha and r["mean"] >= min_excess_pp)
    r["min_excess_pp"] = min_excess_pp
    return r
