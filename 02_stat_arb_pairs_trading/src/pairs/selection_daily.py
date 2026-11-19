"""Monthly, point-in-time pair selection on daily CRSP bars (Part A of the study).

Same rules as `selection.py` (Engle-Granger in both directions with 2 x min p, Johansen trace with a restricted
constant, Hurst < 0.5, hedge ratio 0.2-5, rank by p, each stock in at most 2 pairs), with the changes the plan
lists in section 5 and the review's fixes:
  * daily bars of the log total-return index, formation = the 6 calendar months before the month;
  * pairs are formed inside an industry label of the month's universe (FF48 by default; "Other" never pairs);
  * a pair needs >= MIN_OBS_SHARE x formation days of joint observations (counted, never imputed);
  * half-lives are in trading days (1-10) and the time stop is 3 x hl_days (review B5);
  * Johansen critical values are simulated at T = 126 / 252 (review m7), not asymptotic;
  * same-PERMCO pairs (two share classes of one firm) are flagged and never eligible (review M4);
  * Engle-Granger runs batched over pairs (`coint.engle_granger_batch`), months run in parallel (review M8).

Point in time: `test_month` only reads rows of `logp` dated before the first day of the month. The universe passed
in must itself have been decided at formation end (`universe.select`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import coint as K
from . import config_crsp as CC

CHUNK = 20000          # pairs per batched Engle-Granger call (bounds memory at ~2 x n x CHUNK floats)


@dataclass(frozen=True)
class SelectionCfg:
    """Selection settings; the defaults are the pre-registered ones from config_crsp."""
    formation_months: int = CC.FORMATION_MONTHS
    min_obs_share: float = CC.MIN_OBS_SHARE
    alpha: float = CC.ALPHA
    fdr: float = CC.FDR
    half_life_days: tuple = CC.HALF_LIFE_DAYS
    max_hurst: float = CC.MAX_HURST
    hedge_ratio_range: tuple = CC.HEDGE_RATIO_RANGE
    eg_maxlag: int = CC.EG_MAXLAG
    exclude_same_permco: bool = CC.EXCLUDE_SAME_PERMCO
    ff48_exclude: tuple = tuple(CC.FF48_EXCLUDE)
    max_pairs_per_stock: int = CC.MAX_PAIRS_PER_STOCK
    n_pairs: int = CC.N_PAIRS


def formation_window(days: pd.DatetimeIndex, month: pd.Period, formation_months: int) -> np.ndarray:
    """Row indices of the trading days of the `formation_months` calendar months before `month`."""
    start = (month - formation_months).start_time
    end = month.start_time
    return np.nonzero((days >= start) & (days < end))[0]


def _groups(universe: pd.DataFrame, group_col, cfg: SelectionCfg) -> pd.Series:
    """Industry label per universe row, NaN where the stock may not form within-industry pairs."""
    if group_col in (None, "none"):
        return pd.Series("all", index=universe.index, dtype=object)
    g = universe[group_col]
    bad = g.isna()
    if group_col == "ff48":
        bad |= g.isin(cfg.ff48_exclude)
    return g.astype(object).where(~bad, np.nan)


def _candidate_index(universe: pd.DataFrame, grp: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All within-group pairs as row positions (a, b) of the universe table plus the group label."""
    aa, bb, gg = [], [], []
    pos = np.arange(len(universe))
    for label, idx in pd.Series(pos, index=grp.index).groupby(grp.to_numpy(), dropna=True):
        p = idx.to_numpy()
        if len(p) < 2:
            continue
        i, j = np.triu_indices(len(p), 1)
        aa.append(p[i]); bb.append(p[j]); gg.append(np.full(len(i), label, dtype=object))
    if not aa:
        e = np.zeros(0, dtype=int)
        return e, e, np.zeros(0, dtype=object)
    return np.concatenate(aa), np.concatenate(bb), np.concatenate(gg)


def _test_chunk(A: np.ndarray, B: np.ndarray, M: np.ndarray, cfg: SelectionCfg, n_form: int) -> dict:
    """Formation statistics for a block of pairs. A, B are (n_days, P) log prices, M the joint-observation mask.
    Pairs are grouped by their number of joint observations so each batch is NaN-free and rectangular."""
    P = A.shape[1]
    nobs = M.sum(0)
    out: dict[str, np.ndarray] = {k: np.full(P, np.nan) for k in
                                  ["p_eg", "eg_stat", "beta", "sigma", "mean_ly", "mean_lx", "sd_lx", "var_beta",
                                   "var_alpha", "hl_days", "hurst", "ssd", "johansen"]}
    out["flip"] = np.zeros(P, bool)
    out["degenerate"] = np.zeros(P, bool)
    out["n_obs"] = nobs
    cv95 = K.johansen_cv(n_form)[1]
    for n in np.unique(nobs):
        idx = np.nonzero(nobs == n)[0]
        n = int(n)
        if n < 8:
            out["degenerate"][idx] = True
            continue
        order = np.argsort(~M[:, idx], axis=0, kind="stable")[:n]       # first n joint rows, in time order
        a = np.take_along_axis(A[:, idx], order, 0)
        b = np.take_along_axis(B[:, idx], order, 0)
        r1 = K.engle_granger_batch(a, b, cfg.eg_maxlag)                 # a on b
        r2 = K.engle_granger_batch(b, a, cfg.eg_maxlag)                 # b on a
        bad = r1["degenerate"] | r2["degenerate"]
        p1, p2 = np.nan_to_num(r1["pvalue"], nan=1.0), np.nan_to_num(r2["pvalue"], nan=1.0)
        flip = p2 < p1                                                  # ties keep a as y, as selection.py
        ly, lx = np.where(flip, b, a), np.where(flip, a, b)
        beta = np.where(flip, r2["beta"], r1["beta"])
        u = np.where(flip, r2["resid"], r1["resid"])
        stat = np.where(flip, r2["stat"], r1["stat"])
        mx, my = lx.mean(0), ly.mean(0)
        v2 = (u * u).sum(0) / (n - 2)                                   # OLS residuals have mean zero
        sxx = ((lx - mx) ** 2).sum(0)
        pe = np.minimum(1.0, 2 * np.minimum(p1, p2))
        ssd = ((np.exp(a - a[0]) - np.exp(b - b[0])) ** 2).sum(0)
        hl = K.half_life_batch(u)
        res = {"p_eg": pe, "eg_stat": stat, "beta": beta, "sigma": np.sqrt(v2), "mean_ly": my, "mean_lx": mx,
               "sd_lx": lx.std(0), "var_beta": v2 / sxx, "var_alpha": v2 / n, "hl_days": hl, "ssd": ssd}
        for k, v in res.items():
            out[k][idx] = v
        out["flip"][idx] = flip
        out["degenerate"][idx] = bad
        for q in np.nonzero((pe <= cfg.alpha) & ~bad)[0]:               # slow tests only where they can matter
            g = idx[q]
            out["hurst"][g] = K.hurst(u[:, q])
            out["johansen"][g] = K.johansen_trace(np.column_stack([ly[:, q], lx[:, q]]))["trace"]
    out["cv95"] = np.full(P, cv95)
    return out


def test_month(logp: pd.DataFrame, universe: pd.DataFrame, month: pd.Period, group_col="ff48",
               cfg: SelectionCfg | None = None) -> pd.DataFrame:
    """Formation statistics for every within-industry candidate pair of `month` (one row per pair).

    logp      DataFrame (trading days x panel columns) of log total-return prices; columns are positional
              (`universe["col"]` indexes them). Only rows dated before the month are read.
    universe  `universe.select` output for this month (contracts.UNIVERSE_COLUMNS).
    Returns contracts.CANDIDATE_COLUMNS. Pairs below the minimum joint observations or numerically degenerate are
    dropped and counted in ``df.attrs`` (n_pairs, n_dropped_min_obs, n_dropped_degenerate); hurst and johansen are
    computed only for pairs with p_eg <= alpha, and are NaN elsewhere."""
    cfg = cfg or SelectionCfg()
    if not isinstance(logp.index, pd.DatetimeIndex):
        raise TypeError("logp must be indexed by trading days")
    rows = formation_window(logp.index, month, cfg.formation_months)
    if len(rows) == 0:
        raise ValueError(f"no formation days before {month}")
    n_form = len(rows)
    min_obs = int(np.ceil(cfg.min_obs_share * n_form))
    universe = universe.reset_index(drop=True)
    grp = _groups(universe, group_col, cfg)
    ia, ib, gl = _candidate_index(universe, grp)
    cols = universe["col"].to_numpy()
    W = logp.to_numpy(dtype=float)[rows]                    # (n_form, n_columns): the only data that is read
    cand_cols = []
    frames = []
    n_total = n_min = n_deg = 0
    for s in range(0, len(ia), CHUNK):
        sl = slice(s, s + CHUNK)
        A, B = W[:, cols[ia[sl]]], W[:, cols[ib[sl]]]
        M = np.isfinite(A) & np.isfinite(B)
        n_total += A.shape[1]
        keep = M.sum(0) >= min_obs
        n_min += int((~keep).sum())
        if not keep.any():
            continue
        r = _test_chunk(A[:, keep], B[:, keep], M[:, keep], cfg, n_form)
        ok = ~r["degenerate"]
        n_deg += int((~ok).sum())
        a_pos, b_pos, g_ = ia[sl][keep][ok], ib[sl][keep][ok], gl[sl][keep][ok]
        flip = r["flip"][ok]
        y_pos, x_pos = np.where(flip, b_pos, a_pos), np.where(flip, a_pos, b_pos)
        df = pd.DataFrame({"y": universe["permno"].to_numpy()[y_pos], "x": universe["permno"].to_numpy()[x_pos],
                           "iy": cols[y_pos], "ix": cols[x_pos], "group": g_,
                           "same_permco": universe["permco"].to_numpy()[y_pos] == universe["permco"].to_numpy()[x_pos]})
        for k in ["p_eg", "eg_stat", "beta", "sigma", "mean_ly", "mean_lx", "sd_lx", "var_beta", "var_alpha",
                  "hl_days", "hurst", "ssd", "johansen"]:
            df[k] = r[k][ok]
        df["n_obs"] = r["n_obs"][ok].astype("int64")
        df["_cv95"] = r["cv95"][ok]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["y", "x", "iy", "ix", "group", "same_permco", "p_eg", "eg_stat", "beta", "sigma", "mean_ly",
                 "mean_lx", "sd_lx", "var_beta", "var_alpha", "hl_days", "hurst", "ssd", "johansen", "n_obs", "_cv95"])
    df.insert(0, "month", str(month))
    if len(df):
        df["bh_pass"] = K.benjamini_hochberg(df["p_eg"].to_numpy(), cfg.fdr)
        df["johansen_pass"] = (df["johansen"] > df["_cv95"]).to_numpy()
        lo, hi = cfg.half_life_days
        blo, bhi = cfg.hedge_ratio_range
        df["plausible"] = df["beta"].between(blo, bhi)
        df["eligible"] = ((df["p_eg"] <= cfg.alpha) & df["johansen_pass"] & df["hl_days"].between(lo, hi)
                          & (df["hurst"] < cfg.max_hurst) & df["plausible"])
        if cfg.exclude_same_permco:
            df["eligible"] &= ~df["same_permco"]
    else:
        for c in ["bh_pass", "johansen_pass", "plausible", "eligible"]:
            df[c] = pd.Series(dtype=bool)
    df = df.drop(columns="_cv95")
    df["n_universe"] = len(universe)
    for c, t in {"y": "int64", "x": "int64", "iy": "int64", "ix": "int64", "n_obs": "int64"}.items():
        df[c] = df[c].astype(t)
    df.attrs.update({"n_pairs": n_total, "n_dropped_min_obs": n_min, "n_dropped_degenerate": n_deg,
                     "n_formation_days": n_form, "min_obs": min_obs})
    return df


def choose(cands: pd.DataFrame, method: str = "coint", n: int | None = None, seed: int = 0,
           cfg: SelectionCfg | None = None) -> pd.DataFrame:
    """Pick up to `n` pairs for one month from its candidate table (same rules as `selection.choose`).

    coint    : eligible pairs ranked by Engle-Granger p-value, ties by half-life (the strategy)
    distance : lowest sum of squared differences of normalised prices (Gatev et al. 2006)
    random   : random pairs from the plausible set (placebo)
    Each stock appears in at most `max_pairs_per_stock` pairs; same-PERMCO pairs are skipped when configured."""
    cfg = cfg or SelectionCfg()
    n = cfg.n_pairs if n is None else n
    if cands.empty:
        return cands
    ok = ~cands["same_permco"] if cfg.exclude_same_permco else pd.Series(True, index=cands.index)
    if method == "coint":
        pool = cands[cands["eligible"]].sort_values(["p_eg", "hl_days", "y", "x"], kind="mergesort")
    elif method == "distance":
        pool = cands[cands["plausible"] & ok].sort_values(["ssd", "y", "x"], kind="mergesort")
    elif method == "random":
        pool = cands[cands["plausible"] & ok].sample(frac=1.0, random_state=seed)
    else:
        raise ValueError(method)
    used: dict[int, int] = {}
    keep = []
    for k in pool.index:
        y, x = cands.at[k, "y"], cands.at[k, "x"]
        if used.get(y, 0) >= cfg.max_pairs_per_stock or used.get(x, 0) >= cfg.max_pairs_per_stock:
            continue
        keep.append(k)
        used[y] = used.get(y, 0) + 1
        used[x] = used.get(x, 0) + 1
        if len(keep) >= n:
            break
    return cands.loc[keep]


def _one_month(sub, universe, m, group_col, cfg, out_dir):
    df = test_month(sub, universe, m, group_col, cfg)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(Path(out_dir) / f"cands_{m}.parquet")
    return str(m), df


def run_months(logp: pd.DataFrame, universes: dict, months, group_col="ff48", cfg: SelectionCfg | None = None,
               n_jobs: int = 1, out_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """`test_month` for every month, in parallel over months (joblib). `universes` maps str(month) -> universe table.
    Each worker gets only the formation-window rows, so nothing dated on or after the month can reach it.
    With `out_dir`, each month's candidates are also written as parquet (one file per month)."""
    cfg = cfg or SelectionCfg()
    jobs = []
    for m in (pd.Period(x, freq="M") for x in months):
        rows = formation_window(logp.index, m, cfg.formation_months)
        jobs.append((logp.iloc[rows], universes[str(m)], m, group_col, cfg, out_dir))
    if n_jobs == 1:
        res = [_one_month(*j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(_one_month)(*j) for j in jobs)
    return dict(res)
