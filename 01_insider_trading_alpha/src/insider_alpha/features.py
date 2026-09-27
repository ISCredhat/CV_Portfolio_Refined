"""Point-in-time features for each insider purchase.

Every feature uses only information public before the entry (open of the session
after the filing date): stock data up to the close of the filing date's session
at the latest, and insider track records built only from trades whose outcome
window had fully ended before the filing date.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ price features
def _windows(arr: np.ndarray, end_row: np.ndarray, col: np.ndarray, length: int) -> np.ndarray:
    """(n, length) block of arr[end_row-length+1 .. end_row, col]; NaN before the data start."""
    rows = end_row[:, None] - np.arange(length)[::-1][None, :]
    out = arr[np.clip(rows, 0, None), col[:, None]].astype("float64")
    out[rows < 0] = np.nan
    return out


def attach_stock_features(ev: pd.DataFrame, px: dict[str, pd.DataFrame], chunk: int = 20_000) -> pd.DataFrame:
    """Stock features read at the session BEFORE entry (the filing date's close or earlier).

    Computed point-wise per event (not as full date x ticker matrices) to keep memory low.
    """
    ev = ev.copy()
    adj = px["adj_close"].to_numpy()
    close = px["close"].to_numpy()
    dvol = (px["close"] * px["volume"]).to_numpy()
    r_all = ev["entry_row"].to_numpy() - 1
    c_all = ev["col"].to_numpy()
    feats = {k: np.full(len(ev), np.nan) for k in
             ["mom_1m", "mom_6m_skip1m", "mom_12m", "vol_3m", "dist_52w_high", "dist_52w_low", "log_price", "log_adv"]}
    for s0 in range(0, len(ev), chunk):
        r, c = r_all[s0:s0 + chunk], c_all[s0:s0 + chunk]
        ok = (r >= 0) & (c >= 0)
        r, c = np.where(ok, r, 0), np.where(ok, c, 0)
        w = _windows(adj, r, c, 253)                     # 253 closes -> 252 returns
        last = w[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f = {
                "mom_1m": last / w[:, -22] - 1,
                "mom_6m_skip1m": w[:, -22] / w[:, -127] - 1,
                "mom_12m": last / w[:, 0] - 1,
                "vol_3m": np.nanstd(np.diff(np.log(w[:, -64:]), axis=1), axis=1, ddof=1) * np.sqrt(252),
                "dist_52w_high": last / np.nanmax(w[:, 1:], axis=1) - 1,
                "dist_52w_low": last / np.nanmin(w[:, 1:], axis=1) - 1,
                "log_price": np.log(close[r, c].astype("float64")),
                "log_adv": np.log1p(np.nanmean(_windows(dvol, r, c, 20), axis=1)),
            }
        valid_hist = np.sum(np.isfinite(w[:, -64:]), axis=1) >= 40
        f["vol_3m"] = np.where(valid_hist, f["vol_3m"], np.nan)
        for k, v in f.items():
            feats[k][s0:s0 + chunk] = np.where(ok, v, np.nan)
    for k, v in feats.items():
        ev[k] = v
    ev["adv"] = np.expm1(ev["log_adv"])
    ev["price_pre"] = np.exp(ev["log_price"])
    return ev


def attach_market_features(ev: pd.DataFrame, px: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ev = ev.copy()
    spy = px["adj_close"][C.MARKET].astype("float64")
    mret = spy / spy.shift(21) - 1
    mvol = np.log(spy).diff().rolling(21).std() * np.sqrt(252)
    r = ev["entry_row"].to_numpy() - 1
    ev["mkt_ret_1m"] = mret.to_numpy()[r]
    ev["mkt_vol_1m"] = mvol.to_numpy()[r]
    return ev


# ------------------------------------------------------------------ activity features
def _trailing_count(query: pd.DataFrame, pool: pd.DataFrame, by: str, window_days: int,
                    count_col: str = "", distinct: str | None = None, chunk_keys: int = 400) -> np.ndarray:
    """For each query row, count pool rows with the same `by` key whose filing_date lies in
    [query filing_date - window, query filing_date] (inclusive: same-day filings are public
    before our next-day entry). If `distinct` is set, count distinct values of that column.
    Processed in chunks of keys so memory stays bounded for issuers with thousands of filings."""
    q = query[[by, "filing_date"]].reset_index()
    p = pool[[by, "filing_date"] + ([distinct] if distinct else [])]
    p = p[p[by].isin(set(q[by]))]
    keys = q[by].unique()
    counts = []
    for i in range(0, len(keys), chunk_keys):
        ks = set(keys[i:i + chunk_keys])
        merged = q[q[by].isin(ks)].merge(p[p[by].isin(ks)], on=by, suffixes=("", "_p"))
        lo = merged["filing_date"] - pd.Timedelta(days=window_days)
        merged = merged[(merged["filing_date_p"] >= lo) & (merged["filing_date_p"] <= merged["filing_date"])]
        if distinct:
            counts.append(merged.groupby("index")[distinct].nunique())
        else:
            counts.append(merged.groupby("index").size())
    cnt = pd.concat(counts) if counts else pd.Series(dtype=float)
    return cnt.reindex(query.index, fill_value=0).to_numpy()


def attach_activity_features(ev: pd.DataFrame, buys: pd.DataFrame, sells: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["cluster_insiders_30d"] = _trailing_count(ev, buys, "issuer_cik", 30, "", distinct="owner_cik")
    ev["is_cluster"] = ev["cluster_insiders_30d"] >= 2
    ev["sellers_90d"] = _trailing_count(ev, sells, "issuer_cik", 90, "", distinct="owner_cik")
    ev["prior_buys_issuer_1y"] = _trailing_count(ev, buys, "issuer_cik", 365, "") - 1  # exclude itself
    ev["log_value"] = np.log(ev["value"])
    ev["late_filing"] = ev["filing_lag"] > 4
    return ev


def routine_opportunistic(ev: pd.DataFrame, all_trades: pd.DataFrame) -> pd.Series:
    """Cohen, Malloy & Pomorski (2012) classification using trades BEFORE the event year.

    routine       : insider traded (buy or sell, any issuer) in the same calendar month in
                    each of the three preceding calendar years
    opportunistic : insider traded in each of the three preceding years, but not routinely
    unclassified  : less than three years of trading history
    """
    t = all_trades[["owner_cik", "filing_date"]].copy()
    t["y"] = t["filing_date"].dt.year
    t["m"] = t["filing_date"].dt.month
    ym = set(zip(t["owner_cik"], t["y"], t["m"]))
    yy = set(zip(t["owner_cik"], t["y"]))
    out = []
    for o, d in zip(ev["owner_cik"], ev["filing_date"]):
        y, m = d.year, d.month
        if all((o, y - k) in yy for k in (1, 2, 3)):
            out.append("routine" if all((o, y - k, m) in ym for k in (1, 2, 3)) else "opportunistic")
        else:
            out.append("unclassified")
    return pd.Series(out, index=ev.index)


def attach_track_record(ev: pd.DataFrame, outcome_col: str, realise_col: str = "realise_date",
                        prior_strength: float = 5.0) -> pd.DataFrame:
    """Point-in-time insider track record.

    Uses only the insider's earlier purchases whose outcome window (entry + LABEL_HORIZON
    sessions) had already *ended* strictly before the current filing date. The shrunk
    mean pulls small samples towards zero: sum / (n + prior_strength).
    """
    ev = ev.copy()
    done = ev.loc[ev[outcome_col].notna() & ev[realise_col].notna(),
                  ["owner_cik", realise_col, outcome_col]].copy()
    done = done.sort_values(realise_col)
    done["hit"] = (done[outcome_col] > 0).astype(float)
    g = done.groupby("owner_cik")
    done["cum_n"] = g.cumcount() + 1
    done["cum_sum"] = g[outcome_col].cumsum()
    done["cum_hit"] = g["hit"].cumsum()
    q = ev[["owner_cik", "filing_date"]].reset_index().sort_values("filing_date")
    q["filing_date"] = q["filing_date"].astype("datetime64[ns]")
    done = done.rename(columns={realise_col: "rd"})
    done["rd"] = done["rd"].astype("datetime64[ns]")
    merged = pd.merge_asof(q, done[["owner_cik", "rd", "cum_n", "cum_sum", "cum_hit"]],
                           left_on="filing_date", right_on="rd", by="owner_cik",
                           allow_exact_matches=False, direction="backward")
    merged = merged.set_index("index").reindex(ev.index)
    n = merged["cum_n"].fillna(0)
    ev["tr_n"] = n
    ev["tr_mean"] = np.where(n > 0, merged["cum_sum"] / n.replace(0, np.nan), np.nan)
    ev["tr_hit"] = np.where(n > 0, merged["cum_hit"] / n.replace(0, np.nan), np.nan)
    ev["tr_shrunk"] = merged["cum_sum"].fillna(0) / (n + prior_strength)
    return ev


FEATURES = [
    # trade
    "log_value", "own_chg", "n_lines", "filing_lag", "late_filing", "indirect",
    # who
    "is_ceo", "is_cfo", "is_chair", "is_officer", "is_director", "is_tenpct", "opportunistic", "routine",
    # activity
    "cluster_insiders_30d", "sellers_90d", "prior_buys_issuer_1y",
    # track record (point-in-time)
    "tr_n", "tr_mean", "tr_hit", "tr_shrunk",
    # stock
    "mom_1m", "mom_6m_skip1m", "mom_12m", "vol_3m", "dist_52w_high", "dist_52w_low", "log_price", "log_adv",
    # market
    "mkt_ret_1m", "mkt_vol_1m",
]
