"""Gatev, Goetzmann & Rouwenhorst (2006, "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", RFS)
as its own engine, deliberately separate from `backtest.run_month`: the GGR rules differ from the frozen
configuration in ways that matter (normalised-price distance, no hedge ratio, several round trips, no month-end
exit, overlapping portfolios), so reusing the main backtester would not be a replication (review M7).

Rules implemented (all gross of costs, as in the paper's Table 1):
  * Formation: `formation_months` (12) of daily prices on the total-return scale, normalised to 1 on the first
    formation day. Only stocks with a valid price on every formation day, traded on every formation day when a
    `traded` mask is given, and inside the optional universe (share codes 10/11 are the caller's filter). Pairs of
    two share classes of one company (same `permco`) are dropped (Do & Faff 2010).
  * Pairs: ranked by the sum of squared differences (SSD) of the two normalised price series. `selection=(a, b)`
    keeps ranks a..b (1-based): (1, 5) top 5, (1, 20) top 20, (101, 120) pairs 101-120, (1, None) all pairs.
  * Trading period: `trading_months` (6) after formation, normalised with the formation base. A pair opens when the
    normalised-price spread exceeds `open_sd` formation standard deviations (ddof = 1): $1 short the high-priced
    stock, $1 long the low-priced one, shares fixed at opening. It closes at the first crossing (spread reaches 0
    or changes sign). Round trips repeat; open positions close on the last trading day, or on a leg's last CRSP
    row (delisting, at its TR close, which includes the delisting return). No new position opens on the last day.
  * Prices are re-scaled to $1 at the start of the trading period (Do & Faff 2010); the trigger is `open_sd` SDs of
    the formation-period spread of prices normalised at the start of formation.
  * `wait=1` (one-day wait, which removes bid-ask bounce): open at the close of the day after the divergence and
    close at the close of the day after the convergence (Do & Faff 2010). `wait=0`: both on the trigger day's
    close. `wait_close` overrides the closing half.
  * A new portfolio starts every month; each trades for 6 months, so six overlap. The monthly return is the equal
    weight average of the portfolios active that month, on committed capital (the pairs selected: payoff / N) and
    on employed capital (payoff / pairs that actually opened).
Point in time: the formation window ends before the first trading day, and pair selection reads nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GGRConfig:
    formation_months: int = 12
    trading_months: int = 6
    open_sd: float = 2.0
    wait: int = 1                      # days between the divergence and opening: 0 or 1
    wait_close: bool | None = None     # None: as `wait` (Do & Faff: "wait one day" delays the close too)
    selection: tuple = (1, 20)         # inclusive 1-based SSD ranks; None as the upper end = all pairs


@dataclass
class GGRResult:
    monthly: pd.DataFrame              # index PeriodIndex: committed, employed, n_portfolios
    trades: pd.DataFrame               # one row per round trip
    portfolios: pd.DataFrame           # one row per portfolio: pairs selected / traded / round trips / days open
    pairs: pd.DataFrame = None         # one row per selected pair: opened, crossing round trips, payoff
    stats: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------ formation
def ssd_matrix(P: np.ndarray) -> np.ndarray:
    """Sum of squared differences between all columns of P (n_form, K): one matrix product."""
    sq = (P * P).sum(0)
    G = P.T @ P
    return np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)


def select_pairs(Pn: np.ndarray, selection: tuple = (1, 20), group: np.ndarray | None = None,
                 permco: np.ndarray | None = None) -> tuple:
    """Rank all pairs of the columns of the normalised formation prices Pn by SSD; return (i, j, ssd) for the
    requested rank slice (ties broken by column order). `group` (K,) restricts pairs to equal labels; pairs with
    equal `permco` (share classes of one firm) are dropped before ranking."""
    K = Pn.shape[1]
    S = ssd_matrix(Pn)
    ii, jj = np.triu_indices(K, 1)
    if group is not None:
        keep = group[ii] == group[jj]
        ii, jj = ii[keep], jj[keep]
    if permco is not None:
        keep = permco[ii] != permco[jj]
        ii, jj = ii[keep], jj[keep]
    d = S[ii, jj]
    order = np.lexsort((jj, ii, d))
    lo, hi = selection
    sl = order[lo - 1: (None if hi is None else hi)]
    return ii[sl], jj[sl], d[sl]


# ------------------------------------------------------------------------------------------ trading
def simulate_pairs(Pi: np.ndarray, Pj: np.ndarray, sd: np.ndarray, open_sd: float = 2.0, wait: int = 1,
                   wait_close: bool = False, dead_i: np.ndarray | None = None, dead_j: np.ndarray | None = None):
    """Trade a set of pairs over one trading period.

    Pi, Pj  (T, P) normalised TR prices of the two legs (NaN after a delisting); sd (P,) formation SD of the spread
            Pi - Pj. Returns (pnl (T, P) daily $ payoff per $1 per leg, trades list of dicts)."""
    T, P = Pi.shape
    s = (Pi - Pj) / sd
    dead = np.zeros((T, P), bool)
    for d in (dead_i, dead_j):
        if d is not None:
            dead |= d
    pos = np.zeros(P, dtype=int)                 # +1: spread > 0 so short i / long j
    hi_ = np.zeros(P)
    hj_ = np.zeros(P)
    t_open = np.full(P, -1)
    p_open = np.zeros((P, 2))
    pend_open = np.zeros(P, dtype=int)           # side waiting to open at the next close
    pend_close = np.zeros(P, dtype=bool)
    pnl = np.zeros((T, P))
    trades = []

    def close(mask, t, reason):
        for p in np.nonzero(mask)[0]:
            sgn = pos[p]
            ret_i = Pi[t, p] / p_open[p, 0] - 1
            ret_j = Pj[t, p] / p_open[p, 1] - 1
            ret = -sgn * ret_i + sgn * ret_j     # short the high-priced, long the low-priced
            trades.append({"pair": p, "open": int(t_open[p]), "close": t, "side": int(sgn), "ret": ret,
                           "days": t - int(t_open[p]), "reason": reason})
        pos[mask] = 0
        hi_[mask] = 0.0
        hj_[mask] = 0.0

    for t in range(T):
        if t > 0:
            held = pos != 0
            if held.any():
                dPi, dPj = Pi[t] - Pi[t - 1], Pj[t] - Pj[t - 1]
                bad = held & ~(np.isfinite(dPi) & np.isfinite(dPj))
                if bad.any():
                    raise AssertionError(f"NaN price while a position is open (pair {int(np.nonzero(bad)[0][0])}, row {t})")
                pnl[t] = np.where(held, hi_ * np.where(held, dPi, 0.0) + hj_ * np.where(held, dPj, 0.0), 0.0)
        held = pos != 0
        last = t == T - 1
        fin = np.isfinite(s[t])
        # ---- closes: delisting and end of period first, then the crossing
        forced = held & (dead[t] if dead is not None else False)
        if forced.any():
            close(forced, t, "delisting")
            pend_close[forced] = False
        held = pos != 0
        cross = held & fin & ((pos * s[t]) <= 0)
        if wait_close:
            do_close = held & pend_close
            pend_close = cross & ~do_close
            cross = do_close
        if cross.any():
            close(cross, t, "crossing")
        held = pos != 0
        if last:
            if held.any():
                close(held, t, "end")
            break
        # ---- opens
        if wait:
            opening = (pend_open != 0) & (pos == 0) & fin & ~dead[t]
            for side_, m in ((1, opening & (pend_open == 1)), (-1, opening & (pend_open == -1))):
                if m.any():
                    pos[m] = side_
                    t_open[m] = t
                    p_open[m, 0], p_open[m, 1] = Pi[t, m], Pj[t, m]
                    hi_[m] = -side_ / Pi[t, m]
                    hj_[m] = side_ / Pj[t, m]
            pend_open[:] = 0
        sig = (pos == 0) & fin & ~dead[t] & (np.abs(s[t]) > open_sd)
        if wait:
            pend_open = np.where(sig & ~(pos != 0), np.sign(s[t]).astype(int), 0)
        else:
            for side_ in (1, -1):
                m = sig & (np.sign(s[t]) == side_)
                if m.any():
                    pos[m] = side_
                    t_open[m] = t
                    p_open[m, 0], p_open[m, 1] = Pi[t, m], Pj[t, m]
                    hi_[m] = -side_ / Pi[t, m]
                    hj_[m] = side_ / Pj[t, m]
    return pnl, trades


# ------------------------------------------------------------------------------------------ portfolios
def _window(days: pd.DatetimeIndex, start: pd.Period, months: int) -> np.ndarray:
    a, b = start.start_time, (start + months).start_time
    return np.nonzero((days >= a) & (days < b))[0]


def run_portfolio(tr: np.ndarray, days: pd.DatetimeIndex, start: pd.Period, cfg: GGRConfig,
                  dead: np.ndarray | None = None, universe: np.ndarray | None = None,
                  group: np.ndarray | None = None, traded: np.ndarray | None = None,
                  permco: np.ndarray | None = None) -> dict | None:
    """One GGR portfolio formed from the `formation_months` before `start` and traded for `trading_months`."""
    f_rows = _window(days, start - cfg.formation_months, cfg.formation_months)
    t_rows = _window(days, start, cfg.trading_months)
    if len(f_rows) < 20 or len(t_rows) < 2:
        return None
    F = tr[f_rows]
    ok = np.isfinite(F).all(0) & (F > 0).all(0)
    first = tr[t_rows[0]]
    ok &= np.isfinite(first) & (first > 0)               # must have a price on the day trading starts
    if traded is not None:
        ok &= traded[f_rows].all(0)
    if universe is not None:
        ok &= universe
    cols = np.nonzero(ok)[0]
    if len(cols) < 2:
        return None
    Pn = F[:, cols] / F[0, cols]
    ia, ja, ssd = select_pairs(Pn, cfg.selection, None if group is None else group[cols],
                              None if permco is None else permco[cols])
    if len(ia) == 0:
        return None
    sd = (Pn[:, ia] - Pn[:, ja]).std(axis=0, ddof=1)
    live = sd > 0
    ia, ja, ssd, sd = ia[live], ja[live], ssd[live], sd[live]
    base = first[cols]                                   # trading-period prices are re-scaled to $1 on day one
    Ti = tr[t_rows][:, cols[ia]] / base[ia]
    Tj = tr[t_rows][:, cols[ja]] / base[ja]
    di = None if dead is None else dead[t_rows][:, cols[ia]]
    dj = None if dead is None else dead[t_rows][:, cols[ja]]
    wc = bool(cfg.wait) if cfg.wait_close is None else cfg.wait_close
    pnl, trades = simulate_pairs(Ti, Tj, sd, cfg.open_sd, cfg.wait, wc, di, dj)
    return {"pnl": pnl, "trades": trades, "rows": t_rows, "n_selected": len(ia), "ia": cols[ia], "ja": cols[ja],
            "ssd": ssd, "n_universe": len(cols)}


def aggregate(monthly_by_portfolio: dict[pd.Period, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight average over the portfolios active in each calendar month (the paper's six overlapping
    portfolios). Input: start month -> DataFrame(index months, columns committed/employed)."""
    stack = pd.concat(monthly_by_portfolio.values(), keys=monthly_by_portfolio.keys(), names=["start", "month"])
    g = stack.groupby(level="month")
    out = g.mean()
    out["n_portfolios"] = g.size()
    return out.sort_index()


def run(tr: pd.DataFrame, first_start: str, last_start: str, cfg: GGRConfig | None = None,
        dead: pd.DataFrame | np.ndarray | None = None, universe=None, group: np.ndarray | None = None,
        traded: pd.DataFrame | np.ndarray | None = None, permco: np.ndarray | None = None) -> GGRResult:
    """GGR over portfolios starting in each month from `first_start` to `last_start`.

    tr        DataFrame (trading days x stocks) of total-return prices (contracts convention, NaN = no price).
    dead      optional bool frame/array, True on a stock's last row (forces the delisting close).
    universe  optional callable(start_period) -> bool array over columns (e.g. share codes 10/11, or the top-500
              for the bridge run); traded  optional bool frame, False on days without a trade or a valid return;
    permco    optional (n_stocks,) firm id: pairs of share classes of one firm are excluded."""
    return assemble(run_parts(tr, first_start, last_start, cfg, dead, universe, group, traded, permco))


def run_parts(tr: pd.DataFrame, first_start: str, last_start: str, cfg: GGRConfig | None = None,
              dead: pd.DataFrame | np.ndarray | None = None, universe=None, group: np.ndarray | None = None,
              traded: pd.DataFrame | np.ndarray | None = None, permco: np.ndarray | None = None) -> dict:
    """The per-portfolio pieces of `run` (arguments as there): each portfolio's monthly returns and its trade,
    portfolio and pair rows. Runs over separate blocks of start months join with `assemble` (memory: the
    replication then never needs the whole 1962-2009 panel at once)."""
    cfg = cfg or GGRConfig()
    days = tr.index
    X = tr.to_numpy(dtype=float)
    D = None if dead is None else np.asarray(dead, dtype=bool)
    TD = None if traded is None else np.asarray(traded, dtype=bool)
    pc = None if permco is None else np.asarray(permco)
    by_port, trades, ports, pairs = {}, [], [], []
    for start in pd.period_range(first_start, last_start, freq="M"):
        u = None if universe is None else universe(start)
        r = run_portfolio(X, days, start, cfg, D, u, group, TD, pc)
        if r is None:
            continue
        pnl, rows = r["pnl"], r["rows"]
        months = pd.PeriodIndex(days[rows], freq="M")
        n_sel = r["n_selected"]
        opened = {t["pair"] for t in r["trades"]}
        n_open = len(opened)
        daily = pnl.sum(1)
        m = pd.Series(daily, index=months).groupby(level=0).sum()
        by_port[start] = pd.DataFrame({"committed": m / n_sel, "employed": m / n_open if n_open else 0.0})
        per_pair = pnl.sum(0)
        for p_ in range(n_sel):
            tp = [t for t in r["trades"] if t["pair"] == p_]
            cr = [t for t in tp if t["reason"] == "crossing"]
            pairs.append({"start": start, "y": int(tr.columns[r["ia"][p_]]), "x": int(tr.columns[r["ja"][p_]]),
                          "opened": len(tp) > 0, "round_trips": len(cr), "payoff": float(per_pair[p_]),
                          "rt_days": float(np.mean([t["days"] for t in cr])) if cr else np.nan})
        for t in r["trades"]:
            trades.append({"start": str(start), "y": int(tr.columns[r["ia"][t["pair"]]]),
                           "x": int(tr.columns[r["ja"][t["pair"]]]), "open": days[rows[t["open"]]],
                           "close": days[rows[t["close"]]], "side": t["side"], "ret": t["ret"],
                           "days": t["days"], "reason": t["reason"]})
        ports.append({"start": str(start), "n_selected": n_sel, "n_traded": n_open, "round_trips": len(r["trades"]),
                      "days_open": sum(t["days"] for t in r["trades"]), "n_universe": r["n_universe"]})
    return {"by_port": by_port, "trades": trades, "ports": ports, "pairs": pairs}


def assemble(parts: dict | list[dict]) -> GGRResult:
    """GGRResult from one `run_parts` output or a list of them in start-month order (blocks must not overlap)."""
    by_port, trades, ports, pairs = {}, [], [], []
    for p in [parts] if isinstance(parts, dict) else parts:
        twice = set(by_port) & set(p["by_port"])
        if twice:
            raise ValueError(f"portfolio start(s) in two blocks: {sorted(map(str, twice))[:3]}")
        by_port.update(p["by_port"])
        trades += p["trades"]
        ports += p["ports"]
        pairs += p["pairs"]
    if not by_port:
        raise ValueError("no GGR portfolio could be formed")
    monthly = aggregate(by_port)
    tdf, pdf, prdf = pd.DataFrame(trades), pd.DataFrame(ports), pd.DataFrame(pairs)
    st = {"portfolios": len(pdf), "pairs_selected": int(pdf["n_selected"].sum()), "pairs_traded": int(pdf["n_traded"].sum()),
          "round_trips": int(pdf["round_trips"].sum()),
          "round_trips_per_traded_pair": float(pdf["round_trips"].sum() / max(pdf["n_traded"].sum(), 1)),
          "avg_days_open": float(tdf["days"].mean()) if len(tdf) else float("nan"),
          "share_selected_traded": float(pdf["n_traded"].sum() / pdf["n_selected"].sum()),
          "delisting_exits": int((tdf["reason"] == "delisting").sum()) if len(tdf) else 0}
    return GGRResult(monthly=monthly, trades=tdf, portfolios=pdf, pairs=prdf, stats=st)


# ------------------------------------------------------------------------------------------ Do & Faff tables
def dofaff_table1(monthly: pd.DataFrame, periods: dict, lags: int = 6, periods_per_year: int = 12) -> pd.DataFrame:
    """Do & Faff (2010) Table 1 statistics on the monthly GGR series, for each labelled (first, last) month range:
    mean and median of employed-capital returns, Newey-West(`lags`) t of the mean, Sharpe ratio (employed), share
    of months < 0, and the committed-capital mean. Returns one row per period.

    D&F's Sharpe ratio is monthly, not annualised: their wait-one-day full-sample mean 0.0062 and Sharpe 0.59
    imply a monthly SD of 1.05%, in line with GGR (1999) Table III (top 20, annualised SD 3.8% = 1.1% a month);
    an annualised 0.59 would need a 3.6% monthly SD. Compare `sharpe_monthly` with their table."""
    from . import stats as ST
    rows = {}
    for label, (a, b) in periods.items():
        m = monthly.loc[pd.Period(a, "M"):pd.Period(b, "M")]
        e = m["employed"].to_numpy(float)
        if len(e) < 3:
            raise ValueError(f"period {label}: only {len(e)} months")
        nw = ST.newey_west_mean_test(e, lags, "two")
        rows[label] = {"months": len(e), "mean_employed": nw["mean"], "t_nw": nw["t"], "se_nw": nw["se"],
                       "median_employed": float(np.median(e)),
                       "sharpe_monthly": float(e.mean() / e.std(ddof=1)),
                       "sharpe_annual": float(e.mean() / e.std(ddof=1) * np.sqrt(periods_per_year)),
                       "share_neg": float((e < 0).mean()), "mean_committed": float(m["committed"].mean())}
    return pd.DataFrame(rows).T


def dofaff_table2(pairs: pd.DataFrame, periods: dict, trading_months: int = 6) -> pd.DataFrame:
    """Do & Faff (2010) Table 2: share of selected pairs by group, per sub-period of portfolio start months.
    1 never traded, 2 opened but never converged (no crossing close), 3 one round trip, 4 several. Also the
    average monthly payoff per pair of each group and the average round-trip length (days) of groups 3 and 4."""
    out = {}
    for label, (a, b) in periods.items():
        p = pairs[(pairs["start"] >= pd.Period(a, "M")) & (pairs["start"] <= pd.Period(b, "M"))]
        if p.empty:
            raise ValueError(f"period {label}: no pairs")
        g = np.where(~p["opened"], 1, np.where(p["round_trips"] == 0, 2, np.where(p["round_trips"] == 1, 3, 4)))
        for k in (1, 2, 3, 4):
            sel = p[g == k]
            out[(label, k)] = {"share": float((g == k).mean()), "n": len(sel),
                               "avg_monthly_payoff": float(sel["payoff"].mean() / trading_months) if len(sel) else np.nan,
                               "avg_rt_days": float(sel["rt_days"].mean()) if k >= 3 and len(sel) else np.nan}
    return pd.DataFrame(out).T.rename_axis(["period", "group"])
