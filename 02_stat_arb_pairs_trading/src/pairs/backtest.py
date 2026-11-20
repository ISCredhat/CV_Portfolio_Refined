"""Event-driven pairs backtest, vectorised across pairs and parameter settings ("lanes").

Timing
  * The signal for bar t uses prices up to the close of bar t.
  * Orders fill at the close of the 1-minute bar *after* the signal bar closes (1-minute latency),
    at that minute's last trade price. Month-end exits use the closing price (a market-on-close order).
Positions
  * A trade buys/sells fixed share quantities at entry: gross exposure = LEVERAGE x slot capital,
    split $1 : $beta between the legs (beta = hedge ratio at entry). No rebalancing while open.
  * Exit when the z-score reverts through +/- exit_z, on a stop-loss (|z| >= entry_z + 2), after
    3 formation half-lives, or at month end. After a stop or time exit the pair cannot re-enter until
    |z| has come back inside the entry band.
Costs
  * Every fill pays the leg's one-way cost (half-spread floored at half a tick, commission, impact).
  * The short leg pays a borrow fee on its market value while the trade is open.
P&L is booked on committed capital (1.0 = the whole portfolio), so returns are additive across pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import costs as TC
from . import data as D
from . import hedge as HG
from .selection import trade_months

REASONS = {1: "reversion", 2: "stop-loss", 3: "time stop", 4: "month end", 5: "delisting"}


# --------------------------------------------------------------------------------- bar data
class BarData:
    """All bar-level arrays for one bar size (and latency) over the whole sample."""

    def __init__(self, panel: D.MinutePanel, bar: str, latency_min: int = C.LATENCY_MIN,
                 latency_bars: int = 0):
        b = panel.bars(bar, latency_min=latency_min)
        self.bar = bar
        self.ends = b["ends"]
        self.ts = b["ts"]
        self.day = b["day"]
        self.close = b["close"].astype("float64")
        self.exec = b["exec"].astype("float64")
        n_min = panel.close.shape[0]
        # trading day on which each bar's orders are filled (the next morning for a day's last bar)
        self.fill_day = panel.cal["day_of_min"][np.minimum(self.ends + latency_min, n_min - 1)]
        if latency_bars:                      # fill at the close of a later bar instead
            idx = np.minimum(np.arange(len(self.ends)) + latency_bars, len(self.ends) - 1)
            self.exec = self.close[idx]
            self.fill_day = self.day[idx]
        cum = np.cumsum(~np.isnan(panel.close), axis=0, dtype="int32")
        prev = np.concatenate([[-1], self.ends[:-1]])
        start = np.where((prev >= 0)[:, None], cum[np.maximum(prev, 0)], 0)
        self.fresh = (cum[self.ends] - start) > 0          # the ticker traded inside the bar
        self.factor = panel.split_factor_day[self.day]      # raw price = adjusted * factor
        self.month = pd.PeriodIndex(self.ts, freq="M")
        self.dt_days = 1.0 if bar in ("1D", "1d", "D") else pd.Timedelta(bar).total_seconds() / 60 / 390
        self.bars_per_hour = 1 / 6.5 if bar in ("1D", "1d", "D") else 60 / (pd.Timedelta(bar).total_seconds() / 60)
        self.bars_per_day = D.bars_per_day(bar)
        self.days = panel.cal["days"]


# --------------------------------------------------------------------------------- simulation
@dataclass
class MonthResult:
    daily: pd.DataFrame            # (days x lanes) net P&L components are in separate frames
    gross: pd.DataFrame = field(default=None)
    cost: pd.DataFrame = field(default=None)
    borrow: pd.DataFrame = field(default=None)
    trades: pd.DataFrame = field(default=None)
    signals: dict = field(default=None)


def simulate(z, beta, Cy, Cx, Ey, Ex, fresh, entry, exit_, stop, tstop, w_gross,
             dead=None, short_ok=None, no_entry=None):
    """Run the entry/exit state machine. All inputs are (n_bars, lanes) or (lanes,).
    Returns share holdings of each leg after each bar's orders, and the trade list.

    CRSP extensions (all optional, so the minute-bar path is unchanged):
      dead      (n, lanes) bool: a leg of the pair has its last row here. An open pair is closed on this row
                (reason 5, "delisting"); the caller prices both legs at this row's close (review B1).
      short_ok  (ok_y, ok_x), each (n, lanes) bool: False forbids opening a position that is short that leg
                (2008 short-sale ban proxy, review M13). Positions already open are not touched.
      no_entry  (n, lanes) bool: no new position may be opened on this row (dead leg at the fill, missing fill price)."""
    n, L = z.shape
    pos = np.zeros(L, dtype=np.int8)
    age = np.zeros(L)
    blocked = np.zeros(L, dtype=bool)
    t0 = np.full(L, -1)
    cur_y = np.zeros(L)
    cur_x = np.zeros(L)
    hy = np.zeros((n, L))
    hx = np.zeros((n, L))
    lo, hi = C.HEDGE_RATIO_RANGE
    trades = []
    for t in range(n):
        zt = z[t]
        fin = np.isfinite(zt)
        az = np.abs(np.where(fin, zt, 0.0))
        open_ = pos != 0
        if open_.any():
            rev = fin & (((pos == 1) & (zt >= -exit_)) | ((pos == -1) & (zt <= exit_)))
            stp = fin & (az >= stop)
            tim = age >= tstop
            end = np.full(L, t == n - 1)
            dl = dead[t] if dead is not None else np.zeros(L, dtype=bool)
            ex = open_ & (rev | stp | tim | end | dl)
            if ex.any():
                reason = np.where(dl, 5, np.where(stp, 2, np.where(tim, 3, np.where(rev, 1, 4))))
                for lane in np.nonzero(ex)[0]:
                    trades.append((lane, t0[lane], t, reason[lane], pos[lane]))
                blocked |= ex & (stp | tim) & ~dl
                pos[ex] = 0
                cur_y[ex] = 0.0
                cur_x[ex] = 0.0
        else:
            ex = np.zeros(L, dtype=bool)
        blocked &= ~(fin & (az < entry))
        if t < n - 1:
            bt = beta[t]
            can = (pos == 0) & ~ex & ~blocked & fin & fresh[t] & (bt >= lo) & (bt <= hi)
            if no_entry is not None:
                can &= ~no_entry[t]
            sh = can & (zt > entry) & (zt < stop)             # short y, long x
            lg = can & (zt < -entry) & (zt > -stop)           # long y, short x
            if short_ok is not None:
                sh &= short_ok[0][t]
                lg &= short_ok[1][t]
            new = sh | lg
            if new.any():
                p = np.where(lg, 1, -1)
                wy = w_gross / (1 + bt)
                wx = w_gross * bt / (1 + bt)
                cur_y = np.where(new, p * wy / Ey[t], cur_y)
                cur_x = np.where(new, -p * wx / Ex[t], cur_x)
                pos = np.where(new, p, pos).astype(np.int8)
                age = np.where(new, 0, age)
                t0 = np.where(new, t, t0)
        age = age + (pos != 0)
        hy[t] = cur_y
        hx[t] = cur_x
    return hy, hx, trades


def _open_nan_check(a: np.ndarray, held: np.ndarray, what: str, lane_pair, pairs, rows, bd) -> None:
    """Raise if `a` is not finite where a position is involved (no silent NaN handling, review B1)."""
    bad = held & ~np.isfinite(a)
    if bad.any():
        r, l = map(int, np.argwhere(bad)[0])
        p = pairs.iloc[lane_pair[l]]
        raise AssertionError(f"NaN {what} while a position is open: pair ({p['y']}, {p['x']}) "
                             f"on {bd.days[rows[r]].date()} ({int(bad.sum())} cells)")


def run_month(bd, month: pd.Period, pairs: pd.DataFrame, methods, entries, exits,
              half_spread, cost_mult: float = 1.0, return_signals: bool = False,
              cost_fn=None, n_pairs: int | None = None) -> MonthResult | None:
    """Backtest the selected `pairs` of one month for every (method, entry, exit) combination.

    `bd` is a BarData (minute path, unchanged) or a contracts.DailyBars (``bars_per_hour is None``). Extras:
      cost_fn   contracts.CostFn replacing the default `costs.leg_cost_bps` path;
      n_pairs   slot count that sets the leg weight LEVERAGE / n_pairs (default C.N_PAIRS; 20 for the
                pre-registered secondary series);
    and for daily bars: the time stop is 3 x `pairs["hl_days"]` trading days, `bd.dead` force-closes a pair
    on a leg's last row (reason "delisting", both legs at that row's close), `bd.short_ok` blocks new short
    legs, `bd.borrow_fee` sets per-stock borrow, and any NaN price while a position is open raises.
    `half_spread` is positional over the panel columns (bps, previous-month estimate)."""
    daily = bd.bars_per_hour is None
    rows = np.nonzero(bd.month == month)[0]
    if len(rows) < 2 or pairs.empty:
        return None
    iy, ix = pairs["iy"].to_numpy(), pairs["ix"].to_numpy()
    P = len(pairs)
    f = {k: pairs[k].to_numpy(dtype=float) for k in ["beta", "sigma", "sd_lx", "var_beta", "var_alpha"]}
    my, mx = pairs["mean_ly"].to_numpy(float), pairs["mean_lx"].to_numpy(float)
    win = int(max(20, round(C.ROLLING_WINDOW_DAYS * bd.bars_per_day)))
    r0 = max(rows[0] - win, 0)
    ext = np.arange(r0, rows[-1] + 1)
    ly = np.log(bd.close[ext][:, iy]) - my
    lx = np.log(bd.close[ext][:, ix]) - mx
    k0 = rows[0] - r0                                   # first bar of the month inside `ext`
    zs, bs = {}, {}
    for m in methods:
        if m == "static":
            z, b = HG.static(ly[k0:], lx[k0:], f)
        elif m == "rolling":
            z, b = HG.rolling(ly, lx, win)
            z, b = z[k0:], b[k0:]
        elif m == "kalman":
            z, b = HG.kalman(ly, lx, f, bd.dt_days, win)
            z, b = z[k0:], b[k0:]
        else:
            raise ValueError(m)
        zs[m], bs[m] = z, b
    combos = [(m, e, x) for m in methods for e in entries for x in exits]
    Q = len(combos)
    L = P * Q
    lane_pair = np.tile(np.arange(P), Q)
    lane_combo = np.repeat(np.arange(Q), P)
    z = np.column_stack([zs[combos[q][0]][:, p] for q, p in zip(lane_combo, lane_pair)])
    beta = np.column_stack([bs[combos[q][0]][:, p] for q, p in zip(lane_combo, lane_pair)])
    entry = np.array([combos[q][1] for q in lane_combo])
    exit_ = np.array([combos[q][2] for q in lane_combo])
    stop = entry + C.STOP_Z_ADD
    if daily:       # half-lives in trading days, one bar per day: time stop = 3 x hl_days bars
        hl = pairs["hl_days"].to_numpy(float)[lane_pair]
        tstop = np.maximum(1, hl * C.TIME_STOP_HALF_LIVES)
    else:
        hl = pairs["hl_hours"].to_numpy(float)[lane_pair]
        tstop = np.maximum(1, hl * C.TIME_STOP_HALF_LIVES * bd.bars_per_hour)

    Cm, Em = bd.close[rows], bd.exec[rows].copy()
    Em[-1] = Cm[-1]                                      # month-end exit on the close
    Cy, Cx = Cm[:, iy][:, lane_pair], Cm[:, ix][:, lane_pair]
    Ey, Ex = Em[:, iy][:, lane_pair], Em[:, ix][:, lane_pair]
    fresh = (bd.fresh[rows][:, iy] & bd.fresh[rows][:, ix])[:, lane_pair]
    w_gross = C.LEVERAGE / (C.N_PAIRS if n_pairs is None else n_pairs)
    dead_l = ok_y = ok_x = no_entry = None
    if daily:
        fd = bd.fill_day[rows]                                      # row each order is filled on
        dead_all = bd.dead[rows]
        dead_l = (dead_all[:, iy] | dead_all[:, ix])[:, lane_pair]
        dead_fill = (bd.dead[fd][:, iy] | bd.dead[fd][:, ix])[:, lane_pair]
        Ey, Ex = np.where(dead_l, Cy, Ey), np.where(dead_l, Cx, Ex)  # delisting exit: both legs at this close
        ok_y = bd.short_ok[fd][:, iy][:, lane_pair]
        ok_x = bd.short_ok[fd][:, ix][:, lane_pair]
        no_entry = dead_l | dead_fill | ~np.isfinite(Ey) | ~np.isfinite(Ex)
        no_entry[-1] = True
    hy, hx, trades = simulate(z, beta, Cy, Cx, Ey, Ex, fresh, entry, exit_, stop, tstop, w_gross,
                              dead=dead_l, short_ok=None if not daily else (ok_y, ok_x), no_entry=no_entry)

    # ---- P&L ----------------------------------------------------------------------------
    n = len(rows)
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    hy0 = np.vstack([np.zeros((1, L)), hy[:-1]])
    hx0 = np.vstack([np.zeros((1, L)), hx[:-1]])
    fac = bd.factor[rows]
    if daily:
        # NaN prices are only allowed where no position is involved: 0 x NaN is not a P&L
        for h, c, e, nm in [(hy0, Cy, Ey, "y"), (hx0, Cx, Ex, "x")]:
            _open_nan_check(c, h != 0, f"close of {nm}", lane_pair, pairs, rows, bd)
            _open_nan_check(e, (h != 0) | (np.vstack([h[1:], np.zeros((1, L))]) != 0), f"fill of {nm}",
                            lane_pair, pairs, rows, bd)
        for h, c, nm in [(hy, Cy, "y"), (hx, Cx, "x")]:
            _open_nan_check(c[nxt], (h != 0) & (np.arange(n) < n - 1)[:, None], f"next close of {nm}",
                            lane_pair, pairs, rows, bd)
        sf = lambda h, d: np.where(h != 0, h * np.where(h != 0, d, 0.0), 0.0)
        g = (sf(hy0, Ey - Cy) + sf(hy, Cy[nxt] - Ey)) + (sf(hx0, Ex - Cx) + sf(hx, Cx[nxt] - Ex))
        g[-1] = sf(hy0[-1], Ey[-1] - Cy[-1]) + sf(hx0[-1], Ex[-1] - Cx[-1])
    else:
        g = (hy0 * (Ey - Cy) + hy * (Cy[nxt] - Ey)) + (hx0 * (Ex - Cx) + hx * (Cx[nxt] - Ex))
        g[-1] = hy0[-1] * (Ey[-1] - Cy[-1]) + hx0[-1] * (Ex[-1] - Cx[-1])
    hs = np.asarray(half_spread, dtype=float)
    if daily:
        assert hs.shape == (len(bd.permnos),), "half_spread must be positional over the panel columns"
        if not np.isfinite(hs[np.concatenate([iy, ix])]).all():
            raise ValueError(f"missing half-spread for a selected stock in {month}")
        if cost_fn is None:
            cost_fn = lambda h, px, rr, cc, mult: TC.leg_cost_bps(h[None, :], px, mult)   # TC cap/tick: minute defaults
        ly_, lx_ = iy[lane_pair], ix[lane_pair]            # per lane, so delisting exits are priced at the close
        cy_bps = cost_fn(hs[ly_], Ey * fac[:, ly_], rows, ly_, cost_mult)
        cx_bps = cost_fn(hs[lx_], Ex * fac[:, lx_], rows, lx_, cost_mult)
        dy, dx_ = np.abs(hy - hy0), np.abs(hx - hx0)
        # a worthless delisting (TR 0) has nothing left to trade: no cost (0 x the undefined bps is not a cost)
        cost = (np.where(dy > 0, dy * np.where((dy > 0) & (Ey > 0), Ey * cy_bps, 0.0), 0.0)
                + np.where(dx_ > 0, dx_ * np.where((dx_ > 0) & (Ex > 0), Ex * cx_bps, 0.0), 0.0)) / 1e4
        if not np.isfinite(cost).all():
            raise AssertionError("non-finite transaction cost on a traded leg")
        if bd.borrow_fee is not None:
            fee_y, fee_x = bd.borrow_fee[iy][lane_pair][None, :], bd.borrow_fee[ix][lane_pair][None, :]
        else:
            fee_y = fee_x = C.BORROW_FEE_ANNUAL
        short_val = (np.where(hy < 0, -hy * np.where(hy < 0, Cy, 0.0), 0.0) * fee_y
                     + np.where(hx < 0, -hx * np.where(hx < 0, Cx, 0.0), 0.0) * fee_x)
        borrow = short_val * bd.dt_days / C.TRADING_DAYS
    else:
        if cost_fn is None:
            cy_bps = TC.leg_cost_bps(hs[iy][None, :], Em[:, iy] * fac[:, iy], cost_mult)[:, lane_pair]
            cx_bps = TC.leg_cost_bps(hs[ix][None, :], Em[:, ix] * fac[:, ix], cost_mult)[:, lane_pair]
        else:
            cy_bps = cost_fn(hs[iy], Em[:, iy] * fac[:, iy], rows, iy, cost_mult)[:, lane_pair]
            cx_bps = cost_fn(hs[ix], Em[:, ix] * fac[:, ix], rows, ix, cost_mult)[:, lane_pair]
        cost = (np.abs(hy - hy0) * Ey * cy_bps + np.abs(hx - hx0) * Ex * cx_bps) / 1e4
        short_val = np.abs(np.minimum(hy, 0)) * Cy + np.abs(np.minimum(hx, 0)) * Cx
        borrow = short_val * C.BORROW_FEE_ANNUAL * bd.dt_days / C.TRADING_DAYS

    day_g = bd.day[rows][nxt]
    day_c = getattr(bd, "fill_day", bd.day)[rows].copy()      # costs are booked when the fill happens
    day_c[-1] = bd.day[rows][-1]                               # month-end exit: on the close
    ud = np.unique(np.concatenate([day_g, day_c, bd.day[rows]]))
    pos_g, pos_c = np.searchsorted(ud, day_g), np.searchsorted(ud, day_c)
    G = np.zeros((len(ud), L)); Cc = np.zeros((len(ud), L)); B = np.zeros((len(ud), L))
    np.add.at(G, pos_g, g if daily else np.nan_to_num(g))
    if daily:       # a delisting exit is filled on its own row's close, not on the next day
        cost_dead = np.where(dead_l, cost, 0.0)
        np.add.at(Cc, pos_c, cost - cost_dead)
        np.add.at(Cc, np.searchsorted(ud, bd.day[rows]), cost_dead)
    else:
        np.add.at(Cc, pos_c, cost)
    np.add.at(B, np.searchsorted(ud, bd.day[rows]), borrow)
    idx = bd.days[ud]
    slot = pairs["slot"].to_numpy() if "slot" in pairs else np.arange(P)
    cols = pd.MultiIndex.from_tuples([(combos[q][0], combos[q][1], combos[q][2], slot[p])
                                      for q, p in zip(lane_combo, lane_pair)],
                                     names=["method", "entry", "exit", "slot"])
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    gross, costf, borf = mk(G), mk(Cc), mk(B)

    # ---- trade list -----------------------------------------------------------------------
    hl_name = "hl_days" if daily else "hl_hours"
    tr = []
    for lane, a, b_, reason, side in trades:
        p = lane_pair[lane]
        pnl = hy[a, lane] * (Ey[b_, lane] - Ey[a, lane]) + hx[a, lane] * (Ex[b_, lane] - Ex[a, lane])
        c = cost[a, lane] + cost[b_, lane]
        bw = borrow[a:b_, lane].sum()
        m, e, x = combos[lane_combo[lane]]
        tr.append({"month": str(month), "method": m, "entry": e, "exit": x, "slot": slot[p], "y": pairs["y"].iloc[p],
                   "x": pairs["x"].iloc[p], "side": int(side), "open": bd.ts[rows[a]], "close": bd.ts[rows[b_]],
                   "bars": int(b_ - a), "reason": REASONS[int(reason)], "gross": pnl, "cost": c, "borrow": bw,
                   hl_name: pairs[hl_name].iloc[p], "p_eg": pairs["p_eg"].iloc[p],
                   "z_entry": z[a, lane]})
    sig = None
    if return_signals:
        sig = {"ts": bd.ts[rows], "z": z, "hy": hy, "lanes": cols, "Cy": Cy, "Cx": Cx}
    return MonthResult(daily=gross - costf - borf, gross=gross, cost=costf, borrow=borf,
                       trades=pd.DataFrame(tr), signals=sig)


def run(bd, selected: dict[str, pd.DataFrame], half_spread: pd.DataFrame,
        methods=C.HEDGE_METHODS, entries=C.ENTRY_Z, exits=C.EXIT_Z, cost_mult: float = 1.0,
        keep_slot: bool = False, months=None, cost_fn=None, n_pairs: int | None = None) -> dict:
    """Walk forward over all trading months. `selected[month]` = pairs chosen for that month;
    `half_spread` = (month x column) half-spread estimates *from the previous month*.
    Returns daily P&L frames (net, gross, cost, borrow) with one column per parameter combination
    (or per combination and `slot` label if keep_slot, e.g. to run many random portfolios at once).

    `months` defaults to `selection.trade_months()` (the minute-bar study). For DailyBars a month that has
    pairs but no row in `half_spread` raises instead of trading at zero cost."""
    daily = bd.bars_per_hour is None
    months = trade_months() if months is None else pd.PeriodIndex(months, freq="M")
    parts = {"net": [], "gross": [], "cost": [], "borrow": []}
    trades = []
    levels = ["method", "entry", "exit"] + (["slot"] if keep_slot else [])
    if keep_slot:
        slots = sorted({s for v in selected.values() if v is not None and len(v) for s in v["slot"]})
        combos = pd.MultiIndex.from_tuples([(m, e, x, s) for m in methods for e in entries for x in exits
                                            for s in slots], names=levels)
    else:
        combos = pd.MultiIndex.from_tuples([(m, e, x) for m in methods for e in entries for x in exits],
                                           names=levels)
    for month in months:
        key = str(month)
        pairs = selected.get(key)
        rows = np.nonzero(bd.month == month)[0]
        days = bd.days[np.unique(bd.day[rows])]
        has = pairs is not None and len(pairs) > 0
        if month in half_spread.index:
            hs = half_spread.loc[month]
        elif daily and has:
            raise KeyError(f"no half-spread row for {month}: refusing to trade at zero cost")
        else:
            hs = pd.Series(0.0, index=half_spread.columns)
        res = run_month(bd, month, pairs if pairs is not None else pd.DataFrame(), methods, entries, exits,
                        hs, cost_mult, cost_fn=cost_fn, n_pairs=n_pairs) if has else None
        for k, frame in zip(["net", "gross", "cost", "borrow"],
                            [None if res is None else getattr(res, a) for a in ["daily", "gross", "cost", "borrow"]]):
            if frame is None:
                agg = pd.DataFrame(0.0, index=days, columns=combos)
            else:
                agg = frame.T.groupby(level=levels).sum().T.reindex(columns=combos, fill_value=0.0)
                agg = agg.reindex(days, fill_value=0.0)
            parts[k].append(agg)
        if res is not None and not res.trades.empty:
            trades.append(res.trades)
    out = {k: pd.concat(v).sort_index() for k, v in parts.items()}
    out = {k: v.groupby(level=0).sum() for k, v in out.items()}
    out["trades"] = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return out
