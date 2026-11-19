"""backtest.run_month / run on contracts.DailyBars: delisting exits, short ban, time stop in days, fills, NaN rule."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import backtest as B  # noqa: E402
from pairs import config as C  # noqa: E402
from pairs import contracts as CT  # noqa: E402

BETA, SIGMA = 1.2, 0.01
MY = BETA * np.log(50) + 0.3
MX = np.log(50)
N = 65
DAYS = pd.bdate_range("2023-01-02", periods=N)
MONTH = DAYS[-1].to_period("M")
K0 = int(np.nonzero(DAYS.to_period("M") == MONTH)[0][0])      # first row of the traded month


def make_bars(zpath, exec_mode="t1_close", dead=None, short_ok=None, nan_y_from=None, y_scale=None,
              borrow_fee=None):
    """Two stocks. Stock 0 (y) is built so that the *static* z equals `zpath` (x is flat), indexed from K0."""
    ly = np.full(N, MY)
    z = np.zeros(N)
    z[K0:K0 + len(zpath)] = zpath
    ly = MY + z * SIGMA
    if y_scale is not None:
        ly = ly + y_scale
    close = np.exp(np.column_stack([ly, np.full(N, MX)]))
    if nan_y_from is not None:
        close[nan_y_from:, 0] = np.nan
    n = N
    if exec_mode == "t_close":
        ex, fd = close.copy(), np.arange(n)
    else:
        ex, fd = np.vstack([close[1:], close[-1:]]), np.minimum(np.arange(n) + 1, n - 1)
    d = np.zeros((n, 2), bool) if dead is None else dead
    so = np.ones((n, 2), bool) if short_ok is None else short_ok
    bars = CT.DailyBars(permnos=np.array([10001, 10002]), days=DAYS, close=close, exec=ex, fill_day=fd,
                        fresh=np.ones((n, 2), bool), factor=np.ones((n, 2)), dead=d, short_ok=so,
                        exch=np.full((n, 2), "N", dtype=object), exec_mode=exec_mode, borrow_fee=borrow_fee)
    CT.validate_bars(bars)
    return bars


def pair(**kw):
    row = {"iy": 0, "ix": 1, "y": 10001, "x": 10002, "beta": BETA, "sigma": SIGMA, "sd_lx": 0.05,
           "var_beta": 1e-6, "var_alpha": 1e-8, "mean_ly": MY, "mean_lx": MX, "hl_days": 5.0, "p_eg": 0.01}
    row.update(kw)
    return pd.DataFrame([row])


HS = pd.Series([0.0, 0.0])
W = C.LEVERAGE / C.N_PAIRS
WY, WX = W / (1 + BETA), W * BETA / (1 + BETA)


def go(bd, pr=None, hs=HS, mult=0.0, **kw):
    return B.run_month(bd, MONTH, pair() if pr is None else pr, ["static"], [2.0], [0.5], hs, cost_mult=mult, **kw)


def entry_path(n=22, hold=2.5):
    z = np.zeros(n)
    z[3:] = hold                                   # signal on month-row 3 -> fill on row 4
    return z


# ------------------------------------------------------------------------------------------------
def test_delisting_closes_both_legs_with_right_pnl():
    n = len(DAYS) - K0
    z = entry_path(n)
    z[8] = 1.0                                     # the delisting return: y falls, our short gains
    dead = np.zeros((N, 2), bool)
    dead[K0 + 8, 0] = True
    bd = make_bars(z, dead=dead, nan_y_from=K0 + 9)
    res = go(bd)
    tr = res.trades
    assert len(tr) == 1 and tr["reason"].iloc[0] == "delisting"
    assert tr["close"].iloc[0] == DAYS[K0 + 8] and tr["open"].iloc[0] == DAYS[K0 + 3]
    cy = bd.close[:, 0]
    expected = -WY * (cy[K0 + 8] / cy[K0 + 4] - 1)             # short y filled at close of row 4, x flat
    assert np.isclose(tr["gross"].iloc[0], expected, atol=1e-12) and expected > 0
    assert np.isclose(res.gross.sum().sum(), tr["gross"].sum(), atol=1e-12)
    assert np.isfinite(res.daily.to_numpy()).all()
    assert res.gross.index.max() >= DAYS[K0 + 8]
    # the live leg (x) exits the same day, so nothing is held afterwards
    assert (res.gross.loc[DAYS[K0 + 10]:] == 0).all().all()


def test_worthless_delisting_of_the_long_leg_loses_it_all_at_no_cost():
    """CIZ delisting row with a -100% return: TR close 0. The long leg loses its whole value; the cost of a
    worthless exit is 0 even when the cost function gives no finite bps at price 0 (tick floor / 0)."""
    n = len(DAYS) - K0
    z = entry_path(n, hold=-2.5)                   # y cheap: long y, short x
    dead = np.zeros((N, 2), bool)
    dead[K0 + 8, 0] = True
    bd = make_bars(z, dead=dead, nan_y_from=K0 + 9)
    bd.close[K0 + 8, 0] = 0.0
    bd.exec[K0 + 7, 0] = 0.0

    def cost_fn(hs, px, rows, cols, mult):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(px > 0, 5.0, 0.005 / px * 1e4)       # inf / nan at price 0, like the tick floor

    with np.errstate(divide="ignore", invalid="ignore"):
        res = go(bd, hs=pd.Series([1.0, 1.0]), mult=1.0, cost_fn=cost_fn)
    tr = res.trades
    assert len(tr) == 1 and tr["reason"].iloc[0] == "delisting"
    assert np.isclose(tr["gross"].iloc[0], -WY, atol=1e-12)                    # long y bought at row 4, worth 0
    assert np.isfinite(tr["cost"].iloc[0]) and np.isclose(tr["cost"].iloc[0], (W + WX) * 5.0 / 1e4)
    assert np.isfinite(res.daily.to_numpy()).all()


def test_no_entry_when_leg_dies_before_the_fill():
    n = len(DAYS) - K0
    z = np.zeros(n)
    z[5] = 2.5
    dead = np.zeros((N, 2), bool)
    dead[K0 + 6, 0] = True                         # the t+1 fill row is the last row of y
    bd = make_bars(z, dead=dead, nan_y_from=K0 + 7)
    assert go(bd) is None or go(bd).trades.empty


def test_short_ban_blocks_new_short_legs_only():
    n = len(DAYS) - K0
    z = entry_path(n)
    so = np.ones((N, 2), bool)
    so[:, 0] = False                               # y may not be shorted; z > 0 means short y
    assert go(make_bars(z, short_ok=so)).trades.empty
    so = np.ones((N, 2), bool)
    so[:, 1] = False                               # x may not be shorted; z > 0 means long x: fine
    assert len(go(make_bars(z, short_ok=so)).trades) == 1
    so = np.ones((N, 2), bool)
    so[:, 1] = False
    assert go(make_bars(-z, short_ok=so)).trades.empty        # z < 0 shorts x
    so = np.ones((N, 2), bool)
    so[K0 + 4:, 0] = False                         # ban starts on the fill row of an entry signalled on row 3
    assert go(make_bars(z, short_ok=so)).trades.empty
    so = np.ones((N, 2), bool)
    so[K0 + 5:, 0] = False                         # ban after the fill: the open trade is not touched
    assert len(go(make_bars(z, short_ok=so)).trades) == 1


def test_time_stop_is_three_half_lives_in_days():
    n = len(DAYS) - K0
    z = entry_path(n)                              # never reverts, stays inside the stop band
    for hl, bars in [(2.0, 6), (4.0, 12)]:
        tr = go(make_bars(z), pair(hl_days=hl)).trades
        assert tr["reason"].iloc[0] == "time stop" and tr["bars"].iloc[0] == bars, (hl, tr)
        assert "hl_days" in tr.columns


def test_fill_modes_t1_close_vs_t_close():
    n = len(DAYS) - K0
    z = entry_path(n)
    z[3:8] = 2.5 + 0.1 * np.arange(5)              # drifting so that the two fill rules give different prices
    z[8:] = 0.0                                    # reversion: exit on row 8
    out = {}
    for mode in ("t1_close", "t_close"):
        bd = make_bars(z, exec_mode=mode)
        res = go(bd, hs=pd.Series([10.0, 10.0]), mult=1.0)
        tr = res.trades.iloc[0]
        cy = bd.close[:, 0]
        k = 1 if mode == "t1_close" else 0
        expected = -WY * (cy[K0 + 8 + k] / cy[K0 + 3 + k] - 1)
        assert np.isclose(tr["gross"], expected, atol=1e-12), mode
        # costs are booked on the day the fill happens
        paid = res.cost.sum(axis=1)
        assert (paid > 0).sum() == 2
        assert paid.index[paid > 0].tolist() == [DAYS[K0 + 3 + k], DAYS[K0 + 8 + k]]
        out[mode] = tr["gross"]
    assert out["t1_close"] != out["t_close"]


def test_nan_price_while_open_raises_but_flat_nan_is_fine():
    n = len(DAYS) - K0
    z = entry_path(n)
    bd = make_bars(z)
    bd.close[K0 + 6, 0] = np.nan
    bd.exec[K0 + 5, 0] = np.nan
    try:
        go(bd)
        raise AssertionError("expected a NaN assertion")
    except AssertionError as e:
        assert "NaN" in str(e)
    z2 = np.zeros(n)                               # never in a trade: NaN prices are harmless
    bd2 = make_bars(z2, nan_y_from=K0 + 10)
    res = go(bd2)
    assert res is not None and res.trades.empty and np.isfinite(res.daily.to_numpy()).all()


def test_cost_fn_plumbing_and_missing_spread_raises():
    n = len(DAYS) - K0
    z = entry_path(n)
    z[8] = 1.0
    dead = np.zeros((N, 2), bool)
    dead[K0 + 8, 0] = True
    bd = make_bars(z, dead=dead, nan_y_from=K0 + 9)
    seen = []

    def cost_fn(hs, px, rows, cols, mult):
        seen.append((hs.shape, px.shape, rows.shape, cols.tolist(), mult))
        return np.full(px.shape, 7.0) * mult

    res = go(bd, hs=pd.Series([3.0, 4.0]), mult=2.0, cost_fn=cost_fn)
    assert len(seen) == 2 and seen[0][3] == [0] and seen[1][3] == [1] and seen[0][4] == 2.0
    assert seen[0][1] == (len(np.nonzero(bd.month == MONTH)[0]), 1)
    tr = res.trades.iloc[0]
    cy = bd.close[:, 0]
    exit_notional = WY / cy[K0 + 4] * cy[K0 + 8] + WX                          # y at its delisting close, x flat
    assert np.isclose(tr["cost"], (W + exit_notional) * 14.0 / 1e4)            # entry + exit, both legs, 7 bps x mult 2
    assert np.isclose(res.cost.sum().sum(), res.trades["cost"].sum())
    try:
        go(bd, hs=pd.Series([np.nan, 1.0]))
        raise AssertionError("missing half-spread must raise")
    except ValueError:
        pass


def test_borrow_fee_per_stock_and_n_pairs_scaling():
    n = len(DAYS) - K0
    z = entry_path(n)
    z[8:] = 0.0
    fee = np.array([0.0252, 0.0])                  # only y is shorted (z > 0)
    bd = make_bars(z, borrow_fee=fee)
    res = go(bd)
    cy = bd.close[:, 0]
    exp = sum(WY / cy[K0 + 4] * cy[t] * 0.0252 / 252 for t in range(K0 + 3, K0 + 8))
    assert np.isclose(res.borrow.sum().sum(), exp, rtol=1e-9) and np.isclose(res.trades["borrow"].iloc[0], exp, rtol=1e-9)
    r5 = go(make_bars(z)).trades["gross"].iloc[0]
    r20 = go(make_bars(z), n_pairs=20).trades["gross"].iloc[0]
    assert np.isclose(r20 * 4, r5)                 # 20 slots: each leg carries 1/4 of the 5-slot weight


def test_run_months_argument_and_missing_spread_month_raises():
    n = len(DAYS) - K0
    z = entry_path(n)
    z[8:] = 0.0
    bd = make_bars(z)
    sel = {str(MONTH): pair().assign(slot=0)}
    hs = pd.DataFrame(0.0, index=pd.PeriodIndex([MONTH], freq="M"), columns=[10001, 10002])
    out = B.run(bd, sel, hs, ["static"], [2.0], [0.5], months=[MONTH])
    assert len(out["trades"]) == 1 and out["net"].shape[1] == 1
    assert np.isclose(out["gross"].sum().sum(), out["trades"]["gross"].sum())
    try:
        B.run(bd, sel, hs.iloc[0:0], ["static"], [2.0], [0.5], months=[MONTH])
        raise AssertionError("missing spread month must raise for DailyBars")
    except KeyError:
        pass
    out2 = B.run(bd, {}, hs.iloc[0:0], ["static"], [2.0], [0.5], months=[MONTH])      # no pairs: no spread needed
    assert out2["trades"].empty and (out2["net"] == 0).all().all()
