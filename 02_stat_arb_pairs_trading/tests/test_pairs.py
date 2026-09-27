"""Unit tests: statistical tests, data handling, look-ahead and P&L accounting.

    python tests/test_pairs.py        (or: pytest tests)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import backtest as B  # noqa: E402
from pairs import coint as K  # noqa: E402
from pairs import config as C  # noqa: E402
from pairs import costs as TC  # noqa: E402
from pairs import data as D  # noqa: E402


# ------------------------------------------------------------------------ statistics
def test_mackinnon_p_values_at_critical_values():
    # MacKinnon (2010) 5% / 1% critical values, constant, N = 1 and N = 2
    assert abs(K.mackinnon_p(-2.86154, 1) - 0.05) < 0.002
    assert abs(K.mackinnon_p(-3.43035, 1) - 0.01) < 0.001
    assert abs(K.mackinnon_p(-3.33613, 2) - 0.05) < 0.002
    assert abs(K.mackinnon_p(-3.89644, 2) - 0.01) < 0.001


def test_adf_and_engle_granger_size_and_power():
    rng = np.random.default_rng(0)
    n, reps = 800, 300
    size_adf = np.mean([K.adf(np.cumsum(rng.standard_normal(n)), maxlag=6)["pvalue"] < 0.05 for _ in range(reps)])
    size_eg = np.mean([K.engle_granger(np.cumsum(rng.standard_normal(n)), np.cumsum(rng.standard_normal(n)),
                                       maxlag=6)["pvalue"] < 0.05 for _ in range(reps)])
    assert 0.02 < size_adf < 0.09, size_adf
    assert 0.02 < size_eg < 0.09, size_eg
    x = np.cumsum(rng.standard_normal(n))
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = 0.9 * u[i - 1] + rng.standard_normal()
    r = K.engle_granger(1.5 * x + u, x, maxlag=6)
    assert r["pvalue"] < 0.01 and abs(r["beta"] - 1.5) < 0.05


def test_johansen_critical_value_matches_simulation():
    rng = np.random.default_rng(1)
    st = [K.johansen_trace(np.cumsum(rng.standard_normal((600, 2)), 0))["trace"] for _ in range(1500)]
    q95 = np.quantile(st, 0.95)
    assert abs(q95 - K.JOHANSEN_CV_R0[1]) < 1.2, q95


class Skip(Exception):
    pass


def test_against_statsmodels_if_installed():
    try:
        from statsmodels.tsa.stattools import adfuller, coint
    except ImportError:
        try:
            import pytest
            pytest.skip("statsmodels not installed")
        except ImportError:
            raise Skip("statsmodels not installed")
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(500))
    y = 0.8 * x + np.cumsum(rng.standard_normal(500)) * 0.3
    a = K.adf(y, maxlag=8)
    s = adfuller(y, maxlag=8, autolag="BIC", regression="c")
    assert abs(a["stat"] - s[0]) < 1e-6 and abs(a["pvalue"] - s[1]) < 1e-6
    e = K.engle_granger(y, x, maxlag=8)
    c = coint(y, x, trend="c", maxlag=8, autolag="bic")
    assert abs(e["stat"] - c[0]) < 1e-6 and abs(e["pvalue"] - c[1]) < 1e-6


def test_half_life_hurst_and_bh():
    rng = np.random.default_rng(3)
    phi = 0.95
    u = np.zeros(20000)
    for i in range(1, len(u)):
        u[i] = phi * u[i - 1] + rng.standard_normal()
    true_hl = -np.log(2) / np.log(phi)
    assert abs(K.half_life(u) - true_hl) / true_hl < 0.1
    assert K.hurst(u) < 0.45 and abs(K.hurst(np.cumsum(rng.standard_normal(20000))) - 0.5) < 0.05
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert K.benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 8


# ------------------------------------------------------------------------ data
def test_fill_helpers_and_bar_ends():
    a = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, np.nan], [4.0, 5.0]])
    assert np.allclose(D.ffill(a), [[np.nan, 1], [2, 1], [2, 1], [4, 5]], equal_nan=True)
    assert np.allclose(D.bfill(a), [[2, 1], [2, 5], [4, 5], [4, 5]], equal_nan=True)
    cal = {"day_start": np.array([0, 390, 720]), "n_per_day": np.array([390, 330, 210])}
    e = D.bar_ends(cal, "1h")
    assert e[0] == 59 and e[6] == 389                     # 15:30-16:00 is a half bar
    assert (e[7:13] == 390 + np.array([59, 119, 179, 239, 299, 329])).all()   # 15:00 session end
    assert (D.bar_ends(cal, "1D") == [389, 719, 929]).all()


def test_split_factors_back_adjust():
    days = pd.bdate_range("2022-07-14", periods=4)
    sp = pd.DataFrame({"ticker": ["GOOG"], "date": [pd.Timestamp("2022-07-18")], "ratio": [20.0]})
    f = D.split_factors(sp, ["GOOG", "MSFT"], days)
    assert f[:, 0].tolist() == [20.0, 20.0, 1.0, 1.0] and (f[:, 1] == 1).all()


def test_cost_floor_is_half_a_tick():
    c = TC.leg_cost_bps(np.array([0.0]), np.array([5.0]))
    expected = 0.5 * 0.01 / 5 * 1e4 + 0.0035 / 5 * 1e4 + C.IMPACT_BPS
    assert abs(c[0] - expected) < 1e-9


# ------------------------------------------------------------------------ backtest
class _FakeBars:
    """Minimal BarData stand-in: 1 month of 30-minute bars for a synthetic pair."""

    def __init__(self, ly, lx, bpd=13):
        n = len(ly)
        self.close = np.exp(np.column_stack([ly, lx]))
        self.exec = np.vstack([self.close[1:], self.close[-1:]])      # fill at next bar's close
        n_days = n // bpd
        self.days = pd.bdate_range("2023-03-01", periods=n_days)
        self.day = np.repeat(np.arange(n_days), bpd)
        self.ts = self.days[self.day] + pd.to_timedelta(np.tile(np.arange(bpd) * 30, n_days), unit="m")
        self.month = pd.PeriodIndex(self.ts, freq="M")
        self.fresh = np.ones_like(self.close, bool)
        self.factor = np.ones_like(self.close)
        self.dt_days, self.bars_per_hour, self.bars_per_day = 30 / 390, 2.0, bpd


def _pair(beta=1.2, sigma=0.01, hl=5.0):
    return pd.DataFrame([{"iy": 0, "ix": 1, "y": "Y", "x": "X", "beta": beta, "sigma": sigma, "sd_lx": 0.05,
                          "var_beta": 1e-6, "var_alpha": 1e-8, "mean_ly": beta * np.log(50) + 0.3,
                          "mean_lx": np.log(50), "hl_hours": hl, "p_eg": 0.01}])


def _synthetic(seed, phi=0.9, n_days=21, bpd=13, spread_rw=False):
    rng = np.random.default_rng(seed)
    n = n_days * bpd
    lx = np.log(50) + np.cumsum(rng.standard_normal(n)) * 0.005
    if spread_rw:
        u = np.cumsum(rng.standard_normal(n)) * 0.003
    else:
        u = np.zeros(n)
        for i in range(1, n):
            u[i] = phi * u[i - 1] + rng.standard_normal() * 0.01 * np.sqrt(1 - phi ** 2)
    return 1.2 * lx + u + 0.3, lx


def test_mean_reverting_spread_makes_money_and_accounting_adds_up():
    tot = {m: 0.0 for m in C.HEDGE_METHODS}
    for s in range(15):
        bd = _FakeBars(*_synthetic(s))
        res = B.run_month(bd, bd.month[0], _pair(), C.HEDGE_METHODS, [2.0], [0.0], pd.Series([0.0, 0.0]), cost_mult=0.0)
        g = res.gross.sum().groupby(level="method").sum()
        t = res.trades.groupby("method")["gross"].sum().reindex(g.index, fill_value=0.0)
        assert np.allclose(g, t, atol=1e-12)                  # daily P&L == sum of trade P&Ls
        for m in C.HEDGE_METHODS:
            tot[m] += g[m]
    assert all(v > 0 for v in tot.values()), tot


def test_random_walk_spread_has_no_edge():
    vals = []
    for s in range(150):
        bd = _FakeBars(*_synthetic(100 + s, spread_rw=True))
        res = B.run_month(bd, bd.month[0], _pair(), ["static", "rolling"], [2.0], [0.0], pd.Series([0.0, 0.0]),
                          cost_mult=0.0)
        vals.append(res.gross.sum().sum())
    v = np.array(vals)
    assert abs(v.mean() / (v.std() / np.sqrt(len(v)))) < 3.0


def test_costs_charged_on_every_fill():
    bd = _FakeBars(*_synthetic(7))
    hs = pd.Series([5.0, 5.0])                                 # 5 bps half-spread on both legs
    res = B.run_month(bd, bd.month[0], _pair(), ["static"], [2.0], [0.0], hs, cost_mult=1.0)
    tr = res.trades
    assert len(tr) > 0
    assert np.isclose(tr["cost"].sum(), res.cost.sum().sum())
    px = bd.exec[:, 0]
    per_leg = TC.leg_cost_bps(np.array([5.0]), np.array([px.mean()]))[0]
    approx = len(tr) * 2 * (C.LEVERAGE / C.N_PAIRS) * per_leg / 1e4
    assert 0.8 < tr["cost"].sum() / approx < 1.25


def test_no_look_ahead_in_signals_or_positions():
    """Changing prices after bar k must not change anything decided at or before bar k."""
    ly, lx = _synthetic(11)
    k = 150
    bd1 = _FakeBars(ly, lx)
    ly2 = ly.copy()
    ly2[k + 1:] += np.linspace(0, 0.2, len(ly) - k - 1)       # perturb the future only
    bd2 = _FakeBars(ly2, lx)
    for m in C.HEDGE_METHODS:
        r1 = B.run_month(bd1, bd1.month[0], _pair(), [m], [1.5], [0.0], pd.Series([0.0, 0.0]), return_signals=True)
        r2 = B.run_month(bd2, bd2.month[0], _pair(), [m], [1.5], [0.0], pd.Series([0.0, 0.0]), return_signals=True)
        z1, z2 = r1.signals["z"][: k + 1], r2.signals["z"][: k + 1]
        assert np.allclose(z1, z2, equal_nan=True), m
        # decisions up to bar k are identical; share counts too, except that an order sent at bar k
        # is filled at a (perturbed) price after k
        assert (np.sign(r1.signals["hy"][: k + 1]) == np.sign(r2.signals["hy"][: k + 1])).all(), m
        assert np.allclose(r1.signals["hy"][:k], r2.signals["hy"][:k]), m


def test_month_end_exit_and_no_open_positions():
    bd = _FakeBars(*_synthetic(5))
    res = B.run_month(bd, bd.month[0], _pair(), ["static"], [1.5], [0.0], pd.Series([0.0, 0.0]), return_signals=True)
    assert res.signals["hy"][-1, 0] == 0.0                    # flat at the end of every month


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Skip as e:
                print("SKIP", name, f"({e})")
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
