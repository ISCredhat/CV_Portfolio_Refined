"""Inference helpers added to stats.py: Holm, NW test, stationary bootstrap, Ledoit-Wolf, MDEs, trend with a break."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)
from pairs import stats as S  # noqa: E402


def test_holm_hand_example_and_optional_statsmodels():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj, rej = S.holm(p, 0.05)
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02]) and rej.tolist() == [True, False, False, True]
    try:
        from statsmodels.stats.multitest import multipletests
    except Exception:
        return
    rng = np.random.default_rng(0)
    q = rng.random(20) ** 2
    assert np.allclose(S.holm(q)[0], multipletests(q, method="holm")[1])


def test_newey_west_mean_test():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(400) * 0.02 + 0.004
    r = S.newey_west_mean_test(x, 6, "greater")
    ols = S.newey_west_ols(x, np.ones((400, 1)), 6)
    assert np.isclose(r["t"], ols["t"][0]) and np.isclose(r["mean"], x.mean())
    assert np.isclose(r["p"] + S.newey_west_mean_test(x, 6, "less")["p"], 1.0)
    # AR(1) data: NW se is larger than the naive se
    e = np.zeros(2000)
    z = rng.standard_normal(2000)
    for i in range(1, 2000):
        e[i] = 0.6 * e[i - 1] + z[i]
    assert S.newey_west_mean_test(e, 12)["se"] > 1.3 * e.std() / np.sqrt(2000)
    try:
        S.newey_west_mean_test(np.array([1.0, np.nan, 2.0]))
        raise AssertionError
    except ValueError:
        pass
    try:
        import statsmodels.api as sm
        m = sm.OLS(x, np.ones(400)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        assert np.isclose(m.bse[0], r["se"], rtol=1e-8)
    except ImportError:
        pass


def test_stationary_bootstrap_block_lengths():
    rng = np.random.default_rng(2)
    n, reps = 1000, 400
    for b in (5.0, 21.0):
        idx = S.stationary_bootstrap_indices(n, reps, b, rng)
        assert idx.shape == (reps, n) and idx.min() >= 0 and idx.max() < n
        cont = (idx[:, 1:] == (idx[:, :-1] + 1) % n).mean()          # continuation prob = 1 - 1/b (+ 1/n by chance)
        assert abs(cont - (1 - 1 / b + 1 / (b * n))) < 0.01, (b, cont)
    z = rng.standard_normal(n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.7 * ar[i - 1] + z[i]
    assert S.politis_white_block(z) < 5 < S.politis_white_block(ar)


def test_sharpe_stationary_bootstrap_ci():
    rng = np.random.default_rng(3)
    good = rng.standard_normal(2500) * 0.01 + 0.0008            # annual Sharpe ~1.27
    noise = rng.standard_normal(2500) * 0.01
    g = S.sharpe_stationary_bootstrap(good, reps=1000, seed=1)
    n_ = S.sharpe_stationary_bootstrap(noise, reps=1000, seed=1, mean_block=21)
    assert g["lo"] < g["sharpe"] < g["hi"] and g["lower_one_sided"] > 0
    assert n_["lower_one_sided"] < 0 < n_["hi"] and n_["block"] == 21
    again = S.sharpe_stationary_bootstrap(good, reps=1000, seed=1)
    assert again["lo"] == g["lo"]                                # seeded
    try:
        S.sharpe_stationary_bootstrap(np.array([0.1, np.nan]))
        raise AssertionError
    except ValueError:
        pass


def test_ledoit_wolf_size_power_and_se():
    rng = np.random.default_rng(4)
    rej, T = 0, 500
    for _ in range(400):
        c = rng.standard_normal(T)
        a = 0.0006 + 0.01 * (0.6 * c + 0.8 * rng.standard_normal(T))      # same Sharpe, correlated
        b = 0.0006 + 0.01 * (0.6 * c + 0.8 * rng.standard_normal(T))
        rej += S.ledoit_wolf_sharpe_diff(a, b)["p_two_sided"] < 0.05
    assert 0.02 < rej / 400 < 0.10, rej / 400
    a = 0.002 + 0.01 * rng.standard_normal(2000)
    b = 0.0 + 0.01 * rng.standard_normal(2000)
    r = S.ledoit_wolf_sharpe_diff(a, b)
    assert r["p_one_sided"] < 0.001 and r["diff"] > 0 and abs(r["diff"] - (r["sr1"] - r["sr2"])) < 1e-9
    assert S.ledoit_wolf_sharpe_diff(b, a)["p_one_sided"] > 0.99
    same = S.ledoit_wolf_sharpe_diff(a, a)
    assert abs(same["diff"]) < 1e-12 or same["se"] == 0 or abs(same["z"]) < 1e-6
    # independent iid normals: Var(SR1 - SR2) = 2 (1 + SR^2 / 2) / T per period
    x, y = 0.01 * rng.standard_normal(20000) + 0.0005, 0.01 * rng.standard_normal(20000) + 0.0005
    out = S.ledoit_wolf_sharpe_diff(x, y, lags=0, periods=1)
    assert abs(out["se"] / np.sqrt(2 * (1 + 0.05 ** 2 / 2) / 20000) - 1) < 0.08


def test_mde_values_match_the_review():
    assert abs(S.mde_sharpe(30.8) - 0.45) < 0.01
    assert abs(S.mde_sharpe(3.0) - 1.4) < 0.05
    assert abs(S.mde_sharpe(30.8, corr=0.3) - 0.53) < 0.01
    assert abs(S.mde_trend_change(432) - 0.41) < 0.01


def test_trend_with_break():
    rng = np.random.default_rng(5)
    idx = pd.period_range("1990-01", "2025-12", freq="M")
    yrs = np.arange(len(idx)) / 12
    post = idx >= pd.Period("2001-04", "M")
    flat_then_fall = pd.Series(np.where(post, -0.002 * (yrs - yrs[post][0]), 0.0) + 0.01 * rng.standard_normal(len(idx)), idx)
    r = S.trend_test_with_break(flat_then_fall, "2001-04", 12)
    assert r["slope"] < 0 and r["slope_p_neg"] < 0.01
    assert abs(r["slope_pre"]) < 0.003 and abs(r["slope_post"] + 0.002) < 0.0005 and r["slope_post_t"] < -3
    assert r["n_pre"] + r["n_post"] == len(idx) == r["n"]
    noise = pd.Series(0.01 * rng.standard_normal(len(idx)), idx)
    assert S.trend_test_with_break(noise)["slope_p_neg"] > 0.01


def test_no_statsmodels_import_in_src_modules():
    code = ("import sys; sys.path.insert(0, %r)\n"
            "import pairs.stats, pairs.coint, pairs.backtest, pairs.selection_daily\n"
            "import importlib\n"
            "for m in ['ggr', 'nulls', 'prereg']:\n"
            "    importlib.import_module('pairs.' + m)\n"
            "assert 'statsmodels' not in sys.modules, 'statsmodels imported'\n") % SRC
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
