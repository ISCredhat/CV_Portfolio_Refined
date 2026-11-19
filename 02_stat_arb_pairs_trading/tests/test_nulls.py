"""nulls.py: the resampled null keeps size at ~5% on random walks, ignores the future, and caches by key."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import nulls as N  # noqa: E402
from pairs import selection_daily as SD  # noqa: E402

DAYS = pd.bdate_range("2022-10-03", periods=300)
MONTH = pd.Period("2023-07", "M")
CUT = MONTH.start_time


def make(seed=0, n_stocks=60, n_coint=12):
    """Random walks in 3 industries; the first `n_coint` stocks of each industry follow their partner's walk."""
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    X = np.cumsum(rng.standard_normal((n, n_stocks)) * 0.015, 0) + np.log(40)
    for j in range(1, n_coint, 2):
        u = np.zeros(n)
        for i in range(1, n):
            u[i] = 0.8 * u[i - 1] + rng.standard_normal() * 0.01
        X[:, j] = 1.1 * X[:, j - 1] + u
    uni = pd.DataFrame({"permno": np.arange(n_stocks) + 1, "permco": np.arange(n_stocks) + 1,
                        "col": np.arange(n_stocks), "ff48": np.arange(n_stocks) // 20})
    return pd.DataFrame(X, index=DAYS), uni


def test_resampled_null_has_size_on_random_walks_and_kills_cointegration():
    logp, uni = make(1, 60, 0)                                   # no cointegration at all
    c = SD.test_month(logp, uni, MONTH)
    obs = (c["p_eg"] <= 0.05).mean()
    for kind, block in [("stationary", 21), ("stationary", 5), ("iid", 1)]:
        r = N.month_null_rate(logp, c, MONTH, kind, block, seed=7, reps_per_pair=5)
        # iid resampling has nominal size; block resampling of one 126-day sample over-rejects (~8% at block 21)
        assert (abs(r["rate"] - 0.05) < 0.02) if kind == "iid" else (0.04 < r["rate"] < 0.13), (kind, r)
    assert obs < 0.12
    logp2, uni2 = make(2, 60, 40)
    c2 = SD.test_month(logp2, uni2, MONTH)
    rate_obs = (c2["p_eg"] <= 0.05).mean()
    rate_null = N.month_null_rate(logp2, c2, MONTH, "iid", 1, seed=7, reps_per_pair=5)["rate"]
    assert rate_obs > rate_null + 0.015, (rate_obs, rate_null)    # injected cointegration shows up as excess
    assert N.iid_gaussian_null_rate(126, reps=3000, seed=3)["rate"] < 0.075


def test_null_is_seeded_and_cache_is_keyed(tmp_path):
    logp, uni = make(3)
    c = SD.test_month(logp, uni, MONTH)
    a = N.month_null_rate(logp, c, MONTH, "stationary", 21, seed=1, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 1
    b = N.month_null_rate(logp, c, MONTH, "stationary", 21, seed=1, cache_dir=tmp_path)
    assert a == b and len(list(tmp_path.glob("*.json"))) == 1         # hit
    fresh = N.month_null_rate(logp, c, MONTH, "stationary", 21, seed=1)
    assert fresh == a                                                  # a cache hit equals recomputation
    for kw in [dict(seed=2), dict(block=5), dict(kind="iid")]:
        args = dict(kind="stationary", block=21, seed=1) | kw
        N.month_null_rate(logp, c, MONTH, cache_dir=tmp_path, **args)
    assert len(list(tmp_path.glob("*.json"))) == 4                     # seed, block and type each get their own key
    assert N.cache_key(126, "stationary", 21, 1) != N.cache_key(252, "stationary", 21, 1)
    try:
        N.resampled_null_flags(logp.to_numpy()[:100], np.array([0]), np.array([1]), kind="bogus")
        raise AssertionError
    except ValueError:
        pass


def test_null_ignores_data_after_formation_end():
    logp, uni = make(4)
    c = SD.test_month(logp, uni, MONTH)
    base = N.month_null_rate(logp, c, MONTH, "stationary", 21, seed=5)
    pert = logp.copy()
    pert.loc[pert.index >= CUT] += np.random.default_rng(0).standard_normal((int((pert.index >= CUT).sum()), logp.shape[1]))
    assert N.month_null_rate(pert, c, MONTH, "stationary", 21, seed=5) == base
    early = logp.copy()
    early.iloc[:] = early.to_numpy() + np.random.default_rng(1).standard_normal(early.shape) * 0.1
    assert N.month_null_rate(early, c, MONTH, "stationary", 21, seed=5) != base      # formation data do matter


def test_monthly_excess_and_h2_decision_rule():
    logp, uni = make(5, 60, 12)
    months = [MONTH, MONTH - 1]
    cands = SD.run_months(logp, {str(m): uni for m in months}, months)
    ex = N.monthly_excess(logp, cands, "stationary", 21, seed=9)
    assert list(ex.columns) == ["n_pairs", "observed", "null", "excess_pp"] and len(ex) == 2
    assert np.allclose(ex["excess_pp"], 100 * (ex["observed"] - ex["null"]))
    rng = np.random.default_rng(0)
    strong = rng.standard_normal(200) * 3 + 2.0
    weak = rng.standard_normal(2000) * 2 + 0.5                         # significant but below 1 pp
    none = rng.standard_normal(200) * 3
    r = N.h2_test(strong)
    assert r["supported"] and r["p"] < 0.05 and r["lags"] == 6
    w = N.h2_test(weak)
    assert w["p"] < 0.05 and not w["supported"]
    assert not N.h2_test(none)["supported"]


def test_shifted_null_is_size_correct_and_breaks_cointegration():
    """Time-shifted partner null (primary H2 null): ~5% on random walks, and cointegrated pairs lose their link."""
    days = pd.bdate_range("2015-01-02", "2023-12-29")
    month = pd.Period("2023-07", "M")
    rng = np.random.default_rng(5)
    n, k = len(days), 400
    X = np.cumsum(rng.standard_normal((n, k)) * 0.015, 0) + np.log(40)
    for j in range(1, 40, 2):                                     # 20 strongly cointegrated pairs
        u = np.zeros(n)
        for i in range(1, n):
            u[i] = 0.5 * u[i - 1] + rng.standard_normal() * 0.01
        X[:, j] = X[:, j - 1] + u
    logp = pd.DataFrame(X, index=days)
    ia, ib = np.arange(40, k, 2), np.arange(41, k, 2)
    rw = pd.DataFrame({"iy": ia, "ix": ib})
    f = N.shifted_null_flags(logp, rw, month, 6, 100)
    assert len(f) == len(rw) and abs(f.mean() - 0.05) < 0.035, f.mean()
    co = pd.DataFrame({"iy": np.arange(1, 40, 2), "ix": np.arange(0, 40, 2)})
    assert N.shifted_null_flags(logp, co, month, 6, 100).mean() < 0.3      # real pairs pass ~100% unshifted
    r = N.month_null_rate(logp, rw, month, "shifted")
    assert r["n"] == len(rw)
