"""Batched Engle-Granger must reproduce the per-pair loop; simulated Johansen critical values."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import coint as K  # noqa: E402


def _random_pairs(n, P, seed):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.standard_normal((n, P)), 0) * 0.02
    Y = np.cumsum(rng.standard_normal((n, P)), 0) * 0.02
    for j in range(0, P, 3):                       # a third of the pairs are cointegrated
        u = np.zeros(n)
        for i in range(1, n):
            u[i] = 0.85 * u[i - 1] + rng.standard_normal() * 0.01
        Y[:, j] = 1.3 * X[:, j] + u
    return Y + 4.0, X + 3.0


def test_batch_matches_loop_to_1e8():
    for n, maxlag in [(126, 12), (100, 12), (252, 12), (60, 4), (126, None)]:
        Y, X = _random_pairs(n, 30, n)
        b = K.engle_granger_batch(Y, X, maxlag)
        for j in range(Y.shape[1]):
            r = K.engle_granger(Y[:, j], X[:, j], maxlag)
            assert abs(b["stat"][j] - r["stat"]) < 1e-8, (n, j)
            assert abs(b["pvalue"][j] - r["pvalue"]) < 1e-8
            assert abs(b["beta"][j] - r["beta"]) < 1e-8 and abs(b["alpha"][j] - r["alpha"]) < 1e-8
            assert b["lags"][j] == r["lags"]
            assert np.allclose(b["resid"][:, j], r["resid"], atol=1e-10)


def test_batch_flags_degenerate_and_rejects_nan():
    Y, X = _random_pairs(126, 4, 1)
    Y[:, 1] = X[:, 1]                               # identical series
    b = K.engle_granger_batch(Y, X, 12)
    assert b["degenerate"].tolist() == [False, True, False, False] and np.isnan(b["pvalue"][1])
    assert np.isfinite(b["pvalue"][[0, 2, 3]]).all()
    Y[3, 0] = np.nan
    try:
        K.engle_granger_batch(Y, X, 12)
        raise AssertionError("NaN input must raise")
    except ValueError:
        pass


def test_mackinnon_vec_and_half_life_batch():
    s = np.linspace(-6, 3, 50)
    for nv in (1, 2):
        assert np.allclose(K.mackinnon_p_vec(s, nv), [K.mackinnon_p(v, nv) for v in s], atol=1e-12)
    rng = np.random.default_rng(3)
    S = np.cumsum(rng.standard_normal((126, 5)), 0)
    S[:, 0] = np.convolve(rng.standard_normal(126), [1, 0.5], "same")
    hb = K.half_life_batch(S)
    assert np.allclose(hb, [K.half_life(S[:, j]) for j in range(5)], atol=1e-9, equal_nan=False)


def test_simulated_johansen_cv_cached_and_sane():
    assert set(K.JOHANSEN_CV_SIM) == {126, 252}
    c126, c252 = K.johansen_cv(126), K.johansen_cv(252)
    assert c126[0] < c126[1] < c126[2] and c252[0] < c252[1] < c252[2]
    assert c126[1] > c252[1] > 19.5                 # finite-sample CVs exceed the asymptotic 20.26 at small T
    assert K.johansen_cv(130) == c126 and K.johansen_cv(240) == c252
    fresh = K.johansen_cv_sim(126, reps=3000, seed=5)        # a fresh small simulation agrees loosely
    assert abs(fresh[1] - c126[1]) < 0.8
