"""Tests for the parts of the pipeline where a silent bug would fake alpha.

Run:  python -m pytest tests        (or, without pytest:  python tests/test_pipeline.py)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from insider_alpha import backtest as B  # noqa: E402
from insider_alpha import config as C  # noqa: E402
from insider_alpha import events as E  # noqa: E402
from insider_alpha import features as F  # noqa: E402
from insider_alpha import load as L  # noqa: E402
from insider_alpha import model as M  # noqa: E402
from insider_alpha import returns as R  # noqa: E402


# ------------------------------------------------------------------ data hygiene
def test_unadjusted_close_undoes_split():
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"X": [50.0, 50.0, 51.0, 52.0]}, index=idx)      # Yahoo: split-adjusted
    splits = pd.DataFrame({"X": [0.0, 0.0, 2.0, 0.0]}, index=idx)         # 2:1 on day 3
    raw = L.unadjusted_close(close, splits)["X"].to_numpy()
    assert np.allclose(raw, [100.0, 100.0, 51.0, 52.0])                   # as traded


def test_ticker_validation_rejects_mismatched_price():
    idx = pd.bdate_range("2024-01-01", periods=5)
    unadj = pd.DataFrame({"OLD": [10.0] * 5, "NEW": [100.0] * 5}, index=idx)
    ev = pd.DataFrame({"filed_symbol": ["OLD", "OLD"], "issuer_cik": ["1", "2"],
                       "last_trans_date": [idx[2], idx[2]], "price": [100.5, 10.2]})
    out = E.map_to_prices(ev, {}, {"1": "NEW"}, unadj)
    # event 0: filed symbol price is 10x off, current ticker matches -> NEW
    assert out.loc[0, "ticker"] == "NEW" and out.loc[0, "price_validated"]
    # event 1: filed symbol matches
    assert out.loc[1, "ticker"] == "OLD" and out.loc[1, "price_validated"]
    ev2 = ev.assign(price=[500.0, 500.0], issuer_cik=["9", "9"])
    assert not E.map_to_prices(ev2, {}, {}, unadj)["price_validated"].any()


def test_entry_is_strictly_after_filing_date():
    idx = pd.bdate_range("2024-01-01", periods=10)
    px = {"close": pd.DataFrame({"A": np.arange(10.0) + 10}, index=idx),
          "open": pd.DataFrame({"A": np.arange(10.0) + 10}, index=idx)}
    ev = pd.DataFrame({"ticker": ["A", "A"], "filing_date": [idx[3], idx[3] + pd.Timedelta(hours=0)]})
    out = E.add_entry(ev, px)
    assert (out["entry_date"] > out["filing_date"]).all()
    assert (out["entry_row"] == 4).all()


# ------------------------------------------------------------------ returns
def test_value_paths_and_bhar():
    adj_close = np.array([[10.0], [11.0], [12.1], [np.nan], [np.nan]])
    adj_open = np.array([[10.0], [10.0], [11.0], [np.nan], [np.nan]])
    p = R.value_paths(np.array([1]), np.array([0]), adj_close, adj_open, horizon=3)
    # entry at open 10 on row 1: closes 11, 12.1, then delisted -> last price carried
    assert np.allclose(p, [[1.1, 1.21, 1.21]])
    p_end = R.value_paths(np.array([3]), np.array([0]), adj_close, adj_open * 0 + 10, horizon=3)
    assert np.isnan(p_end[0, 2])  # beyond the data -> not realised


def test_calendar_portfolio_costs():
    T = 6
    idx = pd.bdate_range("2024-01-01", periods=T)
    adj_close = np.array([[10.0, 20.0]] * T)
    adj_close[2:, 0] = 11.0                   # stock 0 jumps +10% on day 2 (close-to-close)
    adj_open = adj_close.copy()
    adj_open[1, 0] = 10.0
    sel = pd.DataFrame({"entry_row": [1, 1], "col": [0, 1], "adv": [1e9, 1e9]})  # 5 bps one-way
    pf = B.calendar_portfolio(sel, adj_close, adj_open, idx, hold=3, cost_mult=1.0)
    assert np.isclose(pf.loc[idx[1], "gross"], 0.0)
    assert np.isclose(pf.loc[idx[1], "net"], -0.0005)          # entry cost on both legs
    assert np.isclose(pf.loc[idx[2], "gross"], 0.05)           # half the book up 10%
    # exit day (row 3): stock0 weight 1.1, stock1 weight 1.0 -> cost 5bp on each exit value
    assert np.isclose(pf.loc[idx[3], "net"], -0.0005)


# ------------------------------------------------------------------ point-in-time
def test_track_record_uses_only_realised_outcomes():
    d = pd.to_datetime(["2020-01-10", "2020-02-10", "2020-03-10", "2020-06-10"])
    ev = pd.DataFrame({"owner_cik": ["o"] * 4, "filing_date": d,
                       "bhar_63": [0.10, -0.20, 0.30, np.nan],
                       "realise_date": pd.to_datetime(["2020-04-15", "2020-05-15", "2020-06-15", pd.NaT])})
    out = F.attach_track_record(ev, "bhar_63")
    # at the 4th filing (2020-06-10) only the first two outcomes had been realised
    assert out.loc[3, "tr_n"] == 2
    assert np.isclose(out.loc[3, "tr_mean"], -0.05)
    assert out.loc[0, "tr_n"] == 0 and out.loc[2, "tr_n"] == 0


def test_walk_forward_never_trains_on_unrealised_labels():
    rng = np.random.default_rng(0)
    n = 4000
    fd = pd.Timestamp("2013-01-01") + pd.to_timedelta(rng.integers(0, 365 * 6, n), unit="D")
    ev = pd.DataFrame({"filing_date": fd, "x": rng.normal(size=n)})
    ev["realise_date"] = ev["filing_date"] + pd.Timedelta(days=95)
    ev["y"] = rng.normal(size=n)
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(X.index)
            return self

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.5)])

    orig = M.make_model
    M.make_model = lambda name: Spy()
    try:
        scores, diags = M.walk_forward(ev.sort_values("filing_date").reset_index(drop=True), ["x"], "y",
                                       ["gbm"], start="2016-01-01", verbose=False)
    finally:
        M.make_model = orig
    evs = ev.sort_values("filing_date").reset_index(drop=True)
    for idx, dg in zip(seen, diags):
        assert (evs.loc[idx, "realise_date"] < dg["train_end"]).all()
    assert scores.loc[evs["filing_date"] < "2016-01-01", "score_gbm"].isna().all()


# ------------------------------------------------------------------ end to end
def test_pipeline_recovers_planted_effect():
    """On synthetic data with a known effect, the full pipeline must find it."""
    from tests.synthetic import make_synthetic_raw
    from insider_alpha import pipeline
    import pickle
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_synthetic_raw(tmp / "raw", n_tickers=60, n_insiders=300)
        pipeline.build(tmp / "raw", tmp / "proc", tmp / "res")
        b = pickle.load(open(tmp / "proc" / "events.pkl", "rb"))["buys"]
        assert (b["entry_date"] > b["filing_date"]).all()
        g = b.groupby("role")["bhar_63"].mean()
        assert g["CEO/CFO"] > g["Director"]
        assert b.loc[b["is_cluster"], "bhar_63"].mean() > b.loc[~b["is_cluster"], "bhar_63"].mean()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
