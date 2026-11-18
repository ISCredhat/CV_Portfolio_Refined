"""Offline tests for phase 2 builder B: taker dedup, signals, combine, plots."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pmdetect.markets.combine import combine, percentiles_by_type, rank_agreement
from pmdetect.markets.signals import compute_signals, taker_vs_all_check
from pmdetect.report.plots import plot_d_histogram, plot_market

NEWS = datetime(2025, 6, 10, 12, 0)


def _trade(cid, ts, wallet, label, side, price, size, tx):
    return {
        "trade_id": f"{tx}-{wallet}-{label}-{side}",
        "tx_hash": tx,
        "condition_id": cid,
        "token_id": f"{cid}-{label}",
        "wallet": wallet,
        "side": side,
        "outcome_label": label,
        "size": size,
        "price": price,
        "usd": size * price,
        "ts": ts,
    }


def _tape(cid, n_days=9, drift=False, rng=None):
    rng = rng or np.random.default_rng(1)
    rows, start = [], NEWS - timedelta(days=n_days)
    for h in range(0, n_days * 24 - 1):
        ts = start + timedelta(hours=h)
        frac = h / (n_days * 24)
        px = 0.3 + (0.6 * frac if drift else 0.0) + rng.normal(0, 0.005)
        if drift and (NEWS - ts) < timedelta(hours=30):
            w = "fresh_whale"
            rows.append(_trade(cid, ts, w, "Yes", "BUY", px, 800, f"{cid}-{h}"))
        else:
            rows.append(_trade(cid, ts, f"w{h % 7}", "Yes", "BUY" if h % 2 else "SELL", px, 50, f"{cid}-{h}"))
    return pl.DataFrame(rows)


def _markets(cids, info="unscheduled_event"):
    return pl.DataFrame(
        {
            "condition_id": cids,
            "news_ts": [NEWS] * len(cids),
            "outcome": ["YES"] * len(cids),
            "info_type": [info] * len(cids),
        }
    )


def _wallets():
    return pl.DataFrame(
        {
            "wallet": ["fresh_whale"] + [f"w{i}" for i in range(7)],
            "first_seen": [NEWS - timedelta(days=2)] + [NEWS - timedelta(days=300)] * 7,
            "n_markets": [1] + [20] * 7,
        }
    )


def test_taker_dedup_counts_fill_once():
    # a fill listed twice (taker + maker row, same tx) counts once on the taker tape
    taker = pl.DataFrame([_trade("c", NEWS, "t", "Yes", "BUY", 0.5, 10, "0xa")])
    allrows = pl.concat(
        [taker, pl.DataFrame([_trade("c", NEWS, "m", "Yes", "SELL", 0.5, 10, "0xa")])]
    )
    chk = taker_vs_all_check(allrows, taker).row(0, named=True)
    assert chk["n_all"] == 2 and chk["n_tx_all"] == 1 and chk["n_taker"] == 1
    assert chk["rows_eq_unique_tx"]
    assert abs(chk["size_ratio"] - 1.0) < 1e-9 and abs(chk["usd_ratio"] - 1.0) < 1e-9


def test_signals_planted_vs_flat_and_empty():
    cids = ["drift", "flat", "empty"]
    tapes = pl.concat([_tape("drift", drift=True), _tape("flat")])
    sig = compute_signals(_markets(cids), tapes, _wallets()).sort("condition_id")
    d, f, e = (sig.filter(pl.col("condition_id") == c).row(0, named=True) for c in cids)
    assert d["ofi_24h"] > 0.9 and f["ofi_24h"] < 0.5
    assert d["fresh_share"] > 0.5 and f["fresh_share"] == 0.0
    assert d["hhi"] > f["hhi"]
    assert d["vol_surge_z"] > f["vol_surge_z"]
    assert d["fresh_method"] == "wallets_first_seen"
    assert np.isnan(e["ofi_24h"]) and e["nan_reason"] == "no taker trades"


def test_signals_missing_news_ts_is_nan():
    m = _markets(["x"]).with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("news_ts"))
    sig = compute_signals(m, _tape("x"), _wallets())
    assert sig["nan_reason"][0] == "no news_ts" and np.isnan(sig["hhi"][0])


def test_fresh_fallback_to_store_when_wallet_unknown():
    tape = _tape("drift", drift=True)
    sig = compute_signals(_markets(["drift"]), tape, _wallets().head(0))
    assert sig["fresh_method"][0] == "store_fallback"
    assert sig["fresh_share"][0] > 0  # first trades seen in the store are within 7d of the trade


def _scores(n=30, plant="m0"):
    rng = np.random.default_rng(0)
    ids = [f"m{i}" for i in range(n)]
    D = rng.uniform(0, 0.05, n)
    D[0] = 0.4
    return pl.DataFrame(
        {"condition_id": ids, "D": D, "info_type": ["unscheduled_event"] * n, "question": ids}
    ), ids


def test_planted_drift_ranks_first_in_combine():
    sc, ids = _scores()
    rng = np.random.default_rng(2)
    sig = pl.DataFrame(
        {
            "condition_id": ids,
            "ofi_24h": rng.uniform(-0.5, 0.5, 30),
            "vol_surge_z": rng.normal(0, 1, 30),
            "max_jump": rng.uniform(0, 0.05, 30),
            "fresh_share": rng.uniform(0, 0.3, 30),
            "hhi": rng.uniform(0.05, 0.2, 30),
        }
    )
    sig = sig.with_columns(
        pl.when(pl.col("condition_id") == "m0").then(0.95).otherwise(pl.col("ofi_24h")).alias("ofi_24h"),
        pl.when(pl.col("condition_id") == "m0").then(9.0).otherwise(pl.col("vol_surge_z")).alias("vol_surge_z"),
        pl.when(pl.col("condition_id") == "m0").then(0.5).otherwise(pl.col("max_jump")).alias("max_jump"),
    )
    out = combine(sc, sig)
    top = out.sort("fisher_score", descending=True)["condition_id"][0]
    assert top == "m0"
    assert out.filter(pl.col("condition_id") == "m0")["flag_fisher"][0]
    assert out.filter(pl.col("flag_fisher")).height == 1  # ceil(1% of 30) = 1
    assert out.filter(pl.col("condition_id") == "m0")["main_driver"][0] in (
        "D", "ofi_24h", "vol_surge_z", "max_jump",
    )
    ag = rank_agreement(out)
    assert ag["spearman"]["fisher_score~rank_mean"] > 0.9


def test_percentiles_by_type_and_nan():
    df = pl.DataFrame(
        {
            "info_type": ["a", "a", "a", "b", "b"],
            "x": [1.0, 2.0, float("nan"), 5.0, 5.0],
        }
    )
    out = percentiles_by_type(df, ["x"])
    assert out["x_pct"].to_list() == [0.5, 1.0, None, 0.75, 0.75]  # ties share the average rank


def test_plots_with_stub_baseline(tmp_path):
    tape = _tape("drift", drift=True)
    taus = np.arange(1, 169, dtype=float)
    series = pl.DataFrame(
        {
            "condition_id": "drift",
            "t": [NEWS - timedelta(hours=float(t)) for t in taus],
            "tau_h": taus,
            "q": np.linspace(0.9, 0.3, 168),
        }
    )
    base = series.select("condition_id", "t", "tau_h").with_columns(
        pl.lit(0.2).alias("m10"), pl.lit(0.3).alias("m50"), pl.lit(0.4).alias("m90")
    )
    row = {"condition_id": "drift", "question": "Will it?", "news_ts": NEWS, "outcome": "YES",
           "D": 0.2, "info_type": "deadline"}  # fmt: skip
    p = plot_market(row, series, base, tape, tmp_path)
    assert p.exists() and p.stat().st_size > 1000
    h = plot_d_histogram(pl.DataFrame({"D": [0.0, 0.1, None, 0.3]}), tmp_path / "h.png", 0.3)
    assert h.exists()
