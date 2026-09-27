"""Stage 1: raw downloads -> clean, validated, point-in-time event tables."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import events as E
from . import features as F
from . import load as L
from . import returns as R


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def clean_price_spikes(adj: pd.DataFrame, up: float = 1.5, down: float = -0.6) -> tuple[pd.DataFrame, int]:
    """Remove one-day spikes that fully reverse (classic bad ticks in free data):
    a +150% day followed by a -60% day (or the mirror image) -> the spike close is NaN."""
    a = adj.to_numpy(dtype="float64", copy=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a[1:] / a[:-1] - 1
    r_next = np.vstack([r[1:], np.full((1, a.shape[1]), np.nan)])
    spike = ((r > up) & (r_next < down)) | ((r < down) & (r_next > up))
    n = int(np.nansum(spike))
    a[1:][spike] = np.nan
    return pd.DataFrame(a.astype("float32"), index=adj.index, columns=adj.columns), n


def coverage_report(stages: dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(stages).fillna(0).astype(int).sort_index()
    df.index = df.index.astype(str)
    df.loc["total"] = df.sum()
    return df


def build(raw_dir: Path = C.DATA_RAW, proc_dir: Path = C.DATA_PROC, results_dir: Path = C.RESULTS) -> None:
    raw_dir, proc_dir, results_dir = Path(raw_dir), Path(proc_dir), Path(results_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log("Loading SEC Form 4 lines ...")
    raw = L.load_sec(raw_dir / "form4_ps_all.csv.gz")
    tx = E.clean_transactions(raw)
    tx = tx[tx["filing_date"] >= C.TRACK_RECORD_START]
    log(f"  {len(raw):,} raw P/S lines -> {len(tx):,} clean lines")
    buys = E.aggregate_events(tx, "P")
    sells = E.aggregate_events(tx, "S")
    log(f"  events: {len(buys):,} purchases, {len(sells):,} sales")

    cik_map = L.load_cik_ticker_map(raw_dir / "sec_company_tickers.json")
    wanted = set(buys["filed_symbol"].dropna()) | set(sells["filed_symbol"].dropna())
    wanted |= {cik_map[c] for c in set(buys["issuer_cik"]) | set(sells["issuer_cik"]) if c in cik_map}
    wanted |= {C.MARKET, C.BENCHMARK}
    log(f"Loading prices for up to {len(wanted):,} tickers ...")
    px = L.load_prices(raw_dir / "prices", tickers=wanted)
    log(f"  price matrix: {px['close'].shape[0]:,} days x {px['close'].shape[1]:,} tickers")
    px["adj_close"], n_spikes = clean_price_spikes(px["adj_close"])
    px["close"] = px["close"].where(px["adj_close"].notna())
    log(f"  removed {n_spikes:,} reversing one-day price spikes")
    unadj = L.unadjusted_close(px["close"], px["splits"])

    out = {}
    for name, ev in [("P", buys), ("S", sells)]:
        ev = E.map_to_prices(ev, px, cik_map, unadj)
        ev = E.add_entry(ev, px)
        out[name] = ev
    buys, sells = out["P"], out["S"]
    # every filing, matched to prices or not, was public -> use all of them for activity features
    pool = {k: v[["issuer_cik", "owner_cik", "filing_date"]].copy() for k, v in [("P", buys), ("S", sells)]}

    # --- coverage report (how many events survive each step, by year) ------------
    y = buys["filing_date"].dt.year
    rep = coverage_report({
        "purchase_events": y.value_counts(),
        "has_ticker_candidate": y[buys["filed_symbol"].notna() | buys["current_symbol"].notna()].value_counts(),
        "has_price_data": y[buys["ticker"].notna()].value_counts(),
        "price_validated": y[buys["price_validated"]].value_counts(),
        "tradeable_entry": y[buys["price_validated"] & (buys["entry_row"] >= 0)].value_counts(),
    })
    rep.to_csv(results_dir / "data_coverage.csv")
    # why are events missing? mostly companies that no longer exist (Yahoo drops delisted stocks)
    missing = buys[~(buys["price_validated"] & (buys["entry_row"] >= 0))]
    miss_delisted = float((~missing["issuer_cik"].isin(set(cik_map))).mean()) if len(missing) else 0.0
    log("Coverage of purchase events:\n" + rep.tail(8).to_string())

    dates = px["close"].index
    adj_close = px["adj_close"].to_numpy()
    adj_open = R.adjusted_open(px)
    col_of = {t: i for i, t in enumerate(px["close"].columns)}
    bc, bo = adj_close[:, col_of[C.BENCHMARK]], adj_open[:, col_of[C.BENCHMARK]]
    mc, mo = adj_close[:, col_of[C.MARKET]], adj_open[:, col_of[C.MARKET]]

    H = max(C.PLOT_HORIZON, max(C.HORIZONS))
    paths_store = {}
    rng = np.random.default_rng(0)
    for name in ["P", "S"]:
        ev = out[name]
        ev = ev[ev["price_validated"] & (ev["entry_row"] >= 0)].reset_index(drop=True)
        tabs, keep_paths, keep_idx = [], [], []
        # sales can number >1M: process in chunks; keep abnormal paths for plots only for a
        # random subsample of sales (all purchases are kept)
        keep_mask = np.ones(len(ev), bool) if name == "P" else rng.random(len(ev)) < min(1.0, 150_000 / max(len(ev), 1))
        for s0 in range(0, len(ev), 100_000):
            sl = slice(s0, s0 + 100_000)
            er, col = ev["entry_row"].to_numpy()[sl], ev["col"].to_numpy()[sl]
            paths = R.value_paths(er, col, adj_close, adj_open, H)
            bpaths = R.benchmark_paths(er, bc, bo, H)
            mpaths = R.benchmark_paths(er, mc, mo, H)
            t1 = R.bhar_table(paths, bpaths, C.HORIZONS, "bhar")
            t2 = R.bhar_table(paths, mpaths, C.HORIZONS, "bhar_mkt").filter(like="bhar_mkt")
            tabs.append(pd.concat([t1, t2], axis=1))
            km = keep_mask[sl]
            keep_paths.append((paths - bpaths)[km].astype("float32"))
            keep_idx.append(np.arange(s0, s0 + len(er))[km])
            del paths, bpaths, mpaths
        ev = pd.concat([ev, pd.concat(tabs, ignore_index=True)], axis=1)
        rr = ev["entry_row"].to_numpy() + C.LABEL_HORIZON - 1
        ev["realise_date"] = pd.NaT
        ev.loc[rr < len(dates), "realise_date"] = dates[rr[rr < len(dates)]]
        ev["realise_date"] = ev["realise_date"].astype("datetime64[ns]")
        ev["path_row"] = -1
        idx = np.concatenate(keep_idx)
        ev.loc[idx, "path_row"] = np.arange(len(idx))
        paths_store[name] = np.concatenate(keep_paths)
        out[name] = ev
        log(f"  {name}: {len(ev):,} tradeable events with returns")

    # what the insider earned before outsiders could see the trade (trade date -> filing date),
    # and the overnight gap between the filing-day close and our entry at the next open
    for name in ["P", "S"]:
        ev = out[name]
        er, col = ev["entry_row"].to_numpy(), ev["col"].to_numpy()
        tr = np.clip(dates.searchsorted(ev["last_trans_date"].to_numpy(), side="right") - 1, 0, None)
        fr = er - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            stock_pre = adj_close[fr, col] / adj_close[tr, col] - 1
            bench_pre = bc[fr] / bc[tr] - 1
            gap = adj_open[er, col] / adj_close[fr, col] - 1 - (bo[er] / bc[fr] - 1)
        ev["abn_trade_to_filing"] = np.where(fr > tr, stock_pre - bench_pre, 0.0)
        ev["abn_overnight_gap"] = gap
        out[name] = ev

    out["P"] = F.attach_stock_features(out["P"], px)
    out["P"] = F.attach_market_features(out["P"], px)
    del unadj

    buys, sells = out["P"], out["S"]
    log("Point-in-time features ...")
    buys = F.attach_activity_features(buys, pool["P"], pool["S"])
    all_trades = pd.concat([pool["P"][["owner_cik", "filing_date"]], pool["S"][["owner_cik", "filing_date"]]])
    cls = F.routine_opportunistic(buys, all_trades)
    buys["trader_type"] = cls
    buys["opportunistic"] = cls.eq("opportunistic")
    buys["routine"] = cls.eq("routine")
    buys = F.attach_track_record(buys, f"bhar_{C.LABEL_HORIZON}")
    buys["investable"] = (buys["price_pre"] >= C.MIN_PRICE) & (buys["adv"] >= C.MIN_ADV)

    with open(proc_dir / "events.pkl", "wb") as fh:
        pickle.dump({"buys": buys, "sells": sells}, fh, protocol=4)
    np.savez_compressed(proc_dir / "abnormal_paths.npz", P=paths_store["P"], S=paths_store["S"])
    with open(proc_dir / "prices.pkl", "wb") as fh:
        pickle.dump({"dates": dates, "tickers": list(px["close"].columns), "adj_close": adj_close.astype("float32"),
                     "adj_open": adj_open.astype("float32")}, fh, protocol=4)
    factors = L.load_factors(raw_dir / "factors")
    factors.to_pickle(proc_dir / "factors.pkl")
    bench = pd.DataFrame({C.MARKET: px["adj_close"][C.MARKET], C.BENCHMARK: px["adj_close"][C.BENCHMARK]})
    bench.to_pickle(proc_dir / "benchmarks.pkl")
    import json
    (results_dir / "build_meta.json").write_text(json.dumps({
        "n_raw_lines": int(len(raw)), "n_clean_lines": int(len(tx)),
        "n_purchase_events_all": int(len(pool["P"])), "n_sale_events_all": int(len(pool["S"])),
        "n_tickers_priced": int(px["close"].shape[1]), "n_price_spikes_removed": int(n_spikes),
        "missing_share_delisted": miss_delisted,
        "price_start": str(dates[0].date()), "price_end": str(dates[-1].date()),
    }, indent=2))
    log(f"Saved processed data to {proc_dir}: {len(buys):,} purchase events, {len(sells):,} sale events")
