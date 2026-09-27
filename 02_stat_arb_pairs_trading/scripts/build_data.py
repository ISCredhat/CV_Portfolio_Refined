#!/usr/bin/env python3
"""Build the clean minute panel from the raw Databento CSVs.

    python scripts/build_data.py

Input : data/raw/minute_bars/*.csv   (Databento XNAS.ITCH 1-minute OHLCV, UTC, unadjusted)
        data/raw/splits.csv          (split / spin-off history from Yahoo, scripts/download_data.py)
        data/raw/yahoo_daily.csv.gz  (daily closes, used only to cross-check the minute data)
Output: data/processed/minute_*.npy, calendar.pkl, tickers.json
        results/data_*.csv           (coverage, split audit, cross-check vs Yahoo)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config as C  # noqa: E402
from pairs import data as D  # noqa: E402

# Splits for tickers Yahoo does not cover, found by the overnight-jump audit below and
# checked by hand (the jump matches a round ratio and the company announced the split).
MANUAL_SPLITS = [
    {"ticker": "NVO", "date": "2023-09-20", "ratio": 2.0, "source": "detected + verified (2-for-1 ADS split)"},
]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _read(path: Path):
    df = D.read_minute_csv(path)
    return path.stem.upper(), df


def main() -> None:
    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    files = sorted(C.MINUTE_DIR.glob("*.csv"))
    log(f"Reading {len(files)} minute files ...")
    parsed = dict(Parallel(n_jobs=2)(delayed(_read)(f) for f in files))
    tickers = sorted(parsed)

    for t, (d0, why) in C.SYMBOL_REUSE.items():
        if t in parsed:
            df = parsed[t]
            log(f"  {t}: dropping {int((df['date'] < pd.Timestamp(d0)).sum()):,} bars before {d0} ({why})")
            parsed[t] = df[df["date"] >= pd.Timestamp(d0)].reset_index(drop=True)

    counts = pd.Series(np.concatenate([df["date"].unique() for df in parsed.values()])).value_counts()
    ends = D.session_ends([parsed[t] for t in C.CORE_TICKERS if t in parsed])
    cal = D.make_calendar(counts, ends)
    n, k = len(cal["ts"]), len(tickers)
    se = pd.Series(cal["session_end"]).value_counts()
    log(f"Calendar: {len(cal['days'])} trading days, {n:,} RTH minutes; {k} tickers")
    log("Session ends (minute of day -> days): " + ", ".join(f"{int(m) // 60:02d}:{int(m) % 60:02d} -> {c}"
                                                            for m, c in se.items()))

    panels = {f: np.full((n, k), np.nan, dtype="float32") for f in D.FIELDS}
    for j, t in enumerate(tickers):
        df = parsed.pop(t)
        pos = D.grid_positions(cal, df["date"].to_numpy(), df["minute"].to_numpy())
        ok = pos >= 0
        for f in D.FIELDS:
            panels[f][pos[ok], j] = df[f].to_numpy()[ok]
    panels["volume"] = np.nan_to_num(panels["volume"])

    # ---- split audit: every large overnight jump, matched against the split file ------------
    raw = D.MinutePanel(cal, tickers, panels["close"], panels["high"], panels["low"], panels["volume"])
    dly = raw.daily()
    last = pd.DataFrame(dly["last"], index=cal["days"], columns=tickers)
    first = pd.DataFrame(dly["first"], index=cal["days"], columns=tickers)
    jumps = D.overnight_jumps(last, first)
    splits = pd.read_csv(C.DATA_RAW / "splits.csv", parse_dates=["date"])
    splits = pd.concat([splits, pd.DataFrame(MANUAL_SPLITS).assign(date=lambda d: pd.to_datetime(d["date"]))],
                       ignore_index=True)
    splits = splits[(splits["date"] > cal["days"][0]) & (splits["date"] <= cal["days"][-1])]
    key = set(zip(splits["ticker"], splits["date"]))
    jumps["in_split_file"] = [(t, d) in key for t, d in zip(jumps["ticker"], jumps["date"])]
    jumps["group"] = jumps["ticker"].map(C.GROUP_OF).fillna("(not in a pair group)")
    jumps.sort_values(["ticker", "date"]).to_csv(C.RESULTS / "data_split_audit.csv", index=False)
    missed = splits[[(t, d) not in set(zip(jumps["ticker"], jumps["date"]))
                     for t, d in zip(splits["ticker"], splits["date"])]]
    log(f"Overnight jumps > 45%: {len(jumps)}; explained by a split: {int(jumps['in_split_file'].sum())}")
    log(f"Split events with no visible jump (small ratios / spin-offs): {len(missed)}")

    # ---- apply split factors (back-adjust: latest prices stay as traded) ---------------------
    f_day = D.split_factors(splits, tickers, cal["days"])
    f_min = f_day[cal["day_of_min"]].astype("float32")
    for fld in ["close", "high", "low"]:
        panels[fld] /= f_min
    panels["volume"] *= f_min
    del f_min

    for fld, arr in panels.items():
        np.save(C.DATA_PROC / f"minute_{fld}.npy", arr)
    np.save(C.DATA_PROC / "split_factor_day.npy", f_day.astype("float64"))   # raw price = adjusted * factor
    with open(C.DATA_PROC / "calendar.pkl", "wb") as fh:
        pickle.dump(cal, fh, protocol=4)
    (C.DATA_PROC / "tickers.json").write_text(json.dumps(tickers))
    splits.to_csv(C.DATA_PROC / "splits_applied.csv", index=False)

    # ---- data summary + cross-check against Yahoo daily closes -----------------------------
    adj = D.MinutePanel(cal, tickers, panels["close"], panels["high"], panels["low"], panels["volume"])
    d2 = adj.daily()
    traded_days = (d2["minutes_traded"] > 0)
    first_day = [cal["days"][np.argmax(traded_days[:, j])].date() for j in range(k)]
    summ = pd.DataFrame({
        "ticker": tickers,
        "group": [C.GROUP_OF.get(t, "") for t in tickers],
        "first_day": first_day,
        "days_traded": traded_days.sum(0),
        "avg_minutes_with_trade": np.where(traded_days.sum(0) > 0,
                                           d2["minutes_traded"].sum(0) / np.maximum(traded_days.sum(0), 1), 0),
        "median_nasdaq_dollar_vol_m": np.nanmedian(np.where(traded_days, d2["dollar_volume"], np.nan), 0) / 1e6,
        "median_price": np.nanmedian(d2["last"], 0),
    })
    yp = C.DATA_RAW / "yahoo_daily.csv.gz"
    if yp.exists():
        y = pd.read_csv(yp, parse_dates=["date"])
        y = y[y["ticker"].isin(tickers)]
        yc = y.pivot(index="date", columns="ticker", values="close")
        ys = y.pivot(index="date", columns="ticker", values="splits").fillna(0)
        ys = ys.where(ys > 0, 1.0)
        # Yahoo 'close' is split-adjusted to today: undo adjustments after our sample end
        after = np.exp(np.log(ys.loc[ys.index > cal["days"][-1]]).sum())
        yc = yc * after.reindex(yc.columns).fillna(1.0)
        ours = pd.DataFrame(d2["last"], index=cal["days"], columns=tickers)
        ours = ours[cal["session_end"] == C.RTH_END_MIN]          # compare full sessions only
        both = ours.reindex(index=yc.index.intersection(ours.index), columns=yc.columns)
        diff = (np.log(both / yc.loc[both.index])).abs() * 1e4
        chk = pd.DataFrame({"median_abs_diff_bps": diff.median(), "share_within_50bps": (diff < 50).mean(),
                            "n_days": diff.notna().sum()})
        summ = summ.merge(chk, left_on="ticker", right_index=True, how="left")
        log("Cross-check vs Yahoo closes: median |diff| = "
            f"{np.nanmedian(chk['median_abs_diff_bps']):.1f} bps; tickers with median > 25 bps: "
            f"{chk.index[chk['median_abs_diff_bps'] > 25].tolist()}")
    summ.to_csv(C.RESULTS / "data_summary.csv", index=False, float_format="%.4g")
    meta = {"n_tickers": k, "n_days": int(len(cal["days"])), "n_minutes": int(n),
            "first_day": str(cal["days"][0].date()), "last_day": str(cal["days"][-1].date()),
            "n_split_events": int(len(splits)), "n_overnight_jumps": int(len(jumps)),
            "n_jumps_explained": int(jumps["in_split_file"].sum()),
            "n_days_session_to_1500": int((cal["session_end"] == 900).sum()),
            "n_days_early_close": int((cal["session_end"] == C.HALF_DAY_END_MIN).sum()),
            "yahoo_median_abs_diff_bps": float(np.nanmedian(summ["median_abs_diff_bps"]))
            if "median_abs_diff_bps" in summ else None}
    (C.RESULTS / "data_meta.json").write_text(json.dumps(meta, indent=2))
    log("Done: " + json.dumps(meta))


if __name__ == "__main__":
    main()
