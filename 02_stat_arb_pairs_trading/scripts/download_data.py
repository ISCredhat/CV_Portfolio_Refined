#!/usr/bin/env python3
"""Download the raw data into data/raw/.

    python scripts/download_data.py --databento-key YOUR_KEY     # minute bars (paid) + free data
    python scripts/download_data.py --skip-minute                 # only the free daily data

1. Minute bars: Databento Nasdaq TotalView-ITCH (`XNAS.ITCH`), schema `ohlcv-1m`, one CSV per ticker
   in data/raw/minute_bars/ with columns
   ts_event, rtype, publisher_id, instrument_id, open, high, low, close, volume, symbol.
   Databento charges for historical data (check the cost with the `metadata.get_cost` call first).
2. Daily prices and split history from Yahoo Finance (yfinance): data/raw/yahoo_daily.csv.gz and
   data/raw/splits.csv. Used for split adjustment, a cross-check of the minute closes, and SPY.
3. Fama-French 5 factors + momentum (daily) from Kenneth French's data library:
   data/raw/factors_daily.csv.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config as C  # noqa: E402

FRENCH = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------ minute bars
def download_minute(key: str, tickers: list[str], start: str, end: str) -> None:
    import databento as db
    out = C.MINUTE_DIR
    out.mkdir(parents=True, exist_ok=True)
    client = db.Historical(key)
    for t in tickers:
        path = out / f"{t}.csv"
        if path.exists():
            continue
        # request whole UTC days so that the full regular session (13:30/14:30-21:00 UTC) is included
        data = client.timeseries.get_range(dataset="XNAS.ITCH", schema="ohlcv-1m", symbols=[t],
                                           stype_in="raw_symbol", start=start, end=end)
        df = data.to_df().reset_index()
        cols = ["ts_event", "rtype", "publisher_id", "instrument_id", "open", "high", "low", "close", "volume", "symbol"]
        df[[c for c in cols if c in df.columns]].to_csv(path, index=False)
        log(f"  {t}: {len(df):,} bars")


# ------------------------------------------------------------------------------ Yahoo
def download_yahoo(tickers: list[str], start: str, end: str) -> None:
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i + 20]
        data = yf.download(batch, start=start, end=end, auto_adjust=False, actions=True, group_by="ticker",
                           threads=False, progress=False)
        for t in batch:
            try:
                sub = data[t].dropna(subset=["Close"])
            except KeyError:
                log(f"  no Yahoo data for {t}")
                continue
            sub = sub.rename(columns={"Open": "open", "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
                                      "Stock Splits": "splits"})
            sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
            sub.index.name = "date"
            sub = sub[[c for c in ["open", "close", "adj_close", "volume", "splits"] if c in sub]].assign(ticker=t)
            frames.append(sub.reset_index())
        time.sleep(2)
    y = pd.concat(frames, ignore_index=True)
    y.to_csv(C.DATA_RAW / "yahoo_daily.csv.gz", index=False, float_format="%.6g")
    sp = y[(y["splits"] > 0) & (y["splits"] != 1)][["ticker", "date", "splits"]].rename(columns={"splits": "ratio"})
    sp.assign(source="yahoo").sort_values(["date", "ticker"]).to_csv(C.DATA_RAW / "splits.csv", index=False)
    log(f"Yahoo: {y['ticker'].nunique()} tickers, {len(sp)} split events")


# ------------------------------------------------------------------------------ factors
def _parse_french(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    head = next(i for i, ln in enumerate(lines) if re.match(r"^\s*,\s*\w", ln))   # e.g. ",Mkt-RF,SMB,..."
    cols = [c.strip() for c in lines[head].split(",")[1:]]
    rows = []
    for ln in lines[head + 1:]:
        if not re.match(r"^\s*\d{8}\s*,", ln):
            break
        p = [x.strip() for x in ln.split(",")]
        rows.append([p[0]] + p[1:len(cols) + 1])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").apply(pd.to_numeric, errors="coerce") / 100.0


def download_factors() -> None:
    import requests
    parts = []
    for name, url in FRENCH.items():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            text = zf.read([n for n in zf.namelist() if n.lower().endswith(".csv")][0]).decode("latin-1")
        parts.append(_parse_french(text))
    f = parts[0].join(parts[1], how="inner")
    f.columns = [c.lower().replace("-", "_") for c in f.columns]
    f = f.rename(columns={"umd": "mom"}).loc["2020-01-01":]
    f.to_csv(C.DATA_RAW / "factors_daily.csv", float_format="%.6g")
    log(f"Factors: {f.index.min().date()} to {f.index.max().date()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--databento-key", default=os.environ.get("DATABENTO_API_KEY"))
    ap.add_argument("--skip-minute", action="store_true")
    ap.add_argument("--start", default=C.DATA_START)
    ap.add_argument("--end", default=C.DATA_END)
    a = ap.parse_args()
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if not a.skip_minute:
        if not a.databento_key:
            sys.exit("Minute bars need a Databento API key (--databento-key or DATABENTO_API_KEY), or use --skip-minute")
        download_minute(a.databento_key, C.ALL_TICKERS, a.start, a.end)
    download_yahoo(C.ALL_TICKERS + ["SPY"], "2020-10-01", "2026-01-01")
    download_factors()


if __name__ == "__main__":
    main()
