"""Loading raw downloads (SEC Form 4, Yahoo prices, Fama-French factors)."""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------- SEC
SEC_USECOLS = ["ACCESSION_NUMBER", "SECURITY_TITLE", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
               "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS", "DIRECT_INDIRECT_OWNERSHIP", "FILING_DATE",
               "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL", "RPTOWNERCIK",
               "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "N_OWNERS"]


def load_sec(path: Path | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    """Read the combined Form 4 file in chunks, keeping only the columns we use and only
    original Form 4 open-market purchases/sales (keeps memory low for ~3M lines)."""
    path = path or C.DATA_RAW / "form4_ps_all.csv.gz"
    parts = []
    for chunk in pd.read_csv(path, dtype=str, usecols=lambda c: c.upper() in SEC_USECOLS,
                             chunksize=chunksize, low_memory=False):
        chunk.columns = [c.upper() for c in chunk.columns]
        code = chunk["TRANS_CODE"].str.strip()
        keep = (chunk["DOCUMENT_TYPE"].str.strip() == "4") & code.isin(["P", "S"])
        parts.append(chunk[keep])
    df = pd.concat(parts, ignore_index=True)
    for c in ["SECURITY_TITLE", "ISSUERNAME", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "ISSUERTRADINGSYMBOL"]:
        df[c] = df[c].astype("category")
    return df


def load_cik_ticker_map(path: Path | None = None) -> dict[str, str]:
    """CIK (no leading zeros) -> current Yahoo-style ticker from SEC company_tickers.json."""
    path = path or C.DATA_RAW / "sec_company_tickers.json"
    if not Path(path).exists():
        return {}
    out: dict[str, str] = {}
    for rec in json.loads(Path(path).read_text()).values():
        out.setdefault(str(int(rec["cik_str"])), rec["ticker"].upper().replace(".", "-"))
    return out


# -------------------------------------------------------------------------- prices
PRICE_FIELDS = ["open", "close", "adj_close", "volume", "splits"]


def load_prices(price_dir: Path | None = None, tickers: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read the yfinance batch files into wide (date x ticker) float32 matrices.

    Returns dict with keys open, close (split-adjusted, as Yahoo reports it),
    adj_close (split + dividend adjusted), volume, splits.
    """
    price_dir = price_dir or C.DATA_RAW / "prices"
    wide_parts: dict[str, list[pd.DataFrame]] = {f: [] for f in PRICE_FIELDS}
    for f in sorted(Path(price_dir).glob("batch_*.csv.gz")):
        df = pd.read_csv(f, usecols=lambda c: c in ["date", "ticker"] + PRICE_FIELDS)
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(["date", "ticker"], keep="last")
        for field in PRICE_FIELDS:
            if field in df.columns:
                wide_parts[field].append(df.pivot(index="date", columns="ticker", values=field).astype("float32"))
    out = {}
    for field, parts in wide_parts.items():
        if not parts:
            continue
        w = pd.concat(parts, axis=1).sort_index()
        dup = w.columns[w.columns.duplicated()].unique()
        if len(dup):  # a ticker split across batches (e.g. re-downloaded): combine, last wins
            fixed = {t: w.loc[:, [t]].ffill(axis=1).iloc[:, -1] for t in dup}
            w = pd.concat([w.loc[:, ~w.columns.isin(dup)], pd.DataFrame(fixed)], axis=1)
        out[field] = w
    cols = out["close"].columns
    out = {k: v.reindex(columns=cols) for k, v in out.items()}
    # keep only trading days on which the market ETF traded (removes stray holiday rows)
    if C.MARKET in cols:
        cal = out["close"][C.MARKET].dropna().index
        out = {k: v.reindex(cal) for k, v in out.items()}
    return out


def unadjusted_close(close: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Undo Yahoo's split adjustment so prices are comparable with Form 4 prices.

    Yahoo's `close` is divided by every split that happens *after* each date, so the
    as-traded price is close(t) * prod(split ratios with ex-date > t).
    """
    s = splits.reindex_like(close).fillna(0.0).astype("float64")
    s = s.where(s > 0, 1.0)
    log_s = np.log(s)
    total = log_s.sum(axis=0)
    after = total - log_s.cumsum(axis=0)          # sum over ex-dates strictly after t
    return (close.astype("float64") * np.exp(after)).astype("float32")


# ------------------------------------------------------------------------- factors
def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV (text preamble, header row, YYYYMMDD rows in %)."""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip().startswith(",") and re.match(r"^\s*\d{8}\s*,", nxt):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("could not find the data header in the Fama-French file")
    cols = [c.strip() for c in lines[header_idx].split(",")[1:]]
    rows = []
    for line in lines[header_idx + 1:]:
        if not re.match(r"^\s*\d{8}\s*,", line):
            break
        parts = [p.strip() for p in line.split(",")]
        rows.append([parts[0]] + parts[1:len(cols) + 1])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce") / 100.0
    return df


def _read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        return zf.read(name).decode("latin-1")


def load_factors(factor_dir: Path | None = None) -> pd.DataFrame:
    """Daily FF5 + momentum + risk-free, in decimals, columns lower-case."""
    factor_dir = factor_dir or C.DATA_RAW / "factors"
    ff5 = _parse_french_csv(_read_zip_text(Path(factor_dir) / "ff5_daily.zip"))
    mom = _parse_french_csv(_read_zip_text(Path(factor_dir) / "mom_daily.zip"))
    df = ff5.join(mom, how="inner")
    df.columns = [c.lower().replace("-", "_").strip() for c in df.columns]
    df = df.rename(columns={"mom": "mom", "umd": "mom"})
    return df
