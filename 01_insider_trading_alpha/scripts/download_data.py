#!/usr/bin/env python3
"""
Download all raw data for the Insider Trading Alpha project (run once, locally).

    cd 01_insider_trading_alpha
    pip install pandas requests yfinance
    python scripts/download_data.py --email you@example.com

The SEC asks every automated client to identify itself with a contact e-mail in
the User-Agent header, so --email is required (it is only sent to sec.gov).

Steps (the script is RESUMABLE - re-running skips anything already on disk):
  1. SEC EDGAR "Insider Transactions Data Sets" (Forms 3/4/5): one zip per
     quarter, parsed down to open-market purchases (code P) and sales (code S).
  2. SEC company_tickers.json (CIK -> current ticker), used to recover prices for
     companies that changed ticker after the filing.
  3. Daily prices from Yahoo Finance (yfinance): open, raw close, adjusted close,
     volume, splits and dividends for every ticker found in step 1, plus
     benchmark ETFs.
  4. Fama-French 5 factors + momentum, daily, from Kenneth French's data library.

Everything is written to data/raw/. Expect ~30-90 minutes and ~1-2 GB on disk
(the quarterly SEC zips can be deleted afterwards with --delete-zips).
Keep the computer awake while it runs. If it stops, just run it again.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEC_DIR = RAW / "sec_zips"
SEC_PARSED = RAW / "sec_parsed"
PRICE_DIR = RAW / "prices"
FACTOR_DIR = RAW / "factors"

SEC_URL_PATTERNS = [
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip",
    "https://www.sec.gov/files/dera/data/form-345/{q}_form345.zip",
]
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FF_URLS = {
    "ff5_daily.zip": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_daily.zip": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}
BENCHMARKS = ["SPY", "IWM", "IWC", "IJR", "MDY", "QQQ", "XLB", "XLE", "XLF", "XLI",
              "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]

TRANS_COLS = ["ACCESSION_NUMBER", "SECURITY_TITLE", "TRANS_DATE", "TRANS_FORM_TYPE",
              "TRANS_CODE", "EQUITY_SWAP_INVOLVED", "TRANS_TIMELINESS", "TRANS_SHARES",
              "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
              "DIRECT_INDIRECT_OWNERSHIP", "NATURE_OF_OWNERSHIP"]
SUB_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE",
            "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL", "AFF10B5ONE"]
OWNER_COLS = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
              "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------- SEC
def sec_session(email: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": f"InsiderAlphaResearch {email}",
                      "Accept-Encoding": "gzip, deflate"})
    return s


def quarters(start_year: int, end_year: int) -> list[str]:
    return [f"{y}q{q}" for y in range(start_year, end_year + 1) for q in range(1, 5)]


def download_sec_quarter(sess: requests.Session, q: str) -> Path | None:
    out = SEC_DIR / f"{q}_form345.zip"
    if out.exists() and out.stat().st_size > 1000:
        return out
    for pattern in SEC_URL_PATTERNS:
        url = pattern.format(q=q)
        for attempt in range(4):
            try:
                r = sess.get(url, timeout=180)
            except requests.RequestException as e:
                log(f"  {q}: network error ({e}); retrying")
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200 and r.content[:2] == b"PK":
                out.write_bytes(r.content)
                time.sleep(0.5)  # stay far below SEC's 10 requests/second limit
                return out
            if r.status_code == 403:
                sys.exit("SEC returned 403 Forbidden. Check that --email is a real address "
                         "and try again in 10 minutes (SEC rate limiting).")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10 * (attempt + 1))
                continue
            break  # 404 etc -> try next URL pattern
    return None


def _read_tsv(zf: zipfile.ZipFile, name: str, keep: list[str]) -> pd.DataFrame:
    member = next((n for n in zf.namelist() if n.upper().endswith(name.upper())), None)
    if member is None:
        raise FileNotFoundError(f"{name} not found in {zf.filename}: {zf.namelist()}")
    with zf.open(member) as fh:
        df = pd.read_csv(io.TextIOWrapper(fh, encoding="latin-1"), sep="\t", dtype=str,
                         quoting=csv.QUOTE_NONE, on_bad_lines="skip", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df[[c for c in keep if c in df.columns]]


def parse_sec_quarter(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        tr = _read_tsv(zf, "NONDERIV_TRANS.tsv", TRANS_COLS)
        tr = tr[tr["TRANS_CODE"].str.strip().isin(["P", "S"])]
        sub = _read_tsv(zf, "SUBMISSION.tsv", SUB_COLS)
        own = _read_tsv(zf, "REPORTINGOWNER.tsv", OWNER_COLS)
    own = own[own["ACCESSION_NUMBER"].isin(tr["ACCESSION_NUMBER"])]
    own_agg = own.groupby("ACCESSION_NUMBER").agg(
        RPTOWNERCIK=("RPTOWNERCIK", "first"),
        RPTOWNERNAME=("RPTOWNERNAME", "first"),
        RPTOWNER_RELATIONSHIP=("RPTOWNER_RELATIONSHIP",
                               lambda s: "|".join(sorted(set(",".join(s.dropna()).split(","))))),
        RPTOWNER_TITLE=("RPTOWNER_TITLE", lambda s: next((x for x in s if isinstance(x, str) and x.strip()), "")),
        N_OWNERS=("RPTOWNERCIK", "nunique"),
    ).reset_index()
    df = tr.merge(sub, on="ACCESSION_NUMBER", how="inner").merge(own_agg, on="ACCESSION_NUMBER", how="left")
    df.insert(0, "QUARTER", zip_path.name[:6])
    return df


def run_sec(email: str, start_year: int, end_year: int) -> pd.DataFrame:
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    SEC_PARSED.mkdir(parents=True, exist_ok=True)
    sess = sec_session(email)
    frames, missing = [], []
    for q in quarters(start_year, end_year):
        parsed = SEC_PARSED / f"form4_ps_{q}.csv.gz"
        if parsed.exists():
            frames.append(pd.read_csv(parsed, dtype=str))
            continue
        zp = download_sec_quarter(sess, q)
        if zp is None:
            missing.append(q)
            log(f"SEC {q}: not available (not yet published?)")
            continue
        df = parse_sec_quarter(zp)
        df.to_csv(parsed, index=False, compression="gzip")
        n_p = (df["TRANS_CODE"] == "P").sum()
        log(f"SEC {q}: {len(df):>7,} P/S transaction lines ({n_p:,} purchases)")
        frames.append(df)
    if not frames:
        sys.exit("No SEC data could be downloaded - see messages above.")
    allq = pd.concat(frames, ignore_index=True)
    allq.to_csv(RAW / "form4_ps_all.csv.gz", index=False, compression="gzip")
    log(f"SEC total: {len(allq):,} lines, quarters missing: {missing or 'none'}")

    # CIK -> current ticker map
    out = RAW / "sec_company_tickers.json"
    if not out.exists():
        r = sess.get(SEC_TICKERS_URL, timeout=60)
        if r.status_code == 200:
            out.write_bytes(r.content)
        else:
            log(f"company_tickers.json: HTTP {r.status_code} (skipped)")
    return allq


# ------------------------------------------------------------------ ticker list
def clean_symbol(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip().upper().replace("$", "")
    for sep in [",", ";", "/", " ", "|"]:
        s = s.split(sep)[0]
    s = s.replace(".", "-")
    if not s or s in {"NONE", "NA", "N/A", "NULL", "-", "0"} or len(s) > 8:
        return None
    return s


def build_universe(allq: pd.DataFrame) -> pd.DataFrame:
    allq = allq.copy()
    allq["SYM"] = allq["ISSUERTRADINGSYMBOL"].map(clean_symbol)
    cik_map = {}
    p = RAW / "sec_company_tickers.json"
    if p.exists():
        for rec in json.loads(p.read_text()).values():
            cik_map.setdefault(str(int(rec["cik_str"])), rec["ticker"].upper().replace(".", "-"))
    allq["CUR"] = allq["ISSUERCIK"].map(lambda c: cik_map.get(str(int(c))) if str(c).strip().isdigit() else None)
    rows = []
    for col, src in [("SYM", "filed"), ("CUR", "current_cik")]:
        g = allq.dropna(subset=[col]).groupby(col)["TRANS_CODE"].agg(
            n_P=lambda s: (s == "P").sum(), n_S=lambda s: (s == "S").sum())
        g["source"] = src
        rows.append(g.reset_index().rename(columns={col: "ticker"}))
    uni = (pd.concat(rows).groupby("ticker")
           .agg(n_P=("n_P", "max"), n_S=("n_S", "max"), source=("source", "|".join)).reset_index())
    bench = pd.DataFrame({"ticker": BENCHMARKS, "n_P": 10**9, "n_S": 0, "source": "benchmark"})
    uni = pd.concat([bench, uni[~uni["ticker"].isin(BENCHMARKS)]], ignore_index=True)
    # purchases first: if the run is interrupted, the most important prices exist
    uni = uni.sort_values(["n_P", "n_S"], ascending=False).reset_index(drop=True)
    uni.to_csv(RAW / "ticker_universe.csv", index=False)
    log(f"Ticker universe: {len(uni):,} symbols ({(uni.n_P > 0).sum():,} with purchases)")
    return uni


# ---------------------------------------------------------------------- prices
FIELD_MAP = {"Open": "open", "Close": "close", "Adj Close": "adj_close",
             "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits"}


def _normalise(sub: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub = sub.rename(columns=FIELD_MAP)
    keep = [c for c in FIELD_MAP.values() if c in sub.columns]
    sub = sub[keep].dropna(subset=["close"]) if "close" in keep else sub.iloc[0:0]
    if sub.empty:
        return sub
    idx = pd.to_datetime(sub.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    sub = sub.copy()
    sub.index = idx.normalize()
    sub.index.name = "date"
    sub.insert(0, "ticker", ticker)
    return sub.reset_index()


def yf_to_long(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if data is None or len(data) == 0:
        return pd.DataFrame()
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = set(data.columns.get_level_values(0))
        by_ticker = bool(set(tickers) & lvl0)
        for t in tickers:
            try:
                sub = data[t] if by_ticker else data.xs(t, axis=1, level=1)
            except KeyError:
                continue
            frames.append(_normalise(sub, t))
    else:
        frames.append(_normalise(data, tickers[0]))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Always-available tickers added to every batch. Yahoo throttling is often *partial* (some
# requests succeed, most fail), so one canary is not enough: all of them must come back.
CANARIES = ["SPY", "AAPL", "MSFT"]


def _manifest() -> pd.DataFrame:
    """Status of every ticker attempted so far.

    v1 (first run) and v2 (single-canary run) were partly rate-limited, so only their
    successes are trusted. v3 (gentle, multi-canary run) failures are final: the ticker
    really has no data on Yahoo (mostly delisted companies).
    """
    parts = []
    for name, trust_fail in [("prices_manifest.csv", False), ("prices_manifest_v2.csv", False),
                             ("prices_manifest_v3.csv", True)]:
        f = RAW / name
        if f.exists():
            m = pd.read_csv(f, usecols=[0, 1], names=["ticker", "status"], header=0)
            m["final_fail"] = trust_fail & (m["status"] != "ok")
            parts.append(m)
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=["ticker", "status", "final_fail"]))


def _done_tickers() -> set:
    m = _manifest()
    if m.empty:
        return set()
    return set(m.loc[m["status"] == "ok", "ticker"]) | set(m.loc[m["final_fail"].astype(bool), "ticker"])


def _listed_tickers() -> set:
    p = RAW / "sec_company_tickers.json"
    if not p.exists():
        return set()
    return {r["ticker"].upper().replace(".", "-") for r in json.loads(p.read_text()).values()}


def _fetch(yf, batch: list[str], start: str, end: str) -> tuple[pd.DataFrame, bool]:
    extra = [c for c in CANARIES if c not in batch]
    tickers = batch + extra
    data = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=False, actions=True,
                       group_by="ticker", threads=False, progress=False)   # sequential = gentle
    long = yf_to_long(data, tickers)
    got = set(long["ticker"]) if len(long) else set()
    healthy = all(c in got for c in CANARIES)
    if len(long):
        long = long[~long["ticker"].isin(extra)]
    return long, healthy


def run_prices(uni: pd.DataFrame, start: str, batch_size: int, max_tickers: int | None,
               listed_only: bool = False) -> None:
    import logging
    import yfinance as yf  # imported here so the SEC step works without it
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # silence 'possibly delisted' spam

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_p = RAW / "prices_manifest_v3.csv"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    done = _done_tickers()
    listed = _listed_tickers()
    todo = [t for t in uni["ticker"] if t not in done]
    # currently listed companies first: they almost always have data
    todo = [t for t in todo if t in listed] + ([] if listed_only else [t for t in todo if t not in listed])
    if max_tickers:
        todo = todo[:max_tickers]
    log(f"Prices: {len(done):,} tickers finished, {len(todo):,} to download "
        f"({sum(t in listed for t in todo):,} currently listed first)")
    batch_no = max([int(f.name.split("_")[1].split(".")[0]) for f in PRICE_DIR.glob("batch_*.csv.gz")] or [0])
    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        n_listed = sum(t in listed for t in batch)
        long, healthy, wait = pd.DataFrame(), False, 60
        for attempt in range(8):
            try:
                long, healthy = _fetch(yf, batch, start, end)
            except Exception as e:  # network hiccups
                log(f"  batch error: {type(e).__name__}: {str(e)[:120]}")
                healthy = False
            got_listed = sum(t in listed for t in (set(long["ticker"]) if len(long) else set()))
            if healthy and not (n_listed >= 3 and got_listed == 0):
                break
            log(f"  Yahoo seems to be rate-limiting: pausing {wait}s ...")
            time.sleep(wait)
            wait = min(wait * 2, 900)
        else:
            sys.exit("Yahoo is still rate-limiting after ~1 hour of pauses. Progress is saved - "
                     "run the same command again later and it will continue.")
        got = set(long["ticker"]) if len(long) else set()
        if len(long):
            batch_no += 1
            long.to_csv(PRICE_DIR / f"batch_{batch_no:04d}.csv.gz", index=False,
                        compression="gzip", float_format="%.6g")
        status = pd.DataFrame({"ticker": batch, "status": ["ok" if t in got else "no_data" for t in batch]})
        status.to_csv(manifest_p, mode="a", header=not manifest_p.exists(), index=False)
        n_done = i + len(batch)
        rate = n_done / max(time.time() - t0, 1)
        eta = (len(todo) - n_done) / max(rate, 1e-9) / 60
        log(f"Prices {n_done:,}/{len(todo):,}: batch got {len(got)}/{len(batch)} tickers, ~{eta:.0f} min left")
        time.sleep(2.0)
    m = _manifest()
    n_ok = m.loc[m["status"] == "ok", "ticker"].nunique()
    log(f"Prices finished: {n_ok:,} tickers have data (the rest are mostly delisted - this is expected)")


# --------------------------------------------------------------------- factors
def run_factors() -> None:
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in FF_URLS.items():
        out = FACTOR_DIR / fname
        if out.exists():
            continue
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            out.write_bytes(r.content)
            log(f"Factors: saved {fname}")
        else:
            log(f"Factors: HTTP {r.status_code} for {url}")


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True, help="contact e-mail for the SEC User-Agent header")
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=pd.Timestamp.today().year)
    ap.add_argument("--price-start", default="2009-01-01")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=None, help="for a quick test run")
    ap.add_argument("--listed-only", action="store_true",
                    help="only (re)try tickers of currently listed companies (quick top-up, ~20 min)")
    ap.add_argument("--skip-sec", action="store_true")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--skip-factors", action="store_true")
    ap.add_argument("--delete-zips", action="store_true", help="remove SEC zips once parsed")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    log(f"Writing to {RAW}")
    if not args.skip_factors:
        run_factors()
    if args.skip_sec:
        allq = pd.read_csv(RAW / "form4_ps_all.csv.gz", dtype=str)
    else:
        allq = run_sec(args.email, args.start_year, args.end_year)
    uni = build_universe(allq)
    if not args.skip_prices:
        run_prices(uni, args.price_start, args.batch_size, args.max_tickers, args.listed_only)
    if args.delete_zips:
        for z in SEC_DIR.glob("*.zip"):
            z.unlink()
    log("Done. Raw data is in data/raw/. Next: python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
