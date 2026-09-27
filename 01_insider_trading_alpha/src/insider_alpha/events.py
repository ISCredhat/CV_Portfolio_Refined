"""Turn raw Form 4 transaction lines into clean, point-in-time insider trading events.

One *event* = one insider (reporting owner) trading one issuer's stock in one Form 4
filing on one filing date. Several transaction lines (partial fills at different
prices) are aggregated into a single event.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C


def parse_sec_date(s: pd.Series) -> pd.Series:
    """SEC data sets use DD-MON-YYYY; fall back to flexible parsing for odd rows."""
    out = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        out[bad] = pd.to_datetime(s[bad], errors="coerce", format="mixed", dayfirst=True)
    out = out.where((out >= pd.Timestamp("1990-01-01")) & (out <= pd.Timestamp("2035-12-31")))  # typos
    return out.astype("datetime64[ns]")


def clean_symbol(s) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.strip().upper().replace("$", "")
    for sep in [",", ";", "/", " ", "|"]:
        s = s.split(sep)[0]
    s = s.replace(".", "-")
    if not s or s in {"NONE", "NA", "N/A", "NAN", "NULL", "-", "0"} or len(s) > 8:
        return None
    return s


# non-common securities and pooled vehicles, where "insider buying" means something else
_NOT_COMMON = re.compile(r"PREFERRED|PFD|NOTE|DEBENTURE|WARRANT|RIGHT|BOND|OPTION|PHANTOM|DEPOSITARY UNIT")
_POOLED = re.compile(r"\bFUND\b|\bETF\b|ACQUISITION CORP|ACQUISITION CO\b|ACQUISITION LTD|ACQUISITION INC|"
                     r"BLANK CHECK|\bSPAC\b|CAPITAL TRUST|MUNICIPAL")
_CEO = re.compile(r"\bCEO\b|CHIEF EXECUTIVE|PRINCIPAL EXECUTIVE")
_CFO = re.compile(r"\bCFO\b|CHIEF FINANCIAL|PRINCIPAL FINANCIAL")
_CHAIR = re.compile(r"CHAIR")


def classify_role(relationship: pd.Series, title: pd.Series) -> pd.DataFrame:
    rel = relationship.astype("string").fillna("").str.upper()
    tit = title.astype("string").fillna("").str.upper()
    out = pd.DataFrame(index=relationship.index)
    out["is_director"] = rel.str.contains("DIRECTOR")
    out["is_officer"] = rel.str.contains("OFFICER")
    out["is_tenpct"] = rel.str.contains("TENPERCENT") | rel.str.contains("10%")
    out["is_ceo"] = out["is_officer"] & tit.str.contains(_CEO)
    out["is_cfo"] = out["is_officer"] & tit.str.contains(_CFO)
    out["is_chair"] = tit.str.contains(_CHAIR)
    role = np.select(
        [out["is_ceo"] | out["is_cfo"], out["is_officer"], out["is_director"], out["is_tenpct"]],
        ["CEO/CFO", "Other officer", "Director", "10% owner"], default="Other")
    out["role"] = role
    return out


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Validated open-market purchase (P) and sale (S) lines from original Form 4s."""
    df = raw
    df.columns = [c.upper() for c in df.columns]
    n0 = len(df)
    df = df[df["DOCUMENT_TYPE"].str.strip() == "4"].copy()          # drop amendments (4/A) & Form 5
    title = df["SECURITY_TITLE"].astype("string").fillna("").str.upper() if "SECURITY_TITLE" in df else None
    if title is not None:
        df = df[~(title.str.contains(_NOT_COMMON) & ~title.str.contains("COMMON"))]
    name = df["ISSUERNAME"].astype("string").fillna("").str.upper()
    df = df[~name.str.contains(_POOLED)]
    df["TRANS_CODE"] = df["TRANS_CODE"].str.strip()
    ad = df["TRANS_ACQUIRED_DISP_CD"].str.strip()
    df = df[((df["TRANS_CODE"] == "P") & (ad == "A")) | ((df["TRANS_CODE"] == "S") & (ad == "D"))]
    for c in ["TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["filing_date"] = parse_sec_date(df["FILING_DATE"])
    df["trans_date"] = parse_sec_date(df["TRANS_DATE"])
    df = df[(df["TRANS_SHARES"] > 0) & (df["TRANS_PRICEPERSHARE"] > 0)
            & df["filing_date"].notna() & df["trans_date"].notna()]
    # guard against keying errors: trade must precede filing, and not by years
    lag = (df["filing_date"] - df["trans_date"]).dt.days
    df = df[(lag >= 0) & (lag <= 365)]
    df["value"] = df["TRANS_SHARES"] * df["TRANS_PRICEPERSHARE"]
    df = df[(df["value"] >= 1_000) & (df["value"] <= 2e9)]          # drop token trades & keying errors
    df["issuer_cik"] = pd.to_numeric(df["ISSUERCIK"], errors="coerce").astype("Int64").astype(str)
    df["owner_cik"] = pd.to_numeric(df["RPTOWNERCIK"], errors="coerce").astype("Int64").astype(str)
    sym = df["ISSUERTRADINGSYMBOL"].astype(str)
    uniq = {v: clean_symbol(v) for v in sym.unique()}
    df["filed_symbol"] = sym.map(uniq)
    df["indirect"] = df["DIRECT_INDIRECT_OWNERSHIP"].str.strip().eq("I")
    df.attrs["n_raw_lines"] = n0
    return df


def aggregate_events(tx: pd.DataFrame, code: str) -> pd.DataFrame:
    """Collapse transaction lines into one row per (issuer, insider, filing date)."""
    t = tx[tx["TRANS_CODE"] == code].copy()
    t["px_x_sh"] = t["TRANS_PRICEPERSHARE"] * t["TRANS_SHARES"]
    keys = ["issuer_cik", "owner_cik", "filing_date"]
    g = t.sort_values("trans_date").groupby(keys, sort=False)
    ev = g.agg(
        accession=("ACCESSION_NUMBER", "first"),
        issuer_name=("ISSUERNAME", "first"),
        filed_symbol=("filed_symbol", "first"),
        owner_name=("RPTOWNERNAME", "first"),
        relationship=("RPTOWNER_RELATIONSHIP", "first"),
        title=("RPTOWNER_TITLE", "first"),
        first_trans_date=("trans_date", "min"),
        last_trans_date=("trans_date", "max"),
        shares=("TRANS_SHARES", "sum"),
        value=("value", "sum"),
        px_x_sh=("px_x_sh", "sum"),
        shares_after=("SHRS_OWND_FOLWNG_TRANS", "max"),
        indirect=("indirect", "max"),
        n_lines=("TRANS_SHARES", "size"),
        n_owners=("N_OWNERS", "first"),
    ).reset_index()
    ev["price"] = ev["px_x_sh"] / ev["shares"]
    ev = ev.drop(columns="px_x_sh")
    ev["code"] = code
    ev = pd.concat([ev, classify_role(ev["relationship"], ev["title"])], axis=1)
    before = ev["shares_after"] - ev["shares"] if code == "P" else ev["shares_after"] + ev["shares"]
    ev["own_chg"] = np.where(before > 0, ev["shares"] / before, 5.0)   # 5.0 = new/huge position
    ev["own_chg"] = ev["own_chg"].clip(0, 5.0)
    ev["filing_lag"] = (ev["filing_date"] - ev["last_trans_date"]).dt.days
    return ev.sort_values("filing_date").reset_index(drop=True)


# ----------------------------------------------------------------------------------
def map_to_prices(ev: pd.DataFrame, px: dict[str, pd.DataFrame], cik_map: dict[str, str],
                  unadj: pd.DataFrame) -> pd.DataFrame:
    """Choose the Yahoo ticker for each event and validate it against the Form 4 price.

    Candidates: the symbol written on the filing and the issuer's *current* ticker from
    SEC's CIK map (catches renames such as FB -> META). A candidate is accepted only if
    the market close on the trade date is within PRICE_MATCH_TOL (log distance) of the
    price the insider reported - this rejects recycled tickers that now belong to a
    different company, and bad splits data.
    """
    ev = ev.copy()
    ev["current_symbol"] = ev["issuer_cik"].map(cik_map)
    dates = unadj.index
    cols = {t: i for i, t in enumerate(unadj.columns)}
    arr = unadj.to_numpy()
    # row of the last trading day <= trade date
    row = dates.searchsorted(ev["last_trans_date"].to_numpy(), side="right") - 1
    best_t = np.full(len(ev), None, dtype=object)
    best_d = np.full(len(ev), np.inf)
    for cand in ["filed_symbol", "current_symbol"]:
        sym = ev[cand].to_numpy()
        ci = np.array([cols.get(s, -1) if isinstance(s, str) else -1 for s in sym])
        ok = (ci >= 0) & (row >= 0)
        mkt = np.full(len(ev), np.nan)
        # look back up to 3 sessions in case the stock did not trade on the trade date
        for back in range(4):
            r = np.clip(row - back, 0, None)
            val = np.where(ok, arr[r, np.where(ci >= 0, ci, 0)], np.nan)
            mkt = np.where(np.isnan(mkt), val, mkt)
        d = np.abs(np.log(ev["price"].to_numpy() / mkt))
        better = ok & np.isfinite(d) & (d < best_d)
        best_d = np.where(better, d, best_d)
        best_t = np.where(better, sym, best_t)
    ev["ticker"] = best_t
    ev["price_match_dist"] = best_d
    ev["price_validated"] = ev["price_match_dist"] < C.PRICE_MATCH_TOL
    return ev


def add_entry(ev: pd.DataFrame, px: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Entry = open of the first session strictly AFTER the filing date.

    Form 4s can be filed until 10pm ET and still get that day's filing date, so trading
    on the filing date itself would be look-ahead. We require a valid open/close within
    MAX_ENTRY_SEARCH_DAYS sessions, otherwise the event is dropped (untradeable).
    """
    ev = ev.copy()
    dates = px["close"].index
    cols = {t: i for i, t in enumerate(px["close"].columns)}
    close = px["close"].to_numpy()
    open_ = px["open"].to_numpy()
    first = dates.searchsorted(ev["filing_date"].to_numpy(), side="right")
    ci = np.array([cols.get(t, -1) if isinstance(t, str) else -1 for t in ev["ticker"]])
    entry = np.full(len(ev), -1)
    for k in range(C.MAX_ENTRY_SEARCH_DAYS):
        r = first + k
        valid = (entry < 0) & (ci >= 0) & (r < len(dates))
        rr = np.clip(r, 0, len(dates) - 1)
        cc = np.where(ci >= 0, ci, 0)
        has = valid & np.isfinite(close[rr, cc]) & (close[rr, cc] > 0) & np.isfinite(open_[rr, cc]) & (open_[rr, cc] > 0)
        entry = np.where(has, r, entry)
    ev["col"] = ci
    ev["entry_row"] = entry
    ev["entry_date"] = pd.NaT
    m = entry >= 0
    ev.loc[m, "entry_date"] = dates[entry[m]]
    return ev
