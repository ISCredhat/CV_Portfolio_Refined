"""Industry classifications: Fama-French 48 from SIC, SIC2, and GICS industry as of a date (CCM + co_hgic).

FF48 map: ``data/reference/Siccodes48.txt`` (format of Ken French's ``Siccodes48.zip``: a header line
"<n> <Name> <label>" followed by lines starting with an "aaaa-bbbb" SIC range). The vendored file is the official
one from Ken French's site, supplied by Isaac on 2026-11-19 (it replaced a copy rebuilt from memory, which mapped
96 SIC codes differently, e.g. 7370-7372 software to Computers instead of Business Services). ``load_ff48``
refuses a file whose sha256 differs from ``SICCODES48_SHA256`` (``python -m pairs.industries`` prints the digest).

SIC 0, missing or unmapped codes map to FF48 48 ("Other"), which ``config_crsp.FF48_EXCLUDE`` keeps out of
within-industry pairs (review m5; CRSP SIC differs from Compustat, Kahle & Walkling 1996).

GICS as of date d (review m14): CCM link with ``linkprim IN ('P','C')``, ``linktype IN ('LC','LU')`` and
``linkdt <= d <= coalesce(linkenddt, inf)``, then ``co_hgic`` row with ``indfrom <= d <= coalesce(indthru, inf)``.
The 6-digit ``gind`` is the industry. Several valid links: primary ('P') first, then the latest ``linkdt``.
Verify on WRDS: co_hgic column names (gind, gsubind) and when the history starts (check query 11).
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config_crsp as CC

SICCODES48_PATH = CC.REFERENCE / "Siccodes48.txt"
SICCODES48_SHA256 = "af5dd86f6a2014e76a1fa78ce8194e85fb13d05c6094547d1b3081b39e6c37b8"
OTHER = 48


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def load_ff48(path: str | None = None, sha256: str | None = None) -> tuple[np.ndarray, dict[int, str]]:
    """(lookup table of length 10000 -> FF48 code, {code: short name}). Raises if the sha256 differs."""
    p = Path(path) if path else SICCODES48_PATH
    want = sha256 or SICCODES48_SHA256
    got = sha256_file(p)
    if got != want:
        raise ValueError(f"{p.name}: sha256 {got} != expected {want}")
    table = np.full(10000, OTHER, dtype="int64")
    seen = np.zeros(10000, bool)
    names: dict[int, str] = {}
    cur = None
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*(\d+)\s+(\w+)\s", line + " ")
        r = re.match(r"^\s+(\d{4})-(\d{4})", line)
        if r and cur is not None:
            a, b = int(r.group(1)), int(r.group(2))
            if seen[a:b + 1].any():
                raise ValueError(f"overlapping SIC range {a}-{b} in {p.name}")
            table[a:b + 1] = cur
            seen[a:b + 1] = True
        elif m and not r:
            cur = int(m.group(1))
            names[cur] = m.group(2)
    if sorted(names) != list(range(1, 49)):
        raise ValueError(f"{p.name}: expected 48 industries, found {sorted(names)}")
    return table, names


def ff48_from_sic(sic) -> np.ndarray:
    """FF48 code for each SIC; NaN, <=0, >9999 and unmapped codes -> 48 ("Other")."""
    table, _ = load_ff48()
    s = pd.to_numeric(pd.Series(np.asarray(sic).ravel()), errors="coerce").to_numpy(float)
    ok = np.isfinite(s) & (s > 0) & (s <= 9999)
    out = np.full(s.shape, OTHER, dtype="int64")
    out[ok] = table[s[ok].astype("int64")]
    return out.reshape(np.shape(sic))


def sic2(sic) -> np.ndarray:
    """Two-digit SIC (0 for missing)."""
    s = pd.to_numeric(pd.Series(np.asarray(sic).ravel()), errors="coerce").fillna(0).to_numpy("int64")
    return (s // 100).reshape(np.shape(sic))


def gics_asof(permnos, date, co_hgic: pd.DataFrame, ccm: pd.DataFrame, level: str = "gind") -> pd.Series:
    """GICS code (string) per PERMNO as of `date`; None when no valid link or no GICS record yet."""
    d = pd.Timestamp(date)
    link = ccm.copy()
    link.columns = [c.lower() for c in link.columns]
    for c in ("gvkey", "lpermno", "linkprim", "linktype", "linkdt", "linkenddt"):
        if c not in link.columns:
            raise KeyError(f"ccmxpf_lnkhist: required column {c!r} missing")
    link = link[link["linkprim"].isin(["P", "C"]) & link["linktype"].isin(["LC", "LU"])]
    link = link[link["lpermno"].isin(np.asarray(permnos))]
    link = link[(pd.to_datetime(link["linkdt"]) <= d) & (pd.to_datetime(link["linkenddt"]).fillna(pd.Timestamp.max) >= d)]
    h = co_hgic.copy()
    h.columns = [c.lower() for c in h.columns]
    if level not in h.columns:
        raise KeyError(f"co_hgic: column {level!r} missing")
    h = h[(pd.to_datetime(h["indfrom"]) <= d) & (pd.to_datetime(h["indthru"]).fillna(pd.Timestamp.max) >= d)]
    h = h.sort_values("indfrom").drop_duplicates("gvkey", keep="last")[["gvkey", level]]
    m = link.merge(h, on="gvkey", how="inner")
    m["_p"] = (m["linkprim"] == "P").astype(int)
    m = m.sort_values(["lpermno", "_p", "linkdt"]).drop_duplicates("lpermno", keep="last")
    got = pd.Series(m[level].astype(str).to_numpy(), index=m["lpermno"].to_numpy())
    return pd.Series([got.get(int(p)) for p in permnos], index=np.asarray(permnos), dtype=object)


if __name__ == "__main__":
    print(sha256_file(SICCODES48_PATH))
