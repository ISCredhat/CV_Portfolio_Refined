"""WRDS access for the CRSP daily study: table discovery, SQL builders, yearly chunk downloads, manifest, French data.

Adapted from ``reference_05/wrds_io.py``. ``connect`` is the only network call to WRDS here; everything else is
pure or works on a connection-like object (``raw_sql``, ``describe_table``, ``list_tables``), so tests use a fake.

Layout: ``config_crsp.WRDS_RAW/<key>/<chunk>.parquet`` (``dsf_v2``: one file per calendar year ``y1961`` ..
``y2025``; every other key: ``all.parquet``) and ``WRDS_RAW/manifest.json`` with, per chunk, rows, seconds, sha256
of the file and sha256 of the normalised SQL text. A chunk that exists is skipped (resumable) unless ``force``.

Choices / assumptions (verify on WRDS; see ``scripts/check_wrds_crsp.py``):
  - ``dsf_v2`` is downloaded for ALL ``securitytype = 'EQTY'`` rows with every classification column that
    ``describe_table`` reports (review M3: point-in-time filters are applied locally from ``stocknames_v2``).
    Only the columns in ``REQUIRED`` raise when absent; the rest are skipped with a logged warning. Cumulative
    adjustment factors (``dlycumfacpr``/``dlycumfacshr``) are not needed and not selected (review m12).
  - ``wrds_username`` is passed explicitly: ``wrds.Connection()`` ignores the WRDS_USERNAME variable.
  - The market index is ``crsp.dsi_v2`` or ``crsp.dsi``, whichever is readable; all its columns are kept.
  - Legacy ``dsedelist`` (event test, plan B2) and a legacy-``dsf`` tail (last 15 days before each delisting) are
    downloaded when readable; the tail is optional.
  - Fama-French 5, MOM and STREV daily come from Ken French's site (``fetch_french``), not WRDS.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_START = "1961-07-01"
DEFAULT_END = "2025-12-31"

CANDIDATES: dict[str, list[str]] = {
    "dsf_v2": ["crsp.dsf_v2"],
    "stocknames_v2": ["crsp.stocknames_v2"],
    "dsedelist": ["crsp.dsedelist", "crsp_a_stock.dsedelist"],
    "co_hgic": ["comp.co_hgic"],
    "ccmxpf_lnkhist": ["crsp.ccmxpf_lnkhist", "crsp_a_ccm.ccmxpf_lnkhist"],
    "ff_factors_daily": ["ff.factors_daily"],
    "market_index": ["crsp.dsi_v2", "crsp.dsi"],
    "dsf_legacy_tail": ["crsp.dsf", "crsp_a_stock.dsf"],
}
# keys the study cannot run without; the rest are optional or have fallbacks
MUST_HAVE = ("dsf_v2", "stocknames_v2", "dsedelist")

# Wish list in a fixed order. Columns missing from the table are skipped unless in REQUIRED.
SELECT_COLUMNS: dict[str, list[str]] = {
    "dsf_v2": ["permno", "permco", "securitybegindt", "securityenddt", "securitytype", "securitysubtype",
               "sharetype", "usincflg", "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg",
               "dlycaldt", "dlydelflg", "dlyprc", "dlyprcflg", "dlycap", "dlycapflg", "dlyret", "dlyretx",
               "dlyreti", "dlyretmissflg", "dlyretdurflg", "dlyorddivamt", "dlynonorddivamt", "dlyfacprc",
               "dlydistretflg", "dlyvol", "dlyclose", "dlylow", "dlyhigh", "dlybid", "dlyask", "dlyopen",
               "dlynumtrd", "shrout", "siccd"],
    "stocknames_v2": ["permno", "permco", "namedt", "nameenddt", "nameendt", "ticker", "tsymbol", "comnam",
                      "ncusip", "cusip", "primaryexch", "siccd", "naics", "securitytype", "securitysubtype",
                      "sharetype", "usincflg", "issuertype", "conditionaltype", "tradingstatusflg", "shrcls"],
    "dsedelist": ["permno", "dlstdt", "dlstcd", "dlret", "dlretx", "dlprc", "nwperm", "nwcomp", "nextdt"],
    "co_hgic": ["gvkey", "indfrom", "indthru", "gsector", "ggroup", "gind", "gsubind", "sic", "naics"],
    "ccmxpf_lnkhist": ["gvkey", "linkprim", "linktype", "liid", "lpermno", "lpermco", "usedflag", "linkdt",
                       "linkenddt"],
    "ff_factors_daily": ["date", "mktrf", "smb", "hml", "rf", "umd"],
    "dsf_legacy_tail": ["permno", "date", "ret", "retx", "prc"],
}
REQUIRED: dict[str, tuple[str, ...]] = {
    "dsf_v2": ("permno", "dlycaldt", "dlyret", "dlyprc", "dlyvol", "securitytype"),
    "stocknames_v2": ("permno", "namedt", "primaryexch", "siccd"),
    "dsedelist": ("permno", "dlstdt", "dlstcd", "dlret"),
    "co_hgic": ("gvkey", "indfrom", "indthru", "gind"),
    "ccmxpf_lnkhist": ("gvkey", "linkprim", "linktype", "lpermno", "linkdt", "linkenddt"),
    "ff_factors_daily": ("date", "mktrf", "smb", "hml", "rf"),
    "dsf_legacy_tail": ("permno", "date", "ret"),
}
_DATE_COLUMNS = frozenset({"dlycaldt", "namedt", "nameenddt", "nameendt", "dlstdt", "indfrom", "indthru", "linkdt",
                           "linkenddt", "date", "caldt", "securitybegindt", "securityenddt", "nextdt"})
_INT_COLUMNS = frozenset({"permno", "permco", "lpermno", "lpermco", "nwperm", "dlstcd", "siccd"})
_STR_COLUMNS = frozenset({"ticker", "tsymbol", "comnam", "ncusip", "cusip", "primaryexch", "securitytype",
                          "securitysubtype", "sharetype", "usincflg", "issuertype", "conditionaltype",
                          "tradingstatusflg", "dlydelflg", "dlyprcflg", "dlyretmissflg", "dlyretdurflg",
                          "dlycapflg", "dlydistretflg", "shrcls", "linkprim", "linktype", "liid", "naics",
                          "gsector", "ggroup", "gind", "gsubind"})

_TEMPLATES = {
    "dsf_v2": "SELECT {select} FROM {table} WHERE securitytype = 'EQTY' AND dlycaldt BETWEEN '{start}' AND "
              "'{end}' ORDER BY permno, dlycaldt",
    "stocknames_v2": "SELECT {select} FROM {table} ORDER BY permno, namedt",
    "dsedelist": "SELECT {select} FROM {table} ORDER BY permno, dlstdt",
    "co_hgic": "SELECT {select} FROM {table}",
    "ccmxpf_lnkhist": "SELECT {select} FROM {table}",
    "ff_factors_daily": "SELECT {select} FROM {table} WHERE date BETWEEN '{start}' AND '{end}' ORDER BY date",
    "market_index": "SELECT {select} FROM {table}",
    "dsf_legacy_tail": "SELECT {select} FROM {table} f WHERE EXISTS (SELECT 1 FROM {delist} e WHERE "
                       "e.permno = f.permno AND f.date BETWEEN e.dlstdt - 15 AND e.dlstdt)",
}


def connect(username: str | None = None):
    """Lazy ``import wrds``; ``wrds.Connection(wrds_username=...)`` (the WRDS_USERNAME variable is read here,
    because wrds.Connection() ignores it). The password is handled by the wrds package (~/.pgpass or prompt)."""
    import wrds
    user = username or os.environ.get("WRDS_USERNAME")
    return wrds.Connection(wrds_username=user) if user else wrds.Connection()


# ------------------------------------------------------------------------------------------------ probing
def describe_columns(conn, table: str) -> list[str]:
    """Lower-case column names of ``lib.table`` from ``conn.describe_table``."""
    lib, _, tab = table.partition(".")
    desc = conn.describe_table(lib, tab)
    col = "name" if "name" in desc.columns else desc.columns[0]
    return [str(c).lower() for c in desc[col].tolist()]


def probe(conn, keys=None) -> dict[str, dict]:
    """First readable candidate per key: {"table", "columns"}; ``{"table": None, "columns": []}`` if none."""
    out = {}
    for key in (keys or CANDIDATES):
        found = {"table": None, "columns": []}
        for cand in CANDIDATES[key]:
            try:
                cols = describe_columns(conn, cand)
            except Exception as e:                                      # not readable / not subscribed
                log.info("probe: %s not usable (%s)", cand, type(e).__name__)
                continue
            if cols:
                found = {"table": cand, "columns": cols}
                break
        out[key] = found
    return out


def missing_required(info: dict, keys=None) -> list[str]:
    return [k for k in (keys or MUST_HAVE) if not info.get(k, {}).get("table")]


# ------------------------------------------------------------------------------------------------ SQL
def _norm_ws(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def sql_digest(sql: str) -> str:
    return hashlib.sha256(_norm_ws(sql).encode("utf-8")).hexdigest()


def build_sql(key: str, table: str, columns: list[str], **params) -> str:
    """Pure. Query for `key`, selecting the wish-list columns that exist (``market_index``: every column the
    table reports). Missing required columns raise KeyError; missing optional ones are skipped with a warning."""
    if key not in _TEMPLATES:
        raise KeyError(f"unknown query key {key!r}")
    have = [str(c).lower() for c in columns]
    if key == "market_index":
        chosen = have
    else:
        order = SELECT_COLUMNS[key]
        miss = [c for c in REQUIRED.get(key, ()) if c not in have]
        if key == "stocknames_v2" and "nameenddt" not in have and "nameendt" not in have:
            miss.append("nameenddt")
        if miss:
            raise KeyError(f"{key}: required column(s) {miss} missing from {table} (have {sorted(have)})")
        chosen = [c for c in order if c in have]
        if key == "stocknames_v2" and "nameenddt" in chosen and "nameendt" in chosen:
            chosen.remove("nameendt")
        skipped = [c for c in order if c not in have]
        if skipped:
            log.warning("%s: optional column(s) %s not in %s; skipped", key, skipped, table)
    fmt = {"table": table, "select": ", ".join(chosen), "start": params.get("start", DEFAULT_START)[:10],
           "end": params.get("end", DEFAULT_END)[:10], "delist": params.get("delist", "crsp.dsedelist")}
    return _norm_ws(_TEMPLATES[key].format(**fmt))


def chunk_plan(key: str, start: str = DEFAULT_START, end: str = DEFAULT_END) -> list[dict]:
    """``dsf_v2``: one chunk per calendar year (the first and last clipped to start/end); others one chunk."""
    s, e = str(start)[:10], str(end)[:10]
    if key == "dsf_v2":
        out = []
        for y in range(int(s[:4]), int(e[:4]) + 1):
            a = max(s, f"{y:04d}-01-01")
            b = min(e, f"{y:04d}-12-31")
            out.append({"name": f"y{y}", "params": {"start": a, "end": b}})
        return out
    return [{"name": "all", "params": {"start": s, "end": e}}]


# ------------------------------------------------------------------------------------------------ dtypes
def normalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns; dates datetime64[ns]; ids int64 (nullable Int64 if missing values); the rest floats
    stay float64 and flag/label columns object."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    for c in out.columns:
        if c in _DATE_COLUMNS:
            out[c] = pd.to_datetime(out[c], errors="coerce").astype("datetime64[ns]")
        elif c in _INT_COLUMNS:
            num = pd.to_numeric(out[c], errors="coerce")
            out[c] = num.astype("int64") if not num.isna().any() else num.round().astype("Int64")
        elif c == "gvkey":
            out[c] = out[c].astype("object").map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                                                 else str(v).strip().zfill(6))
        elif c in _STR_COLUMNS:
            out[c] = out[c].astype("object").where(out[c].notna(), None)
        elif out[c].dtype.kind in "fiu":
            out[c] = out[c].astype("float64")
    return out


# ------------------------------------------------------------------------------------------------ download
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_manifest(raw_dir: Path) -> dict:
    p = Path(raw_dir) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_manifest(raw_dir: Path, manifest: dict) -> None:
    p = Path(raw_dir) / "manifest.json"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_parquet(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def download_key(conn, key: str, table_info: dict, raw_dir: Path, start: str = DEFAULT_START,
                 end: str = DEFAULT_END, force: bool = False, extra: dict | None = None) -> list[dict]:
    """Download every chunk of `key` (skip existing unless `force`), update the manifest after each chunk.
    Returns one dict per chunk: name, path, rows, seconds, sha256, sql_sha256, skipped."""
    raw_dir = Path(raw_dir)
    out_dir = raw_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)
    table, columns = table_info["table"], list(table_info["columns"])
    manifest = read_manifest(raw_dir)
    rec = manifest.setdefault(key, {"table": table, "chunks": {}})
    rec["table"] = table
    info = []
    for ch in chunk_plan(key, start, end):
        dest = out_dir / f"{ch['name']}.parquet"
        if dest.exists() and not force:
            old = rec["chunks"].get(ch["name"], {})
            rec["chunks"][ch["name"]] = {**old, "sha256": sha256_file(dest)}
            info.append({"name": ch["name"], "path": dest, "rows": old.get("rows"), "seconds": 0.0,
                         "sha256": rec["chunks"][ch["name"]]["sha256"], "sql_sha256": old.get("sql_sha256"),
                         "skipped": True})
            continue
        sql = build_sql(key, table, columns, **{**ch["params"], **(extra or {})})
        sel = re.match(r"SELECT (.*?) FROM", sql).group(1).split(", ")
        t0 = time.perf_counter()
        df = conn.raw_sql(sql, date_cols=[c for c in sel if c in _DATE_COLUMNS])
        df = normalise_dtypes(df)
        secs = time.perf_counter() - t0
        write_parquet(df, dest)
        digest = sha256_file(dest)
        rec["chunks"][ch["name"]] = {"rows": int(len(df)), "seconds": round(secs, 2), "sha256": digest,
                                     "sql_sha256": sql_digest(sql), "columns": list(df.columns)}
        write_manifest(raw_dir, manifest)
        log.info("%s/%s: %d rows in %.1f s", key, ch["name"], len(df), secs)
        info.append({"name": ch["name"], "path": dest, "rows": int(len(df)), "seconds": round(secs, 2),
                     "sha256": digest, "sql_sha256": sql_digest(sql), "skipped": False})
    write_manifest(raw_dir, manifest)
    return info


# ------------------------------------------------------------------------------------------------ French
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {"ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
                "mom": "F-F_Momentum_Factor_daily_CSV.zip",
                "strev": "F-F_ST_Reversal_Factor_daily_CSV.zip"}
_FRENCH_NAMES = {"mkt-rf": "mktrf", "smb": "smb", "hml": "hml", "rmw": "rmw", "cma": "cma", "rf": "rf",
                 "mom": "mom", "st_rev": "strev", "strev": "strev"}


def parse_french_csv(text: str) -> pd.DataFrame:
    """Daily rows of a French CSV (``YYYYMMDD, v1, v2, ...``; percent -> fraction). The header is the last line
    before the first data row. Footer text and monthly/annual blocks (non 8-digit keys) are ignored."""
    lines = text.splitlines()
    first = next(i for i, l in enumerate(lines) if re.match(r"^\s*\d{8}\s*,", l))
    header = [h.strip().lower() for h in lines[first - 1].split(",")][1:]
    names = [_FRENCH_NAMES.get(h.replace(" ", "_").replace("-", "-"), h) for h in header]
    rows = []
    for l in lines[first:]:
        if not re.match(r"^\s*\d{8}\s*,", l):
            break
        parts = [p.strip() for p in l.split(",")]
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + names)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in names:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df = df.replace(-0.9999, np.nan)                                   # French missing code -99.99 %
    return df


def fetch_french(cache_dir: Path, force: bool = False, opener=urllib.request.urlopen) -> pd.DataFrame:
    """Download (or reuse cached) FF5, MOM and STREV daily zips from Ken French's site, record zip sha256 values
    in ``french_manifest.json``, return the merged frame (date, mktrf, smb, hml, rmw, cma, rf, mom, strev)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    man, frames = {}, []
    for key, fname in FRENCH_FILES.items():
        zp = cache_dir / fname
        if force or not zp.exists():
            with opener(FRENCH_BASE + fname, timeout=60) as r:
                zp.write_bytes(r.read())
        man[fname] = sha256_file(zp)
        with zipfile.ZipFile(zp) as z:
            member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            frames.append(parse_french_csv(io.TextIOWrapper(z.open(member), encoding="latin-1").read()))
    (cache_dir / "french_manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f.drop(columns=[c for c in f.columns if c in out.columns and c != "date"]), on="date",
                        how="outer")
    return out.sort_values("date").reset_index(drop=True)
