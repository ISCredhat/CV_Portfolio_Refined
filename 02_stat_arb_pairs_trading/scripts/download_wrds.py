"""Download the CRSP daily study data from WRDS (one connection, one Duo push) and Ken French's site.

    python scripts/download_wrds.py [--username U] [--only dsf_v2 stocknames_v2 ...] [--force]
                                    [--start 1961-07-01] [--end 2025-12-31] [--french] [--dry-run]

Yearly parquet chunks go to ``config_crsp.WRDS_RAW/<key>/``; ``manifest.json`` records rows, seconds, sha256 and
the SQL hash per chunk. Existing chunks are skipped (resumable) unless --force. Required columns that a table
lacks abort that key with a message (exit 2); optional ones are skipped with a warning.
Exit codes: 0 ok, 1 connection failed, 2 a required table/column is missing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import wrds_io as W  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--username", default=None)
    ap.add_argument("--only", nargs="*", default=None, help=f"keys: {', '.join(W.CANDIDATES)}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--start", default=CC.DOWNLOAD_START)
    ap.add_argument("--end", default=CC.DOWNLOAD_END)
    ap.add_argument("--raw-dir", default=str(CC.WRDS_RAW))
    ap.add_argument("--french", action="store_true", help="also fetch FF5/MOM/STREV daily from Ken French's site")
    ap.add_argument("--french-only", action="store_true", help="only the French data (no WRDS connection)")
    ap.add_argument("--dry-run", action="store_true", help="probe tables and print the plan, download nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = Path(args.raw_dir)
    if args.french or args.french_only:
        df = W.fetch_french(raw / "french", force=args.force)
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "french").mkdir(exist_ok=True)
        W.write_parquet(df, raw / "french" / "french_daily.parquet")
        print(f"french: {len(df)} rows {df['date'].min().date()}..{df['date'].max().date()}")
        if args.french_only:
            return 0
    keys = args.only or list(W.CANDIDATES)
    bad = [k for k in keys if k not in W.CANDIDATES]
    if bad:
        print(f"unknown key(s) {bad}", file=sys.stderr)
        return 2
    try:
        conn = W.connect(args.username)
    except Exception as e:
        print(f"could not connect to WRDS ({type(e).__name__}: {e})", file=sys.stderr)
        return 1
    rc = 0
    try:
        info = W.probe(conn, keys)
        for k in keys:
            t = info[k]
            print(f"{k:18s} {t['table'] or 'NOT FOUND (tried ' + ', '.join(W.CANDIDATES[k]) + ')'}")
        for k in W.missing_required(info, [k for k in keys if k in W.MUST_HAVE]):
            print(f"MISSING required table {k}", file=sys.stderr)
            rc = 2
        if args.dry_run:
            return rc
        for k in keys:
            if not info[k]["table"]:
                continue
            extra = {"delist": info["dsedelist"]["table"]} if k == "dsf_legacy_tail" and info.get("dsedelist") else None
            try:
                W.download_key(conn, k, info[k], raw, args.start, args.end, force=args.force, extra=extra)
            except KeyError as e:
                print(f"{k}: {e}", file=sys.stderr)
                rc = 2
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
