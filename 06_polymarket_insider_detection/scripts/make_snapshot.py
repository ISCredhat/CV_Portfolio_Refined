"""Parquet snapshot of the store for read-only analysis (never takes the write lock).

Usage: python scripts/make_snapshot.py   -> data/snapshots/{markets,tokens,trades,prices,wallets,labels}.parquet
Retries while the wallet backfill holds the DuckDB write lock.
"""

import time

import duckdb

from pmdetect.store.db import DEFAULT_DB, PROJECT_ROOT

OUT = PROJECT_ROOT / "data" / "snapshots"
OUT.mkdir(parents=True, exist_ok=True)
for _attempt in range(120):
    try:
        con = duckdb.connect(str(DEFAULT_DB), read_only=True)
        break
    except duckdb.IOException:
        time.sleep(2)
else:
    raise SystemExit("could not open the DB read-only")
for t in ["markets", "tokens", "trades", "prices", "wallets", "labels"]:
    con.execute(
        f"COPY (SELECT * FROM {t}) TO '{(OUT / (t + '.parquet')).as_posix()}' (FORMAT parquet)"
    )
    print(t, con.execute(f"SELECT count(*) FROM {t}").fetchall()[0][0])
con.close()
