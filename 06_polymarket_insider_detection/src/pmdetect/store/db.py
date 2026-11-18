"""DuckDB store: connect (applies schema) and idempotent upsert."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pyarrow  # noqa: F401  # eager import: polars' to_arrow() imports it lazily mid-run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB: Path = PROJECT_ROOT / "data" / "pmdetect.duckdb"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: Path = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the DuckDB file and apply schema.sql."""
    if str(path) != ":memory:":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    if str(path) != ":memory:":  # keep any spill files inside the project's data folder
        tmp = Path(path).parent / "duckdb_tmp"
        tmp.mkdir(exist_ok=True)
        con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    return con


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame, keys: list[str]) -> int:
    """Insert rows; on a key clash update only the columns present in df.

    Columns not in df keep their stored values (so a re-collect never wipes tags added
    later, e.g. info_type). Duplicate keys inside df keep the last row. Returns rows written.
    """
    if df is None or df.height == 0:
        return 0
    df = df.unique(subset=keys, keep="last", maintain_order=True)
    cols = list(df.columns)
    col_sql = ", ".join(f'"{c}"' for c in cols)
    key_sql = ", ".join(f'"{k}"' for k in keys)
    sets = [c for c in cols if c not in keys]
    if sets:
        upd = ", ".join(f'"{c}" = excluded."{c}"' for c in sets)
        conflict = f"DO UPDATE SET {upd}"
    else:
        conflict = "DO NOTHING"
    con.register("_upsert_src", df.to_arrow())
    try:
        con.execute(
            f'INSERT INTO "{table}" ({col_sql}) SELECT {col_sql} FROM _upsert_src '
            f"ON CONFLICT ({key_sql}) {conflict}"
        )
    finally:
        con.unregister("_upsert_src")
    return df.height
