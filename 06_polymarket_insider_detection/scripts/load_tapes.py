"""Load the bulk tapes (data/tapes/**) into DuckDB / snapshots. Idempotent; one step per call.

  python scripts/load_tapes.py prices   # tapes/prices -> DuckDB prices (INSERT OR IGNORE)
  python scripts/load_tapes.py taker    # pilot taker tape + tapes/trades_taker -> snapshots/trades_taker.parquet (pilot copy kept as trades_taker_pilot.parquet)
  python scripts/load_tapes.py trades   # tapes/trades_all -> DuckDB trades, batch by batch (needs write lock)
  python scripts/load_tapes.py status   # part counts per stage
Run `trades` only after bulk_fetch has finished that stage (it checks the part count).
"""

import glob
import math
import sys
import time
from datetime import datetime

import duckdb
import polars as pl

from pmdetect.store import db
from pmdetect.store.db import PROJECT_ROOT

TAPES = PROJECT_ROOT / "data" / "tapes"
SNAP = PROJECT_ROOT / "data" / "snapshots"


def log(m):
    print(datetime.now().isoformat(timespec="seconds"), m, flush=True)


def expected_parts():
    import json

    ids = json.loads((TAPES / "universe_ids.json").read_text(encoding="utf-8"))
    return {
        "prices": math.ceil(len(ids["winner_tokens"]) / 25),
        "trades_taker": math.ceil(len(ids["conditions"]) / 20),
        "trades_all": math.ceil(len(ids["conditions"]) / 20),
    }


def parts(stage):
    return sorted(glob.glob(str(TAPES / stage / "part-*.parquet")))


def status():
    for k, v in expected_parts().items():
        log(f"{k}: {len(parts(k))}/{v}")


def load_prices():
    fs = parts("prices")
    con = db.connect()
    n0 = con.execute("SELECT count(*) FROM prices").fetchall()[0][0]
    for i in range(0, len(fs), 50):
        grp = [f.replace("\\", "/") for f in fs[i : i + 50]]
        lst = ",".join(f"'{g}'" for g in grp)
        con.execute(
            f"INSERT OR IGNORE INTO prices SELECT token_id, ts, price FROM read_parquet([{lst}])"
        )
    n1 = con.execute("SELECT count(*) FROM prices").fetchall()[0][0]
    log(f"prices: {n0:,} -> {n1:,} rows ({len(fs)} parts)")
    con.close()


def build_taker():
    fs = parts("trades_taker")
    frames = [pl.read_parquet(f) for f in fs]
    pilot = SNAP / "trades_taker_pilot.parquet"
    if pilot.exists():
        frames.append(pl.read_parquet(pilot))
    df = pl.concat(frames, how="vertical_relaxed").unique(subset=["trade_id"], keep="first")
    df = df.sort(["condition_id", "ts"])
    out = SNAP / "trades_taker.parquet"
    df.write_parquet(out, compression="zstd")
    log(f"taker: {df.height:,} rows, {df['condition_id'].n_unique():,} markets -> {out.name}")


def load_trades():
    exp = expected_parts()["trades_all"]
    fs = parts("trades_all")
    if len(fs) < exp:
        raise SystemExit(f"trades_all incomplete: {len(fs)}/{exp}")
    for _attempt in range(600):
        try:
            con = db.connect()
            break
        except duckdb.IOException:
            time.sleep(3)
    n0 = con.execute("SELECT count(*) FROM trades").fetchall()[0][0]
    for i in range(0, len(fs), 20):
        grp = [f.replace("\\", "/") for f in fs[i : i + 20]]
        lst = ",".join(f"'{g}'" for g in grp)
        con.execute(
            "INSERT OR IGNORE INTO trades SELECT trade_id, tx_hash, condition_id, token_id, wallet, "
            f"side, outcome_label, size, price, usd, ts FROM read_parquet([{lst}])"
        )
        log(f"trades: loaded parts {i + len(grp)}/{len(fs)}")
    n1 = con.execute("SELECT count(*) FROM trades").fetchall()[0][0]
    log(f"trades: {n0:,} -> {n1:,} rows")
    con.close()


if __name__ == "__main__":
    {"prices": load_prices, "taker": build_taker, "trades": load_trades, "status": status}[sys.argv[1]]()
