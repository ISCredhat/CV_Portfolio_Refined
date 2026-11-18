"""Resumable bulk tape fetch for the full universe -> partitioned Parquet under data/tapes/.

Stages (run one at a time or several: `python scripts/bulk_fetch.py prices taker trades`):
  prices : /v2/prices-history of the WINNING token of every universe market (q(t) needs only that)
  taker  : /v2/trades taker_only=true  (one row per fill)
  trades : /v2/trades taker_only=false (taker + maker rows, every fill twice)
A part file is written atomically when its batch is complete; a restart skips existing parts. One
process, N worker threads, one shared ~5 req/s budget (http._throttle) with a shared 429 back-off.
Bulk pages are never written to data/raw_cache (use_cache=False). Pilot markets (already in the
snapshots) are included for uniformity unless --skip-pilot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl

from pmdetect.collect.data_api import (
    PRICE_BUCKET_LADDER,
    PRICES_URL,
    TRADES_URL,
    _ts,
    trades_to_frame,
)
from pmdetect.collect.http import paginate_cursor
from pmdetect.store.db import PROJECT_ROOT

SNAP = PROJECT_ROOT / "data" / "snapshots"
TAPES = PROJECT_ROOT / "data" / "tapes"
IDS_FILE = TAPES / "universe_ids.json"
BATCH = 20
PRICE_GROUP = 25


def log(msg: str) -> None:
    print(datetime.now().isoformat(timespec="seconds"), msg, flush=True)


def universe_ids() -> dict:
    """Frozen id lists so part numbering is stable across restarts."""
    if IDS_FILE.exists():
        return json.loads(IDS_FILE.read_text(encoding="utf-8"))
    m = pl.read_parquet(SNAP / "markets.parquet").filter(
        pl.col("in_universe") & pl.col("outcome").is_not_null()
    )
    t = pl.read_parquet(SNAP / "tokens.parquet")
    w = (
        m.select("condition_id", "outcome")
        .join(t, on="condition_id")
        .filter(pl.col("outcome_label").str.to_uppercase() == pl.col("outcome"))
    )
    conds = sorted(m["condition_id"].to_list())
    tokens = sorted(set(w["token_id"].to_list()))
    TAPES.mkdir(parents=True, exist_ok=True)
    out = {"conditions": conds, "winner_tokens": tokens}
    IDS_FILE.write_text(json.dumps(out), encoding="utf-8")
    return out


def atomic_write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def fetch_trade_batch(batch: list[str], taker_only: bool) -> pl.DataFrame:
    params = {
        "condition": ",".join(batch),
        "limit": 1000,
        "taker_only": "true" if taker_only else "false",
        "filter_amount": 0,
    }
    rows = list(paginate_cursor(TRADES_URL, params, use_cache=False))
    return trades_to_frame(rows, Counter())


def fetch_price_group(tokens: list[str]) -> pl.DataFrame:
    frames = []
    for tid in tokens:
        pts = []
        for bucket in PRICE_BUCKET_LADDER:
            params = {"token_id": tid, "interval": "max", "bucket_seconds": bucket}
            try:
                pts = list(paginate_cursor(PRICES_URL, params, use_cache=False))
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    continue
                raise
            if pts:
                break
        if pts:
            frames.append(
                pl.DataFrame(
                    {
                        "token_id": [tid] * len(pts),
                        "ts": [_ts(p["timestamp"]) for p in pts],
                        "price": [float(p["price"]) for p in pts],
                    },
                    schema={"token_id": pl.Utf8, "ts": pl.Datetime("us"), "price": pl.Float64},
                )
            )
    if frames:
        return pl.concat(frames)
    return pl.DataFrame(schema={"token_id": pl.Utf8, "ts": pl.Datetime("us"), "price": pl.Float64})


def run_stage(stage: str, ids: dict, workers: int, skip: set[str]) -> None:
    if stage == "prices":
        items = [ids["winner_tokens"][i : i + PRICE_GROUP] for i in range(0, len(ids["winner_tokens"]), PRICE_GROUP)]
        fn = fetch_price_group
        sub = TAPES / "prices"
    else:
        conds = [c for c in ids["conditions"] if c not in skip]
        items = [conds[i : i + BATCH] for i in range(0, len(conds), BATCH)]
        taker = stage == "taker"
        fn = lambda b: fetch_trade_batch(b, taker)  # noqa: E731
        sub = TAPES / ("trades_taker" if taker else "trades_all")
    todo = [(i, b) for i, b in enumerate(items) if not (sub / f"part-{i:05d}.parquet").exists()]
    log(f"stage {stage}: {len(items)} parts, {len(todo)} to do (workers={workers})")
    t0, done, rows = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, b): i for i, b in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            df = fut.result()
            atomic_write(df, sub / f"part-{i:05d}.parquet")
            done += 1
            rows += df.height
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                eta = el / done * (len(todo) - done)
                log(f"{stage}: {done}/{len(todo)} parts ({done / max(len(todo), 1):.1%}) rows={rows:,} elapsed={el / 60:.1f}m eta={eta / 60:.1f}m")
    log(f"stage {stage} complete")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["prices", "taker", "trades"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-pilot", action="store_true")
    a = ap.parse_args()
    ids = universe_ids()
    log(f"universe: {len(ids['conditions'])} markets, {len(ids['winner_tokens'])} winner tokens")
    skip: set[str] = set()
    if a.skip_pilot:
        skip = set(pl.read_parquet(SNAP / "trades_taker.parquet")["condition_id"].unique().to_list())
    for st in a.stages:
        run_stage(st, ids, a.workers, skip)


if __name__ == "__main__":
    sys.exit(main())
