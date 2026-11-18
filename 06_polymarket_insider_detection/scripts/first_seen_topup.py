"""Resumable first_seen top-up for the one-shot candidate set (Phase 3b).

Usage: python scripts/first_seen_topup.py [max_wallets]

Reads only Parquet (never DuckDB). Candidate set = pmdetect.wallets.oneshot.topup_candidates (top winning
positions by the score without the freshness component, valid cutoff or not). Wallets already present in
data/snapshots/wallets.parquet / extra_wallets.parquet or in the top-up file are skipped. One
/v2/activity?type=TRADE&sort_direction=ASC&limit=1 call per wallet (~5 req/s via the shared throttle,
use_cache=False so the raw cache does not grow). Output: data/private/first_seen_topup.parquet
(wallet, first_seen (null if the wallet has no trades), fetched_at), rewritten every 100 wallets.
"""

import sys
from datetime import UTC, datetime

import polars as pl

from pmdetect.collect.data_api import ACTIVITY_URL, _ts
from pmdetect.collect.http import get_json
from pmdetect.store.db import PROJECT_ROOT
from pmdetect.wallets.oneshot import topup_candidates

SNAP = PROJECT_ROOT / "data" / "snapshots"
OUT = PROJECT_ROOT / "data" / "private" / "first_seen_topup.parquet"
SCHEMA = {"wallet": pl.String, "first_seen": pl.Datetime("us"), "fetched_at": pl.Datetime("us")}


def log(m: str) -> None:
    print(datetime.now().isoformat(timespec="seconds"), m, flush=True)


def rd(name: str):
    f = SNAP / name
    return pl.read_parquet(f) if f.exists() else None


def main() -> None:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else None
    pos = pl.read_parquet(PROJECT_ROOT / "data" / "private" / "positions.parquet")
    markets = rd("markets.parquet")
    q = markets.select("condition_id", "event_id", "question")
    extra = rd("extra_markets.parquet")
    if extra is not None:
        q = pl.concat(
            [q, extra.select("condition_id", pl.col("condition_id").alias("event_id"), "question")]
        ).unique("condition_id")
    ms_path = PROJECT_ROOT / "results" / "market_scores.csv"
    ms = pl.read_csv(ms_path, infer_schema_length=5000) if ms_path.exists() else None
    known = pl.concat(
        [
            d.select("wallet", "first_seen")
            for d in (rd("wallets.parquet"), rd("extra_wallets.parquet"))
            if d is not None
        ]
    )
    done = pl.read_parquet(OUT) if OUT.exists() else pl.DataFrame(schema=SCHEMA)
    cand = topup_candidates(pos, q, ms, rd("within_event.parquet"), known)
    n_cand_all = cand.height
    todo = cand.filter(~pl.col("wallet").is_in(done["wallet"].implode()))["wallet"].to_list()
    if cap:
        todo = todo[:cap]
    log(
        f"candidates lacking first_seen={n_cand_all} already_topped_up={n_cand_all - len(todo)} todo={len(todo)}"
    )
    rows = done.to_dicts()
    for i, w in enumerate(todo, 1):
        try:
            page = get_json(
                ACTIVITY_URL,
                {"user": w, "type": "TRADE", "start": 1, "sort_direction": "ASC", "limit": 1},
                use_cache=False,
            )
        except Exception as e:  # noqa: BLE001 - keep going, wallet is retried on the next run
            log(f"error {type(e).__name__} (wallet skipped, retried next run)")
            continue
        data = page.get("data") or []
        rows.append(
            {
                "wallet": w,
                "first_seen": _ts(data[0]["timestamp"]) if data else None,
                "fetched_at": datetime.now(UTC).replace(tzinfo=None),
            }
        )
        if i % 100 == 0 or i == len(todo):
            pl.DataFrame(rows, schema=SCHEMA).write_parquet(OUT)
            log(f"{i}/{len(todo)} saved={len(rows)}")
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(OUT)
    log("done")


if __name__ == "__main__":
    main()
