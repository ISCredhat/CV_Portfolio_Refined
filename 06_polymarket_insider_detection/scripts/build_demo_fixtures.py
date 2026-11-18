"""Build the offline demo fixtures (tests/fixtures/demo/) from the real Parquet snapshots.

Offline and read-only on data/snapshots (never DuckDB, never the network). Run once by the maintainer;
users of the repo only need the committed fixtures (`pmdetect demo run`).

Selection (seed 20261113, deterministic): the Nobel Peace Prize 2025 winner market, 22 multi-outcome
scheduled-reveal events (the winner market plus its sibling outcomes), 12 single-market scheduled
reveals (6 resolved YES, 6 NO), 5 deadline, 5 unscheduled-event and 5 scheduled-reveal markets without an
independent news time. Thin markets only (100 to 3000 taker fills; Nobel keeps its 6143).

Anonymisation: every wallet becomes ``wallet-NNNNN`` (assigned in the order of a seeded hash, so the
same wallet gets the same id in every table and the ids carry no information about the address); trade ids
become ``trade-NNNNNNN``; the on-chain transaction hash column is dropped. Market, token and event ids
are public market identifiers and are kept.

Usage:  python scripts/build_demo_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20261113
ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshots"
OUT = ROOT / "tests" / "fixtures" / "demo"
NOBEL = "0x14a3dfeba8b22a32feb0f10763db68bc4d2abeb5bff90e9ae20de53793b35a1d"
N_MULTI, N_BIN_PER_SIDE, N_DEADLINE, N_UNSCHED, N_SCHED_NI = 22, 6, 5, 5, 5
MIN_TRADES, MAX_TRADES = 100, 3000


def pick(rng: np.random.RandomState, df: pl.DataFrame, n: int, key: str = "condition_id") -> list[str]:
    ids = sorted(df[key].unique().to_list())
    if len(ids) < n:
        raise SystemExit(f"only {len(ids)} candidates for a group of {n}")
    return [ids[i] for i in rng.permutation(len(ids))[:n]]


def main() -> None:
    rng = np.random.RandomState(SEED)
    mk = pl.read_parquet(SNAP / "markets.parquet")
    ni = pl.read_parquet(SNAP / "news_independent.parquet")
    em = pl.read_parquet(SNAP / "event_markets.parquet")
    tk = (
        pl.scan_parquet(SNAP / "trades_taker.parquet")
        .group_by("condition_id")
        .agg(pl.len().alias("n_fills"))
        .collect()
    )
    base = (
        mk.filter(pl.col("in_universe") & pl.col("outcome").is_in(["YES", "NO"]))
        .join(tk, on="condition_id")
        .filter(pl.col("n_fills").is_between(MIN_TRADES, MAX_TRADES))
        .join(ni.select("condition_id", "independent"), on="condition_id", how="left")
        .join(em.select("condition_id", "neg_risk", "n_event_markets", "prices_fetched"), on="condition_id", how="left")
        .with_columns(pl.col("independent").fill_null(False))
    )
    sched = base.filter((pl.col("info_type") == "scheduled_reveal") & pl.col("independent"))
    multi = sched.filter(
        pl.col("neg_risk").fill_null(False)
        & pl.col("n_event_markets").is_between(4, 14)
        & (pl.col("outcome") == "YES")
        & pl.col("prices_fetched").fill_null(False)
    ).sort("condition_id").unique("event_id", keep="first", maintain_order=True)
    binary = sched.filter(pl.col("n_event_markets") == 1)
    groups: dict[str, list[str]] = {
        "nobel": [NOBEL],
        "multi_event_winner": pick(rng, multi, N_MULTI),
        "binary_yes": pick(rng, binary.filter(pl.col("outcome") == "YES"), N_BIN_PER_SIDE),
        "binary_no": pick(rng, binary.filter(pl.col("outcome") == "NO"), N_BIN_PER_SIDE),
        "deadline": pick(rng, base.filter(pl.col("info_type") == "deadline"), N_DEADLINE),
        "unscheduled_event": pick(rng, base.filter(pl.col("info_type") == "unscheduled_event"), N_UNSCHED),
        "scheduled_not_independent": pick(
            rng, base.filter((pl.col("info_type") == "scheduled_reveal") & ~pl.col("independent")), N_SCHED_NI
        ),
    }
    chosen = [c for g in groups.values() for c in g]
    assert len(set(chosen)) == len(chosen)
    events = mk.filter(pl.col("condition_id").is_in(chosen))["event_id"].unique().to_list()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    markets = mk.filter(pl.col("condition_id").is_in(chosen)).sort("condition_id")
    tokens = pl.read_parquet(SNAP / "tokens.parquet").filter(pl.col("condition_id").is_in(chosen))
    trades = (
        pl.scan_parquet(SNAP / "trades_taker.parquet")
        .filter(pl.col("condition_id").is_in(chosen))
        .collect()
        .sort("ts", "trade_id")
    )
    # prices: the winner token's series (and the Nobel market's both tokens), as stored
    prices = (
        pl.scan_parquet(SNAP / "prices.parquet")
        .filter(pl.col("token_id").is_in(tokens["token_id"].implode()))
        .collect()
        .sort("token_id", "ts")
    )
    ev_m = em.filter(pl.col("event_id").is_in(events)).sort("event_id", "condition_id")
    ev_p = (
        pl.scan_parquet(SNAP / "event_prices.parquet")
        .filter(pl.col("event_id").is_in(events))
        .collect()
        .sort("event_id", "condition_id", "ts")
    )
    wallets = pl.read_parquet(SNAP / "wallets.parquet")

    # --- anonymise: stable, information-free ids from a seeded hash ordering
    ws = sorted(trades["wallet"].unique().to_list(), key=lambda w: hashlib.sha256(f"{SEED}:{w}".encode()).hexdigest())
    wmap = {w: f"wallet-{i + 1:05d}" for i, w in enumerate(ws)}
    trades = trades.with_columns(
        pl.col("wallet").replace_strict(wmap, return_dtype=pl.String),
        pl.format("trade-{}", pl.int_range(1, pl.len() + 1).cast(pl.String).str.zfill(7)).alias("trade_id"),
    ).drop("tx_hash")
    wallets = (
        wallets.filter(pl.col("wallet").is_in(list(wmap)))
        .with_columns(pl.col("wallet").replace_strict(wmap, return_dtype=pl.String))
        .select("wallet", "first_seen", "n_markets", "total_usd")
        .sort("wallet")
    )
    markets = markets.drop("fetched_at")

    for name, df in [
        ("markets", markets), ("tokens", tokens), ("trades_taker", trades), ("prices", prices),
        ("wallets", wallets), ("event_markets", ev_m), ("event_prices", ev_p),
    ]:  # fmt: skip
        df.write_parquet(OUT / f"{name}.parquet", compression="zstd", compression_level=15)
    # the single override row in data/manual that concerns a demo market (Nobel announcement time)
    src = ROOT / "data" / "manual" / "news_time_overrides.csv"
    ov = pl.read_csv(src).filter(pl.col("condition_id").is_in(chosen))
    ov.write_csv(OUT / "news_time_overrides.csv")
    manifest = {
        "seed": SEED,
        "n_markets": len(chosen),
        "n_events_with_siblings": int(ev_m["event_id"].n_unique()),
        "n_sibling_markets": ev_m.height,
        "n_taker_fills": trades.height,
        "n_wallets": len(wmap),
        "groups": {k: len(v) for k, v in groups.items()},
        "condition_ids": groups,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print({k: v for k, v in manifest.items() if k != "condition_ids"}, f"size {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
