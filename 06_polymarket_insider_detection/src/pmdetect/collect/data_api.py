"""Data API v2 collectors: trades, prices, resolutions, tokens and wallet activity.

Field names confirmed against https://data-api.polymarket.com/v2/openapi.json (2026-11-11):

* ``/v2/trades``: rows carry ``proxy_wallet``, ``condition_id``, ``token_id``, ``side``, ``size``
  (shares), ``price``, ``timestamp`` (epoch s), ``transaction_hash``, ``outcome``. There is no
  trade id and no USD field, so ``usd = size * price`` and ``trade_id`` is a content hash.
  ``taker_only`` defaults to true (one row per fill); we pass ``taker_only=false`` so that
  maker-side rows (limit-order wallets) are in the tape too.
* ``/v2/prices-history``: ``interval=max`` + ``bucket_seconds``; old markets only keep the 3 h
  and 12 h grains, so we walk a ladder from fine to coarse and keep the first non-empty one.
* ``/v2/resolutions``: ``payouts`` in micro-USDC per outcome slot.
* ``/v2/tokens``: token ids, outcome labels and ``final_price`` per condition.
* ``/v2/activity``: ``usdc_size``, ``type``; keyset-paged, ``start=1`` for full history.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import duckdb
import httpx
import polars as pl

from pmdetect.collect.http import get_json, paginate_cursor
from pmdetect.store import db as _db
from pmdetect.store.db import upsert

BASE = "https://data-api.polymarket.com/v2"
TRADES_URL = f"{BASE}/trades"
PRICES_URL = f"{BASE}/prices-history"
RESOLUTIONS_URL = f"{BASE}/resolutions"
TOKENS_URL = f"{BASE}/tokens"
ACTIVITY_URL = f"{BASE}/activity"

MAX_CONDITIONS_PER_TRADES_CALL = 20
MAX_CONDITIONS_PER_TOKENS_CALL = 10
TRADES_PAGE_LIMIT = 1000
# fine to coarse; 10800 (3 h) and 43200 (12 h) exist for the whole life of every market
PRICE_BUCKET_LADDER = (600, 1800, 10800, 43200)
ACTIVITY_MAX_PAGES = 2  # per wallet; 1000 rows per page


LOCK_WAIT_SECONDS = 1800


def connect(path=None) -> duckdb.DuckDBPyConnection:
    """db.connect that waits for another process (e.g. the Gamma collector) to release the file."""
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            return _db.connect(path) if path else _db.connect()
        except duckdb.IOException:
            if time.monotonic() > deadline:
                raise
            time.sleep(3)


@contextmanager
def session():
    """Short-lived connection so long HTTP loops do not hold the single-writer lock."""
    con = connect()
    try:
        yield con
    finally:
        con.close()


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _ts(epoch: int | float | None) -> datetime | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(int(epoch), UTC).replace(tzinfo=None)


def _chunks(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


# --------------------------------------------------------------------------- fetch_log helpers


def _completed(con, endpoint: str) -> set[str]:
    rows = con.execute(
        "SELECT key FROM fetch_log WHERE endpoint = ? AND completed", [endpoint]
    ).fetchall()
    return {r[0] for r in rows}


def _mark(con, endpoint: str, keys: Iterable[str], cursor: str | None = None) -> None:
    now = _now()
    keys = list(keys)
    df = pl.DataFrame(
        {
            "endpoint": [endpoint] * len(keys),
            "key": keys,
            "last_cursor": [cursor] * len(keys),
            "completed": [True] * len(keys),
            "fetched_at": [now] * len(keys),
        },
        schema={
            "endpoint": pl.Utf8,
            "key": pl.Utf8,
            "last_cursor": pl.Utf8,
            "completed": pl.Boolean,
            "fetched_at": pl.Datetime("us"),
        },
    )
    upsert(con, "fetch_log", df, ["endpoint", "key"])


# --------------------------------------------------------------------------- trades


def trade_id(row: dict, occurrence: int = 0) -> str:
    """Content hash used as trades.trade_id (the API exposes no unique trade id)."""
    raw = "|".join(
        str(row.get(k, ""))
        for k in ("transaction_hash", "proxy_wallet", "token_id", "side", "size", "price")
    )
    raw += f"|{row.get('timestamp', '')}|{occurrence}"
    return hashlib.sha1(raw.encode()).hexdigest()


def trades_to_frame(rows: list[dict], seen: Counter | None = None) -> pl.DataFrame:
    """Convert raw /v2/trades rows into the `trades` table layout.

    `seen` carries identical-row counts across calls so repeated identical fills keep distinct
    ids even when they land in different write buffers.
    """
    seen = Counter() if seen is None else seen
    out = []
    for r in rows:
        base = trade_id(r)
        occ = seen[base]
        seen[base] += 1
        size = float(r.get("size") or 0.0)
        price = float(r.get("price") or 0.0)
        out.append(
            {
                "trade_id": base if occ == 0 else trade_id(r, occ),
                "tx_hash": r.get("transaction_hash"),
                "condition_id": r.get("condition_id"),
                "token_id": r.get("token_id"),
                "wallet": (r.get("proxy_wallet") or "").lower(),
                "side": r.get("side"),
                "outcome_label": r.get("outcome"),
                "size": size,
                "price": price,
                "usd": size * price,
                "ts": _ts(r.get("timestamp")),
            }
        )
    schema = {
        "trade_id": pl.Utf8,
        "tx_hash": pl.Utf8,
        "condition_id": pl.Utf8,
        "token_id": pl.Utf8,
        "wallet": pl.Utf8,
        "side": pl.Utf8,
        "outcome_label": pl.Utf8,
        "size": pl.Float64,
        "price": pl.Float64,
        "usd": pl.Float64,
        "ts": pl.Datetime("us"),
    }
    return pl.DataFrame(out, schema=schema)


def collect_trades(condition_ids: list[str]) -> int:
    """Load the full trade tape of each condition (batches of 20, cursor-followed).

    Idempotent: conditions already marked complete in fetch_log are skipped, trades upsert on
    trade_id. Also loads tokens and resolutions. Returns the number of trade rows written.
    """
    ids = _dedupe(c.lower() for c in condition_ids)
    with session() as con:
        done = _completed(con, "trades")
    todo = [c for c in ids if c not in done]
    written = 0
    for batch in _chunks(todo, MAX_CONDITIONS_PER_TRADES_CALL):
        params = {
            "condition": ",".join(batch),
            "limit": TRADES_PAGE_LIMIT,
            "taker_only": "false",
            "filter_amount": 0,
        }
        buf: list[dict] = []
        seen: Counter = Counter()
        for row in paginate_cursor(TRADES_URL, params):
            buf.append(row)
            if len(buf) >= 20000:
                with session() as con:
                    written += upsert(con, "trades", trades_to_frame(buf, seen), ["trade_id"])
                buf = []
        with session() as con:
            if buf:
                written += upsert(con, "trades", trades_to_frame(buf, seen), ["trade_id"])
            _mark(con, "trades", batch)
    with session() as con:
        _collect_tokens(con, ids)
        _collect_resolutions(con, ids)
    return written


# --------------------------------------------------------------------------- tokens


def _collect_tokens(con, condition_ids: list[str]) -> int:
    done = _completed(con, "tokens")
    todo = [c for c in condition_ids if c not in done]
    n = 0
    for batch in _chunks(todo, MAX_CONDITIONS_PER_TOKENS_CALL):
        data = get_json(TOKENS_URL, {"condition": ",".join(batch)}).get("data") or []
        rows = [
            {
                "token_id": t["token_id"],
                "condition_id": (t.get("condition_id") or "").lower(),
                "outcome_label": t.get("outcome"),
            }
            for t in data
            if t.get("condition_id") and (t.get("condition_id") or "").lower() in set(batch)
        ]
        if rows:
            n += upsert(con, "tokens", pl.DataFrame(rows), ["token_id"])
        _mark(con, "tokens", batch)
    return n


def collect_tokens(condition_ids: list[str]) -> int:
    """Load token ids / outcome labels for conditions into `tokens`."""
    with session() as con:
        return _collect_tokens(con, _dedupe(c.lower() for c in condition_ids))


# --------------------------------------------------------------------------- resolutions


def outcome_from_payouts(payouts: list[int] | None) -> str | None:
    """Map [outcome0, outcome1] payouts to YES / NO (slot 0 = Yes); split payouts give None."""
    if not payouts or len(payouts) != 2:
        return None
    p0, p1 = payouts
    if p0 > 0 and p1 == 0:
        return "YES"
    if p1 > 0 and p0 == 0:
        return "NO"
    return None


def outcome_from_tokens(tokens: list[dict]) -> str | None:
    """Fallback: use final prices of 1 and 0 from /tokens (Yes/No labels or slot order)."""
    won = [t for t in tokens if t.get("final_price") == 1.0]
    if len(won) != 1 or len(tokens) != 2:
        return None
    label = (won[0].get("outcome") or "").strip().lower()
    if label in ("yes", "no"):
        return label.upper()
    return "YES" if won[0].get("outcome_index") == 0 else "NO"


def _collect_resolutions(con, condition_ids: list[str]) -> int:
    n = 0
    for batch in _chunks(condition_ids, MAX_CONDITIONS_PER_TRADES_CALL):
        res = get_json(RESOLUTIONS_URL, {"condition": ",".join(batch)}).get("data") or []
        by_cond = {(r.get("condition_id") or "").lower(): r for r in res if r.get("condition_id")}
        tok_by_cond: dict[str, list[dict]] = {}
        for sub in _chunks(batch, MAX_CONDITIONS_PER_TOKENS_CALL):
            for t in get_json(TOKENS_URL, {"condition": ",".join(sub)}).get("data") or []:
                tok_by_cond.setdefault((t.get("condition_id") or "").lower(), []).append(t)
        for cid in batch:
            r = by_cond.get(cid)
            outcome = outcome_from_payouts(r.get("payouts")) if r else None
            if outcome is None:
                outcome = outcome_from_tokens(tok_by_cond.get(cid, []))
            if outcome is None:
                continue
            con.execute("UPDATE markets SET outcome = ? WHERE condition_id = ?", [outcome, cid])
            n += 1
    return n


def collect_resolutions(condition_ids: list[str]) -> int:
    """Set markets.outcome (YES/NO) from /resolutions payouts, else final token prices.

    Only updates existing `markets` rows (direct UPDATE so Gamma columns are untouched).
    Returns the number of markets with a resolved binary outcome.
    """
    ids = _dedupe(c.lower() for c in condition_ids)
    with session() as con:
        return _collect_resolutions(con, ids)


# --------------------------------------------------------------------------- prices


def fetch_price_series(
    token_id: str, ladder: tuple[int, ...] = PRICE_BUCKET_LADDER
) -> tuple[list[dict], int | None]:
    """Finest available price history for a token: (points, bucket_seconds used)."""
    for bucket in ladder:
        params = {"token_id": token_id, "interval": "max", "bucket_seconds": bucket}
        try:
            pts = list(paginate_cursor(PRICES_URL, params))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:  # bucket too fine for this window
                continue
            raise
        if pts:
            return pts, bucket
    return [], None


def collect_prices(token_ids: list[str]) -> int:
    """Load /prices-history per token at the finest fidelity the API still holds."""
    ids = _dedupe(token_ids)
    with session() as con:
        done = _completed(con, "prices")
    written = 0
    for group in _chunks([t for t in ids if t not in done], 10):
        frames, keys = [], []
        for tid in group:
            pts, _bucket = fetch_price_series(tid)
            keys.append(tid)
            if pts:
                frames.append(
                    pl.DataFrame(
                        {
                            "token_id": [tid] * len(pts),
                            "ts": [_ts(p["timestamp"]) for p in pts],
                            "price": [float(p["price"]) for p in pts],
                        },
                        schema={
                            "token_id": pl.Utf8,
                            "ts": pl.Datetime("us"),
                            "price": pl.Float64,
                        },
                    )
                )
        with session() as con:
            if frames:
                written += upsert(con, "prices", pl.concat(frames), ["token_id", "ts"])
            _mark(con, "prices", keys)
    return written


# --------------------------------------------------------------------------- wallet activity


def summarize_activity(rows: list[dict]) -> dict | None:
    """first_seen / n_markets / total_usd from TRADE activity rows (ascending by time)."""
    trades = [r for r in rows if r.get("type", "TRADE") == "TRADE"]
    if not trades:
        return None
    return {
        "first_seen": _ts(min(r["timestamp"] for r in trades)),
        "n_markets": len({r.get("condition_id") for r in trades if r.get("condition_id")}),
        "total_usd": float(sum(float(r.get("usdc_size") or 0.0) for r in trades)),
    }


def fetch_wallet_activity(wallet: str, max_pages: int = ACTIVITY_MAX_PAGES) -> list[dict]:
    params = {
        "user": wallet,
        "type": "TRADE",
        "start": 1,
        "sort_direction": "ASC",
        "limit": 1000,
    }
    rows: list[dict] = []
    pages = 0
    cursor = None
    while pages < max_pages:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        # not cached: ~0.5 MB per page and only the summary is kept
        page = get_json(ACTIVITY_URL, p, use_cache=False)
        rows.extend(page.get("data") or [])
        pages += 1
        cursor = (page.get("pagination") or {}).get("next_cursor")
        if not cursor:
            break
    return rows


def collect_wallet_activity(wallets: list[str]) -> int:
    """Derive wallets.first_seen / n_markets / total_usd from /activity (TRADE rows).

    first_seen is exact (ascending order). n_markets and total_usd cover up to
    ACTIVITY_MAX_PAGES * 1000 trades per wallet, so they are lower bounds for very busy wallets.
    """
    ids = _dedupe(w.lower() for w in wallets)
    with session() as con:
        done = _completed(con, "activity")
    written = 0
    for group in _chunks([w for w in ids if w not in done], 100):
        batch: list[dict] = []
        for w in group:
            s = summarize_activity(fetch_wallet_activity(w))
            if s:
                batch.append({"wallet": w, **s, "funding_source": None, "fetched_at": _now()})
        with session() as con:
            if batch:
                written += upsert(
                    con, "wallets", pl.DataFrame(batch, schema=_WALLET_SCHEMA), ["wallet"]
                )
            _mark(con, "activity", group)
    return written


PRIORITY_SQL = """
WITH t AS (
    SELECT tr.wallet, tr.usd, tr.side, tr.ts, tr.condition_id,
           (upper(tr.outcome_label) = m.outcome AND tr.side = 'BUY'
            AND m.news_ts IS NOT NULL AND tr.ts < m.news_ts) AS early_winner
    FROM trades tr JOIN markets m USING (condition_id)
    WHERE tr.wallet IS NOT NULL
)
SELECT wallet,
       COALESCE(SUM(usd) FILTER (WHERE early_winner), 0) >= 50 AS prio,
       COUNT(DISTINCT condition_id) AS n_markets,
       SUM(usd) AS total_usd
FROM t GROUP BY wallet
ORDER BY prio DESC, total_usd DESC
"""


def collect_wallet_first_seen(limit: int | None = None, log=print) -> int:
    """Cheap wallet pass: one /activity call per wallet (limit=1, ascending) gives first_seen.

    n_markets and total_usd are then the in-store (pilot tapes) values, i.e. lower bounds.
    Wallets already completed by collect_wallet_activity ('activity') are skipped. Resumable
    through fetch_log endpoint 'activity_first'. Order: wallets that bought the eventual winner
    before news_ts with >= $50 staked first, then by total USD.
    """
    with session() as con:
        done = _completed(con, "activity_first") | _completed(con, "activity")
        rows = con.execute(PRIORITY_SQL).fetchall()
    todo = [r for r in rows if r[0] not in done]
    if limit:
        todo = todo[:limit]
    log(f"wallets total={len(rows)} todo={len(todo)}")
    written = 0
    for i, group in enumerate(_chunks(todo, 100)):
        batch, keys = [], []
        for wallet, _prio, n_markets, total_usd in group:
            keys.append(wallet)
            page = get_json(
                ACTIVITY_URL,
                {"user": wallet, "type": "TRADE", "start": 1, "sort_direction": "ASC", "limit": 1},
                use_cache=False,
            )
            data = page.get("data") or []
            if not data:
                continue
            batch.append(
                {
                    "wallet": wallet,
                    "first_seen": _ts(data[0]["timestamp"]),
                    "n_markets": int(n_markets),
                    "total_usd": float(total_usd),
                    "funding_source": None,
                    "fetched_at": _now(),
                }
            )
        with session() as con:
            if batch:
                written += upsert(
                    con, "wallets", pl.DataFrame(batch, schema=_WALLET_SCHEMA), ["wallet"]
                )
            _mark(con, "activity_first", keys)
        log(f"batch {i + 1}: {min((i + 1) * 100, len(todo))}/{len(todo)} written={written}")
    return written


_WALLET_SCHEMA = {
    "wallet": pl.Utf8,
    "first_seen": pl.Datetime("us"),
    "n_markets": pl.Int64,
    "total_usd": pl.Float64,
    "funding_source": pl.Utf8,
    "fetched_at": pl.Datetime("us"),
}


# --------------------------------------------------------------------------- pilot selection

NOBEL_2025_CONDITION = "0x14a3dfeba8b22a32feb0f10763db68bc4d2abeb5bff90e9ae20de53793b35a1d"


PILOT_FILE = _db.PROJECT_ROOT / "data" / "manual" / "pilot_conditions.csv"


def pilot_condition_ids(n: int = 200, seed: int = 42, extra: Iterable[str] = ()) -> list[str]:
    """The pilot set: pinned in data/manual/pilot_conditions.csv when that file exists.

    Otherwise n random universe markets (seeded) plus `extra`; the Nobel 2025 market is always
    added. The draw is written to PILOT_FILE so later universe re-runs cannot change the pilot.
    """
    import csv
    import random

    if PILOT_FILE.exists():
        with PILOT_FILE.open(encoding="utf-8", newline="") as fh:
            pinned = [r["condition_id"].lower() for r in csv.DictReader(fh)]
        return _dedupe([*pinned, NOBEL_2025_CONDITION, *extra])
    with session() as con:
        rows = con.execute(
            "SELECT condition_id FROM markets WHERE in_universe ORDER BY condition_id"
        ).fetchall()
    ids = [r[0] for r in rows]
    random.Random(seed).shuffle(ids)
    pilot = _dedupe([*ids[:n], NOBEL_2025_CONDITION, *extra])
    PILOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PILOT_FILE.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(["condition_id", *pilot]) + "\n")
    return pilot
