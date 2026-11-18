"""Gamma collector: closed event markets via /events/keyset into the markets/tokens tables.

Why events, not /markets: `/markets` stops paging at offset ~5000 (HTTP 422, "use
/markets/keyset") and its `tags` are almost always just ["all"]; the event object carries the
real topical tags. Each event nests its markets with `conditionId`, `clobTokenIds`, etc.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from pmdetect.collect.http import get_json
from pmdetect.store.db import connect, upsert

GAMMA = "https://gamma-api.polymarket.com"
EVENTS_KEYSET = f"{GAMMA}/events/keyset"
FETCH_ENDPOINT = "gamma/events/keyset"

# Tags dropped server-side to keep the pilot tractable (sports, crypto-prices, weather,
# up-or-down, esports, games). The universe rules in markets/universe.py re-check these.
DEFAULT_EXCLUDE_TAG_IDS = [1, 1312, 84, 102127, 64, 100639]


def parse_ts(value: Any) -> datetime | None:
    """Parse Gamma timestamps ('2026-11-11 10:32:06.1+00', ISO 'Z', date) to naive UTC."""
    if value in (None, ""):
        return None
    s = str(value).strip().replace("Z", "+00:00")
    if len(s) == 10:
        s += "T00:00:00+00:00"
    s = s.replace(" ", "T", 1)
    if s.endswith("+00") or s.endswith("-00"):
        s += ":00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _loads(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            out = json.loads(value)
            return out if isinstance(out, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_event(event: dict) -> tuple[list[dict], list[dict]]:
    """Flatten one Gamma event into market rows and token rows (schema.sql columns)."""
    tags = sorted({t.get("slug") for t in event.get("tags") or [] if t.get("slug")} - {"all"})
    now = datetime.now(UTC).replace(tzinfo=None)
    markets, tokens = [], []
    for m in event.get("markets") or []:
        cid = m.get("conditionId")
        if not cid:
            continue
        outcomes = [str(o) for o in _loads(m.get("outcomes"))]
        prices = [_num(p) for p in _loads(m.get("outcomePrices"))]
        token_ids = [str(t) for t in _loads(m.get("clobTokenIds"))]
        outcome = None
        if outcomes == ["Yes", "No"] and len(prices) == 2 and None not in prices:
            if prices[0] == 1 and prices[1] == 0:
                outcome = "YES"
            elif prices[0] == 0 and prices[1] == 1:
                outcome = "NO"
        statuses = _loads(m.get("umaResolutionStatuses"))
        uma = m.get("umaResolutionStatus")
        if "disputed" in [str(s).lower() for s in statuses] and uma != "disputed":
            uma = "disputed" if not uma else f"{uma}|disputed"
        markets.append(
            {
                "condition_id": cid,
                "event_id": str(event.get("id")) if event.get("id") is not None else None,
                "slug": m.get("slug"),
                "question": m.get("question"),
                "category": m.get("category") or event.get("category"),
                "tags": tags,
                "volume_usd": _num(m.get("volumeNum", m.get("volume"))),
                "liquidity_usd": _num(m.get("liquidityNum", m.get("liquidity"))),
                "start_ts": parse_ts(m.get("startDate") or event.get("startDate")),
                "end_ts": parse_ts(m.get("endDate") or event.get("endDate")),
                "closed_ts": parse_ts(m.get("closedTime") or event.get("closedTime")),
                "uma_status": uma,
                "outcome": outcome,
                "fetched_at": now,
            }
        )
        for tid, label in zip(token_ids, outcomes, strict=False):
            tokens.append({"token_id": tid, "condition_id": cid, "outcome_label": label})
    return markets, tokens


def _window_key(params: dict) -> str:
    items = sorted((k, str(v)) for k, v in params.items() if k not in ("after_cursor", "limit"))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:16]  # noqa: S324


def iter_event_pages(
    params: dict, *, start_cursor: str | None = None
) -> Iterator[tuple[list[dict], str | None]]:
    """Yield (events, next_cursor) pages of /events/keyset."""
    cursor = start_cursor
    while True:
        p = dict(params)
        if cursor:
            p["after_cursor"] = cursor
        page = get_json(EVENTS_KEYSET, p)
        events = page.get("events") or []
        cursor = page.get("next_cursor") or None
        yield events, cursor
        if not cursor or not events:
            return


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur < end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def collect_markets(
    start: date = date(2025, 1, 1),
    end: date | None = None,
    *,
    volume_min: float = 5000,
    max_pages_per_window: int = 3,
    window: str = "month",
    exclude_tag_ids: list[int] | None = None,
    resume: bool = True,
    con=None,
) -> dict:
    """Page closed events per end-date window into markets/tokens. Resumable via fetch_log.

    Returns counters: windows, pages, events, markets.
    """
    end = end or datetime.now(UTC).date()
    own = con is None
    con = con or connect()
    excl = DEFAULT_EXCLUDE_TAG_IDS if exclude_tag_ids is None else exclude_tag_ids
    stats = {"windows": 0, "pages": 0, "events": 0, "markets": 0}
    wins = month_windows(start, end) if window == "month" else [(start, end)]
    try:
        for w0, w1 in wins:
            params: dict[str, Any] = {
                "limit": 100,
                "closed": "true",
                "volume_min": volume_min,
                "end_date_min": f"{w0.isoformat()}T00:00:00Z",
                "end_date_max": f"{w1.isoformat()}T00:00:00Z",
            }
            if excl:
                params["exclude_tag_id"] = list(excl)
            key = _window_key(params)
            rows = con.execute(
                "SELECT last_cursor, completed FROM fetch_log WHERE endpoint=? AND key=?",
                [FETCH_ENDPOINT, key],
            ).fetchall()  # fetchall: duckdb 2.x errors if a result is left unfinished
            row = rows[0] if rows else None
            if resume and row and row[1]:
                stats["windows"] += 1
                continue
            cursor = row[0] if (resume and row) else None
            stats["windows"] += 1
            pages = 0
            for events, nxt in iter_event_pages(params, start_cursor=cursor):
                pages += 1
                m_rows, t_rows = [], []
                for ev in events:
                    mr, tr = parse_event(ev)
                    m_rows += mr
                    t_rows += tr
                if m_rows:
                    upsert(con, "markets", _markets_df(m_rows), ["condition_id"])
                if t_rows:
                    upsert(con, "tokens", pl.DataFrame(t_rows), ["token_id"])
                stats["pages"] += 1
                stats["events"] += len(events)
                stats["markets"] += len(m_rows)
                done = nxt is None
                hit_cap = pages >= max_pages_per_window and not done
                upsert(
                    con,
                    "fetch_log",
                    pl.DataFrame(
                        {
                            "endpoint": [FETCH_ENDPOINT],
                            "key": [key],
                            "last_cursor": [nxt],
                            "completed": [done],
                            "fetched_at": [datetime.now(UTC).replace(tzinfo=None)],
                        }
                    ),
                    ["endpoint", "key"],
                )
                if hit_cap:
                    break
            print(f"  window {w0}..{w1}: {pages} page(s)", flush=True)
    finally:
        if own:
            con.close()
    return stats


def _markets_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema_overrides={
            "tags": pl.List(pl.String),
            "start_ts": pl.Datetime("us"),
            "end_ts": pl.Datetime("us"),
            "closed_ts": pl.Datetime("us"),
            "fetched_at": pl.Datetime("us"),
        },
        infer_schema_length=None,
    )


def find_markets(query: str, *, limit: int = 10) -> list[dict]:
    """Search Gamma /public-search for events matching `query` (used to seed known cases)."""
    res = get_json(f"{GAMMA}/public-search", {"q": query, "limit_per_type": limit})
    return res.get("events") or []
