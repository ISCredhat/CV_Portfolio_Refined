"""Offline: extract candidate event/resolution time fields from cached Gamma event pages.

Reads data/raw_cache/*.json (no network), writes data/snapshots/gamma_time_fields.parquet
(condition_id, one row per market). Candidates: umaEndDate, closedTime (market+event), endDate
(market+event), eventStartTime, gameStartTime, event eventDate / startTime, first date in the
description text (regex, with optional ET time).
"""
from __future__ import annotations

import glob
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(
    rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})"
    r"(?:[,\s]+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\s*(ET|EST|EDT|UTC|GMT)?)?"
)
NY = ZoneInfo("America/New_York")


def parse_ts(s):
    if not s:
        return None
    try:
        s = str(s).replace("Z", "+00:00").replace(" ", "T", 1) if "T" not in str(s) else str(s).replace("Z", "+00:00")
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=UTC)
        return d.astimezone(UTC).replace(tzinfo=None)
    except Exception:  # noqa: BLE001
        try:
            d = datetime.fromisoformat(str(s)[:10])
            return d
        except Exception:  # noqa: BLE001
            return None


def desc_dates(text):
    """(first_date, has_time, n_dates): first date mentioned in the description (UTC naive)."""
    if not text:
        return None, False, 0
    ms = list(DATE_RE.finditer(text))
    if not ms:
        return None, False, 0
    m = ms[0]
    mon = MONTHS.split("|").index(m.group(1)) + 1
    try:
        d = datetime(int(m.group(3)), mon, int(m.group(2)))
    except ValueError:
        return None, False, len(ms)
    has_t = m.group(4) is not None
    if has_t:
        hh = int(m.group(4)) % 12 if m.group(6) and m.group(6).lower() else int(m.group(4))
        if m.group(6) and m.group(6).lower() == "pm":
            hh += 12
        d = d.replace(hour=hh % 24, minute=int(m.group(5) or 0))
        tz = (m.group(7) or "ET").upper()
        if tz in ("ET", "EST", "EDT"):
            d = d.replace(tzinfo=NY).astimezone(UTC).replace(tzinfo=None)
    return d, has_t, len(ms)


def main():
    rows = {}
    for f in glob.glob(str(ROOT / "data" / "raw_cache" / "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and "events" in d:
            evs = d["events"]
        elif isinstance(d, dict) and "markets" in d and "closedTime" in d:
            evs = [d]
        else:
            continue
        for e in evs:
            for m in e.get("markets", []) or []:
                cid = (m.get("conditionId") or "").lower()
                if not cid:
                    continue
                dd, has_t, nd = desc_dates(m.get("description"))
                r = {
                    "condition_id": cid,
                    "uma_end": parse_ts(m.get("umaEndDate")),
                    "m_closed": parse_ts(m.get("closedTime")),
                    "m_end": parse_ts(m.get("endDate")),
                    "m_event_start": parse_ts(m.get("eventStartTime")),
                    "m_game_start": parse_ts(m.get("gameStartTime")),
                    "e_event_date": parse_ts(e.get("eventDate")),
                    "e_start_time": parse_ts(e.get("startTime")),
                    "e_end": parse_ts(e.get("endDate")),
                    "e_closed": parse_ts(e.get("closedTime")),
                    "desc_date": dd, "desc_has_time": has_t, "desc_n_dates": nd,
                    "uma_status": m.get("umaResolutionStatus"),
                    "resolved_by": (m.get("resolvedBy") or "")[:2],  # presence only, no address stored
                    "has_resolved_by": bool(m.get("resolvedBy")),
                }
                old = rows.get(cid)
                if old is None or (r["m_closed"] and not old["m_closed"]):
                    rows[cid] = r
    df = pl.DataFrame(list(rows.values()), infer_schema_length=None).drop("resolved_by")
    df = df.with_columns([pl.col(c).cast(pl.Datetime("us")) for c in df.columns if df[c].dtype == pl.Datetime("ms") or df[c].dtype == pl.Datetime("us")])
    out = ROOT / "data" / "snapshots" / "gamma_time_fields.parquet"
    df.write_parquet(out)
    print(df.height, "markets ->", out)
    print(df.null_count())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
