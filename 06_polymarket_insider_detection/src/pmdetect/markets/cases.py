"""Seed `labels` from data/manual/known_cases.csv by resolving event slugs through Gamma."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import polars as pl

from pmdetect.collect.gamma import EVENTS_KEYSET, _markets_df, parse_event
from pmdetect.collect.http import get_json
from pmdetect.store.db import PROJECT_ROOT, connect, upsert

CASES_CSV = PROJECT_ROOT / "data" / "manual" / "known_cases.csv"


def fetch_event_by_slug(slug: str) -> dict | None:
    page = get_json(EVENTS_KEYSET, {"slug": slug, "limit": 5})
    for ev in page.get("events") or []:
        if ev.get("slug") == slug:
            return ev
    return None


def seed_labels(path: Path = CASES_CSV, con=None) -> dict:
    """Upsert case markets (+tokens) and label rows. Returns counts and unresolved slugs."""
    own = con is None
    con = con or connect()
    out = {"labels": 0, "markets": 0, "unresolved": []}
    try:
        with path.open(encoding="utf-8", newline="") as f:
            cases = list(csv.DictReader(f))
        label_rows = []
        for c in cases:
            ev = fetch_event_by_slug(c["event_slug"])
            if ev is None:
                out["unresolved"].append(c["event_slug"])
                continue
            m_rows, t_rows = parse_event(ev)
            rx = c.get("market_match") or ""
            if rx:
                keep = {
                    m["condition_id"]
                    for m in m_rows
                    if re.search(rx, f"{m['question']} {m['slug']}", re.I)
                }
                m_rows = [m for m in m_rows if m["condition_id"] in keep]
                t_rows = [t for t in t_rows if t["condition_id"] in keep]
            if m_rows:
                upsert(con, "markets", _markets_df(m_rows), ["condition_id"])
                out["markets"] += len(m_rows)
            if t_rows:
                upsert(con, "tokens", pl.DataFrame(t_rows), ["token_id"])
            for m in m_rows:
                wallet = (c.get("wallet") or "").lower() or None
                key = f"{c['case_name']}|{m['condition_id']}|{wallet}"
                label_rows.append(
                    {
                        "label_id": hashlib.sha1(key.encode()).hexdigest()[:16],  # noqa: S324
                        "condition_id": m["condition_id"],
                        "wallet": wallet,
                        "case_name": c["case_name"],
                        "label": c["label"],
                        "source_url": c.get("source_url") or None,
                        "notes": c.get("notes") or None,
                    }
                )
        if label_rows:
            out["labels"] = upsert(con, "labels", pl.DataFrame(label_rows, infer_schema_length=None), ["label_id"])
    finally:
        if own:
            con.close()
    return out
