"""Information-type tagging from tags and keywords.

scheduled_reveal   : the answer is revealed at a pre-announced time (awards, elections, data
                     releases, rankings/snapshots on a date, earnings, scheduled votes/speeches).
deadline           : "by <date>" / "before <date>" style; resolves YES whenever the event
                     happens before the cutoff, NO if it never does.
unscheduled_event  : an event with no announced reveal time (resignation, strike, arrest,
                     capture, statement, leak ...). Default when nothing else matches.
Priority: scheduled_reveal > deadline > unscheduled_event.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

INFO_TYPES = ("scheduled_reveal", "unscheduled_event", "deadline")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS_CSV = PROJECT_ROOT / "data" / "manual" / "info_type_labels.csv"

SCHEDULED_TAGS = {
    "awards", "oscars", "grammys", "emmys", "golden-globes", "bafta", "eurovision", "nobel",
    "nobel-prize", "elections", "global-elections", "us-election", "primaries",
    "fed", "fed-rates", "cpi", "inflation", "jobs-report", "earnings", "box-office",
    "year-in-review", "google-year-in-search", "gdp", "unemployment", "rankings",
    "us-presidential-election", "midterms", "referendum",
}
SCHEDULED_RX = re.compile(
    r"\b(win|wins|winner|nominee|nominated|nomination|elected|election|primary|runoff|"
    r"referendum|award|oscar|grammy|emmy|bafta|golden globe|eurovision|nobel|"
    r"person of the year|year in search|cpi|inflation|unemployment|jobs report|"
    r"nonfarm|payrolls|gdp|fed (rate|decision)|interest rate|fomc|rate (cut|hike)|"
    r"earnings|box office|opening weekend|rotten tomatoes|top (song|artist|movie|"
    r"searched)|#1|number one|billboard|spotify|ranked|ranking|rank|"
    r"album of the year|best (picture|actor|actress|director)|seats|vote share|"
    r"approval rating|market cap on|close (above|below)|finish (above|below)|"
    r"confirm|confirmation|vote to|farewell address|state of the union|speech|debate|"
    r"public sale|committed|top \d+|(best|top) .{0,40}(model|lab)|at the .{0,12}meeting|largest|most valuable|index)\b",
    re.I,
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
DEADLINE_RX = re.compile(
    rf"\bby\s+(\w+\s+)?(\d{{1,2}}(st|nd|rd|th)?|end of|the end of|year.?end|eoy|q[1-4]|{_MONTHS})"
    rf"|\bbefore\s+(\w+\s+)?(\d{{4}}|\d{{1,2}}|{_MONTHS}|the end|next)"
    rf"|\bin\s+(20\d\d|{_MONTHS})\b|\bby\s+20\d\d|\bwithin\s+\d+|\bon or before\b"
    r"|\bthis (week|month|year)\b"
    r"|\b(by|before|until)\s+(mon|tues|wednes|thurs|fri|satur|sun)day\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\s*-\s*\d{1,2}\b",
    re.I,
)
UNSCHEDULED_RX = re.compile(
    r"\b(resign|fired|out as|step down|removed|ousted|captured|arrest|indict|charged|"
    r"convicted|pardon|strike|invade|invasion|attack|ceasefire|deal|announce|launch|"
    r"release|say|says|tweet|post|meet|visit|sign|veto|declare|die|dies|leave|leaves|"
    r"exposed|hack|leak|default|bankrupt|acquire|merger|ban|sanction|impose|tariff)\b",
    re.I,
)


def classify(question: str, tags: list[str] | None = None, slug: str | None = None) -> str:
    q = question or ""
    tag_sched = bool(set(tags or []) & SCHEDULED_TAGS)
    dead, unsched = DEADLINE_RX.search(q), UNSCHEDULED_RX.search(q)
    if SCHEDULED_RX.search(q):
        # "will X happen by <date>" that merely contains a scheduled-sounding word stays a deadline
        if not tag_sched and dead and unsched:
            return "deadline"
        return "scheduled_reveal"
    if tag_sched and not dead:
        return "scheduled_reveal"
    if dead:
        return "deadline"
    return "unscheduled_event"


def tag_markets(con) -> dict[str, int]:
    """Write info_type for every stored market. Returns counts per type."""
    rows = con.execute("SELECT condition_id, question, tags, slug FROM markets").fetchall()
    out: dict[str, list[str]] = {t: [] for t in INFO_TYPES}
    for cid, q, tags, slug in rows:
        out[classify(q or "", tags or [], slug)].append(cid)
    for t, ids in out.items():
        if ids:
            con.execute(
                "UPDATE markets SET info_type = ? WHERE condition_id IN "
                "(SELECT unnest(?::VARCHAR[]))",
                [t, ids],
            )
    return {t: len(v) for t, v in out.items()}


def load_labels(path: Path = LABELS_CSV) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("info_type")]


def agreement(labels: list[dict] | None = None, con=None) -> dict:
    """Compare classify() against the hand labels (rows carry condition_id, question, tags)."""
    labels = load_labels() if labels is None else labels
    tags_by_cid: dict = {}
    if con is not None:
        tags_by_cid = {
            r[0]: (r[1] or [])
            for r in con.execute("SELECT condition_id, tags FROM markets").fetchall()
        }
    hit, misses = 0, []
    for r in labels:
        tags = tags_by_cid.get(r.get("condition_id")) or _split(r)
        pred = classify(r["question"], tags)
        if pred == r["info_type"]:
            hit += 1
        else:
            misses.append((r["question"], r["info_type"], pred))
    n = len(labels)
    return {"n": n, "agree": hit, "rate": (hit / n) if n else float("nan"), "misses": misses}


def _split(row: dict) -> list[str]:
    return [t for t in (row.get("tags") or "").split("|") if t]
