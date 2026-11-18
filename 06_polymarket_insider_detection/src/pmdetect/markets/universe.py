"""Universe filter: thin, cleanly resolved binary event markets.

All rules live in UNIVERSE_CONFIG. `drop_reason` returns the first failing rule (or None).
The "at least 20 trades" rule is applied later, after trades are loaded (integrity report).
"""

from __future__ import annotations

import re
from collections import Counter

import polars as pl

UNIVERSE_CONFIG: dict = {
    "volume_min": 5_000.0,
    "volume_max": 500_000.0,
    "allowed_uma_status": {"resolved"},
    # tag slug sets: any hit drops the market
    "exclude_tags": {
        "sports": {
            "sports", "nfl", "nba", "mlb", "nhl", "soccer", "epl", "premier-league",
            "champions-league", "la-liga", "mls", "tennis", "golf", "ufc", "mma", "boxing",
            "f1", "formula-1", "cricket", "esports", "games", "cfb", "college-football",
            "college-basketball", "ncaa", "wnba", "rugby", "nascar", "cs2", "dota-2", "lol",
            "valorant", "atp", "wta",
        },
        "crypto_price": {"crypto-prices", "up-or-down", "hit-price", "price-markets"},
        "weather": {"weather", "temperature", "daily-temperature", "hurricane-season"},
    },
    # regexes on question / slug (case-insensitive)
    "exclude_question_regex": {
        "crypto_updown": r"\bup or down\b|\bup/down\b",
        "crypto_price": (
            r"\b(bitcoin|btc|ethereum|eth|solana|sol|xrp|dogecoin|doge)\b.*"
            r"(above|below|reach|dip to|hit|price)\b.*(\$|\d)"
        ),
        "sports_game": r"\bvs\.?\s|\bspread:|\bo/u\b|\bover/under\b|moneyline",
        "weather": r"\b(highest|lowest) temperature\b|\bdegrees?\b.*\b(°|fahrenheit|celsius)",
    },
    "exclude_slug_regex": {
        "crypto_updown": r"-updown-|up-or-down",
        "sports_game": r"^(nba|nfl|mlb|nhl|epl|ufc|atp|wta|cfb|cbb|lol|cs2|dota2)-",
        "weather": r"highest-temperature|lowest-temperature",
    },
}


def drop_reason(row: dict, cfg: dict = UNIVERSE_CONFIG) -> str | None:
    """First failing universe rule for a market row, or None if it is in the universe."""
    vol = row.get("volume_usd")
    if vol is None or vol < cfg["volume_min"]:
        return "volume_below_min"
    if vol > cfg["volume_max"]:
        return "volume_above_max"
    uma = (row.get("uma_status") or "").lower()
    if "disputed" in uma:
        return "disputed"
    if uma not in cfg["allowed_uma_status"]:
        return "not_resolved"
    if row.get("outcome") not in ("YES", "NO"):
        return "not_clean_binary"
    tags = set(row.get("tags") or [])
    for name, tagset in cfg["exclude_tags"].items():
        if tags & tagset:
            return f"excluded_{name}_tag"
    q = row.get("question") or ""
    for name, rx in cfg["exclude_question_regex"].items():
        if re.search(rx, q, re.I):
            return f"excluded_{name}_text"
    slug = row.get("slug") or ""
    for name, rx in cfg["exclude_slug_regex"].items():
        if re.search(rx, slug, re.I):
            return f"excluded_{name}_slug"
    return None


def build_universe(con, cfg: dict = UNIVERSE_CONFIG) -> dict:
    """Set markets.in_universe for every stored market. Returns {'total','kept','drops'}."""
    df = pl.from_arrow(
        con.sql(
            "SELECT condition_id, slug, question, tags, volume_usd, uma_status, outcome "
            "FROM markets"
        ).to_arrow_table()
    )
    flags, reasons = [], Counter()
    for row in df.iter_rows(named=True):
        r = drop_reason(row, cfg)
        flags.append(r is None)
        reasons[r or "kept"] += 1
    if df.height:
        out = pl.DataFrame({"condition_id": df["condition_id"], "in_universe": flags})
        con.register("_uni", out.to_arrow())
        try:
            con.execute(
                "UPDATE markets SET in_universe = u.in_universe "
                "FROM _uni u WHERE markets.condition_id = u.condition_id"
            )
        finally:
            con.unregister("_uni")
    kept = reasons.pop("kept", 0)
    return {"total": df.height, "kept": kept, "drops": dict(reasons.most_common())}
