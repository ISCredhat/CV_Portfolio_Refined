"""Shared pre-registered constants and helpers for stage 2 (frozen 2026-11-14, see docs/STAGE2_PLAN.md).

Periods are assigned by a market's ``closed_ts`` (UTC):
  fit    : closed_ts <= 2025-12-31 23:59:59
  tune   : 2026-01-01 <= closed_ts < 2026-07-01
  sealed : closed_ts >= 2026-07-01   (NOT opened this round: ``assert_not_sealed`` / ``drop_sealed``)

Costs: entry at the next traded price after a delay (5 min default; 1 and 30 min reported), taker fee
per share = rate * p * (1 - p) with the rate by category (a SECONDARY-SOURCE schedule to confirm; the
conservative max over a market's matching tag classes is used; unmatched markets pay 5%).
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

FIT_END = datetime(2026, 1, 1)  # exclusive
TUNE_END = datetime(2026, 7, 1)  # exclusive; sealed from here on
DELAYS_MIN = (5, 1, 30)  # 5 min is the headline, 1 and 30 reported
SEED = 20261113
N_BOOT = 1000

# fee rate by class (SECONDARY SOURCE, to confirm against the venue's schedule)
FEE_RATES = {"geopolitics": 0.00, "sports": 0.03, "core": 0.04, "other": 0.05, "crypto": 0.07}
_TAG_CLASS = {
    "crypto": "crypto",
    "bitcoin": "crypto",
    "ethereum": "crypto",
    "sports": "sports",
    "nba": "sports",
    "nfl": "sports",
    "soccer": "sports",
    "mlb": "sports",
    "nhl": "sports",
    "tennis": "sports",
    "geopolitics": "geopolitics",
    "world": "geopolitics",
    "politics": "core",
    "elections": "core",
    "global-elections": "core",
    "finance": "core",
    "tech": "core",
    "big-tech": "core",
    "ai": "core",
    "trump": "core",
    "mention-markets": "core",
    "tweets-markets": "core",
    "economy": "other",
    "economic-policy": "other",
    "culture": "other",
    "pop-culture": "other",
    "awards": "other",
    "movies": "other",
    "music": "other",
    "weather": "other",
}


def period_of(closed_ts: datetime | None) -> str:
    if closed_ts is None:
        return "unknown"
    if closed_ts < FIT_END:
        return "fit"
    if closed_ts < TUNE_END:
        return "tune"
    return "sealed"


def assert_not_sealed(df: pl.DataFrame, col: str = "closed_ts") -> None:
    """Raise if any row belongs to the sealed period (closed on or after 2026-07-01)."""
    if df.height and df.filter(pl.col(col) >= TUNE_END).height:
        raise AssertionError("sealed period (>= 2026-07-01) must not be opened in this round")


def drop_sealed(df: pl.DataFrame, col: str = "closed_ts") -> pl.DataFrame:
    """Keep only fit and tune rows; rows with a null close time are dropped too."""
    return df.filter(pl.col(col).is_not_null() & (pl.col(col) < TUNE_END))


def fee_class(tags: list[str] | None) -> str:
    """Conservative class: the highest-fee class among the market's matching tags, else 'other'."""
    classes = {_TAG_CLASS[t] for t in (tags or []) if t in _TAG_CLASS}
    if not classes:
        return "other"
    return max(classes, key=lambda c: FEE_RATES[c])


def fee_rate(tags: list[str] | None) -> float:
    return FEE_RATES[fee_class(tags)]


def fee_per_share(p: float, rate: float) -> float:
    """Taker fee per share at price p: rate * p * (1 - p)."""
    return rate * p * (1.0 - p)


def net_return_per_dollar(entry_price: float, won: bool, rate: float, cost_mult: float = 1.0) -> float:
    """Buy one share at entry_price, hold to resolution (payout 1 if won else 0).

    Cost per share = price + fee * cost_mult; return per dollar = (payout - cost) / cost.
    """
    cost = entry_price + cost_mult * fee_per_share(entry_price, rate)
    if cost <= 0:
        return float("nan")
    return ((1.0 if won else 0.0) - cost) / cost
