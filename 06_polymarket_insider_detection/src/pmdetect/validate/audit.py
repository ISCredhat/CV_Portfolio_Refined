"""Phase 4E: offline hand audit of the extremes of the two detectors (descriptive, no ground truth).

Two lists are audited:

* the top 30 WITHIN-EVENT extremes (24 h window): real multi-outcome events with >= 3 comparable outcomes and a
  computed statistic (the same 809-event population as the 4A simulation), ordered by the winner's within-event
  percentile (descending), then the winner-minus-median-loser log-odds difference ``diff_24h`` (descending), then
  ``n_comparable`` (descending), then the winner's condition id (ascending, only to make the order deterministic);
* the top 30 ONE-SHOT positions OUTSIDE the Maduro market: real winning, valid-cutoff positions of non-excluded
  wallets with USD >= 100 (the watchlist's own size rule), ordered by the one-shot score (descending), then USD
  (descending), then the wallet address (ascending, deterministic only).

Each row is put into one of four classes using only data on disk (market question, the winner's price path and its
timing relative to the reveal or cutoff, trade count and volume around the move, wallet first_seen / market count
from the store):

* ``plausible_leak``: an early, sharp, one-sided move before any visible public trigger;
* ``predictable_convergence``: a gradual rise, a favourite firming up, or no visible move at all in a market whose
  percentile comes from its siblings;
* ``public_news_reaction``: a sharp step that follows a visible public trigger (a related market that moved first,
  a market-wide repricing at the same hour, a known release time, the market's own opening);
* ``unclear``: the default, whenever the offline data cannot decide.

The classifier is a transparent rule set (``Rules``), fixed before looking at the results, applied identically to
every row. It is NOT ground truth: a human reading the news or the source feed would classify some rows
differently, and a class is not evidence about any person. Wallet addresses appear only in the one-shot private
file ``data/private/audit_oneshot.csv``; the within-event file carries none; ``results/validation_hand_audit.md`` is
aggregate-only (counts, shares, market questions).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.store.db import PROJECT_ROOT
from pmdetect.watchlist import MIN_USD, ONE_SHOT_Q, assert_no_addresses

SNAP = PROJECT_ROOT / "data" / "snapshots"
PRIVATE = PROJECT_ROOT / "data" / "private"
RESULTS = PROJECT_ROOT / "results"

CLASSES = ("plausible_leak", "predictable_convergence", "public_news_reaction", "unclear")
N_AUDIT = 30
MADURO_Q = "Maduro out by January 31, 2026?"
TOPIC_RE = re.compile(
    r"earnings|beat quarterly|\bEPS\b|box office|opening weekend|\bpoll|approval rating|\bAI\b|arena|leaderboard|"
    r"market cap|largest company|\belect|primary|senate|governor|mayor|deputy|presidential|eurovision|oscar|"
    r"grammy|emmy",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rules:
    """Thresholds of the classifier, fixed in advance (probability points unless stated)."""

    min_rise: float = 0.02  # a winner that rose less than this has no visible move
    min_step: float = 0.10  # a step is "sharp" at >= 10 points ...
    step_share: float = 0.60  # ... and >= 60% of the whole rise happens inside one short step
    max_res_h: float = 6.0  # observation spacing above this cannot show a step
    early_h: float = 3.0  # a step ending within this many hours of the window end is "late"
    high_p: float = 0.5  # a "favourite"
    sib_min: float = 0.05  # a related move must be at least this big ...
    sib_rel: float = 0.5  # ... and this share of the winner's step to count as "moved first"
    open_h: float = 12.0  # a step within this many hours after the market opened is price discovery
    concurrent_min: int = 3  # this many other markets repricing the same hour = a common public trigger
    fresh_days: float = 14.0  # "fresh wallet" for the one-shot leak class
    flow_share: float = 0.25  # the wallet's share of the winner-side flow around its entry for "informed-looking"
    pre_chase: float = 0.10  # a visible rise before the one-shot entry = bought a move already underway
    flat_after: float = 0.03  # no visible rise between entry and the window end


RULES = Rules()


# ------------------------------------------------------------------------------------------ path utilities


def to_grid(ts: np.ndarray, px: np.ndarray, anchor, h_from: int, h_to: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward-filled price at ``anchor - h`` hours for h = h_from, h_from - 1, ..., h_to (NaN before the first
    observation; never backward-filled)."""
    hs = np.arange(h_from, h_to - 1, -1)
    anchor = np.datetime64(anchor, "us")
    t = anchor - (hs.astype(np.int64) * 3_600_000_000).astype("timedelta64[us]")
    if len(ts) == 0:
        return hs, np.full(len(hs), np.nan)
    order = np.argsort(ts, kind="stable")
    ts, px = np.asarray(ts, "datetime64[us]")[order], np.asarray(px, float)[order]
    idx = np.searchsorted(ts, t, side="right") - 1
    out = np.where(idx >= 0, px[np.clip(idx, 0, None)], np.nan)
    return hs, out


def obs_gap_h(ts: np.ndarray, anchor, h_from: int, h_to: int) -> float:
    """Median gap in hours between observations inside the grid range (inf with fewer than 2)."""
    anchor = np.datetime64(anchor, "us")
    lo = anchor - np.timedelta64(int(h_from * 3600), "s")
    hi = anchor - np.timedelta64(int(h_to * 3600), "s")
    t = np.sort(np.asarray(ts, "datetime64[us]"))
    t = t[(t >= lo) & (t <= hi)]
    if len(t) < 2:
        return float("inf")
    return float(np.median(np.diff(t).astype("timedelta64[s]").astype(float)) / 3600.0)


def best_step(p: np.ndarray, i0: int, i1: int, lag: int) -> tuple[float, int, int]:
    """Largest rise ``p[i + lag] - p[i]`` with ``i0 <= i`` and ``i + lag <= i1`` (indices in time order)."""
    best, bi = -np.inf, -1
    for i in range(max(i0, 0), i1 - lag + 1):
        a, b = p[i], p[i + lag]
        if np.isfinite(a) and np.isfinite(b) and b - a > best:
            best, bi = b - a, i
    if bi < 0:
        return float("nan"), -1, -1
    return float(best), bi, bi + lag


def _idx(hours: np.ndarray, h: float) -> int:
    return int(round(float(hours[0]) - h))


def _res(res_h: float) -> int:
    return max(2, int(np.ceil(res_h)))


def _sibling_first(
    hours: np.ndarray, siblings, a: int, step: float, lag: int, i_lo: int, i_hi: int, rules: Rules
) -> bool:
    """True when some other outcome made a big move (either direction) that ENDED before the winner's step began,
    within the 24 h before it: the market was already repricing on something else."""
    for s in siblings or []:
        s = np.asarray(s, float)
        for sign in (1.0, -1.0):
            sz, sa, sb = best_step(sign * s, max(i_lo, a - 24), i_hi, lag)
            if np.isfinite(sz) and sz >= max(rules.sib_min, rules.sib_rel * step) and sb <= a:
                return True
    return False


def _flow(flow, step_start_h: float, step_end_h: float) -> dict | None:
    return flow(step_start_h, step_end_h) if callable(flow) else flow


# ------------------------------------------------------------------------------------- the within-event rules


def classify_winner_path(
    hours: np.ndarray,
    price: np.ndarray,
    *,
    win_start_h: int = 25,
    win_end_h: int = 1,
    res_h: float = 1.0,
    siblings=None,
    release_h=(),
    market_open_h: float | None = None,
    common_trigger: bool = False,
    flow=None,
    topic: bool = False,
    require_flow: bool = False,
    rules: Rules = RULES,
) -> dict:
    """Classify the winner's price path over the 24 h window ending 1 h before the reveal.

    ``hours``: hours before the reveal, consecutive, descending (e.g. 97 down to 1); ``price``: the winner's
    forward-filled price at those hours. ``siblings``: other outcomes' price arrays on the same grid.
    ``release_h``: known release times as hours before the reveal; ``market_open_h``: hours before the reveal at
    which the market opened; ``common_trigger``: many other markets repriced in the same hour; ``flow``: a dict
    (or a callable ``(step_start_h, step_end_h) -> dict``) with ``one_sided`` and ``informed_looking``;
    ``require_flow``: with no flow information a sharp step stays ``unclear`` rather than ``plausible_leak``.
    Returns ``{"cls", "reason", ...features}``.
    """
    hours = np.asarray(hours)
    price = np.asarray(price, float)
    i0, i1 = _idx(hours, win_start_h), _idx(hours, win_end_h)
    out: dict = {"cls": "unclear", "reason": "", "res_h": res_h}
    if i0 < 0 or i1 >= len(price) or not (np.isfinite(price[i0]) and np.isfinite(price[i1])):
        out["reason"] = "winner price missing at the window edges: cannot judge"
        return out
    if res_h > rules.max_res_h:
        out["reason"] = f"price observations too sparse ({res_h:.1f} h spacing) to see a step: cannot judge"
        return out
    lag = _res(res_h)
    p0, p1 = float(price[i0]), float(price[i1])
    rise = p1 - p0
    size, a, b = best_step(price, i0, i1, lag)
    share = float(np.clip(size / rise, 0.0, 1.0)) if rise > 1e-9 and np.isfinite(size) else 0.0
    j48 = _idx(hours, win_start_h + 48)
    pre48 = float(p0 - price[j48]) if 0 <= j48 < len(price) and np.isfinite(price[j48]) else float("nan")
    out.update(p_start=p0, p_end=p1, rise=rise, step_size=size, step_share=share, pre48_rise=pre48)
    if a >= 0:
        out.update(step_start_h=float(hours[a]), step_end_h=float(hours[b]))
    topic_txt = "; scheduled-results/polls/leaderboard-type market" if topic else ""
    if rise < rules.min_rise:
        if p0 >= rules.high_p and p1 >= rules.high_p:
            out.update(cls="predictable_convergence",
                       reason="favourite already high, no visible move; the percentile comes from siblings" + topic_txt)  # fmt: skip
        else:
            out["reason"] = "no visible rise of the winner in the window (percentile from siblings' falls or stale ties): cannot judge"  # fmt: skip
        return out
    sharp = np.isfinite(size) and size >= rules.min_step and share >= rules.step_share
    if not sharp:
        why = "favourite firming up" if p0 >= rules.high_p else "gradual rise"
        out.update(cls="predictable_convergence", reason=f"{why} (largest step {size:.2f}, {share:.0%} of the rise)" + topic_txt)  # fmt: skip
        return out
    if np.isfinite(pre48) and pre48 >= 0.5 * rise and pre48 >= rules.min_step:
        out.update(cls="predictable_convergence", reason="price already rising in the 48 h before the window; the step continues a trend" + topic_txt)  # fmt: skip
        return out
    sh, eh = float(hours[a]), float(hours[b])
    why = []
    if any(sh + 1 >= r >= eh - 1 for r in (release_h or ())):
        why.append("step coincides with a known release time")
    if _sibling_first(hours, siblings, a, size, lag, max(i0 - 24, 0), i1, rules):
        why.append("a related outcome moved first")
    if common_trigger:
        why.append("other markets repriced in the same hours")
    if market_open_h is not None and 0 <= market_open_h - sh <= rules.open_h:
        why.append("step within hours of the market opening")
    out["triggers"] = why
    if why:
        out.update(cls="public_news_reaction", reason="sharp step with a visible public trigger: " + "; ".join(why))
        return out
    if topic:
        out["reason"] = "sharp step in a market type with frequent public intermediate data (polls, leaderboards, results): cannot rule out a public trigger"  # fmt: skip
        return out
    if eh < win_end_h + rules.early_h:
        out["reason"] = f"sharp step only {eh - win_end_h + 1:.0f} h before the reveal: cannot separate from first public results"  # fmt: skip
        return out
    fl = _flow(flow, sh, eh)
    out["flow"] = fl
    if fl is None:
        if require_flow:
            out["reason"] = "sharp early step but no tape for the winner: one-sidedness cannot be judged"
            return out
        out.update(cls="plausible_leak", reason=f"sharp early step ({size:.2f}, {eh:.0f} h before the reveal), no related move first, no known release time (flow unknown)")  # fmt: skip
        return out
    if fl.get("one_sided") and fl.get("informed_looking", True):
        out.update(cls="plausible_leak", reason=f"sharp early step ({size:.2f}, {eh:.0f} h before the reveal), one-sided concentrated or fresh-wallet flow, no related move first, no known release time")  # fmt: skip
        return out
    out["reason"] = "sharp early step but the flow is not one-sided and concentrated: cannot judge"
    return out


# ---------------------------------------------------------------------------------------- the one-shot rules


def classify_oneshot_path(
    hours: np.ndarray,
    price: np.ndarray,
    *,
    entry_h: float,
    end_h: float,
    entry_price: float,
    age_days: float | None = None,
    only_market: bool = False,
    res_h: float = 1.0,
    siblings=None,
    release_h=(),
    market_open_h: float | None = None,
    concurrent_jumps: int = 0,
    flow=None,
    topic: bool = False,
    require_flow: bool = False,
    rules: Rules = RULES,
) -> dict:
    """Classify the winner-token path around a one-shot position.

    ``hours``: hours before the cutoff (may go negative when the window runs past a price-based cutoff),
    consecutive, descending; ``entry_h``: hours before the cutoff at the position's first trade; ``end_h``: the
    last hour of the window (1 for a scheduled cutoff; below zero to see the move that defines a jump-start
    cutoff). ``flow`` is a dict (or callable) with ``wallet_share`` (the wallet's share of winner-side buy USD
    around its entry) and ``n_trades``.
    """
    hours = np.asarray(hours)
    price = np.asarray(price, float)
    ie, i1 = _idx(hours, entry_h), _idx(hours, end_h)
    out: dict = {"cls": "unclear", "reason": "", "res_h": res_h}
    if ie < 0 or i1 >= len(price) or ie >= i1 or not np.isfinite(price[ie]):
        out["reason"] = "winner price missing around the entry: cannot judge"
        return out
    if res_h > rules.max_res_h:
        out["reason"] = f"price observations too sparse ({res_h:.1f} h spacing) to see a step: cannot judge"
        return out
    lag = _res(res_h)
    pe = float(price[ie])
    jp = _idx(hours, entry_h + 24)
    pre_rise = float(pe - price[jp]) if 0 <= jp < len(price) and np.isfinite(price[jp]) else float("nan")
    seg = price[ie : i1 + 1]
    post_total = float(np.nanmax(seg) - pe) if np.isfinite(seg).any() else float("nan")
    size, a, b = best_step(price, ie, i1, lag)
    share = float(np.clip(size / post_total, 0.0, 1.0)) if post_total > 1e-9 and np.isfinite(size) else 0.0
    out.update(p_entry=pe, pre_rise=pre_rise, post_rise=post_total, step_size=size, step_share=share)
    if a >= 0:
        out.update(step_start_h=float(hours[a]), step_end_h=float(hours[b]), step_after_entry_h=float(entry_h - hours[a]))  # fmt: skip
    topic_txt = "; scheduled-results/polls/leaderboard-type market" if topic else ""
    if np.isfinite(pre_rise) and pre_rise >= rules.pre_chase:
        out.update(cls="public_news_reaction", reason=f"bought after a visible rise of {pre_rise:.2f} in the 24 h before the entry: a move already underway" + topic_txt)  # fmt: skip
        return out
    if not np.isfinite(post_total) or post_total < rules.flat_after:
        if entry_price >= rules.high_p:
            out.update(cls="predictable_convergence", reason="favourite bought, no visible move before the window end" + topic_txt)  # fmt: skip
        else:
            out["reason"] = "no visible price move between the entry and the window end (the win came with the reveal): cannot judge"  # fmt: skip
        return out
    sharp = np.isfinite(size) and size >= rules.min_step and share >= rules.step_share
    if not sharp:
        why = "favourite firming up" if entry_price >= rules.high_p else "gradual rise after the entry"
        out.update(cls="predictable_convergence", reason=f"{why} (largest step {size:.2f}, {share:.0%} of the rise)" + topic_txt)  # fmt: skip
        return out
    sh, eh = float(hours[a]), float(hours[b])
    why, ambiguous = [], []
    if any(r >= entry_h for r in (release_h or ())):
        why.append("a known release time before the entry")
    elif any(sh + 1 >= r >= eh - 1 for r in (release_h or ())):
        ambiguous.append("step coincides with a known release time after the entry")
    if _sibling_first(hours, siblings, ie, size, lag, max(ie - 24, 0), i1, rules):
        why.append("a related outcome had already moved before the entry")
    elif _sibling_first(hours, siblings, a, size, lag, ie, i1, rules):
        ambiguous.append("a related outcome moved between the entry and the step")
    if concurrent_jumps >= rules.concurrent_min:
        ambiguous.append(f"{concurrent_jumps} other markets repriced in the same hours")
    if market_open_h is not None and 0 <= market_open_h - sh <= rules.open_h:
        why.append("step within hours of the market opening")
    out["triggers"] = why + ambiguous
    if why:
        out.update(cls="public_news_reaction", reason="visible public trigger before the entry: " + "; ".join(why))
        return out
    if ambiguous:
        out["reason"] = (
            "entered before a sharp step that was also visible elsewhere (" + "; ".join(ambiguous)
            + "): consistent with public news, but whether the information was public at the entry cannot be judged offline"
        )
        return out
    if topic:
        out["reason"] = "sharp step in a market type with frequent public intermediate data: cannot rule out a public trigger"  # fmt: skip
        return out
    fresh = bool(only_market) or (age_days is not None and np.isfinite(age_days) and age_days <= rules.fresh_days)
    if not fresh:
        out["reason"] = "sharp step after the entry but the wallet is neither fresh nor single-market: cannot judge"
        return out
    fl = _flow(flow, sh, eh)
    out["flow"] = fl
    if fl is None:
        if require_flow:
            out["reason"] = "sharp step after a fresh wallet's entry but no tape: cannot judge the flow"
            return out
    elif not fl.get("wallet_share", 0.0) >= rules.flow_share:
        out["reason"] = "sharp step but the wallet is a small share of the winner-side flow around its entry: cannot judge"  # fmt: skip
        return out
    out.update(cls="plausible_leak", reason=f"fresh/single-market wallet bought {entry_h - sh:.0f} h before a sharp step ({size:.2f}); no related move first, no known release time")  # fmt: skip
    return out


# --------------------------------------------------------------------------------------------- flow helpers


def flow_stats(trades: pl.DataFrame, t_lo, t_hi, first_seen: dict | None = None, fresh_days: float = 14.0) -> dict:
    """Aggregate YES-token taker flow in [t_lo, t_hi] from rows ``ts, side, usd, wallet`` (no addresses returned)."""
    sub = trades.filter((pl.col("ts") >= t_lo) & (pl.col("ts") <= t_hi))
    buys = sub.filter(pl.col("side") == "BUY")
    sells = sub.filter(pl.col("side") == "SELL")
    b, s = float(buys["usd"].sum() or 0.0), float(sells["usd"].sum() or 0.0)
    n_w = int(sub["wallet"].n_unique()) if sub.height else 0
    by_w = buys.group_by("wallet").agg(pl.col("usd").sum()).sort("usd", descending=True)
    top3 = float(by_w["usd"].head(3).sum() / b) if b > 0 else float("nan")
    fresh = float("nan")
    only_mkt = float("nan")
    known_share = float("nan")
    if first_seen is not None and b > 0:
        fs = buys.with_columns(pl.col("wallet").replace_strict(first_seen["first_seen"], default=None).alias("fs"))
        known = fs.filter(pl.col("fs").is_not_null())
        known_share = float(known["usd"].sum() / b)
        if known.height:
            age = (known["ts"] - known["fs"]).dt.total_seconds() / 86400.0
            fresh = float(known.filter(age <= fresh_days)["usd"].sum() / known["usd"].sum())
            nm = known.with_columns(pl.col("wallet").replace_strict(first_seen["n_markets"], default=None).alias("nm"))
            only_mkt = float(nm.filter(pl.col("nm") == 1)["usd"].sum() / known["usd"].sum())
    ratio = (b - s) / (b + s) if b + s > 0 else float("nan")
    one_sided = bool(sub.height >= 3 and np.isfinite(ratio) and ratio >= 0.8)
    informed = one_sided and ((np.isfinite(top3) and top3 >= 0.5) or (np.isfinite(fresh) and fresh >= 0.5))
    return {
        "n_trades": int(sub.height), "usd": b + s, "buy_usd": b, "sell_usd": s, "net_ratio": ratio,
        "n_wallets": n_w, "top3_buy_share": top3, "fresh_buy_share": fresh, "only_market_buy_share": only_mkt,
        "age_known_share": known_share, "one_sided": one_sided, "informed_looking": bool(informed),
    }  # fmt: skip


def _concat_existing(snap: Path, names: tuple[str, ...], cols: list[str], tokens: list[str]) -> pl.DataFrame:
    """Rows of the named snapshot files (those that exist) for the given token ids."""
    parts = [
        pl.scan_parquet(snap / f"{n}.parquet").filter(pl.col("token_id").is_in(tokens)).select(cols).collect()
        for n in names
        if (snap / f"{n}.parquet").exists()
    ]
    return pl.concat(parts).unique() if parts else pl.DataFrame(schema={c: pl.Null for c in cols})


def _first_seen_dict(snap: Path) -> dict:
    parts = [
        pl.read_parquet(snap / f, columns=["wallet", "first_seen", "n_markets"]).with_columns(pl.col("n_markets").cast(pl.Int64))
        for f in ("wallets.parquet", "extra_wallets.parquet")
        if (snap / f).exists()
    ]
    w = pl.concat(parts).unique("wallet", keep="first").drop_nulls("first_seen")
    return {
        "first_seen": dict(zip(w["wallet"].to_list(), w["first_seen"].to_list(), strict=True)),
        "n_markets": dict(zip(w["wallet"].to_list(), w["n_markets"].to_list(), strict=True)),
    }


# ----------------------------------------------------------------------------------------- within-event audit


def within_event_population(snap: Path = SNAP) -> pl.DataFrame:
    """The 4A population (multi events, a statistic, >= 3 comparable outcomes), ordered for the audit."""
    we = pl.read_parquet(snap / "within_event.parquet")
    ni = pl.read_parquet(snap / "news_independent.parquet").select(
        pl.col("condition_id").alias("ref_pilot_id"), pl.col("news_ts_indep").alias("reveal_ts")
    )
    pop = we.filter(
        (pl.col("group") == "multi") & pl.col("winner_pct_24h").is_not_null() & (pl.col("n_comparable") >= 3)
    ).join(ni, on="ref_pilot_id", how="left")
    return pop.sort(
        ["winner_pct_24h", "diff_24h", "n_comparable", "condition_id"], descending=[True, True, True, False]
    )


def _series_by_cid(df: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for (cid,), g in df.sort("ts", maintain_order=True).group_by("condition_id", maintain_order=True):
        out[cid] = (g["ts"].to_numpy(), g["price"].to_numpy())
    return out


def audit_within_event(snap: Path = SNAP, n: int = N_AUDIT, log: Callable[[str], None] = print) -> pl.DataFrame:
    """Audit the top ``n`` within-event extremes (24 h window). Returns one row per event (no wallet addresses)."""
    t0 = time.time()
    pop = within_event_population(snap)
    top = pop.head(n)
    em = pl.read_parquet(snap / "event_markets.parquet")
    mk = pl.read_parquet(snap / "markets.parquet").select("condition_id", "question", "start_ts")
    om = pl.read_parquet(snap / "event_outcome_moves.parquet").filter(pl.col("reason").is_null())
    pep = pl.scan_parquet(snap / "event_prices.parquet")
    # winners of the whole population: step times, to see whether many events moved in the same hours
    all_w = pop["condition_id"].to_list()
    ser_w = _series_by_cid(
        pep.filter(pl.col("condition_id").is_in(all_w)).select("condition_id", "ts", "price").collect()
    )
    steps: list[tuple[object, object]] = []  # (reveal, absolute time of the sharp step end) per event
    for r in pop.iter_rows(named=True):
        ts, px = ser_w.get(r["condition_id"], (np.array([], "datetime64[us]"), np.array([])))
        hs, p = to_grid(ts, px, r["reveal_ts"], 97, 1)
        sz, a, b = best_step(p, _idx(hs, 25), _idx(hs, 1), 3)
        rise = p[_idx(hs, 1)] - p[_idx(hs, 25)] if len(p) > 96 else np.nan
        if np.isfinite(sz) and sz >= RULES.min_step and rise > 0 and sz / rise >= RULES.step_share:
            steps.append((r["reveal_ts"], np.datetime64(r["reveal_ts"], "us") - np.timedelta64(int(hs[b]) * 3600, "s")))
        else:
            steps.append((r["reveal_ts"], None))
    log(f"[{time.time() - t0:.0f}s] population {pop.height}; top {top.height} selected; peer step times built")
    win_cids = top["condition_id"].to_list()
    tok = em.filter(pl.col("condition_id").is_in(win_cids)).select("condition_id", "yes_token_id")
    lo = top["reveal_ts"].min() - timedelta(hours=110)
    tape = (
        pl.scan_parquet(snap / "trades_taker.parquet")
        .filter(pl.col("condition_id").is_in(win_cids) & (pl.col("ts") >= lo))
        .select("condition_id", "token_id", "ts", "side", "usd", "size", "wallet")
        .collect()
    )
    fsd = _first_seen_dict(snap)
    log(f"[{time.time() - t0:.0f}s] winner tape rows {tape.height}")
    pos_in_pop = {c: i for i, c in enumerate(all_w)}
    rows = []
    for rk, r in enumerate(top.iter_rows(named=True), 1):
        cid, reveal, eid = r["condition_id"], r["reveal_ts"], r["event_id"]
        anchor = np.datetime64(reveal, "us")
        sib_ids = [c for c in om.filter(pl.col("pilot_condition_id") == r["ref_pilot_id"])["condition_id"].to_list() if c != cid]  # fmt: skip
        ev_px = pep.filter(pl.col("event_id") == eid).select("condition_id", "ts", "price").collect()
        ser = _series_by_cid(ev_px)
        wts, wpx = ser.get(cid, (np.array([], "datetime64[us]"), np.array([])))
        ytok = tok.filter(pl.col("condition_id") == cid)["yes_token_id"].to_list()
        tw = tape.filter((pl.col("condition_id") == cid) & (pl.col("token_id") == (ytok[0] if ytok else "")))
        tape_prices = tw.filter(pl.col("size") > 0).select("ts", (pl.col("usd") / pl.col("size")).alias("price"))
        all_ts = np.concatenate([wts, tape_prices["ts"].to_numpy()]) if tape_prices.height else wts
        all_px = np.concatenate([wpx, tape_prices["price"].to_numpy()]) if tape_prices.height else wpx
        hs, p = to_grid(all_ts, all_px, anchor, 97, 1)
        res = obs_gap_h(all_ts, anchor, 49, 1)
        sibs = [to_grid(*ser[c], anchor, 97, 1)[1] for c in sib_ids if c in ser]
        q = mk.filter(pl.col("condition_id") == cid)
        question = q["question"][0] if q.height else str(r.get("ref_pilot_id", ""))
        start_ts = q["start_ts"][0] if q.height else None
        mo = float((anchor - np.datetime64(start_ts, "us")) / np.timedelta64(1, "h")) if start_ts is not None else None
        peers = sum(
            1 for j, (rv, st) in enumerate(steps)
            if j != pos_in_pop[cid] and st is not None and steps[pos_in_pop[cid]][1] is not None
            and abs((np.datetime64(rv, "us") - np.datetime64(reveal, "us")) / np.timedelta64(1, "h")) <= 1
            and abs((st - steps[pos_in_pop[cid]][1]) / np.timedelta64(1, "h")) <= 3
        )  # fmt: skip

        def fl(sh, eh, tw=tw, anchor=anchor):
            return flow_stats(tw, anchor - np.timedelta64(int((sh + 1) * 3600), "s"), anchor - np.timedelta64(int((eh - 1) * 3600), "s"), fsd)  # fmt: skip

        res_d = classify_winner_path(
            hs, p, res_h=res, siblings=sibs, market_open_h=mo, common_trigger=peers >= RULES.concurrent_min,
            flow=fl, topic=bool(TOPIC_RE.search(question)), require_flow=True,
        )  # fmt: skip
        f24 = flow_stats(tw, anchor - np.timedelta64(25 * 3600, "s"), anchor - np.timedelta64(3600, "s"), fsd)
        sf = res_d.get("flow") or (fl(res_d["step_start_h"], res_d["step_end_h"]) if "step_start_h" in res_d else None)
        rows.append({
            "rank": rk, "event_id": eid, "winner_condition_id": cid, "question": question, "reveal_ts": reveal,
            "n_comparable": r["n_comparable"], "winner_pct_24h": r["winner_pct_24h"], "diff_24h": r["diff_24h"],
            "winner_pct_12h": r["winner_pct_12h"], "win_move_24h": r["win_move_24h"],
            "med_lose_move_24h": r["med_lose_move_24h"],
            "p_start": res_d.get("p_start"), "p_end": res_d.get("p_end"), "rise": res_d.get("rise"),
            "step_size": res_d.get("step_size"), "step_share": res_d.get("step_share"),
            "step_start_h_before_reveal": res_d.get("step_start_h"), "step_end_h_before_reveal": res_d.get("step_end_h"),
            "pre48_rise": res_d.get("pre48_rise"), "obs_spacing_h": res,
            "n_siblings": len(sibs), "peer_events_same_step": peers, "market_open_h_before_reveal": mo,
            "topic_flag": bool(TOPIC_RE.search(question)),
            "n_trades_window": f24["n_trades"], "usd_window": f24["usd"],
            "n_trades_step": sf["n_trades"] if sf else None, "usd_step": sf["usd"] if sf else None,
            "step_net_buy_ratio": sf["net_ratio"] if sf else None, "step_n_wallets": sf["n_wallets"] if sf else None,
            "step_top3_buy_share": sf["top3_buy_share"] if sf else None,
            "step_fresh_buy_share": sf["fresh_buy_share"] if sf else None,
            "step_only_market_buy_share": sf["only_market_buy_share"] if sf else None,
            "triggers": "; ".join(res_d.get("triggers", [])),
            "class": res_d["cls"], "reason": res_d["reason"],
        })  # fmt: skip
        log(f"[{time.time() - t0:.0f}s] {rk}/{top.height} {res_d['cls']}")
    return pl.DataFrame(rows, infer_schema_length=None)


# -------------------------------------------------------------------------------------------- one-shot audit


def oneshot_top(private: Path = PRIVATE, n: int = N_AUDIT) -> tuple[pl.DataFrame, float, int, int]:
    """Top ``n`` valid-cutoff, non-excluded, USD >= 100 winning positions outside the Maduro market."""
    sc = pl.read_parquet(private / "oneshot_scores.parquet")
    flags = pl.read_parquet(private / "wallet_flags.parquet", columns=["wallet", "excluded"])
    ok = sc.join(flags, on="wallet", how="left").filter(~pl.col("excluded").fill_null(False) & ~pl.col("no_valid_cutoff"))
    q = float(ok["score"].quantile(ONE_SHOT_Q))
    ok = ok.with_columns(pl.col("question").str.contains(re.escape(MADURO_Q)).alias("maduro"))
    cand = ok.filter(~pl.col("maduro") & (pl.col("usd") >= MIN_USD)).sort(
        ["score", "usd", "wallet"], descending=[True, True, False]
    )
    return cand.head(n), q, ok.height, int(ok["maduro"].sum())


def audit_oneshot(
    snap: Path = SNAP, private: Path = PRIVATE, results: Path = RESULTS, n: int = N_AUDIT,
    log: Callable[[str], None] = print,
) -> pl.DataFrame:
    """Audit the top one-shot positions. The returned frame CONTAINS wallet addresses: write it to data/private only."""
    t0 = time.time()
    top, q99, _n_ok, _n_mad = oneshot_top(private, n)
    pos = pl.read_parquet(private / "positions.parquet", columns=["wallet", "event_id", "condition_id", "token_id"])
    top = top.join(pos, on=["wallet", "event_id", "condition_id"], how="left")
    mk = pl.read_parquet(snap / "markets.parquet").select("condition_id", "start_ts")
    ms = pl.read_csv(results / "market_scores.csv", columns=["condition_id", "jump_start_ts"], infer_schema_length=0)
    ms = ms.with_columns(pl.col("jump_start_ts").str.to_datetime(strict=False))
    jumps = ms.drop_nulls("jump_start_ts")
    fsd = _first_seen_dict(snap)
    toks = top["token_id"].unique().to_list()
    events = top["event_id"].unique().to_list()
    tape = _concat_existing(snap, ("trades_taker", "extra_trades_taker"), ["token_id", "ts", "side", "usd", "size", "wallet"], toks)  # fmt: skip
    series = _concat_existing(snap, ("prices", "extra_prices"), ["token_id", "ts", "price"], toks)
    pep = (
        pl.scan_parquet(snap / "event_prices.parquet").filter(pl.col("event_id").is_in(events))
        .select("event_id", "condition_id", "ts", "price").collect()
    )  # fmt: skip
    log(f"[{time.time() - t0:.0f}s] tier threshold {q99:.4f}; tape rows {tape.height}; series rows {series.height}")
    rows = []
    for rk, r in enumerate(top.iter_rows(named=True), 1):
        entry_ts = np.datetime64(r["first_ts"], "us")
        cutoff = entry_ts + np.timedelta64(int(r["hours_before_cutoff"] * 3600), "s")
        jump = r["cutoff_method"] == "jump_start"
        end_h = -6 if jump else 1  # a price-based cutoff is the START of the move: look past it
        h_from = int(np.ceil(r["hours_before_cutoff"])) + 48
        tk = tape.filter(pl.col("token_id") == r["token_id"])
        sr = series.filter(pl.col("token_id") == r["token_id"])
        tp = tk.filter(pl.col("size") > 0)
        all_ts = np.concatenate([sr["ts"].to_numpy(), tp["ts"].to_numpy()])
        all_px = np.concatenate([sr["price"].to_numpy(), (tp["usd"] / tp["size"]).to_numpy()])
        hs, p = to_grid(all_ts, all_px, cutoff, h_from, end_h)
        res = obs_gap_h(all_ts, cutoff, h_from, end_h)
        sib = pep.filter((pl.col("event_id") == r["event_id"]) & (pl.col("condition_id") != r["condition_id"]))
        sibs = [to_grid(*v, cutoff, h_from, end_h)[1] for v in _series_by_cid(sib).values()]
        st = mk.filter(pl.col("condition_id") == r["condition_id"])["start_ts"]
        mo = float((cutoff - np.datetime64(st[0], "us")) / np.timedelta64(1, "h")) if st.len() and st[0] is not None else None  # fmt: skip
        js = jumps.filter(pl.col("condition_id") == r["condition_id"])["jump_start_ts"]
        conc = 0
        if jump and js.len():
            t = js[0]
            conc = jumps.filter(
                (pl.col("condition_id") != r["condition_id"])
                & ((pl.col("jump_start_ts") - t).abs() <= pl.duration(hours=2))
            ).height
        topic = bool(TOPIC_RE.search(r["question"]))
        w_ent = tk.filter((pl.col("ts") >= r["first_ts"] - timedelta(hours=3))
                          & (pl.col("ts") <= r["first_ts"] + timedelta(hours=3)))  # fmt: skip
        buys = w_ent.filter(pl.col("side") == "BUY")
        wbuy = float(buys.filter(pl.col("wallet") == r["wallet"])["usd"].sum() or 0.0)
        tot = float(buys["usd"].sum() or 0.0)
        flow = {"wallet_share": wbuy / tot if tot > 0 else float("nan"), "n_trades": int(w_ent.height), "usd": float(w_ent["usd"].sum() or 0.0)}  # fmt: skip
        res_d = classify_oneshot_path(
            hs, p, entry_h=float(r["hours_before_cutoff"]), end_h=end_h, entry_price=float(r["entry_price"]),
            age_days=r["wallet_age_days_at_entry"], only_market=bool(r["only_market_store"]), res_h=res,
            siblings=sibs, market_open_h=mo, concurrent_jumps=conc, flow=flow, topic=topic, require_flow=True,
        )  # fmt: skip
        if jump:
            res_d["reason"] += " [cutoff = price jump start: outcome-selected, the move itself follows the cutoff]"
        fs = fsd["first_seen"].get(r["wallet"])
        rows.append({
            "rank": rk, "wallet": r["wallet"], "question": r["question"], "condition_id": r["condition_id"],
            "score": r["score"], "oneshot_rank_in_pool": r["rank"], "usd": r["usd"], "entry_price": r["entry_price"],
            "hours_before_cutoff": r["hours_before_cutoff"], "cutoff_method": r["cutoff_method"],
            "wallet_age_days_at_entry": r["wallet_age_days_at_entry"], "only_market_store": r["only_market_store"],
            "n_markets_global": r["n_markets_global"], "first_seen_known": fs is not None,
            "p_entry": res_d.get("p_entry"), "pre_rise_24h": res_d.get("pre_rise"), "post_rise": res_d.get("post_rise"),
            "step_size": res_d.get("step_size"), "step_share": res_d.get("step_share"),
            "step_after_entry_h": res_d.get("step_after_entry_h"), "obs_spacing_h": res_d["res_h"],
            "n_siblings": len(sibs), "concurrent_jump_markets": conc, "topic_flag": topic,
            "trades_entry_pm3h": flow["n_trades"], "usd_entry_pm3h": flow["usd"], "wallet_share_of_buys": flow["wallet_share"],
            "triggers": "; ".join(res_d.get("triggers", [])), "class": res_d["cls"], "reason": res_d["reason"],
        })  # fmt: skip
        log(f"[{time.time() - t0:.0f}s] {rk}/{top.height} {res_d['cls']}")
    return pl.DataFrame(rows, infer_schema_length=None)


# ------------------------------------------------------------------------------------------------- report


def _mix(df: pl.DataFrame) -> list[tuple[str, int, float]]:
    n = df.height
    return [(c, int((df["class"] == c).sum()), (df["class"] == c).sum() / n if n else 0.0) for c in CLASSES]


def _examples(df: pl.DataFrame, k: int = 4) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in CLASSES:
        qs = df.filter(pl.col("class") == c).sort("rank")["question"].unique(maintain_order=True).to_list()
        out[c] = [q.replace("|", "/") for q in qs[:k]]
    return out


def _count(df: pl.DataFrame, text: str) -> int:
    return int(df["reason"].str.contains(text, literal=True).sum())


def _notes(we: pl.DataFrame, osh: pl.DataFrame) -> list[str]:
    peers = int(((we["class"] == "public_news_reaction") & (we["peer_events_same_step"] >= RULES.concurrent_min)).sum())
    return [
        f"* Within-event: {peers} of the {int((we['class'] == 'public_news_reaction').sum())} public-news rows are events that "
        f"share a reveal time with >= {RULES.concurrent_min} other events whose winners stepped in the same hours (election "
        "markets that all reprice together on the same count or poll), not an individual market's signal.",
        f"* Within-event `unclear`: {_count(we, 'too sparse')} rows have price observations 12 h apart (no step can be "
        f"seen), {_count(we, 'frequent public')} are sharp steps in poll/results/leaderboard-type markets, "
        f"{_count(we, 'before the reveal: cannot')} are sharp steps in the last 3 h, {_count(we, 'not one-sided')} are "
        f"sharp steps whose tape is balanced or spread over many wallets, {_count(we, 'no visible rise')} show no visible "
        "rise (the percentile comes from siblings).",
        f"* Within-event `predictable_convergence`: {_count(we, 'favourite already high')} are favourites already high "
        f"with no visible move, {_count(we, 'firming up') + _count(we, 'gradual rise')} are gradual or favourite-firming "
        f"rises, {_count(we, 'already rising')} were already rising before the window.",
        f"* One-shot: {int((osh['cutoff_method'] == 'jump_start').sum())} of {osh.height} rows have a jump-start (outcome-selected) "
        f"cutoff; {_count(osh, 'visible elsewhere')} entered before a sharp step that other markets or related outcomes "
        "also made (consistent with public news; whether it was public at the entry cannot be judged); "
        f"{_count(osh, 'no visible price move')} show no price move before the window end.",
        "* No row's class rests on a news or social-media source: a row labelled `plausible_leak` is one the offline "
        "rules could not explain by public triggers, nothing more.",
    ]


def render_md(we: pl.DataFrame, osh: pl.DataFrame, meta: dict) -> str:
    L = ["# Hand audit of the detectors' extremes (phase 4E)", ""]
    L += [
        "Aggregate-only: counts, shares and market questions; no wallet addresses and no event or wallet identifiers. "
        "Row-level reasons are in `data/private/audit_within_event.csv` and `data/private/audit_oneshot.csv` "
        "(the second contains wallet addresses and stays in `data/private/`).",
        "",
        "**This is a descriptive audit with no ground truth.** The classes are produced by a fixed rule set over offline "
        "data (see the criteria), written before looking at the results. A human reading the news or the source feed "
        "would classify some rows differently. A class says what the offline price, trade and wallet data look like; "
        "it is not evidence about any person or wallet, and 'plausible leak' means only that the offline data cannot "
        "exclude one.",
        "",
        "## The two lists",
        "",
        f"* **Within-event, top {we.height}**: real multi-outcome events with >= 3 comparable outcomes and a computed "
        f"statistic ({meta['pop_we']} events), ordered by the winner's 24 h within-event percentile (descending), then "
        "the winner-minus-median-loser log-odds difference `diff_24h`, then the number of comparable outcomes (all "
        "descending), then the winner's condition id ascending (only to make the order deterministic). The percentile "
        "is a mid-rank, so large events tie near 0.97-0.98 and the tie-break decides the order. Percentile values "
        f"in the list: {we['winner_pct_24h'].min():.3f} to {we['winner_pct_24h'].max():.3f}.",
        f"* **One-shot, top {osh.height}**: real winning positions with a valid cutoff of non-excluded wallets, USD >= "
        f"{MIN_USD:.0f} (the watchlist's size rule), outside the Maduro market (which holds {meta['n_maduro']} of the "
        f"{meta['n_valid']} valid positions and the top ranks), ordered by the one-shot score (descending), then USD "
        f"(descending), then wallet (ascending, deterministic only). Tier threshold (99th percentile of valid-cutoff "
        f"scores): {meta['q99']:.4f}. Cutoff method of the list: "
        + ", ".join(f"{k} {v}" for k, v in osh["cutoff_method"].value_counts().rows())
        + ". A `jump_start` cutoff is the start of the price move itself, so those rows are outcome-selected and "
        "the move follows the cutoff by construction.",
        "",
        "## Mix of classes",
        "",
        "| class | within-event n | within-event share | one-shot n | one-shot share |",
        "|---|---|---|---|---|",
    ]
    mw, mo = _mix(we), _mix(osh)
    for (c, a, sa), (_, b, sb) in zip(mw, mo, strict=True):
        L.append(f"| {c} | {a} | {sa:.0%} | {b} | {sb:.0%} |")
    L += [
        f"| total | {we.height} | 100% | {osh.height} | 100% |",
        "",
        "## Criteria (fixed in advance, same for every row)",
        "",
        "Price path: the winner's forward-filled price on an hourly grid, built from the Gamma price history "
        "(3-hourly for most events) plus the winner's YES-token taker trades where they exist. A path with observations "
        f"more than {RULES.max_res_h:.0f} h apart cannot show a step and is `unclear`. A **step** is the largest rise "
        f"over a short lag (>= 2 h, or the observation spacing); it is **sharp** when it is >= {RULES.min_step:.2f} "
        f"probability points and >= {RULES.step_share:.0%} of the rise over the whole window.",
        "",
        f"* `predictable_convergence`: the winner rose by less than {RULES.min_rise:.2f} while already a favourite "
        f"(>= {RULES.high_p:.1f}); or the rise is gradual (no sharp step), including a favourite firming up; or it "
        "was already rising strongly in the 48 h before the window. One-shot: a gradual rise after the entry, or a "
        "favourite bought with no visible move.",
        "* `public_news_reaction`: a sharp step with at least one visible public trigger: a related outcome of the "
        "same event moved (either direction) before the winner's step, other events with the same reveal time stepped "
        f"in the same hours (>= {RULES.concurrent_min} peers; for jump-start cutoffs, >= {RULES.concurrent_min} other "
        f"markets repricing within 2 h), a supplied known release time, or the step within {RULES.open_h:.0f} h of the "
        "market opening. One-shot: the entry came after a rise of >= "
        f"{RULES.pre_chase:.2f} in the preceding 24 h.",
        f"* `plausible_leak`: a sharp step that ends at least {RULES.early_h:.0f} h before the end of the window, no "
        "public trigger above, not a polls/results/leaderboard/earnings-type market, AND tape evidence of one-sided "
        "buying (>= 3 trades, net buy ratio >= 0.8) that is concentrated (top-3 wallets >= 50% of buy USD) or from "
        f"fresh wallets (>= 50% of known-age buy USD from wallets first seen <= {RULES.fresh_days:.0f} days earlier). "
        "One-shot: a fresh (<= 14 days) or single-market wallet bought before a sharp step, with the wallet at least "
        f"{RULES.flow_share:.0%} of winner-side buy USD within 3 h of its entry, and no trigger above. Without tape "
        "the row stays `unclear`.",
        "* `unclear`: everything else, including sparse or missing price data, steps in the last 3 h before the reveal "
        "(cannot be separated from the first public results), sharp steps in markets of a type with frequent public "
        "intermediate data, one-shot entries that preceded a step also made by other markets (not separable from early "
        "public information), and rows where the winner did not visibly move at all (the percentile or score then comes "
        "from the siblings, the entry price, the wallet's freshness or the size).",
        "",
        "## Examples (market questions only)",
        "",
    ]
    for title, df in (("Within-event", we), ("One-shot", osh)):
        L.append(f"### {title}")
        L.append("")
        for c, qs in _examples(df).items():
            L.append(f"* **{c}**: " + ("; ".join(f"\"{q}\"" for q in qs) if qs else "(none)"))
        L.append("")
    L += ["## Notes on the mix", ""] + _notes(we, osh) + [
        "",
        "## What the offline data cannot decide",
        "",
        "* Whether a step followed public information: no news, social-media or source feed was read; the only "
        "public-trigger evidence is another market moving first, a market-wide repricing at the same hour, a supplied "
        "release time (none exists offline) and the market opening time.",
        "* Whether the buyer knew anything: a one-sided, fresh-wallet buy before a step is also what an early, correct "
        "but public-information buyer looks like. Trade-time resolution is the tape's; the Gamma price history is "
        "3-hourly for most events, so 'sharp' means 'within about 3 h' there.",
        "* Wallet history is partial: first_seen is known for only part of the wallets (store backfill and top-up); "
        "unknown ages are not treated as fresh.",
        "* Events with many siblings put most of the percentile on the siblings: with >= 10 comparable outcomes and "
        "stale low-probability siblings (a zero median loser move), a winner that merely moves a little ranks at the "
        "top. The within-event validation found that 53.7% of such real winners already exceed percentile 0.95; "
        "the placebo shows the same ranking two days earlier.",
        "* Jump-start cutoffs (part of the one-shot list) are chosen from the price path itself, so the 'before any "
        "visible public trigger' test is weakest exactly there.",
        "",
        "## Limitations",
        "",
        "* No ground truth. A human reading news would classify differently; the mix is a description of what these "
        "lists look like offline, not an estimate of how many rows are real leaks.",
        "* The classes are not evidence about any person, wallet or market operator. 'Anomalous' is a statistical "
        "statement.",
        "* Rule thresholds are fixed, simple and coarse; different thresholds move rows between classes (the default "
        "for an undecidable row is `unclear`, which is therefore expected to be large).",
        "* Tiny lists (30 each): shares carry large sampling noise and say nothing about the rest of either ranking.",
        "* The within-event detector failed its own criterion (false-positive rate 11.7% > 10%, both placebos fail) "
        "and the one-shot screen reaches no cell with >= 50% detection; this audit does not rescue either.",
        "",
    ]
    return "\n".join(L)


def run(
    snap: Path = SNAP, private: Path = PRIVATE, results: Path = RESULTS, n: int = N_AUDIT,
    log: Callable[[str], None] = print,
) -> dict:
    """Run both audits; write the two private CSVs and the aggregate-only markdown (checked for addresses)."""
    t0 = time.time()
    we = audit_within_event(snap, n, log)
    osh = audit_oneshot(snap, private, results, n, log)
    _, q99, n_valid, n_mad = oneshot_top(private, 1)
    meta = {"pop_we": within_event_population(snap).height, "q99": q99, "n_valid": n_valid, "n_maduro": n_mad}
    private.mkdir(parents=True, exist_ok=True)
    we.write_csv(private / "audit_within_event.csv")
    osh.write_csv(private / "audit_oneshot.csv")
    text = render_md(we, osh, meta)
    assert_no_addresses(text, "validation_hand_audit.md")
    results.mkdir(parents=True, exist_ok=True)
    (results / "validation_hand_audit.md").write_text(text, encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s")
    return {"within_event": we, "oneshot": osh, "meta": meta}
