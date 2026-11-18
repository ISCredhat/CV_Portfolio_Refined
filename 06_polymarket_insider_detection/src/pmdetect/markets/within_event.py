"""Within-event pre-reveal test (phase 2c): "rational evolution", no peer band.

For each market with an independent news time (the pilot run: 115 of the 399 pilot markets; now the whole universe) (``news_independent.parquet``,
``independent == True``) the Gamma event is fetched with ALL sibling outcome markets and the
YES-token price history of each. Per outcome the statistic is the change in log-odds over the
last 24 h and last 12 h before the reveal, with the window ending ``BUFFER_H`` = 1 h before
``news_ts_indep`` (same buffer as ``series.py``)::

    move_w = logit(p(end)) - logit(p(end - w)),   p clipped to [0.005, 0.995]

where p(t) is the last observation at or before t (forward fill, never backward fill; no
observation at or before the window start -> null move, reason ``no_price_at_window_start``).
Observations are the ``/v2/prices-history`` points of the YES token, plus the YES-token trade
prices of the pilot market itself when ``trades.parquet`` has them.

Groups
------
* ``multi``: a mutually exclusive event (Gamma ``negRisk``) with exactly one sibling resolved YES.
  Per event: winner's move minus the median losing outcome's move (``diff_*``) and the winner's
  percentile among all comparable outcomes (``winner_pct_*``; mid-rank, 0..1).
  One row per event, keyed on the event WINNER's condition_id (``winner_is_pilot`` says whether
  it is a pilot market; the window uses the winner's reveal time if it is a pilot, else the
  reveal time of ``ref_pilot_id``, a pilot sibling). Every other pilot market of the event gets
  a row with null statistics and reason ``pilot_market_not_event_winner``, so nothing is
  dropped; downstream joins on condition_id therefore only score markets that actually won.
* ``binary``: single-market events, and markets of non-exclusive events (``reason`` says
  ``non_exclusive_event``, each treated as its own yes/no question). ``win_move_*`` is the
  log-odds move of the eventual winning SIDE (YES price if YES won, minus the YES move if NO won).
  ``winner_pct_*`` is the percentile among the other binary markets that resolved to the same
  side (a NO winner drifts up through time decay; mixing sides would flatter YES winners).
  ``diff_*`` is the move minus the median of those other markets; ``med_lose_move_*`` is null.

Outcomes that cannot be compared are NOT dropped silently: they stay in ``event_markets.parquet``
and ``outcome_moves`` with a reason: ``resolved_before_window_end`` (closed before the window
end: frozen price, uninformative), ``no_price_history``, ``no_price_at_window_start``,
``stale_price`` (last observation older than ``MAX_STALE_H``).

Placebo (pooled over events): share of events whose winner move exceeds the 95th percentile of
that event's comparable losers, a permutation null (the winner drawn uniformly at random among
the event's comparable outcomes, ``n_perm`` draws) giving a one-sided p-value for the observed
mean and median of ``winner - median loser`` and for that share.

Nothing here opens DuckDB; fetched data goes to Parquet under ``data/snapshots``.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.collect.data_api import _ts, fetch_price_series
from pmdetect.collect.gamma import GAMMA, parse_ts
from pmdetect.collect.http import get_json

SNAPSHOTS = Path(__file__).resolve().parents[3] / "data" / "snapshots"
EVENTS_DIR = SNAPSHOTS.parent / "tapes" / "events"  # per-event part files (incremental fetch)
BUFFER_H = 1
WINDOWS_H = (24, 12)
CLIP = (0.005, 0.995)
MAX_STALE_H = 48.0
EVENT_MARKETS = "event_markets.parquet"
EVENT_PRICES = "event_prices.parquet"
WITHIN_EVENT = "within_event.parquet"
OUTCOME_MOVES = "event_outcome_moves.parquet"
# Cost model (measured on the 108-event pilot): the price ladder (600 / 1800 / 10800 / 43200 s)
# costs the rung index + 1 requests per series (400 on a too-fine bucket), 4 when no rung has
# data; over the pilot's 1,694 series that averaged 3.43 requests, plus one Gamma call per event.
REQ_PER_SERIES = 3.43  # measured with the full 4-rung ladder on the pilot
# The within-event test only uses 24 h / 12 h windows, so the 3 h and 12 h grains are enough:
# the coarse ladder costs 1 request when the 3 h grain has data, 2 when only 12 h (or 400 + data),
# 2 when none; with ~46% empty series that averages about 1.5 requests per series.
COARSE_LADDER = (10800, 43200)
REQ_PER_SERIES_COARSE = 1.6
RATE_RPS = 5.0

MARKETS_SCHEMA = {
    "event_id": pl.String,
    "condition_id": pl.String,
    "question": pl.String,
    "label": pl.String,
    "yes_token_id": pl.String,
    "outcome": pl.String,
    "end_ts": pl.Datetime("us"),
    "closed_ts": pl.Datetime("us"),
    "neg_risk": pl.Boolean,
    "is_pilot": pl.Boolean,
    "n_event_markets": pl.Int64,
    "prices_fetched": pl.Boolean,
    "n_price_points": pl.Int64,
    "bucket_seconds": pl.Int64,
}
PRICES_SCHEMA = {
    "event_id": pl.String,
    "condition_id": pl.String,
    "token_id": pl.String,
    "ts": pl.Datetime("us"),
    "price": pl.Float64,
    "source": pl.String,
}


# --------------------------------------------------------------------------- math


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), *CLIP)
    return np.log(p / (1.0 - p))


def last_at_or_before(
    ts: np.ndarray, px: np.ndarray, t: np.datetime64
) -> tuple[float, float] | None:
    """(price, age_hours) of the last observation at or before ``t`` (forward fill), else None."""
    if len(ts) == 0:
        return None
    i = int(np.searchsorted(ts, t, side="right")) - 1
    if i < 0:
        return None
    return float(px[i]), float((t - ts[i]) / np.timedelta64(1, "h"))


def log_odds_move(
    ts: np.ndarray, px: np.ndarray, end: np.datetime64, window_h: float
) -> tuple[float | None, str | None]:
    """Change in clipped log-odds over (end - window_h, end]; (None, reason) if not computable."""
    if len(ts) == 0:
        return None, "no_price_history"
    start = end - np.timedelta64(int(window_h * 3600), "s")
    a = last_at_or_before(ts, px, start)
    b = last_at_or_before(ts, px, end)
    if a is None or b is None:
        return None, "no_price_at_window_start"
    if b[1] > MAX_STALE_H:
        return None, "stale_price"
    return float(logit(b[0]) - logit(a[0])), None


def mid_percentile(x: float, pool: np.ndarray) -> float:
    """Mid-rank percentile of x within pool (pool includes x): (#less + 0.5 #equal) / n."""
    pool = np.asarray(pool, dtype=float)
    return float(((pool < x).sum() + 0.5 * (pool == x).sum()) / len(pool))


def event_stats(moves: np.ndarray, winner_idx: int) -> dict:
    """Winner-vs-losers statistics for one event; ``moves`` are comparable outcomes only."""
    moves = np.asarray(moves, dtype=float)
    losers = np.delete(moves, winner_idx)
    w = float(moves[winner_idx])
    return {
        "win": w,
        "med_lose": float(np.median(losers)) if len(losers) else None,
        "diff": w - float(np.median(losers)) if len(losers) else None,
        "pct": mid_percentile(w, moves),
        "q95_lose": float(np.quantile(losers, 0.95)) if len(losers) else None,
    }


def permutation_null(
    events: list[tuple[np.ndarray, int]], n_perm: int = 10000, seed: int = 0
) -> dict:
    """Winner chosen uniformly at random within each event, versus the observed winners.

    events: list of (moves of comparable outcomes, observed winner index). Statistics: mean and
    median of (winner - median loser) over events and the share of events with winner above
    the 95th percentile of its losers. p-values are one-sided (>= observed).

    The per-event statistic of EVERY candidate winner is precomputed once, so each permutation is a
    gather over a flat array (the draws, ``rng.integers(0, n)`` event by event, are unchanged, so
    results equal the earlier loop implementation for the same seed).
    """
    rng = np.random.default_rng(seed)
    ev = [(np.asarray(m, float), w) for m, w in events if len(m) >= 2]
    if not ev:
        return {"n_events": 0}
    diffs, flags, offs, sizes = [], [], [], []
    off = 0
    for m, _ in ev:
        st = [event_stats(m, k) for k in range(len(m))]
        diffs.append([x["diff"] for x in st])
        flags.append([x["win"] > x["q95_lose"] for x in st])
        offs.append(off)
        sizes.append(len(m))
        off += len(m)
    diff_flat = np.concatenate(diffs).astype(float)
    flag_flat = np.concatenate(flags).astype(float)
    offs_a = np.array(offs)
    obs_idx = offs_a + np.array([w for _, w in ev])
    obs = (
        float(diff_flat[obs_idx].mean()),
        float(np.median(diff_flat[obs_idx])),
        float(flag_flat[obs_idx].mean()),
    )
    null = np.empty((n_perm, 3))
    for b in range(n_perm):
        idx = offs_a + np.array([rng.integers(0, n) for n in sizes])
        d = diff_flat[idx]
        null[b] = (d.mean(), np.median(d), flag_flat[idx].mean())
    p = [(1 + float((null[:, i] >= obs[i]).sum())) / (n_perm + 1) for i in range(3)]
    return {
        "n_events": len(ev),
        "obs_mean_diff": obs[0],
        "obs_median_diff": obs[1],
        "obs_share_above_q95": obs[2],
        "null_mean_diff": float(null[:, 0].mean()),
        "null_median_diff": float(null[:, 1].mean()),
        "null_share_above_q95": float(null[:, 2].mean()),
        "p_mean_diff": p[0],
        "p_median_diff": p[1],
        "p_share_above_q95": p[2],
    }


# --------------------------------------------------------------------------- fetching


def _loads(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v:
        try:
            out = json.loads(v)
            return out if isinstance(out, list) else []
        except json.JSONDecodeError:
            return []
    return []


def parse_event_markets(event: dict, event_id: str, pilot_ids: set[str]) -> list[dict]:
    """Flatten a Gamma event into one row per sibling market with its YES token and outcome."""
    mk = event.get("markets") or []
    neg = bool(event.get("negRisk") or event.get("enableNegRisk"))
    rows = []
    for m in mk:
        cid = (m.get("conditionId") or "").lower()
        if not cid:
            continue
        outs = [str(o) for o in _loads(m.get("outcomes"))]
        toks = [str(t) for t in _loads(m.get("clobTokenIds"))]
        prices = []
        for p in _loads(m.get("outcomePrices")):
            try:
                prices.append(float(p))
            except (TypeError, ValueError):
                prices.append(float("nan"))
        yi = outs.index("Yes") if "Yes" in outs else 0
        outcome = None
        if len(prices) == len(outs) == 2 and outs[yi] == "Yes":
            if prices[yi] >= 0.999:
                outcome = "YES"
            elif prices[yi] <= 0.001:
                outcome = "NO"
        rows.append(
            {
                "event_id": event_id,
                "condition_id": cid,
                "question": m.get("question"),
                "label": m.get("groupItemTitle"),
                "yes_token_id": toks[yi] if len(toks) > yi else None,
                "outcome": outcome,
                "end_ts": parse_ts(m.get("endDate")),
                "closed_ts": parse_ts(m.get("closedTime")),
                "neg_risk": neg,
                "is_pilot": cid in pilot_ids,
                "n_event_markets": len(mk),
            }
        )
    return rows


def independent_markets(snapshot_dir: Path = SNAPSHOTS) -> pl.DataFrame:
    """Every market with an independent news time (universe-wide), with event_id, outcome, question.

    Reads ``news_independent.parquet`` (all universe markets) joined to ``markets.parquet``.
    """
    m = pl.read_parquet(snapshot_dir / "markets.parquet").select(
        "condition_id", "event_id", "question", "outcome"
    )
    n = pl.read_parquet(snapshot_dir / "news_independent.parquet")
    return n.filter(pl.col("independent")).join(m, on="condition_id", how="left")


pilot_independent = independent_markets  # name used by the 108-event pilot run


def _atomic_write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# --- incremental per-event part files ---------------------------------------------------------
#
# data/tapes/events/markets/<event_id>.parquet   one row per sibling market (completion marker)
# data/tapes/events/prices/<event_id>.parquet    YES-token price history of the fetched siblings
#
# A part is written atomically (tmp file + rename). The prices part is written BEFORE the markets
# part, so an event counts as done only if its markets part exists; a crash in between leaves an
# orphan prices part that is simply overwritten on the retry. Cost of one event is O(its size), not
# O(everything fetched so far). ``consolidate`` concatenates the parts into the snapshot files.


def _part_paths(parts_dir: Path, eid: str) -> tuple[Path, Path]:
    safe = "".join(ch for ch in str(eid) if ch.isalnum() or ch in "-_")
    return parts_dir / "markets" / f"{safe}.parquet", parts_dir / "prices" / f"{safe}.parquet"


def done_events(parts_dir: Path = EVENTS_DIR) -> set[str]:
    """Event ids with a completed markets part."""
    d = parts_dir / "markets"
    return {p.stem for p in d.glob("*.parquet")} if d.exists() else set()


def write_event_parts(parts_dir: Path, eid: str, mk: pl.DataFrame, pr: pl.DataFrame) -> None:
    mpath, ppath = _part_paths(parts_dir, eid)
    if pr.height:
        _atomic_write(pr.cast(PRICES_SCHEMA), ppath)
    elif ppath.exists():
        ppath.unlink()
    _atomic_write(mk.cast(MARKETS_SCHEMA), mpath)  # completion marker: last


def fetch_one_event(
    eid: str, independent_ids: set[str], cutoff, get=None, series_fn=None
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Gamma event + YES-token price histories of its siblings -> (markets, prices, stats).

    ``cutoff`` = latest independent reveal of the event minus the buffer: a sibling that closed
    before it and is not itself an independent market is skipped (frozen price, uninformative; it
    stays in the table with ``prices_fetched`` False). ``get`` / ``series_fn`` default to the
    module-level ``get_json`` / ``fetch_price_series`` (looked up at call time so tests can mock
    them); the price series uses the same bucket ladder as ``collect_prices``.
    """
    get = get or get_json
    series_fn = series_fn or (lambda t: fetch_price_series(t, COARSE_LADDER))
    ev = get(f"{GAMMA}/events/{eid}")
    rows = parse_event_markets(ev, eid, independent_ids)
    frames = []
    calls = 0
    for r in rows:
        r["prices_fetched"] = False
        r["n_price_points"] = 0
        r["bucket_seconds"] = None
        early = r["closed_ts"] is not None and r["closed_ts"] < cutoff
        if not r["yes_token_id"] or (early and not r["is_pilot"]):
            continue
        pts, bucket = series_fn(r["yes_token_id"])
        calls += 1
        r["prices_fetched"] = True
        r["n_price_points"] = len(pts)
        r["bucket_seconds"] = bucket
        if pts:
            frames.append(
                pl.DataFrame(
                    {
                        "event_id": [eid] * len(pts),
                        "condition_id": [r["condition_id"]] * len(pts),
                        "token_id": [r["yes_token_id"]] * len(pts),
                        "ts": [_ts(p["timestamp"]) for p in pts],
                        "price": [float(p["price"]) for p in pts],
                        "source": ["history"] * len(pts),
                    },
                    schema=PRICES_SCHEMA,
                )
            )
    mk = pl.DataFrame(rows, schema=MARKETS_SCHEMA)
    pr = pl.concat(frames) if frames else pl.DataFrame(schema=PRICES_SCHEMA)
    return mk, pr, {"siblings": len(rows), "price_calls": calls, "with_prices": len(frames)}


def migrate_snapshot_to_parts(snapshot_dir: Path = SNAPSHOTS, parts_dir: Path = EVENTS_DIR) -> int:
    """Import events from an existing ``event_markets``/``event_prices`` snapshot (the 108-event
    pilot) as part files, history rows only, skipping events that already have parts. Returns the
    number of events imported."""
    mpath, ppath = snapshot_dir / EVENT_MARKETS, snapshot_dir / EVENT_PRICES
    if not mpath.exists():
        return 0
    mk = pl.read_parquet(mpath)
    pr = pl.read_parquet(ppath) if ppath.exists() else pl.DataFrame(schema=PRICES_SCHEMA)
    pr = pr.filter(pl.col("source") == "history")
    have = done_events(parts_dir)
    pr_by = {k[0]: g for k, g in pr.partition_by("event_id", as_dict=True).items()}
    n = 0
    for (eid,), g in mk.partition_by("event_id", as_dict=True).items():
        if str(eid) in have:
            continue
        write_event_parts(parts_dir, str(eid), g, pr_by.get(eid, pl.DataFrame(schema=PRICES_SCHEMA)))
        n += 1
    return n


def fetch_event_data(
    snapshot_dir: Path = SNAPSHOTS,
    parts_dir: Path = EVENTS_DIR,
    log=print,
    max_events: int | None = None,
    event_ids: list[str] | None = None,
    migrate: bool = True,
) -> dict:
    """Fetch every event of every independent market, incrementally and resumably.

    One Gamma event call plus one price-ladder call per fetched sibling, a pair of small
    atomic part files per event (see above). Events with completed parts are skipped, so the
    script can be stopped and restarted at any time. Existing pilot snapshot events are imported
    first (``migrate``) and never refetched. Returns counts; failures are logged and retried on
    the next run (no part is written for them).
    """
    ind = independent_markets(snapshot_dir).filter(pl.col("event_id").is_not_null())
    independent_ids = set(ind["condition_id"])
    imported = migrate_snapshot_to_parts(snapshot_dir, parts_dir) if migrate else 0
    events = ind["event_id"].unique().sort().to_list()
    if event_ids is not None:
        events = [e for e in events if e in set(event_ids)]
    last_end = ind.group_by("event_id").agg(pl.col("news_ts_indep").max().alias("t"))
    last_end = dict(zip(last_end["event_id"], last_end["t"], strict=True))
    done = done_events(parts_dir)
    todo = [e for e in events if e not in done]
    if max_events is not None:
        todo = todo[:max_events]
    stats = {"events": 0, "siblings": 0, "price_calls": 0, "failed": [], "imported": imported,
             "already_done": len(events) - len([e for e in events if e not in done]), "todo": len(todo)}  # fmt: skip
    log(f"events: {len(events)} total, {stats['already_done']} done, {len(todo)} to fetch")
    t0 = time.time()
    for i, eid in enumerate(todo):
        cutoff = last_end[eid] - timedelta(hours=BUFFER_H)
        try:
            mk, pr, st = fetch_one_event(eid, independent_ids, cutoff)
        except Exception as exc:  # noqa: BLE001 - record and continue; retried on the next run
            stats["failed"].append((eid, str(exc)[:80]))
            log(f"[{i + 1}/{len(todo)}] event {eid} FAILED {exc}")
            continue
        write_event_parts(parts_dir, eid, mk, pr)
        stats["events"] += 1
        stats["siblings"] += st["siblings"]
        stats["price_calls"] += st["price_calls"]
        el = time.time() - t0
        eta = el / (i + 1) * (len(todo) - i - 1) / 3600
        log(
            f"[{i + 1}/{len(todo)}] event {eid}: {st['siblings']} siblings, {st['with_prices']} "
            f"with prices, calls={stats['price_calls']} elapsed={el / 60:.1f}min eta={eta:.1f}h "
            f"t={time.strftime('%H:%M:%S')}"
        )
    return stats


def trade_price_rows(snapshot_dir: Path, mk: pl.DataFrame) -> pl.DataFrame:
    """YES-token trade prices (source='trade') of every sibling market that has a tape.

    Two sources, no double counting: markets present in the all-rows ``trades.parquet`` use its
    YES-token rows (what the pilot run used); markets absent from it but present in the full taker
    tape ``trades_taker.parquet`` use that tape mapped to the YES price (a fill on the NO token at
    price p is a YES price of 1 - p). Markets without any tape get none.
    """
    mk = mk.filter(pl.col("yes_token_id").is_not_null())
    meta = mk.select("condition_id", "event_id", "yes_token_id").unique(subset="condition_id")
    frames = []
    covered: set[str] = set()
    tpath = snapshot_dir / "trades.parquet"
    if tpath.exists():
        tr = (
            pl.scan_parquet(tpath)
            .filter(pl.col("token_id").is_in(meta["yes_token_id"].implode()))
            .select("condition_id", "token_id", pl.col("ts").cast(pl.Datetime("us")), "price")
            .collect()
        )
        if tr.height:
            covered = set(tr["condition_id"].unique())
            frames.append(tr.select("condition_id", "ts", "price"))
        all_cids = pl.scan_parquet(tpath).select("condition_id").unique().collect()["condition_id"]
        covered |= set(all_cids)
    kpath = snapshot_dir / "trades_taker.parquet"
    if kpath.exists():
        rest = meta.filter(~pl.col("condition_id").is_in(list(covered)))
        if rest.height:
            tk = (
                pl.scan_parquet(kpath)
                .filter(pl.col("condition_id").is_in(rest["condition_id"].implode()))
                .select("condition_id", "token_id", pl.col("ts").cast(pl.Datetime("us")), "price")
                .collect()
                .join(rest.select("condition_id", "yes_token_id"), on="condition_id")
                .select(
                    "condition_id",
                    "ts",
                    pl.when(pl.col("token_id") == pl.col("yes_token_id"))
                    .then(pl.col("price"))
                    .otherwise((1.0 - pl.col("price")).round(6))
                    .alias("price"),
                )
            )
            if tk.height:
                frames.append(tk)
    if not frames:
        return pl.DataFrame(schema=PRICES_SCHEMA)
    tr = pl.concat(frames).join(meta, on="condition_id")
    return (
        tr.select(
            "event_id",
            "condition_id",
            pl.col("yes_token_id").alias("token_id"),
            "ts",
            "price",
            pl.lit("trade").alias("source"),
        )
        .cast(PRICES_SCHEMA)
    )


def consolidate(snapshot_dir: Path = SNAPSHOTS, parts_dir: Path = EVENTS_DIR, log=print) -> dict:
    """Concatenate the part files into ``event_markets.parquet`` / ``event_prices.parquet``.

    The price file = history rows from the parts + trade-price rows rebuilt from the tapes
    (``trade_price_rows``), so the step is idempotent and can be repeated after more events or a
    refreshed ``trades.parquet``. Both files are replaced atomically.
    """
    mdir, pdir = parts_dir / "markets", parts_dir / "prices"
    mfiles = sorted(mdir.glob("*.parquet")) if mdir.exists() else []
    if not mfiles:
        raise FileNotFoundError(f"no event parts under {mdir}")
    mk = pl.concat([pl.read_parquet(f) for f in mfiles]).cast(MARKETS_SCHEMA)
    pfiles = sorted(pdir.glob("*.parquet")) if pdir.exists() else []
    done = {f.stem for f in mfiles}
    pr = (
        pl.concat([pl.read_parquet(f) for f in pfiles if f.stem in done])
        if pfiles
        else pl.DataFrame(schema=PRICES_SCHEMA)
    ).cast(PRICES_SCHEMA)
    tr = trade_price_rows(snapshot_dir, mk)
    _atomic_write(mk, snapshot_dir / EVENT_MARKETS)
    _atomic_write(pl.concat([pr, tr]), snapshot_dir / EVENT_PRICES)
    out = {"events": len(mfiles), "markets": mk.height, "history_rows": pr.height, "trade_rows": tr.height}
    log(f"consolidated: {out}")
    return out


def add_trade_prices(snapshot_dir: Path = SNAPSHOTS) -> int:
    """Rebuild the trade-price rows of event_prices.parquet in place (history rows kept)."""
    mpath, ppath = snapshot_dir / EVENT_MARKETS, snapshot_dir / EVENT_PRICES
    if not mpath.exists() or not ppath.exists():
        return 0
    mk = pl.read_parquet(mpath)
    pr = pl.read_parquet(ppath).filter(pl.col("source") != "trade")
    tr = trade_price_rows(snapshot_dir, mk)
    _atomic_write(pl.concat([pr, tr]), ppath)
    return tr.height


def estimate_fetch_cost(snapshot_dir: Path = SNAPSHOTS, parts_dir: Path = EVENTS_DIR) -> dict:
    """Request and hour estimate for ``fetch_event_data`` from the stored data (no network).

    Events = distinct event_id of independent markets. Siblings per event are counted from
    ``markets.parquet`` (every stored market of the event) times ``sibling_scale``, the ratio of
    Gamma siblings to stored siblings measured on events already fetched (1.0 without data).
    Price series are fetched for siblings that did not close before the event's last independent
    reveal (stored ``closed_ts``), plus the independent markets themselves.
    Each series costs ``REQ_PER_SERIES`` HTTP requests (the 10 min / 30 min / 3 h / 12 h bucket
    ladder tries finer buckets first; measured on the pilot as wall time at ~5 req/s over series
    calls) and each event one Gamma call. Cached responses are free, so events already fetched
    are excluded.
    """
    ind = independent_markets(snapshot_dir).filter(pl.col("event_id").is_not_null())
    m = pl.read_parquet(snapshot_dir / "markets.parquet").select(
        "condition_id", "event_id", "closed_ts"
    )
    last_end = ind.group_by("event_id").agg(pl.col("news_ts_indep").max().alias("_end"))
    sib = (
        m.join(last_end, on="event_id", how="inner")
        .with_columns(
            pl.col("condition_id").is_in(ind["condition_id"].implode()).alias("_ind"),
            (pl.col("closed_ts") < pl.col("_end") - pl.duration(hours=BUFFER_H)).fill_null(False).alias("_early"),
        )
    )
    per = sib.group_by("event_id").agg(
        pl.len().alias("n_stored"),
        (pl.col("_ind") | ~pl.col("_early")).sum().alias("n_series"),
    )
    done = done_events(parts_dir)
    scale = 1.0
    emp = snapshot_dir / EVENT_MARKETS
    if emp.exists():
        done = done | set(pl.read_parquet(emp, columns=["event_id"])["event_id"].unique())
        em = pl.read_parquet(emp).group_by("event_id").agg(pl.len().alias("n_gamma"))
        j = em.join(per, on="event_id", how="inner")
        if j.height and j["n_stored"].sum():
            scale = float(j["n_gamma"].sum() / j["n_stored"].sum())
    todo = per.filter(~pl.col("event_id").is_in(list(done)))
    n_ev = todo.height
    n_series = float(todo["n_series"].sum()) * scale
    req = n_ev * 1.0 + n_series * REQ_PER_SERIES_COARSE
    return {
        "events_total": per.height,
        "events_done": per.height - n_ev,
        "events_todo": n_ev,
        "stored_siblings_todo": int(todo["n_stored"].sum()),
        "sibling_scale": scale,
        "series_calls_todo": n_series,
        "req_per_series": REQ_PER_SERIES_COARSE,
        "requests_todo": req,
        "hours_at_5rps": req / RATE_RPS / 3600,
    }


# --------------------------------------------------------------------------- computing


def _series(prices: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for (cid,), g in prices.sort("ts").group_by("condition_id", maintain_order=True):
        out[cid] = (g["ts"].to_numpy(), g["price"].to_numpy())
    return out


def outcome_moves(
    event_markets: pl.DataFrame, prices: pl.DataFrame, reveal_end: dict[str, np.datetime64]
) -> pl.DataFrame:
    """Per (pilot reveal, sibling outcome) 24 h / 12 h moves with reasons.

    ``reveal_end`` maps each pilot condition_id to its window end (news_ts_indep - buffer);
    every sibling of the pilot's event is evaluated against that end.
    """
    ser = _series(prices)
    by_event = {k[0]: g for k, g in event_markets.group_by("event_id")}
    rows = []
    for pilot, end in reveal_end.items():
        ev = event_markets.filter(pl.col("condition_id") == pilot)
        if ev.is_empty():
            continue
        eid = ev["event_id"][0]
        for r in by_event[eid].iter_rows(named=True):
            rec = {
                "pilot_condition_id": pilot,
                "event_id": eid,
                "condition_id": r["condition_id"],
                "label": r["label"] or r["question"],
                "outcome": r["outcome"],
                "reason": None,
            }
            closed = r["closed_ts"]
            if closed is not None and np.datetime64(closed, "us") < end and not r["is_pilot"]:
                rec["reason"] = "resolved_before_window_end"
                rec["move_24h"] = rec["move_12h"] = None
            else:
                ts, px = ser.get(r["condition_id"], (np.array([], "datetime64[us]"), np.array([])))
                reasons = []
                for w in WINDOWS_H:
                    mv, why = log_odds_move(ts, px, end, w)
                    rec[f"move_{w}h"] = mv
                    if why:
                        reasons.append(why)
                rec["reason"] = reasons[0] if reasons else None
            rows.append(rec)
    schema = {
        "pilot_condition_id": pl.String,
        "event_id": pl.String,
        "condition_id": pl.String,
        "label": pl.String,
        "outcome": pl.String,
        "reason": pl.String,
        "move_24h": pl.Float64,
        "move_12h": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


OUT_SCHEMA = {
    "condition_id": pl.String,
    "event_id": pl.String,
    "n_outcomes": pl.Int64,
    "n_comparable": pl.Int64,
    "win_move_24h": pl.Float64,
    "win_move_12h": pl.Float64,
    "med_lose_move_24h": pl.Float64,
    "med_lose_move_12h": pl.Float64,
    "diff_24h": pl.Float64,
    "diff_12h": pl.Float64,
    "winner_pct_24h": pl.Float64,
    "winner_pct_12h": pl.Float64,
    "group": pl.String,
    "reason": pl.String,
    "ref_pilot_id": pl.String,
    "winner_is_pilot": pl.Boolean,
}


def classify_event(sib: pl.DataFrame) -> tuple[str, str | None]:
    """('multi'|'binary', reason) for the sibling rows of one event."""
    n = sib.height
    if n == 1:
        return "binary", None
    n_yes = int((sib["outcome"] == "YES").sum())
    if not bool(sib["neg_risk"][0]):
        return "binary", "non_exclusive_event"
    if n_yes != 1:
        return "binary", f"exclusive_event_with_{n_yes}_yes_winners"
    return "multi", None


def compute_within_event(
    pilots: pl.DataFrame, event_markets: pl.DataFrame, prices: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build within_event rows (one per independent pilot market) and the outcome-move table.

    pilots: condition_id, news_ts_indep, outcome (columns of news_independent + markets).
    """
    end = {
        r["condition_id"]: np.datetime64(r["news_ts_indep"] - timedelta(hours=BUFFER_H), "us")
        for r in pilots.iter_rows(named=True)
    }
    om = outcome_moves(event_markets, prices, end)
    sib_by_event = {k[0]: g for k, g in event_markets.group_by("event_id")}
    out = []
    done_events: set[str] = set()
    # pass 1: binary pool (winner-side move) -- needs all binaries before percentiles
    binary_rows = []
    for r in pilots.iter_rows(named=True):
        cid, eid = r["condition_id"], r["event_id"]
        base = {k: None for k in OUT_SCHEMA}
        base.update(condition_id=cid, event_id=eid)
        sib = sib_by_event.get(eid)
        if sib is None:
            base.update(group=None, reason="event_not_fetched")
            out.append(base)
            continue
        base["n_outcomes"] = sib.height
        group, why = classify_event(sib)
        base["group"] = group
        mv = om.filter(pl.col("pilot_condition_id") == cid)
        base["winner_is_pilot"] = None
        if group == "binary":
            me = mv.filter(pl.col("condition_id") == cid).row(0, named=True)
            reason = me["reason"] or why
            if r["outcome"] not in ("YES", "NO"):
                reason = "outcome_unresolved"
            sign = 1.0 if r["outcome"] == "YES" else -1.0
            base["reason"] = reason
            base["n_comparable"] = 1 if me["reason"] is None else 0
            rec = {**base, "_sign": sign, "_side": r["outcome"], "_pilot_reason": reason}
            for w in WINDOWS_H:
                v = me[f"move_{w}h"]
                rec[f"win_move_{w}h"] = (
                    None if v is None or reason in ("outcome_unresolved",) else sign * v
                )
            binary_rows.append(rec)
            continue
        # multi: statistics belong to the event winner (which need not be a pilot market)
        win_cid = sib.filter(pl.col("outcome") == "YES")["condition_id"][0]
        if cid != win_cid:
            base["reason"] = "pilot_market_not_event_winner"
            out.append(base)
        if eid in done_events:
            continue
        done_events.add(eid)
        ref = win_cid if win_cid in end else cid  # pilot whose reveal time defines the window
        mv = om.filter(pl.col("pilot_condition_id") == ref)
        comp = mv.filter(pl.col("reason").is_null())
        wrow = {k: None for k in OUT_SCHEMA}
        wrow.update(
            condition_id=win_cid,
            event_id=eid,
            n_outcomes=sib.height,
            n_comparable=comp.height,
            group="multi",
            ref_pilot_id=ref,
            winner_is_pilot=win_cid in end,
        )
        reasons = []
        for w in WINDOWS_H:
            c = comp.filter(pl.col(f"move_{w}h").is_not_null())
            ids = c["condition_id"].to_list()
            if win_cid not in ids:
                wr = mv.filter(pl.col("condition_id") == win_cid)["reason"].to_list()
                reasons.append(f"winner_{wr[0] if wr and wr[0] else 'move'}_{w}h")
                continue
            if len(ids) < 2:
                reasons.append(f"no_comparable_losers_{w}h")
                continue
            st = event_stats(c[f"move_{w}h"].to_numpy(), ids.index(win_cid))
            wrow[f"win_move_{w}h"] = st["win"]
            wrow[f"med_lose_move_{w}h"] = st["med_lose"]
            wrow[f"diff_{w}h"] = st["diff"]
            wrow[f"winner_pct_{w}h"] = st["pct"]
        wrow["reason"] = ";".join(reasons) or None
        out.append(wrow)
    # binary percentiles within same winning side (vectorised: sorted pool + leave-one-out median)
    for w in WINDOWS_H:
        for side in ("YES", "NO"):
            pool = [b for b in binary_rows if b["_side"] == side and b[f"win_move_{w}h"] is not None]
            if not pool:
                continue
            x = np.array([b[f"win_move_{w}h"] for b in pool], dtype=float)
            n = len(x)
            if n < 3:
                for b in pool:
                    b["reason"] = b["reason"] or f"binary_pool_too_small_{side}"
                continue
            sv = np.sort(x)
            less = np.searchsorted(sv, x, side="left")
            eq = np.searchsorted(sv, x, side="right") - less
            pct = (less + 0.5 * eq) / n  # mid-rank among all n including the market itself
            m = n - 1  # leave-one-out median of the other markets
            lo, hi = (m - 1) // 2, m // 2
            pos = less  # index of (the first copy of) x[i] in sv; removing it leaves the same multiset

            def kth(k, pos=pos, sv=sv, n=n):
                return np.where(k < pos, sv[k], sv[np.minimum(k + 1, n - 1)])

            med_others = 0.5 * (kth(np.full(n, lo)) + kth(np.full(n, hi)))
            for i, b in enumerate(pool):
                b[f"winner_pct_{w}h"] = float(pct[i])
                b[f"diff_{w}h"] = float(x[i] - med_others[i])
    for b in binary_rows:
        for k in ("_sign", "_side", "_pilot_reason"):
            b.pop(k)
        out.append(b)
    return pl.DataFrame(out, schema=OUT_SCHEMA), om


# --------------------------------------------------------------------------- pooled placebo


def pooled_placebo(
    within: pl.DataFrame,
    om: pl.DataFrame,
    event_markets: pl.DataFrame,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict:
    """Pooled placebo over multi events (winner vs losers) for 24 h and 12 h."""
    res = {}
    wins = within.filter((pl.col("group") == "multi") & pl.col("winner_pct_24h").is_not_null())
    wins12 = within.filter((pl.col("group") == "multi") & pl.col("winner_pct_12h").is_not_null())
    for w, ws in ((24, wins), (12, wins12)):
        events = []
        for r in ws.iter_rows(named=True):
            sib = om.filter(
                (pl.col("pilot_condition_id") == r["ref_pilot_id"])
                & pl.col(f"move_{w}h").is_not_null()
            )
            ids = sib["condition_id"].to_list()
            if r["condition_id"] not in ids or len(ids) < 3:
                continue
            events.append((sib[f"move_{w}h"].to_numpy(), ids.index(r["condition_id"])))
        res[f"{w}h"] = permutation_null(events, n_perm=n_perm, seed=seed)
        res[f"{w}h"]["median_winner_pct"] = (
            float(ws[f"winner_pct_{w}h"].median()) if ws.height else None
        )
    return res


def run(
    snapshot_dir: Path = SNAPSHOTS,
    n_perm: int = 10000,
    log=print,
    parts_dir: Path = EVENTS_DIR,
    consolidate_first: bool | None = None,
    independent_ids: set[str] | list[str] | None = None,
) -> dict:
    """Compute from the fetched events and write within_event.parquet (+ outcome moves, placebo).

    If per-event parts exist (``parts_dir``) they are consolidated into ``event_markets.parquet`` /
    ``event_prices.parquet`` first (``consolidate_first=False`` skips this and reads the existing
    snapshot files, e.g. the 108-event pilot). ``independent_ids`` restricts the markets scored
    (the pilot run used the pilot's independent markets only).
    """
    have_parts = bool(done_events(parts_dir))
    if consolidate_first or (consolidate_first is None and have_parts):
        consolidate(snapshot_dir, parts_dir, log=log)
    em = pl.read_parquet(snapshot_dir / EVENT_MARKETS)
    pr = pl.read_parquet(snapshot_dir / EVENT_PRICES)
    pil = independent_markets(snapshot_dir).filter(pl.col("event_id").is_in(em["event_id"].implode()))
    if independent_ids is not None:
        pil = pil.filter(pl.col("condition_id").is_in(list(independent_ids)))
    within, om = compute_within_event(pil, em, pr)
    _atomic_write(within, snapshot_dir / WITHIN_EVENT)
    _atomic_write(om, snapshot_dir / OUTCOME_MOVES)
    placebo = pooled_placebo(within, om, em, n_perm=n_perm)
    log(json.dumps(placebo, indent=1, default=float))
    (snapshot_dir / "within_event_placebo.json").write_text(
        json.dumps(placebo, indent=1, default=float), encoding="utf-8"
    )
    return placebo
