"""Resolving-jump detection on the winner price q(t) (phase 2b), independent of any news time.

q(t) is the price of the eventual winning outcome (YES price if YES won, else 1 - YES price, from
the price history and the winner/loser taker trades, quoted as winner price). The detector works on
an hourly forward-filled grid over the market's life up to ``closed_ts``:

1. *Candidate*: the largest rise of q over any window of at most ``MAX_RISE_H`` (12) hours,
   ``rise = max_L q[i+L] - q[i]``; its start hour is ``i*``.
2. *Pre-jump level*: median of q over the (up to) 72 h ending at ``i*``.
3. *Clear single jump* (null with a ``jump_reason`` otherwise)::

       rise >= MIN_JUMP (0.20)                                      else "no_jump"
       pre_level <= MAX_PRE_LEVEL (0.80)                            else "pre_level_high"
       rise >= DOMINANCE (0.5) * (q_end - pre_level)                else "gradual_or_multi_step"
       q stays >= pre_level + 0.5 * rise for 24 h after the peak
           (or until the end of the series)                         else "reverted"
       at least MIN_PRE_H (12) hours of history before i*           else "jump_at_start"
       at least MIN_HOURS (36) hours of series                      else "short_history"

   The thresholds are round, principled numbers (a 20-point move that carries at least half of the
   total move to the final level); they were NOT tuned on any labelled case.
4. *Jump start*: the band is ``pre_level + tol`` with ``tol = max(0.02, 3 * robust sigma of hourly
   changes in the pre window)``. ``jump_start_ts`` is the first raw observation (price point or
   trade) after the last in-band grid hour whose price is above the band and is confirmed by the
   next raw observation also being above it (a lone bad print does not start a jump). So it is
   "when q first leaves its pre-jump level", at sub-hour resolution where trades exist.

Outputs per market: ``jump_start_ts, jump_peak_ts, jump_size, jump_hours, pre_level, post_level,
q_end, jump_reason`` (reason is None when a clear jump was found).
"""

from __future__ import annotations

import numpy as np
import polars as pl

MAX_RISE_H = 12
MIN_JUMP = 0.20
MAX_PRE_LEVEL = 0.80
DOMINANCE = 0.5
PERSIST_H = 24
PRE_LOOKBACK_H = 72
MIN_PRE_H = 12
MIN_HOURS = 36
HOUR_US = 3_600_000_000

JUMP_SCHEMA = {
    "condition_id": pl.String,
    "jump_start_ts": pl.Datetime("us"),
    "jump_peak_ts": pl.Datetime("us"),
    "jump_size": pl.Float64,
    "jump_hours": pl.Float64,
    "pre_level": pl.Float64,
    "post_level": pl.Float64,
    "q_end": pl.Float64,
    "jump_reason": pl.String,
}


def _hourly(ts: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-filled hourly grid (datetime64[us] hour starts, q) from the first observation."""
    h0 = ts[0].astype("datetime64[h]")
    h1 = ts[-1].astype("datetime64[h]")
    grid = np.arange(h0, h1 + np.timedelta64(1, "h"), dtype="datetime64[h]").astype("datetime64[us]")
    idx = np.searchsorted(ts, grid, side="right") - 1
    idx = np.clip(idx, 0, None)  # first hour starts before the first obs: carry the first obs
    return grid, q[idx]


def detect_jump(ts: np.ndarray, q: np.ndarray) -> dict:
    """Detect the resolving jump. ``ts`` datetime64[us] ascending, ``q`` winner price in [0, 1]."""
    out = {k: None for k in JUMP_SCHEMA if k != "condition_id"}
    if ts.size < 3:
        out["jump_reason"] = "no_price_data"
        return out
    grid, qh = _hourly(ts, q)
    n = len(qh)
    out["q_end"] = float(qh[-1])
    if n < MIN_HOURS:
        out["jump_reason"] = "short_history"
        return out
    # largest rise over <= MAX_RISE_H hours
    best, bi, bl = -1.0, 0, 1
    for lag in range(1, MAX_RISE_H + 1):
        d = qh[lag:] - qh[:-lag]
        i = int(np.argmax(d))
        if d[i] > best + 1e-12:
            best, bi, bl = float(d[i]), i, lag
    out["jump_size"] = best
    out["jump_hours"] = float(bl)
    lo = max(0, bi - PRE_LOOKBACK_H)
    pre_win = qh[lo : bi + 1]
    pre = float(np.median(pre_win))
    out["pre_level"] = pre
    out["post_level"] = float(qh[bi + bl])
    out["jump_peak_ts"] = grid[bi + bl]
    if best < MIN_JUMP:
        out["jump_reason"] = "no_jump"
        return out
    if bi < MIN_PRE_H:
        out["jump_reason"] = "jump_at_start"
        return out
    if pre > MAX_PRE_LEVEL:
        out["jump_reason"] = "pre_level_high"
        return out
    total = float(qh[-1]) - pre
    if total > 0 and best < DOMINANCE * total:
        out["jump_reason"] = "gradual_or_multi_step"
        return out
    after = qh[bi + bl : bi + bl + PERSIST_H + 1]
    if after.size and float(after.min()) < pre + 0.5 * best:
        out["jump_reason"] = "reverted"
        return out
    # jump start: last in-band hour, then first confirmed out-of-band raw observation
    dpre = np.diff(pre_win)
    sigma = 1.4826 * float(np.median(np.abs(dpre - np.median(dpre)))) if dpre.size else 0.0
    band = pre + max(0.02, 3 * sigma)
    k = bi
    while k > lo and qh[k] > band:
        k -= 1
    k = k if qh[k] <= band else lo
    t_k = grid[k]
    cand = np.where((ts > t_k) & (q > band))[0]
    start = None
    for c in cand:
        if c + 1 >= ts.size or q[c + 1] > band:
            start = ts[c]
            break
    out["jump_start_ts"] = start if start is not None else grid[min(k + 1, n - 1)]
    return out


def winner_points(
    markets: pl.DataFrame, prices: pl.DataFrame, taker: pl.DataFrame | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per market (ts, q) of the winner price over the whole tape, ascending, up to closed_ts.

    ``prices``: condition_id, outcome_label, ts, price. ``taker``: taker trades (price of the
    traded token; a loser-token trade at p counts as winner price 1 - p).
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    p_by = prices.partition_by("condition_id", as_dict=True) if prices.height else {}
    t_by = taker.partition_by("condition_id", as_dict=True) if taker is not None else {}
    for cid, outcome, closed in markets.select("condition_id", "outcome", "closed_ts").iter_rows():
        win = {"YES": "yes", "NO": "no"}.get(outcome or "")
        if win is None:
            continue
        ts_l, q_l = [], []
        for src in (p_by.get((cid,)), t_by.get((cid,))):
            if src is None or src.height == 0:
                continue
            lab = src["outcome_label"].str.to_lowercase()
            keep = lab.is_in(["yes", "no"])
            src = src.filter(keep)
            lab = src["outcome_label"].str.to_lowercase()
            px = np.where((lab == win).to_numpy(), src["price"].to_numpy(), 1 - src["price"].to_numpy())
            ts_l.append(src["ts"].to_numpy().astype("datetime64[us]"))
            q_l.append(px.astype(float))
        if not ts_l:
            continue
        ts, q = np.concatenate(ts_l), np.concatenate(q_l)
        order = np.argsort(ts, kind="stable")
        ts, q = ts[order], np.clip(q[order], 0, 1)
        if closed is not None:
            m = ts <= np.datetime64(closed, "us")
            if m.sum() >= 3:
                ts, q = ts[m], q[m]
        out[cid] = (ts, q)
    return out


def detect_jumps(
    markets: pl.DataFrame,
    prices: pl.DataFrame,
    taker: pl.DataFrame | None,
    pts: dict | None = None,
) -> pl.DataFrame:
    """One row per market (JUMP_SCHEMA); markets without outcome/price data get a reason.

    ``pts`` (from ``winner_points``) can be passed in to avoid computing the paths twice."""
    if pts is None:
        pts = winner_points(markets, prices, taker)
    rows = []
    for cid, outcome in markets.select("condition_id", "outcome").iter_rows():
        if cid not in pts:
            r = {k: None for k in JUMP_SCHEMA}
            r["jump_reason"] = "no_outcome" if outcome not in ("YES", "NO") else "no_price_data"
        else:
            r = detect_jump(*pts[cid])
        r["condition_id"] = cid
        for k in ("jump_start_ts", "jump_peak_ts"):
            if r[k] is not None:
                r[k] = r[k].astype("datetime64[us]").item()
        rows.append(r)
    return pl.DataFrame(rows, schema=JUMP_SCHEMA, infer_schema_length=None)


def price_news_ts(ts: np.ndarray, q: np.ndarray, threshold: float = 0.9):
    """First time q is above ``threshold`` and stays above to the end (price-based news time)."""
    if ts.size == 0:
        return None
    below = np.where(q <= threshold)[0]
    i = 0 if below.size == 0 else int(below[-1]) + 1
    return ts[i] if i < ts.size else None
