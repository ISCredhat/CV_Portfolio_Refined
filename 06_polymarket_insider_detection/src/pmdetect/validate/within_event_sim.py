"""Phase 4A/4C: synthetic-insider power test and placebos for the within-event pre-reveal test.

WHAT THIS IS, AND IS NOT
------------------------
A *simulation*: an informed buyer of the eventual winner is planted into real, otherwise untouched
price paths, and the real within-event pipeline statistic (``markets/within_event.py``: change in
clipped log-odds over the 24 h / 12 h window ending ``BUFFER_H`` = 1 h before the reveal, winner's
mid-rank percentile among all comparable outcomes of its event) is recomputed. It shows what the
statistic *could* see if a buyer of a given stake and lead time moved prices the way ordinary
taker flow in that same market has moved them. It is NOT proof of real-world detectability.
Known limits, stated plainly:

* The price impact is empirical but linear in signed USD (per market, in log-odds), estimated from
  public taker flow in 1 h bins. Real impact is concave and depends on the book at that moment;
  linear extrapolation to $20k-$50k stakes in thin markets can overstate the move (a half-impact
  sensitivity run is written next to the main table). Hourly "last trade" prices include bid-ask
  bounce and unrelated news, which biases the estimated slope towards zero (conservative).
* The planted buy is ONE order at ``reveal - k h`` whose effect is a PERMANENT level shift (no decay
  inside the window). A buyer who splits the order over hours or days, who hides behind other
  wallets, or whose information is already public news (so everybody moves) is not modelled.
* A level shift that happens before the window START (reveal - 1 h - 24 h, i.e. k > 25 h for the
  24 h window; k > 13 h for the 12 h window) leaves the log-odds CHANGE inside the window almost
  untouched, so the statistic is blind to it by construction. That is a property of the test
  ("what moved in the last day"), reported as such, not a bug of the simulation.
* The winner is the only outcome that gets a buyer; siblings are lowered mechanically (below).
* Events are real scheduled-reveal multi-outcome (negRisk) events with >= 3 comparable outcomes;
  results do not transfer to unscheduled news or binary markets.

MODEL
-----
Impact (per market, ``estimate_impacts``): hourly signed net taker USD ``x_h`` (buy YES and sell NO
count +, sell YES and buy NO count -; $1k units) against the 1-h change in clipped log-odds of the
last YES-equivalent trade price ``y_h`` between CONSECUTIVE trading hours, using the tape strictly
before the window end (reveal - 1 h) only. Slope through the origin by Huber IRLS (c = 1.345, scale
from the residual MAD, floored). The market's own slope is used when it has >= 30 hour-pairs, a
positive slope and a robust t-statistic >= 2; otherwise (and for winners without a tape) the
FALLBACK is the pooled median of valid slopes by price bucket (bucket of the market's median price
for the pool; of the winner's price at the window start for the lookup).

Planting (``plant_moves``): at ``t0 = reveal - k h`` the winner's YES log-odds are raised by
``Delta = beta * stake/1000`` (clipped to the price clip bounds, [0.005, 0.995]); all later price
observations/forward fills carry the shift (no decay). Siblings are repriced at the same times so
the sum of the comparable outcomes' YES prices is conserved: ``p_i' = p_i * (S - p_w') / (S - p_w)``
with ``S`` the sum of comparable YES prices at that time (floored at 0; the statistic clips again).
A sibling whose last print is older than ``t0`` is treated as repriced at ``t0`` (its stale age is
measured from ``t0``); this mirrors neg-risk arbitrage.

Detection (PLAN threshold, fixed before any run): winner percentile (mid-rank, 24 h window) >= 0.95.
False-positive rate: share of UNTOUCHED real events (same population: >= 3 comparable outcomes)
with a real winner percentile >= 0.95; also reported on all multi events (n_comparable >= 2) as
a sensitivity. Plantable events are the "unremarkable" ones: real winner percentile < 0.75
(fixed in advance, per window). A cell (s, k) is *detected* if detection >= 50% at FPR <= 10%.
Because the mid-rank percentile of the winner among n outcomes is at most (n - 0.5)/n, an event
needs n >= 10 comparable outcomes to ever reach 0.95; detection is therefore capped by the share
of such events (the table is also given for the n >= 10 stratum).

Bootstrap: events are resampled with replacement (1,000 resamples; one resample feeds both the
detection rate over the plantable events in it and the FPR over all its events), 95% percentile
intervals. Seeds fixed (master seed 20261113).

Placebos (4C): (a) the same statistic on a window ending 48 h before the reveal; (b) 100
shuffles of the winner label within events. Pass rules are in PLAN.md and are used as written.

Privacy: only aggregates are written; event and market identifiers are not written to results/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.markets.within_event import (
    BUFFER_H,
    CLIP,
    MAX_STALE_H,
    OUTCOME_MOVES,
    WINDOWS_H,
    WITHIN_EVENT,
    event_stats,
    last_at_or_before,
    log_odds_move,
    logit,
    outcome_moves,
    permutation_null,
)

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS = ROOT / "data" / "snapshots"
RESULTS = ROOT / "results"

SEED = 20261113
STAKES_USD = (500, 2000, 5000, 20000, 50000)
LEADS_H = (2, 6, 12, 24, 48, 72)
UNREMARKABLE_PCT = 0.75  # plantable if the real winner percentile is below this (per window)
DETECT_PCT = 0.95  # PLAN: detection / false positive threshold
MIN_COMPARABLE = 3
BIG_EVENT = 10  # smallest n for which a mid-rank percentile can reach 0.95
CRIT_DETECT, CRIT_FPR = 0.50, 0.10  # PLAN criterion
MIN_FIT_PAIRS = 30
MIN_T = 2.0
PRICE_EDGES = (0.1, 0.3, 0.7, 0.9)  # fallback price buckets
HOUR = np.timedelta64(1, "h")

# --------------------------------------------------------------------------- impact estimation


def huber_slope(x: np.ndarray, y: np.ndarray, c: float = 1.345, iters: int = 50) -> tuple[float, float]:
    """Through-origin Huber IRLS slope of y on x and a sandwich standard error."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sxx = float(np.sum(x * x))
    if len(x) < 3 or sxx <= 0:
        return float("nan"), float("nan")
    b = float(np.sum(x * y) / sxx)
    w = np.ones_like(x)
    for _ in range(iters):
        r = y - b * x
        s = max(1.4826 * float(np.median(np.abs(r))), 0.1 * float(np.std(r)), 1e-9)
        u = np.abs(r) / (c * s)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        den = float(np.sum(w * x * x))
        if den <= 0:
            break
        b_new = float(np.sum(w * x * y) / den)
        if abs(b_new - b) < 1e-10 * (1 + abs(b)):
            b = b_new
            break
        b = b_new
    r = y - b * x
    den = float(np.sum(w * x * x))
    se = float(np.sqrt(np.sum((w * x * r) ** 2)) / den) if den > 0 else float("nan")
    return b, se


def hourly_flow(tape: pl.DataFrame) -> pl.DataFrame:
    """Hourly signed net taker USD and last YES-equivalent price per market.

    ``tape``: condition_id, outcome_label ('Yes'/'No'), side ('BUY'/'SELL'), price, usd, ts.
    """
    yes = pl.col("outcome_label") == "Yes"
    buy = pl.col("side") == "BUY"
    return (
        tape.filter(pl.col("outcome_label").is_in(["Yes", "No"]))
        .with_columns(
            pl.when(yes == buy).then(1.0).otherwise(-1.0).alias("_sign"),
            pl.when(yes).then(pl.col("price")).otherwise(1.0 - pl.col("price")).alias("_p"),
            pl.col("ts").dt.truncate("1h").alias("hour"),
        )
        .sort("condition_id", "ts")
        .group_by("condition_id", "hour", maintain_order=True)
        .agg((pl.col("_sign") * pl.col("usd")).sum().alias("flow_usd"), pl.col("_p").last().alias("p"))
    )


def hour_pairs(hourly: pl.DataFrame) -> pl.DataFrame:
    """Consecutive-trading-hour pairs: x = net flow in $k in hour h, y = d logit(last price)."""
    h = hourly.sort("condition_id", "hour").with_columns(
        pl.col("hour").shift(1).over("condition_id").alias("_ph"),
        pl.col("p").shift(1).over("condition_id").alias("_pp"),
    )
    h = h.filter((pl.col("hour") - pl.col("_ph")) == pl.duration(hours=1)).filter(
        pl.col("_pp").is_not_null()
    )
    return h.select(
        "condition_id",
        (pl.col("flow_usd") / 1000.0).alias("x"),
        pl.col("p"),
        pl.col("_pp"),
    ).with_columns(
        pl.Series("y", logit(h["p"].to_numpy()) - logit(h["_pp"].to_numpy()))
    )


def estimate_impacts(tape: pl.DataFrame, cutoffs: dict[str, object]) -> pl.DataFrame:
    """Per-market log-odds impact per $1k of net taker flow (see module docstring).

    ``cutoffs``: condition_id -> datetime; only trades strictly before the cutoff are used.
    Returns condition_id, beta, se, t, n_pairs, med_price, valid.
    """
    cut = pl.DataFrame(
        {"condition_id": list(cutoffs), "_cut": list(cutoffs.values())},
        schema={"condition_id": pl.String, "_cut": pl.Datetime("us")},
    )
    tp = tape.join(cut, on="condition_id").filter(pl.col("ts") < pl.col("_cut")).drop("_cut")
    pairs = hour_pairs(hourly_flow(tp))
    rows = []
    for (cid,), g in pairs.group_by("condition_id", maintain_order=True):
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        n = len(x)
        if n >= MIN_FIT_PAIRS:
            b, se = huber_slope(x, y)
        else:
            b, se = float("nan"), float("nan")
        t = b / se if se and se > 0 and np.isfinite(b) else float("nan")
        rows.append(
            {
                "condition_id": cid,
                "beta": b,
                "se": se,
                "t": t,
                "n_pairs": n,
                "med_price": float(np.median(g["p"].to_numpy())),
                "valid": bool(n >= MIN_FIT_PAIRS and np.isfinite(t) and b > 0 and t >= MIN_T),
            }
        )
    schema = {
        "condition_id": pl.String,
        "beta": pl.Float64,
        "se": pl.Float64,
        "t": pl.Float64,
        "n_pairs": pl.Int64,
        "med_price": pl.Float64,
        "valid": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


def price_bucket(p: float) -> int:
    return int(np.searchsorted(PRICE_EDGES, p, side="right"))


def pooled_fallback(impacts: pl.DataFrame, min_n: int = 20) -> dict:
    """Pooled median of valid slopes by price bucket (+ 'all'); buckets thinner than min_n use 'all'."""
    ok = impacts.filter(pl.col("valid"))
    out: dict = {"all": float(ok["beta"].median()) if ok.height else float("nan"), "n": {}}
    if ok.height:
        b = np.array([price_bucket(p) for p in ok["med_price"].to_numpy()])
        beta = ok["beta"].to_numpy()
        for k in range(len(PRICE_EDGES) + 1):
            sel = beta[b == k]
            out["n"][k] = int(len(sel))
            out[k] = float(np.median(sel)) if len(sel) >= min_n else out["all"]
    return out


def impact_for(cid: str, p_ref: float, own: dict[str, float], fb: dict) -> tuple[float, str]:
    """(beta per $1k, 'own' | 'fallback') for the winner market."""
    if cid in own:
        return own[cid], "own"
    return fb.get(price_bucket(p_ref), fb["all"]), "fallback"


# --------------------------------------------------------------------------- events and planting


@dataclass
class SimEvent:
    event_id: str
    winner_cid: str
    win_idx: int
    n: int
    end: np.datetime64  # window end = reveal - BUFFER_H
    pre: dict  # window_h -> dict(start, end, ps, as_, pe, ae)  per-outcome arrays
    raw_pct: dict  # window_h -> real winner percentile
    p_ref: float  # winner price at the 24 h window start


def make_sim_event(
    event_id: str,
    winner_cid: str,
    cids: list[str],
    series: list[tuple[np.ndarray, np.ndarray]],
    end: np.datetime64,
) -> SimEvent | None:
    """Pre-compute the window-edge prices of every comparable outcome; None if any is missing.

    Uses ``last_at_or_before`` (forward fill only) exactly as the pipeline does, and
    ``log_odds_move`` for the real (untouched) moves, so ``raw_pct`` equals the pipeline value.
    """
    pre: dict = {}
    raw_pct: dict = {}
    p_ref = float("nan")
    for w in WINDOWS_H:
        start = end - np.timedelta64(int(w * 3600), "s")
        ps, as_, pe, ae = [], [], [], []
        moves = []
        for ts, px in series:
            a = last_at_or_before(ts, px, start)
            b = last_at_or_before(ts, px, end)
            mv, why = log_odds_move(ts, px, end, w)
            if a is None or b is None or why is not None:
                return None
            ps.append(a[0])
            as_.append(a[1])
            pe.append(b[0])
            ae.append(b[1])
            moves.append(mv)
        win_idx = cids.index(winner_cid)
        raw_pct[w] = event_stats(np.array(moves), win_idx)["pct"]
        pre[w] = {
            "start": start,
            "end": end,
            "ps": np.array(ps),
            "as": np.array(as_),
            "pe": np.array(pe),
            "ae": np.array(ae),
        }
        if w == max(WINDOWS_H):
            p_ref = float(ps[win_idx])
    return SimEvent(event_id, winner_cid, cids.index(winner_cid), len(cids), end, pre, raw_pct, p_ref)


def _expit(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def plant_moves(ev: SimEvent, w: int, delta: float, lead_h: float) -> np.ndarray:
    """Log-odds moves of all comparable outcomes after planting a buyer ``lead_h`` hours before
    the reveal (reveal = window end + BUFFER_H). ``delta`` = log-odds shift of the winner (>= 0);
    delta = 0 returns the real moves. Siblings keep the sum of comparable YES prices."""
    p = ev.pre[w]
    t0 = ev.end + np.timedelta64(BUFFER_H * 3600, "s") - np.timedelta64(int(round(lead_h * 3600)), "s")
    lo, hi = CLIP
    out = []
    for t, price, age in ((p["start"], p["ps"], p["as"]), (p["end"], p["pe"], p["ae"])):
        price = price.copy()
        age = age.copy()
        if delta > 0 and t >= t0:
            pw = price[ev.win_idx]
            pwm = float(np.clip(_expit(float(logit(pw)) + delta), lo, hi))
            s = float(price.sum())
            den = s - pw
            f = max((s - pwm) / den, 0.0) if den > 1e-9 else 1.0
            price = price * f
            price[ev.win_idx] = pwm
            age = np.minimum(age, float((t - t0) / HOUR))
        out.append((price, age))
    (p_s, _), (p_e, a_e) = out
    if np.any(a_e > MAX_STALE_H):
        return np.full(ev.n, np.nan)
    return logit(p_e) - logit(p_s)


def planted_percentile(ev: SimEvent, w: int, delta: float, lead_h: float) -> float:
    mv = plant_moves(ev, w, delta, lead_h)
    if not np.all(np.isfinite(mv)):
        return float("nan")
    return event_stats(mv, ev.win_idx)["pct"]


# --------------------------------------------------------------------------- bootstrap


def bootstrap_grid(
    det: np.ndarray,
    elig: np.ndarray,
    fp: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
    crit_detect: float = CRIT_DETECT,
    crit_fpr: float = CRIT_FPR,
) -> dict:
    """Event bootstrap for detection rates (per cell, over plantable events) and the FPR.

    det: (N, C) bool, detection of the planted winner per event and cell (only rows with ``elig``
    count). elig: (N,) bool, plantable. fp: (N,) bool, untouched winner percentile >= 0.95.
    One multinomial weight matrix per call: detection and FPR come from the SAME resample.
    """
    n = len(elig)
    rng = np.random.default_rng(seed)
    wts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    d = det.astype(float) * elig[:, None]
    den = wts @ elig.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (wts @ d) / den[:, None]
    fpr = (wts @ fp.astype(float)) / n
    q = lambda a, p: np.nanpercentile(a, p, axis=0)  # noqa: E731
    meets = (rate >= crit_detect) & (fpr[:, None] <= crit_fpr)
    return {
        "det_lo": q(rate, 2.5),
        "det_hi": q(rate, 97.5),
        "fpr_lo": float(np.percentile(fpr, 2.5)),
        "fpr_hi": float(np.percentile(fpr, 97.5)),
        "p_meets": meets.mean(axis=0),
    }


# --------------------------------------------------------------------------- loading


def load_sim_events(
    snapshot_dir: Path = SNAPSHOTS,
    n_events: int | None = None,
    seed: int = SEED,
    log=print,
) -> tuple[list[SimEvent], dict, dict]:
    """Real multi events with >= 3 comparable outcomes -> (events, ref/winner info, exclusions)."""
    w = pl.read_parquet(snapshot_dir / WITHIN_EVENT)
    multi = w.filter(
        (pl.col("group") == "multi") & (pl.col("reason") != "pilot_market_not_event_winner").fill_null(True)
    )
    excl = {
        "multi_events_total": multi.height,
        "no_pipeline_statistic": multi.filter(pl.col("winner_pct_24h").is_null()).height,
    }
    have = multi.filter(pl.col("winner_pct_24h").is_not_null())
    excl["statistic_n_comparable_lt3"] = have.filter(pl.col("n_comparable") < MIN_COMPARABLE).height
    ok = have.filter(pl.col("n_comparable") >= MIN_COMPARABLE).sort("event_id")
    # real winner percentile among ALL multi events with a statistic (>= 2 comparable): FPR sensitivity
    alt = {f"pct_{w_}h": have[f"winner_pct_{w_}h"].drop_nulls().to_numpy() for w_ in WINDOWS_H}
    if n_events is not None and n_events < ok.height:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(ok.height, size=n_events, replace=False))
        ok = ok[keep.tolist()]
    refs = ok["ref_pilot_id"].to_list()
    eids = ok["event_id"].to_list()
    news = pl.read_parquet(snapshot_dir / "news_independent.parquet").select(
        pl.col("condition_id").alias("ref_pilot_id"), "news_ts_indep"
    )
    ok = ok.join(news, on="ref_pilot_id", how="left")
    om = pl.read_parquet(snapshot_dir / OUTCOME_MOVES).filter(
        pl.col("pilot_condition_id").is_in(refs) & pl.col("reason").is_null()
    )
    om_by = {k[0]: g["condition_id"].to_list() for k, g in om.group_by("pilot_condition_id", maintain_order=True)}
    pr = (
        pl.scan_parquet(snapshot_dir / "event_prices.parquet")
        .filter(pl.col("event_id").is_in(eids))
        .select("condition_id", "ts", "price")
        .collect()
        .sort("ts", maintain_order=True)
    )
    ser: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (cid,), g in pr.group_by("condition_id", maintain_order=True):
        ser[cid] = (g["ts"].to_numpy(), g["price"].to_numpy())
    events: list[SimEvent] = []
    info: dict = {}
    excl["pipeline_parity_or_missing_series"] = 0
    for r in ok.iter_rows(named=True):
        cids = om_by.get(r["ref_pilot_id"], [])
        if r["condition_id"] not in cids or len(cids) < MIN_COMPARABLE or r["news_ts_indep"] is None:
            excl["pipeline_parity_or_missing_series"] += 1
            continue
        end = np.datetime64(r["news_ts_indep"] - timedelta(hours=BUFFER_H), "us")
        empty = (np.array([], "datetime64[us]"), np.array([]))
        ev = make_sim_event(
            r["event_id"], r["condition_id"], cids, [ser.get(c, empty) for c in cids], end
        )
        if ev is None or abs(ev.raw_pct[24] - r["winner_pct_24h"]) > 1e-9 or abs(
            ev.raw_pct[12] - r["winner_pct_12h"]
        ) > 1e-9:
            excl["pipeline_parity_or_missing_series"] += 1
            continue
        events.append(ev)
        info[ev.event_id] = {
            "ref": r["ref_pilot_id"],
            "winner": r["condition_id"],
            "news": r["news_ts_indep"],
            "cids": cids,
        }
    excl["used"] = len(events)
    log(f"events: {excl}")
    return events, info, {"excl": excl, "alt": alt}


def load_impacts(info: dict, snapshot_dir: Path = SNAPSHOTS, log=print) -> tuple[dict, dict, pl.DataFrame]:
    """Per-market impacts for the event winners plus the pooled fallback from independent markets."""
    news = pl.read_parquet(snapshot_dir / "news_independent.parquet").filter(pl.col("independent"))
    cutoffs = {
        c: t - timedelta(hours=BUFFER_H) for c, t in zip(news["condition_id"], news["news_ts_indep"], strict=True)
    }
    for v in info.values():  # winners that are not themselves independent: the reference reveal
        cutoffs.setdefault(v["winner"], v["news"] - timedelta(hours=BUFFER_H))
    tape = (
        pl.scan_parquet(snapshot_dir / "trades_taker.parquet")
        .filter(pl.col("condition_id").is_in(list(cutoffs)))
        .select("condition_id", "outcome_label", "side", "price", "usd", "ts")
        .collect()
    )
    log(f"impact tape: {tape.height:,} rows, {len(cutoffs):,} markets")
    imp = estimate_impacts(tape, cutoffs)
    fb = pooled_fallback(imp)
    own = {c: b for c, b, v in zip(imp["condition_id"], imp["beta"], imp["valid"], strict=True) if v}
    log(
        f"impact: {imp.height} fitted, {int(imp['valid'].sum())} valid; pooled median beta "
        f"{fb['all']:.4f} per $1k; buckets {fb['n']}"
    )
    return own, fb, imp


# --------------------------------------------------------------------------- power grid


def run_grid(
    events: list[SimEvent],
    betas: np.ndarray,
    w: int,
    impact_scale: float = 1.0,
    stakes=STAKES_USD,
    leads=LEADS_H,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(det (N,C) bool, pct (N,C) float, delta (N,C) float, cells list as (s,k) array)."""
    cells = [(s, k) for s in stakes for k in leads]
    n = len(events)
    det = np.zeros((n, len(cells)), bool)
    pct = np.full((n, len(cells)), np.nan)
    dlt = np.zeros((n, len(cells)))
    for i, ev in enumerate(events):
        for j, (s, k) in enumerate(cells):
            d = impact_scale * betas[i] * s / 1000.0
            dlt[i, j] = d
            q = planted_percentile(ev, w, d, k)
            pct[i, j] = q
            det[i, j] = bool(q >= DETECT_PCT)
    return det, pct, dlt, np.array(cells)


def power_table(
    events: list[SimEvent],
    betas: np.ndarray,
    w: int,
    subset: str,
    impact_scale: float,
    alt_fpr: float | None,
    n_boot: int,
    seed: int,
    stakes=STAKES_USD,
    leads=LEADS_H,
) -> pl.DataFrame:
    sel = [i for i, e in enumerate(events) if subset == "all" or e.n >= BIG_EVENT]
    ev = [events[i] for i in sel]
    b = betas[sel]
    elig = np.array([e.raw_pct[w] < UNREMARKABLE_PCT for e in ev])
    fp = np.array([e.raw_pct[w] >= DETECT_PCT for e in ev])
    det, pct, dlt, cells = run_grid(ev, b, w, impact_scale, stakes, leads)
    n = len(ev)
    if n == 0 or elig.sum() == 0:
        return pl.DataFrame()
    bs = bootstrap_grid(det, elig, fp, n_boot=n_boot, seed=seed)
    rate = det[elig].mean(axis=0)
    fpr = float(fp.mean())
    rows = []
    for j, (s, k) in enumerate(cells):
        meets = bool(rate[j] >= CRIT_DETECT and fpr <= CRIT_FPR)
        meets_alt = None if alt_fpr is None else bool(rate[j] >= CRIT_DETECT and alt_fpr <= CRIT_FPR)
        rows.append(
            {
                "subset": subset,
                "window_h": w,
                "impact_scale": impact_scale,
                "stake_usd": int(s),
                "lead_h": int(k),
                "n_events": n,
                "n_plantable": int(elig.sum()),
                "detection_rate": float(rate[j]),
                "det_ci_lo": float(bs["det_lo"][j]),
                "det_ci_hi": float(bs["det_hi"][j]),
                "fpr_untouched": fpr,
                "fpr_ci_lo": bs["fpr_lo"],
                "fpr_ci_hi": bs["fpr_hi"],
                "fpr_alt_all_multi_events": alt_fpr,
                "meets_criterion": meets,
                "meets_criterion_alt_fpr": meets_alt,
                "p_meets_bootstrap": float(bs["p_meets"][j]),
                "median_delta_logodds": float(np.median(dlt[elig, j])),
                "median_planted_pct": float(np.nanmedian(pct[elig, j])),
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- placebos


def events_from_moves(om: pl.DataFrame, win: dict[str, tuple[str, str]], w: int) -> dict:
    """event_id -> (moves of comparable outcomes, winner index), pooled_placebo's selection rule."""
    out = {}
    for eid, (ref, wcid) in win.items():
        sib = om.filter((pl.col("pilot_condition_id") == ref) & pl.col(f"move_{w}h").is_not_null())
        ids = sib["condition_id"].to_list()
        if wcid not in ids or len(ids) < 3:
            continue
        out[eid] = (sib[f"move_{w}h"].to_numpy(), ids.index(wcid))
    return out


def shuffle_shares(events: list[tuple[np.ndarray, int]], n_shuffles: int, seed: int) -> dict:
    """Winner label redrawn uniformly among comparable outcomes, n_shuffles times: distribution of
    (share of events with the 'winner' above the 95th percentile of the others) and of
    (share with percentile >= 0.95). The observed values are returned alongside."""
    rng = np.random.default_rng(seed)
    flag_rows, pct_rows, obs_flag, obs_pct = [], [], [], []
    for m, wi in events:
        st = [event_stats(m, k) for k in range(len(m))]
        flag_rows.append(np.array([x["win"] > x["q95_lose"] for x in st], float))
        pct_rows.append(np.array([x["pct"] >= DETECT_PCT for x in st], float))
        obs_flag.append(flag_rows[-1][wi])
        obs_pct.append(pct_rows[-1][wi])
    sizes = [len(r) for r in flag_rows]
    f = np.empty(n_shuffles)
    p = np.empty(n_shuffles)
    for b in range(n_shuffles):
        pick = [int(rng.integers(0, s)) for s in sizes]
        f[b] = np.mean([flag_rows[i][k] for i, k in enumerate(pick)])
        p[b] = np.mean([pct_rows[i][k] for i, k in enumerate(pick)])
    return {
        "n_events": len(events),
        "obs_share_above_q95": float(np.mean(obs_flag)),
        "obs_share_pct95": float(np.mean(obs_pct)),
        "shuffle_share_above_q95": f,
        "shuffle_share_pct95": p,
    }


def run_placebos(
    info: dict,
    snapshot_dir: Path = SNAPSHOTS,
    n_perm: int = 2000,
    n_shuffles: int = 100,
    seed: int = SEED,
    log=print,
) -> pl.DataFrame:
    em = pl.read_parquet(snapshot_dir / "event_markets.parquet").filter(
        pl.col("event_id").is_in(list(info))
    )
    pr = pl.read_parquet(snapshot_dir / "event_prices.parquet").filter(
        pl.col("event_id").is_in(list(info))
    )
    end_real = {
        v["ref"]: np.datetime64(v["news"] - timedelta(hours=BUFFER_H), "us") for v in info.values()
    }
    end_48 = {
        v["ref"]: np.datetime64(v["news"] - timedelta(hours=48), "us") for v in info.values()
    }
    win = {eid: (v["ref"], v["winner"]) for eid, v in info.items()}
    log("placebo: outcome moves, real window")
    om_real = outcome_moves(em, pr, end_real)
    log("placebo: outcome moves, window ending 48 h before the reveal")
    om_48 = outcome_moves(em, pr, end_48)
    rows = []
    for w in WINDOWS_H:
        ev_r = events_from_moves(om_real, win, w)
        ev_p = events_from_moves(om_48, win, w)
        common = sorted(set(ev_r) & set(ev_p))
        sets = {"intersection": common, "own": None}
        res = {}
        for name, ids in sets.items():
            r_list = [ev_r[e] for e in (ids if ids is not None else sorted(ev_r))]
            p_list = [ev_p[e] for e in (ids if ids is not None else sorted(ev_p))]
            real = permutation_null(r_list, n_perm=n_perm, seed=seed)
            plc = permutation_null(p_list, n_perm=n_perm, seed=seed)
            ex_real = real["obs_share_above_q95"] - real["null_share_above_q95"]
            ex_plc = plc["obs_share_above_q95"] - plc["null_share_above_q95"]
            ok = bool(plc["p_share_above_q95"] > 0.01 or ex_plc <= 0.5 * ex_real)
            res[name] = ok
            rows.append(
                {
                    "placebo": "a_window_ending_48h_before_reveal",
                    "window_h": w,
                    "event_set": name,
                    "n_events": plc["n_events"],
                    "n_events_real_window": real["n_events"],
                    "observed_share_above_q95": plc["obs_share_above_q95"],
                    "null_share_above_q95": plc["null_share_above_q95"],
                    "excess": ex_plc,
                    "permutation_p": plc["p_share_above_q95"],
                    "real_window_observed": real["obs_share_above_q95"],
                    "real_window_null": real["null_share_above_q95"],
                    "real_window_excess": ex_real,
                    "real_window_p": real["p_share_above_q95"],
                    "excess_ratio_vs_real": ex_plc / ex_real if ex_real > 0 else None,
                    "pass": ok,
                }
            )
        log(f"placebo a, {w} h: pass(intersection)={res['intersection']} pass(own)={res['own']}")
        # (b) shuffled winners on the real window
        sh = shuffle_shares([ev_r[e] for e in sorted(ev_r)], n_shuffles, seed + w)
        for stat, key_o, key_s in (
            ("share_above_q95_of_losers", "obs_share_above_q95", "shuffle_share_above_q95"),
            ("share_percentile_ge_0.95", "obs_share_pct95", "shuffle_share_pct95"),
        ):
            dist = sh[key_s]
            lo, hi = np.percentile(dist, [2.5, 97.5])
            obs = sh[key_o]
            rows.append(
                {
                    "placebo": "b_shuffled_winners",
                    "window_h": w,
                    "event_set": stat,
                    "n_events": sh["n_events"],
                    "observed_share_above_q95": obs,
                    "null_share_above_q95": float(dist.mean()),
                    "excess": obs - float(dist.mean()),
                    "shuffle_sd": float(dist.std(ddof=1)),
                    "shuffle_lo95": float(lo),
                    "shuffle_hi95": float(hi),
                    "real_percentile_in_shuffles": float((dist < obs).mean() + 0.5 * (dist == obs).mean()),
                    "pass": bool(lo <= obs <= hi),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


# --------------------------------------------------------------------------- outputs


def heatmap(tab: dict[int, pl.DataFrame], path: Path, title_extra: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tab), figsize=(6.2 * len(tab), 4.6), squeeze=False)
    for ax, (w, df) in zip(axes[0], sorted(tab.items(), reverse=True), strict=True):
        stakes = sorted(df["stake_usd"].unique().to_list())
        leads = sorted(df["lead_h"].unique().to_list())
        m = np.full((len(stakes), len(leads)), np.nan)
        for r in df.iter_rows(named=True):
            m[stakes.index(r["stake_usd"]), leads.index(r["lead_h"])] = r["detection_rate"]
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
        for r in df.iter_rows(named=True):
            i, j = stakes.index(r["stake_usd"]), leads.index(r["lead_h"])
            col = "white" if m[i, j] < 0.6 else "black"
            ax.text(
                j,
                i,
                f"{r['detection_rate']:.0%}\n[{r['det_ci_lo']:.0%}, {r['det_ci_hi']:.0%}]",
                ha="center",
                va="center",
                fontsize=7,
                color=col,
            )
            if r["meets_criterion"]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2.5))
        ax.set_xticks(range(len(leads)), [str(k) for k in leads])
        ax.set_yticks(range(len(stakes)), [f"${s:,}" for s in stakes])
        ax.set_xlabel("lead time k before the reveal (hours)")
        ax.set_ylabel("planted stake s")
        r0 = df.row(0, named=True)
        ax.set_title(
            f"{w} h window: detection rate (95% bootstrap CI)\n"
            f"FPR on untouched events = {r0['fpr_untouched']:.1%} "
            f"[{r0['fpr_ci_lo']:.1%}, {r0['fpr_ci_hi']:.1%}]; criterion: >= 50% at FPR <= 10%\n"
            f"{r0['n_plantable']} plantable of {r0['n_events']} events{title_extra}",
            fontsize=8.5,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="detection rate")
    fig.suptitle(
        "Within-event test, synthetic informed buyer of the winner (simulation, not proof)", fontsize=10
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _fmt_grid(df: pl.DataFrame) -> str:
    leads = sorted(df["lead_h"].unique().to_list())
    lines = ["| stake \\ lead (h) | " + " | ".join(str(k) for k in leads) + " |", "|---|" + "---|" * len(leads)]
    for s in sorted(df["stake_usd"].unique().to_list()):
        cells = []
        for k in leads:
            r = df.filter((pl.col("stake_usd") == s) & (pl.col("lead_h") == k)).row(0, named=True)
            mark = " **(meets)**" if r["meets_criterion"] else ""
            cells.append(f"{r['detection_rate']:.0%} [{r['det_ci_lo']:.0%}, {r['det_ci_hi']:.0%}]{mark}")
        lines.append(f"| ${s:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def smallest_cells(df: pl.DataFrame, col: str = "meets_criterion") -> list[tuple[int, int]]:
    """For each lead, the smallest stake meeting the criterion; (stake, lead) pairs, Pareto-minimal."""
    ok = df.filter(pl.col(col))
    best = []
    for k in sorted(ok["lead_h"].unique().to_list()):
        best.append((int(ok.filter(pl.col("lead_h") == k)["stake_usd"].min()), int(k)))
    return best


def write_report(
    path: Path,
    tabs: dict,
    plc: pl.DataFrame,
    meta: dict,
) -> None:
    t24 = tabs[("all", 24, 1.0)]
    t12 = tabs[("all", 12, 1.0)]
    L = []
    a = L.append
    a("# Within-event test: synthetic-insider power and placebos (phase 4A / 4C)\n")
    a("Aggregate-only output of `pmdetect validate within-event` (no wallet addresses, no market or "
      "event identifiers). Thresholds are the PLAN.md acceptance thresholds written before any run; "
      "they were not changed.\n")
    a("## What this is, and what it is not\n")
    a("A **simulation**: a buyer of the eventual winner is planted into real, otherwise untouched "
      "price paths with an empirical, linear-in-USD (log-odds) price impact estimated from each "
      "market's own taker tape, and the real pipeline statistic is recomputed. It is **not proof of "
      "real-world detectability**. It does not model buyers who split orders over time or across "
      "wallets, impact that is concave in size, or leaks that are already public news (then every "
      "outcome moves and there is nothing to single out). Planted buys are one order with a "
      "permanent price effect; buys earlier than the window start are invisible to a 'change in the "
      "last day' statistic by construction.\n")
    a("## Method\n")
    a(f"* Population: real scheduled-reveal multi-outcome events from `within_event.parquet` with a "
      f"pipeline statistic and >= {MIN_COMPARABLE} comparable outcomes. Used: **{meta['n_events']}**. "
      f"Excluded: {meta['excl']}.")
    a(f"* Plantable ('unremarkable', fixed in advance): real winner percentile < {UNREMARKABLE_PCT} "
      f"in the window being tested (24 h: {t24['n_plantable'][0]} events; 12 h: {t12['n_plantable'][0]}).")
    a(f"* Impact: per-market Huber slope of 1-h log-odds change on signed net taker USD ($1k), tape "
      f"before the window end; own slope if >= {MIN_FIT_PAIRS} hour-pairs, positive and t >= {MIN_T}, "
      f"else the pooled median by price bucket. Own-market impact used for "
      f"**{meta['n_own']}** of {meta['n_events']} events, pooled fallback for {meta['n_fallback']}. "
      f"Median slope used: {meta['median_beta']:.3f} log-odds per $1k "
      f"(pooled median {meta['fb_all']:.3f}; {meta['n_valid']} of {meta['n_fitted']} fitted markets valid).")
    a("* Planting: winner log-odds +beta*stake/1000 from reveal - k h onward (clipped to [0.005, 0.995]), "
      "no decay; siblings scaled by (S - p_w')/(S - p_w) so the comparable YES prices keep their sum.")
    a("* Statistic: exactly `within_event.py` (24 h and 12 h windows ending 1 h before the reveal, "
      "forward-filled last price, same clip, same stale rule, mid-rank winner percentile).")
    a("* Detection: planted winner percentile >= 0.95. FPR: share of untouched real events (same "
      f"population) with real winner percentile >= 0.95. Criterion: detection >= 50% at FPR <= 10%. "
      f"Bootstrap: {meta['n_boot']} event resamples, 95% percentile intervals; seed {meta['seed']}.\n")
    for w, t in ((24, t24), (12, t12)):
        r0 = t.row(0, named=True)
        a(f"## Power table, {w} h window (detection rate, 95% CI)\n")
        a(f"FPR on untouched events: **{r0['fpr_untouched']:.1%}** (95% CI {r0['fpr_ci_lo']:.1%} to "
          f"{r0['fpr_ci_hi']:.1%}, n = {r0['n_events']}); on all multi events incl. 2-outcome ones "
          f"(sensitivity): {r0['fpr_alt_all_multi_events']:.1%}.\n")
        a(_fmt_grid(t))
        cells = smallest_cells(t)
        cells_alt = smallest_cells(t, "meets_criterion_alt_fpr")
        a("")
        if cells:
            a(f"Cells meeting the criterion: {int(t['meets_criterion'].sum())} of {t.height}; smallest stake "
              f"per lead: " + ", ".join(f"${s:,} at {k} h" for s, k in cells) + ".")
        else:
            a(f"Cells meeting the criterion: **none** (0 of {t.height}).")
        if cells_alt:
            a("With the alternative FPR denominator (all multi events, incl. n = 2): "
              + ", ".join(f"${s:,} at {k} h" for s, k in cells_alt) + ".")
        else:
            a("With the alternative FPR denominator (all multi events): none either.")
        a("")
    big = tabs.get(("n10", 24, 1.0))
    if big is not None and big.height:
        r0 = big.row(0, named=True)
        a("## Supplementary: events with >= 10 comparable outcomes only (24 h)\n")
        a(f"Only these can reach a mid-rank percentile of 0.95. FPR among them is "
          f"**{r0['fpr_untouched']:.1%}** (95% CI {r0['fpr_ci_lo']:.1%} to {r0['fpr_ci_hi']:.1%}, "
          f"n = {r0['n_events']}), so real winners of large events are routinely already at the top "
          f"of the pre-reveal ranking.\n")
        a(_fmt_grid(big))
        a("")
    half = tabs.get(("all", 24, 0.5))
    if half is not None and half.height:
        a("## Sensitivity: half the estimated impact (24 h)\n")
        a(_fmt_grid(half))
        a("")
    a("## Verdict for the README\n")
    meets_any = int(t24["meets_criterion"].sum())
    small = [s for s, _ in smallest_cells(t24)]
    r24 = t24.row(0, named=True)
    if meets_any and min(small) <= 20000:
        a("At least one cell with stake <= $20k meets the PLAN criterion (see the table); only those "
          "cells may be called 'working'.")
    else:
        why = (
            f"the false-positive rate on untouched events ({r24['fpr_untouched']:.1%}) already exceeds "
            f"{CRIT_FPR:.0%}" if r24["fpr_untouched"] > CRIT_FPR else "no cell reaches the detection rate"
        )
        a(f"**No cell with stake <= $20k meets the PLAN criterion** ({why}). Per the PLAN, the "
          "within-event test is reported as a **descriptive result, not a detector**.")
    a("")
    a("## Placebos (4C)\n")
    a("Pass rules (PLAN): (a) window ending 48 h before the reveal passes if its permutation p > 0.01 "
      "or its excess share above the 95th percentile of losers (observed minus null) is at most half "
      "that of the real window; (b) 100 shuffled winners pass if the real share sits inside the "
      "central 95% interval of the shuffle distribution.\n")
    a("| placebo | window | set | n | observed | null | excess | p | real-window excess | pass |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in plc.iter_rows(named=True):
        if r["placebo"].startswith("a"):
            a(f"| (a) 48 h-earlier window | {r['window_h']} h | {r['event_set']} | {r['n_events']} | "
              f"{r['observed_share_above_q95']:.3f} | {r['null_share_above_q95']:.3f} | {r['excess']:+.3f} | "
              f"{r['permutation_p']:.4f} | {r['real_window_excess']:+.3f} | **{'PASS' if r['pass'] else 'FAIL'}** |")
        else:
            a(f"| (b) shuffled winners: {r['event_set']} | {r['window_h']} h | real window | {r['n_events']} | "
              f"{r['observed_share_above_q95']:.3f} | {r['null_share_above_q95']:.3f} | {r['excess']:+.3f} | "
              f"central 95%: [{r['shuffle_lo95']:.3f}, {r['shuffle_hi95']:.3f}] | "
              f"real at pct {r['real_percentile_in_shuffles']:.2f} of shuffles | **{'PASS' if r['pass'] else 'FAIL'}** |")
    a("")
    a("Reading: a real winner that is already high in the pre-reveal ranking is not by itself "
      "evidence of an informed buyer, because markets converge on the eventual winner as the reveal "
      "nears (predictable convergence, public news). The placebo (a) shows how much of the excess "
      "exists two days earlier, when a window-limited informed buyer is not expected to dominate.\n")
    a("## Caveats\n")
    a("* Simulation with an empirical, linear-ish impact; not real-world detectability.")
    a("* The 24 h / 12 h statistic cannot see buys placed before the window start (about 25 h / 13 h "
      "before the reveal); long leads show only the false-positive floor.")
    a("* The statistic uses only the two window end points, so every lead inside the window (24 h window: k = 2 to 24 h) gives the same rate; identical columns are expected, not a bug.")
    a("* Percentile detection needs >= 10 comparable outcomes; most events have fewer.")
    a("* Hourly last-price impact estimates carry bid-ask bounce (attenuation) and endogeneity; fits are "
      "per market from thin tapes; fallback markets use a pooled median.")
    a("* Splitting, wallet fragmentation and public-news leaks are not modelled.")
    a(f"* Wall time {meta['wall_s']:.0f} s.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def run_within_event_validation(
    n_events: int | None = None,
    seed: int = SEED,
    n_boot: int = 1000,
    n_perm: int = 2000,
    out_dir: Path | None = None,
    snapshot_dir: Path = SNAPSHOTS,
    log=print,
) -> dict:
    t0 = time.time()
    out = Path(out_dir) if out_dir else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    events, info, aux = load_sim_events(snapshot_dir, n_events, seed, log)
    own, fb, imp = load_impacts(info, snapshot_dir, log)
    betas, src, p_refs = [], [], []
    for ev in events:
        b, s = impact_for(info[ev.event_id]["winner"], ev.p_ref, own, fb)
        betas.append(b)
        src.append(s)
        p_refs.append(ev.p_ref)
    betas = np.array(betas)
    meta = {
        "n_events": len(events),
        "excl": aux["excl"],
        "n_own": src.count("own"),
        "n_fallback": src.count("fallback"),
        "median_beta": float(np.median(betas)) if len(betas) else float("nan"),
        "fb_all": fb["all"],
        "n_valid": int(imp["valid"].sum()),
        "n_fitted": imp.height,
        "n_boot": n_boot,
        "seed": seed,
        "wall_s": 0.0,
    }
    tabs: dict = {}
    for scale in (1.0, 0.5):
        for w in WINDOWS_H:
            alt = aux["alt"][f"pct_{w}h"]
            alt_fpr = float((alt >= DETECT_PCT).mean()) if len(alt) else None
            subsets = ("all", "n10") if (scale == 1.0 and w == 24) else ("all",)
            for sub in subsets:
                t = power_table(events, betas, w, sub, scale, alt_fpr if sub == "all" else None, n_boot, seed)
                tabs[(sub, w, scale)] = t
                log(f"grid subset={sub} window={w} scale={scale}: done")
    main = pl.concat([t for (sub, w, sc), t in tabs.items() if sc == 1.0 and t.height], how="vertical_relaxed")
    main.write_csv(out / "validation_within_event_power.csv")
    half = pl.concat([t for (sub, w, sc), t in tabs.items() if sc == 0.5 and t.height], how="vertical_relaxed")
    half.write_csv(out / "validation_within_event_power_half_impact.csv")
    heatmap({w: tabs[("all", w, 1.0)] for w in WINDOWS_H}, out / "figures" / "validation_within_event_power.png")
    plc = run_placebos(info, snapshot_dir, n_perm=n_perm, seed=seed, log=log)
    plc.write_csv(out / "validation_within_event_placebos.csv")
    meta["wall_s"] = time.time() - t0
    write_report(out / "validation_within_event.md", tabs, plc, meta)
    # placebo markdown (aggregate only)
    pm = ["# Within-event placebos (phase 4C)\n", "Aggregate-only; see validation_within_event_placebos.csv.\n"]
    for r in plc.iter_rows(named=True):
        pm.append(f"* {r['placebo']} / {r['window_h']} h / {r['event_set']}: n={r['n_events']}, "
                  f"observed {r['observed_share_above_q95']:.3f}, null {r['null_share_above_q95']:.3f}, "
                  f"**{'PASS' if r['pass'] else 'FAIL'}**")
    (out / "validation_within_event_placebos.md").write_text("\n".join(pm) + "\n", encoding="utf-8")
    t24 = tabs[("all", 24, 1.0)]
    summary = (
        f"events={len(events)} plantable24={t24['n_plantable'][0]} FPR24={t24['fpr_untouched'][0]:.3f} "
        f"cells_meeting_24h={int(t24['meets_criterion'].sum())}/{t24.height} wall={meta['wall_s']:.0f}s"
    )
    return {"summary": summary, "tables": tabs, "placebos": plc, "meta": meta}
