"""Direction-free burst / repricing news times (phase 3c, Part B method 2).

Problem: ``deadline`` and ``unscheduled_event`` markets have no independent news time (the
``scheduled_end`` rule only covers ``scheduled_reveal`` markets), so they only have outcome-selected
jump times (``changepoint.detect_jump`` looks for a rise toward the winner) and cannot enter the
calibration or the repeat test. This module finds a time WITHOUT looking at which side won.

Method (all parameters fixed in advance, none tuned)
----------------------------------------------------
1. YES-price coordinates. Every price point / taker fill is converted to the YES price
   (``price`` on a Yes token, ``1 - price`` on a No token). The outcome is NOT used to pick a window or a
   direction; it is used only afterwards to split the validation (YES-winner vs NO-winner twins).
2. Hourly grid of bins from the first to the last observation: ``P[k]`` = last YES price observed at
   or before the end of bin k (forward filled, never backward), ``C[k]`` = number of taker fills in bin k.
3. Sliding window of ``W = 6`` hourly bins. For each start bin s (s >= 1) the repricing is
   ``delta = P[s+W-1] - P[s-1]`` (signed, in YES price). Burst trade z of the window::

       n      = fills in the W bins;   lam = (N - n) / (H - W)   (fills per bin outside the window)
       z      = (n - W*lam) / sqrt(max(W*lam, 1))

   A window qualifies if ``|delta| >= 0.10``, ``n >= 5`` and ``z >= 3``. The burst is the qualifying
   window with the largest ``|delta|`` in EITHER direction (ties: the earliest). No qualifying window:
   ``news_ts_burst`` null (reason ``no_burst``; ``short_history`` if fewer than 12 bins).
4. ``news_ts_burst`` = start of the first bin k of the window whose price has covered at least 20 % of
   the window's move away from the pre-level ``P[s-1]`` (``|P[k] - P[s-1]| >= 0.2 |delta|``). This is
   "when the YES price first leaves its pre-burst level", at hour resolution (news at time T inside a
   bin gives the start of that bin, i.e. 0 to 1 h early). ``burst_size`` = signed ``delta`` (YES
   price units, positive = YES repriced up), ``burst_trade_z`` = z.
5. Flow BEFORE the burst: ``flow.market_flow`` (legacy window ``[burst - 24 h, burst)``, history blocks
   of the market's own earlier 24 h, robust z) with the burst start as ``js``. The window is chosen by the
   direction-free burst start only; the flow measures themselves are signed toward the eventual winner
   (a measure, not a selector), as in flow.py. Columns ``bf_*``.

Validation (set up before looking at any result)
------------------------------------------------
Criteria fixed 2026-11-13T18:15Z (UTC; wall clock when the task started, before any result or any
labelled-case rank was computed or looked at); they are the criteria given by the requester:

  Validation set = markets with an independent time in ``news_independent.parquet`` (``override`` or
  ``scheduled_end``; the latter is date-level, so hours of noise are expected, the override market is
  exact). error = ``news_ts_burst - news_ts_indep`` in hours. Markets where no burst was found count
  as an infinite error (they are neither within 6 h nor within 24 h and push the median up); the
  denominator is ALL validation markets.

  burst_validated = (median |error| <= 6 h) AND (share |error| <= 24 h >= 70 %) AND
                    (median |error| of YES-winner markets <= 2 * max(median |error| of NO-winner
                    markets, 1 h)).

  If met, ``deadline`` / ``unscheduled_event`` markets with a burst get ``independent_burst = True``,
  ``indep_method_burst = 'burst'`` (in ``news_burst.parquet``; ``news_independent.parquet`` is NEVER
  touched) and the wallet pipeline may use ``news_ts_burst`` as a calibration-eligible cutoff
  (``cutoff_method = 'independent_burst'``) via ``positions.market_cutoffs(..., news_burst=...)``,
  only if ``burst_validated`` is true. Otherwise those markets stay screen-only.

Caveat that holds even when validated: the burst time is price-selected (the biggest abrupt repricing
exists only in markets where something happened), so a burst cutoff is "independent of the winning
direction", not independent of the price path.

Run: ``python -m pmdetect.markets.newstime_burst`` writes ``data/snapshots/news_burst.parquet``
(offline, Parquet only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.markets.series import SNAPSHOTS

W = 6  # window, hourly bins
MIN_DELTA = 0.10
MIN_TRADES = 5
MIN_Z = 3.0
START_FRAC = 0.20
MIN_BINS = 12

CRIT_MEDIAN_ABS_H = 6.0
CRIT_SHARE_24H = 0.70
CRIT_YES_NO_RATIO = 2.0
CRIT_NO_FLOOR_H = 1.0
CRITERIA_FIXED_AT = "2026-11-13T18:15Z"

BURST_COLS = [
    "condition_id", "news_ts_burst", "burst_size", "burst_trade_z", "method", "validation_group",
    "burst_reason", "burst_validated", "independent_burst", "indep_method_burst",
]  # fmt: skip


# --------------------------------------------------------------------------- detection


def detect_burst(P: np.ndarray, C: np.ndarray) -> dict:
    """Largest abrupt repricing (either direction) with a trade burst on an hourly grid.

    ``P`` forward-filled YES price per bin, ``C`` fills per bin (same length). Returns
    ``{start_bin, delta, z, n, reason}``; ``start_bin`` is None when nothing qualifies.
    """
    H = len(P)
    out = {"start_bin": None, "delta": None, "z": None, "n": None, "reason": None}
    if H < max(MIN_BINS, W + 2):
        out["reason"] = "short_history"
        return out
    cs = np.concatenate([[0.0], np.cumsum(C, dtype=float)])
    N = cs[-1]
    nw = H - W  # candidate windows j = s-1 in 0 .. H-W-1 cover bins j+1 .. j+W
    delta = P[W:] - P[:nw]
    n = cs[W + 1 : W + 1 + nw] - cs[1 : 1 + nw]
    lam = (N - n) / max(H - W, 1)
    z = (n - W * lam) / np.sqrt(np.maximum(W * lam, 1.0))
    ok = (np.abs(delta) >= MIN_DELTA) & (n >= MIN_TRADES) & (z >= MIN_Z)
    if not ok.any():
        out["reason"] = "no_burst"
        return out
    j = int(np.argmax(np.where(ok, np.abs(delta), -1.0)))  # first (earliest) maximum
    d = float(delta[j])
    seg = np.abs(P[j + 1 : j + W + 1] - P[j])
    k = int(np.argmax(seg >= START_FRAC * abs(d)))  # first bin that covers >= 20 % of the move
    out.update(start_bin=j + 1 + k, delta=d, z=float(z[j]), n=float(n[j]))
    return out


def yes_points(prices: pl.DataFrame, taker: pl.DataFrame) -> pl.DataFrame:
    """condition_id, ts, ypx (YES price), n (1 for a taker fill, 0 for a price-history point).

    Only Yes/No-labelled tokens contribute (others have no YES coordinate). ``prices`` needs
    condition_id, outcome_label, ts, price; ``taker`` the same columns per fill.
    """

    def conv(df: pl.DataFrame, n: int) -> pl.DataFrame:
        lab = pl.col("outcome_label").str.to_lowercase()
        return (
            df.filter(lab.is_in(["yes", "no"]) & pl.col("price").is_between(0.0, 1.0))
            .select(
                "condition_id",
                "ts",
                pl.when(lab == "yes").then(pl.col("price")).otherwise(1.0 - pl.col("price")).alias("ypx"),
                pl.lit(n, dtype=pl.Int32).alias("n"),
            )
        )

    return pl.concat([conv(prices, 0), conv(taker, 1)])


def hourly_bins(points: pl.DataFrame) -> pl.DataFrame:
    """condition_id, hour (bin start), ypx (last in the bin), n (fills in the bin); sorted."""
    return (
        points.sort(["condition_id", "ts"])
        .with_columns(pl.col("ts").dt.truncate("1h").alias("hour"))
        .group_by("condition_id", "hour", maintain_order=True)
        .agg(pl.col("ypx").last(), pl.col("n").sum())
        .sort(["condition_id", "hour"])
    )


def detect_bursts(hourly: pl.DataFrame) -> pl.DataFrame:
    """One row per market in ``hourly``: condition_id, news_ts_burst, burst_size, burst_trade_z,
    burst_n_trades, burst_reason."""
    cids = hourly["condition_id"].to_numpy()
    hrs = hourly["hour"].dt.epoch("us").to_numpy() // 3_600_000_000  # hours since epoch
    ypx = hourly["ypx"].to_numpy()
    cnt = hourly["n"].to_numpy().astype(float)
    starts = np.flatnonzero(np.concatenate([[True], cids[1:] != cids[:-1]]))
    ends = np.concatenate([starts[1:], [len(cids)]])
    rows = []
    for a, b in zip(starts, ends, strict=True):
        h = hrs[a:b]
        H = int(h[-1] - h[0]) + 1
        idx = (h - h[0]).astype(int)
        P = np.full(H, np.nan)
        P[idx] = ypx[a:b]
        C = np.zeros(H)
        C[idx] = cnt[a:b]
        last = np.maximum.accumulate(np.where(np.isnan(P), -1, np.arange(H)))
        P = P[last]  # first bin always observed, so last >= 0
        r = detect_burst(P, C)
        ts = None
        if r["start_bin"] is not None:
            ts = (int(h[0]) + r["start_bin"]) * 3_600_000_000
        rows.append((cids[a], ts, r["delta"], r["z"], r["n"], r["reason"]))
    df = pl.DataFrame(
        rows,
        schema={"condition_id": pl.String, "ts": pl.Int64, "burst_size": pl.Float64,
                "burst_trade_z": pl.Float64, "burst_n_trades": pl.Float64, "burst_reason": pl.String},
        orient="row",
    )  # fmt: skip
    return df.select(
        "condition_id",
        pl.from_epoch("ts", time_unit="us").cast(pl.Datetime("us")).alias("news_ts_burst"),
        "burst_size", "burst_trade_z", "burst_n_trades", "burst_reason",
    )  # fmt: skip


# --------------------------------------------------------------------------- validation


def validation_errors(
    burst: pl.DataFrame, indep: pl.DataFrame, markets: pl.DataFrame
) -> pl.DataFrame:
    """Validation set: condition_id, indep_method, outcome, err_h (nan = no burst found).

    ``indep``: condition_id, news_ts_indep, independent, indep_method; ``markets``: condition_id,
    outcome. Only markets with an independent time AND a YES/NO outcome are in the set."""
    v = (
        indep.filter(pl.col("independent").fill_null(False))
        .select("condition_id", "news_ts_indep", "indep_method")
        .join(markets.select("condition_id", "outcome"), on="condition_id", how="inner")
        .filter(pl.col("outcome").is_in(["YES", "NO"]))
        .join(burst.select("condition_id", "news_ts_burst"), on="condition_id", how="left")
    )
    return v.select(
        "condition_id", "indep_method", "outcome",
        ((pl.col("news_ts_burst") - pl.col("news_ts_indep")).dt.total_seconds() / 3600.0)
        .fill_null(float("nan")).alias("err_h"),
    )  # fmt: skip


def error_stats(err_h: np.ndarray) -> dict:
    """Distribution of signed errors; undetected (nan) counts as infinite absolute error."""
    e = np.asarray(err_h, dtype=float)
    n = e.size
    det = e[~np.isnan(e)]
    a_all = np.where(np.isnan(e), np.inf, np.abs(e))
    out = {
        "n": int(n), "n_detected": int(det.size),
        "detected_share": float(det.size / n) if n else float("nan"),
        "median_abs_h_all": float(np.median(a_all)) if n else float("nan"),
        "share_within_6h": float((a_all <= 6).mean()) if n else float("nan"),
        "share_within_24h": float((a_all <= 24).mean()) if n else float("nan"),
    }
    if det.size:
        q1, q2, q3 = np.quantile(det, [0.25, 0.5, 0.75])
        out.update(
            median_signed_h=float(q2), q25_h=float(q1), q75_h=float(q3), mean_signed_h=float(det.mean()),
            median_abs_h_detected=float(np.median(np.abs(det))),
            share_within_6h_detected=float((np.abs(det) <= 6).mean()),
            share_within_24h_detected=float((np.abs(det) <= 24).mean()),
        )  # fmt: skip
    return out


def criterion(err_h: np.ndarray, outcome: np.ndarray) -> dict:
    """The pre-registered acceptance criterion (see the module docstring). Returns the three
    component values, booleans and ``burst_validated``."""
    e = np.asarray(err_h, dtype=float)
    o = np.asarray(outcome)

    def med_abs(mask) -> float:
        a = np.where(np.isnan(e[mask]), np.inf, np.abs(e[mask]))
        return float(np.median(a)) if a.size else float("nan")

    st = error_stats(e)
    m_yes, m_no = med_abs(o == "YES"), med_abs(o == "NO")
    c1 = bool(st["median_abs_h_all"] <= CRIT_MEDIAN_ABS_H)
    c2 = bool(st["share_within_24h"] >= CRIT_SHARE_24H)
    c3 = bool(m_yes <= CRIT_YES_NO_RATIO * max(m_no, CRIT_NO_FLOOR_H))
    return {
        "median_abs_h_all": st["median_abs_h_all"], "share_within_24h": st["share_within_24h"],
        "median_abs_yes_h": m_yes, "median_abs_no_h": m_no,
        "ratio_yes_over_no": m_yes / max(m_no, CRIT_NO_FLOOR_H) if np.isfinite(m_yes) else float("inf"),
        "crit_median_ok": c1, "crit_share24_ok": c2, "crit_symmetry_ok": c3,
        "burst_validated": c1 and c2 and c3,
    }  # fmt: skip


# --------------------------------------------------------------------------- flow before the burst


def burst_flow(
    markets: pl.DataFrame, burst: pl.DataFrame, taker: pl.DataFrame, wallets: pl.DataFrame | None
) -> pl.DataFrame:
    """Flow measures in ``[burst - 24 h, burst)`` per market with a burst (flow.market_flow, legacy
    window): condition_id and ``bf_*`` columns (the window is chosen by the burst start only)."""
    from pmdetect.markets.flow import market_flow

    fs = None
    if wallets is not None and wallets.height:
        fs = wallets.select(
            "wallet", "first_seen",
            (pl.col("n_markets").cast(pl.Int64) if "n_markets" in wallets.columns
             else pl.lit(None, dtype=pl.Int64).alias("n_markets")),
        ).unique(subset=["wallet"])  # fmt: skip
    outc = dict(markets.select("condition_id", "outcome").iter_rows())
    bts = {r[0]: r[1] for r in burst.filter(pl.col("news_ts_burst").is_not_null())
           .select("condition_id", "news_ts_burst").iter_rows()}
    parts = {k[0]: v for k, v in taker.filter(pl.col("condition_id").is_in(list(bts))).partition_by(
        "condition_id", as_dict=True).items()}
    keep = ["fb_ofi_z", "fb_fresh_z", "fb_hhi_z", "fb_vol_z", "fb_n_trades", "fb_usd", "fb_reason"]
    rows = []
    for cid, js in bts.items():
        o = outc.get(cid)
        if o not in ("YES", "NO"):
            r = {c: None for c in keep}
            r["fb_reason"] = "no_outcome"
        else:
            r = market_flow(cid, o, js, parts.get(cid, pl.DataFrame()), fs, legacy=True)
        rows.append({"condition_id": cid, **{"bf_" + c[3:]: r.get(c) for c in keep}})
    schema = {"condition_id": pl.String, "bf_ofi_z": pl.Float64, "bf_fresh_z": pl.Float64,
              "bf_hhi_z": pl.Float64, "bf_vol_z": pl.Float64, "bf_n_trades": pl.Int64,
              "bf_usd": pl.Float64, "bf_reason": pl.String}  # fmt: skip
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None)


# --------------------------------------------------------------------------- assembly


def assemble(
    burst: pl.DataFrame, markets: pl.DataFrame, indep: pl.DataFrame, verdict: bool
) -> pl.DataFrame:
    """Final table: BURST_COLS (+ burst_n_trades). ``validation_group``:
    ``scheduled_validation`` (independent time known), ``scheduled_no_indep``, ``deadline``,
    ``unscheduled_event``, ``other``. Only deadline / unscheduled_event markets can become
    ``independent_burst`` (and only when ``verdict``)."""
    df = (
        burst.join(markets.select("condition_id", "info_type"), on="condition_id", how="left")
        .join(indep.select("condition_id", pl.col("independent").fill_null(False).alias("_ind")),
              on="condition_id", how="left")
        .with_columns(pl.col("_ind").fill_null(False))
    )
    grp = (
        pl.when((pl.col("info_type") == "scheduled_reveal") & pl.col("_ind")).then(pl.lit("scheduled_validation"))
        .when(pl.col("info_type") == "scheduled_reveal").then(pl.lit("scheduled_no_indep"))
        .when(pl.col("info_type") == "deadline").then(pl.lit("deadline"))
        .when(pl.col("info_type") == "unscheduled_event").then(pl.lit("unscheduled_event"))
        .otherwise(pl.lit("other"))
    )  # fmt: skip
    cand = pl.col("validation_group").is_in(["deadline", "unscheduled_event"]) & pl.col(
        "news_ts_burst"
    ).is_not_null()
    return (
        df.with_columns(grp.alias("validation_group"), pl.lit("hourly_window_burst").alias("method"),
                        pl.lit(bool(verdict)).alias("burst_validated"))
        .with_columns((pl.col("burst_validated") & cand).alias("independent_burst"))
        .with_columns(
            pl.when(pl.col("independent_burst")).then(pl.lit("burst")).otherwise(pl.lit("none"))
            .alias("indep_method_burst")
        )
        .drop("info_type", "_ind")
        .select(*BURST_COLS, "burst_n_trades")
    )


def build(snapshot_dir: Path = SNAPSHOTS, write: bool = True, log=print) -> dict:
    """Detect bursts for every universe market, validate, write news_burst.parquet."""
    from pmdetect.markets.series import load_inputs

    markets, prices = load_inputs(snapshot_dir)
    ni = pl.read_parquet(snapshot_dir / "news_independent.parquet")
    taker = pl.read_parquet(
        snapshot_dir / "trades_taker.parquet",
        columns=["condition_id", "wallet", "side", "outcome_label", "size", "usd", "price", "ts", "token_id"],
    )
    pts = yes_points(prices.select("condition_id", "outcome_label", "ts", "price"), taker)
    hourly = hourly_bins(pts)
    del pts
    log(f"hourly bins: {hourly.height} markets: {hourly['condition_id'].n_unique()}")
    burst = detect_bursts(hourly)
    del hourly
    val = validation_errors(burst, ni, markets)
    crit = criterion(val["err_h"].to_numpy(), val["outcome"].to_numpy())
    out = assemble(burst, markets, ni, crit["burst_validated"])
    wallets = pl.read_parquet(snapshot_dir / "wallets.parquet")
    flow = burst_flow(markets, burst, taker, wallets)
    out = out.join(flow, on="condition_id", how="left")
    if write:
        out.write_parquet(snapshot_dir / "news_burst.parquet")
    return {"burst": out, "validation": val, "criterion": crit,
            "stats": error_stats(val["err_h"].to_numpy())}  # fmt: skip


SCORE_COLS = [
    "news_ts_burst", "burst_size", "burst_trade_z", "burst_validated", "independent_burst",
    "validation_group", "bf_ofi_z", "bf_fresh_z", "bf_hhi_z", "bf_vol_z", "bf_n_trades", "bf_reason",
]  # fmt: skip


def burst_score_columns(snapshot_dir: Path = SNAPSHOTS) -> pl.DataFrame | None:
    """condition_id + SCORE_COLS from news_burst.parquet for ``score-markets`` (None if absent)."""
    p = snapshot_dir / "news_burst.parquet"
    if not p.exists():
        return None
    b = pl.read_parquet(p)
    return b.select("condition_id", *[c for c in SCORE_COLS if c in b.columns]).unique("condition_id")


def summary_section(snapshot_dir: Path = SNAPSHOTS) -> str:
    """Markdown section (aggregate only) for results/phase2_summary.md."""
    from pmdetect.markets.series import load_inputs

    p = snapshot_dir / "news_burst.parquet"
    if not p.exists():
        return ""
    b = pl.read_parquet(p)
    markets, _ = load_inputs(snapshot_dir)
    ni = pl.read_parquet(snapshot_dir / "news_independent.parquet")
    v = validation_errors(b, ni, markets)
    e = v["err_h"].to_numpy()
    st, cr = error_stats(e), criterion(e, v["outcome"].to_numpy())
    ok = bool(cr["burst_validated"])
    f = lambda x, n=1: "NaN" if x is None else f"{x:.{n}f}"  # noqa: E731
    g = b.group_by("validation_group").agg(
        pl.len().alias("n"), pl.col("news_ts_burst").is_not_null().sum().alias("burst")
    ).sort("validation_group")
    L = ["## Direction-free burst news times (phase 3c Part B)", "",
         f"Method and the acceptance criterion (fixed {CRITERIA_FIXED_AT}, before any result) are in "
         "`markets/newstime_burst.py`. Validation set = scheduled reveals with an independent time "
         f"(n = {st['n']}; one exact override, the rest date-level `scheduled_end`).", "",
         f"- Burst found for {st['n_detected']} of {st['n']} ({100 * st['detected_share']:.1f} %); error = "
         f"burst - independent time (h): median {f(st.get('median_signed_h'))}, quartiles "
         f"{f(st.get('q25_h'))} / {f(st.get('q75_h'))}.",
         f"- Median |error| over all (undetected = infinite): {f(cr['median_abs_h_all'])} h "
         f"(criterion <= {CRIT_MEDIAN_ABS_H:g} h: {'met' if cr['crit_median_ok'] else 'NOT met'}); "
         f"within 6 h: {100 * st['share_within_6h']:.1f} %, within 24 h: {100 * st['share_within_24h']:.1f} % "
         f"(criterion >= {100 * CRIT_SHARE_24H:.0f} %: {'met' if cr['crit_share24_ok'] else 'NOT met'}).",
         f"- YES-winner median |error| {f(cr['median_abs_yes_h'])} h vs NO-winner {f(cr['median_abs_no_h'])} h "
         f"(criterion YES <= {CRIT_YES_NO_RATIO:g} x NO: {'met' if cr['crit_symmetry_ok'] else 'NOT met'}).",
         f"- **burst_validated = {ok}**. "
         + ("deadline / unscheduled_event markets with a burst are independent (method `burst`) for the wallet "
            "calibration and repeat test." if ok else
            "deadline / unscheduled_event markets stay screen-only (one-shot screen); the wallet calibration is "
            "unchanged."), "",
         "Markets per group (burst found): "
         + ", ".join(f"{r['validation_group']} {r['burst']}/{r['n']}" for r in g.iter_rows(named=True)) + ".", ""]
    return "\n".join(L)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    res = build()
    print(res["criterion"])
    print(res["stats"])
