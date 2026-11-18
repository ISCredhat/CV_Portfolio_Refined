"""Phase 2 / 2b pipeline: series -> baseline -> score -> jumps -> flow -> combine -> plots -> summary.

Everything in the markdown summary is computed from the data (no hand-typed numbers).
Reads only Parquet snapshots; never opens DuckDB.

Extra labelled cases (``data/snapshots/extra_*.parquet``, see ``markets.cases_extra``) are scored
next to the pilot but are ``out_of_universe``: the level baseline is fitted on pilot markets with
an independent news time only (an independent extra such as Maduro gets an out-of-sample baseline
from the model fitted on all pilot independents), and combine excludes them from every percentile
pool and flag cut-off.
"""

from __future__ import annotations

import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.markets import series as S
from pmdetect.markets.baseline import _fit_fold, _predict, fit_predict_cross, fit_predict_timesplit
from pmdetect.markets.cases_extra import (
    EXTRA_WALLETS,
    extra_prices_labelled,
    load_extra,
)
from pmdetect.markets.changepoint import detect_jumps, price_news_ts, winner_points
from pmdetect.markets.combine import (
    COMPONENTS,
    FLOW_Z_COLS,
    MIN_COMPONENTS,
    combine,
    rank_agreement,
)
from pmdetect.markets.flow import FLOW_COLS, flow_before_move, jump_initiators
from pmdetect.markets.score import score_markets
from pmdetect.markets.signals import compute_signals, load_taker_trades
from pmdetect.report.plots import plot_d_histogram, plot_market
from pmdetect.store.db import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
TAKER_COLS = ["condition_id", "wallet", "side", "outcome_label", "price", "usd", "ts"]
PX_COLS = ["condition_id", "outcome_label", "ts", "price"]
NOBEL_CASE = "nobel_peace_2025"
MARKET_COLS = [
    "condition_id", "question", "info_type", "outcome", "news_ts", "news_ts_method", "closed_ts",
    "in_universe", "news_ts_indep", "indep_method", "independent", "news_ts_best",
    "out_of_universe",
]  # fmt: skip


def peak_memory_mb() -> float:
    """Peak resident set of this process in MB (Windows PeakWorkingSetSize; POSIX ru_maxrss)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t)
                    for n in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                    )
                ]  # fmt: skip

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / 2**20
        else:
            import resource

            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return kb / (1024 if sys.platform != "darwin" else 1024 * 1024)
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    return float("nan")


class Timings(dict):
    """Wall time per pipeline stage; ``stage(name)`` is a context manager (also logs to stderr)."""

    @contextmanager
    def stage(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self[name] = time.time() - t0
            print(f"[phase2] {name}: {self[name]:.1f}s  peak {peak_memory_mb():.0f} MB",
                  file=sys.stderr, flush=True)  # fmt: skip


def _fmt(x, nd: int = 3) -> str:
    return "NaN" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- inputs


def build_inputs(snapshot_dir: Path) -> dict:
    """Universe + extra markets, prices (with outcome labels), price points and the taker tape."""
    from pmdetect.markets.newstime_indep import build

    markets, prices = S.load_inputs(snapshot_dir)
    ni_path = snapshot_dir / "news_independent.parquet"
    ni = pl.read_parquet(ni_path) if ni_path.exists() else build(snapshot_dir)
    markets = (
        markets.join(ni, on="condition_id", how="left")
        .with_columns(
            pl.col("independent").fill_null(False),
            pl.coalesce("news_ts_indep", "news_ts").alias("news_ts_best"),
            pl.lit(False).alias("out_of_universe"),
        )
    )
    pilot_ids = markets["condition_id"].to_list()
    ind_ids = markets.filter(pl.col("independent"))["condition_id"].to_list()
    taker = load_taker_trades(columns=TAKER_COLS)
    trades = taker.select(*PX_COLS)  # price points (zero-copy columns of the tape)
    wallets = pl.read_parquet(snapshot_dir / "wallets.parquet")
    extra = load_extra(snapshot_dir)
    extra_meta = None
    if extra is not None:
        xm, xt, xp = extra
        xpl = extra_prices_labelled(xm, xp)
        pts = winner_points(xm, xpl, xt)
        news_px = {}
        for cid in xm["condition_id"].to_list():
            if cid in pts:
                t = price_news_ts(*pts[cid])
                news_px[cid] = None if t is None else t.astype("datetime64[us]").item()
        xm2 = xm.with_columns(
            pl.col("condition_id").replace_strict(news_px, default=None, return_dtype=pl.Datetime("us"))
            .alias("news_ts"),
            pl.lit("price_cross_0.9").alias("news_ts_method"),
            pl.lit(False).alias("in_universe"),
            pl.col("news_ts_indep").is_not_null().alias("independent"),
        ).with_columns(
            pl.coalesce("news_ts_indep", "news_ts").alias("news_ts_best"),
            pl.lit(True).alias("out_of_universe"),
        )
        extra_meta = xm2  # keeps every extra row (case mapping / source for the labels and summary)
        # With the full universe loaded, many extra candidate markets are universe markets already
        # (scored once, as universe members; their case label still comes from extra_meta). Only
        # the genuinely out-of-universe extras are appended, so no market is counted twice.
        uni = set(pilot_ids)
        keep = pl.col("condition_id").is_in(list(uni)).not_()
        xm_new, xt_new, xpl_new = xm2.filter(keep), xt.filter(keep), xpl.filter(keep)
        markets = pl.concat(
            [markets.select(MARKET_COLS), xm_new.select(MARKET_COLS)], how="vertical_relaxed"
        )
        px_cols = ["condition_id", "token_id", "outcome_label", "ts", "price"]
        prices = pl.concat([prices.select(px_cols), xpl_new.select(px_cols)])
        trades = pl.concat([trades.select(PX_COLS), xt_new.select(PX_COLS)])
        taker = pl.concat([taker, xt_new.select(TAKER_COLS)])
        if EXTRA_WALLETS.exists():
            xw = pl.read_parquet(EXTRA_WALLETS).select(
                "wallet", "first_seen",
                pl.lit(None, dtype=pl.Int32).alias("n_markets"),  # in-tape count is not global
                pl.lit(None, dtype=pl.Float64).alias("total_usd"),
            )
            wallets = pl.concat(
                [wallets.select("wallet", "first_seen", pl.col("n_markets").cast(pl.Int32),
                                pl.col("total_usd")), xw], how="vertical_relaxed"
            ).unique(subset=["wallet"], keep="first")
    return {"markets": markets, "prices": prices, "trades": trades, "taker": taker,
            "wallets": wallets, "ni_ids": ind_ids, "pilot_ids": pilot_ids, "extra_meta": extra_meta,
            "snapshot_dir": snapshot_dir}  # fmt: skip


def load_within_event(snapshot_dir: Path) -> pl.DataFrame | None:
    """Builder A's within_event.parquet with every column but condition_id prefixed ``we_``."""
    p = snapshot_dir / "within_event.parquet"
    if not p.exists():
        return None
    w = pl.read_parquet(p).unique(subset=["condition_id"], keep="first")
    return w.rename({c: f"we_{c}" for c in w.columns if c != "condition_id"})


def _timesplit_with_extras(series, q0, markets, ind_pilot_ids):
    """Time-split baseline: pilot independents and extra independents are scored by expanding-window
    models fitted on EARLIER pilot independents only (extras never enter a fit).
    Returns (baseline, plan)."""
    it = markets.select("condition_id", "info_type")
    ext_ids = markets.filter(pl.col("out_of_universe") & pl.col("independent"))["condition_id"].to_list()
    news = markets.select("condition_id", pl.col("news_ts_best").alias("news_ts"))
    return fit_predict_timesplit(
        series, q0, it, news, score_ids=[*ind_pilot_ids, *ext_ids], train_ids=ind_pilot_ids
    )


def _baseline_with_extras(
    series, q0, markets, ind_pilot_ids, seed: int = 0, scheme: str = "timesplit"
) -> pl.DataFrame:
    """Baseline frame for the chosen scheme. ``crossfit``: cross-fitted on pilot independents +
    out-of-sample prediction for extra independents (fitted on all pilot independents).
    ``timesplit``: see ``_timesplit_with_extras``."""
    if scheme == "timesplit":
        return _timesplit_with_extras(series, q0, markets, ind_pilot_ids)[0]
    it = markets.select("condition_id", "info_type")
    base = fit_predict_cross(series, q0, it, seed=seed, independent_ids=ind_pilot_ids)
    ext_ids = markets.filter(pl.col("out_of_universe") & pl.col("independent"))["condition_id"].to_list()
    if not ext_ids or base.height == 0:
        return base
    df = (
        series.join(q0.select("condition_id", "q0"), on="condition_id", how="inner")
        .join(it, on="condition_id", how="inner")
        .filter(pl.col("q0").is_not_null() & pl.col("info_type").is_not_null())
    )
    train = df.filter(pl.col("condition_id").is_in(ind_pilot_ids))
    test = df.filter(pl.col("condition_id").is_in(ext_ids))
    if test.height == 0:
        return base
    pred = _predict(_fit_fold(train), test)
    extra = test.select("condition_id", "t", "tau_h", pl.lit(-1, dtype=pl.Int64).alias("fold")).with_columns(
        pl.Series("m10", pred[:, 0]), pl.Series("m50", pred[:, 1]), pl.Series("m90", pred[:, 2])
    )
    return pl.concat([base, extra.select(base.columns)], how="vertical_relaxed")


def _pool_rank(df: pl.DataFrame, col: str) -> pl.Series:
    """1 + number of universe-pool markets with a strictly larger ``col``; null where ``col`` is null."""
    pool = np.sort(df.filter(~pl.col("out_of_universe") & pl.col(col).is_not_null())[col].to_numpy())
    v = df[col].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    ok = ~np.isnan(v)
    r = np.full(len(v), -1, dtype=np.int64)
    r[ok] = pool.size - np.searchsorted(pool, v[ok], side="right") + 1
    return pl.Series(f"rank_{col}", [int(x) if x >= 0 else None for x in r], dtype=pl.Int64)


def label_table(df: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Labelled cases joined to the scored table."""
    lab = (
        labels.filter(pl.col("condition_id").is_not_null())
        .group_by("condition_id")
        .agg(
            pl.col("case_name").unique().sort().str.join(", ").alias("lab_case"),
            pl.col("label").unique().sort().str.join(", ").alias("lab_label"),
        )
    )
    return (
        lab.join(df, on="condition_id", how="inner")
        .drop([c for c in ("case_name", "case_label") if c in df.columns])
        .rename({"lab_case": "case_name", "lab_label": "case_label"})
        .sort(["case_label", "case_name"], descending=[True, False])
    )


def run_phase2(
    results_dir: Path = RESULTS,
    snapshot_dir: Path = S.SNAPSHOTS,
    max_case_plots: int = 80,
    max_flag_plots: int = 60,
    private_dir: Path | None = None,
    baseline_scheme: str = "timesplit",
) -> dict:
    """Full-universe pipeline. Figures: flagged markets (at most ``max_flag_plots``, best Fisher
    scores first) and labelled cases (at most ``max_case_plots``) only, never one per market."""
    t_all = time.time()
    tm = Timings()
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("*.png"):  # figures are regenerated; drop stale ones
        old.unlink()

    with tm.stage("inputs"):
        inp = build_inputs(snapshot_dir)
        markets, prices, trades, taker, wallets = (
            inp[k] for k in ("markets", "prices", "trades", "taker", "wallets")
        )
    with tm.stage("series"):
        series = S.winner_series(markets, prices, trades, news_col="news_ts_best")
        q0 = S.q0_table(series)
    with tm.stage("baseline"):
        # both schemes are always fitted: the chosen one drives every score, the other only fills
        # D_crossfit / D_timesplit for the comparison
        base_ts, plan = _timesplit_with_extras(series, q0, markets, inp["ni_ids"])
        base_cf = _baseline_with_extras(series, q0, markets, inp["ni_ids"], scheme="crossfit")
        baseline = base_ts if baseline_scheme == "timesplit" else base_cf
    with tm.stage("score"):
        ni = markets.select("condition_id", "news_ts_indep", "indep_method", "independent")
        unscor = plan.filter(~pl.col("scorable"))
        scores = score_markets(
            markets, series, baseline, q0, news_indep=ni, scheme=baseline_scheme,
            unscorable=unscor if baseline_scheme == "timesplit" else None,
        )
        d_ts = score_markets(markets, series, base_ts, q0, news_indep=ni, scheme="timesplit",
                             unscorable=unscor).select("condition_id", pl.col("D").alias("D_timesplit"))
        d_cf = score_markets(markets, series, base_cf, q0, news_indep=ni, scheme="crossfit").select(
            "condition_id", pl.col("D").alias("D_crossfit"))
        scores = scores.join(d_ts, on="condition_id", how="left").join(d_cf, on="condition_id", how="left")
        del base_ts, base_cf, d_ts, d_cf
    with tm.stage("signals"):
        sig_markets = markets.with_columns(pl.col("news_ts_best").alias("news_ts"))
        signals = compute_signals(sig_markets, taker, wallets)
    with tm.stage("jumps"):
        pts = winner_points(markets, prices, taker)
        jumps = detect_jumps(markets, prices, taker, pts=pts)
    with tm.stage("flow"):
        flow = flow_before_move(markets, jumps, taker, wallets, pts)
        old = flow_before_move(markets, jumps, taker, wallets, pts, legacy=True)
        flow = flow.join(
            old.select("condition_id", *[pl.col(c).alias(f"old_{c}") for c in FLOW_Z_COLS],
                       pl.col("fb_reason").alias("old_fb_reason")),
            on="condition_id", how="left",
        )  # fmt: skip
    with tm.stage("combine"):
        extra_cols = markets.select("condition_id", "out_of_universe", "closed_ts")
        sig_all = signals.join(jumps, on="condition_id", how="left").join(
            flow, on="condition_id", how="left"
        )
        we = load_within_event(snapshot_dir)
        if we is not None:
            sig_all = sig_all.join(we, on="condition_id", how="left")
        scores = scores.join(extra_cols, on="condition_id", how="left")
        df = combine(scores, sig_all)
        from pmdetect.markets.newstime_burst import burst_score_columns

        bcols = burst_score_columns(snapshot_dir)  # news_ts_burst, burst_validated, bf_* (columns only)
        if bcols is not None:
            df = df.join(bcols, on="condition_id", how="left")

    labels = pl.read_parquet(snapshot_dir / "labels.parquet")
    em = inp["extra_meta"]
    if em is not None:
        xl = em.select(
            "condition_id", "case_name",
            pl.when(pl.col("mapping") == "unambiguous").then(pl.lit("insider_suspected"))
            .otherwise(pl.lit("case_candidate")).alias("label"),
        )  # fmt: skip
        labels = pl.concat(
            [labels.select("condition_id", "case_name", "label"), xl], how="vertical_relaxed"
        )
        df = df.join(em.select("condition_id", "mapping", "news_source"), on="condition_id", how="left")
    lt = label_table(df, labels)
    lab_by = {r["condition_id"]: (r["case_name"], r["case_label"]) for r in lt.iter_rows(named=True)}
    case_ids = list(lab_by)
    df = df.with_columns(
        pl.col("condition_id").replace_strict({k: v[0] for k, v in lab_by.items()}, default=None,
                                              return_dtype=pl.Utf8).alias("case_name"),
        pl.col("condition_id").replace_strict({k: v[1] for k, v in lab_by.items()}, default=None,
                                              return_dtype=pl.Utf8).alias("case_label"),
    )  # fmt: skip
    pool_d = df.filter(~pl.col("out_of_universe") & pl.col("D").is_not_null())["D"].to_numpy()
    d_thr = float(np.quantile(pool_d, 0.99)) if pool_d.size else float("nan")
    n_scored_D = int(pool_d.size)
    for c in ("D_timesplit", "D_crossfit"):  # ranks vs the universe pool (extras: would-be rank)
        df = df.with_columns(_pool_rank(df, c).alias(f"rank_{c}"))

    with tm.stage("write_scores"):
        df.write_csv(results_dir / "market_scores.csv")
    with tm.stage("initiators"):
        xw_path = snapshot_dir / "extra_wallets.parquet"
        init = jump_initiators(
            markets, jumps, taker, pl.read_parquet(snapshot_dir / "wallets.parquet"), pts,
            labelled={k: v[0] for k, v in lab_by.items()},
            extra_wallets=pl.read_parquet(xw_path) if xw_path.exists() else None,
        )  # fmt: skip
        private_dir = Path(private_dir) if private_dir else PROJECT_ROOT / "data" / "private"
        private_dir.mkdir(parents=True, exist_ok=True)
        init.write_csv(private_dir / "jump_initiators.csv")  # wallet addresses: never in results/
    with tm.stage("plots"):
        hist = plot_d_histogram(
            df.filter(~pl.col("out_of_universe")), fig_dir / "D_histogram.png", d_thr
        )
        have_series = set(series["condition_id"].unique().to_list())
        figs: list[tuple[str, str, Path]] = []
        flagged = df.filter(pl.col("flag_fisher") | pl.col("flag_rank_mean")).sort(
            "fisher_score", descending=True, nulls_last=True
        )
        n_flagged = flagged.height
        todo = [("flagged_", r) for r in flagged.head(max_flag_plots).iter_rows(named=True)]
        seen = {r["condition_id"] for _, r in todo}
        cases = list(df.filter(pl.col("condition_id").is_in(case_ids)).iter_rows(named=True))
        cases.sort(key=lambda r: (not (r["case_name"] and NOBEL_CASE in r["case_name"]),
                                  r["case_label"] != "insider_suspected", r["case_name"] or ""))  # fmt: skip
        todo += [("case_", r) for r in cases[:max_case_plots]]
        for tag, r in todo:
            if r["condition_id"] not in have_series:
                continue
            if tag == "case_" and r["condition_id"] in seen:
                tag = "flagged_case_"
            if r.get("case_name") and NOBEL_CASE in r["case_name"]:
                tag = "case_NOBEL_"
            elif tag == "case_" and r.get("out_of_universe"):
                tag = "case_extra_"
            figs.append((tag, r["condition_id"], plot_market(r, series, baseline, taker, fig_dir, tag=tag)))

    tm["total"] = time.time() - t_all
    perf = {"timings": dict(tm), "peak_mb": peak_memory_mb(), "n_flagged": n_flagged,
            "max_flag_plots": max_flag_plots}  # fmt: skip
    with tm.stage("summary"):
        summary = write_summary(df, lt, d_thr, hist, figs, results_dir / "phase2_summary.md", wallets,
                                taker, series, baseline, inp, init=init, we=we, perf=perf,
                                plan=plan, scheme=baseline_scheme)  # fmt: skip
        from pmdetect.markets.newstime_burst import summary_section

        sec = summary_section(snapshot_dir)
        if sec:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write("\n" + sec)
    return {"scores": df, "summary": summary, "d_threshold": d_thr, "n_figs": len(figs),
            "n_scored_D": n_scored_D, "timings": dict(tm), "peak_mb": peak_memory_mb()}  # fmt: skip


# --------------------------------------------------------------------------- summary


def _rank_str(r: dict, col: str, n_by: dict) -> str:
    v = r.get(col)
    if v is None:
        return "NaN"
    return f"{v}/{n_by[col]}"


def within_event_coverage(snapshot_dir: Path) -> dict:
    """Events of independent markets vs events with fetched siblings (event_markets.parquet)."""
    out = {"events_independent": None, "events_fetched": None}
    try:
        from pmdetect.markets.within_event import EVENT_MARKETS, independent_markets

        ind = independent_markets(snapshot_dir)
        out["events_independent"] = int(ind["event_id"].drop_nulls().n_unique())
        p = snapshot_dir / EVENT_MARKETS
        if p.exists():
            out["events_fetched"] = int(pl.read_parquet(p, columns=["event_id"])["event_id"].n_unique())
    except Exception:  # noqa: BLE001 - coverage line is informative only
        pass
    return out


def within_event_status(df: pl.DataFrame, we: pl.DataFrame | None, snapshot_dir: Path) -> str:
    """One honest sentence: absent / partial (pilot-only) / complete."""
    if we is None or "within_event" not in df.columns:
        return ("within-event component ABSENT (no data/snapshots/within_event.parquet): the sibling "
                "fetch `scripts/event_fetch_full.py` has not been run, so the combined score uses "
                "level D, late-window D, max_jump and flow only.")  # fmt: skip
    cov = within_event_coverage(snapshot_dir)
    pool = df.filter(~pl.col("out_of_universe"))
    n_we = pool.filter(pl.col("within_event").is_not_null()).height
    n_ind = pool.filter(pl.col("independent")).height
    ev_i, ev_f = cov["events_independent"], cov["events_fetched"]
    partial = ev_i is not None and ev_f is not None and ev_f < ev_i
    txt = (f"within-event component present for {n_we} of {n_ind} independent markets; sibling data "
           f"fetched for {ev_f} of {ev_i} independent-market events")
    if partial:
        txt += (" (PARTIAL: only the events fetched so far, i.e. the 108-event pilot unless "
                "`scripts/event_fetch_full.py` has been run; markets outside them have no within-event "
                "score and may use fewer components)")
    return txt + "."


def within_event_section(
    df: pl.DataFrame, we: pl.DataFrame | None, snapshot_dir: Path | None = None,
    results_dir: Path | None = None,
) -> list[str]:
    snapshot_dir = snapshot_dir or (PROJECT_ROOT / "data" / "snapshots")
    results_dir = results_dir or (PROJECT_ROOT / "results")
    L = ["## Within-event pre-reveal test and placebo", ""]
    L += [within_event_status(df, we, snapshot_dir), ""]
    if we is None or "within_event" not in df.columns:
        return L
    pool = df.filter(~pl.col("out_of_universe"))
    n_we_all = pool.filter(pl.col("within_event").is_not_null()).height
    w = pool.filter(pl.col("we_diff_24h").is_not_null() | pl.col("we_winner_pct_24h").is_not_null())
    grp = w["we_group"].fill_null("none").value_counts().sort("we_group")
    L.append(f"Markets with a within-event statistic: {w.height} ("
             + ", ".join(f"{g}: {n}" for g, n in grp.iter_rows()) + ").")
    plc_path = snapshot_dir / "within_event_placebo.json"
    if plc_path.exists():
        import json

        plc = json.loads(plc_path.read_text(encoding="utf-8"))
        L.append(
            "Pooled placebo over all multi-outcome events with a comparable-loser set "
            "(`within_event.run()`; winner drawn at random within the event for the null):"
        )
        for hz in ("24h", "12h"):
            r = plc.get(hz) or {}
            L.append(f"- Final {hz}: " + "; ".join(f"{k} = {v:.4g}" if isinstance(v, (int, float)) else f"{k} = {v}" for k, v in r.items() if v is not None))
    else:
        L.append("Placebo not available: run `within_event.run()` to write `within_event_placebo.json`.")
    for f in sorted(results_dir.glob("within_event*.md")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if "placebo" in ln.lower() or "permutation" in ln.lower():
                L.append(f"- Builder A report (`{f.name}`): {ln.strip()[:400]}")
    L.append("")
    top = (pool.filter(pl.col("D").is_not_null()).sort("D", descending=True).head(12))
    t_rows = [{"market": (r["question"] or "")[:55], "info_type": r["info_type"], "D": _fmt(r["D"], 4),
               "rank_D": r["rank_D"], "group": r["we_group"] or "none",
               "diff_24h": _fmt(r["we_diff_24h"], 2), "pct_24h": _fmt(r["we_winner_pct_24h"], 2),
               "within": _fmt(r["within_event"], 2), "rank_within": r["rank_within_event"] or "NaN"}
              for r in top.iter_rows(named=True)]  # fmt: skip
    L.append("Highest level-D markets and what the within-event test says about them (a market whose "
             "level D is high but whose winner is not an outlier among its event's outcomes is "
             "explained by event structure; NaN = no sibling outcomes, not a scheduled reveal or no "
             "price data):")
    L.append(_md_table(t_rows, ["market", "info_type", "D", "rank_D", "group", "diff_24h", "pct_24h",
                                "within", "rank_within"]))
    bo = pool.filter(pl.col("question").str.contains("(?i)box office|gross more than")
                     & pl.col("D").is_not_null())
    nb = bo.filter(pl.col("within_event").is_not_null())
    hi = bo.filter(pl.col("rank_D") <= 10)
    L.append(
        f"Box-office markets with level D: {bo.height}, of which {nb.height} have a within-event reading "
        f"(the rest have no usable sibling outcome path). Among the {hi.height} in the level-D top 10, "
        + ", ".join(
            f"{(r['question'] or '')[:30]} (D rank {r['rank_D']}, within-event "
            f"{'NaN' if r['within_event'] is None else format(r['within_event'], '.2f')}, rank "
            f"{r['rank_within_event'] or 'NaN'}/{n_we_all})"
            for r in hi.iter_rows(named=True)
        ) + ". A within-event percentile near 1 would mean the winner stood out among its siblings; "
        "values around 0.5 mean it did not, i.e. the level-D flag is not confirmed by the within-event test."
    )
    L.append("")
    return L


def _flag_list(g: pl.DataFrame, col: str) -> list[dict]:
    return [{"question": (r["question"] or "")[:75], "info_type": r["info_type"],
             "D_timesplit": _fmt(r["D_timesplit"], 4), "D_crossfit": _fmt(r["D_crossfit"], 4)}
            for r in g.sort(col, descending=True, nulls_last=True).iter_rows(named=True)]  # fmt: skip


def baseline_comparison_section(df: pl.DataFrame, plan: pl.DataFrame | None, scheme: str) -> list[str]:
    """Time split vs cross-fit: coverage, rank correlation, top-1% thresholds, flag changes, labelled
    cases (held out: the scheme was fixed before any labelled-case rank was looked at)."""
    L = ["## Baseline scheme: time split vs cross-fit", ""]
    if "D_timesplit" not in df.columns or "D_crossfit" not in df.columns:
        return L + ["comparison columns not available.", ""]
    pool = df.filter(~pl.col("out_of_universe"))
    ind = pool.filter(pl.col("independent"))
    ts = pool.filter(pl.col("D_timesplit").is_not_null())
    cf = pool.filter(pl.col("D_crossfit").is_not_null())
    both = pool.filter(pl.col("D_timesplit").is_not_null() & pl.col("D_crossfit").is_not_null())
    L.append(
        "Time split (default): expanding calendar-month windows over the independent news time; window W "
        "is scored by quantile models fitted on all independent markets whose news time is strictly "
        "before W starts, so no market and no later information sets its own baseline. A window needs "
        ">= 100 earlier training markets and >= 500 thinned training rows, otherwise its markets are "
        "unscorable (reported, not scored in-sample). Windows, minimums and the model spec were fixed "
        "before any labelled-case rank was inspected. Cross-fit (5 folds by market, a market's model "
        "may be fitted on LATER markets) is kept for comparison as `D_crossfit`."
    )
    L.append("")
    n_unsc = ind.filter(pl.col("D_timesplit").is_null() & pl.col("D_crossfit").is_not_null()).height
    L.append(f"Independent markets: {ind.height}. With a time-split D: {ts.height}. With a cross-fit D: "
             f"{cf.height}. Cross-fit D but no time-split D (unscorable early window): {n_unsc}.")
    if plan is not None and plan.height:
        w = (plan.join(pool.select("condition_id"), on="condition_id", how="inner").group_by("window")
             .agg(pl.len().alias("n_scored_markets"), pl.col("n_train_markets").first(),
                  pl.col("n_train_rows").first(), pl.col("scorable").first()).sort("window"))
        L += ["", _md_table([{"window": str(r["window"])[:7], "markets": r["n_scored_markets"],
                              "train_markets": r["n_train_markets"], "train_rows": r["n_train_rows"],
                              "scorable": "yes" if r["scorable"] else "NO (unscorable)"}
                             for r in w.iter_rows(named=True)],
                            ["window", "markets", "train_markets", "train_rows", "scorable"])]  # fmt: skip
    if both.height < 3:
        return L + ["Too few markets with both scores for a comparison.", ""]
    rho = both.select(pl.corr("D_timesplit", "D_crossfit", method="spearman")).item()
    L.append(f"Spearman rank correlation of D_timesplit vs D_crossfit over the {both.height} markets "
             f"with both: {rho:.3f}.")
    thr_ts = float(np.quantile(ts["D_timesplit"].to_numpy(), 0.99))
    thr_cf = float(np.quantile(cf["D_crossfit"].to_numpy(), 0.99))
    L.append(f"Top-1% D thresholds (99th percentile of the scheme's own pool): time split {thr_ts:.4f} "
             f"(n={ts.height}), cross-fit {thr_cf:.4f} (n={cf.height}).")
    f_ts = ts.filter(pl.col("D_timesplit") >= thr_ts)
    f_cf = cf.filter(pl.col("D_crossfit") >= thr_cf)
    ids_ts, ids_cf = set(f_ts["condition_id"]), set(f_cf["condition_id"])
    L.append(f"Markets at or above the own top-1% D threshold: time split {len(ids_ts)}, cross-fit "
             f"{len(ids_cf)}, in both {len(ids_ts & ids_cf)}.")
    drop = pool.filter(pl.col("condition_id").is_in(list(ids_cf - ids_ts)))
    ent = pool.filter(pl.col("condition_id").is_in(list(ids_ts - ids_cf)))
    cols = ["question", "info_type", "D_timesplit", "D_crossfit"]
    L += ["", f"Drop out of the level-D top 1% under the time split ({drop.height}); D_timesplit is NaN "
          "where the market is unscorable:", "", _md_table(_flag_list(drop, "D_crossfit"), cols),
          f"Enter the level-D top 1% under the time split ({ent.height}):", "",
          _md_table(_flag_list(ent, "D_timesplit"), cols)]  # fmt: skip
    L += ["Held-out check on the labelled cases (rank 1 = highest D; the scheme was fixed before these "
          "ranks were looked at; extras are ranked against the universe pool):", ""]
    lab = df.filter(pl.col("case_label").is_not_null() & pl.col("case_label").str.contains("insider_suspected"))
    n_ts, n_cf = ts.height, cf.height
    L.append(_md_table([
        {"case": r["case_name"], "market": (r["question"] or "")[:55],
         "type": "extra" if r["out_of_universe"] else "universe", "indep": r["independent"],
         "D_timesplit": _fmt(r["D_timesplit"], 4),
         "rank_timesplit": f"{r['rank_D_timesplit']}/{n_ts}" if r["rank_D_timesplit"] else "NaN",
         "D_crossfit": _fmt(r["D_crossfit"], 4),
         "rank_crossfit": f"{r['rank_D_crossfit']}/{n_cf}" if r["rank_D_crossfit"] else "NaN",
         "reason": r["reason"] or ""}
        for r in lab.sort("case_name").iter_rows(named=True)],
        ["case", "market", "type", "indep", "D_timesplit", "rank_timesplit", "D_crossfit",
         "rank_crossfit", "reason"]))  # fmt: skip
    return L


def initiator_section(init: pl.DataFrame | None) -> list[str]:
    L = ["## Jump-initiating trades (`data/private/jump_initiators.csv`, not in results/ because it holds wallet addresses)", ""]
    if init is None or init.height == 0:
        return L + ["not computed.", ""]
    mk = init.filter(pl.col("wallet").is_not_null())
    L.append(f"Top winner-side taker wallets inside the flow window (24 h before the jump start to the "
             f"half-jump point, at most 2 h in) for {mk['condition_id'].n_unique()} markets with a jump; "
             "labelled cases are listed with up to 5 wallets, others 3. `first_seen_src` shows coverage "
             "(wallets.parquet, else extra_wallets.parquet, else none); `only_mkt` = the wallet traded "
             "only this market in the taker store (a subset of Polymarket, so a lower bound on breadth).")
    r1 = mk.filter(pl.col("rank") == 1)
    cov = r1.filter(pl.col("first_seen").is_not_null()).height
    only = r1.filter(pl.col("only_market_store").fill_null(False)).height
    L.append(f"Rank-1 initiator has a known first_seen for {cov} of {r1.height} markets; {only} of them "
             "traded only that market in the store.")
    L.append("Per-wallet rows (addresses, USD, first_seen) are in `data/private/jump_initiators.csv`; "
             "only aggregates are kept in this summary.")
    return L


def write_summary(df, lt, d_thr, hist, figs, path: Path, wallets, taker, series, baseline, inp,
                  init=None, we=None, perf=None, plan=None, scheme="timesplit") -> Path:
    pool = df.filter(~pl.col("out_of_universe"))
    ext = df.filter(pl.col("out_of_universe"))
    n_pool = pool.height
    n_ind = pool.filter(pl.col("independent")).height
    n_D = pool.filter(pl.col("D").is_not_null()).height
    n_late = pool.filter(pl.col("D_last24").is_not_null()).height
    n_mj = pool.filter(pl.col("max_jump").is_not_null()).height
    n_jump = pool.filter(pl.col("jump_start_ts").is_not_null()).height
    n_flow = pool.filter(pl.col("flow_score").is_not_null()).height
    n_comb = pool.filter(pl.col("fisher_score").is_not_null()).height
    n_flow_old = pool.filter(pl.col("flow_score_old").is_not_null()).height
    n_we = pool.filter(pl.col("within_event").is_not_null()).height
    k_flag = pool.filter(pl.col("flag_fisher")).height
    n_by = {
        "rank_D": n_D, "rank_D_last24": n_late, "rank_max_jump": n_mj, "rank_flow_score": n_flow,
        "rank_fisher_score": n_comb,
        "rank_flow_score_old": n_flow_old, "rank_within_event": n_we,
    }
    L: list[str] = ["# Phase 2c summary: market-level detector (independent news times, within-event test)", ""]
    L += [f"Markets scored (full universe): {n_pool}; extra out-of-universe cases: {ext.height}. "
          f"Markets with an independent news time: {n_ind} "
          f"({', '.join(f'{m} {n}' for m, n in pool.filter(pl.col('independent'))['indep_method'].value_counts().sort('indep_method').iter_rows())}). "
          f"Level D: {n_D}; D_last24: {n_late}; max_jump: {n_mj}; resolving jump detected: {n_jump}; "
          f"flow-before-move score (new window): {n_flow} (old window: {n_flow_old}); within-event score: {n_we}; combined score (>= {MIN_COMPONENTS} components): {n_comb}.", "",
          "All percentiles, ranks and flag cut-offs below use the universe pool only; extra cases get "
          "the percentile / rank / flag they would have had against that pool.", ""]  # fmt: skip

    # D distribution
    snap = inp.get("snapshot_dir") or (PROJECT_ROOT / "data" / "snapshots")
    L += ["Within-event status: " + within_event_status(df, we, snap), ""]
    if perf:
        L += ["## Run performance (this run)", "",
              "Wall time per stage (s): " + ", ".join(f"{k} {v:.0f}" for k, v in perf["timings"].items())
              + f". Peak working set: {perf['peak_mb'] / 1024:.1f} GB. Flagged markets: "
              f"{perf['n_flagged']}; figures are drawn for at most {perf['max_flag_plots']} of them "
              "plus the labelled cases (never one per market).", ""]  # fmt: skip
    L += ["## Level D (independent-news-time markets only)", "",
          f"Baseline scheme for every score below: **{scheme}**."
          + (" Independent markets in early windows without enough earlier training markets are "
             "UNSCORABLE (D null, reason `no_prior_training_window`), see the next section."
             if scheme == "timesplit" else ""), ""]  # fmt: skip
    d = pool.filter(pl.col("D").is_not_null())["D"].to_numpy()
    if d.size:
        qs = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        L += ["| quantile | D |", "|---|---|"]
        L += [f"| {q:.2f} | {np.quantile(d, q):.4f} |" for q in qs]
        L += ["", f"Share with D == 0: {(d == 0).mean():.1%}. New top-1% threshold (99th percentile "
              f"of D over {d.size} independent markets): {d_thr:.4f}; markets at or above it: "
              f"{int((d >= d_thr).sum())}.", ""]  # fmt: skip
        by = (pool.filter(pl.col("D").is_not_null()).group_by("info_type")
              .agg(pl.len().alias("n"), pl.col("D").median().alias("median_D"),
                   pl.col("D").max().alias("max_D")).sort("info_type"))  # fmt: skip
        L.append(_md_table([{**r, "median_D": _fmt(r["median_D"], 4), "max_D": _fmt(r["max_D"], 4)}
                            for r in by.iter_rows(named=True)], ["info_type", "n", "median_D", "max_D"]))
        L.append(f"Histogram: `{hist.relative_to(path.parent).as_posix()}`\n")
    for c, nm in (("D_last24", "late-window D (last 24 h)"), ("max_jump", "max_jump")):
        v = pool.filter(pl.col(c).is_not_null())[c].to_numpy()
        if v.size:
            L.append(f"- {nm}: n={v.size}, median {np.median(v):.4f}, p90 {np.quantile(v, 0.9):.4f}, "
                     f"p99 {np.quantile(v, 0.99):.4f}, max {v.max():.4f}.")
    L.append("")

    L += baseline_comparison_section(df, plan, scheme)

    # jumps and flow
    L += ["## Resolving-jump detection and flow before the move", ""]
    jr = pool["jump_reason"].fill_null("clear jump").value_counts().sort("count", descending=True)
    L.append("Jump detection over universe markets: " + ", ".join(f"{r} ({n})" for r, n in jr.iter_rows()) + ".")
    jj = pool.filter(pl.col("jump_start_ts").is_not_null())
    if jj.height:
        lead = ((jj["news_ts_best"] - jj["jump_start_ts"]).dt.total_seconds() / 3600.0).drop_nulls()
        L.append(f"Jump size: median {jj['jump_size'].median():.3f}; hours from jump start to the "
                 f"market's best news time: median {lead.median():.1f}, p10 {lead.quantile(0.1):.1f}, "
                 f"p90 {lead.quantile(0.9):.1f} (independent news times only: "
                 f"{_fmt(((jj.filter(pl.col('independent'))['news_ts_best'] - jj.filter(pl.col('independent'))['jump_start_ts']).dt.total_seconds() / 3600.0).median(), 1)}).")  # fmt: skip
    fr = pool.filter(pl.col("jump_start_ts").is_not_null())["fb_reason"].fill_null("computed").value_counts().sort("count", descending=True)
    L.append("Flow-before-move among markets with a detected jump: " + ", ".join(f"{r} ({n})" for r, n in fr.iter_rows()) + ".")
    wc = pool.filter(pl.col("fb_fresh_cov").is_not_null())
    L.append(f"fresh-wallet share available (>= 50% winner-side USD with known first_seen) for "
             f"{pool.filter(pl.col('fb_fresh_z').is_not_null()).height} of {wc.height} markets with a window.")
    wl = pool.filter(pl.col("fb_win_end").is_not_null())
    if wl.height:
        hrs = (wl["fb_win_end"] - wl["jump_start_ts"]).dt.total_seconds() / 3600.0
        L.append(f"Phase 2c window: [jump_start - 24 h, min(half-jump point, jump_start + 2 h)]; the part "
                 f"after the jump start lasts median {hrs.median():.2f} h (p90 {hrs.quantile(0.9):.2f} h) "
                 f"over {wl.height} markets, so it contains the trades that caused the first half of the "
                 "jump (the 2b window ended at jump_start and missed them).")
    L += ["", "The earlier history is unchanged in kind (24 h blocks before the window start). Flow and "
          "volume measured here therefore include the jump-initiating trades; no threshold was tuned "
          "on a labelled case.", ""]  # fmt: skip

    # flagged
    L += ["## Flagged markets (top 1% of the Fisher combination, universe pool)", "",
          f"Rule: ceil(1% of {n_comb} scored markets) = {k_flag} highest `fisher_score` (ties "
          f"included); a market is also listed if it is in the rank-mean top 1%. IsolationForest is a "
          "cross-check only.", ""]  # fmt: skip
    fl = pool.filter(pl.col("flag_fisher") | pl.col("flag_rank_mean")).sort("fisher_score", descending=True)
    rows = [
        {"rank": r["rank_fisher_score"], "question": (r["question"] or "")[:70],
         "info_type": r["info_type"], "indep": r["independent"], "D": _fmt(r["D"], 4),
         "D_last24": _fmt(r["D_last24"], 4), "max_jump": _fmt(r["max_jump"], 3),
         "flow": _fmt(r["flow_score"], 2), "within": _fmt(r["within_event"], 2),
         "we_diff24": _fmt(r.get("we_diff_24h"), 2),
         "jump_start": str(r["jump_start_ts"] or "none")[:16], "fisher_p": f"{r['fisher_p']:.1e}", "iso": r["flag_iso"], "driver": r["main_driver"]}
        for r in fl.iter_rows(named=True)
    ]  # fmt: skip
    L.append(_md_table(rows, ["rank", "question", "info_type", "indep", "D", "D_last24", "max_jump",
                              "flow", "within", "we_diff24", "jump_start", "fisher_p", "iso",
                              "driver"]))

    # labelled cases
    L += ["## Labelled cases: rank on every score", ""]
    if lt.height == 0:
        L.append("No labelled case market is in the scoring table.\n")
    else:
        L.append(f"Labelled markets in the table: {lt.height}. Rank 1 = most suspicious; denominators "
                 "are the number of universe markets with that score. Extra (out-of-universe) cases are "
                 "ranked against the universe pool, never inside it. NaN = measure not available "
                 "(see the reason columns in market_scores.csv).\n")  # fmt: skip
        # every insider_suspected case in full; context/candidate cases (hundreds of sibling
        # markets: the full table is in market_scores.csv) capped to the 10 best-ranked per case
        sus = lt.filter(pl.col("case_label").str.contains("insider_suspected"))
        oth = (
            lt.filter(~pl.col("case_label").str.contains("insider_suspected"))
            .with_columns(
                pl.col("rank_fisher_score").fill_null(10**9).alias("_k1"),
                pl.col("rank_max_jump").fill_null(10**9).alias("_k2"),
            )
            .sort("case_name", "case_label", "_k1", "_k2")
            .group_by(["case_name", "case_label"], maintain_order=True)
            .head(10)
            .drop("_k1", "_k2")
        )
        if oth.height < lt.height - sus.height:
            L.append(f"Context / candidate cases are capped at 10 markets per case here "
                     f"({lt.height - sus.height - oth.height} more are in `market_scores.csv`).\n")
        rows = []
        for r in pl.concat([sus, oth.select(sus.columns)]).iter_rows(named=True):
            rows.append({
                "case": r["case_name"], "label": r["case_label"],
                "market": (r["question"] or "")[:52], "type": "extra" if r["out_of_universe"] else "universe",
                "indep": r["independent"],
                "rank_D": _rank_str(r, "rank_D", n_by), "rank_D24": _rank_str(r, "rank_D_last24", n_by),
                "rank_maxjump": _rank_str(r, "rank_max_jump", n_by),
                "rank_flow_old": _rank_str(r, "rank_flow_score_old", n_by),
                "rank_flow_new": _rank_str(r, "rank_flow_score", n_by),
                "rank_within": _rank_str(r, "rank_within_event", n_by),
                "rank_combined": _rank_str(r, "rank_fisher_score", n_by),
                "jump": "yes" if r["jump_start_ts"] is not None else (r["jump_reason"] or ""),
            })
        L.append(_md_table(rows, ["case", "label", "market", "type", "indep", "rank_D", "rank_D24",
                                  "rank_maxjump", "rank_flow_old", "rank_flow_new", "rank_within",
                                  "rank_combined", "jump"]))

    # Machado
    L += ["### Nobel Peace Prize 2025 (Machado)", ""]
    nb = lt.filter(pl.col("case_name").str.contains("nobel"))
    if nb.height == 0:
        L.append("The Machado market is not in the scoring table.\n")
    for r in nb.iter_rows(named=True):
        cid = r["condition_id"]
        fig = next((p for t, c, p in figs if c == cid), None)
        j = baseline.join(series.filter(pl.col("condition_id") == cid).select("condition_id", "t", "q"),
                          on=["condition_id", "t"]).filter(pl.col("condition_id") == cid)
        above = float((j["q"] > j["m90"]).mean()) if j.height else float("nan")
        last = j.filter(pl.col("tau_h") <= 24)
        above24 = float((last["q"] > last["m90"]).mean()) if last.height else float("nan")
        mx = float((j["q"] - j["m90"]).max()) if j.height else float("nan")
        L.append(f"- News time {r['news_ts_best']} ({r['indep_method']}), independent={r['independent']}. "
                 f"Level D {_fmt(r['D'], 4)} (rank {r['rank_D']}/{n_D}), D_last24 {_fmt(r['D_last24'], 4)} "
                 f"(rank {r['rank_D_last24']}/{n_late}), max_jump {_fmt(r['max_jump'], 3)} at "
                 f"{_fmt(r['max_jump_hours_before'], 0)} h before news (rank {r['rank_max_jump']}/{n_mj}), "
                 f"flow_score {_fmt(r['flow_score'], 2)} (new window rank {r['rank_flow_score']}/{n_flow}; old window "
                 f"rank {r['rank_flow_score_old']}/{n_flow_old}), within-event {_fmt(r['within_event'], 2)} "
                 f"(rank {r['rank_within_event']}/{n_we}), combined "
                 f"rank {r['rank_fisher_score']}/{n_comb}, main driver {r['main_driver']}.")  # fmt: skip
        L.append(f"- Peer band: share of the 168 hourly points with q above the peer m90: {above:.1%} "
                 f"(last 24 h: {above24:.1%}); largest q - m90 = {mx:.3f}. Detected jump: start "
                 f"{r['jump_start_ts']}, size {_fmt(r['jump_size'], 3)}, pre-level {_fmt(r['pre_level'], 3)}.")  # fmt: skip
        L.append(f"- Chart: `{fig.relative_to(path.parent).as_posix() if fig else 'not drawn'}`; the "
                 "visual verdict (does q(t) clear the band?) must be read from the image together "
                 "with these shares.\n")  # fmt: skip

    # extras
    L += ["## Extra labelled cases (out of universe)", ""]
    em = inp["extra_meta"]
    if em is None:
        L.append("No extra cases fetched (run `python -m pmdetect.markets.cases_extra`).\n")
    else:
        rows = []
        for r in em.iter_rows(named=True):
            x = df.filter(pl.col("condition_id") == r["condition_id"]).row(0, named=True)
            rows.append({
                "case": r["case_name"], "mapping": r["mapping"], "market": r["question"][:55],
                "taker_rows": r["tape_rows"], "tape_start": str(r["tape_start"])[:16],
                "news_ts": str(x["news_ts_best"])[:16], "news_source": ("independent: " if r["independent"] else "") + r["news_source"][:70],
                "jump_start": str(x["jump_start_ts"] or x["jump_reason"] or "")[:19],
                "D": _fmt(x["D"], 3), "D_last24": _fmt(x["D_last24"], 3), "max_jump": _fmt(x["max_jump"], 3),
                "flow": _fmt(x["flow_score"], 2), "flow_old": _fmt(x["flow_score_old"], 2),
                "within": _fmt(x["within_event"], 2), "combined": x["rank_fisher_score"] or "NaN",
            })
        L.append(_md_table(rows, ["case", "mapping", "market", "taker_rows", "tape_start", "news_ts",
                                  "news_source", "jump_start", "D", "D_last24", "max_jump", "flow",
                                  "flow_old", "within", "combined"]))
        L.append("`candidate` mapping: the press names accounts but not the market, so every candidate is "
                 "scored and none is singled out. Tapes are capped to the 12 days before resolution "
                 "(`/v2/trades` has no time filter for the market feed). Short-lived markets (created "
                 "< 36 h before resolution) cannot get a resolving-jump or flow-before-move reading.\n")  # fmt: skip

    L += within_event_section(df, we, snap, path.parent)
    L += initiator_section(init)

    # agreement
    ag = rank_agreement(df.filter(~pl.col("out_of_universe")))
    L += ["## Agreement between rankings", "",
          f"Over {ag['n']} universe markets with all three scores. Spearman:", ""]  # fmt: skip
    for k, v in ag["spearman"].items():
        L.append(f"- {k}: {_fmt(v)}")
    L += ["", f"Flag counts: {ag['flag_counts']}; overlap: {ag['flag_overlap']}.", ""]

    # coverage
    L += ["## Coverage and missing data", ""]
    reasons = pool.filter(pl.col("reason").is_not_null())["reason"].value_counts().sort("count", descending=True)
    L.append("Level D is NaN for " + (", ".join(f"{r} ({n})" for r, n in reasons.iter_rows()) or "no market") + ".")
    for c in [*COMPONENTS, *FLOW_COLS]:
        if c in pool.columns:
            L.append(f"- {c}: available for {pool.filter(pl.col(c).is_not_null() & ~pl.col(c).cast(pl.Float64).is_nan()).height} of {n_pool}")  # fmt: skip
    nc = pool["n_components"].value_counts().sort("n_components")
    L.append("Components available per market: " + ", ".join(f"{k}: {n}" for k, n in nc.iter_rows()) + ".")
    wf = wallets.filter(pl.col("first_seen").is_not_null()).height
    L.append(f"Wallet first_seen: {wf} wallet rows with first_seen (backfill may be incomplete).")
    L += [""]

    L += ["## Caveats", "",
          f"- Only {n_ind} of {n_pool} markets have an independent news time ({n_D} with level D); "
          f"every other market is scored on max_jump / flow only. A 1% flag is {k_flag} market(s).",
          "- Independent times come from the override CSV and the `scheduled_end` rule (hand-checked "
          "sample in the README); end_ts is often date-level.",
          "- Markets with a price-based news time only have max_jump (cut off by construction before the "
          "0.9 crossing) and, if a clear jump with enough history exists, flow-before-move; most get no "
          "combined score.",
          "- Jump detection needs >= 36 h of series and >= 12 h before the jump; many markets are "
          "short-lived or start at the jump, so only a minority have a flow reading. Thresholds are round "
          "numbers, not tuned on any labelled case.",
          "- Flow z-scores use 24 h blocks of the market's own earlier history (needs >= 3); fresh share "
          "needs >= 50% wallet coverage. Extra markets use first_seen fetched for the top winner-side "
          "wallets only (extra_wallets.parquet).",
          "- Signals use taker rows only. Fisher p-values are rank-based pseudo p-values, signals are "
          "correlated. The level baseline uses a time split (see the baseline-scheme section): early independent markets "
          "are unscorable by design, and each scored market sees only earlier peers, so early windows "
          "have a smaller and more non-stationary training set than the cross-fit's.",
          f"- Figures: {len(figs)} charts in `figures/` (flagged_*, case_*, jump_start_ts dotted).", ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return path
