"""`pmdetect score-wallets` at full-universe scale (no network, Parquet only).

Inputs (data/snapshots, never the DuckDB write lock): ``trades_taker.parquet`` (+ ``extra_trades_taker``),
``markets``, ``wallets`` (+ ``extra_wallets``), ``news_independent``, ``trades.parquet`` (all rows, for
the maker rule), ``within_event``; ``results/market_scores.csv`` (jump starts + context percentiles) or
``data/snapshots/jumps.parquet`` (condition_id, jump_start_ts) when present; the first_seen top-up file
``data/private/first_seen_topup.parquet``.

Scale design: the taker tape is only ever scanned lazily with the columns a stage needs (positions,
two-sidedness and the maker rule stream; clusters hold only the BUY rows as integer codes); the
Poisson-binomial tail is a bucketed vectorised DP; co-trading pairs come from one composite-key
``searchsorted`` with a per-anchor cap. Every stage's wall time and the process peak working set are
recorded in the returned dict and the aggregate-only summary.

Positions are rebuilt automatically when ``positions.parquet`` is older than any of its inputs, so a
re-run after the market pipeline rewrites ``news_independent`` / ``market_scores`` picks the new cutoffs up.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path

import polars as pl

from pmdetect.store.db import PROJECT_ROOT
from pmdetect.wallets import filters

SNAP = PROJECT_ROOT / "data" / "snapshots"
RESULTS = PROJECT_ROOT / "results"
PRIVATE = PROJECT_ROOT / "data" / "private"
TAKER_COLS = ["trade_id", "wallet", "condition_id", "token_id", "outcome_label", "side", "size", "usd", "ts"]
CTX_COLS = ("within_event_pct", "flow_score_pct", "D_last24_pct")


def peak_mb() -> float:
    """Peak working set of this process in MB (Windows; 0.0 elsewhere)."""
    try:

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]  # fmt: skip

        k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**20
    except Exception:  # noqa: BLE001 - not on Windows
        return 0.0


class Timer:
    def __init__(self, log: Callable[[str], None]):
        self.log, self.rows, self.t0 = log, [], time.time()

    @contextmanager
    def stage(self, name: str):
        t = time.time()
        yield
        dt = time.time() - t
        self.rows.append({"stage": name, "seconds": round(dt, 1), "peak_mb": round(peak_mb())})
        self.log(f"[{time.time() - self.t0:7.0f}s] {name}: {dt:.1f}s (peak RSS {peak_mb():.0f} MB)")


# --------------------------------------------------------------------------------------------- inputs


def _rd(snap: Path, name: str) -> pl.DataFrame | None:
    f = snap / name
    return pl.read_parquet(f) if f.exists() else None


def load_taker_lazy(snap: Path = SNAP) -> pl.LazyFrame:
    """Full taker tape (+ extra labelled markets) as a LazyFrame; extra rows already in the main tape
    (same trade_id) are dropped, so nothing is counted twice."""
    main_f, extra_f = snap / "trades_taker.parquet", snap / "extra_trades_taker.parquet"
    main = pl.scan_parquet(main_f).select(TAKER_COLS)
    if not extra_f.exists():
        return main
    extra = pl.read_parquet(extra_f).select(TAKER_COLS)
    overlap = extra["condition_id"].unique()
    seen = (
        main.filter(pl.col("condition_id").is_in(overlap.implode())).select("trade_id").collect()["trade_id"]
    )
    extra = extra.filter(~pl.col("trade_id").is_in(seen.implode()))
    return pl.concat([main, extra.lazy()])


def load_wallets(snap: Path = SNAP, private: Path = PRIVATE) -> pl.DataFrame:
    from pmdetect.wallets import oneshot

    parts = [
        d.select("wallet", "first_seen", pl.col("n_markets").cast(pl.Int64))
        for d in (_rd(snap, "wallets.parquet"), _rd(snap, "extra_wallets.parquet"))
        if d is not None
    ]
    wallets = pl.concat(parts).unique("wallet", keep="first")
    tp = private / "first_seen_topup.parquet"
    return oneshot.merge_first_seen(wallets, pl.read_parquet(tp) if tp.exists() else None)


def load_market_scores(results: Path = RESULTS) -> pl.DataFrame | None:
    """Only the columns the wallet side uses (condition_id, jump_start_ts, context percentiles)."""
    f = results / "market_scores.csv"
    if not f.exists():
        return None
    head = pl.read_csv(f, n_rows=0).columns
    cols = [c for c in ("condition_id", "jump_start_ts", *CTX_COLS) if c in head]
    ov = {"jump_start_ts": pl.String} if "jump_start_ts" in cols else {}
    return pl.read_csv(f, columns=cols, infer_schema_length=5000, schema_overrides=ov)


def load_cutoff_inputs(snap: Path = SNAP, results: Path = RESULTS):
    """(markets, news_indep, jumps, market_scores, within_event, qmarkets) for positions and scoring.

    Cutoff rule (phase 3b): independent news time (``news_independent.parquet`` plus out-of-universe
    labelled markets that carry a documented independent time), else the changepoint jump start
    (``jumps.parquet`` or ``market_scores.csv``), else no valid cutoff."""
    markets_all = pl.read_parquet(snap / "markets.parquet")
    extra = _rd(snap, "extra_markets.parquet")
    mk = markets_all.select("condition_id", "event_id", "question", "outcome", "info_type", "news_ts")
    ni = _rd(snap, "news_independent.parquet")
    ni = ni.select("condition_id", "news_ts_indep", "independent") if ni is not None else None
    if extra is not None:
        ex = extra.select(
            "condition_id", pl.col("condition_id").alias("event_id"), "question", "outcome",
            "info_type", pl.col("news_ts_indep").alias("news_ts"),
        ).filter(~pl.col("condition_id").is_in(mk["condition_id"].implode()))  # fmt: skip
        mk = pl.concat([mk, ex])
        exi = extra.select(
            "condition_id", "news_ts_indep",
            (pl.col("indep_method").is_in(["override", "scheduled_end"]) & pl.col("news_ts_indep").is_not_null())
            .alias("independent"),
        )  # fmt: skip
        ni = exi if ni is None else pl.concat([ni, exi.filter(~pl.col("condition_id").is_in(ni["condition_id"].implode()))])
    ms = load_market_scores(results)
    jumps = None
    jf = snap / "jumps.parquet"
    if jf.exists():
        jumps = pl.read_parquet(jf).select("condition_id", pl.col("jump_start_ts").cast(pl.Datetime("us")))
    elif ms is not None and "jump_start_ts" in ms.columns:
        jumps = ms.select("condition_id", pl.col("jump_start_ts").str.to_datetime(strict=False)).drop_nulls()
    jumps = jumps.drop_nulls() if jumps is not None else None
    we = _rd(snap, "within_event.parquet")
    q = mk.select("condition_id", "event_id", "question")
    return mk, ni, jumps, ms, we, q


def load_news_burst(snap: Path = SNAP) -> pl.DataFrame | None:
    """data/snapshots/news_burst.parquet (None if absent). ``positions.usable_burst`` keeps only the
    rows with ``burst_validated`` and ``independent_burst`` true, so an unvalidated file changes nothing."""
    return _rd(snap, "news_burst.parquet")


def _inputs_mtime(snap: Path, results: Path) -> float:
    fs = [snap / n for n in ("trades_taker.parquet", "extra_trades_taker.parquet", "markets.parquet",
                             "news_independent.parquet", "news_burst.parquet", "extra_markets.parquet",
                             "jumps.parquet")]  # fmt: skip
    fs.append(results / "market_scores.csv")
    return max((f.stat().st_mtime for f in fs if f.exists()), default=0.0)


# ------------------------------------------------------------------------------------------- the run


def run_score_wallets(
    snap: Path = SNAP,
    private: Path = PRIVATE,
    results: Path = RESULTS,
    rebuild_positions: bool | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run positions -> calibration -> FDR -> filters -> one-shot -> clusters -> watchlist -> summary.

    ``rebuild_positions``: True/False force; None (default) rebuilds when missing or older than the inputs."""
    from pmdetect.wallets import calibration, clusters, oneshot
    from pmdetect.wallets import positions as posmod
    from pmdetect.wallets.fdr import apply_fdr, stage_counts
    from pmdetect.watchlist import assert_results_clean, build_watchlist, write_phase3_summary

    tm = Timer(log)
    private.mkdir(parents=True, exist_ok=True)
    with tm.stage("load inputs"):
        taker = load_taker_lazy(snap)
        wallets = load_wallets(snap, private)
        mk, ni, jumps, ms, we, qmarkets = load_cutoff_inputs(snap, results)
        all_f = snap / "trades.parquet"
        all_trades = None
        if all_f.exists():
            all_trades = pl.scan_parquet(all_f)
            want = ("trade_id", "wallet", "condition_id", "token_id", "side", "size")
            all_trades = all_trades.select([c for c in want if c in all_trades.collect_schema().names()])

    pos_path = private / "positions.parquet"
    stale = (not pos_path.exists()) or pos_path.stat().st_mtime < _inputs_mtime(snap, results)
    with tm.stage("positions" + (" (build)" if rebuild_positions or (rebuild_positions is None and stale) else " (load)")):
        if rebuild_positions or (rebuild_positions is None and stale):
            positions = posmod.build_positions(taker.select(*[c for c in TAKER_COLS if c != "trade_id"]),
                                               mk, ni, jumps, write=True, out=pos_path,
                                               news_burst=load_news_burst(snap))  # fmt: skip
        else:
            positions = pl.read_parquet(pos_path)
    log(f"positions {positions.height}: {positions['cutoff_method'].value_counts().sort('cutoff_method').rows()}")

    with tm.stage("calibration (isotonic, cross-fitted)"):
        calibrated, _curve, _obs = calibration.fit_isotonic(positions)
    with tm.stage("wallet tests (Poisson-binomial + bootstrap) + FDR"):
        tests = apply_fdr(calibration.wallet_tests(positions, write=False, calibrated=calibrated))
    with tm.stage("filters (two-sided, maker, contract, broad)"):
        flags = filters.flag_non_directional(taker, positions, wallets, all_trades)
        cov = filters.maker_coverage(all_trades, taker)
    excl = flags.filter(pl.col("excluded"))["wallet"]
    with tm.stage("one-shot scoring"):
        scored = oneshot.score_oneshot(positions, qmarkets, wallets, ms, we)
    with tm.stage("co-trading clusters"):
        cl = clusters.cotrading_clusters(
            taker.select("wallet", "condition_id", "token_id", "side", "ts"),
            min_markets=2, window_minutes=10, exclude_wallets=set(excl.to_list()),
        )  # fmt: skip
    with tm.stage("cluster calibration"):
        ctests = clusters.calibrate_clusters(calibrated, cl) if cl.height else None
    with tm.stage("watchlist"):
        wl_path = build_watchlist(tests, scored, flags, cl, ctests, positions, qmarkets,
                                  out_path=private / "watchlist.csv")  # fmt: skip
        wl = pl.read_csv(wl_path, infer_schema_length=None)
    with tm.stage("write outputs + summary"):
        flags.write_parquet(private / "wallet_flags.parquet")
        scored.write_parquet(private / "oneshot_scores.parquet")
        cl.write_parquet(private / "clusters.parquet")
        calibrated.write_parquet(private / "positions_calibrated.parquet")
        tests.write_parquet(private / "wallet_tests_fdr.parquet")
        if ctests is not None:
            ctests.write_parquet(private / "cluster_tests.parquet")
        n_fdr = int(tests["flag"].sum())
        notes = scale_notes(positions, tests, flags, cov, jumps, ni, scored, tm.rows, stage_counts(tests), cl)
        out = write_phase3_summary(
            results / "phase3_summary.md", n_positions=positions.height, n_wallets_tested=tests.height,
            exclusion=filters.exclusion_counts(flags), rule_doc=filters.RULE_DOC, tests=tests,
            n_fdr_pass=n_fdr, oneshot=scored, watchlist=wl, clusters=cl, cluster_tests=ctests, notes=notes,
        )  # fmt: skip
        filters.ntg_histogram(flags).write_csv(results / "maker_ntg_histogram.csv")
        n_clean = assert_results_clean(results)
    res = {
        "positions": positions.height, "tested": tests.height, "fdr_pass": n_fdr,
        "excluded": excl.len(), "clusters": cl["cluster_id"].n_unique() if cl.height else 0,
        "watchlist": wl.height, "summary": out, "privacy_files": n_clean, "maker_coverage": cov,
        "timings": tm.rows, "total_s": round(time.time() - tm.t0, 1), "peak_mb": round(peak_mb()),
        "positions_df": positions, "tests": tests, "scored": scored, "flags": flags, "clusters_df": cl,
        "cluster_tests": ctests, "watchlist_df": wl,
    }  # fmt: skip
    return res


def scale_notes(positions, tests, flags, cov, jumps, ni, scored, timings, sc, cl=None) -> list[str]:
    """Aggregate-only caveat lines for the summary (no addresses)."""
    cm = positions["cutoff_method"].value_counts().sort("cutoff_method")
    n_ind = 0 if ni is None else int(ni["independent"].fill_null(False).sum())
    lines = [
        "- Scale run: positions by cutoff method: "
        + ", ".join(f"{r['cutoff_method']}={r['count']}" for r in cm.iter_rows(named=True))
        + f"; markets with an independent news time: {n_ind}; markets with a jump start: "
        + f"{0 if jumps is None else jumps.height}.",
        f"- Repeat test and calibration: {sc['wallets_with_resolved_positions']} wallets with >= 1 independent-news "
        f"position, {sc['wallets_ge5_positions']} eligible (>= 5), BH passes {sc['bh_flag_pb']} (exact tail) / "
        f"{sc['bh_flag_boot']} (bootstrap) / {sc['bh_flag_both']} (both).",
        f"- Maker rule coverage: the all-rows (maker+taker) tape covers {cov['covered']} of {cov['taker_markets']} "
        f"taker-tape markets ({cov['share']:.1%}); "
        + (
            "complete."
            if cov["complete"]
            else "INCOMPLETE: maker fills are only visible in covered markets, so the maker exclusion "
            "under-counts and is not final until the all-rows tape is loaded."
        ),
        "- Stage timings (s): " + ", ".join(f"{r['stage']}={r['seconds']}" for r in timings),
    ]
    if "maker_old" in flags.columns:
        ec = filters.exclusion_counts(flags)
        lines.append(
            f"- Maker rule redesign (two-sided, inventory-neutral; see filters.py): activity-only rule flagged "
            f"{ec['maker_old']} wallets, the new rule {ec['maker']} (overlap {ec['maker_both']}); wallets excluded "
            f"overall {ec['excluded_old']} under the old rule vs {ec['excluded']} now; {ec['moved_to_included']} "
            "wallets move from excluded to included."
        )
        h = filters.ntg_histogram(flags)
        lines.append(
            "- Net-to-gross histogram (gross-weighted |net|/gross over maker markets, wallets passing the maker "
            "activity screen; threshold fixed a priori at <= "
            f"{filters.MAKER_MAX_NTG}): "
            + ", ".join(f"{r['ntg_bin']}: {r['wallets']}" for r in h.iter_rows(named=True))
        )
    if cl is not None and cl.height and "link_level" in cl.columns:
        lv = cl.group_by("cluster_id").agg(pl.col("link_level").first()).group_by("link_level").len().sort("link_level")
        lines.append(
            "- Cluster link level (shared markets needed; oversized components are re-linked upward from 2): "
            + ", ".join(f"{r['link_level']}: {r['len']}" for r in lv.iter_rows(named=True))
        )
    if scored.height and "wallet_age_days_at_entry" in scored.columns:
        miss = int(scored["wallet_age_days_at_entry"].is_null().sum())
        lines.append(
            f"- first_seen missing for {miss} of {scored.height} winning positions (store backfill covers "
            "only part of the wallets; the first_seen top-up fills the candidate set)."
        )
    return lines
