"""Phase 3A runner: positions -> data/private/positions.parquet, then calibration/FDR outputs.

Usage: python scripts/run_phase3_a.py [positions|tests|all]
Same inputs as `pmdetect score-wallets` (pmdetect.wallets.pipeline loaders). Reads data/snapshots only (never the DuckDB write lock). Wallet-level outputs go to data/private/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

from pmdetect.wallets import pipeline
from pmdetect.wallets.positions import PRIVATE, build_positions

SNAP = Path(__file__).resolve().parents[1] / "data" / "snapshots"


def main(what: str = "all") -> None:
    t0 = time.time()
    # full-universe loaders shared with `pmdetect score-wallets` (lazy taker scan, no whole-tape load)
    taker = pipeline.load_taker_lazy(SNAP)
    markets, ni, jumps, _ms, _we, _q = pipeline.load_cutoff_inputs(SNAP)
    print(f"inputs {time.time() - t0:.0f}s; markets {markets.height}", flush=True)
    pos = build_positions(
        taker.select(*[c for c in pipeline.TAKER_COLS if c != "trade_id"]), markets, ni, jumps, write=True
    )
    print(
        f"positions {pos.height} wallets {pos['wallet'].n_unique()} -> {PRIVATE / 'positions.parquet'}",
        flush=True,
    )
    print(pos["cutoff_method"].value_counts(), flush=True)
    print(
        f"valid-cutoff positions {int(pos['valid_cutoff'].sum())} wallets "
        f"{pos.filter(pl.col('valid_cutoff'))['wallet'].n_unique()}",
        flush=True,
    )
    if what in ("tests", "all"):
        from pmdetect.wallets.calibration import (
            MIN_TYPE_N,
            decile_table,
            fit_isotonic,
            wallet_tests,
        )
        from pmdetect.wallets.fdr import apply_fdr, stage_counts

        cal, curve, obs = fit_isotonic(pos)
        cal.write_parquet(PRIVATE / "positions_calibrated.parquet")
        tests = wallet_tests(pos, write=True, calibrated=cal)  # B can read this immediately
        res = apply_fdr(tests)
        res.write_parquet(PRIVATE / "wallet_tests.parquet")
        sc = stage_counts(res)
        print(sc, flush=True)
        results = Path(__file__).resolve().parents[1] / "results"
        curve.write_csv(results / "phase3a_calibration_curve.csv")
        obs.write_csv(results / "phase3a_calibration_bins.csv")
        # decile specs: before (old rule, previous snapshot), after_rule_as_specified (fit on all valid-cutoff
        # positions incl. jump_start), after_independent_only (deployed: calib_eligible)
        before_p = PRIVATE / "_before_positions_calibrated.parquet"
        parts = []
        if before_p.exists():
            b = pl.read_parquet(before_p).with_columns(pl.col("won").cast(pl.Boolean))
            parts.append(decile_table(b).with_columns(pl.lit("before").alias("spec")))
        spec_fit, _, _ = fit_isotonic(pos.with_columns(pl.col("valid_cutoff").alias("calib_eligible")))
        for name, frame in (("after_rule_as_specified", spec_fit), ("after_independent_only", cal)):
            t = decile_table(frame.with_columns(pl.col("won").cast(pl.Boolean)))
            parts.append(t.with_columns(pl.lit(name).alias("spec")))
        dec = pl.concat(parts).select(
            "spec", "decile", "n", "price_lo", "price_hi", "mean_entry", "win_rate", "mean_p_tilde"
        )
        dec.write_csv(results / "phase3b_calibration.csv")
        n_by_type = cal.group_by("info_type").agg(pl.len().alias("n")).sort("n", descending=True)
        print(dec, n_by_type, f"MIN_TYPE_N={MIN_TYPE_N}", flush=True)
        el = res.filter(pl.col("eligible"))
        best = (
            el.sort("p_pb")
            .head(1)
            .select("n_pos", "wins", "expected", "p_pb", "q_pb", "p_boot", "q_boot", "excess_pnl")
        )
        print("best wallet (no address):", best.row(0, named=True), flush=True)
        hist = res.filter(pl.col("eligible"))["p_pb"].hist(bin_count=10)
        lines = ["# Phase 3A: wallet tests (aggregate only; statistical flags, no addresses)", ""]
        lines += [f"- {k}: {v}" for k, v in sc.items()]
        lines += [
            "",
            "Phase 3b: calibration fit and repeat test use calib_eligible positions only (independent-news cutoff; jump_start cutoffs are outcome-selected and excluded); "
            "see results/phase3b_calibration.csv for the entry-price decile table before/after.",
        ]
        lines += [
            f"- calib_eligible resolved positions in fit: {cal.height}; wallets: {cal['wallet'].n_unique()}"
        ]
        lines += [
            f"- info_type counts in fit (min per-type training size {MIN_TYPE_N}, else pooled fallback): "
            + ", ".join(f"{r['info_type']}={r['n']}" for r in n_by_type.iter_rows(named=True))
        ]
        lines += [f"- best wallet (aggregate stats only): {best.row(0, named=True)}"]
        lines += ["", "p_pb histogram over eligible wallets (10 bins on [0,1]):", ""]
        lines += [f"- {c}" for c in hist["count"].to_list()]
        lines += [
            "",
            "Caveat (phase 3b): jump_start cutoffs are outcome-informed (a jump is detected toward the winner), so "
            "cheap winners / expensive losers are over-represented in that subset (win rate ~0.97-0.99 at prices 0.1-0.3, "
            "~0.03 at 0.8+); the independent-news-only subset is close to the diagonal. See the after_* rows in "
            "results/phase3b_calibration.csv.",
        ]
        lines += [
            "",
            "Power is limited: only independent-news positions enter the repeat test; most wallets have 1-2 positions.",
        ]
        (results / "phase3a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
