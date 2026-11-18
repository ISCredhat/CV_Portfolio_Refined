"""Offline end-to-end demo on about 50 fixture markets (`pmdetect demo run`).

The fixtures in ``tests/fixtures/demo/`` are a deterministic (seed 20261113) sample of the real Parquet
snapshots with anonymised wallet ids (``wallet-NNNNN``), built by ``scripts/build_demo_fixtures.py``.
``run_demo`` copies them into a scratch snapshot directory and runs the two packaged detectors with the
real pipeline code, never touching ``data/`` or ``results/`` and never the network or DuckDB:

1. news times      ``newstime.bulk_news_times`` (price-based, plus the manual override CSV)
2. independent     ``newstime_indep.independent_news_times`` (override or scheduled end time)
3. jumps           ``changepoint.detect_jumps`` (cutoff fallback for markets without an independent time)
4. within-event    ``within_event.run`` (winner against its sibling outcomes, final 24 h / 12 h, placebo)
5. one-shot screen ``wallets.pipeline.run_score_wallets`` (positions, calibration, filters, one-shot score)
6. combined summary ``demo_summary.md`` (aggregate counts; fake wallet ids only; a 40-hex scan of every output)

The sample is far too small to say anything statistically; it shows that the code runs and what the
outputs look like. Level D, repeat-wallet FDR, burst times and clusters are not part of the packaged
detectors; the wallet pipeline still computes them (they are documented as negative results), so the
scratch run contains their files too.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import polars as pl

from pmdetect.store.db import PROJECT_ROOT

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "demo"
DEFAULT_OUT = PROJECT_ROOT / "demo_output"
SEED = 20261113
HEX40 = re.compile(r"\b(?:0x)?[0-9a-fA-F]{40}\b")
SNAP_FILES = [
    "markets", "tokens", "trades_taker", "prices", "wallets", "event_markets", "event_prices",
]  # fmt: skip


def _guard_out(out: Path) -> Path:
    """The scratch directory must not be (inside) the real data store or the curated results."""
    out = out.resolve()
    for protected in (PROJECT_ROOT / "data", PROJECT_ROOT / "results", PROJECT_ROOT / "src", FIXTURES):
        p = protected.resolve()
        if out == p or p in out.parents:
            raise ValueError(f"refusing to write the demo into {out} (inside {p.name}/); use another --out")
    return out


def scan_hex40(paths: list[Path]) -> list[str]:
    """Files under ``paths`` that contain a 40-hex string (wallet-address shape): text files and every
    string column of Parquet files. Empty list = clean."""
    hits: list[str] = []
    files = [f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*"))) if f.is_file()]
    for f in files:
        suffix = f.suffix.lower()
        if suffix == ".parquet":
            df = pl.read_parquet(f)
            for c, dt in df.schema.items():
                if dt == pl.String and df[c].str.contains(HEX40.pattern).any():
                    hits.append(f"{f.name}:{c}")
        elif suffix in {".csv", ".md", ".json", ".txt", ".log"}:
            if HEX40.search(f.read_text(encoding="utf-8", errors="ignore")):
                hits.append(f.name)
    return hits


def _step(log, name: str, t0: float) -> None:
    log(f"  {name}: {time.time() - t0:.1f}s")


def run_demo(
    out: Path = DEFAULT_OUT,
    fixtures: Path = FIXTURES,
    n_perm: int = 2000,
    log=print,
) -> dict:
    """Run the demo and return a dict of aggregate results (paths and counts)."""
    from pmdetect.markets import within_event
    from pmdetect.markets.changepoint import detect_jumps
    from pmdetect.markets.newstime import bulk_news_times, load_bulk_inputs, load_overrides
    from pmdetect.markets.newstime_indep import independent_news_times, last_trade_times
    from pmdetect.markets.series import load_inputs
    from pmdetect.wallets.pipeline import run_score_wallets

    out = _guard_out(Path(out))
    fixtures = Path(fixtures)
    if not (fixtures / "markets.parquet").exists():
        raise FileNotFoundError(f"fixtures not found in {fixtures} (scripts/build_demo_fixtures.py builds them)")
    snap, private, results = out / "snapshots", out / "private", out / "results"
    for d in (snap, private, results):  # only our own three sub-folders are ever cleared
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    t_all = time.time()
    for name in SNAP_FILES:
        shutil.copy2(fixtures / f"{name}.parquet", snap / f"{name}.parquet")
    overrides = load_overrides(fixtures / "news_time_overrides.csv")
    log(f"demo run: scratch directory {out}")

    t0 = time.time()
    inp = load_bulk_inputs(snap)
    nt = bulk_news_times(inp["markets"], inp["tokens"], inp["prices"], inp["trades"], overrides)
    mk = pl.read_parquet(snap / "markets.parquet").drop("news_ts", "news_ts_method").join(
        nt, on="condition_id", how="left"
    )
    mk.write_parquet(snap / "markets.parquet")
    _step(log, "1 news times (price-based + override)", t0)

    t0 = time.time()
    markets, prices = load_inputs(snap)
    ni = independent_news_times(markets, last_trade_times(snap), overrides)
    ni.write_parquet(snap / "news_independent.parquet")
    _step(log, "2 independent news times", t0)

    t0 = time.time()
    taker = pl.read_parquet(snap / "trades_taker.parquet")
    jumps = detect_jumps(markets, prices, taker).filter(pl.col("jump_start_ts").is_not_null())
    jumps.select("condition_id", "jump_start_ts").write_parquet(snap / "jumps.parquet")
    _step(log, "3 resolving-jump start times", t0)

    t0 = time.time()
    within_event.run(
        snapshot_dir=snap, n_perm=n_perm, log=lambda *_: None, parts_dir=snap / "_no_parts",
        consolidate_first=False,
    )  # fmt: skip
    _step(log, "4 within-event test + placebo", t0)

    t0 = time.time()
    res = run_score_wallets(snap, private, results, rebuild_positions=True, log=lambda *_: None)
    _step(log, "5 one-shot screen (positions, calibration, filters, scores)", t0)

    summary = write_demo_summary(out, snap, private, results, res, nt, ni, time.time() - t_all)
    hits = scan_hex40([out])
    if hits:
        raise RuntimeError(f"40-hex strings found in demo outputs: {hits}")
    log(f"  6 combined summary: {summary}")
    log(f"demo done in {time.time() - t_all:.0f}s; 40-hex scan of {out.name}/: clean")
    return {"out": out, "summary": summary, "seconds": time.time() - t_all, "positions": res["positions"]}


def _md(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_none_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows) + "\n"


def _f(x, nd=2) -> str:
    return "NaN" if x is None else f"{x:.{nd}f}"


def write_demo_summary(out, snap, private, results, res, nt, ni, seconds) -> Path:
    """Aggregate-only markdown (market questions are public; wallet ids are the anonymised fixtures)."""
    mk = pl.read_parquet(snap / "markets.parquet")
    we = pl.read_parquet(snap / "within_event.parquet")
    plc = json.loads((snap / "within_event_placebo.json").read_text(encoding="utf-8"))
    L = ["# Demo summary (fixture sample, offline)", ""]
    L.append(
        f"Sample: {mk.height} markets; {mk['info_type'].value_counts().sort('info_type').rows()} by info type; "
        f"wall time {seconds:.0f} s. Fixture wallet ids are anonymised (`wallet-NNNNN`); results are "
        "descriptive only (a 50-market sample has no statistical power)."
    )
    L += ["", "## News times", ""]
    L.append("Price-based news time (first stable 0.9 crossing of the winner price), method counts: "
             + ", ".join(f"{m}={n}" for m, n in nt["news_ts_method"].value_counts().sort("news_ts_method").rows()) + ".")
    L.append("Independent news time (override or scheduled end time), method counts: "
             + ", ".join(f"{m}={n}" for m, n in ni["indep_method"].value_counts().sort("indep_method").rows()) + ".")
    L += ["", "## Within-event pre-reveal test", ""]
    multi = we.filter(pl.col("group") == "multi").filter(pl.col("winner_pct_24h").is_not_null())
    binary = we.filter(pl.col("group") == "binary").filter(pl.col("winner_pct_24h").is_not_null())
    L.append(f"Events scored: {we.height} rows; multi-outcome with a comparable-loser set: {multi.height}; "
             f"binary reveals with a pooled percentile: {binary.height}. Winner percentile = rank of the "
             "winner's 24 h log-odds move among all outcomes of its event, window ending 1 h before the reveal.")
    q = mk.select("condition_id", "question")
    top = multi.join(q, on="condition_id", how="left").sort("winner_pct_24h", "diff_24h", descending=True).head(8)
    L += ["", _md([{"market": (r["question"] or "")[:70], "n_outcomes": r["n_outcomes"],
                    "winner_move_24h": _f(r["win_move_24h"]), "median_loser_move_24h": _f(r["med_lose_move_24h"]),
                    "pct_24h": _f(r["winner_pct_24h"])} for r in top.iter_rows(named=True)],
                   ["market", "n_outcomes", "winner_move_24h", "median_loser_move_24h", "pct_24h"])]
    for hz in ("24h", "12h"):
        r = plc.get(hz) or {}
        L.append(f"- Pooled placebo, final {hz}: " + "; ".join(
            f"{k} = {v:.4g}" if isinstance(v, (int, float)) else f"{k} = {v}" for k, v in r.items() if v is not None))
    L += ["", "## One-shot bet screen", ""]
    sc, tests = res["scored"], res["tests"]
    L.append(f"Positions built: {res['positions']} (cutoff method counts: "
             + ", ".join(f"{m}={n}" for m, n in res["positions_df"]["cutoff_method"].value_counts().sort("cutoff_method").rows())
             + f"); wallets tested for repeat anomalies: {tests.height}; passing FDR: {res['fdr_pass']} "
             "(a sample this small has almost no power to pass FDR); wallets excluded as non-directional: " + f"{res['excluded']}.")
    L.append(f"Winning positions scored by the one-shot screen: {sc.height}.")
    score_col = next((c for c in ("oneshot_score", "score") if c in sc.columns), None)
    if score_col:
        top = sc.sort(score_col, descending=True).head(5).join(q, on="condition_id", how="left") \
            if "condition_id" in sc.columns else sc.head(0)
        rows = [{"market": (r.get("question") or "")[:60], "entry_price": _f(r.get("entry_price"), 3),
                 "usd": _f(r.get("usd"), 0), "score": _f(r[score_col], 3)} for r in top.iter_rows(named=True)]
        L += ["", "Top positions by one-shot score (fake wallet ids omitted):", "", _md(rows, ["market", "entry_price", "usd", "score"])]
    wl = res["watchlist_df"]
    if wl.height and "tier" in wl.columns:
        L.append("Watchlist tiers: " + ", ".join(f"{t}={n}" for t, n in wl["tier"].value_counts().sort("tier").rows()) + ".")
    L += ["", "## Files", "",
          f"- scratch snapshots (inputs copied from the fixtures plus derived Parquet): `{snap.name}/`",
          f"- wallet-level outputs (anonymised ids): `{private.name}/`",
          f"- aggregate wallet-pipeline summary: `{results.name}/phase3_summary.md`"]
    path = out / "demo_summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


FETCH_PLAN = r"""Fresh ~50-market sample from the live API (commands only; nothing is run here).
Run from the project folder in a Python 3.11+ venv with `pip install -e .[dev]`. Free, keyless APIs,
about 5 requests/s; the whole sample takes a few minutes of fetching. All state lands in data/ (gitignored).

 1. Closed Gamma markets, resumable (2 pages per month keeps it small):
      pmdetect collect markets --start 2025-06-01 --pages-per-month 1
 2. Universe filter and information-type tags:
      pmdetect universe
      pmdetect infotype
 3. Pick ~50 markets (one per event, spread over time) into data/manual/pilot_markets.csv:
      pmdetect pilot --n 50 --seed demo1
 4. Their taker/maker trades, resolutions and price histories into data/pmdetect.duckdb:
      pmdetect collect trades --pilot
      pmdetect collect prices --pilot
 5. Parquet snapshots (read-only analysis never takes the DuckDB write lock):
      python scripts/make_snapshot.py
 6. Taker-only tape for those markets (one row per fill; the all-rows tape lists every fill twice):
      # bulk_fetch.py covers every in-universe market in the snapshot (frozen in data/tapes/universe_ids.json).
      # After a 1-page-per-month collect that is small; to restrict it to the 50 pilot markets write that
      # JSON first as {"conditions": [<50 ids>], "winner_tokens": []} (prices come from step 4 here).
      python scripts/bulk_fetch.py taker --workers 2
      python scripts/load_tapes.py taker
 7. News times, independent news times:
      pmdetect newstime
      python -m pmdetect.markets.newstime_indep
 8. Sibling outcomes of each independent market's event, for the within-event test (small slice):
      python scripts/event_fetch_full.py --estimate         # request and hour estimate, no network
      python scripts/event_fetch_full.py --max-events 40
 9. One-shot screen and aggregate summary:
      pmdetect score-wallets

The offline demo (`pmdetect demo run`) needs none of this: it uses the committed fixtures.
"""


def fetch_plan() -> str:
    return FETCH_PLAN
