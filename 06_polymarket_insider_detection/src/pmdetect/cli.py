"""pmdetect command line interface. Builder B modules are imported lazily inside commands."""

from __future__ import annotations

import csv
from datetime import date

import typer

from pmdetect.store.db import PROJECT_ROOT

app = typer.Typer(no_args_is_help=True, help="Polymarket insider-trading detector (phase 1).")
collect_app = typer.Typer(no_args_is_help=True, help="Collect data from Polymarket APIs.")
app.add_typer(collect_app, name="collect")

from pmdetect.validate.cli import app as validate_app  # noqa: E402  (light: heavy imports are lazy)

app.add_typer(validate_app, name="validate")

from pmdetect.stage2.cli import app as stage2_app  # noqa: E402  (light: heavy imports are lazy)

app.add_typer(stage2_app, name="stage2")

demo_app = typer.Typer(no_args_is_help=True, help="Offline end-to-end demo on ~50 fixture markets.")
app.add_typer(demo_app, name="demo")


@demo_app.command("run")
def demo_run_cmd(
    out: str = typer.Option("", help="Scratch output directory (default demo_output/, gitignored)."),
    n_perm: int = typer.Option(2000, help="Permutations for the within-event placebo."),
) -> None:
    """Run news times, within-event test and one-shot screen on the committed fixtures (offline)."""
    from pathlib import Path

    from pmdetect.demo import DEFAULT_OUT, run_demo

    r = run_demo(Path(out) if out else DEFAULT_OUT, n_perm=n_perm, log=typer.echo)
    typer.echo(f"summary: {r['summary']}")


@demo_app.command("fetch-plan")
def demo_fetch_plan_cmd() -> None:
    """Print (no network) the commands that collect a fresh ~50-market sample from the live API."""
    from pmdetect.demo import fetch_plan

    typer.echo(fetch_plan())


PILOT_CSV = PROJECT_ROOT / "data" / "manual" / "pilot_markets.csv"


def _pilot_ids() -> list[str]:
    if not PILOT_CSV.exists():
        typer.echo(f"{PILOT_CSV} missing: run `pmdetect pilot` first", err=True)
        raise typer.Exit(1)
    with PILOT_CSV.open(encoding="utf-8", newline="") as f:
        return [r["condition_id"] for r in csv.DictReader(f)]


def _ids(pilot: bool, limit: int | None) -> list[str]:
    from pmdetect.store.db import connect

    if pilot:
        ids = _pilot_ids()
    else:
        con = connect()
        ids = [
            r[0]
            for r in con.execute(
                "SELECT condition_id FROM markets WHERE in_universe ORDER BY condition_id"
            ).fetchall()
        ]
        con.close()
    return ids[:limit] if limit else ids


@collect_app.command("markets")
def collect_markets_cmd(
    start: str = typer.Option("2025-01-01", help="Earliest market end date (YYYY-MM-DD)."),
    end: str = typer.Option("", help="Latest end date; default today."),
    pages_per_month: int = typer.Option(2, help="Max 100-event pages per monthly window."),
    include_sports: bool = typer.Option(False, help="Do not exclude sports/crypto-price tags."),
) -> None:
    """Collect closed Gamma event markets (resumable, raw JSON cached)."""
    from pmdetect.collect.gamma import collect_markets

    stats = collect_markets(
        date.fromisoformat(start),
        date.fromisoformat(end) if end else None,
        max_pages_per_window=pages_per_month,
        exclude_tag_ids=[] if include_sports else None,
    )
    typer.echo(stats)


@collect_app.command("trades")
def collect_trades_cmd(
    pilot: bool = typer.Option(True, help="Use data/manual/pilot_markets.csv (else universe)."),
    limit: int | None = typer.Option(None),
) -> None:
    """Collect trades + resolutions (Builder B)."""
    from pmdetect.collect.data_api import collect_resolutions, collect_trades

    ids = _ids(pilot, limit)
    typer.echo(f"trades: {collect_trades(ids)}  resolutions: {collect_resolutions(ids)}")


@collect_app.command("prices")
def collect_prices_cmd(
    pilot: bool = typer.Option(True), limit: int | None = typer.Option(None)
) -> None:
    """Collect price history per token (Builder B)."""
    from pmdetect.collect.data_api import collect_prices
    from pmdetect.store.db import connect

    ids = _ids(pilot, limit)
    con = connect()
    toks = [
        r[0]
        for r in con.execute(
            "SELECT token_id FROM tokens WHERE condition_id IN (SELECT unnest(?::VARCHAR[]))", [ids]
        ).fetchall()
    ]
    con.close()
    typer.echo(f"prices: {collect_prices(toks)}")


@collect_app.command("wallets")
def collect_wallets_cmd(
    limit: int = typer.Option(None, help="Stop after this many wallets (default: all)."),
    full: bool = typer.Option(False, help="Full /activity history per wallet (slow)."),
) -> None:
    """Wallet first_seen for every wallet in stored trades; resumable, priority-ordered."""
    from pmdetect.collect import data_api

    if not full:
        typer.echo(f"wallets: {data_api.collect_wallet_first_seen(limit)}")
        return
    from pmdetect.store.db import connect

    con = connect()
    wallets = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT wallet FROM trades WHERE wallet IS NOT NULL"
        ).fetchall()
    ]
    con.close()
    typer.echo(f"wallets: {data_api.collect_wallet_activity(wallets[:limit])}")


@app.command("universe")
def universe_cmd() -> None:
    """Apply universe rules to stored markets and report drops."""
    from pmdetect.markets.universe import build_universe
    from pmdetect.store.db import connect

    con = connect()
    res = build_universe(con)
    con.close()
    typer.echo(f"markets={res['total']} in_universe={res['kept']}")
    for reason, n in res["drops"].items():
        typer.echo(f"  dropped {n:>6}  {reason}")


@app.command("infotype")
def infotype_cmd() -> None:
    """Tag info_type on all markets and report agreement with hand labels."""
    from pmdetect.markets.info_type import agreement, load_labels, tag_markets
    from pmdetect.store.db import connect

    con = connect()
    typer.echo(f"tagged: {tag_markets(con)}")
    labels = load_labels()
    for split in sorted({r.get("split", "all") for r in labels}):
        sub = [r for r in labels if r.get("split", "all") == split]
        a = agreement(sub, con=con)
        typer.echo(f"agreement[{split}]: {a['agree']}/{a['n']} = {a['rate']:.1%}")
        for q, hand, pred in a["misses"]:
            typer.echo(f"   miss: {q[:70]!r} hand={hand} auto={pred}")
    con.close()


@app.command("seed-labels")
def seed_labels_cmd() -> None:
    """Resolve data/manual/known_cases.csv through Gamma and seed `labels`."""
    from pmdetect.markets.cases import seed_labels

    typer.echo(seed_labels())


@app.command("pilot")
def pilot_cmd(n: int = typer.Option(200), seed: str = typer.Option("pilot1")) -> None:
    """Pick ~n universe markets (one per event, spread over time) plus case markets."""
    from pmdetect.store.db import connect

    con = connect()
    rows = con.execute(
        """
        SELECT condition_id, question, volume_usd FROM (
          SELECT *, row_number() OVER (PARTITION BY event_id ORDER BY hash(condition_id || ?)) rn
          FROM markets WHERE in_universe
        ) WHERE rn = 1 ORDER BY hash(condition_id || ?) LIMIT ?
        """,
        [seed, seed, n],
    ).fetchall()
    cases = con.execute(
        "SELECT DISTINCT m.condition_id, m.question, m.volume_usd FROM labels l "
        "JOIN markets m USING (condition_id) WHERE l.label = 'insider_suspected'"
    ).fetchall()
    con.close()
    seen, out = set(), []
    for cid, q, v in cases + rows:
        if cid not in seen:
            seen.add(cid)
            out.append((cid, q, v, "case" if (cid, q, v) in cases else "universe"))
    PILOT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PILOT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition_id", "question", "volume_usd", "source"])
        w.writerows(out)
    typer.echo(f"pilot: {len(out)} markets -> {PILOT_CSV}")


@app.command("newstime")
def newstime_cmd(
    legacy: bool = typer.Option(
        False, help="Per-market DuckDB estimator (pilot only; needs fetch_log) instead of the bulk one."
    ),
) -> None:
    """Estimate news times: vectorised over the Parquet snapshots for every universe market, one UPDATE."""
    from pmdetect.markets.newstime import estimate_news_times, estimate_news_times_bulk

    if legacy:
        typer.echo(f"news times set: {estimate_news_times()}")
        return
    res = estimate_news_times_bulk()
    typer.echo(f"news times set: {res['news_ts'].is_not_null().sum()} of {res.height} markets")
    typer.echo("refresh snapshots next: python scripts/make_snapshot.py")


@app.command("score-markets")
def score_markets_cmd(
    refresh_taker: bool = typer.Option(False, help="Disabled (the full taker tape comes from bulk_fetch.py)."),
    baseline: str = typer.Option(
        "timesplit", "--baseline", help="Level baseline scheme: timesplit (default) or crossfit."
    ),
) -> None:
    """Phase 2: series -> baseline -> score -> signals -> combine -> plots (Parquet only)."""
    from pmdetect.report.phase2 import run_phase2

    if refresh_taker:
        # The taker tape is now the FULL universe tape built by scripts/bulk_fetch.py; the old
        # in-process refresh only knew the 399 pilot markets and would overwrite it.
        typer.echo(
            "--refresh-taker is disabled: data/snapshots/trades_taker.parquet is the full tape "
            "(8.5M rows) built by scripts/bulk_fetch.py + scripts/load_tapes.py.",
            err=True,
        )
        raise typer.Exit(2)
    if baseline not in ("timesplit", "crossfit"):
        typer.echo("--baseline must be timesplit or crossfit", err=True)
        raise typer.Exit(2)
    res = run_phase2(baseline_scheme=baseline)
    typer.echo(
        f"scored D: {res['n_scored_D']}  top-1% D threshold: {res['d_threshold']:.4f}  "
        f"figures: {res['n_figs']}  summary: {res['summary']}"
    )
    typer.echo(f"wall time {res['timings']['total']:.0f}s  peak memory {res['peak_mb'] / 1024:.1f} GB")


@app.command("score-wallets")
def score_wallets_cmd(
    rebuild_positions: bool = typer.Option(
        False, "--rebuild-positions", help="Force a positions rebuild (default: auto when inputs are newer)."
    ),
) -> None:
    """Phase 3: positions -> tests -> FDR -> filters -> one-shot -> clusters -> watchlist.

    Full-universe scale, Parquet only, no network. Wallet-level outputs go to data/private/ only;
    results/phase3_summary.md is aggregate-only (and checked for addresses).
    """
    from pmdetect.wallets.pipeline import run_score_wallets

    r = run_score_wallets(rebuild_positions=True if rebuild_positions else None, log=typer.echo)
    typer.echo(
        f"positions={r['positions']} tested={r['tested']} fdr_pass={r['fdr_pass']} "
        f"excluded={r['excluded']} clusters={r['clusters']} watchlist={r['watchlist']} "
        f"total={r['total_s']}s peak_rss={r['peak_mb']}MB summary={r['summary']} "
        f"(privacy-checked {r['privacy_files']} files under results/)"
    )


@app.command("first-seen-candidates")
def first_seen_candidates_cmd(
    n_top: int = typer.Option(5000, help="Top one-shot positions (valid cutoff) by score without freshness."),
    n_novalid: int = typer.Option(1000, help="Top no-valid-cutoff positions by the proximity-free screen."),
) -> None:
    """Select (locally, no network) the wallets whose first_seen should be fetched; writes
    data/private/first_seen_candidates.parquet and prints the counts and the hours estimate."""
    from pmdetect.wallets.firstseen import select_candidates

    r = select_candidates(n_top=n_top, n_novalid=n_novalid, log=typer.echo)
    typer.echo(r["report"])


@app.command("integrity")
def integrity_cmd() -> None:
    """Run the trade/volume integrity report (Builder B)."""
    from pmdetect.quality.integrity import run_integrity_report

    typer.echo(f"report: {run_integrity_report()}")


if __name__ == "__main__":
    app()
