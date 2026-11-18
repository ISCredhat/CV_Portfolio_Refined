"""``pmdetect stage2`` command group (stage 2 research; offline only, sealed period never opened).

Each subcommand imports its heavy module lazily. T1 adds ``convergence`` here or registers it directly.
"""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, help="Stage 2: markouts, forecasting benchmark, convergence.")


@app.command("markouts")
def markouts_cmd(
    n_draws: int = typer.Option(2000, help="Sign-randomisation draws per period."),
    n_boot: int = typer.Option(1000, help="Event-cluster bootstrap resamples."),
    n_shuffle: int = typer.Option(200, help="Shuffled-wallet baseline draws."),
    threads: int = typer.Option(8, help="Worker threads for the sign randomisation."),
    rebuild: bool = typer.Option(False, help="Recompute the trade-level markouts even if cached."),
) -> None:
    """Task B: taker-trade markouts, wallet skill v2 (sign randomisation, EB shrinkage), persistence."""
    from pmdetect.stage2.markouts import run_markouts

    r = run_markouts(
        n_draws=n_draws, n_boot=n_boot, n_shuffle=n_shuffle, threads=threads, rebuild=rebuild, log=typer.echo
    )
    typer.echo(f"verdict: {r['verdict']}")


@app.command("forecast")
def forecast_cmd(
    n_boot: int = typer.Option(1000, help="Event-cluster bootstrap resamples."),
    rebuild: bool = typer.Option(False, help="Recompute the snapshot features even if cached."),
) -> None:
    """Task C: forecasting benchmark against the market price (the stop gate)."""
    from pmdetect.stage2.forecast import run_forecast

    r = run_forecast(n_boot=n_boot, rebuild=rebuild, log=typer.echo)
    typer.echo(f"verdict: {r['verdict']}")


@app.command("convergence")
def convergence_cmd() -> None:
    """Task A: tradable convergence backtest (strategy 2), fit and tune periods only."""
    from pmdetect.stage2.convergence import run

    rep = run(log=typer.echo)
    for r in rep.passes:
        typer.echo(f"{r['rule']} {r['horizon']}: {'PASS' if r['PASS'] else 'FAIL'}")
