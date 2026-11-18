"""``pmdetect validate`` command group. Other validation builders add their subcommands here or
register their own ``typer.Typer`` with ``app.add_typer``; heavy imports stay inside the commands."""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, help="Phase 4 validation: simulations and placebos.")


@app.command("within-event")
def within_event_cmd(
    n_events: int | None = typer.Option(None, help="Use only this many events (fast tests)."),
    seed: int = typer.Option(20261113, help="Master seed."),
    n_boot: int = typer.Option(1000, help="Bootstrap resamples (events)."),
    n_perm: int = typer.Option(2000, help="Permutation draws for the placebo p-values."),
    out_dir: str | None = typer.Option(None, help="Output folder (default: results/)."),
) -> None:
    """Synthetic-insider power grid + placebos for the within-event pre-reveal test."""
    from pathlib import Path

    from pmdetect.validate.within_event_sim import run_within_event_validation

    res = run_within_event_validation(
        n_events=n_events,
        seed=seed,
        n_boot=n_boot,
        n_perm=n_perm,
        out_dir=Path(out_dir) if out_dir else None,
        log=typer.echo,
    )
    typer.echo(res["summary"])


@app.command("oneshot")
def oneshot_cmd(
    n_markets: int = typer.Option(0, help="Random draw of this many eligible markets (0 = all)."),
    seed: int = typer.Option(20261113, help="Master seed."),
    n_boot: int = typer.Option(1000, help="Bootstrap resamples (markets)."),
    out_dir: str | None = typer.Option(None, help="Results folder (default: results/)."),
    parity: bool = typer.Option(True, help="Check the fast scorer against the full score_oneshot."),
) -> None:
    """Synthetic-insider power grid + label-flip placebo for the one-shot screen."""
    from pmdetect.validate.cli_oneshot import oneshot_cmd as _run

    _run(n_markets=n_markets, seed=seed, n_boot=n_boot, out_dir=out_dir, parity=parity)


@app.command("cases")
def cases_cmd(out_dir: str | None = typer.Option(None, help="Results folder (default: results/).")) -> None:
    """Labelled cases on every score (held out): results/validation_labelled_cases.csv + .md."""
    from pmdetect.validate.cli_cases import cases_cmd as _run

    _run(out_dir=out_dir)


@app.command("audit")
def audit_cmd(
    n: int = typer.Option(30, help="Rows per list."),
    out_dir: str | None = typer.Option(None, help="Results folder (default: results/)."),
) -> None:
    """Hand audit of the top within-event extremes and top one-shot positions (offline, descriptive)."""
    from pmdetect.validate.cli_cases import audit_cmd as _run

    _run(n=n, out_dir=out_dir)
