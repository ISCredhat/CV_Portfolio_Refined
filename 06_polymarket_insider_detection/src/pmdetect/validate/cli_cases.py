"""``pmdetect validate cases`` and ``pmdetect validate audit`` (phase 4D labelled cases, 4E hand audit)."""

from __future__ import annotations

from pathlib import Path

import typer


def cases_cmd(
    out_dir: str | None = typer.Option(None, help="Results folder (default: results/)."),
) -> None:
    """Phase 4D: the labelled cases on every score (held out), aggregate-only table."""
    from pmdetect.validate import cases

    kw = {"results": Path(out_dir)} if out_dir else {}
    res = cases.run(log=typer.echo, **kw)
    typer.echo(f"rows: {len(res['table'])}")


def audit_cmd(
    n: int = typer.Option(30, help="Rows per list."),
    out_dir: str | None = typer.Option(None, help="Results folder (default: results/)."),
) -> None:
    """Phase 4E: hand audit of the top within-event extremes and top one-shot positions (offline)."""
    from pmdetect.validate import audit

    kw = {"results": Path(out_dir)} if out_dir else {}
    res = audit.run(n=n, log=typer.echo, **kw)
    for name in ("within_event", "oneshot"):
        typer.echo(f"{name}: " + ", ".join(f"{c} {n_}" for c, n_, _ in audit._mix(res[name])))
