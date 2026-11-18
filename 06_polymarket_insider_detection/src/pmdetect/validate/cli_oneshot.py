"""``pmdetect validate oneshot``: synthetic-insider power grid and label-flip placebo for the one-shot screen."""

from __future__ import annotations

from pathlib import Path

import typer


def oneshot_cmd(
    n_markets: int = typer.Option(0, help="Random draw of this many eligible markets (0 = all available)."),
    seed: int = typer.Option(20261113, help="Master seed."),
    n_boot: int = typer.Option(1000, help="Bootstrap resamples (markets)."),
    out_dir: str | None = typer.Option(None, help="Results folder (default: results/)."),
    parity: bool = typer.Option(True, help="Check the fast scorer against the full score_oneshot."),
) -> None:
    """Phase 4B/4C: inject fresh winning wallets, report detection by stake and lead, and the flipped-label placebo."""
    from pmdetect.validate import oneshot_sim

    kw = {"results": Path(out_dir)} if out_dir else {}
    res = oneshot_sim.run(n_markets=n_markets, seed=seed, n_boot=n_boot, parity=parity, log=typer.echo, **kw)["result"]
    meet = res.cells.filter((res.cells["grid"] == "plan") & res.cells["meets_criterion"])
    typer.echo(f"cells meeting >= 50%: {meet.height}; flipped-label share {res.placebo['flipped']['share']:.2%}")
