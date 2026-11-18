"""Rebuild the curated result figures offline (no network, no DuckDB).

    python scripts/make_figures.py            # all figures that can be built
    python scripts/make_figures.py placebo    # one of: placebo, nobel, calibration

Outputs (results/figures/, committable):
  within_event_placebo.png   real winner share above the 95th percentile of losers vs the shuffled-winner
                             range and the 48 h-earlier placebo window (from results/validation_within_event_placebos.csv)
  calibration_curve.png      observed win rate by entry-price decile, before and after the cutoff fix
                             (from results/phase3b_calibration.csv)
  nobel_event_moves.png      24 h log-odds move of every comparable outcome of the Nobel Peace Prize 2025
                             event (outcome names, no wallet data; from data/snapshots/event_outcome_moves.parquet
                             and markets.parquet; skipped with a message if the snapshots are absent)

The two power heatmaps (validation_within_event_power.png, validation_oneshot_power.png) are written by
`pmdetect validate within-event` and `pmdetect validate oneshot`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
SNAP = ROOT / "data" / "snapshots"

BLUE = "#1f5fa8"
ORANGE = "#d9701a"
GREY = "#8a8f98"
TEXT = "#222222"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def within_event_placebo() -> Path:
    df = pl.read_csv(RESULTS / "validation_within_event_placebos.csv", infer_schema_length=0)
    num = [
        "window_h", "observed_share_above_q95", "null_share_above_q95", "shuffle_lo95", "shuffle_hi95",
        "n_events",
    ]  # fmt: skip
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
    rows = []
    for w in (24, 12):
        shuf = df.filter(
            (pl.col("placebo") == "b_shuffled_winners")
            & (pl.col("event_set") == "share_above_q95_of_losers")
            & (pl.col("window_h") == w)
        ).row(0, named=True)
        p48 = df.filter(
            (pl.col("placebo") == "a_window_ending_48h_before_reveal")
            & (pl.col("event_set") == "intersection")
            & (pl.col("window_h") == w)
        ).row(0, named=True)
        rows.append(
            (
                f"{w} h window before the reveal\n(n = {int(shuf['n_events'])} events)",
                shuf,
                "real",
                w,
            )
        )
        rows.append(
            (
                f"Same window, ending 48 h earlier\n(n = {int(p48['n_events'])} events)",
                p48,
                "p48",
                w,
            )
        )

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
    ys = []
    for i, (label, r, kind, _w) in enumerate(rows):
        y = len(rows) - 1 - i - (0.4 if i >= 2 else 0)
        ys.append((y, label))
        null = r["null_share_above_q95"]
        obs = r["observed_share_above_q95"]
        if kind == "real":  # central 95% range of 100 shuffled-winner draws
            ax.hlines(
                y, r["shuffle_lo95"], r["shuffle_hi95"], color=GREY, lw=7, alpha=0.55, zorder=1
            )
        ax.plot([null], [y], marker="o", mfc="white", mec=GREY, mew=1.6, ms=8, ls="none", zorder=2)
        ax.plot(
            [obs],
            [y],
            marker="o",
            color=ORANGE if kind == "real" else BLUE,
            ms=9,
            ls="none",
            zorder=3,
        )
        ax.annotate(
            f"{obs:.3f}",
            (obs, y),
            xytext=(8, -3),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
        )
    ax.set_yticks([y for y, _ in ys])
    ax.set_yticklabels([lab for _, lab in ys], fontsize=8.5)
    ax.set_xlim(0.1, 0.62)
    ax.set_xlabel(
        "share of events whose winner's move exceeds the 95th percentile of the losers' moves"
    )
    ax.set_title(
        "Within-event test: winners already stand out two days before the reveal\n"
        "(all four rows fail the pre-registered placebo rules)", fontsize=10, color=TEXT, loc="left",
    )  # fmt: skip
    handles = [
        plt.Line2D([], [], marker="o", color=ORANGE, ls="none", ms=8, label="observed, real window"),
        plt.Line2D([], [], marker="o", color=BLUE, ls="none", ms=8, label="observed, 48 h-earlier window"),
        plt.Line2D([], [], marker="o", mfc="white", mec=GREY, mew=1.6, ls="none", ms=8, label="null (winner drawn at random)"),
        plt.Line2D([], [], color=GREY, lw=7, alpha=0.55, label="central 95% of 100 shuffles"),
    ]  # fmt: skip
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=False)
    _style(ax)
    fig.tight_layout()
    out = FIG / "within_event_placebo.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def calibration_curve() -> Path:
    d = pl.read_csv(RESULTS / "phase3b_calibration.csv")
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=150)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="perfectly calibrated")
    spec = [
        ("before", "all cutoff types, incl. price-based news times (before the fix)", ORANGE, "o"),
        ("after_independent_only", "independent news time only (after the fix)", BLUE, "s"),
    ]
    for key, lab, col, mk in spec:
        s = d.filter(pl.col("spec") == key).sort("decile")
        ax.plot(s["mean_entry"], s["win_rate"], marker=mk, color=col, lw=1.6, ms=5, label=lab)
    ax.set_xlabel("mean entry price in the decile")
    ax.set_ylabel("observed win rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        "Calibration of positions by entry-price decile", fontsize=10, color=TEXT, loc="left"
    )
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    _style(ax)
    fig.tight_layout()
    out = FIG / "calibration_curve.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def nobel_event_moves() -> Path | None:
    f_moves, f_mk = SNAP / "event_outcome_moves.parquet", SNAP / "markets.parquet"
    if not (f_moves.exists() and f_mk.exists()):
        print("nobel: data/snapshots not present, skipped")
        return None
    mk = pl.read_parquet(f_mk).filter(
        pl.col("question").str.contains("Machado win the Nobel Peace Prize in 2025")
    )
    if mk.height == 0:
        print("nobel: market not found, skipped")
        return None
    cid = mk["condition_id"][0]
    m = (
        pl.read_parquet(f_moves)
        .filter((pl.col("pilot_condition_id") == cid) & pl.col("move_24h").is_not_null())
        .with_columns(
            pl.col("label").str.strip_chars().alias("name"),
            (pl.col("condition_id") == cid).alias("winner"),
        )
        .sort("move_24h")
    )
    names = [n if len(n) <= 34 else n[:33] + "..." for n in m["name"].to_list()]
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * m.height + 1.6), dpi=150)
    cols = [ORANGE if w else BLUE for w in m["winner"].to_list()]
    ax.barh(range(m.height), m["move_24h"].to_list(), color=cols, height=0.7)
    ax.axvline(0, color=TEXT, lw=0.8)
    ax.set_yticks(range(m.height))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(
        "24 h change in log-odds of the YES price (window ends 1 h before the announcement)",
        fontsize=8.5,
    )
    for i, v in enumerate(m["move_24h"].to_list()):
        ax.annotate(
            f"{v:+.2f}",
            (max(v, 0), i),
            xytext=(4, -3),
            textcoords="offset points",
            fontsize=7,
            color=TEXT,
        )
    ax.set_title(
        f"Nobel Peace Prize 2025: 24 h move of the {m.height} comparable outcomes\n"
        "(eventual winner in orange; the largest mover is what convergence predicts)",
        fontsize=9.5, color=TEXT, loc="left",
    )  # fmt: skip
    ax.set_xlim(min(m["move_24h"].min() - 0.4, -0.5), m["move_24h"].max() + 1.0)
    _style(ax)
    fig.tight_layout()
    out = FIG / "nobel_event_moves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


ALL = {
    "placebo": within_event_placebo,
    "calibration": calibration_curve,
    "nobel": nobel_event_moves,
}


def main(argv: list[str]) -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    keys = argv or list(ALL)
    for k in keys:
        if k not in ALL:
            print(f"unknown figure '{k}', choose from {sorted(ALL)}")
            return 2
        out = ALL[k]()
        if out:
            print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
