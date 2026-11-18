"""Paper-style charts (Agg backend): per-market q(t) vs peer band, and the D histogram."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

N_LARGE = 12  # largest taker trades marked per market


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-")[:n].lower() or "market"


def plot_market(
    row: dict,
    series: pl.DataFrame,
    baseline: pl.DataFrame,
    taker_trades: pl.DataFrame,
    out_dir: Path,
    tag: str = "",
) -> Path:
    """Figure 3 style: winner price q(t) against the cross-fitted m10/m50/m90 band.

    x axis is hours relative to news_ts (negative = before). Large taker trades are drawn at the
    winner-token price (triangle up = toward the winner, down = away), sized by USD.
    """
    cid = row["condition_id"]
    news = row.get("news_ts_best") or row.get("news_ts")
    js = row.get("jump_start_ts")
    x_js = None if (js is None or news is None) else (js - news).total_seconds() / 3600.0
    x_min = -168.0 if x_js is None or x_js >= -168 else x_js - 6.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = series.filter(pl.col("condition_id") == cid).sort("tau_h", descending=True)
    b = baseline.filter(pl.col("condition_id") == cid).sort("tau_h", descending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if b.height:
        x = -b["tau_h"].to_numpy()
        ax.fill_between(x, b["m10"], b["m90"], color="#9ecae1", alpha=0.6, label="peer m10-m90")
        ax.plot(x, b["m50"], color="#2171b5", lw=1.2, label="peer median m50")
    if s.height:
        ax.plot(-s["tau_h"].to_numpy(), s["q"], color="#d94801", lw=1.8, label="winner price q(t)")
    if news is not None and taker_trades is not None and taker_trades.height:
        outcome = (row.get("outcome") or "").lower()
        tr = taker_trades.filter(pl.col("condition_id") == cid)
        tr = tr.with_columns(pl.col("outcome_label").str.to_lowercase().alias("_lab"))
        tr = tr.filter(pl.col("_lab").is_in(["yes", "no"]))
        tr = tr.with_columns(
            ((pl.col("ts") - news).dt.total_seconds() / 3600.0).alias("x"),
            pl.when(pl.col("_lab") == outcome)
            .then(pl.col("price"))
            .otherwise(1 - pl.col("price"))
            .alias("pw"),
            (
                ((pl.col("_lab") == outcome) & (pl.col("side").str.to_uppercase() == "BUY"))
                | ((pl.col("_lab") != outcome) & (pl.col("side").str.to_uppercase() == "SELL"))
            ).alias("toward"),
        ).filter((pl.col("x") >= x_min) & (pl.col("x") < 6))
        big = tr.sort("usd", descending=True).head(N_LARGE)
        for toward, color, marker, name in (
            (True, "#238b45", "^", "large taker trade toward winner"),
            (False, "#cb181d", "v", "large taker trade away"),
        ):
            sub = big.filter(pl.col("toward") == toward)
            if sub.height:
                ax.scatter(
                    sub["x"], sub["pw"], s=np.clip(sub["usd"].to_numpy() / 20, 20, 400),
                    c=color, marker=marker, edgecolor="k", linewidth=0.4, alpha=0.85,
                    zorder=5, label=name,
                )  # fmt: skip
    src = "independent" if row.get("independent") else "price-based"
    ax.axvline(0, color="k", ls="--", lw=1, label=f"news time ({src})")
    if x_js is not None:
        ax.axvline(x_js, color="#6a51a3", ls=":", lw=1.6, label="jump_start_ts (q leaves pre-jump level)")
    ax.axvspan(-168, -1, color="grey", alpha=0.05)
    ax.set_xlim(x_min - 2, 6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("hours relative to the news time (scored window: -168 h to -1 h)")
    ax.set_ylabel("price of the eventual winner")

    def _f(name):
        v = row.get(name)
        return f"{name}=NaN" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{name}={v:.3f}"

    ax.set_title(
        f"{(row.get('question') or cid)[:90]}\n{_f('D')}  {_f('D_last24')}  {_f('max_jump')}  "
        f"[{row.get('info_type')}]",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    path = out_dir / f"{tag}{_slug(row.get('question') or cid)}_{cid[:8]}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_d_histogram(scores: pl.DataFrame, out_path: Path, threshold: float | None = None) -> Path:
    """Figure 2 style: histogram of D across scored markets (optionally marking the top-1% cut)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = scores["D"].drop_nulls().drop_nans().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    if d.size:
        ax.hist(d, bins=min(40, max(5, d.size // 4)), color="#6baed6", edgecolor="white")
    if threshold is not None and not np.isnan(threshold):
        ax.axvline(threshold, color="#cb181d", ls="--", label=f"top-1% cut D={threshold:.3f}")
        ax.legend()
    ax.set_xlabel("discrepancy score D (mean excess of winner price over peer median)")
    ax.set_ylabel("markets")
    ax.set_title(f"Distribution of D across {d.size} scored markets")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
