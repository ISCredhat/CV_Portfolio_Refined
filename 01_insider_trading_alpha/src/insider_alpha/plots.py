"""Figure helpers: one consistent, accessible style for every chart in the project."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order - colour follows the entity, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "font.size": 10, "figure.dpi": 110, "savefig.dpi": 150,
        "axes.axisbelow": True,
    })


def _finish(fig, path, note: str | None = None):
    for ax in fig.axes:
        _place_labels(ax)
    if note:
        note = note.replace("$", r"\$")
        fig.text(0.01, 0.005, note, fontsize=8, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _direct_label(ax, x, y, text, color):
    """Queue an end-of-line label; call _place_labels(ax) once all lines are drawn."""
    ax.plot([x], [y], "o", ms=5, color=color, mec=SURFACE, mew=1.5)
    if not hasattr(ax, "_pending_labels"):
        ax._pending_labels = []
    ax._pending_labels.append((x, y, text))


def _place_labels(ax, min_gap_px: float = 13.0):
    """Draw queued end-of-line labels, nudged apart vertically so they never overlap."""
    items = getattr(ax, "_pending_labels", [])
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    rows = []
    for x, y, t in items:
        xp, yp = ax.transData.transform((mdates_num(x), y))
        rows.append([yp, xp, t])
    rows.sort(key=lambda r: r[0])
    for i in range(1, len(rows)):
        if rows[i][0] - rows[i - 1][0] < min_gap_px:
            rows[i][0] = rows[i - 1][0] + min_gap_px
    offset = matplotlib.transforms.ScaledTranslation(6 / 72, 0, fig.dpi_scale_trans)
    inv = ax.transAxes.inverted()
    for yp, xp, t in rows:
        xa, ya = inv.transform((xp, yp))
        ax.text(xa, ya, t, transform=ax.transAxes + offset, va="center", fontsize=9, color=INK,
                fontweight="semibold", clip_on=False)
    ax._pending_labels = []


def mdates_num(x):
    import matplotlib.dates as mdates
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return mdates.date2num(pd.Timestamp(x))
    return x


def car_paths(paths: dict[str, np.ndarray], title: str, path, note: str | None = None, ylabel="Mean abnormal return vs IWM"):
    """paths: label -> mean abnormal-return path (days 0..H-1); plotted from day 0 = 0%."""
    style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (lab, p) in enumerate(paths.items()):
        y = np.concatenate([[0.0], p])
        x = np.arange(len(y))
        ax.plot(x, y, color=SERIES[i], label=lab)
        _direct_label(ax, x[-1], y[-1], f"{y[-1]:+.1%}", SERIES[i])
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Trading days after entry (open of the session after the filing)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)
    ax.set_xlim(0, len(y) + 12)
    _finish(fig, path, note)


def bars(labels, values, title, path, ylabel, errors=None, note=None, fmt="{:+.1%}", highlight=None):
    style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(values))
    cols = [SERIES[0] if (highlight is None or i in highlight) else "#9ec5f4" for i in range(len(values))]
    ax.bar(x, values, color=cols, width=0.62, yerr=errors, ecolor=INK_2, capsize=3,
           error_kw={"elinewidth": 1})
    for xi, v in zip(x, values):
        ax.annotate(fmt.format(v), (xi, v), xytext=(0, 4 if v >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    _finish(fig, path, note)


def equity_curves(curves: dict[str, pd.Series], title: str, path, note: str | None = None):
    """Growth of $1 (log scale) with a drawdown panel underneath (same x, separate panels)."""
    style()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for i, (lab, r) in enumerate(curves.items()):
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, color=SERIES[i], label=lab, lw=2.2 if i == 0 else 1.6)
        _direct_label(ax, eq.index[-1], eq.values[-1], f"{eq.values[-1]:.2f}x", SERIES[i])
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd.values, color=SERIES[i], lw=1.2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mtick.LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_locator(mtick.NullLocator())
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax2.set_ylabel("Drawdown")
    ax2.axhline(0, color=NEUTRAL, lw=1)
    _finish(fig, path, note)


def lines_xy(series: dict[str, tuple[np.ndarray, np.ndarray]], title, path, xlabel, ylabel, note=None,
             yfmt=None, xfmt=None, ref: tuple[float, str] | None = None):
    style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    if ref is not None:
        ax.axhline(ref[0], color=NEUTRAL, lw=1.2, ls="--", label=f"{ref[1]} ({ref[0]:.2f})")
    for i, (lab, (x, y)) in enumerate(series.items()):
        ax.plot(x, y, color=SERIES[i], label=lab, marker="o", ms=5, mec=SURFACE, mew=1.5)
        _direct_label(ax, x[-1], y[-1], f"{y[-1]:.2f}", SERIES[i])
    ax.axhline(0, color=NEUTRAL, lw=1)
    if xfmt:
        ax.xaxis.set_major_formatter(xfmt)
    if yfmt:
        ax.yaxis.set_major_formatter(yfmt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * 0.10)
    ax.legend(loc="lower left")
    _finish(fig, path, note)


def hbars(labels, values, title, path, xlabel, note=None, errors=None):
    style()
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(values) + 1.4))
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, color=SERIES[0], height=0.62, xerr=errors, ecolor=INK_2, capsize=2,
            error_kw={"elinewidth": 1})
    ax.set_yticks(y, labels)
    ax.axvline(0, color=NEUTRAL, lw=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _finish(fig, path, note)


def grouped_bars(labels, series: dict[str, np.ndarray], title, path, ylabel, errors: dict | None = None, note=None):
    style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    k = len(series)
    width = 0.8 / k
    x = np.arange(len(labels))
    for i, (lab, vals) in enumerate(series.items()):
        err = errors.get(lab) if errors else None
        ax.bar(x + (i - (k - 1) / 2) * width, vals, width=width * 0.92, color=SERIES[i], label=lab,
               yerr=err, ecolor=INK_2, capsize=2, error_kw={"elinewidth": 0.8})
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=k)
    _finish(fig, path, note)
