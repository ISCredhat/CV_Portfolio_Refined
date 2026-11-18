"""Aggregate outputs of stage 2 task C (no market ids, no wallet addresses)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.stage2 import forecast as F


def _f(x, nd=5):
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}"


def write_outputs(res: dict, log: Callable = F._noop, results: Path = F.RESULTS) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(res["rows"], infer_schema_length=None)
    cal = pl.DataFrame(res["calib"], infer_schema_length=None) if res["calib"] else pl.DataFrame()
    grp = pl.DataFrame(res["groups"], infer_schema_length=None) if res["groups"] else pl.DataFrame()
    parts = [df.with_columns(pl.lit("improvement").alias("table"), pl.lit("").alias("group"), pl.lit(None, pl.Int64).alias("bin"))]
    if grp.height:
        parts.append(grp.with_columns(pl.lit("by_info_type").alias("table"), pl.lit("pooled").alias("sample"), pl.lit(None, pl.Int64).alias("bin")))
    if cal.height:
        parts.append(
            cal.with_columns(pl.lit("calibration_price").alias("table"), pl.lit("pooled").alias("sample"), pl.lit("price").alias("model"), pl.lit("").alias("group"))
        )
    pl.concat(parts, how="diagonal_relaxed").write_csv(results / "stage2_forecast.csv")
    (results / "stage2_forecast.md").write_text(_markdown(res, df, cal, grp), encoding="utf-8")
    _figure(res, df, cal, results / "figures" / "stage2_forecast.png")
    log("wrote results/stage2_forecast.csv, .md and figures/stage2_forecast.png")


def _row(df, h, model, sample):
    r = df.filter((pl.col("H") == h) & (pl.col("model") == model) & (pl.col("sample") == sample))
    return r.to_dicts()[0] if r.height else None


def _markdown(res, df, cal, grp) -> str:
    L = ["# Stage 2, task C: forecasting benchmark against the market price (the stop gate)", "",
         "Aggregate results only. Frozen rule (docs/STAGE2_PLAN.md section 9): a model's out-of-sample log-loss below "
         "the price benchmark in BOTH halves of the tune period (end_ts Jan-Mar and Apr-Jun 2026) with the lower 95% "
         "event-cluster bootstrap bound of the improvement above zero, for at least one horizon. The sealed period was "
         "never loaded.", "",
         f"## Verdict: {res['verdict']}", "",
         "Improvement = mean(price loss - model loss); positive means the model beats the price. Prices clipped to "
         "[0.001, 0.999].", "",
         "## Sample", "", "| horizon | all snapshots (>= 20 trades, open at t) | test rows | markets | events | Jan-Mar | Apr-Jun | YES rate |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for c in res["counts"]:
        L.append(f"| {c['H']} | {c['snapshots_all']:,} | {c['test_rows']:,} | {c['test_markets']:,} | {c['test_events']:,} | {c['JanMar']:,} | {c['AprJun']:,} | {100 * c['yes_rate_test']:.1f}% |")
    L += ["", "## Out-of-sample log-loss and Brier: price vs models, with improvement and 95% event-cluster interval", "",
          "| horizon | model | sample | n | price LL | model LL | LL improvement [95%] | Brier improvement [95%] |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for h in F.HORIZONS_D:
        for mod in ("recal", "hgb_all"):
            for s in ("JanMar", "AprJun", "pooled"):
                r = _row(df, h, mod, s)
                if r:
                    L.append(f"| {h} | {mod} | {s} | {r['n']:,} | {r['logloss_price']:.4f} | {r['logloss_model']:.4f} | {_f(r['dll'])} [{_f(r['dll_lo'])}, {_f(r['dll_hi'])}] | {_f(r['dbr'])} [{_f(r['dbr_lo'])}, {_f(r['dbr_hi'])}] |")
    L += ["", "## Pass rule by horizon and model", "", "| horizon | recalibration passes | feature model (HGB) passes |", "| --- | --- | --- |"]
    for h in F.HORIZONS_D:
        L.append(f"| {h} | {'PASS' if res['passes'].get((h, 'recal')) else 'fail'} | {'PASS' if res['passes'].get((h, 'hgb_all')) else 'fail'} |")
    L += ["", "Six horizon-by-model comparisons are made, uncorrected for multiplicity (the frozen rule is 'at least one "
          "horizon'); a marginal pass should be read with that in mind. A pass by the price-only recalibration "
          "would show calibration, not information beyond the price.", "",
          "## Feature-group ablation (pooled test period; log-loss improvement vs price, HGB depth 3 / 200 trees / lr 0.05)", "",
          "'plus' = logit price plus that group only; 'minus' = all features without that group.", "",
          "| horizon | model | LL improvement [95%] | Brier improvement |", "| --- | --- | --- | --- |"]  # fmt: skip
    names = ["hgb_price_only", *[f"hgb_plus_{g}" for g in F.GROUPS], *[f"hgb_minus_{g}" for g in F.GROUPS], "hgb_all"]
    for h in F.HORIZONS_D:
        for mod in names:
            r = _row(df, h, mod, "pooled")
            if r:
                L.append(f"| {h} | {mod} | {_f(r['dll'])} [{_f(r['dll_lo'])}, {_f(r['dll_hi'])}] | {_f(r['dbr'])} |")
    L += ["", "## Stress test: dropping the largest 10% of test markets by volume to date", "",
          "| horizon | model | sample | LL improvement [95%] |", "| --- | --- | --- | --- |"]  # fmt: skip
    for h in F.HORIZONS_D:
        for mod in ("recal", "hgb_all"):
            for s in ("JanMar_drop_top10pct_volume", "AprJun_drop_top10pct_volume", "pooled_drop_top10pct_volume"):
                r = _row(df, h, mod, s)
                if r:
                    L.append(f"| {h} | {mod} | {s.replace('_drop_top10pct_volume', '')} | {_f(r['dll'])} [{_f(r['dll_lo'])}, {_f(r['dll_hi'])}] (n {r['n']:,}) |")
    if grp.height:
        L += ["", "## By information type (pooled test period)", "", "| horizon | info type | model | n | LL improvement [95%] |", "| --- | --- | --- | --- | --- |"]
        for r in grp.to_dicts():
            L.append(f"| {r['H']} | {r['group']} | {r['model']} | {r['n']:,} | {_f(r['dll'])} [{_f(r['dll_lo'])}, {_f(r['dll_hi'])}] |")
    if cal.height:
        L += ["", "## Calibration of the price benchmark (pooled test rows, quantile bins)", "", "| horizon | bin | n | mean price | outcome rate |", "| --- | --- | --- | --- | --- |"]
        for r in cal.to_dicts():
            L.append(f"| {r['H']} | {r['bin'] + 1} | {r['n']:,} | {r['mean_price']:.3f} | {r['outcome_rate']:.3f} |")
    L += ["", "## Construction and caveats", "",
          "- Snapshot t = end_ts - H; markets with closed_ts <= t, fewer than 20 taker trades before t, or an unknown YES/NO outcome are excluded. "
          "Selection on 'still open at t' uses only information knowable at t.",
          "- Test rows: market closed in the tune period (closed_ts 2026-01-01 to 2026-06-30), end_ts month Jan to Jun 2026, t >= 2026-01-01. Training rows for test month M: "
          "closed_ts at least 7 days before the start of M (labels known), fit period and earlier tune months. Markets whose end_ts falls in the tune window but that "
          "closed after 2026-06-30 are sealed and excluded, which selects on survival to the cut-off.",
          "- Wallet-skill features: fit-period 24 h markout classification (analytic sign-randomisation p with Benjamini-Hochberg 5% and EB shrunk skill > 0); for fit-period training rows "
          "the classification is cross-fitted by event so a market never feeds its own feature. The classification uses markouts of other markets that can be later than t "
          "for training rows only (never for test rows).",
          "- 'Buying wallets' in the Herfindahl are the wallets pushing the YES price up (buy Yes or sell No); the original BUY/SELL flag is not used because the Yes/No sides are folded to YES terms.",
          "- Hours to end equals H by construction (a constant within a horizon).",
          "- Price history is the stored winner-token history plus taker fills; no order-book data, so the price is a last-trade price, not a mid.", ""]  # fmt: skip
    return "\n".join(L)


def _figure(res, df, cal, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    samples = ["JanMar", "AprJun", "pooled"]
    cols = {"recal": "#4c72b0", "hgb_all": "#dd8452"}
    k = 0
    xt, xl = [], []
    for i, h in enumerate(F.HORIZONS_D):
        for j, s in enumerate(samples):
            xpos = i * 4 + j
            xt.append(xpos)
            xl.append(f"{h}\n{s}")
            for mi, mod in enumerate(("recal", "hgb_all")):
                r = _row(df, h, mod, s)
                if r:
                    ax[0].errorbar(xpos + 0.15 * (mi - 0.5) * 2, r["dll"], yerr=[[r["dll"] - r["dll_lo"]], [r["dll_hi"] - r["dll"]]], fmt="o", color=cols[mod], capsize=2, label=mod if k < 2 else None)
                    k += 1
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(xt, xl, fontsize=6)
    ax[0].set_ylabel("log-loss improvement vs price")
    ax[0].set_title("Out-of-sample improvement (95% event-cluster CI)")
    ax[0].legend(fontsize=7)
    if cal.height:
        for h, c in zip(F.HORIZONS_D, ("#4c72b0", "#dd8452", "#55a868"), strict=False):
            sub = cal.filter(pl.col("H") == h)
            ax[1].plot(sub["mean_price"], sub["outcome_rate"], "o-", color=c, label=h, ms=4)
        ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[1].set_xlabel("mean price in bin")
        ax[1].set_ylabel("YES frequency")
        ax[1].set_title("Calibration of the price benchmark")
        ax[1].legend(fontsize=7)
    names = ["hgb_price_only", *[f"hgb_plus_{g}" for g in F.GROUPS], "hgb_all"]
    w = 0.25
    for i, h in enumerate(F.HORIZONS_D):
        y = [(_row(df, h, n, "pooled") or {"dll": np.nan})["dll"] for n in names]
        ax[2].bar(np.arange(len(names)) + (i - 1) * w, y, w, label=h)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(np.arange(len(names)), [n.replace("hgb_", "").replace("plus_", "+") for n in names], rotation=30, fontsize=7)
    ax[2].set_title("Feature-group ablation (pooled LL improvement)")
    ax[2].legend(fontsize=7)
    fig.suptitle(res["verdict"][:120], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
