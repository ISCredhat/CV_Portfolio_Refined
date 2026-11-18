"""Descriptive tables, aggregate outputs and the figure for stage 2 task B (no wallet-level rows here)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.stage2 import markouts as M
from pmdetect.stage2.common import SEED


def descriptive(markets: pl.DataFrame, n_w: int, n_events: int, n_boot: int, log: Callable = M._noop) -> list[dict]:
    """Pooled size-weighted markout per dollar by horizon and period (gross and spread-adjusted) with
    event-cluster bootstrap intervals, share of wallets (>= 10 trades) with positive skill, share of
    markouts exactly zero (no later observation), and the share of trades kept for the horizon."""
    scan = pl.scan_parquet(str(M.PARTS_DIR / "part_*.parquet"))
    totals = scan.group_by("period").agg(pl.len()).collect()
    total = {r["period"]: r["len"] for r in totals.to_dicts()}
    rows = []
    for h in M.HORIZON_COLS:
        df = M.load_trade_arrays(h)
        for period in ("fit", "tune"):
            sub = df.filter(pl.col("period") == period)
            d = M._arrays(sub, h)
            _, e = np.unique(d["e"], return_inverse=True)
            ne = int(e.max()) + 1
            rng = np.random.default_rng(SEED)
            counts = M.bootstrap_counts(ne, n_boot, rng)
            g = d["q"] * d["a"]
            gross = M.pooled_ratio_ci(g, d["usd"], e, ne, counts)
            adj = M.pooled_ratio_ci(g - d["hs_usd"], d["usd"], e, ne, counts)
            st = M.wallet_period_stats(d, n_w)
            ok = st["n"] >= M.MIN_EB
            skill = st["num"][ok] / st["den"][ok]
            dd = sub[f"d_{h}"].to_numpy()
            rows.append(
                {
                    "horizon": h, "period": period, "n_trades": int(sub.height),
                    "kept_share": sub.height / total[period],
                    "markets": int(sub["mcode"].n_unique()),
                    "gross": gross[0], "gross_lo": gross[1], "gross_hi": gross[2],
                    "adj": adj[0], "adj_lo": adj[1], "adj_hi": adj[2],
                    "wallets_ge10": int(ok.sum()),
                    "share_wallets_positive": float((skill > 0).mean()) if skill.size else float("nan"),
                    "median_wallet_skill": float(np.median(skill)) if skill.size else float("nan"),
                    "zero_share": float((dd == 0).mean()) if dd.size else float("nan"),
                }
            )  # fmt: skip
            log(f"  descriptive {h} {period}: gross {gross[0]:+.5f} [{gross[1]:+.5f}, {gross[2]:+.5f}]")
        del df
    return rows


def _fmt(x: float, nd: int = 5) -> str:
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def write_outputs(res: dict, log: Callable = M._noop, results: Path = M.RESULTS) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def add(section, period, horizon, metric, value, lo=None, hi=None, n=None):
        rows.append({"section": section, "period": period, "horizon": horizon, "metric": metric,
                     "value": value, "lo95": lo, "hi95": hi, "n": n})  # fmt: skip

    for r in res["desc"]:
        h, p = r["horizon"], r["period"]
        add("pooled_markout_per_dollar", p, h, "gross", r["gross"], r["gross_lo"], r["gross_hi"], r["n_trades"])
        add("pooled_markout_per_dollar", p, h, "spread_adjusted", r["adj"], r["adj_lo"], r["adj_hi"], r["n_trades"])
        add("coverage", p, h, "kept_share", r["kept_share"], n=r["n_trades"])
        add("coverage", p, h, "zero_markout_share", r["zero_share"], n=r["n_trades"])
        add("wallets", p, h, "share_positive_skill_wallets_ge10", r["share_wallets_positive"], n=r["wallets_ge10"])
        add("wallets", p, h, "median_wallet_skill", r["median_wallet_skill"], n=r["wallets_ge10"])
    for p, c in res["counts"].items():
        if isinstance(c, dict) and "trades" in c and "skilled" in c:
            for k in ("trades", "wallets", "wallets_ge10", "markets", "events", "skilled"):
                add("counts_24h", p, "24h", k, c[k])
            add("eb", p, "24h", "prior_mean", c["mu"])
            add("eb", p, "24h", "prior_tau2", c["tau2"])
            add("skilled_share", p, "24h", "skilled_share_MC_BH", c["skilled"] / max(1, c["wallets_ge10"]),
                n=c["wallets_ge10"])  # fmt: skip
            add("skilled_share", p, "24h", "skilled_share_analytic_BH",
                c["skilled_analytic"] / max(1, c["wallets_ge10"]), n=c["wallets_ge10"])  # fmt: skip
            add("skilled_share", p, "24h", "min_possible_MC_p", c["min_mc_p"])
    ret = res["retention"]
    for k, v in ret.items():
        add("retention", "fit->tune", "24h", k, v)
    per = res["persistence"]
    add("persistence", "tune", "24h", "n_eligible_wallets", per.get("n_elig"))
    add("persistence", "tune", "24h", "embargo_markets_dropped", res["counts"]["embargo_markets_dropped"])
    if "deciles" in per:
        for d in per["deciles"]:
            add("persistence_decile_gross", "tune", "24h", f"fit_decile_{d['decile']:02d}", d["mean"], d["lo"], d["hi"], d["n_wallets"])
            add("persistence_decile_spread_adj", "tune", "24h", f"fit_decile_{d['decile']:02d}", d["mean_adj"], d["lo_adj"], d["hi_adj"], d["n_wallets"])
        a = per["all"]
        add("persistence_decile_gross", "tune", "24h", "all_eligible", a["mean"], a["lo"], a["hi"], per["n_elig"])
        add("persistence_decile_spread_adj", "tune", "24h", "all_eligible", a["mean_adj"], a["lo_adj"], a["hi_adj"], per["n_elig"])
        t = per["top_minus_bottom"]
        add("persistence", "tune", "24h", "top_minus_bottom_decile", t["mean"], t["lo"], t["hi"])
        add("persistence", "fit->tune", "24h", "spearman_fit_shrunk_vs_tune_skill", per["rho"], per["rho_lo"], per["rho_hi"], per["n_elig"])
        add("placebo", "tune", "24h", "shuffled_top_decile_mean", per["base_top_mean"], hi=per["base_top_p95"], n=len(per["base_top"]))
        add("placebo", "tune", "24h", "shuffled_spearman_p95", per["base_rho_p95"], n=len(per["base_rho"]))
        for k, v in per["concentration"].items():
            add("concentration", "tune", "24h", k, v)
        for k, v in per["checks"].items():
            add("pass_rule", "tune", "24h", k, int(v))
        add("pass_rule", "tune", "24h", "PASS", int(per["pass"]))
    pl.DataFrame(rows, schema_overrides={"n": pl.Float64, "lo95": pl.Float64, "hi95": pl.Float64}).write_csv(
        results / "stage2_markouts.csv"
    )
    (results / "stage2_markouts.md").write_text(_markdown(res), encoding="utf-8")
    _figure(res, results / "figures" / "stage2_markouts.png")
    log("wrote results/stage2_markouts.csv, .md and figures/stage2_markouts.png")


def _markdown(res: dict) -> str:
    per, ret, c = res["persistence"], res["retention"], res["counts"]
    L = ["# Stage 2, task B: taker-trade markouts and wallet skill v2", "",
         "Aggregate results only (no wallet addresses). Pre-registered pass rule: docs/STAGE2_PLAN.md section 9. "
         "Periods by market `closed_ts`: fit <= 2025-12-31, tune 2026-01-01 to 2026-06-30; the sealed period was "
         "never loaded. Wallets are described as 'anomalous' or 'skilled', never as having acted on non-public "
         "information.", "",
         f"## Verdict: {res['verdict']}", ""]  # fmt: skip
    for p in ("fit", "tune"):
        k = c[p]
        L.append(f"- {p}: {k['trades']:,} trades with a 24 h markout, {k['wallets']:,} wallets ({k['wallets_ge10']:,} with >= 10), "
                 f"{k['markets']:,} markets, {k['events']:,} events.")  # fmt: skip
    L += ["", "## Pre-registered pass rule (24 h markout per dollar, tune period)", "",
          "| check | value | result |", "| --- | --- | --- |"]  # fmt: skip
    if "deciles" in per:
        top, bot = per["deciles"][9], per["deciles"][0]
        ch = per["checks"]
        L += [
            f"| fit-top-decile tune mean, lower 95% bound > 0 | {_fmt(top['mean'])} [{_fmt(top['lo'])}, {_fmt(top['hi'])}] | {'pass' if ch['top_decile_gt0_lower_bound'] else 'FAIL'} |",
            f"| above fit-bottom decile | top {_fmt(top['mean'])} vs bottom {_fmt(bot['mean'])} (diff {_fmt(per['top_minus_bottom']['mean'])} [{_fmt(per['top_minus_bottom']['lo'])}, {_fmt(per['top_minus_bottom']['hi'])}]) | {'pass' if ch['top_above_bottom'] else 'FAIL'} |",
            f"| Spearman fit-shrunk vs tune skill, lower 95% bound > 0 | {_fmt(per['rho'], 4)} [{_fmt(per['rho_lo'], 4)}, {_fmt(per['rho_hi'], 4)}] | {'pass' if ch['rank_corr_lower_bound_gt0'] else 'FAIL'} |",
            f"| shuffled-wallet baseline 95th pct below observed top decile | baseline p95 {_fmt(per['base_top_p95'])} (mean {_fmt(per['base_top_mean'])}, {len(per['base_top'])} shuffles) vs observed {_fmt(top['mean'])} | {'pass' if ch['baseline_p95_below_observed'] else 'FAIL'} |",
            "",
            f"Eligible wallets (>= 20 trades in each period after the 7-day embargo): {per['n_elig']:,}; "
            f"{c['embargo_markets_dropped']:,} markets overlapping the boundary dropped; trades used: fit "
            f"{c['persistence_trades']['fit']:,}, tune {c['persistence_trades']['tune']:,}. The bootstrap resamples tune-period events "
            "(1,000 resamples, seed 20261113); the fit classification is held fixed.", "",
            "Concentration check of the top decile (tune period): largest wallet "
            f"{100 * per['concentration']['top_decile_largest_wallet_usd_share']:.1f}% of its dollars, top five wallets "
            f"{100 * per['concentration']['top_decile_top5_wallet_usd_share']:.1f}%; equal-weighted wallet mean "
            f"{_fmt(per['concentration']['top_decile_wallet_mean_equal_weight'])}, median "
            f"{_fmt(per['concentration']['top_decile_wallet_median'])} (bottom decile median "
            f"{_fmt(per['concentration']['bottom_decile_wallet_median'])}); share of wallets with positive tune skill "
            f"{100 * per['concentration']['top_decile_share_wallets_positive']:.0f}% in the top vs "
            f"{100 * per['concentration']['bottom_decile_share_wallets_positive']:.0f}% in the bottom decile.", "",
            "### Tune-period 24 h markout per dollar by fit-period decile (gross, then spread-adjusted)", "",
            "| fit decile | wallets | gross | 95% interval | spread-adjusted | 95% interval |", "| --- | --- | --- | --- | --- | --- |",
        ]  # fmt: skip
        for d in per["deciles"]:
            L.append(f"| {d['decile']} | {d['n_wallets']:,} | {_fmt(d['mean'])} | [{_fmt(d['lo'])}, {_fmt(d['hi'])}] | {_fmt(d['mean_adj'])} | [{_fmt(d['lo_adj'])}, {_fmt(d['hi_adj'])}] |")
        a = per["all"]
        L.append(f"| all eligible | {per['n_elig']:,} | {_fmt(a['mean'])} | [{_fmt(a['lo'])}, {_fmt(a['hi'])}] | {_fmt(a['mean_adj'])} | [{_fmt(a['lo_adj'])}, {_fmt(a['hi_adj'])}] |")
    L += ["", "## Share of wallets classed skilled (descriptive; definitions differ from the literature)", "",
          "Skilled = empirical-Bayes shrunk 24 h skill > 0 AND the one-sided sign-randomisation p, adjusted by "
          "Benjamini-Hochberg across wallets with >= 10 trades, <= 5%. Published figures for comparison: 3.14% of "
          "1.72M wallets 'skilled winners', 44% of those retained out of sample (different test, different "
          "outcome measure, different universe).", "",
          "Two versions of the p-value: (i) the pre-registered Monte-Carlo sign-randomisation p with 2,000 draws, whose "
          "smallest possible value is 1/2001, so Benjamini-Hochberg over tens of thousands of wallets cannot reject anyone "
          "unless several hundred wallets sit at that floor (a resolution limit, not evidence of no skill); (ii) a normal p "
          "from the exact mean and variance of the same sign-shuffle null (no resolution floor), used for the retention "
          "figures and for the forecasting features.", "",
          "| period | wallets with >= 10 trades | skilled (MC p, 2,000 draws) | share | skilled (analytic p) | share |",
          "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for p in ("fit", "tune"):
        k = c[p]
        L.append(f"| {p} | {k['wallets_ge10']:,} | {k['skilled']:,} | {100 * k['skilled'] / max(1, k['wallets_ge10']):.2f}% | {k['skilled_analytic']:,} | {100 * k['skilled_analytic'] / max(1, k['wallets_ge10']):.2f}% |")
    L += ["",
          f"Retention (analytic classification): of {ret['fit_skilled_with_tune_ge10']:,} fit-skilled wallets with >= 10 tune trades, "
          f"{100 * ret['tune_positive_share']:.1f}% have a positive tune skill and "
          f"{100 * ret['tune_skilled_share']:.1f}% are again classed skilled in the tune period "
          f"(for reference {100 * ret['tune_skilled_share_all_wallets']:.2f}% of all tune wallets with >= 10 trades are classed skilled).", "",
          "## Markouts by horizon (pooled, size-weighted, per dollar of notional)", "",
          "Gross is side-signed in the traded token's price; spread-adjusted subtracts the half-spread proxy per "
          "trade. 'zero' = share of markouts that are exactly zero (no later observation before t+h). 5 min and 1 h "
          "are the pre-registered short horizons, 6 h the extra sign check, 24 h the headline, 'res' the final payoff.", "",
          "| horizon | period | trades | kept | gross [95%] | spread-adjusted [95%] | zero | wallets > 0 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for r in res["desc"]:
        L.append(f"| {r['horizon']} | {r['period']} | {r['n_trades']:,} | {100 * r['kept_share']:.1f}% | {_fmt(r['gross'])} [{_fmt(r['gross_lo'])}, {_fmt(r['gross_hi'])}] | {_fmt(r['adj'])} [{_fmt(r['adj_lo'])}, {_fmt(r['adj_hi'])}] | {100 * r['zero_share']:.1f}% | {100 * r['share_wallets_positive']:.1f}% |")
    L += ["", "## Construction and caveats", "",
          "- Price at t+h: last observation at or before t+h of the YES-equivalent series (all taker fills plus the stored "
          "winner-token price history converted to YES terms, observations after closed_ts removed). Horizon dropped "
          "(never padded) if t+h is after min(last observation, closed_ts) or the last observation is older than 6 h "
          "(h <= 1 h), 24 h (6 h) or 48 h (24 h).",
          "- Because the last observation can be the trade itself, short-horizon markouts contain many exact zeros; this "
          "dilutes them toward zero but does not bias the sign.",
          "- The sign-randomisation null conditions on which markets, prices and sign counts a wallet faced; the null mean "
          "is not zero (a market that drifts up rewards buyers in the null as well). Skill itself is the raw pre-registered "
          "statistic; the p-value is against the shuffled null.",
          "- A taker's own entry price already contains its half spread, so subtracting a further half spread "
          "(pre-registered) is conservative.",
          "- The persistence test uses the embargoed sets; the skilled share and the retention figures use all fit and "
          "tune trades without embargo.",
          "- Markets with no resolved YES/NO outcome are excluded; fills after a market's closed_ts are dropped.", ""]  # fmt: skip
    return "\n".join(L)


def _figure(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = res["persistence"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # panel 1: pooled markout by horizon
    hs = M.HORIZON_COLS
    x = np.arange(len(hs))
    for k, (period, col) in enumerate((("fit", "#4c72b0"), ("tune", "#dd8452"))):
        rr = {r["horizon"]: r for r in res["desc"] if r["period"] == period}
        y = np.array([rr[h]["gross"] for h in hs])
        lo = np.array([rr[h]["gross_lo"] for h in hs])
        hi = np.array([rr[h]["gross_hi"] for h in hs])
        ya = np.array([rr[h]["adj"] for h in hs])
        ax[0].errorbar(x + 0.12 * (k - 0.5) * 2, y, yerr=[y - lo, hi - y], fmt="o", color=col, label=f"{period} gross", capsize=3)
        ax[0].plot(x + 0.12 * (k - 0.5) * 2, ya, "x", color=col, label=f"{period} spread-adjusted")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(x, hs)
    ax[0].set_ylabel("pooled markout per dollar")
    ax[0].set_title("Taker markouts by horizon")
    ax[0].legend(fontsize=7)
    # panel 2: tune markout by fit decile
    if "deciles" in per:
        d = per["deciles"]
        xs = np.arange(1, 11)
        y = np.array([r["mean"] for r in d])
        lo = np.array([r["lo"] for r in d])
        hi = np.array([r["hi"] for r in d])
        ax[1].errorbar(xs, y, yerr=[y - lo, hi - y], fmt="o", capsize=3, color="#4c72b0", label="observed")
        ax[1].plot(xs, [r["mean_adj"] for r in d], "x", color="#c44e52", label="spread-adjusted")
        ax[1].axhline(per["base_top_p95"], color="grey", ls="--", lw=1, label="shuffled top-decile p95")
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_xlabel("fit-period skill decile (10 = top)")
        ax[1].set_ylabel("tune 24 h markout per dollar")
        ax[1].set_title(f"Persistence: rho {per['rho']:+.3f} [{per['rho_lo']:+.3f}, {per['rho_hi']:+.3f}]")
        ax[1].legend(fontsize=7)
        # panel 3: shuffled baseline distribution vs observed
        ax[2].hist(per["base_top"], bins=25, color="#999999")
        ax[2].axvline(d[9]["mean"], color="#c44e52", lw=2, label="observed top decile")
        ax[2].axvline(per["base_top_p95"], color="k", ls="--", label="shuffled p95")
        ax[2].set_xlabel("tune 24 h markout per dollar of a random decile")
        ax[2].set_title("Shuffled-wallet baseline")
        ax[2].legend(fontsize=7)
    fig.suptitle(f"Stage 2 task B: {res['verdict']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
