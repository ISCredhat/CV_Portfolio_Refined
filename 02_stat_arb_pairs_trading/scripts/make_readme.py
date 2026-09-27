#!/usr/bin/env python3
"""Fill README.template.md with numbers from results/ so the README never drifts from the code.

    python scripts/make_readme.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BAR = {"1min": "1 min", "5min": "5 min", "15min": "15 min", "30min": "30 min", "1h": "1 hour", "1D": "1 day"}
METHOD = {"static": "Static OLS", "rolling": "Rolling OLS", "kalman": "Kalman filter"}


def pct(x, d=1, sign=False):
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


def main() -> None:
    meta = json.loads((RES / "run_meta.json").read_text())
    dm = meta["data"]
    perf = pd.read_csv(RES / "performance.csv").set_index(["period", "series"])
    grid = pd.read_csv(RES / "grid_results.csv")
    bars = pd.read_csv(RES / "bar_size_study.csv").set_index("bar")
    lat = pd.read_csv(RES / "latency.csv")
    selm = pd.read_csv(RES / "selection_methods.csv")
    ts = pd.read_csv(RES / "trade_stats.csv").set_index("period")
    pers = pd.read_csv(RES / "cointegration_persistence.csv").set_index("eligible")
    yr = pd.read_csv(RES / "by_year.csv")
    ind = pd.read_csv(RES / "by_industry.csv")
    funnel = pd.read_csv(RES / "selection_funnel.csv")
    hedge = pd.read_csv(RES / "hedge_method_study.csv")
    n = meta["n_trials"]
    S, G, E, M = "Strategy (net)", "Strategy (gross)", f"Average of all {n} variants (net)", "SPY buy & hold"
    p = lambda per, s, k: perf.loc[(per, s), k]
    best = meta["best_config"]
    el, ot = "Eligible (passed all filters)", "Other candidates (random sample)"
    intraday = bars.loc[["1min", "5min", "15min", "30min", "1h"]]
    v = {
        "n_tickers": dm["n_tickers"], "n_minutes": f"{dm['n_minutes']:,}", "n_days": f"{dm['n_days']:,}",
        "first_day": dm["first_day"], "last_day": dm["last_day"], "n_winter": dm["n_days_session_to_1500"],
        "n_splits": dm["n_split_events"], "yahoo_bps": num(dm["yahoo_median_abs_diff_bps"], 1),
        "config": meta["config_label"], "n_trials": n,
        "val_sharpe": num(p("validation", S, "sharpe")), "oos_sharpe": num(p("oos", S, "sharpe")),
        "full_sharpe": num(p("full", S, "sharpe")),
        "oos_ci": f"[{p('oos', S, 'sharpe_ci_low'):.2f}, {p('oos', S, 'sharpe_ci_high'):.2f}]",
        "val_ci": f"[{p('validation', S, 'sharpe_ci_low'):.2f}, {p('validation', S, 'sharpe_ci_high'):.2f}]",
        "oos_gross": num(p("oos", G, "sharpe")), "val_gross": num(p("validation", G, "sharpe")),
        "val_cagr": pct(p("validation", S, "cagr")), "oos_cagr": pct(p("oos", S, "cagr")),
        "oos_dd": pct(p("oos", S, "max_dd"), 0), "full_dd": pct(p("full", S, "max_dd"), 0),
        "oos_beta": num(p("oos", S, "beta_spy")), "full_beta": num(p("full", S, "beta_spy")),
        "full_corr": num(perf.loc[("full", S), "corr_spy"]),
        "oos_alpha": pct(p("oos", S, "ff6_alpha_ann"), 1, True), "oos_alpha_t": num(p("oos", S, "ff6_alpha_t"), 1),
        "val_alpha": pct(p("validation", S, "ff6_alpha_ann"), 1, True), "val_alpha_t": num(p("validation", S, "ff6_alpha_t"), 1),
        "dsr": num(meta["deflated_sharpe_validation"]),
        "ens_val": num(p("validation", E, "sharpe")), "ens_oos": num(p("oos", E, "sharpe")),
        "spy_oos": num(p("oos", M, "sharpe")),
        "rank_corr": num(meta["rank_corr_val_oos"]),
        "share_pos_oos": pct((grid["oos_sharpe"] > 0).mean(), 0),
        "n_pos_oos": int((grid["oos_sharpe"] > 0).sum()),
        "n_pair_months": f"{meta['n_pair_months']:,}", "n_unc": f"{meta['n_unconstrained_pair_months']:,}",
        "share_p5": pct(meta["share_p5"]), "null_p5": pct(meta["null_share_p5"]),
        "share_p1": pct(meta["share_p1"]), "null_p1": pct(meta["null_share_p1"]),
        "n_bh": meta["n_bh"], "n_bh_unc": meta["n_bh_unconstrained"],
        "avg_pairs": num(meta["avg_pairs_per_month"], 1), "util": pct(meta["capital_utilisation"], 0),
        "months_none": meta["months_with_no_pair"], "n_months": meta["n_months"],
        "pers_el": pct(pers.loc[el, "share_p5_next"]), "pers_ot": pct(pers.loc[ot, "share_p5_next"]),
        "n_pers_el": int(pers.loc[el, "n"]),
        "hl_form": num(pers.loc[el, "median_hl_form"] / 6.5, 1), "hl_next": num(pers.loc[el, "median_hl_next"] / 6.5, 1),
        "g1m": num(bars.loc["1min", "full_gross_sharpe_avg"]), "n1m": num(bars.loc["1min", "full_net_sharpe_avg"]),
        "g1d": num(bars.loc["1D", "full_gross_sharpe_avg"]), "n1d": num(bars.loc["1D", "full_net_sharpe_avg"]),
        "g_intra_max": num(intraday["full_gross_sharpe_avg"].max()),
        "bps1m": num(bars.loc["1min", "gross_bps_per_trade"], 1), "cost1m": num(bars.loc["1min", "cost_bps_per_trade"], 1),
        "tpm1m": num(bars.loc["1min", "trades_per_month"], 1),
        "leg_cost": num(meta["avg_leg_cost_bps"], 1),
        "be_full": num(meta["break_even_multiplier_full"], 1), "be_oos": num(meta["break_even_multiplier_oos"], 1),
        "be_full_bps": num(meta["break_even_multiplier_full"] * meta["avg_leg_cost_bps"], 0),
        "be_oos_bps": num(meta["break_even_multiplier_oos"] * meta["avg_leg_cost_bps"], 0),
        "lat0": num(lat.iloc[0]["full_net_sharpe"]), "lat1": num(lat.iloc[1]["full_net_sharpe"]),
        "latbar": num(lat.iloc[-1]["full_net_sharpe"]),
        "placebo_full": pct(meta["placebo_pct_full"], 0), "placebo_oos": pct(meta["placebo_pct_oos"], 0),
        "n_placebo": meta["n_placebo"],
        "rnd_oos": num(selm.iloc[3]["oos"]), "dist_oos": num(selm.iloc[1]["oos"]),
        "unc_oos": num(selm.iloc[2]["oos"]), "unc_val": num(selm.iloc[2]["validation"]),
        "n_trades": int(ts.loc["full", "n_trades"]), "tpm": num(ts.loc["full", "trades_per_month"], 1),
        "win": pct(ts.loc["full", "win_rate"], 0), "hold_days": num(ts.loc["full", "median_hold_hours"] / 24, 1),
        "gross_bps": num(ts.loc["full", "gross_bps_per_trade"], 0), "cost_bps": num(ts.loc["full", "cost_bps_per_trade"], 0),
        "gross_bps_val": num(ts.loc["validation", "gross_bps_per_trade"], 0),
        "gross_bps_oos": num(ts.loc["oos", "gross_bps_per_trade"], 0),
        "exit_rev": pct(ts.loc["full", "exit_reversion"], 0), "exit_time": pct(ts.loc["full", "exit_time_stop"], 0),
        "exit_me": pct(ts.loc["full", "exit_month_end"], 0),
        "exit_stop": pct(ts.loc["full"]["exit_stop_loss"] if "exit_stop_loss" in ts.columns and pd.notna(ts.loc["full"]["exit_stop_loss"]) else 0.0, 0),
        "bar": BAR[best["bar"]], "method": METHOD[best["method"]], "entry": f"{best['entry']:g}", "exit": f"{best['exit']:g}",
    }
    semi = ind.set_index("group").loc["Semiconductors"] if "Semiconductors" in set(ind["group"]) else None
    v["semi_val"] = pct(semi["net_pnl_pct_of_capital_validation"] / 100, 1, True) if semi is not None else "n/a"
    v["semi_oos"] = pct(semi["net_pnl_pct_of_capital_oos"] / 100, 1, True) if semi is not None else "n/a"
    y = yr.set_index("year")
    v["yr_list"] = ", ".join(f"{int(k)} {r['strategy_net_return'] * 100:+.1f}%" for k, r in y.iterrows())

    # ---- tables
    rows = []
    for per, lab in [("validation", "Validation (2021-05 to 2022-12)"), ("oos", "**Out of sample (2023-01 to 2025-10)**"),
                     ("full", "Full walk-forward")]:
        for s, sl in [(S, "Strategy, net"), (G, "Strategy, gross"), (E, f"Average of all {n} variants, net"), (M, "SPY buy & hold")]:
            r = perf.loc[(per, s)]
            rows.append({"Period": lab if s == S else "", "Series": sl, "CAGR": pct(r["cagr"]), "Vol": pct(r["vol"], 0),
                         "Sharpe": num(r["sharpe"]), "Sharpe 95% CI": f"[{r['sharpe_ci_low']:.2f}, {r['sharpe_ci_high']:.2f}]",
                         "Max DD": pct(r["max_dd"], 0),
                         "FF5+Mom alpha p.a. (t)": f"{r['ff6_alpha_ann'] * 100:+.1f}% ({r['ff6_alpha_t']:.1f})" if s != M else "",
                         "Beta to SPY": num(r["beta_spy"])})
    v["table_perf"] = md(pd.DataFrame(rows))
    v["table_funnel"] = md(pd.DataFrame({"Step": funnel["step"], "Pair-months": funnel["count"].map(lambda x: f"{x:,}")}))
    g2 = grid.sort_values("validation_sharpe", ascending=False).head(8)
    v["table_top"] = md(pd.DataFrame({"Bar": g2["bar"].map(BAR), "Hedge": g2["method"].map(METHOD),
                                      "Entry z": g2["entry"].map(lambda x: f"{x:g}"), "Exit z": g2["exit"].map(lambda x: f"{x:g}"),
                                      "Validation Sharpe": g2["validation_sharpe"].map(num),
                                      "OOS Sharpe": g2["oos_sharpe"].map(num), "Trades": g2["n_trades"]}))
    b = bars.reset_index()
    v["table_bars"] = md(pd.DataFrame({
        "Bar size": b["bar"].map(BAR),
        "Gross Sharpe (avg of 18)": b["full_gross_sharpe_avg"].map(num), "Net Sharpe (avg of 18)": b["full_net_sharpe_avg"].map(num),
        "OOS net (avg of 18)": b["oos_net_sharpe_avg"].map(num),
        "Trades / month*": b["trades_per_month"].map(lambda x: num(x, 1)),
        "Gross bps / trade*": b["gross_bps_per_trade"].map(lambda x: num(x, 1)),
        "Cost bps / trade*": b["cost_bps_per_trade"].map(lambda x: num(x, 1)),
        "Break-even cost / leg*": b["break_even_leg_cost_bps"].map(lambda x: "none" if pd.isna(x) else f"{x:.0f} bps")}))
    h = hedge[hedge["bar"] == best["bar"]].set_index("method")
    ha = hedge.groupby("method")[["validation_sharpe", "oos_sharpe", "oos_gross_sharpe"]].mean()
    v["table_hedge"] = md(pd.DataFrame({
        "Hedge ratio": [METHOD[m] for m in ha.index],
        "Validation net (all bars)": ha["validation_sharpe"].map(num).values,
        "OOS net (all bars)": ha["oos_sharpe"].map(num).values, "OOS gross (all bars)": ha["oos_gross_sharpe"].map(num).values,
        f"OOS net ({BAR[best['bar']]} bars)": [num(h.loc[m, "oos_sharpe"]) if m in h.index else "" for m in ha.index]}))
    v["table_latency"] = md(pd.DataFrame({"Order filled": lat["fill"], "Net Sharpe (full)": lat["full_net_sharpe"].map(num),
                                          "Gross Sharpe (full)": lat["full_gross_sharpe"].map(num),
                                          "Net Sharpe (OOS)": lat["oos_net_sharpe"].map(num)}))
    v["table_selection"] = md(pd.DataFrame({"Pair selection (same trading rules)": selm["selection"],
                                            "Validation": selm["validation"].map(num), "Out of sample": selm["oos"].map(num),
                                            "Full sample": selm["full"].map(num)}))
    v["table_years"] = md(pd.DataFrame({"Year": [f"{int(k)} ({m})" for k, m in zip(y.index, y["months"])],
                                        "Strategy net": y["strategy_net_return"].map(lambda x: pct(x, 1, True)),
                                        "Sharpe": y["strategy_net_sharpe"].map(num),
                                        f"All {n} variants net": y["all_variants_net_return"].map(lambda x: pct(x, 1, True)),
                                        "SPY": y["spy_return"].map(lambda x: pct(x, 1, True))}))
    pers_t = pers.reset_index()
    v["table_persist"] = md(pd.DataFrame({"Formation-window result": pers_t["eligible"], "Pairs re-tested": pers_t["n"],
                                          "Still p <= 5% next 6 months": pers_t["share_p5_next"].map(pct),
                                          "Median half-life then -> next (days)":
                                              [f"{a / 6.5:.1f} -> {b_ / 6.5:.1f}" for a, b_ in zip(pers_t["median_hl_form"], pers_t["median_hl_next"])]}))
    ind2 = ind.copy()
    v["table_industry"] = md(pd.DataFrame({"Industry": ind2["group"],
                                           "Trades (val / OOS)": [f"{int(a)} / {int(b_)}" for a, b_ in zip(ind2["trades_validation"], ind2["trades_oos"])],
                                           "Net P&L, validation": ind2["net_pnl_pct_of_capital_validation"].map(lambda x: f"{x:+.1f}%"),
                                           "Net P&L, OOS": ind2["net_pnl_pct_of_capital_oos"].map(lambda x: f"{x:+.1f}%")}))
    tmpl = (ROOT / "README.template.md").read_text(encoding="utf-8")
    missing = set(re.findall(r"\{\{(\w+)\}\}", tmpl)) - set(v)
    if missing:
        sys.exit(f"template placeholders without values: {sorted(missing)}")
    (ROOT / "README.md").write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: str(v[m.group(1)]), tmpl), encoding="utf-8")
    print("README.md written")


if __name__ == "__main__":
    main()
