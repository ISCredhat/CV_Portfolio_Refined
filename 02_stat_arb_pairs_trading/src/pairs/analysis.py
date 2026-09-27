"""Stage `analyse`: tables (results/*.csv), figures (results/figures/) and run_meta.json."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from . import backtest as B
from . import config as C
from . import hedge as HG  # noqa: F401  (documented dependency of the signal plots)
from . import pipeline as PL
from . import plots as P
from . import selection as S
from . import stats as ST

FF6 = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
BAR_LABEL = {"1min": "1 min", "5min": "5 min", "15min": "15 min", "30min": "30 min", "1h": "1 hour", "1D": "1 day"}
METHOD_LABEL = {"static": "Static OLS", "rolling": "Rolling OLS", "kalman": "Kalman filter"}


# ------------------------------------------------------------------------------ inputs
def load_market() -> tuple[pd.Series, pd.DataFrame]:
    y = pd.read_csv(C.DATA_RAW / "yahoo_daily.csv.gz", parse_dates=["date"])
    spy = y[y["ticker"] == "SPY"].set_index("date")["adj_close"].sort_index().pct_change()
    fac = pd.read_csv(C.DATA_RAW / "factors_daily.csv", parse_dates=["date"]).set_index("date")
    return spy.rename("SPY"), fac


def perf_row(r: pd.Series, fac: pd.DataFrame, spy: pd.Series, label: str, n_boot: int = 2000,
             already_excess: bool = True) -> dict:
    """Performance of a daily return series. Strategy P&L is booked on committed capital *excluding*
    interest on that capital (collateral would earn roughly the T-bill rate), so it is already an
    excess return: the risk-free rate is not subtracted again. For SPY it is."""
    r = r.dropna()
    rf = fac["rf"].reindex(r.index).fillna(0) * (0.0 if already_excess else 1.0)
    ps = ST.perf_stats(r, rf)
    lo, hi = ST.sharpe_bootstrap_ci(r - rf, n_boot=n_boot, block=C.BOOTSTRAP_BLOCK)
    ff = ST.factor_regression(r, fac.assign(rf=fac["rf"] * (0.0 if already_excess else 1.0)), FF6)
    both = pd.concat([r, spy], axis=1, join="inner").dropna()
    beta_spy = np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / both.iloc[:, 1].var() if len(both) > 20 else np.nan
    return {"series": label, **{k: ps[k] for k in ["start", "end", "years", "cagr", "vol", "sharpe", "sortino",
                                                     "max_dd", "hit_rate_daily", "skew", "kurt"]},
            "sharpe_ci_low": lo, "sharpe_ci_high": hi, "ff6_alpha_ann": ff["alpha_ann"], "ff6_alpha_t": ff["alpha_t"],
            "beta_mkt": ff["beta_mkt_rf"], "beta_spy": beta_spy, "corr_spy": both.corr().iloc[0, 1] if len(both) > 20 else np.nan}


def trade_stats(tr: pd.DataFrame, n_months: int | None = None) -> dict:
    """Trade statistics. P&L per trade is in bps of the position's gross notional
    (LEVERAGE / N_PAIRS of capital: long leg + short leg)."""
    if tr.empty:
        return {}
    notional = 2 * C.LEVERAGE / C.N_PAIRS                  # entry + exit, both legs
    net = tr["gross"] - tr["cost"] - tr["borrow"]
    hold_h = (tr["close"] - tr["open"]).dt.total_seconds() / 3600
    return {"n_trades": int(len(tr)), "trades_per_month": len(tr) / (n_months or tr["month"].nunique()),
            "win_rate": float((net > 0).mean()), "median_hold_hours": float(hold_h.median()),
            "gross_bps_per_trade": float(tr["gross"].mean() / (C.LEVERAGE / C.N_PAIRS) * 1e4),
            "cost_bps_per_trade": float(tr["cost"].mean() / (C.LEVERAGE / C.N_PAIRS) * 1e4),
            "avg_leg_cost_bps": float(tr["cost"].sum() / (len(tr) * notional) * 1e4),
            **{f"exit_{k.replace(' ', '_').replace('-', '_')}": float(v)
               for k, v in tr["reason"].value_counts(normalize=True).items()}}


def cost_curve(g: dict, col, mults=C.COST_MULTIPLIERS) -> pd.DataFrame:
    rows = []
    for m in mults:
        r = g["gross"][col] - m * g["cost"][col] - g["borrow"][col]
        for per, x in PL.split_periods(r).items():
            rows.append({"cost_multiplier": m, "period": per, "sharpe": PL.sharpe(x), "ann_return": x.mean() * 252})
    return pd.DataFrame(rows)


def break_even_multiplier(g: dict, col, period: str = "full") -> float:
    gr, co, bo = (PL.split_periods(g[k][col])[period].mean() for k in ["gross", "cost", "borrow"])
    return float((gr - bo) / co) if co > 0 and gr > bo else np.nan      # NaN: loses money even at zero cost


def null_rates(n_sims: int = 2000, n_obs: int = 1500, seed: int = 0) -> dict:
    """Share of pairs passing the selection test when there is NO cointegration: two independent random
    walks, tested exactly as real pairs are (both orderings, 2 x the smaller p-value). Cached."""
    path = C.DATA_PROC / "null_pvalue_rates.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(seed)
    p = np.array([S.test_pair(np.cumsum(rng.standard_normal(n_obs)) * 0.01,
                              np.cumsum(rng.standard_normal(n_obs)) * 0.01, 0.5)["p_eg"] for _ in range(n_sims)])
    out = {str(a): float((p <= a).mean()) for a in [0.01, 0.05, 0.10]}
    out["n_sims"] = n_sims
    path.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------------------ main
def run_all(ctx: PL.Context) -> None:
    out, fig = C.RESULTS, C.FIGURES
    fig.mkdir(parents=True, exist_ok=True)
    spy, fac = load_market()
    meta = json.loads((out / "data_meta.json").read_text())
    best = json.loads((C.DATA_PROC / "best_config.json").read_text())
    col = (best["method"], best["entry"], best["exit"])
    PL.log(f"Analysing chosen configuration {best}")

    # ---------------------------------------------------------------- selection
    cands = PL.load_candidates()
    allc = pd.concat(cands.values(), ignore_index=True)
    sel = PL.selections(cands)
    by_month = pd.DataFrame({
        "month": list(cands),
        "universe": [int(c["n_universe"].iloc[0]) if len(c) else 0 for c in cands.values()],
        "candidates": [len(c) for c in cands.values()],
        "p_below_5pct": [int((c["p_eg"] <= C.ALPHA).sum()) if len(c) else 0 for c in cands.values()],
        "bh_discoveries": [int(c["bh_pass"].sum()) if len(c) else 0 for c in cands.values()],
        "eligible": [int(c["eligible"].sum()) if len(c) else 0 for c in cands.values()],
        "selected": [len(sel[m]) for m in cands],
    })
    by_month.to_csv(out / "selection_by_month.csv", index=False)
    lo, hi = C.HALF_LIFE_HOURS
    p5 = allc["p_eg"] <= C.ALPHA
    funnel = pd.DataFrame({"step": ["Within-industry candidate pair-months", "Engle-Granger p <= 5% (both orderings, Bonferroni)",
                                    "... and Johansen trace test rejects r = 0", "... and half-life 2 h - 10 days",
                                    "... and Hurst < 0.5 and plausible hedge ratio (eligible)", "Selected (top 5 / month, ticker cap)",
                                    "Benjamini-Hochberg discoveries at 5% FDR (all candidates)"],
                           "count": [len(allc), int(p5.sum()), int((p5 & allc["johansen_pass"]).sum()),
                                     int((p5 & allc["johansen_pass"] & allc["hl_hours"].between(lo, hi)).sum()),
                                     int(allc["eligible"].sum()), int(sum(len(v) for v in sel.values())),
                                     int(allc["bh_pass"].sum())]})
    funnel.to_csv(out / "selection_funnel.csv", index=False)
    selected = pd.concat([v for v in sel.values() if len(v)], ignore_index=True)
    selected[["month", "y", "x", "group", "p_eg", "beta", "hl_hours", "hurst", "johansen", "bh_pass"]].to_csv(
        out / "selected_pairs.csv", index=False, float_format="%.4g")
    group_counts = selected["group"].value_counts()

    # ---------------------------------------------------------------- grid
    table = pd.read_pickle(C.DATA_PROC / "grid_table.pkl")
    table.to_csv(out / "grid_results.csv", index=False, float_format="%.4f")
    n_trials = len(table)
    rank_corr = table[["validation_sharpe", "oos_sharpe"]].corr(method="spearman").iloc[0, 1]
    table["ensemble_member"] = True

    # ---------------------------------------------------------------- headline performance
    g = PL.load_grid(best["bar"])
    net, gross = g["net"][col], g["gross"][col]
    ens = pd.concat([PL.load_grid(b)["net"] for b in C.BAR_SIZES], axis=1).mean(axis=1)
    rows = []
    for per in ["validation", "oos", "full"]:
        rows.append({"period": per, **perf_row(PL.split_periods(net)[per], fac, spy, "Strategy (net)")})
        rows.append({"period": per, **perf_row(PL.split_periods(gross)[per], fac, spy, "Strategy (gross)")})
        rows.append({"period": per, **perf_row(PL.split_periods(ens)[per], fac, spy, f"Average of all {n_trials} variants (net)")})
        rows.append({"period": per, **perf_row(PL.split_periods(spy.reindex(net.index))[per], fac, spy, "SPY buy & hold",
                                               already_excess=False)})
    perf = pd.DataFrame(rows)
    val_net = PL.split_periods(net)["validation"]
    vs = ST.perf_stats(val_net)
    dsr = ST.deflated_sharpe(vs["sharpe"], len(val_net), vs["skew"], vs["kurt"], n_trials,
                             table["validation_sharpe"].dropna().tolist())
    perf["deflated_sharpe_prob"] = np.where((perf["period"] == "validation") & (perf["series"] == "Strategy (net)"), dsr, np.nan)
    perf.to_csv(out / "performance.csv", index=False, float_format="%.4f")

    tr_all = g["trades"]
    tr = tr_all[(tr_all["method"] == col[0]) & (tr_all["entry"] == col[1]) & (tr_all["exit"] == col[2])].copy()
    tr["period"] = np.where(tr["open"] <= pd.Period(C.VALIDATION_END, freq="M").end_time, "validation", "oos")
    n_m = {"validation": int((pd.PeriodIndex(list(cands), freq="M") <= pd.Period(C.VALIDATION_END, freq="M")).sum())}
    n_m["full"] = len(cands)
    n_m["oos"] = n_m["full"] - n_m["validation"]
    ts = pd.DataFrame({per: trade_stats(tr[tr["period"] == per] if per != "full" else tr, n_m[per])
                       for per in ["validation", "oos", "full"]}).T
    ts.index.name = "period"
    ts.to_csv(out / "trade_stats.csv", float_format="%.4f")
    tr.to_csv(out / "trades_chosen_config.csv", index=False, float_format="%.6g")
    held = selected.groupby("month").size().reindex(list(cands), fill_value=0)
    # where did the P&L come from? by calendar year and by industry group
    yr = pd.DataFrame({"strategy_net": net, "strategy_gross": gross, "all_variants_net": ens, "spy": spy.reindex(net.index)})
    yearly = yr.groupby(yr.index.year).agg(lambda r: (1 + r.fillna(0)).prod() - 1).rename(columns=lambda c: c + "_return")
    yearly = yearly.join(yr.groupby(yr.index.year).agg(PL.sharpe).rename(columns=lambda c: c + "_sharpe"))
    yearly.index.name = "year"
    first, last = net.index[0], net.index[-1]
    yearly["months"] = [f"{max(first, pd.Timestamp(y, 1, 1)):%b}-{min(last, pd.Timestamp(y, 12, 31)):%b}" for y in yearly.index]
    yearly.to_csv(out / "by_year.csv", float_format="%.4f")
    tr["group"] = tr["y"].map(C.GROUP_OF)
    tr["net"] = tr["gross"] - tr["cost"] - tr["borrow"]
    by_group = tr.groupby(["group", "period"])["net"].agg(["count", "sum"]).unstack("period").fillna(0)
    by_group.columns = [f"{a}_{b}" for a, b in by_group.columns]
    by_group = by_group.rename(columns=lambda c: c.replace("count", "trades").replace("sum", "net_pnl"))
    for c_ in [c for c in by_group.columns if c.startswith("net_pnl")]:
        by_group[c_.replace("net_pnl", "net_pnl_pct_of_capital")] = by_group.pop(c_) * 100
    by_group.sort_values("net_pnl_pct_of_capital_oos").to_csv(out / "by_industry.csv", float_format="%.2f")
    util = float(held.mean() / C.N_PAIRS)

    # ---------------------------------------------------------------- bar size, hedge method, costs
    bs_rows = []
    for bar in C.BAR_SIZES:
        t = table[table["bar"] == bar]
        gb = PL.load_grid(bar)
        bc = (best["method"], best["entry"], best["exit"])
        be = break_even_multiplier(gb, bc, "full")
        trb = gb["trades"]
        trb = trb[(trb["method"] == bc[0]) & (trb["entry"] == bc[1]) & (trb["exit"] == bc[2])]
        tsb = trade_stats(trb, len(cands))
        bs_rows.append({"bar": bar, "oos_gross_sharpe_avg": t["oos_gross_sharpe"].mean(),
                        "oos_net_sharpe_avg": t["oos_sharpe"].mean(),
                        "full_gross_sharpe_avg": t["full_gross_sharpe"].mean(), "full_net_sharpe_avg": t["full_sharpe"].mean(),
                        "chosen_rules_full_gross": PL.sharpe(gb["gross"][bc]), "chosen_rules_full_net": PL.sharpe(gb["net"][bc]),
                        "chosen_rules_oos_net": PL.sharpe(PL.split_periods(gb["net"][bc])["oos"]),
                        "trades_per_month": tsb.get("trades_per_month", np.nan),
                        "gross_bps_per_trade": tsb.get("gross_bps_per_trade", np.nan),
                        "cost_bps_per_trade": tsb.get("cost_bps_per_trade", np.nan),
                        "break_even_cost_multiplier": be,
                        "break_even_leg_cost_bps": be * tsb.get("avg_leg_cost_bps", np.nan)})
    bar_tab = pd.DataFrame(bs_rows)
    bar_tab.to_csv(out / "bar_size_study.csv", index=False, float_format="%.4f")
    hedge_tab = (table.groupby(["bar", "method"])[["validation_sharpe", "oos_sharpe", "oos_gross_sharpe"]].mean()
                 .reset_index())
    hedge_tab.to_csv(out / "hedge_method_study.csv", index=False, float_format="%.4f")
    cc = cost_curve(g, col)
    cc.to_csv(out / "cost_sensitivity.csv", index=False, float_format="%.4f")
    be_mult = break_even_multiplier(g, col, "full")
    be_mult_oos = break_even_multiplier(g, col, "oos")
    avg_leg_cost = trade_stats(tr).get("avg_leg_cost_bps", np.nan)

    # ---------------------------------------------------------------- selection methods / placebo
    sm = pd.read_pickle(C.DATA_PROC / "selection_methods.pkl")
    sm_rows = [{"selection": "Cointegration, within industry (strategy)", "full": PL.sharpe(net),
                "oos": PL.sharpe(PL.split_periods(net)["oos"]), "validation": PL.sharpe(PL.split_periods(net)["validation"])}]
    for k, lab in [("distance", "Distance / SSD (Gatev et al.)"), ("unconstrained", "Cointegration, any industry")]:
        r = sm[k]["net"].iloc[:, 0]
        sm_rows.append({"selection": lab, "full": PL.sharpe(r), "oos": PL.sharpe(PL.split_periods(r)["oos"]),
                        "validation": PL.sharpe(PL.split_periods(r)["validation"])})
    rnd = sm["random"]["net"]
    rnd_full = rnd.apply(PL.sharpe)
    rnd_oos = PL.split_periods(rnd)["oos"].apply(PL.sharpe)
    rnd_val = PL.split_periods(rnd)["validation"].apply(PL.sharpe)
    sm_rows.append({"selection": f"Random within-industry pairs (median of {rnd.shape[1]})", "full": rnd_full.median(),
                    "oos": rnd_oos.median(), "validation": rnd_val.median()})
    sm_tab = pd.DataFrame(sm_rows)
    pct_full = float((rnd_full < sm_rows[0]["full"]).mean())
    pct_oos = float((rnd_oos < sm_rows[0]["oos"]).mean())
    sm_tab.to_csv(out / "selection_methods.csv", index=False, float_format="%.4f")
    uc = PL.load_candidates(unconstrained=True)
    uc_all = pd.concat(uc.values(), ignore_index=True)

    # ---------------------------------------------------------------- latency
    lat = pd.read_pickle(C.DATA_PROC / "latency.pkl")
    lat_names = {"0 min": "At the signal bar's closing price (no delay, optimistic)",
                 "1 min": "Close of the next 1-minute bar (base case)", "5 min": "5 minutes after the signal",
                 "15 min": "15 minutes after the signal", "next bar close": "Close of the next bar"}
    lat_tab = pd.DataFrame([{"fill": lat_names.get(k, k), "full_net_sharpe": PL.sharpe(v["net"].iloc[:, 0]),
                             "full_gross_sharpe": PL.sharpe(v["gross"].iloc[:, 0]),
                             "oos_net_sharpe": PL.sharpe(PL.split_periods(v["net"].iloc[:, 0])["oos"])}
                            for k, v in lat.items()])
    lat_tab.to_csv(out / "latency.csv", index=False, float_format="%.4f")

    # ---------------------------------------------------------------- persistence
    pers = pd.read_pickle(C.DATA_PROC / "persistence.pkl")
    pers_tab = pers.groupby("eligible").agg(n=("p_next", "size"), share_p5_next=("p_next", lambda s: (s <= C.ALPHA).mean()),
                                            median_p_next=("p_next", "median"),
                                            median_hl_form=("hl_form", "median"), median_hl_next=("hl_next", "median"),
                                            median_abs_beta_change=("beta_next", "median"))
    pers["beta_change"] = (pers["beta_next"] - pers["beta_form"]).abs()
    pers_tab["median_abs_beta_change"] = pers.groupby("eligible")["beta_change"].median()
    pers_tab.index = pers_tab.index.map({True: "Eligible (passed all filters)", False: "Other candidates (random sample)"})
    pers_tab.to_csv(out / "cointegration_persistence.csv", float_format="%.4f")

    # ---------------------------------------------------------------- figures
    split = (pd.Timestamp(pd.Period(C.VALIDATION_END, freq="M").end_time.date()), "Validation (parameters chosen here)",
             "Out of sample")
    cfg_label = (f"{METHOD_LABEL[best['method']]} hedge, {BAR_LABEL[best['bar']]} bars, enter at z = ±{best['entry']:g}, "
                 f"exit at ±{best['exit']:g}")
    P.equity_curves({"Strategy, net of costs": net, "Strategy, before costs": gross,
                     f"Average of all {n_trials} variants, net": ens},
                    "Walk-forward pairs portfolio", fig / "fig1_equity.png", split=split, log=False,
                    note=f"{cfg_label}. {C.N_PAIRS} pair slots, $1 long + $1 short per $1 of slot capital; excludes interest on capital.")
    nr = null_rates()
    _fig_funnel(by_month, nr, fig / "fig2_selection_by_month.png")
    rates = _fig_pvalues(allc, uc_all, nr, fig / "fig3_pvalue_rates.png")
    rates.to_csv(out / "pvalue_rates_vs_null.csv", index=False, float_format="%.4f")
    _fig_grid(table, best, rank_corr, fig / "fig4_validation_vs_oos.png")
    _fig_bar_size(bar_tab, fig / "fig5_bar_size.png")
    P.lines_xy({per.replace("oos", "Out of sample").replace("full", "Full sample").replace("validation", "Validation"):
                (g_["cost_multiplier"].to_numpy(), g_["sharpe"].to_numpy())
                for per, g_ in cc.groupby("period", sort=False)},
               "How much trading cost can the strategy absorb?", fig / "fig6_cost_sensitivity.png",
               "Cost multiplier (1.0x = base case)", "Net Sharpe ratio",
               xfmt=mtick.FuncFormatter(lambda v, _: f"{v:.1f}x"),
               note=f"Base case: average {avg_leg_cost:.1f} bps per leg per fill (half-spread >= half a tick, "
                    f"commission, 1 bp impact) + {C.BORROW_FEE_ANNUAL:.0%} p.a. borrow.")
    _fig_placebo(rnd_full, rnd_oos, sm_tab, fig / "fig7_selection_placebo.png")
    _fig_persistence(pers, fig / "fig8_persistence.png")
    ex = _fig_example(ctx, sel, best, tr, fig / "fig9_example_pair.png")
    _fig_data_quality(ctx, fig / "fig10_data_quality.png")

    # ---------------------------------------------------------------- meta
    pv = perf.set_index(["period", "series"])
    run_meta = {
        "best_config": best, "config_label": cfg_label, "n_trials": n_trials, "rank_corr_val_oos": rank_corr,
        "deflated_sharpe_validation": dsr, "break_even_multiplier_full": be_mult, "break_even_multiplier_oos": be_mult_oos,
        "avg_leg_cost_bps": avg_leg_cost, "capital_utilisation": util, "avg_pairs_per_month": float(held.mean()),
        "months_with_no_pair": int((held == 0).sum()), "n_months": int(len(held)),
        "placebo_pct_full": pct_full, "placebo_pct_oos": pct_oos, "n_placebo": int(rnd.shape[1]),
        "n_pair_months": int(len(allc)), "n_unconstrained_pair_months": int(len(uc_all)),
        "share_p5": float(p5.mean()), "share_p1": float((allc["p_eg"] <= 0.01).mean()),
        "n_bh": int(allc["bh_pass"].sum()), "n_bh_unconstrained": int(uc_all["bh_pass"].sum()),
        "share_p5_unconstrained": float((uc_all["p_eg"] <= C.ALPHA).mean()),
        "null_share_p5": nr["0.05"], "null_share_p1": nr["0.01"],
        "top_groups": group_counts.head(5).to_dict(), "example_pair": ex,
        "data": meta,
    }
    for per in ["validation", "oos", "full"]:
        for s_ in ["Strategy (net)", "Strategy (gross)", f"Average of all {n_trials} variants (net)", "SPY buy & hold"]:
            run_meta.setdefault("perf", {}).setdefault(per, {})[s_] = {k: (str(v) if k in ("start", "end") else float(v))
                                                                      for k, v in pv.loc[(per, s_)].items()
                                                                      if k not in ("deflated_sharpe_prob",)}
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))
    PL.log("Analysis written to results/")


# ------------------------------------------------------------------------------ figures
def _fig_funnel(bm: pd.DataFrame, nr: dict, path) -> None:
    P.style()
    f, ax = plt.subplots(figsize=(9, 4.4))
    x = pd.PeriodIndex(bm["month"], freq="M").to_timestamp()
    ax.bar(x, bm["candidates"], width=25, color="#d9e6f7", label="Within-industry candidates")
    ax.bar(x, bm["p_below_5pct"], width=25, color=P.SERIES[0], label="Engle-Granger p <= 5%")
    ax.plot(x, bm["candidates"] * nr["0.05"], color=P.INK_2, ls="--", lw=1.2, label="Expected by chance alone")
    ax.plot(x, bm["selected"], color=P.SERIES[1], marker="o", ms=4, lw=1.6, label="Traded (after all filters)")
    ax.set_ylabel("Pairs per month")
    ax.set_title("Few pairs pass the cointegration screen each month")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    P._finish(f, path, note=f"Formation window: previous 6 months of 30-minute bars. Chance level from {nr['n_sims']:,} "
                            f"simulated unrelated random-walk pairs ({nr['0.05']:.1%} pass).")


def _fig_pvalues(allc: pd.DataFrame, uc: pd.DataFrame, nr: dict, path) -> pd.DataFrame:
    rows = []
    for a in [0.01, 0.05, 0.10]:
        rows.append({"threshold": a, "within_industry": float((allc["p_eg"] <= a).mean()),
                     "any_two_stocks": float((uc["p_eg"] <= a).mean()), "no_cointegration_null": nr[str(a)]})
    t = pd.DataFrame(rows)
    P.style()
    f, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(t))
    w = 0.27
    for i, (c_, lab) in enumerate([("within_industry", f"Within industry ({len(allc):,} pair-months)"),
                                   ("any_two_stocks", f"Any two stocks ({len(uc):,} pair-months)"),
                                   ("no_cointegration_null", "Unrelated random walks (simulated null)")]):
        col = P.NEUTRAL if i == 2 else P.SERIES[i]
        ax.bar(x + (i - 1) * w, t[c_], width=w * 0.92, color=col, label=lab)
        for xi, v in zip(x, t[c_]):
            ax.annotate(f"{v:.1%}", (xi + (i - 1) * w, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, [f"p <= {a:.0%}" for a in t["threshold"]])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of pairs passing")
    ax.set_title("How much more often than chance do pairs look cointegrated?")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    P._finish(f, path, note="Engle-Granger on 6 months of 30-minute log prices, both orderings, p = 2 x the smaller p-value.")
    return t


def _fig_grid(table: pd.DataFrame, best: dict, rc: float, path) -> None:
    P.style()
    f, ax = plt.subplots(figsize=(8, 5.2))
    for i, bar in enumerate(C.BAR_SIZES):
        t = table[table["bar"] == bar]
        ax.scatter(t["validation_sharpe"], t["oos_sharpe"], s=28, color=P.SERIES[i], label=BAR_LABEL[bar],
                   edgecolor=P.SURFACE, lw=0.6, zorder=3)
    b = table[(table["bar"] == best["bar"]) & (table["method"] == best["method"]) & (table["entry"] == best["entry"])
              & (table["exit"] == best["exit"])].iloc[0]
    ax.scatter([b["validation_sharpe"]], [b["oos_sharpe"]], s=160, facecolor="none", edgecolor=P.INK, lw=1.6, zorder=4)
    ax.annotate("chosen", (b["validation_sharpe"], b["oos_sharpe"]), xytext=(8, 6), textcoords="offset points", fontsize=9)
    lim = [min(table["validation_sharpe"].min(), table["oos_sharpe"].min()) - 0.1,
           max(table["validation_sharpe"].max(), table["oos_sharpe"].max()) + 0.1]
    ax.plot(lim, lim, color=P.NEUTRAL, lw=1, ls="--")
    ax.axhline(0, color=P.NEUTRAL, lw=1)
    ax.axvline(0, color=P.NEUTRAL, lw=1)
    ax.set_xlabel(f"Validation net Sharpe ({C.FIRST_TRADE_MONTH} to {C.VALIDATION_END})")
    ax.set_ylabel(f"Out-of-sample net Sharpe ({pd.Period(C.VALIDATION_END, freq='M') + 1} to {C.LAST_TRADE_MONTH})")
    ax.set_title(f"Does the validation winner stay a winner? Rank correlation = {rc:.2f}")
    ax.legend(loc="upper left", ncol=2, title="Bar size", title_fontsize=9)
    P._finish(f, path, note=f"{len(table)} variants: 6 bar sizes x 3 hedge methods x 3 entry thresholds x 2 exit thresholds.")


def _fig_bar_size(bt: pd.DataFrame, path) -> None:
    labels = [BAR_LABEL[b] for b in bt["bar"]]
    P.style()
    f, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, bt["full_gross_sharpe_avg"], width=w * 0.92, color=P.SERIES[0], label="Before costs")
    ax.bar(x + w / 2, bt["full_net_sharpe_avg"], width=w * 0.92, color=P.SERIES[1], label="After costs")
    for xi, (a, b_) in enumerate(zip(bt["full_gross_sharpe_avg"], bt["full_net_sharpe_avg"])):
        ax.annotate(f"{a:.2f}", (xi - w / 2, a), xytext=(0, 3 if a >= 0 else -11), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{b_:.2f}", (xi + w / 2, b_), xytext=(0, 3 if b_ >= 0 else -11), textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0, color=P.NEUTRAL, lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Sharpe ratio (average of 18 variants)")
    ax.set_title("Trading frequency: gross edge vs what survives costs")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    P._finish(f, path, note="Same pairs every month (selected on 30-minute bars); only the bar size used for signals and fills changes.")


def _fig_placebo(rnd_full: pd.Series, rnd_oos: pd.Series, sm: pd.DataFrame, path) -> None:
    P.style()
    f, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (per, r) in zip(axes, [("full", rnd_full), ("oos", rnd_oos)]):
        ax.hist(r.dropna(), bins=20, color="#c9d9ef", edgecolor=P.SURFACE, label="Random within-industry pairs")
        for i, row in enumerate(sm.iloc[:3].itertuples()):
            v = getattr(row, per)
            ax.axvline(v, color=P.SERIES[i + 1 if i else 1], lw=2 if i == 0 else 1.4, ls="-" if i == 0 else "--",
                       label=row.selection)
        ax.set_title("Full sample" if per == "full" else "Out of sample")
        ax.set_xlabel("Net Sharpe ratio")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Number of random portfolios")
    h, lab = axes[0].get_legend_handles_labels()
    f.legend(h, lab, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.04))
    f.subplots_adjust(bottom=0.25)
    f.suptitle("Does the statistical screen beat picking pairs at random?", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    f.tight_layout(rect=(0, 0.1, 1, 0.95))
    f.text(0.01, 0.005, "Every portfolio uses the same trading rules, costs and 5 pair slots; only the pair choice differs.",
           fontsize=8, color=P.INK_2)
    f.savefig(path, bbox_inches="tight")
    plt.close(f)


def _fig_persistence(pers: pd.DataFrame, path) -> None:
    P.style()
    f, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(0, 1, 11)
    for i, (lab, d) in enumerate([("Pairs that passed all filters", pers[pers["eligible"]]),
                                  ("Other candidates", pers[~pers["eligible"]])]):
        h, _ = np.histogram(d["p_next"].clip(upper=0.999), bins=bins)
        ax.bar(bins[:-1] + (i - 0.5) * 0.045 + 0.05, h / max(len(d), 1), width=0.042, color=P.SERIES[i], label=f"{lab} (n = {len(d)})")
    ax.axhline(0.1, color=P.NEUTRAL, ls="--", lw=1.2, label="Uniform (no persistence)")
    ax.set_xlabel("Engle-Granger p-value re-tested on the NEXT 6 months")
    ax.set_ylabel("Share of pairs")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_title("Is cointegration still there six months later?")
    ax.legend(loc="upper center")
    ax.grid(axis="x", visible=False)
    P._finish(f, path)


def _fig_example(ctx, sel: dict, best: dict, tr: pd.DataFrame, path) -> dict:
    """z-score and positions of the chosen configuration for its busiest out-of-sample pair-month."""
    oos = tr[tr["period"] == "oos"]
    if oos.empty:
        return {}
    key = oos.groupby(["month", "y", "x"]).size().sort_values(ascending=False).index[0]
    month, y, x = key
    pairs = sel[month]
    pairs = pairs[(pairs["y"] == y) & (pairs["x"] == x)]
    bd = B.BarData(ctx.panel, best["bar"])
    res = B.run_month(bd, pd.Period(month, freq="M"), pairs, [best["method"]], [best["entry"]], [best["exit"]],
                      PL.load_half_spread().loc[pd.Period(month, freq="M")], return_signals=True)
    sg = res.signals
    z, pos = sg["z"][:, 0], np.sign(sg["hy"][:, 0])
    t = np.arange(len(z))
    P.style()
    f, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 2]})
    ax.plot(t, sg["Cy"][:, 0] / sg["Cy"][0, 0] - 1, color=P.SERIES[0], lw=1.4, label=y)
    ax.plot(t, sg["Cx"][:, 0] / sg["Cx"][0, 0] - 1, color=P.SERIES[1], lw=1.4, label=x)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax.set_ylabel("Return in month")
    ax.set_title(f"{y} vs {x}, {month}: z-score and trades ({METHOD_LABEL[best['method']]}, {BAR_LABEL[best['bar']]} bars)")
    ax.legend(loc="upper left", ncol=2)
    ax2.plot(t, z, color=P.INK_2, lw=1.0)
    e = best["entry"]
    for lvl, ls in [(e, "--"), (-e, "--"), (e + C.STOP_Z_ADD, ":"), (-e - C.STOP_Z_ADD, ":")]:
        ax2.axhline(lvl, color=P.NEUTRAL, lw=1, ls=ls)
    ax2.axhline(0, color=P.NEUTRAL, lw=1)
    # a position decided at bar a and still held at bar b is exited at bar b + 1: shade [a, b + 1]
    labelled = set()
    k = 0
    while k < len(pos):
        if pos[k] == 0:
            k += 1
            continue
        j = k
        while j + 1 < len(pos) and pos[j + 1] == pos[k]:
            j += 1
        side = "Long spread" if pos[k] > 0 else "Short spread"
        ax2.axvspan(k, min(j + 1, len(pos) - 1), color=P.SERIES[2] if pos[k] > 0 else P.SERIES[7], alpha=0.15, lw=0,
                    label=None if side in labelled else side)
        labelled.add(side)
        k = j + 1
    zz = z[np.isfinite(z)]
    ax2.set_ylim(min(-e - C.STOP_Z_ADD - 0.5, zz.min() - 0.3), max(e + C.STOP_Z_ADD + 0.5, zz.max() + 0.3))
    ax2.set_ylabel("Spread z-score")
    ticks = np.linspace(0, len(t) - 1, 6).astype(int)
    ax2.set_xticks(ticks, [pd.Timestamp(sg["ts"][i]).strftime("%d %b") for i in ticks])
    ax2.legend(loc="lower right", ncol=2)
    P._finish(f, path, note=f"Dashed: entry at |z| = {e:g}; dotted: stop-loss. Shading: position held (orders fill at the first "
                            f"minute after the signal bar closes).")
    sub = tr[(tr["month"] == month) & (tr["y"] == y) & (tr["x"] == x)]
    return {"month": month, "y": y, "x": x, "n_trades": int(len(sub)),
            "net_pnl_bps_of_capital": float((sub["gross"] - sub["cost"] - sub["borrow"]).sum() * 1e4)}


def _fig_data_quality(ctx, path) -> None:
    pnl = ctx.panel
    d = pnl.daily()
    days = pnl.cal["days"]
    j1, j2 = pnl.col["GOOGL"], pnl.col["GOOG"]
    adj = pd.Series(d["last"][:, j1], index=days)
    raw = adj * pnl.split_factor_day[:, j1]
    spread = pd.Series(np.log(d["last"][:, j2] / d["last"][:, j1]) * 1e4, index=days)
    P.style()
    f, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    w = (days >= "2022-03-01") & (days <= "2022-10-31")
    ax.plot(days[w], raw[w], color=P.SERIES[7], lw=1.6, label="As traded (vendor file)")
    ax.plot(days[w], adj[w], color=P.SERIES[0], lw=1.6, label="Split-adjusted (used)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mtick.FixedLocator([100, 200, 500, 1000, 2000, 3000]))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_locator(mtick.NullLocator())
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_title("GOOGL 20-for-1 split, 18 July 2022")
    ax.legend(loc="center left")
    ax2.plot(days, spread, color=P.SERIES[0], lw=1.0)
    ax2.axhline(0, color=P.NEUTRAL, lw=1)
    ax2.set_title("Even share classes drift: GOOG minus GOOGL")
    ax2.set_ylabel("Log price gap (bps)")
    P._finish(f, path, note="Left: unadjusted vendor prices would book a ~95% 'loss' on the split date. Right: the class C premium "
                            "reached ~4% in mid-2021 before collapsing.")
