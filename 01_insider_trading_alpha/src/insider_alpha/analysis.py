"""Stage 2: event study, skill persistence, walk-forward ML and portfolio backtests."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from . import config as C
from . import model as M
from . import plots as P
from . import stats as S
from .backtest import calendar_portfolio
from .features import FEATURES


FEATURE_LABELS = {
    "log_value": "Trade size ($, log)", "own_chg": "Holdings increase (%)", "n_lines": "Transaction lines in filing",
    "filing_lag": "Days from trade to filing", "late_filing": "Late filing", "indirect": "Indirect ownership",
    "is_ceo": "CEO", "is_cfo": "CFO", "is_chair": "Chair", "is_officer": "Officer", "is_director": "Director",
    "is_tenpct": "10% owner", "opportunistic": "Opportunistic trader", "routine": "Routine trader",
    "cluster_insiders_30d": "Insiders buying (30d)", "sellers_90d": "Insiders selling (90d)",
    "prior_buys_issuer_1y": "Prior buys at company (1y)", "tr_n": "Track record: # past trades",
    "tr_mean": "Track record: mean return", "tr_hit": "Track record: hit rate",
    "tr_shrunk": "Track record: shrunk mean", "mom_1m": "1-month momentum", "mom_6m_skip1m": "6-1 month momentum",
    "mom_12m": "12-month momentum", "vol_3m": "3-month volatility", "dist_52w_high": "Distance from 52w high",
    "dist_52w_low": "Distance from 52w low", "log_price": "Share price (log)", "log_adv": "Dollar volume (log)",
    "mkt_ret_1m": "Market 1-month return", "mkt_vol_1m": "Market volatility",
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def md_table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    """Minimal markdown table writer (no extra dependency)."""
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = floatfmt.get(c)
            if f and isinstance(v, (int, float, np.floating, np.integer)) and pd.notna(v):
                cells.append(f.format(v))
            else:
                cells.append("" if (isinstance(v, float) and np.isnan(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def winsorize(x: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    a, b = x.quantile([lo, hi])
    return x.clip(a, b)


# ------------------------------------------------------------------------- loading
def load_processed(proc_dir: Path) -> dict:
    with open(proc_dir / "events.pkl", "rb") as fh:
        d = pickle.load(fh)
    with open(proc_dir / "prices.pkl", "rb") as fh:
        d["prices"] = pickle.load(fh)
    npz = np.load(proc_dir / "abnormal_paths.npz")
    d["paths_P"], d["paths_S"] = npz["P"], npz["S"]
    d["factors"] = pd.read_pickle(proc_dir / "factors.pkl")
    d["bench"] = pd.read_pickle(proc_dir / "benchmarks.pkl")
    return d


# --------------------------------------------------------------------- event study
def _groups(b: pd.DataFrame, s: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    size = pd.cut(b["value"], [0, 50e3, 250e3, 1e6, np.inf], labels=["<$50k", "$50k-250k", "$250k-1M", ">$1M"])
    liq = pd.qcut(b["log_adv"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    g = {
        "All purchases": (b, b.index == b.index),
        "Purchases - investable": (b, b["investable"]),
        "CEO/CFO": (b, b["role"] == "CEO/CFO"),
        "Other officer": (b, b["role"] == "Other officer"),
        "Director": (b, b["role"] == "Director"),
        "10% owner": (b, b["role"] == "10% owner"),
        "Cluster buy (>=2 insiders, 30d)": (b, b["is_cluster"]),
        "Single-insider buy": (b, ~b["is_cluster"]),
        "Opportunistic trader": (b, b["trader_type"] == "opportunistic"),
        "Routine trader": (b, b["trader_type"] == "routine"),
        "Holdings +>=10%": (b, b["own_chg"] >= 0.10),
        "Holdings +<10%": (b, b["own_chg"] < 0.10),
    }
    for lab in size.cat.categories:
        g[f"Trade size {lab}"] = (b, size == lab)
    for lab in ["Q1", "Q5"]:
        g[f"Liquidity {lab} ({'least' if lab == 'Q1' else 'most'} liquid)"] = (b, liq == lab)
    g["All sales"] = (s, s.index == s.index)
    g["Sales - CEO/CFO"] = (s, s["role"] == "CEO/CFO")
    return g


def event_study(d: dict, out: Path, fig: Path) -> pd.DataFrame:
    b = d["buys"][d["buys"]["filing_date"] >= C.SAMPLE_START]
    s = d["sells"][d["sells"]["filing_date"] >= C.SAMPLE_START]
    rows = []
    for name, (df, mask) in _groups(b, s).items():
        sub = df[np.asarray(mask, dtype=bool)]
        row = {"group": name, "n_events": len(sub)}
        for h in C.HORIZONS:
            col = f"bhar_{h}"
            x = winsorize(df[col].dropna()).reindex(sub.index).dropna()
            m, se, t = S.clustered_mean(x, sub.loc[x.index, "filing_date"].dt.to_period("M"))
            row[f"bhar_{h}d"] = m
            row[f"t_{h}d"] = t
        x63 = sub[f"bhar_{C.LABEL_HORIZON}"].dropna()
        row["median_63d"] = x63.median()
        row["pct_positive_63d"] = (x63 > 0).mean()
        xm = winsorize(df[f"bhar_mkt_{C.LABEL_HORIZON}"].dropna()).reindex(sub.index).dropna()
        row["bhar_vs_spy_63d"] = xm.mean()
        for c in ["abn_trade_to_filing", "abn_overnight_gap"]:
            xx = winsorize(df[c].dropna()).reindex(sub.index).dropna()
            row[c] = xx.mean()
        rows.append(row)
    es = pd.DataFrame(rows)
    es.to_csv(out / "event_study.csv", index=False)

    # figures: mean abnormal path (winsorised per day) by group
    def mean_path(paths, idx):
        p = paths[idx]
        lo, hi = np.nanpercentile(paths, [1, 99], axis=0)
        return np.nanmean(np.clip(p, lo, hi), axis=0)

    pb, ps = d["paths_P"], d["paths_S"]
    B, SL = d["buys"], d["sells"]
    in_b = (B["filing_date"] >= C.SAMPLE_START) & (B["path_row"] >= 0)
    ib = B.loc[in_b, "path_row"].to_numpy()
    isl = SL.loc[(SL["filing_date"] >= C.SAMPLE_START) & (SL["path_row"] >= 0), "path_row"].to_numpy()

    def rows_where(mask):
        return B.loc[in_b & mask, "path_row"].to_numpy()

    note = (f"Events filed {C.SAMPLE_START[:4]}-{b['filing_date'].max().year}. Buy-and-hold return minus {C.BENCHMARK}, "
            "winsorised at 1%/99% per day. Source: SEC Form 4 data sets, Yahoo Finance.")
    P.car_paths({r: mean_path(pb, rows_where(B["role"] == r))
                 for r in ["CEO/CFO", "Other officer", "Director", "10% owner"]},
                "Insider purchases: abnormal return by insider role", fig / "fig1_car_by_role.png", note)
    P.car_paths({
        "Cluster buys": mean_path(pb, rows_where(B["is_cluster"])),
        "Opportunistic buys": mean_path(pb, rows_where(B["trader_type"] == "opportunistic")),
        "All buys": mean_path(pb, ib),
        "All sales": mean_path(ps, isl),
    }, "Purchases vs sales: who is informative?", fig / "fig2_buys_vs_sells.png", note)

    # how has the signal changed over time? (5-day vs 63-day abnormal return by filing year)
    yrs = []
    for y, g in b.groupby(b["filing_date"].dt.year):
        row = {"year": y, "n": len(g)}
        for h in [5, 63]:
            x = winsorize(b[f"bhar_{h}"].dropna()).reindex(g.index).dropna()
            m, se, t = S.clustered_mean(x, g.loc[x.index, "filing_date"].dt.to_period("M"))
            row[f"bhar_{h}d"], row[f"se_{h}d"], row[f"t_{h}d"] = m, se, t
        yrs.append(row)
    yrs = pd.DataFrame(yrs)
    yrs = yrs[yrs["n"] >= 500]
    yrs.to_csv(out / "event_study_by_year.csv", index=False)
    done = yrs[yrs["year"] <= b["filing_date"].max().year - (1 if b["filing_date"].max().month < 7 else 0)]
    P.grouped_bars([str(y) for y in done["year"]],
                   {"First 5 days": done["bhar_5d"].to_numpy(), "First 63 days (~3 months)": done["bhar_63d"].to_numpy()},
                   "A reliable short-term pop, an unreliable 3-month drift", fig / "fig9_signal_by_year.png",
                   "Mean abnormal return vs IWM",
                   errors={"First 5 days": 1.96 * done["se_5d"].to_numpy(),
                           "First 63 days (~3 months)": 1.96 * done["se_63d"].to_numpy()},
                   note="All purchase events by filing year; winsorised 1%/99%; whiskers: 95% CI clustered by month.")

    # who captures the return? insider's own window vs what an outsider can trade
    def cm(col, df=b):
        x = winsorize(df[col].dropna())
        return S.clustered_mean(x, df.loc[x.index, "filing_date"].dt.to_period("M"))
    parts = [("Insider's trade date -> filing date close", "abn_trade_to_filing")]
    tl = []
    for lab, col in parts:
        m, se, t = cm(col)
        tl.append({"window": lab, "mean": m, "t": t})
    m, se, t = cm("abn_overnight_gap")
    tl.append({"window": "Filing-day close -> next open (overnight)", "mean": m, "t": t})
    for lab, h in [("Entry open -> day 5 close", 5), ("Entry open -> day 21 close", 21), ("Entry open -> day 63 close", 63)]:
        m, se, t = cm(f"bhar_{h}")
        tl.append({"window": lab, "mean": m, "t": t})
    pd.DataFrame(tl).to_csv(out / "return_timeline.csv", index=False)
    return es


# --------------------------------------------------------------- skill persistence
def skill_persistence(d: dict, out: Path, fig: Path) -> dict:
    b = d["buys"]
    b = b[(b["filing_date"] >= C.SAMPLE_START) & b[f"bhar_{C.LABEL_HORIZON}"].notna()].copy()
    y = f"bhar_{C.LABEL_HORIZON}"
    b["y_w"] = winsorize(b[y])
    rep = b[b["tr_n"] >= 3].copy()
    rep["q"] = pd.qcut(rep["tr_mean"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    rows = []
    for q, g in rep.groupby("q", observed=True):
        m, se, t = S.clustered_mean(g["y_w"], g["filing_date"].dt.to_period("M"))
        rows.append({"track_record_quintile": str(q), "n": len(g), "past_mean_bhar": g["tr_mean"].mean(),
                     "future_bhar_63d": m, "t_stat": t, "se": se})
    first = b[b["tr_n"] == 0]
    m, se, t = S.clustered_mean(first["y_w"], first["filing_date"].dt.to_period("M"))
    rows.append({"track_record_quintile": "No realised history", "n": len(first), "past_mean_bhar": np.nan,
                 "future_bhar_63d": m, "t_stat": t, "se": se})
    per = pd.DataFrame(rows)
    # monthly Q5-Q1 long-short spread series -> Newey-West t
    rep["month"] = rep["filing_date"].dt.to_period("M")
    mq = rep.groupby(["month", "q"], observed=True)["y_w"].mean().unstack()
    spread = (mq["Q5"] - mq["Q1"]).dropna()
    nw = S.newey_west_ols(spread.to_numpy(), np.ones((len(spread), 1)), lags=3)
    # Fama-MacBeth: monthly cross-sections, coefficient on the shrunk track record
    ctrl = ["tr_shrunk", "log_value", "own_chg", "cluster_insiders_30d", "log_adv", "mom_6m_skip1m", "vol_3m"]
    fm = []
    for mth, g in b.assign(month=b["filing_date"].dt.to_period("M")).groupby("month"):
        g = g.dropna(subset=ctrl + ["y_w"])
        if len(g) < 30:
            continue
        X = np.column_stack([np.ones(len(g))] + [(g[c] - g[c].mean()) / (g[c].std() + 1e-12) for c in ctrl])
        beta = np.linalg.lstsq(X, g["y_w"].to_numpy(), rcond=None)[0]
        fm.append(beta)
    fm = np.array(fm)
    fm_res = {}
    for i, c in enumerate(["const"] + ctrl):
        r = S.newey_west_ols(fm[:, i], np.ones((len(fm), 1)), lags=3)
        fm_res[c] = {"coef_per_1sd": float(r["beta"][0]), "t": float(r["t"][0])}
    res = {"quintiles": per, "q5_minus_q1_monthly_mean": float(nw["beta"][0]),
           "q5_minus_q1_t": float(nw["t"][0]), "n_months": len(spread), "fama_macbeth": fm_res}
    per.to_csv(out / "skill_persistence.csv", index=False)
    pd.DataFrame(fm_res).T.to_csv(out / "fama_macbeth.csv")
    lab = per["track_record_quintile"].tolist()
    lab = [l.replace("No realised history", "No history") for l in lab]
    P.bars(lab, per["future_bhar_63d"].to_numpy(), "Does an insider's past record predict the next trade?",
           fig / "fig3_skill_persistence.png", "Next purchase: 63-day abnormal return",
           errors=1.96 * per["se"].to_numpy(), highlight=[0, 4],
           note="Quintiles of the insider's point-in-time mean abnormal return on >=3 earlier purchases "
                "whose 63-day windows had closed. Bars: mean; whiskers: 95% CI (clustered by month).")
    return res


# ----------------------------------------------------------------------- ML model
def run_models(d: dict, out: Path, fig: Path, models=("logit", "adaboost", "gbm")) -> pd.DataFrame:
    b = d["buys"]
    b = b[(b["filing_date"] >= C.SAMPLE_START) & b["investable"]].copy()
    label = f"bhar_{C.LABEL_HORIZON}"
    log(f"Walk-forward models on {len(b):,} investable purchases ...")
    scores, diags = M.walk_forward(b, FEATURES, label, list(models))
    b = b.join(scores)
    metrics = []
    for m in models:
        mm = M.oos_metrics(b, f"score_{m}", label)
        mm.insert(0, "model", m)
        metrics.append(mm)
    metrics = pd.concat(metrics)
    metrics.to_csv(out / "model_oos_metrics.csv", index=False)
    pd.DataFrame(diags).to_csv(out / "model_folds.csv", index=False)
    log("Permutation importance ...")
    imp = M.permutation_importance_last_fold(b, FEATURES, label, "gbm")
    imp.to_csv(out / "feature_importance.csv", index=False)
    top = imp.head(12)
    P.hbars([FEATURE_LABELS.get(f, f) for f in top["feature"]], top["auc_drop"], "What drives the model? (out-of-sample permutation importance)",
            fig / "fig6_feature_importance.png", "Drop in AUC when the feature is shuffled",
            errors=top["std"], note="Gradient-boosting model trained on data before the final year, evaluated on the final year.")
    # decile spread of the primary model's OOS score (model fixed in config, not picked ex post)
    best = C.PRIMARY_MODEL
    oos = b[b[f"score_{best}"].notna() & b[label].notna()].copy()
    oos["dec"] = pd.qcut(oos[f"score_{best}"].rank(method="first"), 10, labels=False) + 1
    oos["y_w"] = winsorize(oos[label])
    dec = oos.groupby("dec")["y_w"].agg(["mean", "count"])
    dec["se"] = oos.groupby("dec").apply(lambda g: S.clustered_mean(g["y_w"], g["filing_date"].dt.to_period("M"))[1])
    dec.to_csv(out / "score_deciles.csv")
    P.bars([str(i) for i in dec.index], dec["mean"].to_numpy(),
           "Out-of-sample: 3-month abnormal return by model score decile",
           fig / "fig5_score_deciles.png", "Mean 63-day abnormal return vs IWM", highlight=[0, 9],
           errors=1.96 * dec["se"].to_numpy(),
           note=f"Gradient boosting, walk-forward predictions {C.WALK_FORWARD_START[:4]}+, retrained every "
                f"{C.RETRAIN_MONTHS} months on realised data only. Decile 10 = highest score. Whiskers: 95% CI.")
    d["buys_scored"] = b
    d["best_model"] = best
    return metrics


# ------------------------------------------------------------------------ backtest
def _row(name, r, gross, pf_sel_n, years_n, rf, fac, bench, sr_trials, n_trials):
    ps = S.perf_stats(r, rf)
    ff = S.factor_regression(r, fac, ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
    ex = r - rf.reindex(r.index).fillna(0)
    lo, hi = S.sharpe_bootstrap_ci(ex)
    hedged = r - bench[C.BENCHMARK].reindex(r.index).fillna(0)
    return {
        "strategy": name, **{k: ps[k] for k in ["start", "end", "cagr", "vol", "sharpe", "sortino", "max_dd", "calmar"]},
        "sharpe_ci_low": lo, "sharpe_ci_high": hi,
        "deflated_sharpe_prob": S.deflated_sharpe(ps["sharpe"], len(r), ps["skew"], ps["kurt"],
                                                 n_trials=n_trials, sr_trials_ann=sr_trials),
        "gross_sharpe": S.perf_stats(gross, rf)["sharpe"] if gross is not None else np.nan,
        "hedged_vs_iwm_ann_return": hedged.mean() * 252,
        "hedged_vs_iwm_sharpe": hedged.mean() / hedged.std() * np.sqrt(252),
        "ff6_alpha_ann": ff["alpha_ann"], "ff6_alpha_t": ff["alpha_t"], "beta_mkt": ff["beta_mkt_rf"],
        "beta_smb": ff["beta_smb"], "avg_positions": pf_sel_n, "trades_per_year": years_n,
    }


def run_backtests(d: dict, out: Path, fig: Path) -> pd.DataFrame:
    px = d["prices"]
    dates, adj_close, adj_open = px["dates"], px["adj_close"].astype("float64"), px["adj_open"].astype("float64")
    fac = d["factors"]
    rf = fac["rf"]
    b = d["buys_scored"]
    b = b[b["filing_date"] >= C.WALK_FORWARD_START]
    best = d["best_model"]
    ml_long, ml_short = f"ML top 20% ({best})", f"ML bottom 20% ({best})"
    strategies = {
        ml_long: b[f"sel_{best}"] == 1,
        "All investable buys": b.index == b.index,
        "CEO/CFO buys": b["role"] == "CEO/CFO",
        "Cluster buys": b["is_cluster"],
        "Opportunistic buys": b["trader_type"] == "opportunistic",
        "Track-record insiders (point-in-time)": (b["tr_n"] >= 3) & (b["tr_hit"] >= 0.57),
        ml_short: b[f"sel_low_{best}"] == 1,
    }
    for m in ["logit", "adaboost", "gbm"]:
        if m != best and f"sel_{m}" in b:
            strategies[f"ML top 20% ({m})"] = b[f"sel_{m}"] == 1
    start = pd.Timestamp(C.WALK_FORWARD_START)
    bench = d["bench"].pct_change()
    rets, gross_rets, pfs, sels = {}, {}, {}, {}
    for name, mask in strategies.items():
        sel = b[np.asarray(mask, dtype=bool)]
        pf = calendar_portfolio(sel, adj_close, adj_open, dates, rf=rf)
        pf = pf[pf.index >= start]
        rets[name], gross_rets[name], pfs[name], sels[name] = pf["net"], pf["gross"], pf, sel
    common_start = max(r.index[0] for r in rets.values())
    # stop when the last filing in the data has completed its holding period (afterwards: cash)
    active = pfs["All investable buys"]["n_open"]
    common_end = active[active > 0].index[-1]
    cut = lambda x: x[(x.index >= common_start) & (x.index <= common_end)]
    rets = {k: cut(v) for k, v in rets.items()}
    gross_rets = {k: cut(v) for k, v in gross_rets.items()}
    bench = cut(bench)
    # market-neutral spread: long top quintile, short bottom quintile (short leg pays its trading
    # costs and a borrow fee), per $1 long + $1 short
    short_cost = gross_rets[ml_short] - rets[ml_short]
    borrow = C.SHORT_BORROW_APR / 252
    ls_name = f"Long-short: ML top - bottom 20% ({best})"
    rets[ls_name] = rets[ml_long] - gross_rets[ml_short] - short_cost - borrow
    gross_rets[ls_name] = gross_rets[ml_long] - gross_rets[ml_short]
    n_trials = len(rets)
    sr_trials = [S.perf_stats(r, rf)["sharpe"] for r in rets.values()]
    summary = []
    for name in rets:
        if name == ls_name:
            pos = pfs[ml_long]["n_open"].mean() + pfs[ml_short]["n_open"].mean()
            tpy = (len(sels[ml_long]) + len(sels[ml_short])) / S.perf_stats(rets[name], rf)["years"]
        else:
            pos = pfs[name]["n_open"].mean()
            tpy = len(sels[name]) / S.perf_stats(rets[name], rf)["years"]
        summary.append(_row(name, rets[name], gross_rets[name], pos, tpy, rf, fac, bench, sr_trials, n_trials))
    for bname in [C.BENCHMARK, C.MARKET]:
        r = bench[bname].dropna()
        ps = S.perf_stats(r, rf)
        ff = S.factor_regression(r, fac, ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
        summary.append({"strategy": f"{bname} (buy & hold)",
                        **{k: ps[k] for k in ["start", "end", "cagr", "vol", "sharpe", "sortino", "max_dd", "calmar"]},
                        "ff6_alpha_ann": ff["alpha_ann"], "ff6_alpha_t": ff["alpha_t"], "beta_mkt": ff["beta_mkt_rf"],
                        "beta_smb": ff["beta_smb"]})
    summ = pd.DataFrame(summary)
    summ.to_csv(out / "backtest_summary.csv", index=False)
    allr = pd.DataFrame({**rets, **{f"{k} (buy & hold)": bench[k] for k in [C.BENCHMARK, C.MARKET]}})
    allr.to_csv(out / "daily_returns_net.csv", float_format="%.6f")
    annual = (1 + allr.fillna(0)).groupby(allr.index.year).prod() - 1
    annual.index.name = "year"
    annual.to_csv(out / "annual_returns.csv")

    # cost sensitivity (x base-case cost schedule)
    cs_rows = []
    for name in [ml_long, "Opportunistic buys", "All investable buys"]:
        for mlt in [0, 0.5, 1, 1.5, 2, 3]:
            pf = calendar_portfolio(sels[name], adj_close, adj_open, dates, rf=rf, cost_mult=mlt)
            r = cut(pf["net"])
            ps = S.perf_stats(r, rf)
            cs_rows.append({"strategy": name, "cost_multiplier": mlt, "sharpe": ps["sharpe"], "cagr": ps["cagr"]})
    cs = pd.DataFrame(cs_rows)
    cs.to_csv(out / "cost_sensitivity.csv", index=False)
    P.lines_xy({n: (g["cost_multiplier"].to_numpy(), g["sharpe"].to_numpy()) for n, g in cs.groupby("strategy", sort=False)},
               "How much trading cost can the edge absorb?", fig / "fig7_cost_sensitivity.png",
               "Cost multiplier (1.0x = base case: 5-60 bps per side by liquidity)", "Net Sharpe ratio",
               xfmt=mtick.FuncFormatter(lambda v, _: f"{v:.1f}x"),
               ref=(S.perf_stats(bench[C.BENCHMARK].dropna(), rf)["sharpe"], f"{C.BENCHMARK} buy & hold"),
               note="Base-case one-way costs: 60 bps (ADV < $1M) to 5 bps (ADV > $100M), charged on entry and exit.")

    # holding-period robustness (post hoc: chosen after seeing the event study, reported for transparency)
    hp_rows = []
    for name in [ml_long, "All investable buys", "Opportunistic buys"]:
        for hold in [5, 10, 21, 63, 126]:
            pf = calendar_portfolio(sels[name], adj_close, adj_open, dates, rf=rf, hold=hold)
            r = cut(pf["net"])
            ff = S.factor_regression(r, fac, ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
            hp_rows.append({"strategy": name, "hold_days": hold, "sharpe": S.perf_stats(r, rf)["sharpe"],
                            "gross_sharpe": S.perf_stats(cut(pf["gross"]), rf)["sharpe"],
                            "ff6_alpha_ann": ff["alpha_ann"], "ff6_alpha_t": ff["alpha_t"]})
    pd.DataFrame(hp_rows).to_csv(out / "holding_period_sensitivity.csv", index=False)

    end = allr.index[-1].date()
    curves = {ml_long: rets[ml_long], "All investable buys": rets["All investable buys"],
              f"{C.BENCHMARK} (small caps)": bench[C.BENCHMARK], f"{C.MARKET} (S&P 500)": bench[C.MARKET]}
    P.equity_curves(curves, "Out-of-sample long-only portfolios, net of costs", fig / "fig4_equity_curves.png",
                    note=f"Walk-forward, {common_start.date()} to {end}. Enter at next open after filing, "
                         f"hold {C.LABEL_HORIZON} sessions, liquidity-based costs. Not investment advice.")
    P.equity_curves({ls_name.replace(f" ({best})", ""): rets[ls_name],
                     "Top 20% minus IWM (beta ~1 hedge)": rets[ml_long] - bench[C.BENCHMARK]},
                    "Market-neutral: does the model rank insider buys?", fig / "fig8_long_short.png",
                    note=f"Long top-quintile / short bottom-quintile insider purchases by model score; short leg pays "
                         f"trading costs + {C.SHORT_BORROW_APR:.0%} p.a. borrow. {common_start.date()} to {end}.")
    d["common_start"] = common_start
    return summ


# --------------------------------------------------------------------------- run
def run_all(proc_dir: Path = C.DATA_PROC, results_dir: Path = C.RESULTS) -> dict:
    proc_dir, out = Path(proc_dir), Path(results_dir)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    d = load_processed(proc_dir)
    log("Event study ...")
    es = event_study(d, out, fig)
    log("Skill persistence ...")
    sp = skill_persistence(d, out, fig)
    log("Models ...")
    mm = run_models(d, out, fig)
    log("Backtests ...")
    bt = run_backtests(d, out, fig)
    meta = {
        "n_purchase_events": int((d["buys"]["filing_date"] >= C.SAMPLE_START).sum()),
        "n_sale_events": int((d["sells"]["filing_date"] >= C.SAMPLE_START).sum()),
        "first_filing": str(d["buys"]["filing_date"].min().date()),
        "last_filing": str(d["buys"]["filing_date"].max().date()),
        "best_model": d["best_model"], "oos_start": str(d["common_start"].date()),
        "skill_q5_minus_q1": sp["q5_minus_q1_monthly_mean"], "skill_q5_minus_q1_t": sp["q5_minus_q1_t"],
        "fama_macbeth": sp["fama_macbeth"],
    }
    bm = out / "build_meta.json"
    if bm.exists():
        meta.update(json.loads(bm.read_text()))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log("All results written to " + str(out))
    return {"event_study": es, "skill": sp, "models": mm, "backtest": bt, "meta": meta}
