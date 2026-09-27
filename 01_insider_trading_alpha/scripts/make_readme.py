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


def pct(x, d=1, sign=True):
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return f"{x:.{d}f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


def main() -> None:
    global RES
    if len(sys.argv) > 1:
        RES = Path(sys.argv[1])
    es = pd.read_csv(RES / "event_study.csv").set_index("group")
    tl = pd.read_csv(RES / "return_timeline.csv")
    yr = pd.read_csv(RES / "event_study_by_year.csv")
    cov = pd.read_csv(RES / "data_coverage.csv", index_col=0)
    mm = pd.read_csv(RES / "model_oos_metrics.csv")
    dec = pd.read_csv(RES / "score_deciles.csv")
    bt = pd.read_csv(RES / "backtest_summary.csv").set_index("strategy")
    hp = pd.read_csv(RES / "holding_period_sensitivity.csv")
    cs = pd.read_csv(RES / "cost_sensitivity.csv")
    sp = pd.read_csv(RES / "skill_persistence.csv").set_index("track_record_quintile")
    meta = json.loads((RES / "run_meta.json").read_text())
    best = meta["best_model"]
    top, bot = f"ML top 20% ({best})", f"ML bottom 20% ({best})"
    ls = f"Long-short: ML top - bottom 20% ({best})"
    allb = "All investable buys"
    ap, op, sl = es.loc["All purchases"], es.loc["Opportunistic trader"], es.loc["All sales"]
    g = mm[(mm["model"] == best) & (mm["year"] == "all")].iloc[0]
    tot = cov.loc["total"]
    yrs_full = yr[yr["year"] <= int(meta["last_filing"][:4]) - 1]
    v = {
        "n_lines": f"{meta.get('n_raw_lines', 0):,}",
        "n_buy_events": f"{int(tot['purchase_events']):,}",
        "n_buy_tradeable": f"{int(tot['tradeable_entry']):,}",
        "coverage": pct(tot["tradeable_entry"] / tot["purchase_events"], 0, False),
        "n_sample_buys": f"{meta['n_purchase_events']:,}",
        "n_sample_sells": f"{meta['n_sale_events']:,}",
        "n_sell_events": f"{meta.get('n_sale_events_all', meta['n_sale_events']):,}",
        "first_filing": meta["first_filing"], "last_filing": meta["last_filing"],
        "oos_start": meta["oos_start"], "oos_end": str(bt.loc[top, "end"]),
        "n_tickers": f"{meta.get('n_tickers_priced', 0):,}",
        "b5": pct(ap["bhar_5d"], 2), "t5": num(ap["t_5d"], 1),
        "b21": pct(ap["bhar_21d"], 2), "t21": num(ap["t_21d"], 1),
        "b63": pct(ap["bhar_63d"], 2), "t63": num(ap["t_63d"], 1),
        "b126": pct(ap["bhar_126d"], 2), "t126": num(ap["t_126d"], 1),
        "opp63": pct(op["bhar_63d"], 2), "opp63t": num(op["t_63d"], 1),
        "opp126": pct(op["bhar_126d"], 2), "opp126t": num(op["t_126d"], 1),
        "sell63": pct(sl["bhar_63d"], 2), "sell63t": num(sl["t_63d"], 1),
        "pre_filing": pct(tl.iloc[0]["mean"], 2), "overnight": pct(tl.iloc[1]["mean"], 2),
        "priced_before": pct(tl.iloc[0]["mean"] + tl.iloc[1]["mean"], 1),
        "years_pos5": f"{int((yrs_full['t_5d'] > 1.96).sum())} of {len(yrs_full)}",
        "auc": num(g["auc"], 3), "ic": num(g["rank_ic"], 3),
        "dec10": pct(dec.iloc[-1]["mean"], 1), "dec1": pct(dec.iloc[0]["mean"], 1),
        "dec_spread": f"{(dec.iloc[-1]['mean'] - dec.iloc[0]['mean']) * 100:.1f}",
        "top_sharpe": num(bt.loc[top, "sharpe"]), "top_cagr": pct(bt.loc[top, "cagr"], 1, False),
        "top_dd": pct(bt.loc[top, "max_dd"], 0), "top_alpha": pct(bt.loc[top, "ff6_alpha_ann"], 1),
        "top_alpha_t": num(bt.loc[top, "ff6_alpha_t"], 1), "top_beta": num(bt.loc[top, "beta_mkt"]),
        "top_ci": f"[{bt.loc[top, 'sharpe_ci_low']:.2f}, {bt.loc[top, 'sharpe_ci_high']:.2f}]",
        "top_dsr": pct(bt.loc[top, "deflated_sharpe_prob"], 0, False),
        "all_sharpe": num(bt.loc[allb, "sharpe"]), "all_alpha": pct(bt.loc[allb, "ff6_alpha_ann"], 1),
        "all_alpha_t": num(bt.loc[allb, "ff6_alpha_t"], 1),
        "iwm_sharpe": num(bt.loc["IWM (buy & hold)", "sharpe"]), "spy_sharpe": num(bt.loc["SPY (buy & hold)", "sharpe"]),
        "iwm_cagr": pct(bt.loc["IWM (buy & hold)", "cagr"], 1, False),
        "bot_alpha": pct(bt.loc[bot, "ff6_alpha_ann"], 1), "bot_alpha_t": num(bt.loc[bot, "ff6_alpha_t"], 1),
        "ls_sharpe": num(bt.loc[ls, "sharpe"]), "ls_gross": num(bt.loc[ls, "gross_sharpe"]),
        "ls_alpha": pct(bt.loc[ls, "ff6_alpha_ann"], 1), "ls_alpha_t": num(bt.loc[ls, "ff6_alpha_t"], 1),
        "ls_beta": num(bt.loc[ls, "beta_mkt"]),
        "q1_future": pct(sp.loc["Q1", "future_bhar_63d"], 2), "q5_future": pct(sp.loc["Q5", "future_bhar_63d"], 2),
        "q5_t": num(sp.loc["Q5", "t_stat"], 1),
        "fm_coef": pct(meta["fama_macbeth"]["tr_shrunk"]["coef_per_1sd"], 2),
        "fm_t": num(meta["fama_macbeth"]["tr_shrunk"]["t"], 1),
        "skill_spread_t": num(meta["skill_q5_minus_q1_t"], 1),
    }
    cov_ratio = tot["tradeable_entry"] / tot["purchase_events"]
    v["miss_delisted"] = pct(meta.get("missing_share_delisted", float("nan")), 0, False)
    h5 = hp[(hp["strategy"] == allb) & (hp["hold_days"] == 5)].iloc[0]
    v["h5_gross"], v["h5_net"] = num(h5["gross_sharpe"]), num(h5["sharpe"])
    c2 = cs[(cs["strategy"] == top) & (cs["cost_multiplier"] == 2)].iloc[0]
    v["top_sharpe_2x"] = num(c2["sharpe"])

    # ---- tables
    rows = [top, allb, "Opportunistic buys", "Cluster buys", "CEO/CFO buys",
            "Track-record insiders (point-in-time)", bot, ls, "IWM (buy & hold)", "SPY (buy & hold)"]
    t = bt.loc[[r for r in rows if r in bt.index]].reset_index()
    tab = pd.DataFrame({
        "Strategy (net of costs)": t["strategy"].str.replace(f" ({best})", "", regex=False),
        "CAGR": t["cagr"].map(lambda x: pct(x, 1, False)), "Vol": t["vol"].map(lambda x: pct(x, 0, False)),
        "Sharpe": t["sharpe"].map(num),
        "Sharpe 95% CI": [f"[{a:.2f}, {b:.2f}]" if pd.notna(a) else "" for a, b in zip(t["sharpe_ci_low"], t["sharpe_ci_high"])],
        "Max DD": t["max_dd"].map(lambda x: pct(x, 0)),
        "FF5+Mom alpha p.a. (t)": [f"{a * 100:+.1f}% ({b:.1f})" if pd.notna(a) else "" for a, b in zip(t["ff6_alpha_ann"], t["ff6_alpha_t"])],
        "Market beta": t["beta_mkt"].map(lambda x: "" if pd.isna(x) else num(x)),
    })
    v["table_backtest"] = md(tab)
    groups = ["All purchases", "CEO/CFO", "Director", "10% owner", "Cluster buy (>=2 insiders, 30d)",
              "Opportunistic trader", "Routine trader", "Trade size >$1M", "All sales"]
    e = es.loc[[x for x in groups if x in es.index]].reset_index()
    fmt = lambda m, tt: f"{m * 100:+.2f}% ({tt:.1f})"
    v["table_event"] = md(pd.DataFrame({
        "Group": e["group"], "Events": e["n_events"].map(lambda x: f"{x:,}"),
        "+5 days (t)": [fmt(a, b) for a, b in zip(e["bhar_5d"], e["t_5d"])],
        "+21 days (t)": [fmt(a, b) for a, b in zip(e["bhar_21d"], e["t_21d"])],
        "+63 days (t)": [fmt(a, b) for a, b in zip(e["bhar_63d"], e["t_63d"])],
        "+126 days (t)": [fmt(a, b) for a, b in zip(e["bhar_126d"], e["t_126d"])],
    }))
    v["table_timeline"] = md(pd.DataFrame({"Window": tl["window"],
                                           "Mean abnormal return": tl["mean"].map(lambda x: pct(x, 2)),
                                           "t-stat": tl["t"].map(lambda x: num(x, 1))}))
    m2 = mm[mm["year"] == "all"]
    v["table_models"] = md(pd.DataFrame({"Model": m2["model"], "OOS events": m2["n"].map(lambda x: f"{int(x):,}"),
                                         "AUC": m2["auc"].map(lambda x: num(x, 3)),
                                         "Rank IC": m2["rank_ic"].map(lambda x: num(x, 3)),
                                         "Top-bottom quintile spread (63d)": m2["q5_minus_q1"].map(lambda x: pct(x, 1))}))
    h = hp.pivot(index="strategy", columns="hold_days", values="sharpe")
    hg = hp.pivot(index="strategy", columns="hold_days", values="gross_sharpe")
    v["table_hold"] = md(pd.DataFrame({"Strategy": [s.replace(f" ({best})", "") for s in h.index],
                                       **{f"{c}d net (gross)": [f"{h.loc[s, c]:.2f} ({hg.loc[s, c]:.2f})" for s in h.index]
                                          for c in h.columns}}))
    tmpl = (ROOT / "README.template.md").read_text(encoding="utf-8")
    missing = set(re.findall(r"\{\{(\w+)\}\}", tmpl)) - set(v)
    if missing:
        sys.exit(f"template placeholders without values: {sorted(missing)}")
    out = re.sub(r"\{\{(\w+)\}\}", lambda m_: str(v[m_.group(1)]), tmpl)
    (ROOT / "README.md").write_text(out, encoding="utf-8")
    print("README.md written")


if __name__ == "__main__":
    main()
