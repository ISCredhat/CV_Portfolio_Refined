"""Render the README section "Stage two: can any of this be traded?" from the stage-2 result CSVs.

Reads results/stage2_convergence.csv, stage2_markouts.csv and stage2_forecast.csv (offline, aggregates
only) and replaces the block between the markers
    <!-- STAGE2:BEGIN --> ... <!-- STAGE2:END -->
in README.md (the markers must already exist). Every number in the tables comes from the CSVs; the
surrounding prose describes the pre-registered design and caveats. Rerun after regenerating the results:

    python scripts/make_readme_stage2.py            # rewrite README.md
    python scripts/make_readme_stage2.py --check    # exit 1 if README.md is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"
BEGIN, END = "<!-- STAGE2:BEGIN -->", "<!-- STAGE2:END -->"

RULE_NAMES = {"logodds_rise": "largest log-odds rise", "highest_price": "highest price"}
MODEL_NAMES = {"recal": "logistic recalibration of the price", "hgb_all": "gradient-boosted trees, all features"}


def _f(x, nd=3, sign=False) -> str:
    v = float(x)
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _ci(v, lo, hi, nd=3) -> str:
    return f"{_f(v, nd, True)} [{_f(lo, nd, True)}, {_f(hi, nd, True)}]"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def convergence_tables() -> tuple[str, dict]:
    c = pl.read_csv(RESULTS / "stage2_convergence.csv", infer_schema_length=0)
    head = c.filter((pl.col("group") == "overall") & (pl.col("delay") == "5") & (pl.col("cost_mult") == "1.0"))
    stress = c.filter(pl.col("group") == "stress")
    rows, n_pass = [], 0
    for r in head.iter_rows(named=True):
        s = stress.filter((pl.col("rule") == r["rule"]) & (pl.col("horizon") == r["horizon"]))
        s_ret = float(s["net_ret"][0]) if s.height else float("nan")
        ok = float(r["ret_lo"]) > 0 and s_ret > 0
        n_pass += ok
        rows.append(
            [
                RULE_NAMES[r["rule"]],
                {"7d": "7 days", "72h": "72 h", "48h": "48 h"}[r["horizon"]],
                r["n_events"],
                _f(r["win_rate"], 3),
                _f(r["mean_entry"], 3),
                _f(r["win_minus_entry"], 3, True),
                _ci(r["net_ret"], r["ret_lo"], r["ret_hi"]),
                _f(s_ret, 3, True) if s.height else "n/a",
                "PASS" if ok else "FAIL",
            ]
        )
    t = _table(
        [
            "Leader chosen by", "Bought at T minus", "Events", "Win rate", "Mean entry price",
            "Win rate minus entry", "Net return per dollar [95%]", "Doubled-cost stress", "Frozen rule",
        ],
        rows,
    )
    return t, {"n_cells": len(rows), "n_pass": n_pass, "events": [int(r["n_events"]) for r in head.iter_rows(named=True)]}


def markouts_tables() -> tuple[str, str, dict]:
    m = pl.read_csv(RESULTS / "stage2_markouts.csv", infer_schema_length=0)

    def row(section, metric, period=None, horizon=None):
        q = (pl.col("section") == section) & (pl.col("metric") == metric)
        if period is not None:
            q &= pl.col("period") == period
        if horizon is not None:
            q &= pl.col("horizon") == horizon
        return m.filter(q).row(0, named=True)

    g10 = row("persistence_decile_gross", "fit_decile_10")
    s10 = row("persistence_decile_spread_adj", "fit_decile_10")
    g01 = row("persistence_decile_gross", "fit_decile_01")
    gall = row("persistence_decile_gross", "all_eligible")
    sall = row("persistence_decile_spread_adj", "all_eligible")
    diff = row("persistence", "top_minus_bottom_decile")
    rho = row("persistence", "spearman_fit_shrunk_vs_tune_skill")
    elig = int(float(row("persistence", "n_eligible_wallets")["value"]))
    shuf = row("placebo", "shuffled_top_decile_mean")
    top5 = float(row("concentration", "top_decile_top5_wallet_usd_share")["value"])
    pos_top = float(row("concentration", "top_decile_share_wallets_positive")["value"])
    pos_bot = float(row("concentration", "bottom_decile_share_wallets_positive")["value"])
    passed = float(row("pass_rule", "PASS")["value"]) == 1.0
    persist = _table(
        ["Tune-period 24 h markout per dollar", "Gross [95%]", "After spread proxy [95%]"],
        [
            ["Wallets in the top fit-period skill decile", _ci(g10["value"], g10["lo95"], g10["hi95"]), _ci(s10["value"], s10["lo95"], s10["hi95"])],
            ["Wallets in the bottom fit-period skill decile", _ci(g01["value"], g01["lo95"], g01["hi95"]), "negative (see CSV)"],
            ["All eligible wallets", _ci(gall["value"], gall["lo95"], gall["hi95"]), _ci(sall["value"], sall["lo95"], sall["hi95"])],
        ],
    )
    rows = []
    for h, label in (("5m", "5 minutes"), ("1h", "1 hour"), ("6h", "6 hours"), ("24h", "24 hours"), ("res", "to resolution")):
        g = row("pooled_markout_per_dollar", "gross", "tune", h)
        s = row("pooled_markout_per_dollar", "spread_adjusted", "tune", h)
        rows.append([label, _ci(g["value"], g["lo95"], g["hi95"], 4), _ci(s["value"], s["lo95"], s["hi95"], 4)])
    pooled = _table(["Horizon", "Gross, all taker trades [95%]", "After spread proxy [95%]"], rows)
    skilled = {
        "mc_fit": float(row("skilled_share", "skilled_share_MC_BH", "fit")["value"]),
        "an_fit": float(row("skilled_share", "skilled_share_analytic_BH", "fit")["value"]),
        "n_fit_wallets": int(float(row("skilled_share", "skilled_share_MC_BH", "fit")["n"])),
        "fit_skilled": int(float(row("retention", "fit_skilled")["value"])),
        "fit_skilled_ge10": int(float(row("retention", "fit_skilled_with_tune_ge10")["value"])),
        "retained_positive": float(row("retention", "tune_positive_share")["value"]),
    }
    info = dict(
        elig=elig, rho=rho, diff=diff, shuf_p95=float(shuf["hi95"]), top5=top5, pos_top=pos_top, pos_bot=pos_bot,
        passed=passed, g10=g10, s10=s10, **skilled,
    )
    return persist, pooled, info


def forecast_table() -> tuple[str, dict]:
    f = pl.read_csv(RESULTS / "stage2_forecast.csv", infer_schema_length=0)
    imp = f.filter((pl.col("table") == "improvement") & pl.col("model").is_in(list(MODEL_NAMES)))
    rows, any_pass = [], False
    for h, hl in (("7d", "7 days"), ("3d", "3 days"), ("1d", "1 day")):
        for model in ("recal", "hgb_all"):
            cells, halves_ok = [], []
            for sample in ("JanMar", "AprJun"):
                r = imp.filter((pl.col("H") == h) & (pl.col("model") == model) & (pl.col("sample") == sample)).row(0, named=True)
                cells.append(_ci(r["dll"], r["dll_lo"], r["dll_hi"], 4))
                halves_ok.append(float(r["dll_lo"]) > 0)
            any_pass |= all(halves_ok)
            n = imp.filter((pl.col("H") == h) & (pl.col("model") == model) & (pl.col("sample") == "pooled")).row(0, named=True)
            rows.append([hl, MODEL_NAMES[model], f"{n['n']} / {n['events']}", _f(n["logloss_price"], 3), *cells])
    t = _table(
        ["Forecast time before end", "Model", "Test rows / events", "Price log-loss", "Improvement over price, Jan-Mar 2026 [95%]", "Improvement, Apr-Jun 2026 [95%]"],
        rows,
    )
    return t, {"any_pass": any_pass}


def render() -> str:
    conv, c = convergence_tables()
    persist, pooled, b = markouts_tables()
    fc, f = forecast_table()
    verdict_b = "passes" if b["passed"] else "does not pass"
    lines = f"""## Stage two: can any of this be traded?

Stage one produced no validated detector, so the natural next question was whether any signal in this data
is worth trading. I wrote the pass marks down **before running anything** (frozen on 2026-11-14 at 12:50 UTC in
[docs/STAGE2_PLAN.md](docs/STAGE2_PLAN.md), section 9) and ran three tests, all offline on the data already
collected. The short answer is **no**: the pre-registered rule stopped the strategy track. This is research,
**not financial advice**, and nothing here places or recommends a trade.

**Design.** Markets are assigned to periods by their close time: *fit* up to 2025-12-31, *tune* 2026-01-01 to
2026-06-30, and a *sealed* period from 2026-07-01 onward that was **never opened** (the code drops it at load
and a test enforces it; no hyperparameters were searched, so "tune" was used purely as an evaluation period).
Costs: entry at the next traded price after a 5-minute delay (1 and 30 minutes are also in the CSV), plus a
taker fee of rate x p x (1 - p) per share, with the rate set by category (0% geopolitics and world, 3% sports,
4% finance, politics, tech and mentions, 5% economics, culture, weather and other, 7% crypto). That fee schedule
comes from a secondary source and still needs confirming with the venue. A strategy passes only if the lower
95% bound of its net return per dollar (bootstrap resampling whole events) is above zero, it beats a placebo,
and it stays positive at doubled costs.

| Test | Verdict | In one line |
|---|---|---|
| A. Tradable convergence (buy the price-selected leader of a multi-outcome event days before the reveal) | **Fails**: {c['n_pass']} of {c['n_cells']} versions pass | The leader is not underpriced (win rate within about 0.05 of its entry price); net returns are negative or about zero |
| B. Wallet skill from markouts (do wallets whose trades moved the price their way keep doing so?) | **{verdict_b.capitalize()}** the frozen rule, narrowly | Top fit-period decile earns {_f(b['g10']['value'], 3, True)} per dollar over 24 h on the tune period ({_f(b['s10']['value'], 3, True)} after the spread proxy); it is concentrated and is not evidence that copying pays |
| C. Forecast benchmark (do flow, wallet-skill and impact features beat the market price?) | **Fails**: no gain in either half, at any horizon | The price is already well calibrated; the full model is worse than the price |

### A. Tradable convergence

At 7 days, 72 hours and 48 hours before the reveal of a multi-outcome scheduled event, buy the outcome that
looks like the leader **from prices alone** (largest log-odds rise over the prior window, or highest price; never
the eventual winner) and hold to resolution. Fit and tune periods only, 5-minute entry delay.

{conv}

None of the six versions has a lower bound above zero, so none passes (and none would after correcting for six
cells). The best-looking number is the highest-price leader at 7 days: its tune-period-only return is
positive but its fit-period return is not, and its pooled interval spans zero. A placebo that buys a random
non-leader matched on price exists for only part of the trades (the leader is often much pricier than every
sibling), so "beats the placebo" is weak evidence either way.

### B. Wallet skill from markouts

A **markout** is how far the price of the traded token moved in the trade's favour afterwards, per dollar, so it
needs neither the final outcome nor a news time. Wallets were classified on the fit period and followed into the
tune period ({b['elig']} wallets had at least 20 trades in each period).

{persist}

* Rank correlation of fit-period and tune-period skill: {_ci(b['rho']['value'], b['rho']['lo95'], b['rho']['hi95'])}.
  The shuffled-wallet baseline's 95th percentile for the top-decile mean is {_f(b['shuf_p95'], 3, True)}, below the observed
  {_f(b['g10']['value'], 3, True)}. All four frozen checks pass.
* The result is narrow: only {b['elig']} wallets are eligible, about 74 per decile; the **top 5 wallets hold
  {b['top5'] * 100:.0f}% of the top decile's dollars**; {b['pos_top'] * 100:.0f}% of the top decile's wallets have positive
  tune skill against {b['pos_bot'] * 100:.0f}% of the bottom decile's; and across all eligible wallets the markout is about zero
  after the spread proxy.
* Taker trades lose on average. Pooled over all taker trades in the tune period:

{pooled}

* **Share of wallets classed skilled.** With the pre-registered Monte-Carlo test (2,000 draws) and
  Benjamini-Hochberg correction, {b['mc_fit'] * 100:.2f}% of {b['n_fit_wallets']:,} fit-period wallets qualify; that zero is a
  resolution limit of the test (its smallest possible p-value is 1/2001), not proof of no skill. An analytic version
  without that limit classes {b['an_fit'] * 100:.2f}% ({b['fit_skilled']} wallets) as skilled. For comparison, a published
  study reports 3.14% of accounts as skilled with 44% staying skilled out of sample; the definitions differ, so this
  is descriptive only. Of the {b['fit_skilled_ge10']} fit-skilled wallets with at least 10 tune trades,
  {b['retained_positive'] * 100:.0f}% have positive tune skill, but none is classed skilled again.

Reading: some active wallets show persistent gross markout skill over this period. That is a statement about the
past, not evidence that copying them is profitable (the spread proxy absorbs most of it, the group is small and
concentrated, and a copier would trade later than the wallet did).

### C. Forecast benchmark (the stop gate)

At 7 days, 3 days and 1 day before a market's scheduled end, for open markets with at least 20 earlier taker
trades, predict the outcome using only data before that time. Models: a logistic recalibration of the price and
shallow gradient-boosted trees (fixed hyperparameters, no tuning) with price dynamics, order-flow imbalance, volume
surge, buyer concentration, fresh-wallet share, net flow of wallets classed skilled on the fit period, impact per
dollar, and market descriptors. Purged walk-forward validation (train on months before the test month with a 7-day
embargo). Positive improvement means the model beats the market price on log-loss.

{fc}

No model beats the price in both halves at any horizon, so the frozen rule fails and **the strategy track stops**
(the remaining candidate strategies, such as following informed flow or skilled wallets, were not run). The
all-features model is significantly worse than the price at 7 days. The price is well calibrated; feature-group
ablation and a stress test that drops the largest 10% of markets by volume agree
([results/stage2_forecast.md](results/stage2_forecast.md)).

### Caveats and what was not done

* The fee schedule is from a secondary source. There is no historical order book, so spreads and impact are
  inferred from the taker tape.
* About half of the leader trades in test A were skipped because no price traded within 6 hours, and the volume cap
  (1% of trailing 24-hour volume) shrank 91% of the stakes, so the backtest says little about size.
* The wallet-skill feature in the forecast is nearly empty (only 42 wallets qualify under the analytic rule), so
  test C is a weak test of that feature in particular.
* The builders added a few definitions beyond the frozen text, listed here for transparency: a 6-hour markout
  horizon with a 24-hour staleness limit; a spread proxy equal to the gap to the preceding opposite-side fill within
  10 minutes (else the market median, else the global median); a purge rule for training rows (closed at least 7 days
  before the test month); test rows must have forecast time on or after 2026-01-01; and "beats the placebo" read as
  exceeding the upper 95% limit of 200 price-matched placebo draws.
* `scripts/orderbook_recorder.py` (a polite live order-book recorder for a forward paper test) exists and is tested
  offline, but it was **never run**. Funding-chain data and the Hugging Face on-chain dataset were not used; the
  steps are written in [docs/STAGE2_PLAN.md](docs/STAGE2_PLAN.md) section 10.
* Reproduce: `pmdetect stage2 convergence`, `pmdetect stage2 markouts` and `pmdetect stage2 forecast` (they need the
  local data store); this section is generated from the result CSVs by `scripts/make_readme_stage2.py`.
"""
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if README.md differs from the rendered block")
    a = ap.parse_args(argv)
    text = README.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"README.md must contain {BEGIN} and {END}")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + BEGIN + "\n" + render().rstrip("\n") + "\n" + END + tail
    if a.check:
        return 0 if new == text else 1
    README.write_text(new, encoding="utf-8", newline="\n")
    print("README.md: stage-two section regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
