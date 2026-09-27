# Insider Trading Alpha

**Do corporate insiders' open-market purchases predict stock returns, and can an outsider still profit from them after costs?**

> **Headline:** insider purchases are informative (**+0.58% abnormal return in 5 days, t = 11.9**), but +1.3% is priced in before an outsider can trade. A walk-forward gradient-boosting model ranks them out of sample (AUC 0.530, rank IC 0.068), and its real value is as a filter: the buys it ranks worst earn **-14.8% p.a. FF5+momentum alpha (t = -3.0)**. Copying every insider buy earns no alpha after costs (-1.1% p.a., t = -0.6).

| Data | Signal decay | ML model (out of sample) | Tradable result | Copy-all-buys |
|---|---|---|---|---|
| 1,970,299 Form 4 lines, 2010-01-04 to 2026-03-31 | +0.58% in 5 days (t = 11.9); +0.24% at 3 months (t = 0.8) | AUC 0.530, top-bottom decile spread 6.5 pp over 3 months | Worst-quintile buys: -14.8% p.a. alpha (t = -3.0) | -1.1% p.a. alpha (t = -0.6), net |

An end-to-end, point-in-time research pipeline built on **1,970,299 SEC Form 4 transaction lines** (every open-market purchase and sale filed 2010-01-04 to 2026-03-31). It covers event studies, insider-skill tests, a walk-forward machine-learning ranking model and calendar-time portfolio backtests with realistic costs.

> Every number below is produced by `python scripts/run_pipeline.py` and inserted by `scripts/make_readme.py`. Nothing is typed by hand.

## Key findings

| | Result |
|---|---|
| **Insider purchases carry information** | Stocks earn **+0.58%** abnormal return (vs IWM) in the 5 days after we can trade (t = 11.9). This is significant in 13 of 13 calendar years. |
| **...but most of it is gone before an outsider can act** | **+0.55%** accrues between the insider's trade and the filing, and another **+0.73%** overnight after the filing, so **+1.3%** is priced before the next open. |
| **The longer-run drift is weak** | Abnormal return is +0.24% at 3 months (t = 0.8) and -0.12% at 6 months (t = -0.3). Sales carry no signal (+0.15%, t = 0.5). |
| **Who is informative** | *Opportunistic* insiders (Cohen, Malloy & Pomorski 2012) drift for months: +0.88% at 3 months (t = 2.6), +1.18% at 6 months (t = 2.6). |
| **ML separates winners from losers** | Walk-forward gradient boosting reaches out-of-sample AUC 0.530 and rank IC 0.068. The top score decile earns +2.2% and the bottom decile -4.3% over 3 months, a **6.5 pp spread**. |
| **The strongest tradable result is knowing what to avoid** | Insider buys in the model's bottom quintile have FF5+momentum alpha of **-14.8% p.a. (t = -3.0)**. |
| **Naively copying insiders doesn't pay** | Buying every investable insider purchase gives FF5+momentum alpha of -1.1% p.a. (t = -0.6) after costs. A 5-day hold has Sharpe 1.27 gross but only 0.01 net. |
| **Long-only ML portfolio** | Top-quintile portfolio: Sharpe 0.46 (95% CI [-0.19, 1.15]) vs 0.49 for IWM, alpha +1.2% (t = 0.3) and market beta 0.94. It's mostly small-cap beta. |
| **Market-neutral ML spread** | Long top / short bottom quintile: Sharpe 0.65 gross, **0.26 net** of short-side costs and 3% borrow fee, alpha +4.9% (t = 0.8), beta 0.04. |

**Bottom line:** the insider signal is real and statistically robust, but it is fast and small. After you account for when outsiders can actually trade, transaction costs and factor exposures, it is most useful as a **filter**: avoiding the insider purchases the model flags as losers. It is not a stand-alone long-only strategy.

## Pipeline

```mermaid
flowchart LR
  A["SEC EDGAR Form 4 data sets<br/>(quarterly, 2010-2026)"] --> B["Clean + aggregate<br/>insider x issuer x filing day"]
  C["Yahoo Finance daily prices<br/>(raw + adjusted, splits)"] --> D["Ticker validation:<br/>market price vs Form 4 price"]
  B --> D --> E["Entry = next open after filing<br/>+ point-in-time features"]
  E --> F["Event study<br/>(BHAR vs IWM / SPY)"]
  E --> G["Walk-forward ML<br/>(retrain quarterly)"]
  G --> H["Calendar-time portfolios<br/>+ liquidity-based costs"]
  I["Fama-French 5 + momentum"] --> H
```

**Data:** 224,096 purchase events and 628,422 sale events (2010 to 2026). 127,767 purchases have validated prices and a tradeable entry; see `results/data_coverage.csv`.

- **Source.** Filings come from the SEC's official Form 3/4/5 data sets, so the sample is complete and reproducible. Only original (not amended) Form 4s are used, and only open-market purchases (code `P`) and sales (code `S`) of common stock. Funds, SPACs and preferred stock are excluded.
- **One event.** An event is one insider buying one company's stock on one filing day; partial fills are combined.
- **Ticker matching.** Tickers are matched using both the symbol on the filing and the company's current ticker, which catches renames such as FB to META. A match is **accepted only if the market price on the trade date is within 30% of the price the insider reported.** This rejects recycled tickers and bad split data.
- **Coverage.** 57% of purchase events end up with validated prices and a tradeable entry. Of the rest, 92% are companies that no longer exist (delisted or acquired), which Yahoo doesn't serve (see Limitations).

## 1 · Event study

![Abnormal return by role](results/figures/fig1_car_by_role.png)
![Buys vs sells](results/figures/fig2_buys_vs_sells.png)

Mean buy-and-hold abnormal return vs IWM from the open of the session **after** the filing date. Values are winsorised at 1%/99%; t-statistics are clustered by filing month.

| Group | Events | +5 days (t) | +21 days (t) | +63 days (t) | +126 days (t) |
|---|---:|---:|---:|---:|---:|
| All purchases | 108,751 | +0.58% (11.9) | +0.47% (3.1) | +0.24% (0.8) | -0.12% (-0.3) |
| CEO/CFO | 26,542 | +0.73% (9.2) | +0.61% (3.1) | +0.28% (0.6) | +0.08% (0.1) |
| Director | 51,351 | +0.49% (9.8) | +0.45% (2.8) | +0.54% (2.0) | +0.63% (1.6) |
| 10% owner | 11,448 | +0.70% (5.6) | +0.19% (0.6) | -0.72% (-1.0) | -2.43% (-2.8) |
| Cluster buy (>=2 insiders, 30d) | 64,799 | +0.62% (10.2) | +0.50% (2.9) | +0.32% (0.8) | -0.24% (-0.5) |
| Opportunistic trader | 22,345 | +0.53% (8.2) | +0.59% (3.7) | +0.88% (2.6) | +1.18% (2.6) |
| Routine trader | 6,888 | +0.22% (2.3) | +0.19% (0.7) | +0.77% (1.3) | +0.74% (0.8) |
| Trade size >$1M | 9,356 | +0.97% (7.3) | +0.64% (2.1) | -0.20% (-0.3) | -1.39% (-1.9) |
| All sales | 370,719 | -0.05% (-1.5) | +0.02% (0.2) | +0.15% (0.5) | +0.45% (1.0) |

**Who captures the return?** All purchase events:

| Window | Mean abnormal return | t-stat |
|---|---:|---:|
| Insider's trade date -> filing date close | +0.55% | 21.9 |
| Filing-day close -> next open (overnight) | +0.73% | 25.8 |
| Entry open -> day 5 close | +0.58% | 11.9 |
| Entry open -> day 21 close | +0.47% | 3.1 |
| Entry open -> day 63 close | +0.24% | 0.8 |

![Signal by year](results/figures/fig9_signal_by_year.png)

## 2 · Is insider skill persistent?

Do some insiders have persistent skill? An insider's track record uses **only earlier purchases whose 3-month outcome window had already closed** before the new filing.

- Insiders in the top quintile of past performance earn +0.63% (t = 0.9) on their next purchase, against -0.65% for the bottom quintile. The Q5 minus Q1 spread has t = 1.0.
- In Fama-MacBeth regressions with controls, a one-standard-deviation better track record adds +0.21% (t = 0.9).
- Skill persistence is therefore weak: measured point-in-time, an insider's past winners say little about their next purchase.

![Skill persistence](results/figures/fig3_skill_persistence.png)

## 3 · Walk-forward ML ranking

- **Setup.** 31 point-in-time features, grouped as:
  - trade: size, holdings change, filing lag, indirect ownership
  - insider: role, routine vs opportunistic, track record
  - activity: cluster buying, insider selling
  - stock: momentum, volatility, 52-week range, liquidity
  - market: SPY return and volatility
- **Label.** 63-day abnormal return > 0.
- **Retraining.** Every quarter, on labels that were already realised. The first prediction is for 2016-01-05.
- **Trading rule.** The top-20% threshold comes from the training-period score distribution, so no future information is used.
- **Model choice.** The primary model (gradient boosting) was fixed in `config.py` before any out-of-sample results were seen.

| Model | OOS events | AUC | Rank IC | Top-bottom quintile spread (63d) |
|---|---:|---:|---:|---:|
| logit | 65,115 | 0.516 | 0.053 | +2.6% |
| adaboost | 65,115 | 0.517 | 0.045 | +1.8% |
| gbm | 65,115 | 0.530 | 0.068 | +4.9% |

![Score deciles](results/figures/fig5_score_deciles.png)
![Feature importance](results/figures/fig6_feature_importance.png)

## 4 · Portfolio backtest (2016-01-05 to 2026-07-01)

- **Entry and hold.** Enter at the next open after the filing and hold 63 sessions; positions drift with their own prices, and the portfolio is not rebalanced daily.
- **Costs.** One-way costs by 20-day average dollar volume: **60 bps** (under $1M/day) down to **5 bps** (over $100M/day), charged on both entry and exit.
- **Universe.** Price at least $2 and dollar volume at least $250k per day.
- **Statistics.** Sharpe confidence intervals come from a block bootstrap. The deflated Sharpe ratio adjusts for all variants tried; the ML top-quintile's deflated Sharpe gives a 70% probability that its true Sharpe is above zero.

| Strategy (net of costs) | CAGR | Vol | Sharpe | Sharpe 95% CI | Max DD | FF5+Mom alpha p.a. (t) | Market beta |
|---|---:|---:|---:|---:|---:|---:|---:|
| ML top 20% | 11.5% | 27% | 0.46 | [-0.19, 1.15] | -57% | +1.2% (0.3) | 0.94 |
| All investable buys | 9.7% | 23% | 0.42 | [-0.20, 1.08] | -48% | -1.1% (-0.6) | 0.93 |
| Opportunistic buys | 11.2% | 23% | 0.48 | [-0.14, 1.14] | -44% | +0.4% (0.2) | 0.89 |
| Cluster buys | 9.3% | 24% | 0.40 | [-0.24, 1.04] | -49% | -1.2% (-0.5) | 0.91 |
| CEO/CFO buys | 7.5% | 25% | 0.33 | [-0.32, 1.03] | -51% | -2.6% (-0.9) | 0.93 |
| Track-record insiders (point-in-time) | 8.6% | 23% | 0.38 | [-0.24, 1.02] | -51% | -2.3% (-0.9) | 0.91 |
| ML bottom 20% | -5.9% | 28% | -0.16 | [-0.75, 0.45] | -85% | -14.8% (-3.0) | 0.91 |
| Long-short: ML top - bottom 20% | 5.6% | 21% | 0.26 | [-0.31, 0.91] | -46% | +4.9% (0.8) | 0.04 |
| IWM (buy & hold) | 11.5% | 23% | 0.49 |  | -41% | -1.4% (-1.7) | 1.00 |
| SPY (buy & hold) | 15.2% | 18% | 0.76 |  | -34% | -0.2% (-0.5) | 0.99 |

![Equity curves](results/figures/fig4_equity_curves.png)
![Long-short](results/figures/fig8_long_short.png)
![Cost sensitivity](results/figures/fig7_cost_sensitivity.png)

**Holding-period robustness.** Net Sharpe (gross in brackets). This analysis was run after seeing the event study, so treat it as descriptive rather than a strategy choice:

| Strategy | 5d net (gross) | 10d net (gross) | 21d net (gross) | 63d net (gross) | 126d net (gross) |
|---|---:|---:|---:|---:|---:|
| All investable buys | 0.01 (1.27) | 0.30 (0.94) | 0.30 (0.61) | 0.42 (0.53) | 0.43 (0.49) |
| ML top 20% | 0.52 (1.18) | 0.39 (0.80) | 0.44 (0.69) | 0.46 (0.55) | 0.46 (0.50) |
| Opportunistic buys | -0.05 (1.13) | 0.35 (0.96) | 0.42 (0.75) | 0.48 (0.60) | 0.49 (0.55) |

## Biases handled

| Pitfall | How this project handles it |
|---|---|
| Look-ahead in features | Every feature is point-in-time (track records use only closed outcome windows); unit tests enforce it (`tests/`) |
| Trading before the information is public | Entry at the open of the session *after* the filing date |
| Wrong or recycled tickers | A ticker is accepted only if the market price matches the price the insider reported (within 30%) |
| Overlapping returns inflating the Sharpe ratio | Daily calendar-time portfolio P&L |
| Ignoring costs and factor exposure | IWM/SPY benchmarks, Fama-French 5 + momentum alpha, liquidity-based costs, cost sensitivity |
| Data snooping across variants | Walk-forward only; model fixed in advance; deflated Sharpe over all variants tried |

## Limitations

- **Survivorship.** Free Yahoo data lacks most delisted stocks: 92% of the unpriced purchase events are at companies that no longer exist. The sample is therefore biased towards survivors, which likely *flatters* the long-only results. The negative bottom-quintile result is probably understated rather than overstated, since delisted firms tend to be the worst performers. CRSP (via WRDS) would fix this, and the loaders only need a date × ticker price table.
- **Filing time.** Filing timestamps are daily. Entering at the next open is conservative for after-hours filings but gives up intraday reactions.
- **Mixed transactions.** Code `P` also includes some private placements. 10b5-1 plan flags only exist from 2023, so they are not used.
- **Costs.** The cost model is a schedule, not a market-impact model. The short leg assumes shares can be borrowed at 3% p.a.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py --email you@example.com   # ~1 h: SEC data sets, prices, factors -> data/raw/
python scripts/run_pipeline.py                            # build + analyse -> results/
python scripts/make_readme.py                             # refresh README numbers
python tests/test_pipeline.py                             # or: pytest tests
```

```
src/insider_alpha/
  events.py     Form 4 cleaning, event aggregation, ticker validation, entry timing
  features.py   point-in-time stock, activity, role and track-record features
  returns.py    event return paths and BHARs
  model.py      walk-forward training, OOS metrics, permutation importance
  backtest.py   calendar-time portfolio with liquidity-based costs
  stats.py      Newey-West, clustered t-stats, bootstrap and deflated Sharpe
  analysis.py   event study, skill persistence, models, backtests, figures
notebooks/      walkthrough notebook (reads results/)
results/        CSV tables + figures (committed)
tests/          look-ahead and P&L unit tests + synthetic end-to-end test
```

## References

- Lakonishok & Lee (2001), *Are Insider Trades Informative?*, Review of Financial Studies.
- Cohen, Malloy & Pomorski (2012), *Decoding Inside Information*, Journal of Finance.
- Lyon, Barber & Tsai (1999), *Improved Methods for Tests of Long-Run Abnormal Stock Returns*, Journal of Finance.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management.
- López de Prado (2018), *Advances in Financial Machine Learning*.
