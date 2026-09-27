# Data

Raw and processed data are git-ignored (the minute bars are licensed from Databento). Everything can be
rebuilt with:

```bash
python scripts/download_data.py --databento-key YOUR_KEY   # data/raw/
python scripts/build_data.py                               # data/processed/
```

If the minute CSVs already exist elsewhere, point to them instead of downloading:
`PAIRS_MINUTE_DIR=/path/to/csvs python scripts/build_data.py` (and `download_data.py --skip-minute` for the free files).

| File | Source | Content |
|---|---|---|
| `raw/minute_bars/{TICKER}.csv` | Databento, dataset `XNAS.ITCH`, schema `ohlcv-1m` | 1-minute OHLCV bars, UTC timestamps (bar start), as-traded (unadjusted) prices, 100 tickers, 2020-11-06 to 2025-11-05 |
| `raw/splits.csv` | Yahoo Finance via `yfinance` | Split and spin-off ratios by ex-date; one extra split (NVO, 2-for-1, 2023-09-20) was found by the overnight-jump audit and added in `scripts/build_data.py` |
| `raw/yahoo_daily.csv.gz` | Yahoo Finance via `yfinance` | Daily closes used to cross-check the minute data, and SPY for market beta |
| `raw/factors_daily.csv` | Kenneth French data library | Daily Fama-French 5 factors, momentum and the risk-free rate |
| `processed/minute_{close,high,low,volume}.npy` | `scripts/build_data.py` | Regular-session minute panel (minutes x tickers), split-adjusted, NaN where no trade |
| `processed/calendar.pkl`, `tickers.json`, `split_factor_day.npy` | `scripts/build_data.py` | Minute grid, column order, and the factor that converts adjusted prices back to as-traded prices |

## Data-quality checks (all in `scripts/build_data.py`, results in `results/data_*.csv`)
- **Session hours.** The vendor files end at 20:00 UTC. That is 16:00 New York time in summer but 15:00 in winter, so the last trading hour is missing on standard-time days. Each day's session end is detected from the data, and bars never span the gap.
- **Splits.** Every overnight move larger than 45% is listed in `results/data_split_audit.csv`. Each is either explained by a split in the split file or checked as a genuine price move.
- **Recycled symbols.** The `META` symbol belonged to a Roundhill ETF until 9 June 2022. The `BHVN` symbol passed from the old Biohaven, bought by Pfizer, to its spin-off on 4 October 2022. Data before those dates is dropped.
- **Cross-check.** On full-session days, the last regular-session trade matches Yahoo's official close to within a median of a few basis points (`results/data_summary.csv`).
- **Single venue.** The data is Nasdaq-only (`XNAS.ITCH`). Prices are representative, but volume is roughly 15-20% of consolidated volume, so the liquidity filter uses a correspondingly lower dollar-volume threshold.
