# Data — 01 Insider Trading Alpha

Raw and processed data are git-ignored. Everything can be rebuilt with two commands:

```bash
python scripts/download_data.py --email you@example.com   # -> data/raw/
python scripts/run_pipeline.py --stage build              # -> data/processed/
```

| File / folder | Source | Notes |
|---|---|---|
| `raw/sec_zips/*_form345.zip` | [SEC insider transactions data sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets) (Forms 3/4/5, quarterly) | About 1 GB. Can be deleted after parsing (`--delete-zips`). |
| `raw/sec_parsed/form4_ps_<quarter>.csv.gz` | Parsed from the zips | Open-market purchases (P) and sales (S) only, joined to filer and issuer details |
| `raw/form4_ps_all.csv.gz` | All quarters combined | About 2M transaction lines, 2010 to the latest published quarter |
| `raw/sec_company_tickers.json` | SEC CIK-to-ticker map | Used to recover prices for companies that changed ticker |
| `raw/prices/batch_*.csv.gz` | Yahoo Finance via `yfinance` | Daily open, split-adjusted close, total-return adjusted close, volume, splits and dividends |
| `raw/prices_manifest*.csv` | Download log | Which tickers returned data. A `clean=True` failure means the ticker really has no data, not that the download was rate-limited. |
| `raw/factors/*.zip` | [Kenneth French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Fama-French 5 factors plus momentum, daily |
| `processed/` | `run_pipeline.py --stage build` | Event tables (pickled), abnormal-return paths and price matrices |

**Known gap:** Yahoo Finance does not serve most delisted stocks, so they drop out of the sample. `results/data_coverage.csv` shows how many purchase events survive each step. CRSP via WRDS would close this gap, and the loaders only need a date × ticker price table.
