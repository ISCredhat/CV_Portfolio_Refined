"""Project-wide paths and research parameters (one place to change every assumption)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
# Databento 1-minute OHLCV CSVs, one per ticker (set PAIRS_MINUTE_DIR to use a folder elsewhere)
MINUTE_DIR = Path(os.environ.get("PAIRS_MINUTE_DIR", DATA_RAW / "minute_bars"))
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

TZ = "America/New_York"
RTH_START_MIN = 9 * 60 + 30                    # 09:30 ET
RTH_END_MIN = 16 * 60                          # 16:00 ET (bars start at 09:30 ... 15:59)
HALF_DAY_END_MIN = 13 * 60                     # 13:00 ET on NYSE early-close days
# NYSE early closes inside the sample (13:00 ET). After-hours prints on these days are dropped.
EARLY_CLOSES = ["2020-11-27", "2020-12-24", "2021-11-26", "2022-11-25", "2023-07-03",
                "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03"]

# Symbols that were reused by a different security inside the sample: data before the date is dropped.
SYMBOL_REUSE = {
    "META": ("2022-06-09", "Roundhill Metaverse ETF traded as META until Facebook took the symbol"),
    "BHVN": ("2022-10-04", "old Biohaven was bought by Pfizer; the symbol passed to the spin-off"),
}
CORE_TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "AMD", "GOOGL", "INTC", "CSCO", "PYPL",
                "QCOM", "MU", "BAC", "PFE", "CMCSA"]   # used to detect each day's session end

# --- industry groups ----------------------------------------------------------------------
# Pairs are only formed *within* a group (economic link first, statistics second).
# Groups are hand-labelled from each company's main line of business.
INDUSTRY = {
    "Semiconductors": ["AMD", "AVGO", "INTC", "LRCX", "MCHP", "MRVL", "MU", "NVDA", "NVTS", "ON", "QCOM"],
    "Bitcoin miners / hosting": ["APLD", "BITF", "CAN", "CIFR", "CLSK", "IREN", "MARA", "RIOT", "WULF"],
    "Interactive media": ["GOOG", "GOOGL", "META", "PINS", "SNAP"],
    "Internet retail": ["AMZN", "CPNG"],
    "Software": ["MSFT", "PLTR", "FTNT", "U", "APPS", "SOUN"],
    "Tech hardware": ["AAPL", "CSCO", "SMCI"],
    "Fintech": ["PYPL", "SOFI", "HOOD", "UPST"],
    "Banks": ["BAC", "HBAN"],
    "Pharma": ["PFE", "NVO", "TEVA", "VTRS"],
    "Biotech": ["MRNA", "SRPT", "BHVN", "RXRX"],
    "Solar": ["ENPH", "SEDG", "CSIQ"],
    "EVs, autos & batteries": ["TSLA", "RIVN", "F", "GT", "QS", "SLDP", "AUR", "ECX"],
    "Quantum computing": ["QBTS", "QUBT", "RGTI"],
    "Packaged food & drinks": ["KHC", "MDLZ", "KDP", "BYND"],
    "Household products": ["KMB", "KVUE"],
    "Ride-hailing": ["LYFT", "GRAB"],
    "Media & cable": ["CMCSA", "WBD"],
    "Metals & mining": ["VALE", "BTG"],
    "AI cloud": ["CRWV", "NBIS"],
    "Travel": ["AAL", "HTZ", "HST"],
    "Hydrogen & EV charging": ["PLUG", "NVVE"],
}
GROUP_OF = {t: g for g, ts in INDUSTRY.items() for t in ts}
UNGROUPED = ["ASST", "CHR", "CSX", "DKNG", "DVLT", "EXC", "LNAI", "MTC", "NFE", "ONDS", "OPEN", "RKLB",
             "RUBI", "SBUX", "SMX", "TREX", "YDKG"]            # in the data, used only in the "any industry" test
ALL_TICKERS = sorted(set(GROUP_OF) | set(UNGROUPED))
DATA_START, DATA_END = "2020-11-06", "2025-11-06"

# --- walk-forward design ------------------------------------------------------------------
FORMATION_MONTHS = 6            # test for cointegration on the previous 6 months ...
SELECTION_BAR = "30min"         # ... of 30-minute bars (test power depends on span, not bar size)
FIRST_TRADE_MONTH = "2021-05"   # first month with a full 6-month formation window
LAST_TRADE_MONTH = "2025-10"    # last full month of data
VALIDATION_END = "2022-12"      # parameters are chosen on 2021-05..2022-12 only
                                # and reported out of sample on 2023-01..2025-10

# --- universe filter (point-in-time, evaluated on each formation window) --------------------
MIN_DAYS_COVERAGE = 0.95        # traded on >= 95% of formation days
MIN_PRICE = 3.0                 # median price >= $3
MIN_DOLLAR_VOLUME = 5e6         # median daily *Nasdaq* $ volume >= $5M (~$25M consolidated)

# --- pair selection --------------------------------------------------------------------------
ALPHA = 0.05                    # pair-level significance for the Engle-Granger test
FDR = 0.05                      # Benjamini-Hochberg false discovery rate (reported, see README)
JOHANSEN_LEVEL = "5%"           # Johansen trace test must also reject r = 0
HALF_LIFE_HOURS = (2.0, 65.0)   # OU half-life between 2 trading hours and 10 trading days
MAX_HURST = 0.5                 # spread must be mean-reverting (H < 0.5)
HEDGE_RATIO_RANGE = (0.2, 5.0)  # discard pairs whose formation hedge ratio is implausible
N_PAIRS = 5                     # capital is split into 5 equal pair slots
MAX_PAIRS_PER_TICKER = 2        # diversification: a ticker can appear in at most 2 pairs

# --- trading rules -----------------------------------------------------------------------------
BAR_SIZES = ["1min", "5min", "15min", "30min", "1h", "1D"]
HEDGE_METHODS = ["static", "rolling", "kalman"]
ENTRY_Z = [1.5, 2.0, 2.5]
EXIT_Z = [0.0, 0.5]
STOP_Z_ADD = 2.0                # stop loss when |z| > entry_z + 2
TIME_STOP_HALF_LIVES = 3.0      # close a trade after 3 formation half-lives
ROLLING_WINDOW_DAYS = 10        # rolling-OLS hedge window (at least 20 bars)
KALMAN_MEMORY_DAYS = 10         # Kalman state noise set so the filter "remembers" ~10 days at any bar size
LEVERAGE = 2.0                  # $1 long + $1 short per $1 of slot capital (Reg-T style)
LATENCY_MIN = 1                 # orders fill at the close of the next 1-minute bar

# --- costs -------------------------------------------------------------------------------------
COMMISSION_PER_SHARE = 0.0035   # IBKR-style per-share commission
TICK = 0.01                     # US equities quote in cents: the spread is at least one tick
IMPACT_BPS = 1.0                # extra slippage / impact per leg, one way
MAX_HALF_SPREAD_BPS = 50.0      # cap on the estimated half-spread
BORROW_FEE_ANNUAL = 0.01        # 1% p.a. on the short leg's value
COST_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# --- statistics ------------------------------------------------------------------------------
TRADING_DAYS = 252
BOOTSTRAP_BLOCK = 21
