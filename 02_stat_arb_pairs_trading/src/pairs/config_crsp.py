"""Settings for the CRSP (WRDS) daily study: Part A of the README.

The minute-bar Databento study (Part B) keeps using ``config.py``. Everything here is read-only for the
build agents. The values that the pre-registration freezes are mirrored in ``config/preregistration.yaml``.
``prereg.py`` asserts that the two agree.

Plan: /mnt/project-files/plans/02_stat_arb_wrds_PLAN.md (v2, section 12 overrides earlier sections).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRDS_RAW = Path(os.environ.get("PAIRS_WRDS_DIR", ROOT / "data" / "raw" / "wrds"))   # yearly parquet chunks
CRSP_PROC = ROOT / "data" / "processed" / "crsp"
REFERENCE = ROOT / "data" / "reference"
CONFIG_DIR = ROOT / "config"
PREREG_YAML = CONFIG_DIR / "preregistration.yaml"
RESULTS_CRSP = ROOT / "results" / "crsp"

# --- download -------------------------------------------------------------------------------
DOWNLOAD_START = "1961-07-01"          # GGR replication needs 1962-07 onward; main study needs 1989-01 onward
DOWNLOAD_END = "2025-12-31"

# --- periods ----------------------------------------------------------------------------------
MAIN_FIRST_TRADE_MONTH = "1990-01"
MAIN_LAST_TRADE_MONTH = "2025-12"
PERIODS = {                                      # label -> (first trade month, last trade month)
    "backward_oos": ("1990-01", "2020-10"),      # H1 window: the frozen configuration never traded it
    "seen": ("2020-11", "2022-12"),              # formation and selection data of the original study
    "forward": ("2023-01", "2025-12"),           # observed before on 100 stocks; new universe here
}
# GGR / Do & Faff (2010) replication: D&F's trading months. CRSP daily began on 1962-07-02 when GGR and D&F used it,
# so their first 12-month formation ran 1962-07 to 1963-06 and the first trading month was 1963-07 (GGR 1999 Table
# III starts 8/1963). Today's CRSP goes back to 1925, so the start is set here rather than taken from the data.
GGR_FIRST_TRADE_MONTH = "1963-07"
GGR_LAST_TRADE_MONTH = "2009-06"

# --- universe (point in time, decided at formation end only) ------------------------------------
UNIVERSE_TOP_N = 500
UNIVERSE_TOP_N_SENSITIVITY = [100, 1000, 3000]
MIN_PRICE_LAST_DAY = 5.0               # raw closing price on the last formation day
MIN_TRADED_SHARE = 0.95                # traded (trade-price flag and volume > 0) on >= 95% of formation days
SHARE_FILTER = {                       # applied from stocknames_v2 spells valid on the formation-end date
    "securitytype": ["EQTY"],
    "securitysubtype": ["COM"],
    "sharetype": ["NS"],
    "usincflg": ["Y"],
    "issuertype": ["ACOR", "CORP"],
    "primaryexch": ["N", "A", "Q"],
    "conditionaltype": ["RW"],
    "tradingstatusflg": ["A"],
}
EXCLUDE_SAME_PERMCO = True             # share classes of one company are reported separately, never traded

# --- industries --------------------------------------------------------------------------------
INDUSTRY_PRIMARY = "ff48"              # from CRSP historical SIC as of formation end
INDUSTRY_SENSITIVITY = ["gics_ind", "sic2", "none"]
FF48_EXCLUDE = [48]                    # "Other" (and unmapped SIC) never forms within-industry pairs

# --- selection (daily bars) -----------------------------------------------------------------------
FORMATION_MONTHS = 6
FORMATION_MONTHS_SENSITIVITY = [12]
MIN_OBS_SHARE = 0.8                    # a pair needs >= 0.8 x formation days of joint observations
ALPHA = 0.05
FDR = 0.05
HALF_LIFE_DAYS = (1.0, 10.0)
MAX_HURST = 0.5
HEDGE_RATIO_RANGE = (0.2, 5.0)
EG_MAXLAG = 12
N_PAIRS = 5
N_PAIRS_SECONDARY = 20                 # pre-registered secondary series
N_PAIRS_SENSITIVITY = [50]
MAX_PAIRS_PER_STOCK = 2

# --- frozen trading configuration (chosen on 2021-22 Databento data) ---------------------------------
FROZEN = {"bar": "1D", "method": "rolling", "entry": 2.0, "exit": 0.5}
STOP_Z_ADD = 2.0                       # stop when |z| >= entry + 2
TIME_STOP_HALF_LIVES = 3.0             # time stop = 3 x hl_days trading days
ROLLING_WINDOW_BARS = 20               # what the current code uses for daily bars: max(20, 10 x bars_per_day)
KALMAN_MEMORY_DAYS = 10
LEVERAGE = 2.0
VARIANTS = {"method": ["static", "rolling", "kalman"], "entry": [1.5, 2.0, 2.5], "exit": [0.0, 0.5]}   # 18

# --- execution --------------------------------------------------------------------------------------
EXEC_PRIMARY = "t1_close"              # H1 is tested on this: defined every day since 1926 (= GGR's one-day wait)
EXEC_CO_REPORTED = "t1_open"           # from 1993-01; open fills pay OPEN_SPREAD_MULT x the closing half-spread
EXEC_NO_DELAY = "t_close"
OPEN_FIRST_MONTH = "1993-01"
OPEN_SPREAD_MULT = 1.5

# --- delisting ----------------------------------------------------------------------------------------
# Missing delisting return for performance delistings (dlstcd 500-599): Shumway (1997), Shumway & Warther (1999)
DELIST_IMPUTE = {"N": -0.30, "A": -0.30, "Q": -0.55}
DELIST_IMPUTE_BOUNDS = [0.0, -1.0]
MERGER_CODES = (200, 399)              # tagged for attribution only, never filtered

# --- costs ----------------------------------------------------------------------------------------------
# Tick size by exchange: list of (first effective date, tick in $). Before the first date: $1/8.
TICKS = {
    "N": [("1900-01-01", 0.125), ("1997-06-24", 0.0625), ("2001-01-29", 0.01)],
    "A": [("1900-01-01", 0.125), ("1997-05-07", 0.0625), ("2001-01-29", 0.01)],
    "Q": [("1900-01-01", 0.125), ("1997-06-02", 0.0625), ("2001-04-09", 0.01)],
}
# One-way commission in bps of traded value, institutional trades: Do & Faff (2012, J. Financial Research 35(2),
# Table 2), = 80% of Jones (2002) for 1963-2000, ITG (2008, 2010) for 2005-09, 10 bps interpolated for 2001-04.
# Years after 2009 hold the 2009 value (9 bps): conservative, since no later estimate is pre-registered. Years before
# 1963 use the 1963 value. Flat $0.0035/share is a sensitivity.
COMMISSION_BPS = [(1963, 70), (1964, 71), (1965, 71), (1966, 66), (1967, 67), (1968, 68), (1969, 64), (1970, 63), (1971, 66), (1972, 61), (1973, 56), (1974, 72), (1975, 68), (1976, 57), (1977, 54), (1978, 55), (1979, 48), (1980, 45), (1981, 41), (1982, 33), (1983, 28), (1984, 26), (1985, 25), (1986, 22), (1987, 20), (1988, 19), (1989, 20), (1990, 20), (1991, 21), (1992, 19), (1993, 17), (1994, 17), (1995, 16), (1996, 14), (1997, 10), (1998, 10), (1999, 10), (2000, 10), (2001, 10), (2002, 10), (2003, 10), (2004, 10), (2005, 10), (2006, 9), (2007, 7), (2008, 8), (2009, 9)]
COMMISSION_BPS_SOURCE = "Do & Faff (2012) Table 2, institutional trades"
COMMISSION_FLAT_SENSITIVITY = 0.0035
IMPACT_BPS = 1.0
BORROW_FEE_ANNUAL = 0.01
BORROW_FEE_HARD = 0.05                 # top-1000/3000 only: shorts below the NYSE 20th size percentile
SPREAD_SOURCE_PRIMARY = "quoted"       # half of the median closing quoted spread over the previous month
SPREAD_QUOTED_FACTOR_SENSITIVITY = 0.5
SPREAD_FALLBACK_MULT = 1.25            # missing estimate -> 1.25 x median of same exchange x price decile
QUOTES_RELIABLE_FROM = "1993-01"       # before this NYSE/AMEX use calibrated Abdi-Ranaldo / Corwin-Schultz
COST_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# --- short-sale ban 2008 ------------------------------------------------------------------------------
SHORT_BAN = ("2008-09-19", "2008-10-08")
SHORT_BAN_FF48 = [44, 45, 46, 47]      # banks, insurance, real estate, trading (proxy for the SEC list)

# --- statistics ------------------------------------------------------------------------------------------
TRADING_DAYS = 252
BOOT_REPS = 2000
BOOT_MEAN_BLOCK = 21
BOOT_BLOCK_SENSITIVITY = [5, 63]
NULL_REPS_PER_PAIR = 1
H2_MIN_EXCESS_PP = 1.0
NW_LAGS_MONTHLY = 6
NW_LAGS_TREND = 12
TREND_BREAK = "2001-04"
SEED = 20261118
N_PLACEBO = 100
