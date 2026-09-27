"""Project-wide paths and research parameters (one place to change assumptions)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# --- sample -----------------------------------------------------------------
TRACK_RECORD_START = "2010-01-01"   # filings used only to build insider track records
SAMPLE_START = "2013-01-01"         # first filing date in the analysis sample
WALK_FORWARD_START = "2016-01-01"   # first out-of-sample month for the ML model
BENCHMARK = "IWM"                   # small-cap benchmark (insider buys are mostly small caps)
MARKET = "SPY"

# --- event study ------------------------------------------------------------
HORIZONS = [1, 5, 21, 63, 126]      # trading days after entry
PLOT_HORIZON = 126
LABEL_HORIZON = 63                  # ~3 months: holding period and ML label horizon

# --- data hygiene -----------------------------------------------------------
PRICE_MATCH_TOL = 0.30              # |log(Form 4 price / market price)| to accept a ticker match
MAX_ENTRY_SEARCH_DAYS = 5           # stock must trade within 5 sessions after filing
MIN_PRICE = 2.0                     # investable universe: price >= $2 at filing
MIN_ADV = 250_000                   # investable universe: 20d avg dollar volume >= $250k

# --- costs: one-way cost in basis points by 20d average dollar volume ---------
# (half-spread + impact; deliberately conservative for small caps)
COST_SCHEDULE = [          # (ADV upper bound in $, one-way cost in bps)
    (1_000_000, 60.0),
    (5_000_000, 35.0),
    (25_000_000, 20.0),
    (100_000_000, 10.0),
    (float("inf"), 5.0),
]

# --- model ------------------------------------------------------------------
PRIMARY_MODEL = "gbm"               # fixed before looking at out-of-sample results
RETRAIN_MONTHS = 3                  # walk-forward retraining frequency
TOP_FRACTION = 0.20                 # trade the top 20% of scores (threshold from training data)
RANDOM_STATE = 42
SHORT_BORROW_APR = 0.03             # borrow fee on the short leg of the long-short portfolio
