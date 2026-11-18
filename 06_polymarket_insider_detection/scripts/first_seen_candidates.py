"""first_seen candidate selection (offline) and top-up fetch (NETWORK, restartable).

  python scripts/first_seen_candidates.py select            # local, no network: writes
                                                            # data/private/first_seen_candidates.parquet and prints counts
  python scripts/first_seen_candidates.py fetch [--max N] [--threads 3]
                                                            # NETWORK: /v2/activity (asc, limit 1) per candidate,
                                                            # use_cache=False, ~5 req/s shared throttle, one process;
                                                            # writes data/private/first_seen_topup.parquet atomically
                                                            # every 200 wallets; rerun to resume
  python scripts/first_seen_candidates.py all               # select, then fetch

Log: logs/first_seen_candidates.log (also stdout). Never touches DuckDB. Run `pmdetect score-wallets` afterwards
(it merges the top-up file) and refresh the selection if the candidate set should follow the new scores.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from pmdetect.store.db import PROJECT_ROOT
from pmdetect.wallets.firstseen import fetch_candidates, select_candidates

# not the same file the batch script redirects stdout to (Windows would refuse the second open)
LOG = PROJECT_ROOT / "logs" / "first_seen_candidates.run.log"


def log(msg: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["select", "fetch", "all"])
    ap.add_argument("--max", type=int, default=None, help="fetch at most N wallets this run")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--n-top", type=int, default=5000)
    ap.add_argument("--n-novalid", type=int, default=1000)
    a = ap.parse_args(argv)
    if a.mode in ("select", "all"):
        select_candidates(n_top=a.n_top, n_novalid=a.n_novalid, log=log)
    if a.mode in ("fetch", "all"):
        fetch_candidates(max_n=a.max, threads=a.threads, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
