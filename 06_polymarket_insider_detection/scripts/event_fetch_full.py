"""Within-event fetch for EVERY independent market (restartable; makes network calls).

For each Gamma event that contains at least one market with an independent news time
(``news_independent.parquet``) this fetches the event (all sibling outcome markets) and the YES-token
price history of each sibling (same bucket ladder as ``collect_prices``; siblings that closed before
the event's last independent reveal and are not independent themselves are skipped). Output is
incremental: two small atomic part files per event under ``data/tapes/events/{markets,prices}/``;
an event is done once its markets part exists, so the script can be stopped (Ctrl-C) and restarted
at any time and cached pilot events are never refetched. The final step consolidates the parts into
``data/snapshots/event_markets.parquet`` / ``event_prices.parquet`` (plus trade-price rows built
from the tapes) and recomputes ``within_event.parquet`` + the placebo (no network).

Usage (from the project folder, with the venv):

    .venv\\Scripts\\python.exe scripts\\event_fetch_full.py --estimate          # cost only, no network
    .venv\\Scripts\\python.exe scripts\\event_fetch_full.py                     # fetch + consolidate + compute
    .venv\\Scripts\\python.exe scripts\\event_fetch_full.py --max-events 200    # a slice (smoke test)
    .venv\\Scripts\\python.exe scripts\\event_fetch_full.py --consolidate-only  # no fetching, rebuild snapshots

It shares the ~5 req/s API budget with every other fetch process: do not run two at once.
"""

from __future__ import annotations

import argparse
import json
import sys

from pmdetect.markets import within_event as we


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--estimate", action="store_true", help="print the request/hour estimate and exit")
    ap.add_argument("--max-events", type=int, default=None, help="fetch at most this many new events")
    ap.add_argument("--consolidate-only", action="store_true", help="skip fetching")
    ap.add_argument("--no-compute", action="store_true", help="stop after consolidating the snapshots")
    ap.add_argument("--n-perm", type=int, default=10000, help="permutations for the placebo")
    args = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)  # noqa: E731

    est = we.estimate_fetch_cost()
    log("estimate: " + json.dumps(est, default=float))
    if args.estimate:
        return 0
    if not args.consolidate_only:
        stats = we.fetch_event_data(log=log, max_events=args.max_events)
        log("fetch done: " + json.dumps(stats, default=str))
    we.consolidate(log=log)
    if not args.no_compute:
        we.run(n_perm=args.n_perm, log=log, consolidate_first=False)
    log("DONE")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
