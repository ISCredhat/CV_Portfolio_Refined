"""Live order-book recorder for the forward paper test (NETWORK: run by the user in a terminal).

Polls https://clob.polymarket.com/books (public, no key, max 500 tokens per request) every --interval
seconds for the tokens in --tokens-file (one token id per line) and appends compact top-of-book rows to
hourly Parquet parts under data/orderbook/. Polite by design: one request per poll per 500 tokens, never
faster than one request per second, honours Retry-After on 429, restartable (each poll is an independent
part; Ctrl+C is safe). It records the book only; it places no orders and needs no credentials.

  .venv\Scripts\python.exe scripts\orderbook_recorder.py --tokens-file data\manual\watch_tokens.txt --interval 60 --hours 168

Columns: ts_utc, token_id, best_bid, best_ask, bid_size, ask_size, bid_depth_5c, ask_depth_5c,
last_trade_price, tick_size, book_hash. Bids come sorted ascending and asks descending (best is last).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

BOOKS_URL = "https://clob.polymarket.com/books"
MAX_PER_REQUEST = 500
MIN_GAP_S = 1.0
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "orderbook"
SCHEMA = {
    "ts_utc": pl.Datetime("us"),
    "token_id": pl.Utf8,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "bid_size": pl.Float64,
    "ask_size": pl.Float64,
    "bid_depth_5c": pl.Float64,
    "ask_depth_5c": pl.Float64,
    "last_trade_price": pl.Float64,
    "tick_size": pl.Float64,
    "book_hash": pl.Utf8,
}


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize_book(book: dict, ts: datetime) -> dict:
    """Top-of-book and 5-cent depth from one /books entry (best bid is the LAST bid, best ask the LAST ask)."""
    bids = [(float(b["price"]), float(b["size"])) for b in book.get("bids") or []]
    asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks") or []]
    bb = max(bids, default=None, key=lambda x: x[0])
    ba = min(asks, default=None, key=lambda x: x[0])
    return {
        "ts_utc": ts,
        "token_id": book.get("asset_id"),
        "best_bid": bb[0] if bb else None,
        "best_ask": ba[0] if ba else None,
        "bid_size": bb[1] if bb else None,
        "ask_size": ba[1] if ba else None,
        "bid_depth_5c": sum(s for p, s in bids if bb and p >= bb[0] - 0.05) if bb else 0.0,
        "ask_depth_5c": sum(s for p, s in asks if ba and p <= ba[0] + 0.05) if ba else 0.0,
        "last_trade_price": _f(book.get("last_trade_price")),
        "tick_size": _f(book.get("tick_size")),
        "book_hash": book.get("hash"),
    }


def poll(client: httpx.Client, tokens: list[str]) -> list[dict]:
    rows: list[dict] = []
    for i in range(0, len(tokens), MAX_PER_REQUEST):
        chunk = tokens[i : i + MAX_PER_REQUEST]
        while True:
            resp = client.post(BOOKS_URL, json=[{"token_id": t} for t in chunk], timeout=30)
            if resp.status_code == 429:
                time.sleep(min(float(resp.headers.get("Retry-After", "2")), 120.0))
                continue
            resp.raise_for_status()
            break
        ts = datetime.now(UTC).replace(tzinfo=None)
        rows.extend(summarize_book(b, ts) for b in resp.json())
        time.sleep(MIN_GAP_S)
    return rows


def write_part(rows: list[dict], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    path = out / f"books-{now:%Y%m%dT%H}-{now:%M%S}.parquet"
    tmp = path.with_suffix(".tmp")
    pl.DataFrame(rows, schema=SCHEMA).write_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens-file", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between polls (min 5)")
    ap.add_argument("--hours", type=float, default=24.0, help="stop after this many hours")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    tokens = [t.strip() for t in a.tokens_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    if not tokens:
        raise SystemExit("no tokens in --tokens-file")
    interval = max(a.interval, 5.0)
    end = time.monotonic() + a.hours * 3600
    print(f"recording {len(tokens)} tokens every {interval:.0f}s for {a.hours:g} h into {a.out}", flush=True)
    with httpx.Client(headers={"User-Agent": "pmdetect-orderbook-recorder/0.1 (research)"}) as client:
        while time.monotonic() < end:
            t0 = time.monotonic()
            try:
                path = write_part(poll(client, tokens), a.out)
                print(f"{datetime.now(UTC):%H:%M:%S} wrote {path.name}", flush=True)
            except (httpx.HTTPError, OSError) as exc:  # keep going on transient errors
                print(f"{datetime.now(UTC):%H:%M:%S} poll failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
