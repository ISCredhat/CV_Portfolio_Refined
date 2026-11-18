"""Offline tests for scripts/orderbook_recorder.py (no network: a stub client)."""

import importlib.util
from datetime import datetime
from pathlib import Path

import polars as pl

_spec = importlib.util.spec_from_file_location(
    "orderbook_recorder", Path(__file__).resolve().parents[1] / "scripts" / "orderbook_recorder.py"
)
rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rec)

BOOK = {
    "asset_id": "tok1",
    "bids": [{"price": "0.40", "size": "10"}, {"price": "0.45", "size": "5"}, {"price": "0.30", "size": "99"}],
    "asks": [{"price": "0.60", "size": "7"}, {"price": "0.50", "size": "3"}],
    "last_trade_price": "0.47",
    "tick_size": "0.01",
    "hash": "abc",
}


def test_summarize_book_best_prices_and_depth():
    r = rec.summarize_book(BOOK, datetime(2026, 1, 1))
    assert (r["best_bid"], r["best_ask"]) == (0.45, 0.50)
    assert (r["bid_size"], r["ask_size"]) == (5.0, 3.0)
    assert r["bid_depth_5c"] == 15.0  # 0.45 and 0.40 within 5 cents, 0.30 excluded
    assert r["ask_depth_5c"] == 3.0 + 0.0  # 0.60 is 10 cents above the best ask
    assert r["last_trade_price"] == 0.47


def test_empty_book_does_not_crash_and_part_is_written(tmp_path):
    r = rec.summarize_book({"asset_id": "tok2", "bids": [], "asks": []}, datetime(2026, 1, 1))
    assert r["best_bid"] is None and r["bid_depth_5c"] == 0.0
    p = rec.write_part([r, rec.summarize_book(BOOK, datetime(2026, 1, 1))], tmp_path)
    df = pl.read_parquet(p)
    assert df.height == 2 and set(df.columns) == set(rec.SCHEMA)
