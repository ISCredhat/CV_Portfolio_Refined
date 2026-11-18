import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import httpx
import pytest

from pmdetect.collect import data_api
from pmdetect.store import db

FIX = Path(__file__).parent / "fixtures"
COND = "0xbc5edc340f1cf273b29f8fc1ccfdb17aa946f05ab14e377b5449ae511d7026af"
KHABIB = "0x128dd10748d92137d5057bed92b6c187c0a000a9a94f1a75d4a618b09eac629b"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture
def con_factory(tmp_path, monkeypatch):
    path = tmp_path / "t.duckdb"
    monkeypatch.setattr(data_api, "connect", partial(db.connect, path))
    con = db.connect(path)
    con.close()
    return lambda: db.connect(path)


def test_trades_to_frame_fields():
    rows = load("trades_page.json")["data"]
    df = data_api.trades_to_frame(rows)
    assert df.height == 5
    r = df.row(0, named=True)
    assert r["usd"] == pytest.approx(rows[0]["size"] * rows[0]["price"])
    assert r["wallet"] == rows[0]["proxy_wallet"]
    assert r["tx_hash"] == rows[0]["transaction_hash"]
    assert isinstance(r["ts"], datetime)
    assert df["trade_id"].n_unique() == 5


def test_trade_id_disambiguates_identical_rows():
    row = load("trades_page.json")["data"][0]
    df = data_api.trades_to_frame([row, dict(row)])
    assert df["trade_id"].n_unique() == 2


def test_collect_trades_paginates_and_is_idempotent(con_factory, monkeypatch):
    rows = load("trades_page.json")["data"]
    calls = []

    def fake_paginate(url, params, **kw):
        calls.append(params)
        yield from rows

    def fake_get(url, params=None, **kw):
        if url == data_api.TOKENS_URL:
            return load("tokens.json")
        return load("resolutions.json")

    monkeypatch.setattr(data_api, "paginate_cursor", fake_paginate)
    monkeypatch.setattr(data_api, "get_json", fake_get)
    assert data_api.collect_trades([COND]) == 5
    assert calls[0]["condition"] == COND and calls[0]["taker_only"] == "false"
    assert data_api.collect_trades([COND]) == 0  # completed in fetch_log
    con = con_factory()
    assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 5
    con.execute("DELETE FROM fetch_log")
    con.close()
    data_api.collect_trades([COND])  # re-run upserts, no duplicates
    con = con_factory()
    assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 5


def test_trade_batches_of_20(con_factory, monkeypatch):
    seen = []
    monkeypatch.setattr(
        data_api, "paginate_cursor", lambda url, params, **kw: seen.append(params) or iter([])
    )
    monkeypatch.setattr(data_api, "get_json", lambda *a, **k: {"data": []})
    ids = [f"0x{i:064x}" for i in range(45)]
    data_api.collect_trades(ids)
    assert [len(p["condition"].split(",")) for p in seen] == [20, 20, 5]


def test_outcome_from_payouts():
    assert data_api.outcome_from_payouts([1000000, 0]) == "YES"
    assert data_api.outcome_from_payouts([0, 1000000]) == "NO"
    assert data_api.outcome_from_payouts([500000, 500000]) is None
    assert data_api.outcome_from_payouts(None) is None


def test_collect_resolutions_updates_markets_and_tokens(con_factory, monkeypatch):
    con = con_factory()
    con.execute(
        "INSERT INTO markets (condition_id, slug, volume_usd, in_universe) "
        "VALUES (?, 's', 1, TRUE)",
        [KHABIB],
    )
    con.close()

    def fake_get(url, params=None, **kw):
        return load("tokens.json") if url == data_api.TOKENS_URL else load("resolutions.json")

    monkeypatch.setattr(data_api, "get_json", fake_get)
    monkeypatch.setattr(data_api, "paginate_cursor", lambda *a, **k: iter([]))
    assert data_api.collect_resolutions([KHABIB]) == 1
    data_api.collect_tokens([KHABIB])
    con = con_factory()
    assert con.execute("SELECT outcome, slug, volume_usd FROM markets").fetchone() == (
        "YES",
        "s",
        1.0,
    )
    assert con.execute("SELECT COUNT(*) FROM tokens").fetchone()[0] == 2


def test_outcome_from_tokens_fallback():
    toks = load("tokens.json")["data"]
    assert data_api.outcome_from_tokens(toks) == "YES"


def test_price_ladder_skips_400_and_empty(monkeypatch):
    series = load("prices_12h.json")["data"]
    tried = []

    def fake_paginate(url, params, **kw):
        tried.append(params["bucket_seconds"])
        if params["bucket_seconds"] == 600:
            req = httpx.Request("GET", url)
            resp = httpx.Response(400, request=req)
            raise httpx.HTTPStatusError("bad", request=req, response=resp)
        if params["bucket_seconds"] == 1800:
            return iter([])
        return iter(series)

    monkeypatch.setattr(data_api, "paginate_cursor", fake_paginate)
    pts, bucket = data_api.fetch_price_series("tok")
    assert tried == [600, 1800, 10800] and bucket == 10800 and len(pts) == len(series)


def test_collect_prices_writes_and_resumes(con_factory, monkeypatch):
    series = load("prices_12h.json")["data"]
    monkeypatch.setattr(data_api, "paginate_cursor", lambda *a, **k: iter(series))
    assert data_api.collect_prices(["tok"]) == len(series)
    assert data_api.collect_prices(["tok"]) == 0
    con = con_factory()
    assert con.execute("SELECT MAX(price) FROM prices").fetchone()[0] == 1.0


def test_summarize_activity_and_collect_wallets(con_factory, monkeypatch):
    rows = load("activity_page.json")["data"]
    s = data_api.summarize_activity(rows)
    trades = [r for r in rows if r["type"] == "TRADE"]  # REDEEM row must be ignored
    assert len(trades) == 2 and len(rows) == 3
    assert s["n_markets"] == len({r["condition_id"] for r in trades})
    assert s["total_usd"] == pytest.approx(sum(r["usdc_size"] for r in trades))
    assert s["first_seen"] == datetime.fromtimestamp(
        min(r["timestamp"] for r in trades), tz=UTC
    ).replace(tzinfo=None)
    monkeypatch.setattr(
        data_api, "get_json", lambda *a, **k: {"data": rows, "pagination": {"next_cursor": None}}
    )
    assert data_api.collect_wallet_activity(["0xABC"]) == 1
    con = con_factory()
    row = con.execute("SELECT wallet, n_markets FROM wallets").fetchone()
    assert row == ("0xabc", s["n_markets"])
