"""Offline tests for Builder A modules (http, db, gamma parsing, universe, info type)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from pmdetect.collect import gamma, http
from pmdetect.markets import info_type, universe
from pmdetect.store.db import connect, upsert

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def con():
    c = connect(Path(":memory:"))
    yield c
    c.close()


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "raw_cache")
    monkeypatch.setattr(http, "MIN_INTERVAL", 0.0)


# ---- http -------------------------------------------------------------------------------
@respx.mock
def test_get_json_caches():
    route = respx.get("https://x.test/a").mock(return_value=httpx.Response(200, json={"k": 1}))
    assert http.get_json("https://x.test/a", {"b": 2, "a": 1}) == {"k": 1}
    assert http.get_json("https://x.test/a", {"a": 1, "b": 2}) == {"k": 1}  # param order irrelevant
    assert route.call_count == 1


@respx.mock
def test_retry_after_and_5xx(monkeypatch):
    sleeps = []
    monkeypatch.setattr(http.time, "sleep", lambda s: sleeps.append(s))
    respx.get("https://x.test/r").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    assert http.get_json("https://x.test/r") == {"ok": True}
    assert any(abs(s - 7.0) < 0.05 for s in sleeps)  # shared back-off sleeps in _throttle


@respx.mock
def test_paginate_cursor():
    def handler(request):
        cur = request.url.params.get("cursor")
        if cur is None:
            return httpx.Response(200, json={"data": [1, 2], "pagination": {"next_cursor": "c2"}})
        return httpx.Response(200, json={"data": [3], "pagination": {"next_cursor": None}})

    respx.get("https://x.test/p").mock(side_effect=handler)
    assert list(http.paginate_cursor("https://x.test/p", {"limit": 2})) == [1, 2, 3]


# ---- db ---------------------------------------------------------------------------------
def test_upsert_merges_and_keeps_other_columns(con):
    df = pl.DataFrame({"condition_id": ["a"], "question": ["q"], "info_type": ["deadline"]})
    assert upsert(con, "markets", df, ["condition_id"]) == 1
    upsert(
        con, "markets", pl.DataFrame({"condition_id": ["a"], "question": ["q2"]}), ["condition_id"]
    )
    row = con.execute("SELECT question, info_type FROM markets").fetchall()
    assert row == [("q2", "deadline")]


# ---- gamma ------------------------------------------------------------------------------
def test_parse_event_fixture():
    ev = json.loads((FIX / "gamma_events_nobel.json").read_text(encoding="utf-8"))["events"][0]
    markets, tokens = gamma.parse_event(ev)
    assert markets and len(tokens) == 2 * len(markets)
    machado = next(m for m in markets if "Machado" in m["question"])
    assert machado["outcome"] == "YES"
    assert machado["condition_id"].startswith("0x")
    assert "all" not in machado["tags"]
    assert machado["closed_ts"] is not None and machado["closed_ts"].tzinfo is None


def test_parse_ts_variants():
    assert gamma.parse_ts("2026-11-11 10:32:06.111533+00").hour == 10
    assert gamma.parse_ts("2025-10-10T09:00:00Z").day == 10
    assert gamma.parse_ts("2025-10-10").year == 2025
    assert gamma.parse_ts(None) is None


@respx.mock
def test_collect_markets_resumable(con):
    page = json.loads((FIX / "gamma_events_nobel.json").read_text(encoding="utf-8"))
    route = respx.get(gamma.EVENTS_KEYSET).mock(return_value=httpx.Response(200, json=page))
    from datetime import date

    s = gamma.collect_markets(date(2025, 10, 1), date(2025, 11, 1), window="all", con=con)
    assert s["markets"] > 0
    n = con.execute("SELECT count(*) FROM markets").fetchall()[0][0]
    assert n == s["markets"]
    gamma.collect_markets(date(2025, 10, 1), date(2025, 11, 1), window="all", con=con)
    assert route.call_count == 1  # completed window skipped on resume


# ---- universe ---------------------------------------------------------------------------
BASE = {
    "volume_usd": 20_000.0,
    "uma_status": "resolved",
    "outcome": "YES",
    "tags": ["politics"],
    "question": "Will X happen?",
    "slug": "will-x-happen",
}


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({}, None),
        ({"volume_usd": 100.0}, "volume_below_min"),
        ({"volume_usd": 9e6}, "volume_above_max"),
        ({"uma_status": "resolved|disputed"}, "disputed"),
        ({"outcome": None}, "not_clean_binary"),
        ({"tags": ["nba"]}, "excluded_sports_tag"),
        ({"question": "Bitcoin Up or Down - 5pm?"}, "excluded_crypto_updown_text"),
        ({"question": "Highest temperature in NYC on May 1?"}, "excluded_weather_text"),
        ({"slug": "nba-lal-bos-2026"}, "excluded_sports_game_slug"),
    ],
)
def test_drop_reason(patch, reason):
    assert universe.drop_reason({**BASE, **patch}) == reason


def test_build_universe(con):
    upsert(
        con,
        "markets",
        pl.DataFrame(
            [
                {"condition_id": "a", **BASE},
                {"condition_id": "b", **{**BASE, "volume_usd": 10.0}},
            ],
            schema_overrides={"tags": pl.List(pl.String)},
        ),
        ["condition_id"],
    )
    res = universe.build_universe(con)
    assert res["kept"] == 1 and res["drops"] == {"volume_below_min": 1}
    assert con.execute("SELECT in_universe FROM markets WHERE condition_id='a'").fetchall() == [
        (True,)
    ]


# ---- info type --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("Will X win Best Director at the 98th Academy Awards?", "scheduled_reveal"),
        ("Will the Bank of Mexico announce no change at the May meeting?", "scheduled_reveal"),
        ("Bolsonaro arrested before May?", "deadline"),
        ("Will the captain of the crashed flight be a woman?", "unscheduled_event"),
    ],
)
def test_classify(q, expected):
    assert info_type.classify(q, []) == expected


def test_label_file_agreement_reasonable():
    labels = info_type.load_labels()
    assert len(labels) >= 30
    assert info_type.agreement(labels)["rate"] >= 0.8
