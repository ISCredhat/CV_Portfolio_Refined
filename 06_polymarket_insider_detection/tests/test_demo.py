"""Offline demo: the fixtures run end to end and nothing looks like a wallet address."""

from __future__ import annotations

import json
import re

import polars as pl
import pytest
from typer.testing import CliRunner

from pmdetect import demo
from pmdetect.cli import app

HEX40 = re.compile(r"\b(?:0x)?[0-9a-fA-F]{40}\b")


def test_fixtures_are_anonymised_and_small():
    files = list(demo.FIXTURES.iterdir())
    assert sum(f.stat().st_size for f in files) < 5_000_000
    assert demo.scan_hex40([demo.FIXTURES]) == []
    tr = pl.read_parquet(demo.FIXTURES / "trades_taker.parquet")
    assert tr["wallet"].str.starts_with("wallet-").all()
    assert tr["wallet"].n_unique() > 1000
    man = json.loads((demo.FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    assert man["n_markets"] == 50


def test_demo_run_end_to_end(tmp_path):
    out = tmp_path / "scratch"
    r = demo.run_demo(out, n_perm=200, log=lambda *_: None)
    assert r["seconds"] < 180
    for rel in (
        "demo_summary.md",
        "snapshots/news_independent.parquet",
        "snapshots/within_event.parquet",
        "snapshots/within_event_placebo.json",
        "private/oneshot_scores.parquet",
        "private/watchlist.csv",
        "results/phase3_summary.md",
    ):
        assert (out / rel).exists(), rel
    assert demo.scan_hex40([out]) == []
    for f in out.rglob("*.md"):
        assert not HEX40.search(f.read_text(encoding="utf-8"))


def test_demo_refuses_real_store():
    with pytest.raises(ValueError):
        demo.run_demo(demo.PROJECT_ROOT / "results" / "x", log=lambda *_: None)
    with pytest.raises(ValueError):
        demo.run_demo(demo.PROJECT_ROOT / "data", log=lambda *_: None)


def test_fetch_plan_cli_prints_commands():
    res = CliRunner().invoke(app, ["demo", "fetch-plan"])
    assert res.exit_code == 0
    assert "pmdetect collect markets" in res.output and "pmdetect pilot" in res.output
