"""Offline, synthetic tests for the labelled-case table (4D) and the hand-audit classifier (4E)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pmdetect.validate import audit, cases

HEX40 = re.compile(r"(?<![0-9a-fA-F])(?:0x)?[0-9a-fA-F]{40}(?![0-9a-fA-F])")
RESULTS = Path(__file__).resolve().parents[1] / "results"
HOURS = np.arange(97, 0, -1)  # hours before the reveal, descending


def _flat(level: float = 0.1) -> np.ndarray:
    return np.full(len(HOURS), level)


def _jump_at(p: np.ndarray, h_start: int, h_end: int, to: float) -> np.ndarray:
    """Linear jump from the level at h_start to ``to`` at h_end (hours before reveal), then flat."""
    out = p.copy()
    i0, i1 = int(HOURS[0] - h_start), int(HOURS[0] - h_end)
    out[i0 : i1 + 1] = np.linspace(p[i0], to, i1 - i0 + 1)
    out[i1:] = to
    return out


ONE_SIDED = {"one_sided": True, "informed_looking": True}


def test_gradual_rise_is_convergence():
    p = np.linspace(0.20, 0.60, len(HOURS))
    r = audit.classify_winner_path(HOURS, p, siblings=[_flat(0.3)], flow=ONE_SIDED)
    assert r["cls"] == "predictable_convergence"


def test_sharp_early_jump_without_related_move_is_plausible_leak():
    p = _jump_at(_flat(0.10), 14, 12, 0.55)
    r = audit.classify_winner_path(HOURS, p, siblings=[_flat(0.3), _flat(0.2)], flow=ONE_SIDED)
    assert r["cls"] == "plausible_leak", r["reason"]
    assert r["step_size"] >= 0.4


def test_leak_needs_tape_when_required_and_balanced_flow_is_unclear():
    p = _jump_at(_flat(0.10), 14, 12, 0.55)
    assert audit.classify_winner_path(HOURS, p, flow=None, require_flow=True)["cls"] == "unclear"
    bal = {"one_sided": False, "informed_looking": False}
    assert audit.classify_winner_path(HOURS, p, flow=bal)["cls"] == "unclear"
    # a callable flow gets the step window
    seen = {}

    def fl(sh, eh):
        seen["w"] = (sh, eh)
        return ONE_SIDED

    assert audit.classify_winner_path(HOURS, p, flow=fl)["cls"] == "plausible_leak"
    assert 11 <= seen["w"][1] <= seen["w"][0] <= 15


def test_jump_at_known_release_time_is_public_news():
    p = _jump_at(_flat(0.10), 14, 12, 0.55)
    r = audit.classify_winner_path(HOURS, p, release_h=(13,), flow=ONE_SIDED)
    assert r["cls"] == "public_news_reaction" and "release" in r["reason"]


def test_related_market_moving_first_is_public_news():
    p = _jump_at(_flat(0.10), 10, 8, 0.55)
    sib = _jump_at(_flat(0.50), 16, 14, 0.20)  # the favourite collapsed first
    r = audit.classify_winner_path(HOURS, p, siblings=[sib], flow=ONE_SIDED)
    assert r["cls"] == "public_news_reaction"
    # same step, a sibling that moves AFTER it is not a trigger
    late = _jump_at(_flat(0.50), 6, 4, 0.20)
    assert audit.classify_winner_path(HOURS, p, siblings=[late], flow=ONE_SIDED)["cls"] == "plausible_leak"


def test_common_trigger_and_market_opening_are_public_news():
    p = _jump_at(_flat(0.10), 14, 12, 0.55)
    assert audit.classify_winner_path(HOURS, p, common_trigger=True, flow=ONE_SIDED)["cls"] == "public_news_reaction"
    assert audit.classify_winner_path(HOURS, p, market_open_h=20.0, flow=ONE_SIDED)["cls"] == "public_news_reaction"


def test_defaults_to_unclear():
    p = _jump_at(_flat(0.10), 14, 12, 0.55)
    # too sparse to see a step
    assert audit.classify_winner_path(HOURS, p, res_h=12.0, flow=ONE_SIDED)["cls"] == "unclear"
    # step in the last hours before the reveal
    late = _jump_at(_flat(0.10), 4, 2, 0.55)
    assert audit.classify_winner_path(HOURS, late, flow=ONE_SIDED)["cls"] == "unclear"
    # missing prices at the window edges
    nan = p.copy()
    nan[:] = np.nan
    assert audit.classify_winner_path(HOURS, nan)["cls"] == "unclear"
    # a long shot that did not move at all in the window
    assert audit.classify_winner_path(HOURS, _flat(0.10))["cls"] == "unclear"
    # a poll/results-type market with a sharp step: cannot rule out a public trigger
    assert audit.classify_winner_path(HOURS, p, topic=True, flow=ONE_SIDED)["cls"] == "unclear"


def test_favourite_already_high_is_convergence():
    assert audit.classify_winner_path(HOURS, _flat(0.97))["cls"] == "predictable_convergence"


def test_rising_before_window_is_convergence_even_with_a_step():
    p = _flat(0.10)
    p = _jump_at(p, 70, 40, 0.40)  # rising well before the window
    p = _jump_at(p, 14, 12, 0.60)
    assert audit.classify_winner_path(HOURS, p, flow=ONE_SIDED)["cls"] in {
        "predictable_convergence",
        "plausible_leak",
    }
    # the 48 h before the window rose by most of the window rise: convergence
    q = _jump_at(_flat(0.10), 60, 55, 0.50)
    q = _jump_at(q, 14, 12, 0.60)
    assert audit.classify_winner_path(HOURS, q, flow=ONE_SIDED)["cls"] == "predictable_convergence"


# ------------------------------------------------------------------------------------- one-shot rules

H2 = np.arange(100, -7, -1)  # hours before the cutoff, going past it


def _p2(entry_h: int, jump: tuple[int, int, float], base: float = 0.10, pre: float = 0.0) -> np.ndarray:
    p = np.full(len(H2), base)
    if pre:
        i = int(H2[0] - (entry_h + 6))
        p[i:] = base + pre
    i0, i1 = int(H2[0] - jump[0]), int(H2[0] - jump[1])
    p[i0 : i1 + 1] = np.linspace(p[i0], jump[2], i1 - i0 + 1)
    p[i1:] = jump[2]
    return p


def _os(p, **kw):
    args = {"entry_h": 30.0, "end_h": 1, "entry_price": 0.1, "age_days": 1.0, "only_market": True, "res_h": 1.0}
    args.update(kw)
    return audit.classify_oneshot_path(H2, p, **args)


def test_oneshot_fresh_wallet_before_sharp_step_is_leak():
    p = _p2(30, (12, 10, 0.60))
    r = _os(p, flow={"wallet_share": 0.6, "n_trades": 10})
    assert r["cls"] == "plausible_leak", r["reason"]
    assert _os(p, flow=None, require_flow=True)["cls"] == "unclear"
    assert _os(p, flow={"wallet_share": 0.05, "n_trades": 50})["cls"] == "unclear"
    assert _os(p, age_days=900.0, only_market=False, flow={"wallet_share": 0.6})["cls"] == "unclear"


def test_oneshot_chasing_a_move_and_triggers():
    chase = _p2(30, (12, 10, 0.60), pre=0.2)
    assert _os(chase, flow={"wallet_share": 0.6})["cls"] == "public_news_reaction"
    p = _p2(30, (12, 10, 0.60))
    # a related outcome that had already moved before the entry is a public trigger
    sib = np.full(len(H2), 0.5)
    sib[int(H2[0] - 50) :] = 0.2
    assert _os(p, siblings=[sib], flow={"wallet_share": 0.6})["cls"] == "public_news_reaction"
    # a market-wide repricing at the same hour does not show the information was public at the entry
    r = _os(p, concurrent_jumps=5, flow={"wallet_share": 0.6})
    assert r["cls"] == "unclear" and "cannot be judged" in r["reason"]
    assert _os(p, topic=True, flow={"wallet_share": 0.6})["cls"] == "unclear"


def test_oneshot_gradual_flat_and_favourite():
    grad = np.linspace(0.10, 0.5, len(H2))
    assert _os(grad, flow={"wallet_share": 0.6})["cls"] == "predictable_convergence"
    flat = np.full(len(H2), 0.10)
    assert _os(flat)["cls"] == "unclear"
    assert _os(np.full(len(H2), 0.8), entry_price=0.8)["cls"] == "predictable_convergence"
    # a jump-start cutoff: the window runs past the cutoff (end_h < 0)
    j = _p2(30, (-2, -4, 0.7))
    assert _os(j, end_h=-6, flow={"wallet_share": 0.6})["cls"] == "plausible_leak"


# ------------------------------------------------------------------------------------------ utilities


def test_to_grid_forward_fills_and_never_backfills():
    ts = np.array(["2026-01-01T12:00", "2026-01-01T15:00"], dtype="datetime64[us]")
    hs, p = audit.to_grid(ts, np.array([0.2, 0.6]), np.datetime64("2026-01-01T18:00", "us"), 8, 1)
    assert hs[0] == 8 and hs[-1] == 1
    assert np.isnan(p[0])  # 10:00 is before the first observation: no backward fill
    assert p[2] == pytest.approx(0.2)  # 12:00
    assert p[-1] == pytest.approx(0.6)  # 17:00


def test_best_step_and_rank_helpers():
    p = np.array([0.1, 0.1, 0.2, 0.6, 0.6])
    size, a, b = audit.best_step(p, 0, 4, 2)
    assert size == pytest.approx(0.5) and (a, b) == (1, 3)
    pool = np.array([0.1, 0.5, 0.9, 0.9])
    rank, pct = cases.pool_rank(pool, 0.9)
    assert rank == 1.0 and pct == pytest.approx(100.0)
    rank, pct = cases.pool_rank(pool, 0.1)
    assert rank == 4.0 and pct == pytest.approx(25.0)
    assert np.isnan(cases.pool_rank(pool, float("nan"))[0])


def test_flow_stats_aggregates_without_addresses():
    t0 = datetime(2026, 1, 1)
    rows = [{"ts": t0 + timedelta(minutes=i), "side": "BUY", "usd": 100.0, "wallet": f"w{i % 2}"} for i in range(6)]
    df = pl.DataFrame(rows)
    fs = {"first_seen": {"w0": df["ts"][0], "w1": df["ts"][0]}, "n_markets": {"w0": 1, "w1": 5}}
    out = audit.flow_stats(df, df["ts"].min(), df["ts"].max(), fs)
    assert out["one_sided"] and out["informed_looking"]
    assert out["n_wallets"] == 2 and out["only_market_buy_share"] == pytest.approx(0.5)
    assert not any(isinstance(v, str) for v in out.values())


# --------------------------------------------------------------------------------- 4D verdicts / privacy


def test_within_event_verdicts():
    assert cases.verdict_within_event({"we_winner_pct_24h": 0.97})[0] == "FLAGS"
    assert cases.verdict_within_event({"we_winner_pct_24h": 0.5})[0] == "MISSES"
    assert cases.verdict_within_event({"we_winner_pct_24h": float("nan"), "out_of_universe": True})[0] == "CANNOT_TEST"
    assert cases.verdict_within_event({"we_winner_pct_24h": float("nan"), "independent": False})[0] == "CANNOT_TEST"


def test_oneshot_verdicts():
    best = {"cutoff_method": "independent_news"}
    assert cases.verdict_oneshot("x", None, best, 1, 0)[0] == "FLAGS"
    assert cases.verdict_oneshot("x", None, best, 0, 2)[0] == "CANNOT_TEST"  # jump-start tier is outcome-selected
    assert cases.verdict_oneshot("x", None, best, 0, 0)[0] == "MISSES"
    assert cases.verdict_oneshot("x", None, None, 0, 0)[0] == "CANNOT_TEST"
    assert cases.verdict_oneshot("axiom_expose_feb_2026", "candidate", best, 1, 0)[0] == "CANNOT_TEST"
    assert cases.verdict_price({}, "maduro_removal_jan_2026", [], True)[0] == "CANNOT_TEST"
    assert cases.verdict_price({"independent": False}, "google", [], False)[0] == "CANNOT_TEST"
    assert cases.verdict_price({}, "x", ["D"], True)[0] == "FLAGS"


def test_audit_report_is_aggregate_only():
    we = pl.DataFrame(
        {"rank": [1, 2], "question": ["Will A win?", "Will B win?"], "winner_pct_24h": [0.98, 0.97],
         "class": ["unclear", "predictable_convergence"], "reason": ["x", "gradual rise"],
         "peer_events_same_step": [0, 0]}
    )  # fmt: skip
    osh = pl.DataFrame(
        {"rank": [1], "wallet": ["0x" + "ab" * 20], "question": ["Will C win?"], "class": ["unclear"],
         "reason": ["y"], "cutoff_method": ["independent_news"]}
    )  # fmt: skip
    text = audit.render_md(we, osh, {"pop_we": 809, "q99": 0.77, "n_valid": 10, "n_maduro": 1})
    assert not HEX40.search(text)
    assert "unclear" in text and "plausible_leak" in text


@pytest.mark.parametrize(
    "name", ["validation_labelled_cases.md", "validation_labelled_cases.csv", "validation_hand_audit.md"]
)
def test_generated_results_have_no_wallet_address(name):
    f = RESULTS / name
    if not f.exists():
        pytest.skip("results not generated yet")
    assert not HEX40.search(f.read_text(encoding="utf-8"))


def test_private_within_event_audit_has_no_wallet_column():
    f = audit.PRIVATE / "audit_within_event.csv"
    if not f.exists():
        pytest.skip("audit not generated yet")
    text = f.read_text(encoding="utf-8")
    assert "wallet" not in text.splitlines()[0].replace("n_wallets", "").replace("wallet_", "")
    assert not HEX40.search(text)


def test_labelled_cases_csv_has_clean_line_endings_and_loads_in_polars():
    """Regression: write_text() on Windows turned CRLF into CR CR LF, which polars rejects."""
    path = Path(__file__).resolve().parents[1] / "results" / "validation_labelled_cases.csv"
    if not path.exists():
        pytest.skip("results/validation_labelled_cases.csv not generated")
    raw = path.read_bytes()
    assert b"\r\r" not in raw
    import csv
    import io

    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    assert len({len(r) for r in rows}) == 1  # rectangular
    df = pl.read_csv(path, infer_schema_length=0)
    assert df.height == len(rows) - 1
