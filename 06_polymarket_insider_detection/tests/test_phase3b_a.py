"""Phase 3b-A tests (offline): no-valid-cutoff positions are kept but excluded from fit and repeat test."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from tests.test_phase3_a import T0, _markets, _synthetic_positions, _trades

from pmdetect.wallets.calibration import decile_table, fit_isotonic, wallet_tests
from pmdetect.wallets.fdr import apply_fdr
from pmdetect.wallets.positions import build_positions


def _ni():
    return pl.DataFrame(
        {"condition_id": ["A", "B", "C"], "news_ts_indep": [T0 + timedelta(hours=5), None, None],
         "independent": [True, False, False]},
        schema_overrides={"news_ts_indep": pl.Datetime("us")},
    )  # fmt: skip


def test_no_valid_cutoff_kept_but_flagged_and_late_entries_dropped():
    tr = _trades([
        ("w1", "A", "No", "BUY", 100, 0.3, 1),     # valid (independent news at +5h)
        ("w1", "A", "No", "BUY", 900, 0.9, 6),     # after the valid cutoff: must be dropped
        ("w2", "C", "Yes", "BUY", 100, 0.2, 1),    # market C: no independent news, no jump -> none
    ])  # fmt: skip
    pos = build_positions(tr, _markets(), _ni(), None).sort("wallet")
    a, c = pos.row(0, named=True), pos.row(1, named=True)
    assert a["valid_cutoff"] is True and a["cutoff_method"] == "independent_news"
    assert a["entry_price"] == 0.3 and a["size"] == 100  # late buy never enters
    assert c["valid_cutoff"] is False and c["cutoff_method"] == "none" and c["cutoff_ts"] is None
    assert pos["first_ts"].max() < datetime(2025, 1, 2)


def test_invalid_cutoff_positions_excluded_from_fit_and_repeat_test():
    pos = _synthetic_positions().with_columns(
        pl.lit(True).alias("valid_cutoff"), pl.lit(True).alias("calib_eligible")
    )
    # a wallet whose only positions have no valid cutoff and are all long-shot wins: must not be tested
    extra = pl.DataFrame({
        "wallet": ["cheat"] * 10, "event_id": [f"x{i}" for i in range(10)], "entry_price": [0.05] * 10,
        "size": [100.0] * 10, "won": [True] * 10, "info_type": ["deadline"] * 10,
        "valid_cutoff": [False] * 10,
    })  # fmt: skip
    both = pl.concat([pos, extra], how="diagonal_relaxed")
    d, _, _ = fit_isotonic(both)
    assert d.height == pos.height and "cheat" not in d["wallet"].to_list()
    res = apply_fdr(wallet_tests(both, write=False))
    assert "cheat" not in res["wallet"].to_list()
    planted = res.filter(pl.col("wallet") == "planted").row(0, named=True)
    assert planted["flag"] is True  # the planted long-shot wallet still passes FDR


def test_decile_table_aggregate():
    pos = _synthetic_positions()
    d, _, _ = fit_isotonic(pos)
    t = decile_table(d)
    assert t["n"].sum() == d.height and "wallet" not in t.columns


def test_jump_start_positions_excluded_from_fit_and_repeat_test():
    base = _synthetic_positions()
    pos = base.with_columns(
        pl.lit(True).alias("valid_cutoff"), pl.lit(True).alias("calib_eligible")
    )
    jump = pl.DataFrame({
        "wallet": ["jumper"] * 10, "event_id": [f"j{i}" for i in range(10)], "entry_price": [0.05] * 10,
        "size": [100.0] * 10, "won": [True] * 10, "info_type": ["deadline"] * 10,
        "valid_cutoff": [True] * 10, "calib_eligible": [False] * 10,
    })  # fmt: skip
    both = pl.concat([pos, jump], how="diagonal_relaxed")
    d, _, _ = fit_isotonic(both)
    assert d.height == pos.height and "jumper" not in d["wallet"].to_list()
    res = apply_fdr(wallet_tests(both, write=False))
    assert "jumper" not in res["wallet"].to_list()
    assert res.filter(pl.col("wallet") == "planted")["flag"][0] is True
