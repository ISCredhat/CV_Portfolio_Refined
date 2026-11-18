from datetime import datetime

import polars as pl
import pytest

from pmdetect.stage2 import common


def test_periods_and_sealed_guard():
    assert common.period_of(datetime(2025, 12, 31, 23, 59)) == "fit"
    assert common.period_of(datetime(2026, 1, 1)) == "tune"
    assert common.period_of(datetime(2026, 6, 30, 23, 59)) == "tune"
    assert common.period_of(datetime(2026, 7, 1)) == "sealed"
    df = pl.DataFrame({"closed_ts": [datetime(2025, 5, 1), datetime(2026, 7, 2), None]})
    assert common.drop_sealed(df).height == 1
    with pytest.raises(AssertionError):
        common.assert_not_sealed(df)


def test_fee_classes_are_conservative():
    assert common.fee_rate(["geopolitics", "world"]) == 0.0
    assert common.fee_rate(["politics", "world"]) == 0.04  # max over matching classes
    assert common.fee_rate(["crypto", "politics"]) == 0.07
    assert common.fee_rate(["sports"]) == 0.03
    assert common.fee_rate(["unknown-tag"]) == 0.05 and common.fee_rate(None) == 0.05


def test_net_return_arithmetic_and_doubled_costs():
    r1 = common.net_return_per_dollar(0.5, True, 0.04)
    cost = 0.5 + 0.04 * 0.25
    assert r1 == pytest.approx((1 - cost) / cost)
    assert common.net_return_per_dollar(0.5, False, 0.04) == pytest.approx(-1.0)
    assert common.net_return_per_dollar(0.5, True, 0.04, cost_mult=2.0) < r1
