"""prereg.py: agreement with config_crsp, hash verification, freezing (on temporary copies only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import prereg as P  # noqa: E402
from pairs import stats as ST  # noqa: E402


def _copy(tmp_path):
    """An unfrozen copy of the real file (meta reset if the real one has been frozen), so the tests keep working
    after the freeze."""
    import re
    p = tmp_path / "preregistration.yaml"
    t = CC.PREREG_YAML.read_text()
    for key, val in (("frozen", "false"), ("frozen_at", "null"), ("git_commit", "null")):
        t = re.sub(rf"^(  {key}:)\s*\S+", rf"\g<1> {val}", t, count=1, flags=re.M)
    p.write_text(t)
    return p


def _raises(fn, exc=P.PreregError):
    try:
        fn()
    except exc:
        return True
    return False


def test_real_yaml_loads_and_agrees_with_config():
    cfg = P.load()
    frozen = cfg["meta"]["frozen"]
    assert frozen in (True, False)
    assert P.frozen_configuration(cfg) == {"method": "rolling", "entry": 2.0, "exit": 0.5, "bar": "1D"}
    assert P.open_checks(cfg) == []                                       # all pre-freeze checks closed 2026-11-19
    if frozen:                                                  # frozen on Isaac's PC before the download
        assert CC.PREREG_YAML.with_suffix(".sha256").exists() and cfg["meta"]["git_commit"] and cfg["meta"]["frozen_at"]
    else:
        assert not CC.PREREG_YAML.with_suffix(".sha256").exists()


def test_hypotheses_have_statistic_side_alpha_holm_and_mdes():
    cfg = P.load()
    h = cfg["hypotheses"]
    for k in ("H1", "H2", "H3", "H4"):
        assert {"statistic", "side", "alpha", "holm"} <= set(h[k]) and h[k]["alpha"] == 0.05
    assert h["H4"]["side"] == "less" and h["H4"]["break"] == "2001-04" and h["H2"]["min_excess_pp"] == 1.0
    pw = cfg["power"]
    assert abs(pw["H1_full_backward_oos_sharpe"] - ST.mde_sharpe(30.8)) < 0.01
    assert abs(pw["H1_forward_2023_25_sharpe"] - ST.mde_sharpe(3.0)) < 0.05
    assert abs(pw["H3_sharpe_difference_corr_0_3"] - ST.mde_sharpe(30.8, 0.3)) < 0.01
    assert abs(pw["H4_total_change_in_monthly_sd_units"] - ST.mde_trend_change(432)) < 0.01


def test_disagreement_with_config_raises(tmp_path):
    for old, new in [("  entry: 2.0\n  exit: 0.5", "  entry: 2.5\n  exit: 0.5"), ("  top_n: 500", "  top_n: 400"),
                     ("backward_oos: [\"1990-01\", \"2020-10\"]", "backward_oos: [\"1990-01\", \"2021-04\"]"),
                     ("  primary: t1_close", "  primary: t1_open"), ("min_price_last_formation_day: 5.0", "min_price_last_formation_day: 3.0"),
                     ('ggr: ["1963-07", "2009-06"]', 'ggr: ["1962-07", "2009-06"]')]:
        p = _copy(tmp_path)
        t = p.read_text()
        assert old in t, old
        p.write_text(t.replace(old, new, 1))
        assert _raises(lambda: P.load(p)), old


def test_freeze_records_hash_and_detects_tampering(tmp_path):
    p = _copy(tmp_path)
    t = p.read_text()
    assert "  signed_off_by_isaac: true" in t
    p.write_text(t.replace("  signed_off_by_isaac: true", "  signed_off_by_isaac: false", 1))
    assert _raises(lambda: P.freeze(p, commit="abc"))          # an open pre-freeze check blocks it
    out = P.freeze(p, commit="abc123", allow_open_checks=True)
    cfg = P.load(p)
    assert cfg["meta"]["frozen"] is True and cfg["meta"]["git_commit"] == "abc123" and cfg["meta"]["frozen_at"]
    assert (tmp_path / "preregistration.sha256").read_text().split()[0] == out["sha256"] == P.sha256_file(p)
    assert _raises(lambda: P.freeze(p, commit="x", allow_open_checks=True))      # no second freeze
    p.write_text(p.read_text().replace("seed: 20261118", "seed: 20261119"))
    assert _raises(lambda: P.load(p))                                            # edited after freezing
    (tmp_path / "preregistration.sha256").unlink()
    assert _raises(lambda: P.load(p))                                            # frozen without a hash file


def test_hash_file_without_frozen_flag_is_rejected(tmp_path):
    p = _copy(tmp_path)
    (tmp_path / "preregistration.sha256").write_text("00  preregistration.yaml\n")
    assert _raises(lambda: P.load(p))


def test_freeze_script_check_mode_runs():
    import subprocess
    r = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve().parents[1] / "scripts" / "freeze_prereg.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "ok: frozen=" in r.stdout, r.stderr
