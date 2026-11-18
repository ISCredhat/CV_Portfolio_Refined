"""The README's stage-two section is generated from results/stage2_*.csv; keep it in sync and clean."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_readme_stage2.py"
NEEDED = [ROOT / "results" / f"stage2_{n}.csv" for n in ("convergence", "markouts", "forecast")]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="stage-2 result CSVs not generated")
def test_readme_stage2_section_is_up_to_date():
    res = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert res.returncode == 0, "README.md is out of date: run python scripts/make_readme_stage2.py"


def test_readme_has_stage2_markers_and_no_addresses():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<!-- STAGE2:BEGIN -->" in text and "<!-- STAGE2:END -->" in text
    assert "## Stage two: can any of this be traded?" in text
    assert "not financial advice" in text
    assert not re.search(r"\b0x[0-9a-fA-F]{40}\b", text)
