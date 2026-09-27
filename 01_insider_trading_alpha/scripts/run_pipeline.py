#!/usr/bin/env python3
"""Rebuild every result in results/ from the raw downloads.

    python scripts/download_data.py --email you@example.com   # once (network)
    python scripts/run_pipeline.py                            # build + analyse
    python scripts/run_pipeline.py --stage analyse            # re-run analysis only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insider_alpha import analysis, pipeline  # noqa: E402
from insider_alpha import config as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "build", "analyse"], default="all")
    ap.add_argument("--raw", default=str(C.DATA_RAW))
    ap.add_argument("--processed", default=str(C.DATA_PROC))
    ap.add_argument("--results", default=str(C.RESULTS))
    a = ap.parse_args()
    if a.stage in ("all", "build"):
        pipeline.build(Path(a.raw), Path(a.processed), Path(a.results))
    if a.stage in ("all", "analyse"):
        analysis.run_all(Path(a.processed), Path(a.results))


if __name__ == "__main__":
    main()
