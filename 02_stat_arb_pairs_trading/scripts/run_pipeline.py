#!/usr/bin/env python3
"""Recompute every result in results/ from the processed minute panel.

    python scripts/build_data.py                 # once: raw CSVs -> data/processed/
    python scripts/run_pipeline.py               # all stages (about 30 minutes on a laptop)
    python scripts/run_pipeline.py --stage analyse

Stages: select -> costs -> grid -> extras -> analyse (each stage caches to data/processed/).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

from pairs import pipeline as PL  # noqa: E402

STAGES = ["select", "costs", "grid", "extras", "analyse"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--from-stage", choices=STAGES, help="run this stage and every later one")
    args = ap.parse_args()
    todo = STAGES if args.stage == "all" else [args.stage]
    if args.from_stage:
        todo = STAGES[STAGES.index(args.from_stage):]
    t0 = time.time()
    ctx = PL.Context()
    for s in todo:
        PL.log(f"=== stage: {s}")
        if s == "select":
            PL.stage_select(ctx)
        elif s == "costs":
            PL.stage_costs(ctx)
        elif s == "grid":
            PL.stage_grid(ctx)
        elif s == "extras":
            PL.stage_extras(ctx)
        elif s == "analyse":
            from pairs import analysis as A
            A.run_all(ctx)
    PL.log(f"finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
