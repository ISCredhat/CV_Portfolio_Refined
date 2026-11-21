"""Part A: run the CRSP study on Isaac's PC after `scripts/download_wrds.py`.

    python scripts/run_crsp.py [--top-n 500] [--first 1990-01] [--last 2025-12] [--jobs 4] [--ggr] [--ggr-only]

Loads only PERMNOs ever in the top N (review M9), runs the pre-registered primary analysis and writes
results/crsp/. Refuses to run unless config/preregistration.yaml is frozen and its hash verifies, so no result
can exist before the registration (use --allow-unfrozen only for a dry run, which is labelled as such).
Memory: the top-N PERMNOs are found a year at a time, the panel reads only those PERMNOs and the columns it uses,
and the GGR replication runs in blocks of portfolio start months (pipeline_crsp.run_ggr_blocks), so a 16 GB PC is
enough. --ggr-only skips the primary analysis (for example after it has finished once).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import crsp as R  # noqa: E402
from pairs import ggr as G  # noqa: E402
from pairs import pipeline_crsp as P  # noqa: E402
from pairs import prereg  # noqa: E402
from pairs import universe as U  # noqa: E402


def write_ggr(res: dict, out_dir: Path) -> None:
    for label, r in res.items():
        r["table1"].to_csv(out_dir / f"ggr_{label}_dofaff_table1.csv")
        r["table2"].to_csv(out_dir / f"ggr_{label}_dofaff_table2.csv")
        r["monthly"].to_csv(out_dir / f"ggr_{label}_monthly.csv")
        (out_dir / f"ggr_{label}_stats.json").write_text(json.dumps(r["stats"], indent=2, default=str))
        P.log(f"GGR {label} written")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top-n", type=int, default=CC.UNIVERSE_TOP_N)
    ap.add_argument("--first", default=CC.MAIN_FIRST_TRADE_MONTH)
    ap.add_argument("--last", default=CC.MAIN_LAST_TRADE_MONTH)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--ggr", action="store_true", help="also run the GGR / Do & Faff replication (1962-2009)")
    ap.add_argument("--ggr-only", action="store_true", help="run only the GGR replication")
    ap.add_argument("--allow-unfrozen", action="store_true", help="dry run before the registration is frozen")
    a = ap.parse_args(argv)

    cfg = prereg.load(verify=not a.allow_unfrozen)
    frozen = bool(cfg["meta"]["frozen"])
    if not frozen and not a.allow_unfrozen:
        print("preregistration.yaml is not frozen: run scripts/freeze_prereg.py first (or --allow-unfrozen).")
        return 2
    raw = CC.WRDS_RAW
    out_dir = CC.RESULTS_CRSP / f"top{a.top_n}{'' if frozen else '_DRYRUN_unfrozen'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_commit": prereg.git_commit(),
            "prereg_frozen": frozen, "prereg_frozen_at": cfg["meta"].get("frozen_at"),
            "prereg_git_commit": cfg["meta"].get("git_commit"), "args": vars(a)}
    (out_dir / f"run_info{'_ggr' if a.ggr_only else ''}.json").write_text(json.dumps(info, indent=2))
    names = R.read_chunks(raw, "stocknames_v2")
    delist = R.read_chunks(raw, "dsedelist") if (raw / "dsedelist").exists() else None

    if not a.ggr_only:
        months = pd.period_range(a.first, a.last, freq="M")
        start = str((months[0] - CC.FORMATION_MONTHS - 1).start_time.date())
        end = str(months[-1].end_time.date())
        P.log(f"PERMNOs ever in the top {a.top_n}")
        keep = U.ever_top_n_chunked(raw, names, months, a.top_n)
        P.log(f"{len(keep)} PERMNOs; loading {start}..{end}")
        dsf = R.load_dsf(raw, start, end, permnos=keep, columns=R.panel_columns(raw))
        P.log(f"{len(dsf)} rows; building the panel")
        inp = P.build_inputs(dsf, names, delist)
        del dsf
        gc.collect()
        out = P.run_main(inp, months, out_dir=out_dir, top_n=a.top_n, n_jobs=a.jobs,
                         fz=prereg.frozen_configuration(cfg))
        for k, v in out["hypotheses"].items():
            print(k, {x: v.get(x) for x in ("p", "p_holm", "reject_holm")}, flush=True)
        del inp, out
        gc.collect()
    if a.ggr or a.ggr_only:
        # the registered target (top 20, wait one day) and its no-wait twin first, then the other selections
        for group in (("top20",), ("top5", "p101_120")):
            cfgs = {f"{s}_wait{w}": G.GGRConfig(wait=w, selection=P.GGR_SELECTIONS[s]) for s in group for w in (1, 0)}
            write_ggr(P.run_ggr_blocks(raw, names, delist, cfgs), out_dir)
            gc.collect()
    print("results in", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
