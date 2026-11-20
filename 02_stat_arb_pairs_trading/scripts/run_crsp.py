"""Part A: run the CRSP study on Isaac's PC after `scripts/download_wrds.py`.

    python scripts/run_crsp.py [--top-n 500] [--first 1990-01] [--last 2025-12] [--jobs 4] [--ggr]

Loads only PERMNOs ever in the top N (review M9), runs the pre-registered primary analysis and writes
results/crsp/. Refuses to run unless config/preregistration.yaml is frozen and its hash verifies, so no result
can exist before the registration (use --allow-unfrozen only for a dry run, which is labelled as such).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import crsp as R  # noqa: E402
from pairs import pipeline_crsp as P  # noqa: E402
from pairs import prereg  # noqa: E402
from pairs import universe as U  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top-n", type=int, default=CC.UNIVERSE_TOP_N)
    ap.add_argument("--first", default=CC.MAIN_FIRST_TRADE_MONTH)
    ap.add_argument("--last", default=CC.MAIN_LAST_TRADE_MONTH)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--ggr", action="store_true", help="also run the GGR / Do & Faff replication (1962-2009)")
    ap.add_argument("--allow-unfrozen", action="store_true", help="dry run before the registration is frozen")
    a = ap.parse_args(argv)

    cfg = prereg.load(verify=not a.allow_unfrozen)
    if not cfg.get("frozen") and not a.allow_unfrozen:
        print("preregistration.yaml is not frozen: run scripts/freeze_prereg.py first (or --allow-unfrozen).")
        return 2
    months = pd.period_range(a.first, a.last, freq="M")
    start = str((months[0] - CC.FORMATION_MONTHS - 1).start_time.date())
    end = str(months[-1].end_time.date())
    names = R.read_chunks(CC.WRDS_RAW, "stocknames_v2")
    dsf = R.load_dsf(CC.WRDS_RAW, start, end)
    keep = U.ever_top_n(dsf, names, months, a.top_n)
    dsf = dsf[dsf["permno"].isin(keep)]
    dd = CC.WRDS_RAW / "dsedelist"
    delist = R.read_chunks(CC.WRDS_RAW, "dsedelist") if dd.exists() else None
    inp = P.build_inputs(dsf, names, delist)
    tag = "" if cfg.get("frozen") else "_DRYRUN_unfrozen"
    out_dir = CC.RESULTS_CRSP / f"top{a.top_n}{tag}"
    out = P.run_main(inp, months, out_dir=out_dir, top_n=a.top_n, n_jobs=a.jobs, fz=prereg.frozen_configuration(cfg))
    for k, v in out["hypotheses"].items():
        print(k, {x: v.get(x) for x in ("p", "p_holm", "reject_holm")})
    if a.ggr:
        first = pd.Period(CC.GGR_FIRST_TRADE_MONTH, "M")             # trading months 1963-07 to 2009-06 (Do & Faff)
        last_start = pd.Period(CC.GGR_LAST_TRADE_MONTH, "M") - 5    # last 6-month portfolio ends in the last month
        g = R.load_dsf(CC.WRDS_RAW, str((first - 12).start_time.date()), str(pd.Period(CC.GGR_LAST_TRADE_MONTH, "M").end_time.date()))
        gp = R.build_panel(g, names, delist)
        tr = pd.DataFrame(gp.tr, index=gp.days, columns=gp.permnos)
        for wait in (1, 0):                                      # wait one day = primary target; no wait = secondary
            r = P.run_ggr(tr, str(first), str(last_start), wait=wait, dead=gp.dead, traded=gp.fresh, permco=gp.permco)
            r["table1"].to_csv(out_dir / f"ggr_dofaff_table1_wait{wait}.csv")
            r["table2"].to_csv(out_dir / f"ggr_dofaff_table2_wait{wait}.csv")
    print("results in", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
