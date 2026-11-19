"""Freeze config/preregistration.yaml: set frozen: true, record the git commit and time, write the sha256.

    python scripts/freeze_prereg.py            # refuses while a pre_freeze_checks item is false
    python scripts/freeze_prereg.py --check    # only verify hash + agreement with config_crsp, change nothing

Run it once, after Isaac signs off and before the WRDS download; then commit and tag the YAML and its .sha256.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import prereg as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", default=str(CC.PREREG_YAML))
    ap.add_argument("--check", action="store_true", help="verify only")
    ap.add_argument("--allow-open-checks", action="store_true", help="freeze although pre_freeze_checks are open (not advised)")
    a = ap.parse_args()
    path = Path(a.yaml)
    cfg = P.load(path)
    if a.check:
        print(f"ok: frozen={cfg['meta']['frozen']}, sha256={P.sha256_file(path)[:16]}..., open checks: {P.open_checks(cfg) or 'none'}")
        return 0
    try:
        out = P.freeze(path, allow_open_checks=a.allow_open_checks)
    except P.PreregError as e:
        print("not frozen:", e)
        return 1
    print(f"frozen at {out['frozen_at']}, commit {out['git_commit']}, sha256 {out['sha256']}")
    print("Now commit and tag preregistration.yaml and preregistration.sha256 before downloading any CRSP data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
