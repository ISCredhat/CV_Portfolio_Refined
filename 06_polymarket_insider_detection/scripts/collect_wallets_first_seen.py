"""Background wallet pass (resumable). Usage: python scripts/collect_wallets_first_seen.py [limit]"""

import sys
from datetime import datetime

from pmdetect.collect.data_api import collect_wallet_first_seen

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
collect_wallet_first_seen(
    limit, log=lambda m: print(datetime.now().isoformat(timespec="seconds"), m, flush=True)
)
