"""Cross-check ~10 sampled wallets' token entries against /v2/positions (output in data/private/)."""

import polars as pl
from run_phase3_a import load_phase3_inputs  # noqa: I001

from pmdetect.wallets.positions import PRIVATE, crosscheck_positions

pos = pl.read_parquet(PRIVATE / "positions.parquet")
inp, *_ = load_phase3_inputs()
cand = (
    pos.group_by("wallet").len().filter(pl.col("len") >= 3).sample(10, seed=1)["wallet"].to_list()
)
res = crosscheck_positions(pos, cand, n=10, taker_trades=inp["taker"])
res.write_csv(PRIVATE / "crosscheck_positions.csv")
print(res.height, res["status"].value_counts())
ok = res.filter(pl.col("price_ok").is_not_null())
print(
    "compared",
    ok.height,
    "price_ok",
    int(ok["price_ok"].sum()),
    "size_ok",
    int(ok["size_ok"].sum()),
)
print(
    res.select(
        pl.col("wallet").str.slice(0, 6),
        "status",
        "our_entry",
        "api_avg_price",
        "our_buy_size",
        "api_total_size",
    ).head(40)
)
