"""Cross-check taker-only vs taker+maker BUY entries against /v2/positions (restartable; output in data/private/).

For the ~10 wallets of the earlier cross-check (data/private/crosscheck_positions.csv, same wallets and the
same API request parameters, so every response is already in data/raw_cache) or a fresh sample, compare per
(wallet, token): the BUY VWAP and bought size from (a) the taker tape only and (b) the all-rows tape
(taker + maker rows; each row is the wallet's own fill, so maker BUYs count as buys) with the API's
avg_price / total_size.

    python scripts/crosscheck_positions_maker.py --cache-only      # offline: skips any uncached request
    python scripts/crosscheck_positions_maker.py                   # may call the network (a few dozen requests)

Expected: entry prices agree for both (avg_price is a VWAP either way); the SIZE agreement should improve
markedly for wallets that rest limit orders (taker-only undercounts bought size, all-rows matches the API),
and be unchanged for purely-taker wallets. Aggregate-only printout; per-wallet rows go to data/private/.
"""

from __future__ import annotations

import sys

import polars as pl

from pmdetect.collect import http
from pmdetect.wallets import positions as posmod
from pmdetect.wallets.pipeline import PRIVATE, SNAP, load_taker_lazy

CACHE_ONLY = "--cache-only" in sys.argv


def _guarded_get_json(url, params=None, *, use_cache=True):
    if CACHE_ONLY and not http._cache_path(url, params).exists():
        raise RuntimeError("uncached request skipped (--cache-only)")
    return _orig(url, params, use_cache=use_cache)


_orig = http.get_json
http.get_json = _guarded_get_json  # _api_positions imports get_json lazily from this module


def _buys(lf: pl.LazyFrame, w: str, label: str) -> pl.DataFrame:
    return (
        lf.filter((pl.col("wallet") == w) & (pl.col("side") == "BUY"))
        .group_by("token_id", "condition_id")
        .agg(pl.col("size").sum().alias(f"{label}_size"), (pl.col("size") * pl.col("price")).sum().alias("_u"))
        .with_columns((pl.col("_u") / pl.col(f"{label}_size")).alias(f"{label}_entry"))
        .drop("_u")
        .collect()
    )


def main() -> None:
    prev = PRIVATE / "crosscheck_positions.csv"
    if prev.exists():
        wallets = pl.read_csv(prev)["wallet"].unique().to_list()[:10]
    else:
        pos = pl.read_parquet(PRIVATE / "positions.parquet")
        wallets = pos.group_by("wallet").len().filter(pl.col("len") >= 3).sample(10, seed=1)["wallet"].to_list()
    taker = load_taker_lazy().select("wallet", "condition_id", "token_id", "side", "size", "price")
    allr = pl.scan_parquet(SNAP / "trades.parquet").select("wallet", "condition_id", "token_id", "side", "size", "price")
    rows = []
    for w in wallets:
        t, a = _buys(taker, w, "taker"), _buys(allr, w, "all")
        conds = a["condition_id"].unique().to_list()
        try:
            api = posmod._api_positions(w, conds)
        except Exception as e:  # noqa: BLE001
            rows.append({"wallet": w, "status": f"api_error:{type(e).__name__}"})
            continue
        by = {r["token_id"]: r for r in api}
        j = a.join(t.drop("condition_id"), on="token_id", how="left")
        for r in j.iter_rows(named=True):
            x = by.get(r["token_id"])
            if x is None:
                rows.append({"wallet": w, **r, "status": "missing_in_api"})
                continue
            ap, asz = float(x["avg_price"]), float(x["total_size"])
            rows.append({
                "wallet": w, **r, "api_avg_price": ap, "api_total_size": asz, "status": x["_status"],
                "size_ok_taker": abs(asz - (r["taker_size"] or 0)) <= 0.15 * max(asz, r["taker_size"] or 0),
                "size_ok_all": abs(asz - r["all_size"]) <= 0.15 * max(asz, r["all_size"]),
                "price_ok_taker": r["taker_entry"] is not None and abs(ap - r["taker_entry"]) <= 0.02,
                "price_ok_all": abs(ap - r["all_entry"]) <= 0.02,
            })  # fmt: skip
    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_csv(PRIVATE / "crosscheck_positions_maker.csv")
    ok = res.filter(pl.col("size_ok_all").is_not_null()) if "size_ok_all" in res.columns else res.head(0)
    print("compared tokens:", ok.height)
    for c in ("price_ok_taker", "price_ok_all", "size_ok_taker", "size_ok_all"):
        if c in ok.columns:
            print(c, int(ok[c].sum()))


if __name__ == "__main__":
    main()
