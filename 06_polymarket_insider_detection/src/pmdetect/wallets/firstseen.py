"""first_seen candidates and the (network) top-up fetch.

:func:`select_candidates` is local and offline: it only reads Parquet and writes
``data/private/first_seen_candidates.parquet`` (wallet-level, therefore private). The candidate set is
the union of

* ``top_oneshot``     wallets behind the top ``n_top`` (default 5,000) winning positions with a valid cutoff,
                      ranked by the one-shot score WITHOUT the freshness component (``score_nofresh``);
* ``novalid_screen``  wallets behind the top ``n_novalid`` (default 1,000) winning positions with no valid
                      cutoff, ranked by the proximity-free screen (these never reach the one-shot tier, so
                      this is the smallest component and can be set to 0);
* ``jump_initiator``  every wallet listed in ``data/private/jump_initiators.csv`` (market pipeline);
* ``calib_ge5``       every wallet with at least ``MIN_CALIB`` (5) calibration-eligible resolved positions
                      (the repeat-test population, whose freshness feeds nothing but the watchlist evidence).

Wallets already holding a first_seen (``wallets.parquet``, ``extra_wallets.parquet``) or already fetched
(any row in ``first_seen_topup.parquet``, including "no trades" rows) are dropped, as are wallets the
non-directional filters flagged in the last ``score-wallets`` run (market makers etc.).
:func:`fetch_candidates` does the network part (``/v2/activity`` asc, limit 1) and is only run by hand.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from pmdetect.store.db import PROJECT_ROOT
from pmdetect.wallets import oneshot, pipeline

PRIVATE = PROJECT_ROOT / "data" / "private"
SNAP = PROJECT_ROOT / "data" / "snapshots"
RESULTS = PROJECT_ROOT / "results"
CANDIDATES = PRIVATE / "first_seen_candidates.parquet"
TOPUP = PRIVATE / "first_seen_topup.parquet"
MIN_CALIB = 5
REQ_PER_S = 5.0  # the shared throttle in collect.http (MIN_INTERVAL = 0.2 s), whatever the thread count
SCHEMA = {"wallet": pl.String, "first_seen": pl.Datetime("us"), "fetched_at": pl.Datetime("us")}


def _atomic_write(df: pl.DataFrame, path: Path, tries: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    for i in range(tries):  # a reader (score-wallets) may hold the target open for a moment on Windows
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.5 * (i + 1))
    tmp.replace(path)


def known_first_seen(snap: Path = SNAP, private: Path = PRIVATE) -> tuple[pl.Series, pl.Series]:
    """(wallets with a non-null first_seen in the store, wallets already fetched into the top-up file)."""
    have = []
    for n in ("wallets.parquet", "extra_wallets.parquet"):
        f = snap / n
        if f.exists():
            have.append(pl.read_parquet(f, columns=["wallet", "first_seen"]).drop_nulls("first_seen")["wallet"])
    store = pl.concat(have).unique() if have else pl.Series("wallet", [], dtype=pl.String)
    tp = private / "first_seen_topup.parquet"
    done = pl.read_parquet(tp, columns=["wallet"])["wallet"] if tp.exists() else pl.Series("wallet", [], dtype=pl.String)
    return store, done


def candidate_table(
    positions: pl.DataFrame,
    scored: pl.DataFrame,
    jump_wallets: pl.Series | None,
    n_top: int = 5000,
    n_novalid: int = 1000,
    exclude: pl.Series | None = None,
) -> pl.DataFrame:
    """Pure function behind :func:`select_candidates` (no I/O): one row per candidate wallet with the
    four membership flags, ``best_score`` (best candidate-position score or null) and a fetch ``priority``
    (1 = first): jump initiators, then top one-shot, then calib_ge5, then the no-valid screen."""
    sc = scored
    if exclude is not None and exclude.len():
        sc = sc.filter(~pl.col("wallet").is_in(exclude.implode()))
    top = (
        sc.filter(~pl.col("no_valid_cutoff")).sort("score_nofresh", descending=True).head(n_top)
        .select("wallet", pl.col("score_nofresh").alias("_s")).unique("wallet", keep="first", maintain_order=True)
    )  # fmt: skip
    nov = (
        sc.filter(pl.col("no_valid_cutoff")).sort("score_cand", descending=True).head(n_novalid)
        .select("wallet", pl.col("score_cand").alias("_s")).unique("wallet", keep="first", maintain_order=True)
    )  # fmt: skip
    ge5 = (
        positions.filter(pl.col("calib_eligible").fill_null(False) & pl.col("won").is_not_null())
        .group_by("wallet").agg(pl.len().alias("_n")).filter(pl.col("_n") >= MIN_CALIB).select("wallet")
    )  # fmt: skip
    jw = (
        pl.DataFrame({"wallet": jump_wallets.drop_nulls().unique()}) if jump_wallets is not None
        else pl.DataFrame(schema={"wallet": pl.String})
    )  # fmt: skip
    if exclude is not None and exclude.len():
        ge5 = ge5.filter(~pl.col("wallet").is_in(exclude.implode()))
        jw = jw.filter(~pl.col("wallet").is_in(exclude.implode()))
    allw = pl.concat([top.select("wallet"), nov.select("wallet"), ge5, jw]).unique()
    out = (
        allw.join(top.rename({"_s": "_top"}), on="wallet", how="left")
        .join(nov.rename({"_s": "_nov"}), on="wallet", how="left")
        .with_columns(
            pl.col("wallet").is_in(top["wallet"].implode()).alias("top_oneshot"),
            pl.col("wallet").is_in(nov["wallet"].implode()).alias("novalid_screen"),
            pl.col("wallet").is_in(jw["wallet"].implode()).alias("jump_initiator"),
            pl.col("wallet").is_in(ge5["wallet"].implode()).alias("calib_ge5"),
            pl.coalesce("_top", "_nov").alias("best_score"),
        )
        .drop("_top", "_nov")
    )
    return out.sort(
        ["jump_initiator", "top_oneshot", "calib_ge5", "novalid_screen", "best_score", "wallet"],
        descending=[True, True, True, True, True, False],
        nulls_last=True,
    ).with_columns(pl.int_range(1, pl.len() + 1).alias("priority"))


def hours_estimate(n: int, req_per_s: float = REQ_PER_S) -> float:
    return n / req_per_s / 3600.0


def select_candidates(
    snap: Path = SNAP,
    private: Path = PRIVATE,
    results: Path = RESULTS,
    n_top: int = 5000,
    n_novalid: int = 1000,
    log=print,
) -> dict:
    """Local selection step (no network). Writes ``first_seen_candidates.parquet`` (only wallets that still
    lack a first_seen, in fetch order) and returns counts plus a printable ``report``."""
    pos = pl.read_parquet(private / "positions.parquet")
    _mk, _ni, _jumps, ms, we, qm = pipeline.load_cutoff_inputs(snap, results)
    scored = oneshot.score_oneshot(pos, qm, None, ms, we)  # no first_seen: freshness absent by construction
    ex = None
    ff = private / "wallet_flags.parquet"
    if ff.exists():
        ex = pl.read_parquet(ff, columns=["wallet", "excluded"]).filter(pl.col("excluded"))["wallet"]
    jf = private / "jump_initiators.csv"
    jw = pl.read_csv(jf, columns=["wallet"], infer_schema_length=0)["wallet"] if jf.exists() else None
    cand = candidate_table(pos, scored, jw, n_top, n_novalid, ex)
    store, done = known_first_seen(snap, private)
    have = pl.concat([store, done]).unique()
    todo = cand.filter(~pl.col("wallet").is_in(have.implode())).with_columns(
        pl.int_range(1, pl.len() + 1).alias("priority")
    )
    _atomic_write(todo, private / "first_seen_candidates.parquet")
    cnt = {k: int(cand[k].sum()) for k in ("top_oneshot", "novalid_screen", "jump_initiator", "calib_ge5")}
    cnt_todo = {k: int(todo[k].sum()) for k in cnt}
    res = {
        "candidates_total": cand.height, "already_known_store": int(cand["wallet"].is_in(store.implode()).sum()),
        "already_fetched_topup": int(cand["wallet"].is_in(done.implode()).sum()),
        "to_fetch": todo.height, "by_component_total": cnt, "by_component_to_fetch": cnt_todo,
        "flagged_excluded": 0 if ex is None else ex.len(), "jump_initiators_file": jw is not None,
        "winning_positions_scored": scored.height,
        "requests": todo.height, "hours_at_5rps": round(hours_estimate(todo.height), 2),
    }  # fmt: skip
    res["report"] = (
        f"first_seen candidates: {cand.height} wallets in the union; "
        f"{res['already_known_store']} already have first_seen in the store, {res['already_fetched_topup']} already "
        f"in the top-up file; TO FETCH {todo.height} -> ~{todo.height} requests, "
        f"{res['hours_at_5rps']:.1f} h at {REQ_PER_S:.0f} req/s.\n"
        f"  components (total / still to fetch): "
        + ", ".join(f"{k} {cnt[k]}/{cnt_todo[k]}" for k in cnt)
        + f"\n  jump_initiators.csv present: {res['jump_initiators_file']}; non-directional wallets dropped: "
        f"{res['flagged_excluded']} flagged\n  written: {private / 'first_seen_candidates.parquet'}"
    )
    log(res["report"])
    return res


def fetch_candidates(
    max_n: int | None = None,
    threads: int = 3,
    private: Path = PRIVATE,
    log=print,
    save_every: int = 200,
    max_consecutive_errors: int = 40,
) -> dict:
    """NETWORK. Fetch first_seen for the wallets in ``first_seen_candidates.parquet`` that are not yet in the
    top-up file, in priority order, with ``threads`` worker threads (the shared throttle keeps the rate at
    ~5 req/s). Restartable: ``first_seen_topup.parquet`` is rewritten atomically every ``save_every``
    wallets and at the end; errors are skipped and retried on the next run."""
    from pmdetect.collect.data_api import ACTIVITY_URL, _ts
    from pmdetect.collect.http import get_json

    cand = pl.read_parquet(private / "first_seen_candidates.parquet")
    tp = private / "first_seen_topup.parquet"
    done = pl.read_parquet(tp) if tp.exists() else pl.DataFrame(schema=SCHEMA)
    todo = cand.filter(~pl.col("wallet").is_in(done["wallet"].implode()))["wallet"].to_list()
    if max_n:
        todo = todo[:max_n]
    log(f"candidates={cand.height} already_fetched={done.height} todo={len(todo)} "
        f"(~{hours_estimate(len(todo)):.1f} h at {REQ_PER_S:.0f} req/s)")  # fmt: skip
    rows = done.to_dicts()
    n_ok = n_err = streak = 0

    def one(w: str):
        page = get_json(
            ACTIVITY_URL, {"user": w, "type": "TRADE", "start": 1, "sort_direction": "ASC", "limit": 1},
            use_cache=False,
        )  # fmt: skip
        data = page.get("data") or []
        return w, (_ts(data[0]["timestamp"]) if data else None)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for a in range(0, len(todo), save_every):
            batch = todo[a : a + save_every]
            for fut in [ex.submit(one, w) for w in batch]:
                try:
                    w, fs = fut.result()
                except Exception as e:  # noqa: BLE001 - skipped, retried on the next run
                    n_err += 1
                    streak += 1
                    log(f"error {type(e).__name__} (wallet skipped, retried next run)")
                    continue
                streak = 0
                n_ok += 1
                rows.append({"wallet": w, "first_seen": fs, "fetched_at": datetime.now(UTC).replace(tzinfo=None)})
            _atomic_write(pl.DataFrame(rows, schema=SCHEMA), tp)
            el = time.time() - t0
            log(f"{min(a + save_every, len(todo))}/{len(todo)} saved={len(rows)} ok={n_ok} err={n_err} "
                f"{(a + len(batch)) / max(el, 1e-9):.2f} req/s")  # fmt: skip
            if streak >= max_consecutive_errors:
                log("too many consecutive errors: stopping (rerun to resume)")
                break
    _atomic_write(pl.DataFrame(rows, schema=SCHEMA), tp)
    log("done")
    return {"ok": n_ok, "errors": n_err, "saved": len(rows)}
