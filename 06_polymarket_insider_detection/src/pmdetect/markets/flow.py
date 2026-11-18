"""Informed flow BEFORE the resolving move (phase 2b), no price-defined news time needed.

For each market with a detected jump (``changepoint.detect_jump``), the window (phase 2c) is
``[js - 24 h, end]`` with ``js = jump_start_ts`` and ``end`` = the earlier of (a) the first
observation after ``js`` at which the winner price q has covered half of the jump
(``q >= pre_level + 0.5 * jump_size``) and (b) ``js + 2 h``. So the window contains the 24 h of
positioning before the move AND the trades that caused the first part of the jump (the 2b window
stopped at ``js`` and missed them). Without a price path the end falls back to ``js + 2 h``
(capped at ``jump_peak_ts``). Over the taker tape (one row per fill), in that window:

* ``fb_ofi``    net winner-side order-flow imbalance: (USD toward the winner - USD away) / total USD,
                where "toward" = buy the winning token or sell the losing token;
* ``fb_fresh``  share of winner-side USD from fresh wallets (first_seen less than 7 days before the
                trade, or a wallet seen in a single market); needs wallet coverage (below);
* ``fb_hhi``    Herfindahl concentration of winner-side USD across wallets;
* ``fb_vol``    mean hourly USD volume (window length is 24-26 h, so per hour).

Each is compared with the market's OWN earlier history: the same statistic computed on the
consecutive 24 h blocks before the window start ``js - 24 h`` (up to ``MAX_BLOCKS``, only blocks fully inside the
market's tape), giving a robust z = (x - median) / (1.4826 * MAD) (std when MAD is 0). The z
columns (``fb_ofi_z`` ...) are what ``combine`` ranks within info_type. Nulls come with
``fb_reason`` ("no_jump", "short_history" (< 3 usable history blocks), "thin_window" (< 3 trades),
"wallet_coverage_low" (< 50 % of the winner-side USD has a known first_seen)).

Question answered: was there unusual, one-sided, concentrated or fresh-money flow just before the
price started to move, measured against how this market behaved before?
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

MAD_SCALE = 1.4826
WINDOW = timedelta(hours=24)
MAX_BLOCKS = 14
MIN_BLOCKS = 3
MIN_TRADES = 3
MIN_COVERAGE = 0.5
FRESH_DAYS = timedelta(days=7)
FLOW_COLS = ["fb_ofi_z", "fb_fresh_z", "fb_hhi_z", "fb_vol_z"]
RAW_COLS = ["fb_ofi", "fb_fresh", "fb_hhi", "fb_vol"]
OUT_COLS = [
    "condition_id", *RAW_COLS, *FLOW_COLS, "fb_n_trades", "fb_usd", "fb_hist_blocks",
    "fb_fresh_cov", "fb_reason", "fb_win_start", "fb_win_end",
]  # fmt: skip
CAUSE_H = 2.0  # cap on the part of the jump included in the window


def robust_z(
    x: float, hist: list[float], floor: float = 0.0, rel_floor: float = 0.0
) -> float | None:
    """Robust z of ``x`` against ``hist`` (median/MAD, std fallback); None if degenerate."""
    h = np.asarray(hist, dtype=float)
    med = float(np.median(h))
    scale = MAD_SCALE * float(np.median(np.abs(h - med)))
    if scale <= 0:
        scale = float(np.std(h))
    scale = max(scale, floor, rel_floor * abs(med))
    if scale <= 0:
        return None
    return (x - med) / scale


def flow_window(js, jump: dict | None = None, pts: tuple | None = None) -> tuple:
    """(window_start, window_end) for a jump starting at ``js``.

    end = first observation after js with q >= pre_level + 0.5 * jump_size, but at most
    js + CAUSE_H hours. ``pts`` = (ts, q) winner price path (numpy); ``jump`` = dict with
    pre_level, jump_size, jump_peak_ts (any may be None).
    """
    cap = js + timedelta(hours=CAUSE_H)
    jump = jump or {}
    end = cap
    pre, size = jump.get("pre_level"), jump.get("jump_size")
    if pts is not None and pre is not None and size is not None:
        ts, q = pts
        m = (ts > np.datetime64(js, "us")) & (q >= pre + 0.5 * size)
        idx = np.where(m)[0]
        if idx.size:
            t = ts[idx[0]].astype("datetime64[us]").item()
            end = min(end, t)
    elif jump.get("jump_peak_ts") is not None:
        end = min(end, jump["jump_peak_ts"])
    return js - WINDOW, max(end, js)


def _prep(tr: pl.DataFrame, outcome: str, fs: pl.DataFrame | None) -> pl.DataFrame:
    win = {"YES": "yes", "NO": "no"}[outcome]
    tr = tr.with_columns(pl.col("outcome_label").str.to_lowercase().alias("_lab"))
    tr = tr.filter(pl.col("_lab").is_in(["yes", "no"]) & (pl.col("usd") > 0)).sort("ts")
    toward = ((pl.col("_lab") == win) & (pl.col("side").str.to_uppercase() == "BUY")) | (
        (pl.col("_lab") != win) & (pl.col("side").str.to_uppercase() == "SELL")
    )
    tr = tr.with_columns(toward.alias("toward"))
    if fs is not None:
        tr = tr.join(fs, on="wallet", how="left")
    else:
        tr = tr.with_columns(
            pl.lit(None, dtype=pl.Datetime("us")).alias("first_seen"),
            pl.lit(None, dtype=pl.Int64).alias("n_markets"),
        )
    known = pl.col("first_seen").is_not_null()
    fresh = (pl.col("first_seen") > pl.col("ts") - FRESH_DAYS) | (
        pl.col("n_markets").fill_null(0) == 1
    )
    return tr.with_columns(known.alias("_known"), (known & fresh).alias("_fresh"))


def _stats(b: pl.DataFrame) -> dict:
    """ofi, hhi, fresh (None if coverage low), fresh_cov, usd of one block of prepared trades."""
    usd = float(b["usd"].sum())
    out = {"usd": usd, "n": b.height, "ofi": None, "hhi": None, "fresh": None, "cov": None}
    if b.height == 0 or usd <= 0:
        return out
    signed = b.select(
        pl.when(pl.col("toward")).then(pl.col("usd")).otherwise(-pl.col("usd")).sum()
    ).item()
    out["ofi"] = float(signed) / usd
    w = b.filter(pl.col("toward"))
    wt = float(w["usd"].sum())
    if wt > 0:
        per = w.group_by("wallet").agg(pl.col("usd").sum())
        out["hhi"] = float(((per["usd"] / wt) ** 2).sum())
        cov = float(w.filter(pl.col("_known"))["usd"].sum()) / wt
        out["cov"] = cov
        if cov >= MIN_COVERAGE:
            out["fresh"] = float(w.filter(pl.col("_fresh"))["usd"].sum()) / wt
    return out


def market_flow(
    cid: str, outcome: str, js, tr: pl.DataFrame, fs: pl.DataFrame | None,
    jump: dict | None = None, pts: tuple | None = None, legacy: bool = False,
) -> dict:
    row = {c: None for c in OUT_COLS}
    row["condition_id"] = cid
    if js is None:
        row["fb_reason"] = "no_jump"
        return row
    if tr.height == 0:
        row["fb_reason"] = "no_taker_trades"
        return row
    tr = _prep(tr, outcome, fs)
    if tr.height == 0:
        row["fb_reason"] = "no_taker_trades"
        return row
    ws, we = flow_window(js, jump, pts)
    if legacy:  # phase 2b window [js - 24 h, js), kept for the old-vs-new comparison
        we = js - timedelta(microseconds=1)
    row["fb_win_start"], row["fb_win_end"] = ws, we
    hours = (we - ws).total_seconds() / 3600.0
    first = tr["ts"].min()
    first_hour = first.replace(minute=0, second=0, microsecond=0)
    win = tr.filter((pl.col("ts") >= ws) & (pl.col("ts") <= we))
    row["fb_n_trades"] = win.height
    row["fb_usd"] = float(win["usd"].sum())
    # history blocks fully inside the tape, newest first
    blocks = []
    for b in range(MAX_BLOCKS):
        b_end = ws - WINDOW * b
        b_start = b_end - WINDOW
        if b_start < first_hour:
            break
        blocks.append(_stats(tr.filter((pl.col("ts") >= b_start) & (pl.col("ts") < b_end))))
    row["fb_hist_blocks"] = len(blocks)
    if len(blocks) < MIN_BLOCKS:
        row["fb_reason"] = "short_history"
        return row
    s = _stats(win)
    row["fb_vol"] = s["usd"] / hours
    hv = [x["usd"] / 24.0 for x in blocks]
    row["fb_vol_z"] = robust_z(row["fb_vol"], hv, rel_floor=0.1)
    row["fb_fresh_cov"] = s["cov"]
    reasons = []
    if win.height < MIN_TRADES:
        reasons.append("thin_window")
    else:
        for key, zcol, rawcol, floor in (
            ("ofi", "fb_ofi_z", "fb_ofi", 0.05),
            ("hhi", "fb_hhi_z", "fb_hhi", 0.02),
            ("fresh", "fb_fresh_z", "fb_fresh", 0.05),
        ):
            row[rawcol] = s[key]
            hist = [x[key] for x in blocks if x[key] is not None]
            if s[key] is None:
                if key == "fresh":
                    reasons.append("wallet_coverage_low")
                continue
            if len(hist) >= MIN_BLOCKS:
                row[zcol] = robust_z(s[key], hist, floor)
            else:
                reasons.append(f"{key}_short_history")
    row["fb_reason"] = "; ".join(reasons) if reasons else None
    return row


def flow_before_move(
    markets: pl.DataFrame,
    jumps: pl.DataFrame,
    taker: pl.DataFrame,
    wallets: pl.DataFrame | None,
    pts: dict | None = None,
    legacy: bool = False,
) -> pl.DataFrame:
    """One row per market in ``markets`` (condition_id, outcome) joined with ``jumps``.

    ``pts``: optional {condition_id: (ts, q)} winner price paths (``changepoint.winner_points``)
    used to find the half-jump point that ends the window.
    """
    fs = None
    if wallets is not None and wallets.height:
        fs = wallets.select(
            "wallet",
            "first_seen",
            (pl.col("n_markets").cast(pl.Int64) if "n_markets" in wallets.columns
             else pl.lit(None, dtype=pl.Int64).alias("n_markets")),
        ).unique(subset=["wallet"])
    jrows = {r["condition_id"]: r for r in jumps.iter_rows(named=True)}
    jmap = {k: v["jump_start_ts"] for k, v in jrows.items()}
    pts = pts or {}
    parts = {k[0]: v for k, v in taker.partition_by("condition_id", as_dict=True).items()}
    rows = []
    for cid, outcome in markets.select("condition_id", "outcome").iter_rows():
        if outcome not in ("YES", "NO"):
            r = {**{c: None for c in OUT_COLS}, "condition_id": cid, "fb_reason": "no_outcome"}
        else:
            r = market_flow(cid, outcome, jmap.get(cid), parts.get(cid, pl.DataFrame()), fs,
                          jrows.get(cid), pts.get(cid), legacy)
        rows.append(r)
    schema = {c: pl.Float64 for c in OUT_COLS}
    schema.update({"condition_id": pl.String, "fb_reason": pl.String, "fb_n_trades": pl.Int64,
                   "fb_hist_blocks": pl.Int64, "fb_win_start": pl.Datetime("us"),
                   "fb_win_end": pl.Datetime("us")})
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None)


INIT_COLS = [
    "condition_id", "case_name", "is_labelled", "question", "jump_start_ts", "win_start", "win_end",
    "jump_reason", "rank", "wallet", "winner_usd", "share_of_winner_usd", "n_trades",
    "first_trade_in_window", "first_seen", "first_seen_src", "days_first_seen_to_first_trade",
    "n_markets_store", "only_market_store", "n_markets_global", "window_winner_usd",
    "first_seen_in_window",
]  # fmt: skip


def jump_initiators(
    markets: pl.DataFrame,
    jumps: pl.DataFrame,
    taker: pl.DataFrame,
    wallets: pl.DataFrame | None,
    pts: dict | None = None,
    labelled: dict[str, str] | None = None,
    top_n: int = 3,
    top_n_labelled: int = 5,
    extra_wallets: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Top wallets by winner-side taker USD inside the flow window (the jump-initiating trades).

    ``labelled``: {condition_id: case_name}; these markets get ``top_n_labelled`` rows and a row
    with the reason even when no clear jump exists (then no window, no wallet).
    ``first_seen``: from ``wallets`` (first_seen), else ``extra_wallets``; ``first_seen_src`` says
    which ("wallets", "extra_wallets" or "none"; none = not covered by either backfill, the
    wallet's first trade in the tape is then the only lower bound on its age, in
    ``first_trade_in_window``/``n_markets_store``). ``n_markets_store`` = distinct markets the
    wallet traded in the taker tape provided (pilot + extra store); ``only_market_store`` is
    true when that is 1 (a lower bound on breadth: the store is a subset of Polymarket).
    ``n_markets_global`` = ``n_markets`` of the wallets table when known (not for extra wallets).
    """
    labelled = labelled or {}
    pts = pts or {}
    fs_main = fs_extra = None
    if wallets is not None and wallets.height:
        fs_main = wallets.select(
            "wallet", "first_seen",
            (pl.col("n_markets").cast(pl.Int64) if "n_markets" in wallets.columns
             else pl.lit(None, dtype=pl.Int64).alias("n_markets")),
        ).unique(subset=["wallet"])
    if extra_wallets is not None and extra_wallets.height:
        fs_extra = extra_wallets.select("wallet", "first_seen").unique(subset=["wallet"])
    breadth = taker.group_by("wallet").agg(pl.col("condition_id").n_unique().alias("n_markets_store"))
    bmap = dict(breadth.iter_rows())
    main_map = {} if fs_main is None else {r[0]: (r[1], r[2]) for r in fs_main.iter_rows()}
    extra_map = {} if fs_extra is None else dict(fs_extra.iter_rows())
    jrows = {r["condition_id"]: r for r in jumps.iter_rows(named=True)}
    parts = {k[0]: v for k, v in taker.partition_by("condition_id", as_dict=True).items()}
    rows = []
    for m in markets.select("condition_id", "outcome", "question").iter_rows(named=True):
        cid = m["condition_id"]
        j = jrows.get(cid)
        is_lab = cid in labelled
        base = {c: None for c in INIT_COLS}
        base.update(condition_id=cid, case_name=labelled.get(cid), is_labelled=is_lab,
                    question=m["question"])  # fmt: skip
        js = None if j is None else j["jump_start_ts"]
        tr = parts.get(cid)
        if m["outcome"] not in ("YES", "NO") or js is None or tr is None or tr.height == 0:
            if is_lab:
                reason = (
                    "no_outcome" if m["outcome"] not in ("YES", "NO")
                    else (j["jump_reason"] if j and j["jump_reason"] else "no_taker_trades")
                )
                rows.append({**base, "jump_reason": reason})
            continue
        ws, we = flow_window(js, j, pts.get(cid))
        t = _prep(tr, m["outcome"], None).filter(
            pl.col("toward") & (pl.col("ts") >= ws) & (pl.col("ts") <= we)
        )
        base.update(jump_start_ts=js, win_start=ws, win_end=we)
        tot = float(t["usd"].sum())
        base["window_winner_usd"] = tot
        if t.height == 0:
            rows.append({**base, "jump_reason": "no_winner_side_trades_in_window"})
            continue
        agg = (
            t.group_by("wallet")
            .agg(pl.col("usd").sum().alias("usd"), pl.len().alias("n"),
                 pl.col("ts").min().alias("t0"))
            .sort("usd", descending=True)
            .head(top_n_labelled if is_lab else top_n)
        )  # fmt: skip
        for rk, r in enumerate(agg.iter_rows(named=True), start=1):
            w = r["wallet"]
            fs, gm = main_map.get(w, (None, None))
            src = "wallets" if fs is not None else None
            if fs is None and w in extra_map:
                fs, src = extra_map[w], "extra_wallets"
            nst = bmap.get(w)
            rows.append({
                **base, "rank": rk, "wallet": w, "winner_usd": r["usd"],
                "share_of_winner_usd": r["usd"] / tot if tot > 0 else None, "n_trades": r["n"],
                "first_trade_in_window": r["t0"], "first_seen": fs, "first_seen_src": src or "none",
                "days_first_seen_to_first_trade": (
                    None if fs is None else (r["t0"] - fs).total_seconds() / 86400.0
                ),
                "n_markets_store": nst, "only_market_store": None if nst is None else nst == 1,
                "n_markets_global": gm, "first_seen_in_window": (
                    None if fs is None else (ws <= fs <= we)
                ),
            })  # fmt: skip
    schema = {c: pl.String for c in INIT_COLS}
    schema.update({
        "is_labelled": pl.Boolean, "jump_start_ts": pl.Datetime("us"), "win_start": pl.Datetime("us"),
        "win_end": pl.Datetime("us"), "rank": pl.Int64, "winner_usd": pl.Float64,
        "share_of_winner_usd": pl.Float64, "n_trades": pl.Int64,
        "first_trade_in_window": pl.Datetime("us"), "first_seen": pl.Datetime("us"),
        "days_first_seen_to_first_trade": pl.Float64, "n_markets_store": pl.Int64,
        "only_market_store": pl.Boolean, "n_markets_global": pl.Int64,
        "window_winner_usd": pl.Float64, "first_seen_in_window": pl.Boolean,
    })  # fmt: skip
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None)
