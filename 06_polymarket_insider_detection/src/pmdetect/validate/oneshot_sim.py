"""Phase 4B / 4C validation of the one-shot bet screen (synthetic insiders and the label-flip placebo).

Offline, read-only: reads ``data/snapshots/*.parquet``, ``data/private/*.parquet`` and ``results/market_scores.csv``;
writes only aggregate files to ``results/`` (no wallet-level rows; synthetic wallets are ``synthetic-0001``-style ids).
Thresholds are the PLAN's (written before any run): tier = score >= the 0.99 quantile of the real
valid-cutoff winning-position scores AND USD >= 100; a grid cell is "detected" at a rate >= 50%; the flipped-label
placebo passes when no more than 2% of the flipped losers land in the tier.

Model of a synthetic insider (idealised on purpose)
---------------------------------------------------
For a market drawn from the markets with an INDEPENDENT news time (so the cutoff is valid and not price-based), one
fresh wallet buys the market's WINNING token for ``s`` USD at ``k`` hours before the cutoff, at the then-market price
(last winner-token taker trade within 6 h before the entry time, else the last winner-token price observation within
24 h; no price, no injection for that cell). Wallet age at entry is Uniform(0, 3) days (``first_seen`` = entry
time minus age), the wallet trades only this market (``n_markets_global`` = 1, only market in the store) and the
position wins. The position is scored with the REAL one-shot components (``wallets/oneshot.py``).

Independent scoring (the documented equivalent of "re-score one injected row at a time")
-----------------------------------------------------------------------------------------
``score_oneshot`` turns every component into a rank percentile (rank / n, average ties) of the real population:
within the market (event) for improbability and size (global fallback when the group has fewer than 5 rows,
min with the global percentile), global for wallet age and number of markets. A synthetic row is scored as if it
were the ONLY extra row: its percentile is ``(L + (E + 2) / 2) / (n + 1)`` where ``L`` / ``E`` are the numbers of
real values strictly below / equal to it among the ``n`` real rows of the relevant reference set (exactly the
average rank of the row in the set plus itself). The real rows' own percentiles shift by at most one rank, which
is ignored (that is the point of one-at-a-time scoring). ``parity_check`` verifies the fast path against the real
``score_oneshot`` on the real positions plus a handful of injected rows.
The real 99th-percentile tier threshold is fixed from the REAL winners before and independent of any injection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.store.db import PROJECT_ROOT
from pmdetect.wallets import oneshot
from pmdetect.watchlist import MIN_USD, ONE_SHOT_Q

SEED = 20261113
S_GRID = (500.0, 2000.0, 5000.0, 20000.0, 50000.0)
K_GRID = (2.0, 6.0, 12.0, 24.0, 48.0, 72.0)
K_EXTENDED = (120.0, 168.0, 240.0)  # supplementary: beyond the flat part of the proximity function
CRITERION = 0.50  # PLAN 4B: a cell is detected at a rate >= 50%
PLACEBO_MAX = 0.02  # PLAN 4C(c)
N_BOOT = 1000
AGE_MAX_DAYS = 3.0
TAPE_TOL = timedelta(hours=6)
SERIES_TOL = timedelta(hours=24)
COMP_NAMES = oneshot.COMPONENTS

SNAP = PROJECT_ROOT / "data" / "snapshots"
PRIVATE = PROJECT_ROOT / "data" / "private"
RESULTS = PROJECT_ROOT / "results"


# --------------------------------------------------------------------------------------- reference sets


def avg_rank(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Average rank of x if inserted into ``sorted_arr`` (ties share the mean rank, x counted as one more row)."""
    lo = np.searchsorted(sorted_arr, x, side="left")
    hi = np.searchsorted(sorted_arr, x, side="right")
    return lo + (hi - lo + 2) / 2.0


def pct_in(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile (rank / n) of x among sorted_arr plus itself."""
    return avg_rank(sorted_arr, x) / (len(sorted_arr) + 1)


def prepare_reference_frame(positions: pl.DataFrame, wallets: pl.DataFrame | None) -> pl.DataFrame:
    """The pre-cutoff position table exactly as ``score_oneshot`` builds it before percentiles:
    ``mkey`` (event key), ``_improb``, ``usd``, ``_age_days`` (null without first_seen) and ``n_markets_global``."""
    p = positions
    if "event_id" not in p.columns:
        p = p.with_columns(pl.col("condition_id").alias("event_id"))
    p = p.with_columns(pl.col("event_id").fill_null(pl.col("condition_id")).alias("mkey"))
    p = p.filter(pl.col("hours_before_cutoff").is_null() | (pl.col("hours_before_cutoff") > 0))
    p = p.with_columns((-pl.col("entry_price").clip(0.01, 0.99).log()).alias("_improb"))
    if wallets is not None and wallets.height:
        w = wallets.select(
            "wallet", "first_seen", pl.col("n_markets").cast(pl.Float64).alias("n_markets_global")
        ).unique("wallet")
        p = p.join(w, on="wallet", how="left")
    else:
        p = p.with_columns(
            pl.lit(None, pl.Datetime("us")).alias("first_seen"),
            pl.lit(None, pl.Float64).alias("n_markets_global"),
        )
    return p.with_columns(
        ((pl.col("first_ts") - pl.col("first_seen")).dt.total_seconds() / 86400.0)
        .clip(lower_bound=0)
        .alias("_age_days")
    )


@dataclass
class Reference:
    """Sorted real reference sets for percentile look-ups, plus the context table."""

    g_improb: np.ndarray
    g_usd: np.ndarray
    g_age: np.ndarray
    g_nm: np.ndarray
    groups: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    ctx: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls, frame: pl.DataFrame, ctx: pl.DataFrame | None, keys: list[str] | None = None
    ) -> Reference:
        def srt(col: str) -> np.ndarray:
            return np.sort(frame[col].drop_nulls().drop_nans().to_numpy().astype(float))

        sub = frame if keys is None else frame.filter(pl.col("mkey").is_in(pl.Series(keys).implode()))
        agg = sub.group_by("mkey").agg(pl.col("_improb"), pl.col("usd"))
        groups = {
            r["mkey"]: (np.sort(np.asarray(r["_improb"], float)), np.sort(np.asarray(r["usd"], float)))
            for r in agg.iter_rows(named=True)
        }
        cmap = {}
        if ctx is not None and ctx.height:
            cmap = dict(zip(ctx["key"].to_list(), ctx["c_context"].to_list(), strict=True))
        return cls(srt("_improb"), srt("usd"), srt("_age_days"), srt("n_markets_global"), groups, cmap)

    def context_for(self, event_id: str, condition_id: str) -> float:
        v = self.ctx.get(event_id)
        if v is None or v != v:
            v = self.ctx.get(condition_id)
        return oneshot.NEUTRAL if v is None or v != v else float(v)


def score_injections(
    ref: Reference,
    event_ids: list[str],
    condition_ids: list[str],
    entry_price: np.ndarray,  # (M, K) nan = unavailable
    ks: np.ndarray,  # (K,)
    ss: np.ndarray,  # (S,)
    age_days: np.ndarray,  # (M, K, S)
) -> dict[str, np.ndarray]:
    """Components and score of every injected position, each scored independently (see module docstring).

    Returns ``comps`` (M, K, S, 5) in ``COMP_NAMES`` order and ``score`` (M, K, S); unavailable cells are nan."""
    M, K, S = len(event_ids), len(ks), len(ss)
    comps = np.full((M, K, S, len(COMP_NAMES)), np.nan)
    n_age, n_nm = len(ref.g_age), len(ref.g_nm)
    # global-only quantities
    s_nm = 1.0 - (avg_rank(ref.g_nm, np.array([1.0])) / (n_nm + 1))[0] if n_nm else np.nan
    prox = np.array([oneshot.proximity(float(k)) for k in ks])
    usd_g = pct_in(ref.g_usd, ss)  # (S,)
    for i in range(M):
        pr = entry_price[i]  # (K,)
        ok = np.isfinite(pr)
        if not ok.any():
            continue
        imp = -np.log(np.clip(np.where(ok, pr, 0.5), 0.01, 0.99))
        g = ref.groups.get(event_ids[i])
        n_g = 0 if g is None else len(g[0])
        if g is not None and n_g + 1 >= oneshot.MIN_GROUP:
            w_imp = pct_in(g[0], imp)
            w_usd = pct_in(g[1], ss)
        else:
            w_imp = pct_in(ref.g_improb, imp)
            w_usd = usd_g
        c_imp = np.minimum(w_imp, pct_in(ref.g_improb, imp))  # (K,)
        c_size = np.minimum(w_usd, usd_g)  # (S,)
        ctx = ref.context_for(event_ids[i], condition_ids[i])
        a = age_days[i]  # (K, S)
        s_age = 1.0 - avg_rank(ref.g_age, a.ravel()).reshape(a.shape) / (n_age + 1) if n_age else np.full(a.shape, np.nan)
        parts = np.stack([s_age, np.full(a.shape, s_nm), np.ones(a.shape)])  # (3, K, S)
        c_fresh = np.nanmean(parts, axis=0)
        comps[i, :, :, 0] = c_imp[:, None]
        comps[i, :, :, 1] = c_size[None, :]
        comps[i, :, :, 2] = prox[:, None]
        comps[i, :, :, 3] = c_fresh
        comps[i, :, :, 4] = ctx
        comps[i, ~ok] = np.nan
    comps = np.where(np.isnan(comps) & ~np.isnan(comps[..., :1]), oneshot.NEUTRAL, comps)
    score = np.prod(comps, axis=-1) ** (1.0 / len(COMP_NAMES))
    return {"comps": comps, "score": score}


# ------------------------------------------------------------------------------------------ tier / FPR


def real_tier(scored: pl.DataFrame, excluded: pl.Series | None) -> tuple[pl.DataFrame, float]:
    """Real valid-cutoff winning positions (excluded wallets removed, as in the watchlist) and the tier
    threshold q = ``ONE_SHOT_Q`` quantile of their scores."""
    ok = scored if excluded is None else scored.filter(~pl.col("wallet").is_in(excluded.implode()))
    ok = ok.filter(~pl.col("no_valid_cutoff"))
    return ok, float(ok["score"].quantile(ONE_SHOT_Q))


def in_tier(score: np.ndarray, usd: np.ndarray, q: float) -> np.ndarray:
    return (score >= q) & (usd >= MIN_USD)


def bootstrap_rate(
    det: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rate, 2.5% and 97.5% percentile of the market-resampled rate. ``det`` is (M, ...) with nan = unavailable."""
    M = det.shape[0]
    valid = np.isfinite(det).astype(float)
    d = np.nan_to_num(det).astype(float)
    flat_d, flat_v = d.reshape(M, -1), valid.reshape(M, -1)
    rate = flat_d.sum(0) / np.maximum(flat_v.sum(0), 1)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, flat_d.shape[1]))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
        out[b] = (w @ flat_d) / np.maximum(w @ flat_v, 1)
    lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
    shape = det.shape[1:]
    return rate.reshape(shape), lo.reshape(shape), hi.reshape(shape)


# ------------------------------------------------------------------------------------- market candidates


def eligible_markets(
    markets: pl.DataFrame, news_indep: pl.DataFrame, tokens: pl.DataFrame
) -> pl.DataFrame:
    """Markets with an independent news time (valid cutoff = that time), a YES/NO outcome and a winner token.
    Columns: condition_id, event_id, outcome, token_id (winner), cutoff_ts, info_type."""
    ni = news_indep.filter(pl.col("independent").fill_null(False) & pl.col("news_ts_indep").is_not_null())
    m = markets.select("condition_id", "event_id", "outcome", "info_type").join(
        ni.select("condition_id", pl.col("news_ts_indep").alias("cutoff_ts")), on="condition_id"
    )
    m = m.filter(pl.col("outcome").is_in(["YES", "NO"]))
    tk = tokens.with_columns(pl.col("outcome_label").str.to_uppercase().alias("_o"))
    m = m.join(tk, left_on=["condition_id", "outcome"], right_on=["condition_id", "_o"], how="inner")
    m = m.with_columns(pl.col("event_id").fill_null(pl.col("condition_id")))
    return m.select("condition_id", "event_id", "outcome", "token_id", "cutoff_ts", "info_type").unique(
        "condition_id"
    )


def draw_markets(elig: pl.DataFrame, n_markets: int, seed: int = SEED) -> pl.DataFrame:
    """Random draw (fixed seed) of ``n_markets`` eligible markets; 0 or more than available = all."""
    elig = elig.sort("condition_id")
    if n_markets and n_markets < elig.height:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(elig.height, size=n_markets, replace=False))
        return elig[idx.tolist()]
    return elig


def entry_prices(
    cands: pl.DataFrame, ks: tuple[float, ...], tape: pl.DataFrame, series: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Then-market winner-token price at cutoff - k h for each candidate and k.

    ``tape``: token_id, ts, price (taker fills of the winner tokens); ``series``: token_id, ts, price.
    Order of preference: last tape trade within 6 h before the entry time, else the last series observation within
    24 h. Returns (price (M, K) with nan, source (M, K) with 0 none / 1 tape / 2 series)."""
    M, K = cands.height, len(ks)
    base = cands.select("token_id", "cutoff_ts").with_row_index("_i")
    rows = pl.concat(
        [base.with_columns(pl.lit(j).alias("_j"), (pl.col("cutoff_ts") - pl.duration(hours=k)).alias("t"))
         for j, k in enumerate(ks)]
    ).sort("t")  # fmt: skip

    def asof(src: pl.DataFrame, tol: timedelta, name: str) -> pl.DataFrame:
        s = src.select("token_id", pl.col("ts").alias("t"), pl.col("price").alias(name)).sort("t")
        return rows.join_asof(s, on="t", by="token_id", strategy="backward", tolerance=tol).select(
            "_i", "_j", name
        )

    a = asof(tape, TAPE_TOL, "p_tape")
    b = asof(series, SERIES_TOL, "p_ser")
    j = a.join(b, on=["_i", "_j"], how="left")
    price = np.full((M, K), np.nan)
    src = np.zeros((M, K), dtype=int)
    ii, jj = j["_i"].to_numpy(), j["_j"].to_numpy()
    pt, ps = j["p_tape"].to_numpy(), j["p_ser"].to_numpy()
    use_t = np.isfinite(pt)
    use_s = ~use_t & np.isfinite(ps)
    price[ii[use_t], jj[use_t]] = pt[use_t]
    src[ii[use_t], jj[use_t]] = 1
    price[ii[use_s], jj[use_s]] = ps[use_s]
    src[ii[use_s], jj[use_s]] = 2
    return price, src


# ------------------------------------------------------------------------------------------------ run


@dataclass
class SimResult:
    cells: pl.DataFrame
    diag: dict
    placebo: dict
    markets_used: int
    n_candidates: int
    q99: float
    fpr: dict
    rank_pct: np.ndarray
    ks: tuple[float, ...]
    ss: tuple[float, ...]
    arrays: dict = field(default_factory=dict)


def run_injection(
    ref: Reference,
    cands: pl.DataFrame,
    price: np.ndarray,
    ks: tuple[float, ...],
    ss: tuple[float, ...],
    real_scores: np.ndarray,
    q99: float,
    seed: int = SEED,
    n_boot: int = N_BOOT,
) -> dict:
    """Score all injections and summarise per (s, k) cell. ``real_scores``: scores of the real valid-cutoff winners."""
    rng = np.random.default_rng(seed)
    M, K, S = cands.height, len(ks), len(ss)
    age = rng.uniform(0.0, AGE_MAX_DAYS, size=(M, K, S))
    res = score_injections(
        ref, cands["event_id"].to_list(), cands["condition_id"].to_list(), price,
        np.array(ks), np.array(ss), age,
    )  # fmt: skip
    score = res["score"]
    usd = np.broadcast_to(np.array(ss)[None, None, :], score.shape)
    avail = np.isfinite(score)
    det = np.where(avail, in_tier(np.nan_to_num(score), usd, q99).astype(float), np.nan)
    rs = np.sort(real_scores)
    rank = np.where(avail, np.searchsorted(rs, np.nan_to_num(score), side="right") / len(rs), np.nan)
    rate, lo, hi = bootstrap_rate(det, n_boot, seed)
    return {"score": score, "comps": res["comps"], "det": det, "rank": rank, "rate": rate, "lo": lo,
            "hi": hi, "age": age, "avail": avail}  # fmt: skip


def cells_table(out: dict, ks: tuple[float, ...], ss: tuple[float, ...], plan_ks: tuple[float, ...]) -> pl.DataFrame:
    rows = []
    for ik, k in enumerate(ks):
        for js, s in enumerate(ss):
            av = out["avail"][:, ik, js]
            sc = out["score"][:, ik, js][av]
            rk = out["rank"][:, ik, js][av]
            rate = float(out["rate"][ik, js])
            rows.append({
                "s_usd": s, "k_hours": k, "grid": "plan" if k in plan_ks else "extended",
                "n_markets": int(av.sum()), "detection_rate": rate,
                "ci_lo": float(out["lo"][ik, js]), "ci_hi": float(out["hi"][ik, js]),
                "median_rank_pct": float(np.median(rk) * 100) if av.any() else float("nan"),
                "mean_rank_pct": float(np.mean(rk) * 100) if av.any() else float("nan"),
                "median_score": float(np.median(sc)) if av.any() else float("nan"),
                "meets_criterion": bool(rate >= CRITERION),
            })  # fmt: skip
    return pl.DataFrame(rows)


# ------------------------------------------------------------------------------------------- placebo


def flip_labels(positions: pl.DataFrame) -> pl.DataFrame:
    """Treat losers as winners: ``won`` is inverted (null stays null). Every other field (entry price, size,
    timing, wallet) is untouched, so the losers are scored with the unchanged real reference distributions."""
    return positions.with_columns(
        pl.when(pl.col("won").is_null()).then(None).otherwise(~pl.col("won")).alias("won")
    )


def tier_share(
    scored: pl.DataFrame, excluded: pl.Series | None, q: float, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """Share of valid-cutoff (non-excluded) scored positions in the tier (score >= q and USD >= 100), with a
    market-resampled 95% interval, and the share ignoring the USD rule."""
    ok = scored if excluded is None else scored.filter(~pl.col("wallet").is_in(excluded.implode()))
    ok = ok.filter(~pl.col("no_valid_cutoff"))
    key = "event_id" if "event_id" in ok.columns else "condition_id"
    tier = (pl.col("score") >= q) & (pl.col("usd") >= MIN_USD)
    g = ok.group_by(key).agg(
        tier.sum().alias("t"), (pl.col("score") >= q).sum().alias("t_nousd"), pl.len().alias("n")
    )
    t, tn, n = (g[c].to_numpy().astype(float) for c in ("t", "t_nousd", "n"))
    rng = np.random.default_rng(seed)
    sh = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(n), len(n)), minlength=len(n)).astype(float)
        sh[b] = (w @ t) / max(w @ n, 1.0)
    lo, hi = np.percentile(sh, [2.5, 97.5])
    return {"n": int(n.sum()), "n_tier": int(t.sum()), "share": float(t.sum() / n.sum()),
            "ci_lo": float(lo), "ci_hi": float(hi), "share_nousd": float(tn.sum() / n.sum())}  # fmt: skip


# --------------------------------------------------------------------------------------------- parity


def injected_positions(
    cands: pl.DataFrame, price: float, k: float, s: float, age_days: float, idx: int
) -> pl.DataFrame:
    """One synthetic position row in the schema of ``positions.parquet`` (fake wallet id)."""
    c = cands.row(idx, named=True)
    ft = c["cutoff_ts"] - timedelta(hours=k)
    return pl.DataFrame({
        "wallet": f"synthetic-{idx + 1:04d}", "event_id": c["event_id"], "condition_id": c["condition_id"],
        "token_id": c["token_id"], "outcome_label": c["outcome"].title(), "entry_price": price,
        "usd": s, "size": s / price, "won": True, "first_ts": ft, "cutoff_ts": c["cutoff_ts"],
        "cutoff_method": "independent_news", "info_type": c["info_type"], "hours_before_cutoff": k,
        "n_tokens": 1, "usd_all": s, "valid_cutoff": True, "legacy_cutoff_ts": c["cutoff_ts"],
        "legacy_cutoff_method": "independent_news", "hours_before_legacy_cutoff": k, "calib_eligible": True,
    }).with_columns(pl.col("first_ts").cast(pl.Datetime("us")), pl.col("cutoff_ts").cast(pl.Datetime("us")),
                    pl.col("legacy_cutoff_ts").cast(pl.Datetime("us")))  # fmt: skip


def parity_check(
    positions: pl.DataFrame, wallets: pl.DataFrame, qmarkets, ms, we,
    cands: pl.DataFrame, price: np.ndarray, out: dict, ks, ss, picks: list[tuple[int, int, int]],
) -> dict:
    """Score a few injections with the REAL ``score_oneshot`` (real positions + injected rows appended) and compare
    with the fast independent path. ``picks`` = (market index, k index, s index)."""
    rows, wal, fast = [], [], []
    for n, (i, jk, js) in enumerate(picks):
        age = float(out["age"][i, jk, js])
        r = injected_positions(cands, float(price[i, jk]), ks[jk], ss[js], age, i).with_columns(
            pl.lit(f"synthetic-{n + 1:04d}").alias("wallet")
        )
        rows.append(r)
        ft = r["first_ts"][0] - timedelta(days=age)
        wal.append({"wallet": f"synthetic-{n + 1:04d}", "first_seen": ft, "n_markets": 1})
        fast.append(float(out["score"][i, jk, js]))
    pos2 = pl.concat([positions, *rows], how="diagonal_relaxed")
    w2 = pl.concat(
        [wallets.select("wallet", "first_seen", pl.col("n_markets").cast(pl.Int64)),
         pl.DataFrame(wal).with_columns(pl.col("first_seen").cast(pl.Datetime("us")))],
        how="diagonal_relaxed",
    )  # fmt: skip
    sc = oneshot.score_oneshot(pos2, qmarkets, w2, ms, we)
    slow = [
        float(sc.filter(pl.col("wallet") == f"synthetic-{n + 1:04d}")["score"][0]) for n in range(len(picks))
    ]
    d = np.abs(np.array(slow) - np.array(fast))
    return {"n": len(picks), "fast": fast, "full": slow, "max_abs_diff": float(d.max())}


# ---------------------------------------------------------------------------------------------- driver


def run(
    n_markets: int = 0,
    seed: int = SEED,
    n_boot: int = N_BOOT,
    snap: Path = SNAP,
    private: Path = PRIVATE,
    results: Path = RESULTS,
    parity: bool = True,
    write: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Full 4B + 4C run on the real data; writes the aggregate outputs when ``write``."""
    from pmdetect.wallets import pipeline

    t0 = time.time()
    positions = pl.read_parquet(private / "positions.parquet")
    wallets = pipeline.load_wallets(snap, private)
    mk, ni, _jumps, ms, we, qmarkets = pipeline.load_cutoff_inputs(snap, results)
    flags = pl.read_parquet(private / "wallet_flags.parquet", columns=["wallet", "excluded"])
    excluded = flags.filter(pl.col("excluded"))["wallet"]
    log(f"[{time.time() - t0:.0f}s] loaded: positions={positions.height} wallets={wallets.height}")

    scored = oneshot.score_oneshot(positions, qmarkets, wallets, ms, we)
    ok, q99 = real_tier(scored, excluded)
    saved = private / "oneshot_scores.parquet"
    reproduced = None
    if saved.exists():
        s0 = pl.read_parquet(saved, columns=["wallet", "event_id", "score"])
        j = scored.select("wallet", "event_id", "score").join(s0, on=["wallet", "event_id"], suffix="_saved")
        reproduced = {"rows": j.height, "max_abs_diff": float((j["score"] - j["score_saved"]).abs().max())}
    real_scores = ok["score"].to_numpy()
    n_tier_real = int(((ok["score"] >= q99) & (ok["usd"] >= MIN_USD)).sum())
    log(f"[{time.time() - t0:.0f}s] real winners scored: valid={ok.height} q99={q99:.4f} in tier={n_tier_real}")

    frame = prepare_reference_frame(positions, wallets)
    ctx = oneshot._context(qmarkets, ms, we)
    elig = eligible_markets(mk, ni, pl.read_parquet(snap / "tokens.parquet"))
    cands = draw_markets(elig, n_markets, seed)
    log(f"[{time.time() - t0:.0f}s] eligible independent-cutoff markets: {elig.height}; drawn: {cands.height}")

    tok = cands["token_id"].unique()
    tape = (
        pl.scan_parquet(snap / "trades_taker.parquet")
        .filter(pl.col("token_id").is_in(tok.implode()) & (pl.col("size") > 0))
        .select("token_id", "ts", (pl.col("usd") / pl.col("size")).alias("price"))
        .collect()
    )
    series = pl.read_parquet(snap / "prices.parquet").filter(pl.col("token_id").is_in(tok.implode()))
    ks_all = K_GRID + K_EXTENDED
    price, src = entry_prices(cands, ks_all, tape, series)
    has_any = np.isfinite(price[:, : len(K_GRID)]).any(axis=1)
    keep = np.flatnonzero(has_any)
    cands, price, src = cands[keep.tolist()], price[keep], src[keep]
    log(f"[{time.time() - t0:.0f}s] markets with an entry price at >= 1 plan lead: {cands.height}")

    ref = Reference.build(frame, ctx, keys=cands["event_id"].unique().to_list())
    out = run_injection(ref, cands, price, ks_all, S_GRID, real_scores, q99, seed, n_boot)
    cells = cells_table(out, ks_all, S_GRID, K_GRID)
    log(f"[{time.time() - t0:.0f}s] injections scored")

    pr = None
    if parity:
        rng = np.random.default_rng(seed + 1)
        picks = []
        for _ in range(6):
            i = int(rng.integers(0, cands.height))
            jk = int(rng.integers(0, len(K_GRID)))
            js = int(rng.integers(0, len(S_GRID)))
            if np.isfinite(price[i, jk]):
                picks.append((i, jk, js))
        pr = parity_check(positions, wallets, qmarkets, ms, we, cands, price, out, ks_all, S_GRID, picks)
        log(f"[{time.time() - t0:.0f}s] parity vs full score_oneshot: max |diff| = {pr['max_abs_diff']:.2e}")

    # 4C placebo: losers with the outcome flipped, real winners for comparison
    flipped = oneshot.score_oneshot(flip_labels(positions), qmarkets, wallets, ms, we)
    pl_flip = tier_share(flipped, excluded, q99, n_boot, seed)
    pl_win = tier_share(scored, excluded, q99, n_boot, seed)
    log(f"[{time.time() - t0:.0f}s] placebo: flipped {pl_flip['share']:.4f}, winners {pl_win['share']:.4f}")
    placebo = {"flipped": pl_flip, "winners": pl_win, "pass": pl_flip["share"] <= PLACEBO_MAX}
    placebo["flipped_by_cutoff"] = {}
    comp_means = {
        "tier_real": {c: float(ok.filter((pl.col("score") >= q99) & (pl.col("usd") >= MIN_USD))[c].mean())
                      for c in COMP_NAMES},
        "all_real": {c: float(ok[c].mean()) for c in COMP_NAMES},
        "tier_flipped": {},
    }  # fmt: skip
    fl_ok = flipped.filter(~pl.col("wallet").is_in(excluded.implode()) & ~pl.col("no_valid_cutoff"))
    fl_tier = fl_ok.filter((pl.col("score") >= q99) & (pl.col("usd") >= MIN_USD))
    comp_means["tier_flipped"] = {c: float(fl_tier[c].mean()) for c in COMP_NAMES} if fl_tier.height else {}
    comp_means["all_flipped"] = {c: float(fl_ok[c].mean()) for c in COMP_NAMES}

    diag = {
        "reproduced": reproduced, "parity": pr, "n_real_valid": ok.height, "n_real_tier": n_tier_real,
        "elig": elig.height, "price_source": {
            "tape": float((src[:, : len(K_GRID)] == 1).mean()), "series": float((src[:, : len(K_GRID)] == 2).mean()),
            "none": float((src[:, : len(K_GRID)] == 0).mean())},
        "comp_means": comp_means, "ctx_share": float(np.mean([ref.context_for(e, c) != oneshot.NEUTRAL
                                                             for e, c in zip(cands["event_id"], cands["condition_id"], strict=True)])),
        "n_real_scores_in_events": len(ref.groups),
    }  # fmt: skip
    res = SimResult(cells, diag, placebo, cands.height, elig.height, q99,
                    {"n": ok.height, "tier": n_tier_real, "share": n_tier_real / ok.height},
                    out["rank"], ks_all, S_GRID, {"out": out, "price": price, "cands": cands, "real_scores": real_scores})  # fmt: skip
    if write:
        write_outputs(res, results, seed, n_boot)
        log(f"[{time.time() - t0:.0f}s] outputs written")
    return {"result": res}


# ------------------------------------------------------------------------------------------ outputs


def heatmap(cells: pl.DataFrame, path: Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = cells.filter(pl.col("grid") == "plan")
    ss = sorted(c["s_usd"].unique().to_list())
    ks = sorted(c["k_hours"].unique().to_list())
    z = np.full((len(ss), len(ks)), np.nan)
    ann = [[""] * len(ks) for _ in ss]
    for r in c.iter_rows(named=True):
        a, b = ss.index(r["s_usd"]), ks.index(r["k_hours"])
        z[a, b] = r["detection_rate"]
        ann[a][b] = f"{r['detection_rate']:.0%}\n[{r['ci_lo']:.0%}, {r['ci_hi']:.0%}]"
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=140)
    im = ax.imshow(z, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for a in range(len(ss)):
        for b in range(len(ks)):
            v = z[a, b]
            ax.text(b, a, ann[a][b], ha="center", va="center", fontsize=7.5,
                    color="white" if v < 0.6 else "black", fontweight="bold" if v >= CRITERION else "normal")  # fmt: skip
            if v >= CRITERION:
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="white", lw=2))
    ax.set_xticks(range(len(ks)), [f"{k:g}" for k in ks])
    ax.set_yticks(range(len(ss)), [f"${s:,.0f}" for s in ss])
    ax.set_xlabel("hours before the cutoff (independent news time)")
    ax.set_ylabel("stake of the synthetic position")
    ax.set_title(title or "One-shot screen: detection rate of a synthetic fresh-wallet winner\n"
                 "(tier = real 99th percentile and USD >= 100; white box = cell meets >= 50%)", fontsize=9)  # fmt: skip
    fig.colorbar(im, ax=ax, label="detection rate")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _fmt_table(cells: pl.DataFrame, value: str = "rate") -> str:
    c = cells.filter(pl.col("grid") == "plan")
    ks = sorted(c["k_hours"].unique().to_list())
    lines = ["| stake \\ lead (h) | " + " | ".join(f"{k:g}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for s in sorted(c["s_usd"].unique().to_list(), reverse=True):
        row = []
        for k in ks:
            r = c.filter((pl.col("s_usd") == s) & (pl.col("k_hours") == k)).row(0, named=True)
            if value == "rate":
                row.append(f"{r['detection_rate']:.0%} [{r['ci_lo']:.0%}, {r['ci_hi']:.0%}]")
            else:
                row.append(f"{r['median_rank_pct']:.1f}")
        lines.append(f"| ${s:,.0f} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_outputs(res: SimResult, results: Path, seed: int, n_boot: int) -> dict[str, Path]:
    results.mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    csv = results / "validation_oneshot_power.csv"
    res.cells.write_csv(csv)
    png = heatmap(res.cells, results / "figures" / "validation_oneshot_power.png")
    md = results / "validation_oneshot.md"
    md.write_text(render_md(res, seed, n_boot), encoding="utf-8")
    return {"csv": csv, "png": png, "md": md}


def render_md(res: SimResult, seed: int, n_boot: int) -> str:
    c = res.cells
    plan = c.filter(pl.col("grid") == "plan")
    meet = plan.filter(pl.col("meets_criterion")).sort(["s_usd", "k_hours"])
    d = res.diag
    cm = d["comp_means"]
    out = res.arrays["out"]
    comps, avail, det = out["comps"], out["avail"], out["det"]
    names = list(COMP_NAMES)
    ref_cell = (K_GRID.index(24.0), S_GRID.index(20000.0))
    miss = avail[:, ref_cell[0], ref_cell[1]] & (det[:, ref_cell[0], ref_cell[1]] == 0)
    cc = comps[:, ref_cell[0], ref_cell[1], :]
    lim = np.argmin(cc[miss], axis=1) if miss.any() else np.array([], dtype=int)
    lim_share = {n: float((lim == i).mean()) if lim.size else float("nan") for i, n in enumerate(names)}
    comp_ref = {n: float(np.nanmean(cc[avail[:, ref_cell[0], ref_cell[1]], i])) for i, n in enumerate(names)}
    fl, wn = res.placebo["flipped"], res.placebo["winners"]
    cells_txt = (
        ", ".join(f"(${r['s_usd']:,.0f}, {r['k_hours']:g} h)" for r in meet.iter_rows(named=True))
        if meet.height else "none"
    )
    smallest = ""
    if meet.height:
        mn = meet.sort(["s_usd", "k_hours"]).row(0, named=True)
        smallest = f" The smallest stake that meets it is ${mn['s_usd']:,.0f} (at {mn['k_hours']:g} h)."
    verdict = "PASS" if res.placebo["pass"] else "FAIL"
    cmp_rows = "\n".join(
        f"| {n} | {cm['all_real'][n]:.3f} | {cm['tier_real'][n]:.3f} | {comp_ref[n]:.3f} | "
        f"{lim_share[n]:.0%} |" for n in names
    )
    flip_cmp = (
        "\n".join(
            f"| {n} | {cm['all_flipped'][n]:.3f} | {cm['tier_flipped'].get(n, float('nan')):.3f} |" for n in names
        )
    )
    rep = d["reproduced"]
    par = d["parity"]
    ext = c.filter(pl.col("grid") == "extended").sort(["k_hours", "s_usd"])
    ext_rows = "\n".join(
        f"| ${r['s_usd']:,.0f} | {r['k_hours']:g} | {r['detection_rate']:.0%} [{r['ci_lo']:.0%}, {r['ci_hi']:.0%}] |"
        for r in ext.iter_rows(named=True)
    )
    ps = d["price_source"]
    return f"""# One-shot screen validation (PLAN phase 4B and 4C)

Aggregate results only: no wallet addresses; the synthetic wallets are fake ids. Generated by
`pmdetect validate oneshot` (seed {seed}, {n_boot} bootstrap resamples of markets, 95% percentile intervals).
Files: `results/validation_oneshot_power.csv`, `results/figures/validation_oneshot_power.png`.

## What was tested, and what it does not show

A synthetic wallet is injected into real markets and scored by the real one-shot code against the real score
distribution. The injected position is deliberately idealised: a brand-new wallet (age 0 to 3 days, uniform), only
one market, a clean entry `k` hours before an independent news time, and it always wins. This tests whether the
SCREEN can rank such a pattern, **not** whether real anomalous traders look like this, and it says nothing about
how often real informed trading is split across wallets, smaller, earlier, or in markets without an independent
time (those markets cannot be given a valid cutoff at all and are screen-only, outside this test). The result is a
property of the score, not a detection rate in the wild.

## Method

* Markets: the {res.n_candidates} universe markets with an independent news time and a YES/NO outcome; all of them
  were used (a random draw with the fixed seed is available through `--n-markets`); {res.markets_used} have a
  winner-token price at one or more plan leads (a market that did not yet trade at `cutoff - k h` gets no
  injection for that lead, so each cell has its own `n_markets`, {int(plan['n_markets'].min())} to {int(plan['n_markets'].max())}).
  All are `scheduled_reveal` markets, because only those carry an independent time.
* Grid: stake s in {{500, 2,000, 5,000, 20,000, 50,000}} USD, lead k in {{2, 6, 12, 24, 48, 72}} h before the cutoff
  (supplementary leads 120, 168, 240 h below).
* Entry price: the winner token's price at the entry time (last taker fill within 6 h, else the last price
  observation within 24 h; shares at the plan leads: tape {ps['tape']:.0%}, series {ps['series']:.0%}, none {ps['none']:.0%}).
  No market impact or slippage is modelled, so large stakes in thin markets are optimistic.
* Winning side: the market's actual winner. In a market where NO won, the winner is the NO token, usually a high
  price and therefore low improbability: those injections are hard to detect by construction, and they are kept
  because the screen cannot know the side in advance.
* Scoring: independent, one injected row at a time against the real reference sets (module docstring of
  `oneshot_sim.py`). Parity: {('full score_oneshot on the real positions plus ' + str(par['n']) + ' injected rows agrees with the fast path to a max absolute score difference of ' + format(par['max_abs_diff'], '.1e')) if par else 'not run'}.
  The re-computed real scores {('match the saved oneshot_scores.parquet on ' + str(rep['rows']) + ' rows (max abs diff ' + format(rep['max_abs_diff'], '.1e') + ')') if rep else 'were not compared with a saved file'}.
* Tier: the existing `one_shot_anomalous` rule: score >= the 99th percentile of the real valid-cutoff winning-position
  scores (wallets excluded by the non-directional filters removed, as in the watchlist; threshold {res.q99:.4f}) and
  USD >= 100. The 99th percentile and the real score distribution come from the real winners only.
* Detection rate: share of injections in the tier. Intervals: markets resampled with replacement, {n_boot} times.

## Real positions: false-positive rate

{res.fpr['n']} real winning valid-cutoff positions; {res.fpr['tier']} are in the tier ({res.fpr['share']:.2%}; the 99th-percentile rule alone gives 1% of positions, the USD >= 100 rule removes most of
those, so the effective false-positive rate is this lower number).

## 4B power table (detection rate [95% interval])

{_fmt_table(c)}

Median rank percentile of the injected position among the {res.fpr['n']} real winning valid-cutoff positions
(100 = above all real positions):

{_fmt_table(c, 'rank')}

**Cells meeting the criterion (detection rate >= {CRITERION:.0%}): {len(meet)} of {plan.height}: {cells_txt}.**{smallest}
Cells not listed do not meet it. Words for the README: the screen "works" (ranks the idealised pattern into the
top 1%) only in the listed cells; elsewhere the pattern is not separable from real winners by this score.

### Supplementary leads beyond 72 h (proximity decays with e-folding 72 h; not part of the criterion)

| stake | lead (h) | detection rate [95% interval] |
|---|---|---|
{ext_rows}

## What the screen is made of, and what dominates

The score is the geometric mean of five components in [0, 1]: improbability (low entry price), size (USD percentile
within the market and globally), proximity (1 up to 72 h before the cutoff, then exp(-(h-72)/72)), freshness (age at
entry, number of markets, only-market) and market context (within-event / flow / late-window percentile of the
market, neutral 0.5 when missing; present for {d['ctx_share']:.0%} of the injected markets).

Mean components (real winners; real tier; injected cell $20,000 at 24 h; limiting component = lowest component
among the injections of that cell that missed the tier):

| component | real winners (all) | real tier | injected ($20k, 24 h) | share of misses it limits |
|---|---|---|---|---|
{cmp_rows}

Reading: a fresh, large, late winner gets a near-maximal proximity, size and freshness by design; whether it
reaches the top 1% is decided mostly by the components that depend on the market, improbability (the winner's entry
price) and context. Proximity is flat for every plan lead (all leads <= 72 h score 1), so within the plan grid lead
only matters through the price the position gets and how much price movement exists; the supplementary table shows
the decay beyond 72 h. These are properties of the score as defined, not tuned on any labelled case.

## 4C placebo: label-flipped losers (criterion: no more than {PLACEBO_MAX:.0%} in the tier)

Real losing positions are scored as if they had won (own entry price, size, timing, wallet; same real reference
distributions), then compared to the real-winner tier threshold above.

| group | positions (valid cutoff) | in the tier | share [95% interval, markets resampled] | share ignoring USD rule |
|---|---|---|---|---|
| flipped losers | {fl['n']} | {fl['n_tier']} | {fl['share']:.2%} [{fl['ci_lo']:.2%}, {fl['ci_hi']:.2%}] | {fl['share_nousd']:.2%} |
| real winners | {wn['n']} | {wn['n_tier']} | {wn['share']:.2%} [{wn['ci_lo']:.2%}, {wn['ci_hi']:.2%}] | {wn['share_nousd']:.2%} |

**Verdict (flipped-label share <= 2%): {verdict}** ({fl['share']:.2%} observed).

Mean components of flipped losers (all, and those in the tier):

| component | flipped (all) | flipped in tier |
|---|---|---|
{flip_cmp}

The 2% criterion is applied to the tier as defined (score >= threshold AND USD >= 100). The score uses only the
position's price, size, timing, freshness and the market's context, never the outcome, so the label flip changes
only WHICH positions are scored. Losing positions tend to have low entry prices (high improbability), so they
are not scored lower than winners: ignoring the USD rule, flipped losers sit above the real-winner 99th percentile
more often than real winners do (see the last column). That is a property of the score, not a failure of the
criterion: it passes as written, but it means entering the tier is NOT evidence of winning-by-luck-or-skill on its
own; the evidence comes from combining the tier with the (separate) outcome and with the repeat tests. Both numbers
are reported, not only the pass.

## Caveats

* The injected pattern is a best case (always wins, fresh, clean timing). Real anomalous trading may be spread over
  wallets, over time, or in markets with no independent time, where this test cannot say anything.
* Percentile components make the score relative to this particular population of 14k markets; a different universe
  moves every threshold.
* No price impact: a $50,000 buy in a thin market would move the price against the buyer.
* First_seen is known only for part of the real wallets (store backfill plus the top-up); missing ages are left out of
  the freshness percentile reference, which the injection reproduces.
* Cells are not independent (the same markets are used in every cell) and intervals resample markets only.
"""
