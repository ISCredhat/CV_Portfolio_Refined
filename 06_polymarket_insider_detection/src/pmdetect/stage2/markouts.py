"""Stage 2, task B: taker-trade markouts and wallet skill v2 (frozen definition: docs/STAGE2_PLAN.md s.9).

Definitions (all fixed in advance, nothing tuned)
-------------------------------------------------
* Every taker fill is expressed in YES terms: ``y`` = YES-equivalent price (price of a Yes fill, 1 - price of a
  No fill) and ``q`` = +1 if the fill pushes the YES price up (buy Yes or sell No), else -1. The side-signed
  markout in the TRADED token's price at horizon h is ``q * (Y(t+h) - y)``: for a No fill this equals
  (P_no(t+h) - P_no(t)) * (+1 if buy else -1) exactly, because P_no = 1 - Y.
* ``Y(t+h)`` = last observation at or before t+h of the market's YES-equivalent series, built from (a) all taker
  fills of the market (a fill on the No token at 1 - price) merged on time with (b) the stored winner-token
  price history (``prices.parquet``, converted to YES terms), observations later than ``closed_ts`` removed.
  Ties in time: the fill wins over the history point.
* Horizons 5 min, 1 h, 6 h, 24 h and resolution (final payoff of the traded token, 0 or 1). A horizon is DROPPED
  (null, never padded) for a trade when t+h is after min(last observation, closed_ts), or when the last
  observation is stale: older than 6 h for h <= 1 h, 24 h for h = 6 h, 48 h for h = 24 h. Resolution has no
  staleness rule (the payoff is known). The 6 h horizon and its 24 h limit are additions for the sign report.
* "Per dollar": the markout in dollars (size * markout per share) divided by the notional (usd). A wallet's
  skill = sum(q * size * dY) / sum(usd) over its taker trades in a period (size-weighted mean markout per dollar).
* Null: event-level sign randomisation: within each market the YES-direction signs ``q`` are shuffled among
  that market's trades (cluster = market), ``n_draws`` times; z and p per wallet. The exact null mean and
  variance of a wallet's numerator under this shuffle are also computed analytically (used as the sampling
  variance in the empirical-Bayes step; the Monte-Carlo draws give the p-values).
* Empirical Bayes (normal-normal) over wallets with at least 10 trades: skill_w ~ N(theta_w, v_w) with v_w the
  null variance, theta_w ~ N(mu, tau^2), tau^2 by the method of moments.
* Periods by the market's ``closed_ts`` (see common.py); the sealed period is dropped at load and asserted away.
* Spread proxy: for each trade, the gap |y - y_prev| to the immediately preceding fill of the same market when
  that fill has the opposite YES direction and is within 10 minutes; otherwise the market's median of such gaps;
  otherwise the global median. Half of it is the per-share spread cost deducted in the spread-adjusted version.
  (A taker's own entry price already includes its half spread, so deducting a further half spread is
  conservative; it is the pre-registered adjustment.)
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.stage2.common import FIT_END, SEED, assert_not_sealed, drop_sealed
from pmdetect.store.db import PROJECT_ROOT
from pmdetect.wallets.fdr import bh_adjusted

SNAP = PROJECT_ROOT / "data" / "snapshots"
PRIVATE = PROJECT_ROOT / "data" / "private"
RESULTS = PROJECT_ROOT / "results"
PARTS_DIR = PRIVATE / "markout_trades"
HORIZONS = {"5m": 300, "1h": 3600, "6h": 6 * 3600, "24h": 24 * 3600}
STALE_S = {"5m": 6 * 3600, "1h": 6 * 3600, "6h": 24 * 3600, "24h": 48 * 3600}
HORIZON_COLS = ["5m", "1h", "6h", "24h", "res"]
HEADLINE = "24h"
EMBARGO = timedelta(days=7)
SPREAD_WINDOW_S = 600
MIN_EB = 10
MIN_PERSIST = 20
N_PARTS = 16
TAPE_COLS = ["condition_id", "wallet", "side", "outcome_label", "size", "price", "usd", "ts"]


warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")


def _noop(msg: str) -> None:
    pass


# --------------------------------------------------------------------------------------------- load


def load_markets(snap: Path = SNAP) -> pl.DataFrame:
    """Non-sealed markets with a YES/NO outcome; adds period, dense market code and event code."""
    m = pl.read_parquet(
        snap / "markets.parquet",
        columns=["condition_id", "event_id", "tags", "info_type", "volume_usd", "start_ts", "end_ts",
                 "closed_ts", "outcome"],
    )  # fmt: skip
    m = drop_sealed(m)
    assert_not_sealed(m)
    m = m.with_columns(pl.col("outcome").str.strip_chars().str.to_uppercase()).filter(
        pl.col("outcome").is_in(["YES", "NO"])
    )
    m = m.with_columns(
        pl.when(pl.col("closed_ts") < FIT_END).then(pl.lit("fit")).otherwise(pl.lit("tune")).alias("period"),
        pl.coalesce(pl.col("event_id"), pl.col("condition_id")).alias("_ev"),
    )
    m = m.sort("condition_id").with_row_index("mcode")
    ev = m.select("_ev").unique().sort("_ev").with_row_index("ecode")
    return m.join(ev, on="_ev", how="left").drop("_ev").sort("mcode")


def wallet_table(snap: Path = SNAP, markets: pl.DataFrame | None = None) -> pl.DataFrame:
    """Dense integer codes for the wallets of the non-sealed tape (kept in data/private)."""
    lf = pl.scan_parquet(snap / "trades_taker.parquet").select("condition_id", "wallet")
    if markets is not None:
        lf = lf.filter(pl.col("condition_id").is_in(markets["condition_id"].implode()))
    w = lf.select("wallet").unique().collect().sort("wallet")
    return w.with_row_index("wcode").with_columns(pl.col("wcode").cast(pl.Int32))


# --------------------------------------------------------------------------------- trade-level markouts


def prep_trades(tr: pl.DataFrame, mk: pl.DataFrame) -> pl.DataFrame:
    """YES-direction sign q and YES-equivalent price y; drops fills after the market's closed_ts."""
    t = tr.with_columns(pl.col("outcome_label").str.strip_chars().str.to_uppercase().alias("_lab"))
    t = t.filter(pl.col("_lab").is_in(["YES", "NO"]))
    t = t.join(mk.select("condition_id", "closed_ts"), on="condition_id", how="inner")
    t = t.filter(pl.col("ts") <= pl.col("closed_ts"))
    is_yes = pl.col("_lab") == "YES"
    return t.with_columns(
        pl.when((pl.col("side") == "BUY") == is_yes).then(1).otherwise(-1).cast(pl.Int8).alias("q"),
        pl.when(is_yes).then(pl.col("price")).otherwise(1.0 - pl.col("price")).alias("y"),
    ).drop("_lab")


def build_series(t: pl.DataFrame, hist: pl.DataFrame | None, mk: pl.DataFrame) -> pl.DataFrame:
    """YES-equivalent observation series (condition_id, ots, y_obs): taker fills + winner-token history.

    ``hist`` has condition_id, outcome_label, ts, price. History points later than closed_ts are removed;
    on equal timestamps the fill sorts after the history point so it wins an as-of lookup.
    """
    parts = [t.select("condition_id", pl.col("ts").alias("ots"), pl.col("y").alias("y_obs"), pl.lit(1).alias("_src"))]
    if hist is not None and hist.height:
        lab = pl.col("outcome_label").str.strip_chars().str.to_uppercase()
        h = hist.with_columns(lab.alias("_lab")).filter(pl.col("_lab").is_in(["YES", "NO"]))
        h = h.join(mk.select("condition_id", "closed_ts"), on="condition_id", how="inner")
        h = h.filter(pl.col("ts") <= pl.col("closed_ts"))
        parts.append(
            h.select(
                "condition_id",
                pl.col("ts").alias("ots"),
                pl.when(pl.col("_lab") == "YES").then(pl.col("price")).otherwise(1.0 - pl.col("price")).alias("y_obs"),
                pl.lit(0).alias("_src"),
            )
        )
    return pl.concat(parts).sort(["ots", "_src"]).drop("_src")


def spread_proxy(t: pl.DataFrame) -> pl.DataFrame:
    """Per-trade half-spread proxy (price units); see the module docstring. Adds ``hs``."""
    s = t.sort(["condition_id", "ts"], maintain_order=True)
    prev_q = pl.col("q").shift(1).over("condition_id")
    prev_ts = pl.col("ts").shift(1).over("condition_id")
    prev_y = pl.col("y").shift(1).over("condition_id")
    ok = (prev_q == -pl.col("q")) & ((pl.col("ts") - prev_ts).dt.total_seconds() <= SPREAD_WINDOW_S)
    s = s.with_columns(pl.when(ok).then((pl.col("y") - prev_y).abs()).otherwise(None).alias("_gap"))
    med = s.group_by("condition_id").agg(pl.col("_gap").median().alias("_mmed"))
    gmed = s["_gap"].drop_nulls().median()
    gmed = 0.0 if gmed is None else float(gmed)
    s = s.join(med, on="condition_id", how="left")
    return s.with_columns(
        (pl.coalesce(pl.col("_gap"), pl.col("_mmed"), pl.lit(gmed)) / 2.0).alias("hs")
    ).drop("_gap", "_mmed")


def markouts_for_trades(
    t: pl.DataFrame, series: pl.DataFrame, mk: pl.DataFrame
) -> pl.DataFrame:
    """Add YES-unit price changes ``d_<h>`` (Y(t+h) - y, null when dropped) and ``d_res`` to prepped trades.

    ``t`` needs condition_id, ts, y, closed_ts; ``series`` is :func:`build_series`; ``mk`` has outcome.
    The side-signed markout is ``q * d``.
    """
    t = t.with_row_index("_rid")
    last = series.group_by("condition_id").agg(pl.col("ots").max().alias("_last"))
    t = t.join(last, on="condition_id", how="left").with_columns(
        pl.min_horizontal("closed_ts", "_last").alias("_lim")
    )
    ser = series.with_columns(pl.col("ots").alias("obs_ts"))
    for name, sec in HORIZONS.items():
        left = t.select(
            "_rid", "condition_id", "y", "_lim", (pl.col("ts") + pl.duration(seconds=sec)).alias("_tt")
        ).sort("_tt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            j = left.join_asof(ser, left_on="_tt", right_on="ots", by="condition_id", strategy="backward")
        ok = (
            pl.col("obs_ts").is_not_null()
            & (pl.col("_tt") <= pl.col("_lim"))
            & ((pl.col("_tt") - pl.col("obs_ts")).dt.total_seconds() <= STALE_S[name])
        )
        j = j.select("_rid", pl.when(ok).then(pl.col("y_obs") - pl.col("y")).otherwise(None).alias(f"d_{name}"))
        t = t.join(j, on="_rid", how="left")
    yfin = mk.select("condition_id", (pl.col("outcome") == "YES").cast(pl.Float64).alias("_yfin"))
    t = t.join(yfin, on="condition_id", how="left").with_columns((pl.col("_yfin") - pl.col("y")).alias("d_res"))
    return t.drop("_rid", "_last", "_lim", "_yfin")


def _build_part(
    k: int, mk: pl.DataFrame, wtab: pl.DataFrame, snap: Path, hist_all: pl.DataFrame, out: Path
) -> int:
    cids = mk["condition_id"]
    tr = (
        pl.scan_parquet(snap / "trades_taker.parquet")
        .select(TAPE_COLS)
        .filter(pl.col("condition_id").is_in(cids.implode()))
        .collect()
    )
    t = prep_trades(tr, mk)
    del tr
    ser = build_series(t, hist_all.filter(pl.col("condition_id").is_in(cids.implode())), mk)
    t = spread_proxy(t)
    t = markouts_for_trades(t, ser, mk)
    t = t.join(wtab, on="wallet", how="left").join(
        mk.select("condition_id", "mcode", "ecode", "period"), on="condition_id", how="left"
    )
    keep = ["wcode", "mcode", "ecode", "period", "ts", "closed_ts", "q", "size", "usd", "y", "hs"] + [
        f"d_{h}" for h in HORIZON_COLS
    ]
    t = t.select(
        [pl.col(c).cast(pl.Float32) if c.startswith("d_") or c in ("y", "hs") else pl.col(c) for c in keep]
    )
    t.write_parquet(out / f"part_{k:02d}.parquet")
    return t.height


def build_trade_markouts(
    markets: pl.DataFrame, snap: Path = SNAP, out: Path = PARTS_DIR, n_parts: int = N_PARTS, log: Callable = _noop
) -> int:
    """Compute markouts for every non-sealed taker fill in ``n_parts`` market partitions (bounded memory)."""
    assert_not_sealed(markets)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("part_*.parquet"):
        f.unlink()
    wtab = wallet_table(snap, markets)
    wtab.write_parquet(PRIVATE / "markout_wallets.parquet")
    tokens = pl.read_parquet(snap / "tokens.parquet")
    hist = pl.read_parquet(snap / "prices.parquet").join(tokens, on="token_id", how="inner")
    hist = hist.filter(pl.col("condition_id").is_in(markets["condition_id"].implode())).select(
        "condition_id", "outcome_label", "ts", "price"
    )
    total = 0
    t0 = time.time()
    for k in range(n_parts):
        mk = markets.filter((pl.col("mcode") % n_parts) == k)
        n = _build_part(k, mk, wtab, snap, hist, out)
        total += n
        log(f"markout part {k + 1}/{n_parts}: {mk.height} markets, {n} trades ({time.time() - t0:.0f}s)")
    return total


# ------------------------------------------------------------------------------------ numeric helpers


def null_moments(w: np.ndarray, m: np.ndarray, q: np.ndarray, a: np.ndarray, n_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and variance, per wallet, of ``sum(q_pi * a)`` when the signs q are shuffled within markets.

    Per (wallet, market) group with n trades in the market, market sign mean mu and variance 1 - mu^2:
    mean = mu * sum(a); var = (1 - mu^2) * (n * sum(a^2) - sum(a)^2) / (n - 1) (0 if n == 1).
    """
    m = m.astype(np.int64)
    n_m = np.bincount(m)
    mu = np.bincount(m, weights=q.astype(np.float64)) / np.maximum(n_m, 1)
    key = w.astype(np.int64) * len(n_m) + m
    order = np.argsort(key, kind="stable")
    ks = key[order]
    a_s = a[order].astype(np.float64)
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    A = np.add.reduceat(a_s, start)
    A2 = np.add.reduceat(a_s * a_s, start)
    gk = ks[start]
    gm = gk % len(n_m)
    gw = gk // len(n_m)
    n = n_m[gm].astype(np.float64)
    s2 = 1.0 - mu[gm] ** 2
    var_g = np.where(n > 1, s2 * (n * A2 - A * A) / np.maximum(n - 1.0, 1.0), 0.0)
    var_g = np.maximum(var_g, 0.0)
    mean = np.bincount(gw, weights=mu[gm] * A, minlength=n_w)
    var = np.bincount(gw, weights=var_g, minlength=n_w)
    return mean, var


def sign_randomization(
    w: np.ndarray,
    m: np.ndarray,
    q: np.ndarray,
    a: np.ndarray,
    n_w: int,
    n_draws: int = 2000,
    seed: int = SEED,
    threads: int = 8,
) -> dict[str, np.ndarray]:
    """Monte-Carlo event-level sign randomisation of the per-wallet numerator ``sum(q * a)``.

    Within each market the signs are shuffled among its trades (``n_draws`` draws). Returns the observed
    numerator, the empirical null mean/sd, the upper-tail p ``(1 + #{draw >= obs}) / (n_draws + 1)`` and the
    two-sided p about the null mean.
    """
    order = np.argsort(m, kind="stable")
    w, m, q, a = w[order], m[order], q[order].astype(np.float64), a[order].astype(np.float64)
    mf = m.astype(np.float64)
    n = len(w)
    obs = np.bincount(w, weights=q * a, minlength=n_w)
    nm_mean, _ = null_moments(w, m, q, a, n_w)
    dev_obs = np.abs(obs - nm_mean)
    tol = 1e-9 * max(1.0, float(np.abs(obs).max()) if n else 1.0)
    seeds = np.random.SeedSequence(seed).spawn(max(1, threads))
    chunks = np.array_split(np.arange(n_draws), max(1, threads))

    def work(i: int):
        rng = np.random.default_rng(seeds[i])
        ge = np.zeros(n_w, np.int64)
        ge2 = np.zeros(n_w, np.int64)
        s = np.zeros(n_w)
        s2 = np.zeros(n_w)
        for _ in range(len(chunks[i])):
            perm = np.argsort(mf + rng.random(n))
            st = np.bincount(w, weights=q[perm] * a, minlength=n_w)
            ge += st >= obs - tol
            ge2 += np.abs(st - nm_mean) >= dev_obs - tol
            s += st
            s2 += st * st
        return ge, ge2, s, s2

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        res = list(ex.map(work, range(len(chunks))))
    ge = sum(r[0] for r in res)
    ge2 = sum(r[1] for r in res)
    s = sum(r[2] for r in res)
    s2 = sum(r[3] for r in res)
    mean = s / n_draws
    sd = np.sqrt(np.maximum(s2 / n_draws - mean**2, 0.0))
    return {
        "obs": obs,
        "null_mean": mean,
        "null_sd": sd,
        "p_upper": (1.0 + ge) / (n_draws + 1.0),
        "p_two": (1.0 + ge2) / (n_draws + 1.0),
    }


def eb_shrink(skill: np.ndarray, var: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray | float]:
    """Normal-normal empirical-Bayes shrinkage over the wallets in ``mask`` (method-of-moments tau^2).

    shrunk = mu + tau^2 / (tau^2 + v) * (skill - mu); wallets outside the mask get NaN.
    """
    s, v = skill[mask], var[mask]
    out = np.full(skill.shape, np.nan)
    if s.size < 2:
        return {"shrunk": out, "mu": float("nan"), "tau2": float("nan")}
    tau2 = max(0.0, float(np.var(s, ddof=1) - np.mean(v)))
    wgt = 1.0 / np.maximum(v + tau2, 1e-12)
    mu = float(np.sum(wgt * s) / np.sum(wgt))
    denom = tau2 + v
    gain = np.where(denom > 0, tau2 / np.where(denom > 0, denom, 1.0), 0.0)
    out[mask] = mu + gain * (s - mu)
    return {"shrunk": out, "mu": mu, "tau2": tau2}


def bootstrap_counts(n_events: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Event-resampling multiplicities (n_boot x n_events), multinomial with equal probabilities."""
    return rng.multinomial(n_events, np.full(n_events, 1.0 / n_events), size=n_boot).astype(np.float64)


def pooled_ratio_ci(
    num: np.ndarray, den: np.ndarray, ecode: np.ndarray, n_events: int, counts: np.ndarray
) -> tuple[float, float, float]:
    """Pooled ratio sum(num)/sum(den) with a 95% event-cluster bootstrap interval."""
    ne = np.bincount(ecode, weights=num, minlength=n_events)
    de = np.bincount(ecode, weights=den, minlength=n_events)
    point = float(ne.sum() / de.sum()) if de.sum() > 0 else float("nan")
    bn, bd = counts @ ne, counts @ de
    ok = bd > 0
    if not ok.any():
        return point, float("nan"), float("nan")
    b = bn[ok] / bd[ok]
    return point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------------- persistence test


def _decile_sets(score: np.ndarray, elig: np.ndarray) -> list[np.ndarray]:
    """Wallet masks for the 10 deciles (0 = lowest) of ``score`` among eligible wallets."""
    idx = np.flatnonzero(elig)
    order = idx[np.argsort(score[idx], kind="stable")]
    parts = np.array_split(order, 10)
    out = []
    for p in parts:
        mask = np.zeros(score.shape, dtype=bool)
        mask[p] = True
        out.append(mask)
    return out


def wallet_period_stats(d: dict, n_w: int) -> dict[str, np.ndarray]:
    """Per-wallet counts and sums for a trade set: n, den (usd), num (q*a), num_adj (q*a - size*hs)."""
    w = d["w"]
    g = d["q"] * d["a"]
    return {
        "n": np.bincount(w, minlength=n_w),
        "den": np.bincount(w, weights=d["usd"], minlength=n_w),
        "num": np.bincount(w, weights=g, minlength=n_w),
        "num_adj": np.bincount(w, weights=g - d["hs_usd"], minlength=n_w),
    }


def rank_score(fs: dict) -> np.ndarray:
    """Score used for deciles/ranks: the EB-shrunk skill; if tau^2 = 0 (all shrunk values equal) the raw skill."""
    sh = fs["shrunk"]
    if not np.isfinite(fs["tau2"]) or fs["tau2"] <= 0:
        return np.where(np.isfinite(sh), fs["skill"], np.nan)
    return sh


def fit_skill(d: dict, n_w: int) -> dict[str, np.ndarray]:
    """Fit-period wallet skill: raw, analytic null variance (of the per-dollar skill), shrunk, counts."""
    st = wallet_period_stats(d, n_w)
    den = np.where(st["den"] > 0, st["den"], np.nan)
    skill = st["num"] / den
    nm, nv = null_moments(d["w"], d["m"], d["q"], d["a"], n_w)
    v = nv / den**2
    mask = (st["n"] >= MIN_EB) & np.isfinite(skill) & np.isfinite(v)
    eb = eb_shrink(skill, v, mask)
    return {"skill": skill, "var": v, "null_mean_skill": nm / den, "shrunk": eb["shrunk"], "mu": eb["mu"],
            "tau2": eb["tau2"], "n": st["n"], "den": st["den"], "eb_mask": mask}  # fmt: skip


def persistence_test(
    fit: dict,
    tune: dict,
    n_w: int,
    n_events: int,
    n_boot: int = 1000,
    n_shuffle: int = 200,
    seed: int = SEED,
    log: Callable = _noop,
) -> dict:
    """Fit-period classification, tune-period persistence (24 h markout per dollar), shuffled baseline.

    ``fit`` / ``tune``: dicts of equal-length arrays w (wallet code), m (market code), e (event code), q, a
    (size * price change in YES units), usd, hs_usd (size * half-spread proxy). Wallets need >= 20 trades in
    EACH period; deciles are of the fit EB-shrunk skill among them.
    """
    from scipy import sparse

    rng = np.random.default_rng(seed)
    uniq_e, dense_e = np.unique(tune["e"], return_inverse=True)  # bootstrap clusters = events present in tune
    tune = {**tune, "e": dense_e}
    n_events = int(uniq_e.size)
    fs = fit_skill(fit, n_w)
    ts_ = wallet_period_stats(tune, n_w)
    score = rank_score(fs)
    elig = (fs["n"] >= MIN_PERSIST) & (ts_["n"] >= MIN_PERSIST) & np.isfinite(score) & (ts_["den"] > 0)
    out: dict = {"n_elig": int(elig.sum()), "fit_mu": fs["mu"], "fit_tau2": fs["tau2"]}
    if elig.sum() < 20:
        out["verdict"] = "insufficient wallets"
        return out
    tune_skill = ts_["num"] / np.where(ts_["den"] > 0, ts_["den"], np.nan)
    deciles = _decile_sets(score, elig)
    counts = bootstrap_counts(n_events, n_boot, rng)
    gt = tune["q"] * tune["a"]
    gt_adj = gt - tune["hs_usd"]

    def group_stat(sel: np.ndarray, adj: bool = False) -> tuple[float, float, float]:
        rows = sel[tune["w"]]
        num = (gt_adj if adj else gt)[rows]
        return pooled_ratio_ci(num, tune["usd"][rows], tune["e"][rows], n_events, counts)

    dec_rows = []
    for i, sel in enumerate(deciles):
        p, lo, hi = group_stat(sel)
        pa, loa, hia = group_stat(sel, adj=True)
        dec_rows.append({"decile": i + 1, "n_wallets": int(sel.sum()), "mean": p, "lo": lo, "hi": hi,
                         "mean_adj": pa, "lo_adj": loa, "hi_adj": hia})  # fmt: skip
    all_sel = elig
    allp = group_stat(all_sel)
    alla = group_stat(all_sel, adj=True)
    # top - bottom difference with event bootstrap
    def diff_ci(sa: np.ndarray, sb: np.ndarray) -> tuple[float, float, float]:
        def arr(sel):
            rows = sel[tune["w"]]
            e = tune["e"][rows]
            return (np.bincount(e, weights=gt[rows], minlength=n_events),
                    np.bincount(e, weights=tune["usd"][rows], minlength=n_events))  # fmt: skip

        na, da = arr(sa)
        nb, db = arr(sb)
        point = na.sum() / da.sum() - nb.sum() / db.sum()
        bn_a, bd_a, bn_b, bd_b = counts @ na, counts @ da, counts @ nb, counts @ db
        ok = (bd_a > 0) & (bd_b > 0)
        b = bn_a[ok] / bd_a[ok] - bn_b[ok] / bd_b[ok]
        return float(point), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    dtb = diff_ci(deciles[9], deciles[0])
    # rank correlation fit shrunk vs tune skill, with event bootstrap of the tune side
    idx = np.flatnonzero(elig)
    pos = np.full(n_w, -1, np.int64)
    pos[idx] = np.arange(idx.size)
    rows = elig[tune["w"]]
    cw = pos[tune["w"][rows]]
    ce = tune["e"][rows]
    shp = (idx.size, n_events)
    Mn = sparse.coo_matrix((gt[rows], (cw, ce)), shape=shp).tocsr()
    Md = sparse.coo_matrix((tune["usd"][rows], (cw, ce)), shape=shp).tocsr()
    rho = _spearman(score[idx], tune_skill[idx])
    NUM = Mn @ counts.T
    DEN = Md @ counts.T
    rhos = []
    xs = score[idx]
    for b in range(n_boot):
        den_b = DEN[:, b]
        ok = den_b > 0
        sk = np.full(idx.size, np.nan)
        sk[ok] = NUM[ok, b] / den_b[ok]
        rhos.append(_spearman(xs, sk))
    rhos = np.array(rhos)
    rho_lo, rho_hi = float(np.nanpercentile(rhos, 2.5)), float(np.nanpercentile(rhos, 97.5))
    # shuffled-wallet baseline: permute wallet identities over the fit trades, redo EB + deciles
    base_top, base_rho = [], []
    for s in range(n_shuffle):
        perm = rng.permutation(fit["w"].size)
        d2 = dict(fit)
        d2["w"] = fit["w"][perm]
        f2 = fit_skill(d2, n_w)
        sc2 = rank_score(f2)
        el2 = elig & np.isfinite(sc2)
        if el2.sum() < 20:
            continue
        dec = _decile_sets(sc2, el2)
        top = dec[9]
        base_top.append(ts_["num"][top].sum() / ts_["den"][top].sum())
        base_rho.append(_spearman(sc2[idx], tune_skill[idx]))
        if (s + 1) % 50 == 0:
            log(f"  shuffled baseline {s + 1}/{n_shuffle}")
    base_top = np.array(base_top)
    base_rho = np.array(base_rho)
    top, bot = dec_rows[9], dec_rows[0]
    topw = np.flatnonzero(deciles[9])
    botw = np.flatnonzero(deciles[0])
    usd_top = np.sort(ts_["den"][topw])[::-1]
    sk_top, sk_bot = tune_skill[topw], tune_skill[botw]
    conc = {
        "top_decile_largest_wallet_usd_share": float(usd_top[0] / usd_top.sum()),
        "top_decile_top5_wallet_usd_share": float(usd_top[:5].sum() / usd_top.sum()),
        "top_decile_wallet_mean_equal_weight": float(np.nanmean(sk_top)),
        "top_decile_wallet_median": float(np.nanmedian(sk_top)),
        "bottom_decile_wallet_median": float(np.nanmedian(sk_bot)),
        "top_decile_share_wallets_positive": float(np.nanmean(sk_top > 0)),
        "bottom_decile_share_wallets_positive": float(np.nanmean(sk_bot > 0)),
    }
    checks = {
        "top_decile_gt0_lower_bound": bool(top["lo"] > 0),
        "top_above_bottom": bool(top["mean"] > bot["mean"]),
        "rank_corr_lower_bound_gt0": bool(rho_lo > 0),
        "baseline_p95_below_observed": bool(base_top.size > 0 and np.percentile(base_top, 95) < top["mean"]),
    }
    out.update(
        {
            "deciles": dec_rows,
            "all": {"mean": allp[0], "lo": allp[1], "hi": allp[2], "mean_adj": alla[0], "lo_adj": alla[1], "hi_adj": alla[2]},
            "top_minus_bottom": {"mean": dtb[0], "lo": dtb[1], "hi": dtb[2]},
            "rho": rho, "rho_lo": rho_lo, "rho_hi": rho_hi,
            "base_top": base_top, "base_rho": base_rho,
            "base_top_p95": float(np.percentile(base_top, 95)) if base_top.size else float("nan"),
            "base_top_mean": float(base_top.mean()) if base_top.size else float("nan"),
            "base_rho_p95": float(np.percentile(base_rho, 95)) if base_rho.size else float("nan"),
            "checks": checks, "concentration": conc,
            "pass": bool(all(checks.values())),
            "verdict": "PASS (skill persists)" if all(checks.values()) else "FAIL (no persistent markout skill)",
        }
    )  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------- driver


def load_trade_arrays(horizon: str, parts: Path = PARTS_DIR) -> pl.DataFrame:
    cols = ["wcode", "mcode", "ecode", "period", "ts", "closed_ts", "q", "size", "usd", "hs", f"d_{horizon}"]
    df = pl.scan_parquet(str(parts / "part_*.parquet")).select(cols).filter(pl.col(f"d_{horizon}").is_not_null()).collect()
    assert_not_sealed(df)
    return df


def _arrays(df: pl.DataFrame, horizon: str) -> dict:
    size = df["size"].to_numpy().astype(np.float64)
    d = df[f"d_{horizon}"].to_numpy().astype(np.float64)
    return {
        "w": df["wcode"].to_numpy().astype(np.int64),
        "m": df["mcode"].to_numpy().astype(np.int64),
        "e": df["ecode"].to_numpy().astype(np.int64),
        "q": df["q"].to_numpy().astype(np.float64),
        "a": size * d,
        "usd": df["usd"].to_numpy().astype(np.float64),
        "hs_usd": size * df["hs"].to_numpy().astype(np.float64),
    }


def _sub(d: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in d.items()}


def embargo_markets(df: pl.DataFrame, boundary: datetime = FIT_END, embargo: timedelta = EMBARGO) -> pl.Series:
    """Market codes whose [first trade, closed_ts] interval intersects [boundary - 7 d, boundary + 7 d]."""
    per = df.group_by("mcode").agg(pl.col("ts").min().alias("_first"), pl.col("closed_ts").first())
    bad = per.filter((pl.col("_first") < boundary + embargo) & (pl.col("closed_ts") > boundary - embargo))
    return bad["mcode"]


def run_markouts(
    n_draws: int = 2000,
    n_boot: int = 1000,
    n_shuffle: int = 200,
    threads: int = 8,
    rebuild: bool = False,
    snap: Path = SNAP,
    log: Callable = print,
) -> dict:
    """Full task B; writes data/private/wallet_skill.parquet and aggregate results (see module docstring)."""
    t0 = time.time()
    markets = load_markets(snap)
    assert_not_sealed(markets)
    log(f"markets (non-sealed, YES/NO): {markets.height}  by period: {markets['period'].value_counts().to_dicts()}")
    if rebuild or not list(PARTS_DIR.glob("part_*.parquet")):
        n = build_trade_markouts(markets, snap, log=log)
        log(f"trade-level markouts built: {n} trades ({time.time() - t0:.0f}s)")
    wtab = pl.read_parquet(PRIVATE / "markout_wallets.parquet")
    n_w = int(wtab.height)
    n_events = int(markets["ecode"].max()) + 1
    from scipy.stats import norm

    from pmdetect.stage2 import markouts_report as rep

    # --- descriptive: every horizon, pooled and per wallet
    desc = rep.descriptive(markets, n_w, n_events, n_boot, log)
    # --- headline horizon
    df = load_trade_arrays(HEADLINE)
    log(f"24h-eligible trades: {df.height}  ({time.time() - t0:.0f}s)")
    res: dict = {"desc": desc, "counts": {}}
    skill_rows = []
    per_period = {}
    for period in ("fit", "tune"):
        sub = df.filter(pl.col("period") == period)
        d = _arrays(sub, HEADLINE)
        n_trade_w = np.bincount(d["w"], minlength=n_w)
        log(f"{period}: {sub.height} trades, {int((n_trade_w > 0).sum())} wallets, {sub['mcode'].n_unique()} markets")
        sr = sign_randomization(d["w"], d["m"], d["q"], d["a"], n_w, n_draws=n_draws, threads=threads)
        log(f"{period}: sign randomisation done ({time.time() - t0:.0f}s)")
        fs = fit_skill(d, n_w)
        den = np.where(fs["den"] > 0, fs["den"], np.nan)
        elig = fs["n"] >= MIN_EB
        p_up = np.where(elig, sr["p_upper"], np.nan)
        q_bh = bh_adjusted(p_up)
        z = np.where(sr["null_sd"] > 0, (sr["obs"] - sr["null_mean"]) / np.where(sr["null_sd"] > 0, sr["null_sd"], 1.0), np.nan)
        skilled = elig & (fs["shrunk"] > 0) & (q_bh <= 0.05)
        # resolution-free variant: normal p from the exact null mean and variance (the Monte-Carlo p cannot go
        # below 1 / (n_draws + 1), so BH over tens of thousands of wallets cannot reject with 2,000 draws)
        sd_an = np.sqrt(fs["var"])
        z_an = np.where(sd_an > 0, (fs["skill"] - fs["null_mean_skill"]) / np.where(sd_an > 0, sd_an, 1.0), np.nan)
        p_an = np.where(elig & np.isfinite(z_an), norm.sf(z_an), np.nan)
        q_an = bh_adjusted(p_an)
        skilled_an = elig & (fs["shrunk"] > 0) & (q_an <= 0.05)
        per_period[period] = {"d": d, "fs": fs, "skilled": skilled, "skilled_an": skilled_an, "elig": elig,
                              "q_bh": q_bh, "z": z}  # fmt: skip
        res["counts"][period] = {"trades": int(sub.height), "wallets": int((n_trade_w > 0).sum()),
                                 "wallets_ge10": int(elig.sum()), "markets": int(sub["mcode"].n_unique()),
                                 "events": int(sub["ecode"].n_unique()), "skilled": int(skilled.sum()),
                                 "skilled_analytic": int(skilled_an.sum()), "min_mc_p": float(np.nanmin(p_up)),
                                 "mu": fs["mu"], "tau2": fs["tau2"]}  # fmt: skip
        st = wallet_period_stats(d, n_w)
        idx = np.flatnonzero(n_trade_w > 0)
        skill_rows.append(
            pl.DataFrame(
                {
                    "wcode": idx.astype(np.int32),
                    "period": period,
                    "n_trades": n_trade_w[idx],
                    "usd": st["den"][idx],
                    "skill_24h": (st["num"] / den)[idx],
                    "skill_24h_spread_adj": (st["num_adj"] / den)[idx],
                    "null_mean_skill": fs["null_mean_skill"][idx],
                    "null_sd_skill": (sr["null_sd"] / den)[idx],
                    "z": z[idx],
                    "p_upper": p_up[idx],
                    "p_two": np.where(elig, sr["p_two"], np.nan)[idx],
                    "q_bh": q_bh[idx],
                    "shrunk_skill": fs["shrunk"][idx],
                    "eb_eligible": elig[idx],
                    "skilled": skilled[idx],
                    "z_analytic": z_an[idx],
                    "p_analytic": p_an[idx],
                    "q_bh_analytic": q_an[idx],
                    "skilled_analytic": skilled_an[idx],
                }
            )
        )
        del sr
    sk = pl.concat(skill_rows).join(wtab, on="wcode", how="left")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    sk.write_parquet(PRIVATE / "wallet_skill.parquet")
    log(f"wrote data/private/wallet_skill.parquet: {sk.height} rows ({time.time() - t0:.0f}s)")
    # --- persistence with the 7-day embargo
    bad = embargo_markets(df).to_numpy()
    res["counts"]["embargo_markets_dropped"] = int(bad.size)
    fit_d = per_period["fit"]["d"]
    tune_d = per_period["tune"]["d"]
    fit_p = _sub(fit_d, ~np.isin(fit_d["m"], bad))
    tune_p = _sub(tune_d, ~np.isin(tune_d["m"], bad))
    res["counts"]["persistence_trades"] = {"fit": int(fit_p["w"].size), "tune": int(tune_p["w"].size)}
    log(f"persistence sets after embargo: fit {fit_p['w'].size}, tune {tune_p['w'].size}")
    res["persistence"] = persistence_test(fit_p, tune_p, n_w, n_events, n_boot, n_shuffle, log=log)
    res["verdict"] = res["persistence"].get("verdict", "n/a")
    f_, t_ = per_period["fit"], per_period["tune"]
    f_sk = f_["skilled_an"]
    t_el = t_["elig"]
    t_skill = t_["fs"]["skill"]
    both = f_sk & t_el
    res["retention"] = {
        "fit_skilled": int(f_sk.sum()),
        "fit_skilled_with_tune_ge10": int(both.sum()),
        "tune_positive_share": float((both & (t_skill > 0)).sum() / max(1, both.sum())),
        "tune_skilled_share": float((both & t_["skilled_an"]).sum() / max(1, both.sum())),
        "tune_skilled_share_all_wallets": float(t_["skilled_an"].sum() / max(1, t_el.sum())),
    }
    res["skilled_share"] = {
        p: per_period[p]["skilled"].sum() / max(1, per_period[p]["elig"].sum()) for p in per_period
    }
    # skilled fit wallets' share of tune-period dollars etc. is reported by the report module
    rep.write_outputs(res, log)
    log(f"done in {time.time() - t0:.0f}s: {res['verdict']}")
    return res
