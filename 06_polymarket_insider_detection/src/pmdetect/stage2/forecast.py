"""Stage 2, task C: forecasting benchmark against the market price (the STOP GATE; docs/STAGE2_PLAN.md s.9).

Task: at t = end_ts - H (H in 7 d, 3 d, 1 d), for markets still open at t (closed_ts > t) with >= 20 taker trades
strictly before t, predict the YES outcome. Benchmark = the market price at t (last observation strictly before t of
the YES-equivalent series: taker fills merged with the stored winner-token history). Prices are clipped to
[0.001, 0.999] for every loss.

Features (all computed from rows strictly before t; a test perturbs everything at/after t and checks invariance):
  price dynamics : logit price; logit change over 1 h, 24 h, 7 d (last observation at/before t - w)
  flow           : taker order-flow imbalance (24 h, YES-direction signed USD / USD), volume surge (24 h USD vs the
                   earlier daily mean, log ratio), Herfindahl of the wallets pushing the YES price up (buy Yes / sell
                   No) over all pre-t trades, fresh-wallet share of 24 h USD (wallet's first tape trade < 7 d earlier)
  wallet skill   : net YES-direction flow of wallets classed skilled on the FIT period (24 h and all pre-t, as a
                   share of USD) and the skilled share of USD. 'Skilled' = EB-shrunk 24 h markout skill > 0 and a
                   Benjamini-Hochberg (5%) adjusted analytic sign-randomisation p. For rows whose market is itself in
                   the fit period the classification is CROSS-FITTED (5 folds by event hash; the market's own fold is
                   left out) so that its own markouts never enter its feature; rows from the tune period use the full
                   fit-period classification.
  impact / other : Kyle-style lambda (1 h price change on signed net taker USD, through the origin, 1 h bins before
                   t, >= 5 bins, scaled per $1,000), median gap between consecutive opposite-side fills within 10 min
                   (spread proxy from pre-t rows only), info_type (one-hot), log volume to date, log trade count,
                   hours since start, hours to end (= H).

Models: (a) logistic recalibration of logit price, (b) HistGradientBoosting depth 3, 200 iterations, learning rate
0.05, no tuning, no early stopping. Purged walk-forward by month of end_ts: test month M in Jan-Jun 2026 (tune
period: market closed_ts in the tune period, t >= 2026-01-01); training rows are those whose label is known 7 days
before the start of M (closed_ts <= start of M - 7 d), from the fit period and earlier tune months. Improvements are
mean(loss_price - loss_model) per row (positive = the model beats the price) with 95% event-cluster bootstrap
intervals (1,000 resamples, by event).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.stage2 import markouts as M
from pmdetect.stage2.common import FIT_END, N_BOOT, SEED, TUNE_END, assert_not_sealed

SNAP = M.SNAP
PRIVATE = M.PRIVATE
RESULTS = M.RESULTS
HORIZONS_D = {"7d": 7, "3d": 3, "1d": 1}
MIN_TRADES = 20
CLIP = 0.001
K_FOLDS = 5
EMBARGO = timedelta(days=7)
HALVES = {"JanMar": (datetime(2026, 1, 1), datetime(2026, 4, 1)), "AprJun": (datetime(2026, 4, 1), datetime(2026, 7, 1))}
PRICE_FEATS = ["logit_p", "dp_1h", "dp_24h", "dp_7d"]
FLOW_FEATS = ["ofi_24h", "vol_surge", "hhi_yes", "fresh_share"]
SKILL_FEATS = ["skill_flow_24h", "skill_flow_all", "skill_share"]
OTHER_FEATS = ["lambda_1k", "spread_gap", "it_deadline", "it_scheduled", "it_unscheduled", "log_vol", "log_n", "hours_since_start", "hours_to_end"]  # fmt: skip
GROUPS = {"price_dynamics": PRICE_FEATS[1:], "flow": FLOW_FEATS, "wallet_skill": SKILL_FEATS, "impact_other": OTHER_FEATS}
ALL_FEATS = PRICE_FEATS + FLOW_FEATS + SKILL_FEATS + OTHER_FEATS
HGB_PARAMS = dict(max_depth=3, max_iter=200, learning_rate=0.05, early_stopping=False, random_state=SEED)  # fmt: skip

warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")


def _noop(msg: str) -> None:
    pass


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------ candidates


def candidate_snapshots(markets: pl.DataFrame) -> pl.DataFrame:
    """(mcode, H, t) rows for markets open at t = end_ts - H (closed_ts > t, start_ts < t). Sealed markets never enter."""
    assert_not_sealed(markets)
    parts = []
    for name, days in HORIZONS_D.items():
        parts.append(
            markets.select(
                "mcode",
                pl.lit(name).alias("H"),
                (pl.col("end_ts") - pl.duration(days=days)).alias("t"),
                "closed_ts",
                "start_ts",
            )
        )
    s = pl.concat(parts).filter((pl.col("closed_ts") > pl.col("t")) & (pl.col("start_ts") < pl.col("t")))
    return s.select("mcode", "H", "t")


# ------------------------------------------------------------------------------------------- skill flags


def classify_skilled(d: dict, n_w: int) -> np.ndarray:
    """Boolean per wallet: EB-shrunk skill > 0 and BH-adjusted (5%) analytic one-sided sign-randomisation p."""
    from scipy.stats import norm

    fs = M.fit_skill(d, n_w)
    sd = np.sqrt(fs["var"])
    z = np.where(sd > 0, (fs["skill"] - fs["null_mean_skill"]) / np.where(sd > 0, sd, 1.0), np.nan)
    elig = fs["eb_mask"] & np.isfinite(z)
    p = np.where(elig, norm.sf(z), np.nan)
    q = bh_adj(p)
    return elig & (fs["shrunk"] > 0) & (q <= 0.05)


def bh_adj(p: np.ndarray) -> np.ndarray:
    from pmdetect.wallets.fdr import bh_adjusted

    return bh_adjusted(p)


def event_fold(ecode: np.ndarray, k: int = K_FOLDS) -> np.ndarray:
    """Deterministic fold of an event code (multiplicative hash; stable across runs)."""
    return ((ecode.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)) % np.uint64(k)


def skill_matrix(fit_d: dict, fold_of_mcode: np.ndarray, n_w: int, k: int = K_FOLDS) -> np.ndarray:
    """(n_w x (k+1)) skilled flags: column f = classified on fit trades outside fold f, column k = all fit trades."""
    out = np.zeros((n_w, k + 1), dtype=bool)
    fold = fold_of_mcode[fit_d["m"]]
    for f in range(k):
        keep = fold != f
        out[:, f] = classify_skilled({key: v[keep] for key, v in fit_d.items()}, n_w)
    out[:, k] = classify_skilled(fit_d, n_w)
    return out


# ----------------------------------------------------------------------------------------- features


def snapshot_features(tr: pl.DataFrame, hist: pl.DataFrame, snap: pl.DataFrame) -> pl.DataFrame:
    """Features at each snapshot (mcode, H, t) from rows strictly before t.

    ``tr``: mcode, wcode, ts, q (+-1, YES direction), usd, y (YES-equivalent price), first_ts (wallet's first tape
    trade), sk (skilled flag, 0/1) with the fills of the snapshot markets; ``hist``: mcode, ots, y (price history).
    Rows at or after t are filtered out first, so nothing later can reach a feature.
    """
    tr = tr.sort(["mcode", "ts"], maintain_order=True)
    prev_q = pl.col("q").shift(1).over("mcode")
    gap_ok = (prev_q == -pl.col("q")) & ((pl.col("ts") - pl.col("ts").shift(1).over("mcode")).dt.total_seconds() <= 600)
    tr = tr.with_columns(
        pl.when(gap_ok).then((pl.col("y") - pl.col("y").shift(1).over("mcode")).abs()).otherwise(None).alias("gap")
    )
    j = tr.join(snap, on="mcode", how="inner").filter(pl.col("ts") < pl.col("t"))
    in24 = pl.col("ts") >= pl.col("t") - pl.duration(hours=24)
    fresh = (pl.col("ts") - pl.col("first_ts")).dt.total_seconds() < 7 * 86400
    qu = pl.col("q") * pl.col("usd")
    base = j.group_by(["mcode", "H"]).agg(
        pl.col("t").first(),
        pl.len().alias("n_pre"),
        pl.col("usd").sum().alias("vol"),
        pl.col("ts").min().alias("first_trade"),
        pl.col("usd").filter(in24).sum().alias("v24"),
        qu.filter(in24).sum().alias("q24"),
        pl.col("usd").filter(~in24).sum().alias("vprior"),
        pl.col("usd").filter(in24 & fresh).sum().alias("fresh24"),
        (qu * pl.col("sk")).sum().alias("skq_all"),
        (qu * pl.col("sk")).filter(in24).sum().alias("skq_24"),
        (pl.col("usd") * pl.col("sk")).sum().alias("sku_all"),
        pl.col("gap").median().alias("spread_gap"),
    )
    base = base.with_columns(
        pl.when(pl.col("v24") > 0).then(pl.col("q24") / pl.col("v24")).otherwise(None).alias("ofi_24h"),
        pl.when(pl.col("v24") > 0).then(pl.col("fresh24") / pl.col("v24")).otherwise(None).alias("fresh_share"),
        pl.when(pl.col("v24") > 0).then(pl.col("skq_24") / pl.col("v24")).otherwise(None).alias("skill_flow_24h"),
        (pl.col("skq_all") / pl.col("vol")).alias("skill_flow_all"),
        (pl.col("sku_all") / pl.col("vol")).alias("skill_share"),
        (
            (pl.col("t") - pl.duration(hours=24) - pl.col("first_trade")).dt.total_seconds() / 86400.0
        ).clip(lower_bound=1.0).alias("_span_d"),
    ).with_columns(
        (
            (pl.col("v24") + 1).log() - (pl.col("vprior") / pl.col("_span_d") + 1).log()
        ).alias("vol_surge"),
        (pl.col("vol") + 1).log().alias("log_vol"),
        pl.col("n_pre").cast(pl.Float64).log().alias("log_n"),
    )
    # Herfindahl of the wallets pushing the YES price up
    up = j.filter(pl.col("q") > 0).group_by(["mcode", "H", "wcode"]).agg(pl.col("usd").sum().alias("u"))
    hhi = up.with_columns((pl.col("u") / pl.col("u").sum().over(["mcode", "H"])).alias("s")).group_by(["mcode", "H"]).agg(
        (pl.col("s") ** 2).sum().alias("hhi_yes")
    )
    # Kyle-style lambda: hourly bins going back from t
    jb = j.with_columns(((pl.col("t") - pl.col("ts")).dt.total_seconds() // 3600).cast(pl.Int64).alias("bin")).sort("ts")
    bins = (
        jb.group_by(["mcode", "H", "bin"])
        .agg(qu.sum().alias("x"), pl.col("y").last().alias("p_last"))
        .sort(["mcode", "H", "bin"], descending=[False, False, True])
    )
    bins = bins.with_columns(pl.col("p_last").diff().over(["mcode", "H"]).alias("dp")).filter(pl.col("dp").is_not_null())
    lam = bins.group_by(["mcode", "H"]).agg(
        pl.len().alias("n_bins"),
        (pl.col("x") * pl.col("dp")).sum().alias("_xd"),
        (pl.col("x") ** 2).sum().alias("_xx"),
    )
    lam = lam.with_columns(
        pl.when((pl.col("n_bins") >= 5) & (pl.col("_xx") > 0))
        .then(pl.col("_xd") / pl.col("_xx") * 1000.0)
        .otherwise(None)
        .alias("lambda_1k")
    ).select("mcode", "H", "lambda_1k")
    out = base.join(hhi, on=["mcode", "H"], how="left").join(lam, on=["mcode", "H"], how="left")
    # prices: last observation at or before t - 1 us (strictly before t), and at t-1h, t-24h, t-7d
    series = pl.concat(
        [
            tr.select("mcode", pl.col("ts").alias("ots"), pl.col("y").alias("y_obs"), pl.lit(1).alias("_s")),
            hist.select("mcode", "ots", pl.col("y").alias("y_obs"), pl.lit(0).alias("_s")),
        ]
    ).sort(["ots", "_s"]).drop("_s")
    pt = out.select("mcode", "H", "t")
    for name, delta in (("p0", timedelta(microseconds=1)), ("p1h", timedelta(hours=1)), ("p24h", timedelta(hours=24)), ("p7d", timedelta(days=7))):
        q_ = pt.select("mcode", "H", (pl.col("t") - delta).alias("_tt")).sort("_tt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = q_.join_asof(series, left_on="_tt", right_on="ots", by="mcode", strategy="backward")
        out = out.join(r.select("mcode", "H", pl.col("y_obs").alias(name)), on=["mcode", "H"], how="left")
    cl = lambda c: pl.col(c).clip(CLIP, 1 - CLIP)  # noqa: E731
    lg = lambda c: (cl(c) / (1 - cl(c))).log()  # noqa: E731
    out = out.with_columns(
        cl("p0").alias("price"),
        lg("p0").alias("logit_p"),
        (lg("p0") - lg("p1h")).alias("dp_1h"),
        (lg("p0") - lg("p24h")).alias("dp_24h"),
        (lg("p0") - lg("p7d")).alias("dp_7d"),
    )
    return out.select("mcode", "H", "t", "n_pre", "price", *[c for c in ALL_FEATS if c in out.columns or c == "logit_p"])


# -------------------------------------------------------------------------------------- build features


def build_features(
    markets: pl.DataFrame, snap_dir: Path = SNAP, parts: Path = M.PARTS_DIR, n_parts: int = 8, log: Callable = _noop
) -> pl.DataFrame:
    """Features for every (market, H) snapshot over fit + tune markets (sealed markets are not in ``markets``)."""
    t0 = time.time()
    assert_not_sealed(markets)
    n_w = pl.read_parquet(PRIVATE / "markout_wallets.parquet").height
    snaps = candidate_snapshots(markets)
    log(f"candidate snapshots (closed_ts > t): {snaps.height}")
    scan = pl.scan_parquet(str(parts / "part_*.parquet"))
    # skilled flags from the FIT period 24 h markouts (cross-fitted for fit-period markets)
    df = M.load_trade_arrays(M.HEADLINE, parts).filter(pl.col("period") == "fit")
    fit_d = M._arrays(df, M.HEADLINE)
    n_m = int(markets["mcode"].max()) + 1
    ecode_of_m = np.zeros(n_m, dtype=np.int64)
    ecode_of_m[markets["mcode"].to_numpy()] = markets["ecode"].to_numpy()
    fold_of_m = event_fold(ecode_of_m).astype(np.int64)
    smat = skill_matrix(fit_d, fold_of_m, n_w)
    log(f"skill matrix: {smat.sum(axis=0).tolist()} skilled wallets per fold column + full ({time.time() - t0:.0f}s)")
    del df, fit_d
    col_of_m = np.full(n_m, K_FOLDS, dtype=np.int64)
    fit_m = markets.filter(pl.col("closed_ts") < FIT_END)["mcode"].to_numpy()
    col_of_m[fit_m] = fold_of_m[fit_m]
    first = scan.group_by("wcode").agg(pl.col("ts").min().alias("first_ts")).collect()
    tokens = pl.read_parquet(snap_dir / "tokens.parquet")
    hist_all = pl.read_parquet(snap_dir / "prices.parquet").join(tokens, on="token_id", how="inner")
    cmap = markets.select("condition_id", "mcode")
    hist_all = (
        hist_all.join(cmap, on="condition_id", how="inner")
        .with_columns(pl.col("outcome_label").str.strip_chars().str.to_uppercase().alias("_l"))
        .filter(pl.col("_l").is_in(["YES", "NO"]))
        .select("mcode", pl.col("ts").alias("ots"), pl.when(pl.col("_l") == "YES").then(pl.col("price")).otherwise(1.0 - pl.col("price")).alias("y"))
    )
    outs = []
    for k in range(n_parts):
        sk_m = snaps.filter(pl.col("mcode") % n_parts == k)
        if sk_m.height == 0:
            continue
        tr = (
            scan.filter(pl.col("mcode").is_in(sk_m["mcode"].unique().implode()))
            .select("mcode", "wcode", "ts", "q", "usd", "y")
            .collect()
        )
        w = tr["wcode"].to_numpy().astype(np.int64)
        mc = tr["mcode"].to_numpy().astype(np.int64)
        tr = tr.with_columns(
            pl.Series("sk", smat[w, col_of_m[mc]].astype(np.int8)),
            pl.col("q").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
        )
        tr = tr.join(first, on="wcode", how="left")
        h = hist_all.filter(pl.col("mcode").is_in(sk_m["mcode"].unique().implode()))
        outs.append(snapshot_features(tr, h, sk_m))
        log(f"  features part {k + 1}/{n_parts}: {outs[-1].height} rows ({time.time() - t0:.0f}s)")
    feat = pl.concat(outs).filter(pl.col("n_pre") >= MIN_TRADES)
    meta = markets.select("mcode", "ecode", "event_id", "info_type", "outcome", "start_ts", "end_ts", "closed_ts", "volume_usd")
    feat = feat.join(meta, on="mcode", how="left")
    feat = feat.with_columns(
        (pl.col("outcome") == "YES").cast(pl.Int8).alias("y"),
        (pl.col("info_type") == "deadline").cast(pl.Float64).alias("it_deadline"),
        (pl.col("info_type") == "scheduled_reveal").cast(pl.Float64).alias("it_scheduled"),
        (pl.col("info_type") == "unscheduled_event").cast(pl.Float64).alias("it_unscheduled"),
        ((pl.col("t") - pl.col("start_ts")).dt.total_seconds() / 3600.0).alias("hours_since_start"),
        ((pl.col("end_ts") - pl.col("t")).dt.total_seconds() / 3600.0).alias("hours_to_end"),
    )
    assert_not_sealed(feat)
    return feat


# ---------------------------------------------------------------------------------------- evaluation


def month_start(ts: datetime) -> datetime:
    return datetime(ts.year, ts.month, 1)


def train_cutoff(m_start: datetime, embargo: timedelta = EMBARGO) -> datetime:
    """Labels must be known (closed_ts) at least ``embargo`` before the start of the test month."""
    return m_start - embargo


def split_masks(feat: pl.DataFrame, m_start: datetime, m_end: datetime) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) row masks for the test month [m_start, m_end) by end_ts.

    test: end_ts in the month, market closed in the tune period, t >= 2026-01-01. train: closed_ts <= m_start - 7 d.
    """
    closed = feat["closed_ts"].to_numpy()
    end = feat["end_ts"].to_numpy()
    t = feat["t"].to_numpy()
    c = np.datetime64(train_cutoff(m_start))
    train = closed <= c
    test = (
        (end >= np.datetime64(m_start))
        & (end < np.datetime64(m_end))
        & (closed >= np.datetime64(FIT_END))
        & (closed < np.datetime64(TUNE_END))
        & (t >= np.datetime64(FIT_END))
    )
    return train, test & ~train


def losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(p, CLIP, 1 - CLIP)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ll, (p - y) ** 2


def fit_predict(kind: str, cols: list[str], x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    """kind 'recal' (logistic on logit price, column 0) or 'hgb' (fixed hyperparameters)."""
    if len(np.unique(y_tr)) < 2:
        return 1.0 / (1.0 + np.exp(-x_te[:, 0]))  # identity: the price itself
    if kind == "recal":
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=1e6, max_iter=1000)
        m.fit(x_tr[:, :1], y_tr)
        return m.predict_proba(x_te[:, :1])[:, 1]
    from sklearn.ensemble import HistGradientBoostingClassifier

    m = HistGradientBoostingClassifier(**HGB_PARAMS)
    m.fit(x_tr, y_tr)
    return m.predict_proba(x_te)[:, 1]


def model_specs() -> dict[str, tuple[str, list[str]]]:
    """name -> (kind, feature columns); logit price is always first."""
    s: dict[str, tuple[str, list[str]]] = {"recal": ("recal", ["logit_p"]), "hgb_all": ("hgb", ALL_FEATS)}
    s["hgb_price_only"] = ("hgb", ["logit_p"])
    for g, cols in GROUPS.items():
        s[f"hgb_plus_{g}"] = ("hgb", ["logit_p", *cols])
        s[f"hgb_minus_{g}"] = ("hgb", [c for c in ALL_FEATS if c not in cols])
    return s


def walk_forward(feat: pl.DataFrame, h: str, specs: dict | None = None) -> pl.DataFrame:
    """Out-of-sample predictions for horizon ``h`` over test months Jan-Jun 2026 (one row per test snapshot)."""
    specs = specs or model_specs()
    fh = feat.filter(pl.col("H") == h).sort("t")
    outs = []
    for mo in range(1, 7):
        m0, m1 = datetime(2026, mo, 1), datetime(2026, mo + 1, 1) if mo < 12 else datetime(2027, 1, 1)
        tr_m, te_m = split_masks(fh, m0, m1)
        if te_m.sum() == 0 or tr_m.sum() < 30:
            continue
        assert fh.filter(pl.Series(tr_m))["closed_ts"].max() <= train_cutoff(m0)  # purge / embargo guard
        te = fh.filter(pl.Series(te_m))
        y_tr = fh.filter(pl.Series(tr_m))["y"].to_numpy()
        res = te.select("mcode", "ecode", "H", "t", "end_ts", "closed_ts", "volume_usd", "info_type", "price", "y", "log_vol")
        res = res.with_columns(pl.lit(mo).alias("month"))
        for name, (kind, cols) in specs.items():
            x_tr = fh.filter(pl.Series(tr_m)).select(cols).to_numpy().astype(float)
            x_te = te.select(cols).to_numpy().astype(float)
            if kind == "recal":
                x_tr, x_te = np.nan_to_num(x_tr), np.nan_to_num(x_te)
            res = res.with_columns(pl.Series(f"p_{name}", fit_predict(kind, cols, x_tr, y_tr, x_te)))
        outs.append(res)
    return pl.concat(outs) if outs else pl.DataFrame()


def improvement_ci(
    y: np.ndarray, p_price: np.ndarray, p_model: np.ndarray, ecode: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """Mean per-row improvement (price loss - model loss) for log-loss and Brier with event-cluster bootstrap."""
    ll0, br0 = losses(y, p_price)
    ll1, br1 = losses(y, p_model)
    _, e = np.unique(ecode, return_inverse=True)
    ne = int(e.max()) + 1 if e.size else 0
    out = {"n": int(y.size), "events": ne, "logloss_price": float(ll0.mean()) if y.size else np.nan,
           "logloss_model": float(ll1.mean()) if y.size else np.nan, "brier_price": float(br0.mean()) if y.size else np.nan,
           "brier_model": float(br1.mean()) if y.size else np.nan}  # fmt: skip
    if y.size == 0:
        return out | {"dll": np.nan, "dll_lo": np.nan, "dll_hi": np.nan, "dbr": np.nan, "dbr_lo": np.nan, "dbr_hi": np.nan}
    counts = M.bootstrap_counts(ne, n_boot, np.random.default_rng(seed))
    n_e = np.bincount(e, minlength=ne).astype(float)
    for key, diff in (("dll", ll0 - ll1), ("dbr", br0 - br1)):
        s_e = np.bincount(e, weights=diff, minlength=ne)
        b = (counts @ s_e) / (counts @ n_e)
        out[key] = float(diff.mean())
        out[key + "_lo"], out[key + "_hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return out


def evaluate(oos: pl.DataFrame, n_boot: int = N_BOOT) -> list[dict]:
    """Improvement rows per model x sample (pooled, each half, stress test without the largest 10% by volume)."""
    rows = []
    models = [c[2:] for c in oos.columns if c.startswith("p_")]
    t = oos["t"].to_numpy()
    end = oos["end_ts"].to_numpy()
    vol = oos["log_vol"].to_numpy()
    cut = np.nanpercentile(vol, 90) if vol.size else np.inf
    samples = {
        "pooled": np.ones(oos.height, bool),
        "JanMar": (end >= np.datetime64(HALVES["JanMar"][0])) & (end < np.datetime64(HALVES["JanMar"][1])),
        "AprJun": (end >= np.datetime64(HALVES["AprJun"][0])) & (end < np.datetime64(HALVES["AprJun"][1])),
    }
    del t
    base = dict(samples)
    for k, m in base.items():
        samples[f"{k}_drop_top10pct_volume"] = m & (vol < cut)
    for sname, mask in samples.items():
        sub = oos.filter(pl.Series(mask))
        for mod in models:
            r = improvement_ci(sub["y"].to_numpy().astype(float), sub["price"].to_numpy(), sub[f"p_{mod}"].to_numpy(), sub["ecode"].to_numpy(), n_boot)
            rows.append({"model": mod, "sample": sname, **r})
    return rows


def calibration_table(oos: pl.DataFrame, n_bins: int = 10) -> list[dict]:
    """Calibration of the price benchmark: quantile bins of price, mean price vs outcome frequency."""
    if oos.height == 0:
        return []
    p = oos["price"].to_numpy()
    y = oos["y"].to_numpy().astype(float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [
        {"bin": int(i), "n": int((b == i).sum()), "mean_price": float(p[b == i].mean()), "outcome_rate": float(y[b == i].mean())}
        for i in range(len(edges) - 1)
        if (b == i).any()
    ]


# ------------------------------------------------------------------------------------------- driver


def by_group(oos: pl.DataFrame, col: str, models: list[str], n_boot: int) -> list[dict]:
    rows = []
    for g in oos[col].unique().sort().to_list():
        sub = oos.filter(pl.col(col) == g)
        for mod in models:
            r = improvement_ci(sub["y"].to_numpy().astype(float), sub["price"].to_numpy(), sub[f"p_{mod}"].to_numpy(), sub["ecode"].to_numpy(), n_boot)
            rows.append({"group": str(g), "model": mod, **r})
    return rows


def apply_pass_rule(rows: list[dict], horizons: list[str]) -> dict:
    """Frozen rule: model log-loss below the price in BOTH halves with the lower 95% bound > 0, for >= 1 horizon."""
    out: dict = {}
    for h in horizons:
        for mod in ("recal", "hgb_all"):
            r = {x["sample"]: x for x in rows if x["H"] == h and x["model"] == mod}
            ok = all(
                k in r and r[k]["dll"] > 0 and r[k]["dll_lo"] > 0 for k in ("JanMar", "AprJun")
            )
            out[(h, mod)] = bool(ok)
    return out


def run_forecast(n_boot: int = N_BOOT, rebuild: bool = False, snap: Path = SNAP, log: Callable = print) -> dict:
    t0 = time.time()
    markets = M.load_markets(snap)
    assert_not_sealed(markets)
    if not list(M.PARTS_DIR.glob("part_*.parquet")):
        raise RuntimeError("trade-level markouts missing: run `pmdetect stage2 markouts` first")
    fpath = PRIVATE / "forecast_features.parquet"
    if rebuild or not fpath.exists():
        feat = build_features(markets, snap, log=log)
        feat.write_parquet(fpath)
    else:
        feat = pl.read_parquet(fpath)
    assert_not_sealed(feat)
    log(f"features: {feat.height} snapshots ({time.time() - t0:.0f}s)")
    rows: list[dict] = []
    calib: list[dict] = []
    groups: list[dict] = []
    counts: list[dict] = []
    oos_all = []
    for h in HORIZONS_D:
        oos = walk_forward(feat, h)
        if oos.height == 0:
            continue
        oos_all.append(oos)
        ev = evaluate(oos, n_boot)
        rows += [{"H": h, **r} for r in ev]
        calib += [{"H": h, **r} for r in calibration_table(oos)]
        groups += [{"H": h, **r} for r in by_group(oos, "info_type", ["recal", "hgb_all"], n_boot)]
        fh = feat.filter(pl.col("H") == h)
        counts.append({"H": h, "snapshots_all": fh.height, "test_rows": oos.height,
                       "test_markets": int(oos["mcode"].n_unique()), "test_events": int(oos["ecode"].n_unique()),
                       "JanMar": int((oos["end_ts"] < datetime(2026, 4, 1)).sum()),
                       "AprJun": int((oos["end_ts"] >= datetime(2026, 4, 1)).sum()),
                       "yes_rate_test": float(oos["y"].mean())})  # fmt: skip
        log(f"horizon {h}: {oos.height} test rows ({time.time() - t0:.0f}s)")
        pl.concat(oos_all).drop("mcode").write_parquet(PRIVATE / "forecast_oos.parquet")
    passes = apply_pass_rule(rows, list(HORIZONS_D))
    info_pass = any(v for (h, m), v in passes.items() if m == "hgb_all")
    recal_pass = any(v for (h, m), v in passes.items() if m == "recal")
    if info_pass:
        verdict = "PASS (a model with information beyond the price clears the frozen rule for at least one horizon)"
    elif recal_pass:
        verdict = "PASS on recalibration only (the price-only logistic recalibration clears the rule; the feature model does not): calibration, not information"
    else:
        verdict = "FAIL: no measurable information beyond the price. The strategy track STOPS (S2/S3/S5 not run)."
    res = {"rows": rows, "calib": calib, "groups": groups, "counts": counts, "passes": passes, "verdict": verdict,
           "info_pass": info_pass, "recal_pass": recal_pass, "n_features": feat.height}  # fmt: skip
    from pmdetect.stage2 import forecast_report as rep

    rep.write_outputs(res, log)
    log(f"done in {time.time() - t0:.0f}s: {verdict}")
    return res
