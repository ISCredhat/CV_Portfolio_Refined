"""Benjamini-Hochberg control over the per-wallet tests (phase 3A).

Statistical flags only: a wallet that passes is anomalous relative to the recalibrated null, not
proven to have acted on non-public information. Thresholds are the conventional 5% and are never
lowered to obtain hits.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def benjamini_hochberg(p, alpha: float = 0.05) -> np.ndarray:
    """Boolean rejection mask of the BH step-up procedure (NaN p-values are never rejected)."""
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    rej = np.zeros(p.shape, dtype=bool)
    m = int(ok.sum())
    if m == 0:
        return rej
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    thr = alpha * (np.arange(1, m + 1) / m)
    below = p[order] <= thr
    if below.any():
        kmax = int(np.where(below)[0].max())
        rej[order[: kmax + 1]] = True
    return rej


def bh_adjusted(p) -> np.ndarray:
    """BH q-values (NaN stays NaN)."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = idx.size
    if m == 0:
        return q
    order = idx[np.argsort(p[idx], kind="stable")]
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1.0)
    return q


def apply_fdr(tests: pl.DataFrame, min_positions: int = 5, alpha: float = 0.05) -> pl.DataFrame:
    """Add ``eligible``, ``q_pb``, ``flag`` (BH on the exact Poisson-binomial p over eligible wallets),
    ``q_boot``, ``flag_boot`` and ``flag_both`` (both pass independently)."""
    elig = (tests["n_pos"] >= min_positions).to_numpy()
    p_pb = np.where(elig, tests["p_pb"].to_numpy(), np.nan)
    p_bt = np.where(
        elig, tests["p_boot"].cast(pl.Float64).fill_null(float("nan")).to_numpy(), np.nan
    )
    flag = benjamini_hochberg(p_pb, alpha)
    flag_b = benjamini_hochberg(p_bt, alpha)
    return tests.with_columns(
        pl.Series("eligible", elig),
        pl.Series("q_pb", bh_adjusted(p_pb)),
        pl.Series("flag", flag),
        pl.Series("q_boot", bh_adjusted(p_bt)),
        pl.Series("flag_boot", flag_b),
        pl.Series("flag_both", flag & flag_b),
    )


def stage_counts(res: pl.DataFrame, alpha: float = 0.05) -> dict:
    """Aggregate funnel (no wallet identifiers)."""
    e = res.filter(pl.col("eligible"))
    return {
        "wallets_with_resolved_positions": res.height,
        "wallets_ge5_positions": e.height,
        "raw_p_pb_lt_alpha": int((e["p_pb"] < alpha).sum()),
        "bh_flag_pb": int(res["flag"].sum()),
        "raw_p_boot_lt_alpha": int((e["p_boot"] < alpha).sum()),
        "bh_flag_boot": int(res["flag_boot"].sum()),
        "bh_flag_both": int(res["flag_both"].sum()),
        "expected_false_raw": alpha * e.height,
    }
