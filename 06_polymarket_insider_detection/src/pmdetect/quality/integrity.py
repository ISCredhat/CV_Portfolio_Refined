"""Data-integrity report (plan 1.5).

Per market whose tape is fully loaded (fetch_log 'trades' complete):

* summed trade USD vs Gamma ``volume_usd`` (``ratio``; reconciles when within TOLERANCE),
* the largest gap in the tape and whether it is a long interior gap,
* row-level sanity (non-positive size, price outside [0, 1], missing wallet, trades dated
  before market start or long after close),
* the at-least-20-trades rule (``keep_20``); optionally applied to ``markets.in_universe``.

Writes ``results/integrity_report.csv`` and ``results/integrity_summary.md``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pmdetect.collect.data_api import TOKENS_URL, connect
from pmdetect.collect.http import get_json
from pmdetect.store.db import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_CSV = RESULTS_DIR / "integrity_report.csv"
SUMMARY_MD = RESULTS_DIR / "integrity_summary.md"

MIN_TRADES = 20
TOLERANCE = 0.15  # |notional_usd / gamma_volume - 1| <= 15% counts as reconciling
LONG_GAP_HOURS = 24 * 7
LONG_GAP_SHARE = 0.5  # ... and more than half of the market's traded span

_SQL = """
WITH done AS (SELECT key AS condition_id FROM fetch_log WHERE endpoint = 'trades' AND completed),
t AS (
    SELECT condition_id, ts, usd, size, price, wallet,
           ts - LAG(ts) OVER (PARTITION BY condition_id ORDER BY ts) AS gap
    FROM trades
),
agg AS (
    SELECT condition_id,
           COUNT(*) AS n_trades,
           COUNT(DISTINCT wallet) AS n_wallets,
           SUM(usd) AS trade_usd,
           SUM(size) / 2.0 AS notional_usd,
           MIN(ts) AS first_trade_ts,
           MAX(ts) AS last_trade_ts,
           MAX(epoch(gap)) / 3600.0 AS max_gap_hours,
           SUM(CASE WHEN size <= 0 THEN 1 ELSE 0 END) AS n_bad_size,
           SUM(CASE WHEN price < 0 OR price > 1 THEN 1 ELSE 0 END) AS n_bad_price,
           SUM(CASE WHEN wallet IS NULL OR wallet = '' THEN 1 ELSE 0 END) AS n_no_wallet
    FROM t GROUP BY condition_id
)
SELECT d.condition_id, m.slug, m.in_universe, m.outcome, m.volume_usd AS gamma_volume_usd,
       m.start_ts, m.closed_ts,
       COALESCE(a.n_trades, 0) AS n_trades, COALESCE(a.n_wallets, 0) AS n_wallets,
       COALESCE(a.trade_usd, 0.0) AS trade_usd,
       COALESCE(a.notional_usd, 0.0) AS notional_usd,
       a.first_trade_ts, a.last_trade_ts, a.max_gap_hours,
       COALESCE(a.n_bad_size, 0) AS n_bad_size, COALESCE(a.n_bad_price, 0) AS n_bad_price,
       COALESCE(a.n_no_wallet, 0) AS n_no_wallet
FROM done d
LEFT JOIN agg a USING (condition_id)
LEFT JOIN markets m USING (condition_id)
ORDER BY d.condition_id
"""


def fetch_neg_risk(condition_ids: list[str]) -> dict[str, bool | None]:
    """neg_risk flag per condition from /v2/tokens (cached by get_json)."""
    out: dict[str, bool | None] = {}
    for i in range(0, len(condition_ids), 10):
        chunk = condition_ids[i : i + 10]
        for t in get_json(TOKENS_URL, {"condition": ",".join(chunk)}).get("data") or []:
            out[(t.get("condition_id") or "").lower()] = t.get("neg_risk")
    return out


def build_report(
    con, tolerance: float = TOLERANCE, neg_risk: dict[str, bool | None] | None = None
) -> pl.DataFrame:
    """Compute the per-market integrity table from the store."""
    df = pl.from_arrow(con.sql(_SQL).to_arrow_table())
    if df.height == 0:
        return df
    neg_risk = neg_risk or {}
    df = df.with_columns(
        pl.Series("neg_risk", [neg_risk.get(c) for c in df["condition_id"]], dtype=pl.Boolean)
    )
    span_h = (
        (pl.col("last_trade_ts") - pl.col("first_trade_ts")).dt.total_seconds() / 3600.0
    ).alias("span_hours")
    df = df.with_columns(span_h).with_columns(
        (pl.col("notional_usd") / pl.col("gamma_volume_usd")).alias("ratio"),
        (pl.col("trade_usd") / pl.col("gamma_volume_usd")).alias("cash_ratio"),
        (pl.col("n_trades") >= MIN_TRADES).alias("keep_20"),
        (
            (pl.col("max_gap_hours") > LONG_GAP_HOURS)
            & (pl.col("max_gap_hours") > LONG_GAP_SHARE * pl.col("span_hours"))
        )
        .fill_null(False)
        .alias("long_gap"),
        (pl.col("first_trade_ts") < pl.col("start_ts") - pl.duration(days=1))
        .fill_null(False)
        .alias("trades_before_start"),
        (pl.col("last_trade_ts") > pl.col("closed_ts") + pl.duration(days=1))
        .fill_null(False)
        .alias("trades_after_close"),
    )
    df = df.with_columns(
        (pl.col("ratio").is_not_null() & ((pl.col("ratio") - 1).abs() <= tolerance)).alias(
            "reconciles"
        ),
        ((pl.col("n_bad_size") + pl.col("n_bad_price") + pl.col("n_no_wallet")) > 0).alias(
            "row_issues"
        ),
    )
    return df.with_columns(
        pl.when(pl.col("n_trades") == 0)
        .then(pl.lit("no_trades"))
        .when(pl.col("gamma_volume_usd").is_null())
        .then(pl.lit("no_gamma_volume"))
        .when(pl.col("reconciles"))
        .then(pl.lit("ok"))
        .when((pl.col("ratio") < 1 - tolerance) & pl.col("neg_risk").fill_null(False))
        .then(pl.lit("tape_below_gamma_negrisk"))
        .when(pl.col("ratio") < 1 - tolerance)
        .then(pl.lit("tape_below_gamma"))
        .otherwise(pl.lit("tape_above_gamma"))
        .alias("status")
    )


def _neg_risk_lines(with_vol: pl.DataFrame) -> list[str]:
    out = []
    for label, flag in (("non-neg-risk", False), ("neg-risk", True)):
        g = with_vol.filter(pl.col("neg_risk") == flag)
        if g.height:
            k = g.filter(pl.col("reconciles")).height
            out.append(
                f"  - {label}: {k} of {g.height} ({k / g.height:.1%}); "
                f"median ratio {g['ratio'].median():.2f}"
            )
    return out


def summarize(df: pl.DataFrame, tolerance: float = TOLERANCE, rule_applied: bool = False) -> str:
    n = df.height
    lines = ["# Integrity report", ""]
    if n == 0:
        lines.append("No markets with a completed trade tape yet.")
        return "\n".join(lines) + "\n"
    kept = df.filter(pl.col("keep_20"))
    with_vol = kept.filter(pl.col("gamma_volume_usd").is_not_null())
    n_rec = with_vol.filter(pl.col("reconciles")).height
    share = n_rec / with_vol.height if with_vol.height else float("nan")
    lines += [
        f"- Markets with a completed trade tape: **{n}**",
        f"- Dropped by the at-least-{MIN_TRADES}-trades rule: **{n - kept.height}**"
        + (" (in_universe set to false)" if rule_applied else " (report only)"),
        f"- Markets kept: **{kept.height}**; trades kept: **{int(kept['n_trades'].sum()):,}**; "
        f"distinct wallet-market pairs: {int(kept['n_wallets'].sum()):,}",
        f"- Reconciling with Gamma volume (|tape notional / Gamma volume - 1| <= {tolerance:.0%}), "
        f"among kept markets with a Gamma volume: **{n_rec} of {with_vol.height} "
        f"({share:.1%})**",
        *_neg_risk_lines(with_vol),
        f"- Markets with a long interior gap in the tape (> {LONG_GAP_HOURS // 24} days and "
        f"> {LONG_GAP_SHARE:.0%} of the span): {kept.filter(pl.col('long_gap')).height}",
        f"- Markets with row-level issues (bad size/price/wallet): "
        f"{kept.filter(pl.col('row_issues')).height}",
        "",
        "## Status of kept markets",
        "",
        "| status | markets |",
        "| --- | ---: |",
    ]
    for status, cnt in kept.group_by("status").len().sort("len", descending=True).iter_rows():
        lines.append(f"| {status} | {cnt} |")
    r = with_vol.filter(pl.col("ratio").is_not_null())["ratio"]
    if r.len():
        q = r.quantile
        lines += [
            "",
            "## Tape notional / Gamma volume (kept markets)",
            "",
            f"- min {r.min():.2f}, p10 {q(0.1):.2f}, median {q(0.5):.2f}, "
            f"p90 {q(0.9):.2f}, max {r.max():.2f}",
            "- Gamma `volumeNum` is the *share-notional* volume: the sum of `size` (shares, "
            "each worth $1 at resolution) over every tape row divided by 2, because "
            "`taker_only=false` lists each fill twice (taker row and maker row). So "
            "`notional_usd = SUM(size) / 2` reconciles with Gamma, for binary and neg-risk "
            "markets alike. The cash volume `SUM(size * price)` (`trade_usd`, `cash_ratio`) is "
            "not what Gamma reports and is not used for reconciliation; a large gap between the "
            "two just reflects the typical price of the traded side.",
        ]
    dropped = df.filter(~pl.col("keep_20"))
    if dropped.height:
        lines += ["", f"## Dropped by the {MIN_TRADES}-trade rule", ""]
        for row in dropped.head(50).iter_rows(named=True):
            lines.append(f"- `{row['slug'] or row['condition_id']}`: {row['n_trades']} trades")
        if dropped.height > 50:
            lines.append(f"- ... and {dropped.height - 50} more (see the CSV)")
    return "\n".join(lines) + "\n"


def run_integrity_report(
    apply_rule: bool = True, tolerance: float = TOLERANCE, with_neg_risk: bool = True
) -> Path:
    """Write results/integrity_report.csv and integrity_summary.md; returns the CSV path.

    With apply_rule, markets with fewer than 20 trades get in_universe = false.
    """
    con = connect()
    try:
        nr = None
        if with_neg_risk:
            done = [
                r[0]
                for r in con.execute(
                    "SELECT key FROM fetch_log WHERE endpoint = 'trades' AND completed"
                ).fetchall()
            ]
            nr = fetch_neg_risk(done)
        df = build_report(con, tolerance, nr)
        if apply_rule and df.height:
            drop = df.filter(~pl.col("keep_20"))["condition_id"].to_list()
            if drop:
                con.execute(
                    "UPDATE markets SET in_universe = FALSE "
                    "WHERE condition_id IN (SELECT UNNEST(?::VARCHAR[]))",
                    [drop],
                )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.write_csv(REPORT_CSV)
        SUMMARY_MD.write_text(summarize(df, tolerance, apply_rule), encoding="utf-8")
        return REPORT_CSV
    finally:
        con.close()
