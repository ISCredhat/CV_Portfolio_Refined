"""Three-tier anomalous-wallet watchlist (statistical flags, not findings of wrongdoing).

Tiers (highest wins):
* ``repeat_anomalous``    : wallet passes the Benjamini-Hochberg FDR on the calibrated win test and is not
                            excluded by the non-directional filters.
* ``one_shot_anomalous``  : best winning position WITH A VALID CUTOFF has a one-shot score (proximity and
                            wallet-age-at-entry definitions, see oneshot.py) >= the ``ONE_SHOT_Q`` quantile
                            of all valid-cutoff scored winning positions of non-excluded wallets and
                            USD >= ``MIN_USD``. Positions without a valid cutoff never reach this tier.
* ``watch``               : not in the above, but valid-cutoff one-shot score >= the ``WATCH_Q`` quantile, or
                            (no valid cutoff) the proximity-free ``score_screen`` >= the ``WATCH_Q`` quantile
                            of such positions (column ``no_valid_cutoff`` is true), or an unadjusted win-test
                            p < ``WATCH_P``, or member of a co-trading cluster whose pooled test has
                            p < ``WATCH_P``.

Output (wallet-level, therefore only ever written under data/private/): ``data/private/watchlist.csv``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pmdetect.store.db import PROJECT_ROOT

ONE_SHOT_Q = 0.99
WATCH_Q = 0.95
MIN_USD = 100.0
WATCH_P = 0.01
PRIVATE_DIR = PROJECT_ROOT / "data" / "private"
TIERS = ("repeat_anomalous", "one_shot_anomalous", "watch")

P_CANDIDATES = ("p_pb", "p_value", "p_exact", "p_pb", "p_binom", "p_tail", "pval", "p")
Q_CANDIDATES = ("q_pb", "q_value", "q", "p_adj", "p_bh", "qval", "fdr_q")
PASS_CANDIDATES = ("flag", "fdr_pass", "passes_fdr", "reject", "significant", "fdr_reject", "pass_fdr")


def pick(df: pl.DataFrame, cands: tuple[str, ...]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _fmt(vals: list, nd: int = 3) -> str:
    return ";".join(f"{v:.{nd}f}" if isinstance(v, float) else str(v) for v in vals)


def build_watchlist(
    tests: pl.DataFrame,
    oneshot: pl.DataFrame,
    flags: pl.DataFrame,
    clusters: pl.DataFrame,
    cluster_tests: pl.DataFrame | None,
    positions: pl.DataFrame,
    markets: pl.DataFrame | None = None,
    out_path: Path | None = None,
    one_shot_q: float = ONE_SHOT_Q,
    watch_q: float = WATCH_Q,
    min_usd: float = MIN_USD,
) -> Path:
    """Write the watchlist CSV (data/private/watchlist.csv by default) and return its path."""
    out_path = out_path or PRIVATE_DIR / "watchlist.csv"
    wl = assign_tiers(tests, oneshot, flags, clusters, cluster_tests, positions, markets,
                      one_shot_q, watch_q, min_usd)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wl.write_csv(out_path)
    return out_path


def assign_tiers(
    tests: pl.DataFrame,
    oneshot: pl.DataFrame,
    flags: pl.DataFrame,
    clusters: pl.DataFrame,
    cluster_tests: pl.DataFrame | None,
    positions: pl.DataFrame,
    markets: pl.DataFrame | None = None,
    one_shot_q: float = ONE_SHOT_Q,
    watch_q: float = WATCH_Q,
    min_usd: float = MIN_USD,
) -> pl.DataFrame:
    ex_s = flags.filter(pl.col("excluded"))["wallet"]
    excluded = set(ex_s.to_list())
    ex_imp = ex_s.implode()
    pcol, qcol, fcol = pick(tests, P_CANDIDATES), pick(tests, Q_CANDIDATES), pick(tests, PASS_CANDIDATES)
    t = tests.filter(~pl.col("wallet").is_in(ex_imp))
    repeat = set(t.filter(pl.col(fcol).fill_null(False))["wallet"].to_list()) if fcol else set()
    nominal = (
        set(t.filter(pl.col(pcol) < WATCH_P)["wallet"].to_list()) if pcol else set()
    )
    os_ok = oneshot.filter(~pl.col("wallet").is_in(ex_imp))
    if "no_valid_cutoff" not in os_ok.columns:
        os_ok = os_ok.with_columns(pl.lit(False).alias("no_valid_cutoff"))
    if "score_screen" not in os_ok.columns:
        os_ok = os_ok.with_columns(pl.col("score").alias("score_screen"))
    os_nv = os_ok.filter(pl.col("no_valid_cutoff"))
    os_ok = os_ok.filter(~pl.col("no_valid_cutoff"))
    q_hi = os_ok["score"].quantile(one_shot_q) if os_ok.height else None
    q_wa = os_ok["score"].quantile(watch_q) if os_ok.height else None
    q_nv = os_nv["score_screen"].quantile(watch_q) if os_nv.height else None
    best_nv = os_nv.sort("score_screen", descending=True).unique("wallet", keep="first", maintain_order=True)
    watch_nv = (
        set(best_nv.filter(pl.col("score_screen") >= q_nv)["wallet"].to_list()) if q_nv is not None else set()
    )
    best = os_ok.sort("score", descending=True).unique("wallet", keep="first", maintain_order=True)
    one_shot = (
        set(best.filter((pl.col("score") >= q_hi) & (pl.col("usd") >= min_usd))["wallet"].to_list())
        if q_hi is not None
        else set()
    )
    watch_os = set(best.filter(pl.col("score") >= q_wa)["wallet"].to_list()) if q_wa is not None else set()
    clus_flag: set[str] = set()
    cl_p: dict[int, float] = {}
    if cluster_tests is not None and cluster_tests.height and clusters.height:
        cp = pick(cluster_tests, P_CANDIDATES)
        if cp:
            cl_p = dict(zip(cluster_tests["cluster_id"], cluster_tests[cp], strict=True))
            hot = {c for c, p in cl_p.items() if p is not None and p < WATCH_P}
            clus_flag = set(clusters.filter(pl.col("cluster_id").is_in(list(hot)))["wallet"].to_list())

    tier: dict[str, str] = {}
    for w in nominal | watch_os | clus_flag | watch_nv:
        tier[w] = "watch"
    for w in one_shot:
        tier[w] = "one_shot_anomalous"
    for w in repeat:
        tier[w] = "repeat_anomalous"
    tier = {w: t_ for w, t_ in tier.items() if w not in excluded}
    if not tier:
        return pl.DataFrame(schema={"tier": pl.String, "wallet": pl.String})

    names = (
        dict(zip(markets["condition_id"], markets["question"], strict=False))
        if markets is not None and "question" in markets.columns
        else {}
    )
    key = "event_id" if "event_id" in positions.columns else "condition_id"
    tier_imp = pl.Series("wallet", list(tier), dtype=pl.String).implode()
    cl_map = dict(zip(clusters["wallet"], clusters["cluster_id"], strict=True)) if clusters.height else {}
    tmap = {r["wallet"]: r for r in tests.filter(pl.col("wallet").is_in(tier_imp)).iter_rows(named=True)}
    best_map = {r["wallet"]: r for r in best_nv.filter(pl.col("wallet").is_in(tier_imp)).iter_rows(named=True)}
    best_map.update(  # valid-cutoff best wins
        {r["wallet"]: r for r in best.filter(pl.col("wallet").is_in(tier_imp)).iter_rows(named=True)}
    )
    rank_map = dict(
        best.with_row_index("_rk", offset=1)
        .filter(pl.col("wallet").is_in(tier_imp))
        .select("wallet", "_rk")
        .iter_rows()
    )
    cond_col = "condition_id" if "condition_id" in positions.columns else key
    pw = (
        positions.filter(pl.col("wallet").is_in(tier_imp))
        .group_by("wallet")
        .agg(
            pl.len().alias("n_pos"),
            pl.col("won").fill_null(False).sum().alias("n_wins"),
            pl.col("usd").sum().alias("usd_total"),
            pl.col(cond_col).head(5).alias("_c5"),
            pl.col(key).n_unique().alias("n_markets"),
            pl.col("entry_price").head(8).alias("_e8"),
        )
    )
    pos_by_w = {r["wallet"]: r for r in pw.iter_rows(named=True)}
    rows = []
    for w, tr in tier.items():
        g = pos_by_w.get(w)
        tt, bb = tmap.get(w, {}), best_map.get(w, {})
        row = {
            "tier": tr,
            "wallet": w,
            "n_pos": g["n_pos"] if g else None,
            "n_wins": int(g["n_wins"]) if g else None,
            "usd_total": float(g["usd_total"]) if g else None,
            "markets": ";".join(str(names.get(c, c))[:60] for c in (g["_c5"] if g else [])),
            "n_markets": g["n_markets"] if g else None,
            "entry_prices": _fmt(g["_e8"], 3) if g else "",
            "p_value": tt.get(pcol) if pcol else None,
            "q_value": tt.get(qcol) if qcol else None,
            "oneshot_rank": rank_map.get(w),
            "oneshot_score": bb.get("score") if bb and not bb.get("no_valid_cutoff") else bb.get("score_screen"),
            "best_usd": bb.get("usd"),
            "best_entry_price": bb.get("entry_price"),
            "best_hours_before_cutoff": bb.get("hours_before_cutoff"),
            "best_market": bb.get("question") or names.get(bb.get("condition_id"), ""),
            "no_valid_cutoff": bool(bb.get("no_valid_cutoff")) if bb else None,
            "best_wallet_age_days_at_entry": bb.get("wallet_age_days_at_entry") if bb else None,
            "first_seen": bb.get("first_seen") if bb else None,
            "only_market_store": bb.get("only_market_store") if bb else None,
            "n_markets_global": bb.get("n_markets_global") if bb else None,
            "cluster_id": cl_map.get(w),
            "cluster_p": cl_p.get(cl_map.get(w)) if w in cl_map else None,
        }
        rows.append(row)
    wl = pl.DataFrame(rows, infer_schema_length=None)
    order = {t_: i for i, t_ in enumerate(TIERS)}
    return wl.sort(
        [pl.col("tier").replace_strict(order), pl.col("oneshot_score").fill_null(-1.0)],
        descending=[False, True],
    )


# ---------------------------------------------------------------- aggregate-only summary + privacy guard
import re  # noqa: E402

ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")


def assert_no_addresses(text: str, where: str = "output") -> None:
    """Raise if a 0x + 40-hex wallet address appears (64-hex condition ids are not matched)."""
    if ADDRESS_RE.search(text):
        raise AssertionError(f"privacy violation: wallet address found in {where}")


def assert_results_clean(results_dir: Path) -> int:
    """Scan every text file under ``results_dir``; return files checked, raise on any address."""
    n = 0
    for f in results_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".md", ".csv", ".txt", ".json", ".log", ".svg"}:
            assert_no_addresses(f.read_text(encoding="utf-8", errors="ignore"), str(f.name))
            n += 1
    return n


def _dist(s: pl.Series) -> str:
    s = s.drop_nulls().drop_nans() if s.dtype.is_float() else s.drop_nulls()
    if s.len() == 0:
        return "n=0"
    q = [s.quantile(x) for x in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)]
    return "n={} min={:.3g} q25={:.3g} median={:.3g} q75={:.3g} q90={:.3g} q99={:.3g} max={:.3g}".format(
        s.len(), *q
    )


def write_phase3_summary(
    path: Path,
    *,
    n_positions: int,
    n_wallets_tested: int,
    exclusion: dict[str, int],
    rule_doc: dict[str, str],
    tests: pl.DataFrame,
    n_fdr_pass: int,
    oneshot: pl.DataFrame,
    watchlist: pl.DataFrame,
    clusters: pl.DataFrame,
    cluster_tests: pl.DataFrame | None,
    notes: list[str] | None = None,
) -> Path:
    """Write results/phase3_summary.md: counts and distributions only, never addresses or per-wallet rows."""
    pcol = pick(tests, P_CANDIDATES)
    L = [
        "# Phase 3 summary: anomalous wallet detector (aggregate only)",
        "",
        "These are statistical flags of anomalous trading, not findings of wrongdoing. "
        "No wallet addresses or per-wallet rows appear here; wallet-level output is in data/private/.",
        "",
        "## Coverage",
        f"- Positions built: {n_positions}",
        f"- Wallets tested: {n_wallets_tested}",
        f"- Wallets passing FDR (before non-directional exclusion): {n_fdr_pass}",
        "",
        "## Non-directional exclusions (rules, thresholds, counts)",
        f"- Wallets screened: {exclusion.get('wallets')}; excluded (any rule): {exclusion.get('excluded')}",
    ]
    for r, d in rule_doc.items():
        L.append(f"- `{r}`: {exclusion.get(r)} wallets. Rule: {d}")
    L += ["", "## Win-test p-value histogram (all tested wallets, 10 equal bins on [0,1])"]
    if pcol:
        p = tests[pcol].drop_nulls().drop_nans()
        h = [int(((p >= i / 10) & (p < (i + 1) / 10 if i < 9 else p <= 1)).sum()) for i in range(10)]
        L.append("- counts: " + ", ".join(map(str, h)))
        L.append(f"- p < 0.05: {int((p < 0.05).sum())}; p < 0.01: {int((p < 0.01).sum())} of {p.len()}")
    L += ["", "## One-shot scoring (winning pre-cutoff positions)",
          f"- Scored winning positions: {oneshot.height}",
          f"- Score distribution: {_dist(oneshot['score'])}" if oneshot.height else "- none",
          f"- Positions with >=1 imputed component: {int((oneshot['n_missing'] > 0).sum())}"
          if oneshot.height else "",
          ]
    if oneshot.height and "no_valid_cutoff" in oneshot.columns:
        nv = int(oneshot["no_valid_cutoff"].sum())
        L += [
            f"- Winning positions with no valid cutoff (no proximity credit, screened only): {nv}",
            f"- Winning positions with wallet first_seen missing: "
            f"{int(oneshot['wallet_age_days_at_entry'].is_null().sum())} of {oneshot.height}",
            "- Proximity = 1 for entries <= 72 h before the cutoff, exp(-(h-72)/72) beyond; freshness = "
            "wallet age at entry (see oneshot.py).",
        ]
    L += ["", "## Watchlist tier sizes"]
    for t in TIERS:
        L.append(f"- {t}: {int((watchlist['tier'] == t).sum()) if watchlist.height else 0}")
    L += ["", "## Co-trading clusters"]
    if clusters.height:
        sizes = clusters.group_by("cluster_id").agg(pl.len().alias("n"), pl.col("oversized").first())
        L += [
            f"- Clusters: {sizes.height}; wallets in clusters: {clusters.height}",
            f"- Size distribution: {_dist(sizes['n'])}",
            f"- Largest cluster size: {int(sizes['n'].max())}; oversized (> cap, not calibrated): "
            f"{int(sizes['oversized'].sum())}",
        ]
        cp = pick(cluster_tests, P_CANDIDATES) if cluster_tests is not None else None
        if cp:
            pv = cluster_tests[cp].drop_nulls()
            L.append(f"- Pooled-cluster calibration tests: {pv.len()}; p < 0.05: {int((pv < 0.05).sum())}")
    else:
        L.append("- No clusters found at the configured thresholds.")
    L += ["", "## Caveats",
          "- Power is limited by the number of calibration-eligible positions (independent news time "
          "only); most wallets have one or two positions, so the repeat test covers few wallets.",
          "- Positions without a valid cutoff and jump-start positions are screened by the one-shot "
          "score only (the jump start is outcome-selected).",
          "- Funding-chain linkage is a stub (`link_by_funding`); clusters use co-trading only.",
          *(notes or [])]
    text = "\n".join(x for x in L) + "\n"
    assert_no_addresses(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
