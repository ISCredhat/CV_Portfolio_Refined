"""Phase 4D: where the labelled cases land on every score (HELD OUT: nothing was ever tuned on them).

One row per labelled market of ``results/market_scores.csv`` (75) plus the two labelled wallet classes that have a
known wallet, referred to ONLY as ``maduro_labelled_wallet`` and ``machado_jump_wallet`` (the addresses stay in
``data/private/validation_labelled_wallets.csv``). For every row:

* within-event: the winner's 24 h / 12 h within-event percentile and the winner-minus-median-loser difference, each
  with its rank and percentile among the markets that have the statistic (rank 1 = most extreme);
* one-shot: the best (highest-score) real winning position of the market among all valid-cutoff positions of
  non-excluded wallets (the pool behind the tier threshold), its rank and percentile in that pool, and whether any
  position of the market is in the tier (score >= the pool's 99th percentile and USD >= 100);
* the price measures that were not packaged as detectors (level D, late-window D24, max_jump, flow before the move,
  Fisher-combined score), with rank/N from ``market_scores.csv`` against the universe pool;
* a plain verdict per detector: FLAGS, MISSES or CANNOT_TEST, with the reason.

"Flags" uses the pre-registered definitions: within-event winner percentile >= 0.95 (24 h window); one-shot tier
membership with a scheduled (independent) cutoff; a price measure is flagged in the top 1% of its own pool.
Nothing here is evidence of a leak: the within-event test failed its own criterion (false-positive rate 11.7%, both
placebos fail) and the one-shot screen reaches no >= 50% power cell, with flipped-label losers entering the tier more
than twice as often as real winners.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from pmdetect.store.db import PROJECT_ROOT
from pmdetect.watchlist import MIN_USD, ONE_SHOT_Q, assert_no_addresses

SNAP = PROJECT_ROOT / "data" / "snapshots"
PRIVATE = PROJECT_ROOT / "data" / "private"
RESULTS = PROJECT_ROOT / "results"

WE_DETECT = 0.95
HELD_OUT = "HELD OUT (never tuned on)"
MADURO_ALIAS = "maduro_labelled_wallet"
MACHADO_ALIAS = "machado_jump_wallet"
UNTESTABLE_CASES = {
    "axiom_expose_feb_2026": "Axiom / ZachXBT candidates: markets are short-lived (short history), mapped to the "
    "labelled event only as candidates (no wallet or exact market is published), and the news time is price-based, "
    "so there is no independent time",
    "bubblemaps_military_cluster": "Bubblemaps military-cluster candidates: the mapping from the reported accounts to "
    "these markets is uncertain (candidates only), short or price-based histories, and no independent news time",
}
PRICE_MEASURES = (  # label, value column, rank column, universe-pool column for N
    ("D", "D_timesplit", "rank_D_timesplit"),
    ("D24", "D_last24", "rank_D_last24"),
    ("max_jump", "max_jump", "rank_max_jump"),
    ("flow", "flow_score", "rank_flow_score"),
    ("combined", "fisher_score", "rank_fisher_score"),
)
WE_STATS = (("winner_pct_24h", "we_winner_pct_24h"), ("winner_pct_12h", "we_winner_pct_12h"),
            ("diff_24h", "we_diff_24h"), ("diff_12h", "we_diff_12h"))  # fmt: skip


# ----------------------------------------------------------------------------------------------- helpers


def pool_rank(pool: np.ndarray, x: float) -> tuple[float, float]:
    """(rank, percentile) of ``x`` in ``pool``: rank 1 = highest value (ties share the best rank), percentile =
    100 * (N - rank + 1) / N so that 100 is the most extreme. NaN for a missing value."""
    if x is None or not np.isfinite(x) or len(pool) == 0:
        return float("nan"), float("nan")
    rank = 1.0 + float((pool > x).sum())
    n = float(len(pool))
    return rank, 100.0 * (n - rank + 1.0) / n


def _f(x, nd: int = 3) -> str:
    return "NA" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _rn(rank, n) -> str:
    return "NA" if rank is None or not (isinstance(rank, float | int) and math.isfinite(rank)) else f"{int(rank)}/{int(n)}"


def load_scores(results: Path = RESULTS) -> pd.DataFrame:
    return pd.read_csv(results / "market_scores.csv", low_memory=False)


def oneshot_pool(private: Path = PRIVATE) -> tuple[pl.DataFrame, float]:
    """Valid-cutoff winning positions of non-excluded wallets (the tier pool) and the 99th-percentile score."""
    sc = pl.read_parquet(private / "oneshot_scores.parquet")
    flags = pl.read_parquet(private / "wallet_flags.parquet", columns=["wallet", "excluded"])
    ok = sc.join(flags, on="wallet", how="left").filter(~pl.col("excluded").fill_null(False) & ~pl.col("no_valid_cutoff"))
    return ok, float(ok["score"].quantile(ONE_SHOT_Q))


# --------------------------------------------------------------------------------------------- verdicts


def verdict_within_event(r: dict) -> tuple[str, str]:
    p = r.get("we_winner_pct_24h")
    if p is not None and np.isfinite(p):
        if p >= WE_DETECT:
            return "FLAGS", f"24 h winner percentile {p:.3f} >= {WE_DETECT} (the test's false-positive rate on untouched events is 11.7%)"  # fmt: skip
        return "MISSES", f"24 h winner percentile {p:.3f} < {WE_DETECT}"
    if r.get("out_of_universe"):
        return "CANNOT_TEST", "out-of-universe market: no sibling-outcome price data"
    if not r.get("independent"):
        return "CANNOT_TEST", "no independent reveal time (the pre-reveal window needs one)"
    why = r.get("we_reason")
    if isinstance(why, str) and why:
        return "CANNOT_TEST", "within-event statistic not computed: " + why.replace("_", " ")
    return "CANNOT_TEST", "market lost its event or has no comparable siblings (the statistic scores event winners only)"


def verdict_oneshot(case: str, mapping: str | None, best: dict | None, tier_idx: int, tier_jump: int) -> tuple[str, str]:
    if case in UNTESTABLE_CASES:
        return "CANNOT_TEST", UNTESTABLE_CASES[case].split(":")[0] + " (uninterpretable here)"
    if best is None:
        return "CANNOT_TEST", "no valid-cutoff winning position (no independent news time and no price jump start)"
    if tier_idx > 0:
        return "FLAGS", f"{tier_idx} position(s) in the tier with a scheduled (independent) cutoff"
    if tier_jump > 0:
        return "CANNOT_TEST", (f"{tier_jump} tier position(s) exist only with a jump-start cutoff (chosen from the "
                               "price move itself: outcome-selected), so the tier is not interpretable")  # fmt: skip
    return "MISSES", f"best position not in the tier (cutoff {best['cutoff_method']})"


def verdict_price(r: dict, case: str, flagged: list[str], any_value: bool) -> tuple[str, str]:
    if case == "maduro_removal_jan_2026":
        return "CANNOT_TEST", (
            "out-of-universe market ranked against the universe pool; D = 0 and max_jump 0.024 are not informative: "
            "the news time is a documented press time (about 06:00 UTC), the recorded price move starts at 03:03 UTC "
            "and the tape starts 2025-12-22, so the 'pre-news' path is ill-defined"
        )
    if not any_value:
        why = "no independent time (level D needs one)" if not r.get("independent") else "price measures not computed"
        js = r.get("jump_reason")
        if isinstance(js, str) and js == "short_history":
            why += "; short history"
        return "CANNOT_TEST", why
    if flagged:
        return "FLAGS", "top 1% of its pool on " + ", ".join(flagged) + " (descriptive: not a packaged detector)"
    return "MISSES", "not in the top 1% on any computable price measure (not packaged detectors)"


# ------------------------------------------------------------------------------------------------- build


def build_cases(
    results: Path = RESULTS, snap: Path = SNAP, private: Path = PRIVATE, log: Callable[[str], None] = print
) -> dict:
    t0 = time.time()
    ms = load_scores(results)
    uni = ms[~ms["out_of_universe"].astype(bool)]
    lab = ms[ms["case_name"].notna()].copy()
    pool_we = {k: uni[c].dropna().to_numpy(float) for k, c in WE_STATS}
    pool_pm = {name: (uni[col].dropna().shape[0]) for name, col, _ in PRICE_MEASURES}
    ok, q99 = oneshot_pool(private)
    pool_scores = ok["score"].to_numpy()
    n_pool = len(pool_scores)
    log(f"[{time.time() - t0:.0f}s] labelled markets {len(lab)}; one-shot pool {n_pool}, q99 {q99:.4f}")

    by_cid = {k[0]: g for k, g in ok.group_by("condition_id")}
    rows: list[dict] = []
    for _, m in lab.iterrows():
        r = m.to_dict()
        case = r["case_name"]
        row = {
            "case_name": case, "case_label": r["case_label"], "row_type": "market", "alias": "",
            "market": r["question"], "market_key": str(r["condition_id"])[:10],
            "in_universe": not bool(r["out_of_universe"]), "independent_time": bool(r["independent"]),
            "mapping": r["mapping"] if isinstance(r["mapping"], str) else "", "held_out": HELD_OUT,
            "outcome": r["outcome"],
        }  # fmt: skip
        row["we_n_comparable"] = r.get("we_n_comparable")
        for key, col in WE_STATS:
            rank, pct = pool_rank(pool_we[key], r.get(col))
            row.update({f"we_{key}": r.get(col), f"we_{key}_rank": rank, f"we_{key}_n": len(pool_we[key]),
                        f"we_{key}_percentile": pct})  # fmt: skip
        g = by_cid.get(r["condition_id"])
        best = None
        tier_idx = tier_jump = 0
        if g is not None and g.height:
            g2 = g.sort("score", descending=True)
            b = g2.row(0, named=True)
            rank, pct = pool_rank(pool_scores, b["score"])
            best = {"score": b["score"], "rank": rank, "pct": pct, "usd": b["usd"], "entry": b["entry_price"],
                    "cutoff_method": b["cutoff_method"], "hours": b["hours_before_cutoff"]}  # fmt: skip
            t = g2.filter((pl.col("score") >= q99) & (pl.col("usd") >= MIN_USD))
            tier_idx = int((t["cutoff_method"] == "independent_news").sum())
            tier_jump = int((t["cutoff_method"] != "independent_news").sum())
            row.update(os_n_positions=g.height, os_n_tier=t.height)
        else:
            row.update(os_n_positions=0, os_n_tier=0)
        row.update(
            os_best_score=best["score"] if best else float("nan"), os_best_rank=best["rank"] if best else float("nan"),
            os_pool_n=n_pool, os_best_percentile=best["pct"] if best else float("nan"),
            os_best_usd=best["usd"] if best else float("nan"), os_best_entry_price=best["entry"] if best else float("nan"),
            os_cutoff_method=best["cutoff_method"] if best else "", os_tier=("yes" if (tier_idx + tier_jump) else "no") if best else "NA",
        )  # fmt: skip
        flagged = []
        any_value = False
        for name, col, rcol in PRICE_MEASURES:
            v, rk, n = r.get(col), r.get(rcol), pool_pm[name]
            has = v is not None and np.isfinite(v) and rk is not None and np.isfinite(rk)
            any_value |= bool(has)
            top = bool(has and rk <= math.ceil(0.01 * n))
            if top:
                flagged.append(name)
            row.update({f"{name}_value": v if has else float("nan"), f"{name}_rank": rk if has else float("nan"),
                        f"{name}_n": n, f"{name}_percentile": (100.0 * (n - rk + 1) / n) if has else float("nan"),
                        f"{name}_top1pct": ("yes" if top else "no") if has else "NA"})  # fmt: skip
        r["we_winner_pct_24h"] = r.get("we_winner_pct_24h")
        v_we = verdict_within_event(r)
        v_os = verdict_oneshot(case, row["mapping"], best, tier_idx, tier_jump)
        v_pm = verdict_price(r, case, flagged, any_value)
        if case in UNTESTABLE_CASES:
            v_we = ("CANNOT_TEST", v_we[1] + "; " + UNTESTABLE_CASES[case].split(":")[0])
            if v_pm[0] != "CANNOT_TEST":
                v_pm = ("CANNOT_TEST", "values shown for information only; " + UNTESTABLE_CASES[case].split(":")[0])
        row.update(verdict_within_event=v_we[0], why_within_event=v_we[1], verdict_oneshot=v_os[0], why_oneshot=v_os[1],
                   verdict_price=v_pm[0], why_price=v_pm[1])  # fmt: skip
        rows.append(row)

    # ---- the two labelled wallet classes (addresses stay in data/private)
    labels = pl.read_parquet(snap / "labels.parquet")
    mad = labels.filter((pl.col("case_name") == "maduro_removal_jan_2026") & pl.col("wallet").is_not_null())
    ji = pl.read_csv(private / "jump_initiators.csv", infer_schema_length=0)
    mac = ji.filter((pl.col("case_name") == "nobel_peace_2025") & (pl.col("is_labelled").str.to_lowercase() == "true")
                     & (pl.col("rank").cast(pl.Float64, strict=False) == 1.0))
    wallets = {MADURO_ALIAS: mad["wallet"][0] if mad.height else None, MACHADO_ALIAS: mac["wallet"][0] if mac.height else None}
    wl = pl.read_csv(private / "watchlist.csv", infer_schema_length=0) if (private / "watchlist.csv").exists() else None
    private_rows = []
    for alias, w in wallets.items():
        case = "maduro_removal_jan_2026" if alias == MADURO_ALIAS else "nobel_peace_2025"
        mrow = next(x for x in rows if x["case_name"] == case)
        base = {k: ("" if isinstance(v, str) else float("nan")) for k, v in mrow.items()}
        base.update(case_name=case, case_label="insider_suspected", row_type="wallet", alias=alias,
                    market=mrow["market"], market_key=mrow["market_key"], in_universe=mrow["in_universe"],
                    independent_time=mrow["independent_time"], held_out=HELD_OUT, os_pool_n=n_pool)  # fmt: skip
        for key, _ in WE_STATS:
            base[f"we_{key}_n"] = mrow[f"we_{key}_n"]
        for name, _, _ in PRICE_MEASURES:
            base[f"{name}_n"] = mrow[f"{name}_n"]
            base[f"{name}_top1pct"] = "NA"
        base["os_tier"] = "NA"
        gw = ok.filter(pl.col("wallet") == w) if w else ok.head(0)
        tier_w = "NA"
        if gw.height:
            b = gw.sort("score", descending=True).row(0, named=True)
            rank, pct = pool_rank(pool_scores, b["score"])
            in_tier = b["score"] >= q99 and b["usd"] >= MIN_USD
            tier_w = "yes" if in_tier else "no"
            base.update(os_n_positions=gw.height, os_best_score=b["score"], os_best_rank=rank, os_best_percentile=pct,
                        os_best_usd=b["usd"], os_best_entry_price=b["entry_price"], os_cutoff_method=b["cutoff_method"],
                        os_tier=tier_w, os_n_tier=int(in_tier))  # fmt: skip
            v_os = ("FLAGS", f"in the tier (rank {int(rank)} of {n_pool}, percentile {pct:.2f}; cutoff {b['cutoff_method']})") if (in_tier and b["cutoff_method"] == "independent_news") else (("CANNOT_TEST", "in the tier but the cutoff is a price jump start (outcome-selected)") if in_tier else ("MISSES", f"not in the tier (rank {int(rank)} of {n_pool})"))  # fmt: skip
        else:
            v_os = ("CANNOT_TEST", "wallet has no valid-cutoff winning position in the scored pool")
        wl_tier = ""
        if wl is not None and w:
            x = wl.filter(pl.col("wallet") == w)
            wl_tier = x["tier"][0] if x.height else "not on watchlist"
        base.update(
            verdict_within_event="CANNOT_TEST", why_within_event="wallet-level class: the within-event test scores markets, not wallets",
            verdict_oneshot=v_os[0], why_oneshot=v_os[1] + (f"; watchlist tier: {wl_tier}" if wl_tier else ""),
            verdict_price="CANNOT_TEST", why_price="wallet-level class: price measures score markets, not wallets",
        )  # fmt: skip
        rows.append(base)
        private_rows.append({"alias": alias, "wallet": w, "case_name": case, "os_best_rank": base.get("os_best_rank"),
                             "os_best_percentile": base.get("os_best_percentile"), "os_tier": tier_w, "watchlist_tier": wl_tier})  # fmt: skip
    tab = pd.DataFrame(rows)

    # ---- context numbers for the summary
    mad_cid = lab[lab["case_name"] == "maduro_removal_jan_2026"]["condition_id"].tolist()
    tier_all = ok.filter((pl.col("score") >= q99) & (pl.col("usd") >= MIN_USD))
    n_tier_mad = int(tier_all.filter(pl.col("condition_id").is_in(mad_cid)).height) if mad_cid else 0
    ctx = {"n_pool": n_pool, "q99": q99, "n_tier": tier_all.height, "n_tier_maduro": n_tier_mad,
           "top10_maduro": int(ok.sort("score", descending=True).head(10).filter(pl.col("condition_id").is_in(mad_cid)).height) if mad_cid else 0}  # fmt: skip
    return {"table": tab, "private": pd.DataFrame(private_rows), "ctx": ctx}


# --------------------------------------------------------------------------------------------- reporting


def _cell_we(r, key: str, nd: int = 3) -> str:
    if r["row_type"] == "wallet" or not np.isfinite(r.get(f"we_{key}", float("nan")) if r.get(f"we_{key}") is not None else float("nan")):
        return "NA"
    return f"{_f(r[f'we_{key}'], nd)} ({_rn(r[f'we_{key}_rank'], r[f'we_{key}_n'])}; {r[f'we_{key}_percentile']:.0f}%)"


def _cell_os(r) -> str:
    if not np.isfinite(r["os_best_rank"]):
        return "NA"
    return f"{_rn(r['os_best_rank'], r['os_pool_n'])}; {r['os_best_percentile']:.2f}%; tier {r['os_tier']}"


def _cell_pm(r, name: str) -> str:
    v = r.get(f"{name}_value")
    if v is None or not np.isfinite(v):
        return "NA"
    return f"{_f(v)} ({_rn(r[f'{name}_rank'], r[f'{name}_n'])})" + (" *" if r[f"{name}_top1pct"] == "yes" else "")


def render_md(res: dict) -> str:
    tab: pd.DataFrame = res["table"]
    ctx = res["ctx"]
    mk = tab[tab["row_type"] == "market"]
    L = ["# Labelled cases on every score (phase 4D): HELD OUT, never tuned on", ""]
    L += [
        "Aggregate/market-level only: no wallet addresses; the two labelled wallets are referred to as "
        f"`{MADURO_ALIAS}` and `{MACHADO_ALIAS}` (addresses stay in `data/private/`). Full numeric table: "
        "`results/validation_labelled_cases.csv`. **Every number below is HELD OUT: none of these cases set a "
        "threshold, a baseline or a rule; they are reported after the fact.**",
        "",
        "## How to read it",
        "",
        "* Rank 1 is the most extreme. 'Percentile' is 100 x (N - rank + 1) / N within the pool (100 = most extreme).",
        f"* Within-event: pool = the universe markets that have the statistic in market_scores.csv (event winners, multi and binary groups; wider than the 809-event validation population). "
        f"Flags if the 24 h winner percentile is >= {WE_DETECT} (the validation's detection rule). The validation "
        "found a 11.7% false-positive rate for exactly this rule on untouched events, and both placebos fail.",
        f"* One-shot: best position of the market among {ctx['n_pool']} real valid-cutoff winning positions of "
        f"non-excluded wallets; tier = score >= {ctx['q99']:.4f} (99th percentile) and USD >= {MIN_USD:.0f}. Flags only "
        "when a tier position has a scheduled (independent) cutoff; a jump-start cutoff is chosen from the price move itself "
        "(outcome-selected), so such a tier membership is shown but not interpreted.",
        "* Price measures (level D, late-window D24, max_jump, flow before the move, Fisher-combined): rank/N in the "
        "universe pool; `*` = top 1% of that pool. They are documented negative or untested results, not packaged "
        "detectors, so 'flags' there is descriptive.",
        "* CANNOT_TEST means the measure does not exist for that case or cannot be interpreted; the reason is in the "
        "CSV (`why_*` columns) and summarised below.",
        "",
    ]
    # ---- counts
    def cnt(col):
        v = mk[col].value_counts()
        return ", ".join(f"{k} {int(v.get(k, 0))}" for k in ("FLAGS", "MISSES", "CANNOT_TEST"))
    L += ["## Verdict counts over the 75 labelled markets", "",
          f"* Within-event: {cnt('verdict_within_event')}",
          f"* One-shot: {cnt('verdict_oneshot')}",
          f"* Price measures (descriptive): {cnt('verdict_price')}", ""]  # fmt: skip
    # ---- summary
    nob = mk[mk["case_name"] == "nobel_peace_2025"].iloc[0]
    mad = mk[mk["case_name"] == "maduro_removal_jan_2026"].iloc[0]
    wmad = tab[tab["alias"] == MADURO_ALIAS].iloc[0]
    wmac = tab[tab["alias"] == MACHADO_ALIAS].iloc[0]
    L += [
        "## Summary (read with the validation results)",
        "",
        "* **Nothing here is evidence of a leak.** The labelled insider-suspected cases are two markets and two wallet "
        "classes; the context cases (Google Year in Search, Axiom/ZachXBT, Bubblemaps military cluster) have no "
        "confirmed informed trader at all, so a flag or a miss on them carries no information about the detectors "
        "beyond their base rates.",
        f"* **Nobel Peace Prize 2025 (insider-suspected market)**: within-event 24 h winner percentile "
        f"{_f(nob['we_winner_pct_24h'])} (rank {_rn(nob['we_winner_pct_24h_rank'], nob['we_winner_pct_24h_n'])}) -> "
        f"{nob['verdict_within_event']}. That is what the validation says to expect of a winner that was already high in the "
        "pre-reveal ranking (convergence): 11.7% of untouched events do the same, and the 48-hour-earlier placebo "
        "fails. The level-D and combined scores do not put it in the top 1% "
        f"(D {_rn(nob['D_rank'], nob['D_n'])}, combined {_rn(nob['combined_rank'], nob['combined_n'])}). The hand audit "
        "classifies the move as unclear (the buying was balanced across many wallets).",
        f"* **Maduro (insider-suspected market)**: one-shot {mad['verdict_oneshot']} (best position rank "
        f"{_rn(mad['os_best_rank'], mad['os_pool_n'])}); within-event and the price measures cannot be tested "
        "(out-of-universe, no sibling data; price measures uninformative, see the CSV). The Maduro market holds "
        f"{ctx['n_tier_maduro']} of the {ctx['n_tier']} tier positions and {ctx['top10_maduro']} of the 10 highest "
        "scores. The label-flip placebo shows flipped losers entering the tier 0.74% against 0.35% for real winners "
        "(more than twice as often) while passing the 2% rule as written, so tier membership by itself does not "
        "separate informed from uninformed positions.",
        f"* **{MADURO_ALIAS}**: {wmad['verdict_oneshot']} ({wmad['why_oneshot']}). "
        f"**{MACHADO_ALIAS}**: {wmac['verdict_oneshot']} ({wmac['why_oneshot']}). A fresh wallet buying a "
        "low-priced winner shortly before the news is exactly what the score rewards. Both are in the tier with a "
        f"scheduled cutoff, but the tier holds {ctx['n_tier']} real positions and flipped-label losers enter it more "
        "than twice as often as real winners, so this is a consistency check on two publicly reported wallets, not "
        "a validation, and it cannot show that the screen separates informed from lucky buyers.",
        "* **Context cases**: Axiom and Bubblemaps candidates are untestable (short history, uncertain mapping, no "
        "independent time); the Google Year in Search markets have no independent time and mostly no computable "
        "measure. Where a number is shown for them it is informational. The one-shot tier does contain positions in "
        "the Iran-strike markets, but with jump-start cutoffs only, which is outcome-selected.",
        "* **Compatibility**: among the informative labelled rows the within-event test flags the Nobel winner "
        "market (a winner that already ranked at the top before the reveal, which the placebos show is common) and "
        "the one-shot tier contains the Maduro market, the Nobel market and the two named wallets. All of this is "
        "compatible with convergence plus the score rewarding low-priced, fresh, late winners; it is not evidence "
        "of leaks. Two insider-suspected markets cannot establish detection power, and the validation already shows "
        "that neither detector meets its pre-registered criterion.",
        "",
        "## Per-case table (HELD OUT)",
        "",
        "Cells: value (rank/N; percentile). Within-event columns are the winner's percentile in its event and the "
        "log-odds difference to the median loser. Order: insider-suspected first, then candidates, then context.",
        "",
        "| case | label | row | market | WE pct24 | WE pct12 | WE diff24 | one-shot best | D | D24 | max_jump | flow | combined | WE | one-shot | price |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    order = {"insider_suspected": 0, "case_candidate": 1, "case_context": 2}
    t = tab.copy()
    t["_o"] = t["case_label"].map(lambda s: min(order.get(x.strip(), 3) for x in str(s).split(",")))
    t = t.sort_values(["_o", "case_name", "row_type", "market"], kind="stable")
    for _, r in t.iterrows():
        who = r["alias"] if r["row_type"] == "wallet" else "market"
        q = str(r["market"]).replace("|", "/")[:70]
        L.append(
            f"| {r['case_name']} | {r['case_label']} | {who} | {q} | {_cell_we(r, 'winner_pct_24h')} | "
            f"{_cell_we(r, 'winner_pct_12h')} | {_cell_we(r, 'diff_24h', 2)} | {_cell_os(r)} | "
            + " | ".join(_cell_pm(r, n) for n, _, _ in PRICE_MEASURES)
            + f" | {r['verdict_within_event']} | {r['verdict_oneshot']} | {r['verdict_price']} |"
        )
    L += ["", "## Reasons, per case", ""]
    for (case, _), g in t.groupby(["case_name", "_o"], sort=False):
        L.append(f"### {case}")
        L.append("")
        for col, title in (("why_within_event", "Within-event"), ("why_oneshot", "One-shot"), ("why_price", "Price measures")):
            reasons = g[col].astype(str).str.replace(r"\s+", " ", regex=True).value_counts()
            L.append(f"* {title}: " + " | ".join(f"{k} (x{v})" if v > 1 else k for k, v in reasons.items()))
        L.append("")
    L += [
        "## Caveats",
        "",
        "* Two insider-suspected markets and two wallet classes cannot calibrate or validate anything; nothing was tuned on them.",
        "* Maduro and the context cases rely on press-documented times (hour precision) rather than a feed.",
        "* Ranks of out-of-universe markets are computed against the universe pool, not inside it.",
        "* A `NA` is a missing measure, not a zero; CANNOT_TEST is not a miss.",
        "",
    ]
    return "\n".join(L)


def run(results: Path = RESULTS, snap: Path = SNAP, private: Path = PRIVATE, log: Callable[[str], None] = print) -> dict:
    res = build_cases(results, snap, private, log)
    tab = res["table"]
    text = render_md(res)
    assert_no_addresses(text, "validation_labelled_cases.md")
    # LF only: write_text() would turn CRLF into CR CR LF on Windows
    csv_text = tab.to_csv(index=False, lineterminator="\n")
    assert_no_addresses(csv_text, "validation_labelled_cases.csv")
    results.mkdir(parents=True, exist_ok=True)
    (results / "validation_labelled_cases.md").write_text(text, encoding="utf-8")
    with (results / "validation_labelled_cases.csv").open("w", encoding="utf-8", newline="") as fh:
        fh.write(csv_text)
    private.mkdir(parents=True, exist_ok=True)
    res["private"].to_csv(private / "validation_labelled_wallets.csv", index=False, lineterminator="\n")
    log(f"wrote validation_labelled_cases.md/.csv ({len(tab)} rows)")
    return res
