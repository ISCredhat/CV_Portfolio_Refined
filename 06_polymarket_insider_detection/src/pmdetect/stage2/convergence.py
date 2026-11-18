"""Stage 2, strategy 2: tradable convergence backtest (frozen definition: docs/STAGE2_PLAN.md s.9 A).

For mutually exclusive (neg-risk) multi-outcome events with an independent scheduled news time T
(``news_independent.parquet``) and at least 3 comparable outcomes, at T-7d / T-72h / T-48h the
LEADER is chosen from prices only (never the eventual winner):
  (i)  ``logodds_rise``: the largest clipped-log-odds rise over the prior 7 d / 72 h / 48 h window;
  (ii) ``highest_price``: the highest price.
Buy that outcome's YES token at the next traded price after a delay (5 min headline, 1, 30) and hold
to resolution. Fee per share = rate * p * (1 - p) (common.fee_rate, tags of the market). Stake $100
capped at 1% of the outcome market's trailing 24 h taker volume before entry.

Sealed period (market closed_ts >= 2026-07-01) is dropped at load; events with any sibling closed
in or after it (or with no close time) are excluded as a whole, and a guard asserts it.
Aggregate results only: no wallets and no market ids are ever written.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from pmdetect.markets.within_event import log_odds_move
from pmdetect.stage2 import common

SNAPSHOTS = Path(__file__).resolve().parents[3] / "data" / "snapshots"
RESULTS = SNAPSHOTS.parents[1] / "results"
HORIZONS_H = {"7d": 168.0, "72h": 72.0, "48h": 48.0}
RULES = ("logodds_rise", "highest_price")
STAKE = 100.0
CAP_FRAC = 0.01
MIN_STAKE = 1.0  # a trade whose volume cap leaves less than this is skipped
MAX_WAIT_H = 6.0
PLACEBO_BAND = 0.05
N_PLACEBO = 200
MIN_OUTCOMES = 3
SIZE_BUCKETS = (("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, 10**6))
US = np.timedelta64(1, "us")


def _td(hours: float) -> np.timedelta64:
    return np.timedelta64(int(round(hours * 3600 * 1e6)), "us")


def size_bucket(n: int) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= n <= hi:
            return name
    return "<3"


# --------------------------------------------------------------------------- selection (prices only)


def pick_leader(cands: list[dict], rule: str) -> str | None:
    """Pick the leader id from candidates ``{cid, move, price}``. Uses NO outcome information.

    Ties break on the other quantity, then on the id (deterministic).
    """
    if not cands:
        return None
    if rule == "logodds_rise":
        key = lambda c: (c["move"], c["price"], c["cid"])  # noqa: E731
    elif rule == "highest_price":
        key = lambda c: (c["price"], c["move"], c["cid"])  # noqa: E731
    else:
        raise ValueError(rule)
    return max(cands, key=key)["cid"]


# --------------------------------------------------------------------------- entry price


@dataclass
class Tape:
    ts: np.ndarray  # datetime64[us], sorted
    px: np.ndarray  # YES-token-equivalent price
    cum_usd: np.ndarray  # prefix sums of usd, length n + 1


def make_tape(ts, px, usd) -> Tape:
    ts = np.asarray(ts, dtype="datetime64[us]")
    o = np.argsort(ts, kind="stable")
    return Tape(ts[o], np.asarray(px, float)[o], np.concatenate([[0.0], np.cumsum(np.asarray(usd, float)[o])]))


def find_entry(
    tape: Tape | None,
    series: tuple[np.ndarray, np.ndarray] | None,
    t_entry: np.datetime64,
    closed_ts: np.datetime64,
    max_wait_h: float = MAX_WAIT_H,
) -> dict | str:
    """Next traded price at or after ``t_entry`` within ``max_wait_h``; else a skip reason.

    The taker tape of the outcome market is used when it has any trades; the snapshot price series is
    a fallback ONLY for markets with no tape at all (then the 24 h volume is unknown: ``vol24`` None).
    """
    wait = _td(max_wait_h)
    if tape is not None and len(tape.ts):
        i = int(np.searchsorted(tape.ts, t_entry, side="left"))
        if i >= len(tape.ts) or tape.ts[i] - t_entry > wait:
            return "no_price_within_6h"
        te = tape.ts[i]
        if te >= closed_ts:
            return "closed_before_entry"
        lo = int(np.searchsorted(tape.ts, te - _td(24), side="left"))
        vol = float(tape.cum_usd[i] - tape.cum_usd[lo])  # trades strictly before entry (lo..i-1)
        return {"px": float(tape.px[i]), "ts": te, "source": "tape", "vol24": vol}
    if series is not None and len(series[0]):
        ts, px = series
        i = int(np.searchsorted(ts, t_entry, side="left"))
        if i >= len(ts) or ts[i] - t_entry > wait:
            return "no_price_within_6h"
        if ts[i] >= closed_ts:
            return "closed_before_entry"
        return {"px": float(px[i]), "ts": ts[i], "source": "snapshot_fallback", "vol24": None}
    return "no_price_data"


def stake_for(entry: dict) -> float | None:
    """$100 capped at 1% of trailing 24 h taker volume; None when the cap leaves under MIN_STAKE.

    Snapshot-fallback entries have no volume information: full stake (flagged by the caller).
    """
    if entry["vol24"] is None:
        return STAKE
    s = min(STAKE, CAP_FRAC * entry["vol24"])
    return s if s >= MIN_STAKE else None


# --------------------------------------------------------------------------- events


def build_events(em: pl.DataFrame, ni: pl.DataFrame, mk: pl.DataFrame) -> tuple[list[dict], Counter]:
    """Eligible events from event_markets / news_independent / markets. Returns (events, exclusions).

    Event T = median independent news time of its markets. The sealed guard is applied to the event
    close time (max sibling closed_ts): events touching the sealed period are dropped.
    """
    ex: Counter = Counter()
    indep = ni.filter(pl.col("independent")).select("condition_id", "news_ts_indep")
    tags = mk.select("condition_id", "tags")
    d = em.join(indep, on="condition_id", how="left").join(tags, on="condition_id", how="left")
    ev = d.group_by("event_id").agg(
        pl.len().alias("n"),
        pl.col("neg_risk").all().alias("neg_risk"),
        pl.col("closed_ts").max().alias("closed_ts"),
        pl.col("closed_ts").null_count().alias("n_null_close"),
        pl.col("news_ts_indep").median().alias("T"),
        pl.col("news_ts_indep").drop_nulls().len().alias("n_indep"),
    )
    ex["events_total"] = ev.height
    ev = ev.filter(pl.col("n_indep") > 0)
    ex["no_independent_time"] = ex["events_total"] - ev.height
    unresolved = ev.filter(pl.col("n_null_close") > 0)
    ex["sibling_without_close_time"] = unresolved.height
    ev = ev.filter(pl.col("n_null_close") == 0)
    sealed_n = ev.filter(pl.col("closed_ts") >= common.TUNE_END).height
    ex["sealed_period_dropped"] = sealed_n
    ev = common.drop_sealed(ev)
    common.assert_not_sealed(ev)
    ex["not_neg_risk"] = ev.filter(~pl.col("neg_risk")).height
    ev = ev.filter(pl.col("neg_risk"))
    ex["fewer_than_3_outcomes"] = ev.filter(pl.col("n") < MIN_OUTCOMES).height
    ev = ev.filter(pl.col("n") >= MIN_OUTCOMES)
    keep = ev["event_id"].to_list()
    rows = d.filter(pl.col("event_id").is_in(keep))
    common.assert_not_sealed(rows)
    by_event: dict[str, list[dict]] = {}
    for r in rows.iter_rows(named=True):
        by_event.setdefault(r["event_id"], []).append(
            {
                "cid": r["condition_id"],
                "yes_token": r["yes_token_id"],
                "won": None if r["outcome"] is None else r["outcome"] == "YES",
                "closed_ts": np.datetime64(r["closed_ts"], "us"),
                "rate": common.fee_rate(r["tags"]),
            }
        )
    events = []
    for r in ev.iter_rows(named=True):
        events.append(
            {
                "event_id": r["event_id"],
                "T": np.datetime64(r["T"], "us"),
                "closed_ts": r["closed_ts"],
                "period": common.period_of(r["closed_ts"]),
                "n_outcomes": r["n"],
                "markets": by_event[r["event_id"]],
            }
        )
    ex["events_eligible"] = len(events)
    return events, ex


# --------------------------------------------------------------------------- trade construction


def build_trades(
    events: list[dict],
    tapes: dict[str, Tape],
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    delays_min=common.DELAYS_MIN,
) -> tuple[list[dict], Counter]:
    """One leader trade per (event, horizon, rule, delay), with matched placebo candidates."""
    skips: Counter = Counter()
    trades: list[dict] = []
    for ev in events:
        T = ev["T"]
        for hname, hh in HORIZONS_H.items():
            t_sel = T - _td(hh)
            comp = []
            for m in ev["markets"]:
                if m["closed_ts"] <= t_sel:
                    continue  # already closed at selection time (knowable)
                s = series.get(m["cid"])
                if s is None or len(s[0]) == 0:
                    continue
                mv, why = log_odds_move(s[0], s[1], t_sel, hh)
                if mv is None:
                    continue
                i = int(np.searchsorted(s[0], t_sel, side="right")) - 1
                comp.append({"cid": m["cid"], "move": mv, "price": float(s[1][i])})
            if len(comp) < MIN_OUTCOMES:
                skips[(hname, "all", "all", "fewer_than_3_comparable_outcomes")] += 1
                continue
            leaders = {rule: pick_leader(comp, rule) for rule in RULES}
            for dmin in delays_min:
                t_entry = t_sel + np.timedelta64(dmin * 60 * 10**6, "us")
                ents: dict[str, dict | str] = {}
                for m in ev["markets"]:
                    ents[m["cid"]] = find_entry(
                        tapes.get(m["cid"]), series.get(m["cid"]), t_entry, m["closed_ts"]
                    )
                by_cid = {m["cid"]: m for m in ev["markets"]}
                for rule, lid in leaders.items():
                    e = ents[lid]
                    key = (hname, rule, dmin)
                    if isinstance(e, str):
                        skips[(*key, e)] += 1
                        continue
                    m = by_cid[lid]
                    if m["won"] is None:
                        skips[(*key, "unresolved_outcome")] += 1
                        continue
                    stake = stake_for(e)
                    if stake is None:
                        skips[(*key, "volume_cap_below_min")] += 1
                        continue
                    cands = []
                    for cid2, e2 in ents.items():
                        if cid2 == lid or isinstance(e2, str) or by_cid[cid2]["won"] is None:
                            continue
                        if abs(e2["px"] - e["px"]) > PLACEBO_BAND:
                            continue
                        s2 = stake_for(e2)
                        if s2 is None:
                            continue
                        m2 = by_cid[cid2]
                        cands.append(
                            (s2, common.net_return_per_dollar(e2["px"], m2["won"], m2["rate"]))
                        )
                    trades.append(
                        {
                            "event_id": ev["event_id"],
                            "period": ev["period"],
                            "size": size_bucket(ev["n_outcomes"]),
                            "horizon": hname,
                            "rule": rule,
                            "delay": dmin,
                            "px": e["px"],
                            "won": bool(m["won"]),
                            "rate": m["rate"],
                            "stake": stake,
                            "capped": stake < STAKE,
                            "source": e["source"],
                            "cands": np.array(cands, float).reshape(-1, 2),
                        }
                    )
    return trades, skips


# --------------------------------------------------------------------------- statistics


def _rets(rows: list[dict], cost_mult: float) -> np.ndarray:
    return np.array(
        [common.net_return_per_dollar(r["px"], r["won"], r["rate"], cost_mult) for r in rows], float
    )


def event_aggregates(rows: list[dict], cost_mult: float = 1.0) -> dict:
    """Per-event sums (events are the bootstrap clusters)."""
    ret = _rets(rows, cost_mult)
    ids = [r["event_id"] for r in rows]
    uniq = {e: i for i, e in enumerate(dict.fromkeys(ids))}
    k = len(uniq)
    stake = np.zeros(k)
    pnl = np.zeros(k)
    win = np.zeros(k)
    ent = np.zeros(k)
    n = np.zeros(k)
    for r, x, e in zip(rows, ret, ids, strict=True):
        j = uniq[e]
        stake[j] += r["stake"]
        pnl[j] += r["stake"] * x
        win[j] += r["won"]
        ent[j] += r["px"]
        n[j] += 1
    return {"stake": stake, "pnl": pnl, "win": win, "ent": ent, "n": n}


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, n_boot: int = common.N_BOOT, seed: int = common.SEED):
    """Resample whole EVENTS with replacement; statistic = sum(num)/sum(den). Returns (point, draws)."""
    k = len(num)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    point = float(num.sum() / den.sum()) if den.sum() else float("nan")
    d = den[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.where(d > 0, num[idx].sum(axis=1) / d, np.nan)
    return point, draws


def placebo_draws(rows: list[dict], n_draws: int = N_PLACEBO, seed: int = common.SEED) -> dict:
    """Random non-leader outcome matched on entry price (within 0.05), cost multiplier 1.

    Per draw each trade with at least one candidate picks one uniformly; statistic = pooled net
    return per dollar (candidate's own stake). Also returns the leaders' value on the SAME matched trades.
    """
    m = [r for r in rows if len(r["cands"])]
    if not m:
        return {"n_matched": 0}
    rng = np.random.default_rng(seed + 1)
    ncand = np.array([len(r["cands"]) for r in m])
    offs = np.concatenate([[0], np.cumsum(ncand)[:-1]])
    flat = np.vstack([r["cands"] for r in m])
    stakes, rets = flat[:, 0], flat[:, 1]
    vals = np.empty(n_draws)
    for b in range(n_draws):
        pick = offs + np.floor(rng.random(len(m)) * ncand).astype(int)
        vals[b] = float((stakes[pick] * rets[pick]).sum() / stakes[pick].sum())
    lr = _rets(m, 1.0)
    st = np.array([r["stake"] for r in m])
    return {
        "n_matched": len(m),
        "obs_matched": float((st * lr).sum() / st.sum()),
        "placebo_mean": float(vals.mean()),
        "placebo_lo": float(np.quantile(vals, 0.025)),
        "placebo_hi": float(np.quantile(vals, 0.975)),
    }


def group_stats(rows: list[dict], cost_mult: float = 1.0, with_placebo: bool = True) -> dict:
    if not rows:
        return {"n_events": 0, "n_trades": 0}
    a = event_aggregates(rows, cost_mult)
    pt, dr = bootstrap_ratio(a["pnl"], a["stake"])
    dr_ok = dr[~np.isnan(dr)]
    out = {
        "n_events": len(a["n"]),
        "n_trades": len(rows),
        "stake_total": float(a["stake"].sum()),
        "median_stake": float(np.median([r["stake"] for r in rows])),
        "share_capped": float(np.mean([r["capped"] for r in rows])),
        "win_rate": float(a["win"].sum() / a["n"].sum()),
        "mean_entry": float(a["ent"].sum() / a["n"].sum()),
        "net_ret": pt,
        "ret_lo": float(np.quantile(dr_ok, 0.025)),
        "ret_hi": float(np.quantile(dr_ok, 0.975)),
        "p_one_sided": float((np.sum(dr_ok <= 0) + 1) / (len(dr_ok) + 1)),
    }
    # is the leader underpriced? win rate minus mean entry price, event bootstrap
    gap, gd = bootstrap_ratio(a["win"] - a["ent"], a["n"], seed=common.SEED + 7)
    out["win_minus_entry"] = gap
    out["gap_lo"] = float(np.nanquantile(gd, 0.025))
    out["gap_hi"] = float(np.nanquantile(gd, 0.975))
    if with_placebo:
        out.update(placebo_draws(rows))
    return out


def evaluate(trades: list[dict]) -> list[dict]:
    """All reported rows: cells x (overall, period, size) x delay, plus the stress rows."""
    res: list[dict] = []

    def add(rows, **kw):
        res.append({**kw, **group_stats(rows, kw.get("cost_mult", 1.0))})

    for rule in RULES:
        for h in HORIZONS_H:
            base = [t for t in trades if t["rule"] == rule and t["horizon"] == h]
            for delay in common.DELAYS_MIN:
                sub = [t for t in base if t["delay"] == delay]
                add(sub, rule=rule, horizon=h, delay=delay, cost_mult=1.0, group="overall", value="all")
                if delay != 5:
                    continue
                for p in ("fit", "tune"):
                    add([t for t in sub if t["period"] == p], rule=rule, horizon=h, delay=delay,
                        cost_mult=1.0, group="period", value=p)
                for name, _, _ in SIZE_BUCKETS:
                    add([t for t in sub if t["size"] == name], rule=rule, horizon=h, delay=delay,
                        cost_mult=1.0, group="size", value=name)
                add([t for t in sub if t["source"] == "tape"], rule=rule, horizon=h, delay=delay,
                    cost_mult=1.0, group="tape_only", value="tape_only")
            add([t for t in base if t["delay"] == 30], rule=rule, horizon=h, delay=30,
                cost_mult=2.0, group="stress", value="fee_x2_delay30")
    return res


def pass_table(res: list[dict], n_cells: int = 6) -> list[dict]:
    """Frozen pass rule per (rule, horizon): lo95 > 0 AND beats placebo AND positive under stress."""
    out = []
    for rule in RULES:
        for h in HORIZONS_H:
            g = next((r for r in res if r["rule"] == rule and r["horizon"] == h
                      and r["group"] == "overall" and r["delay"] == 5), {})
            s = next((r for r in res if r["rule"] == rule and r["horizon"] == h
                      and r["group"] == "stress"), {})
            ok_lo = g.get("ret_lo", float("nan")) > 0
            ok_pl = g.get("obs_matched", float("nan")) > g.get("placebo_hi", float("nan"))
            ok_st = s.get("net_ret", float("nan")) > 0
            p = g.get("p_one_sided", 1.0)
            out.append(
                {
                    "rule": rule, "horizon": h, "lo95_gt0": bool(ok_lo), "beats_placebo": bool(ok_pl),
                    "stress_positive": bool(ok_st), "PASS": bool(ok_lo and ok_pl and ok_st),
                    "p_one_sided": p, "bonferroni_p": min(1.0, p * n_cells),
                    "bonferroni_pass": bool(ok_lo and ok_pl and ok_st and p * n_cells < 0.05),
                }
            )
    return out


# --------------------------------------------------------------------------- IO / driver


def load_inputs(snap: Path = SNAPSHOTS):
    em = pl.read_parquet(snap / "event_markets.parquet")
    ni = pl.read_parquet(snap / "news_independent.parquet")
    mk = pl.read_parquet(snap / "markets.parquet", columns=["condition_id", "tags"])
    events, ex = build_events(em, ni, mk)
    cids = [m["cid"] for e in events for m in e["markets"]]
    tt = (
        pl.scan_parquet(snap / "trades_taker.parquet")
        .filter(pl.col("condition_id").is_in(cids))
        .select("condition_id", "token_id", "price", "usd", "ts")
        .collect()
    )
    yes_tok = {m["cid"]: m["yes_token"] for e in events for m in e["markets"]}
    tt = tt.with_columns(
        pl.when(pl.col("token_id") == pl.col("condition_id").replace_strict(yes_tok, default=None))
        .then(pl.col("price"))
        .otherwise(1.0 - pl.col("price"))
        .alias("yes_px")
    )
    tapes = {}
    for (cid,), g in tt.partition_by("condition_id", as_dict=True).items():
        tapes[cid] = make_tape(g["ts"].to_numpy(), g["yes_px"].to_numpy(), g["usd"].to_numpy())
    ep = (
        pl.scan_parquet(snap / "event_prices.parquet")
        .filter(pl.col("condition_id").is_in(cids))
        .select("condition_id", "ts", "price")
        .sort("condition_id", "ts")
        .collect()
    )
    series = {}
    for (cid,), g in ep.partition_by("condition_id", as_dict=True).items():
        series[cid] = (g["ts"].to_numpy().astype("datetime64[us]"), g["price"].to_numpy())
    return events, ex, tapes, series


def _err(x, lo, hi):
    return [[max(0.0, x - lo)], [max(0.0, hi - x)]]


def _fmt(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


@dataclass
class Report:
    events: list[dict] = field(default_factory=list)
    exclusions: Counter = field(default_factory=Counter)
    skips: Counter = field(default_factory=Counter)
    results: list[dict] = field(default_factory=list)
    passes: list[dict] = field(default_factory=list)


def run(out_dir: Path = RESULTS, snap: Path = SNAPSHOTS, log=print) -> Report:
    events, ex, tapes, series = load_inputs(snap)
    log(f"events eligible: {len(events)}; tapes {len(tapes)}; price series {len(series)}")
    trades, skips = build_trades(events, tapes, series)
    res = evaluate(trades)
    rep = Report(events, ex, skips, res, pass_table(res))
    write_outputs(rep, trades, out_dir)
    return rep


def write_outputs(rep: Report, trades: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rep.results, infer_schema_length=None).write_csv(out_dir / "stage2_convergence.csv")
    _figure(rep, out_dir / "figures" / "stage2_convergence.png")
    (out_dir / "stage2_convergence.md").write_text(_markdown(rep, trades), encoding="utf-8")


def _markdown(rep: Report, trades: list[dict]) -> str:
    L = ["# Stage 2A: tradable convergence (strategy 2)", "",
         "Frozen definition and pass rule: docs/STAGE2_PLAN.md section 9. Research only, not advice. "
         "Fit and tune periods only (sealed period not opened). Aggregate results, no wallets, no market ids.", ""]
    ev = rep.events
    L += ["## Events", "", f"Eligible events: {len(ev)}", ""]
    for p in ("fit", "tune"):
        e = [x for x in ev if x["period"] == p]
        L.append(f"- {p}: {len(e)} events (" + ", ".join(
            f"{n}: {sum(1 for x in e if size_bucket(x['n_outcomes']) == n)}" for n, _, _ in SIZE_BUCKETS) + ")")
    L += ["", "Exclusions: " + ", ".join(f"{k}={v}" for k, v in rep.exclusions.items()), ""]
    L += ["## Trades skipped by reason (event-horizon-rule-delay attempts)", "",
          "| horizon | rule | delay | reason | count |", "|---|---|---|---|---|"]
    for k, v in sorted(rep.skips.items(), key=lambda kv: tuple(map(str, kv[0]))):
        L.append("| " + " | ".join(map(str, k)) + f" | {v} |")
    t5 = [t for t in trades if t["delay"] == 5]
    if t5:
        L += ["", f"Trades at the 5-min headline delay: {len(t5)}; stake scaled down by the 1% volume cap: "
              f"{100 * np.mean([t['capped'] for t in t5]):.1f}%; entry from the snapshot price series "
              f"(sibling with no taker tape): {100 * np.mean([t['source'] != 'tape' for t in t5]):.1f}%."]
    L += ["", "## Pass table (frozen rule: lower 95% bound > 0, beats placebo, positive at fee x2 + 30-min entry)", "",
          "| rule | horizon | lo95>0 | beats placebo | stress>0 | PASS | one-sided p | Bonferroni p (6 cells) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rep.passes:
        L.append(f"| {r['rule']} | {r['horizon']} | {r['lo95_gt0']} | {r['beats_placebo']} | {r['stress_positive']} | "
                 f"{'PASS' if r['PASS'] else 'FAIL'} | {_fmt(r['p_one_sided'])} | {_fmt(r['bonferroni_p'])} |")
    n_pass = sum(r["PASS"] for r in rep.passes)
    n_bon = sum(r["bonferroni_pass"] for r in rep.passes)
    L += ["", f"Cells tested: 6. Passing the frozen rule: {n_pass}. Passing after Bonferroni over 6 cells: {n_bon}. "
          "By chance alone at a one-sided 2.5% level about 0.15 cells (0.30 at 5%) are expected to show a positive "
          "lower bound. 'Beats placebo' is read as: leaders' net return on the matched trades exceeds the upper "
          "95% limit of the 200 placebo draws.", ""]
    L += ["## Results (net return per dollar, 95% event-bootstrap interval)", "",
          "| group | value | rule | horizon | delay | cost x | events | trades | win rate | mean entry | win-entry [95%] | "
          "net ret [95%] | placebo mean [95%] | leaders on matched | matched trades | median stake |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep.results:
        if not r.get("n_trades"):
            L.append(f"| {r['group']} | {r['value']} | {r['rule']} | {r['horizon']} | {r['delay']} | {r['cost_mult']} | 0 | 0 |||||||")
            continue
        L.append(
            f"| {r['group']} | {r['value']} | {r['rule']} | {r['horizon']} | {r['delay']} | {r['cost_mult']:g} | "
            f"{r['n_events']} | {r['n_trades']} | {_fmt(r['win_rate'])} | {_fmt(r['mean_entry'])} | "
            f"{_fmt(r['win_minus_entry'])} [{_fmt(r['gap_lo'])}, {_fmt(r['gap_hi'])}] | "
            f"{_fmt(r['net_ret'])} [{_fmt(r['ret_lo'])}, {_fmt(r['ret_hi'])}] | "
            f"{_fmt(r.get('placebo_mean'))} [{_fmt(r.get('placebo_lo'))}, {_fmt(r.get('placebo_hi'))}] | "
            f"{_fmt(r.get('obs_matched'))} | {r.get('n_matched', 0)} | {_fmt(r['median_stake'], 1)} |")
    L += ["", "## Caveats", "",
          "- Scheduled-reveal events (independent news time = scheduled end) are a selected subset of markets.",
          "- Leaders are chosen from prices only; the eventual winner is never used. Events are required to be "
          "neg-risk with at least 3 outcomes comparable at selection time, and all siblings must have a close time.",
          "- Entry uses the outcome market's own taker tape (any trade, the NO-token price converted to a YES price); "
          "the snapshot price series is used only for markets with no tape at all (then the volume cap is unknown and "
          "the full $100 stake is assumed; see the tape_only rows).",
          "- Thin sibling tapes: entry within 6 h of the delayed time, else the trade is skipped; fills are not "
          "guaranteed at displayed size.",
          "- The fee schedule is from a secondary source and is to be confirmed with the venue.",
          "- The price-matched placebo (non-leader within 0.05 of the leader's entry price) exists for only a "
          "subset of trades (column 'matched trades'); where that subset is small the placebo interval is "
          "degenerate and not informative, and 'beats placebo' there is weak evidence.",
          "- Several horizons and two rules are tested on overlapping events: results are not independent."]
    return "\n".join(L) + "\n"


def _figure(rep: Report, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [(r, h) for r in RULES for h in HORIZONS_H]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (rule, h) in enumerate(cells):
        g = next((x for x in rep.results if x["rule"] == rule and x["horizon"] == h
                  and x["group"] == "overall" and x["delay"] == 5), None)
        if not g or not g.get("n_trades"):
            continue
        c = "tab:blue" if rule == RULES[0] else "tab:orange"
        ax[0].errorbar(g["net_ret"], k, xerr=_err(g["net_ret"], g["ret_lo"], g["ret_hi"]),
                       fmt="o", color=c)
        if g.get("placebo_mean") is not None:
            ax[0].errorbar(g["placebo_mean"], k + 0.25, xerr=_err(g["placebo_mean"], g["placebo_lo"], g["placebo_hi"]), fmt="s", color="gray", alpha=0.8)
        ax[1].errorbar(g["win_minus_entry"], k, xerr=_err(g["win_minus_entry"], g["gap_lo"], g["gap_hi"]), fmt="o", color=c)
    for a, t in zip(ax, ("Net return per dollar (5-min entry); grey = placebo", "Win rate minus mean entry price"),
                    strict=True):
        a.axvline(0, color="k", lw=0.8)
        a.set_yticks(range(len(cells)))
        a.set_yticklabels([f"{r}, T-{h}" for r, h in cells], fontsize=8)
        a.set_title(t, fontsize=10)
    fig.suptitle("Stage 2A tradable convergence: fit + tune, 95% event-bootstrap intervals", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

