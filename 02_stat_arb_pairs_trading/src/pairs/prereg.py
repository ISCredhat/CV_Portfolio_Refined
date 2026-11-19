"""Load and verify the pre-registration (``config/preregistration.yaml``).

`load()` does two things that make the file binding:
  1. if the file is frozen, the sha256 of its bytes must equal the one recorded by `freeze()` in
     ``config/preregistration.sha256`` (an edit after freezing raises);
  2. the YAML must agree with `config_crsp` (frozen configuration, periods, N, execution, filters, selection
     settings, seeds). A disagreement raises, so the code can never run a configuration other than the registered one.
Nothing re-selects the configuration: `frozen_configuration()` returns it from the YAML.

Post-freeze fixes go to ``config/deviations_log.yaml`` (not hashed), never into the registered file.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import config_crsp as CC

HASH_FILE = CC.CONFIG_DIR / "preregistration.sha256"


class PreregError(AssertionError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_path(path: Path) -> Path:
    return Path(path).with_suffix(".sha256")


def _expect(cond: bool, what: str, yaml_val, cfg_val) -> None:
    if not cond:
        raise PreregError(f"preregistration.yaml disagrees with config_crsp on {what}: yaml={yaml_val!r}, config={cfg_val!r}")


def _tup(x):
    return tuple(tuple(v) if isinstance(v, list) else v for v in x)


def check_against_config(cfg: dict) -> None:
    """Assert that the YAML values equal config_crsp (frozen config, periods, N, execution, filters, selection...)."""
    c, u, s, e = cfg["configuration"], cfg["universe"], cfg["selection"], cfg["execution"]
    F = CC.FROZEN
    for k_yaml, k_cfg in [("bar", "bar"), ("method", "method"), ("entry", "entry"), ("exit", "exit")]:
        _expect(c[k_yaml] == F[k_cfg], f"configuration.{k_yaml}", c[k_yaml], F[k_cfg])
    for k, v in {"stop_z_add": CC.STOP_Z_ADD, "time_stop_half_lives": CC.TIME_STOP_HALF_LIVES,
                 "rolling_window_bars": CC.ROLLING_WINDOW_BARS, "kalman_memory_days": CC.KALMAN_MEMORY_DAYS,
                 "leverage": CC.LEVERAGE, "n_pairs": CC.N_PAIRS, "n_pairs_secondary": CC.N_PAIRS_SECONDARY,
                 "max_pairs_per_stock": CC.MAX_PAIRS_PER_STOCK}.items():
        _expect(c[k] == v, f"configuration.{k}", c[k], v)
    n_var = 1
    for lst in CC.VARIANTS.values():
        n_var *= len(lst)
    _expect(c["variants"] == CC.VARIANTS and c["variants_total"] == n_var, "configuration.variants", c["variants"], CC.VARIANTS)
    _expect(u["top_n"] == CC.UNIVERSE_TOP_N and u["top_n_sensitivity"] == CC.UNIVERSE_TOP_N_SENSITIVITY,
            "universe.top_n", u["top_n"], CC.UNIVERSE_TOP_N)
    _expect(u["min_price_last_formation_day"] == CC.MIN_PRICE_LAST_DAY and u["min_traded_share"] == CC.MIN_TRADED_SHARE,
            "universe price / traded filters", (u["min_price_last_formation_day"], u["min_traded_share"]),
            (CC.MIN_PRICE_LAST_DAY, CC.MIN_TRADED_SHARE))
    _expect(u["share_filter"] == CC.SHARE_FILTER, "universe.share_filter", u["share_filter"], CC.SHARE_FILTER)
    _expect(u["exclude_same_permco"] == CC.EXCLUDE_SAME_PERMCO, "universe.exclude_same_permco", u["exclude_same_permco"],
            CC.EXCLUDE_SAME_PERMCO)
    _expect(tuple(u["merger_codes"]) == tuple(CC.MERGER_CODES), "universe.merger_codes", u["merger_codes"], CC.MERGER_CODES)
    ind = cfg["industry"]
    _expect(ind["primary"] == CC.INDUSTRY_PRIMARY and ind["sensitivity"] == CC.INDUSTRY_SENSITIVITY
            and ind["exclude_ff48"] == CC.FF48_EXCLUDE, "industry", ind, CC.INDUSTRY_PRIMARY)
    for k, v in {"formation_months": CC.FORMATION_MONTHS, "min_obs_share": CC.MIN_OBS_SHARE, "alpha": CC.ALPHA,
                 "fdr": CC.FDR, "eg_maxlag": CC.EG_MAXLAG, "max_hurst": CC.MAX_HURST}.items():
        _expect(s[k] == v, f"selection.{k}", s[k], v)
    _expect(tuple(s["half_life_days"]) == tuple(CC.HALF_LIFE_DAYS), "selection.half_life_days", s["half_life_days"], CC.HALF_LIFE_DAYS)
    _expect(tuple(s["hedge_ratio_range"]) == tuple(CC.HEDGE_RATIO_RANGE), "selection.hedge_ratio_range",
            s["hedge_ratio_range"], CC.HEDGE_RATIO_RANGE)
    _expect(s["formation_months_sensitivity"] == CC.FORMATION_MONTHS_SENSITIVITY, "selection.formation_months_sensitivity",
            s["formation_months_sensitivity"], CC.FORMATION_MONTHS_SENSITIVITY)
    _expect(s["random_placebo_portfolios"] == CC.N_PLACEBO, "selection.random_placebo_portfolios", s["random_placebo_portfolios"], CC.N_PLACEBO)
    _expect(e["primary"] == CC.EXEC_PRIMARY and e["co_reported"] == CC.EXEC_CO_REPORTED and e["no_delay"] == CC.EXEC_NO_DELAY,
            "execution modes", e, CC.EXEC_PRIMARY)
    _expect(e["open_first_month"] == CC.OPEN_FIRST_MONTH and e["open_spread_mult"] == CC.OPEN_SPREAD_MULT,
            "execution open settings", e, CC.OPEN_FIRST_MONTH)
    per = {k: tuple(v) for k, v in cfg["periods"].items()}
    want = {k: tuple(v) for k, v in CC.PERIODS.items()}
    want["main"] = (CC.MAIN_FIRST_TRADE_MONTH, CC.MAIN_LAST_TRADE_MONTH)
    want["ggr"] = (CC.GGR_FIRST_TRADE_MONTH, CC.GGR_LAST_TRADE_MONTH)
    _expect(per == want, "periods", per, want)
    d = cfg["delisting"]
    _expect(d["impute"] == CC.DELIST_IMPUTE and tuple(d["bounds"]) == tuple(CC.DELIST_IMPUTE_BOUNDS), "delisting", d, CC.DELIST_IMPUTE)
    k = cfg["costs"]
    _expect(k["spread_fallback_mult"] == CC.SPREAD_FALLBACK_MULT and k["impact_bps"] == CC.IMPACT_BPS
            and k["borrow_fee_annual"] == CC.BORROW_FEE_ANNUAL and k["borrow_fee_hard"] == CC.BORROW_FEE_HARD
            and k["multipliers"] == CC.COST_MULTIPLIERS and k["quotes_reliable_from"] == CC.QUOTES_RELIABLE_FROM
            and k["commission_flat_sensitivity"] == CC.COMMISSION_FLAT_SENSITIVITY, "costs", k, "config_crsp costs")
    _expect(_tup(k["commission_bps"]) == _tup(CC.COMMISSION_BPS), "costs.commission_bps",
            k["commission_bps"], CC.COMMISSION_BPS)
    _expect({x: [tuple(r) for r in v] for x, v in k["ticks"].items()} == {x: [tuple(r) for r in v] for x, v in CC.TICKS.items()},
            "costs.ticks", k["ticks"], CC.TICKS)
    b = cfg["short_sale_ban"]
    _expect(tuple(b["window"]) == tuple(CC.SHORT_BAN) and b["ff48"] == CC.SHORT_BAN_FF48, "short_sale_ban", b, CC.SHORT_BAN)
    t = cfg["statistics"]
    for kk, v in {"trading_days": CC.TRADING_DAYS, "boot_reps": CC.BOOT_REPS, "boot_mean_block": CC.BOOT_MEAN_BLOCK,
                  "nw_lags_monthly": CC.NW_LAGS_MONTHLY, "nw_lags_trend": CC.NW_LAGS_TREND, "seed": CC.SEED,
                  "n_placebo": CC.N_PLACEBO}.items():
        _expect(t[kk] == v, f"statistics.{kk}", t[kk], v)
    _expect(t["boot_block_sensitivity"] == CC.BOOT_BLOCK_SENSITIVITY, "statistics.boot_block_sensitivity",
            t["boot_block_sensitivity"], CC.BOOT_BLOCK_SENSITIVITY)
    h = cfg["hypotheses"]
    _expect(h["H2"]["min_excess_pp"] == CC.H2_MIN_EXCESS_PP and h["H2"]["nw_lags"] == CC.NW_LAGS_MONTHLY
            and h["H4"]["nw_lags"] == CC.NW_LAGS_TREND and h["H4"]["break"] == CC.TREND_BREAK, "hypotheses settings", h, "config_crsp")
    for name in ("H1", "H2", "H3", "H4"):
        for key in ("statistic", "side", "alpha", "holm"):
            _expect(key in h[name], f"hypotheses.{name}.{key} present", None, key)
    _expect(h["holm_across"] == ["H1", "H2", "H3", "H4"], "hypotheses.holm_across", h["holm_across"], "H1-H4")


def load(path: str | Path = CC.PREREG_YAML, verify: bool = True) -> dict:
    """Parse the pre-registration, verify its hash if frozen, assert agreement with config_crsp."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if verify:
        hp = _hash_path(path)
        if cfg["meta"]["frozen"]:
            if not hp.exists():
                raise PreregError(f"{path.name} is marked frozen but {hp.name} is missing")
            recorded = hp.read_text().split()[0]
            if recorded != sha256_file(path):
                raise PreregError(f"{path.name} changed after freezing: sha256 {sha256_file(path)[:12]} != recorded {recorded[:12]}")
        elif hp.exists():
            raise PreregError(f"{hp.name} exists but {path.name} is not marked frozen")
        check_against_config(cfg)
    return cfg


def frozen_configuration(cfg: dict | None = None) -> dict:
    """The registered trading configuration (method, entry, exit, bar): what the pipeline must run, never re-picked."""
    c = (cfg or load())["configuration"]
    return {"method": c["method"], "entry": c["entry"], "exit": c["exit"], "bar": c["bar"]}


def open_checks(cfg: dict) -> list[str]:
    return [k for k, v in cfg.get("pre_freeze_checks", {}).items() if not v]


def git_commit(repo: Path = CC.ROOT) -> str:
    """HEAD commit (with a '-dirty' suffix if the tree has uncommitted changes); read-only git calls."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def freeze(path: str | Path = CC.PREREG_YAML, commit: str | None = None, now: datetime | None = None,
           allow_open_checks: bool = False) -> dict:
    """Mark the file frozen: set frozen: true, frozen_at, git_commit (text edits, comments are kept), then record the
    sha256 of the final bytes beside it. Refuses while a `pre_freeze_checks` item is false, or if already frozen."""
    path = Path(path)
    cfg = load(path)
    if cfg["meta"]["frozen"]:
        raise PreregError("already frozen")
    todo = open_checks(cfg)
    if todo and not allow_open_checks:
        raise PreregError("open pre-freeze checks: " + ", ".join(todo))
    commit = commit or git_commit()
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = path.read_text()
    for key, val in [("frozen", "true"), ("frozen_at", f'"{stamp}"'), ("git_commit", f'"{commit}"')]:
        text, n = re.subn(rf"^(  {key}:)\s*\S+", rf"\g<1> {val}", text, count=1, flags=re.M)
        if n != 1:
            raise PreregError(f"cannot set meta.{key}")
    path.write_text(text)
    digest = sha256_file(path)
    _hash_path(path).write_text(f"{digest}  {path.name}\n")
    load(path)
    return {"sha256": digest, "git_commit": commit, "frozen_at": stamp}
