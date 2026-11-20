"""CRSP daily panel (CIZ ``dsf_v2``) -> total-return prices, delisting events and ``contracts.DailyBars``.

CIZ format (the "verify on WRDS" list, review items 1-8; checked on WRDS on 2026-11-19 with
``scripts/check_wrds_crsp.py`` and follow-up queries):
  1. ``dsf_v2`` columns used: permno, dlycaldt, dlyret (total return), dlyprc (positive), dlyvol (shares), and,
     when present, permco, dlyprcflg ('TR' = closing trade price; 'BA' = bid/ask average; others such as NT, MP,
     HA, SU = no trade), dlyretmissflg (null/empty when the return is valid, a reason code when missing),
     dlyretdurflg ('D1'-'D4' = one trading day spanning 1-4 calendar days; 'P1'.. = the return reaches back past
     a gap; 'MR' = missing), dlydelflg, dlyopen, dlyhigh, dlylow, dlybid, dlyask, dlycap ($ thousands), shrout
     (thousands), primaryexch.
     A delisting is its own row, dated the trading day after the last trade (legacy ``dlstdt``), with
     dlydelflg 'Y', dlyret = the legacy ``dlret``, dlyvol null, and dlyprcflg 'DP' (dlyprc = the delisting price),
     'DA' (dlyprc 0, return present) or 'DM' (price and return missing). No row follows it. So that row's return
     counts even without a valid price.
     Missing optional columns degrade features and are counted in ``Panel.diag``; missing required ones raise.
  2. Missing-return policy (item 7). A return that is missing on a row that is not the security's first row is
     not treated as zero: the TR index is NaN that day (no valid price) and the level carries over, so the next
     return, if it spans the gap (a 'P' duration flag), restores the right level. Gap days whose next row is
     a one-day return lose that move; they are counted (``missing_ret_lost``). Nothing is dropped silently.
  3. Delisting (item 6): CIZ puts the delisting return in ``dlyret`` of the delisting row. If that return
     is missing: (a) legacy ``dsedelist.dlret`` when present, else (b) for codes 500-599 the Shumway (1997) /
     Shumway-Warther (1999) imputation ``DELIST_IMPUTE`` by exchange, bounded to [-100%, 0], else (c) 0 with a
     count (``dlret_missing_other``). Delisting codes come from legacy ``dsedelist`` (CIZ has no code column
     that this module relies on).
  4. ``dead`` = a security's last row, unless that row is the panel's last day and ``dlydelflg`` != 'Y'.
  5. Open on the TR scale = TR close x dlyopen / dlyprc on the same day (review m12): no cumulative factors.
     Opens before ``OPEN_FIRST_MONTH`` are ignored for fills.
  6. mcap = ``dlycap`` ($ thousands) else |price| x shrout; both are in $ thousands.
Gatev, Goetzmann & Rouwenhorst (2006) use the same one-day-wait convention that ``t1_close`` mirrors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config_crsp as CC
from . import contracts as K

DSF_REQUIRED = ("permno", "dlycaldt", "dlyret", "dlyprc", "dlyvol")
DSF_OPTIONAL = ("permco", "dlyprcflg", "dlyretmissflg", "dlyretdurflg", "dlydelflg", "dlyopen", "dlyhigh",
                "dlylow", "dlybid", "dlyask", "dlyclose", "dlycap", "shrout", "primaryexch", "siccd")
NAMES_REQUIRED = ("permno", "namedt", "nameenddt")
TRADE_FLAG = "TR"
SINGLE_DAY = re.compile(r"^D\d$")           # D1-D4: one trading day (n = calendar days)
FAR_FUTURE = pd.Timestamp("2200-01-01")


# ------------------------------------------------------------------------------------------------ loading
def _need(df: pd.DataFrame, cols, what: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{what}: required column(s) {miss} missing (have {sorted(df.columns)})")


def read_chunks(raw_dir: Path, key: str, years: range | None = None, columns=None, permnos=None) -> pd.DataFrame:
    """Read ``raw_dir/<key>/*.parquet`` (yearly ``yYYYY.parquet`` for dsf_v2, ``all.parquet`` otherwise).
    A requested year without a chunk raises: no silent holes."""
    d = Path(raw_dir) / key
    if years is not None:
        files = [d / f"y{y}.parquet" for y in years]
        absent = [f.name for f in files if not f.exists()]
        if absent:
            raise FileNotFoundError(f"{key}: missing chunk(s) {absent} under {d}")
    else:
        files = sorted(d.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{key}: no parquet files under {d}")
    parts = [_read_parquet(f, columns, permnos) for f in files]
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def _read_parquet(path: Path, columns=None, permnos=None) -> pd.DataFrame:
    """One chunk, optionally only some PERMNOs. Pushes the PERMNO filter into the parquet reader when
    ``pyarrow.dataset`` is usable; otherwise (for example Windows Smart App Control blocking one of pyarrow's DLLs)
    reads the chunk and filters in pandas, with the same result."""
    if permnos is None:
        return pd.read_parquet(path, columns=columns)
    keep = [int(p) for p in permnos]
    try:
        return pd.read_parquet(path, columns=columns, filters=[("permno", "in", keep)])
    except (ImportError, ValueError) as e:
        if "dataset" not in str(e) and "filters" not in str(e):
            raise
    cols = None if columns is None else list(dict.fromkeys([*columns, "permno"]))
    df = pd.read_parquet(path, columns=cols)
    df = df[df["permno"].isin(keep)].reset_index(drop=True)
    return df if columns is None else df[list(columns)]


def normalise_dsf(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, check the required ones, types, sort, and refuse duplicate (permno, date)."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    _need(out, DSF_REQUIRED, "dsf_v2")
    out["permno"] = out["permno"].astype("int64")
    out["dlycaldt"] = pd.to_datetime(out["dlycaldt"])
    out = out.sort_values(["permno", "dlycaldt"], kind="stable").reset_index(drop=True)
    dup = out.duplicated(["permno", "dlycaldt"]).sum()
    if dup:
        raise ValueError(f"dsf_v2: {dup} duplicate (permno, dlycaldt) rows")
    return out


def load_dsf(raw_dir: Path, start: str, end: str, permnos=None, columns=None) -> pd.DataFrame:
    """Rows of ``dsf_v2`` with start <= dlycaldt <= end, optionally for a PERMNO subset (review M9)."""
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    df = read_chunks(raw_dir, "dsf_v2", range(y0, y1 + 1), columns=columns, permnos=permnos)
    df = normalise_dsf(df)
    m = (df["dlycaldt"] >= pd.Timestamp(start)) & (df["dlycaldt"] <= pd.Timestamp(end))
    return df[m].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ names
def name_spells(names: pd.DataFrame) -> pd.DataFrame:
    """stocknames_v2 as point-in-time spells: ``namedt <= d <= nameenddt`` (open end -> far future).
    Accepts ``nameendt`` for ``nameenddt``."""
    out = names.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "nameenddt" not in out.columns and "nameendt" in out.columns:
        out = out.rename(columns={"nameendt": "nameenddt"})
    _need(out, NAMES_REQUIRED, "stocknames_v2")
    out["permno"] = out["permno"].astype("int64")
    out["namedt"] = pd.to_datetime(out["namedt"])
    out["nameenddt"] = pd.to_datetime(out["nameenddt"]).fillna(FAR_FUTURE)
    bad = out["nameenddt"] < out["namedt"]
    if bad.any():
        raise ValueError(f"stocknames_v2: {int(bad.sum())} spells end before they start")
    return out.sort_values(["permno", "namedt"], kind="stable").reset_index(drop=True)


def names_asof(spells: pd.DataFrame, date, permnos=None) -> pd.DataFrame:
    """The spell valid on `date` for each PERMNO (latest ``namedt`` if two overlap); no valid spell -> absent."""
    d = pd.Timestamp(date)
    s = spells[(spells["namedt"] <= d) & (spells["nameenddt"] >= d)]
    if permnos is not None:
        s = s[s["permno"].isin(np.asarray(permnos, dtype="int64"))]
    return s.sort_values(["permno", "namedt"]).drop_duplicates("permno", keep="last").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ panel
@dataclass
class Panel:
    """Dense day x PERMNO arrays (see contracts.py indexing convention). ``tr`` is the TR index (base 1 on a
    security's first row), ``prc`` the raw positive price."""
    days: pd.DatetimeIndex
    permnos: np.ndarray
    permco: np.ndarray
    tr: np.ndarray
    prc: np.ndarray
    ret: np.ndarray            # as used (delisting folded/imputed); NaN where missing
    vol: np.ndarray
    cap: np.ndarray            # $ thousands
    open: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    high: np.ndarray
    low: np.ndarray
    fresh: np.ndarray
    dead: np.ndarray
    exch: np.ndarray           # object, forward-filled
    events: pd.DataFrame = field(default_factory=pd.DataFrame)   # contracts.DELIST_COLUMNS
    diag: dict = field(default_factory=dict)


def _col(df: pd.DataFrame, name: str, default=np.nan):
    return df[name].to_numpy() if name in df.columns else np.full(len(df), default, dtype=object if isinstance(default, str) else float)


def _scatter(shape, ri, ci, vals, fill=np.nan, dtype=float):
    a = np.full(shape, fill, dtype=dtype)
    a[ri, ci] = vals
    return a


def build_panel(dsf: pd.DataFrame, names: pd.DataFrame | None = None, delist: pd.DataFrame | None = None,
                days: pd.DatetimeIndex | None = None) -> Panel:
    """Build the panel from normalised ``dsf_v2`` rows. `delist` is legacy ``dsedelist`` (permno, dlstcd, dlret);
    `names` supplies the exchange where ``primaryexch`` is absent. See the module docstring for every policy."""
    df = normalise_dsf(dsf)
    if days is None:
        days = pd.DatetimeIndex(np.sort(df["dlycaldt"].unique()))
    else:
        df = df[df["dlycaldt"].isin(days)]
    permnos = np.sort(df["permno"].unique())
    n, k = len(days), len(permnos)
    ri = days.get_indexer(df["dlycaldt"])
    ci = np.searchsorted(permnos, df["permno"].to_numpy())
    sh = (n, k)
    diag: dict = {"n_rows": int(len(df))}
    for c in DSF_OPTIONAL:
        if c not in df.columns:
            diag.setdefault("missing_optional_columns", []).append(c)

    ret_r = pd.to_numeric(df["dlyret"], errors="coerce").to_numpy(float).copy()
    prc_r = np.abs(pd.to_numeric(df["dlyprc"], errors="coerce").to_numpy(float))
    prc_r = np.where(prc_r > 0, prc_r, np.nan)
    vol_r = np.nan_to_num(pd.to_numeric(df["dlyvol"], errors="coerce").to_numpy(float), nan=0.0)
    flag = df["dlyprcflg"].to_numpy(object) if "dlyprcflg" in df.columns else None
    trade = (flag == TRADE_FLAG) if flag is not None else np.ones(len(df), bool)
    fresh_r = trade & (vol_r > 0) & np.isfinite(prc_r)

    def pos(name):
        if name not in df.columns:
            return np.full(len(df), np.nan)
        v = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
        return np.where(v > 0, v, np.nan)

    # ---- first / last row of each security
    first_row = np.r_[True, df["permno"].to_numpy()[1:] != df["permno"].to_numpy()[:-1]]
    last_row = np.r_[df["permno"].to_numpy()[1:] != df["permno"].to_numpy()[:-1], True]
    delflg = (df["dlydelflg"].to_numpy(object) == "Y") if "dlydelflg" in df.columns else np.zeros(len(df), bool)
    is_last_day = df["dlycaldt"].to_numpy() == days.values[-1]
    dead_r = (last_row & ~(is_last_day & ~delflg)) | delflg
    exch_r = df["primaryexch"].to_numpy(object) if "primaryexch" in df.columns else np.full(len(df), None, object)

    # ---- delisting folding / imputation on dead rows with a missing return
    ev_rows = []
    leg = None
    if delist is not None and len(delist):
        leg = delist.copy()
        leg.columns = [str(c).lower() for c in leg.columns]
        _need(leg, ("permno", "dlstcd", "dlret"), "dsedelist")
        leg = leg.sort_values(["permno", "dlstdt"] if "dlstdt" in leg.columns else ["permno"]) \
                 .drop_duplicates("permno", keep="last").set_index("permno")
    cnt = dict(dlret_ciz=0, dlret_legacy=0, dlret_imputed=0, dlret_missing_other=0, no_code=0)
    ex_fill = {}
    if names is not None:
        sp = name_spells(names)
    for r in np.nonzero(dead_r)[0]:
        p = int(df["permno"].iat[r])
        code = int(leg.at[p, "dlstcd"]) if leg is not None and p in leg.index else 0
        legret = float(leg.at[p, "dlret"]) if leg is not None and p in leg.index else np.nan
        ex = exch_r[r]
        if ex is None or (isinstance(ex, float) and np.isnan(ex)):
            ex = "N"
            if names is not None:
                a = names_asof(sp, df["dlycaldt"].iat[r], [p])
                if len(a) and "primaryexch" in a.columns:
                    ex = a["primaryexch"].iat[0]
        imputed = False
        if np.isfinite(ret_r[r]):
            cnt["dlret_ciz"] += 1
            dl = np.nan if not np.isfinite(legret) else legret
        elif np.isfinite(legret):
            ret_r[r] = legret
            dl = legret
            cnt["dlret_legacy"] += 1
        elif 500 <= code <= 599:
            lo, hi = sorted(CC.DELIST_IMPUTE_BOUNDS)
            ret_r[r] = float(np.clip(CC.DELIST_IMPUTE.get(ex, CC.DELIST_IMPUTE["Q"]), lo, hi))
            dl = ret_r[r]
            imputed = True
            cnt["dlret_imputed"] += 1
        else:
            ret_r[r] = 0.0
            dl = 0.0
            cnt["dlret_missing_other"] += 1
            if code == 0:
                cnt["no_code"] += 1
        ev_rows.append((p, df["dlycaldt"].iat[r], code, ex, dl if np.isfinite(dl) else ret_r[r], imputed,
                        CC.MERGER_CODES[0] <= code <= CC.MERGER_CODES[1], r))
    diag.update(cnt)
    ev = pd.DataFrame(ev_rows, columns=["permno", "last_date", "dlstcd", "exch", "dlret", "dlret_imputed",
                                        "is_merger", "_row"]).drop(columns="_row")
    ev = ev.astype({"permno": "int64", "last_date": "datetime64[ns]", "dlstcd": "int64", "exch": object,
                    "dlret": "float64", "dlret_imputed": bool, "is_merger": bool})

    # ---- missing returns on ordinary rows
    miss = ~np.isfinite(ret_r) & ~first_row
    lost = 0
    if "dlyretdurflg" in df.columns:
        one_day = np.array([isinstance(f, str) and bool(SINGLE_DAY.match(f)) for f in df["dlyretdurflg"]])
        nxt_single = np.r_[one_day[1:], False] & ~last_row
        lost = int((miss & nxt_single).sum())
    diag["missing_ret_rows"] = int(miss.sum())
    diag["missing_ret_lost"] = lost
    r1 = np.where(np.isfinite(ret_r) & ~first_row, ret_r, 0.0)
    r1 = np.where(first_row, 0.0, r1)
    grow = _scatter(sh, ri, ci, 1.0 + r1, fill=1.0)
    tr = np.cumprod(grow, axis=0)
    ok_r = (np.isfinite(ret_r) | first_row) & (np.isfinite(prc_r) | dead_r)   # delisting rows: price may be 0/NaN
    diag["tr_nan_rows_bad_price_or_ret"] = int((~ok_r).sum())
    tr = np.where(_scatter(sh, ri, ci, ok_r, fill=False, dtype=bool), tr, np.nan)
    prc = _scatter(sh, ri, ci, prc_r)
    cap_r = pos("dlycap") if "dlycap" in df.columns else (prc_r * pos("shrout") if "shrout" in df.columns
                                                       else np.full(len(df), np.nan))
    exch = _scatter(sh, ri, ci, exch_r, fill=None, dtype=object)
    exch = _ffill_obj(exch)
    if names is not None:                                 # fill still-empty exchanges from name spells
        for j in range(k):
            if (exch[:, j] == None).any():  # noqa: E711
                a = names_asof(sp, days[0], [permnos[j]])
                if len(a) and "primaryexch" in a.columns:
                    exch[:, j] = np.where(exch[:, j] == None, a["primaryexch"].iat[0], exch[:, j])  # noqa: E711
    n_unknown = int(sum((exch[:, j] == None).all() for j in range(k)))  # noqa: E711
    diag["permnos_without_exchange"] = n_unknown
    exch = np.where(exch == None, "", exch)  # noqa: E711
    pcol = df.groupby("permno")["permco"].first().reindex(permnos).to_numpy() if "permco" in df.columns \
        else np.full(k, -1)
    return Panel(days=days, permnos=permnos, permco=np.nan_to_num(pcol, nan=-1).astype("int64"), tr=tr, prc=prc,
                 ret=_scatter(sh, ri, ci, ret_r), vol=_scatter(sh, ri, ci, vol_r, fill=0.0),
                 cap=_scatter(sh, ri, ci, cap_r), open=_scatter(sh, ri, ci, pos("dlyopen")),
                 bid=_scatter(sh, ri, ci, pos("dlybid")), ask=_scatter(sh, ri, ci, pos("dlyask")),
                 high=_scatter(sh, ri, ci, pos("dlyhigh")), low=_scatter(sh, ri, ci, pos("dlylow")),
                 fresh=_scatter(sh, ri, ci, fresh_r, fill=False, dtype=bool),
                 dead=_scatter(sh, ri, ci, dead_r, fill=False, dtype=bool), exch=exch, events=ev, diag=diag)


def _ffill_obj(a: np.ndarray) -> np.ndarray:
    """Forward fill an object array along axis 0 (None = empty), then back fill leading gaps."""
    out = a.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        valid = np.nonzero(col != None)[0]  # noqa: E711
        if len(valid) == 0:
            continue
        idx = np.searchsorted(valid, np.arange(len(col)), side="right") - 1
        idx = np.where(idx < 0, 0, idx)
        out[:, j] = col[valid[idx]]
    return out


def delisting_events(panel: Panel) -> pd.DataFrame:
    """One row per delisting event (contracts.DELIST_COLUMNS): the panel's dead rows that are delistings or
    end-of-data exits, with the delisting return used (CIZ, legacy or imputed) and the imputation flag."""
    return panel.events.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ bars
def short_ok_mask(days: pd.DatetimeIndex, permnos: np.ndarray, ff48: pd.Series | dict | None) -> np.ndarray:
    """False where a new short is forbidden: 2008-09-19..2008-10-08 for FF48 in ``SHORT_BAN_FF48`` (proxy for
    the SEC list; the uptick rule is not modelled). `ff48` maps permno -> FF48 as of formation (point in time).
    Raises when the window is inside `days` and no map is given."""
    ok = np.ones((len(days), len(permnos)), bool)
    a, b = (pd.Timestamp(x) for x in CC.SHORT_BAN)
    rows = np.nonzero((days >= a) & (days <= b))[0]
    if len(rows) == 0:
        return ok
    if ff48 is None:
        raise ValueError("short_ok: panel spans the 2008 short-sale ban but no ff48 map was given")
    m = pd.Series(ff48).reindex(permnos)
    banned = m.isin(CC.SHORT_BAN_FF48).to_numpy()
    ok[np.ix_(rows, np.nonzero(banned)[0])] = False
    return ok


def ffill_within_life(tr: np.ndarray) -> np.ndarray:
    """Carry the last TR level over days with no valid price (halts, missing returns) between a security's first
    and last finite rows, so an open position is marked flat that day instead of producing NaN P&L. Those days
    keep ``fresh = False`` (no new entries) and are counted in ``Panel.diag["stale_price_cells"]``. Nothing is
    filled before listing or after the last row (delisting), so dead stocks never come back."""
    a = tr.copy()
    fin = np.isfinite(a)
    n = len(a)
    idx = np.where(fin, np.arange(n)[:, None], -1)
    last = np.maximum.accumulate(idx, axis=0)                      # last finite row at or before t
    first = np.where(fin.any(0), fin.argmax(0), n)
    lastrow = np.where(fin.any(0), n - 1 - fin[::-1].argmax(0), -1)
    t = np.arange(n)[:, None]
    inside = (t >= first) & (t <= lastrow) & ~fin & (last >= 0)
    cols = np.broadcast_to(np.arange(a.shape[1]), a.shape)
    a[inside] = a[last[inside], cols[inside]]
    return a


def build_daily_bars(panel: Panel, exec_mode: str, ff48: pd.Series | dict | None = None) -> K.DailyBars:
    """contracts.DailyBars from a panel for one execution mode (contract docstring has the definitions)."""
    if exec_mode not in K.EXEC_MODES:
        raise ValueError(exec_mode)
    n, k = panel.tr.shape
    close = ffill_within_life(panel.tr)
    panel.diag["stale_price_cells"] = int((np.isfinite(close) & ~np.isfinite(panel.tr)).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = ffill_within_life(panel.prc) / close
        open_ok = (panel.open > 0) & (panel.days.values[:, None] >= np.datetime64(CC.OPEN_FIRST_MONTH + "-01"))
        open_tr = np.where(open_ok, close * panel.open / panel.prc, np.nan)
    ex = close.copy()
    open_fill = np.zeros((n, k), bool)
    fill_day = np.arange(n)
    if exec_mode != "t_close":
        fill_day = np.minimum(np.arange(n) + 1, n - 1)
        ex[:-1] = close[1:]
        if exec_mode == "t1_open":
            o = open_tr[1:]
            use = np.isfinite(o)
            ex[:-1] = np.where(use, o, close[1:])
            open_fill[:-1] = use
    bd = K.DailyBars(permnos=panel.permnos, days=panel.days, close=close, exec=ex, fill_day=fill_day,
                     fresh=panel.fresh, factor=factor, dead=panel.dead,
                     short_ok=short_ok_mask(panel.days, panel.permnos, ff48), exch=panel.exch,
                     exec_mode=exec_mode, open_fill=open_fill)
    K.validate_bars(bd)
    return bd


# ------------------------------------------------------------------------------------------------ B2 check
def delisting_crosscheck(dsf: pd.DataFrame, dsedelist: pd.DataFrame, legacy_dsf: pd.DataFrame | None = None,
                         tol: float = 1e-4) -> pd.DataFrame:
    """Plan B2: for every legacy delisting event compare CIZ returns with the legacy ``(1+ret)(1+dlret)-1``.
    CIZ side: compound ``dlyret`` over rows from the legacy last trading date to ``dlstdt``, plus the CIZ
    delisting row (dlydelflg 'Y', dated the trading day after ``dlstdt``) when there is one. Legacy side needs
    `legacy_dsf` (permno, date, ret); without it only ``dlret`` is compared with the CIZ delisting-row return
    (else the last row on or before ``dlstdt``; ``mode='dlret_only'``, a weak check). Events with a missing ``dlret`` are listed (``dlret_missing``) and
    excluded from the match rate. Returns one row per event."""
    z = normalise_dsf(dsf)
    ev = dsedelist.copy()
    ev.columns = [str(c).lower() for c in ev.columns]
    _need(ev, ("permno", "dlstdt", "dlstcd", "dlret"), "dsedelist")
    ev["dlstdt"] = pd.to_datetime(ev["dlstdt"])
    if "dlydelflg" not in z.columns:
        z["dlydelflg"] = "N"
    zg = {p: (g["dlycaldt"].to_numpy(), pd.to_numeric(g["dlyret"], errors="coerce").to_numpy(float),
              (g["dlydelflg"] == "Y").to_numpy()) for p, g in z.groupby("permno")}
    lg = None
    if legacy_dsf is not None:
        l = legacy_dsf.copy()
        l.columns = [str(c).lower() for c in l.columns]
        l["date"] = pd.to_datetime(l["date"])
        lg = {p: (g["date"].to_numpy(), pd.to_numeric(g["ret"], errors="coerce").to_numpy(float))
              for p, g in l.sort_values(["permno", "date"]).groupby("permno")}
    rows = []
    for e in ev.itertuples(index=False):
        p, d = int(e.permno), np.datetime64(e.dlstdt)
        zd, zr, zy = zg.get(p, (np.array([], "datetime64[ns]"), np.array([]), np.array([], bool)))
        cut = np.searchsorted(zd, d, side="right")
        ydel = cut < len(zd) and bool(zy[cut])              # CIZ delisting row right after dlstdt
        dend = zd[cut] if ydel else d
        if cut == 0:
            rows.append((p, e.dlstdt, int(e.dlstcd), np.nan, np.nan, np.nan, "no_ciz_rows", bool(np.isnan(e.dlret)))); continue
        if lg is not None and p in lg:
            ld, lr = lg[p]
            lc = np.searchsorted(ld, d, side="right")
            lt = ld[lc - 1] if lc else zd[cut - 1]
            lret = 0.0 if not lc or np.isnan(lr[lc - 1]) else lr[lc - 1]
            sel = (zd >= lt) & (zd <= dend)
            ciz = np.prod(1 + np.nan_to_num(zr[sel])) - 1
            leg = (1 + lret) * (1 + e.dlret) - 1 if not np.isnan(e.dlret) else np.nan
            mode = "compound"
        else:
            ciz = zr[cut] if ydel else zr[cut - 1]
            leg = e.dlret
            mode = "dlret_only"
        rows.append((p, e.dlstdt, int(e.dlstcd), ciz, leg, ciz - leg if np.isfinite(leg) else np.nan, mode,
                     bool(np.isnan(e.dlret))))
    t = pd.DataFrame(rows, columns=["permno", "dlstdt", "dlstcd", "ciz_ret", "legacy_ret", "diff", "mode",
                                    "dlret_missing"])
    t["match"] = t["diff"].abs() <= tol
    t["decade"] = (t["dlstdt"].dt.year // 10) * 10
    return t


def crosscheck_summary(t: pd.DataFrame) -> dict:
    """Match rate over events with a non-missing dlret, and missing-dlret counts by code/100 and decade."""
    ok = t[~t["dlret_missing"]]
    miss = t[t["dlret_missing"]].groupby([t["dlstcd"] // 100, "decade"]).size()
    return {"n_events": int(len(t)), "n_with_dlret": int(len(ok)),
            "match_rate": float(ok["match"].mean()) if len(ok) else float("nan"),
            "passes_99pct": bool(len(ok) and ok["match"].mean() >= 0.99),
            "missing_dlret_by_code_decade": {f"{a}xx/{b}s": int(v) for (a, b), v in miss.items()}}


def load_panel(raw_dir: Path, start: str, end: str, permnos=None) -> Panel:
    """Convenience: chunks -> panel (dsf_v2 + stocknames_v2 + dsedelist when present)."""
    dsf = load_dsf(raw_dir, start, end, permnos)
    names = read_chunks(raw_dir, "stocknames_v2")
    dd = Path(raw_dir) / "dsedelist"
    delist = read_chunks(raw_dir, "dsedelist") if dd.exists() else None
    return build_panel(dsf, names, delist)
