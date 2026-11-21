"""Point-in-time universe: filters on name spells valid at formation end, price, trading days, then top N by size.

Order (review m6): (1) ``config_crsp.SHARE_FILTER`` on the ``stocknames_v2`` spell valid on the last formation
day, (2) raw close >= $5 on that day, (3) traded (trade-price flag and volume > 0) on >= 95% of the formation
days, (4) top N by market cap on the last formation day (ties by PERMNO). Nothing after the last formation day
enters (perturbation-tested). The formation window is ``FORMATION_MONTHS`` calendar months before the first day
of `month`. A stock must therefore have been listed for the whole window to reach 95% traded share, which is the
intended reading. ``mcap`` is in $ thousands (CRSP ``dlycap``), ``price`` in $.
Verify on WRDS: the ``stocknames_v2`` columns of ``SHARE_FILTER`` exist (a missing one raises).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config_crsp as CC
from . import crsp as R
from . import industries as I


def _month_start(month) -> pd.Timestamp:
    return pd.Period(month, freq="M").start_time


def formation_rows(days: pd.DatetimeIndex, month, formation_months: int = CC.FORMATION_MONTHS):
    """(first row, last row) of the formation window, or raises when `days` does not cover it."""
    ms = _month_start(month)
    t0 = ms - pd.DateOffset(months=formation_months)
    e = int(days.searchsorted(ms)) - 1
    s = int(days.searchsorted(t0))
    if e < 0 or days[0] > t0 + pd.Timedelta(days=7):
        raise ValueError(f"panel starts {days[0].date()}, formation window for {month} starts {t0.date()}")
    return s, e


def _share_filter(spells: pd.DataFrame, share_filter: dict) -> pd.Series:
    ok = pd.Series(True, index=spells.index)
    for col, allowed in share_filter.items():
        if col not in spells.columns:
            raise KeyError(f"stocknames_v2: filter column {col!r} missing (verify on WRDS)")
        ok &= spells[col].isin(allowed)
    return ok


def _core(days, permnos, permco, prc, cap, fresh, exch, spells, month, top_n, formation_months, gics, share_filter):
    s, e = formation_rows(days, month, formation_months)
    last = days[e]
    sp = R.names_asof(spells, last, permnos)
    funnel = {"in_panel": int(len(permnos)), "has_spell": int(len(sp))}
    sp = sp[_share_filter(sp, share_filter)]
    funnel["share_filter"] = int(len(sp))
    col = np.searchsorted(permnos, sp["permno"].to_numpy())
    px = prc[e, col]
    keep = np.isfinite(px) & (px >= CC.MIN_PRICE_LAST_DAY)
    sp, col, px = sp[keep], col[keep], px[keep]
    funnel["price"] = int(len(sp))
    share = fresh[s:e + 1, col].mean(axis=0) if len(col) else np.array([])
    keep = share >= CC.MIN_TRADED_SHARE - 1e-12
    sp, col, px, share = sp[keep], col[keep], px[keep], share[keep]
    funnel["traded_share"] = int(len(sp))
    mc = cap[e, col]
    keep = np.isfinite(mc)
    sp, col, px, share, mc = sp[keep], col[keep], px[keep], share[keep], mc[keep]
    funnel["has_mcap"] = int(len(sp))
    order = np.lexsort((sp["permno"].to_numpy(), -mc))[:top_n]
    sp, col, px, share, mc = sp.iloc[order], col[order], px[order], share[order], mc[order]
    sic = pd.to_numeric(sp["siccd"], errors="coerce").fillna(0).astype("int64").to_numpy() \
        if "siccd" in sp.columns else np.zeros(len(sp), "int64")
    g = [None] * len(sp)
    if gics is not None:
        g = [gics.get(int(p)) if hasattr(gics, "get") else None for p in sp["permno"]]
    out = pd.DataFrame({"permno": sp["permno"].to_numpy("int64"), "permco": permco[col].astype("int64"),
                        "col": col.astype("int64"), "mcap": mc.astype(float), "price": px.astype(float),
                        "exch": exch[e, col].astype(object), "siccd": sic, "ff48": I.ff48_from_sic(sic),
                        "sic2": I.sic2(sic), "gics_ind": np.array(g, dtype=object),
                        "traded_share": share.astype(float)}).reset_index(drop=True)
    funnel["selected"] = int(len(out))
    out.attrs["funnel"] = funnel
    out.attrs["formation_end"] = last
    return out


def select(panel: R.Panel, names: pd.DataFrame, month, top_n: int = CC.UNIVERSE_TOP_N,
           formation_months: int = CC.FORMATION_MONTHS, gics: pd.Series | dict | None = None,
           share_filter: dict = CC.SHARE_FILTER) -> pd.DataFrame:
    """Universe of `month` (contracts.UNIVERSE_COLUMNS), decided at the formation end. `gics` is an optional
    permno -> GICS industry map (``industries.gics_asof`` at the formation end); None before GICS exists."""
    spells = R.name_spells(names)
    return _core(panel.days, panel.permnos, panel.permco, panel.prc, panel.cap, panel.fresh, panel.exch, spells,
                 month, top_n, formation_months, gics, share_filter)


def ever_top_n(dsf: pd.DataFrame, names: pd.DataFrame, months, top_n: int = CC.UNIVERSE_TOP_N,
               formation_months: int = CC.FORMATION_MONTHS, share_filter: dict = CC.SHARE_FILTER) -> np.ndarray:
    """PERMNOs that are in the top `top_n` universe of at least one month (review M9): load only these into the
    big panel. `dsf` is the raw ``dsf_v2`` frame (normalised or not); each month needs only its formation window."""
    df = R.normalise_dsf(dsf)
    spells = R.name_spells(names)
    keep: set[int] = set()
    for m in months:
        ms = _month_start(m)
        w = df[(df["dlycaldt"] >= ms - pd.DateOffset(months=formation_months)) & (df["dlycaldt"] < ms)]
        if w.empty:
            continue
        pan = R.build_panel(w, names=None, delist=None)
        try:
            u = _core(pan.days, pan.permnos, pan.permco, pan.prc, pan.cap, pan.fresh, pan.exch, spells, m, top_n,
                      formation_months, None, share_filter)
        except ValueError:
            continue
        keep.update(int(p) for p in u["permno"])
    return np.array(sorted(keep), dtype="int64")


EVER_COLUMNS = ("permno", "permco", "dlycaldt", "dlyret", "dlyprc", "dlyvol", "dlyprcflg", "dlydelflg", "dlycap",
                "shrout", "primaryexch")      # what the universe filters read (price, cap, traded share, exchange)


def ever_top_n_chunked(raw_dir, names: pd.DataFrame, months, top_n: int = CC.UNIVERSE_TOP_N,
                       formation_months: int = CC.FORMATION_MONTHS,
                       share_filter: dict = CC.SHARE_FILTER) -> np.ndarray:
    """`ever_top_n` without holding the whole sample in memory: the months are taken one calendar year at a time,
    reading only that year's formation windows and only ``EVER_COLUMNS`` from the ``dsf_v2`` chunks under
    `raw_dir`. Each month still sees exactly its formation-window rows, so the PERMNOs are the same."""
    cols = R.panel_columns(raw_dir, EVER_COLUMNS)
    months = pd.PeriodIndex([pd.Period(m, freq="M") for m in months])
    keep: set[int] = set()
    for y in sorted(set(months.year)):
        ms = months[months.year == y]
        a = ms.min().start_time - pd.DateOffset(months=formation_months)
        b = ms.max().start_time - pd.Timedelta(days=1)
        dsf = R.load_dsf(raw_dir, str(a.date()), str(b.date()), columns=cols)
        keep.update(int(p) for p in ever_top_n(dsf, names, list(ms), top_n, formation_months, share_filter))
    return np.array(sorted(keep), dtype="int64")
