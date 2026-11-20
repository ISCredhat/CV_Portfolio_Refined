"""Synthetic CRSP-shaped data (CIZ column names) for tests and the integration run. No WRDS needed.

``make_crsp(seed, start, end, n_permno) -> dict[str, DataFrame]`` with keys

  dsf_v2          daily file: permno, permco, securitytype, securitysubtype, sharetype, usincflg, issuertype,
                  primaryexch, conditionaltype, tradingstatusflg, dlycaldt, dlydelflg, dlyprc (positive, raw),
                  dlyprcflg ('TR' trade, 'BA' bid/ask average), dlycap ($ thousands), dlyret (total return),
                  dlyretx, dlyretmissflg (null when the return is valid), dlyretdurflg ('D1' single day, 'MD'
                  return spans a gap), dlyvol, dlyopen, dlyhigh, dlylow, dlybid, dlyask, dlyclose, shrout
                  (thousands), siccd. A delisting is a separate row on the trading day after the last trade, as in
                  CIZ (checked on WRDS 2026-11-19): dlydelflg 'Y', dlyret = the delisting return, dlyvol null, and
                  dlyprcflg 'DP' (delisting price), 'DA' (price 0) or 'DM' (price and return missing).
  stocknames_v2   name spells: permno, permco, namedt, nameenddt, ticker, comnam, primaryexch, siccd, and the
                  classification columns above
  dsedelist       legacy delisting file: permno, dlstdt, dlstcd, dlret, dlretx, dlprc
  dsf_legacy      legacy daily returns (ret excludes the delisting return): permno, date, ret, retx, prc
  co_hgic         GICS history: gvkey, indfrom, indthru, gsector, ggroup, gind, gsubind
  ccmxpf_lnkhist CCM links: gvkey, linkprim, linktype, lpermno, lpermco, linkdt, linkenddt
  truth_pairs / truth_events / truth_twins   what was injected (not CRSP tables)

Features: market + industry factors, exchange and SIC changes, share classes (same PERMCO), splits and dividends
(TR != price), halts (no trade, bid/ask-average price), thin-trading stocks, performance delistings with and
without a delisting return, merger delistings, missing bid/ask before 1993 for NYSE/AMEX, missing opens before
1992-07, dropped rows (multi-day returns), a few cointegrated pairs inside an industry, a few ADRs.
The CIZ semantics here (flag values, units) are assumptions: see ``crsp.py`` "verify on WRDS".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (SIC, FF48 it maps to, GICS industry code)
INDUSTRIES = [(2834, 13, "352020"), (3674, 36, "453010"), (6021, 44, "401010"), (4911, 31, "551010"),
              (1311, 30, "101020"), (5812, 43, "253010"), (3711, 23, "251020"), (7372, 34, "451030")]
PERF_CODES = [550, 560, 574, 580, 584]
MERGER_CODES = [231, 241, 331]


def _ou(rng, n, hl, sd):
    phi = 0.5 ** (1.0 / hl)
    e = rng.standard_normal(n) * sd * np.sqrt(1 - phi ** 2)
    x = np.zeros(n)
    x[0] = rng.standard_normal() * sd
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def make_crsp(seed: int = 0, start: str = "1990-01-02", end: str = "1995-12-29", n_permno: int = 300,
              n_pairs: int = 6, n_twins: int = 8) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    T = len(days)
    n_base = n_permno - n_twins
    n_ind = len(INDUSTRIES)
    ind = rng.integers(0, n_ind, n_base)
    ind[:n_ind] = np.arange(n_ind)                      # every industry is populated
    # ---- log-price paths: market + industry + idiosyncratic
    mkt = rng.standard_normal(T) * 0.008
    indf = rng.standard_normal((T, n_ind)) * 0.008
    idv = rng.uniform(0.010, 0.020, n_base)
    lp = np.cumsum(mkt[:, None] + indf[:, ind] + rng.standard_normal((T, n_base)) * idv, axis=0)
    clean = np.zeros(n_base, bool)                      # stocks that live the full window, no splits/drops
    truth_pairs = []
    order = rng.permutation(n_base)
    used = set()
    for _ in range(n_pairs):
        for k in range(n_ind):
            pool = [i for i in order if ind[i] == k and i not in used]
            if len(pool) >= 2:
                break
        else:
            break
        used.update(pool[:2])
        x, y = pool[0], pool[1]
        beta, hl = float(rng.uniform(0.7, 1.4)), 4.0
        lp[:, y] = 0.3 + beta * lp[:, x] + _ou(rng, T, hl, 0.04)
        clean[[x, y]] = True
        truth_pairs.append((x, y, beta, hl))
    twin_base = [i for i in order if i not in used][:n_twins]
    for b in twin_base:
        clean[b] = True
    # twin columns appended after the base stocks
    lp = np.hstack([lp, np.column_stack([lp[:, b] + _ou(rng, T, 2.0, 0.004) for b in twin_base])]) \
        if twin_base else lp
    n = lp.shape[1]
    base_of = np.r_[np.arange(n_base), np.array(twin_base, int)]
    ind_all = ind[base_of]
    permno = 10001 + np.arange(n) * 3
    permco = np.r_[20001 + np.arange(n_base), 20001 + np.array(twin_base, int)]
    ex0 = rng.choice(["N", "A", "Q"], n, p=[0.45, 0.15, 0.40])
    ex0[n_base:] = ex0[base_of[n_base:]]
    p0 = np.clip(np.exp(rng.normal(np.log(30), 0.6, n)), 8, None)
    sh0 = np.exp(rng.normal(10.5, 1.0, n))
    hs = np.where(ex0 == "Q", rng.uniform(0.002, 0.006, n), rng.uniform(0.0008, 0.003, n))
    free = ~clean_ext(clean, n_base, n)
    adr = (rng.random(n) < 0.04) & free
    thin = (rng.random(n) < 0.03) & free
    kind = np.array(["none"] * n, dtype=object)
    r = rng.random(n)
    kind[free & (r < 0.07)] = "perf"
    kind[free & (r >= 0.07) & (r < 0.14)] = "merger"
    last_ix = np.full(n, T - 1)
    first_ix = np.zeros(n, int)
    late = free & (rng.random(n) < 0.3)
    first_ix[late] = rng.integers(1, max(2, int(T * 0.6)), late.sum())
    for i in np.nonzero(kind != "none")[0]:
        last_ix[i] = rng.integers(min(first_ix[i] + 150, T - 2), T - 1) if first_ix[i] + 150 < T - 1 else T - 2
    ex_change_ix = np.full(n, -1)
    chg = free & (ex0 != "N") & (rng.random(n) < 0.2)
    ex_change_ix[chg] = rng.integers(T // 4, 3 * T // 4, chg.sum())
    sic_change_ix = np.full(n, -1)
    schg = free & (rng.random(n) < 0.15)
    sic_change_ix[schg] = rng.integers(T // 4, 3 * T // 4, schg.sum())
    split_ix = np.full(n, -1)
    spl = free & (rng.random(n) < 0.15)
    split_ix[spl] = rng.integers(T // 4, 3 * T // 4, spl.sum())
    sic0 = np.array([INDUSTRIES[k][0] for k in ind_all])
    sic1 = np.array([INDUSTRIES[rng.integers(0, n_ind)][0] for _ in range(n)])
    pair_leg = np.zeros(n, bool)
    for x, y, *_ in truth_pairs:
        pair_leg[[x, y]] = True

    frames, spells, dels, legacy, truth_ev = [], [], [], [], []
    ob_open_from = np.datetime64("1992-07-01")
    q_from = np.datetime64("1993-01-01")
    for i in range(n):
        idx = np.arange(first_ix[i], last_ix[i] + 1)
        if free[i]:                                      # dropped rows -> multi-day returns
            keep = rng.random(len(idx)) > 0.002
            keep[0] = keep[-1] = True
            idx = idx[keep]
        m = len(idx)
        d = days.values[idx]
        l = lp[idx, i]
        retx = np.r_[np.nan, np.exp(np.diff(l)) - 1]
        gap = np.r_[False, np.diff(idx) > 1]
        phase = int(rng.integers(0, 63))
        paying = (idx % 63 == phase) & ~(pair_leg[i] | (i >= n_base))
        div = np.where(paying, 0.003 + 0.003 * rng.random(m), 0.0)
        ret = np.where(np.isnan(retx), np.nan, retx + div)
        sf = np.where((split_ix[i] >= 0) & (idx >= split_ix[i]), 2.0, 1.0)
        prc = p0[i] * np.exp(l - lp[first_ix[i], i]) / sf
        shr = sh0[i] * sf
        exch = np.where((ex_change_ix[i] >= 0) & (idx >= ex_change_ix[i]), "N", ex0[i]).astype(object)
        sic = np.where((sic_change_ix[i] >= 0) & (idx >= sic_change_ix[i]), sic1[i], sic0[i])
        halt = rng.random(m) < (0.10 if thin[i] else 0.004)
        halt[0] = False
        h = np.maximum(hs[i] * prc, 0.005)
        bid, ask = prc - h, prc + h
        noq = (exch != "Q") & (d < q_from)
        bid = np.where(noq, np.nan, bid)
        ask = np.where(noq, np.nan, ask)
        opn = prc * (1 + 0.004 * rng.standard_normal(m))
        opn = np.where((d < ob_open_from) | halt | (rng.random(m) < 0.03), np.nan, opn)
        sd = idv[base_of[i]] if i < n_base else 0.01
        pr_open = np.where(np.isnan(opn), prc, opn)
        hi = np.maximum(prc, pr_open) * np.exp(0.5 * sd * np.abs(rng.standard_normal(m))) + h
        lo = np.maximum(np.minimum(prc, pr_open) * np.exp(-0.5 * sd * np.abs(rng.standard_normal(m))) - h, 0.01)
        vol = np.round(np.exp(13 + 0.5 * rng.standard_normal(m)))
        vol = np.where(halt, 0.0, vol)
        flag = np.where(halt, "BA", "TR").astype(object)
        hi = np.where(halt, np.nan, hi)
        lo = np.where(halt, np.nan, lo)
        close = np.where(halt, np.nan, prc)
        retmiss = np.full(m, None, dtype=object)
        retmiss[0] = "NP"
        dur = np.where(gap, "MD", "D1").astype(object)
        dur[0] = "NA"
        dlydel = np.full(m, "N", dtype=object)
        retx_out = retx.copy()
        ycol = None
        if kind[i] != "none":
            if kind[i] == "perf":
                code = int(rng.choice(PERF_CODES))
                miss = rng.random() < 0.5
                dlret = np.nan if miss else float(np.clip(rng.normal(-0.35, 0.25), -0.95, -0.02))
            else:
                code = int(rng.choice(MERGER_CODES))
                miss = rng.random() < 0.1
                dlret = np.nan if miss else float(rng.uniform(-0.02, 0.10))
            if kind[i] == "perf" and not miss and code == 574:
                dlret = -1.0                                 # worthless: the TR index goes to 0
            lastret = ret[-1]
            last_date = pd.Timestamp(d[-1])
            rng.random()                                     # kept so the random stream is unchanged
            # CIZ delisting row: next trading day, delisting price (DP), price 0 (DA) or nothing (DM)
            if miss:
                yflag, yprc, ydur = "DM", np.nan, "MR"
            elif kind[i] == "merger" and permno[i] % 2 == 0:
                yflag, yprc, ydur = "DA", 0.0, "DD"
            else:
                yflag, yprc, ydur = "DP", float(prc[-1] * (1 + dlret)), "D1"
            ydate = days.values[last_ix[i] + 1]
            ycol = {"dlycaldt": ydate, "dlydelflg": "Y", "dlyprc": yprc, "dlyprcflg": yflag, "dlycap": np.nan,
                    "dlyret": dlret, "dlyretx": dlret, "dlyretmissflg": "DM" if miss else None,
                    "dlyretdurflg": ydur, "dlyvol": np.nan, "dlyopen": np.nan, "dlyhigh": np.nan,
                    "dlylow": np.nan, "dlybid": np.nan, "dlyask": np.nan, "dlyclose": np.nan}
            dels.append((permno[i], last_date, code, dlret, dlret, yprc))
            truth_ev.append((permno[i], pd.Timestamp(ydate), code, dlret, kind[i]))
            legacy_last = lastret
        else:
            legacy_last = None
        sec = pd.DataFrame({
            "permno": permno[i], "permco": permco[i], "securitytype": "EQTY", "securitysubtype": "COM",
            "sharetype": "AD" if adr[i] else "NS", "usincflg": "N" if adr[i] else "Y", "issuertype": "CORP",
            "primaryexch": exch, "conditionaltype": "RW", "tradingstatusflg": "A",
            "dlycaldt": d, "dlydelflg": dlydel, "dlyprc": prc, "dlyprcflg": flag, "dlycap": prc * shr,
            "dlyret": ret, "dlyretx": retx_out, "dlyretmissflg": retmiss, "dlyretdurflg": dur, "dlyvol": vol,
            "dlyopen": opn, "dlyhigh": hi, "dlylow": lo, "dlybid": bid, "dlyask": ask, "dlyclose": close,
            "shrout": shr, "siccd": sic})
        if ycol is not None:
            yrow = sec.iloc[[-1]].copy()
            for c, v in ycol.items():
                yrow[c] = [v]
            sec = pd.concat([sec, yrow.astype({c: object for c in ("dlyretmissflg",)})], ignore_index=True)
        frames.append(sec)
        lg = pd.DataFrame({"permno": permno[i], "date": d, "ret": ret.copy(),
                           "retx": retx, "prc": prc})
        if legacy_last is not None:
            lg.loc[lg.index[-1], "ret"] = legacy_last
        legacy.append(lg)
        # name spells: break at exchange / SIC change days
        cuts = sorted({0} | {int(np.searchsorted(idx, c)) for c in (ex_change_ix[i], sic_change_ix[i]) if c >= 0})
        cuts = [c for c in cuts if c < m]
        for a, b in zip(cuts, cuts[1:] + [m]):
            spells.append((permno[i], permco[i], pd.Timestamp(d[a]),
                           pd.Timestamp(d[b]) - pd.Timedelta(days=1) if b < m else pd.Timestamp(end),
                           f"S{i:03d}", f"SYNTH CO {i}", exch[a], int(sic[a]), "EQTY", "COM",
                           "AD" if adr[i] else "NS", "N" if adr[i] else "Y", "CORP", "RW", "A"))
    dsf = pd.concat(frames, ignore_index=True)
    dsf["dlyretmissflg"] = dsf["dlyretmissflg"].astype(object)
    names = pd.DataFrame(spells, columns=["permno", "permco", "namedt", "nameenddt", "ticker", "comnam",
                                          "primaryexch", "siccd", "securitytype", "securitysubtype", "sharetype",
                                          "usincflg", "issuertype", "conditionaltype", "tradingstatusflg"])
    dl = pd.DataFrame(dels, columns=["permno", "dlstdt", "dlstcd", "dlret", "dlretx", "dlprc"])
    # GICS / CCM: one gvkey per PERMCO, GICS from 1992-01 (some switch industry in mid-1993)
    comp = sorted(set(permco))
    gv = {pc: f"{1000 + j:06d}" for j, pc in enumerate(comp)}
    hg, ccm = [], []
    first_row = dsf.groupby("permno")["dlycaldt"].min()
    for i in range(n):
        g = gv[permco[i]]
        if i >= n_base:
            continue
        k = ind_all[i]
        gi = INDUSTRIES[k][2]
        if rng.random() < 0.1 and free[i]:
            k2 = (k + 1) % n_ind
            hg.append((g, pd.Timestamp("1992-01-01"), pd.Timestamp("1993-06-30"), gi[:2], gi[:4], gi, gi + "10"))
            gi2 = INDUSTRIES[k2][2]
            hg.append((g, pd.Timestamp("1993-07-01"), pd.NaT, gi2[:2], gi2[:4], gi2, gi2 + "10"))
        else:
            hg.append((g, pd.Timestamp("1992-01-01"), pd.NaT, gi[:2], gi[:4], gi, gi + "10"))
    for i in range(n):
        g = gv[permco[i]]
        ccm.append((g, "P" if i < n_base else "C", "LC", int(permno[i]), int(permco[i]), first_row[permno[i]],
                    pd.NaT))
        if i % 25 == 0:                                  # a non-qualifying link that must be filtered out
            ccm.append((g, "J", "LD", int(permno[i]), int(permco[i]), first_row[permno[i]], pd.NaT))
    hgic = pd.DataFrame(hg, columns=["gvkey", "indfrom", "indthru", "gsector", "ggroup", "gind", "gsubind"])
    link = pd.DataFrame(ccm, columns=["gvkey", "linkprim", "linktype", "lpermno", "lpermco", "linkdt", "linkenddt"])
    tp = pd.DataFrame([(int(permno[y]), int(permno[x]), b, hl) for x, y, b, hl in truth_pairs],
                      columns=["y", "x", "beta", "hl_days"])
    tw = pd.DataFrame([(int(permno[b]), int(permno[n_base + j])) for j, b in enumerate(twin_base)],
                      columns=["permno_a", "permno_b"])
    te = pd.DataFrame(truth_ev, columns=["permno", "last_date", "dlstcd", "dlret_true", "kind"])
    return {"dsf_v2": dsf, "stocknames_v2": names, "dsedelist": dl,
            "dsf_legacy": pd.concat(legacy, ignore_index=True), "co_hgic": hgic, "ccmxpf_lnkhist": link,
            "truth_pairs": tp, "truth_events": te, "truth_twins": tw}


def clean_ext(clean: np.ndarray, n_base: int, n: int) -> np.ndarray:
    """clean flags for base stocks, extended with True for the twin columns."""
    return np.r_[clean, np.ones(n - n_base, bool)]
