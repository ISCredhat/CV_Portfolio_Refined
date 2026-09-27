"""Synthetic raw data in exactly the format written by scripts/download_data.py.

Used by the tests (and to smoke-test the whole pipeline without network access).
A known insider effect is planted so the tests can check the pipeline recovers it:
purchases by CEOs/CFOs, clusters and 'skilled' insiders are followed by positive drift.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _french_zip(path: Path, name: str, df: pd.DataFrame) -> None:
    lines = ["This file was created synthetically", "", "," + ",".join(df.columns)]
    for d, row in df.iterrows():
        lines.append(d.strftime("%Y%m%d") + "," + ",".join(f"{v * 100:.4f}" for v in row))
    lines += ["", "Copyright synthetic"]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, "\n".join(lines))


def make_synthetic_raw(raw: Path, n_tickers: int = 120, n_insiders: int = 600, seed: int = 7,
                       start: str = "2009-01-02", end: str = "2025-06-30", effect: float = 0.06) -> dict:
    rng = np.random.default_rng(seed)
    raw = Path(raw)
    (raw / "prices").mkdir(parents=True, exist_ok=True)
    (raw / "factors").mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    # market + idiosyncratic log returns
    mkt = rng.normal(0.0003, 0.011, T)
    beta = rng.uniform(0.6, 1.5, n_tickers)
    idio = rng.normal(0, 0.025, (T, n_tickers))
    logret = mkt[:, None] * beta[None, :] + idio
    # ---- insider events (planted effect) --------------------------------------
    insider_skill = rng.normal(0, 1, n_insiders)
    skilled = insider_skill > 1.0
    insider_issuer = rng.integers(0, n_tickers, n_insiders)
    roles = rng.choice(["CEO", "CFO", "Director", "10%", "VP"], n_insiders, p=[.12, .08, .5, .1, .2])
    events = []
    n_buy = 6000
    for k in range(n_buy):
        i = rng.integers(0, n_insiders)
        t_idx = rng.integers(260, T - 5)
        events.append((i, insider_issuer[i], t_idx, "P"))
        if rng.random() < 0.15:  # cluster: a second insider of the same issuer buys soon after
            j = rng.integers(0, n_insiders)
            events.append((j, insider_issuer[i], min(t_idx + rng.integers(1, 10), T - 5), "P"))
    for k in range(9000):
        i = rng.integers(0, n_insiders)
        events.append((i, insider_issuer[i], rng.integers(260, T - 5), "S"))
    ev = pd.DataFrame(events, columns=["ins", "tk", "t", "code"])
    ev["filing_t"] = np.minimum(ev["t"] + rng.integers(0, 3, len(ev)), T - 2)
    # plant drift after purchases filed (starts the session after filing)
    buys = ev[ev["code"] == "P"]
    counts = buys.groupby(["tk"])["t"].apply(list)
    for _, e in buys.iterrows():
        strength = effect * (0.5 + (roles[e.ins] in ("CEO", "CFO")) + 1.5 * skilled[e.ins])
        s, f = e.filing_t + 1, min(e.filing_t + 64, T)
        logret[s:f, e.tk] += strength / 63.0
    close_split = np.exp(np.log(rng.uniform(5, 80, n_tickers))[None, :] + np.cumsum(logret, axis=0))
    # one 2:1 split in a few tickers -> Yahoo 'close' is split-adjusted, raw prices are not
    splits = np.zeros((T, n_tickers))
    for tk in range(0, n_tickers, 10):
        splits[rng.integers(T // 3, T - 10), tk] = 2.0
    s = np.where(splits > 0, splits, 1.0)
    after = np.log(s).sum(0) - np.cumsum(np.log(s), 0)
    raw_px = close_split * np.exp(after)          # as-traded price
    open_px = close_split * np.exp(rng.normal(0, 0.004, (T, n_tickers)))
    vol = rng.lognormal(12, 1, (T, n_tickers)).round()
    # delist the last 10 tickers two-thirds of the way through
    for tk in range(n_tickers - 10, n_tickers):
        cut = int(T * 0.66)
        close_split[cut:, tk] = np.nan
        open_px[cut:, tk] = np.nan
    # benchmarks
    bench = {"SPY": np.exp(np.cumsum(mkt)) * 100, "IWM": np.exp(np.cumsum(mkt * 1.1 + rng.normal(0, 0.003, T))) * 80}
    frames = []
    for j, tk in enumerate(tickers):
        frames.append(pd.DataFrame({"date": dates, "ticker": tk, "open": open_px[:, j], "close": close_split[:, j],
                                    "adj_close": close_split[:, j], "volume": vol[:, j],
                                    "dividends": 0.0, "splits": splits[:, j]}).dropna(subset=["close"]))
    for tk, p in bench.items():
        frames.append(pd.DataFrame({"date": dates, "ticker": tk, "open": p * (1 + rng.normal(0, 0.002, T)), "close": p,
                                    "adj_close": p, "volume": 1e8, "dividends": 0.0, "splits": 0.0}))
    long = pd.concat(frames)
    half = len(long) // 2
    long.iloc[:half].to_csv(raw / "prices" / "batch_0001.csv.gz", index=False, compression="gzip")
    long.iloc[half:].to_csv(raw / "prices" / "batch_0002.csv.gz", index=False, compression="gzip")
    # ---- SEC lines ------------------------------------------------------------
    rel_map = {"CEO": ("Director,Officer", "CEO & President"), "CFO": ("Officer", "Chief Financial Officer"),
               "Director": ("Director", ""), "10%": ("TenPercentOwner", ""), "VP": ("Officer", "VP Sales")}
    rows = []
    for n, e in enumerate(ev.itertuples()):
        rel, title = rel_map[roles[e.ins]]
        p = raw_px[e.t, e.tk] * (1 + rng.normal(0, 0.01))
        if not np.isfinite(p):
            continue
        sh = int(rng.lognormal(8, 1)) + 10
        rows.append({
            "QUARTER": "synthetic", "ACCESSION_NUMBER": f"0000-{n:07d}", "SECURITY_TITLE": "Common Stock",
            "TRANS_DATE": dates[e.t].strftime("%d-%b-%Y").upper(), "TRANS_FORM_TYPE": "4",
            "TRANS_CODE": e.code, "EQUITY_SWAP_INVOLVED": "0", "TRANS_TIMELINESS": "",
            "TRANS_SHARES": sh, "TRANS_PRICEPERSHARE": round(p, 4),
            "TRANS_ACQUIRED_DISP_CD": "A" if e.code == "P" else "D",
            "SHRS_OWND_FOLWNG_TRANS": sh * int(rng.integers(2, 50)), "DIRECT_INDIRECT_OWNERSHIP": "D",
            "NATURE_OF_OWNERSHIP": "", "FILING_DATE": dates[e.filing_t].strftime("%d-%b-%Y").upper(),
            "PERIOD_OF_REPORT": "", "DOCUMENT_TYPE": "4", "ISSUERCIK": str(1000 + e.tk),
            "ISSUERNAME": f"Company {e.tk}", "ISSUERTRADINGSYMBOL": tickers[e.tk].lower() if e.tk % 7 else "NONE",
            "AFF10B5ONE": "", "RPTOWNERCIK": str(50000 + e.ins), "RPTOWNERNAME": f"Insider {e.ins}",
            "RPTOWNER_RELATIONSHIP": rel, "RPTOWNER_TITLE": title, "N_OWNERS": 1})
    pd.DataFrame(rows).to_csv(raw / "form4_ps_all.csv.gz", index=False, compression="gzip")
    # current-ticker map (lets the pipeline recover the 'NONE' symbols)
    cmap = {str(i): {"cik_str": 1000 + i, "ticker": tickers[i], "title": f"Company {i}"} for i in range(n_tickers)}
    (raw / "sec_company_tickers.json").write_text(json.dumps(cmap))
    # ---- factors -------------------------------------------------------------------
    ff = pd.DataFrame({"Mkt-RF": mkt - 0.00008, "SMB": rng.normal(0, .005, T), "HML": rng.normal(0, .005, T),
                       "RMW": rng.normal(0, .004, T), "CMA": rng.normal(0, .004, T), "RF": 0.00008}, index=dates)
    _french_zip(raw / "factors" / "ff5_daily.zip", "F-F_Research_Data_5_Factors_2x3_daily.CSV", ff)
    _french_zip(raw / "factors" / "mom_daily.zip", "F-F_Momentum_Factor_daily.CSV",
                pd.DataFrame({"Mom": rng.normal(0, .006, T)}, index=dates))
    return {"skilled": skilled, "roles": roles}
