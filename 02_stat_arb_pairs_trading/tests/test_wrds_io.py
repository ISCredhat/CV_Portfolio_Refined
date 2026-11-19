"""wrds_io, download_wrds.py and check_wrds_crsp.py against a fake connection (no network)."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pairs import wrds_io as W  # noqa: E402
from pairs.synthetic_crsp import make_crsp  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeConn:
    """Serves synthetic tables; describe_table reports the columns of the frame behind each table."""
    def __init__(self, tables: dict[str, pd.DataFrame], hide_cols=()):
        self.t, self.sql, self.hide = tables, [], set(hide_cols)

    def describe_table(self, lib, table):
        key = f"{lib}.{table}"
        if key not in self.t:
            raise ValueError("no such table")
        return pd.DataFrame({"name": [c for c in self.t[key].columns if c not in self.hide]})

    def list_tables(self, library):
        return [k.split(".")[1] for k in self.t if k.startswith(library + ".")]

    def raw_sql(self, sql, date_cols=None):
        self.sql.append(sql)
        if "dlycaldt BETWEEN" in sql:
            a, b = sql.split("BETWEEN '")[1].split("'")[0], sql.split("AND '")[-1].split("'")[0]
            df = self.t["crsp.dsf_v2"]
            return df[(df["dlycaldt"] >= a) & (df["dlycaldt"] <= b)].copy()
        for key, df in self.t.items():
            if f"FROM {key}" in sql:
                return df.copy()
        return pd.DataFrame({"x": [1]})


@pytest.fixture(scope="module")
def d():
    return make_crsp(1, start="1990-01-02", end="1991-12-31", n_permno=30)


@pytest.fixture
def conn(d):
    return FakeConn({"crsp.dsf_v2": d["dsf_v2"], "crsp.stocknames_v2": d["stocknames_v2"],
                     "crsp.dsedelist": d["dsedelist"], "comp.co_hgic": d["co_hgic"],
                     "crsp.ccmxpf_lnkhist": d["ccmxpf_lnkhist"], "crsp.dsi": pd.DataFrame(
                         {"date": pd.bdate_range("1990-01-02", periods=5), "vwretd": 0.001})})


def test_build_sql_selects_existing_columns_only():
    cols = ["permno", "dlycaldt", "dlyret", "dlyprc", "dlyvol", "securitytype", "dlyopen", "dlycumfacpr"]
    sql = W.build_sql("dsf_v2", "crsp.dsf_v2", cols, start="1990-01-01", end="1990-12-31")
    assert "securitytype = 'EQTY'" in sql and "dlyopen" in sql and "dlybid" not in sql
    assert "dlycumfacpr" not in sql and "BETWEEN '1990-01-01' AND '1990-12-31'" in sql


def test_build_sql_required_missing_raises_and_alias():
    with pytest.raises(KeyError, match="dlyret"):
        W.build_sql("dsf_v2", "crsp.dsf_v2", ["permno", "dlycaldt", "dlyprc", "dlyvol", "securitytype"])
    base = ["permno", "namedt", "primaryexch", "siccd"]
    with pytest.raises(KeyError, match="nameenddt"):
        W.build_sql("stocknames_v2", "crsp.stocknames_v2", base)
    assert "nameendt" in W.build_sql("stocknames_v2", "crsp.stocknames_v2", base + ["nameendt"])
    assert "dlyret" not in W.build_sql("market_index", "crsp.dsi", ["date", "vwretd"])


def test_chunk_plan():
    ch = W.chunk_plan("dsf_v2", "1961-07-01", "2025-12-31")
    assert len(ch) == 65 and ch[0]["name"] == "y1961" and ch[0]["params"]["start"] == "1961-07-01"
    assert ch[-1]["params"]["end"] == "2025-12-31" and ch[1]["params"]["start"] == "1962-01-01"
    assert W.chunk_plan("stocknames_v2")[0]["name"] == "all"


def test_probe_missing_table_and_hidden_column(conn):
    info = W.probe(conn)
    assert info["dsf_v2"]["table"] == "crsp.dsf_v2" and info["market_index"]["table"] == "crsp.dsi"
    assert info["ff_factors_daily"]["table"] is None and W.missing_required(info) == []
    c2 = FakeConn(conn.t, hide_cols=["dlyret"])
    info2 = W.probe(c2)
    with pytest.raises(KeyError, match="dlyret"):
        W.build_sql("dsf_v2", info2["dsf_v2"]["table"], info2["dsf_v2"]["columns"])
    del conn.t["crsp.dsedelist"]
    assert W.missing_required(W.probe(conn)) == ["dsedelist"]


def test_download_resumable_manifest_force(conn, tmp_path):
    info = W.probe(conn, ["dsf_v2"])["dsf_v2"]
    r1 = W.download_key(conn, "dsf_v2", info, tmp_path, "1990-01-01", "1991-12-31")
    assert [r["name"] for r in r1] == ["y1990", "y1991"] and all(not r["skipped"] for r in r1) and len(conn.sql) == 2
    df = pd.read_parquet(tmp_path / "dsf_v2" / "y1990.parquet")
    assert str(df["dlycaldt"].dtype).startswith("datetime64") and df["permno"].dtype == "int64"
    assert df["dlycaldt"].dt.year.eq(1990).all()
    man = json.loads((tmp_path / "manifest.json").read_text())["dsf_v2"]
    c = man["chunks"]["y1990"]
    assert c["rows"] == len(df) and len(c["sha256"]) == 64 and len(c["sql_sha256"]) == 64 and c["seconds"] >= 0
    r2 = W.download_key(conn, "dsf_v2", info, tmp_path, "1990-01-01", "1991-12-31")
    assert all(r["skipped"] for r in r2) and len(conn.sql) == 2                   # resumable: no new query
    assert r2[0]["rows"] == c["rows"] and r2[0]["sql_sha256"] == c["sql_sha256"]
    W.download_key(conn, "dsf_v2", info, tmp_path, "1990-01-01", "1991-12-31", force=True)
    assert len(conn.sql) == 4
    (tmp_path / "dsf_v2" / "y1991.parquet").unlink()                             # partial state: only the hole is fetched
    r3 = W.download_key(conn, "dsf_v2", info, tmp_path, "1990-01-01", "1991-12-31")
    assert [r["skipped"] for r in r3] == [True, False]


def test_dtypes_normalised(d):
    raw = d["co_hgic"].copy()
    raw["gvkey"] = raw["gvkey"].astype(int)
    out = W.normalise_dtypes(raw)
    assert out["gvkey"].str.len().eq(6).all()
    ds = W.normalise_dtypes(d["dsf_v2"].head(100).rename(columns=str.upper))
    assert "dlycaldt" in ds.columns and ds["dlyprc"].dtype == "float64"


FRENCH = """This file was created by CMPT_ME_BEME_RETS using the 202403 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
20240102,  0.50, -0.10, 0.20, 0.00, 0.10, 0.02
20240103, -1.00,  0.30, -0.20, 0.10, -0.10, 0.02

Copyright 2024 Kenneth R. French
"""


def test_parse_french_and_fetch_cache(tmp_path):
    df = W.parse_french_csv(FRENCH)
    assert df.shape == (2, 7) and abs(df["mktrf"].iloc[1] + 0.01) < 1e-12 and "rmw" in df.columns
    mom = "Momentum\n\n,Mom\n20240102, 0.3\n20240103, -0.2\n\nfooter\n"
    strev = "x\n\n,ST_Rev\n20240102, 0.1\n20240103, 0.2\n"
    calls = []

    def opener(url, timeout=0):
        calls.append(url)
        name = url.rsplit("/", 1)[1]
        txt = FRENCH if "5_Factors" in name else mom if "Momentum" in name else strev
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("data.CSV", txt)
        return _Ctx(buf.getvalue())
    out = W.fetch_french(tmp_path, opener=opener)
    assert {"mktrf", "mom", "strev", "rmw"} <= set(out.columns) and len(out) == 2 and len(calls) == 3
    W.fetch_french(tmp_path, opener=opener)
    assert len(calls) == 3                                                         # cached zips reused
    man = json.loads((tmp_path / "french_manifest.json").read_text())
    assert len(man) == 3 and all(len(v) == 64 for v in man.values())


class _Ctx(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_script_main(conn, tmp_path, monkeypatch):
    mod = load_script("download_wrds")
    monkeypatch.setattr(mod.W, "connect", lambda username=None: conn)
    rc = mod.main(["--raw-dir", str(tmp_path), "--start", "1990-01-01", "--end", "1990-12-31",
                   "--only", "dsf_v2", "stocknames_v2", "dsedelist", "market_index", "co_hgic"])
    assert rc == 0 and (tmp_path / "dsf_v2" / "y1990.parquet").exists()
    assert (tmp_path / "market_index" / "all.parquet").exists()
    assert json.loads((tmp_path / "manifest.json").read_text()).keys() >= {"dsf_v2", "dsedelist"}
    n = len(conn.sql)
    assert mod.main(["--raw-dir", str(tmp_path), "--start", "1990-01-01", "--end", "1990-12-31", "--only", "dsf_v2"]) == 0
    assert len(conn.sql) == n                                                      # resumed, nothing re-fetched
    conn.t.pop("crsp.dsedelist")
    assert mod.main(["--raw-dir", str(tmp_path), "--dry-run", "--only", "dsedelist"]) == 2


def test_check_script_runs_and_skips_missing_columns(conn, tmp_path, monkeypatch):
    mod = load_script("check_wrds_crsp")
    cols = list(conn.t["crsp.dsf_v2"].columns)
    q = mod.build_queries(cols, list(conn.t["crsp.stocknames_v2"].columns))
    assert len(q) >= 11 and all(not i["missing"] for i in q if i["id"] in (2, 3, 5, 7))
    q2 = mod.build_queries([c for c in cols if c != "dlyopen"], cols)
    assert [i["missing"] for i in q2 if i["id"] == 3] == [["dlyopen"]]
    res = mod.run(conn, quick=True)
    assert res["checks"]["1"]["dlycumfacpr_exists"] is False and res["checks"]["1"]["status"] == "ok"
    assert res["checks"]["3"]["status"] in ("ok", "error") and "10" in res["checks"]
    md = mod.to_markdown(res)
    assert "## 1." in md and "```sql" in md
    monkeypatch.setattr(mod.W, "connect", lambda username=None: conn)
    assert mod.main(["--out-dir", str(tmp_path), "--quick", "--only", "1", "10"]) == 0
    assert json.loads((tmp_path / "wrds_crsp_check.json").read_text())["checks"]["1"]["status"] == "ok"
    assert (tmp_path / "wrds_crsp_check.md").exists() and (tmp_path / "dsf_v2_columns.json").exists()
