"""Run the 11 "verify on WRDS" checks of the plan review against dsf_v2 / stocknames_v2 and friends.

    python scripts/check_wrds_crsp.py [--username U] [--only 1 3 5] [--quick]

One connection (one Duo push), read-only SELECTs, each wrapped so one failure does not stop the others.
Writes ``data/reference/wrds_crsp_check.json`` and ``wrds_crsp_check.md`` plus the column lists
``dsf_v2_columns.json`` / ``stocknames_v2_columns.json``. ``--quick`` restricts the heavy scans to 2000-2003.
A query whose columns do not exist is recorded as "skipped" with the missing names (discover first, then run).
Exit codes: 0 done, 1 connection failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pairs import config_crsp as CC  # noqa: E402
from pairs import wrds_io as W  # noqa: E402

EXPECT_DSF = ["dlyopen", "dlybid", "dlyask", "dlyhigh", "dlylow", "dlyclose", "dlyprcflg", "dlycap", "dlyfacprc",
              "dlydelflg", "dlyretmissflg", "dlyretdurflg", "shrout", "siccd", "primaryexch", "sharetype",
              "securitytype", "securitysubtype", "usincflg", "issuertype", "conditionaltype", "tradingstatusflg"]
HEAVY_LIMIT = 300


def build_queries(dsf_cols: list[str], names_cols: list[str], quick: bool = False) -> list[dict]:
    """The queries of review 'Verify on WRDS' items 2-11 (item 1 is describe_table). Each: id, title, sql, needs
    (columns required in dsf_v2 / stocknames_v2; the query is skipped when one is missing)."""
    d0, d1 = ("2000-01-01", "2003-12-31") if quick else ("1990-01-01", "2010-12-31")
    o0, o1 = ("2000-01-01", "2003-12-31") if quick else ("1988-01-01", "2005-12-31")
    dsf, nm = "crsp.dsf_v2", "crsp.stocknames_v2"
    end = "nameenddt" if "nameenddt" in names_cols else "nameendt"
    q = [
        dict(id=2, title="Are classification columns point in time in dsf_v2? (count of PERMNOs with >1 value)",
             needs=["primaryexch", "siccd"],
             sql=f"SELECT COUNT(*) AS n_changing FROM (SELECT permno FROM {dsf} WHERE dlycaldt BETWEEN '{d0}' AND "
                 f"'{d1}' GROUP BY permno HAVING COUNT(DISTINCT primaryexch) > 1 OR COUNT(DISTINCT siccd) > 1) t"),
        dict(id=3, title="First date with an open price, by exchange", needs=["dlyopen", "primaryexch"],
             sql=f"SELECT primaryexch, MIN(dlycaldt) AS first_open FROM {dsf} WHERE dlyopen > 0 GROUP BY 1"),
        dict(id="3b", title="Open coverage by year and exchange", needs=["dlyopen", "primaryexch", "securitytype"],
             sql=f"SELECT date_part('year', dlycaldt) AS y, primaryexch, AVG((dlyopen > 0)::int) AS share_open "
                 f"FROM {dsf} WHERE dlycaldt BETWEEN '{o0}' AND '{o1}' AND securitytype = 'EQTY' GROUP BY 1, 2 "
                 f"ORDER BY 1, 2"),
        dict(id=4, title="Closing quote and high/low coverage by year and exchange",
             needs=["dlybid", "dlyask", "dlyhigh", "dlylow", "primaryexch", "securitytype"],
             sql=f"SELECT date_part('year', dlycaldt) AS y, primaryexch, AVG((dlybid > 0 AND dlyask > dlybid)::int) "
                 f"AS share_quote, AVG((dlyhigh > 0 AND dlylow > 0)::int) AS share_hl FROM {dsf} WHERE dlycaldt "
                 f"BETWEEN '{o0}' AND '{o1}' AND securitytype = 'EQTY' GROUP BY 1, 2 ORDER BY 1, 2"),
        dict(id=5, title="Price convention: flag values, counts, minimum price (1995)", needs=["dlyprcflg", "dlyprc"],
             sql=f"SELECT dlyprcflg, COUNT(*) AS n, MIN(dlyprc) AS min_prc FROM {dsf} WHERE dlycaldt BETWEEN "
                 f"'1995-01-01' AND '1995-12-31' GROUP BY 1"),
        dict(id=6, title="Delisting rows: flag vs missing returns", needs=["dlydelflg", "dlyret"],
             sql=f"SELECT dlydelflg, COUNT(*) AS n, AVG((dlyret IS NULL)::int) AS share_ret_null FROM {dsf} WHERE "
                 f"dlycaldt >= '1988-01-01' GROUP BY 1"),
        dict(id="6b", title="CIZ delisting row date vs dsedelist.dlstdt", needs=["dlydelflg"],
             sql=f"SELECT COUNT(*) AS n_events, SUM((l.d <> e.dlstdt)::int) AS n_date_differs FROM (SELECT permno, "
                 f"MAX(dlycaldt) AS d FROM {dsf} WHERE dlydelflg = 'Y' GROUP BY permno) l JOIN crsp.dsedelist e "
                 f"ON e.permno = l.permno"),
        dict(id="6c", title="Missing legacy dlret by dlstcd/100 and decade", needs=[],
             sql="SELECT dlstcd / 100 AS code100, floor(date_part('year', dlstdt) / 10) * 10 AS decade, COUNT(*) "
                 "AS n, SUM((dlret IS NULL)::int) AS n_missing FROM crsp.dsedelist WHERE dlstdt >= '1988-01-01' "
                 "GROUP BY 1, 2 ORDER BY 1, 2"),
        dict(id=7, title="Return duration / missing flags", needs=["dlyretdurflg", "dlyretmissflg"],
             sql=f"SELECT dlyretdurflg, dlyretmissflg, COUNT(*) AS n FROM {dsf} WHERE dlycaldt >= '1988-01-01' "
                 f"GROUP BY 1, 2 ORDER BY 3 DESC"),
        dict(id=8, title="stocknames_v2 classification vs legacy shrcd/exchcd (same permno, overlapping dates)",
             needs=[], names_needs=["sharetype", "securitytype", "securitysubtype", "usincflg", "issuertype"],
             sql=f"SELECT n.sharetype, n.securitytype, n.securitysubtype, n.usincflg, n.issuertype, o.shrcd, "
                 f"COUNT(*) AS n FROM {nm} n JOIN crsp.stocknames o ON o.permno = n.permno AND n.namedt <= "
                 f"o.nameenddt AND n.{end} >= o.namedt GROUP BY 1, 2, 3, 4, 5, 6 ORDER BY 7 DESC LIMIT {HEAVY_LIMIT}"),
        dict(id=9, title="Rows and distinct PERMNOs by year (EQTY)", needs=["securitytype"],
             sql=f"SELECT date_part('year', dlycaldt) AS y, COUNT(*) AS n_rows, COUNT(DISTINCT permno) AS n_permno "
                 f"FROM {dsf} WHERE dlycaldt >= '{('2000-01-01' if quick else CC.DOWNLOAD_START)}' AND "
                 f"securitytype = 'EQTY' GROUP BY 1 ORDER BY 1"),
        dict(id=11, title="GICS history start and companies", needs=[],
             sql="SELECT MIN(indfrom) AS first_indfrom, COUNT(DISTINCT gvkey) AS n_gvkey FROM comp.co_hgic"),
    ]
    for item in q:
        item["missing"] = [c for c in item.get("needs", []) if c not in dsf_cols] + \
                          [c for c in item.get("names_needs", []) if c not in names_cols]
    return q


def library_checks(conn) -> dict:
    """Item 10: list_tables('ff') and ('crsp') filtered to the names that matter."""
    out = {}
    for lib, pats in (("ff", None), ("crsp", ("dsi", "msi", "dsf", "stocknames", "dsedelist", "ccm"))):
        try:
            tabs = sorted(str(t) for t in conn.list_tables(library=lib))
            out[lib] = [t for t in tabs if pats is None or any(p in t for p in pats)]
        except Exception as e:
            out[lib] = f"error: {type(e).__name__}: {e}"
    return out


def run(conn, only=None, quick=False) -> dict:
    res: dict = {"generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
                 "quick": quick, "checks": {}}
    cols = {}
    for tab in ("crsp.dsf_v2", "crsp.stocknames_v2"):
        try:
            cols[tab] = W.describe_columns(conn, tab)
        except Exception as e:
            cols[tab] = []
            res["checks"][f"1:{tab}"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    dsf_cols, names_cols = cols["crsp.dsf_v2"], cols["crsp.stocknames_v2"]
    res["checks"]["1"] = {"status": "ok" if dsf_cols else "error", "title": "describe_table dsf_v2 / stocknames_v2",
                          "dsf_v2_columns": dsf_cols, "stocknames_v2_columns": names_cols,
                          "expected_dsf_present": {c: c in dsf_cols for c in EXPECT_DSF},
                          "dlycumfacpr_exists": "dlycumfacpr" in dsf_cols,
                          "siccd_in": {"dsf_v2": "siccd" in dsf_cols, "stocknames_v2": "siccd" in names_cols},
                          "name_spell_columns": [c for c in names_cols if c in ("namedt", "nameenddt", "nameendt")]}
    if only is None or 10 in only:
        res["checks"]["10"] = {"status": "ok", "title": "tables in ff and crsp", **library_checks(conn)}
    try:
        conn.raw_sql("SET statement_timeout = 900000")                 # 15 min per query; harmless if refused
    except Exception:
        pass
    for item in build_queries(dsf_cols, names_cols, quick):
        base = int(str(item["id"]).rstrip("bc"))
        if only is not None and base not in only:
            continue
        rec = {"title": item["title"], "sql": item["sql"]}
        if item["missing"]:
            rec.update(status="skipped", reason=f"columns missing: {item['missing']}")
        else:
            try:
                df = conn.raw_sql(item["sql"])
                rec.update(status="ok", n_rows=int(len(df)), rows=json.loads(df.head(HEAVY_LIMIT).to_json(
                    orient="records", date_format="iso")))
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
        res["checks"][str(item["id"])] = rec
    return res


def to_markdown(res: dict) -> str:
    L = [f"# WRDS CRSP checks ({res['generated_utc']}{', quick' if res.get('quick') else ''})", ""]
    for k, r in res["checks"].items():
        L += [f"## {k}. {r.get('title', '')}", f"status: {r['status']}", ""]
        if k == "1":
            L += [f"- dlycumfacpr exists: {r['dlycumfacpr_exists']}", f"- siccd in: {r['siccd_in']}",
                  f"- name spell columns: {r['name_spell_columns']}",
                  "- expected dsf_v2 columns missing: " + str([c for c, v in r["expected_dsf_present"].items() if not v]),
                  f"- dsf_v2 has {len(r['dsf_v2_columns'])} columns, stocknames_v2 {len(r['stocknames_v2_columns'])}", ""]
        elif k == "10":
            L += [f"- ff tables: {r.get('ff')}", f"- crsp tables of interest: {r.get('crsp')}", ""]
        elif r["status"] == "ok":
            rows = r["rows"][:40]
            if rows:
                head = list(rows[0])
                L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                L += ["| " + " | ".join(str(x.get(h)) for h in head) + " |" for x in rows]
            L += [f"({r['n_rows']} rows)", ""]
        else:
            L += [f"{r.get('reason') or r.get('error')}", ""]
        if "sql" in r:
            L += ["```sql", r["sql"], "```", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--username", default=None)
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=str(CC.REFERENCE))
    args = ap.parse_args(argv)
    try:
        conn = W.connect(args.username)
    except Exception as e:
        print(f"could not connect to WRDS ({type(e).__name__}: {e})", file=sys.stderr)
        return 1
    try:
        res = run(conn, args.only, args.quick)
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wrds_crsp_check.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    (out / "wrds_crsp_check.md").write_text(to_markdown(res) + "\n")
    c1 = res["checks"].get("1", {})
    for name, key in (("dsf_v2_columns.json", "dsf_v2_columns"), ("stocknames_v2_columns.json", "stocknames_v2_columns")):
        (out / name).write_text(json.dumps(c1.get(key, []), indent=2) + "\n")
    for k, r in res["checks"].items():
        print(f"{k:4s} {r['status']}")
    print(f"wrote {out / 'wrds_crsp_check.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
