from datetime import datetime, timedelta
from functools import partial

import polars as pl

from pmdetect.markets import newstime
from pmdetect.quality import integrity
from pmdetect.store import db

T0 = datetime(2025, 1, 1)


def pts(*vals, step_h=1):
    return [(T0 + timedelta(hours=i * step_h), v) for i, v in enumerate(vals)]


def test_stable_cross_ignores_early_spike():
    p = pts(0.5, 0.95, 0.6, 0.92, 0.97, 1.0)
    ts, status = newstime.stable_cross(p)
    assert status == "ok" and ts == T0 + timedelta(hours=3)


def test_stable_cross_edge_cases():
    assert newstime.stable_cross(pts(0.95, 0.99))[1] == "always_above"
    assert newstime.stable_cross(pts(0.95, 0.5)) == (None, "no_cross")
    assert newstime.stable_cross([]) == (None, "no_data")


def test_prefers_fine_prices_else_trades():
    fine_prices = pts(0.5, 0.95, 1.0, step_h=0.25)
    trades = pts(0.5, 0.6, 0.97, 1.0)
    assert newstime.estimate_for_market(fine_prices, trades)[1] == "price_cross_0.9"
    coarse = pts(0.5, 0.95, 1.0, step_h=12)
    ts, method = newstime.estimate_for_market(coarse, trades)
    assert method == "trade_cross_0.9" and ts == T0 + timedelta(hours=2)
    assert newstime.estimate_for_market(coarse, [])[1] == "price_cross_0.9"


def test_overrides_and_estimate(tmp_path, monkeypatch):
    path = tmp_path / "t.duckdb"
    monkeypatch.setattr(newstime, "connect", partial(db.connect, path))
    csv = tmp_path / "ov.csv"
    csv.write_text(
        "condition_id,slug,news_ts,method,source_url,notes\n"
        ",sched,2025-10-10T09:00:00Z,scheduled_reveal,,\n"
    )
    con = db.connect(path)
    for cid, slug, outcome in [("0xa", "sched", "YES"), ("0xb", "auto", "YES"), ("0xc", "x", None)]:
        con.execute(
            "INSERT INTO markets (condition_id, slug, outcome, in_universe) VALUES (?,?,?,TRUE)",
            [cid, slug, outcome],
        )
    con.execute("INSERT INTO tokens VALUES ('tyes','0xb','Yes'),('tno','0xb','No')")
    for cid in ("0xb", "0xc"):
        con.execute("INSERT INTO fetch_log VALUES ('trades', ?, NULL, TRUE, ?)", [cid, T0])
    for i, p in enumerate([0.4, 0.95, 0.99]):
        con.execute("INSERT INTO prices VALUES ('tyes', ?, ?)", [T0 + timedelta(minutes=10 * i), p])
    con.close()
    assert newstime.estimate_news_times(csv) == 2
    con = db.connect(path)
    rows = {
        r[0]: r[1:]
        for r in con.execute("SELECT condition_id, news_ts, news_ts_method FROM markets").fetchall()
    }
    assert rows["0xa"] == (datetime(2025, 10, 10, 9), "override_scheduled_reveal")
    assert rows["0xb"] == (T0 + timedelta(minutes=10), "price_cross_0.9")
    assert rows["0xc"] == (None, "no_outcome")


def test_integrity_report(tmp_path, monkeypatch):
    path = tmp_path / "t.duckdb"
    monkeypatch.setattr(integrity, "connect", partial(db.connect, path))
    monkeypatch.setattr(integrity, "RESULTS_DIR", tmp_path / "res")
    monkeypatch.setattr(integrity, "REPORT_CSV", tmp_path / "res" / "r.csv")
    monkeypatch.setattr(integrity, "SUMMARY_MD", tmp_path / "res" / "s.md")
    con = db.connect(path)
    markets = [("0xok", 625.0), ("0xlow", 5000.0), ("0xfew", 100.0)]
    for cid, vol in markets:
        con.execute(
            "INSERT INTO markets (condition_id, slug, volume_usd, in_universe, start_ts, "
            "closed_ts) "
            "VALUES (?, ?, ?, TRUE, ?, ?)",
            [cid, cid, vol, T0, T0 + timedelta(days=30)],
        )
        con.execute("INSERT INTO fetch_log VALUES ('trades', ?, NULL, TRUE, ?)", [cid, T0])
    n = 0
    # Gamma volume = SUM(size)/2 (share notional). 0xok: 25 rows x 50 shares / 2 = 625 ;
    # 0xlow: same tape vs Gamma 5000 ; 0xfew: 5 trades
    for cid, count in [("0xok", 25), ("0xlow", 25), ("0xfew", 5)]:
        for i in range(count):
            n += 1
            con.execute(
                "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    f"id{n}",
                    "tx",
                    cid,
                    "tok",
                    f"w{i % 3}",
                    "BUY",
                    "Yes",
                    50.0,
                    0.8,
                    40.0,
                    T0 + timedelta(hours=i),
                ],
            )
    con.close()
    csv_path = integrity.run_integrity_report(with_neg_risk=False)
    df = pl.read_csv(csv_path).sort("condition_id")
    status = dict(zip(df["condition_id"], df["status"], strict=True))
    assert status == {"0xfew": "ok", "0xlow": "tape_below_gamma", "0xok": "ok"} or (
        status["0xok"] == "ok" and status["0xlow"] == "tape_below_gamma"
    )
    keep = dict(zip(df["condition_id"], df["keep_20"], strict=True))
    assert keep == {"0xfew": False, "0xlow": True, "0xok": True}
    con = db.connect(path)
    assert (
        con.execute("SELECT in_universe FROM markets WHERE condition_id='0xfew'").fetchone()[0]
        is False
    )
    assert (
        con.execute("SELECT in_universe FROM markets WHERE condition_id='0xok'").fetchone()[0]
        is True
    )
    md = (tmp_path / "res" / "s.md").read_text()
    assert "Dropped by the at-least-20-trades rule: **1**" in md
    assert "1 of 2" in md


def test_neg_risk_status(tmp_path):
    path = tmp_path / "t.duckdb"
    con = db.connect(path)
    con.execute(
        "INSERT INTO markets (condition_id, slug, volume_usd, in_universe) VALUES ('0xn','n',5000,TRUE)"
    )
    con.execute("INSERT INTO fetch_log VALUES ('trades','0xn',NULL,TRUE,?)", [T0])
    for i in range(25):
        con.execute(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                f"i{i}",
                "tx",
                "0xn",
                "tok",
                "w",
                "BUY",
                "Yes",
                50.0,
                0.8,
                40.0,
                T0 + timedelta(hours=i),
            ],
        )
    df = integrity.build_report(con, neg_risk={"0xn": True})
    assert df["status"].to_list() == ["tape_below_gamma_negrisk"]
    assert df["neg_risk"].to_list() == [True]
