"""Hermetic tests for the Live Fleet reader — registry, SQL, guard, shaping.

Nothing here touches the VPS or the network. The SELECT plan is run against
throwaway in-memory SQLite books built to the real schemas, so a column rename
or a broken filter fails here rather than on the page.
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_DASH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DASH))

from data import fleet_collector_remote as rc      # noqa: E402
from data.fleet_registry import (                  # noqa: E402
    BOOKS, BOOKS_BY_KEY, build_agg_sql, build_open_sql, build_spec, build_trades_sql,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Registry hygiene
# ══════════════════════════════════════════════════════════════════════════════
def test_keys_and_labels_are_unique():
    keys = [b.key for b in BOOKS]
    assert len(keys) == len(set(keys)), "duplicate book key"
    labels = [b.label for b in BOOKS]
    assert len(labels) == len(set(labels)), "duplicate book label"


def test_every_book_declares_a_known_tier():
    assert {b.tier for b in BOOKS} <= {1, 2, 3}


def test_tier3_books_are_dimensionless():
    """Shadow books must not carry dollars — sizing corrupts the edge measure."""
    for b in BOOKS:
        if b.tier == 3:
            assert b.pnl is None, f"{b.key} is Tier 3 but books dollars"


def test_every_book_has_r_or_dollars():
    for b in BOOKS:
        assert b.r or b.pnl, f"{b.key} measures nothing"


def test_spec_covers_every_book():
    ids = {q["id"] for q in build_spec()}
    for b in BOOKS:
        assert f"{b.key}::agg" in ids
        assert f"{b.key}::trades" in ids
        assert f"{b.key}::buckets" in ids


def test_the_alpaca_equities_seat_is_on_the_roster():
    b = BOOKS_BY_KEY["analyst_drift"]
    assert b.tier == 2 and b.seat == "analyst_drift_paper"
    assert b.pnl == "pnl_usd" and b.r is None          # a $ book, no stop-defined R
    assert b.reconcilable is False                     # equities, not ByBit linear


# ══════════════════════════════════════════════════════════════════════════════
#  The read-only guard — it must fail CLOSED
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("sql", [
    "DELETE FROM trades",
    "UPDATE trades SET r=1",
    "DROP TABLE trades",
    "INSERT INTO t VALUES (1)",
    "SELECT 1; DROP TABLE t",
    "SELECT 1 -- comment",
    "PRAGMA writable_schema=ON",
    "ATTACH DATABASE 'x' AS y",
    "VACUUM",
    "SELECT * INTO other FROM t",
])
def test_guard_blocks_writes(sql):
    with pytest.raises(ValueError):
        rc._assert_readonly(sql)


@pytest.mark.parametrize("col", ["created_utc", "updated_at", "into_price"])
def test_guard_allows_innocent_columns(col):
    """`created_utc` is a column, not a CREATE — word boundaries, not substrings."""
    rc._assert_readonly(f"SELECT {col} FROM t WHERE {col} IS NOT NULL")


def test_every_generated_statement_passes_the_guard():
    for q in build_spec():
        rc._assert_readonly(q["sql"])


def test_connection_is_read_only(tmp_path):
    p = tmp_path / "b.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE t (a INT)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    ro = rc._connect_ro(str(p))
    assert ro.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    with pytest.raises(sqlite3.Error):
        ro.execute("INSERT INTO t VALUES (2)")


# ══════════════════════════════════════════════════════════════════════════════
#  SQL against real-shaped books
# ══════════════════════════════════════════════════════════════════════════════
KNIFE_DDL = """
CREATE TABLE funded_trades (
  symbol TEXT, direction TEXT, level REAL, sl REAL, tp REAL, qty REAL,
  risk_usd REAL, placed_at_utc TEXT, filled_at_utc TEXT, closed_at_utc TEXT,
  exit_price REAL, exit_reason TEXT, r_multiple REAL, pnl_usd REAL)
"""


NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _t(**kw) -> str:
    """A UTC timestamp relative to now, in the fleet's usual text format.

    Fixtures are anchored to the clock, not to a hard-coded date: the working-leg
    filter is age-bounded (6h), so a frozen timestamp would quietly stop being
    "recent" and the test would rot into a false pass or a false alarm.
    """
    return (NOW - timedelta(**kw)).strftime("%Y-%m-%d %H:%M:%S")


def _knife_book(tmp_path):
    p = tmp_path / "knife.db"
    con = sqlite3.connect(p)
    con.execute(KNIFE_DDL)
    con.executemany(
        "INSERT INTO funded_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            # closed inside the window: one winner, then a loser an hour later
            ("BTCUSDT", "LONG", 100.0, 99.0, 102.0, 1.0, 100.0,
             _t(hours=27), _t(hours=26), _t(hours=25), 102.0, "WIN", 2.0, 200.0),
            ("ETHUSDT", "SHORT", 50.0, 51.0, 48.0, 2.0, 100.0,
             _t(hours=27), _t(hours=26), _t(hours=24), 51.0, "LOSS", -1.0, -100.0),
            # closed long ago — lifetime only, outside the window
            ("SOLUSDT", "LONG", 20.0, 19.0, 22.0, 5.0, 100.0,
             _t(days=400), _t(days=400), _t(days=400), 22.0, "WIN", 1.0, 100.0),
            # filled, still running
            ("BCHUSDT", "LONG", 240.0, 238.0, 244.0, 3.0, 100.0,
             _t(hours=3), _t(hours=3), None, None, None, None, None),
            # order resting at the exchange, never filled
            ("XRPUSDT", "SHORT", 1.3, 1.32, 1.27, 900.0, 100.0,
             _t(hours=1), None, None, None, None, None, None),
        ],
    )
    con.commit()
    con.close()
    return p


def _run(db, sql):
    con = rc._connect_ro(str(db))
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    return rows


def test_agg_splits_lifetime_from_window(tmp_path):
    db = _knife_book(tmp_path)
    b = BOOKS_BY_KEY["knife_100k"]
    row = _run(db, build_agg_sql(b, days=7))[0]
    assert row["n"] == 3                      # three closed trades ever
    assert row["sum_r"] == pytest.approx(2.0)
    assert row["wins"] == 2
    assert row["sum_pnl"] == pytest.approx(200.0)
    assert row["n_w"] == 2                    # only two inside the 7-day window
    assert row["sum_r_w"] == pytest.approx(1.0)
    assert row["wins_w"] == 1
    assert row["sum_pnl_w"] == pytest.approx(100.0)


def test_trades_query_returns_only_the_window_newest_first(tmp_path):
    db = _knife_book(tmp_path)
    rows = _run(db, build_trades_sql(BOOKS_BY_KEY["knife_100k"], days=7, limit=50))
    assert [r["symbol"] for r in rows] == ["ETHUSDT", "BTCUSDT"]
    assert rows[0]["r"] == pytest.approx(-1.0)
    assert rows[0]["pnl"] == pytest.approx(-100.0)
    assert rows[1]["entry"] == pytest.approx(100.0)


def test_open_query_separates_filled_from_working(tmp_path):
    db = _knife_book(tmp_path)
    rows = _run(db, build_open_sql(BOOKS_BY_KEY["knife_100k"]))
    got = {(r["state"], r["symbol"]) for r in rows}
    assert got == {("FILLED", "BCHUSDT"), ("WORKING", "XRPUSDT")}
    filled = next(r for r in rows if r["state"] == "FILLED")
    assert filled["sl"] == pytest.approx(238.0)
    assert filled["risk_usd"] == pytest.approx(100.0)


def test_closed_trades_never_appear_as_running(tmp_path):
    db = _knife_book(tmp_path)
    rows = _run(db, build_open_sql(BOOKS_BY_KEY["knife_100k"]))
    assert "BTCUSDT" not in {r["symbol"] for r in rows}


LR_DDL = """
CREATE TABLE trades (
  timestamp TEXT, signal_type TEXT, entry_price REAL, stop_loss REAL,
  take_profit REAL, exit_price REAL, position_size REAL, realized_pnl REAL,
  status TEXT, exit_timestamp TEXT)
"""


def test_lr_r_is_pnl_over_dollars_risked(tmp_path):
    """LR books store dollars, not R — R must be reconstructed, not assumed."""
    p = tmp_path / "lr.db"
    con = sqlite3.connect(p)
    con.execute(LR_DDL)
    con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)", [
        # risked |100-99| * 2 units = $2, made $3 → +1.5R
        (_t(days=2), "long", 100.0, 99.0, 103.0, 101.5, 2.0, 3.0,
         "closed", _t(days=1)),
        # a stop that equals entry would divide by zero — the filter must drop it
        (_t(days=2), "long", 100.0, 100.0, 103.0, 101.0, 2.0, 2.0,
         "closed", _t(days=1)),
        (_t(hours=2), "short", 50.0, 51.0, 47.0, None, 1.0, None,
         "open", None),
    ])
    con.commit()
    con.close()
    b = BOOKS_BY_KEY["lr_funded_btc"]
    agg = _run(p, build_agg_sql(b, days=7))[0]
    assert agg["n"] == 1
    assert agg["sum_r"] == pytest.approx(1.5)
    open_rows = _run(p, build_open_sql(b))
    assert [r["state"] for r in open_rows] == ["FILLED"]
    assert open_rows[0]["entry"] == pytest.approx(50.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Collector plumbing
# ══════════════════════════════════════════════════════════════════════════════
def test_run_queries_reports_a_missing_db_without_raising(tmp_path):
    out = rc.run_queries([{"id": "x::agg", "db": str(tmp_path / "nope.db"),
                           "sql": "SELECT 1 AS n"}])
    assert out["x::agg"]["ok"] is False
    assert out["x::agg"].get("missing") is True


def test_run_queries_blocks_a_write_even_if_one_reaches_the_plan(tmp_path):
    db = _knife_book(tmp_path)
    out = rc.run_queries([{"id": "bad", "db": str(db),
                           "sql": "DELETE FROM funded_trades"}])
    assert out["bad"]["ok"] is False
    assert "blocked" in out["bad"]["error"]
    assert _run(db, "SELECT COUNT(*) AS n FROM funded_trades")[0]["n"] == 5


def test_env_reader_takes_only_bybit_fields_and_seat_name(tmp_path):
    d = tmp_path / "some_seat"
    d.mkdir()
    f = d / "some_seat.env"
    f.write_text("# comment\nBYBIT_API_KEY=abc123\nBYBIT_API_SECRET='sh'\n"
                 "BYBIT_ENV=demo\nTELEGRAM_TOKEN=must-not-be-read\n")
    got = rc._read_env(str(f))
    assert got == {"BYBIT_API_KEY": "abc123", "BYBIT_API_SECRET": "sh",
                   "BYBIT_ENV": "demo"}
    assert "TELEGRAM_TOKEN" not in got
    assert rc._seat_name(str(f)).endswith("some_seat/some_seat")


def test_discover_seats_hashes_keys_and_skips_templates(tmp_path, monkeypatch):
    (tmp_path / "live.env").write_text("BYBIT_API_KEY=k1\nBYBIT_API_SECRET=s1\nBYBIT_ENV=demo\n")
    (tmp_path / "twin.env").write_text("BYBIT_API_KEY=k1\nBYBIT_API_SECRET=s1\nBYBIT_ENV=demo\n")
    (tmp_path / "x_template.env").write_text("BYBIT_API_KEY=k9\nBYBIT_API_SECRET=s9\n")
    (tmp_path / "nokeys.env").write_text("TELEGRAM_TOKEN=t\n")
    monkeypatch.chdir(tmp_path)
    seats = rc.discover_seats(["*.env"], ["*template*"])
    assert sorted(s["seat"] for s in seats) == ["live", "twin"]
    digests = {s["key_digest"] for s in seats}
    assert len(digests) == 1                     # same key → same account
    assert all(len(d) == 8 for d in digests)
    assert all("k1" not in json.dumps({k: v for k, v in s.items() if not k.startswith("_")})
               for s in seats)                   # the raw key never rides along


def test_collector_payload_is_valid_python_and_carries_the_plan():
    from data import fleet_live
    spec = {"plan": [{"id": "a", "db": "x.db", "sql": "SELECT 1"}],
            "balances": {"enabled": False}}
    payload = fleet_live._payload(spec)
    compile(payload, "<payload>", "exec")
    assert "main(json.loads(" in payload
    assert "SELECT 1" in payload


# ══════════════════════════════════════════════════════════════════════════════
#  Reconciliation — the book's word against its own account
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd                                    # noqa: E402

from data.fleet_live import _base, _to_dt, reconcile    # noqa: E402


def _legs(rows):
    """rows: list of (bot, symbol) Tier-1 FILLED legs."""
    return pd.DataFrame([
        {"bot": bot, "tier": 1, "state": "FILLED", "since": pd.Timestamp("2026-09-01", tz="UTC"),
         "symbol": sym, "side": "LONG", "entry": 1.0, "sl": 0.9, "tp": 1.2,
         "qty": 1.0, "risk_usd": 100.0, "age_h": 24.0}
        for bot, sym in rows
    ])


def _raw_with(seat, symbols, ok=True):
    return {"balances": {"accounts": [
        {"uid": "1", "seats": [seat], "ok": ok,
         "positions": [{"symbol": s} for s in symbols]}]}}


@pytest.mark.parametrize("book, venue", [("BCHUSDT", "BCH"), ("BCH", "BCHUSDT"),
                                         ("bchusdt", "BCH"), ("BTCUSD", "BTC")])
def test_base_asset_matching_survives_spelling(book, venue):
    assert _base(book) == _base(venue)


def test_confirmed_leg_is_not_an_orphan():
    raw = _raw_with("ofcs_demo/ofcs_demo", ["XRPUSDT"])
    assert reconcile(raw, _legs([("ofcs-demo/challenge", "XRPUSDT")])).empty


def test_leg_its_own_account_does_not_hold_is_flagged():
    raw = _raw_with("ofcs_demo/ofcs_demo", [])
    out = reconcile(raw, _legs([("ofcs-demo/challenge", "XRPUSDT")]))
    assert list(out["symbol"]) == ["XRPUSDT"]


def test_another_account_holding_the_symbol_does_not_absolve_the_seat():
    """The carry seat being long ETH says nothing about the ofcs seat's ETH leg."""
    raw = {"balances": {"accounts": [
        {"uid": "1", "seats": ["ofcs_demo/ofcs_demo"], "ok": True, "positions": []},
        {"uid": "2", "seats": ["funding_carry_demo/funding_carry_demo"], "ok": True,
         "positions": [{"symbol": "ETHUSDT"}]}]}}
    out = reconcile(raw, _legs([("ofcs-demo/challenge", "ETHUSDT")]))
    assert len(out) == 1


def test_a_silent_account_is_unchecked_never_orphaned():
    """A dead key cannot testify — the leg is unverifiable, not proven stale."""
    raw = _raw_with("ofcs_demo/ofcs_demo", [], ok=False)
    assert reconcile(raw, _legs([("ofcs-demo/challenge", "XRPUSDT")])).empty


def test_a_book_with_no_known_seat_is_never_flagged():
    raw = _raw_with("desk_demo/desk_demo", [])
    assert reconcile(raw, _legs([("phantom-conductor", "BTCUSDT")])).empty


def test_option_books_are_never_flagged():
    """Option legs live in a category this snapshot does not read."""
    raw = _raw_with("ironfly", [])
    assert reconcile(raw, _legs([("ironfly-btc(opt)", "BTC")])).empty


def test_balances_off_flags_nothing():
    assert reconcile({}, _legs([("ofcs-demo/challenge", "XRPUSDT")])).empty


def test_mixed_timestamp_formats_all_parse():
    """One book writes `T` and microseconds, another a space — both must land."""
    s = pd.Series(["2026-09-02 12:01:00", "2026-06-16T19:21:17.131143",
                   "2026-05-22T02:17:43.861000+00:00", None])
    out = _to_dt(s)
    assert out.notna().sum() == 3
    assert out.dt.tz is not None


# ══════════════════════════════════════════════════════════════════════════════
#  Per-day aggregation — the charts must not read a capped dump
# ══════════════════════════════════════════════════════════════════════════════
from data.fleet_registry import build_bucket_sql       # noqa: E402


def test_buckets_group_by_utc_hour(tmp_path):
    db = _knife_book(tmp_path)
    rows = _run(db, build_bucket_sql(BOOKS_BY_KEY["knife_100k"], days=7))
    # the two in-window closes are an hour apart → two buckets, not one
    want = [(NOW - timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in (25, 24)]
    assert [r["bucket"] for r in rows] == want
    assert [r["n"] for r in rows] == [1, 1]
    assert rows[0]["sum_r"] == pytest.approx(2.0)      # the winner closed first
    assert rows[1]["sum_pnl"] == pytest.approx(-100.0)


def test_buckets_count_every_row_even_past_the_ledger_cap(tmp_path):
    """The ledger keeps 500 rows per book; the day totals must count all of them."""
    p = tmp_path / "busy.db"
    con = sqlite3.connect(p)
    con.execute(KNIFE_DDL)
    con.executemany(
        "INSERT INTO funded_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [("BTCUSDT", "LONG", 100.0, 99.0, 102.0, 1.0, 100.0,
          _t(hours=27), _t(hours=26),
          (NOW - timedelta(hours=25, minutes=i % 60)).strftime("%Y-%m-%d %H:%M:%S"),
          102.0, "WIN", 1.0, 10.0)
         for i in range(700)])
    con.commit()
    con.close()
    b = BOOKS_BY_KEY["knife_100k"]
    capped = _run(p, build_trades_sql(b, days=7, limit=500))
    buckets = _run(p, build_bucket_sql(b, days=7))
    assert len(capped) == 500                          # the dump is trimmed …
    assert sum(r["n"] for r in buckets) == 700         # … the bucket totals are not
    assert sum(r["sum_r"] for r in buckets) == pytest.approx(700.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Granularity roll-up and the Alpaca seat's credentials
# ══════════════════════════════════════════════════════════════════════════════
from data.fleet_live import GRANULARITY, resample_buckets   # noqa: E402


def _buckets():
    return pd.DataFrame({
        "bucket": pd.to_datetime(
            ["2026-09-01T09:00", "2026-09-01T11:00", "2026-09-02T09:00"], utc=True),
        "n": [1, 2, 3], "sum_r": [0.5, -1.0, 2.0], "sum_pnl": [50.0, -100.0, 200.0],
        "bot": ["knife-funded-100k"] * 3, "tier": [1, 1, 1],
    })


def test_hourly_rollup_keeps_every_bucket():
    out = resample_buckets(_buckets(), "Hour")
    assert len(out) == 3
    assert out["n"].sum() == 6


@pytest.mark.parametrize("grain, buckets, first_n", [("4 hours", 2, 3), ("Day", 2, 3)])
def test_coarser_grains_merge_without_losing_trades(grain, buckets, first_n):
    out = resample_buckets(_buckets(), grain).sort_values("t")
    assert len(out) == buckets
    assert out["n"].sum() == 6                 # nothing is dropped by zooming out
    assert out.iloc[0]["n"] == first_n
    assert out["sum_r"].sum() == pytest.approx(1.5)


def test_week_grain_collapses_the_window():
    out = resample_buckets(_buckets(), "Week")
    assert len(out) == 1 and out.iloc[0]["n"] == 6


def test_every_offered_grain_is_a_valid_pandas_alias():
    for grain in GRANULARITY:
        assert not resample_buckets(_buckets(), grain).empty


def test_empty_buckets_survive_the_rollup():
    out = resample_buckets(pd.DataFrame(columns=["bucket", "n", "sum_r", "sum_pnl",
                                                 "bot", "tier"]), "Day")
    assert out.empty


def test_alpaca_reader_takes_only_the_two_key_fields(tmp_path):
    f = tmp_path / ".env"
    f.write_text("export ALPACA_API_KEY=PKabc\nALPACA_SECRET_KEY='sh'\n"
                 "TELEGRAM_TOKEN=must-not-be-read\n")
    key, sec, src = rc._alpaca_env([str(f)])
    assert (key, sec) == ("PKabc", "sh")
    assert src == str(f)


def test_alpaca_refuses_a_non_paper_key(tmp_path):
    """A live key must fail closed here exactly as it does in the bot itself."""
    f = tmp_path / ".env"
    f.write_text("ALPACA_API_KEY=AKlive\nALPACA_SECRET_KEY=s\n")
    out = rc.collect_alpaca({"env_candidates": [str(f)], "seat": "analyst_drift_paper"})
    assert out["ok"] is False
    assert "PAPER" in out["error"]


def test_alpaca_with_no_credentials_reports_rather_than_raises():
    out = rc.collect_alpaca({"env_candidates": ["/nope/.env"]})
    assert out["ok"] is False and "no Alpaca credentials" in out["error"]


def test_side_and_symbol_are_one_type_per_column(tmp_path):
    """A column mixing str and int cannot cross Arrow into st.dataframe."""
    from data.fleet_live import open_frame, trades_frame
    db = _knife_book(tmp_path)
    cols_t = ["ts", "symbol", "side", "entry", "exit_px", "r", "pnl"]
    cols_o = ["state", "since", "symbol", "side", "entry", "sl", "tp", "qty", "risk_usd"]
    raw = {"results": {
        "knife_100k::trades": {"ok": True, "cols": cols_t,
                               "rows": [[_t(hours=2), "BTCUSDT", "LONG", 1.0, 2.0, 1.0, 10.0],
                                        [_t(hours=3), "ETHUSDT", 1, 1.0, 2.0, -1.0, -10.0]]},
        "knife_100k::open": {"ok": True, "cols": cols_o,
                             "rows": [["FILLED", _t(hours=1), "BCHUSDT", -1,
                                       1.0, 0.9, 1.2, 1.0, 100.0]]}}}
    assert str(trades_frame(raw, days=7)["side"].dtype) == "string"
    assert str(open_frame(raw)["side"].dtype) == "string"


def test_sweep_engine_side_is_spelled_out_not_signed():
    b = BOOKS_BY_KEY["sweep_engine"]
    sql = build_trades_sql(b, days=7)
    assert "'LONG'" in sql and "'SHORT'" in sql


# ══════════════════════════════════════════════════════════════════════════════
#  Headline arithmetic — tiers must never be mixed in the money line
# ══════════════════════════════════════════════════════════════════════════════
from data.fleet_live import window_headline                 # noqa: E402


def _scope():
    """A Tier-1 seat losing real money, a paper book "making" virtual money,
    and a shadow recorder printing a large dimensionless R."""
    return pd.DataFrame([
        {"bot": "knife-funded-100k", "tier": 1, "n_7d": 18, "sum_r_7d": -3.17,
         "pnl_usd_7d": -2834.14, "win_rate_7d": 0.5, "note": ""},
        {"bot": "lrr-paper", "tier": 2, "n_7d": 15, "sum_r_7d": 6.59,
         "pnl_usd_7d": 3222.83, "win_rate_7d": 0.6, "note": ""},
        {"bot": "halt-shadow(era2)", "tier": 3, "n_7d": 900, "sum_r_7d": 590.2,
         "pnl_usd_7d": None, "win_rate_7d": 0.8, "note": "era-1 excluded"},
    ])


def test_realized_dollars_come_from_tier_1_alone():
    """Virtual paper dollars must never land in the realized-$ figure."""
    hl = window_headline(_scope())
    assert hl["pnl_t1"] == pytest.approx(-2834.14)
    assert hl["pnl_t1"] != pytest.approx(-2834.14 + 3222.83)


def test_scope_r_and_tier1_r_are_reported_separately():
    hl = window_headline(_scope())
    assert hl["sum_r"] == pytest.approx(-3.17 + 6.59 + 590.2)
    assert hl["sum_r_t1"] == pytest.approx(-3.17)


def test_win_rate_is_trade_weighted_not_book_weighted():
    hl = window_headline(_scope())
    assert hl["win_rate"] == pytest.approx((18 * .5 + 15 * .6 + 900 * .8) / 933)


def test_best_and_worst_rank_on_r():
    hl = window_headline(_scope())
    assert hl["best"]["bot"] == "halt-shadow(era2)"
    assert hl["worst"]["bot"] == "knife-funded-100k"


def test_empty_scope_returns_zeros_not_an_exception():
    hl = window_headline(pd.DataFrame(columns=["bot", "tier", "n_7d", "sum_r_7d",
                                               "pnl_usd_7d", "win_rate_7d", "note"]))
    assert hl["n"] == 0 and hl["best"] is None and hl["pnl_t1"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  2026-09-03 — fleet registry as the unified universe's source
# ══════════════════════════════════════════════════════════════════════════════
from data.fleet_registry import FAMILIES, build_history_sql, family_symbols   # noqa: E402


def test_every_tier1_and_tier2_book_has_a_family():
    missing = [b.key for b in BOOKS if b.tier <= 2 and not b.family]
    assert missing == [], missing


def test_families_are_the_fleet_not_the_shadows():
    assert "Knife" in FAMILIES and "Options" in FAMILIES and "Analyst Drift" in FAMILIES
    assert all(f for f in FAMILIES)


def test_history_sql_carries_the_unified_fields_and_passes_the_guard():
    for b in BOOKS:
        sql = build_history_sql(b)
        rc._assert_readonly(sql)
        for col in ("entry_ts", "exit_ts", "symbol", "side", "entry", "exit_px",
                    "sl", "tp", "qty", "risk_usd", "r", "pnl", "exit_reason"):
            assert f" AS {col}" in sql, (b.key, col)
        assert "LIMIT" not in sql and "-7 day" not in sql     # whole ledger, no window


def test_history_rows_against_a_real_shaped_book(tmp_path):
    db = _knife_book(tmp_path)
    rows = _run(db, build_history_sql(BOOKS_BY_KEY["knife_100k"]))
    assert len(rows) == 3                          # closed rows only, lifetime
    assert [r["symbol"] for r in rows] == ["SOLUSDT", "BTCUSDT", "ETHUSDT"]  # oldest first
    assert rows[1]["entry_ts"] == _t(hours=26)     # the FILL time, not the arm time
    assert rows[1]["sl"] == pytest.approx(99.0)
    assert rows[1]["exit_reason"] == "WIN"


def test_lr_paper_books_are_registered_but_nq_is_left_to_the_legacy_map():
    keys = {b.key for b in BOOKS}
    assert "lr_paper_btc" in keys and "lr_paper_xrp" in keys
    assert "lr_paper_nq" not in keys


def test_family_symbols_come_from_literal_symbol_columns():
    assert set(family_symbols("Liquidity Raid")) >= {"BTC", "ETH", "SOL"}
