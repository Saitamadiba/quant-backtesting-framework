"""The 2026-09-03 registry work — hermetic.

  * schema_normalizer.normalize_fleet_book: a fleet book → the unified schema,
    with Tier 2 landing as `source="Paper"` and never as Live dollars.
  * shadow_normalisers: the ten bridges registered for the post-July recorders,
    including the era filters that keep a refuted era from being pooled.
  * vps_sync.deploy_guard: the diff-first rule as a pure decision.
  * config: the registries agree with each other.
"""
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

_DASH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DASH))

import config as cfg                                            # noqa: E402
from data import shadow_normalisers as sn                       # noqa: E402
from data.fleet_registry import BOOKS_BY_KEY                    # noqa: E402
from data.schema_normalizer import normalize_fleet_book          # noqa: E402
from data.vps_sync import deploy_guard                           # noqa: E402

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _t(**kw):
    return (NOW - timedelta(**kw)).strftime("%Y-%m-%d %H:%M:%S")


# ── unified-schema bridge ─────────────────────────────────────────────────────
def _seats_db(tmp_path):
    p = tmp_path / "desk_demo.db"
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE seats (setup_key TEXT, symbol TEXT, tf TEXT, ltype TEXT,
        side TEXT, level REAL, sl REAL, tp REAL, atr REAL, dist_atr REAL, mass REAL,
        regime5 TEXT, bar_ts TEXT, first_seen TEXT, status TEXT, order_id TEXT, qty REAL,
        risk_usd REAL, fill_time TEXT, fill_px REAL, closed_at TEXT, realized_pnl REAL,
        realized_r REAL, close_reason TEXT, fill_ts_ms INTEGER)""")
    con.executemany("INSERT INTO seats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("k1", "BTC", "15m", "PDH", "L", 100.0, 99.0, 103.0, 1, 1, 1, "quiet", _t(hours=30),
         _t(hours=30), "closed", "o1", 1.0, 100.0, _t(hours=29), 100.0, _t(hours=27),
         300.0, 3.0, "TP", 0),
        ("k2", "ETH", "15m", "PDL", "S", 50.0, 51.0, 47.0, 1, 1, 1, "quiet", _t(hours=20),
         _t(hours=20), "closed", "o2", 2.0, 100.0, _t(hours=19), 50.0, _t(hours=18),
         -100.0, -1.0, "SL", 0),
        ("k3", "SOL", "15m", "PDH", "L", 20.0, 19.0, 22.0, 1, 1, 1, "quiet", _t(hours=2),
         _t(hours=2), "regime_skip", None, 0, 0, None, None, None, None, None, None, 0),
    ])
    con.commit(); con.close()
    return p


def test_fleet_book_lands_in_the_unified_schema(tmp_path):
    df = normalize_fleet_book(_seats_db(tmp_path), BOOKS_BY_KEY["desk_demo"])
    assert list(df.columns)[: len(cfg.TRADE_SCHEMA_COLS)] == cfg.TRADE_SCHEMA_COLS
    assert len(df) == 2                                   # the skipped seat is not a trade
    assert set(df["strategy"]) == {"Desk"}                # the FAMILY, not the seat label
    assert set(df["bot"]) == {"desk-demo"}
    assert list(df["direction"]) == ["Long", "Short"]     # L / S canonicalised
    assert df["source"].unique().tolist() == ["Live"]     # Tier 1 = real fills
    assert df["pnl_usd"].tolist() == [300.0, -100.0]
    assert df["r_multiple"].tolist() == [3.0, -1.0]
    assert df["stop_loss"].tolist() == [99.0, 51.0]
    assert df["exit_reason"].tolist() == ["TP", "SL"]
    assert (df["duration_minutes"] > 0).all()
    assert set(df["session"]) <= {"Asian", "London", "New York", "Off-Hours"}


def test_tier2_books_are_paper_never_live(tmp_path):
    """Virtual dollars must be filterable away from the money line."""
    p = tmp_path / "lrr_paper.db"
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE paper_trades (id INTEGER, source_id TEXT, asset TEXT,
        direction TEXT, entry_ts_utc TEXT, entry_price REAL, sl REAL, tp REAL, ml_p REAL,
        status TEXT, risk_pct REAL, risk_usd REAL, equity_before REAL, exit_ts_utc TEXT,
        exit_reason TEXT, r_gross REAL, r_net REAL, pnl_usd REAL, opened_at_utc TEXT,
        closed_at_utc TEXT)""")
    con.execute("INSERT INTO paper_trades VALUES (1,'s','BTC','LONG',?,100,99,103,0.5,'CLOSED',"
                "1,1000,100000,?,'TP',3.1,3.0,3000,?,?)", (_t(hours=5), _t(hours=3), _t(hours=5), _t(hours=3)))
    con.commit(); con.close()
    df = normalize_fleet_book(p, BOOKS_BY_KEY["lrr_paper"])
    assert df["source"].tolist() == ["Paper"]
    assert df["strategy"].tolist() == ["LRR"]
    assert df["r_multiple"].tolist() == [3.0]             # net of the toll, not gross


def test_unreadable_fleet_book_returns_empty_not_error(tmp_path):
    p = tmp_path / "nope.db"
    p.write_bytes(b"not a database")
    df = normalize_fleet_book(p, BOOKS_BY_KEY["desk_demo"])
    assert df.empty


# ── shadow bridges ────────────────────────────────────────────────────────────
def test_halt_shadow_keeps_only_era_2():
    df = pd.DataFrame({"era": [None, 1, 2, 2], "r_net": [5.0, 4.0, 0.3, -1.0],
                       "strategy": ["desk"] * 4, "symbol": ["BTC"] * 4,
                       "direction": ["LONG"] * 4, "entry": [1] * 4, "sl": [0.9] * 4,
                       "tp": [1.2] * 4, "recorded_at_utc": ["t"] * 4,
                       "closed_at_utc": ["t", "t", "t", None]})
    out = sn.normalise_halt_shadow(df)
    assert len(out) == 2 and set(out["era"]) == {2}
    assert "family" in out.columns and "strategy" not in out.columns
    assert out["r_multiple"].tolist() == [0.3, -1.0]
    assert out["exit_reason"].iloc[-1] == "OPEN"          # unresolved row is OPEN, not a loss


def test_sweep_engine_side_and_feed_filters():
    df = pd.DataFrame({"side": [1, -1, 1], "is_alias": [0, 0, 1], "feed": ["bybit", "bybit", "bybit"],
                       "event_time": ["a", "b", "c"], "resolved_at": ["a", "b", "c"],
                       "entry": [1, 2, 3], "stop": [0.9, 2.1, 2.9], "r_gross": [1.0, -1.0, 9.0],
                       "symbol": ["BTC", "ETH", "SOL"], "exit_reason": ["TP", "SL", "TP"]})
    out = sn.normalise_sweep_engine(df)
    assert len(out) == 2                                   # alias row dropped
    assert out["direction"].tolist() == ["BUY", "SELL"]    # ±1 spelled out
    assert out["r_multiple"].tolist() == [1.0, -1.0]       # gross, as labelled


def test_lr_signal_shadow_uses_era2_net_r():
    df = pd.DataFrame({"era": [1, 2], "r_multiple": [2.0, 2.0], "r_net": [None, 1.6],
                       "direction": ["LONG", "SHORT"], "opened_at_utc": ["a", "b"],
                       "closed_at_utc": ["a", "b"], "exit_reason": ["TP", "TP"]})
    out = sn.normalise_lr_signal_e2(df)
    assert len(out) == 1 and out["r_multiple"].tolist() == [1.6]


def test_fvg_alts_keeps_only_the_newest_era():
    df = pd.DataFrame({"era": [1, 2, 3, 3], "r_net": [9, 9, 0.5, -1.0],
                       "confirm_bar_utc": ["a"] * 4, "exit_utc": ["a"] * 4,
                       "closed_at_utc": ["a"] * 4, "direction": ["LONG"] * 4,
                       "tp1": [1] * 4, "symbol": ["ADA"] * 4, "outcome": ["win_tp1"] * 4})
    out = sn.normalise_fvg_alts_shadow(df)
    assert set(out["era"]) == {3} and out["r_multiple"].tolist() == [0.5, -1.0]
    assert out["exit_reason"].tolist() == ["TP", "TP"]


def test_gate_books_dispatch_by_filename():
    assert sn.normaliser_for("lr_btcusdt_flow_gate.db") is sn.normalise_gate_book
    assert sn.table_for("lr_btcusdt_flow_gate.db") == "shadow_trades"
    df = pd.DataFrame({"realized_r": [1.0], "direction": ["LONG"], "pnl": [10.0],
                       "opened_at_utc": ["a"], "closed_at_utc": ["a"], "exit_reason": ["TP"]})
    out = sn.normalise_gate_book(df)
    assert out["r_multiple"].tolist() == [1.0] and out["pnl_usd"].tolist() == [10.0]


@pytest.mark.parametrize("fn", ["antiknife_shadow.db", "crossvenue_shadow.db", "gated_lr_shadow.db",
                                "wide_rr_shadow.db", "halt_shadow_book.db", "sweep_engine.db",
                                "fib618_shadow.db", "fvg_alts_shadow.db", "lr_signal_shadow.db",
                                "depth_policy_paper_book.db", "mm_15m_shadow.db", "mm_5m_shadow.db"])
def test_every_new_shadow_book_is_synced_labelled_and_bridged(fn):
    assert fn in cfg.VPS_SHADOW_DB_FILES
    assert fn in cfg.SHADOW_DB_STRATEGY_MAP
    assert sn.normaliser_for(fn) is not None


def test_shadow_registries_agree():
    assert set(cfg.VPS_SHADOW_DB_FILES) == set(cfg.SHADOW_DB_STRATEGY_MAP)


# ── deploy guard ──────────────────────────────────────────────────────────────
LOCAL = {"mtime": 1000.0, "size": 500, "md5": "aaa"}


def test_guard_refuses_a_newer_vps_copy():
    v = deploy_guard(LOCAL, {"exists": True, "reachable": True, "mtime": 2000.0, "size": 400, "md5": "bbb"})
    assert v["ok"] is False and "NEWER" in v["reason"]


def test_guard_refuses_a_longer_vps_copy():
    v = deploy_guard(LOCAL, {"exists": True, "reachable": True, "mtime": 500.0, "size": 900, "md5": "bbb"})
    assert v["ok"] is False and "LONGER" in v["reason"]


def test_guard_passes_identical_as_a_noop():
    v = deploy_guard(LOCAL, {"exists": True, "reachable": True, "mtime": 2000.0, "size": 500, "md5": "aaa"})
    assert v["ok"] is True and v.get("noop") is True


def test_guard_passes_older_not_longer_and_absent():
    assert deploy_guard(LOCAL, {"exists": True, "reachable": True, "mtime": 500.0, "size": 400, "md5": "b"})["ok"]
    assert deploy_guard(LOCAL, {"exists": False, "reachable": True})["ok"]


def test_guard_will_not_deploy_blind():
    v = deploy_guard(LOCAL, {"exists": False, "reachable": False})
    assert v["ok"] is False and "unreachable" in v["reason"]


# ── config registries ─────────────────────────────────────────────────────────
def test_bot_services_have_kind_and_log_and_real_unit_names():
    for unit, info in cfg.BOT_SERVICES.items():
        assert info.get("kind") in ("service", "timer", "cron"), unit
        assert "log" in info, unit
        if info["kind"] == "service":
            assert unit.endswith(".service"), unit
        if info["kind"] == "timer":
            assert unit.endswith(".timer"), unit
    for dead in ("lr-btc", "lr-sol", "sbs-btc", "sbs-eth"):   # units that no longer exist
        assert dead not in cfg.BOT_SERVICES and f"{dead}.service" not in cfg.BOT_SERVICES


def test_research_strategies_are_a_subset_of_the_palette():
    assert set(cfg.RESEARCH_STRATEGIES) <= set(cfg.STRATEGIES)
    assert {"Knife", "Depth", "OFCS", "Options", "Analyst Drift"} <= set(cfg.STRATEGIES)
    assert len(set(cfg.STRATEGY_COLORS.values())) == len(cfg.STRATEGY_COLORS)   # no colour collisions


def test_knife_arm_registry_has_all_seven_books():
    assert len(cfg.VPS_KNIFE_DB_FILES) == 7
    assert "knife_funded_10k.db" in cfg.VPS_KNIFE_DB_FILES


def test_net_r_wins_when_a_book_carries_both_gross_and_net():
    """gated_lr and halt books hold r_multiple (pre-fee) AND r_net; the page
    must read net — a non-destructive alias would have kept the gross print."""
    df = pd.DataFrame({"r_multiple": [1.0, -1.0], "r_net": [0.8, -1.2], "sym": ["BTC", "ETH"],
                       "direction": ["LONG", "SHORT"], "opened_at_utc": ["a", "b"],
                       "closed_at_utc": ["a", "b"], "regime": ["quiet", "vol"]})
    out = sn.normalise_gated_lr(df)
    assert out["r_multiple"].tolist() == [0.8, -1.2]
    assert out["r_gross"].tolist() == [1.0, -1.0]
    assert out["exit_reason"].tolist() == ["WIN", "LOSS"]     # derived from R, no label column


def test_paper_book_drops_stale_rows_and_labels_halted_skips():
    df = pd.DataFrame({"stale_signal": [0, 1, 0], "status": ["CLOSED", "CLOSED", "HALTED_RISK"],
                       "exit_reason": ["TP", "TP", None], "r_net": [1.0, 9.0, None],
                       "direction": ["LONG"] * 3, "sl": [1] * 3, "tp": [2] * 3,
                       "opened_at_utc": ["a"] * 3, "closed_at_utc": ["a", "a", None]})
    out = sn.normalise_paper_book(df)
    assert len(out) == 2 and 9.0 not in out["r_multiple"].tolist()   # the stale +9R row is gone
    assert out["exit_reason"].tolist() == ["TP", "HALTED_RISK"]      # a halted skip is not OPEN


from data.vps_sync import parse_is_active                        # noqa: E402


@pytest.mark.parametrize("out, want", [
    ("active\n", "active"), ("inactive\nunknown\n", "inactive"), ("failed\nunknown", "failed"),
    ("activating\nunknown", "activating"), ("unknown\n", "unknown"), ("", "unknown"),
])
def test_is_active_reads_the_state_line_not_the_fallback(out, want):
    assert parse_is_active(out) == want
