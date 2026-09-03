"""Hermetic tests for WS1's Bayesian fleet-edge reader.

Nothing here touches the VPS, the network, or a real fleet database: every book
is a throwaway SQLite file built to the registry's own schema, so a column
rename, a broken toll convention or a lost clustering haircut fails here rather
than on the page.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DASH = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DASH))
sys.path.insert(0, str(_ROOT))

from data import fleet_edge as fe                              # noqa: E402
from data.fleet_registry import BOOKS, Book, BOOKS_BY_KEY      # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  The cluster key must not drift from the feature spine's
# ══════════════════════════════════════════════════════════════════════════════
def test_cluster_key_matches_the_feature_spine():
    """WS0's spine stores `cluster_id` on every fill; WS1 recomputes it here.
    Two definitions of "one bet" would silently disagree about effective n."""
    ff = pytest.importorskip("fleet_features.features")
    for fam, side, ms in (("Knife", "LONG", 1_787_985_123_456),
                          ("Liquidity Raid", "sell", 0),
                          ("Desk", None, 1_700_000_000_001)):
        assert fe.cluster_id(fam, side, ms) == ff.cluster_id(fam, side, ms)


def test_cluster_buckets_to_fifteen_minutes():
    a = fe.cluster_id("F", "LONG", 1_787_985_000_000)
    b = fe.cluster_id("F", "BUY", 1_787_985_000_000 + 14 * 60 * 1000)
    c = fe.cluster_id("F", "LONG", 1_787_985_000_000 + 16 * 60 * 1000)
    assert a == b, "same 15-min bucket, same direction → one bet"
    assert a != c, "next bucket → a separate bet"
    assert fe.cluster_id("F", "SHORT", 1_787_985_000_000) != a


# ══════════════════════════════════════════════════════════════════════════════
#  Timestamps — the epoch-integer trap
# ══════════════════════════════════════════════════════════════════════════════
def test_epoch_seconds_do_not_land_in_1970():
    """ferryman's `orders` stores epoch SECONDS; a bare to_datetime reads a bare
    integer as nanoseconds and puts every fill in January 1970."""
    out = fe.to_utc([1786081227, 1786081827])
    assert out.dt.year.tolist() == [2026, 2026]


def test_epoch_units_agree_on_the_same_instant():
    s = fe.to_utc([1786081227])[0]
    ms = fe.to_utc([1786081227_000])[0]
    us = fe.to_utc([1786081227_000_000])[0]
    assert s == ms == us


def test_iso_and_sqlite_strings_still_parse():
    out = fe.to_utc(["2026-09-01 10:00:00", "2026-09-01T10:15:00Z"])
    assert out.notna().all()
    assert (out.dt.tz is not None) and out.dt.year.tolist() == [2026, 2026]


def test_unparseable_values_become_nat_not_an_exception():
    out = fe.to_utc(["not a time", None])
    assert out.isna().all()


# ══════════════════════════════════════════════════════════════════════════════
#  The bar is the toll, and only for gross books
# ══════════════════════════════════════════════════════════════════════════════
def test_gross_books_owe_the_toll_and_net_books_do_not():
    gross = BOOKS_BY_KEY["ofcs_paper_e1"]        # label: ofcs-paper(gross,era1)
    net = BOOKS_BY_KEY["ofcs_paper_e2"]          # label: ofcs-paper(net,era2)
    assert fe.is_gross(gross) and not fe.is_gross(net)
    assert fe.bar_for(gross, toll=0.25) == 0.25
    assert fe.bar_for(net, toll=0.25) == 0.0


def test_every_registry_label_saying_gross_is_flagged():
    for b in BOOKS:
        assert fe.is_gross(b) == ("gross" in b.label.lower())


def test_p_above_bar_is_stricter_than_p_above_zero_for_a_gross_book():
    rng = np.random.default_rng(7)
    r = rng.normal(0.10, 1.0, 400)
    s = fe.BookSeries(key="k", label="x(gross)", family="F", tier=3, gross=True,
                      r=r, clusters=[f"c{i}" for i in range(len(r))])
    row = fe.posterior(s, toll=0.25)
    assert row["bar_r"] == 0.25
    assert row["p_above_bar"] < row["p_above_0"]


# ══════════════════════════════════════════════════════════════════════════════
#  Effective n — clustering and re-recorded setups
# ══════════════════════════════════════════════════════════════════════════════
def test_clustered_fills_widen_the_interval():
    """Ten simultaneous same-direction fills are one bet; pretending they are ten
    independent draws narrows the interval it has no right to narrow."""
    rng = np.random.default_rng(11)
    r = rng.normal(0.2, 1.0, 300)
    independent = fe.BookSeries(key="i", label="i", family="F", tier=3, gross=False,
                                r=r, clusters=[f"c{i}" for i in range(300)])
    clustered = fe.BookSeries(key="c", label="c", family="F", tier=3, gross=False,
                              r=r, clusters=[f"c{i // 10}" for i in range(300)])
    wide = fe.posterior(clustered)
    tight = fe.posterior(independent)
    assert clustered.n_eff == 30 and independent.n_eff == 300
    assert (wide["ci_hi"] - wide["ci_lo"]) > (tight["ci_hi"] - tight["ci_lo"])
    assert wide["p_above_bar"] < tight["p_above_bar"]


def test_repeated_setups_collapse_to_one_bet():
    keys = [f"F|L|{i}" for i in range(6)]
    raw = pd.DataFrame({"symbol": ["BTCUSDT"] * 6, "side": ["LONG"] * 6,
                        "entry": [100.0] * 4 + [200.0, 300.0],
                        "sl": [99.0] * 4 + [199.0, 299.0]})
    keep = pd.Series([True] * 6)
    out = fe._collapse_repeated_setups(keys, raw, keep)
    assert len(set(out)) == 3, "four re-records of one resting setup are one bet"
    assert out[4] == keys[4] and out[5] == keys[5], "unique setups keep their bucket"


def test_a_jittering_stop_does_not_defeat_the_collapse():
    """The recorders recompute an ATR stop every cycle, so the SL moves in the
    eighth decimal while the resting LEVEL stays pinned. Keying the signature on
    the bracket would collapse nothing — halt_shadow reads +0.31R deduped that
    way and -0.14R deduped on the level."""
    keys = [f"F|L|{i}" for i in range(4)]
    raw = pd.DataFrame({"symbol": ["DOTUSDT"] * 4, "side": ["LONG"] * 4,
                        "entry": [0.8385] * 4,
                        "sl": [0.825554, 0.825436, 0.825183, 0.824628]})
    out = fe._collapse_repeated_setups(keys, raw, pd.Series([True] * 4))
    assert len(set(out)) == 1


def test_a_book_without_repeats_is_untouched():
    keys = [f"F|L|{i}" for i in range(3)]
    raw = pd.DataFrame({"symbol": ["BTCUSDT"] * 3, "side": ["LONG"] * 3,
                        "entry": [1.0, 2.0, 3.0], "sl": [0.9, 1.9, 2.9]})
    assert fe._collapse_repeated_setups(keys, raw, pd.Series([True] * 3)) == keys


def test_opposite_directions_at_one_level_stay_two_bets():
    keys = ["F|L|0", "F|S|0"]
    raw = pd.DataFrame({"symbol": ["BTCUSDT"] * 2, "side": ["LONG", "SHORT"],
                        "entry": [100.0, 100.0], "sl": [99.0, 101.0]})
    assert len(set(fe._collapse_repeated_setups(keys, raw, pd.Series([True] * 2)))) == 2


def test_missing_entry_column_does_not_break_clustering():
    keys = ["a", "b"]
    assert fe._collapse_repeated_setups(keys, pd.DataFrame({"symbol": ["X", "X"]}),
                                        pd.Series([True, True])) == keys


# ══════════════════════════════════════════════════════════════════════════════
#  Reading a book end to end, against a real-schema throwaway DB
# ══════════════════════════════════════════════════════════════════════════════
def _make_book_db(tmp_path: Path, rows) -> Path:
    p = tmp_path / "toy.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE trades (symbol TEXT, side TEXT, entry REAL, sl REAL,"
                " tp REAL, exit_price REAL, qty REAL, risk_usd REAL, r_net REAL,"
                " pnl REAL, exit_reason TEXT, entry_ts TEXT, closed_at TEXT)")
    con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return p


def _toy_book() -> Book:
    return Book(key="toy", label="toy-book", tier=3, db="toy.db", table="trades",
                ts="closed_at", closed_filter="r_net IS NOT NULL", r="r_net",
                symbol="symbol", side="side", entry="entry", exit="exit_price",
                open_sl="sl", open_tp="tp", entry_ts="entry_ts",
                exit_reason="exit_reason", family="Toy")


def test_load_book_series_reads_r_and_clusters(tmp_path):
    rows = [("BTCUSDT", "LONG", 100.0, 99.0, 102.0, 102.0, 1, 10, 1.0, 10.0, "TP",
             "2026-09-01T10:00:00Z", "2026-09-01T11:00:00Z"),
            ("BTCUSDT", "LONG", 100.0, 99.0, 102.0, 99.0, 1, 10, -1.0, -10.0, "SL",
             "2026-09-01T10:07:00Z", "2026-09-01T11:00:00Z"),
            ("BTCUSDT", "SHORT", 100.0, 101.0, 98.0, 98.0, 1, 10, 0.5, 5.0, "TP",
             "2026-09-01T10:40:00Z", "2026-09-01T12:00:00Z")]
    s = fe.load_book_series(_toy_book(), _make_book_db(tmp_path, rows))
    assert s.error == "" and s.n == 3
    # rows 0/1 share a 15-min bucket AND a setup signature → one bet; row 2 is another
    assert s.n_eff == 2
    assert s.first_ts.isoformat().startswith("2026-09-01T10:00")


def test_rows_with_null_r_are_excluded_not_zeroed(tmp_path):
    rows = [("BTCUSDT", "LONG", 100.0, 99.0, 102.0, 102.0, 1, 10, 1.0, 10.0, "TP",
             "2026-09-01T10:00:00Z", "2026-09-01T11:00:00Z"),
            ("BTCUSDT", "LONG", 100.0, 99.0, 102.0, None, 1, 10, None, None, None,
             "2026-09-01T10:20:00Z", "2026-09-01T11:00:00Z")]
    s = fe.load_book_series(_toy_book(), _make_book_db(tmp_path, rows))
    assert s.n == 1, "an unresolved row is not a zero-R trade"


def test_unreadable_db_is_reported_not_silently_empty(tmp_path):
    s = fe.load_book_series(_toy_book(), tmp_path / "missing.db")
    assert s.n == 0 and s.error.startswith("unreadable")
    row = fe.posterior(s)
    assert row["status"].startswith("unreadable")
    assert "post_mean_r" not in row, "no posterior is claimed without data"


# ══════════════════════════════════════════════════════════════════════════════
#  Posterior behaviour
# ══════════════════════════════════════════════════════════════════════════════
def test_thin_book_is_flagged_and_not_fitted():
    s = fe.BookSeries(key="k", label="k", family="F", tier=1, gross=False,
                      r=np.array([1.0]), clusters=["c0"])
    row = fe.posterior(s)
    assert row["status"] == "thin" and "post_mean_r" not in row


def test_prior_pulls_a_tiny_sample_toward_the_toll():
    """Two lucky trades must not read as an edge: the toll prior (-0.25R) still
    dominates until the data outweighs its pseudo-observations."""
    s = fe.BookSeries(key="k", label="k", family="F", tier=1, gross=False,
                      r=np.array([2.0, 2.0]), clusters=["a", "b"])
    row = fe.posterior(s)
    assert row["raw_mean_r"] == 2.0
    assert row["post_mean_r"] < 1.0
    assert fe.verdict(row) == "prior dominates"


def test_large_negative_book_reads_as_evidence_against():
    rng = np.random.default_rng(3)
    r = rng.normal(-0.25, 0.8, 800)
    s = fe.BookSeries(key="k", label="k", family="F", tier=1, gross=False,
                      r=r, clusters=[f"c{i}" for i in range(len(r))])
    row = fe.posterior(s)
    assert row["p_above_bar"] < 0.01 and fe.verdict(row) == "evidence against"


def test_verdict_never_speaks_for_an_unreadable_row():
    assert fe.verdict({"status": "db not synced locally"}) == "no read"


def test_ci_is_the_requested_width():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 1.0, 500)
    s = fe.BookSeries(key="k", label="k", family="F", tier=3, gross=False,
                      r=r, clusters=[f"c{i}" for i in range(len(r))])
    wide = fe.posterior(s, ci=0.99)
    narrow = fe.posterior(s, ci=0.50)
    assert (wide["ci_hi"] - wide["ci_lo"]) > (narrow["ci_hi"] - narrow["ci_lo"])


def test_sum_r_matches_the_scoreboard_arithmetic():
    r = np.array([1.0, -1.0, 0.5, -0.25])
    s = fe.BookSeries(key="k", label="k", family="F", tier=3, gross=False,
                      r=r, clusters=list("abcd"))
    row = fe.posterior(s)
    assert row["sum_r"] == pytest.approx(0.25)
    assert row["raw_mean_r"] == pytest.approx(0.0625)


# ══════════════════════════════════════════════════════════════════════════════
#  Era hygiene
# ══════════════════════════════════════════════════════════════════════════════
def test_era_split_books_stay_separate_rows():
    """Never pool recorder eras: the registry already splits them, and the sweep
    must not merge two eras of one book back together."""
    keys = {b.key for b in BOOKS}
    assert {"ofcs_paper_e1", "ofcs_paper_e2"} <= keys
    assert {"lr_signal_e1", "lr_signal_e2"} <= keys
    e1, e2 = BOOKS_BY_KEY["ofcs_paper_e1"], BOOKS_BY_KEY["ofcs_paper_e2"]
    assert e1.closed_filter != e2.closed_filter
    assert fe.is_gross(e1) and not fe.is_gross(e2)


def test_family_pooling_never_crosses_tier_or_toll_convention():
    import inspect
    src = inspect.getsource(fe.family_posteriors)
    assert "(s.family, s.tier, s.gross)" in src


def test_a_replicated_winner_cannot_carry_the_book():
    """One winning price path re-stamped forty times is one win, not forty. If
    the posterior were fitted on rows, the replicated cluster would set the
    point estimate; fitted on bets, the book reads as the two bets it is."""
    r = np.array([1.0] * 40 + [-1.0])
    clusters = ["setup::level-A"] * 40 + ["setup::level-B"]
    s = fe.BookSeries(key="k", label="k", family="F", tier=3, gross=False,
                      r=r, clusters=clusters)
    assert s.n == 41 and s.n_eff == 2
    assert list(s.bet_r()) == [1.0, -1.0]
    row = fe.posterior(s)
    assert row["raw_mean_r"] == pytest.approx(0.9512, abs=1e-3), "the row mean is flattering"
    assert row["bet_mean_r"] == pytest.approx(0.0), "the bet mean is the truth"
    assert row["post_mean_r"] < 0.2
    assert fe.verdict(row) == "prior dominates"


def test_bet_r_is_the_raw_series_when_nothing_clusters():
    r = np.array([1.0, -1.0, 0.5])
    s = fe.BookSeries(key="k", label="k", family="F", tier=3, gross=False,
                      r=r, clusters=["a", "b", "c"])
    assert list(s.bet_r()) == list(r)
