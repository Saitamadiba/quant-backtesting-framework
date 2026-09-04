"""Hermetic tests for WS7's analyst-drift reader.

The theme is counting: five headlines about one stock on one day are five rows
and one bet, and the difference decides when a frozen gate opens.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DASH = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DASH))
sys.path.insert(0, str(_ROOT))

from data import analyst_drift as ad                          # noqa: E402


def _events(spec=(("SNOW", "2026-09-03", 5), ("AAPL", "2026-09-01", 1),
                  ("DELL", "2026-09-02", 2))):
    """One row per fill; `m_gross` identical inside a (symbol, date) group,
    exactly as the real store records it."""
    rows, nid = [], 0
    for sym, date, n_fills in spec:
        drift = {"SNOW": -0.059, "AAPL": 0.0258, "DELL": 0.0564}.get(sym, 0.01)
        for k in range(n_fills):
            nid += 1
            rows.append({"news_id": nid, "symbol": sym, "eff_date": date,
                         "status": "closed", "pnl_usd": 100.0 * (k + 1),
                         "m_gross": drift, "m_fill": drift + 0.001 * k,
                         "sent": 1, "pit_rank": 100, "beta": 1.5,
                         "oc_spy": 0.003})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  The counting finding
# ══════════════════════════════════════════════════════════════════════════════
def test_repeated_headlines_about_one_stock_on_one_day_are_one_bet():
    ev = _events()
    assert len(ad.closed_events(ev, cluster=False)) == 8
    assert len(ad.closed_events(ev, cluster=True)) == 3


def test_counting_report_shows_both_readings_and_the_ratio():
    r = ad.counting_report(_events())
    assert r["closed_rows"] == 8 and r["distinct_bets"] == 3
    assert r["distinct_news_id"] == 8, "each fill carries its own news id"
    assert r["overcount_ratio"] == pytest.approx(8 / 3)


def test_the_standard_unit_progresses_slower_than_the_shadow():
    """Counting headlines would open the gate on a smaller sample than it claims;
    since amendment 1 the standard unit is the slower, stricter one."""
    r = ad.counting_report(_events())
    assert r["gate_progress_by_bets"] < r["gate_progress_by_rows"]


def test_clustering_keeps_the_drift_and_sums_the_pnl():
    bets = ad.closed_events(_events(), cluster=True)
    snow = bets[bets.symbol == "SNOW"].iloc[0]
    assert snow["m_gross"] == pytest.approx(-0.059), "the drift is the event's, not a mean"
    assert snow["pnl_usd"] == pytest.approx(100 + 200 + 300 + 400 + 500)
    assert snow["fills"] == 5


def test_open_and_unpriced_rows_are_excluded():
    ev = _events()
    ev.loc[0, "status"] = "expired"
    ev.loc[1, "pnl_usd"] = np.nan
    assert len(ad.closed_events(ev, cluster=False)) == 6


# ══════════════════════════════════════════════════════════════════════════════
#  The posterior, and the honesty of "prior dominates"
# ══════════════════════════════════════════════════════════════════════════════
def test_posterior_is_measured_against_the_seats_own_gate_not_zero():
    out = ad.posterior(_events())
    assert out["status"] == "ok"
    assert "p_above_gate" in out and ad.GATE_M_PP == 0.05


def test_prior_dominates_flag_is_on_below_the_plan_floor():
    out = ad.posterior(_events())
    assert out["n_bets"] < ad.WS4_WS6_N
    assert out["prior_dominates"] is True


def test_the_posterior_hit_rate_is_pulled_toward_the_prior():
    """Three bets cannot move a skeptical Beta(50,50) far from 50%."""
    out = ad.posterior(_events())
    assert abs(out["post_hit_rate"] - 0.5) < abs(out["raw_hit_rate"] - 0.5)


def test_posterior_refuses_a_single_bet():
    out = ad.posterior(_events(spec=(("AAPL", "2026-09-01", 3),)))
    assert out["status"].startswith("only")


def test_posterior_uses_bets_not_rows():
    out = ad.posterior(_events())
    assert out["n_bets"] == 3, "eight fills of three events are three observations"


# ══════════════════════════════════════════════════════════════════════════════
#  The VIX band is refused with reasons, not returned empty
# ══════════════════════════════════════════════════════════════════════════════
def test_vol_band_atlas_refuses_and_says_why():
    out = ad.vol_band_atlas(_events())
    assert out["status"] == "not askable"
    assert len(out["reasons"]) >= 3
    assert any("^VIX" in r for r in out["reasons"])
    assert any("roll" in r for r in out["reasons"]), (
        "a VIX ETF's level percentile is a roll-yield gauge, not a fear gauge")
    assert any("trading day" in r for r in out["reasons"])


# ══════════════════════════════════════════════════════════════════════════════
#  Spine shape: contemporaneous columns must not read as pre-trade
# ══════════════════════════════════════════════════════════════════════════════
def test_oc_spy_is_flagged_post_not_pre():
    """The session's own SPY move is contemporaneous with the trade, so it can
    never be a pre-trade feature however useful it looks."""
    blk = ad.equities_block(_events())
    assert "post__oc_spy" in blk.columns
    assert "pre__oc_spy" not in blk.columns


def test_known_before_the_open_columns_are_pre():
    blk = ad.equities_block(_events())
    for c in ("pre__sent", "pre__pit_rank", "pre__beta", "pre__dow"):
        assert c in blk.columns


def test_readiness_reports_both_gates_in_bets():
    r = ad.readiness(_events())
    assert set(r["gate"]) == {"seat review gate (frozen)", "WS4 / WS6 (plan floor)"}
    assert (r["have"] == 3).all()
    assert r.loc[r.gate == "seat review gate (frozen)", "needs"].iloc[0] == 1000


def test_nothing_here_shortens_the_seats_frozen_gate():
    assert ad.GATE_N == 1000 and ad.GATE_M_PP == 0.05


def test_module_never_touches_the_vps():
    import ast
    tree = ast.parse(Path(ad.__file__).read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert not imported & {"subprocess", "paramiko", "socket", "requests", "urllib"}


# ══════════════════════════════════════════════════════════════════════════════
#  AMENDMENT 1 (2026-09-04, SIGHTED): symbol-days standard, headlines shadow
# ══════════════════════════════════════════════════════════════════════════════
def test_the_standard_unit_is_symbol_days():
    out = ad.dual_read(_events())
    assert "symbol, eff_date" in out["gate_unit"] and "STANDARD" in out["gate_unit"]
    assert out["n_gate"] == 3, "the gate counts distinct symbol-days"


def test_headlines_are_now_the_record_only_shadow():
    out = ad.dual_read(_events())
    assert "record-only" in out["shadow_unit"] and "headlines" in out["shadow_unit"]
    assert out["n_shadow"] == 8
    assert out["ratio"] == pytest.approx(8 / 3)


def test_the_amendment_makes_the_gate_harder_not_easier():
    """A sighted amendment that makes a gate EASIER is the dangerous kind. This
    one delays the review: fewer standard units accrue per calendar day."""
    out = ad.dual_read(_events())
    assert out["gate_pct"] < out["shadow_pct"]


def test_both_series_stay_visible_so_the_ratio_can_be_watched():
    r = ad.counting_report(_events())
    assert r["closed_rows"] == 8 and r["distinct_bets"] == 3
    assert r["overcount_ratio"] == pytest.approx(8 / 3)


def test_the_shadow_series_takes_m_gross_once_because_it_is_identical():
    sd = ad.symbol_day_series(_events())
    snow = sd[sd.symbol == "SNOW"].iloc[0]
    assert snow["m_gross"] == pytest.approx(-0.059)
    assert snow["headlines"] == 5


def test_m_fill_is_notional_weighted_not_a_plain_mean():
    """A 300-share fill and a 3-share fill must not speak equally about what the
    book actually captured."""
    ev = _events(spec=(("XYZ", "2026-09-02", 2),))
    ev["qty"] = [1.0, 99.0]
    ev["entry_fill"] = [100.0, 100.0]
    sd = ad.symbol_day_series(ev)
    m = sd["m_fill"].iloc[0]
    plain = ev["m_fill"].mean()
    assert m != pytest.approx(plain)
    assert m == pytest.approx((0.01 * 100 + 0.011 * 9900) / 10000)


def test_the_shadow_series_is_derived_not_a_second_store():
    """Persisting a recomputable view adds a place to drift out of sync without
    adding information."""
    src = Path(ad.__file__).read_text()
    assert "DERIVED view, not a second store" in src
    assert "to_parquet" not in src and "sqlite3.connect" not in src


def test_the_frozen_aggregation_rule_is_written_down():
    assert ad.SHADOW_AGG["m_gross"].startswith("first")
    assert "notional" in ad.SHADOW_AGG["m_fill"]


# ══════════════════════════════════════════════════════════════════════════════
#  The risk face of the same clustering
# ══════════════════════════════════════════════════════════════════════════════
def test_concentration_is_measured_against_the_design_target():
    ev = _events(spec=(("SNOW", "2026-09-03", 5),))
    ev["qty"] = [100.0] * 5
    ev["entry_fill"] = [120.0] * 5
    rep = ad.concentration_report(ev)
    assert rep["x_design"].iloc[0] == pytest.approx(60000 / ad.DESIGN_NOTIONAL_PER_EVENT)
    assert rep["headlines"].iloc[0] == 5


def test_concentration_reports_share_of_the_book():
    ev = _events(spec=(("SNOW", "2026-09-03", 5),))
    ev["qty"] = [100.0] * 5
    ev["entry_fill"] = [120.0] * 5
    rep = ad.concentration_report(ev)
    assert 0.5 < rep["pct_of_book"].iloc[0] < 0.7
    assert ad.DESIGN_NOTIONAL_PER_EVENT == 10_000.0


def test_the_amendment_is_recorded_as_sighted_in_the_prereg():
    """An amendment made after looking at data must say so, so a reader can
    discount it. Precedent: the FVG era-3 kill-bar amendment."""
    pre = _ROOT / "us_markets" / "ANALYST_DRIFT_PAPER_PREREG.md"
    if not pre.exists():
        pytest.skip("prereg not in this checkout")
    txt = pre.read_text()
    assert "AMENDMENT 1" in txt and "SIGHTED" in txt
    assert "no era break is taken" in txt
    assert "harder" in txt.lower(), "the direction of the change must be stated"


def test_the_seats_trading_rules_are_untouched_by_the_amendment():
    """The amendment is a reading rule. Sizing, halts and the metric are
    era-break-protected and must not appear as changed."""
    assert ad.GATE_N == 1000 and ad.GATE_M_PP == 0.05
    assert ad.DESIGN_NOTIONAL_PER_EVENT == 10_000.0
