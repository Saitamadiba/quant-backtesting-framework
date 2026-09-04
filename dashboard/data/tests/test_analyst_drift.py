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


def test_the_gate_progresses_slower_by_bets_than_by_rows():
    """Counting rows would open a frozen gate on a smaller sample than it claims."""
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
