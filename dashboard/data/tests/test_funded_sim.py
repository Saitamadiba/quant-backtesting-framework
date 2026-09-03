"""Rule-engine tests for the HyroTrader funded simulator.

Each test builds a tiny synthetic trade history engineered to trip exactly one
rule, then asserts the engine flags it (and the overall verdict).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

_DASH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DASH))

from data.funded_sim import (  # noqa: E402
    prepare_trades, simulate, compare_sizes, HYRO_TRIAL, ACCOUNT_SIZES,
)

BAL = 10_000


def _raw(trades):
    """trades: list of (day, direction, entry, sl, exit, reason)."""
    rows = []
    for i, (day, d, e, sl, xp, reason) in enumerate(trades):
        rows.append(dict(
            trade_id=i, strategy="T", symbol="BTC", direction=d,
            entry_price=e, stop_loss=sl, take_profit=None, exit_price=xp,
            exit_reason=reason, pnl_usd=None,
            entry_time=pd.Timestamp(day, tz="UTC"),
            exit_time=pd.Timestamp(day, tz="UTC"),
        ))
    return pd.DataFrame(rows)


def _long_win(day, ret):
    """A long with sl_dist=1% and a given fractional return (ret>0 winner)."""
    return (day, "Long", 100.0, 99.0, 100.0 * (1 + ret), "Take Profit")


def test_pass_when_target_and_days_met():
    # 6 winners on 6 distinct days, +2% each (notional 1.0 × 2% move) → +12%.
    raw = _raw([_long_win(f"2026-01-0{i}", 0.02) for i in range(1, 7)])
    r = simulate(prepare_trades(raw), BAL, risk_pct=0.01)
    assert r.rules["Profit target"]["status"] is True
    assert r.rules["Min trading days"]["value"] == 6
    assert r.verdict == "PASS"


def test_max_loss_breach():
    # 6 full -2R losers (one per day so daily DD stays clean) → −12% cumulative,
    # crossing the −10% floor. Each trade: 2% adverse move on a 1% stop = -2R.
    raw = _raw([("2026-01-0%d" % i, "Long", 100.0, 99.0, 98.0, "Stop Loss")
                for i in range(1, 7)])
    r = simulate(prepare_trades(raw), BAL, risk_pct=0.01)
    assert r.rules["Max loss"]["status"] is False
    assert r.verdict == "BREACH"
    assert "Max loss" in r.terminal_event


def test_extreme_R_is_clamped():
    # a logged 12R blowup (12% move on a 1% stop) is guarded down to the floor.
    raw = _raw([("2026-01-01", "Long", 100.0, 99.0, 88.0, "Stop Loss")])
    p = prepare_trades(raw)
    assert bool(p["_clamped"].iloc[0]) is True
    assert p["R"].iloc[0] == pytest.approx(-2.0)


def test_daily_drawdown_breach():
    # same day: +6% then −7% → intraday high-to-low 7% > 5% limit.
    raw = _raw([_long_win("2026-01-01", 0.06),
                ("2026-01-01", "Long", 100.0, 99.0, 93.0, "Stop Loss")])  # −7%
    r = simulate(prepare_trades(raw), BAL, risk_pct=0.01)
    assert r.rules["Daily drawdown"]["status"] is False
    assert r.verdict == "BREACH"
    assert "Daily" in r.terminal_event


def test_min_trading_days_blocks_pass():
    # target reached (+6%) but only 3 distinct days → not a PASS.
    raw = _raw([_long_win(f"2026-01-0{i}", 0.02) for i in (1, 2, 3)])
    r = simulate(prepare_trades(raw), BAL, risk_pct=0.01)
    assert r.rules["Profit target"]["status"] is True
    assert r.rules["Min trading days"]["status"] is False
    assert r.verdict == "IN PROGRESS"


def test_stop_loss_risk_over_3pct_flagged():
    # risk_pct=5% on a 2% stop → notional 2.5×, risk 5% > 3% cap on every trade
    # (2% stop keeps it off the 3× leverage cap so the over-risk actually shows).
    raw = _raw([("2026-01-0%d" % i, "Long", 100.0, 98.0, 101.0, "Take Profit")
                for i in range(1, 7)])
    r = simulate(prepare_trades(raw), BAL, risk_pct=0.05)
    assert r.rules["Stop-loss obligation"]["status"] is False
    assert r.rules["Stop-loss obligation"]["value"] == 6


def test_exit_reconstruction_from_reason():
    # missing exit_price, reason win_tp2 + a TP set → positive return recovered.
    raw = pd.DataFrame([dict(
        trade_id=0, strategy="T", symbol="BTC", direction="Long",
        entry_price=100.0, stop_loss=99.0, take_profit=103.0, exit_price=None,
        exit_reason="win_tp2", entry_time=pd.Timestamp("2026-01-01", tz="UTC"),
        exit_time=pd.Timestamp("2026-01-01", tz="UTC"))])
    p = prepare_trades(raw)
    assert len(p) == 1
    assert p["ret_frac"].iloc[0] == pytest.approx(0.03, rel=1e-6)


def test_size_invariance_of_verdict():
    # the verdict must not depend on account size (rules are %-based).
    raw = _raw([_long_win(f"2026-01-0{i}", 0.02) for i in range(1, 7)])
    prepared = prepare_trades(raw)
    verdicts = {bal: simulate(prepared, bal, 0.01).verdict for bal in ACCOUNT_SIZES}
    assert len(set(verdicts.values())) == 1
    cmp = compare_sizes(prepared, 0.01)
    assert len(cmp) == len(ACCOUNT_SIZES)
    # dollars scale linearly: $1M net pnl == 100× the $10k net pnl
    by = cmp.set_index("Account")["Net PnL $"]
    assert by["$1,000,000"] == pytest.approx(by["$10,000"] * 100, rel=1e-6)


def test_round_trip_fee_reduces_return():
    # identical +2R winner; a 50 bps round-trip fee drops net return by 0.50%.
    base = dict(trade_id=0, strategy="K", symbol="BTC", direction="Long",
                entry_price=100.0, stop_loss=99.0, take_profit=None,
                exit_price=102.0, exit_reason="Take Profit",
                entry_time=pd.Timestamp("2026-01-01", tz="UTC"),
                exit_time=pd.Timestamp("2026-01-01", tz="UTC"))
    free = prepare_trades(pd.DataFrame([base]))
    fee = prepare_trades(pd.DataFrame([dict(base, fee_bps=50.0)]))
    assert free["ret_frac"].iloc[0] == pytest.approx(0.02)
    assert fee["ret_frac"].iloc[0] == pytest.approx(0.02 - 0.005)


def test_empty_input():
    r = simulate(prepare_trades(pd.DataFrame()), BAL)
    assert r.verdict == "NO DATA"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
