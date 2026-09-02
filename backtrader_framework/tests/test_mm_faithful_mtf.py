#!/usr/bin/env python3
"""Tests for the MMC MTF gate added to FaithfulMMAdapter (2026-09-02).

Three properties matter and each has a test:
  1. DEFAULT-OFF — every existing research/WFO caller must be byte-identical,
     or the change silently invalidates studies that use this adapter.
  2. IT BLOCKS THE THING IT WAS ADDED FOR — the live MMC arm was selling into a
     daily bias of LONG with 4H structure BULLISH.
  3. NO LOOK-AHEAD — the gate's verdict at time t must not move when future bars
     arrive. The live LR copy on the VPS still has this wrong; MM must not
     inherit it.

Hermetic: synthetic OHLC, a stub base adapter, a symbol with no DVOL parquet.

Run:  python3 -m pytest backtrader_framework/tests/test_mm_faithful_mtf.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtrader_framework.optimization.strategy_adapters.mm_faithful_filters import (  # noqa: E402
    FaithfulMMAdapter, _mtf_at, compute_daily_mtf, compute_4h_structure,
)

SYM = "TESTCOIN"          # no dvol_vrp_TESTCOIN.parquet => DVOL stays NaN, band 'n/a'


# ── fixtures ──────────────────────────────────────────────────────────
class _Sig:
    def __init__(self, ts, direction, idx=0):
        self.time, self.direction, self.idx = ts, direction, idx
        self.metadata = {}


class _StubBase:
    """Emits whatever it is handed — isolates the gate from MM signal logic."""
    name = "mm_stub"
    default_timeframes = ("15m",)

    def __init__(self, sigs):
        self._sigs = sigs

    def generate_signals(self, df, params, scan_start_idx, scan_end_idx):
        return list(self._sigs)

    def get_default_params(self):
        return {}

    def get_param_space(self):
        return {}


def _frame(closes, start="2025-01-01"):
    """15m OHLC frame from a per-BAR close series."""
    idx = pd.date_range(start=start, periods=len(closes), freq="15min", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * 0.999,
                         "Close": c, "Volume": 1.0}, index=idx)


BARS_PER_DAY = 96                 # 15m bars
DAYS = 260                        # > 200 daily bars, so the daily EMA200 is mature


def _uptrend(days=DAYS):
    """A long, steady climb: daily EMA50 > EMA200 and close > EMA50 => bias LONG.
    Whole days only, so a per-day fixture lands on a calendar boundary."""
    return _frame(100.0 * np.exp(np.linspace(0, 1.2, days * BARS_PER_DAY)))


def _downtrend(days=DAYS):
    return _frame(100.0 * np.exp(np.linspace(0, -1.2, days * BARS_PER_DAY)))


# ── 1. default-off ────────────────────────────────────────────────────
def test_default_construction_leaves_the_gate_off():
    a = FaithfulMMAdapter(symbol=SYM, base=_StubBase([]))
    assert a.min_mtf == 0.0
    assert a.block_opposite_daily is False
    assert a._mtf_armed() is False


def test_default_adapter_passes_signals_through_unchanged():
    df = _uptrend()
    sigs = [_Sig(df.index[-10], "SHORT"), _Sig(df.index[-9], "LONG")]
    a = FaithfulMMAdapter(symbol=SYM, base=_StubBase(sigs))
    out = a.generate_signals(df, {}, 0, len(df))
    assert len(out) == 2, "default adapter must not filter — WFO studies depend on it"
    assert all("mtf_score" not in (s.metadata or {}) for s in out)


def test_mtf_frames_are_not_built_when_the_gate_is_off():
    """A WFO sweep that never reads the frames must not pay to resample them."""
    df = _uptrend()
    a = FaithfulMMAdapter(symbol=SYM, base=_StubBase([]))
    ctx = a._ctx_for(df)
    assert "daily_mtf" not in ctx and "h4_struct" not in ctx
    b = FaithfulMMAdapter(symbol=SYM, base=_StubBase([]), min_mtf=50.0)
    ctx_b = b._ctx_for(df)
    assert "daily_mtf" in ctx_b and "h4_struct" in ctx_b


# ── 2. it blocks what it was added for ────────────────────────────────
def test_short_into_a_daily_uptrend_is_blocked():
    """The live 2026-08-31 shape: MMC sold ETH four times with daily bias LONG."""
    df = _uptrend()
    ts = df.index[-10]
    daily, h4 = compute_daily_mtf(df), compute_4h_structure(df)
    assert _mtf_at(ts, "LONG", daily, h4)[1] == "LONG", "fixture is not a daily uptrend"

    a = FaithfulMMAdapter(symbol=SYM, base=_StubBase([_Sig(ts, "SHORT")]),
                          min_mtf=50.0, block_opposite_daily=True)
    assert a.generate_signals(df, {}, 0, len(df)) == []


def test_long_with_the_daily_trend_survives():
    df = _uptrend()
    ts = df.index[-10]
    a = FaithfulMMAdapter(symbol=SYM, base=_StubBase([_Sig(ts, "LONG")]),
                          min_mtf=50.0, block_opposite_daily=True)
    out = a.generate_signals(df, {}, 0, len(df))
    assert len(out) == 1
    assert out[0].metadata["daily_bias"] == "LONG"
    assert out[0].metadata["mtf_score"] >= 50.0


def test_short_with_the_daily_downtrend_survives():
    df = _downtrend()
    ts = df.index[-10]
    a = FaithfulMMAdapter(symbol=SYM, base=_StubBase([_Sig(ts, "SHORT")]),
                          min_mtf=50.0, block_opposite_daily=True)
    out = a.generate_signals(df, {}, 0, len(df))
    assert len(out) == 1 and out[0].metadata["daily_bias"] == "SHORT"


def test_block_opposite_daily_closes_the_h4_only_hole():
    """The score is 20 base + 50*daily + 30*h4, so a fully-aligned 4H leg alone
    reaches exactly 50 and clears a floor of 50 while the daily disagrees. That
    is the hole LR has open. With block_opposite_daily the same signal dies."""
    df = _uptrend()
    ts = df.index[-10]
    sig = _Sig(ts, "SHORT")
    daily, h4 = compute_daily_mtf(df), compute_4h_structure(df)
    score, bias = _mtf_at(ts, "SHORT", daily, h4)
    assert bias == "LONG"

    floor_only = FaithfulMMAdapter(symbol=SYM, base=_StubBase([sig]),
                                   min_mtf=50.0, block_opposite_daily=False)
    both = FaithfulMMAdapter(symbol=SYM, base=_StubBase([_Sig(ts, "SHORT")]),
                             min_mtf=50.0, block_opposite_daily=True)
    # whatever the 4H leg says, the opposite-daily block is never laxer
    assert len(both.generate_signals(df, {}, 0, len(df))) \
        <= len(floor_only.generate_signals(df, {}, 0, len(df)))
    assert both.generate_signals(df, {}, 0, len(df)) == []


def test_min_mtf_zero_with_block_opposite_daily_still_arms():
    df = _uptrend()
    ts = df.index[-10]
    a = FaithfulMMAdapter(symbol=SYM, base=_StubBase([_Sig(ts, "SHORT")]),
                          min_mtf=0.0, block_opposite_daily=True)
    assert a._mtf_armed() is True
    assert a.generate_signals(df, {}, 0, len(df)) == []


# ── 3. no look-ahead ──────────────────────────────────────────────────
def test_verdict_does_not_change_when_future_bars_arrive():
    """The general property: what the gate knows at t must be a function of
    bars up to t. Under the pre-2026-08-17 call sites (daily '<=', 4h '<=' ets)
    appending future bars moves the answer."""
    df = _uptrend()
    for offset in (10, 97, 300):
        ts = df.index[-offset]
        past = df.loc[df.index <= ts]
        for direction in ("LONG", "SHORT"):
            full_v = _mtf_at(ts, direction, compute_daily_mtf(df), compute_4h_structure(df))
            past_v = _mtf_at(ts, direction, compute_daily_mtf(past), compute_4h_structure(past))
            assert full_v == past_v, (offset, direction, full_v, past_v)


def test_daily_leg_cannot_read_todays_final_close():
    """Crash the LAST day below its EMA50 after a long climb. Read strictly
    before today's date => still LONG (yesterday's completed row). Read '<='
    => it sees this evening's close and flips. Only the first is knowable."""
    up = 100.0 * np.exp(np.linspace(0, 1.2, DAYS * BARS_PER_DAY))
    # exactly one extra CALENDAR day, crashed — the frame starts at 00:00 and
    # every leg is a whole number of days, so the boundary is unambiguous
    closes = np.concatenate([up, np.full(BARS_PER_DAY, up[-1] * 0.55)])
    df = _frame(closes)
    ts = df.index[-BARS_PER_DAY // 2]        # midday on the crashed day
    daily = compute_daily_mtf(df)

    assert _mtf_at(ts, "LONG", daily, compute_4h_structure(df))[1] == "LONG"
    peeked = daily.loc[daily.index <= ts.normalize()].iloc[-1]["bias"]
    assert peeked != "LONG", "fixture did not create a look-ahead discriminator"


def test_4h_leg_only_reads_closed_bins():
    """A bin labelled T covers [T, T+4h), so at a timestamp INSIDE bin T the
    structure of T is still being written and must not be readable. Built on a
    reversal so the current bin and the last closed one genuinely disagree —
    on a smooth trend every bin says the same thing and the bug hides."""
    from backtrader_framework.optimization.strategy_adapters.lr_faithful_filters import (
        mtf_score as _score)
    up = 100.0 * np.exp(np.linspace(0, 1.2, DAYS * BARS_PER_DAY))
    down = up[-1] * np.exp(np.linspace(0, -0.35, 3 * BARS_PER_DAY))
    df = _frame(np.concatenate([up, down]))
    h4 = compute_4h_structure(df)
    daily = compute_daily_mtf(df)

    # find a bin whose structure differs from the previous (closed) one
    flips = [i for i in range(1, len(h4))
             if h4["h4_structure"].iloc[i] != h4["h4_structure"].iloc[i - 1]]
    assert flips, "fixture produced no 4h structure change to discriminate on"
    i = flips[-1]
    ets = h4.index[i] + pd.Timedelta(hours=1)          # inside the UNCLOSED bin i
    assert ets <= df.index[-1], "flip bin is beyond the frame"

    for direction in ("LONG", "SHORT"):
        try:
            r0 = daily.loc[daily.index < ets.normalize()].iloc[-1]
            d_bias, d_str = str(r0["bias"]), float(r0["strength"])
        except IndexError:
            d_bias, d_str = "NEUTRAL", 0.0
        closed = _score(direction, d_bias, d_str,
                        str(h4["h4_structure"].iloc[i - 1]),
                        float(h4["h4_strength"].iloc[i - 1]))
        peeked = _score(direction, d_bias, d_str,
                        str(h4["h4_structure"].iloc[i]),
                        float(h4["h4_strength"].iloc[i]))
        if closed == peeked:
            continue                                   # not a discriminator here
        assert _mtf_at(ets, direction, daily, h4)[0] == closed, (
            direction, _mtf_at(ets, direction, daily, h4)[0], closed, peeked)
        return
    raise AssertionError("no direction discriminated closed-vs-peeked 4h bin")


def test_naive_timestamp_is_treated_as_utc_not_crashed():
    df = _uptrend()
    ts = df.index[-10]
    a = FaithfulMMAdapter(symbol=SYM, base=_StubBase([_Sig(ts.tz_localize(None), "SHORT")]),
                          min_mtf=50.0, block_opposite_daily=True)
    assert a.generate_signals(df, {}, 0, len(df)) == []


def _main():
    fails = []
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            fails.append((name, e))
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_main())
