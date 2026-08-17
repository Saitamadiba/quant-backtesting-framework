"""
Trade simulation engine for Walk-Forward Optimization.

Provides the TradeSimulator class which simulates trade execution with
spread, commission, slippage, and TP/SL logic on OHLCV data. Uses
conservative same-bar ambiguity resolution (SL checked before TP).

When numba is available, the inner bar-by-bar loops are JIT-compiled
for ~10-100x speedup. The public API is unchanged.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .wfo_engine import TransactionCosts, TradeResult

# ── Numba kernel import (optional) ──────────────────────────
try:
    from .numba_kernels import (
        _simulate_v1_kernel,
        _simulate_v2_kernel,
        HAS_NUMBA,
        OUTCOME_STRINGS,
    )
except ImportError:          # numba_kernels itself unavailable
    HAS_NUMBA = False
    OUTCOME_STRINGS = {}
    _simulate_v1_kernel = None
    _simulate_v2_kernel = None

logger = logging.getLogger(__name__)

_EMPTY_F64 = np.empty(0, dtype=np.float64)


class TradeSimulator:
    """Simulate trade execution with spread, commission, slippage, and TP/SL logic."""

    @staticmethod
    def simulate(
        signal: Dict, df: pd.DataFrame, costs: TransactionCosts,
        max_bars: int = 168, window_id: int = 0,
        is_oos: bool = True, regime: str = 'unknown',
        _highs: np.ndarray = None, _lows: np.ndarray = None,
        _closes: np.ndarray = None, _atrs: np.ndarray = None,
    ) -> Optional[TradeResult]:
        """Walk forward bar-by-bar from signal entry, applying SL/TP1/TP2 and costs.

        Returns a TradeResult or None if the signal is invalid (zero risk or no bars).
        Pre-extracted _highs/_lows/_closes/_atrs arrays can be passed to avoid repeated .values calls.

        ATR trailing: when signal metadata contains ``trail_atr_mult > 0`` and
        ``_atrs`` is provided, the stop-loss trails at ``trail_atr_mult × ATR``
        from the high/low water mark after TP1 is hit (instead of simple breakeven).
        """
        idx = signal['idx']
        direction = signal['direction']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        tp1 = signal['take_profit_1']
        tp2 = signal.get('take_profit_2') or tp1
        risk = signal['risk']

        n = len(df)
        if risk <= 0 or idx + 1 >= n:
            return None

        # Use pre-extracted arrays if provided, else extract
        highs = _highs if _highs is not None else df['High'].values
        lows = _lows if _lows is not None else df['Low'].values
        closes = _closes if _closes is not None else df['Close'].values
        atrs = _atrs if _atrs is not None else (df['ATR'].values if 'ATR' in df.columns else None)

        # ATR trailing config from signal metadata (backward-compatible)
        meta = signal.get('metadata', {}) or {}
        trail_atr_mult = meta.get('trail_atr_mult', 0)

        is_long = direction == 'LONG'

        # ── Numba fast path ─────────────────────────────────
        if HAS_NUMBA:
            has_atrs = atrs is not None
            atrs_arr = atrs if has_atrs else _EMPTY_F64

            result = _simulate_v1_kernel(
                idx, is_long, entry_price, stop_loss, tp1, tp2, risk,
                costs.spread_pct, costs.commission_pct, costs.slippage_pct,
                max_bars, float(trail_atr_mult),
                highs, lows, closes, atrs_arr, has_atrs, n,
            )
            outcome_code, exit_px, bars_held, mfe, mae, raw_r, total_cost, final_sl = result

            if exit_px < 0.0:
                return None

            return TradeResult(
                entry_time=signal['time'],
                exit_time=df.index[min(idx + bars_held, n - 1)],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_px,
                stop_loss=final_sl,
                take_profit_1=tp1,
                take_profit_2=tp2,
                outcome=OUTCOME_STRINGS[int(outcome_code)],
                r_multiple=raw_r,
                r_multiple_after_costs=raw_r - total_cost,
                bars_held=bars_held,
                confidence=signal.get('confidence', 0.5),
                bias=signal.get('bias', 'COUNTER'),
                mfe=mfe,
                mae=mae,
                window_id=window_id,
                is_oos=is_oos,
                regime=regime,
                cost_deducted=total_cost,
            )

        # ── Pure-Python fallback ────────────────────────────
        entry_cost = entry_price * (costs.spread_pct + costs.slippage_pct)
        effective_entry = entry_price + entry_cost if is_long else entry_price - entry_cost

        outcome = 'timeout'
        exit_price = None
        bars_held = 0
        mfe = 0.0
        mae = 0.0
        tp1_hit = False
        high_water = 0.0
        low_water = float('inf')

        end_bar = min(idx + max_bars, n)
        for i in range(idx + 1, end_bar):
            h = highs[i]
            lo = lows[i]
            bars_held += 1

            if is_long:
                favorable = (h - effective_entry) / risk
                adverse = (effective_entry - lo) / risk
                if favorable > mfe:
                    mfe = favorable
                if adverse > mae:
                    mae = adverse

                if lo <= stop_loss:
                    outcome = 'breakeven' if tp1_hit else 'loss'
                    exit_price = stop_loss
                    break
                if not tp1_hit and h >= tp1:
                    tp1_hit = True
                    stop_loss = effective_entry
                    high_water = h
                if tp1_hit:
                    if h > high_water:
                        high_water = h
                    if trail_atr_mult > 0 and atrs is not None and i < len(atrs):
                        trail_level = high_water - trail_atr_mult * atrs[i]
                        if trail_level > stop_loss:
                            stop_loss = trail_level
                if h >= tp2:
                    outcome = 'win_tp2'
                    exit_price = tp2
                    break
            else:
                favorable = (effective_entry - lo) / risk
                adverse = (h - effective_entry) / risk
                if favorable > mfe:
                    mfe = favorable
                if adverse > mae:
                    mae = adverse

                if h >= stop_loss:
                    outcome = 'breakeven' if tp1_hit else 'loss'
                    exit_price = stop_loss
                    break
                if not tp1_hit and lo <= tp1:
                    tp1_hit = True
                    stop_loss = effective_entry
                    low_water = lo
                if tp1_hit:
                    if lo < low_water:
                        low_water = lo
                    if trail_atr_mult > 0 and atrs is not None and i < len(atrs):
                        trail_level = low_water + trail_atr_mult * atrs[i]
                        if trail_level < stop_loss:
                            stop_loss = trail_level
                if lo <= tp2:
                    outcome = 'win_tp2'
                    exit_price = tp2
                    break

        if outcome == 'timeout':
            if tp1_hit:
                outcome = 'win_tp1'
                exit_price = tp1
            else:
                last_idx = min(idx + max_bars - 1, n - 1)
                exit_price = closes[last_idx]

        if exit_price is None:
            return None

        if is_long:
            raw_r = (exit_price - effective_entry) / risk
        else:
            raw_r = (effective_entry - exit_price) / risk

        entry_comm = entry_price * costs.commission_pct
        exit_cost = exit_price * (costs.spread_pct + costs.commission_pct + costs.slippage_pct)
        total_cost = (entry_comm + exit_cost) / risk if risk > 0 else 0

        return TradeResult(
            entry_time=signal['time'],
            exit_time=df.index[min(idx + bars_held, n - 1)],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            outcome=outcome,
            r_multiple=raw_r,
            r_multiple_after_costs=raw_r - total_cost,
            bars_held=bars_held,
            confidence=signal.get('confidence', 0.5),
            bias=signal.get('bias', 'COUNTER'),
            mfe=mfe,
            mae=mae,
            window_id=window_id,
            is_oos=is_oos,
            regime=regime,
            cost_deducted=total_cost,
        )

    @staticmethod
    def simulate_v2(
        signal: Dict, df: pd.DataFrame, costs: TransactionCosts,
        max_bars: int = 168, window_id: int = 0,
        is_oos: bool = True, regime: str = 'unknown',
        _highs: np.ndarray = None, _lows: np.ndarray = None,
        _closes: np.ndarray = None, _atrs: np.ndarray = None,
        tp_target: str = 'tp1',
    ) -> Optional[TradeResult]:
        """V2 simulator replicating the live bot's execution logic.

        Adds over simulate():
        - Initial volatility buffer (virtual wider SL for first N bars)
        - Breakeven trigger at configurable R (0.5R default)
        - Stepped trailing stop (direction-specific ATR multipliers)
        - Time-based exit (close if not in profit after N bars)

        All new parameters are read from signal['metadata'].

        tp_target : which level the kernel races the trailing stop against.
            ``'tp1'`` (default, this class's historical behaviour) or ``'tp2'``
            (what wfo_engine.TradeSimulator uses, on the reasoning that the live
            bot's trail-vs-target race runs against the FINAL target).

            This was previously hardcoded differently in the two TradeSimulator
            classes, which is most of why they returned different results for the
            same signal. Making it an explicit argument turns an accidental
            divergence into a measurable choice; it does not decide which is
            right. Default preserves this class's behaviour exactly.
        """
        if tp_target not in ('tp1', 'tp2'):
            raise ValueError(f"tp_target must be 'tp1' or 'tp2', got {tp_target!r}")
        idx = signal['idx']
        direction = signal['direction']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        tp1 = signal['take_profit_1']
        tp2 = signal.get('take_profit_2') or tp1
        tp = tp1 if tp_target == 'tp1' else tp2
        risk = signal['risk']

        n = len(df)
        if risk <= 0 or idx + 1 >= n:
            return None

        highs = _highs if _highs is not None else df['High'].values
        lows = _lows if _lows is not None else df['Low'].values
        closes = _closes if _closes is not None else df['Close'].values
        atrs = _atrs if _atrs is not None else (df['ATR'].values if 'ATR' in df.columns else None)

        meta = signal.get('metadata', {}) or {}

        # V2 parameters from metadata
        buffer_bars = meta.get('initial_buffer_bars', 0)
        buffer_mult = meta.get('initial_buffer_mult', 1.35)
        be_trigger_r = meta.get('breakeven_trigger_r', 0.5)
        be_buffer_pct = meta.get('breakeven_buffer_pct', 0.001)
        time_exit_bars = meta.get('time_exit_bars', 0)

        is_long = direction == 'LONG'
        trail_atr_mult = meta.get('trail_atr_mult_long' if is_long else 'trail_atr_mult_short', 0)
        trail_step_atr = meta.get('trail_step_atr_long' if is_long else 'trail_step_atr_short', 0)

        # MIN_TRAIL_LOCK_R floor and TRAIL_HEADROOM_FRAC cap. These were added to
        # _simulate_v2_kernel and wired up in wfo_engine.TradeSimulator, but NOT
        # here — this class is a separate, parallel TradeSimulator (see the
        # two-pipelines note below), so the positional call below was two
        # arguments short and simulate_v2() raised TypeError on EVERY call.
        # 0.0 disables both, which is the framework default and reproduces the
        # behaviour this method had before the parameters existed.
        # Both key spellings are accepted: this class reads unprefixed metadata
        # keys for its other v2 params, while the engine injects `v2_`-prefixed
        # ones.
        min_trail_lock_r = float(
            meta.get('v2_min_trail_lock_r', meta.get('min_trail_lock_r', 0.0))
        )
        trail_headroom_frac = float(
            meta.get('v2_trail_headroom_frac', meta.get('trail_headroom_frac', 0.0))
        )

        # ── Numba fast path ─────────────────────────────────
        # ── Single numeric core: _simulate_v2_kernel ────────
        # This was gated on `if HAS_NUMBA:` with a 168-line pure-Python V2
        # fallback below it. The kernel is a plain Python function when numba is
        # absent (numba_kernels applies a no-op decorator), so the gate never
        # chose between "fast" and "slow" — it chose between two DIFFERENT
        # implementations, and they disagreed. Measured over 300 random signals,
        # this class and wfo_engine.TradeSimulator returned identical results for
        # only 135/300 (77/300 under engine-injected metadata), with r_multiple
        # differing by more than 10x on individual trades. Whether numba happened
        # to be installed silently changed research numbers.
        #
        # The kernel is canonical ("matches the live LR position_manager.py logic
        # exactly"), so it is now the only path, and a missing kernel FAILS
        # CLOSED instead of quietly computing something else. The parameter
        # semantics of THIS class are unchanged — same metadata keys, same
        # defaults, same TP1 target — so its configuration surface is untouched.
        if _simulate_v2_kernel is None:
            raise RuntimeError(
                "simulate_v2 needs numba_kernels._simulate_v2_kernel, which "
                "failed to import. Refusing to fall back to a divergent "
                "implementation - fix the import instead."
            )
        has_atrs = atrs is not None
        atrs_arr = atrs if has_atrs else _EMPTY_F64

        result = _simulate_v2_kernel(
            idx, is_long, entry_price, stop_loss, tp, risk,
            costs.spread_pct, costs.commission_pct, costs.slippage_pct,
            max_bars,
            int(buffer_bars), float(buffer_mult), float(be_trigger_r),
            float(be_buffer_pct), int(time_exit_bars),
            float(trail_atr_mult), float(trail_step_atr),
            min_trail_lock_r, trail_headroom_frac,
            highs, lows, closes, atrs_arr, has_atrs, n,
        )
        outcome_code, exit_px, bars_held, mfe, mae, raw_r, total_cost, final_sl = result

        if exit_px < 0.0:
            return None

        return TradeResult(
            entry_time=signal['time'],
            exit_time=df.index[min(idx + bars_held, n - 1)],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_px,
            stop_loss=final_sl,
            # Report the signal's own levels, not the one handed to the kernel —
            # `tp` is the raced target selected by tp_target, and collapsing both
            # fields onto it loses the signal's actual TP1/TP2 geometry.
            take_profit_1=tp1,
            take_profit_2=tp2,
            outcome=OUTCOME_STRINGS[int(outcome_code)],
            r_multiple=raw_r,
            r_multiple_after_costs=raw_r - total_cost,
            bars_held=bars_held,
            confidence=signal.get('confidence', 0.5),
            bias=signal.get('bias', 'COUNTER'),
            mfe=mfe,
            mae=mae,
            window_id=window_id,
            is_oos=is_oos,
            regime=regime,
            cost_deducted=total_cost,
        )
