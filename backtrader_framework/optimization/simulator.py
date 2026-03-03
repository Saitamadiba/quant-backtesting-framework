"""
Trade simulation engine for Walk-Forward Optimization.

Provides the TradeSimulator class which simulates trade execution with
spread, commission, slippage, and TP/SL logic on OHLCV data. Uses
conservative same-bar ambiguity resolution (SL checked before TP).
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .wfo_engine import TransactionCosts, TradeResult

logger = logging.getLogger(__name__)


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

        entry_cost = entry_price * (costs.spread_pct + costs.slippage_pct)
        effective_entry = entry_price + entry_cost if direction == 'LONG' else entry_price - entry_cost

        outcome = 'timeout'
        exit_price = None
        bars_held = 0
        mfe = 0.0
        mae = 0.0
        tp1_hit = False
        is_long = direction == 'LONG'
        high_water = 0.0
        low_water = float('inf')

        end_bar = min(idx + max_bars, n)
        for i in range(idx + 1, end_bar):
            h = highs[i]
            lo = lows[i]
            bars_held += 1

            # Conservative same-bar TP/SL ambiguity resolution:
            # Within a single OHLCV bar we cannot determine whether SL or TP
            # was hit first.  We assume the WORST outcome — SL is always
            # checked before TP for both LONG and SHORT directions.  This
            # avoids overstating backtest performance.
            if is_long:
                favorable = (h - effective_entry) / risk
                adverse = (effective_entry - lo) / risk
                if favorable > mfe:
                    mfe = favorable
                if adverse > mae:
                    mae = adverse

                # SL checked first (conservative: assume SL hit before TP)
                if lo <= stop_loss:
                    outcome = 'breakeven' if tp1_hit else 'loss'
                    exit_price = stop_loss
                    break
                if not tp1_hit and h >= tp1:
                    tp1_hit = True
                    stop_loss = effective_entry  # Breakeven floor
                    high_water = h
                if tp1_hit:
                    if h > high_water:
                        high_water = h
                    # ATR trailing: ratchet SL up from high water mark
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

                # SL checked first (conservative: assume SL hit before TP)
                if h >= stop_loss:
                    outcome = 'breakeven' if tp1_hit else 'loss'
                    exit_price = stop_loss
                    break
                if not tp1_hit and lo <= tp1:
                    tp1_hit = True
                    stop_loss = effective_entry  # Breakeven floor
                    low_water = lo
                if tp1_hit:
                    if lo < low_water:
                        low_water = lo
                    # ATR trailing: ratchet SL down from low water mark
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

        # Entry spread+slippage already in effective_entry,
        # so only charge commission for entry to avoid double-counting.
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
    ) -> Optional[TradeResult]:
        """V2 simulator replicating the live bot's execution logic.

        Adds over simulate():
        - Initial volatility buffer (virtual wider SL for first N bars)
        - Breakeven trigger at configurable R (0.5R default)
        - Stepped trailing stop (direction-specific ATR multipliers)
        - Time-based exit (close if not in profit after N bars)

        All new parameters are read from signal['metadata'].
        """
        idx = signal['idx']
        direction = signal['direction']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        tp = signal['take_profit_1']
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

        # Direction-specific trailing
        is_long = direction == 'LONG'
        trail_atr_mult = meta.get('trail_atr_mult_long' if is_long else 'trail_atr_mult_short', 0)
        trail_step_atr = meta.get('trail_step_atr_long' if is_long else 'trail_step_atr_short', 0)

        # Entry cost
        entry_cost = entry_price * (costs.spread_pct + costs.slippage_pct)
        effective_entry = entry_price + entry_cost if is_long else entry_price - entry_cost

        # SL distance for buffer calculation
        sl_distance = abs(effective_entry - stop_loss)
        original_sl = stop_loss

        outcome = 'timeout'
        exit_price = None
        bars_held = 0
        mfe = 0.0
        mae = 0.0
        be_triggered = False
        trailing_active = False
        high_water = effective_entry if is_long else 0.0
        low_water = effective_entry if not is_long else float('inf')
        last_trail_price = effective_entry  # Price at last trail update

        end_bar = min(idx + max_bars, n)
        for i in range(idx + 1, end_bar):
            h = highs[i]
            lo = lows[i]
            cl = closes[i]
            bars_held += 1

            # Current ATR for trailing
            atr_i = atrs[i] if atrs is not None and i < len(atrs) else 0

            # --- MFE/MAE tracking ---
            if is_long:
                favorable = (h - effective_entry) / risk
                adverse = (effective_entry - lo) / risk
            else:
                favorable = (effective_entry - lo) / risk
                adverse = (h - effective_entry) / risk
            if favorable > mfe:
                mfe = favorable
            if adverse > mae:
                mae = adverse

            # --- 1. INITIAL BUFFER (first N bars) ---
            in_buffer = bars_held <= buffer_bars and buffer_bars > 0
            if in_buffer:
                # Virtual SL = wider by buffer_mult
                virtual_sl_dist = sl_distance * buffer_mult
                if is_long:
                    virtual_sl = effective_entry - virtual_sl_dist
                    if lo <= virtual_sl:
                        outcome = 'loss'
                        exit_price = virtual_sl
                        break
                    # Real SL hit but within buffer → ignore (let trade recover)
                else:
                    virtual_sl = effective_entry + virtual_sl_dist
                    if h >= virtual_sl:
                        outcome = 'loss'
                        exit_price = virtual_sl
                        break
                    # Real SL hit but within buffer → ignore
            else:
                # --- 2. BREAKEVEN CHECK (at be_trigger_r profit) ---
                if not be_triggered:
                    if is_long:
                        current_r = (h - effective_entry) / risk
                    else:
                        current_r = (effective_entry - lo) / risk
                    if current_r >= be_trigger_r:
                        be_triggered = True
                        trailing_active = True
                        if is_long:
                            be_level = effective_entry + effective_entry * be_buffer_pct
                            if be_level > stop_loss:
                                stop_loss = be_level
                        else:
                            be_level = effective_entry - effective_entry * be_buffer_pct
                            if be_level < stop_loss:
                                stop_loss = be_level

                # --- 3. STEPPED TRAILING (after breakeven) ---
                if trailing_active and trail_atr_mult > 0 and atr_i > 0:
                    if is_long:
                        if h > high_water:
                            high_water = h
                        # Only update trail on significant moves
                        step_ok = (trail_step_atr <= 0 or
                                   high_water - last_trail_price >= trail_step_atr * atr_i)
                        if step_ok:
                            trail_level = high_water - trail_atr_mult * atr_i
                            if trail_level > stop_loss:
                                stop_loss = trail_level
                                last_trail_price = high_water
                    else:
                        if lo < low_water:
                            low_water = lo
                        step_ok = (trail_step_atr <= 0 or
                                   last_trail_price - low_water >= trail_step_atr * atr_i)
                        if step_ok:
                            trail_level = low_water + trail_atr_mult * atr_i
                            if trail_level < stop_loss:
                                stop_loss = trail_level
                                last_trail_price = low_water

                # --- 4. SL CHECK (conservative: SL before TP) ---
                if is_long:
                    if lo <= stop_loss:
                        if be_triggered:
                            raw_exit_r = (stop_loss - effective_entry) / risk
                            outcome = 'win_trail' if raw_exit_r > 0.05 else 'breakeven'
                        else:
                            outcome = 'loss'
                        exit_price = stop_loss
                        break
                else:
                    if h >= stop_loss:
                        if be_triggered:
                            raw_exit_r = (effective_entry - stop_loss) / risk
                            outcome = 'win_trail' if raw_exit_r > 0.05 else 'breakeven'
                        else:
                            outcome = 'loss'
                        exit_price = stop_loss
                        break

                # --- 5. TP CHECK ---
                if is_long and h >= tp:
                    outcome = 'win_tp'
                    exit_price = tp
                    break
                elif not is_long and lo <= tp:
                    outcome = 'win_tp'
                    exit_price = tp
                    break

            # --- 6. TIME EXIT (after time_exit_bars) ---
            if time_exit_bars > 0 and bars_held >= time_exit_bars:
                if is_long:
                    in_profit = cl > effective_entry
                else:
                    in_profit = cl < effective_entry
                if not in_profit:
                    outcome = 'time_exit'
                    exit_price = cl
                    break

        # Timeout handling
        if outcome == 'timeout':
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
            take_profit_1=tp,
            take_profit_2=tp,
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
