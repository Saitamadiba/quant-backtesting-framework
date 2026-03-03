"""
Liquidity Raid Live adapter — faithful replication of the VPS bot's execution.

Extends LiquidityRaidAdapter with execute_signals() for stateful sequential
processing, matching the live bot's behavior:
    - Single position constraint (1 trade at a time)
    - Re-entry mechanism (2 attempts after stop-out, 4-bar cooldown)
    - Initial volatility buffer (virtual wider SL for first 4 bars)
    - Stepped trailing stop (direction-specific ATR multipliers)
    - Breakeven at 0.5R (with 0.1% buffer)
    - Time exit after 6h if not in profit
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal
from .liquidity_raid_adapter import (
    LiquidityRaidAdapter,
    _compute_session_levels,
    _utc_to_et_hours,
)

logger = logging.getLogger(__name__)


def _bars_for_hours(hours: float, timeframe: str) -> int:
    """Convert a duration in hours to bar count for the given timeframe."""
    tf_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '1d': 1440,
    }
    minutes_per_bar = tf_minutes.get(timeframe, 15)
    return max(1, int(hours * 60 / minutes_per_bar))


class LiquidityRaidLiveAdapter(StrategyAdapter):
    """Faithful replication of the live VPS bot's trading logic for WFO.

    Signal generation reuses LiquidityRaidAdapter logic. The key addition is
    execute_signals() which processes signals sequentially with:
    - Single position constraint
    - Re-entry mechanism after stop-outs
    - Live-bot execution parameters injected into simulate_v2()
    """

    def __init__(self, timeframe: str = '15m'):
        self._base = LiquidityRaidAdapter()
        self._timeframe = timeframe

    @property
    def name(self) -> str:
        return "LiquidityRaidLive"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m", "5m", "1h"]

    def get_param_space(self) -> List[ParamSpec]:
        return [
            ParamSpec("session_lookback",   12,    6,     18,    6,    'int'),
            ParamSpec("atr_sl_multiplier",  2.5,   1.5,   3.5,   0.5),
            ParamSpec("rr_ratio",           0.5,   0.3,   1.0,   0.5),
            ParamSpec("min_body_pct",       0.15,  0.10,  0.25,  0.05),
            ParamSpec("sweep_tolerance",    0.002, 0.001, 0.003, 0.001),
            ParamSpec("min_confidence",     0.25,  0.15,  0.45,  0.05),
            ParamSpec("reentry_max",        2,     0,     3,     1,    'int'),
        ]

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        """Delegate signal generation to base LiquidityRaidAdapter.

        Maps rr_ratio → min_rr/max_rr since base adapter uses those names.
        """
        base_params = dict(params)
        rr = base_params.pop('rr_ratio', 0.5)
        base_params.pop('reentry_max', None)
        base_params['min_rr'] = rr
        base_params['max_rr'] = rr  # Single TP (live bot uses 1 TP)
        return self._base.generate_signals(df, base_params, scan_start_idx, scan_end_idx)

    def execute_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
        costs: Any,
        max_bars: int = 168,
        window_id: int = 0,
        is_oos: bool = True,
        regime: str = 'unknown',
    ) -> Optional[List[Any]]:
        """Sequential signal processing with single-position + re-entry."""
        from ..simulator import TradeSimulator

        # Generate all signals
        signals = self.generate_signals(df, params, scan_start_idx, scan_end_idx)
        if not signals:
            return []

        signals.sort(key=lambda s: s.idx)

        # Pre-extract arrays for performance
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        atrs = df['ATR'].values if 'ATR' in df.columns else None

        # Timeframe-aware constants
        tf = self._timeframe
        time_exit_bars = _bars_for_hours(6.0, tf)
        buffer_bars = _bars_for_hours(1.0, tf)
        reentry_cooldown = _bars_for_hours(1.0, tf)
        reentry_window = _bars_for_hours(2.5, tf)
        reentry_max = int(params.get('reentry_max', 2))
        rr_ratio = params.get('rr_ratio', 0.5)
        atr_sl_mult = params.get('atr_sl_multiplier', 2.5)

        # Live-bot metadata for simulate_v2
        live_meta = {
            'initial_buffer_bars': buffer_bars,
            'initial_buffer_mult': 1.35,
            'breakeven_trigger_r': 0.5,
            'breakeven_buffer_pct': 0.001,
            'time_exit_bars': time_exit_bars,
            'trail_atr_mult_long': 2.5,
            'trail_atr_mult_short': 3.0,
            'trail_step_atr_long': 0.5,
            'trail_step_atr_short': 0.75,
        }

        trades = []
        position_exit_bar = -1

        # Re-entry tracking: list of (stop_bar, direction, sweep_price, signal)
        stop_outs = []

        # --- Pass 1: Primary signals with single-position constraint ---
        for sig in signals:
            # Single-position constraint
            if sig.idx < position_exit_bar:
                continue

            sig_dict = sig.to_dict()
            sig_dict['metadata'] = {**(sig_dict.get('metadata') or {}), **live_meta}

            trade = TradeSimulator.simulate_v2(
                sig_dict, df, costs, max_bars, window_id,
                is_oos, regime,
                _highs=highs, _lows=lows, _closes=closes, _atrs=atrs,
            )
            if trade is None:
                continue

            trades.append(trade)
            exit_bar = sig.idx + trade.bars_held
            position_exit_bar = exit_bar

            # Track stop-outs for re-entry
            if trade.outcome == 'loss':
                stop_outs.append({
                    'stop_bar': exit_bar,
                    'direction': sig.direction,
                    'sweep_price': sig.entry_price,  # Use entry as proxy
                    'atr_at_entry': sig.atr,
                    'original_signal': sig,
                })

        # --- Pass 2: Re-entry attempts after stop-outs ---
        if reentry_max <= 0 or not stop_outs:
            return trades

        reentry_trades = []
        n = len(df)

        for so in stop_outs:
            attempts = 0
            search_start = so['stop_bar'] + reentry_cooldown
            search_end = min(so['stop_bar'] + reentry_window, n)

            if search_start >= search_end or search_start >= n:
                continue

            for bar_i in range(search_start, search_end):
                if attempts >= reentry_max:
                    break

                # Check if another position would be active at this bar
                # (from primary trades or earlier re-entries)
                conflict = False
                for t in trades:
                    t_entry_bar = _find_entry_bar(t, df)
                    if t_entry_bar is not None:
                        t_exit_bar = t_entry_bar + t.bars_held
                        if t_entry_bar <= bar_i < t_exit_bar:
                            conflict = True
                            break
                for t in reentry_trades:
                    t_entry_bar = _find_entry_bar(t, df)
                    if t_entry_bar is not None:
                        t_exit_bar = t_entry_bar + t.bars_held
                        if t_entry_bar <= bar_i < t_exit_bar:
                            conflict = True
                            break
                if conflict:
                    continue

                h = highs[bar_i]
                lo = lows[bar_i]
                cl = closes[bar_i]
                op = df['Open'].values[bar_i]
                is_bullish = cl > op
                is_bearish = cl < op

                atr_val = atrs[bar_i] if atrs is not None and bar_i < len(atrs) else so['atr_at_entry']
                if atr_val <= 0 or np.isnan(atr_val):
                    continue

                sweep_price = so['sweep_price']
                direction = so['direction']

                # Re-entry conditions (matching live bot)
                if direction == 'LONG':
                    # Price near sweep level (within 0.5%) AND bullish candle
                    near_sweep = abs(lo - sweep_price) / sweep_price < 0.005
                    if not (near_sweep and is_bullish):
                        continue
                    # 50% pullback entry
                    candle_range = h - lo
                    if candle_range <= 0:
                        continue
                    entry = lo + candle_range * 0.5
                    sl = sweep_price - atr_val * 1.0
                    if sl >= entry:
                        continue
                    risk = entry - sl
                    tp = entry + risk * rr_ratio
                else:
                    near_sweep = abs(h - sweep_price) / sweep_price < 0.005
                    if not (near_sweep and is_bearish):
                        continue
                    candle_range = h - lo
                    if candle_range <= 0:
                        continue
                    entry = h - candle_range * 0.5
                    sl = sweep_price + atr_val * 1.0
                    if sl <= entry:
                        continue
                    risk = sl - entry
                    tp = entry - risk * rr_ratio

                reentry_sig = {
                    'idx': bar_i,
                    'time': df.index[bar_i],
                    'direction': direction,
                    'entry_price': entry,
                    'stop_loss': sl,
                    'take_profit_1': tp,
                    'risk': risk,
                    'confidence': 0.5,
                    'bias': 'REENTRY',
                    'atr': atr_val,
                    'metadata': {
                        **live_meta,
                        'signal_type': f'REENTRY_{direction}',
                    },
                }

                trade = TradeSimulator.simulate_v2(
                    reentry_sig, df, costs, max_bars, window_id,
                    is_oos, regime,
                    _highs=highs, _lows=lows, _closes=closes, _atrs=atrs,
                )
                if trade is None:
                    attempts += 1
                    continue

                reentry_trades.append(trade)
                attempts += 1
                # Position is now open, skip to after this trade closes
                break

        trades.extend(reentry_trades)
        # Sort by entry time for consistent ordering
        trades.sort(key=lambda t: t.entry_time)
        return trades


def _find_entry_bar(trade, df: pd.DataFrame) -> Optional[int]:
    """Find the bar index for a trade's entry time."""
    try:
        return df.index.get_loc(trade.entry_time)
    except (KeyError, TypeError):
        return None
