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
        _closes: np.ndarray = None,
    ) -> Optional[TradeResult]:
        """Walk forward bar-by-bar from signal entry, applying SL/TP1/TP2 and costs.

        Returns a TradeResult or None if the signal is invalid (zero risk or no bars).
        Pre-extracted _highs/_lows/_closes arrays can be passed to avoid repeated .values calls.
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

        entry_cost = entry_price * (costs.spread_pct + costs.slippage_pct)
        effective_entry = entry_price + entry_cost if direction == 'LONG' else entry_price - entry_cost

        outcome = 'timeout'
        exit_price = None
        bars_held = 0
        mfe = 0.0
        mae = 0.0
        tp1_hit = False
        is_long = direction == 'LONG'

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
                    stop_loss = effective_entry  # Trail to breakeven
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
                    stop_loss = effective_entry  # Trail to breakeven
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

        # Entry spread+slippage already in effective_entry (line 456-457),
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
