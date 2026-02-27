"""
Fair Value Gap (FVG) Strategy for Backtrader.

15-minute timeframe strategy that trades FVG retests during kill zones.
Strategy logic removed — this stub shows the class structure.
"""

import backtrader as bt

from backtrader_framework.strategies.base_strategy import BaseStrategy


class FVGStrategy(BaseStrategy):
    """
    Fair Value Gap Strategy

    Detects 3-candle imbalance zones (FVGs) and enters on retracement
    into the gap with confirmation. Uses ATR-based risk management.

    Implementation removed for IP protection.
    See base_strategy.py for the shared framework.
    """

    params = (
        ('min_gap_pct', 0.001),
        ('max_fvg_age', 50),
        ('fill_entry_min', 0.30),
        ('fill_entry_max', 0.70),
        ('atr_sl_buffer', 0.5),
        ('rr_target', 2.0),
    )

    def __init__(self):
        super().__init__()

    def next(self):
        super().next()
        # Strategy logic removed
