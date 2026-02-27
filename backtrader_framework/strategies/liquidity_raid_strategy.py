"""
Liquidity Raid Strategy for Backtrader.

15-minute timeframe strategy that trades session sweep setups.
Strategy logic removed — this stub shows the class structure.
"""

import backtrader as bt

from backtrader_framework.strategies.base_strategy import BaseStrategy


class LiquidityRaidStrategy(BaseStrategy):
    """
    Liquidity Raid Strategy

    Detects sweeps of session highs/lows and enters on rejection candle
    confirmation with dynamic R:R based on volatility regime.

    Implementation removed for IP protection.
    See base_strategy.py for the shared framework.
    """

    params = (
        ('session_lookback', 12),
        ('atr_sl_multiplier', 2.5),
        ('min_rr', 1.5),
        ('max_rr', 2.5),
        ('min_body_pct', 0.15),
        ('sweep_tolerance', 0.002),
    )

    def __init__(self):
        super().__init__()

    def next(self):
        super().next()
        # Strategy logic removed
