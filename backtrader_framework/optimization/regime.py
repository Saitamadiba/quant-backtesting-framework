"""
Market regime detection for Walk-Forward Optimization.

Provides the RegimeDetector class which classifies market conditions
(trending_up, trending_down, ranging, volatile) from indicator state
using ADX, ATR, and EMA lookback.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Classify market regime (trending_up/down, ranging, volatile) from indicator state."""

    @staticmethod
    def classify(df: pd.DataFrame, idx: int) -> str:
        """Return the regime string for bar at idx using ADX, ATR, and EMA lookback."""
        if idx < 50:
            return 'unknown'
        window = df.iloc[max(0, idx - 50):idx + 1]
        adx = window['ADX'].iloc[-1] if 'ADX' in window.columns else 20
        atr = window['ATR'].iloc[-1] if 'ATR' in window.columns else 0
        price = window['Close'].iloc[-1]

        atr_pct = atr / price if price > 0 else 0
        avg_atr_pct = (window['ATR'] / window['Close']).mean() if 'ATR' in window.columns else atr_pct

        if atr_pct > avg_atr_pct * 1.8:
            return 'volatile'
        elif adx > 30:
            if 'EMA50' in window.columns and window['Close'].iloc[-1] > window['EMA50'].iloc[-1]:
                return 'trending_up'
            else:
                return 'trending_down'
        else:
            return 'ranging'
