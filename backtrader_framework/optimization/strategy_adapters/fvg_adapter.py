"""
FVG (Fair Value Gap) adapter for WFO engine.

Pure pandas signal generation, no backtrader dependency.
Optimized: uses numpy arrays instead of .iloc[] for fast grid search.

Signal logic removed — this is a public stub showing the adapter interface.
"""

from typing import Dict, List, Any

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal


class FVGAdapter(StrategyAdapter):

    @property
    def name(self) -> str:
        return "FVG"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m", "1h"]

    def get_param_space(self) -> List[ParamSpec]:
        """Parameter space for Fair Value Gap detection and entry."""
        return [
            ParamSpec("min_gap_pct",    0.001,  0.0005, 0.003,  0.0005),
            ParamSpec("max_fvg_age",    50,     20,     80,     20,    'int'),
            ParamSpec("fill_entry_min", 0.30,   0.20,   0.40,   0.10),
            ParamSpec("fill_entry_max", 0.70,   0.60,   0.80,   0.10),
            ParamSpec("atr_sl_buffer",  0.5,    0.3,    0.9,    0.3),
            ParamSpec("rr_target",      2.0,    1.5,    3.0,    0.5),
        ]

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        """
        Generate FVG trade signals over [scan_start_idx, scan_end_idx).

        Strategy overview (logic removed for IP protection):
        - Detects bullish/bearish Fair Value Gaps (3-candle imbalance patterns)
        - Tracks active FVGs with expiry (max_fvg_age)
        - Enters on price retracement into the gap zone (fill_entry_min/max)
        - Requires confirmation candle in the expected direction
        - ATR-based stop loss below/above gap boundary
        - Multi-factor confidence scoring (fill quality, volume, RSI, bias)

        Args:
            df: OHLCV DataFrame with indicator columns (ATR, RSI, EMA, etc.)
            params: Dict of parameter values from get_param_space()
            scan_start_idx: First bar index to scan
            scan_end_idx: Last bar index to scan (exclusive)

        Returns:
            List of Signal objects with entry/exit levels and metadata
        """
        raise NotImplementedError(
            "Signal generation logic is proprietary. "
            "See base_adapter.py for the interface contract."
        )
