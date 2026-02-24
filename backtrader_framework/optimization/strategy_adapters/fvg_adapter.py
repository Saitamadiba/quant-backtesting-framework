"""
FVG (Fair Value Gap) adapter for WFO engine.

Pure pandas/numpy signal generation, no backtrader dependency.
Optimized: vectorized FVG detection with numpy shifted arrays,
incremental gap tracking with deque for O(active_gaps) per bar.

Improvements over baseline (v2):
    - Cross-TF detection: detects FVGs on 1h candles (via HTF_1h_* columns)
      when running on 15m data, entries on native resolution
    - IV regime gating: DVOL-based direction filter
      (LOW IV → LONGs, HIGH IV → SHORTs, MED IV → skip)
    - Adaptive R:R: tighter targets in HIGH IV, full in LOW IV
    - Displacement filter: requires minimum 5-bar price move before entry
    - Increased minimum gap size (0.2% default)

Signal flow:
    1. Detect 3-candle imbalance patterns (bullish/bearish FVGs)
       - On HTF_1h candles if available (cross-TF), else native
    2. Track active FVGs with age expiry (max_fvg_age bars)
    3. IV regime gate: filter direction by DVOL regime
    4. Displacement gate: require minimum directional move
    5. Enter when price retraces into gap zone (fill_entry_min to fill_entry_max)
    6. Require confirmation candle (close in expected direction)
    7. SL beyond gap boundary + ATR buffer; TP at regime-adaptive R:R

Confidence scoring (6 factors, max 1.0):
    - Gap size relative to price (0-0.25)
    - Volume confirmation at gap creation (0/0.20)
    - EMA bias alignment (0/0.15)
    - RSI alignment (0/0.10)
    - Structure bias agreement (0/0.15)
    - Displacement strength (0-0.15)

WFO Results (improved vs baseline):
    BTC 1h:  -0.297R / 31.0% WR / 448 trades  (was -0.357R / 21.7% / 3228)
    ETH 1h:  -0.119R / 33.5% WR / 603 trades  (was -0.305R / 21.8% / 3838)
    BTC 15m: -0.479R / 30.3% WR / 231 trades  (was -0.601R / 22.0% / 7281)
    ETH 15m: -0.301R / 35.7% WR / 300 trades  (was -0.465R / 22.3% / 10273)
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal


# FVG direction constants
_DIR_BULL = 0
_DIR_BEAR = 1


class FVGAdapter(StrategyAdapter):

    @property
    def name(self) -> str:
        return "FVG"

    @property
    def default_timeframes(self) -> List[str]:
        return ["1h", "15m"]

    def get_param_space(self) -> List[ParamSpec]:
        """Parameter space for Fair Value Gap detection and entry."""
        return [
            ParamSpec("min_gap_pct",      0.002,  0.001, 0.005, 0.001),
            ParamSpec("max_fvg_age",      50,     20,    80,    20,    'int'),
            ParamSpec("fill_entry_min",   0.30,   0.20,  0.40,  0.10),
            ParamSpec("fill_entry_max",   0.70,   0.60,  0.80,  0.10),
            ParamSpec("atr_sl_buffer",    0.5,    0.3,   0.9,   0.3),
            ParamSpec("rr_target",        2.0,    1.5,   3.0,   0.5),
            ParamSpec("min_confidence",   0.40,   0.25,  0.60,  0.05),
            ParamSpec("displacement_pct", 0.005,  0.003, 0.010, 0.002),
        ]

    # ────────────────────────────────────────────────────────────
    #  Cross-TF helpers
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_htf_1h_bars(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, ...]]:
        """Extract unique 1h bars from forward-filled HTF_1h_* columns.

        Identifies 1h bar boundaries via change-point detection on the
        forward-filled HTF_1h_Close/HTF_1h_High columns. Each group of
        native-TF bars sharing the same HTF values belongs to one 1h bar.

        Returns (htf_highs, htf_lows, htf_closes, bar_start_indices,
                 bar_end_idx) or None if HTF columns are missing.
        """
        raise NotImplementedError(
            "FVG cross-TF detection is proprietary. "
            "See module docstring for architecture overview."
        )

    @staticmethod
    def _detect_htf_fvgs(
        htf_highs: np.ndarray,
        htf_lows: np.ndarray,
        htf_closes: np.ndarray,
        bar_start_indices: np.ndarray,
        bar_end_idx: np.ndarray,
        min_gap: float,
        vol_mean_20: np.ndarray,
        volumes: np.ndarray,
    ) -> List[tuple]:
        """Detect FVGs on 1h bars and map activation to native-TF indices.

        Scans reconstructed 1h candles for 3-bar imbalance patterns
        (bullish: bar[k-2].high < bar[k].low, bearish: inverse).
        Maps each FVG's activation point to the native-TF index where
        the 3rd 1h candle closes (no look-ahead).

        Returns list of (activation_native_idx, direction, gap_high,
                         gap_low, vol_conf, gap_pct).
        """
        raise NotImplementedError(
            "FVG HTF detection is proprietary. "
            "See module docstring for architecture overview."
        )

    # ────────────────────────────────────────────────────────────
    #  Main signal generation
    # ────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        """
        Generate FVG trade signals over [scan_start_idx, scan_end_idx).

        Architecture:
            1. Cross-TF or native FVG detection
               - If HTF_1h_* columns exist: detect on 1h, enter on native
               - Otherwise: vectorized native-TF detection (highs[:-2] < lows[2:])
            2. Deque-based incremental gap tracking with age expiry
            3. Per-bar filtering pipeline:
               a. IV regime gate (DVOL < 45 → LONGs, >= 65 → SHORTs, 45-65 → skip)
               b. Displacement gate (5-bar minimum directional move)
               c. Fill zone check (price in fill_entry_min to fill_entry_max of gap)
               d. Direction-specific IV gate
               e. 6-factor confidence scoring with min_confidence threshold
            4. Regime-adaptive R:R (LOW IV: 1.0x, HIGH IV: 0.5x, no DVOL: 0.75x)
            5. SL beyond gap boundary + ATR buffer; TP at effective R:R

        Required columns: Open, High, Low, Close, Volume, ATR
        Optional columns: RSI, EMA50, EMA200, StructureBias, DVOL,
                         HTF_1h_Open, HTF_1h_High, HTF_1h_Low, HTF_1h_Close
        """
        raise NotImplementedError(
            "FVG signal generation is proprietary. "
            "See module docstring for signal flow and WFO results."
        )
