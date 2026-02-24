"""
SBS (Swing Breakout Sequence) adapter for WFO engine.

Pure pandas/numpy signal generation, no backtrader dependency.
Optimized: vectorized swing detection with numpy, loop only for Fib signals.

Signal flow:
    1. Detect swing highs/lows within lookback window
    2. Identify liquidity sweep beyond swing (price breaks level + closes back)
    3. Calculate Fibonacci retracement (sweep=1.0, swing=0.0)
    4. Enter when price retraces to 0.618 with confirmation candle
    5. SL at sweep level + ATR buffer, TP1 at 0.236, TP2 at swing (0.0)

Confidence scoring:
    - Sweep depth past swing level (0-0.30)
    - EMA bias alignment (0/0.20)
    - RSI alignment (0/0.15)
    - Volume confirmation (0/0.15)
    - Structure bias agreement (0/0.20)
"""

from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal


class SBSAdapter(StrategyAdapter):

    @property
    def name(self) -> str:
        return "SBS"

    @property
    def default_timeframes(self) -> List[str]:
        return ["4h", "1h"]

    def get_param_space(self) -> List[ParamSpec]:
        """Parameter space for Fibonacci-based swing breakout detection."""
        return [
            ParamSpec("lookback_period",       20,  10,   40,   10, 'int'),
            ParamSpec("sweep_tolerance_pct",  0.002, 0.001, 0.004, 0.001),
            ParamSpec("min_swing_pct",        0.02,  0.01,  0.04,  0.01),
            ParamSpec("min_confidence",       0.48,  0.35,  0.65,  0.10),
            ParamSpec("stop_loss_atr_buffer", 0.3,   0.1,   0.5,   0.2),
        ]

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        """
        Generate SBS trade signals over [scan_start_idx, scan_end_idx).

        Detects swing highs/lows, identifies Fibonacci retracement 0.618 sweeps,
        and enters with rejection candle confirmation.
        """
        lookback = int(params.get('lookback_period', 20))
        sweep_tol = params.get('sweep_tolerance_pct', 0.002)
        min_swing = params.get('min_swing_pct', 0.02)
        min_conf = params.get('min_confidence', 0.48)
        sl_buffer = params.get('stop_loss_atr_buffer', 0.3)

        s = scan_start_idx
        e = min(scan_end_idx, len(df))
        if e <= s + lookback:
            return []

        # Numpy arrays for the full dataframe (need lookback before scan)
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        atrs = df['ATR'].values
        volumes = df['Volume'].values

        has_rsi = 'RSI' in df.columns
        rsi = df['RSI'].values if has_rsi else np.full(len(df), 50.0)

        has_ema = 'EMA50' in df.columns and 'EMA200' in df.columns
        if has_ema:
            ema50 = df['EMA50'].values
            ema200 = df['EMA200'].values

        has_structure = 'StructureBias' in df.columns
        if has_structure:
            struct_bias = df['StructureBias'].values

        # Pre-compute rolling volume mean for the scan range
        vol_series = pd.Series(volumes)
        vol_mean_20 = vol_series.rolling(20, min_periods=10).mean().values

        signals: List[Signal] = []
        min_cooldown = 4
        last_sig_idx = -min_cooldown

        # Scan each bar: look for 0.618 sweep setups
        for i in range(max(s, lookback + 5), e):
            if i - last_sig_idx < min_cooldown:
                continue

            atr_val = atrs[i]
            if not (atr_val > 0) or np.isnan(atr_val):
                continue

            # Find swing high and swing low within lookback window
            window_start = max(0, i - lookback)
            window_highs = highs[window_start:i]
            window_lows = lows[window_start:i]

            if len(window_highs) < 5:
                continue

            swing_high_idx_rel = np.argmax(window_highs)
            swing_low_idx_rel = np.argmin(window_lows)
            swing_high_idx = window_start + swing_high_idx_rel
            swing_low_idx = window_start + swing_low_idx_rel
            swing_high = highs[swing_high_idx]
            swing_low = lows[swing_low_idx]

            # Validate minimum swing size
            swing_range = swing_high - swing_low
            mid_price = (swing_high + swing_low) / 2
            if mid_price <= 0 or swing_range / mid_price < min_swing:
                continue

            # Determine direction: most recent swing extreme defines bias
            # If swing_low is more recent → uptrend retracing → LONG setup
            # If swing_high is more recent → downtrend retracing → SHORT setup
            if swing_low_idx > swing_high_idx:
                direction = 'LONG'
            elif swing_high_idx > swing_low_idx:
                direction = 'SHORT'
            else:
                continue

            # Calculate Fibonacci levels
            # For LONG: swing_high=0.0, swing_low=1.0 (retracement down)
            # For SHORT: swing_low=0.0, swing_high=1.0 (retracement up)
            fib = _compute_fib_levels(swing_high, swing_low, direction)
            if fib is None:
                continue

            # Check 0.618 sweep: price crosses 0.618 then closes back
            if direction == 'LONG':
                # Price should dip below fib_618 then close above
                swept_below = lows[i] < fib['fib_618'] * (1 - sweep_tol)
                closed_above = closes[i] > fib['fib_618']
                is_rejection = closes[i] > opens[i]  # bullish candle

                # Also accept strong wick rejection
                body = abs(closes[i] - opens[i])
                lower_wick = min(opens[i], closes[i]) - lows[i]
                wick_rejection = lower_wick > body * 0.5 if body > 0 else False

                if not (swept_below and closed_above and (is_rejection or wick_rejection)):
                    continue

            else:  # SHORT
                # Price should push above fib_618 then close below
                swept_above = highs[i] > fib['fib_618'] * (1 + sweep_tol)
                closed_below = closes[i] < fib['fib_618']
                is_rejection = closes[i] < opens[i]  # bearish candle

                body = abs(closes[i] - opens[i])
                upper_wick = highs[i] - max(opens[i], closes[i])
                wick_rejection = upper_wick > body * 0.5 if body > 0 else False

                if not (swept_above and closed_below and (is_rejection or wick_rejection)):
                    continue

            # ── Confidence scoring ─────────────────────────────────
            confidence = 0.0

            # 1. Sweep depth past 0.618 (0 to 0.30)
            if direction == 'LONG':
                depth = fib['fib_618'] - lows[i]
            else:
                depth = highs[i] - fib['fib_618']
            depth_atr = depth / atr_val if atr_val > 0 else 0
            confidence += min(depth_atr / 0.5, 1.0) * 0.30

            # 2. EMA bias alignment (+0.20)
            if has_ema:
                if direction == 'LONG' and ema50[i] > ema200[i]:
                    confidence += 0.20
                elif direction == 'SHORT' and ema50[i] < ema200[i]:
                    confidence += 0.20

            # 3. RSI alignment (+0.15)
            if has_rsi:
                r = rsi[i]
                if direction == 'LONG' and 30 <= r <= 50:
                    confidence += 0.15
                elif direction == 'SHORT' and 50 <= r <= 70:
                    confidence += 0.15

            # 4. Volume confirmation (+0.15)
            vm = vol_mean_20[i]
            if not np.isnan(vm) and vm > 0 and volumes[i] > vm * 1.2:
                confidence += 0.15

            # 5. Structure bias (+0.20)
            if has_structure:
                sb = struct_bias[i]
                if direction == 'LONG' and sb > 0:
                    confidence += 0.20
                elif direction == 'SHORT' and sb < 0:
                    confidence += 0.20

            # Max: 0.30 + 0.20 + 0.15 + 0.15 + 0.20 = 1.0

            if confidence < min_conf:
                continue

            # ── Entry / SL / TP ────────────────────────────────────
            entry = closes[i]

            if direction == 'LONG':
                stop_loss = fib['fib_1'] - atr_val * sl_buffer
                if stop_loss >= entry:
                    continue
                risk = entry - stop_loss
                tp1 = fib['fib_236']
                tp2 = fib['fib_0']
            else:
                stop_loss = fib['fib_1'] + atr_val * sl_buffer
                if stop_loss <= entry:
                    continue
                risk = stop_loss - entry
                tp1 = fib['fib_236']
                tp2 = fib['fib_0']

            if risk <= 0:
                continue

            # Determine bias alignment
            ema_aligned = False
            if has_ema:
                ema_aligned = (
                    (direction == 'LONG' and ema50[i] > ema200[i]) or
                    (direction == 'SHORT' and ema50[i] < ema200[i])
                )
            struct_aligned = False
            if has_structure:
                sb = struct_bias[i]
                struct_aligned = (
                    (direction == 'LONG' and sb > 0) or
                    (direction == 'SHORT' and sb < 0)
                )

            signals.append(Signal(
                idx=i,
                time=df.index[i],
                direction=direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk=risk,
                confidence=confidence,
                bias='ALIGNED' if (ema_aligned and struct_aligned) else 'PARTIAL',
                atr=atr_val,
            ))
            last_sig_idx = i

        return signals


# ────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ────────────────────────────────────────────────────────────────────

def _compute_fib_levels(
    swing_high: float, swing_low: float, direction: str,
) -> Optional[Dict[str, float]]:
    """Compute Fibonacci retracement levels from swing range.

    For LONG (uptrend retracing down):
        0.0 = swing_high (TP2), 0.236 = TP1, 0.618 = entry, 1.0 = swing_low (SL)

    For SHORT (downtrend retracing up):
        0.0 = swing_low (TP2), 0.236 = TP1, 0.618 = entry, 1.0 = swing_high (SL)
    """
    fib_range = swing_high - swing_low
    if fib_range <= 0:
        return None

    if direction == 'LONG':
        return {
            'fib_0':   swing_high,                         # 0% — TP2 (swing)
            'fib_236': swing_high - fib_range * 0.236,     # 23.6% — TP1
            'fib_5':   swing_high - fib_range * 0.500,     # 50% — trailing SL
            'fib_618': swing_high - fib_range * 0.618,     # 61.8% — ENTRY
            'fib_1':   swing_low,                          # 100% — SL zone
        }
    else:
        return {
            'fib_0':   swing_low,                          # 0% — TP2 (swing)
            'fib_236': swing_low + fib_range * 0.236,      # 23.6% — TP1
            'fib_5':   swing_low + fib_range * 0.500,      # 50% — trailing SL
            'fib_618': swing_low + fib_range * 0.618,      # 61.8% — ENTRY
            'fib_1':   swing_high,                         # 100% — SL zone
        }
