"""
FVG (Fair Value Gap) adapter for WFO engine.

Pure pandas/numpy signal generation, no backtrader dependency.
Optimized: vectorized FVG detection with numpy shifted arrays,
loop only for gap tracking and fill-entry signals.

Signal flow:
    1. Detect 3-candle imbalance patterns (bullish/bearish FVGs)
    2. Track active FVGs with age expiry (max_fvg_age bars)
    3. Enter when price retraces into gap zone (fill_entry_min to fill_entry_max)
    4. Require confirmation candle (close in expected direction)
    5. SL beyond gap boundary + ATR buffer; TP at R:R target

Confidence scoring:
    - Gap size relative to price (0-0.25)
    - Volume confirmation at gap creation (0/0.20)
    - EMA bias alignment (0/0.20)
    - RSI alignment (0/0.15)
    - Structure bias agreement (0/0.20)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal


@dataclass
class _ActiveFVG:
    """Tracks an active Fair Value Gap awaiting fill."""
    bar_idx: int           # bar where FVG was detected
    direction: str         # 'BULL' or 'BEAR'
    gap_high: float        # upper boundary of gap
    gap_low: float         # lower boundary of gap
    volume_confirmed: bool # whether formation candle had above-avg volume
    gap_pct: float         # gap size as fraction of price


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

        Detects 3-candle imbalance patterns, tracks active gaps, and enters
        on price retracement into the gap zone with confirmation candle.
        """
        min_gap = params.get('min_gap_pct', 0.001)
        max_age = int(params.get('max_fvg_age', 50))
        fill_min = params.get('fill_entry_min', 0.30)
        fill_max = params.get('fill_entry_max', 0.70)
        atr_buf = params.get('atr_sl_buffer', 0.5)
        rr_target = params.get('rr_target', 2.0)

        s = scan_start_idx
        e = min(scan_end_idx, len(df))
        if e <= s + 3:
            return []

        # Numpy arrays
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

        # Rolling 20-bar volume mean
        vol_series = pd.Series(volumes)
        vol_mean_20 = vol_series.rolling(20, min_periods=10).mean().values

        # ── Vectorized FVG detection ──────────────────────────────
        # Bullish FVG: candle[i-2].High < candle[i].Low (gap up)
        # Bearish FVG: candle[i-2].Low > candle[i].High (gap down)
        n = len(df)
        bull_fvg_mask = np.zeros(n, dtype=bool)
        bear_fvg_mask = np.zeros(n, dtype=bool)

        if n >= 3:
            # candle[i-2].High < candle[i].Low → bullish FVG at bar i
            bull_fvg_mask[2:] = highs[:-2] < lows[2:]
            # candle[i-2].Low > candle[i].High → bearish FVG at bar i
            bear_fvg_mask[2:] = lows[:-2] > highs[2:]

        # ── Track active FVGs and generate fill signals ───────────
        active_fvgs: List[_ActiveFVG] = []
        signals: List[Signal] = []
        min_cooldown = 4
        last_sig_idx = -min_cooldown

        for i in range(max(s, 2), e):
            atr_val = atrs[i]
            valid_atr = (atr_val > 0) and not np.isnan(atr_val)

            # Register new FVGs detected at this bar
            if bull_fvg_mask[i]:
                gap_high = lows[i]       # top of gap = candle 3 low
                gap_low = highs[i - 2]   # bottom of gap = candle 1 high
                gap_size = gap_high - gap_low
                mid_p = closes[i - 1]    # middle candle price
                if mid_p > 0 and gap_size > 0:
                    gp = gap_size / mid_p
                    if gp >= min_gap:
                        vm = vol_mean_20[i - 1]
                        vol_conf = (
                            not np.isnan(vm) and vm > 0
                            and volumes[i - 1] > vm * 1.2
                        )
                        active_fvgs.append(_ActiveFVG(
                            bar_idx=i, direction='BULL',
                            gap_high=gap_high, gap_low=gap_low,
                            volume_confirmed=vol_conf, gap_pct=gp,
                        ))

            if bear_fvg_mask[i]:
                gap_high = lows[i - 2]   # top of gap = candle 1 low
                gap_low = highs[i]       # bottom of gap = candle 3 high
                gap_size = gap_high - gap_low
                mid_p = closes[i - 1]
                if mid_p > 0 and gap_size > 0:
                    gp = gap_size / mid_p
                    if gp >= min_gap:
                        vm = vol_mean_20[i - 1]
                        vol_conf = (
                            not np.isnan(vm) and vm > 0
                            and volumes[i - 1] > vm * 1.2
                        )
                        active_fvgs.append(_ActiveFVG(
                            bar_idx=i, direction='BEAR',
                            gap_high=gap_high, gap_low=gap_low,
                            volume_confirmed=vol_conf, gap_pct=gp,
                        ))

            # Expire old FVGs and invalidated ones
            surviving: List[_ActiveFVG] = []
            for fvg in active_fvgs:
                age = i - fvg.bar_idx
                if age > max_age:
                    continue
                # Invalidation: price closes through the gap entirely
                if fvg.direction == 'BULL' and closes[i] < fvg.gap_low:
                    continue
                if fvg.direction == 'BEAR' and closes[i] > fvg.gap_high:
                    continue
                surviving.append(fvg)
            active_fvgs = surviving

            # Check for fill entries (only on scan range bars)
            if i < s or not valid_atr:
                continue
            if i - last_sig_idx < min_cooldown:
                continue

            best_signal: Optional[Signal] = None
            best_conf = -1.0

            for fvg in active_fvgs:
                if i - fvg.bar_idx < 2:
                    continue  # skip same-bar and next-bar

                gap_range = fvg.gap_high - fvg.gap_low
                if gap_range <= 0:
                    continue

                # Check price entered the fill zone
                fill_lo = fvg.gap_low + gap_range * fill_min
                fill_hi = fvg.gap_low + gap_range * fill_max

                if fvg.direction == 'BULL':
                    # Price retraces down into gap
                    price_in_zone = lows[i] <= fill_hi and closes[i] >= fill_lo
                    is_confirmation = closes[i] > opens[i]  # bullish close
                    if not (price_in_zone and is_confirmation):
                        continue
                    direction = 'LONG'
                else:
                    # Price retraces up into gap
                    price_in_zone = highs[i] >= (fvg.gap_high - gap_range * fill_max) and closes[i] <= (fvg.gap_high - gap_range * fill_min)
                    is_confirmation = closes[i] < opens[i]  # bearish close
                    if not (price_in_zone and is_confirmation):
                        continue
                    direction = 'SHORT'

                # ── Confidence scoring ─────────────────────────────
                confidence = 0.0

                # 1. Gap size (0 to 0.25) — larger gaps = stronger imbalance
                confidence += min(fvg.gap_pct / 0.005, 1.0) * 0.25

                # 2. Volume at gap creation (+0.20)
                if fvg.volume_confirmed:
                    confidence += 0.20

                # 3. EMA bias (+0.20)
                if has_ema:
                    if direction == 'LONG' and ema50[i] > ema200[i]:
                        confidence += 0.20
                    elif direction == 'SHORT' and ema50[i] < ema200[i]:
                        confidence += 0.20

                # 4. RSI alignment (+0.15)
                if has_rsi:
                    r = rsi[i]
                    if direction == 'LONG' and 30 <= r <= 55:
                        confidence += 0.15
                    elif direction == 'SHORT' and 45 <= r <= 70:
                        confidence += 0.15

                # 5. Structure bias (+0.20)
                if has_structure:
                    sb = struct_bias[i]
                    if direction == 'LONG' and sb > 0:
                        confidence += 0.20
                    elif direction == 'SHORT' and sb < 0:
                        confidence += 0.20

                # Max: 0.25 + 0.20 + 0.20 + 0.15 + 0.20 = 1.0

                # ── Entry / SL / TP ───────────────────────────────
                entry = closes[i]

                if direction == 'LONG':
                    stop_loss = fvg.gap_low - atr_val * atr_buf
                    if stop_loss >= entry:
                        continue
                    risk = entry - stop_loss
                    tp1 = entry + risk * rr_target
                    tp2 = entry + risk * rr_target * 1.5
                else:
                    stop_loss = fvg.gap_high + atr_val * atr_buf
                    if stop_loss <= entry:
                        continue
                    risk = stop_loss - entry
                    tp1 = entry - risk * rr_target
                    tp2 = entry - risk * rr_target * 1.5

                if risk <= 0:
                    continue

                if confidence > best_conf:
                    best_conf = confidence
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

                    best_signal = Signal(
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
                    )

            if best_signal is not None:
                signals.append(best_signal)
                last_sig_idx = i

        return signals
