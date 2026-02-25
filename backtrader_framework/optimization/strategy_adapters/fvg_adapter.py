"""
FVG (Fair Value Gap) adapter for WFO engine — v4.

Pure pandas/numpy signal generation, no backtrader dependency.
Optimized: vectorized FVG detection with numpy shifted arrays,
incremental gap tracking with deque for O(active_gaps) per bar.

v4 improvements (over v3):
    - Displacement gate REMOVED at entry time. The entry-time displacement
      check was conceptually contradictory: FVG entry is a retracement INTO
      the gap (price moving AGAINST trade direction), while displacement
      required same-direction 5-bar momentum. Displacement strength is
      retained as a soft confidence factor only.
    - Sweep requirement CONVERTED from hard gate to soft confidence boost
      (+0.15). Sweeps are already embedded in the FVG concept (impulsive
      move creates gap). Binary gating on a rare event caused 75% of WFO
      windows to produce 0 OOS trades and overfitted to IS sweep patterns.
    - R:R default raised from 0.5 to 1.5, range 0.5-3.0. NQ MFE data
      shows 50% of trades reach 2.0R vs crypto's 9%. WFO optimizes per
      window, but the higher default better anchors the search for equities.

v3 changes (retained):
    - Trailing stop to breakeven after TP1 (TradeSimulator)
    - Session filter: London + NY only (07-21 UTC)
    - Confidence gate at 0.45 default (lowered from 0.55)
    - IV R:R scaling removed; WFO optimizes rr_target directly

v2 changes (retained):
    - Cross-TF detection: 1h FVGs on 15m data (HTF_1h_* columns)
    - IV regime gating: DVOL direction filter (crypto only)
    - Increased minimum gap size (0.2% default)

Signal flow:
    1. Detect 3-candle imbalance patterns (bullish/bearish FVGs)
       - On HTF_1h candles if available (cross-TF), else native
    2. Track active FVGs with age expiry (max_fvg_age bars)
    3. IV regime gate: filter direction by DVOL regime
    4. Session filter: skip bars outside London + NY (07-21 UTC)
    5. Enter when price retraces into gap zone (fill_entry_min to fill_entry_max)
    6. Require confirmation candle (close in expected direction)
    7. SL beyond gap boundary + ATR buffer; TP at R:R target
    8. After TP1 hit, SL trails to breakeven (simulator-level)

Confidence scoring (7 factors, max 1.0):
    - Gap size relative to price (0-0.20)
    - Volume confirmation at gap creation (0/0.15)
    - EMA bias alignment (0/0.15)
    - RSI alignment (0/0.10)
    - Structure bias agreement (0/0.10)
    - Displacement strength (0-0.15)
    - Liquidity sweep present (0/0.15)

WFO Results history:
    v3:  NQ 1h +0.352R/34.6%WR/26, NQ 15m -0.284R/33.3%WR/9
    v2:  BTC 1h -0.297R/31.0%WR/448, ETH 1h -0.119R/33.5%WR/603
    v1:  BTC 1h -0.357R/21.7%WR/3228, ETH 1h -0.305R/21.8%WR/3838
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
            ParamSpec("rr_target",        1.5,    0.5,   3.0,   0.5),
            ParamSpec("min_confidence",   0.45,   0.30,  0.65,  0.05),
        ]

    # ────────────────────────────────────────────────────────────
    #  Cross-TF helpers
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_htf_1h_bars(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, ...]]:
        """Extract unique 1h bars from forward-filled HTF_1h_* columns.

        Returns (htf_highs, htf_lows, htf_closes, bar_end_idx) where
        bar_end_idx[k] is the last native-TF index belonging to 1h bar k.
        Returns None if HTF columns are missing.
        """
        needed = ['HTF_1h_High', 'HTF_1h_Low', 'HTF_1h_Close', 'HTF_1h_Open']
        if not all(c in df.columns for c in needed):
            return None

        htf_close = df['HTF_1h_Close'].values
        htf_high = df['HTF_1h_High'].values
        htf_low = df['HTF_1h_Low'].values

        # Identify 1h bar boundaries: where HTF_1h_Close changes value
        # (forward-fill means all bars within a 1h candle share the same close)
        n = len(df)
        if n < 4:
            return None

        # Find indices where the 1h bar changes
        change_mask = np.zeros(n, dtype=bool)
        change_mask[0] = True
        change_mask[1:] = (htf_close[1:] != htf_close[:-1]) | (htf_high[1:] != htf_high[:-1])

        bar_start_indices = np.where(change_mask)[0]
        n_bars = len(bar_start_indices)
        if n_bars < 3:
            return None

        # For each 1h bar, grab the OHLC from the first native bar in that group
        htf_highs = htf_high[bar_start_indices]
        htf_lows = htf_low[bar_start_indices]
        htf_closes = htf_close[bar_start_indices]

        # bar_end_idx: last native index of each 1h bar
        bar_end_idx = np.empty(n_bars, dtype=np.int64)
        bar_end_idx[:-1] = bar_start_indices[1:] - 1
        bar_end_idx[-1] = n - 1

        return htf_highs, htf_lows, htf_closes, bar_start_indices, bar_end_idx

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

        Returns list of (activation_native_idx, direction, gap_high, gap_low,
                         vol_conf, gap_pct).
        """
        n_bars = len(htf_highs)
        fvgs = []

        for k in range(2, n_bars):
            mid_p = htf_closes[k - 1]
            if mid_p <= 0:
                continue

            # Bullish FVG: candle[k-2].high < candle[k].low
            if htf_highs[k - 2] < htf_lows[k]:
                gap_high = htf_lows[k]
                gap_low = htf_highs[k - 2]
                gap_size = gap_high - gap_low
                if gap_size > 0:
                    gp = gap_size / mid_p
                    if gp >= min_gap:
                        # Activation: after 3rd 1h candle closes
                        act_idx = int(bar_end_idx[k])
                        # Volume check at mid candle
                        mid_native = int(bar_start_indices[k - 1])
                        vm = vol_mean_20[mid_native] if mid_native < len(vol_mean_20) else np.nan
                        vol_conf = (
                            not np.isnan(vm) and vm > 0
                            and volumes[mid_native] > vm * 1.2
                        )
                        fvgs.append((act_idx, _DIR_BULL, gap_high, gap_low, vol_conf, gp))

            # Bearish FVG: candle[k-2].low > candle[k].high
            if htf_lows[k - 2] > htf_highs[k]:
                gap_high = htf_lows[k - 2]
                gap_low = htf_highs[k]
                gap_size = gap_high - gap_low
                if gap_size > 0:
                    gp = gap_size / mid_p
                    if gp >= min_gap:
                        act_idx = int(bar_end_idx[k])
                        mid_native = int(bar_start_indices[k - 1])
                        vm = vol_mean_20[mid_native] if mid_native < len(vol_mean_20) else np.nan
                        vol_conf = (
                            not np.isnan(vm) and vm > 0
                            and volumes[mid_native] > vm * 1.2
                        )
                        fvgs.append((act_idx, _DIR_BEAR, gap_high, gap_low, vol_conf, gp))

        return fvgs

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

        Detects FVGs on 1h candles (cross-TF) when HTF_1h columns exist,
        otherwise on native timeframe. Applies IV regime gating,
        displacement filter, and adaptive R:R.
        """
        min_gap = params.get('min_gap_pct', 0.002)
        max_age = int(params.get('max_fvg_age', 50))
        fill_min = params.get('fill_entry_min', 0.30)
        fill_max = params.get('fill_entry_max', 0.70)
        atr_buf = params.get('atr_sl_buffer', 0.5)
        rr_target = params.get('rr_target', 1.5)
        min_conf = params.get('min_confidence', 0.45)

        s = scan_start_idx
        e = min(scan_end_idx, len(df))
        if e <= s + 3:
            return []

        # ── Native OHLCV arrays ─────────────────────────────────
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        atrs = df['ATR'].values
        volumes = df['Volume'].values
        n = len(df)

        has_rsi = 'RSI' in df.columns
        rsi = df['RSI'].values if has_rsi else np.full(n, 50.0)

        has_ema = 'EMA50' in df.columns and 'EMA200' in df.columns
        ema50 = df['EMA50'].values if has_ema else None
        ema200 = df['EMA200'].values if has_ema else None

        has_structure = 'StructureBias' in df.columns
        struct_bias = df['StructureBias'].values if has_structure else None

        # ── DVOL for IV regime gating ───────────────────────────
        has_dvol = 'DVOL' in df.columns
        dvol = df['DVOL'].values if has_dvol else None

        # Rolling 20-bar volume mean
        vol_series = pd.Series(volumes)
        vol_mean_20 = vol_series.rolling(20, min_periods=10).mean().values

        # ── FVG Detection: cross-TF or native ──────────────────
        htf_result = self._build_htf_1h_bars(df)
        use_cross_tf = htf_result is not None

        active_fvgs: deque = deque()

        if use_cross_tf:
            # Pre-compute all 1h FVGs and seed the deque
            htf_highs, htf_lows, htf_closes, bar_starts, bar_ends = htf_result
            htf_fvg_list = self._detect_htf_fvgs(
                htf_highs, htf_lows, htf_closes,
                bar_starts, bar_ends,
                min_gap, vol_mean_20, volumes,
            )
            # Sort by activation index and convert to iterator
            htf_fvg_list.sort(key=lambda x: x[0])
            htf_fvg_iter_idx = 0
        else:
            # Native-TF vectorized FVG detection masks
            bull_fvg_mask = np.zeros(n, dtype=bool)
            bear_fvg_mask = np.zeros(n, dtype=bool)
            if n >= 3:
                bull_fvg_mask[2:] = highs[:-2] < lows[2:]
                bear_fvg_mask[2:] = lows[:-2] > highs[2:]

        # ── Main loop ───────────────────────────────────────────
        signals: List[Signal] = []
        min_cooldown = 4
        last_sig_idx = -min_cooldown

        loop_start = max(2, s - max_age)

        for i in range(loop_start, e):
            close_i = closes[i]

            # ── Register new FVGs ───────────────────────────────
            if use_cross_tf:
                # Activate any 1h FVGs whose 3rd candle closed at or before bar i
                while htf_fvg_iter_idx < len(htf_fvg_list):
                    fvg = htf_fvg_list[htf_fvg_iter_idx]
                    if fvg[0] <= i:
                        active_fvgs.append(fvg)
                        htf_fvg_iter_idx += 1
                    else:
                        break
            else:
                # Native-TF FVG registration
                if bull_fvg_mask[i]:
                    gap_high = lows[i]
                    gap_low = highs[i - 2]
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
                            active_fvgs.append((i, _DIR_BULL, gap_high, gap_low, vol_conf, gp))

                if bear_fvg_mask[i]:
                    gap_high = lows[i - 2]
                    gap_low = highs[i]
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
                            active_fvgs.append((i, _DIR_BEAR, gap_high, gap_low, vol_conf, gp))

            # Pop expired from front (deque is sorted by activation index)
            while active_fvgs and (i - active_fvgs[0][0]) > max_age:
                active_fvgs.popleft()

            # Skip fill checks for pre-scan bars
            if i < s:
                continue
            if i - last_sig_idx < min_cooldown:
                continue

            atr_val = atrs[i]
            if not (atr_val > 0) or np.isnan(atr_val):
                continue

            # ── IV regime determination ─────────────────────────
            iv_regime = None  # None = no data, 'LOW', 'MED', 'HIGH'
            rr_scale = 1.0    # No R:R scaling; WFO optimizes rr_target
            if has_dvol:
                dv = dvol[i]
                if not np.isnan(dv):
                    if dv < 45:
                        iv_regime = 'LOW'
                    elif dv < 65:
                        iv_regime = 'MED'
                        # MED IV is a dead zone — skip all signals
                        continue
                    else:
                        iv_regime = 'HIGH'

            # ── Displacement (soft score only, no hard gate) ─────
            disp_lookback = 5
            if i >= disp_lookback:
                ref_close = closes[i - disp_lookback]
                if ref_close > 0:
                    price_change = (close_i - ref_close) / ref_close
                else:
                    price_change = 0.0
            else:
                price_change = 0.0

            # ── Session filter: London + NY only (07-21 UTC) ─────
            bar_time = df.index[i]
            if hasattr(bar_time, 'hour'):
                hour = bar_time.hour
                if not (7 <= hour < 21):
                    continue

            # ── Liquidity sweep detection ─────────────────────
            sweep_lookback = 20
            bull_sweep = True   # Default true if insufficient data
            bear_sweep = True
            if i >= sweep_lookback + 5:
                struct_start = i - sweep_lookback - 5
                struct_end = i - 5
                struct_low = np.min(lows[struct_start:struct_end])
                struct_high = np.max(highs[struct_start:struct_end])
                bull_sweep = bool(np.any(lows[struct_end:i + 1] < struct_low))
                bear_sweep = bool(np.any(highs[struct_end:i + 1] > struct_high))

            # ── Check each active FVG for fill entry ────────────
            best_signal: Optional[Signal] = None
            best_conf = -1.0
            open_i = opens[i]
            high_i = highs[i]
            low_i = lows[i]

            for fvg_tuple in active_fvgs:
                fvg_bar, fvg_dir, fvg_gh, fvg_gl, fvg_vc, fvg_gp = fvg_tuple

                age = i - fvg_bar
                if age < 2:
                    continue

                # Invalidation check
                if fvg_dir == _DIR_BULL and close_i < fvg_gl:
                    continue
                if fvg_dir == _DIR_BEAR and close_i > fvg_gh:
                    continue

                gap_range = fvg_gh - fvg_gl
                if gap_range <= 0:
                    continue

                # Fill zone check + direction
                if fvg_dir == _DIR_BULL:
                    fill_lo = fvg_gl + gap_range * fill_min
                    fill_hi = fvg_gl + gap_range * fill_max
                    if not (low_i <= fill_hi and close_i >= fill_lo and close_i > open_i):
                        continue
                    direction = 'LONG'
                else:
                    bear_fill_lo = fvg_gh - gap_range * fill_max
                    bear_fill_hi = fvg_gh - gap_range * fill_min
                    if not (high_i >= bear_fill_lo and close_i <= bear_fill_hi and close_i < open_i):
                        continue
                    direction = 'SHORT'

                # ── IV regime direction gate ──────────────────
                if iv_regime == 'LOW' and direction == 'SHORT':
                    continue
                if iv_regime == 'HIGH' and direction == 'LONG':
                    continue

                # ── Confidence scoring (7 factors, max 1.0) ──
                # 1. Gap size relative to price (0-0.20)
                confidence = min(fvg_gp / 0.005, 1.0) * 0.20

                # 2. Volume confirmation (0/0.15)
                if fvg_vc:
                    confidence += 0.15

                # 3. EMA bias alignment (0/0.15)
                if has_ema:
                    if direction == 'LONG' and ema50[i] > ema200[i]:
                        confidence += 0.15
                    elif direction == 'SHORT' and ema50[i] < ema200[i]:
                        confidence += 0.15

                # 4. RSI alignment (0/0.10)
                if has_rsi:
                    r = rsi[i]
                    if direction == 'LONG' and 30 <= r <= 55:
                        confidence += 0.10
                    elif direction == 'SHORT' and 45 <= r <= 70:
                        confidence += 0.10

                # 5. Structure bias (0/0.10)
                if has_structure:
                    sb = struct_bias[i]
                    if direction == 'LONG' and sb > 0:
                        confidence += 0.10
                    elif direction == 'SHORT' and sb < 0:
                        confidence += 0.10

                # 6. Displacement strength (0-0.15) — soft score, no gate
                abs_disp = abs(price_change)
                disp_score = min(abs_disp / 0.015, 1.0) * 0.15
                confidence += disp_score

                # 7. Liquidity sweep present (0/0.15) — soft boost, no gate
                if direction == 'LONG' and bull_sweep:
                    confidence += 0.15
                elif direction == 'SHORT' and bear_sweep:
                    confidence += 0.15

                if confidence < min_conf:
                    continue

                # ── Entry / SL / TP (regime-adaptive R:R) ────
                entry = close_i
                eff_rr = rr_target * rr_scale

                if direction == 'LONG':
                    stop_loss = fvg_gl - atr_val * atr_buf
                    if stop_loss >= entry:
                        continue
                    risk = entry - stop_loss
                    tp1 = entry + risk * eff_rr
                    tp2 = entry + risk * eff_rr * 1.5
                else:
                    stop_loss = fvg_gh + atr_val * atr_buf
                    if stop_loss <= entry:
                        continue
                    risk = stop_loss - entry
                    tp1 = entry - risk * eff_rr
                    tp2 = entry - risk * eff_rr * 1.5

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
