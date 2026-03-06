"""
FVG (Fair Value Gap) adapter for WFO engine — v5.

Pure pandas/numpy signal generation, no backtrader dependency.
Optimized: vectorized FVG detection with numpy shifted arrays,
incremental gap tracking with deque for O(active_gaps) per bar.

v5 improvements (over v4) — Profitability Study filters, Mar 2026:
    - iFVG (Inverted FVG) detection: tracks mitigated FVGs and generates
      entry signals when price re-touches the inverted zone.
    - DVOL MED=SKIP removed: MED DVOL is now allowed (study: +0.441R
      for retest, +0.477R for iFVG). Was a dead zone that killed trades.
    - DVOL gates are now strategy-dependent: iFVG reverses direction
      gates (HIGH DVOL is best for iFVG at +0.600R).
    - HMM regime integration: 2-state GaussianHMM (calm/volatile) fit
      on warmup data, forward-filtered causally. Intraday prefers
      volatile (+0.462R), daily prefers calm (+0.714R).
    - Realized Vol percentile boost for iFVG (high RV + volatile HMM
      is the strongest 2-factor filter: +0.642R, n=285).
    - R:R defaults updated: retest 3.0R, iFVG 2.0R (was 1.5 flat).
    - SL buffer default lowered from 0.5 to 0.3 ATR (study finding).
    - All signals carry metadata['signal_type'] = 'RETEST' or 'IFVG'.

v4 changes (retained):
    - Displacement gate REMOVED at entry time (soft confidence only).
    - Sweep requirement CONVERTED to soft confidence boost (+0.15).
    - R:R range 0.5-4.0 (was 0.5-3.0); WFO optimizes per window.

v3 changes (retained):
    - Trailing stop to breakeven after TP1 (TradeSimulator)
    - Session filter: London + NY only (07-21 UTC)
    - Confidence gate at 0.45 default (lowered from 0.55)

v2 changes (retained):
    - Cross-TF detection: 1h FVGs on 15m data (HTF_1h_* columns)
    - IV regime gating: DVOL direction filter (crypto only)
    - Increased minimum gap size (0.2% default)

Signal flow:
    1. Detect 3-candle imbalance patterns (bullish/bearish FVGs)
       - On HTF_1h candles if available (cross-TF), else native
    2. Track active FVGs with age expiry (max_fvg_age bars)
    3. Track mitigated FVGs (price closed through zone) as iFVGs
    4. IV regime gate: strategy-dependent direction filter by DVOL
    5. Session filter: skip bars outside London + NY (07-21 UTC)
    6. RETEST: enter when price retraces into gap zone
    7. iFVG: enter when price re-touches mitigated (inverted) zone
    8. HMM + RV confidence boosts for both signal types
    9. SL beyond gap boundary + ATR buffer; TP at R:R target
   10. After TP1 hit, SL trails to breakeven (simulator-level)

Confidence scoring (9 factors):
    - Gap size relative to price (0-0.20)
    - Volume confirmation at gap creation (0/0.15)
    - EMA bias alignment (0/0.15)
    - RSI alignment (0/0.10)
    - Structure bias agreement (0/0.10)
    - Displacement strength (0-0.15)
    - Liquidity sweep present (0/0.15)
    - HMM regime alignment (0/hmm_conf_boost)
    - Realized vol alignment (0/0.10) — iFVG only

WFO Results history:
    v4:  BTC 1h -0.297R/31.0%WR/448, ETH 1h -0.119R/33.5%WR/603
    v3:  NQ 1h +0.352R/34.6%WR/26, NQ 15m -0.284R/33.3%WR/9
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

    # Study-proven constants (not optimizable — reduces search space)
    _FIXED_ATR_SL_BUFFER = 0.3     # Study: 0.3 ATR beats 0.5 default
    _FIXED_RR_RETEST = 3.0         # Study: retest bounce optimal R:R
    _FIXED_RR_IFVG = 2.0           # Study: iFVG inversion optimal R:R
    _FIXED_HMM_CONF_BOOST = 0.15   # Study: volatile-state boost

    def get_param_space(self) -> List[ParamSpec]:
        """Parameter space for Fair Value Gap detection and entry.

        Study-proven filter values (atr_sl_buffer, rr_target, ifvg_rr_target,
        hmm_conf_boost) are fixed as class constants — not part of the search
        space. This keeps the grid at 5 dims for stable optimization with
        150 random samples.
        """
        return [
            ParamSpec("min_gap_pct",      0.002,  0.001, 0.005, 0.001),
            ParamSpec("max_fvg_age",      50,     20,    80,    20,    'int'),
            ParamSpec("min_confidence",   0.45,   0.30,  0.65,  0.05),
        ]

    # ────────────────────────────────────────────────────────────
    #  HMM helpers (cached per DataFrame to avoid re-fitting per param combo)
    # ────────────────────────────────────────────────────────────

    # Class-level HMM cache: avoids re-fitting when the same df is passed
    # across multiple param combos in a WFO window.
    _hmm_cache_id: int = -1
    _hmm_cache_states: Optional[np.ndarray] = None

    @classmethod
    def _fit_hmm_states(cls, df: pd.DataFrame, warmup_pct: float) -> Optional[np.ndarray]:
        """Fit 2-state GaussianHMM on warmup data and forward-filter the rest.

        Returns array of state indices (0=calm, 1=volatile) or None on failure.
        Uses only causal (forward-filter) inference — no future leakage.
        Results are cached per DataFrame id to avoid refitting 150x per window.
        """
        # Cache hit: same DataFrame object → return cached result
        df_id = id(df)
        if cls._hmm_cache_id == df_id and cls._hmm_cache_states is not None:
            return cls._hmm_cache_states

        if 'LogReturn' not in df.columns or 'RealizedVol20' not in df.columns:
            cls._hmm_cache_id = df_id
            cls._hmm_cache_states = None
            return None
        try:
            from ..hmm_regime import GaussianHMM

            features = df[['LogReturn', 'RealizedVol20']].values
            valid_mask = ~np.isnan(features).any(axis=1)
            n = len(df)

            warmup_end = max(int(n * warmup_pct), 100)
            if warmup_end >= n:
                warmup_end = n // 2
            if valid_mask[:warmup_end].sum() < 50:
                cls._hmm_cache_id = df_id
                cls._hmm_cache_states = None
                return None

            warmup_feat = features[:warmup_end][valid_mask[:warmup_end]]
            mu = np.mean(warmup_feat, axis=0)
            std = np.maximum(np.std(warmup_feat, axis=0), 1e-8)

            hmm = GaussianHMM(n_states=2, max_iter=100, tol=1e-4)
            hmm.fit((warmup_feat - mu) / std)

            # Forward filter full data (standardised with warmup stats)
            X_full = features.copy()
            nan_rows = np.isnan(X_full).any(axis=1)
            X_full[nan_rows] = mu
            probs = hmm.forward_filter((X_full - mu) / std)
            states = np.argmax(probs, axis=1)  # 0=calm, 1=volatile

            cls._hmm_cache_id = df_id
            cls._hmm_cache_states = states
            return states

        except Exception:
            cls._hmm_cache_id = df_id
            cls._hmm_cache_states = None
            return None

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
        otherwise on native timeframe. Produces both RETEST and iFVG signals
        with strategy-dependent IV regime gating and HMM confidence boosts.
        """
        min_gap = params.get('min_gap_pct', 0.002)
        max_age = int(params.get('max_fvg_age', 50))
        min_conf = params.get('min_confidence', 0.45)
        # Fixed study-proven values (not in param search space)
        atr_buf = self._FIXED_ATR_SL_BUFFER
        rr_target = self._FIXED_RR_RETEST
        ifvg_rr = self._FIXED_RR_IFVG
        hmm_boost = self._FIXED_HMM_CONF_BOOST

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

        # ── HMM regime states ──────────────────────────────────
        warmup_pct = params.get('hmm_warmup_pct', 0.30)
        hmm_states = self._fit_hmm_states(df, warmup_pct)
        has_hmm = hmm_states is not None

        # ── Realized vol percentile ─────────────────────────────
        has_rv = 'RV_Percentile' in df.columns
        rv_pctile = df['RV_Percentile'].values if has_rv else None

        # Rolling 20-bar volume mean
        vol_series = pd.Series(volumes)
        vol_mean_20 = vol_series.rolling(20, min_periods=10).mean().values

        # ── FVG Detection: cross-TF or native ──────────────────
        htf_result = self._build_htf_1h_bars(df)
        use_cross_tf = htf_result is not None

        active_fvgs: deque = deque()
        # iFVG tracking: mitigated FVGs with inverted polarity
        # Tuple: (mitigation_bar, inv_direction, gap_high, gap_low, vol_conf, gap_pct)
        mitigated_fvgs: deque = deque()

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

        # First-touch-only: track FVG bars that already generated a signal
        used_fvg_bars: set = set()
        used_ifvg_bars: set = set()

        # Track which FVGs got mitigated this bar (avoid duplicates)
        _mitigated_ids: set = set()

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
            while mitigated_fvgs and (i - mitigated_fvgs[0][0]) > max_age:
                mitigated_fvgs.popleft()

            # ── Check for mitigation → move to iFVG deque ───────
            _mitigated_ids.clear()
            new_active: deque = deque()
            for fvg_tuple in active_fvgs:
                fvg_bar, fvg_dir, fvg_gh, fvg_gl, fvg_vc, fvg_gp = fvg_tuple
                # Bullish FVG mitigated: close below gap_low
                if fvg_dir == _DIR_BULL and close_i < fvg_gl:
                    mitigated_fvgs.append(
                        (i, _DIR_BEAR, fvg_gh, fvg_gl, fvg_vc, fvg_gp)
                    )
                # Bearish FVG mitigated: close above gap_high
                elif fvg_dir == _DIR_BEAR and close_i > fvg_gh:
                    mitigated_fvgs.append(
                        (i, _DIR_BULL, fvg_gh, fvg_gl, fvg_vc, fvg_gp)
                    )
                else:
                    new_active.append(fvg_tuple)
            active_fvgs = new_active

            # Skip fill checks for pre-scan bars
            if i < s:
                continue
            if i - last_sig_idx < min_cooldown:
                continue

            atr_val = atrs[i]
            if not (atr_val > 0) or np.isnan(atr_val):
                continue

            # ── IV regime determination ─────────────────────────
            # MED is no longer a dead zone (profitability study: +0.441R)
            iv_regime = None  # None = no data, 'LOW', 'MED', 'HIGH'
            if has_dvol:
                dv = dvol[i]
                if not np.isnan(dv):
                    if dv < 45:
                        iv_regime = 'LOW'
                    elif dv < 65:
                        iv_regime = 'MED'
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

            # ── Session filter: equity only (NQ = no DVOL data) ──
            # Study showed crypto profits across all hours → no session
            # filter for crypto. NQ/equities use London + NY (07-21 UTC).
            if not has_dvol:
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

            open_i = opens[i]
            high_i = highs[i]
            low_i = lows[i]

            # ── HMM hard gate: only trade during volatile regime ───
            # Study: volatile state dominates profitable combos.
            # Skip calm-state bars entirely (hard gate, not soft boost).
            if has_hmm and hmm_states[i] == 0:  # calm
                continue

            # ── RETEST: Check each active FVG for fill entry ────
            best_signal: Optional[Signal] = None
            best_conf = -1.0
            best_fvg_bar = -1  # track which FVG produced best signal

            for fvg_tuple in active_fvgs:
                fvg_bar, fvg_dir, fvg_gh, fvg_gl, fvg_vc, fvg_gp = fvg_tuple

                age = i - fvg_bar
                if age < 2:
                    continue
                # First-touch-only: skip FVGs that already produced a signal
                if fvg_bar in used_fvg_bars:
                    continue

                gap_range = fvg_gh - fvg_gl
                if gap_range <= 0:
                    continue

                # Zone touch check — matches study methodology
                # Study: any bar whose range touches the gap zone triggers
                if fvg_dir == _DIR_BULL:
                    if not (low_i <= fvg_gh and high_i >= fvg_gl):
                        continue
                    direction = 'LONG'
                else:
                    if not (high_i >= fvg_gl and low_i <= fvg_gh):
                        continue
                    direction = 'SHORT'

                # ── Retest IV regime direction gate ──────────
                if iv_regime == 'LOW' and direction == 'SHORT':
                    continue
                if iv_regime == 'HIGH' and direction == 'LONG':
                    continue

                # ── Confidence scoring (7 base + HMM + RV) ──
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

                # 8. HMM is now a hard gate (above), no soft boost needed

                if confidence < min_conf:
                    continue

                # ── Entry / SL / TP ─────────────────────────
                # Entry at gap midpoint — matches study
                # SL buffer = max(30% gap, 0.3 ATR) — matches study
                entry = (fvg_gh + fvg_gl) / 2.0
                sl_buf = max(gap_range * atr_buf, atr_val * atr_buf)

                if direction == 'LONG':
                    stop_loss = fvg_gl - sl_buf
                    if stop_loss >= entry:
                        continue
                    risk = entry - stop_loss
                    tp1 = entry + risk * rr_target
                    tp2 = entry + risk * rr_target * 1.5
                else:
                    stop_loss = fvg_gh + sl_buf
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

                    hmm_st = int(hmm_states[i]) if has_hmm else -1
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
                        metadata={'signal_type': 'RETEST', 'hmm_state': hmm_st},
                    )
                    best_fvg_bar = fvg_bar

            if best_signal is not None:
                signals.append(best_signal)
                used_fvg_bars.add(best_fvg_bar)  # first-touch-only
                last_sig_idx = i
                continue  # one signal per bar

            # ── iFVG: Check mitigated FVGs for re-touch entry ───
            best_ifvg: Optional[Signal] = None
            best_ifvg_conf = -1.0
            best_ifvg_bar = -1

            for ifvg_tuple in mitigated_fvgs:
                mit_bar, inv_dir, ifvg_gh, ifvg_gl, ifvg_vc, ifvg_gp = ifvg_tuple

                age = i - mit_bar
                if age < 2:
                    continue
                # First-touch-only
                if mit_bar in used_ifvg_bars:
                    continue

                gap_range = ifvg_gh - ifvg_gl
                if gap_range <= 0:
                    continue

                # Re-touch fill zone check (inverted polarity)
                # Re-touch zone check — matches study methodology
                if inv_dir == _DIR_BULL:
                    # Originally bearish, now support → LONG
                    if not (low_i <= ifvg_gh and high_i >= ifvg_gl):
                        continue
                    direction = 'LONG'
                else:
                    # Originally bullish, now resistance → SHORT
                    if not (high_i >= ifvg_gl and low_i <= ifvg_gh):
                        continue
                    direction = 'SHORT'

                # ── iFVG-specific DVOL gates (REVERSED) ──────
                # HIGH DVOL is best for iFVG (+0.600R), allow ALL
                # MED DVOL allows ALL (+0.477R)
                # LOW DVOL: only SHORT (inverse of retest)
                if iv_regime == 'LOW' and direction == 'LONG':
                    continue

                # ── RV hard gate for iFVG: rv>=50th pctile ──
                # Study: hmm=volatile + rv=high → +0.642R, 54.7% WR
                if has_rv and rv_pctile is not None:
                    rv_val = rv_pctile[i]
                    if not np.isnan(rv_val) and rv_val < 0.50:
                        continue

                # ── Confidence scoring (7 base + RV boost) ──
                confidence = min(ifvg_gp / 0.005, 1.0) * 0.20

                if ifvg_vc:
                    confidence += 0.15

                if has_ema:
                    if direction == 'LONG' and ema50[i] > ema200[i]:
                        confidence += 0.15
                    elif direction == 'SHORT' and ema50[i] < ema200[i]:
                        confidence += 0.15

                if has_rsi:
                    r = rsi[i]
                    if direction == 'LONG' and 30 <= r <= 55:
                        confidence += 0.10
                    elif direction == 'SHORT' and 45 <= r <= 70:
                        confidence += 0.10

                if has_structure:
                    sb = struct_bias[i]
                    if direction == 'LONG' and sb > 0:
                        confidence += 0.10
                    elif direction == 'SHORT' and sb < 0:
                        confidence += 0.10

                abs_disp = abs(price_change)
                confidence += min(abs_disp / 0.015, 1.0) * 0.15

                if direction == 'LONG' and bull_sweep:
                    confidence += 0.15
                elif direction == 'SHORT' and bear_sweep:
                    confidence += 0.15

                # 8. HMM is now a hard gate (above), no soft boost needed

                # 9. RV: high realized vol is strongest iFVG filter
                if has_rv and rv_pctile is not None and not np.isnan(rv_pctile[i]):
                    if rv_pctile[i] >= 0.50:
                        confidence += 0.10

                if confidence < min_conf:
                    continue

                # ── Entry / SL / TP (iFVG R:R) ──────────────
                # Entry at gap midpoint — matches study
                # SL buffer = max(30% gap, 0.3 ATR) — matches study
                entry = (ifvg_gh + ifvg_gl) / 2.0
                sl_buf = max(gap_range * atr_buf, atr_val * atr_buf)

                if direction == 'LONG':
                    stop_loss = ifvg_gl - sl_buf
                    if stop_loss >= entry:
                        continue
                    risk = entry - stop_loss
                    tp1 = entry + risk * ifvg_rr
                    tp2 = entry + risk * ifvg_rr * 1.5
                else:
                    stop_loss = ifvg_gh + sl_buf
                    if stop_loss <= entry:
                        continue
                    risk = stop_loss - entry
                    tp1 = entry - risk * ifvg_rr
                    tp2 = entry - risk * ifvg_rr * 1.5

                if risk <= 0:
                    continue

                if confidence > best_ifvg_conf:
                    best_ifvg_conf = confidence
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

                    hmm_st = int(hmm_states[i]) if has_hmm else -1
                    rv_val = float(rv_pctile[i]) if has_rv and not np.isnan(rv_pctile[i]) else -1.0
                    best_ifvg = Signal(
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
                        metadata={
                            'signal_type': 'IFVG',
                            'hmm_state': hmm_st,
                            'rv_pctile': rv_val,
                        },
                    )
                    best_ifvg_bar = mit_bar

            if best_ifvg is not None:
                signals.append(best_ifvg)
                used_ifvg_bars.add(best_ifvg_bar)  # first-touch-only
                last_sig_idx = i

        return signals
