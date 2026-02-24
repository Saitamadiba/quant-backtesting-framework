"""
Liquidity Raid adapter for WFO engine.

Pure pandas/numpy signal generation, no backtrader dependency.
Optimized: vectorized sweep detection with numpy boolean masks.

Selectivity architecture:
    Hard gates (reject if failed):
        - Killzone (London 02-08 ET, NY 08-16 ET)
        - Candle body >= min_body_pct
        - Sweep detection (price crosses session level + closes back)
        - IV-Adaptive min sweep depth (DVOL-based)

    Soft scoring (confidence boosters, not gates):
        - EMA50/200 trend bias (baseline directional filter)
        - Price Structure Bias (HH/HL/LH/LL swing analysis)
        - MTF 4H Alignment (higher-timeframe trend agreement)
        - Volatility-Adaptive SL (ATR percentile scaling)

    Minimum confidence threshold rejects low-quality setups.

Session definitions (Eastern Time, approximated as UTC-5):
    Asia:      19:00-23:59 ET  ->  establishes liquidity pool
    London KZ: 02:00-08:00 ET  ->  trade Asia level sweeps
    NY KZ:     08:00-16:00 ET  ->  trade Asia + London level sweeps
"""

from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal


class LiquidityRaidAdapter(StrategyAdapter):

    @property
    def name(self) -> str:
        return "LiquidityRaid"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m", "1h"]

    def get_param_space(self) -> List[ParamSpec]:
        """Parameter space for liquidity sweep detection.

        R:R ranges calibrated from MFE analysis (WFO 2,119 OOS trades).
        min_confidence threshold controls how many soft-scoring layers
        must agree before a signal is accepted.
        """
        return [
            ParamSpec("session_lookback",   12,    6,     24,    6,    'int'),
            ParamSpec("atr_sl_multiplier",  2.5,   1.5,   3.5,   0.5),
            ParamSpec("min_rr",             1.0,   0.5,   1.5,   0.5),
            ParamSpec("max_rr",             1.5,   1.0,   2.0,   0.5),
            ParamSpec("min_body_pct",       0.15,  0.10,  0.25,  0.05),
            ParamSpec("sweep_tolerance",    0.002, 0.001, 0.004, 0.001),
            ParamSpec("min_confidence",     0.35,  0.20,  0.55,  0.05),
        ]

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        """
        Generate Liquidity Raid trade signals over [scan_start_idx, scan_end_idx).

        Hard gates: killzone, body size, sweep detection, DVOL depth.
        Soft scoring: EMA bias, structure bias, HTF alignment, vol-adaptive SL.
        Confidence threshold (optimizable) controls selectivity.
        """
        # ── Extract parameters ──────────────────────────────────────
        lookback = int(params.get('session_lookback', 12))
        atr_mult = params.get('atr_sl_multiplier', 2.5)
        min_rr = params.get('min_rr', 1.0)
        max_rr = params.get('max_rr', 1.5)
        min_body = params.get('min_body_pct', 0.15)
        sweep_tol = params.get('sweep_tolerance', 0.002)
        min_conf = params.get('min_confidence', 0.35)

        if max_rr < min_rr:
            max_rr = min_rr

        # ── Session levels (loops over sessions, not bars) ──────────
        levels = _compute_session_levels(df, lookback)

        # ── Numpy arrays for vectorized detection ───────────────────
        s = scan_start_idx
        e = min(scan_end_idx, len(df))
        if e <= s:
            return []

        sl = slice(s, e)
        scan_len = e - s
        opens  = df['Open'].values[sl]
        highs  = df['High'].values[sl]
        lows   = df['Low'].values[sl]
        closes = df['Close'].values[sl]
        atrs   = df['ATR'].values[sl]

        asia_hi   = levels['asia_high'][sl]
        asia_lo   = levels['asia_low'][sl]
        london_hi = levels['london_high'][sl]
        london_lo = levels['london_low'][sl]
        is_kz     = levels['is_killzone'][sl]

        # ── HARD GATE: EMA trend bias (baseline direction) ──────────
        ema50 = df['EMA50'].values[sl]
        ema200 = df['EMA200'].values[sl]
        is_long_ema = ema50 > ema200
        is_short_ema = ema50 < ema200

        # ── SOFT LAYER 1: Price Structure Bias ──────────────────────
        # +0.25 confidence if structure agrees, 0 if neutral, -0.10 if against
        has_structure = 'StructureBias' in df.columns
        if has_structure:
            structure_bias = df['StructureBias'].values[sl]
            struct_conf_arr = df['StructureConf'].values[sl]
        else:
            structure_bias = np.zeros(scan_len)
            struct_conf_arr = np.zeros(scan_len)

        # ── SOFT LAYER 2: MTF (4H) Alignment ───────────────────────
        # +0.20 confidence if HTF agrees, 0 if not
        has_htf = 'HTF_Bullish' in df.columns
        if has_htf:
            htf_bullish = df['HTF_Bullish'].values[sl]
            htf_bearish = df['HTF_Bearish'].values[sl]
        else:
            htf_bullish = is_long_ema
            htf_bearish = is_short_ema

        # ── HARD GATE: DVOL-based IV-adaptive sweep depth ───────────
        has_dvol = 'DVOL' in df.columns
        if has_dvol:
            dvol = df['DVOL'].values[sl]
            valid_dvol = ~np.isnan(dvol)
            min_depth_arr = np.full(scan_len, 0.25)
            min_depth_arr = np.where(
                valid_dvol & (dvol >= 45) & (dvol < 65), 0.35, min_depth_arr
            )
            rr_scale_arr = np.ones(scan_len)
            rr_scale_arr = np.where(
                valid_dvol & (dvol >= 45) & (dvol < 65), 0.75, rr_scale_arr
            )
        else:
            min_depth_arr = np.full(scan_len, 0.15)
            rr_scale_arr = np.ones(scan_len)

        # ── SOFT LAYER 3: Volatility-Adaptive SL ───────────────────
        has_pctile = 'ATR_Pctile20' in df.columns
        if has_pctile:
            atr_pctile = df['ATR_Pctile20'].values[sl]
            valid_pctile = ~np.isnan(atr_pctile)
            sl_vol_mult = np.where(
                valid_pctile & (atr_pctile >= 0.80), 1.25,
                np.where(valid_pctile & (atr_pctile <= 0.20), 0.80, 1.0)
            )
        else:
            sl_vol_mult = np.ones(scan_len)

        # ── Vectorized candle properties ────────────────────────────
        candle_range = highs - lows
        candle_range_safe = np.maximum(candle_range, 1e-10)
        body_pct = np.abs(closes - opens) / candle_range_safe
        is_bullish = closes > opens
        is_bearish = closes < opens
        valid_atr = (atrs > 0) & ~np.isnan(atrs)

        # Hard-gate base: KZ + EMA bias + candle direction + body + ATR
        long_base = (
            is_kz & is_long_ema & is_bullish
            & (body_pct >= min_body) & valid_atr
        )
        short_base = (
            is_kz & is_short_ema & is_bearish
            & (body_pct >= min_body) & valid_atr
        )

        # Session level validity
        v_asia_lo   = ~np.isnan(asia_lo) & (asia_lo > 0)
        v_london_lo = ~np.isnan(london_lo) & (london_lo > 0)
        v_asia_hi   = ~np.isnan(asia_hi) & (asia_hi > 0)
        v_london_hi = ~np.isnan(london_hi) & (london_hi > 0)

        # ── Vectorized sweep detection (hard gate) ──────────────────
        asia_lo_sweep = (
            long_base & v_asia_lo
            & (lows <= asia_lo * (1 - sweep_tol))
            & (closes > asia_lo)
        )
        london_lo_sweep = (
            long_base & v_london_lo
            & (lows <= london_lo * (1 - sweep_tol))
            & (closes > london_lo)
        )
        asia_hi_sweep = (
            short_base & v_asia_hi
            & (highs >= asia_hi * (1 + sweep_tol))
            & (closes < asia_hi)
        )
        london_hi_sweep = (
            short_base & v_london_hi
            & (highs >= london_hi * (1 + sweep_tol))
            & (closes < london_hi)
        )

        long_sweep = asia_lo_sweep | london_lo_sweep
        short_sweep = asia_hi_sweep | london_hi_sweep
        any_sweep = long_sweep | short_sweep

        sweep_rel_indices = np.where(any_sweep)[0]
        if len(sweep_rel_indices) == 0:
            return []

        # ── Build Signal objects with cooldown + soft scoring ───────
        signals: List[Signal] = []
        min_cooldown = 4
        last_sig_rel = -min_cooldown

        for rel_i in sweep_rel_indices:
            if rel_i - last_sig_rel < min_cooldown:
                continue

            abs_i = s + rel_i
            atr_val = atrs[rel_i]
            cr = candle_range[rel_i]

            # Direction and sweep depth
            if long_sweep[rel_i]:
                direction = 'LONG'
                level = asia_lo[rel_i] if asia_lo_sweep[rel_i] else london_lo[rel_i]
                depth = level - lows[rel_i]
            else:
                direction = 'SHORT'
                level = asia_hi[rel_i] if asia_hi_sweep[rel_i] else london_hi[rel_i]
                depth = highs[rel_i] - level

            # HARD GATE: IV-adaptive minimum sweep depth
            depth_atr = depth / atr_val
            if depth_atr < min_depth_arr[rel_i]:
                continue

            # ── Composite confidence from soft layers ───────────────
            # Base: sweep depth component (0.0 to 0.40)
            depth_c = min(depth_atr / 1.0, 1.0) * 0.40

            # SOFT LAYER 1: Structure bias (+0.25 agree, +0.05 neutral, 0 against)
            sb = structure_bias[rel_i]
            if direction == 'LONG':
                struct_score = 0.25 if sb > 0 else (0.05 if sb == 0 else 0.0)
            else:
                struct_score = 0.25 if sb < 0 else (0.05 if sb == 0 else 0.0)

            # SOFT LAYER 2: HTF alignment (+0.20 if agrees, 0 if not)
            if direction == 'LONG':
                htf_score = 0.20 if htf_bullish[rel_i] else 0.0
            else:
                htf_score = 0.20 if htf_bearish[rel_i] else 0.0

            # SOFT LAYER 3: Structure confidence bonus (0 to 0.15)
            struct_conf_score = float(struct_conf_arr[rel_i]) * 0.15

            confidence = depth_c + struct_score + htf_score + struct_conf_score
            # Max possible: 0.40 + 0.25 + 0.20 + 0.15 = 1.0

            # Soft gate: reject below minimum confidence
            if confidence < min_conf:
                continue

            # DVOL-based R:R scaling
            eff_min_rr = min_rr * rr_scale_arr[rel_i]
            eff_max_rr = max_rr * rr_scale_arr[rel_i]

            # Volatility-adaptive SL distance
            vol_adjusted_mult = atr_mult * sl_vol_mult[rel_i]

            # Entry at 50% pullback of confirmation candle
            if direction == 'LONG':
                entry = lows[rel_i] + cr * 0.5
                stop_loss = entry - atr_val * vol_adjusted_mult
                if stop_loss >= entry:
                    continue
                risk = entry - stop_loss
                tp1 = entry + risk * eff_min_rr
                tp2 = entry + risk * eff_max_rr
            else:
                entry = highs[rel_i] - cr * 0.5
                stop_loss = entry + atr_val * vol_adjusted_mult
                if stop_loss <= entry:
                    continue
                risk = stop_loss - entry
                tp1 = entry - risk * eff_min_rr
                tp2 = entry - risk * eff_max_rr

            signals.append(Signal(
                idx=abs_i,
                time=df.index[abs_i],
                direction=direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk=risk,
                confidence=confidence,
                bias='ALIGNED' if (struct_score > 0 and htf_score > 0) else 'PARTIAL',
                atr=atr_val,
            ))
            last_sig_rel = rel_i

        return signals


# ────────────────────────────────────────────────────────────────────
#  Module-level helpers (no instance state)
# ────────────────────────────────────────────────────────────────────

def _compute_session_levels(
    df: pd.DataFrame, lookback_bars: int,
) -> Dict[str, np.ndarray]:
    """Compute session H/L via boundary detection.

    Loops over ~365 session boundaries per year (fast), not over each bar.
    """
    n = len(df)
    highs_arr = df['High'].values
    lows_arr  = df['Low'].values

    et_hours = (np.asarray(df.index.hour, dtype=np.int32) - 5) % 24

    is_asia   = et_hours >= 19
    is_london = (et_hours >= 2) & (et_hours < 8)
    is_ny     = (et_hours >= 8) & (et_hours < 16)
    is_kz     = is_london | is_ny

    n_sessions = max(1, lookback_bars // 6)

    asia_high, asia_low = _session_hl(highs_arr, lows_arr, is_asia, n, n_sessions)
    london_high, london_low = _session_hl(highs_arr, lows_arr, is_london, n, n_sessions)

    return {
        'et_hours':     et_hours,
        'asia_high':    asia_high,
        'asia_low':     asia_low,
        'london_high':  london_high,
        'london_low':   london_low,
        'is_killzone':  is_kz,
    }


def _session_hl(
    highs: np.ndarray,
    lows: np.ndarray,
    session_mask: np.ndarray,
    n: int,
    n_sessions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rolling session H/L from session boundary transitions."""
    diff = np.diff(session_mask.astype(np.int8), prepend=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    if len(starts) == 0:
        return np.full(n, np.nan), np.full(n, np.nan)

    if len(ends) == 0 or ends[-1] < starts[-1]:
        ends = np.append(ends, n)
    if starts[0] > ends[0]:
        starts = np.insert(starts, 0, 0)

    n_sess = min(len(starts), len(ends))
    if n_sess == 0:
        return np.full(n, np.nan), np.full(n, np.nan)

    sess_highs = np.array([np.max(highs[starts[i]:ends[i]]) for i in range(n_sess)])
    sess_lows  = np.array([np.min(lows[starts[i]:ends[i]])  for i in range(n_sess)])

    high_arr = np.full(n, np.nan)
    low_arr  = np.full(n, np.nan)

    for s_idx in range(n_sess):
        valid_start = ends[s_idx]
        valid_end = starts[s_idx + 1] if s_idx + 1 < n_sess else n

        agg_start = max(0, s_idx - n_sessions + 1)
        level_high = np.max(sess_highs[agg_start:s_idx + 1])
        level_low  = np.min(sess_lows[agg_start:s_idx + 1])

        high_arr[valid_start:valid_end] = level_high
        low_arr[valid_start:valid_end]  = level_low

    return high_arr, low_arr
