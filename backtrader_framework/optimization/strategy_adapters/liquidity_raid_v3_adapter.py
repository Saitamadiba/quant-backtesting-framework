"""
Liquidity Raid adapter for WFO engine — v3 (audit-driven redesign).

Pure pandas/numpy signal generation, no backtrader dependency.

v3 redesign rationale (quant_skills audit of 1,243 OOS trades, 28.5/100 Grade D):
    1. HMM regime analysis found 34% of trades in a favorable regime
       (+1.67R mean, 88.8% WR) vs 66% in adverse regime (-1.25R, 0% WR).
       → NEW: Per-bar regime score hard gate filters adverse-regime entries.

    2. Signal decay analysis showed holding-period gradient:
       1-4 bars = -1.13R (6.4% WR), 100-398 bars = +0.67R (55.1% WR).
       → NEW: TP widened from 0.5/0.75R to 1.0/3.0R; ATR trailing stop.

    3. Confirmation bars filter fake sweeps that cause immediate losses.
       → NEW: Optional 1-2 bar confirmation before entry.

Entry architecture:
    Hard gates (reject if failed):
        - Killzone (London 02-08 ET, NY 08-16 ET)
        - Candle body >= min_body_pct
        - Sweep detection (price crosses session level + closes back)
        - IV-Adaptive min sweep depth (DVOL-based)
        - Volatile regime filter (ATR_Pctile20 > 0.80)
        - Displacement: 5-bar counter-directional price move
        - Regime score >= regime_threshold (NEW)

    Confirmation (NEW):
        - Next N candles must confirm sweep direction (0-2 bars)

    Soft scoring (recalibrated confidence):
        - Sweep depth (0-0.40, primary quality signal)
        - Regime score (0-0.25, HMM-proxy from observable features)
        - Volume surge (0-0.15, institutional activity signal)
        - Counter-trend Structure Bias (0-0.10)
        - Counter-trend HTF Alignment (0-0.10)

    Minimum confidence threshold rejects low-quality setups.

Exit architecture:
    - TP1 at tp1_rr (default 1.0R, optimizable 0.75-1.5R): triggers breakeven
    - TP2 at 3.0R (fixed): lets winners run
    - ATR trailing stop after TP1: trail_atr_mult × ATR from water mark
    - Max holding period from WFOConfig (168×scale bars)
"""

from typing import Dict, List, Any, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ET = ZoneInfo("America/New_York")

from .base_adapter import StrategyAdapter, ParamSpec, Signal


class LiquidityRaidV3Adapter(StrategyAdapter):

    @property
    def name(self) -> str:
        return "LiquidityRaidV3"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m", "1h"]

    def get_param_space(self) -> List[ParamSpec]:
        """Parameter space for V3 liquidity sweep detection.

        Changes from V2:
            - min_rr/max_rr replaced by tp1_rr (tp2 fixed at 3.0R)
            - regime_threshold: minimum regime score to enter (NEW)
            - trail_atr_mult: ATR trailing distance after TP1 (NEW)
            - confirm_bars: confirmation candle delay (NEW)
        """
        return [
            # Retained from V2
            ParamSpec("session_lookback",   12,    6,     18,    6,    'int'),
            ParamSpec("atr_sl_multiplier",  2.5,   1.5,   3.5,   0.5),
            ParamSpec("min_body_pct",       0.15,  0.10,  0.25,  0.05),
            ParamSpec("sweep_tolerance",    0.002, 0.001, 0.003, 0.001),
            ParamSpec("min_confidence",     0.35,  0.15,  0.55,  0.10),
            # NEW: wider TP (defaults reflect WFO convergence: tp1=1.0R)
            ParamSpec("tp1_rr",             1.0,   0.75,  1.5,   0.25),
            # NEW: regime, trailing, confirmation
            ParamSpec("regime_threshold",   0.5,   0.3,   0.7,   0.1),
            ParamSpec("trail_atr_mult",     2.0,   1.5,   3.0,   0.5),
            ParamSpec("confirm_bars",       0,     0,     1,     1,   'int'),  # WFO: confirm=2 never chosen
        ]

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        """
        Generate Liquidity Raid V3 trade signals over [scan_start_idx, scan_end_idx).

        Hard gates: killzone, body size, sweep detection, DVOL depth, regime score.
        Soft scoring: depth, regime, volume, structure, HTF.
        Confirmation: optional N-bar directional confirmation after sweep.
        """
        # ── Extract parameters ──────────────────────────────────────
        lookback = int(params.get('session_lookback', 12))
        atr_mult = params.get('atr_sl_multiplier', 2.5)
        min_body = params.get('min_body_pct', 0.15)
        sweep_tol = params.get('sweep_tolerance', 0.002)
        min_conf = params.get('min_confidence', 0.35)
        tp1_rr = params.get('tp1_rr', 1.0)
        tp2_rr = 3.0  # Fixed — audit shows edge at 100-398 bar holds
        regime_thresh = params.get('regime_threshold', 0.5)
        trail_mult = params.get('trail_atr_mult', 2.0)
        confirm_bars = int(params.get('confirm_bars', 1))

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
        volumes = df['Volume'].values[sl]

        asia_hi   = levels['asia_high'][sl]
        asia_lo   = levels['asia_low'][sl]
        london_hi = levels['london_high'][sl]
        london_lo = levels['london_low'][sl]
        is_kz     = levels['is_killzone'][sl]

        # Full arrays for confirmation bar access
        all_opens  = df['Open'].values
        all_closes = df['Close'].values
        all_atrs   = df['ATR'].values

        # ── EMA trend (soft layer baseline) ────────────────────────
        ema50 = df['EMA50'].values[sl]
        ema200 = df['EMA200'].values[sl]

        # ── Price Structure Bias ───────────────────────────────────
        has_structure = 'StructureBias' in df.columns
        if has_structure:
            structure_bias = df['StructureBias'].values[sl]
        else:
            structure_bias = np.zeros(scan_len)

        # ── HTF Alignment ──────────────────────────────────────────
        has_htf = 'HTF_Bullish' in df.columns
        if has_htf:
            htf_bullish = df['HTF_Bullish'].values[sl]
            htf_bearish = df['HTF_Bearish'].values[sl]
        else:
            htf_bullish = ema50 > ema200
            htf_bearish = ema50 < ema200

        # ── DVOL-based IV-adaptive sweep depth ─────────────────────
        has_dvol = 'DVOL' in df.columns
        if has_dvol:
            dvol = df['DVOL'].values[sl]
            valid_dvol = ~np.isnan(dvol)
            min_depth_arr = np.full(scan_len, 0.25)
            min_depth_arr = np.where(
                valid_dvol & (dvol >= 45) & (dvol < 65), 0.35, min_depth_arr
            )
        else:
            min_depth_arr = np.full(scan_len, 0.15)

        # ── Volatility-Adaptive SL ─────────────────────────────────
        has_pctile = 'ATR_Pctile20' in df.columns
        if has_pctile:
            atr_pctile = df['ATR_Pctile20'].values[sl]
            valid_pctile = ~np.isnan(atr_pctile)
            sl_vol_mult = np.where(
                valid_pctile & (atr_pctile >= 0.80), 1.25,
                np.where(valid_pctile & (atr_pctile <= 0.20), 0.80, 1.0)
            )
            # HARD GATE: volatile regime filter (top 20% ATR = worst regime)
            is_volatile = valid_pctile & (atr_pctile > 0.80)
        else:
            atr_pctile = np.full(scan_len, 0.5)
            sl_vol_mult = np.ones(scan_len)
            is_volatile = np.zeros(scan_len, dtype=bool)

        # ── Vectorized regime score (NEW) ──────────────────────────
        regime_scores = _compute_regime_scores_vectorized(
            atr_pctile=atr_pctile if has_pctile else np.full(scan_len, 0.5),
            adx=df['ADX'].values[sl] if 'ADX' in df.columns else np.full(scan_len, 25.0),
            rel_vol=volumes / np.maximum(
                df['Volume_SMA'].values[sl] if 'Volume_SMA' in df.columns else np.ones(scan_len),
                1e-10,
            ),
            range_pos=_compute_range_position(df['High'].values, df['Low'].values, closes, s, e),
            vol_ratio=atrs / np.maximum(
                pd.Series(df['ATR'].values).rolling(20, min_periods=1).mean().values[sl],
                1e-10,
            ),
        )

        # ── Vectorized candle properties ───────────────────────────
        candle_range = highs - lows
        candle_range_safe = np.maximum(candle_range, 1e-10)
        body_pct = np.abs(closes - opens) / candle_range_safe
        is_bullish = closes > opens
        is_bearish = closes < opens
        valid_atr = (atrs > 0) & ~np.isnan(atrs)

        # Hard-gate base: KZ + candle direction + body + ATR
        long_base = is_kz & is_bullish & (body_pct >= min_body) & valid_atr
        short_base = is_kz & is_bearish & (body_pct >= min_body) & valid_atr

        # Session level validity
        v_asia_lo   = ~np.isnan(asia_lo) & (asia_lo > 0)
        v_london_lo = ~np.isnan(london_lo) & (london_lo > 0)
        v_asia_hi   = ~np.isnan(asia_hi) & (asia_hi > 0)
        v_london_hi = ~np.isnan(london_hi) & (london_hi > 0)

        # ── Vectorized sweep detection (hard gate) ─────────────────
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

        # ── Volume SMA for confidence scoring ──────────────────────
        vol_sma = df['Volume_SMA'].values[sl] if 'Volume_SMA' in df.columns else np.ones(scan_len)

        # ── Build Signal objects ───────────────────────────────────
        signals: List[Signal] = []
        min_cooldown = 2
        last_sig_rel = -min_cooldown

        for rel_i in sweep_rel_indices:
            if rel_i - last_sig_rel < min_cooldown:
                continue

            abs_i = s + rel_i

            # HARD GATE: volatile regime filter
            if is_volatile[rel_i]:
                continue

            # HARD GATE: regime score (NEW)
            rs = regime_scores[rel_i]
            if rs < regime_thresh:
                continue

            atr_val = atrs[rel_i]

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

            # HARD GATE: Displacement check (5-bar directional move)
            disp_lookback = 5
            if abs_i >= disp_lookback:
                ref_close = all_closes[abs_i - disp_lookback]
                if ref_close > 0:
                    price_change = (closes[rel_i] - ref_close) / ref_close
                else:
                    price_change = 0.0
                min_disp = 0.002
                if direction == 'LONG' and price_change > -min_disp:
                    continue
                if direction == 'SHORT' and price_change < min_disp:
                    continue

            # ── Confirmation bars (NEW) ────────────────────────────
            entry_abs_i = abs_i
            if confirm_bars > 0:
                confirmed = True
                for c in range(1, confirm_bars + 1):
                    confirm_abs = abs_i + c
                    if confirm_abs >= len(all_closes):
                        confirmed = False
                        break
                    if direction == 'LONG':
                        confirmed = all_closes[confirm_abs] > all_opens[confirm_abs]
                    else:
                        confirmed = all_closes[confirm_abs] < all_opens[confirm_abs]
                    if not confirmed:
                        break
                if not confirmed:
                    continue
                # Shift entry to confirmation bar (no look-ahead)
                entry_abs_i = abs_i + confirm_bars
                if entry_abs_i >= len(all_closes):
                    continue
                atr_val = all_atrs[entry_abs_i] if entry_abs_i < len(all_atrs) else atr_val

            entry = all_closes[entry_abs_i]

            # ── Confidence scoring (recalibrated) ──────────────────
            # Sweep depth: 0-0.40
            depth_c = min(depth_atr / 1.0, 1.0) * 0.40

            # Regime score: 0-0.25 (NEW)
            regime_c = rs * 0.25

            # Volume surge: 0/0.15 (NEW)
            rel_vol = volumes[rel_i] / max(vol_sma[rel_i], 1e-10)
            volume_c = 0.15 if rel_vol >= 1.5 else (0.07 if rel_vol >= 1.2 else 0.0)

            # Counter-trend structure bias: 0-0.10
            sb = structure_bias[rel_i]
            if direction == 'LONG':
                struct_score = 0.10 if sb < 0 else (0.03 if sb == 0 else 0.0)
            else:
                struct_score = 0.10 if sb > 0 else (0.03 if sb == 0 else 0.0)

            # Counter-trend HTF alignment: 0/0.10
            if direction == 'LONG':
                htf_score = 0.10 if htf_bearish[rel_i] else 0.0
            else:
                htf_score = 0.10 if htf_bullish[rel_i] else 0.0

            confidence = depth_c + regime_c + volume_c + struct_score + htf_score
            # Max possible: 0.40 + 0.25 + 0.15 + 0.10 + 0.10 = 1.0

            # Soft gate: reject below minimum confidence
            if confidence < min_conf:
                continue

            # Volatility-adaptive SL distance
            vol_adjusted_mult = atr_mult * sl_vol_mult[rel_i]

            # ── Entry / SL / TP ────────────────────────────────────
            if direction == 'LONG':
                stop_loss = entry - atr_val * vol_adjusted_mult
                if stop_loss >= entry:
                    continue
                risk = entry - stop_loss
                tp1 = entry + risk * tp1_rr
                tp2 = entry + risk * tp2_rr
            else:
                stop_loss = entry + atr_val * vol_adjusted_mult
                if stop_loss <= entry:
                    continue
                risk = stop_loss - entry
                tp1 = entry - risk * tp1_rr
                tp2 = entry - risk * tp2_rr

            signals.append(Signal(
                idx=entry_abs_i,
                time=df.index[entry_abs_i],
                direction=direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk=risk,
                confidence=confidence,
                bias='COUNTER' if (struct_score > 0.05 or htf_score > 0) else 'PARTIAL',
                atr=atr_val,
                metadata={
                    'trail_atr_mult': trail_mult,
                    'regime_score': float(rs),
                    'sweep_depth_atr': float(depth_atr),
                    'confirm_bars_used': confirm_bars,
                    'relative_volume': float(rel_vol),
                },
            ))
            last_sig_rel = rel_i

        return signals


# ────────────────────────────────────────────────────────────────────
#  Regime score computation
# ────────────────────────────────────────────────────────────────────

def _compute_regime_scores_vectorized(
    atr_pctile: np.ndarray,
    adx: np.ndarray,
    rel_vol: np.ndarray,
    range_pos: np.ndarray,
    vol_ratio: np.ndarray,
) -> np.ndarray:
    """Vectorized regime favorability score for all bars. Returns array in [0, 1].

    Components (weighted sum):
        - Volatility calm (0.30): lower ATR percentile = calmer = better for mean-reversion
        - ADX sweet spot (0.25): peak at ~25 (clear trend to revert against)
        - Volume confirmation (0.20): higher relative volume = institutional activity
        - Range extremity (0.15): price near 50-bar range edges = better mean-reversion setup
        - Volatility stability (0.10): ATR near its mean = predictable environment
    """
    # Volatility calm (w=0.30)
    vol_calm = np.where(np.isnan(atr_pctile), 0.5, np.maximum(0.0, 1.0 - atr_pctile))

    # ADX sweet spot (w=0.25): peak at 25, decays toward 0 at <15 and >65
    adx_safe = np.where(np.isnan(adx), 25.0, adx)
    adx_score = np.where(
        (adx_safe >= 15) & (adx_safe <= 35),
        1.0 - np.abs(adx_safe - 25) / 20,
        np.where(
            adx_safe < 15,
            adx_safe / 15 * 0.5,
            np.maximum(0.0, 1.0 - (adx_safe - 35) / 30),
        ),
    )

    # Volume confirmation (w=0.20)
    vol_score = np.where(np.isnan(rel_vol), 0.5, np.minimum(rel_vol / 2.0, 1.0))

    # Range extremity (w=0.15): 0 at center, 1.0 at range edges
    range_score = np.where(np.isnan(range_pos), 0.5, 2.0 * np.abs(range_pos - 0.5))

    # Volatility stability (w=0.10): 1.0 when ATR/mean=1, 0 when >=2
    stability = np.where(np.isnan(vol_ratio), 0.5, np.maximum(0.0, np.minimum(1.0, 2.0 - vol_ratio)))

    return (
        0.30 * vol_calm
        + 0.25 * adx_score
        + 0.20 * vol_score
        + 0.15 * range_score
        + 0.10 * stability
    )


def _compute_range_position(
    all_highs: np.ndarray,
    all_lows: np.ndarray,
    scan_closes: np.ndarray,
    scan_start: int,
    scan_end: int,
) -> np.ndarray:
    """Compute position within 50-bar range for each bar in scan window."""
    scan_len = scan_end - scan_start
    range_pos = np.full(scan_len, 0.5)
    for i in range(scan_len):
        abs_idx = scan_start + i
        if abs_idx >= 50:
            h50 = np.max(all_highs[abs_idx - 49:abs_idx + 1])
            l50 = np.min(all_lows[abs_idx - 49:abs_idx + 1])
            rng = h50 - l50
            if rng > 0:
                range_pos[i] = (scan_closes[i] - l50) / rng
    return range_pos


# ────────────────────────────────────────────────────────────────────
#  Session level computation (shared with V2)
# ────────────────────────────────────────────────────────────────────

def _utc_to_et_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert a UTC DatetimeIndex to Eastern Time hours (DST-aware)."""
    if index.tz is None:
        utc_index = index.tz_localize('UTC')
    else:
        utc_index = index.tz_convert('UTC')
    et_index = utc_index.tz_convert(_ET)
    return np.asarray(et_index.hour, dtype=np.int32)


def _compute_session_levels(
    df: pd.DataFrame, lookback_bars: int,
) -> Dict[str, np.ndarray]:
    """Compute session H/L via boundary detection.

    Loops over ~365 session boundaries per year (fast), not over each bar.
    """
    n = len(df)
    highs_arr = df['High'].values
    lows_arr  = df['Low'].values

    et_hours = _utc_to_et_hours(df.index)

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
