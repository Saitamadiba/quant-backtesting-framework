"""
Liquidity Raid adapter for WFO engine — v3 (live-faithful).

Pure pandas/numpy signal generation, no backtrader dependency.

v3 live-faithful rewrite (shadow backtest alignment):
    - Single-session levels: uses most recent Asia/London session only
      (matches session_manager.py:146-187, no rolling aggregation)
    - Multi-bar sweep state machine: sweep detection and confirmation
      can occur on different bars (matches session_manager.py:256-366)
    - Structural bias gate: one direction per bar from StructureBias,
      falling back to EMA50 vs EMA200 (matches strategy.py:371-382)
    - London High shorts disabled (0% historical WR, config_base.py:226)
    - Min sweep depth 0.30 ATR without DVOL (matches config_base.py:244)
    - London sweeps only in NY killzone (matches session_manager.py:306)
    - Price reclaim: SWEEP_DETECTED → WAITING if close returns beyond
      level (matches session_manager.py:192-229)

Selectivity architecture:
    Hard gates (reject if failed):
        - Killzone (London 02-08 ET, NY 08-16 ET)
        - Structural bias (LONG or SHORT per bar; NONE = skip)
        - Sweep state machine (level break → directional confirmation)
        - Candle body >= min_body_pct (on confirmation bar)
        - IV-Adaptive min sweep depth (DVOL-based, 0.30 ATR fallback)
        - London High shorts blocked

    Soft scoring (mean-reversion confidence):
        - Sweep depth (0-0.50, primary quality signal)
        - Counter-trend Structure Bias (0-0.20)
        - Counter-trend HTF Alignment (0-0.15)
        - Structure confidence bonus (0-0.15)

    Confidence score is informational (no hard reject threshold).

Session definitions (Eastern Time, via zoneinfo):
    Asia:      19:00-23:59 ET  ->  establishes liquidity pool
    London KZ: 02:00-08:00 ET  ->  trade Asia level sweeps
    NY KZ:     08:00-16:00 ET  ->  trade Asia + London level sweeps
"""

from datetime import date as date_type, timedelta
from typing import Dict, List, Any, Tuple, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ET = ZoneInfo("America/New_York")

from .base_adapter import StrategyAdapter, ParamSpec, Signal

# Sweep state machine constants
_WAITING = 0
_SWEEP_DETECTED = 1
_TRADED = 2


class LiquidityRaidAdapter(StrategyAdapter):

    @property
    def name(self) -> str:
        return "LiquidityRaid"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m", "1h"]

    def get_param_space(self) -> List[ParamSpec]:
        """Parameter space for liquidity sweep detection.

        R:R calibrated from MFE analysis: 79.6% of trades reach 0.5R,
        68.7% reach 0.75R, only 58.7% reach 1.0R → 0.5R default target.
        """
        return [
            ParamSpec("session_lookback",   12,    6,     18,    6,    'int'),
            ParamSpec("atr_sl_multiplier",  2.5,   1.5,   3.5,   0.5),
            ParamSpec("min_rr",             0.5,   0.3,   1.0,   0.5),
            ParamSpec("max_rr",             0.75,  0.5,   1.5,   0.5),
            ParamSpec("min_body_pct",       0.15,  0.10,  0.25,  0.05),
            ParamSpec("sweep_tolerance",    0.002, 0.001, 0.003, 0.001),
            ParamSpec("min_confidence",     0.25,  0.15,  0.45,  0.05),
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

        Uses a sequential state machine matching the live bot's sweep detection:
        - Single-session levels (no rolling aggregation)
        - Multi-bar sweep: detection on bar N, confirmation on bar N or later
        - Structural bias gate: one direction per bar
        - Price reclaim invalidates unconsumed sweeps
        """
        # ── Extract parameters ──────────────────────────────────────
        atr_mult = params.get('atr_sl_multiplier', 2.5)
        min_rr = params.get('min_rr', 1.0)
        max_rr = params.get('max_rr', 1.5)
        min_body = params.get('min_body_pct', 0.15)
        min_conf = params.get('min_confidence', 0.35)

        if max_rr < min_rr:
            max_rr = min_rr

        # ── Slice to scan range ───────────────────────────────────
        s = scan_start_idx
        e = min(scan_end_idx, len(df))
        if e <= s:
            return []

        sl = slice(s, e)
        scan_len = e - s

        # ── Pre-compute numpy arrays ─────────────────────────────
        opens  = df['Open'].values[sl]
        highs  = df['High'].values[sl]
        lows   = df['Low'].values[sl]
        closes = df['Close'].values[sl]
        atrs   = df['ATR'].values[sl]

        ema50 = df['EMA50'].values[sl]
        ema200 = df['EMA200'].values[sl]

        has_structure = 'StructureBias' in df.columns
        if has_structure:
            structure_bias = df['StructureBias'].values[sl]
            struct_conf_arr = df['StructureConf'].values[sl]
        else:
            structure_bias = np.zeros(scan_len)
            struct_conf_arr = np.zeros(scan_len)

        has_htf = 'HTF_Bullish' in df.columns
        if has_htf:
            htf_bullish = df['HTF_Bullish'].values[sl]
            htf_bearish = df['HTF_Bearish'].values[sl]
        else:
            htf_bullish = ema50 > ema200
            htf_bearish = ema50 < ema200

        # ── DVOL-based IV-adaptive sweep depth ────────────────────
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
            min_depth_arr = np.full(scan_len, 0.30)   # Fix 5: was 0.15
            rr_scale_arr = np.ones(scan_len)

        # ── Volatility-Adaptive SL ────────────────────────────────
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

        # ── Candle properties ─────────────────────────────────────
        candle_range = highs - lows
        candle_range_safe = np.maximum(candle_range, 1e-10)
        body_pct = np.abs(closes - opens) / candle_range_safe
        is_bullish = closes > opens
        is_bearish = closes < opens
        valid_atr = (atrs > 0) & ~np.isnan(atrs)

        # ── Pre-compute ET hours and session info ─────────────────
        et_info = _compute_et_info(df.index)
        et_hours_full = et_info['et_hours']
        et_dates_full = et_info['et_dates']

        et_hours_scan = et_hours_full[sl]
        et_dates_scan = et_dates_full[s:e]

        is_london = (et_hours_scan >= 2) & (et_hours_scan < 8)
        is_ny     = (et_hours_scan >= 8) & (et_hours_scan < 16)
        is_kz     = is_london | is_ny

        # ── Pre-compute single-session levels lookup ──────────────
        all_highs = df['High'].values
        all_lows = df['Low'].values
        asia_sess, london_sess = _build_session_lookups(
            et_hours_full, et_dates_full, all_highs, all_lows
        )

        # ── State machine: sequential scan ────────────────────────
        # States per level (reset each session date)
        asia_lo_state = _WAITING
        asia_hi_state = _WAITING
        london_lo_state = _WAITING
        london_hi_state = _WAITING

        asia_lo_sweep_price = 0.0
        asia_hi_sweep_price = 0.0
        london_lo_sweep_price = 0.0
        london_hi_sweep_price = 0.0

        # Current session levels (scalars)
        cur_asia_hi = np.nan
        cur_asia_lo = np.nan
        cur_london_hi = np.nan
        cur_london_lo = np.nan

        # Track session dates for state reset
        last_asia_date = None
        last_london_date = None

        signals: List[Signal] = []
        min_cooldown = 4
        last_sig_rel = -min_cooldown

        for rel_i in range(scan_len):
            # Skip if not in killzone
            if not is_kz[rel_i]:
                continue

            abs_i = s + rel_i
            if not valid_atr[rel_i]:
                continue

            atr_val = atrs[rel_i]
            h_et = et_hours_scan[rel_i]
            bar_date = et_dates_scan[rel_i]

            # ── Update session levels (single session) ────────────
            # Asia: use yesterday's session if before noon ET, else today's
            asia_date = bar_date - timedelta(days=1) if h_et < 12 else bar_date
            if asia_date != last_asia_date:
                sess = asia_sess.get(asia_date)
                if sess is not None:
                    cur_asia_hi, cur_asia_lo = sess
                    # Reset state machine for new session
                    asia_lo_state = _WAITING
                    asia_hi_state = _WAITING
                    asia_lo_sweep_price = 0.0
                    asia_hi_sweep_price = 0.0
                    last_asia_date = asia_date

            # London: only update when in NY killzone
            london_date = bar_date
            if is_ny[rel_i] and london_date != last_london_date:
                sess = london_sess.get(london_date)
                if sess is not None:
                    cur_london_hi, cur_london_lo = sess
                    london_lo_state = _WAITING
                    london_hi_state = _WAITING
                    london_lo_sweep_price = 0.0
                    london_hi_sweep_price = 0.0
                    last_london_date = london_date

            # ── Determine directional bias (Fix 3) ────────────────
            sb = structure_bias[rel_i]
            if sb > 0:
                bias = 1   # LONG
            elif sb < 0:
                bias = -1  # SHORT
            else:
                # Fall back to EMA
                if ema50[rel_i] > ema200[rel_i]:
                    bias = 1
                elif ema50[rel_i] < ema200[rel_i]:
                    bias = -1
                else:
                    continue  # No bias = no signal

            close_val = closes[rel_i]
            low_val = lows[rel_i]
            high_val = highs[rel_i]

            # ── Price reclaim check (Fix 2) ───────────────────────
            # Reset SWEEP_DETECTED → WAITING if price closes beyond level
            if asia_lo_state == _SWEEP_DETECTED:
                if not np.isnan(cur_asia_lo) and close_val > cur_asia_lo:
                    asia_lo_state = _WAITING
                    asia_lo_sweep_price = 0.0

            if london_lo_state == _SWEEP_DETECTED:
                if not np.isnan(cur_london_lo) and close_val > cur_london_lo:
                    london_lo_state = _WAITING
                    london_lo_sweep_price = 0.0

            if asia_hi_state == _SWEEP_DETECTED:
                if not np.isnan(cur_asia_hi) and close_val < cur_asia_hi:
                    asia_hi_state = _WAITING
                    asia_hi_sweep_price = 0.0

            if london_hi_state == _SWEEP_DETECTED:
                if not np.isnan(cur_london_hi) and close_val < cur_london_hi:
                    london_hi_state = _WAITING
                    london_hi_sweep_price = 0.0

            # ── Sweep detection (Fix 2 + Fix 3 bias gate) ─────────
            # LONG bias: check low sweeps (sellside liquidity)
            if bias == 1:
                # Asia low
                if (asia_lo_state != _TRADED
                        and not np.isnan(cur_asia_lo) and cur_asia_lo > 0):
                    if low_val < cur_asia_lo:
                        if asia_lo_state == _WAITING:
                            asia_lo_state = _SWEEP_DETECTED
                            asia_lo_sweep_price = low_val
                        elif low_val < asia_lo_sweep_price:
                            asia_lo_sweep_price = low_val

                # London low (only in NY killzone — Fix 6)
                if (is_ny[rel_i]
                        and london_lo_state != _TRADED
                        and not np.isnan(cur_london_lo) and cur_london_lo > 0):
                    if low_val < cur_london_lo:
                        if london_lo_state == _WAITING:
                            london_lo_state = _SWEEP_DETECTED
                            london_lo_sweep_price = low_val
                        elif low_val < london_lo_sweep_price:
                            london_lo_sweep_price = low_val

            # SHORT bias: check high sweeps (buyside liquidity)
            elif bias == -1:
                # Asia high
                if (asia_hi_state != _TRADED
                        and not np.isnan(cur_asia_hi) and cur_asia_hi > 0):
                    if high_val > cur_asia_hi:
                        if asia_hi_state == _WAITING:
                            asia_hi_state = _SWEEP_DETECTED
                            asia_hi_sweep_price = high_val
                        elif high_val > asia_hi_sweep_price:
                            asia_hi_sweep_price = high_val

                # London high (only in NY killzone — Fix 6)
                # Fix 4: London High shorts disabled (0% historical WR)
                # Don't even detect London High sweeps since they'd all
                # be SHORT and rejected anyway.

            # ── Confirmation check (Fix 2) ────────────────────────
            # A directional candle confirms a pending sweep.
            # Check all levels that are SWEEP_DETECTED.
            confirmed_level = None
            confirmed_direction = None
            confirmed_sweep_price = 0.0
            confirmed_level_val = np.nan

            # Priority: Asia levels first (checked first in live bot)
            if bias == 1:
                # Check Asia low confirmation (LONG)
                if (asia_lo_state == _SWEEP_DETECTED
                        and is_bullish[rel_i]
                        and body_pct[rel_i] >= min_body):
                    confirmed_level = 'asia_low'
                    confirmed_direction = 'LONG'
                    confirmed_sweep_price = asia_lo_sweep_price
                    confirmed_level_val = cur_asia_lo

                # Check London low confirmation (LONG, NY only)
                elif (is_ny[rel_i]
                      and london_lo_state == _SWEEP_DETECTED
                      and is_bullish[rel_i]
                      and body_pct[rel_i] >= min_body):
                    confirmed_level = 'london_low'
                    confirmed_direction = 'LONG'
                    confirmed_sweep_price = london_lo_sweep_price
                    confirmed_level_val = cur_london_lo

            elif bias == -1:
                # Check Asia high confirmation (SHORT)
                if (asia_hi_state == _SWEEP_DETECTED
                        and is_bearish[rel_i]
                        and body_pct[rel_i] >= min_body):
                    confirmed_level = 'asia_high'
                    confirmed_direction = 'SHORT'
                    confirmed_sweep_price = asia_hi_sweep_price
                    confirmed_level_val = cur_asia_hi

                # London high shorts blocked (Fix 4) — no check

            if confirmed_level is None:
                continue

            # ── Cooldown ──────────────────────────────────────────
            if rel_i - last_sig_rel < min_cooldown:
                continue

            # ── Sweep depth check (Fix 5) ─────────────────────────
            depth = abs(confirmed_sweep_price - confirmed_level_val)
            depth_atr = depth / atr_val
            if depth_atr < min_depth_arr[rel_i]:
                continue

            # ── Mark level as TRADED ──────────────────────────────
            if confirmed_level == 'asia_low':
                asia_lo_state = _TRADED
            elif confirmed_level == 'london_low':
                london_lo_state = _TRADED
            elif confirmed_level == 'asia_high':
                asia_hi_state = _TRADED

            # ── Composite confidence (mean-reversion friendly) ────
            depth_c = min(depth_atr / 1.0, 1.0) * 0.50

            sb_val = structure_bias[rel_i]
            if confirmed_direction == 'LONG':
                struct_score = 0.20 if sb_val < 0 else (0.05 if sb_val == 0 else 0.0)
            else:
                struct_score = 0.20 if sb_val > 0 else (0.05 if sb_val == 0 else 0.0)

            if confirmed_direction == 'LONG':
                htf_score = 0.15 if htf_bearish[rel_i] else 0.0
            else:
                htf_score = 0.15 if htf_bullish[rel_i] else 0.0

            struct_conf_score = float(struct_conf_arr[rel_i]) * 0.15
            confidence = depth_c + struct_score + htf_score + struct_conf_score

            # ── R:R and entry/exit calculation ────────────────────
            eff_min_rr = min_rr * rr_scale_arr[rel_i]
            eff_max_rr = max_rr * rr_scale_arr[rel_i]
            vol_adjusted_mult = atr_mult * sl_vol_mult[rel_i]

            entry = close_val
            if confirmed_direction == 'LONG':
                stop_loss = entry - atr_val * vol_adjusted_mult
                if stop_loss >= entry:
                    continue
                risk = entry - stop_loss
                tp1 = entry + risk * eff_min_rr
                tp2 = entry + risk * eff_max_rr
            else:
                stop_loss = entry + atr_val * vol_adjusted_mult
                if stop_loss <= entry:
                    continue
                risk = stop_loss - entry
                tp1 = entry - risk * eff_min_rr
                tp2 = entry - risk * eff_max_rr

            signals.append(Signal(
                idx=abs_i,
                time=df.index[abs_i],
                direction=confirmed_direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk=risk,
                confidence=confidence,
                bias='COUNTER' if (struct_score > 0.1 or htf_score > 0) else 'PARTIAL',
                atr=atr_val,
            ))
            last_sig_rel = rel_i

        return signals


# ────────────────────────────────────────────────────────────────────
#  Module-level helpers (no instance state)
# ────────────────────────────────────────────────────────────────────

def _utc_to_et_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert a UTC DatetimeIndex to Eastern Time hours (DST-aware).

    If the index is already tz-aware (non-UTC), it is converted to UTC first.
    If the index is tz-naive, it is assumed to be UTC.
    """
    if index.tz is None:
        utc_index = index.tz_localize('UTC')
    else:
        utc_index = index.tz_convert('UTC')
    et_index = utc_index.tz_convert(_ET)
    return np.asarray(et_index.hour, dtype=np.int32)


def _compute_et_info(index: pd.DatetimeIndex) -> Dict[str, Any]:
    """Pre-compute ET hours and dates for the full DataFrame index."""
    if index.tz is None:
        utc_index = index.tz_localize('UTC')
    else:
        utc_index = index.tz_convert('UTC')
    et_index = utc_index.tz_convert(_ET)
    et_hours = np.asarray(et_index.hour, dtype=np.int32)
    et_dates = np.array([d.date() for d in et_index], dtype=object)
    return {'et_hours': et_hours, 'et_dates': et_dates}


def _build_session_lookups(
    et_hours: np.ndarray,
    et_dates: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> Tuple[Dict, Dict]:
    """Build single-session H/L lookup tables.

    Returns:
        (asia_sessions, london_sessions) where each is
        dict[date_type, (high, low)] for completed sessions.
    """
    n = len(et_hours)

    # Detect session boundaries via hour masks
    is_asia = et_hours >= 19       # 19:00-23:59 ET
    is_london = (et_hours >= 2) & (et_hours < 8)  # 02:00-08:00 ET

    asia_sessions: Dict[date_type, Tuple[float, float]] = {}
    london_sessions: Dict[date_type, Tuple[float, float]] = {}

    # Group bars by session date and type
    # Asia: bars with hour >= 19, keyed by that bar's ET date
    # London: bars with hour 2-7, keyed by that bar's ET date
    curr_asia_date = None
    curr_asia_bars: List[int] = []
    curr_london_date = None
    curr_london_bars: List[int] = []

    for i in range(n):
        h = et_hours[i]
        d = et_dates[i]

        # Asia session tracking (19:00-23:59)
        if h >= 19:
            if d != curr_asia_date:
                # Finalize previous Asia session
                if curr_asia_bars:
                    bar_idx = np.array(curr_asia_bars)
                    asia_sessions[curr_asia_date] = (
                        float(np.max(highs[bar_idx])),
                        float(np.min(lows[bar_idx])),
                    )
                curr_asia_date = d
                curr_asia_bars = [i]
            else:
                curr_asia_bars.append(i)
        else:
            # Finalize Asia if transitioning out
            if curr_asia_bars and curr_asia_date is not None:
                bar_idx = np.array(curr_asia_bars)
                asia_sessions[curr_asia_date] = (
                    float(np.max(highs[bar_idx])),
                    float(np.min(lows[bar_idx])),
                )
                curr_asia_bars = []

        # London session tracking (02:00-07:59)
        if 2 <= h < 8:
            if d != curr_london_date:
                if curr_london_bars:
                    bar_idx = np.array(curr_london_bars)
                    london_sessions[curr_london_date] = (
                        float(np.max(highs[bar_idx])),
                        float(np.min(lows[bar_idx])),
                    )
                curr_london_date = d
                curr_london_bars = [i]
            else:
                curr_london_bars.append(i)
        else:
            if curr_london_bars and curr_london_date is not None:
                bar_idx = np.array(curr_london_bars)
                london_sessions[curr_london_date] = (
                    float(np.max(highs[bar_idx])),
                    float(np.min(lows[bar_idx])),
                )
                curr_london_bars = []

    # Finalize any trailing sessions
    if curr_asia_bars and curr_asia_date is not None:
        bar_idx = np.array(curr_asia_bars)
        asia_sessions[curr_asia_date] = (
            float(np.max(highs[bar_idx])),
            float(np.min(lows[bar_idx])),
        )
    if curr_london_bars and curr_london_date is not None:
        bar_idx = np.array(curr_london_bars)
        london_sessions[curr_london_date] = (
            float(np.max(highs[bar_idx])),
            float(np.min(lows[bar_idx])),
        )

    return asia_sessions, london_sessions


# Keep legacy functions for backward compatibility with other adapters
def _compute_session_levels(
    df: pd.DataFrame, lookback_bars: int,
) -> Dict[str, np.ndarray]:
    """Compute session H/L via boundary detection (legacy, rolling aggregation).

    Used by non-LR adapters. LR adapter uses _build_session_lookups() instead.
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
    """Extract rolling session H/L from session boundary transitions (legacy)."""
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
