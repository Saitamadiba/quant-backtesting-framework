"""
Momentum Mastery (ICT Liquidity Sweep) adapter for WFO engine — v1.

Pure pandas/numpy signal generation, no backtrader dependency.
Implements the ICT methodology: session H/L liquidity pools, sweep detection,
confirmation candle validation, pullback entry with hybrid SL.

Signal generation flow (multi-bar pattern):
    1. Pre-compute session levels (Asia/London H/L from boundary detection)
    2. Pre-compute EMA 50/100/200 triple stack for directional bias
    3. Scan bars for liquidity sweeps:
       - Kill zone gate (London 02-08 ET, NY 08-16 ET)
       - Softened EMA triple bias (50 > 100 > 200 ± tolerance = LONG)
       - Sweep: price pierces session level and closes back (reversal wick)
    4. After sweep, scan forward up to max_confirm_bars for confirmation:
       - Candle body >= min_body_atr * ATR
       - Correct direction (bullish for LONG, bearish for SHORT)
       - Displacement past sweep price (high > sweep for LONG)
    5. Build signal at confirmation bar:
       - Entry at pullback into confirmation candle range
       - SL beyond sweep ± ATR buffer (with 0.5%-1.5% floor/cap)
       - TP at ATR-regime-scaled R:R

Confidence scoring (max 1.0):
    - Sweep depth / ATR:        0.30
    - Volume confirmation:      0.15
    - EMA trend alignment:      0.15
    - Confirmation quality:     0.15
    - Structure bias:           0.15
    - ATR regime bonus:         0.10
"""

from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal


class MomentumMasteryAdapter(StrategyAdapter):

    @property
    def name(self) -> str:
        return "MomentumMastery"

    @property
    def default_timeframes(self) -> List[str]:
        return ["1h"]

    def get_param_space(self) -> List[ParamSpec]:
        return [
            ParamSpec("session_lookback",    12,    6,     18,    6,    'int'),
            ParamSpec("sl_atr_buffer",       0.5,   0.3,   0.7,   0.2),
            ParamSpec("base_rr",             2.0,   1.5,   3.0,   0.5),
            ParamSpec("min_body_atr",        0.30,  0.20,  0.40,  0.10),
            ParamSpec("entry_pullback",      0.50,  0.30,  0.70,  0.20),
            ParamSpec("max_confirm_bars",    6,     3,     9,     3,    'int'),
            ParamSpec("min_confidence",      0.35,  0.20,  0.55,  0.05),
        ]

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        """Generate Momentum Mastery signals over [scan_start_idx, scan_end_idx)."""

        # ── Extract parameters ──────────────────────────────────────
        lookback = int(params.get('session_lookback', 12))
        sl_buffer = params.get('sl_atr_buffer', 0.5)
        base_rr = params.get('base_rr', 2.0)
        min_body = params.get('min_body_atr', 0.30)
        entry_pb = params.get('entry_pullback', 0.50)
        max_confirm = int(params.get('max_confirm_bars', 6))
        min_conf = params.get('min_confidence', 0.35)

        # R:R regime scaling (matches MM config: QUIET 1.5, NORMAL 2.0, VOLATILE 3.0)
        min_rr = max(1.5, base_rr - 0.5)
        max_rr = min(3.0, base_rr + 1.0)

        # ── Session levels (reuses boundary-detection approach) ─────
        levels = _compute_session_levels(df, lookback)

        # ── Pre-extract arrays ──────────────────────────────────────
        n = len(df)
        s = scan_start_idx
        e = min(scan_end_idx, n)
        if e <= s:
            return []

        opens  = df['Open'].values
        highs  = df['High'].values
        lows   = df['Low'].values
        closes = df['Close'].values
        atrs   = df['ATR'].values
        volumes = df['Volume'].values

        asia_hi   = levels['asia_high']
        asia_lo   = levels['asia_low']
        london_hi = levels['london_high']
        london_lo = levels['london_low']
        is_kz     = levels['is_killzone']
        et_hours  = levels['et_hours']

        # ── EMA triple stack (50/100/200) ───────────────────────────
        ema50 = df['EMA50'].values
        ema200 = df['EMA200'].values
        # EMA100 not in IndicatorEngine — compute locally
        ema100 = df['Close'].ewm(span=100, adjust=False).mean().values

        # ── Volume SMA for sweep volume scoring ────────────────────
        vol_sma = df['Volume_SMA'].values if 'Volume_SMA' in df.columns else None

        # ── Structure bias (from IndicatorEngine) ──────────────────
        has_structure = 'StructureBias' in df.columns
        if has_structure:
            structure_bias = df['StructureBias'].values
        else:
            structure_bias = np.zeros(n)

        # ── ATR percentile for regime classification ───────────────
        has_pctile = 'ATR_Pctile100' in df.columns
        if has_pctile:
            atr_pctile = df['ATR_Pctile100'].values
        else:
            atr_pctile = np.full(n, 50.0)

        # ── DVOL for IV analysis metadata ──────────────────────────
        has_dvol = 'DVOL' in df.columns
        dvol_arr = df['DVOL'].values if has_dvol else np.full(n, np.nan)

        # ── EMA tolerance for softened bias ────────────────────────
        ema_tol = 0.005  # Fixed at config default

        # ── Scan for sweep → confirmation patterns ─────────────────
        signals: List[Signal] = []
        min_cooldown = 4
        last_sig_idx = -min_cooldown

        i = s
        while i < e:
            # Skip if in cooldown
            if i - last_sig_idx < min_cooldown:
                i += 1
                continue

            # HARD GATE: kill zone
            if not is_kz[i]:
                i += 1
                continue

            # HARD GATE: valid ATR
            atr_val = atrs[i]
            if atr_val <= 0 or np.isnan(atr_val):
                i += 1
                continue

            # ── Softened EMA triple bias ────────────────────────────
            e50, e100, e200 = ema50[i], ema100[i], ema200[i]
            if np.isnan(e50) or np.isnan(e100) or np.isnan(e200):
                i += 1
                continue

            bias = _get_ema_bias(e50, e100, e200, ema_tol)
            if bias == 'NONE':
                i += 1
                continue

            # ── Check for liquidity sweep ───────────────────────────
            sweep_info = _check_sweep(
                i, bias, highs, lows, closes,
                asia_hi, asia_lo, london_hi, london_lo
            )
            if sweep_info is None:
                i += 1
                continue

            sweep_price = sweep_info['sweep_price']
            sweep_level = sweep_info['level']
            sweep_depth = sweep_info['depth']

            # ── Scan forward for confirmation candle ────────────────
            confirmed = False
            confirm_idx = -1

            for j in range(i + 1, min(i + 1 + max_confirm, e)):
                # Still in kill zone?
                if not is_kz[j]:
                    continue

                atr_j = atrs[j]
                if atr_j <= 0 or np.isnan(atr_j):
                    continue

                # Candle body quality
                body = abs(closes[j] - opens[j])
                if body < min_body * atr_j:
                    continue

                if bias == 'LONG':
                    # Must be bullish candle with displacement above sweep
                    if closes[j] <= opens[j]:
                        continue
                    if highs[j] <= sweep_price:
                        continue
                    confirmed = True
                    confirm_idx = j
                    break
                else:  # SHORT
                    # Must be bearish candle with displacement below sweep
                    if closes[j] >= opens[j]:
                        continue
                    if lows[j] >= sweep_price:
                        continue
                    confirmed = True
                    confirm_idx = j
                    break

            if not confirmed:
                i += 1
                continue

            # ── Build signal at confirmation bar ────────────────────
            c_high = highs[confirm_idx]
            c_low = lows[confirm_idx]
            c_close = closes[confirm_idx]
            c_atr = atrs[confirm_idx]
            c_range = c_high - c_low

            if c_range <= 0:
                i += 1
                continue

            # ATR regime classification
            pctile = atr_pctile[confirm_idx]
            if np.isnan(pctile):
                rr_ratio = base_rr
                atr_regime = 'NORMAL'
            elif pctile < 25:
                rr_ratio = min_rr
                atr_regime = 'QUIET'
            elif pctile > 75:
                rr_ratio = max_rr
                atr_regime = 'VOLATILE'
            else:
                rr_ratio = base_rr
                atr_regime = 'NORMAL'

            # Entry at pullback into confirmation candle
            entry = c_low + c_range * entry_pb

            # Hybrid SL: beyond sweep + ATR buffer, with floor/cap
            if bias == 'LONG':
                stop_loss = sweep_price - c_atr * sl_buffer

                # Floor: min 0.5% SL distance
                sl_dist_pct = abs(entry - stop_loss) / entry
                if sl_dist_pct < 0.005:
                    stop_loss = entry * (1 - 0.005)
                    sl_dist_pct = 0.005

                # Cap: max 1.5% SL distance
                if sl_dist_pct > 0.015:
                    stop_loss = entry * (1 - 0.015)
                    sl_dist_pct = 0.015

                if stop_loss >= entry:
                    i = confirm_idx + 1
                    continue

                risk = entry - stop_loss
                tp1 = entry + risk * rr_ratio
                tp2 = entry + risk * rr_ratio * 1.5

            else:  # SHORT
                stop_loss = sweep_price + c_atr * sl_buffer

                sl_dist_pct = abs(stop_loss - entry) / entry
                if sl_dist_pct < 0.005:
                    stop_loss = entry * (1 + 0.005)
                    sl_dist_pct = 0.005
                if sl_dist_pct > 0.015:
                    stop_loss = entry * (1 + 0.015)
                    sl_dist_pct = 0.015

                if stop_loss <= entry:
                    i = confirm_idx + 1
                    continue

                risk = stop_loss - entry
                tp1 = entry - risk * rr_ratio
                tp2 = entry - risk * rr_ratio * 1.5

            # ── Confidence scoring ──────────────────────────────────
            # 1. Sweep depth / ATR (0.0 to 0.30)
            depth_atr = sweep_depth / c_atr
            depth_score = min(depth_atr / 1.0, 1.0) * 0.30

            # 2. Volume confirmation (0.0 to 0.15)
            if vol_sma is not None and vol_sma[i] > 0 and not np.isnan(vol_sma[i]):
                vol_ratio = volumes[i] / vol_sma[i]
                vol_score = min(vol_ratio / 2.0, 1.0) * 0.15
            else:
                vol_score = 0.075  # neutral

            # 3. EMA trend alignment (0.0 to 0.15)
            # Full alignment: all 3 EMAs stacked correctly
            ema_score = 0.15  # We already gated on EMA bias, so this is always aligned

            # 4. Confirmation candle quality (0.0 to 0.15)
            body_ratio = abs(closes[confirm_idx] - opens[confirm_idx]) / c_atr
            confirm_score = min(body_ratio / 0.8, 1.0) * 0.15

            # 5. Structure bias alignment (0.0 to 0.15)
            sb = structure_bias[confirm_idx]
            if bias == 'LONG':
                struct_score = 0.15 if sb > 0 else (0.05 if sb == 0 else 0.0)
            else:
                struct_score = 0.15 if sb < 0 else (0.05 if sb == 0 else 0.0)

            # 6. ATR regime bonus (0.0 to 0.10)
            if atr_regime == 'VOLATILE':
                regime_score = 0.10
            elif atr_regime == 'NORMAL':
                regime_score = 0.05
            else:
                regime_score = 0.0

            confidence = (depth_score + vol_score + ema_score
                          + confirm_score + struct_score + regime_score)
            # Max possible: 0.30 + 0.15 + 0.15 + 0.15 + 0.15 + 0.10 = 1.0

            # Soft gate: minimum confidence
            if confidence < min_conf:
                i = confirm_idx + 1
                continue

            # Determine signal bias label
            if struct_score >= 0.10:
                bias_label = 'ALIGNED'
            elif struct_score > 0:
                bias_label = 'PARTIAL'
            else:
                bias_label = 'COUNTER'

            # Kill zone name
            et_h = et_hours[confirm_idx]
            kz_name = 'LONDON' if 2 <= et_h < 8 else 'NEW_YORK'

            signals.append(Signal(
                idx=confirm_idx,
                time=df.index[confirm_idx],
                direction=bias,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk=risk,
                confidence=confidence,
                bias=bias_label,
                atr=c_atr,
                metadata={
                    'sweep_type': sweep_info['type'],
                    'sweep_price': sweep_price,
                    'sweep_depth_atr': depth_atr,
                    'atr_regime': atr_regime,
                    'rr_ratio': rr_ratio,
                    'killzone': kz_name,
                    'dvol': float(dvol_arr[confirm_idx]) if not np.isnan(dvol_arr[confirm_idx]) else None,
                },
            ))
            last_sig_idx = confirm_idx
            i = confirm_idx + 1
            continue

        return signals


# ────────────────────────────────────────────────────────────────────
#  Module-level helpers (no instance state)
# ────────────────────────────────────────────────────────────────────

def _get_ema_bias(e50: float, e100: float, e200: float, tol: float) -> str:
    """Softened EMA triple stack bias (matches MM strategy logic)."""
    if (e50 >= e100 * (1 - tol)
        and e100 >= e200 * (1 - tol)
        and e50 > e200):
        return 'LONG'
    if (e50 <= e100 * (1 + tol)
        and e100 <= e200 * (1 + tol)
        and e50 < e200):
        return 'SHORT'
    return 'NONE'


def _check_sweep(
    i: int,
    bias: str,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    asia_hi: np.ndarray,
    asia_lo: np.ndarray,
    london_hi: np.ndarray,
    london_lo: np.ndarray,
) -> dict:
    """Check for liquidity sweep at bar i.

    For LONG: low must pierce a sellside level (session low) and close back above.
    For SHORT: high must pierce a buyside level (session high) and close back below.

    Returns dict with sweep details or None.
    """
    if bias == 'LONG':
        # Check sellside levels (session lows)
        best = None
        for lvl_val, lvl_name in [
            (asia_lo[i], 'ASIA_LOW'),
            (london_lo[i], 'LONDON_LOW'),
        ]:
            if np.isnan(lvl_val) or lvl_val <= 0:
                continue
            # Sweep: low pierces level, close recovers above
            if lows[i] < lvl_val and closes[i] > lvl_val:
                depth = lvl_val - lows[i]
                if best is None or depth > best['depth']:
                    best = {
                        'type': lvl_name,
                        'level': lvl_val,
                        'sweep_price': lows[i],
                        'depth': depth,
                    }
        return best

    else:  # SHORT
        best = None
        for lvl_val, lvl_name in [
            (asia_hi[i], 'ASIA_HIGH'),
            (london_hi[i], 'LONDON_HIGH'),
        ]:
            if np.isnan(lvl_val) or lvl_val <= 0:
                continue
            if highs[i] > lvl_val and closes[i] < lvl_val:
                depth = highs[i] - lvl_val
                if best is None or depth > best['depth']:
                    best = {
                        'type': lvl_name,
                        'level': lvl_val,
                        'sweep_price': highs[i],
                        'depth': depth,
                    }
        return best


def _compute_session_levels(
    df: pd.DataFrame, lookback_bars: int,
) -> Dict[str, np.ndarray]:
    """Compute session H/L via boundary detection.

    Same approach as LiquidityRaid adapter: loops over session boundaries
    (fast, ~365/year) rather than each bar.
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
