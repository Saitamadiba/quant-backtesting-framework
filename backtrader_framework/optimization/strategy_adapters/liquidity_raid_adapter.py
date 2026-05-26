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

    # Optional per-regime sweep-depth floor (ATR units). When set, the
    # non-DVOL branch in generate_signals builds min_depth_arr by mapping
    # each bar's regime → its floor (with min_sweep_atr as the fallback for
    # any unmapped regime label). Used by run_lr_nq_regime_adaptive.py.
    _regime_depth_floors: Optional[Dict[str, float]] = None

    # Optional per-asset entry-hour whitelist (ET, 24h ints in 0..23).
    # When set, the signal loop skips bars whose ET hour is not in this
    # set, narrowing the existing killzone (London 02-08, NY 08-16).
    # Motivation: 5y NQ analysis showed entries 04-11 ET are net-negative
    # (−83R over 5y); restricting to {12,13,14,15} captures the positive
    # NY-afternoon window (+199R).  See lr_nq_entry_quality_findings.md.
    _entry_hours_et: Optional[set] = None

    # Optional confidence floor (opt-in).  The composite confidence score
    # is informational by default ("no hard reject threshold" per the
    # module docstring).  When this attr is set, the signal loop rejects
    # signals whose computed `confidence` is below the threshold.  NQ
    # uses 0.55 to cut the Q1 (mR=−0.075) bucket — see
    # lr_nq_entry_quality_findings.md.
    _min_confidence_override: Optional[float] = None

    # Optional per-regime confidence floor (opt-in).  Same mechanic as
    # _min_confidence_override but keyed on the per-bar regime.  Used
    # for narrower gates (e.g., NQ: require confidence ≥ 0.61 in
    # `quiet_trend` only).  Falls back to _min_confidence_override (or
    # no floor) when regime not in dict.
    _per_regime_min_confidence: Optional[Dict[str, float]] = None

    # ── Sweep-quality gate (mirrors live the internal strategy core)
    #  Weights + threshold from the internal strategy core.
    #  Off by default for backward compat with existing runs; enable via
    #  `enable_sweep_quality_gate()` to match live bot behaviour.
    _sq_enabled: bool = False
    _sq_min_score: float = 40.0
    _sq_w_depth: float = 0.40
    _sq_w_volume: float = 0.25
    _sq_w_time: float = 0.15
    _sq_w_confirm: float = 0.20

    def enable_sweep_quality_gate(self, min_score: float = 40.0,
                                   w_depth: float = 0.40,
                                   w_volume: float = 0.25,
                                   w_time: float = 0.15,
                                   w_confirm: float = 0.20):
        """Enable the 4-component sweep-quality gate (live bot default).

        Matches the internal strategy core:
            depth_score:   0.5 → 1.0 as sweep depth grows toward 2× ATR
            volume_score:  0.3 (< avg) → 1.0 (> 2× avg) tiered
            time_score:    1.0 (fresh) → 0.2 (>12 bars stale) tiered
            confirm_score: body_ratio × direction_match tiered
        Weighted → 0-100. Reject if below `min_score` (default 40).
        """
        self._sq_enabled = True
        self._sq_min_score = float(min_score)
        self._sq_w_depth = float(w_depth)
        self._sq_w_volume = float(w_volume)
        self._sq_w_time = float(w_time)
        self._sq_w_confirm = float(w_confirm)

    def disable_sweep_quality_gate(self):
        self._sq_enabled = False

    @staticmethod
    def _sq_depth_score(depth_atr: float) -> float:
        return min(0.5 + depth_atr / 2.0, 1.0) if depth_atr >= 0 else 0.5

    @staticmethod
    def _sq_volume_score(vol_ratio: float) -> float:
        if np.isnan(vol_ratio): return 0.6   # treat missing vol as neutral
        if vol_ratio < 1.0:  return 0.3
        if vol_ratio < 1.5:  return 0.6
        if vol_ratio < 2.0:  return 0.8
        return 1.0

    @staticmethod
    def _sq_time_score(candles_since: int) -> float:
        if candles_since <= 1:  return 1.0
        if candles_since <= 3:  return 0.8
        if candles_since <= 6:  return 0.6
        if candles_since <= 12: return 0.4
        return 0.2

    @staticmethod
    def _sq_confirm_score(body_ratio: float, is_direction_match: bool) -> float:
        if not is_direction_match: return 0.2
        if body_ratio > 0.7: return 1.0
        if body_ratio > 0.5: return 0.8
        if body_ratio > 0.3: return 0.6
        return 0.4

    def _sq_quality(self, depth_atr: float, vol_ratio: float,
                    candles_since: int, body_ratio: float,
                    direction_match: bool) -> float:
        d = self._sq_depth_score(depth_atr)
        v = self._sq_volume_score(vol_ratio)
        t = self._sq_time_score(candles_since)
        c = self._sq_confirm_score(body_ratio, direction_match)
        return (d * self._sq_w_depth
                + v * self._sq_w_volume
                + t * self._sq_w_time
                + c * self._sq_w_confirm) * 100.0

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

        Note: the sweep-quality scorer (live bot Phase-2 feature) is wired
        but intentionally NOT in the param space.  Empirical WFO showed it
        overfits IS when tunable — at live's default threshold=40 it is
        already redundant with min_depth_threshold.  The scorer remains
        available for metadata + live parity via enable_sweep_quality_gate().
        """
        return [
            ParamSpec("session_lookback",   12,    6,     18,    6,    'int'),
            ParamSpec("atr_sl_multiplier",  2.5,   1.5,   3.5,   0.5),
            ParamSpec("min_rr",             0.5,   0.3,   1.0,   0.5),
            ParamSpec("max_rr",             0.75,  0.5,   1.5,   0.5),
            ParamSpec("min_body_pct",       0.15,  0.10,  0.25,  0.05),
            ParamSpec("sweep_tolerance",    0.002, 0.001, 0.003, 0.001),
            ParamSpec("min_confidence",     0.25,  0.15,  0.45,  0.05),
            # Sweep-depth floor (non-DVOL fallback). Live default 0.30 ATR.
            # Exposed for studies that want to test whether shallow sweeps
            # are profitable in specific regimes (e.g., ranging on NQ).
            ParamSpec("min_sweep_atr",      0.30,  0.05,  0.30,  0.05),
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
        # Non-DVOL fallback floor for sweep depth (in ATR units). The
        # DVOL-adaptive branch below still computes its own per-bar floor;
        # this param only affects the non-DVOL fallback (i.e., NQ).
        min_sweep_atr = params.get('min_sweep_atr', 0.30)
        # Research knob — disables the hardcoded regime-direction block at
        # lines 491-494 so counter-trend signals (LONG in trending_down,
        # SHORT in trending_up) come through.  Used by the skipped-sweep
        # redemption study (an internal counter-trend study).
        disable_regime_filter = bool(params.get('disable_regime_direction_filter', False))
        # Research knobs (internal): bias-gate options + EMA-stretch
        # mean-reversion check + configurable ADX threshold.  Used by
        # an internal options/bias study to compare Options
        # A/B/C of the May-23 missed-reversal post-mortem.
        bias_mode = str(params.get('bias_mode', 'structure_default'))
        # 'structure_default' = current logic; 'fast_4h_ema20' = 4H EMA20
        # slope; 'disabled' irrelevant (use force_bias for Option C);
        # 'trend_confidence' = P1-M3b composite gate — reads
        #   df['ComposLong'] / df['ComposShort'] (pre-attached by the
        #   feature_lab study driver) and emits bias when composite ≥
        #   tc_threshold on the dominant side.  See
        #   an internal trend-confidence study.
        tc_threshold = float(params.get('tc_threshold', 55.0))
        force_bias = params.get('force_bias', None)
        # 1 = always LONG-scan (low sweeps); -1 = always SHORT-scan;
        # None = use bias_mode logic.  Option C = run twice w/ +1, -1.
        mean_revers_stretch_min = params.get(
            'mean_reversion_ema_stretch_min_atr', None)
        # When not None: require |price-EMA50|/ATR >= this on the
        # correct side (BUY: price below; SELL: price above).
        regime_adx_threshold = float(params.get('regime_adx_threshold', 30.0))

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

        # ── Per-bar regime (vectorised; mirrors live bot classify_regime) ──
        # Defaults from the internal strategy core.
        # Live bot blocks LONG in trending_down and SHORT in trending_up
        # (REGIME_DIRECTION_FILTER_ENABLED default True, config_base.py:308).
        lookback = 50
        adx_arr_full = df['ADX'].values if 'ADX' in df.columns else None
        atr_full = df['ATR'].values
        close_full = df['Close'].values
        ema50_full = df['EMA50'].values
        # Rolling ATR%/Close mean over lookback using pandas for speed.
        atr_pct_full = atr_full / np.where(close_full > 0, close_full, np.nan)
        avg_atr_pct_full = pd.Series(atr_pct_full).rolling(lookback).mean().values
        regime_full = np.full(len(df), 'unknown', dtype=object)
        valid_regime = ~np.isnan(avg_atr_pct_full) & (close_full > 0) & (atr_full > 0)
        volatile_mask = valid_regime & (atr_pct_full > avg_atr_pct_full * 1.8)
        if adx_arr_full is not None:
            trending_mask = valid_regime & (~volatile_mask) & (adx_arr_full > regime_adx_threshold)
            trend_up = trending_mask & (close_full > ema50_full)
            trend_dn = trending_mask & (close_full <= ema50_full)
        else:
            trend_up = np.zeros(len(df), dtype=bool)
            trend_dn = np.zeros(len(df), dtype=bool)
        ranging_mask = valid_regime & (~volatile_mask) & (~trend_up) & (~trend_dn)
        regime_full[volatile_mask] = 'volatile'
        regime_full[trend_up] = 'trending_up'
        regime_full[trend_dn] = 'trending_down'
        regime_full[ranging_mask] = 'ranging'
        regime_arr = regime_full[sl]

        # ── 4H EMA20 slope (for bias_mode='fast_4h_ema20') ────────
        # Pre-compute once; reindex back to 15m grid via forward-fill.
        # Slope = (EMA20 - EMA20[3 bars ago]) / EMA20[3 bars ago].
        # 3 4H bars = 12h lookback — much faster reaction than EMA50/200
        # stack but still noise-resistant.  Only computed when needed.
        if bias_mode == 'fast_4h_ema20':
            _h4 = df['Close'].resample('4h').last().to_frame('close')
            _h4['ema20'] = _h4['close'].ewm(span=20, adjust=False).mean()
            _h4['slope'] = _h4['ema20'].diff(3) / _h4['ema20'].shift(3)
            _h4_slope_full = _h4['slope'].reindex(df.index, method='ffill').values
            h4_slope_sl = _h4_slope_full[sl]
        else:
            h4_slope_sl = None

        # ── Trend Confidence composite (for bias_mode='trend_confidence') ──
        # The composite_long/short series must be pre-attached to df by
        # the study driver (the WFO engine doesn't synthesize them).
        # See the internal strategy core.
        if bias_mode == 'trend_confidence':
            if 'ComposLong' not in df.columns or 'ComposShort' not in df.columns:
                raise ValueError(
                    "bias_mode='trend_confidence' requires df['ComposLong'] "
                    "and df['ComposShort'] pre-attached. Use "
                    "Liquidity_Raid.core.trend_confidence.compute_scores_vectorized "
                    "to compute and attach them before calling generate_signals."
                )
            compos_long_sl = df['ComposLong'].values[sl]
            compos_short_sl = df['ComposShort'].values[sl]
        else:
            compos_long_sl = None
            compos_short_sl = None

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
            # Non-DVOL fallback (e.g., NQ): use the parameterised floor so
            # studies can sweep this threshold. Default 0.30 matches the
            # live LR config (config_base.py:244, "Fix 5: was 0.15").
            if self._regime_depth_floors:
                # Per-regime depth floor: map each scan bar's regime label
                # to its floor; fall back to the scalar min_sweep_atr for
                # any regime not present in the dict (and for 'unknown').
                floors = self._regime_depth_floors
                fallback = float(min_sweep_atr)
                min_depth_arr = np.array(
                    [float(floors.get(r, fallback)) for r in regime_arr],
                    dtype=float,
                )
            else:
                min_depth_arr = np.full(scan_len, float(min_sweep_atr))
            rr_scale_arr = np.ones(scan_len)

        # Research override: when `min_sweep_atr_override` is set, it
        # replaces *whatever* min_depth_arr the branches above chose —
        # including the hardcoded DVOL-bucket floors (0.25 LOW/HIGH IV,
        # 0.35 MED IV).  Lets sensitivity studies sweep the depth filter
        # without touching the live config.  Default None = identical
        # behaviour to before this knob existed.
        _override = params.get('min_sweep_atr_override', None)
        if _override is not None:
            min_depth_arr = np.full(scan_len, float(_override))

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

        # ── Volume ratio for metadata ────────────────────────────
        has_vol_sma = 'Volume_SMA' in df.columns
        if has_vol_sma:
            _volumes = df['Volume'].values[sl]
            _vol_sma = df['Volume_SMA'].values[sl]
            _vol_sma_safe = np.maximum(_vol_sma, 1e-10)
            volume_ratio_arr = _volumes / _vol_sma_safe
        else:
            volume_ratio_arr = np.full(scan_len, np.nan)

        # ── ADX / RSI for metadata ───────────────────────────────
        has_adx = 'ADX' in df.columns
        adx_arr = df['ADX'].values[sl] if has_adx else np.full(scan_len, np.nan)
        has_rsi = 'RSI' in df.columns
        rsi_arr = df['RSI'].values[sl] if has_rsi else np.full(scan_len, np.nan)

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

        # Bar indices (relative) when each level first transitioned to SWEEP_DETECTED.
        # Used by the sweep-quality scorer to compute time-decay.
        asia_lo_first_bar = -1
        asia_hi_first_bar = -1
        london_lo_first_bar = -1
        london_hi_first_bar = -1

        # Current session levels (scalars)
        cur_asia_hi = np.nan
        cur_asia_lo = np.nan
        cur_london_hi = np.nan
        cur_london_lo = np.nan

        # Track session dates for state reset
        last_asia_date = None
        last_london_date = None

        signals: List[Signal] = []
        # Live bot imposes no cooldown between signals (removed the false
        # 4-bar gate that was deflating signal count vs live). Sweep state
        # machine transitions to _TRADED after a confirmation, which is the
        # real deduplication.

        for rel_i in range(scan_len):
            # Skip if not in killzone
            if not is_kz[rel_i]:
                continue

            # Optional per-asset entry-hour whitelist (ET).
            if self._entry_hours_et is not None and int(et_hours_scan[rel_i]) not in self._entry_hours_et:
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
                    asia_lo_first_bar = -1
                    asia_hi_first_bar = -1
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
                    london_lo_first_bar = -1
                    london_hi_first_bar = -1
                    last_london_date = london_date

            # ── Determine directional bias ────────────────────────
            # Mode 1 (structure_default): current logic — StructureBias
            # primary, EMA50 vs EMA200 fallback.
            # Mode 2 (fast_4h_ema20): 4H EMA20 slope sign.
            # Force-bias overrides (used by Option C: run twice as +1, -1).
            if force_bias is not None:
                bias = int(force_bias)
            elif bias_mode == 'fast_4h_ema20':
                _slope = h4_slope_sl[rel_i] if h4_slope_sl is not None else 0.0
                if np.isnan(_slope) or _slope == 0:
                    continue
                bias = 1 if _slope > 0 else -1
            elif bias_mode == 'trend_confidence':
                _cl = compos_long_sl[rel_i] if compos_long_sl is not None else float('nan')
                _cs = compos_short_sl[rel_i] if compos_short_sl is not None else float('nan')
                if np.isnan(_cl) or np.isnan(_cs):
                    continue
                if _cl >= tc_threshold and _cl > _cs:
                    bias = 1
                elif _cs >= tc_threshold and _cs > _cl:
                    bias = -1
                else:
                    continue
            else:  # structure_default
                sb = structure_bias[rel_i]
                if sb > 0:
                    bias = 1   # LONG
                elif sb < 0:
                    bias = -1  # SHORT
                else:
                    if ema50[rel_i] > ema200[rel_i]:
                        bias = 1
                    elif ema50[rel_i] < ema200[rel_i]:
                        bias = -1
                    else:
                        continue

            # ── Mean-reversion EMA-stretch gate (port from live) ──
            # Foundational LR check: BUY requires price BELOW EMA50;
            # SELL requires price ABOVE.  Default None = disabled.
            # See [[lr_foundational_fixes_may23]].
            if mean_revers_stretch_min is not None:
                _atr_now = atrs[rel_i]
                if _atr_now and _atr_now > 0:
                    _ema50_now = ema50[rel_i]
                    _price_now = closes[rel_i]
                    if bias == 1:
                        _stretch = (_ema50_now - _price_now) / _atr_now
                    else:
                        _stretch = (_price_now - _ema50_now) / _atr_now
                    if _stretch < float(mean_revers_stretch_min):
                        continue

            # ── Regime direction filter (mirrors live bot) ────────
            # REGIME_DIRECTION_FILTER_ENABLED=True by default in live LR bot.
            # Live-WFO justification: internal analysis flagged regime-direction mismatches as negative-EV.  Hard-block them.
            reg = regime_arr[rel_i]
            if not disable_regime_filter:
                if reg == 'trending_up' and bias == -1:
                    continue
                if reg == 'trending_down' and bias == 1:
                    continue

            close_val = closes[rel_i]
            low_val = lows[rel_i]
            high_val = highs[rel_i]

            # ── Price reclaim check (Fix 2) ───────────────────────
            # Reset SWEEP_DETECTED → WAITING if price closes beyond level
            if asia_lo_state == _SWEEP_DETECTED:
                if not np.isnan(cur_asia_lo) and close_val > cur_asia_lo:
                    asia_lo_state = _WAITING
                    asia_lo_sweep_price = 0.0
                    asia_lo_first_bar = -1

            if london_lo_state == _SWEEP_DETECTED:
                if not np.isnan(cur_london_lo) and close_val > cur_london_lo:
                    london_lo_state = _WAITING
                    london_lo_sweep_price = 0.0
                    london_lo_first_bar = -1

            if asia_hi_state == _SWEEP_DETECTED:
                if not np.isnan(cur_asia_hi) and close_val < cur_asia_hi:
                    asia_hi_state = _WAITING
                    asia_hi_sweep_price = 0.0
                    asia_hi_first_bar = -1

            if london_hi_state == _SWEEP_DETECTED:
                if not np.isnan(cur_london_hi) and close_val < cur_london_hi:
                    london_hi_state = _WAITING
                    london_hi_sweep_price = 0.0
                    london_hi_first_bar = -1

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
                            asia_lo_first_bar = rel_i
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
                            london_lo_first_bar = rel_i
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
                            asia_hi_first_bar = rel_i
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

            # ── Sweep depth check (Fix 5) ─────────────────────────
            depth = abs(confirmed_sweep_price - confirmed_level_val)
            depth_atr = depth / atr_val
            if depth_atr < min_depth_arr[rel_i]:
                continue

            # ── Sweep-quality scorer (mirrors live SweepQualityScorer) ──
            # Default adapter behaviour: score every signal for metadata, but
            # the live-default threshold=40 is redundant with min_depth (empirically
            # filters 0% of signals).  Enable enforcement via enable_sweep_quality_gate().
            sq_score = None
            sq_threshold = self._sq_min_score if self._sq_enabled else 0.0
            if self._sq_enabled or True:   # always compute for metadata
                # Resolve the sweep's first-detection bar for time-decay
                if confirmed_level == 'asia_low':
                    first_bar = asia_lo_first_bar
                elif confirmed_level == 'london_low':
                    first_bar = london_lo_first_bar
                else:  # asia_high
                    first_bar = asia_hi_first_bar
                candles_since = (rel_i - first_bar) if first_bar >= 0 else 0
                vol_ratio_val = float(volume_ratio_arr[rel_i]) if has_vol_sma else float('nan')
                is_dir_match = (confirmed_direction == 'LONG'
                                and is_bullish[rel_i]) or \
                               (confirmed_direction == 'SHORT'
                                and is_bearish[rel_i])
                sq_score = self._sq_quality(
                    depth_atr=depth_atr,
                    vol_ratio=vol_ratio_val,
                    candles_since=candles_since,
                    body_ratio=float(body_pct[rel_i]),
                    direction_match=bool(is_dir_match),
                )
                if self._sq_enabled and sq_score < sq_threshold:
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

            # ── Optional confidence floor (opt-in) ────────────────
            # Per-regime floor takes precedence; falls back to the
            # global override; otherwise no filter.
            if self._per_regime_min_confidence is not None:
                regime_floor = self._per_regime_min_confidence.get(regime_arr[rel_i])
                if regime_floor is not None and confidence < regime_floor:
                    continue
                elif regime_floor is None and self._min_confidence_override is not None \
                        and confidence < self._min_confidence_override:
                    continue
            elif self._min_confidence_override is not None and confidence < self._min_confidence_override:
                continue

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
                metadata={
                    'sweep_type': confirmed_level,
                    'depth_atr': float(depth_atr),
                    'body_ratio': float(body_pct[rel_i]),
                    'volume_ratio': float(volume_ratio_arr[rel_i]) if has_vol_sma else None,
                    'hour_et': int(h_et),
                    'session': 'london' if is_london[rel_i] else 'ny',
                    'structure_bias_val': float(sb_val),
                    'structure_conf_val': float(struct_conf_arr[rel_i]),
                    'htf_bullish': bool(htf_bullish[rel_i]),
                    'htf_bearish': bool(htf_bearish[rel_i]),
                    'adx_val': float(adx_arr[rel_i]) if has_adx and not np.isnan(adx_arr[rel_i]) else None,
                    'rsi_val': float(rsi_arr[rel_i]) if has_rsi and not np.isnan(rsi_arr[rel_i]) else None,
                    'atr_pctile20': float(atr_pctile[rel_i]) if has_pctile and not np.isnan(atr_pctile[rel_i]) else None,
                    'sl_vol_mult': float(sl_vol_mult[rel_i]),
                    'rr_scale': float(rr_scale_arr[rel_i]),
                    'min_depth_threshold': float(min_depth_arr[rel_i]),
                    'sweep_quality_score': float(sq_score) if sq_score is not None else None,
                    'candle_range_atr': float(candle_range[rel_i] / atr_val),
                    'close_position_in_range': float((closes[rel_i] - lows[rel_i]) / max(candle_range[rel_i], 1e-10)),
                    'is_bullish_bar': bool(is_bullish[rel_i]),
                },
            ))

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
