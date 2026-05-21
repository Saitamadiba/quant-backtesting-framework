"""
FVG (Fair Value Gap) adapter for WFO engine — v6.

Pure pandas/numpy signal generation, no backtrader dependency.

v6 improvements (over v5) — Entry Quality & Regime Alignment, Mar 2026:

    1. LIMIT-ORDER ENTRY at FVG zone: Signals are generated only when
       price actually retraces INTO the FVG zone (bar wick penetrates),
       with entry_price = touch point clamped to the gap zone.
       Previous: entered at midpoint regardless of actual bar range.

    2. HTF PRIORITIZATION: 4h FVGs are detected alongside 1h, and receive
       a confidence boost. Larger institutional gaps are more reliable.

    3. iFVG MITIGATION BUFFER: FVGs are only considered "mitigated" when
       price closes beyond the zone edge by 10% of gap range (buffer).
       Previous: any close 1 tick beyond killed the FVG.

    4. EMA REGIME DIRECTION FILTER: Hard filter — no longs in downtrends
       (EMA50 < EMA200), no shorts in uptrends. WFO data showed materially better performance in one regime than the other.

    5. R:R FROM ACTUAL ENTRY: SL/TP computed from the retest touch price,
       not the idealized midpoint. Risk reflects where the fill actually is.

v5 changes (retained):
    - iFVG (Inverted FVG) detection and re-touch signals
    - HMM regime hard gate (volatile only)
    - DVOL strategy-dependent direction filters
    - Realized Vol percentile boost for iFVG
    - R:R defaults: retest 3.0R, iFVG 2.0R
    - SL buffer 0.3 ATR (internal study)

WFO Results history:
    v5:  positive WFO result (see internal report)
    v4:  negative WFO result on both assets (see internal report)
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import logging
import os
import pickle

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal


# FVG direction constants
_DIR_BULL = 0
_DIR_BEAR = 1

_logger = logging.getLogger(__name__)

# Location of trained FVG momentum models (one per asset, 2-feature LR)
_FVG_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'FVG_Strategy', 'models',
)

# Matches FVG_Strategy/momentum_signal_model.py
_ML_VOLUME_ROLL = 20
_ML_DISPLACEMENT_LOOKBACK = 5


class FVGAdapter(StrategyAdapter):

    @property
    def name(self) -> str:
        return "FVG"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m", "1h"]

    # Study-proven constants (not optimizable)
    _FIXED_ATR_SL_BUFFER = 0.3
    _FIXED_RR_RETEST = 3.0
    _FIXED_RR_IFVG = 2.0
    _FIXED_HMM_CONF_BOOST = 0.15

    # Optional ML filter (live bot's MomentumSignalModel). Off until enable_ml_filter().
    # NOTE: the production models are fitted on the proprietary training window,
    # so enabling the filter on historical WFO is a counterfactual ("would the
    # live filter have helped on this data?"), not a clean OOS test.  Per-window
    # refit is the clean version; not implemented here.
    _ml_model = None
    _ml_scaler = None
    _ml_threshold: float = 0.55
    _ml_symbol: Optional[str] = None

    # Per-window-refit slots (Phase 2B).  When set via begin_window(), these
    # override the globally-loaded production model for one WFO fold.  This
    # gives an honest out-of-sample test of the ML filter: train on the IS
    # trades of the current window only, evaluate on OOS.
    _refit_mode: bool = False          # True while a per-window model is active
    _refit_model = None
    _refit_scaler = None
    _refit_threshold: float = 0.55
    _refit_min_trades: int = 10        # skip refit below this many IS trades

    def enable_ml_filter(self, symbol: str) -> bool:
        """Load the live bot's momentum LR model for `symbol` (e.g., "BTC").

        Returns True if the model loaded successfully, False otherwise.
        Once loaded, `generate_signals` applies a P(win) >= threshold gate
        on every candidate signal.
        """
        # Map our short symbols to FVG model filenames (underscore form).
        sym_map = {'BTC': 'BTC_USD', 'ETH': 'ETH_USD', 'NQ': 'NQ_USD'}
        full_sym = sym_map.get(symbol, symbol)
        path = os.path.join(_FVG_MODEL_DIR, f'momentum_{full_sym}.pkl')
        if not os.path.exists(path):
            _logger.warning(f"FVG ML model not found at {path} — filter disabled")
            self._ml_model = None
            return False
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            self._ml_model = obj['model']
            self._ml_scaler = obj['scaler']
            self._ml_threshold = float(obj.get('probability_threshold', 0.55))
            self._ml_symbol = symbol
            _logger.info(
                f"FVG ML filter loaded for {symbol}: threshold={self._ml_threshold}, "
                f"model={type(self._ml_model).__name__}"
            )
            return True
        except Exception as e:
            _logger.warning(f"Failed to load FVG ML model for {symbol}: {e}")
            self._ml_model = None
            return False

    def disable_ml_filter(self):
        self._ml_model = None
        self._ml_scaler = None
        self._ml_symbol = None

    def enable_per_window_refit(self, min_trades: int = 10):
        """Enable Phase 2B: refit a fresh 2-feature LR per WFO window.

        This eliminates the look-ahead bias of the live-model approach:
        instead of applying a globally-trained model to historical OOS
        windows (which saw those trades during training), we fit a fresh
        model on each window's IS trades only and evaluate on OOS.

        Caveats:
        - Per-window IS typically yields 10-50 trades, so each model is
          small-sample.  If fewer than `min_trades`, the refit is skipped
          for that window and no ML filter is applied.
        - Feature direction-alignment check still applies (see _ml_accept).
        """
        self._refit_mode = True
        self._refit_min_trades = int(min_trades)
        # Ensure the globally-loaded model (if any) doesn't interfere
        self._ml_model = None
        self._ml_scaler = None
        _logger.info(
            f"FVG per-window-refit ENABLED (min_trades={min_trades})."
        )

    def disable_per_window_refit(self):
        self._refit_mode = False
        self._refit_model = None
        self._refit_scaler = None

    @staticmethod
    def _compute_ml_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute volume_deviation and displacement_pct arrays over the full df.

        Matches FVG_Strategy/momentum_signal_model.py feature extraction exactly:
            volume_deviation = current_vol / rolling-mean-of-prior-20-bars
            displacement_pct = (close - close[-5]) / close[-5]   (signed)
        """
        vol = df['Volume'].values
        close = df['Close'].values
        n = len(df)
        # Rolling mean of PRIOR 20 bars (exclusive of current) to match live.
        vol_shift = pd.Series(vol).shift(1)
        vol_mean = vol_shift.rolling(_ML_VOLUME_ROLL, min_periods=_ML_VOLUME_ROLL).mean().values
        safe_mean = np.where((vol_mean > 0) & ~np.isnan(vol_mean), vol_mean, np.nan)
        vol_dev = vol / safe_mean
        # Replace NaN with 1.0 (live model's fallback when data insufficient)
        vol_dev = np.where(np.isnan(vol_dev), 1.0, vol_dev)
        # 5-bar signed displacement
        close_prev = pd.Series(close).shift(_ML_DISPLACEMENT_LOOKBACK).values
        disp = np.where(
            (close_prev > 0) & ~np.isnan(close_prev),
            (close - close_prev) / close_prev, 0.0,
        )
        return vol_dev, disp

    def _ml_accept(self, vol_dev: float, disp: float, direction: str) -> bool:
        """Return True if the signal passes the ML filter (or filter is off).

        Prefers the per-window refit model when active; falls back to the
        globally-loaded production model; returns True if neither is set.
        Applies direction-alignment check before the probability threshold.
        """
        # Pick which model applies: window-refit overrides global.
        if self._refit_mode:
            model = self._refit_model
            scaler = self._refit_scaler
            thresh = self._refit_threshold
            if model is None:
                # Window had too few IS trades to fit; let all signals through.
                return True
        elif self._ml_model is not None:
            model = self._ml_model
            scaler = self._ml_scaler
            thresh = self._ml_threshold
        else:
            return True
        # Direction alignment (matches momentum_signal_model._model_detect).
        if direction == 'LONG' and disp <= 0:
            return False
        if direction == 'SHORT' and disp >= 0:
            return False
        feat = np.array([[vol_dev, disp]])
        scaled = scaler.transform(feat)
        p_win = model.predict_proba(scaled)[0, 1]
        return bool(p_win >= thresh)

    # ────────────────────────────────────────────────────────────
    #  Per-window hooks (Phase 2B)
    # ────────────────────────────────────────────────────────────

    def begin_window(self, train_df, window_id, is_trades=None):
        """Fit a fresh 2-feature LR on the current window's IS trades.

        Features : (volume_deviation, displacement_pct) at each trade's
                   entry bar — identical to the live model's features.
        Label    : 1 if trade's r_multiple_after_costs > 0 else 0.

        No-op if per-window-refit is not enabled (default).
        """
        if not self._refit_mode:
            return
        # Drop any previous window's model first
        self._refit_model = None
        self._refit_scaler = None
        if is_trades is None or len(is_trades) < self._refit_min_trades:
            _logger.debug(
                f"[window {window_id}] refit skipped: "
                f"{0 if is_trades is None else len(is_trades)} IS trades "
                f"< {self._refit_min_trades} min"
            )
            return
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            _logger.warning("sklearn not available; per-window refit disabled")
            return

        vol_dev_arr, disp_arr = self._compute_ml_features(train_df)
        # Map each trade's entry_time back to a bar index in train_df.
        # (The IS trades were simulated on train_df, so indices align.)
        idx = train_df.index
        X, y = [], []
        for t in is_trades:
            try:
                bar = idx.get_loc(t.entry_time)
            except (KeyError, TypeError):
                continue
            if bar < 0 or bar >= len(vol_dev_arr):
                continue
            v = vol_dev_arr[bar]
            d = disp_arr[bar]
            if not np.isfinite(v) or not np.isfinite(d):
                continue
            X.append([v, d])
            y.append(1 if t.r_multiple_after_costs > 0 else 0)

        if len(X) < self._refit_min_trades:
            _logger.debug(
                f"[window {window_id}] refit skipped: only {len(X)} usable "
                f"feature rows after cleaning"
            )
            return
        # Need both classes present for LR.fit to work
        if len(set(y)) < 2:
            _logger.debug(
                f"[window {window_id}] refit skipped: IS trades all "
                f"{'winners' if y[0]==1 else 'losers'} — LR needs both classes"
            )
            return

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        try:
            scaler = StandardScaler().fit(X_arr)
            X_sc = scaler.transform(X_arr)
            model = LogisticRegression(solver='lbfgs', max_iter=500).fit(X_sc, y_arr)
        except Exception as e:
            _logger.warning(f"[window {window_id}] refit failed: {e}")
            return

        self._refit_model = model
        self._refit_scaler = scaler
        # Pick a threshold: use 0.55 (matches live default) unless that would
        # reject every trade.  Degenerate IS sets can produce always-low or
        # always-high probas; adjust threshold to the median predicted proba
        # in that case so we still filter something.
        try:
            probs = model.predict_proba(X_sc)[:, 1]
            base = 0.55
            if probs.max() < base:
                base = float(np.median(probs))
            self._refit_threshold = base
        except Exception:
            self._refit_threshold = 0.55
        _logger.debug(
            f"[window {window_id}] refit fit on {len(X)} trades "
            f"(win_rate={y_arr.mean():.1%}, threshold={self._refit_threshold:.3f})"
        )

    def end_window(self, window_id):
        """Drop the per-window model after OOS evaluation completes."""
        self._refit_model = None
        self._refit_scaler = None

    def get_param_space(self) -> List[ParamSpec]:
        """Parameter space — kept small for stable WFO (72 combos full grid)."""
        return [
            ParamSpec("min_gap_pct",     0.002,  0.001, 0.004, 0.001),     # 4 values
            ParamSpec("max_fvg_age",     50,     30,    70,    20, 'int'),  # 3 values
            ParamSpec("min_confidence",  0.40,   0.30,  0.50,  0.10),      # 3 values
            ParamSpec("regime_filter",   1,      0,     1,     1,  'int'),  # 2 values
        ]

    # ────────────────────────────────────────────────────────────
    #  HMM helpers (cached per DataFrame)
    # ────────────────────────────────────────────────────────────

    _hmm_cache_id: int = -1
    _hmm_cache_states: Optional[np.ndarray] = None

    @classmethod
    def _fit_hmm_states(cls, df: pd.DataFrame, warmup_pct: float) -> Optional[np.ndarray]:
        """Fit 2-state GaussianHMM on warmup data and forward-filter the rest."""
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

            X_full = features.copy()
            nan_rows = np.isnan(X_full).any(axis=1)
            X_full[nan_rows] = mu
            probs = hmm.forward_filter((X_full - mu) / std)
            states = np.argmax(probs, axis=1)

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
    def _build_htf_bars(df: pd.DataFrame, prefix: str) -> Optional[Tuple[np.ndarray, ...]]:
        """Extract unique HTF bars from forward-filled columns.

        Works for any prefix: 'HTF_1h_' or 'HTF_4h_'.
        Returns (htf_highs, htf_lows, htf_closes, bar_start_indices, bar_end_idx)
        or None if columns are missing.
        """
        needed = [f'{prefix}High', f'{prefix}Low', f'{prefix}Close', f'{prefix}Open']
        if not all(c in df.columns for c in needed):
            return None

        htf_close = df[f'{prefix}Close'].values
        htf_high = df[f'{prefix}High'].values
        htf_low = df[f'{prefix}Low'].values

        n = len(df)
        if n < 4:
            return None

        change_mask = np.zeros(n, dtype=bool)
        change_mask[0] = True
        change_mask[1:] = (htf_close[1:] != htf_close[:-1]) | (htf_high[1:] != htf_high[:-1])

        bar_start_indices = np.where(change_mask)[0]
        n_bars = len(bar_start_indices)
        if n_bars < 3:
            return None

        htf_highs = htf_high[bar_start_indices]
        htf_lows = htf_low[bar_start_indices]
        htf_closes = htf_close[bar_start_indices]

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
        htf_source: str = '1h',
    ) -> List[tuple]:
        """Detect FVGs on HTF bars and map activation to native-TF indices.

        Returns list of (activation_native_idx, direction, gap_high, gap_low,
                         vol_conf, gap_pct, htf_source).
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
                        act_idx = int(bar_end_idx[k])
                        mid_native = int(bar_start_indices[k - 1])
                        vm = vol_mean_20[mid_native] if mid_native < len(vol_mean_20) else np.nan
                        vol_conf = (
                            not np.isnan(vm) and vm > 0
                            and volumes[mid_native] > vm * 1.2
                        )
                        fvgs.append((act_idx, _DIR_BULL, gap_high, gap_low, vol_conf, gp, htf_source))

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
                        fvgs.append((act_idx, _DIR_BEAR, gap_high, gap_low, vol_conf, gp, htf_source))

        return fvgs

    # ────────────────────────────────────────────────────────────
    #  Main signal generation (v6)
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

        v6: Limit-order entry, HTF cascade, regime filter, iFVG buffer,
        entry-based R:R.
        """
        min_gap = params.get('min_gap_pct', 0.002)
        max_age = int(params.get('max_fvg_age', 50))
        min_conf = params.get('min_confidence', 0.40)
        regime_filter = bool(params.get('regime_filter', 1))
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

        has_dvol = 'DVOL' in df.columns
        dvol = df['DVOL'].values if has_dvol else None

        warmup_pct = params.get('hmm_warmup_pct', 0.30)
        hmm_states = self._fit_hmm_states(df, warmup_pct)
        has_hmm = hmm_states is not None

        has_rv = 'RV_Percentile' in df.columns
        rv_pctile = df['RV_Percentile'].values if has_rv else None

        vol_series = pd.Series(volumes)
        vol_mean_20 = vol_series.rolling(20, min_periods=10).mean().values

        # ML filter features (computed once per df; no-op if filter disabled).
        # Required for both the global model and the per-window refit mode.
        _ml_active = (self._ml_model is not None) or self._refit_mode
        if _ml_active:
            ml_vol_dev, ml_disp = self._compute_ml_features(df)
        else:
            ml_vol_dev = ml_disp = None

        # ── FVG Detection: HTF cascade (4h → 1h → native) ─────
        # FVG tuple format: (bar, dir, gh, gl, vol_conf, gap_pct, htf_source)
        htf_fvg_list: List[tuple] = []

        # Try 4h first (strongest institutional signal)
        htf_4h = self._build_htf_bars(df, 'HTF_4h_')
        if htf_4h is not None:
            htf_highs, htf_lows, htf_closes, bar_starts, bar_ends = htf_4h
            fvgs_4h = self._detect_htf_fvgs(
                htf_highs, htf_lows, htf_closes, bar_starts, bar_ends,
                min_gap * 0.5,  # 4h gaps are larger absolute, lower pct threshold
                vol_mean_20, volumes, htf_source='4h',
            )
            htf_fvg_list.extend(fvgs_4h)

        # Then 1h
        htf_1h = self._build_htf_bars(df, 'HTF_1h_')
        if htf_1h is not None:
            htf_highs, htf_lows, htf_closes, bar_starts, bar_ends = htf_1h
            fvgs_1h = self._detect_htf_fvgs(
                htf_highs, htf_lows, htf_closes, bar_starts, bar_ends,
                min_gap, vol_mean_20, volumes, htf_source='1h',
            )
            htf_fvg_list.extend(fvgs_1h)

        use_cross_tf = len(htf_fvg_list) > 0

        active_fvgs: deque = deque()
        mitigated_fvgs: deque = deque()

        if use_cross_tf:
            htf_fvg_list.sort(key=lambda x: x[0])
            htf_fvg_iter_idx = 0

        # Native-TF detection (fallback when no HTF columns)
        bull_fvg_mask = np.zeros(n, dtype=bool)
        bear_fvg_mask = np.zeros(n, dtype=bool)
        if not use_cross_tf and n >= 3:
            bull_fvg_mask[2:] = highs[:-2] < lows[2:]
            bear_fvg_mask[2:] = lows[:-2] > highs[2:]

        # ── Main loop ───────────────────────────────────────────
        signals: List[Signal] = []
        min_cooldown = 4
        last_sig_idx = -min_cooldown
        used_fvg_bars: set = set()
        used_ifvg_bars: set = set()

        loop_start = max(2, s - max_age)

        for i in range(loop_start, e):
            close_i = closes[i]

            # ── Register new FVGs ───────────────────────────────
            if use_cross_tf:
                while htf_fvg_iter_idx < len(htf_fvg_list):
                    fvg = htf_fvg_list[htf_fvg_iter_idx]
                    if fvg[0] <= i:
                        active_fvgs.append(fvg)
                        htf_fvg_iter_idx += 1
                    else:
                        break
            else:
                if bull_fvg_mask[i]:
                    gap_high = lows[i]
                    gap_low = highs[i - 2]
                    gap_size = gap_high - gap_low
                    mid_p = closes[i - 1]
                    if mid_p > 0 and gap_size > 0:
                        gp = gap_size / mid_p
                        if gp >= min_gap:
                            vm = vol_mean_20[i - 1]
                            vol_conf = not np.isnan(vm) and vm > 0 and volumes[i - 1] > vm * 1.2
                            active_fvgs.append((i, _DIR_BULL, gap_high, gap_low, vol_conf, gp, 'native'))

                if bear_fvg_mask[i]:
                    gap_high = lows[i - 2]
                    gap_low = highs[i]
                    gap_size = gap_high - gap_low
                    mid_p = closes[i - 1]
                    if mid_p > 0 and gap_size > 0:
                        gp = gap_size / mid_p
                        if gp >= min_gap:
                            vm = vol_mean_20[i - 1]
                            vol_conf = not np.isnan(vm) and vm > 0 and volumes[i - 1] > vm * 1.2
                            active_fvgs.append((i, _DIR_BEAR, gap_high, gap_low, vol_conf, gp, 'native'))

            # Pop expired
            while active_fvgs and (i - active_fvgs[0][0]) > max_age:
                active_fvgs.popleft()
            while mitigated_fvgs and (i - mitigated_fvgs[0][0]) > max_age:
                mitigated_fvgs.popleft()

            # ── Check for mitigation → iFVG (with buffer) ───────
            new_active: deque = deque()
            for fvg_tuple in active_fvgs:
                fvg_bar, fvg_dir, fvg_gh, fvg_gl, fvg_vc, fvg_gp, fvg_src = fvg_tuple
                gap_range = fvg_gh - fvg_gl
                mit_buffer = gap_range * 0.10  # v6: 10% buffer

                if fvg_dir == _DIR_BULL and close_i < (fvg_gl - mit_buffer):
                    mitigated_fvgs.append(
                        (i, _DIR_BEAR, fvg_gh, fvg_gl, fvg_vc, fvg_gp, fvg_src)
                    )
                elif fvg_dir == _DIR_BEAR and close_i > (fvg_gh + mit_buffer):
                    mitigated_fvgs.append(
                        (i, _DIR_BULL, fvg_gh, fvg_gl, fvg_vc, fvg_gp, fvg_src)
                    )
                else:
                    new_active.append(fvg_tuple)
            active_fvgs = new_active

            # Skip pre-scan bars
            if i < s:
                continue
            if i - last_sig_idx < min_cooldown:
                continue

            atr_val = atrs[i]
            if not (atr_val > 0) or np.isnan(atr_val):
                continue

            # ── IV regime ─────────────────────────────────────
            iv_regime = None
            if has_dvol:
                dv = dvol[i]
                if not np.isnan(dv):
                    if dv < 45:
                        iv_regime = 'LOW'
                    elif dv < 65:
                        iv_regime = 'MED'
                    else:
                        iv_regime = 'HIGH'

            # ── Displacement (soft) ───────────────────────────
            disp_lookback = 5
            if i >= disp_lookback:
                ref_close = closes[i - disp_lookback]
                price_change = (close_i - ref_close) / ref_close if ref_close > 0 else 0.0
            else:
                price_change = 0.0

            # ── Session filter (NQ only) ──────────────────────
            if not has_dvol:
                bar_time = df.index[i]
                if hasattr(bar_time, 'hour'):
                    if not (7 <= bar_time.hour < 21):
                        continue

            # ── Liquidity sweep detection ─────────────────────
            sweep_lookback = 20
            bull_sweep = True
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

            # ── HMM hard gate ─────────────────────────────────
            if has_hmm and hmm_states[i] == 0:
                continue

            # ── EMA regime state (for direction filter) ────────
            ema_bull = has_ema and ema50[i] > ema200[i]
            ema_bear = has_ema and ema50[i] < ema200[i]

            # ═══════════════════════════════════════════════════
            #  RETEST: Check active FVGs for RETRACEMENT entry
            # ═══════════════════════════════════════════════════
            best_signal: Optional[Signal] = None
            best_conf = -1.0
            best_fvg_bar = -1

            for fvg_tuple in active_fvgs:
                fvg_bar, fvg_dir, fvg_gh, fvg_gl, fvg_vc, fvg_gp, fvg_src = fvg_tuple

                age = i - fvg_bar
                if age < 2:
                    continue
                if fvg_bar in used_fvg_bars:
                    continue

                gap_range = fvg_gh - fvg_gl
                if gap_range <= 0:
                    continue

                # v6 IMPROVEMENT 1: RETRACEMENT entry (not just zone touch)
                # Price must retrace INTO the FVG zone.
                fvg_mid = (fvg_gh + fvg_gl) / 2.0

                if fvg_dir == _DIR_BULL:
                    # Bullish: need price to retrace DOWN into the gap
                    if low_i > fvg_gh:
                        continue  # bar never reached the gap
                    direction = 'LONG'
                    # Entry = where price actually touched, clamped to gap zone
                    touch_price = max(fvg_gl, min(fvg_mid, low_i))
                else:
                    # Bearish: need price to retrace UP into the gap
                    if high_i < fvg_gl:
                        continue  # bar never reached the gap
                    direction = 'SHORT'
                    touch_price = min(fvg_gh, max(fvg_mid, high_i))

                # v6 IMPROVEMENT 4: EMA regime direction filter
                if regime_filter and has_ema:
                    if ema_bear and direction == 'LONG':
                        continue
                    if ema_bull and direction == 'SHORT':
                        continue

                # ── Retest IV regime direction gate ──────────
                if iv_regime == 'LOW' and direction == 'SHORT':
                    continue
                if iv_regime == 'HIGH' and direction == 'LONG':
                    continue

                # ── Confidence scoring (9 factors) ───────────
                confidence = min(fvg_gp / 0.005, 1.0) * 0.20  # gap size

                if fvg_vc:
                    confidence += 0.15  # volume

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

                confidence += min(abs(price_change) / 0.015, 1.0) * 0.15  # displacement

                if direction == 'LONG' and bull_sweep:
                    confidence += 0.15
                elif direction == 'SHORT' and bear_sweep:
                    confidence += 0.15

                # v6 IMPROVEMENT 2: HTF confidence boost
                if fvg_src == '4h':
                    confidence += 0.15
                elif fvg_src == '1h':
                    confidence += 0.05

                if confidence < min_conf:
                    continue

                # v6 IMPROVEMENT 5: Entry/SL/TP from actual touch price
                entry = touch_price
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
                    ema_aligned = ema_bull if direction == 'LONG' else ema_bear
                    struct_aligned = False
                    if has_structure:
                        sb = struct_bias[i]
                        struct_aligned = (direction == 'LONG' and sb > 0) or (direction == 'SHORT' and sb < 0)

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
                        metadata={
                            'signal_type': 'RETEST',
                            'hmm_state': hmm_st,
                            'htf_source': fvg_src,
                            'touch_price': float(touch_price),
                            'fvg_mid': float(fvg_mid),
                        },
                    )
                    best_fvg_bar = fvg_bar

            if best_signal is not None:
                # ML gate (global model OR per-window refit)
                if _ml_active and ml_vol_dev is not None:
                    if not self._ml_accept(
                        float(ml_vol_dev[i]), float(ml_disp[i]),
                        best_signal.direction,
                    ):
                        best_signal = None
                if best_signal is not None:
                    signals.append(best_signal)
                    used_fvg_bars.add(best_fvg_bar)
                    last_sig_idx = i
                    continue

            # ═══════════════════════════════════════════════════
            #  iFVG: Check mitigated FVGs for re-touch entry
            # ═══════════════════════════════════════════════════
            best_ifvg: Optional[Signal] = None
            best_ifvg_conf = -1.0
            best_ifvg_bar = -1

            for ifvg_tuple in mitigated_fvgs:
                mit_bar, inv_dir, ifvg_gh, ifvg_gl, ifvg_vc, ifvg_gp, ifvg_src = ifvg_tuple

                age = i - mit_bar
                if age < 2:
                    continue
                if mit_bar in used_ifvg_bars:
                    continue

                gap_range = ifvg_gh - ifvg_gl
                if gap_range <= 0:
                    continue

                # v6: Retracement entry for iFVG too
                ifvg_mid = (ifvg_gh + ifvg_gl) / 2.0

                if inv_dir == _DIR_BULL:
                    if low_i > ifvg_gh:
                        continue
                    direction = 'LONG'
                    touch_price = max(ifvg_gl, min(ifvg_mid, low_i))
                else:
                    if high_i < ifvg_gl:
                        continue
                    direction = 'SHORT'
                    touch_price = min(ifvg_gh, max(ifvg_mid, high_i))

                # v6: EMA regime filter for iFVG
                if regime_filter and has_ema:
                    if ema_bear and direction == 'LONG':
                        continue
                    if ema_bull and direction == 'SHORT':
                        continue

                # iFVG DVOL gates (reversed)
                if iv_regime == 'LOW' and direction == 'LONG':
                    continue

                # RV hard gate for iFVG
                if has_rv and rv_pctile is not None:
                    rv_val = rv_pctile[i]
                    if not np.isnan(rv_val) and rv_val < 0.50:
                        continue

                # Confidence scoring
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

                confidence += min(abs(price_change) / 0.015, 1.0) * 0.15

                if direction == 'LONG' and bull_sweep:
                    confidence += 0.15
                elif direction == 'SHORT' and bear_sweep:
                    confidence += 0.15

                # RV boost
                if has_rv and rv_pctile is not None and not np.isnan(rv_pctile[i]):
                    if rv_pctile[i] >= 0.50:
                        confidence += 0.10

                # HTF boost
                if ifvg_src == '4h':
                    confidence += 0.15
                elif ifvg_src == '1h':
                    confidence += 0.05

                if confidence < min_conf:
                    continue

                # v6: Entry from touch price
                entry = touch_price
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
                    ema_aligned = ema_bull if direction == 'LONG' else ema_bear
                    struct_aligned = False
                    if has_structure:
                        sb = struct_bias[i]
                        struct_aligned = (direction == 'LONG' and sb > 0) or (direction == 'SHORT' and sb < 0)

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
                            'htf_source': ifvg_src,
                            'touch_price': float(touch_price),
                        },
                    )
                    best_ifvg_bar = mit_bar

            if best_ifvg is not None:
                # ML gate (global model OR per-window refit)
                if _ml_active and ml_vol_dev is not None:
                    if not self._ml_accept(
                        float(ml_vol_dev[i]), float(ml_disp[i]),
                        best_ifvg.direction,
                    ):
                        best_ifvg = None
            if best_ifvg is not None:
                signals.append(best_ifvg)
                used_ifvg_bars.add(best_ifvg_bar)
                last_sig_idx = i

        return signals
