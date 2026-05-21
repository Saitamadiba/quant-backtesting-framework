"""
Vol-Edge ETH adapter for the WFO engine.

Implements two stateful, vol-derived strategies in a single adapter:

    Edge B — IV-Spike Spot Reversal (long-only)
        Trigger (end-of-UTC-day):
            DVOL_change_1d in top (1 - edge_b_spike_pctile) of trailing 252d
            AND ETH log_return that day ≤ -edge_b_z_threshold × σ_30d
        Entry: at the OPEN of the first bar of the next UTC day.
        SL:    edge_b_sl_atr_mult × ATR(14d) below entry
        TP:    edge_b_rr_target × risk above entry
        Trail: edge_b_trail_atr_mult × ATR (forwarded to engine TradeSimulator
               via signal metadata).
        Hold:  edge_b_hold_days × bars_per_day  (engine max_bars cap).

    Edge C — Short-Vol Synthetic (long + short)
        Active when:
            DVOL < edge_c_iv_low_threshold  AND
            VRP  > edge_c_min_vrp           (DVOL > RV30)
        Trigger (intraday):
            |close_t − session_open| ≥ edge_c_trigger_frac × expected_daily_move
        Entry: at close_t (next-bar fills are simulated by the engine; no
               look-ahead because session_open and EM are known at session
               start).
        Side:  SHORT if move>0, LONG if move<0 (mean-reversion).
        SL:    edge_c_sl_em_mult × EM beyond SESSION OPEN (fixed; not entry).
        TP:    retrace edge_c_tp_retrace_frac of the move toward session open.
        Cap:   edge_c_max_per_day fades, ≤ edge_c_max_hold_hours hold time.

Anti-leakage discipline:
    All daily aggregates (DVOL, RV30, return, ATR daily, EM) are forward-filled
    onto intraday bars with a one-day shift so that the value visible at
    bar t depends only on data with timestamps strictly < t (the day-D
    aggregate becomes visible at the first bar of day D+1).  Edge B's
    entry index is therefore the first bar of the day AFTER the trigger.
    Edge C reads only the current-day session-open and the day's EM, both
    stamped at the first bar of the day.  No future bar leaks into either.

Param space exposes:
    - risk_reward_ratio (Edge B TP)
    - sl_atr_mult       (Edge B SL)
    - trail_atr_mult    (Edge B trail)
    - hold_days         (Edge B time-stop)
    - edge_b_spike_pctile, edge_b_z_threshold (Edge B trigger sensitivity)
    - edge_c_trigger_frac, edge_c_sl_em_mult, edge_c_tp_retrace_frac,
      edge_c_min_rr, edge_c_max_per_day  (Edge C tuning)
    - edge_c_iv_low_threshold (regime gate)

Position-sizing (% of equity per trade) is a downstream concern handled
by Monte Carlo on the OOS R-multiple series — the WFO engine works in
R-space and is sizing-invariant.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta

import numpy as np
import pandas as pd

from .base_adapter import StrategyAdapter, ParamSpec, Signal


# Bars per UTC day for each supported timeframe.
_BARS_PER_DAY = {
    '5m': 288, '15m': 96, '30m': 48,
    '1h': 24, '2h': 12, '4h': 6, '1d': 1,
}

# Bars per hour (used for Edge C max-hold conversion).
_BARS_PER_HOUR = {
    '5m': 12, '15m': 4, '30m': 2,
    '1h': 1, '2h': 0.5, '4h': 0.25, '1d': 1/24,
}


class VolEdgeAdapter(StrategyAdapter):

    # --------------------------------------------------------------
    # Identity
    # --------------------------------------------------------------

    @property
    def name(self) -> str:
        return "VolEdge"

    @property
    def default_timeframes(self) -> List[str]:
        return ["15m", "1h", "4h"]

    # --------------------------------------------------------------
    # Param space
    # --------------------------------------------------------------

    def get_param_space(self) -> List[ParamSpec]:
        """Tunable param space.  Sprint 1+2 design decisions (partial-take,
        SHORT-only Edge C, hour filter, regime gate, momentum kill, single-
        trade R cap, daily/weekly R caps) are NOT tuned — they are fixed
        engineering choices grounded in the OOS forensics, and tuning them
        adds dimensionality without information.  Only the geometry knobs
        and signal-trigger knobs go to the grid."""
        return [
            # Edge B geometry
            ParamSpec("risk_reward_ratio", 1.0, 0.5,  1.75, 0.25),
            ParamSpec("sl_atr_mult",       1.0, 0.75, 1.75, 0.25),
            ParamSpec("trail_atr_mult",    1.0, 0.5,  2.0,  0.5),
            ParamSpec("hold_days",         5,   3,    8,    1, 'int'),
            ParamSpec("edge_b_spike_pctile", 0.95, 0.85, 0.97, 0.05),
            ParamSpec("edge_b_z_threshold",  2.0, 1.5, 2.5, 0.5),
            # Edge C geometry
            ParamSpec("edge_c_trigger_frac",     0.75, 0.50, 1.00, 0.25),
            ParamSpec("edge_c_sl_em_mult",       1.5,  1.0,  2.0,  0.5),
            ParamSpec("edge_c_tp_retrace_frac",  0.5,  0.25, 1.0,  0.25),
            ParamSpec("edge_c_min_rr",           0.4,  0.3,  0.7,  0.1),
            ParamSpec("edge_c_max_per_day",      2,    1,    3,    1, 'int'),
            ParamSpec("edge_c_iv_low_threshold", 45.0, 40.0, 55.0, 5.0),
        ]

    # --------------------------------------------------------------
    # Signal generation — pure function on a dataframe slice
    # --------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        n = len(df)
        if scan_end_idx <= scan_start_idx or n == 0:
            return []
        tf = self._infer_timeframe(df)
        bars_per_day = _BARS_PER_DAY.get(tf, 24)

        # Decode params
        rr_target  = float(params.get("risk_reward_ratio", 2.0))
        sl_atr     = float(params.get("sl_atr_mult", 2.0))
        trail_atr  = float(params.get("trail_atr_mult", 2.0))
        hold_days  = int(params.get("hold_days", 5))
        b_spk_pct  = float(params.get("edge_b_spike_pctile", 0.95))
        b_z_thr    = float(params.get("edge_b_z_threshold", 2.0))

        c_trigger  = float(params.get("edge_c_trigger_frac", 0.75))
        c_sl_em    = float(params.get("edge_c_sl_em_mult", 1.5))
        c_tp_frac  = float(params.get("edge_c_tp_retrace_frac", 0.5))
        c_min_rr   = float(params.get("edge_c_min_rr", 0.4))
        c_max_day  = int(params.get("edge_c_max_per_day", 2))
        iv_low     = float(params.get("edge_c_iv_low_threshold", 45.0))

        # Sprint 1+2 controls.  Defaults are chosen to PRESERVE sample size
        # while still implementing the highest-confidence optimizations:
        #   #1 partial-take ON         — clear empirical lift, low overfit risk
        #   #5 SHORT-only Edge C       — strong directional asymmetry in OOS data
        #   #6 single-trade R cap      — robustness, no EV cost on existing data
        #   #2/3/8 regime/hour/mom     — set to LOOSE defaults (most permissive
        #     end of the recommended range) so WFO can tighten them per window
        #     if appropriate; tightening too aggressively a-priori starves the
        #     OOS sample and prevents WFO from learning anything.
        partial_tp_r = float(params.get("partial_tp_r", 0.5))    # #1 ON
        c_long_ok    = bool(int(params.get("edge_c_long_enabled", 0)))  # #5 ON
        c_max_vr     = float(params.get("edge_c_max_var_ratio", 1.10))  # #2 LOOSE
        b_max_vr     = float(params.get("edge_b_max_var_ratio", 1.20))  # #2 LOOSE
        c_hour_filter = bool(int(params.get("edge_c_hour_filter", 0)))  # #3 OFF default
        momentum_kill = float(params.get("momentum_kill_pct", 0.05))    # #8 LOOSE

        # Pre-compute causal daily aggregates and intraday context.
        ctx = self._build_context(df, tf, iv_low_threshold=iv_low)

        opens  = df['Open'].values
        highs  = df['High'].values
        lows   = df['Low'].values
        closes = df['Close'].values
        atrs   = df['ATR'].values if 'ATR' in df.columns else np.zeros(n)

        signals: List[Signal] = []

        # ─── Edge B: emit on first bar of each UTC day if previous day
        #     met spike + capitulation conditions ────────────────────
        first_bars = ctx['first_bar_of_day']        # array of bar indices, one per UTC day
        spike_pct  = ctx['dvol_spike_pctile_prev']  # one entry per UTC day
        ret_z      = ctx['return_z30_prev']
        atr_daily  = ctx['atr_daily_prev']

        max_bars_b = hold_days * bars_per_day

        var_ratio_per_day = ctx['variance_ratio_60d_prev']

        for i, fb in enumerate(first_bars):
            if fb < scan_start_idx or fb >= scan_end_idx:
                continue
            if not np.isfinite(spike_pct[i]) or not np.isfinite(ret_z[i]):
                continue
            if spike_pct[i] < b_spk_pct:        # not in top tail
                continue
            # Spike must be UP (positive change) — encoded in ctx
            if ctx['dvol_change_sign_prev'][i] <= 0:
                continue
            if ret_z[i] > -b_z_thr:             # not capitulation
                continue
            atr_d = atr_daily[i]
            if not np.isfinite(atr_d) or atr_d <= 0:
                continue

            # Sprint 2 #2: regime gate for Edge B (more permissive than C).
            # Block if 60d variance ratio above b_max_vr (strongly trending).
            vr = var_ratio_per_day[i]
            if np.isfinite(vr) and vr > b_max_vr:
                continue

            entry = opens[fb]
            risk  = atr_d * sl_atr
            sl    = entry - risk
            # Sprint 1 #1: tp1 at +partial_tp_r × risk, tp2 at +rr_target × risk.
            # Engine moves SL→BE on tp1 hit; we post-process the realised R
            # in execute_signals to split the position 50/50.  If
            # partial_tp_r ≤ 0, fall back to single-target (tp1 = tp2).
            tp2 = entry + risk * rr_target
            if partial_tp_r > 0:
                tp1 = min(entry + risk * partial_tp_r, tp2)
            else:
                tp1 = tp2

            signals.append(Signal(
                idx=fb, time=df.index[fb], direction='LONG',
                entry_price=entry, stop_loss=sl,
                take_profit_1=tp1, take_profit_2=tp2,
                risk=risk,
                confidence=float(spike_pct[i]),
                bias='COUNTER',
                atr=atrs[fb] if fb < len(atrs) else atr_d,
                metadata={
                    'edge': 'B',
                    'trail_atr_mult': trail_atr,
                    'max_bars_override': max_bars_b,
                    'dvol_change_pctile': float(spike_pct[i]),
                    'return_z30': float(ret_z[i]),
                    'partial_tp_r': partial_tp_r,
                    'rr_target': rr_target,
                    'variance_ratio_60d': float(vr) if np.isfinite(vr) else None,
                },
            ))

        # ─── Edge C: per-bar fades while regime active ───────────────
        sess_open = ctx['session_open']
        em_arr    = ctx['expected_daily_move']
        regime_on = ctx['edge_c_regime_active']
        day_idx   = ctx['day_index']                 # one int per bar

        bars_per_hour = _BARS_PER_HOUR.get(tf, 1)
        max_bars_c = max(1, int(round(6 * bars_per_hour)))   # 6h cap

        # Sprint 2 #3 — only-on-these-UTC-hours filter.  At 4h, bar timestamps
        # land on {0,4,8,12,16,20}; we keep 0 and 16 only (NY-close mean-rev
        # window + UTC-day open).  Disabled at non-4h timeframes since those
        # are dropped anyway by Sprint 1 #4.
        ALLOWED_HOURS_4H = {0, 16}
        bar_hours = df.index.hour if isinstance(df.index, pd.DatetimeIndex) else None

        # Variance ratio per bar (yesterday-stamped, ffilled across the day).
        var_ratio_bar = ctx['variance_ratio_60d_per_bar']

        # Track per-day fade count within scan range (without leaking
        # across windows: counter resets at each new UTC day).
        cur_day = -1
        cur_count = 0
        for i in range(max(scan_start_idx, 1), scan_end_idx):
            if not regime_on[i]:
                continue
            di = day_idx[i]
            if di != cur_day:
                cur_day = di
                cur_count = 0
            if cur_count >= c_max_day:
                continue
            so = sess_open[i]
            em = em_arr[i]
            if not np.isfinite(so) or not np.isfinite(em) or em <= 0:
                continue

            # Sprint 2 #2 — regime gate: skip if not mean-reverting enough.
            vr = var_ratio_bar[i]
            if np.isfinite(vr) and vr > c_max_vr:
                continue

            # Sprint 2 #3 — hour filter (4h timeframe only).
            if c_hour_filter and tf == '4h' and bar_hours is not None:
                if int(bar_hours[i]) not in ALLOWED_HOURS_4H:
                    continue

            move = closes[i] - so
            thresh = em * c_trigger
            if abs(move) < thresh:
                continue

            # Don't open near end of UTC day (no time for retrace)
            if (i == n - 1) or (i + max_bars_c > n - 1):
                continue
            # Bars remaining today
            bars_left_today = bars_per_day - (i % bars_per_day) - 1
            if bars_left_today < 2:
                continue

            side = 'SHORT' if move > 0 else 'LONG'

            # Sprint 1 #5 — SHORT-only Edge C unless explicitly enabled.
            if side == 'LONG' and not c_long_ok:
                continue

            # Sprint 2 #8 — momentum kill-switch.  Same-direction 4-bar ROC
            # exceeding momentum_kill is a sign the mean-reversion is not
            # going to happen on this bar; bail out.
            if i >= 4:
                roc4 = (closes[i] - closes[i-4]) / max(closes[i-4], 1e-9)
                if side == 'SHORT' and roc4 > momentum_kill:
                    continue
                if side == 'LONG' and roc4 < -momentum_kill:
                    continue

            entry = closes[i]

            if side == 'LONG':
                sl_anchor = so - em * c_sl_em
                sl = min(sl_anchor, entry - em * 0.25)
                tp = so - abs(move) * (1.0 - c_tp_frac)
            else:
                sl_anchor = so + em * c_sl_em
                sl = max(sl_anchor, entry + em * 0.25)
                tp = so + abs(move) * (1.0 - c_tp_frac)

            risk = abs(entry - sl)
            payoff = abs(tp - entry)
            if risk <= 0:
                continue
            rr = payoff / risk
            if rr < c_min_rr:
                continue

            # Sprint 1 #1 — partial-TP at +partial_tp_r × risk; second half
            # rides to the original retrace target.  TP1 must be < TP2 in
            # price terms; if not feasible (very tight tp), fall back to
            # tp1==tp2 (engine treats this as a single-target trade).
            tp2_price = tp
            if partial_tp_r > 0:
                if side == 'LONG':
                    tp1_price = min(entry + risk * partial_tp_r, tp2_price)
                else:
                    tp1_price = max(entry - risk * partial_tp_r, tp2_price)
            else:
                tp1_price = tp2_price

            cap_bars = min(max_bars_c, bars_left_today)

            signals.append(Signal(
                idx=i, time=df.index[i], direction=side,
                entry_price=entry, stop_loss=sl,
                take_profit_1=tp1_price, take_profit_2=tp2_price,
                risk=risk,
                confidence=min(1.0, abs(move) / em),
                bias='COUNTER',
                atr=atrs[i] if i < len(atrs) else 0.0,
                metadata={
                    'edge': 'C',
                    'trail_atr_mult': 0.0,
                    'max_bars_override': cap_bars,
                    'expected_daily_move': float(em),
                    'dvol': float(ctx['dvol_today_prev'][di]) if 0 <= di < len(ctx['dvol_today_prev']) else float('nan'),
                    'vrp': float(ctx['vrp_prev'][di]) if 0 <= di < len(ctx['vrp_prev']) else float('nan'),
                    'rr': float(rr),
                    'partial_tp_r': partial_tp_r,
                    'rr_target': float(rr),  # for post-processing
                    'variance_ratio_60d': float(vr) if np.isfinite(vr) else None,
                },
            ))
            cur_count += 1

        # Sort by idx so the engine sees them in time order
        signals.sort(key=lambda s: s.idx)
        return signals

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------

    @staticmethod
    def _infer_timeframe(df: pd.DataFrame) -> str:
        """Best-effort timeframe inference from the index spacing."""
        if len(df) < 2:
            return '1h'
        diffs = pd.Series(df.index[1:].asi8 - df.index[:-1].asi8)
        median_ns = diffs.median()
        # Convert ns → seconds
        secs = median_ns / 1e9
        if   secs <= 5*60 + 5: return '5m'
        elif secs <= 15*60 + 5: return '15m'
        elif secs <= 30*60 + 5: return '30m'
        elif secs <= 60*60 + 5: return '1h'
        elif secs <= 2*3600 + 60: return '2h'
        elif secs <= 4*3600 + 120: return '4h'
        return '1d'

    @staticmethod
    def _build_context(df: pd.DataFrame, tf: str,
                       iv_low_threshold: float) -> Dict[str, np.ndarray]:
        """
        Pre-compute every per-bar / per-day quantity Edge B and Edge C need.

        ALL series are constructed so that any value visible at intraday bar t
        depends only on bars strictly before t (or, for "today's session
        open", only on the very first open of the current day, which is
        observable at t).
        """
        n = len(df)
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is None:
            # Ensure UTC interpretation for daily grouping.
            idx_utc = idx.tz_localize('UTC')
        else:
            idx_utc = idx

        day_keys = pd.to_datetime(idx_utc.date) if hasattr(idx_utc, 'date') else pd.to_datetime(idx).normalize()
        day_series = pd.Series(day_keys, index=df.index)
        unique_days = day_series.drop_duplicates().sort_values().values
        day_to_pos = {d: i for i, d in enumerate(unique_days)}

        # Map each bar to a day position (0..len(unique_days)-1)
        day_index = day_series.map(day_to_pos).astype(int).values

        # ── Daily DVOL closes (last DVOL within each UTC day) ─────────
        if 'DVOL' in df.columns:
            dvol_intraday = df['DVOL'].values.astype(float)
        else:
            dvol_intraday = np.full(n, np.nan)

        dvol_daily = pd.Series(dvol_intraday, index=df.index).groupby(day_series).last().reindex(unique_days)
        dvol_today = dvol_daily.values
        dvol_yday  = pd.Series(dvol_today).shift(1).values
        dvol_chg_1d = dvol_today - dvol_yday

        # 252-day rolling abs-percentile of DVOL change.  Uses ONLY past
        # 252 days as of the close of day D — does not peek at day D+1.
        dvol_chg_series = pd.Series(np.abs(dvol_chg_1d))
        spike_pct = dvol_chg_series.rolling(252, min_periods=30).apply(
            lambda w: (w[:-1] <= w.iloc[-1]).mean() if len(w) > 1 else np.nan,
            raw=False,
        ).values

        # Sign of dvol change (so Edge B can require positive spikes)
        dvol_change_sign = np.sign(dvol_chg_1d)
        dvol_change_sign[np.isnan(dvol_chg_1d)] = 0

        # ── Daily ETH log-returns and 30d std ─────────────────────────
        daily_close = df['Close'].groupby(day_series).last().reindex(unique_days)
        daily_logret = np.log(daily_close / daily_close.shift(1))
        rolling_std30 = daily_logret.rolling(30, min_periods=10).std(ddof=1)
        ret_z = (daily_logret / rolling_std30).values

        # ── Daily ATR (range-based, Wilder 14 on daily) ───────────────
        daily_high = df['High'].groupby(day_series).max().reindex(unique_days)
        daily_low  = df['Low' ].groupby(day_series).min().reindex(unique_days)
        prev_close = daily_close.shift(1)
        tr_daily = pd.concat([
            (daily_high - daily_low).abs(),
            (daily_high - prev_close).abs(),
            (daily_low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_daily = tr_daily.ewm(alpha=1.0/14, adjust=False).mean().values

        # ── Realised vol 30d annualised (% of price), VRP = DVOL - RV ──
        rv30 = (rolling_std30 * np.sqrt(365) * 100).values
        vrp = dvol_today - rv30

        # ── Expected daily move = DVOL/100 / sqrt(365) × close ─────────
        em_daily = (dvol_today / 100.0) / math.sqrt(365) * daily_close.values

        # ── 60d rolling variance ratio: var(5d_returns) / (5 × var(1d_returns))
        # < 1.0  → mean-reverting; > 1.0 → trending.  Causal: each day's value
        # uses only that day's data and prior 60d.  Sprint-2 #2 input.
        ret_5d = daily_logret.rolling(5).sum()
        var_1d = daily_logret.rolling(60, min_periods=20).var(ddof=1)
        var_5d = ret_5d.rolling(60, min_periods=20).var(ddof=1)
        variance_ratio_60d = (var_5d / (5.0 * var_1d)).values

        # ── Causality shift: Edge B uses YESTERDAY'S aggregates as of
        # today.  Per-day spikes/zs/atr come from index i-1 (or NaN at i=0).
        spike_pct_prev   = pd.Series(spike_pct).shift(1).values
        ret_z_prev       = pd.Series(ret_z).shift(1).values
        dvol_chg_sign_prev = pd.Series(dvol_change_sign).shift(1).values
        atr_daily_prev   = pd.Series(atr_daily).shift(1).values
        dvol_today_prev  = pd.Series(dvol_today).shift(1).values
        vrp_prev         = pd.Series(vrp).shift(1).values
        em_daily_prev    = pd.Series(em_daily).shift(1).values
        variance_ratio_prev = pd.Series(variance_ratio_60d).shift(1).values

        # ── First bar of each day (= where Edge B fires) ───────────────
        first_bar_mask = (day_series != day_series.shift(1)).values
        first_bar_of_day = np.where(first_bar_mask)[0]
        # Make first_bar_of_day length == len(unique_days). Safety check:
        if len(first_bar_of_day) != len(unique_days):
            # Pad/trim defensively
            first_bar_of_day = first_bar_of_day[:len(unique_days)]

        # ── Per-bar session open (first OPEN of current UTC day) ───────
        session_open = pd.Series(df['Open'].values, index=df.index).groupby(day_series).transform('first').values

        # ── Per-bar Expected Daily Move (yesterday-stamped EM ffilled) ─
        em_per_bar = pd.Series(em_daily_prev, index=unique_days).reindex(day_series.values).values

        # ── Edge C regime gate (active when DVOL_yday < threshold AND
        #    VRP_yday > 0).  Uses *yesterday's* DVOL/VRP so the bar can
        #    legitimately observe it.
        regime_active_per_day = np.where(
            np.isnan(dvol_today_prev) | np.isnan(vrp_prev),
            False,
            (dvol_today_prev < iv_low_threshold) & (vrp_prev > 0.0),
        )
        regime_active_per_bar = pd.Series(
            regime_active_per_day, index=unique_days,
        ).reindex(day_series.values).values.astype(bool)

        # Variance ratio per bar (yesterday-stamped, ffilled across the day).
        var_ratio_per_bar = pd.Series(
            variance_ratio_prev, index=unique_days,
        ).reindex(day_series.values).values

        return {
            # Per-day
            'first_bar_of_day':         first_bar_of_day,
            'dvol_today':               dvol_today,
            'dvol_today_prev':          dvol_today_prev,
            'dvol_change_sign_prev':    dvol_chg_sign_prev,
            'dvol_spike_pctile_prev':   spike_pct_prev,
            'return_z30_prev':          ret_z_prev,
            'atr_daily_prev':           atr_daily_prev,
            'vrp_prev':                 vrp_prev,
            'variance_ratio_60d_prev':  variance_ratio_prev,
            # Per-bar
            'day_index':                day_index,
            'session_open':             session_open,
            'expected_daily_move':      em_per_bar,
            'edge_c_regime_active':     regime_active_per_bar,
            'variance_ratio_60d_per_bar': var_ratio_per_bar,
        }

    # --------------------------------------------------------------
    # We delegate trade simulation to the engine's TradeSimulator by
    # NOT overriding execute_signals().  The engine reads
    # signal['metadata']['trail_atr_mult'] and (if we monkey-patch
    # max_bars) max_bars_override.  Provide a tiny override that
    # respects each signal's per-edge max_bars cap.
    # --------------------------------------------------------------

    def execute_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
        costs: Any,
        max_bars: int = 168,
        window_id: int = 0,
        is_oos: bool = True,
        regime: str = 'unknown',
    ) -> Optional[List[Any]]:
        """
        Generate signals and route through the engine TradeSimulator with
        per-signal max_bars override (Edge B uses hold_days × bars_per_day,
        Edge C uses ≤6h or end-of-day, whichever is shorter).
        """
        from ..wfo_engine import TradeSimulator, TradeResult   # local import

        signals = self.generate_signals(df, params, scan_start_idx, scan_end_idx)
        if not signals:
            return []

        highs  = df['High'].values
        lows   = df['Low'].values
        closes = df['Close'].values
        atrs   = df['ATR'].values if 'ATR' in df.columns else None

        # ── Risk overlay constants (Sprint 1+2 mandatory caps) ──────
        MAX_SINGLE_TRADE_R = float(params.get('max_single_trade_r', 3.0))   # #6
        DAILY_R_CAP        = float(params.get('daily_r_cap',        -3.0))  # overlay
        WEEKLY_R_CAP       = float(params.get('weekly_r_cap',       -7.0))  # overlay

        # Sort by entry index so risk-overlay bookkeeping is chronological.
        signals = sorted(signals, key=lambda s: s.idx)

        results: List[TradeResult] = []
        cur_day_key  = None
        cur_week_key = None
        daily_r  = 0.0
        weekly_r = 0.0

        for sig in signals:
            t = pd.Timestamp(sig.time)
            day_key  = t.date()
            week_key = (t.isocalendar()[0], t.isocalendar()[1])

            if day_key != cur_day_key:
                cur_day_key = day_key
                daily_r = 0.0
            if week_key != cur_week_key:
                cur_week_key = week_key
                weekly_r = 0.0

            # Risk overlay kill-switches: skip the trade once tripped.
            if daily_r  <= DAILY_R_CAP:  continue
            if weekly_r <= WEEKLY_R_CAP: continue

            sd = sig.to_dict()
            sd['metadata'] = sig.metadata
            cap = int(sig.metadata.get('max_bars_override', max_bars))
            mb = max(1, min(max_bars, cap))
            tr = TradeSimulator.simulate(
                sd, df, costs, max_bars=mb, window_id=window_id,
                is_oos=is_oos, regime=regime,
                _highs=highs, _lows=lows, _closes=closes, _atrs=atrs,
            )
            if tr is None:
                continue

            # ── Sprint 1 #1: partial-take post-processing.  TP1 was set
            # at +partial_tp_r × R; engine moves SL→BE on tp1 hit.  We
            # split the position 50/50 in P&L space:
            #   if mfe < partial_tp_r:  realised = engine_r       (full SL)
            #   else:                   realised = 0.5*partial_tp_r + 0.5*engine_r
            partial_tp_r = float(sig.metadata.get('partial_tp_r', 0.0))
            engine_r = tr.r_multiple_after_costs
            if partial_tp_r > 0.0 and tr.mfe >= partial_tp_r:
                realised_r = 0.5 * partial_tp_r + 0.5 * engine_r
            else:
                realised_r = engine_r

            # ── Sprint 1 #6: clip individual trade R at +MAX_SINGLE_TRADE_R
            realised_r = float(min(realised_r, MAX_SINGLE_TRADE_R))

            tr.r_multiple_after_costs = realised_r
            daily_r  += realised_r
            weekly_r += realised_r
            results.append(tr)

        return results
