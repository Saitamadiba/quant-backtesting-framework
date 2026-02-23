"""
Drawdown Regime Analysis & Auto-Disable Rules.

Analyzes historical drawdown patterns from WFO OOS trades and calibrates
data-driven auto-disable thresholds for live trading bots.

Classes:
    DrawdownAnalyzer — static analysis of drawdown episodes, streaks, rolling DD
    DrawdownManager  — runtime tracker for live bots (feed trades, check pause)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


# ================================================================
#  DRAWDOWN ANALYZER — Historical Pattern Analysis
# ================================================================

class DrawdownAnalyzer:
    """Analyze drawdown patterns from OOS trade history."""

    @staticmethod
    def analyze(oos_equity: List[Dict], monte_carlo: Optional[Dict] = None) -> Dict:
        """
        Full drawdown pattern analysis from OOS equity data.

        Args:
            oos_equity: List of dicts with 'r', 'cumulative_r', 'time', 'outcome'
            monte_carlo: Optional MC result dict with 'max_drawdown_ci'

        Returns:
            Dict with episodes, streaks, rolling_dd, depth_stats, recovery, thresholds
        """
        if len(oos_equity) < 5:
            return {'valid': False, 'reason': f'Insufficient trades: {len(oos_equity)}'}

        r_values = np.array([e['r'] for e in oos_equity])
        cum_r = np.cumsum(r_values)
        outcomes = [e.get('outcome', '') for e in oos_equity]
        n = len(r_values)

        result = {'valid': True, 'n_trades': n}

        # ── Drawdown Episodes ───────────────────────────────────
        episodes = DrawdownAnalyzer._find_episodes(cum_r)
        result['episodes'] = episodes
        result['n_episodes'] = len(episodes)

        if episodes:
            depths = [ep['depth'] for ep in episodes]
            result['depth_stats'] = {
                'max': float(max(depths)),
                'mean': float(np.mean(depths)),
                'median': float(np.median(depths)),
                'std': float(np.std(depths, ddof=1)) if len(depths) > 1 else 0.0,
                'p75': float(np.percentile(depths, 75)),
                'p90': float(np.percentile(depths, 90)),
                'p95': float(np.percentile(depths, 95)),
                'p99': float(np.percentile(depths, 99)),
            }

            durations = [ep['duration_to_trough'] for ep in episodes]
            recoveries = [ep['recovery_trades'] for ep in episodes if ep['recovered']]
            result['duration_stats'] = {
                'max_to_trough': int(max(durations)),
                'mean_to_trough': float(np.mean(durations)),
                'max_recovery': int(max(recoveries)) if recoveries else None,
                'mean_recovery': float(np.mean(recoveries)) if recoveries else None,
                'pct_recovered': len(recoveries) / len(episodes) if episodes else 0,
            }
        else:
            result['depth_stats'] = {}
            result['duration_stats'] = {}

        # ── Consecutive Loss Streaks ────────────────────────────
        streaks = DrawdownAnalyzer._find_loss_streaks(r_values, outcomes)
        result['streaks'] = streaks

        if streaks['lengths']:
            lengths = streaks['lengths']
            result['streak_stats'] = {
                'max_length': int(max(lengths)),
                'mean_length': float(np.mean(lengths)),
                'p75': float(np.percentile(lengths, 75)),
                'p90': float(np.percentile(lengths, 90)),
                'p95': float(np.percentile(lengths, 95)) if len(lengths) >= 20 else float(max(lengths)),
                'max_r_lost': float(max(streaks['r_lost'])),
                'mean_r_lost': float(np.mean(streaks['r_lost'])),
            }
        else:
            result['streak_stats'] = {'max_length': 0, 'mean_length': 0}

        # ── Rolling Window Max Drawdown ─────────────────────────
        result['rolling_dd'] = {}
        for window in [10, 20, 50]:
            if n >= window:
                rolling = DrawdownAnalyzer._rolling_max_dd(r_values, window)
                result['rolling_dd'][f'window_{window}'] = {
                    'values': [float(v) for v in rolling],
                    'max': float(max(rolling)),
                    'mean': float(np.mean(rolling)),
                    'p75': float(np.percentile(rolling, 75)),
                    'p90': float(np.percentile(rolling, 90)),
                    'p95': float(np.percentile(rolling, 95)),
                }

        # ── Underwater Series (for plotting) ────────────────────
        running_max = np.maximum.accumulate(cum_r)
        underwater = cum_r - running_max  # negative values = in drawdown
        result['underwater'] = [float(v) for v in underwater]
        result['cumulative_r'] = [float(v) for v in cum_r]
        result['times'] = [e.get('time', str(i)) for i, e in enumerate(oos_equity)]

        # ── Recovery Patterns ───────────────────────────────────
        result['recovery'] = DrawdownAnalyzer._recovery_analysis(cum_r)

        # ── Calibrated Thresholds ───────────────────────────────
        result['thresholds'] = DrawdownAnalyzer.calibrate_thresholds(
            oos_equity, monte_carlo
        )

        return result

    @staticmethod
    def _find_episodes(cum_r: np.ndarray) -> List[Dict]:
        """Identify drawdown episodes (peak → trough → recovery cycles)."""
        n = len(cum_r)
        episodes = []
        running_max = np.maximum.accumulate(cum_r)

        in_dd = False
        ep_start = 0
        ep_trough = 0
        ep_trough_val = 0.0
        ep_peak_val = 0.0

        for i in range(n):
            dd = running_max[i] - cum_r[i]

            if dd > 0 and not in_dd:
                # Entering drawdown
                in_dd = True
                ep_start = i - 1 if i > 0 else 0
                ep_peak_val = running_max[i]
                ep_trough = i
                ep_trough_val = cum_r[i]

            elif dd > 0 and in_dd:
                # Still in drawdown — track deepening
                if cum_r[i] < ep_trough_val:
                    ep_trough = i
                    ep_trough_val = cum_r[i]

            elif dd == 0 and in_dd:
                # Recovery — episode complete
                depth = ep_peak_val - ep_trough_val
                if depth >= 0.5:  # Only track significant episodes (>= 0.5R)
                    episodes.append({
                        'start_idx': int(ep_start),
                        'trough_idx': int(ep_trough),
                        'recovery_idx': int(i),
                        'depth': float(depth),
                        'duration_to_trough': int(ep_trough - ep_start),
                        'recovery_trades': int(i - ep_trough),
                        'total_duration': int(i - ep_start),
                        'recovered': True,
                    })
                in_dd = False

        # Handle open drawdown (not yet recovered)
        if in_dd:
            depth = ep_peak_val - ep_trough_val
            if depth >= 0.5:
                episodes.append({
                    'start_idx': int(ep_start),
                    'trough_idx': int(ep_trough),
                    'recovery_idx': None,
                    'depth': float(depth),
                    'duration_to_trough': int(ep_trough - ep_start),
                    'recovery_trades': None,
                    'total_duration': int(n - 1 - ep_start),
                    'recovered': False,
                })

        return episodes

    @staticmethod
    def _find_loss_streaks(
        r_values: np.ndarray, outcomes: List[str]
    ) -> Dict:
        """Find all consecutive loss streaks."""
        lengths = []
        r_lost_per_streak = []
        current_streak = 0
        current_r_lost = 0.0

        for i, (r, outcome) in enumerate(zip(r_values, outcomes)):
            is_loss = 'loss' in outcome or (outcome == '' and r < 0)
            if is_loss:
                current_streak += 1
                current_r_lost += abs(r)
            else:
                if current_streak > 0:
                    lengths.append(current_streak)
                    r_lost_per_streak.append(current_r_lost)
                current_streak = 0
                current_r_lost = 0.0

        # Capture trailing streak
        if current_streak > 0:
            lengths.append(current_streak)
            r_lost_per_streak.append(current_r_lost)

        return {
            'lengths': lengths,
            'r_lost': r_lost_per_streak,
            'n_streaks': len(lengths),
        }

    @staticmethod
    def _rolling_max_dd(r_values: np.ndarray, window: int) -> List[float]:
        """Compute max drawdown within each rolling window of N trades."""
        n = len(r_values)
        rolling_dds = []

        for start in range(n - window + 1):
            chunk = r_values[start:start + window]
            cum = np.cumsum(chunk)
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            rolling_dds.append(float(np.max(dd)))

        return rolling_dds

    @staticmethod
    def _recovery_analysis(cum_r: np.ndarray) -> Dict:
        """Analyze how many trades it takes to recover from various DD depths."""
        running_max = np.maximum.accumulate(cum_r)
        n = len(cum_r)

        thresholds = [1.0, 2.0, 5.0, 10.0]
        recovery_data = {}

        for thresh in thresholds:
            recoveries = []
            i = 0
            while i < n:
                dd = running_max[i] - cum_r[i]
                if dd >= thresh:
                    # Found a drawdown crossing this threshold
                    cross_idx = i
                    # Find recovery (back to running max at cross point)
                    target = running_max[i]
                    recovered = False
                    for j in range(i + 1, n):
                        if cum_r[j] >= target:
                            recoveries.append(j - cross_idx)
                            recovered = True
                            i = j
                            break
                    if not recovered:
                        recoveries.append(None)  # Never recovered
                        break
                i += 1

            valid_recoveries = [r for r in recoveries if r is not None]
            recovery_data[f'{thresh:.0f}R'] = {
                'n_occurrences': len(recoveries),
                'n_recovered': len(valid_recoveries),
                'recovery_rate': len(valid_recoveries) / len(recoveries) if recoveries else 0,
                'avg_trades_to_recover': float(np.mean(valid_recoveries)) if valid_recoveries else None,
                'max_trades_to_recover': int(max(valid_recoveries)) if valid_recoveries else None,
            }

        return recovery_data

    @staticmethod
    def calibrate_thresholds(
        oos_equity: List[Dict],
        monte_carlo: Optional[Dict] = None,
    ) -> Dict:
        """
        Calibrate data-driven auto-disable thresholds.

        Returns recommended rules based on historical drawdown distribution.
        """
        r_values = np.array([e['r'] for e in oos_equity])
        n = len(r_values)

        thresholds = {}

        # 1. Rolling 20-trade DD threshold
        if n >= 20:
            rolling_dd = DrawdownAnalyzer._rolling_max_dd(r_values, 20)
            p95 = float(np.percentile(rolling_dd, 95))
            # Add 10% buffer
            thresholds['rolling_20_dd'] = {
                'value': round(p95 * 1.1, 1),
                'raw_p95': round(p95, 2),
                'calibration': '95th percentile + 10% buffer of rolling 20-trade max DD',
            }
        else:
            cum_r = np.cumsum(r_values)
            running_max = np.maximum.accumulate(cum_r)
            max_dd = float(np.max(running_max - cum_r))
            thresholds['rolling_20_dd'] = {
                'value': round(max_dd * 1.2, 1),
                'raw_p95': round(max_dd, 2),
                'calibration': 'Overall max DD + 20% buffer (insufficient data for rolling)',
            }

        # 2. Consecutive loss streak threshold
        outcomes = [e.get('outcome', '') for e in oos_equity]
        streaks = DrawdownAnalyzer._find_loss_streaks(r_values, outcomes)
        if streaks['lengths']:
            lengths = streaks['lengths']
            if len(lengths) >= 20:
                p95 = float(np.percentile(lengths, 95))
            else:
                p95 = float(max(lengths))
            thresholds['max_consecutive_losses'] = {
                'value': int(np.ceil(p95)) + 1,
                'raw_p95': round(p95, 1),
                'calibration': '95th percentile of streak lengths + 1',
            }
        else:
            thresholds['max_consecutive_losses'] = {
                'value': 5,
                'raw_p95': 0,
                'calibration': 'Default (no loss streaks found)',
            }

        # 3. Cooldown trades (from recovery analysis)
        cum_r = np.cumsum(r_values)
        recovery = DrawdownAnalyzer._recovery_analysis(cum_r)
        # Use recovery from 2R drawdown as reference
        rec_2r = recovery.get('2R', {})
        avg_recovery = rec_2r.get('avg_trades_to_recover')
        if avg_recovery is not None:
            thresholds['cooldown_trades'] = {
                'value': max(3, int(np.ceil(avg_recovery * 0.5))),
                'avg_recovery_from_2R': round(avg_recovery, 1),
                'calibration': '50% of avg trades to recover from 2R drawdown',
            }
        else:
            thresholds['cooldown_trades'] = {
                'value': 5,
                'calibration': 'Default (no recovery data)',
            }

        # 4. Use Monte Carlo if available
        if monte_carlo and monte_carlo.get('valid'):
            mc_max_dd_ci = monte_carlo.get('max_drawdown_ci')
            if mc_max_dd_ci:
                thresholds['mc_max_dd_95'] = {
                    'value': round(mc_max_dd_ci[1], 1),
                    'ci': [round(mc_max_dd_ci[0], 1), round(mc_max_dd_ci[1], 1)],
                    'calibration': '95% CI upper bound of bootstrap max drawdown',
                }

        # 5. Confidence level
        if n >= 200:
            confidence = 'high'
        elif n >= 50:
            confidence = 'medium'
        else:
            confidence = 'low'
        thresholds['confidence'] = confidence
        thresholds['n_trades'] = n

        return thresholds


# ================================================================
#  DRAWDOWN MANAGER — Live Bot Runtime Tracker
# ================================================================

class DrawdownManager:
    """
    Runtime drawdown tracker for live trading bots.

    Feed trade results as they happen, check if auto-disable should trigger.

    Usage:
        manager = DrawdownManager(thresholds={
            'rolling_20_dd': {'value': 8.5},
            'max_consecutive_losses': {'value': 7},
            'cooldown_trades': {'value': 5},
        })

        # After each trade:
        manager.update(trade_r=-1.0, is_loss=True)
        paused, reason = manager.should_pause()
        if paused:
            print(f"AUTO-DISABLE: {reason}")
    """

    def __init__(self, thresholds: Dict, rolling_window: int = 20):
        self.rolling_window = rolling_window
        self.recent_trades: deque = deque(maxlen=rolling_window)
        self.consecutive_losses = 0
        self.peak_r = 0.0
        self.cumulative_r = 0.0
        self.cooldown_remaining = 0
        self.total_trades = 0
        self.paused = False
        self.pause_reason = ''

        # Extract threshold values
        self.dd_threshold = thresholds.get('rolling_20_dd', {}).get('value', 10.0)
        self.streak_threshold = thresholds.get('max_consecutive_losses', {}).get('value', 7)
        self.cooldown_trades = thresholds.get('cooldown_trades', {}).get('value', 5)

    def update(self, trade_r: float, is_loss: bool = None) -> None:
        """Feed a new trade result into the tracker."""
        if is_loss is None:
            is_loss = trade_r < 0

        self.total_trades += 1

        # Handle cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.paused = False
                self.pause_reason = ''

        # Update rolling window
        self.recent_trades.append(trade_r)

        # Update cumulative
        self.cumulative_r += trade_r
        self.peak_r = max(self.peak_r, self.cumulative_r)

        # Update consecutive losses
        if is_loss:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def should_pause(self) -> Tuple[bool, str]:
        """Check if auto-disable should trigger. Returns (should_pause, reason)."""
        if self.paused:
            return True, self.pause_reason

        # Check 1: Rolling window drawdown
        if len(self.recent_trades) >= self.rolling_window:
            r_arr = np.array(list(self.recent_trades))
            cum = np.cumsum(r_arr)
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            max_dd = float(np.max(dd))

            if max_dd >= self.dd_threshold:
                self.paused = True
                self.pause_reason = (
                    f"Rolling {self.rolling_window}-trade DD = {max_dd:.1f}R "
                    f"(threshold: {self.dd_threshold:.1f}R)"
                )
                self.cooldown_remaining = self.cooldown_trades
                return True, self.pause_reason

        # Check 2: Consecutive loss streak
        if self.consecutive_losses >= self.streak_threshold:
            self.paused = True
            self.pause_reason = (
                f"Consecutive losses: {self.consecutive_losses} "
                f"(threshold: {self.streak_threshold})"
            )
            self.cooldown_remaining = self.cooldown_trades
            return True, self.pause_reason

        # Check 3: Overall drawdown from peak
        overall_dd = self.peak_r - self.cumulative_r
        if overall_dd >= self.dd_threshold * 2:
            self.paused = True
            self.pause_reason = (
                f"Overall DD from peak = {overall_dd:.1f}R "
                f"(threshold: {self.dd_threshold * 2:.1f}R)"
            )
            self.cooldown_remaining = self.cooldown_trades
            return True, self.pause_reason

        return False, ''

    def reset(self) -> None:
        """Force-reset the manager (e.g., after manual override)."""
        self.paused = False
        self.pause_reason = ''
        self.cooldown_remaining = 0
        self.consecutive_losses = 0

    def get_state(self) -> Dict:
        """Get current state (for serialization/logging)."""
        return {
            'total_trades': self.total_trades,
            'cumulative_r': self.cumulative_r,
            'peak_r': self.peak_r,
            'current_dd': self.peak_r - self.cumulative_r,
            'consecutive_losses': self.consecutive_losses,
            'paused': self.paused,
            'pause_reason': self.pause_reason,
            'cooldown_remaining': self.cooldown_remaining,
            'recent_trades': list(self.recent_trades),
        }

    def load_state(self, state: Dict) -> None:
        """Restore state from serialized dict (for bot restarts)."""
        self.total_trades = state.get('total_trades', 0)
        self.cumulative_r = state.get('cumulative_r', 0.0)
        self.peak_r = state.get('peak_r', 0.0)
        self.consecutive_losses = state.get('consecutive_losses', 0)
        self.paused = state.get('paused', False)
        self.pause_reason = state.get('pause_reason', '')
        self.cooldown_remaining = state.get('cooldown_remaining', 0)
        recent = state.get('recent_trades', [])
        self.recent_trades = deque(recent, maxlen=self.rolling_window)
