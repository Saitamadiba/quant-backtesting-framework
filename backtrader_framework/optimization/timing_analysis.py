"""
Execution Timing Optimization.

Analyzes WHEN OOS trades perform best — by hour of day, trading session,
and day of week. Uses MFE/MAE as proxies for execution quality since
backtesting lacks actual slippage data. Produces timing filter
recommendations for signal gating.

Classes:
    TimingAnalyzer — static analysis of trade timing patterns
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime


# Trading sessions (UTC hours)
SESSIONS = {
    'Asia':    (0, 8),    # 00:00 - 08:00
    'London':  (8, 13),   # 08:00 - 13:00
    'Overlap': (13, 16),  # 13:00 - 16:00 (London+NY, highest volume)
    'NY':      (16, 21),  # 16:00 - 21:00
    'Quiet':   (21, 24),  # 21:00 - 00:00
}

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


class TimingAnalyzer:
    """Analyze execution timing patterns from OOS trade history."""

    MIN_TRADES_PER_BUCKET = 30  # Minimum for statistical reliability

    @staticmethod
    def analyze(trades: list, symbol: str = '') -> Dict:
        """
        Full timing pattern analysis from OOS TradeResult objects.

        Args:
            trades: List of TradeResult dataclass instances.
            symbol: Optional symbol name (unused currently, reserved).

        Returns:
            Dict with valid flag and all timing sub-analyses.
        """
        if len(trades) < 10:
            return {'valid': False, 'reason': f'Insufficient trades: {len(trades)}'}

        parsed = []
        for t in trades:
            entry = t.entry_time
            if isinstance(entry, str):
                entry = datetime.fromisoformat(str(entry))
            elif hasattr(entry, 'to_pydatetime'):
                entry = entry.to_pydatetime()

            parsed.append({
                'entry_time': entry,
                'direction': t.direction,
                'outcome': t.outcome,
                'r': t.r_multiple_after_costs,
                'bars_held': t.bars_held,
                'mfe': t.mfe,
                'mae': t.mae,
                'regime': t.regime,
            })

        result = {'valid': True, 'n_trades': len(parsed)}

        result['hourly'] = TimingAnalyzer._hourly_breakdown(parsed)
        result['session'] = TimingAnalyzer._session_breakdown(parsed)
        result['day_of_week'] = TimingAnalyzer._day_of_week_breakdown(parsed)
        result['direction_hour_heatmap'] = TimingAnalyzer._direction_hour_heatmap(parsed)
        result['hold_duration'] = TimingAnalyzer._hold_duration_analysis(parsed)
        result['execution_quality'] = TimingAnalyzer._execution_quality_by_hour(parsed)
        result['recommendations'] = TimingAnalyzer._timing_recommendations(
            result['hourly'], result['session'], result['day_of_week'],
            result['execution_quality'], len(parsed),
        )

        return result

    # ── Shared helper ─────────────────────────────────────────────

    @staticmethod
    def _compute_bucket_metrics(bucket_trades: List[Dict]) -> Dict:
        """Compute standard metrics for a group of trades."""
        n = len(bucket_trades)
        if n == 0:
            return {'n_trades': 0}

        rs = [t['r'] for t in bucket_trades]
        wins = [t for t in bucket_trades if 'win' in t.get('outcome', '')]
        mfes = [t['mfe'] for t in bucket_trades if t.get('mfe') is not None]
        maes = [t['mae'] for t in bucket_trades if t.get('mae') is not None]

        win_rate = len(wins) / n
        mean_r = float(np.mean(rs))
        total_r = float(np.sum(rs))

        metrics = {
            'n_trades': n,
            'win_rate': round(win_rate, 4),
            'mean_r': round(mean_r, 6),
            'total_r': round(total_r, 4),
            'std_r': round(float(np.std(rs, ddof=1)), 6) if n > 1 else 0.0,
            'mean_mfe': round(float(np.mean(mfes)), 6) if mfes else 0.0,
            'mean_mae': round(float(np.mean(maes)), 6) if maes else 0.0,
        }

        if metrics['mean_mae'] > 0:
            metrics['mfe_mae_ratio'] = round(metrics['mean_mfe'] / metrics['mean_mae'], 4)
        else:
            metrics['mfe_mae_ratio'] = 0.0

        return metrics

    # ── Sub-analyses ──────────────────────────────────────────────

    @staticmethod
    def _hourly_breakdown(parsed: List[Dict]) -> Dict:
        """Metrics for each hour 0-23 UTC."""
        buckets = defaultdict(list)
        for t in parsed:
            hour = t['entry_time'].hour
            buckets[hour].append(t)

        result = {}
        for hour in range(24):
            result[hour] = TimingAnalyzer._compute_bucket_metrics(buckets[hour])
        return result

    @staticmethod
    def _get_session(hour: int) -> str:
        """Map hour (0-23) to trading session name."""
        for name, (start, end) in SESSIONS.items():
            if start <= hour < end:
                return name
        return 'Quiet'

    @staticmethod
    def _session_breakdown(parsed: List[Dict]) -> Dict:
        """Metrics grouped by trading session."""
        buckets = defaultdict(list)
        for t in parsed:
            session = TimingAnalyzer._get_session(t['entry_time'].hour)
            buckets[session].append(t)

        result = {}
        for session_name in SESSIONS:
            result[session_name] = TimingAnalyzer._compute_bucket_metrics(buckets[session_name])
        return result

    @staticmethod
    def _day_of_week_breakdown(parsed: List[Dict]) -> Dict:
        """Metrics per weekday (0=Mon through 6=Sun)."""
        buckets = defaultdict(list)
        for t in parsed:
            dow = t['entry_time'].weekday()
            buckets[dow].append(t)

        result = {}
        for day_idx in range(7):
            metrics = TimingAnalyzer._compute_bucket_metrics(buckets[day_idx])
            metrics['day_name'] = DAY_NAMES[day_idx]
            result[day_idx] = metrics
        return result

    @staticmethod
    def _direction_hour_heatmap(parsed: List[Dict]) -> Dict:
        """Win rate and mean R by direction (LONG/SHORT) x hour (0-23)."""
        buckets = defaultdict(list)
        for t in parsed:
            key = (t['direction'], t['entry_time'].hour)
            buckets[key].append(t)

        result = {}
        for direction in ['LONG', 'SHORT']:
            result[direction] = {}
            for hour in range(24):
                result[direction][hour] = TimingAnalyzer._compute_bucket_metrics(
                    buckets[(direction, hour)]
                )
        return result

    @staticmethod
    def _hold_duration_analysis(parsed: List[Dict]) -> Dict:
        """Analyze bars_held distribution and relationship to outcomes."""
        valid_trades = [t for t in parsed if t.get('bars_held') is not None]
        if not valid_trades:
            return {'valid': False}

        bars = np.array([t['bars_held'] for t in valid_trades])
        result = {
            'valid': True,
            'distribution': {
                'mean': round(float(np.mean(bars)), 1),
                'median': round(float(np.median(bars)), 0),
                'std': round(float(np.std(bars, ddof=1)), 1) if len(bars) > 1 else 0.0,
                'min': int(np.min(bars)),
                'max': int(np.max(bars)),
                'p25': round(float(np.percentile(bars, 25)), 0),
                'p75': round(float(np.percentile(bars, 75)), 0),
                'p90': round(float(np.percentile(bars, 90)), 0),
            },
            'by_outcome': {},
        }

        # Break down by outcome
        outcome_buckets = defaultdict(list)
        for t in valid_trades:
            outcome_buckets[t['outcome']].append(t['bars_held'])
        for outcome, durations in outcome_buckets.items():
            d = np.array(durations)
            result['by_outcome'][outcome] = {
                'n': len(d),
                'mean': round(float(np.mean(d)), 1),
                'median': round(float(np.median(d)), 0),
            }

        # Correlation: bars_held vs r_multiple (no scipy, use np.corrcoef)
        rs = np.array([t['r'] for t in valid_trades])
        if len(rs) > 5:
            corr_matrix = np.corrcoef(bars, rs)
            pearson_r = float(corr_matrix[0, 1])
            # Approximate p-value using t-distribution
            n = len(rs)
            if abs(pearson_r) < 1.0:
                t_stat = pearson_r * np.sqrt((n - 2) / (1 - pearson_r ** 2))
                # Two-tailed p-value approximation (normal for large n)
                from math import erf
                p_val = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / np.sqrt(2))))
            else:
                p_val = 0.0
            result['duration_r_correlation'] = {
                'pearson_r': round(pearson_r, 4),
                'p_value': round(p_val, 6),
                'significant': p_val < 0.05,
            }

        return result

    @staticmethod
    def _execution_quality_by_hour(parsed: List[Dict]) -> Dict:
        """MFE/MAE ratio per hour — proxy for execution quality."""
        buckets = defaultdict(list)
        for t in parsed:
            if t.get('mfe') is not None and t.get('mae') is not None and t['mae'] > 0:
                buckets[t['entry_time'].hour].append(t)

        result = {}
        for hour in range(24):
            trades = buckets[hour]
            if not trades:
                result[hour] = {'n_trades': 0, 'mfe_mae_ratio': None}
                continue

            mfes = np.array([t['mfe'] for t in trades])
            maes = np.array([t['mae'] for t in trades])
            ratios = mfes / (maes + 1e-10)

            result[hour] = {
                'n_trades': len(trades),
                'mean_mfe': round(float(np.mean(mfes)), 6),
                'mean_mae': round(float(np.mean(maes)), 6),
                'mfe_mae_ratio': round(float(np.mean(ratios)), 4),
                'median_ratio': round(float(np.median(ratios)), 4),
            }
        return result

    # ── Recommendations ───────────────────────────────────────────

    @staticmethod
    def _timing_recommendations(
        hourly: Dict, session: Dict, day_of_week: Dict,
        exec_quality: Dict, total_trades: int,
    ) -> Dict:
        """
        Generate timing filter recommendations from analysis.

        Applies Bonferroni correction for multiple hypothesis testing across
        all time buckets (hours, sessions, days) to control family-wise
        error rate.
        """
        MIN_N = TimingAnalyzer.MIN_TRADES_PER_BUCKET

        recommendations = {
            'best_hours': [],
            'worst_hours': [],
            'best_sessions': [],
            'worst_sessions': [],
            'best_days': [],
            'worst_days': [],
            'suggested_filters': [],
            'confidence': 'low' if total_trades < 100 else ('medium' if total_trades < 300 else 'high'),
            'disclaimers': [],
        }

        # Bonferroni correction: total hypotheses = hours + sessions + days
        n_hour_buckets = sum(
            1 for h in range(24)
            if hourly.get(h, {}).get('n_trades', 0) >= MIN_N
        )
        n_session_buckets = sum(
            1 for s in session.values()
            if s.get('n_trades', 0) >= MIN_N
        )
        n_day_buckets = sum(
            1 for d in range(7)
            if day_of_week.get(d, {}).get('n_trades', 0) >= MIN_N
        )
        n_hypotheses = n_hour_buckets + n_session_buckets + n_day_buckets
        adjusted_alpha = 0.05 / n_hypotheses if n_hypotheses > 0 else 0.05
        recommendations['bonferroni'] = {
            'n_hypotheses': n_hypotheses,
            'nominal_alpha': 0.05,
            'adjusted_alpha': round(adjusted_alpha, 6),
        }

        # Helper: two-sided t-test p-value for bucket mean vs overall mean
        def _bucket_p_value(bucket_mean_r, bucket_std_r, bucket_n, overall_mean_r):
            """Approximate p-value for bucket mean deviating from overall mean."""
            if bucket_n < 2 or bucket_std_r <= 0:
                return 1.0
            se = bucket_std_r / np.sqrt(bucket_n)
            t_stat = (bucket_mean_r - overall_mean_r) / se
            # Two-tailed p-value approximation using error function
            from math import erf
            p_val = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / np.sqrt(2))))
            return p_val

        # Compute overall mean_r as baseline
        weighted_sum = 0.0
        weighted_n = 0
        for h in range(24):
            m = hourly.get(h, {})
            n = m.get('n_trades', 0)
            if n >= MIN_N:
                weighted_sum += m['mean_r'] * n
                weighted_n += n
        overall_mean_r = weighted_sum / weighted_n if weighted_n > 0 else 0.0

        # ── Hours ──
        for hour in range(24):
            m = hourly.get(hour, {})
            if m.get('n_trades', 0) < MIN_N:
                continue

            p_val = _bucket_p_value(
                m['mean_r'], m.get('std_r', 0.0), m['n_trades'], overall_mean_r,
            )

            entry = {
                'hour': hour,
                'n_trades': m['n_trades'],
                'mean_r': m['mean_r'],
                'win_rate': m['win_rate'],
                'total_r': m['total_r'],
                'p_value': round(p_val, 6),
                'significant_after_correction': p_val < adjusted_alpha,
            }
            if m['mean_r'] > 0 and m['mean_r'] > overall_mean_r:
                recommendations['best_hours'].append(entry)
            elif m['mean_r'] < 0 and m['mean_r'] < overall_mean_r:
                recommendations['worst_hours'].append(entry)

        recommendations['best_hours'].sort(key=lambda x: x['mean_r'], reverse=True)
        recommendations['worst_hours'].sort(key=lambda x: x['mean_r'])

        # ── Sessions ──
        for session_name, m in session.items():
            if m.get('n_trades', 0) < MIN_N:
                continue

            p_val = _bucket_p_value(
                m['mean_r'], m.get('std_r', 0.0), m['n_trades'], overall_mean_r,
            )

            entry = {
                'session': session_name,
                'n_trades': m['n_trades'],
                'mean_r': m['mean_r'],
                'win_rate': m['win_rate'],
                'total_r': m['total_r'],
                'mfe_mae_ratio': m.get('mfe_mae_ratio', 0),
                'p_value': round(p_val, 6),
                'significant_after_correction': p_val < adjusted_alpha,
            }
            if m['mean_r'] > 0:
                recommendations['best_sessions'].append(entry)
            elif m['mean_r'] < 0:
                recommendations['worst_sessions'].append(entry)

        recommendations['best_sessions'].sort(key=lambda x: x['mean_r'], reverse=True)
        recommendations['worst_sessions'].sort(key=lambda x: x['mean_r'])

        # ── Days ──
        for day_idx in range(7):
            m = day_of_week.get(day_idx, {})
            if m.get('n_trades', 0) < MIN_N:
                continue

            p_val = _bucket_p_value(
                m['mean_r'], m.get('std_r', 0.0), m['n_trades'], overall_mean_r,
            )

            entry = {
                'day': m.get('day_name', DAY_NAMES[day_idx]),
                'n_trades': m['n_trades'],
                'mean_r': m['mean_r'],
                'win_rate': m['win_rate'],
                'total_r': m['total_r'],
                'p_value': round(p_val, 6),
                'significant_after_correction': p_val < adjusted_alpha,
            }
            if m['mean_r'] > 0:
                recommendations['best_days'].append(entry)
            elif m['mean_r'] < 0:
                recommendations['worst_days'].append(entry)

        recommendations['best_days'].sort(key=lambda x: x['mean_r'], reverse=True)
        recommendations['worst_days'].sort(key=lambda x: x['mean_r'])

        # ── Concrete filter suggestions ──
        # Only suggest filters for statistically significant findings
        worst_hours = recommendations['worst_hours']
        significant_worst_hours = [
            h for h in worst_hours if h.get('significant_after_correction', False)
        ]
        if len(significant_worst_hours) >= 2:
            avoid_hours = [h['hour'] for h in significant_worst_hours[:3]]
            worst_total_r = sum(h['total_r'] for h in significant_worst_hours[:3])
            recommendations['suggested_filters'].append({
                'type': 'avoid_hours',
                'hours_utc': avoid_hours,
                'rationale': f'Worst {len(avoid_hours)} hours contribute {worst_total_r:.1f}R total (Bonferroni-significant)',
            })
        elif len(worst_hours) >= 2:
            # Fallback: suggest but flag as not significant
            avoid_hours = [h['hour'] for h in worst_hours[:3]]
            worst_total_r = sum(h['total_r'] for h in worst_hours[:3])
            recommendations['suggested_filters'].append({
                'type': 'avoid_hours',
                'hours_utc': avoid_hours,
                'rationale': f'Worst {len(avoid_hours)} hours contribute {worst_total_r:.1f}R total',
                'warning': 'NOT statistically significant after Bonferroni correction',
            })

        worst_sessions = recommendations['worst_sessions']
        if worst_sessions:
            sig_flag = worst_sessions[0].get('significant_after_correction', False)
            filter_entry = {
                'type': 'avoid_session',
                'session': worst_sessions[0]['session'],
                'rationale': f"{worst_sessions[0]['session']} session: mean R = {worst_sessions[0]['mean_r']:.3f}",
            }
            if not sig_flag:
                filter_entry['warning'] = 'NOT statistically significant after Bonferroni correction'
            recommendations['suggested_filters'].append(filter_entry)

        # ── Sample size disclaimers ──
        if total_trades < 200:
            recommendations['disclaimers'].append(
                f'WARNING: Only {total_trades} total trades. Timing patterns with '
                f'<200 trades are unreliable and likely driven by noise. '
                f'Increase sample size before acting on these recommendations.'
            )
        if n_hypotheses > 0:
            recommendations['disclaimers'].append(
                f'Multiple comparison correction applied: {n_hypotheses} hypotheses '
                f'tested, adjusted significance level = {adjusted_alpha:.4f} '
                f'(Bonferroni). Only "significant_after_correction: true" entries '
                f'survived correction.'
            )

        return recommendations
