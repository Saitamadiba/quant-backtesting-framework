"""
Statistical testing and Monte Carlo analysis for Walk-Forward Optimization.

Provides StatisticalTests (binomial, t-test, serial correlation, profit factor,
expectancy) and MonteCarloAnalysis (bootstrap CIs, equity fan charts) classes
for evaluating trade results.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from .wfo_engine import TradeResult

logger = logging.getLogger(__name__)


class StatisticalTests:
    """Run significance tests (binomial, t-test, serial correlation) on trade results."""

    @staticmethod
    def run_all(trades: List[TradeResult]) -> Dict:
        """Compute full statistical summary: win rate CI, t-test, profit factor, expectancy, etc.

        Returns a dict with 'valid' key; False if fewer than 5 trades.
        """
        from scipy import stats as scipy_stats
        if len(trades) < 5:
            return {'valid': False, 'reason': f'Insufficient trades: {len(trades)}', 'n_trades': len(trades)}

        r_values = [t.r_multiple_after_costs for t in trades]
        wins = [t for t in trades if 'win' in t.outcome]
        n = len(trades)
        n_wins = len(wins)

        results = {'valid': True, 'n_trades': n, 'n_wins': n_wins, 'n_losses': n - n_wins}

        win_rate = n_wins / n
        results['win_rate'] = win_rate
        results['win_rate_ci_95'] = StatisticalTests._wilson_ci(n_wins, n, 0.95)

        try:
            binom_p = scipy_stats.binomtest(n_wins, n, 0.5, alternative='greater').pvalue
        except AttributeError:
            binom_p = 1.0
        results['binomial_p_value'] = binom_p
        results['win_rate_significant'] = binom_p < 0.05

        mean_r = float(np.mean(r_values))
        std_r = float(np.std(r_values, ddof=1))
        se_r = std_r / np.sqrt(n)
        ci_r = scipy_stats.t.interval(0.95, n - 1, loc=mean_r, scale=se_r) if se_r > 0 else (mean_r, mean_r)
        results['mean_r'] = mean_r
        results['std_r'] = std_r
        results['mean_r_ci_95'] = (float(ci_r[0]), float(ci_r[1]))

        t_stat, t_p = scipy_stats.ttest_1samp(r_values, 0)
        results['t_statistic'] = float(t_stat)
        results['t_p_value'] = float(t_p / 2)
        results['mean_r_significant'] = (t_p / 2) < 0.05 and mean_r > 0

        results['sharpe_per_trade'] = mean_r / std_r if std_r > 0 else 0

        gross_profit = sum(r for r in r_values if r > 0)
        gross_loss = abs(sum(r for r in r_values if r < 0))
        results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        cumulative_r = np.cumsum(r_values)
        running_max = np.maximum.accumulate(cumulative_r)
        drawdowns = running_max - cumulative_r
        results['max_drawdown_r'] = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        avg_win = float(np.mean([r for r in r_values if r > 0])) if any(r > 0 for r in r_values) else 0
        avg_loss = float(np.mean([r for r in r_values if r <= 0])) if any(r <= 0 for r in r_values) else 0
        results['avg_win_r'] = avg_win
        results['avg_loss_r'] = avg_loss
        results['expectancy'] = win_rate * avg_win + (1 - win_rate) * avg_loss

        if std_r > 0 and mean_r != 0:
            effect_size = abs(mean_r) / std_r
            results['min_n_for_significance'] = int(np.ceil((2.8 / effect_size) ** 2))
        else:
            results['min_n_for_significance'] = 999

        if n > 10:
            binary = [1 if r > 0 else 0 for r in r_values]
            autocorr = float(np.corrcoef(binary[:-1], binary[1:])[0, 1])
            results['serial_correlation'] = autocorr
            results['trades_independent'] = abs(autocorr) < 0.2
        else:
            results['serial_correlation'] = None
            results['trades_independent'] = None

        return results

    @staticmethod
    def _wilson_ci(wins: int, n: int, confidence: float) -> Tuple[float, float]:
        """Compute Wilson score confidence interval for a binomial proportion."""
        from scipy import stats as scipy_stats
        if n == 0:
            return (0.0, 1.0)
        z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
        p_hat = wins / n
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
        return (max(0, center - margin), min(1, center + margin))


class MonteCarloAnalysis:
    """Bootstrap-based confidence intervals and equity path simulation."""

    @staticmethod
    def bootstrap_ci(
        trades: List[TradeResult],
        n_resamples: int = 10000,
        block_size: int = 1,
        confidence: float = 0.95,
    ) -> Dict:
        """
        Compute bootstrap CIs on key metrics by resampling trades.

        Returns dict with percentile-based CIs for mean_r, win_rate,
        expectancy, profit_factor, sharpe, max_drawdown, plus p_profitable.
        """
        r_values = np.array([t.r_multiple_after_costs for t in trades])
        n = len(r_values)
        if n < 5:
            return {'valid': False, 'reason': f'Insufficient trades: {n}'}

        alpha = 1 - confidence
        lo_pct = 100 * alpha / 2
        hi_pct = 100 * (1 - alpha / 2)

        # Pre-allocate metric arrays
        mean_rs = np.empty(n_resamples)
        win_rates = np.empty(n_resamples)
        expectancies = np.empty(n_resamples)
        profit_factors = np.empty(n_resamples)
        sharpes = np.empty(n_resamples)
        max_dds = np.empty(n_resamples)

        max_block_start = max(1, n - block_size + 1)

        for i in range(n_resamples):
            # Resample
            if block_size <= 1:
                sample = r_values[np.random.randint(0, n, size=n)]
            else:
                n_blocks = int(np.ceil(n / block_size))
                blocks = []
                for _ in range(n_blocks):
                    start = np.random.randint(0, max_block_start)
                    blocks.extend(r_values[start:start + block_size].tolist())
                sample = np.array(blocks[:n])

            # Compute metrics on this resample
            mean_rs[i] = np.mean(sample)
            wins = sample > 0
            wr = np.sum(wins) / n
            win_rates[i] = wr

            win_vals = sample[wins]
            loss_vals = sample[~wins]
            avg_w = np.mean(win_vals) if len(win_vals) > 0 else 0.0
            avg_l = np.mean(loss_vals) if len(loss_vals) > 0 else 0.0
            expectancies[i] = wr * avg_w + (1 - wr) * avg_l

            gp = np.sum(sample[sample > 0])
            gl = abs(np.sum(sample[sample < 0]))
            profit_factors[i] = gp / gl if gl > 0 else 0.0

            std = np.std(sample, ddof=1)
            sharpes[i] = mean_rs[i] / std if std > 0 else 0.0

            cum = np.cumsum(sample)
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            max_dds[i] = np.max(dd) if len(dd) > 0 else 0.0

        p_profitable = float(np.mean(mean_rs > 0))

        return {
            'valid': True,
            'n_resamples': n_resamples,
            'confidence': confidence,
            'p_profitable': p_profitable,
            'mean_r_ci': (float(np.percentile(mean_rs, lo_pct)), float(np.percentile(mean_rs, hi_pct))),
            'win_rate_ci': (float(np.percentile(win_rates, lo_pct)), float(np.percentile(win_rates, hi_pct))),
            'expectancy_ci': (float(np.percentile(expectancies, lo_pct)), float(np.percentile(expectancies, hi_pct))),
            'profit_factor_ci': (float(np.percentile(profit_factors, lo_pct)), float(np.percentile(profit_factors, hi_pct))),
            'sharpe_ci': (float(np.percentile(sharpes, lo_pct)), float(np.percentile(sharpes, hi_pct))),
            'max_drawdown_ci': (float(np.percentile(max_dds, lo_pct)), float(np.percentile(max_dds, hi_pct))),
        }

    @staticmethod
    def equity_fan(
        trades: List[TradeResult],
        n_paths: int = 1000,
        block_size: int = 1,
    ) -> Dict:
        """
        Generate resampled equity paths for fan chart visualization.

        Returns percentile bands at each trade index, plus summary stats.
        """
        r_values = np.array([t.r_multiple_after_costs for t in trades])
        n = len(r_values)
        if n < 5:
            return {'valid': False}

        max_block_start = max(1, n - block_size + 1)

        # Generate paths
        all_paths = np.empty((n_paths, n))
        for i in range(n_paths):
            if block_size <= 1:
                sample = r_values[np.random.randint(0, n, size=n)]
            else:
                n_blocks = int(np.ceil(n / block_size))
                blocks = []
                for _ in range(n_blocks):
                    start = np.random.randint(0, max_block_start)
                    blocks.extend(r_values[start:start + block_size].tolist())
                sample = np.array(blocks[:n])
            all_paths[i] = np.cumsum(sample)

        # Compute percentile bands at each trade index
        pct_5 = np.percentile(all_paths, 5, axis=0).tolist()
        pct_25 = np.percentile(all_paths, 25, axis=0).tolist()
        pct_50 = np.percentile(all_paths, 50, axis=0).tolist()
        pct_75 = np.percentile(all_paths, 75, axis=0).tolist()
        pct_95 = np.percentile(all_paths, 95, axis=0).tolist()

        # Final R values across all paths
        final_rs = all_paths[:, -1]

        # Max drawdowns across all paths
        max_dds = []
        for path in all_paths:
            running_max = np.maximum.accumulate(path)
            dd = running_max - path
            max_dds.append(float(np.max(dd)))

        # Actual OOS equity for overlay
        actual_equity = np.cumsum(r_values).tolist()

        # Store up to 200 raw paths for spaghetti overlay
        raw_paths = all_paths[:min(200, n_paths)].tolist()

        return {
            'valid': True,
            'n_trades': n,
            'n_paths': n_paths,
            'percentiles': {
                '5': pct_5, '25': pct_25, '50': pct_50,
                '75': pct_75, '95': pct_95,
            },
            'actual_equity': actual_equity,
            'pct_5_final_r': float(np.percentile(final_rs, 5)),
            'median_final_r': float(np.median(final_rs)),
            'pct_95_final_r': float(np.percentile(final_rs, 95)),
            'pct_95_max_dd': float(np.percentile(max_dds, 95)),
            'median_max_dd': float(np.median(max_dds)),
            'paths': raw_paths,
        }
