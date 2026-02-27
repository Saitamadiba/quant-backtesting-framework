"""
Combinatorial Purged Cross-Validation (CPCV) and Deflated Sharpe Ratio.

Implements institutional-grade overfitting detection:
- CPCV: Generates all combinatorial train/test splits with purging to estimate
  the Probability of Backtest Overfitting (PBO)
- Deflated Sharpe: Corrects Sharpe ratio for multiple testing, non-normality

References:
- Bailey, D.H. & Lopez de Prado, M. (2014) "The Deflated Sharpe Ratio"
- Bailey et al. (2017) "Probability of Backtest Overfitting"
"""

import numpy as np
import logging
from itertools import combinations
from typing import List, Dict, Tuple, Optional
from scipy import stats

logger = logging.getLogger(__name__)


def deflated_sharpe_ratio(
    observed_sr: float,
    num_trials: int,
    n_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> Dict[str, float]:
    """
    Compute the Deflated Sharpe Ratio (DSR) that adjusts for:
    1. Number of parameter combinations tried (multiple testing)
    2. Non-normality of returns (skew and excess kurtosis)

    Args:
        observed_sr: The observed (best) Sharpe ratio from optimization
        num_trials: Number of parameter combinations tested
        n_returns: Number of return observations
        skewness: Skewness of the return distribution
        kurtosis: Kurtosis of the return distribution (3.0 for normal)
        sr_benchmark: Benchmark Sharpe ratio to test against (default 0)

    Returns:
        dict with 'deflated_sr', 'p_value', 'is_significant'
    """
    if num_trials <= 1 or n_returns <= 1:
        return {'deflated_sr': observed_sr, 'p_value': 0.5, 'is_significant': False}

    # Expected maximum Sharpe ratio under null hypothesis (multiple testing correction)
    # E[max(SR)] ≈ sqrt(2 * log(N)) - (log(pi) + log(log(N))) / (2 * sqrt(2 * log(N)))
    euler_mascheroni = 0.5772156649
    log_n = np.log(num_trials)

    if log_n > 0:
        e_max_sr = np.sqrt(2 * log_n) - (np.log(np.pi) + np.log(log_n)) / (2 * np.sqrt(2 * log_n))
    else:
        e_max_sr = 0.0

    # Standard deviation of SR estimator (corrected for non-normality)
    # Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / (n-1)
    excess_kurt = kurtosis - 3.0
    sr_var = (1.0 + 0.5 * observed_sr**2 - skewness * observed_sr +
              excess_kurt / 4.0 * observed_sr**2) / max(n_returns - 1, 1)
    sr_std = np.sqrt(max(sr_var, 1e-10))

    # PSR (Probabilistic Sharpe Ratio) against the expected max
    test_stat = (observed_sr - e_max_sr) / sr_std
    p_value = 1.0 - stats.norm.cdf(test_stat)

    return {
        'deflated_sr': round(test_stat, 4),
        'expected_max_sr': round(e_max_sr, 4),
        'p_value': round(p_value, 4),
        'is_significant': p_value < 0.05,
        'sr_std': round(sr_std, 4),
    }


class CPCV:
    """
    Combinatorial Purged Cross-Validation.

    Generates all C(N, N-k) train/test splits from N groups,
    with purging (embargo) between train and test to prevent leakage.

    Used to estimate the Probability of Backtest Overfitting (PBO):
    the fraction of combinatorial splits where the IS-optimal strategy
    underperforms OOS.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2,
                 purge_groups: int = 1):
        """
        Args:
            n_groups: Total number of temporal groups to split data into
            n_test_groups: Number of groups used for testing in each split
            purge_groups: Number of groups to purge between train/test boundaries
        """
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_groups = purge_groups

    def get_splits(self, n_bars: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate all combinatorial train/test index splits.

        Returns list of (train_indices, test_indices) tuples.
        """
        group_size = n_bars // self.n_groups
        if group_size < 10:
            logger.warning(f"Group size {group_size} is very small for {n_bars} bars / {self.n_groups} groups")

        # Create group boundaries
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_groups - 1 else n_bars
            groups.append(np.arange(start, end))

        # Generate all combinations of test groups
        splits = []
        for test_group_ids in combinations(range(self.n_groups), self.n_test_groups):
            test_set = set(test_group_ids)

            # Purge: exclude groups adjacent to test groups
            purge_set = set()
            for tg in test_group_ids:
                for p in range(1, self.purge_groups + 1):
                    if tg - p >= 0:
                        purge_set.add(tg - p)
                    if tg + p < self.n_groups:
                        purge_set.add(tg + p)

            # Train = all groups not in test or purge
            train_indices = []
            test_indices = []
            for i in range(self.n_groups):
                if i in test_set:
                    test_indices.append(groups[i])
                elif i not in purge_set:
                    train_indices.append(groups[i])

            if train_indices and test_indices:
                splits.append((
                    np.concatenate(train_indices),
                    np.concatenate(test_indices),
                ))

        logger.info(f"CPCV: {len(splits)} combinatorial splits from C({self.n_groups}, {self.n_test_groups})")
        return splits

    def compute_pbo(self, is_performances: np.ndarray, oos_performances: np.ndarray) -> Dict:
        """
        Compute the Probability of Backtest Overfitting (PBO).

        For each combinatorial split, the IS-optimal strategy is identified,
        and we check whether it also performs well OOS. PBO is the fraction
        of splits where the IS-best strategy ranks below median OOS.

        Args:
            is_performances: Array of shape (n_splits, n_strategies) with IS metrics
            oos_performances: Array of shape (n_splits, n_strategies) with OOS metrics

        Returns:
            dict with 'pbo', 'pbo_pct', 'n_splits', 'is_reliable'
        """
        n_splits = is_performances.shape[0]
        n_strategies = is_performances.shape[1]

        if n_splits < 2 or n_strategies < 2:
            return {'pbo': 0.5, 'pbo_pct': 50.0, 'n_splits': n_splits, 'is_reliable': False}

        overfit_count = 0
        logit_values = []

        for s in range(n_splits):
            # Find IS-optimal strategy
            is_best_idx = np.argmax(is_performances[s])

            # Rank of IS-best strategy in OOS (0 = worst, 1 = best)
            oos_rank = stats.rankdata(oos_performances[s])[is_best_idx]
            relative_rank = oos_rank / n_strategies

            # Overfit if IS-best ranks below median OOS
            if relative_rank <= 0.5:
                overfit_count += 1

            # Logit for distribution analysis
            if 0 < relative_rank < 1:
                logit_values.append(np.log(relative_rank / (1 - relative_rank)))

        pbo = overfit_count / n_splits

        return {
            'pbo': round(pbo, 4),
            'pbo_pct': round(pbo * 100, 1),
            'n_splits': n_splits,
            'n_strategies': n_strategies,
            'overfit_count': overfit_count,
            'is_reliable': n_splits >= 10 and n_strategies >= 3,
            'logit_mean': round(np.mean(logit_values), 4) if logit_values else 0.0,
        }


def minimum_backtest_length(
    observed_sr: float,
    num_trials: int,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> int:
    """
    Compute the Minimum Backtest Length (MinBTL) needed for a given
    Sharpe ratio to be statistically significant at the specified confidence level.

    This prevents drawing conclusions from backtests that are too short.

    Args:
        observed_sr: Annualized Sharpe ratio
        num_trials: Number of strategies/parameter sets tested
        confidence: Confidence level (default 0.95)
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis

    Returns:
        Minimum number of return observations needed
    """
    if num_trials <= 1 or observed_sr <= 0:
        return 252  # default 1 year

    z_score = stats.norm.ppf(confidence)

    # Expected max SR from multiple testing
    log_n = np.log(num_trials)
    e_max_sr = np.sqrt(2 * log_n) - (np.log(np.pi) + np.log(log_n)) / (2 * np.sqrt(2 * log_n))

    # Solve for n: (SR - E[max(SR)]) / sqrt(Var(SR)) >= z
    # Where Var(SR) ≈ (1 + 0.5*SR^2 + ...) / (n-1)
    excess_kurt = kurtosis - 3.0
    numerator_factor = 1.0 + 0.5 * observed_sr**2 - skewness * observed_sr + excess_kurt / 4.0 * observed_sr**2

    sr_diff = observed_sr - e_max_sr
    if sr_diff <= 0:
        return 99999  # SR doesn't exceed expected max; need infinite data

    min_n = int(np.ceil(z_score**2 * numerator_factor / sr_diff**2)) + 1
    return max(min_n, 30)  # minimum 30 observations
