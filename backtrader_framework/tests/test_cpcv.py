"""Tests for CPCV and Deflated Sharpe Ratio"""
import pytest
import numpy as np


class TestDeflatedSharpe:
    def test_more_trials_lowers_significance(self):
        """More trials should make the same SR less significant"""
        from backtrader_framework.optimization.cpcv import deflated_sharpe_ratio

        few_trials = deflated_sharpe_ratio(1.5, num_trials=5, n_returns=252)
        many_trials = deflated_sharpe_ratio(1.5, num_trials=1000, n_returns=252)
        assert few_trials['p_value'] < many_trials['p_value']

    def test_higher_sr_more_significant(self):
        """Higher SR should be more significant"""
        from backtrader_framework.optimization.cpcv import deflated_sharpe_ratio

        low_sr = deflated_sharpe_ratio(0.5, num_trials=100, n_returns=252)
        high_sr = deflated_sharpe_ratio(3.0, num_trials=100, n_returns=252)
        assert high_sr['p_value'] < low_sr['p_value']


class TestCPCV:
    def test_split_generation(self):
        """Should generate correct number of splits"""
        from backtrader_framework.optimization.cpcv import CPCV

        cpcv = CPCV(n_groups=6, n_test_groups=2, purge_groups=0)
        splits = cpcv.get_splits(n_bars=600)
        # C(6,2) = 15 splits
        assert len(splits) == 15

    def test_no_overlap_with_purge(self):
        """Train and test indices should not overlap"""
        from backtrader_framework.optimization.cpcv import CPCV

        cpcv = CPCV(n_groups=6, n_test_groups=2, purge_groups=1)
        splits = cpcv.get_splits(n_bars=600)
        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_pbo_random_strategies(self):
        """Random strategies should have PBO near 0.5"""
        from backtrader_framework.optimization.cpcv import CPCV

        np.random.seed(42)
        cpcv = CPCV(n_groups=6, n_test_groups=2)
        n_splits = 15
        n_strategies = 10
        is_perf = np.random.randn(n_splits, n_strategies)
        oos_perf = np.random.randn(n_splits, n_strategies)
        result = cpcv.compute_pbo(is_perf, oos_perf)
        # PBO should be roughly 0.5 for random (+/-0.3 for small sample)
        assert 0.1 < result['pbo'] < 0.9


class TestMinimumBacktestLength:
    def test_more_trials_needs_more_data(self):
        """More trials should require longer backtest"""
        from backtrader_framework.optimization.cpcv import minimum_backtest_length

        few = minimum_backtest_length(1.0, num_trials=5)
        many = minimum_backtest_length(1.0, num_trials=1000)
        assert many > few
