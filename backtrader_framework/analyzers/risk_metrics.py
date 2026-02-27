"""
Institutional-grade risk metrics analyzers for Backtrader.

Provides Sortino, Calmar, Omega ratios and buy-and-hold benchmark comparison
that are missing from Backtrader's built-in analyzers.
"""

import backtrader as bt
import numpy as np
import math


class SortinoRatio(bt.Analyzer):
    """
    Sortino Ratio: like Sharpe but only penalizes downside volatility.
    More appropriate for strategies with asymmetric return distributions.

    Formula: (mean_return - risk_free_rate) / downside_deviation
    """
    params = (
        ('risk_free_rate', 0.0),
        ('period', 252),  # annualization factor (252 trading days)
    )

    def start(self):
        self.daily_returns = []
        self.prev_value = self.strategy.broker.getvalue()

    def next(self):
        current_value = self.strategy.broker.getvalue()
        ret = (current_value - self.prev_value) / self.prev_value if self.prev_value > 0 else 0.0
        self.daily_returns.append(ret)
        self.prev_value = current_value

    def get_analysis(self):
        if len(self.daily_returns) < 2:
            return {'sortino_ratio': 0.0, 'downside_deviation': 0.0}

        returns = np.array(self.daily_returns)
        mean_ret = np.mean(returns)

        # Downside deviation: std of returns below target (risk-free rate)
        daily_rf = self.p.risk_free_rate / self.p.period
        downside = returns[returns < daily_rf] - daily_rf
        downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-10

        annualized_excess = (mean_ret - daily_rf) * self.p.period
        annualized_downside = downside_std * np.sqrt(self.p.period)

        sortino = annualized_excess / annualized_downside if annualized_downside > 1e-10 else 0.0

        return {
            'sortino_ratio': round(sortino, 4),
            'downside_deviation': round(annualized_downside, 6),
        }


class CalmarRatio(bt.Analyzer):
    """
    Calmar Ratio: annualized return / maximum drawdown.
    Measures return per unit of drawdown risk.
    """
    params = (
        ('period', 252),
    )

    def start(self):
        self.start_value = self.strategy.broker.getvalue()
        self.peak_value = self.start_value
        self.max_drawdown = 0.0
        self.values = []

    def next(self):
        current = self.strategy.broker.getvalue()
        self.values.append(current)
        self.peak_value = max(self.peak_value, current)
        dd = (self.peak_value - current) / self.peak_value if self.peak_value > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, dd)

    def get_analysis(self):
        if len(self.values) < 2 or self.start_value <= 0:
            return {'calmar_ratio': 0.0, 'max_drawdown_pct': 0.0, 'annualized_return': 0.0}

        total_return = (self.values[-1] - self.start_value) / self.start_value
        n_periods = len(self.values)
        annualized = (1 + total_return) ** (self.p.period / n_periods) - 1 if n_periods > 0 else 0.0

        calmar = annualized / self.max_drawdown if self.max_drawdown > 1e-10 else 0.0

        return {
            'calmar_ratio': round(calmar, 4),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'annualized_return': round(annualized * 100, 2),
        }


class OmegaRatio(bt.Analyzer):
    """
    Omega Ratio: probability-weighted ratio of gains vs losses above/below threshold.
    More comprehensive than Sharpe as it considers the entire return distribution.

    Formula: sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))
    """
    params = (
        ('threshold', 0.0),  # return threshold (typically 0 or risk-free rate)
    )

    def start(self):
        self.daily_returns = []
        self.prev_value = self.strategy.broker.getvalue()

    def next(self):
        current = self.strategy.broker.getvalue()
        ret = (current - self.prev_value) / self.prev_value if self.prev_value > 0 else 0.0
        self.daily_returns.append(ret)
        self.prev_value = current

    def get_analysis(self):
        if len(self.daily_returns) < 2:
            return {'omega_ratio': 1.0}

        returns = np.array(self.daily_returns)
        gains = np.sum(np.maximum(returns - self.p.threshold, 0))
        losses = np.sum(np.maximum(self.p.threshold - returns, 0))

        omega = (gains / losses) if losses > 1e-10 else float('inf')

        return {'omega_ratio': round(omega, 4)}


class BenchmarkComparison(bt.Analyzer):
    """
    Buy-and-hold benchmark comparison.
    Computes alpha and tracking metrics against a simple buy-and-hold strategy.
    """

    def start(self):
        self.start_value = self.strategy.broker.getvalue()
        self.start_price = self.strategy.data.close[0]
        self.strategy_values = []
        self.benchmark_values = []

    def next(self):
        current_value = self.strategy.broker.getvalue()
        current_price = self.strategy.data.close[0]

        self.strategy_values.append(current_value)
        # Benchmark: what if we just held the asset from day 1
        bm_value = self.start_value * (current_price / self.start_price) if self.start_price > 0 else self.start_value
        self.benchmark_values.append(bm_value)

    def get_analysis(self):
        if len(self.strategy_values) < 2:
            return {'alpha_pct': 0.0, 'strategy_return_pct': 0.0, 'benchmark_return_pct': 0.0,
                    'information_ratio': 0.0, 'tracking_error': 0.0}

        strat_ret = (self.strategy_values[-1] - self.start_value) / self.start_value
        bench_ret = (self.benchmark_values[-1] - self.start_value) / self.start_value
        alpha = strat_ret - bench_ret

        # Information ratio (alpha / tracking error)
        strat_daily = np.diff(self.strategy_values) / np.array(self.strategy_values[:-1])
        bench_daily = np.diff(self.benchmark_values) / np.array(self.benchmark_values[:-1])
        tracking_diff = strat_daily - bench_daily
        tracking_error = np.std(tracking_diff) * np.sqrt(252) if len(tracking_diff) > 1 else 1e-10
        info_ratio = (np.mean(tracking_diff) * 252) / tracking_error if tracking_error > 1e-10 else 0.0

        return {
            'strategy_return_pct': round(strat_ret * 100, 2),
            'benchmark_return_pct': round(bench_ret * 100, 2),
            'alpha_pct': round(alpha * 100, 2),
            'information_ratio': round(info_ratio, 4),
            'tracking_error': round(tracking_error, 4),
        }
