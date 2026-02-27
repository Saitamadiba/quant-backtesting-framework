"""
Portfolio-Level Optimization — combine multiple WFO results.

Loads existing WFO result JSONs, aligns equity curves to daily resolution,
computes pairwise correlation, and optimizes capital allocation across
strategies using risk parity, mean-variance, or Kelly criterion.

Usage:
    optimizer = PortfolioOptimizer(fractional_kelly=0.25)
    result = optimizer.optimize(
        filepaths=['wfo_SBS_BTC_4h.json', 'wfo_FVG_BTC_4h.json'],
        method='risk_parity',
    )
    print(result.portfolio_stats)
    print(result.weights)
    print(result.correlation_matrix)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .persistence import load_wfo_result, list_wfo_results


# ================================================================
#  DATA CLASSES
# ================================================================

@dataclass
class PortfolioComponent:
    """One strategy/symbol/timeframe loaded from a WFO result."""
    label: str               # e.g. "SBS_BTC_4h"
    strategy: str
    symbol: str
    timeframe: str
    filepath: str
    oos_equity: List[Dict]   # raw from JSON [{time, r, cumulative_r, ...}, ...]
    oos_stats: Dict
    daily_returns: Optional[pd.Series] = None


@dataclass
class PortfolioResult:
    """Complete portfolio optimization output."""
    components: List[PortfolioComponent]
    correlation_matrix: pd.DataFrame
    allocation_method: str
    weights: Dict[str, float]
    combined_equity: List[Dict]
    portfolio_stats: Dict
    component_stats: Dict[str, Dict]
    diversification_ratio: float
    kelly_fractions: Optional[Dict[str, float]] = None
    monte_carlo: Optional[Dict] = None


# ================================================================
#  PORTFOLIO OPTIMIZER
# ================================================================

class PortfolioOptimizer:
    """Combine multiple WFO results into an optimally allocated portfolio."""

    def __init__(self, fractional_kelly: float = 0.25):
        self.fractional_kelly = fractional_kelly
        self.components: List[PortfolioComponent] = []
        self._daily_returns_df: Optional[pd.DataFrame] = None

    # ── Loading ─────────────────────────────────────────────────

    def load_results(self, filepaths: List[str]) -> None:
        """Load multiple WFO result JSONs and align equity curves."""
        self.components = []
        for fp in filepaths:
            data = load_wfo_result(fp)
            oos_eq = data.get('oos_equity', [])
            if not oos_eq:
                continue

            label = f"{data['strategy_name']}_{data['symbol']}_{data['timeframe']}"
            self.components.append(PortfolioComponent(
                label=label,
                strategy=data['strategy_name'],
                symbol=data['symbol'],
                timeframe=data['timeframe'],
                filepath=fp,
                oos_equity=oos_eq,
                oos_stats=data.get('oos_stats', {}),
            ))

        if len(self.components) < 2:
            raise ValueError(
                f"Need at least 2 components with OOS trades, got {len(self.components)}"
            )

        # Deduplicate labels (add suffix if same strategy/symbol/timeframe)
        seen = {}
        for comp in self.components:
            if comp.label in seen:
                seen[comp.label] += 1
                comp.label = f"{comp.label}_{seen[comp.label]}"
            else:
                seen[comp.label] = 0

        self._align_equity_curves()

    # ── Equity Curve Alignment ──────────────────────────────────

    def _align_equity_curves(self) -> None:
        """Resample all equity curves to daily, compute daily returns, align dates."""
        daily_series = {}

        for comp in self.components:
            # Build cumulative R time series
            times = pd.to_datetime([e['time'] for e in comp.oos_equity])
            cum_r = pd.Series(
                [e['cumulative_r'] for e in comp.oos_equity],
                index=times,
                dtype=float,
            )

            # Remove duplicate timestamps (keep last)
            cum_r = cum_r[~cum_r.index.duplicated(keep='last')]
            cum_r = cum_r.sort_index()

            # Resample to daily (end of day), forward-fill flat between trades
            daily_cum = cum_r.resample('D').last().ffill()

            # Fill leading NaN with 0 (before first trade, equity = 0)
            daily_cum = daily_cum.fillna(0.0)

            # Daily returns = first difference of cumulative R
            daily_ret = daily_cum.diff().fillna(0.0)

            daily_series[comp.label] = daily_ret

        # Aligned DataFrame — inner join on dates (overlapping period only)
        self._daily_returns_df = pd.DataFrame(daily_series).dropna()

        if len(self._daily_returns_df) < 10:
            raise ValueError(
                f"Insufficient overlapping data: {len(self._daily_returns_df)} days. "
                "Ensure WFO results cover a common date range."
            )

        # Store aligned daily returns back on components
        for comp in self.components:
            if comp.label in self._daily_returns_df.columns:
                comp.daily_returns = self._daily_returns_df[comp.label]

    # ── Correlation ─────────────────────────────────────────────

    def compute_correlation_matrix(self) -> pd.DataFrame:
        """Pairwise Pearson correlation of daily R returns."""
        if self._daily_returns_df is None or self._daily_returns_df.empty:
            raise ValueError("No aligned data. Call load_results() first.")
        return self._daily_returns_df.corr()

    # ── Allocation Methods ──────────────────────────────────────

    def equal_weight_allocation(self) -> Dict[str, float]:
        """Baseline: equal allocation across all components."""
        n = len(self.components)
        return {c.label: 1.0 / n for c in self.components}

    def risk_parity_allocation(self) -> Dict[str, float]:
        """Inverse-volatility weighting: weight_i = (1/std_i) / sum(1/std_j)."""
        inv_vols = {}
        for comp in self.components:
            daily_std = comp.daily_returns.std() if comp.daily_returns is not None else 0.0
            inv_vols[comp.label] = 1.0 / daily_std if daily_std > 0 else 0.0

        total = sum(inv_vols.values())
        if total > 0:
            return {k: v / total for k, v in inv_vols.items()}
        return self.equal_weight_allocation()

    def kelly_allocation(self, risk_per_trade: float = 0.01) -> Dict[str, float]:
        """
        Kelly criterion: f* = mean(capital_returns) / var(capital_returns).

        Kelly requires capital-based returns, not raw R-multiples.
        If R-multiples are available, convert via:
            capital_return = r_multiple * risk_per_trade
        where risk_per_trade is the fraction of capital risked per trade
        (default 1% = 0.01).

        Apply fractional Kelly for safety. Negative Kelly -> 0.
        Normalize to sum to 1.
        """
        raw_kelly = {}
        for comp in self.components:
            r_values = [e['r'] for e in comp.oos_equity]
            if not r_values:
                raw_kelly[comp.label] = 0.0
                continue

            # Convert R-multiples to capital returns
            capital_returns = np.array(r_values) * risk_per_trade

            mean_ret = np.mean(capital_returns)
            var_ret = np.var(capital_returns, ddof=1)
            if var_ret > 0 and mean_ret > 0:
                f_star = mean_ret / var_ret
                raw_kelly[comp.label] = f_star * self.fractional_kelly
            else:
                raw_kelly[comp.label] = 0.0

        total = sum(raw_kelly.values())
        if total > 0:
            return {k: v / total for k, v in raw_kelly.items()}
        # All components have negative edge — fall back to equal weight
        return self.equal_weight_allocation()

    def mean_variance_allocation(
        self, target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Mean-variance optimization (Markowitz) with Ledoit-Wolf shrinkage.

        Uses Ledoit-Wolf shrinkage covariance estimator for better
        conditioning of the covariance matrix, especially with limited
        samples or many components.

        If target_return is None, maximize Sharpe ratio (tangent portfolio).
        Constraints: sum(w) = 1, w_i >= 0 (long-only).
        """
        from sklearn.covariance import LedoitWolf

        df = self._daily_returns_df
        n = len(self.components)
        labels = [c.label for c in self.components]

        mu = df.mean().values
        # Ledoit-Wolf shrinkage covariance estimator
        lw = LedoitWolf().fit(df.dropna().values)
        cov = pd.DataFrame(
            lw.covariance_, index=df.columns, columns=df.columns,
        ).values
        # Regularize for numerical stability
        cov += np.eye(n) * 1e-6

        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n

        if target_return is not None:
            # Minimize variance for target return
            def objective(w):
                return w @ cov @ w

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: w @ mu - target_return},
            ]
        else:
            # Maximize Sharpe: minimize -return/volatility
            def objective(w):
                port_ret = w @ mu
                port_vol = np.sqrt(w @ cov @ w)
                if port_vol < 1e-10:
                    return 1e10
                return -port_ret / port_vol

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        from scipy import optimize as scipy_optimize
        result = scipy_optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
        )

        if result.success:
            weights = result.x
            weights = np.where(weights < 0.001, 0.0, weights)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            return dict(zip(labels, weights.tolist()))

        # Fallback to equal weight
        return self.equal_weight_allocation()

    # ── Portfolio Combination ───────────────────────────────────

    def combine_portfolio(
        self,
        weights: Dict[str, float],
        allocation_method: str = 'custom',
        run_monte_carlo: bool = True,
    ) -> PortfolioResult:
        """Combine component equity curves using given weights. Returns PortfolioResult."""
        df = self._daily_returns_df
        labels = [c.label for c in self.components]
        w = np.array([weights.get(label, 0.0) for label in labels])

        # 1. Weighted daily portfolio returns
        weighted_daily = df.values @ w
        portfolio_daily = pd.Series(weighted_daily, index=df.index)
        portfolio_cum = portfolio_daily.cumsum()

        # 2. Build combined equity time series
        combined_equity = []
        for date, daily_r in portfolio_daily.items():
            combined_equity.append({
                'time': str(date.date()) if hasattr(date, 'date') else str(date),
                'r': float(daily_r),
                'cumulative_r': float(portfolio_cum[date]),
            })

        # 3. Portfolio statistics
        port_mean = float(portfolio_daily.mean())
        port_std = float(portfolio_daily.std(ddof=1))
        port_sharpe_daily = port_mean / port_std if port_std > 0 else 0.0
        port_sharpe_annual = port_sharpe_daily * np.sqrt(252)

        # Max drawdown
        cum_arr = portfolio_cum.values
        running_max = np.maximum.accumulate(cum_arr)
        drawdowns = running_max - cum_arr
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        total_r = float(portfolio_cum.iloc[-1]) if len(portfolio_cum) > 0 else 0.0

        portfolio_stats = {
            'total_r': total_r,
            'daily_mean_r': port_mean,
            'daily_std_r': port_std,
            'sharpe_daily': port_sharpe_daily,
            'sharpe_annual': port_sharpe_annual,
            'max_drawdown_r': max_dd,
            'n_days': len(portfolio_daily),
            'n_components': len(self.components),
            'date_start': str(df.index[0].date()),
            'date_end': str(df.index[-1].date()),
        }

        # 4. Per-component statistics
        component_stats = {}
        for comp in self.components:
            r_vals = [e['r'] for e in comp.oos_equity]
            comp_daily = comp.daily_returns
            comp_std = float(comp_daily.std()) if comp_daily is not None else 0.0
            comp_mean = float(comp_daily.mean()) if comp_daily is not None else 0.0
            comp_sharpe = comp_mean / comp_std if comp_std > 0 else 0.0

            component_stats[comp.label] = {
                'n_trades': len(comp.oos_equity),
                'total_r': float(comp.oos_equity[-1]['cumulative_r']) if comp.oos_equity else 0.0,
                'mean_r_per_trade': float(np.mean(r_vals)) if r_vals else 0.0,
                'std_r_per_trade': float(np.std(r_vals, ddof=1)) if len(r_vals) > 1 else 0.0,
                'daily_sharpe': comp_sharpe,
                'annual_sharpe': comp_sharpe * np.sqrt(252),
                'weight': weights.get(comp.label, 0.0),
                'strategy': comp.strategy,
                'symbol': comp.symbol,
                'timeframe': comp.timeframe,
            }

        # 5. Diversification ratio
        weighted_sharpe_sum = sum(
            weights.get(c.label, 0) * component_stats[c.label]['daily_sharpe']
            for c in self.components
        )
        if abs(weighted_sharpe_sum) > 1e-10:
            diversification_ratio = port_sharpe_daily / weighted_sharpe_sum
        else:
            diversification_ratio = 1.0

        # 6. Correlation matrix
        corr = self.compute_correlation_matrix()

        # 7. Kelly fractions (informational)
        kelly = self.kelly_allocation()

        # 8. Monte Carlo on combined portfolio
        mc_result = None
        if run_monte_carlo and len(portfolio_daily) >= 20:
            mc_result = self._portfolio_monte_carlo(portfolio_daily.values)

        return PortfolioResult(
            components=self.components,
            correlation_matrix=corr,
            allocation_method=allocation_method,
            weights=weights,
            combined_equity=combined_equity,
            portfolio_stats=portfolio_stats,
            component_stats=component_stats,
            diversification_ratio=diversification_ratio,
            kelly_fractions=kelly,
            monte_carlo=mc_result,
        )

    # ── Monte Carlo ─────────────────────────────────────────────

    def _portfolio_monte_carlo(
        self,
        daily_returns: np.ndarray,
        n_resamples: int = 5000,
        confidence: float = 0.95,
    ) -> Dict:
        """Bootstrap Monte Carlo on portfolio daily returns."""
        n = len(daily_returns)
        if n < 10:
            return {'valid': False, 'reason': f'Insufficient data: {n} days'}

        alpha = 1 - confidence
        lo_pct = 100 * alpha / 2
        hi_pct = 100 * (1 - alpha / 2)

        final_rs = np.empty(n_resamples)
        max_dds = np.empty(n_resamples)
        sharpes = np.empty(n_resamples)

        for i in range(n_resamples):
            sample = daily_returns[np.random.randint(0, n, size=n)]
            cum = np.cumsum(sample)
            final_rs[i] = cum[-1]

            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            max_dds[i] = np.max(dd) if len(dd) > 0 else 0.0

            std = np.std(sample, ddof=1)
            sharpes[i] = np.mean(sample) / std if std > 0 else 0.0

        return {
            'valid': True,
            'n_resamples': n_resamples,
            'confidence': confidence,
            'p_profitable': float(np.mean(final_rs > 0)),
            'final_r_ci': [
                float(np.percentile(final_rs, lo_pct)),
                float(np.percentile(final_rs, hi_pct)),
            ],
            'max_dd_ci': [
                float(np.percentile(max_dds, lo_pct)),
                float(np.percentile(max_dds, hi_pct)),
            ],
            'sharpe_ci': [
                float(np.percentile(sharpes, lo_pct)),
                float(np.percentile(sharpes, hi_pct)),
            ],
            'median_final_r': float(np.median(final_rs)),
            'pct_5_final_r': float(np.percentile(final_rs, 5)),
            'pct_95_final_r': float(np.percentile(final_rs, 95)),
            'pct_95_max_dd': float(np.percentile(max_dds, 95)),
        }

    # ── Convenience Entry Points ────────────────────────────────

    def optimize(
        self,
        filepaths: List[str],
        method: str = 'risk_parity',
    ) -> PortfolioResult:
        """
        End-to-end: load results, compute allocation, combine portfolio.

        method: 'equal', 'risk_parity', 'kelly', 'mean_variance', 'max_sharpe'
        """
        self.load_results(filepaths)

        method_map = {
            'equal': self.equal_weight_allocation,
            'risk_parity': self.risk_parity_allocation,
            'kelly': self.kelly_allocation,
            'mean_variance': lambda: self.mean_variance_allocation(target_return=None),
            'max_sharpe': lambda: self.mean_variance_allocation(target_return=None),
        }

        alloc_fn = method_map.get(method, self.risk_parity_allocation)
        weights = alloc_fn()

        return self.combine_portfolio(weights, allocation_method=method)

    def compare_all_methods(self, filepaths: List[str]) -> Dict[str, PortfolioResult]:
        """Run all allocation methods and return results for comparison."""
        self.load_results(filepaths)

        methods = {
            'equal': self.equal_weight_allocation,
            'risk_parity': self.risk_parity_allocation,
            'kelly': self.kelly_allocation,
            'max_sharpe': lambda: self.mean_variance_allocation(target_return=None),
        }

        results = {}
        for name, alloc_fn in methods.items():
            weights = alloc_fn()
            results[name] = self.combine_portfolio(
                weights, allocation_method=name, run_monte_carlo=False,
            )

        return results
