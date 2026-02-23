"""
Cross-Asset Robustness Analysis — test if a strategy's edge is structural.

Run the same strategy on BTC, ETH, and NQ, compare OOS performance metrics,
and determine whether the edge is structural (works across assets) or
asset-specific (fragile / curve-fit).

Usage:
    analyzer = CrossAssetAnalyzer()
    analyzer.load_results(filepaths)
    robustness = analyzer.analyze_strategy('SBS')
    print(robustness.robustness_grade, robustness.verdict)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .persistence import load_wfo_result, list_wfo_results, save_wfo_result


# ================================================================
#  DATA CLASSES
# ================================================================

@dataclass
class AssetResult:
    """Summary of one strategy on one asset."""
    strategy: str
    symbol: str
    timeframe: str
    filepath: str
    n_trades: int
    win_rate: float
    mean_r: float
    std_r: float
    sharpe: float               # sharpe_per_trade from oos_stats
    profit_factor: float
    expectancy: float
    max_drawdown_r: float
    overfit_ratio: Optional[float]
    mean_r_significant: bool
    regime_breakdown: Dict[str, Dict]
    direction_breakdown: Dict[str, Dict]
    monte_carlo: Optional[Dict]
    oos_equity: List[Dict]


@dataclass
class StrategyRobustness:
    """Cross-asset robustness assessment for one strategy."""
    strategy: str
    timeframe: str
    assets: List[str]
    asset_results: Dict[str, AssetResult]
    metrics_table: List[Dict]
    regime_comparison: Dict
    equity_correlation: Optional[pd.DataFrame]
    robustness_score: float
    robustness_grade: str
    verdict: str
    component_scores: Dict[str, float]


# ================================================================
#  GRADE / VERDICT MAPPINGS
# ================================================================

_GRADE_THRESHOLDS = [
    (80, 'A', "Structural edge — strategy works across multiple assets"),
    (60, 'B', "Likely structural — performs on most assets with some variation"),
    (40, 'C', "Mixed — works on some assets but not consistently"),
    (20, 'D', "Asset-specific — edge is mostly concentrated in one asset"),
    (0,  'F', "Not robust — no consistent edge across assets"),
]


def _score_to_grade(score: float) -> Tuple[str, str]:
    for threshold, grade, verdict in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade, verdict
    return 'F', _GRADE_THRESHOLDS[-1][2]


def _linear_score(value: float, best: float, worst: float) -> float:
    """Linear interpolation: returns 100 at best, 0 at worst, clamped [0, 100]."""
    if abs(best - worst) < 1e-12:
        return 50.0
    raw = (value - worst) / (best - worst) * 100
    return max(0.0, min(100.0, raw))


# ================================================================
#  CROSS-ASSET ANALYZER
# ================================================================

class CrossAssetAnalyzer:
    """Compare strategy performance across multiple assets."""

    def __init__(self):
        self._results: Dict[str, Dict[str, AssetResult]] = {}

    # ── Loading ─────────────────────────────────────────────────

    def load_results(self, filepaths: List[str]) -> None:
        """Load WFO results and extract AssetResult summaries."""
        self._results = {}

        for fp in filepaths:
            data = load_wfo_result(fp)
            oos = data.get('oos_stats', {})
            if not oos.get('valid', False):
                continue

            ar = AssetResult(
                strategy=data['strategy_name'],
                symbol=data['symbol'],
                timeframe=data['timeframe'],
                filepath=fp,
                n_trades=oos.get('n_trades', 0),
                win_rate=oos.get('win_rate', 0.0),
                mean_r=oos.get('mean_r', 0.0),
                std_r=oos.get('std_r', 0.0),
                sharpe=oos.get('sharpe_per_trade', 0.0),
                profit_factor=oos.get('profit_factor', 0.0),
                expectancy=oos.get('expectancy', 0.0),
                max_drawdown_r=oos.get('max_drawdown_r', 0.0),
                overfit_ratio=data.get('overfit_ratio'),
                mean_r_significant=oos.get('mean_r_significant', False),
                regime_breakdown=data.get('regime_analysis', {}),
                direction_breakdown=data.get('direction_analysis', {}),
                monte_carlo=data.get('monte_carlo') if isinstance(data.get('monte_carlo'), dict) else None,
                oos_equity=data.get('oos_equity', []),
            )

            strat = ar.strategy
            if strat not in self._results:
                self._results[strat] = {}
            self._results[strat][ar.symbol] = ar

    def get_loaded_strategies(self) -> List[str]:
        return list(self._results.keys())

    def get_loaded_assets(self, strategy: str) -> List[str]:
        return list(self._results.get(strategy, {}).keys())

    # ── Coverage ────────────────────────────────────────────────

    def get_coverage_matrix(self) -> Dict:
        """Return strategy x asset grid showing which combos have results."""
        all_saved = list_wfo_results()

        strategies = sorted(set(r['strategy'] for r in all_saved))
        assets = sorted(set(r['symbol'] for r in all_saved))

        coverage = {}
        for strat in strategies:
            coverage[strat] = {}
            for asset in assets:
                has = any(
                    r['strategy'] == strat and r['symbol'] == asset
                    for r in all_saved
                )
                coverage[strat][asset] = has

        # Determine missing combos (all strategies × all assets minus existing)
        missing = []
        for strat in strategies:
            for asset in assets:
                if not coverage[strat].get(asset, False):
                    # Infer timeframe from existing results for this strategy
                    existing_tf = next(
                        (r['timeframe'] for r in all_saved if r['strategy'] == strat),
                        '4h',
                    )
                    missing.append((strat, asset, existing_tf))

        return {
            'strategies': strategies,
            'assets': assets,
            'coverage': coverage,
            'missing': missing,
        }

    # ── Run Missing WFO ────────────────────────────────────────

    def run_missing_wfo(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        progress_callback=None,
    ) -> str:
        """Run WFO for a missing combo. Returns filepath of saved result."""
        from .strategy_adapters import ADAPTER_REGISTRY
        from .wfo_engine import RegimeAdaptiveWFO, WFOConfig, TransactionCosts

        if strategy not in ADAPTER_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy}")

        adapter = ADAPTER_REGISTRY[strategy]()
        config = WFOConfig.for_timeframe(timeframe, costs=TransactionCosts.for_asset(symbol))
        engine = RegimeAdaptiveWFO(adapter, config)

        result = engine.run(symbol, timeframe, progress_callback)

        if not result or not result.get('oos_stats', {}).get('valid', False):
            raise RuntimeError(
                f"WFO for {strategy}/{symbol}/{timeframe} produced no valid results"
            )

        filepath = save_wfo_result(result)

        # Add to loaded results
        oos = result.get('oos_stats', {})
        ar = AssetResult(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            filepath=filepath,
            n_trades=oos.get('n_trades', 0),
            win_rate=oos.get('win_rate', 0.0),
            mean_r=oos.get('mean_r', 0.0),
            std_r=oos.get('std_r', 0.0),
            sharpe=oos.get('sharpe_per_trade', 0.0),
            profit_factor=oos.get('profit_factor', 0.0),
            expectancy=oos.get('expectancy', 0.0),
            max_drawdown_r=oos.get('max_drawdown_r', 0.0),
            overfit_ratio=result.get('overfit_ratio'),
            mean_r_significant=oos.get('mean_r_significant', False),
            regime_breakdown=result.get('regime_analysis', {}),
            direction_breakdown=result.get('direction_analysis', {}),
            monte_carlo=result.get('monte_carlo') if isinstance(result.get('monte_carlo'), dict) else None,
            oos_equity=result.get('oos_equity', []),
        )
        if strategy not in self._results:
            self._results[strategy] = {}
        self._results[strategy][symbol] = ar

        return filepath

    # ── Analysis ────────────────────────────────────────────────

    def analyze_strategy(self, strategy: str) -> StrategyRobustness:
        """Compare one strategy across all loaded assets."""
        asset_results = self._results.get(strategy, {})
        if len(asset_results) < 2:
            raise ValueError(
                f"Need 2+ assets for {strategy}, have {list(asset_results.keys())}"
            )

        assets = sorted(asset_results.keys())
        timeframe = next(iter(asset_results.values())).timeframe

        # 1. Metrics table
        metrics_table = self._build_metrics_table(asset_results)

        # 2. Regime comparison
        regime_comp = self._regime_comparison(asset_results)

        # 3. Equity curve correlation
        equity_corr = self._align_equity_curves(asset_results)

        # 4. Robustness score
        score, component_scores = self._compute_robustness_score(asset_results)

        # 5. Grade + verdict
        grade, verdict = _score_to_grade(score)

        return StrategyRobustness(
            strategy=strategy,
            timeframe=timeframe,
            assets=assets,
            asset_results=asset_results,
            metrics_table=metrics_table,
            regime_comparison=regime_comp,
            equity_correlation=equity_corr,
            robustness_score=score,
            robustness_grade=grade,
            verdict=verdict,
            component_scores=component_scores,
        )

    def analyze_all(self) -> Dict:
        """Run analysis for all strategies with 2+ assets loaded."""
        results = {}
        for strat, asset_map in self._results.items():
            if len(asset_map) >= 2:
                results[strat] = self.analyze_strategy(strat)

        # Cross-strategy summary
        if not results:
            return {
                'strategies': {},
                'cross_strategy': {
                    'best_overall': None,
                    'most_robust': None,
                    'recommendations': ["No strategies with 2+ assets. Run more WFO combos."],
                },
            }

        # Best overall = highest average Sharpe across assets
        avg_sharpes = {}
        for strat, rob in results.items():
            sharpes = [ar.sharpe for ar in rob.asset_results.values()]
            avg_sharpes[strat] = np.mean(sharpes)
        best_overall = max(avg_sharpes, key=avg_sharpes.get)

        # Most robust = highest robustness score
        most_robust = max(results, key=lambda s: results[s].robustness_score)

        # Recommendations
        recommendations = self._generate_recommendations(results)

        return {
            'strategies': results,
            'cross_strategy': {
                'best_overall': best_overall,
                'most_robust': most_robust,
                'recommendations': recommendations,
            },
        }

    # ── Private Helpers ─────────────────────────────────────────

    def _build_metrics_table(self, asset_results: Dict[str, AssetResult]) -> List[Dict]:
        """Build side-by-side metrics rows for display."""
        metrics = [
            ('Trades', 'n_trades', '{:d}'),
            ('Win Rate', 'win_rate', '{:.1%}'),
            ('Mean R', 'mean_r', '{:+.4f}'),
            ('Std R', 'std_r', '{:.4f}'),
            ('Sharpe', 'sharpe', '{:.3f}'),
            ('Profit Factor', 'profit_factor', '{:.2f}'),
            ('Expectancy', 'expectancy', '{:.4f}'),
            ('Max DD (R)', 'max_drawdown_r', '{:.2f}'),
            ('Overfit Ratio', 'overfit_ratio', '{:.3f}'),
            ('Significant', 'mean_r_significant', '{}'),
        ]

        rows = []
        for label, attr, fmt in metrics:
            row = {'Metric': label}
            for symbol, ar in sorted(asset_results.items()):
                val = getattr(ar, attr)
                if val is None:
                    row[symbol] = '—'
                elif isinstance(val, bool):
                    row[symbol] = 'Yes' if val else 'No'
                elif isinstance(val, int):
                    row[symbol] = fmt.format(val)
                else:
                    try:
                        row[symbol] = fmt.format(val)
                    except (ValueError, TypeError):
                        row[symbol] = str(val)
            rows.append(row)

        return rows

    def _regime_comparison(self, asset_results: Dict[str, AssetResult]) -> Dict:
        """Build regime x asset performance table."""
        regimes = ['trending_up', 'trending_down', 'ranging', 'volatile']
        result = {}

        for regime in regimes:
            result[regime] = {}
            for symbol, ar in sorted(asset_results.items()):
                rb = ar.regime_breakdown.get(regime, {})
                result[regime][symbol] = {
                    'n_trades': rb.get('n_trades', 0),
                    'win_rate': rb.get('win_rate', 0.0),
                    'mean_r': rb.get('mean_r', 0.0),
                    'total_r': rb.get('total_r', 0.0),
                }

        return result

    def _align_equity_curves(
        self, asset_results: Dict[str, AssetResult],
    ) -> Optional[pd.DataFrame]:
        """Resample OOS equity to daily, align by date, return correlation matrix."""
        daily_series = {}

        for symbol, ar in asset_results.items():
            if not ar.oos_equity:
                continue

            times = pd.to_datetime([e['time'] for e in ar.oos_equity])
            cum_r = pd.Series(
                [e['cumulative_r'] for e in ar.oos_equity],
                index=times,
                dtype=float,
            )

            # Deduplicate timestamps
            cum_r = cum_r[~cum_r.index.duplicated(keep='last')].sort_index()

            # Resample to daily, forward fill
            daily_cum = cum_r.resample('D').last().ffill().fillna(0.0)
            daily_ret = daily_cum.diff().fillna(0.0)
            daily_series[symbol] = daily_ret

        if len(daily_series) < 2:
            return None

        df = pd.DataFrame(daily_series).dropna()
        if len(df) < 10:
            return None

        return df.corr()

    def _compute_robustness_score(
        self, asset_results: Dict[str, AssetResult],
    ) -> Tuple[float, Dict]:
        """Compute 0-100 robustness score from 5 sub-components."""
        n_assets = len(asset_results)
        if n_assets < 2:
            return 0.0, {}

        ars = list(asset_results.values())

        # 1. Metric Consistency (30%) — low CV of key metrics across assets
        consistency_score = self._metric_consistency_score(ars)

        # 2. Multi-Asset Profitability (25%) — fraction of assets with mean_r > 0
        profitable_count = sum(1 for ar in ars if ar.mean_r > 0)
        profitability_score = (profitable_count / n_assets) * 100

        # 3. Statistical Significance (20%) — fraction with significant mean_r
        sig_count = sum(1 for ar in ars if ar.mean_r_significant)
        significance_score = (sig_count / n_assets) * 100

        # 4. Regime Stability (15%) — do assets share the same best regime?
        regime_score = self._regime_stability_score(ars)

        # 5. Overfit Consistency (10%) — mean overfit ratio scaled
        overfit_ratios = [
            ar.overfit_ratio for ar in ars
            if ar.overfit_ratio is not None and np.isfinite(ar.overfit_ratio)
        ]
        if overfit_ratios:
            # Clamp to [0, 2] range before scoring
            clamped = [max(0.0, min(2.0, abs(r))) for r in overfit_ratios]
            mean_or = np.mean(clamped)
            # Score: 100 if mean >= 0.5, 0 if mean <= 0.1, linear between
            overfit_score = _linear_score(mean_or, best=0.5, worst=0.1)
        else:
            overfit_score = 50.0  # neutral if no data

        component_scores = {
            'metric_consistency': consistency_score,
            'multi_asset_profitability': profitability_score,
            'statistical_significance': significance_score,
            'regime_stability': regime_score,
            'overfit_consistency': overfit_score,
        }

        total = (
            0.30 * consistency_score
            + 0.25 * profitability_score
            + 0.20 * significance_score
            + 0.15 * regime_score
            + 0.10 * overfit_score
        )

        return total, component_scores

    def _metric_consistency_score(self, ars: List[AssetResult]) -> float:
        """Score 0-100 based on coefficient of variation of key metrics."""
        cvs = []
        for attr in ('sharpe', 'mean_r', 'win_rate'):
            values = [getattr(ar, attr) for ar in ars]
            mean = np.mean(values)
            std = np.std(values)
            if abs(mean) > 1e-10:
                cvs.append(abs(std / mean))
            else:
                # If mean is near zero, metrics are inconsistent
                cvs.append(1.0 if std > 1e-10 else 0.0)

        avg_cv = np.mean(cvs)
        # Score: 100 if avg_cv < 0.15, 0 if avg_cv > 1.0
        return _linear_score(avg_cv, best=0.15, worst=1.0)

    def _regime_stability_score(self, ars: List[AssetResult]) -> float:
        """Score 0-100 based on whether assets share the same best regime."""
        best_regimes = []
        for ar in ars:
            if not ar.regime_breakdown:
                continue
            # Find regime with highest mean_r
            best = max(
                ar.regime_breakdown.items(),
                key=lambda kv: kv[1].get('mean_r', -999),
            )
            best_regimes.append(best[0])

        if len(best_regimes) < 2:
            return 50.0

        # Count most common best regime
        from collections import Counter
        counts = Counter(best_regimes)
        most_common_count = counts.most_common(1)[0][1]
        fraction_agreeing = most_common_count / len(best_regimes)

        return fraction_agreeing * 100

    def _generate_recommendations(
        self, results: Dict[str, StrategyRobustness],
    ) -> List[str]:
        """Generate actionable recommendations from robustness analysis."""
        recs = []

        for strat, rob in sorted(results.items(), key=lambda x: -x[1].robustness_score):
            if rob.robustness_grade in ('A', 'B'):
                assets_str = ', '.join(rob.assets)
                recs.append(
                    f"{strat}: Deploy across {assets_str} — "
                    f"grade {rob.robustness_grade} ({rob.robustness_score:.0f}/100)"
                )
            elif rob.robustness_grade == 'C':
                # Find the best asset
                best_asset = max(
                    rob.asset_results.items(),
                    key=lambda kv: kv[1].sharpe,
                )[0]
                recs.append(
                    f"{strat}: Consider {best_asset} only — "
                    f"inconsistent on other assets (grade {rob.robustness_grade})"
                )
            else:
                best_asset = max(
                    rob.asset_results.items(),
                    key=lambda kv: kv[1].sharpe,
                )[0]
                recs.append(
                    f"{strat}: Restrict to {best_asset} — "
                    f"not robust across assets (grade {rob.robustness_grade})"
                )

        # Diversification note
        if len(results) >= 2:
            grades = [(s, r.robustness_grade) for s, r in results.items()]
            robust = [s for s, g in grades if g in ('A', 'B')]
            if len(robust) >= 2:
                recs.append(
                    f"Portfolio opportunity: combine {', '.join(robust)} for diversification"
                )

        return recs
