"""
Quant Skills — Reusable quantitative analysis modules.

Standalone analytical skill modules derived from the Quant Guild Library
(stochastic calculus, option pricing, Monte Carlo, statistical methods,
portfolio theory, trading strategy, ML, and market structure).

Each skill accepts trade data in multiple formats (TradeResult lists,
oos_equity dicts, or raw R-multiple arrays) and returns JSON-serializable
result dictionaries for dashboard integration.

Skills:
    EdgeExistenceValidator    — Permutation tests, power analysis, multi-test correction
    StationarityAnalyzer      — Kalman filter, CUSUM, KS distributional stability
    ErgodicityAnalyzer        — Geometric growth, Kelly with uncertainty, ruin probability
    TailRiskAnalyzer          — Fat tails, Hill estimator, GARCH, VaR comparison
    RegimeEdgeDecomposer      — HMM regime detection, per-regime edge estimation
    SignalDecayAnalyzer       — MFE/MAE time profiles, optimal holding, signal autocorrelation
    StrategyAuditor           — Orchestrator that runs all skills and produces tradability score
"""

from .base import TradeData, SkillResult
from .edge_existence import EdgeExistenceValidator
from .stationarity import StationarityAnalyzer
from .ergodicity import ErgodicityAnalyzer
from .tail_risk import TailRiskAnalyzer
from .regime_edge import RegimeEdgeDecomposer
from .signal_decay import SignalDecayAnalyzer
from .strategy_auditor import StrategyAuditor

__all__ = [
    "TradeData",
    "SkillResult",
    "EdgeExistenceValidator",
    "StationarityAnalyzer",
    "ErgodicityAnalyzer",
    "TailRiskAnalyzer",
    "RegimeEdgeDecomposer",
    "SignalDecayAnalyzer",
    "StrategyAuditor",
]
