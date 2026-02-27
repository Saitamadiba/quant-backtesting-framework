"""Walk-Forward Optimization engine for strategy parameter tuning.

Key classes are re-exported here for convenience:
- WFOEngine, RegimeAdaptiveWFO: main optimization engines
- WFOConfig, TransactionCosts, TradeResult: data classes
- IndicatorEngine: technical indicator calculation
- RegimeDetector: market regime classification
- TradeSimulator: trade execution simulation
- StatisticalTests, MonteCarloAnalysis: statistical analysis
- DataFetcher: DuckDB data retrieval
"""

from .wfo_engine import (
    WFOEngine,
    RegimeAdaptiveWFO,
    WFOConfig,
    TransactionCosts,
    TradeResult,
    ALL_REGIMES,
)
from .indicators import IndicatorEngine
from .regime import RegimeDetector
from .simulator import TradeSimulator
from .statistics import StatisticalTests, MonteCarloAnalysis
from .data_fetcher import DataFetcher

__all__ = [
    'WFOEngine',
    'RegimeAdaptiveWFO',
    'WFOConfig',
    'TransactionCosts',
    'TradeResult',
    'ALL_REGIMES',
    'IndicatorEngine',
    'RegimeDetector',
    'TradeSimulator',
    'StatisticalTests',
    'MonteCarloAnalysis',
    'DataFetcher',
]
