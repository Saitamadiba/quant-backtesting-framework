"""Quant Backtesting Framework - Institutional-grade backtesting, WFO, and analytics."""

__version__ = "1.0.0"

from backtrader_framework.strategies.base_strategy import BaseStrategy
from backtrader_framework.runners.single_backtest import run_backtest
from backtrader_framework.data.duckdb_manager import DuckDBManager

__all__ = [
    "BaseStrategy",
    "run_backtest",
    "DuckDBManager",
    "__version__",
]
