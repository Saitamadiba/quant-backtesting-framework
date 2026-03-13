"""Quant Backtesting Framework - Institutional-grade backtesting, WFO, and analytics."""

__version__ = "1.0.0"

# Lazy imports — backtrader may not be installed in all environments
# (e.g. the Streamlit dashboard only needs the optimization sub-package).
try:
    from backtrader_framework.strategies.base_strategy import BaseStrategy
    from backtrader_framework.runners.single_backtest import run_backtest
except ImportError:
    BaseStrategy = None
    run_backtest = None

try:
    from backtrader_framework.data.duckdb_manager import DuckDBManager
except ImportError:
    DuckDBManager = None

__all__ = [
    "BaseStrategy",
    "run_backtest",
    "DuckDBManager",
    "__version__",
]
