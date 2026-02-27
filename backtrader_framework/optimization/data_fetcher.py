"""
DuckDB data fetching for Walk-Forward Optimization.

Provides the DataFetcher class which queries the local DuckDB database
for OHLCV data, returning timestamp-indexed Pandas DataFrames.
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# DuckDB path
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DUCKDB_PATH = os.path.join(_BASE, 'duckdb_data', 'trading_data.duckdb')


class DataFetcher:
    """Fetch OHLCV data from the local DuckDB database."""

    @staticmethod
    def fetch(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Query DuckDB for OHLCV data and return a timestamp-indexed DataFrame, or None on failure."""
        try:
            import duckdb
        except ImportError:
            logger.error("duckdb not installed")
            return None

        if not os.path.exists(DUCKDB_PATH):
            logger.error("DuckDB not found at %s", DUCKDB_PATH)
            return None

        db_symbol = symbol.replace('-USD', '').replace('-', '')

        try:
            conn = duckdb.connect(DUCKDB_PATH, read_only=True)
            df = conn.execute(
                "SELECT timestamp, open as Open, high as High, "
                "low as Low, close as Close, volume as Volume "
                "FROM ohlcv_data WHERE symbol = ? AND timeframe = ? "
                "ORDER BY timestamp",
                [db_symbol, timeframe]
            ).fetchdf()
            conn.close()

            if df.empty:
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)

            return df
        except Exception as e:
            logger.error("DuckDB fetch error: %s", e)
            return None
