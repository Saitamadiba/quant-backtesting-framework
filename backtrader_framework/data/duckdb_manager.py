"""
DuckDB Manager for the Backtrader Framework.
Handles database connection, schema creation, and query operations.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backtrader_framework.config.settings import DUCKDB_PATH, DUCKDB_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DuckDBManager:
    """
    Manages DuckDB database for backtesting operations.

    Features:
    - Schema creation for OHLCV, trades, and session data
    - Bulk data insertion with pre-computed indicators
    - Analytical query helpers
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize DuckDB manager.

        Args:
            db_path: Path to DuckDB file. Defaults to DUCKDB_PATH from settings.
        """
        self.db_path = db_path or DUCKDB_PATH

        # Ensure directory exists
        DUCKDB_DIR.mkdir(parents=True, exist_ok=True)

        self._conn = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def initialize_schema(self):
        """Create all tables and indexes."""
        logger.info("Initializing DuckDB schema...")

        # OHLCV data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                atr_14 DOUBLE,
                ema_50 DOUBLE,
                ema_100 DOUBLE,
                ema_200 DOUBLE,
                rsi_14 DOUBLE,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)

        # Backtest trades table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                trade_id VARCHAR PRIMARY KEY,
                strategy_name VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                entry_price DOUBLE NOT NULL,
                entry_reason VARCHAR,
                exit_time TIMESTAMP,
                exit_price DOUBLE,
                exit_reason VARCHAR,
                stop_loss DOUBLE NOT NULL,
                take_profit_1 DOUBLE,
                take_profit_2 DOUBLE,
                pnl_percent DOUBLE,
                r_multiple DOUBLE,
                mfe_price DOUBLE,
                mfe_percent DOUBLE,
                mfe_r DOUBLE,
                mae_price DOUBLE,
                mae_percent DOUBLE,
                mae_r DOUBLE,
                session VARCHAR,
                bars_held INTEGER,
                atr_at_entry DOUBLE,
                strategy_params JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Session levels table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_levels (
                symbol VARCHAR NOT NULL,
                session_date DATE NOT NULL,
                session_name VARCHAR NOT NULL,
                session_high DOUBLE,
                session_low DOUBLE,
                session_open DOUBLE,
                session_close DOUBLE,
                session_start TIMESTAMP,
                session_end TIMESTAMP,
                PRIMARY KEY (symbol, session_date, session_name)
            )
        """)

        # FVG zones table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fvg_zones (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                formed_at TIMESTAMP NOT NULL,
                fvg_type VARCHAR NOT NULL,
                top_price DOUBLE NOT NULL,
                bottom_price DOUBLE NOT NULL,
                midpoint DOUBLE NOT NULL,
                size_percent DOUBLE NOT NULL,
                filled BOOLEAN DEFAULT FALSE,
                fill_percent DOUBLE DEFAULT 0,
                filled_at TIMESTAMP
            )
        """)

        # Create indexes
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf
            ON ohlcv_data(symbol, timeframe)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_strategy
            ON backtest_trades(strategy_name, symbol)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_time
            ON backtest_trades(entry_time)
        """)

        # Create analytical views
        self._create_views()

        logger.info("Schema initialized successfully")

    def _create_views(self):
        """Create analytical views for performance analysis."""

        # Strategy performance view
        self.conn.execute("""
            CREATE OR REPLACE VIEW v_strategy_performance AS
            SELECT
                strategy_name,
                symbol,
                COUNT(*) as total_trades,
                SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN r_multiple <= 0 THEN 1 ELSE 0 END) as losses,
                ROUND(100.0 * SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as win_rate,
                ROUND(SUM(r_multiple), 2) as total_r,
                ROUND(AVG(r_multiple), 3) as avg_r,
                ROUND(AVG(CASE WHEN r_multiple > 0 THEN r_multiple END), 3) as avg_winner,
                ROUND(AVG(CASE WHEN r_multiple <= 0 THEN r_multiple END), 3) as avg_loser,
                ROUND(ABS(SUM(CASE WHEN r_multiple > 0 THEN r_multiple ELSE 0 END) /
                          NULLIF(SUM(CASE WHEN r_multiple < 0 THEN ABS(r_multiple) ELSE 0 END), 0)), 2) as profit_factor,
                ROUND(AVG(mfe_r), 2) as avg_mfe_r,
                ROUND(AVG(mae_r), 2) as avg_mae_r,
                ROUND(AVG(bars_held), 1) as avg_bars_held
            FROM backtest_trades
            WHERE exit_time IS NOT NULL
            GROUP BY strategy_name, symbol
        """)

        # Session performance view
        self.conn.execute("""
            CREATE OR REPLACE VIEW v_session_performance AS
            SELECT
                strategy_name,
                session,
                COUNT(*) as trades,
                ROUND(100.0 * SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as win_rate,
                ROUND(SUM(r_multiple), 2) as total_r
            FROM backtest_trades
            WHERE exit_time IS NOT NULL AND session IS NOT NULL
            GROUP BY strategy_name, session
        """)

        # Monthly performance view
        self.conn.execute("""
            CREATE OR REPLACE VIEW v_monthly_performance AS
            SELECT
                strategy_name,
                DATE_TRUNC('month', entry_time) as month,
                COUNT(*) as trades,
                ROUND(SUM(r_multiple), 2) as total_r,
                ROUND(100.0 * SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as win_rate
            FROM backtest_trades
            WHERE exit_time IS NOT NULL
            GROUP BY strategy_name, DATE_TRUNC('month', entry_time)
            ORDER BY month
        """)

    # ==========================================================================
    # OHLCV Data Operations
    # ==========================================================================

    def insert_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Insert OHLCV data into the database.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
                Optional: atr_14, ema_50, ema_100, ema_200, rsi_14
            symbol: Trading symbol ('BTC', 'ETH')
            timeframe: Candle timeframe ('15m', '1h', '4h')
        """
        df = df.copy()
        df['symbol'] = symbol
        df['timeframe'] = timeframe

        # Ensure required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add optional indicator columns if missing
        for col in ['atr_14', 'ema_50', 'ema_100', 'ema_200', 'rsi_14']:
            if col not in df.columns:
                df[col] = None

        # Reorder columns
        columns = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low',
                   'close', 'volume', 'atr_14', 'ema_50', 'ema_100', 'ema_200', 'rsi_14']
        df = df[columns]

        # Insert with conflict handling
        self.conn.execute("""
            INSERT OR REPLACE INTO ohlcv_data
            SELECT * FROM df
        """)

        logger.info(f"Inserted {len(df)} rows for {symbol} {timeframe}")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLCV data from the database.

        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_indicators: Include pre-computed indicators

        Returns:
            DataFrame with OHLCV data
        """
        columns = "timestamp, open, high, low, close, volume"
        if include_indicators:
            columns += ", atr_14, ema_50, ema_100, ema_200, rsi_14"

        query = f"""
            SELECT {columns}
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        return self.conn.execute(query, params).fetchdf()

    def get_data_range(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get the date range of available data."""
        result = self.conn.execute("""
            SELECT
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as total_rows
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
        """, [symbol, timeframe]).fetchone()

        return {
            'start_date': result[0],
            'end_date': result[1],
            'total_rows': result[2]
        }

    # ==========================================================================
    # Trade Operations
    # ==========================================================================

    def insert_trade(self, trade: Dict[str, Any]):
        """Insert a single trade record."""
        columns = list(trade.keys())
        placeholders = ', '.join(['?' for _ in columns])
        col_names = ', '.join(columns)

        self.conn.execute(
            f"INSERT OR REPLACE INTO backtest_trades ({col_names}) VALUES ({placeholders})",
            list(trade.values())
        )

    def insert_trades(self, trades: List[Dict[str, Any]]):
        """Insert multiple trade records."""
        if not trades:
            return

        df = pd.DataFrame(trades)
        self.conn.execute("""
            INSERT OR REPLACE INTO backtest_trades
            SELECT * FROM df
        """)
        logger.info(f"Inserted {len(trades)} trades")

    def get_trades(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get trades with optional filters."""
        query = "SELECT * FROM backtest_trades WHERE 1=1"
        params = []

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)

        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)

        query += " ORDER BY entry_time"

        return self.conn.execute(query, params).fetchdf()

    def get_strategy_performance(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """Get strategy performance summary."""
        query = "SELECT * FROM v_strategy_performance"
        params = []

        if strategy_name:
            query += " WHERE strategy_name = ?"
            params.append(strategy_name)

        return self.conn.execute(query, params).fetchdf()

    def get_session_performance(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """Get performance by trading session."""
        query = "SELECT * FROM v_session_performance"
        params = []

        if strategy_name:
            query += " WHERE strategy_name = ?"
            params.append(strategy_name)

        return self.conn.execute(query, params).fetchdf()

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def execute(self, query: str, params: Optional[List] = None) -> duckdb.DuckDBPyRelation:
        """Execute a raw SQL query."""
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        result = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = ?
        """, [table_name]).fetchone()
        return result[0] > 0

    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table."""
        result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0]

    def clear_table(self, table_name: str):
        """Clear all data from a table."""
        self.conn.execute(f"DELETE FROM {table_name}")
        logger.info(f"Cleared table: {table_name}")


def initialize_database():
    """Initialize the DuckDB database with schema."""
    with DuckDBManager() as db:
        db.initialize_schema()
        print(f"Database initialized at: {db.db_path}")


if __name__ == "__main__":
    initialize_database()
