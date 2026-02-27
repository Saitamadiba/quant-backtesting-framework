"""
Persistent Candle Storage using DuckDB.

Lightweight storage for live trading candles that persists across bot restarts.
Designed for minimal overhead - writes are batched, reads are fast.

Storage estimate: ~50-100 KB/week for all bots (BTC + ETH, multiple timeframes)
"""

import duckdb
import pandas as pd
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta, timezone
import logging

# Default path for candle storage
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "duckdb_data" / "live_candles.duckdb"

logger = logging.getLogger(__name__)


class CandleStore:
    """
    Persistent candle storage for live trading bots.

    Features:
    - Thread-safe writes with batching
    - Fast reads with in-memory caching
    - Automatic cleanup of old data (configurable retention)
    - Singleton pattern per database path
    """

    _instances: Dict[str, 'CandleStore'] = {}
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None):
        """Singleton per database path."""
        db_path = db_path or DEFAULT_DB_PATH
        db_key = str(db_path)

        with cls._lock:
            if db_key not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[db_key] = instance
            return cls._instances[db_key]

    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return

        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Write buffer for batching
        self._write_buffer: List[Dict] = []
        self._buffer_lock = threading.RLock()
        self._last_flush = time.time()
        self._flush_interval = 30  # Flush every 30 seconds
        self._max_buffer_size = 100  # Or when buffer reaches 100 candles

        # Retention settings
        self._retention_days = 30  # Keep 30 days of data

        # Initialize connection and schema
        self._conn = None
        self._init_schema()

        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self._flush_thread.start()

        self._initialized = True
        logger.info(f"CandleStore initialized at {self.db_path}")

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def _init_schema(self):
        """Create the candles table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS live_candles (
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)

        # Index for fast lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_time
            ON live_candles(symbol, timeframe, timestamp DESC)
        """)

        logger.debug("CandleStore schema initialized")

    def save_candle(self, symbol: str, timeframe: str, candle: Dict):
        """
        Save a single candle to the buffer (will be flushed periodically).

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            timeframe: Candle interval (e.g., "15m")
            candle: Dict with keys: timestamp, open, high, low, close, volume
        """
        record = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': candle.get('timestamp') or candle.get('Timestamp'),
            'open': float(candle.get('open') or candle.get('Open')),
            'high': float(candle.get('high') or candle.get('High')),
            'low': float(candle.get('low') or candle.get('Low')),
            'close': float(candle.get('close') or candle.get('Close')),
            'volume': float(candle.get('volume') or candle.get('Volume', 0))
        }

        with self._buffer_lock:
            self._write_buffer.append(record)

            # Flush if buffer is full
            if len(self._write_buffer) >= self._max_buffer_size:
                self._flush_buffer()

    def save_candles_batch(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """
        Save a batch of candles (e.g., on startup from REST API).

        Args:
            symbol: Trading pair
            timeframe: Candle interval
            df: DataFrame with OHLCV columns (index should be timestamp)
        """
        if df is None or df.empty:
            return

        df = df.copy()

        # Normalize column names
        col_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }
        df.columns = [col_map.get(c, c.lower()) for c in df.columns]

        # Add metadata
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['timestamp'] = df.index

        # Select columns in order
        df = df[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Upsert into database
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO live_candles
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                SELECT symbol, timeframe, timestamp, open, high, low, close, volume FROM df
            """)
            logger.debug(f"Saved {len(df)} candles for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Error saving candles batch: {e}")

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get candles from the database.

        Args:
            symbol: Trading pair
            timeframe: Candle interval
            limit: Maximum number of candles to return
            start_time: Optional start time filter

        Returns:
            DataFrame with OHLCV data, index is timestamp
        """
        # Flush buffer first to ensure latest data
        self._flush_buffer()

        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM live_candles
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            df = self.conn.execute(query, params).fetchdf()

            if df.empty:
                return None

            # Sort ascending and set index
            df = df.sort_values('timestamp')
            df = df.set_index('timestamp')

            # Capitalize columns to match websocket format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            return df

        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            return None

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the timestamp of the most recent candle."""
        try:
            result = self.conn.execute("""
                SELECT MAX(timestamp) FROM live_candles
                WHERE symbol = ? AND timeframe = ?
            """, [symbol, timeframe]).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {e}")
            return None

    def get_candle_count(self, symbol: str, timeframe: str) -> int:
        """Get the number of stored candles."""
        try:
            result = self.conn.execute("""
                SELECT COUNT(*) FROM live_candles
                WHERE symbol = ? AND timeframe = ?
            """, [symbol, timeframe]).fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting candle count: {e}")
            return 0

    def cleanup_old_data(self, days: Optional[int] = None):
        """
        Remove candles older than retention period.

        Args:
            days: Number of days to keep (default: self._retention_days)
        """
        days = days or self._retention_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            result = self.conn.execute("""
                DELETE FROM live_candles WHERE timestamp < ?
            """, [cutoff])
            deleted = result.fetchone()
            logger.info(f"Cleaned up candles older than {days} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        try:
            stats = self.conn.execute("""
                SELECT
                    symbol,
                    timeframe,
                    COUNT(*) as candle_count,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM live_candles
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """).fetchdf()

            total_count = self.conn.execute("SELECT COUNT(*) FROM live_candles").fetchone()[0]

            # Estimate size (rough: ~80 bytes per row)
            estimated_size_kb = (total_count * 80) / 1024

            return {
                'total_candles': total_count,
                'estimated_size_kb': round(estimated_size_kb, 2),
                'by_symbol_timeframe': stats.to_dict('records') if not stats.empty else []
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}

    def _flush_buffer(self):
        """Flush the write buffer to disk."""
        with self._buffer_lock:
            if not self._write_buffer:
                return

            try:
                df = pd.DataFrame(self._write_buffer)
                self.conn.execute("""
                    INSERT OR REPLACE INTO live_candles
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume FROM df
                """)
                logger.debug(f"Flushed {len(self._write_buffer)} candles to disk")
                self._write_buffer = []
                self._last_flush = time.time()
            except Exception as e:
                logger.error(f"Error flushing buffer: {e}")

    def _background_flush(self):
        """Background thread for periodic buffer flushing."""
        while True:
            time.sleep(self._flush_interval)

            with self._buffer_lock:
                if self._write_buffer and (time.time() - self._last_flush) >= self._flush_interval:
                    self._flush_buffer()

            # Periodic cleanup (once per hour)
            if int(time.time()) % 3600 < self._flush_interval:
                self.cleanup_old_data()

    def close(self):
        """Close the database connection."""
        self._flush_buffer()
        if self._conn:
            self._conn.close()
            self._conn = None


# Singleton instance for easy access
_store: Optional[CandleStore] = None

def get_candle_store() -> CandleStore:
    """Get the singleton CandleStore instance."""
    global _store
    if _store is None:
        _store = CandleStore()
    return _store
