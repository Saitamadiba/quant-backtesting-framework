"""
Data Migration Script
- Fetches historical OHLCV data from Binance
- Computes technical indicators
- Migrates existing SQLite trades to DuckDB
"""

import requests
import pandas as pd
import numpy as np
import sqlite3
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backtrader_framework.config.settings import (
    BINANCE_BASE_URL, BINANCE_SYMBOLS, SQLITE_ML_DB,
    ATR_PERIOD, EMA_FAST, EMA_MID, EMA_SLOW, RSI_PERIOD
)
from backtrader_framework.data.duckdb_manager import DuckDBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """Fetches historical OHLCV data from Binance API."""

    INTERVAL_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d'
    }

    def __init__(self):
        self.base_url = BINANCE_BASE_URL

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> List[List]:
        """
        Fetch klines (candlestick) data from Binance.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Candle interval ('15m', '1h', '4h')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Max candles per request (max 1000)

        Returns:
            List of kline data
        """
        url = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_historical_data(
        self,
        symbol: str,
        interval: str,
        days: int = 730
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Internal symbol ('BTC', 'ETH')
            interval: Timeframe ('15m', '1h', '4h')
            days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        binance_symbol = BINANCE_SYMBOLS.get(symbol, f"{symbol}USDT")
        binance_interval = self.INTERVAL_MAP.get(interval, interval)

        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        all_data = []
        current_start = start_time

        logger.info(f"Fetching {symbol} {interval} data for {days} days...")

        while current_start < end_time:
            try:
                klines = self.fetch_klines(
                    binance_symbol,
                    binance_interval,
                    start_time=current_start,
                    end_time=end_time,
                    limit=1000
                )

                if not klines:
                    break

                all_data.extend(klines)

                # Move start to after last candle
                current_start = klines[-1][0] + 1

                # Rate limiting
                time.sleep(0.2)

                if len(all_data) % 5000 == 0:
                    logger.info(f"  Fetched {len(all_data)} candles...")

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                time.sleep(1)
                continue

        if not all_data:
            logger.warning(f"No data fetched for {symbol} {interval}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Process columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Keep only required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Fetched {len(df)} candles for {symbol} {interval}")
        return df


class IndicatorCalculator:
    """Calculates technical indicators for OHLCV data."""

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        return atr

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return df['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @classmethod
    def add_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to DataFrame."""
        df = df.copy()

        df['atr_14'] = cls.calculate_atr(df, ATR_PERIOD)
        df['ema_50'] = cls.calculate_ema(df, EMA_FAST)
        df['ema_100'] = cls.calculate_ema(df, EMA_MID)
        df['ema_200'] = cls.calculate_ema(df, EMA_SLOW)
        df['rsi_14'] = cls.calculate_rsi(df, RSI_PERIOD)

        return df


class SQLiteMigrator:
    """Migrates trade data from SQLite to DuckDB."""

    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path

    def read_ml_training_data(self) -> pd.DataFrame:
        """Read trades from ml_training_data.db."""
        if not self.sqlite_path.exists():
            logger.warning(f"SQLite database not found: {self.sqlite_path}")
            return pd.DataFrame()

        conn = sqlite3.connect(str(self.sqlite_path))

        try:
            df = pd.read_sql_query("""
                SELECT
                    trade_id,
                    strategy_name,
                    symbol,
                    timeframe,
                    direction,
                    entry_time,
                    entry_price,
                    exit_time,
                    exit_price,
                    stop_loss,
                    take_profit,
                    outcome,
                    pnl_percent,
                    risk_reward_actual as r_multiple,
                    mfe_percent,
                    mfe_r_multiple as mfe_r,
                    mae_percent,
                    mae_r_multiple as mae_r,
                    session,
                    duration_seconds,
                    atr_value as atr_at_entry,
                    exit_reason
                FROM ml_training_data
                WHERE outcome IS NOT NULL
            """, conn)

            logger.info(f"Read {len(df)} trades from SQLite")
            return df

        except Exception as e:
            logger.error(f"Error reading SQLite: {e}")
            return pd.DataFrame()

        finally:
            conn.close()

    def transform_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform SQLite trades to DuckDB schema."""
        if df.empty:
            return df

        df = df.copy()

        # Map strategy names
        strategy_map = {
            'FVG_BTC': 'FVG',
            'FVG_ETH': 'FVG',
            'LiquidityRaid_BTC': 'LiquidityRaid',
            'LiquidityRaid_ETH': 'LiquidityRaid',
            'MomentumMastery_BTC': 'MomentumMastery',
            'MomentumMastery_ETH': 'MomentumMastery',
            'SBS_BTC': 'SBS',
            'SBS_ETH': 'SBS',
        }
        df['strategy_name'] = df['strategy_name'].map(strategy_map).fillna(df['strategy_name'])

        # Calculate bars_held from duration
        # Assuming 15m timeframe (900 seconds per bar)
        df['bars_held'] = (df['duration_seconds'] / 900).fillna(0).astype(int)

        # Rename columns for DuckDB
        df = df.rename(columns={
            'take_profit': 'take_profit_1'
        })

        # Add missing columns
        df['take_profit_2'] = None
        df['entry_reason'] = None
        df['mfe_price'] = None
        df['mae_price'] = None
        df['strategy_params'] = None

        # Select final columns
        columns = [
            'trade_id', 'strategy_name', 'symbol', 'timeframe', 'direction',
            'entry_time', 'entry_price', 'entry_reason',
            'exit_time', 'exit_price', 'exit_reason',
            'stop_loss', 'take_profit_1', 'take_profit_2',
            'pnl_percent', 'r_multiple',
            'mfe_price', 'mfe_percent', 'mfe_r',
            'mae_price', 'mae_percent', 'mae_r',
            'session', 'bars_held', 'atr_at_entry', 'strategy_params'
        ]

        return df[[c for c in columns if c in df.columns]]


def run_migration(
    fetch_ohlcv: bool = True,
    migrate_trades: bool = True,
    symbols: List[str] = ['BTC', 'ETH'],
    timeframes: List[str] = ['15m', '1h', '4h'],
    days: int = 730
):
    """
    Run the complete data migration.

    Args:
        fetch_ohlcv: Whether to fetch OHLCV data from Binance
        migrate_trades: Whether to migrate trades from SQLite
        symbols: List of symbols to fetch
        timeframes: List of timeframes to fetch
        days: Number of days of historical data
    """
    db = DuckDBManager()
    db.initialize_schema()

    # Fetch OHLCV data
    if fetch_ohlcv:
        fetcher = BinanceDataFetcher()
        calculator = IndicatorCalculator()

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    logger.info(f"Processing {symbol} {timeframe}...")

                    # Fetch data
                    df = fetcher.fetch_historical_data(symbol, timeframe, days)

                    if df.empty:
                        continue

                    # Add indicators
                    df = calculator.add_indicators(df)

                    # Insert into DuckDB
                    db.insert_ohlcv(df, symbol, timeframe)

                    # Show data range
                    info = db.get_data_range(symbol, timeframe)
                    logger.info(f"  Range: {info['start_date']} to {info['end_date']} ({info['total_rows']} rows)")

                except Exception as e:
                    logger.error(f"Error processing {symbol} {timeframe}: {e}")

    # Migrate trades
    if migrate_trades:
        migrator = SQLiteMigrator(SQLITE_ML_DB)
        trades_df = migrator.read_ml_training_data()

        if not trades_df.empty:
            trades_df = migrator.transform_trades(trades_df)

            # Insert trades
            for _, row in trades_df.iterrows():
                try:
                    db.insert_trade(row.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to insert trade {row.get('trade_id')}: {e}")

            logger.info(f"Migrated {len(trades_df)} trades to DuckDB")

    # Print summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)

    for symbol in symbols:
        for timeframe in timeframes:
            info = db.get_data_range(symbol, timeframe)
            if info['total_rows'] > 0:
                print(f"{symbol} {timeframe}: {info['total_rows']} candles ({info['start_date']} to {info['end_date']})")

    trades_count = db.get_table_count('backtest_trades')
    print(f"\nTotal trades migrated: {trades_count}")
    print("=" * 60)

    db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate data to DuckDB")
    parser.add_argument("--no-ohlcv", action="store_true", help="Skip OHLCV fetch")
    parser.add_argument("--no-trades", action="store_true", help="Skip trade migration")
    parser.add_argument("--days", type=int, default=730, help="Days of history to fetch")
    parser.add_argument("--symbols", nargs="+", default=['BTC', 'ETH'], help="Symbols to fetch")
    parser.add_argument("--timeframes", nargs="+", default=['15m', '1h', '4h'], help="Timeframes to fetch")

    args = parser.parse_args()

    run_migration(
        fetch_ohlcv=not args.no_ohlcv,
        migrate_trades=not args.no_trades,
        symbols=args.symbols,
        timeframes=args.timeframes,
        days=args.days
    )
