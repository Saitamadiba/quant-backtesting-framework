"""
Data Feed Adapters for Backtrader.
Provides data feeds from DuckDB and other sources.
"""

import backtrader as bt
import pandas as pd
from typing import Optional

from backtrader_framework.data.duckdb_manager import DuckDBManager
from backtrader_framework.data.validation import validate_ohlcv


class DuckDBData(bt.feeds.PandasData):
    """
    Backtrader data feed that loads data from DuckDB.

    Usage:
        data = DuckDBData(
            symbol='BTC',
            timeframe='15m',
            fromdate=datetime(2023, 1, 1),
            todate=datetime(2024, 12, 31)
        )
        cerebro.adddata(data)
    """

    params = (
        ('symbol', 'BTC'),
        ('timeframe', '15m'),
        ('include_indicators', True),
    )

    def __init__(self, **kwargs):
        # Extract our custom params before calling parent
        symbol = kwargs.pop('symbol', self.p.symbol)
        timeframe = kwargs.pop('timeframe', self.p.timeframe)
        include_indicators = kwargs.pop('include_indicators', self.p.include_indicators)

        # Get date range
        fromdate = kwargs.get('fromdate')
        todate = kwargs.get('todate')

        # Load data from DuckDB
        start_date = fromdate.strftime('%Y-%m-%d') if fromdate else None
        end_date = todate.strftime('%Y-%m-%d') if todate else None

        with DuckDBManager() as db:
            df = db.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                include_indicators=include_indicators
            )

        if df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")

        # Prepare DataFrame for Backtrader
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)

        # Validate OHLCV data integrity
        df = validate_ohlcv(df)

        # Store metadata
        self._symbol = symbol
        self._timeframe = timeframe

        # Pass to parent
        kwargs['dataname'] = df
        super().__init__(**kwargs)


def load_duckdb_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> bt.feeds.PandasData:
    """
    Load data from DuckDB as a Backtrader data feed.

    Args:
        symbol: Trading symbol ('BTC', 'ETH')
        timeframe: Candle timeframe ('15m', '1h', '4h')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Backtrader PandasData feed
    """
    with DuckDBManager() as db:
        df = db.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            include_indicators=True
        )

    if df.empty:
        raise ValueError(f"No data found for {symbol} {timeframe}")

    # Prepare DataFrame
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)

    # Validate OHLCV data integrity
    df = validate_ohlcv(df)

    # Create data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )

    # Store metadata
    data._name = symbol
    data._timeframe = timeframe

    return data


def get_available_data() -> pd.DataFrame:
    """
    Get information about available data in DuckDB.

    Returns:
        DataFrame with symbol, timeframe, start_date, end_date, row_count
    """
    with DuckDBManager() as db:
        result = db.execute("""
            SELECT
                symbol,
                timeframe,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as row_count
            FROM ohlcv_data
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """).fetchdf()

    return result
