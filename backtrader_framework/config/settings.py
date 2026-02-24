"""
Global settings for the Backtrader Framework.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRAMEWORK_DIR = BASE_DIR / "backtrader_framework"
DUCKDB_DIR = BASE_DIR / "duckdb_data"

# Database paths
DUCKDB_PATH = DUCKDB_DIR / "trading_data.duckdb"
SQLITE_ML_DB = BASE_DIR / "ml_training_data.db"

# Supported symbols and timeframes
SYMBOLS = ['BTC', 'ETH']
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
PRIMARY_TIMEFRAMES = ['15m', '1h', '4h']

# Binance API settings
BINANCE_BASE_URL = "https://api.binance.us/api/v3"
BINANCE_KLINES_ENDPOINT = "/klines"
BINANCE_SYMBOLS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
}

# Session times (ET)
SESSIONS = {
    'ASIA': {'start': 19, 'end': 3},      # 7pm - 3am ET
    'LONDON': {'start': 3, 'end': 8},      # 3am - 8am ET
    'NEW_YORK': {'start': 8, 'end': 16},   # 8am - 4pm ET
}

# Kill zones for trading
KILL_ZONES = {
    'LONDON': {'start': 3, 'end': 5},      # 3am - 5am ET
    'NEW_YORK': {'start': 8, 'end': 16},   # 8am - 4pm ET
}

# Default backtesting parameters
DEFAULT_INITIAL_CASH = 10000.0
DEFAULT_COMMISSION = 0.00055  # 0.055% taker fee (Bybit/Binance perp)
DEFAULT_SLIPPAGE = 0.0001    # 0.01% — overridden per-asset in TransactionCosts.for_asset()

# Indicator defaults
ATR_PERIOD = 14
EMA_FAST = 50
EMA_MID = 100
EMA_SLOW = 200
RSI_PERIOD = 14

# Strategy defaults
COOLDOWN_BARS = 4  # Bars between trades

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
}
