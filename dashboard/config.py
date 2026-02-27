"""Dashboard configuration: paths, VPS config, strategy registry, constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # /Backtesting
DASHBOARD_DIR = BASE_DIR / "dashboard"
VPS_CACHE_DIR = DASHBOARD_DIR / "databases"
BACKTEST_RESULTS_DIR = DASHBOARD_DIR / "backtest_results"
DUCKDB_PATH = BASE_DIR / "duckdb_data" / "trading_data.duckdb"

# ── VPS Connection (from environment) ────────────────────────────────────────
VPS_HOST = os.getenv("VPS_HOST", "")
VPS_PORT = int(os.getenv("VPS_PORT", "22"))
VPS_USER = os.getenv("VPS_USER", "trader")
VPS_REMOTE_BASE = os.getenv("VPS_REMOTE_BASE", "/home/trader/trading_bots")

# ── Remote DB Mapping ─────────────────────────────────────────────────────────
# key = local cache filename, value = full remote path
VPS_DB_FILES = {
    "fvg_btc.db": f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC/btc-usd_enhanced_v3_trades.db",
    "fvg_eth.db": f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH/eth-usd_enhanced_v3_trades.db",
    "fvg_nq.db":  f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ/nq-usd_enhanced_v3_trades.db",
    "lr_btc.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/btc_liquidity_raid_v2.db",
    "lr_eth.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2/eth_liquidity_raid_v2.db",
    "mm_btc.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/btc_momentum_mastery_v2.db",
    "mm_eth.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/eth_momentum_mastery_v2.db",
    "sbs.db":     f"{VPS_REMOTE_BASE}/SBS/bots/core/ml_training_data.db",
}

# ── Remote ML Training DB Mapping ────────────────────────────────────────────
VPS_ML_FILES = {
    "fvg_btc_ml_training.db":  f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC/ml_training_data.db",
    "fvg_eth_ml_training.db":  f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH/ml_training_data.db",
    "fvg_nq_ml_training.db":   f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ/ml_training_data.db",
    "lr_btc_ml_training.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/ml_training_data.db",
    "lr_eth_ml_training.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2/ml_training_data.db",
    "mm_btc_ml_training.db":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/ml_training_data.db",
    "mm_eth_ml_training.db":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/ml_training_data.db",
    "root_ml_training.db":     f"{VPS_REMOTE_BASE}/ml_training_data.db",
}

# ── Strategy Registry ─────────────────────────────────────────────────────────
STRATEGIES = {
    "FVG": {"color": "#2196F3", "symbols": ["BTC", "ETH", "NQ"]},
    "Liquidity Raid": {"color": "#FF9800", "symbols": ["BTC", "ETH"]},
    "Momentum Mastery": {"color": "#4CAF50", "symbols": ["BTC", "ETH"]},
    "SBS": {"color": "#9C27B0", "symbols": ["BTC", "ETH", "NQ"]},
}

STRATEGY_COLORS = {s: v["color"] for s, v in STRATEGIES.items()}

# local DB file -> (strategy, symbol)
DB_STRATEGY_MAP = {
    "fvg_btc.db": ("FVG", "BTC"),
    "fvg_eth.db": ("FVG", "ETH"),
    "fvg_nq.db":  ("FVG", "NQ"),
    "lr_btc.db":  ("Liquidity Raid", "BTC"),
    "lr_eth.db":  ("Liquidity Raid", "ETH"),
    "mm_btc.db":  ("Momentum Mastery", "BTC"),
    "mm_eth.db":  ("Momentum Mastery", "ETH"),
    "sbs.db":     ("SBS", "ALL"),
}

# ── VPS Systemd Services ──────────────────────────────────────────────────────
BOT_SERVICES = {
    "fvg-btc": {"strategy": "FVG", "symbol": "BTC"},
    "fvg-eth": {"strategy": "FVG", "symbol": "ETH"},
    "fvg-nq":  {"strategy": "FVG", "symbol": "NQ"},
    "lr-btc":  {"strategy": "Liquidity Raid", "symbol": "BTC"},
    "lr-eth":  {"strategy": "Liquidity Raid", "symbol": "ETH"},
    "mm-btc":  {"strategy": "Momentum Mastery", "symbol": "BTC"},
    "mm-eth":  {"strategy": "Momentum Mastery", "symbol": "ETH"},
    "sbs-btc": {"strategy": "SBS", "symbol": "BTC"},
    "sbs-eth": {"strategy": "SBS", "symbol": "ETH"},
}

SERVICE_WORK_DIRS = {
    "fvg-btc": f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC",
    "fvg-eth": f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH",
    "fvg-nq":  f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ",
    "lr-btc":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2",
    "lr-eth":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2",
    "mm-btc":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC",
    "mm-eth":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH",
    "sbs-btc": f"{VPS_REMOTE_BASE}/SBS/bots/btc",
    "sbs-eth": f"{VPS_REMOTE_BASE}/SBS/bots/eth",
}

VPS_BACKUP_SCRIPT = f"{VPS_REMOTE_BASE}/backup_dbs.sh"

# ── Session Time Ranges (Eastern Time — America/New_York, DST-aware) ─────────
# Hours are in ET.  Callers must convert UTC timestamps to ET via
# zoneinfo.ZoneInfo("America/New_York") before comparing — do NOT use a
# hardcoded UTC-5 offset (ET is UTC-4 during daylight saving time).
SESSIONS = {
    "Asian":    {"start": 19, "end": 3},
    "London":   {"start": 3,  "end": 8},
    "New York": {"start": 8,  "end": 16},
}

# ── SBS Local Backtest Data ───────────────────────────────────────────────────
SBS_TRAINING_CSV = BASE_DIR / "SBS" / "data" / "training" / "ml_training_data.csv"
SBS_RESULTS_DIR = BASE_DIR / "SBS" / "data" / "results"

# ── ML Paths ──────────────────────────────────────────────────────────────────
ML_TRAINING_SCRIPT = BASE_DIR / "SBS" / "research" / "ml" / "train_ml_model.py"
ML_TRAINING_DATA = BASE_DIR / "SBS" / "data" / "training" / "ml_training_data.csv"
ML_MODELS_DIR = BASE_DIR / "SBS" / "research" / "ml" / "models"
ML_TRAINING_DB = BASE_DIR / "ml_training_data.db"
ML_ROOT_TRAINING_SCRIPT = BASE_DIR / "ml_model_training.py"
ML_PREDICTIONS_DB = VPS_CACHE_DIR / "ml_predictions.db"
ML_PERFORMANCE_SCORER = BASE_DIR / "ml_performance_scorer.py"

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_ANALYSIS_SCRIPT = (
    BASE_DIR / "SBS" / "research" / "analysis" / "monte_carlo"
    / "enhanced_monte_carlo_analysis.py"
)

# ── Binance API (public endpoints, no auth required) ─────────────────────────
BINANCE_REST_BASE = "https://api.binance.us/api/v3"
BINANCE_SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}

# ── Manual Trading ───────────────────────────────────────────────────────────
MANUAL_TRADES_DB = DASHBOARD_DIR / "databases" / "manual_trades.db"

TRADING_PAIRS = {
    "BTC": {"source": "binance", "binance_symbol": "BTCUSDT",
            "tv_symbol": "BINANCEUS:BTCUSDT",
            "base": "BTC", "quote": "USDT", "price_decimals": 2, "qty_decimals": 5},
    "ETH": {"source": "binance", "binance_symbol": "ETHUSDT",
            "tv_symbol": "BINANCEUS:ETHUSDT",
            "base": "ETH", "quote": "USDT", "price_decimals": 2, "qty_decimals": 4},
    "NQ":  {"source": "yahoo", "yahoo_ticker": "NQ=F",
            "tv_symbol": "CME_MINI:NQ1!",
            "price_decimals": 2},
}

TV_INTERVAL_MAP = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240", "1D": "D",
}

TRADING_REFRESH_OPTIONS = {"Off": 0, "5s": 5_000, "10s": 10_000, "30s": 30_000}

CHART_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1D"]

MTF_MAP = {
    "1m":  ["5m", "15m"],
    "5m":  ["15m", "1h"],
    "15m": ["1h", "4h"],
    "30m": ["1h", "4h"],
    "1h":  ["4h", "1D"],
    "4h":  ["1D"],
    "1D":  [],
}

SESSION_COLORS = {
    "Asian":    "rgba(156, 39, 176, 0.06)",
    "London":   "rgba(76, 175, 80, 0.06)",
    "New York": "rgba(255, 152, 0, 0.06)",
}

# ── Bot Deploy Mapping (local file -> VPS remote path) ───────────────────────
DEPLOY_BOT_FILES = {
    "FVG BTC": (
        BASE_DIR / "FVG_Strategy" / "BTC" / "fvg_btc.py",
        f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC/fvg_btc.py",
    ),
    "FVG ETH": (
        BASE_DIR / "FVG_Strategy" / "ETH" / "fvg_eth.py",
        f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH/fvg_eth.py",
    ),
    "FVG NQ": (
        BASE_DIR / "FVG_Strategy" / "NQ" / "fvg_nq.py",
        f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ/fvg_nq.py",
    ),
    "LR BTC": (
        BASE_DIR / "Liquidity_Raid" / "BTC_V2" / "lr_btc.py",
        f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/lr_btc.py",
    ),
    "LR ETH": (
        BASE_DIR / "Liquidity_Raid" / "ETH_V2" / "lr_eth.py",
        f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2/lr_eth.py",
    ),
    "MM BTC": (
        BASE_DIR / "Momentum_Mastery" / "BTC" / "btc_momentum_mastery_v2.py",
        f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/btc_momentum_mastery_v2.py",
    ),
    "MM ETH": (
        BASE_DIR / "Momentum_Mastery" / "ETH" / "eth_momentum_mastery_v2.py",
        f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/eth_momentum_mastery_v2.py",
    ),
    "SBS BTC": (
        BASE_DIR / "SBS" / "bots" / "btc" / "sbs_btc.py",
        f"{VPS_REMOTE_BASE}/SBS/bots/btc/sbs_btc.py",
    ),
    "SBS ETH": (
        BASE_DIR / "SBS" / "bots" / "eth" / "sbs_eth.py",
        f"{VPS_REMOTE_BASE}/SBS/bots/eth/sbs_eth.py",
    ),
}

DEPLOY_SERVICE_MAP = {
    "FVG BTC": "fvg-btc",
    "FVG ETH": "fvg-eth",
    "FVG NQ":  "fvg-nq",
    "LR BTC":  "lr-btc",
    "LR ETH":  "lr-eth",
    "MM BTC":  "mm-btc",
    "MM ETH":  "mm-eth",
    "SBS BTC": "sbs-btc",
    "SBS ETH": "sbs-eth",
}

# ── Account Settings ─────────────────────────────────────────────────────────
INITIAL_BALANCE = int(os.getenv("INITIAL_BALANCE", "10000"))

# ── Strategy Change Log ──────────────────────────────────────────────────────
STRATEGY_CHANGELOG = [
    # Example entry:
    # {
    #     "date": "2026-02-11",
    #     "label": "Strategy Update",
    #     "strategies": ["FVG"],
    #     "color": "#FFEB3B",
    #     "description": "Description of what changed",
    # },
]

# ── Auto-Refresh Options (seconds) ───────────────────────────────────────────
REFRESH_OPTIONS = {"Off": 0, "30s": 30_000, "1m": 60_000, "5m": 300_000}

# ── Unified Trade Schema Columns ──────────────────────────────────────────────
TRADE_SCHEMA_COLS = [
    "trade_id", "strategy", "symbol", "timeframe", "source",
    "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "stop_loss", "take_profit",
    "pnl_usd", "pnl_pct", "r_multiple",
    "session", "exit_reason", "duration_minutes",
    "running_balance", "mfe", "mae", "is_open",
]
