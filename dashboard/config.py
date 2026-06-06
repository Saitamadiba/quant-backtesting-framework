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
    # LR BTC/ETH/SOL point at the HyroTrader $10k FUNDED ByBit sub-account
    # (real fills) to match the funded recap; NQ has no ByBit bot so stays paper.
    "lr_btc.db":  f"{VPS_REMOTE_BASE}/HyroTrader/lr_bybit_funded_10k.db",
    "lr_eth.db":  f"{VPS_REMOTE_BASE}/HyroTrader/lr_eth_bybit_funded_10k.db",
    "lr_nq.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2/nq_liquidity_raid_v2.db",
    "lr_sol.db":  f"{VPS_REMOTE_BASE}/HyroTrader/lr_sol_bybit_funded_10k.db",
    "mm_btc.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/btc_momentum_mastery_v2.db",
    "mm_eth.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/eth_momentum_mastery_v2.db",
    "mm_nq.db":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ/nq_momentum_mastery_v2.db",
    "sbs.db":     f"{VPS_REMOTE_BASE}/SBS/bots/core/ml_training_data.db",
}

# ── LR Shadow DBs (gate-rejected signals tracked as paper trades) ─────────────
# Populated by the internal strategy core — every signal a gate
# rejects (IV gate, London-shorts ban, regime block, counter-trend, etc.)
# becomes a shadow paper trade with TP/SL/TIME_EXIT tracking + a block_reason
# field so practice can be confronted against the WFO claim that justified
# each gate.  See dashboard page 23_Shadow_Trades.
VPS_SHADOW_DB_FILES = {
    "lr_btc_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/btc_shadow_trades.db",
    "lr_eth_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2/eth_shadow_trades.db",
    "lr_nq_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2/nq_shadow_trades.db",
    "lr_sol_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/SOL_V2/sol_shadow_trades.db",
    "mm_btc_shadow.db": f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/btc_shadow_trades.db",
    "mm_eth_shadow.db": f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/eth_shadow_trades.db",
    "mm_nq_shadow.db":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ/nq_shadow_trades.db",
    # ── Hybrid paper-bot shadow DBs (2026-05-31) — collect off-window + BTC-regime-block
    #     signals from the 7 no-Deribit-data LR paper bots (XRP + 6 majors).
    "lr_xrp_paper_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/XRP_V2/xrp_shadow_trades.db",
    "lr_bnb_paper_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/BNB_V2/bnb_shadow_trades.db",
    "lr_doge_paper_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/DOGE_V2/doge_shadow_trades.db",
    "lr_avax_paper_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/AVAX_V2/avax_shadow_trades.db",
    "lr_link_paper_shadow.db": f"{VPS_REMOTE_BASE}/Liquidity_Raid/LINK_V2/link_shadow_trades.db",
    "lr_dot_paper_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/DOT_V2/dot_shadow_trades.db",
    "lr_bch_paper_shadow.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/BCH_V2/bch_shadow_trades.db",
    # ── LRR shadow scanner (2026-06-02) — single multi-asset DB. Different
    #    schema (table=lrr_signals, asset column per-row); the page's
    #    _load_one detects the filename and applies the schema-bridge.
    "lrr_shadow_trades.db":    f"{VPS_REMOTE_BASE}/HyroTrader/lrr_shadow_trades.db",
    # ── Manual trades synced from ByBit funded sub-account (2026-06-04) —
    #    every trade NOT placed by a bot, auto-tagged with the nearest
    #    strategy signal (FVG/LR/MM ±30min) and labelled "Manual <TAG> SYM"
    #    in the dashboard. Schema-bridge in _load_one converts table name
    #    + a few column renames so the existing KPI/RR/regime machinery
    #    applies unchanged.
    "manual_trades.db":        f"{VPS_REMOTE_BASE}/HyroTrader/manual_trades.db",
}

SHADOW_DB_STRATEGY_MAP = {
    "lr_btc_shadow.db": ("Liquidity Raid", "BTC"),
    "lr_eth_shadow.db": ("Liquidity Raid", "ETH"),
    "lr_nq_shadow.db":  ("Liquidity Raid", "NQ"),
    "lr_sol_shadow.db": ("Liquidity Raid", "SOL"),
    "mm_btc_shadow.db": ("Momentum Mastery", "BTC"),
    "mm_eth_shadow.db": ("Momentum Mastery", "ETH"),
    "mm_nq_shadow.db":  ("Momentum Mastery", "NQ"),
    # Hybrid paper-bot shadows — labelled "LR Paper" to keep them visually
    # distinct from the live-bot shadows.
    "lr_xrp_paper_shadow.db":  ("LR Paper", "XRP"),
    "lr_bnb_paper_shadow.db":  ("LR Paper", "BNB"),
    "lr_doge_paper_shadow.db": ("LR Paper", "DOGE"),
    "lr_avax_paper_shadow.db": ("LR Paper", "AVAX"),
    "lr_link_paper_shadow.db": ("LR Paper", "LINK"),
    "lr_dot_paper_shadow.db":  ("LR Paper", "DOT"),
    "lr_bch_paper_shadow.db":  ("LR Paper", "BCH"),
    # LRR scanner is single-DB multi-asset. The sentinel symbol "MULTI" tells
    # the page's loader that the asset is in the row, not the file mapping.
    "lrr_shadow_trades.db":    ("LRR Shadow", "MULTI"),
    # Manual trades — single multi-asset DB; per-row strategy_tag (set by
    # the sync script's auto-tagger) becomes the bot label downstream.
    "manual_trades.db":        ("Manual", "MULTI"),
}

# ── Remote ML Training DB Mapping ────────────────────────────────────────────
VPS_ML_FILES = {
    "fvg_btc_ml_training.db":  f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC/ml_training_data.db",
    "fvg_eth_ml_training.db":  f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH/ml_training_data.db",
    "fvg_nq_ml_training.db":   f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ/ml_training_data.db",
    "lr_btc_ml_training.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/ml_training_data.db",
    "lr_eth_ml_training.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2/ml_training_data.db",
    "lr_nq_ml_training.db":    f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2/ml_training_data.db",
    "lr_sol_ml_training.db":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/SOL_V2/ml_training_data.db",
    "mm_btc_ml_training.db":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC/ml_training_data.db",
    "mm_eth_ml_training.db":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH/ml_training_data.db",
    "mm_nq_ml_training.db":    f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ/ml_training_data.db",
    "root_ml_training.db":     f"{VPS_REMOTE_BASE}/ml_training_data.db",
}

# ── Strategy Registry ─────────────────────────────────────────────────────────
STRATEGIES = {
    "FVG": {"color": "#2196F3", "symbols": ["BTC", "ETH", "NQ"]},
    "Liquidity Raid": {"color": "#FF9800", "symbols": ["BTC", "ETH", "NQ", "SOL"]},
    "Momentum Mastery": {"color": "#4CAF50", "symbols": ["BTC", "ETH", "NQ"]},
    "Vol Edge": {"color": "#00BCD4", "symbols": ["BTC", "ETH"]},
    "SBS": {"color": "#9C27B0", "symbols": ["BTC", "ETH"]},
}

STRATEGY_COLORS = {s: v["color"] for s, v in STRATEGIES.items()}

# local DB file -> (strategy, symbol)
DB_STRATEGY_MAP = {
    "fvg_btc.db": ("FVG", "BTC"),
    "fvg_eth.db": ("FVG", "ETH"),
    "fvg_nq.db":  ("FVG", "NQ"),
    "lr_btc.db":  ("Liquidity Raid", "BTC"),
    "lr_eth.db":  ("Liquidity Raid", "ETH"),
    "lr_nq.db":   ("Liquidity Raid", "NQ"),
    "lr_sol.db":  ("Liquidity Raid", "SOL"),
    "mm_btc.db":  ("Momentum Mastery", "BTC"),
    "mm_eth.db":  ("Momentum Mastery", "ETH"),
    "mm_nq.db":   ("Momentum Mastery", "NQ"),
    "sbs.db":     ("SBS", "ALL"),
}

# ── VPS Systemd Services ──────────────────────────────────────────────────────
BOT_SERVICES = {
    "fvg-btc": {"strategy": "FVG", "symbol": "BTC"},
    "fvg-eth": {"strategy": "FVG", "symbol": "ETH"},
    "fvg-nq":  {"strategy": "FVG", "symbol": "NQ"},
    "lr-btc":  {"strategy": "Liquidity Raid", "symbol": "BTC"},
    "lr-eth":  {"strategy": "Liquidity Raid", "symbol": "ETH"},
    "lr-nq":   {"strategy": "Liquidity Raid", "symbol": "NQ"},
    "lr-sol":  {"strategy": "Liquidity Raid", "symbol": "SOL"},
    "mm-btc":  {"strategy": "Momentum Mastery", "symbol": "BTC"},
    "mm-eth":  {"strategy": "Momentum Mastery", "symbol": "ETH"},
    "mm-nq":   {"strategy": "Momentum Mastery", "symbol": "NQ"},
    "mm-btc-shadow": {"strategy": "Momentum Mastery", "symbol": "BTC-SHADOW"},
    "straddle-btc":  {"strategy": "Vol Edge", "symbol": "BTC"},
    "straddle-eth":  {"strategy": "Vol Edge", "symbol": "ETH"},
    "sbs-btc": {"strategy": "SBS", "symbol": "BTC"},
    "sbs-eth": {"strategy": "SBS", "symbol": "ETH"},
}

SERVICE_WORK_DIRS = {
    "fvg-btc": f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC",
    "fvg-eth": f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH",
    "fvg-nq":  f"{VPS_REMOTE_BASE}/FVG_Strategy/NQ",
    "lr-btc":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2",
    "lr-eth":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/ETH_V2",
    "lr-nq":   f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2",
    "lr-sol":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/SOL_V2",
    "mm-btc":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC",
    "mm-eth":  f"{VPS_REMOTE_BASE}/Momentum_Mastery/ETH",
    "mm-nq":   f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ",
    "mm-btc-shadow": f"{VPS_REMOTE_BASE}/Momentum_Mastery/BTC",
    "straddle-btc":  f"{VPS_REMOTE_BASE}/Vol_Edge/Straddle_V1",
    "straddle-eth":  f"{VPS_REMOTE_BASE}/Vol_Edge/Straddle_V1",
    "sbs-btc": f"{VPS_REMOTE_BASE}/SBS/bots/btc",
    "sbs-eth": f"{VPS_REMOTE_BASE}/SBS/bots/eth",
}

# ── VPS Log Files (read directly — journalctl requires systemd-journal group) ─
SERVICE_LOG_FILES = {
    svc: f"{VPS_REMOTE_BASE}/logs/{svc.replace('-', '_')}.log"
    for svc in BOT_SERVICES
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
        f"{VPS_REMOTE_BASE}/the internal FVG live config",
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
    "MM NQ": (
        BASE_DIR / "Momentum_Mastery" / "NQ" / "nq_momentum_mastery_v2.py",
        f"{VPS_REMOTE_BASE}/Momentum_Mastery/NQ/nq_momentum_mastery_v2.py",
    ),
    "LR NQ": (
        BASE_DIR / "Liquidity_Raid" / "NQ_V2" / "lr_nq.py",
        f"{VPS_REMOTE_BASE}/Liquidity_Raid/NQ_V2/lr_nq.py",
    ),
    "LR SOL": (
        BASE_DIR / "Liquidity_Raid" / "SOL_V2" / "lr_sol.py",
        f"{VPS_REMOTE_BASE}/Liquidity_Raid/SOL_V2/lr_sol.py",
    ),
    "Straddle BTC": (
        BASE_DIR / "Vol_Edge" / "Straddle_V1" / "btc_straddle.py",
        f"{VPS_REMOTE_BASE}/Vol_Edge/Straddle_V1/btc_straddle.py",
    ),
    "Straddle ETH": (
        BASE_DIR / "Vol_Edge" / "Straddle_V1" / "eth_straddle.py",
        f"{VPS_REMOTE_BASE}/Vol_Edge/Straddle_V1/eth_straddle.py",
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
    "FVG BTC":      "fvg-btc",
    "FVG ETH":      "fvg-eth",
    "FVG NQ":       "fvg-nq",
    "LR BTC":       "lr-btc",
    "LR ETH":       "lr-eth",
    "LR NQ":        "lr-nq",
    "LR SOL":       "lr-sol",
    "MM BTC":       "mm-btc",
    "MM ETH":       "mm-eth",
    "MM NQ":        "mm-nq",
    "Straddle BTC": "straddle-btc",
    "Straddle ETH": "straddle-eth",
    "SBS BTC":      "sbs-btc",
    "SBS ETH":      "sbs-eth",
}

# ── Macro Data (Macro Context page) ──────────────────────────────────────────
# FRED needs a free API key (https://fred.stlouisfed.org/docs/api/api_key.html).
# Put it in dashboard/.env as FRED_API_KEY=... — never hardcode it here.
# World Bank + the yfinance market-macro series are keyless.
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Broker — Alpaca (Broker panel, READ-ONLY paper) ──────────────────────────
# Put paper-trading keys in dashboard/.env as ALPACA_API_KEY / ALPACA_SECRET_KEY
# — never hardcode them.  The dashboard panel is read-only; it never places
# orders.  ALPACA_PAPER stays True until you deliberately go live.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes")
ALPACA_BASE_URL = (
    "https://paper-api.alpaca.markets" if ALPACA_PAPER
    else "https://api.alpaca.markets"
)

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
