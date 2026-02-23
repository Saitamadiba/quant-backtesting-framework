"""Normalize per-source trade data to the unified dashboard schema."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import (
    VPS_CACHE_DIR, DB_STRATEGY_MAP, SESSIONS, TRADE_SCHEMA_COLS,
    SBS_TRAINING_CSV,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_session(hour: Optional[int]) -> str:
    """Classify hour (ET, 0-23) into trading session name."""
    if hour is None:
        return "Unknown"
    for name, times in SESSIONS.items():
        start, end = times["start"], times["end"]
        if start > end:  # wraps midnight
            if hour >= start or hour < end:
                return name
        else:
            if start <= hour < end:
                return name
    return "Off-Hours"


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if pd.notna(v) else default
    except (TypeError, ValueError):
        return default


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has all columns from TRADE_SCHEMA_COLS."""
    for col in TRADE_SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    return df[TRADE_SCHEMA_COLS]


# ── FVG SQLite (enhanced_trades table) ────────────────────────────────────────

def normalize_fvg(db_path: Path, strategy: str, symbol: str) -> pd.DataFrame:
    """Normalize FVG enhanced_trades SQLite to unified schema.

    Filters out erroneous "FIXED" trades that resulted from a
    misconfigured bot (position sizes of hundreds of BTC, sub-second
    durations, notional values >$20M). Only MOMENTUM-type trades
    with realistic position sizing are kept.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        raw = pd.read_sql_query("SELECT * FROM enhanced_trades", conn)
        conn.close()
    except Exception:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    if raw.empty:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    # ── Data quality filter: drop bogus FIXED-prefix trades ─────────────
    # A misconfigured "FIXED" bot variant produced trades with absurd
    # position sizes (250-936 BTC / 500 ETH / 250 NQ), sub-second
    # durations, and nonsensical PnL. These trades are identifiable by
    # their trade_id starting with "FIXED_". Only MOMENTUM-prefix trades
    # (the correctly configured bot) are kept.
    if "trade_id" in raw.columns:
        before = len(raw)
        raw = raw[~raw["trade_id"].str.startswith("FIXED", na=False)]
        dropped = before - len(raw)
        if dropped > 0:
            import logging
            logging.getLogger(__name__).info(
                f"FVG {symbol}: dropped {dropped} bogus FIXED-prefix trades"
            )

    df = pd.DataFrame()
    df["trade_id"] = raw.get("trade_id", raw.index.astype(str))
    df["strategy"] = strategy
    df["symbol"] = symbol
    df["timeframe"] = raw.get("timeframes", "15m")
    df["source"] = "Live"

    # Direction mapping
    dir_map = {"bullish": "Long", "bearish": "Short", "long": "Long", "short": "Short"}
    df["direction"] = raw["direction"].str.strip().str.lower().map(dir_map).fillna("Unknown")

    df["entry_time"] = pd.to_datetime(raw["entry_timestamp"], errors="coerce")
    df["exit_time"] = pd.to_datetime(raw["exit_timestamp"], errors="coerce")
    df["entry_price"] = raw["entry_price"].astype(float)
    df["exit_price"] = raw["exit_price"].astype(float)
    df["stop_loss"] = raw.get("stop_loss", pd.Series(dtype=float)).astype(float)
    df["take_profit"] = raw.get("take_profit", pd.Series(dtype=float)).astype(float)
    df["pnl_usd"] = raw.get("net_pnl", raw.get("gross_pnl", pd.Series(0.0))).astype(float)
    df["pnl_pct"] = None  # not directly available; compute later if needed

    # R-multiple from risk_amount
    risk = raw.get("risk_amount", pd.Series(dtype=float)).astype(float)
    df["r_multiple"] = df["pnl_usd"] / risk.replace(0, float("nan"))

    # Session
    if "market_session" in raw.columns:
        session_map = {
            "asian": "Asian", "asia": "Asian",
            "london": "London",
            "new_york": "New York", "new york": "New York", "ny": "New York",
        }
        df["session"] = raw["market_session"].str.strip().str.lower().map(session_map).fillna("Unknown")
    else:
        df["session"] = df["entry_time"].dt.hour.apply(_classify_session)

    df["exit_reason"] = raw.get("exit_reason", "Unknown")
    df["duration_minutes"] = raw.get("trade_duration_minutes", pd.Series(dtype=float)).astype(float)
    df["running_balance"] = raw.get("running_balance", pd.Series(dtype=float)).astype(float)
    df["mfe"] = raw.get("max_favorable_excursion", pd.Series(dtype=float)).astype(float)
    df["mae"] = raw.get("max_adverse_excursion", pd.Series(dtype=float)).astype(float)
    df["is_open"] = df["exit_time"].isna()

    # For open trades, zero out computed fields that need an exit
    df.loc[df["is_open"], ["pnl_usd", "pnl_pct", "r_multiple", "duration_minutes"]] = None

    return _ensure_schema(df)


# ── Liquidity Raid / Momentum Mastery SQLite (trades table) ───────────────────

def normalize_lr_mm(db_path: Path, strategy: str, symbol: str) -> pd.DataFrame:
    """Normalize LR / MM trades SQLite to unified schema."""
    try:
        conn = sqlite3.connect(str(db_path))
        raw = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
    except Exception:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    if raw.empty:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    df = pd.DataFrame()
    df["trade_id"] = raw.get("id", raw.index).astype(str)
    df["strategy"] = strategy
    df["symbol"] = symbol
    df["timeframe"] = "15m"
    df["source"] = "Live"

    # Direction
    sig_map = {"BUY": "Long", "SELL": "Short", "buy": "Long", "sell": "Short",
               "LONG": "Long", "SHORT": "Short", "long": "Long", "short": "Short"}
    df["direction"] = raw["signal_type"].str.strip().map(sig_map).fillna("Unknown")

    df["entry_time"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    df["exit_time"] = pd.to_datetime(raw.get("exit_timestamp"), errors="coerce")
    df["entry_price"] = raw["entry_price"].astype(float)
    df["exit_price"] = raw.get("exit_price", pd.Series(dtype=float)).astype(float)
    df["stop_loss"] = raw.get("stop_loss", pd.Series(dtype=float)).astype(float)
    df["take_profit"] = raw.get("take_profit", pd.Series(dtype=float)).astype(float)
    df["pnl_usd"] = raw.get("realized_pnl", pd.Series(0.0)).astype(float)
    df["pnl_pct"] = raw.get("realized_pnl_pct", pd.Series(dtype=float)).astype(float)

    # R-multiple from risk (entry - SL)
    risk_per_unit = (df["entry_price"] - df["stop_loss"]).abs()
    pnl_per_unit = df["exit_price"] - df["entry_price"]
    pnl_per_unit = pnl_per_unit.where(df["direction"] == "Long", -pnl_per_unit)
    df["r_multiple"] = pnl_per_unit / risk_per_unit.replace(0, float("nan"))

    # Session from killzone column
    if "killzone" in raw.columns:
        kz_map = {
            "asian": "Asian", "asia": "Asian",
            "london": "London", "london_open": "London",
            "new_york": "New York", "ny": "New York", "new york": "New York",
            "ny_open": "New York", "ny_pm": "New York",
        }
        df["session"] = raw["killzone"].str.strip().str.lower().map(kz_map).fillna("Unknown")
    else:
        df["session"] = df["entry_time"].dt.hour.apply(_classify_session)

    df["exit_reason"] = raw.get("exit_reason", raw.get("reason", "Unknown"))
    # Duration
    if df["entry_time"].notna().any() and df["exit_time"].notna().any():
        df["duration_minutes"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60
    else:
        df["duration_minutes"] = None
    df["running_balance"] = None
    df["mfe"] = None
    df["mae"] = None

    # Detect open positions from status column or missing exit
    if "status" in raw.columns:
        df["is_open"] = raw["status"].str.strip().str.lower() == "open"
    else:
        df["is_open"] = df["exit_time"].isna()

    df.loc[df["is_open"], ["pnl_usd", "pnl_pct", "r_multiple", "duration_minutes"]] = None

    return _ensure_schema(df)


# ── SBS CSV (ML training data) ───────────────────────────────────────────────

def normalize_sbs_csv(csv_path: Path = None) -> pd.DataFrame:
    """Normalize SBS ML training CSV to unified schema."""
    csv_path = csv_path or SBS_TRAINING_CSV
    if not csv_path.exists():
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    try:
        raw = pd.read_csv(str(csv_path))
    except Exception:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    if raw.empty:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    df = pd.DataFrame()
    df["trade_id"] = raw.get("record_id", raw.index.astype(str))
    df["strategy"] = "SBS"
    df["symbol"] = raw.get("symbol", "BTC").str.replace("-USD", "")
    df["timeframe"] = raw.get("timeframe", "4H")
    df["source"] = "Backtest"

    # Direction
    if "signal_direction" in raw.columns:
        dir_map = {"LONG": "Long", "SHORT": "Short", "long": "Long", "short": "Short"}
        df["direction"] = raw["signal_direction"].str.strip().map(dir_map).fillna("Unknown")
    else:
        df["direction"] = "Unknown"

    df["entry_time"] = pd.to_datetime(raw.get("signal_time"), errors="coerce")
    df["exit_time"] = None  # not available in training CSV
    df["entry_price"] = raw.get("signal_entry_price", pd.Series(dtype=float)).astype(float)
    df["exit_price"] = None
    df["stop_loss"] = raw.get("signal_stop_loss", pd.Series(dtype=float)).astype(float)
    df["take_profit"] = raw.get("signal_tp1", pd.Series(dtype=float)).astype(float)

    # PnL from R-multiple * risk
    r_mult = raw.get("outcome_r_multiple", pd.Series(0.0)).astype(float)
    risk_amt = raw.get("signal_risk_amount", pd.Series(0.0)).astype(float)
    df["pnl_usd"] = r_mult * risk_amt
    df["pnl_pct"] = raw.get("signal_risk_pct", pd.Series(dtype=float)).astype(float) * r_mult
    df["r_multiple"] = r_mult

    # Session from regime_trading_session
    if "regime_trading_session" in raw.columns:
        sess_map = {
            "ASIAN": "Asian", "LONDON": "London", "NY": "New York",
            "OVERLAP": "Off-Hours",
        }
        df["session"] = raw["regime_trading_session"].str.strip().map(sess_map).fillna("Unknown")
    else:
        df["session"] = "Unknown"

    df["exit_reason"] = raw.get("outcome_result", "Unknown")
    bars = raw.get("outcome_bars_held", pd.Series(dtype=float)).astype(float)
    # Approximate duration from bars * timeframe minutes
    tf_min = df["timeframe"].map({"1H": 60, "4H": 240, "1D": 1440, "15m": 15}).fillna(60)
    df["duration_minutes"] = bars * tf_min
    df["running_balance"] = None
    df["mfe"] = raw.get("outcome_mfe", pd.Series(dtype=float)).astype(float)
    df["mae"] = raw.get("outcome_mae", pd.Series(dtype=float)).astype(float)
    df["is_open"] = False  # backtest data is always closed

    return _ensure_schema(df)


# ── Load all VPS cached DBs ──────────────────────────────────────────────────

def load_all_live_trades() -> pd.DataFrame:
    """Load & normalize all VPS-cached SQLite databases."""
    frames = []
    for db_file, (strategy, symbol) in DB_STRATEGY_MAP.items():
        db_path = VPS_CACHE_DIR / db_file
        if not db_path.exists():
            continue

        if strategy == "FVG":
            df = normalize_fvg(db_path, strategy, symbol)
        else:
            df = normalize_lr_mm(db_path, strategy, symbol)

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)
    return pd.concat(frames, ignore_index=True)


def load_all_backtest_trades() -> pd.DataFrame:
    """Load & normalize all local backtest data."""
    frames = []

    # SBS training CSV
    sbs = normalize_sbs_csv()
    if not sbs.empty:
        frames.append(sbs)

    if not frames:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)
    return pd.concat(frames, ignore_index=True)
