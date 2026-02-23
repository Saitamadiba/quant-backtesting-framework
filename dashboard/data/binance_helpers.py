"""Data helpers: candle fetching (Binance + Yahoo), indicators, ICT tools."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from config import (
    BINANCE_REST_BASE, BINANCE_SYMBOL_MAP,
    SESSIONS, TRADING_PAIRS, MTF_MAP,
)


def fetch_binance_candles(sym: str, tf: str, days: int) -> pd.DataFrame:
    """Fetch historical OHLCV from Binance REST API.

    Args:
        sym: Symbol key (e.g. "BTC", "ETH")
        tf: Timeframe string (e.g. "15m", "1h", "4h", "1d")
        days: Number of historical days to fetch
    """
    binance_sym = BINANCE_SYMBOL_MAP.get(sym, f"{sym}USDT")
    url = f"{BINANCE_REST_BASE}/klines"

    all_candles = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    while start_time < end_time:
        params = {
            "symbol": binance_sym, "interval": tf,
            "startTime": start_time, "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break

        if not data:
            break

        all_candles.extend(data)
        start_time = data[-1][6] + 1  # close_time + 1ms
        if len(data) < 1000:
            break

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "Close_Time", "Quote_Volume", "Trades",
        "Taker_Buy_Base", "Taker_Buy_Quote", "Ignore",
    ])
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = df[col].astype(float)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.set_index("Timestamp")
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA50, EMA200, Bullish_Bias, ATR, and ADX to a candle DataFrame."""
    df = df.copy()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    df["Bullish_Bias"] = df["EMA50"] > df["EMA200"]
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    # ADX calculation (14-period)
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
    df["ADX"] = dx.ewm(alpha=1/14, min_periods=14).mean()

    return df


def classify_regime(row) -> str:
    """Classify market regime from a candle row with ADX and EMA indicators.

    Returns one of: 'Trending Up', 'Trending Down', 'Ranging'
    """
    adx = row.get("ADX", 0)
    bullish = row.get("Bullish_Bias", True)

    if pd.isna(adx) or adx < 20:
        return "Ranging"
    elif bullish:
        return "Trending Up"
    else:
        return "Trending Down"


# ── Yahoo Finance fetcher (NQ) ───────────────────────────────────────────────

_YF_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "4h": "60m", "1D": "1d",
}
_YF_MAX_DAYS = {"1m": 7, "5m": 60, "15m": 60, "30m": 60}


def fetch_yahoo_candles(ticker: str, tf: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance.  Used for NQ=F and other non-Binance assets.

    yfinance rate-limits aggressively — keep *days* modest and cache results.
    """
    import yfinance as yf

    interval = _YF_INTERVAL_MAP.get(tf, "1d")
    max_days = _YF_MAX_DAYS.get(tf)
    if max_days:
        days = min(days, max_days)

    try:
        data = yf.download(
            ticker, period=f"{days}d", interval=interval,
            progress=False, auto_adjust=True,
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Timestamp"

    # Resample to 4h if requested (yfinance has no native 4h interval)
    if tf == "4h":
        df = df.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()

    return df


def fetch_candles(symbol: str, tf: str, days: int) -> pd.DataFrame:
    """Unified candle fetcher — routes to Binance or Yahoo based on config."""
    pair = TRADING_PAIRS.get(symbol, {})
    source = pair.get("source", "binance")
    if source == "yahoo":
        return fetch_yahoo_candles(pair.get("yahoo_ticker", symbol), tf, days)
    return fetch_binance_candles(symbol, tf, days)


# ── ICT Order Blocks ─────────────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> list[dict]:
    """Detect unmitigated ICT order blocks.

    Bullish OB: last bearish candle before a strong bullish move that
    breaks above the prior swing high.
    Bearish OB: last bullish candle before a strong bearish move that
    breaks below the prior swing low.

    Returns list of ``{time, top, bottom, type}``.
    """
    if len(df) < lookback + 3:
        return []

    obs: list[dict] = []
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    closes = df["Close"].values
    times = df.index

    for i in range(lookback, len(df) - 1):
        swing_high = highs[i - lookback : i].max()
        swing_low = lows[i - lookback : i].min()

        # Bullish OB: bearish candle followed by break of swing high
        if closes[i] < opens[i] and highs[i + 1] > swing_high:
            top = opens[i]
            bottom = closes[i]
            # Check mitigation: has price come back into the OB zone?
            future = lows[i + 1 :]
            if len(future) == 0 or future.min() > bottom:
                obs.append({
                    "time": times[i], "top": float(top),
                    "bottom": float(bottom), "type": "bullish",
                })

        # Bearish OB: bullish candle followed by break of swing low
        if closes[i] > opens[i] and lows[i + 1] < swing_low:
            top = closes[i]
            bottom = opens[i]
            future = highs[i + 1 :]
            if len(future) == 0 or future.max() < top:
                obs.append({
                    "time": times[i], "top": float(top),
                    "bottom": float(bottom), "type": "bearish",
                })

    return obs


# ── Fair Value Gaps ──────────────────────────────────────────────────────────

def detect_fvgs(df: pd.DataFrame) -> list[dict]:
    """Detect unmitigated Fair Value Gaps (3-candle imbalance).

    Bullish FVG: candle[i-1].High < candle[i+1].Low  (gap up)
    Bearish FVG: candle[i-1].Low  > candle[i+1].High (gap down)

    Returns list of ``{time, top, bottom, type}``.
    """
    if len(df) < 3:
        return []

    fvgs: list[dict] = []
    highs = df["High"].values
    lows = df["Low"].values
    times = df.index

    for i in range(1, len(df) - 1):
        # Bullish FVG
        if highs[i - 1] < lows[i + 1]:
            top = float(lows[i + 1])
            bottom = float(highs[i - 1])
            # Check mitigation: has price dipped into the gap?
            future_lows = lows[i + 2 :] if i + 2 < len(df) else np.array([])
            if len(future_lows) == 0 or future_lows.min() > bottom:
                fvgs.append({
                    "time": times[i], "top": top,
                    "bottom": bottom, "type": "bullish",
                })

        # Bearish FVG
        if lows[i - 1] > highs[i + 1]:
            top = float(lows[i - 1])
            bottom = float(highs[i + 1])
            future_highs = highs[i + 2 :] if i + 2 < len(df) else np.array([])
            if len(future_highs) == 0 or future_highs.max() < top:
                fvgs.append({
                    "time": times[i], "top": top,
                    "bottom": bottom, "type": "bearish",
                })

    return fvgs


def detect_fvgs_mtf(symbol: str, primary_tf: str, days: int) -> list[dict]:
    """Detect FVGs on higher timeframes and return them for overlay.

    Each returned dict includes an ``htf`` field with the source timeframe.
    """
    htf_list = MTF_MAP.get(primary_tf, [])
    all_fvgs: list[dict] = []
    for htf in htf_list:
        htf_df = fetch_candles(symbol, htf, days)
        if htf_df.empty:
            continue
        fvgs = detect_fvgs(htf_df)
        for f in fvgs:
            f["htf"] = htf
        all_fvgs.extend(fvgs)
    return all_fvgs


# ── Session Tools ────────────────────────────────────────────────────────────

def _to_et(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a timestamp to US/Eastern, handling tz-naive inputs."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("US/Eastern")


def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``session`` column (Asian / London / New York / Off-Hours)."""
    df = df.copy()

    def _classify(ts):
        hour = _to_et(ts).hour
        for name, rng in SESSIONS.items():
            s, e = rng["start"], rng["end"]
            if s > e:  # wraps midnight (Asian)
                if hour >= s or hour < e:
                    return name
            else:
                if s <= hour < e:
                    return name
        return "Off-Hours"

    df["session"] = [_classify(t) for t in df.index]
    return df


def compute_session_levels(df: pd.DataFrame) -> list[dict]:
    """Compute high/low for each trading session block.

    Returns list of ``{session, date, high, low, start_time, end_time}``.
    """
    df = assign_sessions(df)
    levels: list[dict] = []

    # Group consecutive candles in the same session
    df["_date"] = [_to_et(t).date() for t in df.index]
    df["_grp"] = (df["session"] != df["session"].shift()).cumsum()

    for _, grp in df.groupby("_grp"):
        session = grp["session"].iloc[0]
        if session == "Off-Hours":
            continue
        levels.append({
            "session": session,
            "date": str(grp["_date"].iloc[0]),
            "high": float(grp["High"].max()),
            "low": float(grp["Low"].min()),
            "start_time": grp.index[0],
            "end_time": grp.index[-1],
        })

    df.drop(columns=["_date", "_grp"], inplace=True, errors="ignore")
    return levels
