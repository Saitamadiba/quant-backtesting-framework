"""
Technical indicator calculations for Walk-Forward Optimization.

Provides the IndicatorEngine class which computes ATR, RSI, EMAs, ADX,
volume metrics, ML feature columns, higher-timeframe indicators,
price structure bias, and DVOL on OHLCV DataFrames.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """Calculate technical indicators (ATR, RSI, EMAs, ADX, volume, ML features) on OHLCV DataFrames."""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return a copy of df with new columns added."""
        df = df.copy()

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

        df['Bullish_Bias'] = df['EMA50'] > df['EMA200']
        df['Bearish_Bias'] = df['EMA50'] < df['EMA200']

        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['High_Volume'] = df['Volume'] > df['Volume_SMA'] * 1.5

        df['ADX'] = IndicatorEngine._calculate_adx(df)

        df['Manip_Score'] = 0
        daily_range = (df['High'] - df['Low']) / df['Close']
        avg_range = daily_range.rolling(window=20).mean()
        df.loc[daily_range > avg_range * 2, 'Manip_Score'] = 2
        df.loc[(daily_range > avg_range * 1.5) & (daily_range <= avg_range * 2), 'Manip_Score'] = 1

        # ── ML Feature Support Columns ──────────────────────────
        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        df['RealizedVol20'] = df['LogReturn'].rolling(window=20).std(ddof=1)

        # ATR percentile ranks (rolling rank / window)
        atr_series = df['ATR']
        df['ATR_Pctile20'] = atr_series.rolling(window=20).apply(
            lambda w: np.sum(w <= w.iloc[-1]) / len(w), raw=False
        )
        df['ATR_Pctile100'] = atr_series.rolling(window=100).apply(
            lambda w: np.sum(w <= w.iloc[-1]) / len(w), raw=False
        )

        # Momentum: 5-bar rate of change
        df['Momentum5'] = df['Close'] / df['Close'].shift(5) - 1

        # Close vs Range: buying pressure (0 = closed at low, 1 = closed at high)
        bar_range = df['High'] - df['Low']
        df['CloseVsRange'] = np.where(
            bar_range > 0, (df['Close'] - df['Low']) / bar_range, 0.5
        )

        # Candle streak: consecutive same-direction closes
        direction = np.sign(df['Close'].values - np.roll(df['Close'].values, 1))
        direction[0] = 0
        streak = np.zeros(len(direction), dtype=float)
        for i in range(1, len(direction)):
            if direction[i] == 0:
                streak[i] = 0
            elif direction[i] == np.sign(streak[i - 1]) or streak[i - 1] == 0:
                streak[i] = streak[i - 1] + direction[i]
            else:
                streak[i] = direction[i]
        df['CandleStreak'] = streak

        # ── Higher Timeframe (4H) Indicators ──────────────────────
        # Resample 15m→4H, compute EMAs, map back to 15m via ffill.
        # Adds MTF alignment filter: trades must agree with 4H trend.
        try:
            ohlcv_4h = df[['Open', 'High', 'Low', 'Close']].resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            }).dropna()
            if len(ohlcv_4h) >= 200:
                htf_ema50 = ohlcv_4h['Close'].ewm(span=50, adjust=False).mean()
                htf_ema200 = ohlcv_4h['Close'].ewm(span=200, adjust=False).mean()
                df['HTF_EMA50'] = htf_ema50.reindex(df.index, method='ffill')
                df['HTF_EMA200'] = htf_ema200.reindex(df.index, method='ffill')
                df['HTF_Bullish'] = df['HTF_EMA50'] > df['HTF_EMA200']
                df['HTF_Bearish'] = df['HTF_EMA50'] < df['HTF_EMA200']
            else:
                df['HTF_Bullish'] = df['Bullish_Bias']
                df['HTF_Bearish'] = df['Bearish_Bias']
        except Exception:
            df['HTF_Bullish'] = df['Bullish_Bias']
            df['HTF_Bearish'] = df['Bearish_Bias']

        # ── 1H OHLCV for cross-TF FVG detection ──────────────────
        # Resample to 1H and forward-fill back, giving the FVG adapter
        # access to 1h candle structure when running on 15m data.
        try:
            ohlcv_1h = df[['Open', 'High', 'Low', 'Close']].resample('1h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            }).dropna()
            if len(ohlcv_1h) >= 50:
                df['HTF_1h_Open'] = ohlcv_1h['Open'].reindex(df.index, method='ffill')
                df['HTF_1h_High'] = ohlcv_1h['High'].reindex(df.index, method='ffill')
                df['HTF_1h_Low'] = ohlcv_1h['Low'].reindex(df.index, method='ffill')
                df['HTF_1h_Close'] = ohlcv_1h['Close'].reindex(df.index, method='ffill')
        except Exception:
            pass  # columns won't exist; FVG adapter falls back to native TF

        # ── Price Structure Bias (Swing HH/HL/LH/LL) ─────────────
        # Leading indicator: reacts to reversals before EMA cross.
        # +1.0 = LONG, -1.0 = SHORT, 0.0 = NEUTRAL
        struct_bias, struct_conf = IndicatorEngine._compute_structure_bias(
            df['High'].values, df['Low'].values
        )
        df['StructureBias'] = struct_bias
        df['StructureConf'] = struct_conf

        # ── DVOL (Deribit Implied Volatility Index) ───────────────
        # Hourly historical DVOL merged via forward-fill.
        try:
            import json as _json
            _dvol_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)
                ))),
                'Liquidity_Raid', 'Research', 'btc_dvol_hourly.json',
            )
            if os.path.exists(_dvol_path):
                with open(_dvol_path) as _f:
                    _dvol_raw = _json.load(_f)
                _dvol_df = pd.DataFrame(_dvol_raw)
                _dvol_df['timestamp'] = pd.to_datetime(
                    _dvol_df['timestamp'], unit='ms', utc=True,
                ).dt.tz_localize(None)
                _dvol_df = _dvol_df.set_index('timestamp').sort_index()
                df['DVOL'] = _dvol_df['dvol'].reindex(df.index, method='ffill')
            else:
                df['DVOL'] = np.nan
        except Exception:
            df['DVOL'] = np.nan

        return df

    @staticmethod
    def _compute_structure_bias(
        highs: np.ndarray, lows: np.ndarray,
    ) -> tuple:
        """Vectorized price structure bias from swing HH/HL/LH/LL.

        Detects swing points (3-bar lookback = 7-bar centered window),
        then for each bar computes the bullish/bearish structure score
        from the 3 most recent swing highs and lows.

        Returns (structure_bias, structure_conf) arrays of length n.
        """
        n = len(highs)

        # Swing detection: local max/min in 7-bar centered window
        h_series = pd.Series(highs)
        l_series = pd.Series(lows)
        rolling_max_h = h_series.rolling(window=7, center=True).max().values
        rolling_min_l = l_series.rolling(window=7, center=True).min().values

        sh_mask = (highs == rolling_max_h) & ~np.isnan(rolling_max_h)
        sl_mask = (lows == rolling_min_l) & ~np.isnan(rolling_min_l)

        sh_idx = np.where(sh_mask)[0]
        sh_val = highs[sh_idx]
        sl_idx = np.where(sl_mask)[0]
        sl_val = lows[sl_idx]

        if len(sh_idx) < 3 or len(sl_idx) < 3:
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

        # Build per-bar arrays of last 3 swing high/low values via ffill.
        # At each swing point, record its value; then forward-fill.
        def _last_k_ffill(indices, values, n_bars, k=3):
            """For each bar, produce the last k values via forward-fill."""
            arrays = [np.full(n_bars, np.nan) for _ in range(k)]
            for j in range(k, len(indices)):
                idx = indices[j]
                for offset in range(k):
                    arrays[offset][idx] = values[j - (k - 1 - offset)]
            # Forward-fill each array
            return [pd.Series(a).ffill().values for a in arrays]

        sh_2, sh_1, sh_0 = _last_k_ffill(sh_idx, sh_val, n, 3)
        sl_2, sl_1, sl_0 = _last_k_ffill(sl_idx, sl_val, n, 3)

        # Vectorized HH/HL/LH/LL counts
        valid = ~np.isnan(sh_0) & ~np.isnan(sh_2) & ~np.isnan(sl_0) & ~np.isnan(sl_2)
        hh = ((sh_0 > sh_1).astype(np.int8) + (sh_1 > sh_2).astype(np.int8))
        hl = ((sl_0 > sl_1).astype(np.int8) + (sl_1 > sl_2).astype(np.int8))
        lh = ((sh_0 < sh_1).astype(np.int8) + (sh_1 < sh_2).astype(np.int8))
        ll_c = ((sl_0 < sl_1).astype(np.int8) + (sl_1 < sl_2).astype(np.int8))

        bull = hh + hl  # max 4
        bear = lh + ll_c

        # Classify: matching live bot logic exactly
        bias = np.where(
            ~valid, 0.0,
            np.where((bull >= 3) & (bear <= 1), 1.0,
            np.where((bear >= 3) & (bull <= 1), -1.0,
            np.where(bull > bear, 1.0,
            np.where(bear > bull, -1.0, 0.0))))
        ).astype(np.float32)

        conf = np.where(
            ~valid, 0.0,
            np.where((bull >= 3) & (bear <= 1), bull / 4.0,
            np.where((bear >= 3) & (bull <= 1), bear / 4.0,
            np.abs(bull - bear) / 4.0))
        ).astype(np.float32)

        return bias, conf

    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average Directional Index (ADX) from High/Low/Close columns."""
        high, low, close = df['High'], df['Low'], df['Close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            np.abs(high - close.shift()),
            np.abs(low - close.shift())
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
