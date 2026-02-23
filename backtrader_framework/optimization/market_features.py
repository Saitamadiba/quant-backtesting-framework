"""
Market Feature Engine — rich feature computation from OHLCV data.

Computes 20 market microstructure features at any bar index for ML
trade filtering. Features span volatility regime, momentum quality,
volume profile, and price structure.

Used by:
- ml_feature_engineering.py (enriching training data)
- ml_trade_filter.py (live inference)
- WFO engine (signal metadata enrichment)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

# All feature names produced by compute_at_bar()
MARKET_FEATURE_NAMES = [
    # Volatility (5)
    'atr_percentile_20', 'atr_percentile_100', 'realized_vol_20',
    'vol_of_vol', 'atr_ratio',
    # Momentum (5)
    'adx_slope_5', 'rsi_divergence', 'candle_streak',
    'close_vs_range', 'momentum_5',
    # Volume (3)
    'relative_volume', 'volume_trend_5', 'volume_price_confirm',
    # Price Structure (5)
    'dist_from_high_20', 'dist_from_low_20', 'ema_alignment',
    'price_vs_ema200', 'range_position',
    # Cross-Asset (2)
    'btc_eth_corr_20', 'btc_eth_divergence',
]


class MarketFeatureEngine:
    """Computes rich market features from an indicator-enriched DataFrame."""

    @staticmethod
    def compute_at_bar(
        df: pd.DataFrame,
        idx: int,
        other_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Compute all market features at a specific bar index.

        Args:
            df: DataFrame with OHLCV + indicators (from IndicatorEngine.calculate())
            idx: Bar index to compute features at
            other_df: Optional second-asset DataFrame for cross-asset features

        Returns:
            Dict of feature_name -> float value. NaN for features that
            can't be computed (insufficient history).
        """
        features = {}
        n = len(df)

        if idx < 0 or idx >= n:
            return {name: float('nan') for name in MARKET_FEATURE_NAMES}

        # Extract arrays for fast access
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        volumes = df['Volume'].values

        atr_vals = df['ATR'].values if 'ATR' in df.columns else None
        rsi_vals = df['RSI'].values if 'RSI' in df.columns else None
        adx_vals = df['ADX'].values if 'ADX' in df.columns else None
        ema20 = df['EMA20'].values if 'EMA20' in df.columns else None
        ema50 = df['EMA50'].values if 'EMA50' in df.columns else None
        ema200 = df['EMA200'].values if 'EMA200' in df.columns else None
        vol_sma = df['Volume_SMA'].values if 'Volume_SMA' in df.columns else None

        close = closes[idx]
        high = highs[idx]
        low = lows[idx]
        atr = atr_vals[idx] if atr_vals is not None else (high - low)

        # ── Volatility Features ──────────────────────────────────────

        # ATR percentile rank (20-bar)
        if atr_vals is not None and idx >= 20:
            window = atr_vals[idx - 19:idx + 1]
            features['atr_percentile_20'] = float(np.sum(window <= atr) / len(window))
        else:
            features['atr_percentile_20'] = float('nan')

        # ATR percentile rank (100-bar)
        if atr_vals is not None and idx >= 100:
            window = atr_vals[idx - 99:idx + 1]
            features['atr_percentile_100'] = float(np.sum(window <= atr) / len(window))
        else:
            features['atr_percentile_100'] = float('nan')

        # Realized volatility (20-bar std of log returns)
        if idx >= 21:
            log_rets = np.diff(np.log(closes[idx - 20:idx + 1]))
            features['realized_vol_20'] = float(np.std(log_rets, ddof=1))
        else:
            features['realized_vol_20'] = float('nan')

        # Vol-of-vol (std of rolling 5-bar realized vol over 20 bars)
        if idx >= 30:
            rolling_vols = []
            for j in range(idx - 19, idx + 1):
                lr = np.diff(np.log(closes[j - 5:j + 1]))
                rolling_vols.append(np.std(lr, ddof=1) if len(lr) >= 2 else 0.0)
            features['vol_of_vol'] = float(np.std(rolling_vols, ddof=1))
        else:
            features['vol_of_vol'] = float('nan')

        # ATR ratio (current vs 20-bar avg)
        if atr_vals is not None and idx >= 20:
            avg_atr = float(np.mean(atr_vals[idx - 19:idx + 1]))
            features['atr_ratio'] = float(atr / avg_atr) if avg_atr > 0 else 1.0
        else:
            features['atr_ratio'] = float('nan')

        # ── Momentum Features ────────────────────────────────────────

        # ADX slope (5-bar change)
        if adx_vals is not None and idx >= 5:
            features['adx_slope_5'] = float(adx_vals[idx] - adx_vals[idx - 5])
        else:
            features['adx_slope_5'] = float('nan')

        # RSI divergence: price new high but RSI lower, or price new low but RSI higher
        if rsi_vals is not None and idx >= 20:
            price_window = closes[idx - 19:idx + 1]
            rsi_window = rsi_vals[idx - 19:idx + 1]
            price_at_high = close >= np.max(price_window[:-1])
            rsi_at_high = rsi_vals[idx] >= np.max(rsi_window[:-1])
            price_at_low = close <= np.min(price_window[:-1])
            rsi_at_low = rsi_vals[idx] <= np.min(rsi_window[:-1])

            if price_at_high and not rsi_at_high:
                features['rsi_divergence'] = -1.0  # Bearish divergence
            elif price_at_low and not rsi_at_low:
                features['rsi_divergence'] = 1.0   # Bullish divergence
            else:
                features['rsi_divergence'] = 0.0
        else:
            features['rsi_divergence'] = float('nan')

        # Candle streak (consecutive same-direction candles)
        if idx >= 1:
            streak = 0
            for j in range(idx, max(idx - 20, 0), -1):
                if closes[j] > closes[j - 1] if j > 0 else True:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                elif closes[j] < closes[j - 1] if j > 0 else True:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
                else:
                    break
            features['candle_streak'] = float(streak)
        else:
            features['candle_streak'] = 0.0

        # Close vs range (buying pressure indicator)
        bar_range = high - low
        features['close_vs_range'] = float((close - low) / bar_range) if bar_range > 0 else 0.5

        # Momentum 5 (5-bar rate of change)
        if idx >= 5 and closes[idx - 5] > 0:
            features['momentum_5'] = float(close / closes[idx - 5] - 1)
        else:
            features['momentum_5'] = float('nan')

        # ── Volume Features ──────────────────────────────────────────

        # Relative volume
        if vol_sma is not None and vol_sma[idx] > 0:
            features['relative_volume'] = float(volumes[idx] / vol_sma[idx])
        elif idx >= 20:
            avg_vol = float(np.mean(volumes[idx - 19:idx + 1]))
            features['relative_volume'] = float(volumes[idx] / avg_vol) if avg_vol > 0 else 1.0
        else:
            features['relative_volume'] = float('nan')

        # Volume trend (5-bar linear regression slope, normalized)
        if idx >= 5:
            vol_window = volumes[idx - 4:idx + 1].astype(float)
            avg_vol = np.mean(vol_window)
            if avg_vol > 0:
                x = np.arange(5, dtype=float)
                slope = np.polyfit(x, vol_window / avg_vol, 1)[0]
                features['volume_trend_5'] = float(slope)
            else:
                features['volume_trend_5'] = 0.0
        else:
            features['volume_trend_5'] = float('nan')

        # Volume-price confirmation
        if vol_sma is not None and idx >= 1:
            high_vol = volumes[idx] > vol_sma[idx] * 1.2
            price_up = closes[idx] > closes[idx - 1]
            bull_bias = ema50 is not None and ema200 is not None and ema50[idx] > ema200[idx]
            if high_vol and ((price_up and bull_bias) or (not price_up and not bull_bias)):
                features['volume_price_confirm'] = 1.0
            else:
                features['volume_price_confirm'] = 0.0
        else:
            features['volume_price_confirm'] = float('nan')

        # ── Price Structure Features ─────────────────────────────────

        # Distance from 20-bar high/low in ATR units
        if idx >= 20 and atr > 0:
            high_20 = float(np.max(highs[idx - 19:idx + 1]))
            low_20 = float(np.min(lows[idx - 19:idx + 1]))
            features['dist_from_high_20'] = float((high_20 - close) / atr)
            features['dist_from_low_20'] = float((close - low_20) / atr)
        else:
            features['dist_from_high_20'] = float('nan')
            features['dist_from_low_20'] = float('nan')

        # EMA alignment score (-3 to +3)
        if ema20 is not None and ema50 is not None and ema200 is not None:
            score = 0
            if close > ema20[idx]:
                score += 1
            else:
                score -= 1
            if ema20[idx] > ema50[idx]:
                score += 1
            else:
                score -= 1
            if ema50[idx] > ema200[idx]:
                score += 1
            else:
                score -= 1
            features['ema_alignment'] = float(score)
        else:
            features['ema_alignment'] = float('nan')

        # Price vs EMA200 (in ATR units)
        if ema200 is not None and atr > 0:
            features['price_vs_ema200'] = float((close - ema200[idx]) / atr)
        else:
            features['price_vs_ema200'] = float('nan')

        # Range position (where in 50-bar range)
        if idx >= 50:
            high_50 = float(np.max(highs[idx - 49:idx + 1]))
            low_50 = float(np.min(lows[idx - 49:idx + 1]))
            rng = high_50 - low_50
            features['range_position'] = float((close - low_50) / rng) if rng > 0 else 0.5
        else:
            features['range_position'] = float('nan')

        # ── Cross-Asset Features ─────────────────────────────────────

        if other_df is not None and idx < len(other_df) and idx >= 20:
            other_closes = other_df['Close'].values
            # Rolling correlation of returns
            rets_a = np.diff(np.log(closes[idx - 20:idx + 1]))
            rets_b = np.diff(np.log(other_closes[idx - 20:idx + 1]))
            if len(rets_a) == len(rets_b) and len(rets_a) >= 5:
                corr = np.corrcoef(rets_a, rets_b)[0, 1]
                features['btc_eth_corr_20'] = float(corr) if not np.isnan(corr) else 0.0
            else:
                features['btc_eth_corr_20'] = float('nan')

            # Return divergence (5-bar)
            if idx >= 5 and closes[idx - 5] > 0 and other_closes[idx - 5] > 0:
                ret_a = close / closes[idx - 5] - 1
                ret_b = other_closes[idx] / other_closes[idx - 5] - 1
                features['btc_eth_divergence'] = float(ret_a - ret_b)
            else:
                features['btc_eth_divergence'] = float('nan')
        else:
            features['btc_eth_corr_20'] = float('nan')
            features['btc_eth_divergence'] = float('nan')

        return features

    @staticmethod
    def compute_batch(
        df: pd.DataFrame,
        indices: list,
        other_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute features for multiple bar indices. Returns a DataFrame."""
        rows = []
        for idx in indices:
            feats = MarketFeatureEngine.compute_at_bar(df, idx, other_df)
            rows.append(feats)
        return pd.DataFrame(rows, index=indices)
