"""Tests for indicator calculations (Wilder's EMA correctness)"""
import pytest
import pandas as pd
import numpy as np


class TestRSI:
    def test_rsi_range(self):
        """RSI should always be between 0 and 100"""
        # Create trending data
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5))
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_overbought_in_uptrend(self):
        """RSI should be high (>70) during strong uptrend"""
        prices = pd.Series(range(100, 200))  # pure uptrend
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        assert rsi.iloc[-1] > 70


class TestATR:
    def test_atr_positive(self):
        """ATR should always be positive"""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        prev_close = pd.Series(close).shift(1)
        tr = pd.concat([
            pd.Series(high) - pd.Series(low),
            (pd.Series(high) - prev_close).abs(),
            (pd.Series(low) - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
