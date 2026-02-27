"""Tests for backtrader_framework.data.validation"""
import pytest
import pandas as pd
import numpy as np
from backtrader_framework.data.validation import validate_ohlcv, DataValidationError


class TestValidateOHLCV:
    def _make_df(self, n=100):
        """Create a valid OHLCV DataFrame for testing."""
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3
        open_ = close + np.random.randn(n) * 0.1
        volume = np.random.randint(100, 10000, n).astype(float)
        return pd.DataFrame({
            'Open': open_, 'High': high, 'Low': low,
            'Close': close, 'Volume': volume
        }, index=dates)

    def test_valid_data_passes(self):
        df = self._make_df()
        result = validate_ohlcv(df)
        assert len(result) == len(df)

    def test_empty_df(self):
        result = validate_ohlcv(pd.DataFrame())
        assert result.empty

    def test_nan_values_fixed(self):
        df = self._make_df()
        df.iloc[5, 0] = np.nan  # NaN in Open
        result = validate_ohlcv(df, fix=True)
        assert not result['Open'].isna().any()

    def test_strict_mode_raises(self):
        df = self._make_df()
        df.iloc[5, 0] = np.nan
        with pytest.raises(DataValidationError):
            validate_ohlcv(df, strict=True)

    def test_high_low_violation_fixed(self):
        df = self._make_df()
        df.iloc[10, 1] = df.iloc[10, 2] - 1  # High < Low
        result = validate_ohlcv(df, fix=True)
        assert (result['High'] >= result['Low']).all()

    def test_negative_volume_fixed(self):
        df = self._make_df()
        df.iloc[5, 4] = -100
        result = validate_ohlcv(df, fix=True)
        assert (result['Volume'] >= 0).all()

    def test_duplicate_timestamps_removed(self):
        df = self._make_df()
        df = pd.concat([df, df.iloc[[5]]])  # duplicate row
        result = validate_ohlcv(df, fix=True)
        assert not result.index.duplicated().any()

    def test_unsorted_timestamps_fixed(self):
        df = self._make_df()
        df = df.iloc[::-1]  # reverse order
        result = validate_ohlcv(df, fix=True)
        assert result.index.is_monotonic_increasing

    def test_lowercase_columns(self):
        df = self._make_df()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        result = validate_ohlcv(df)
        assert len(result) == len(df)
