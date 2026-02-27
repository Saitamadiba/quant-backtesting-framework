"""
Centralized OHLCV data validation for the backtesting framework.
Ensures data integrity at every ingestion point to prevent garbage-in-garbage-out.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when OHLCV data fails critical validation checks."""
    pass


def validate_ohlcv(df: pd.DataFrame, strict: bool = False, fix: bool = True) -> pd.DataFrame:
    """
    Validate and optionally fix OHLCV data integrity.

    Checks performed:
    1. Required columns exist (open/Open, high/High, low/Low, close/Close, volume/Volume)
    2. No NaN/Inf in OHLCV columns (forward-fills if fix=True)
    3. high >= max(open, close) for every row
    4. low <= min(open, close) for every row
    5. volume >= 0
    6. Timestamps are monotonically increasing
    7. No duplicate timestamps
    8. Gap detection (logs warnings)

    Args:
        df: DataFrame with OHLCV data (index should be datetime)
        strict: If True, raise DataValidationError on any issue. If False, warn and fix.
        fix: If True, attempt to fix issues (ffill NaN, drop duplicates, sort index)

    Returns:
        Validated (and optionally fixed) DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to validation")
        return df

    df = df.copy()

    # Normalize column names (handle both 'open' and 'Open')
    col_map = {}
    for col in df.columns:
        if col.lower() in ('open', 'high', 'low', 'close', 'volume'):
            col_map[col] = col  # keep original casing

    # Detect which casing is used
    has_upper = any(c[0].isupper() for c in col_map.keys())
    o, h, l, c, v = ('Open', 'High', 'Low', 'Close', 'Volume') if has_upper else ('open', 'high', 'low', 'close', 'volume')

    required = [o, h, l, c, v]
    missing = [col for col in required if col not in df.columns]
    if missing:
        msg = f"Missing required OHLCV columns: {missing}"
        if strict:
            raise DataValidationError(msg)
        logger.error(msg)
        return df

    issues = []

    # Check for NaN/Inf
    for col in [o, h, l, c]:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col].astype(float)).sum() if df[col].dtype != object else 0
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values in {col}")
            if fix:
                df[col] = df[col].ffill().bfill()
        if inf_count > 0:
            issues.append(f"{inf_count} Inf values in {col}")
            if fix:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Volume NaN (fill with 0)
    vol_nan = df[v].isna().sum()
    if vol_nan > 0:
        issues.append(f"{vol_nan} NaN values in {v}")
        if fix:
            df[v] = df[v].fillna(0)

    # OHLCV consistency: high >= max(open, close), low <= min(open, close)
    high_violations = (df[h] < df[[o, c]].max(axis=1)).sum()
    low_violations = (df[l] > df[[o, c]].min(axis=1)).sum()
    if high_violations > 0:
        issues.append(f"{high_violations} bars where High < max(Open, Close)")
        if fix:
            df[h] = df[[h, o, c]].max(axis=1)
    if low_violations > 0:
        issues.append(f"{low_violations} bars where Low > min(Open, Close)")
        if fix:
            df[l] = df[[l, o, c]].min(axis=1)

    # Negative volume
    neg_vol = (df[v] < 0).sum()
    if neg_vol > 0:
        issues.append(f"{neg_vol} bars with negative volume")
        if fix:
            df[v] = df[v].abs()

    # Timestamp ordering
    if hasattr(df.index, 'is_monotonic_increasing') and not df.index.is_monotonic_increasing:
        issues.append("Timestamps not monotonically increasing")
        if fix:
            df = df.sort_index()

    # Duplicate timestamps
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        issues.append(f"{dup_count} duplicate timestamps")
        if fix:
            df = df[~df.index.duplicated(keep='last')]

    # Gap detection (warn only, don't fix)
    if len(df) > 1 and hasattr(df.index, 'to_series'):
        diffs = df.index.to_series().diff()
        if len(diffs.dropna()) > 0:
            median_diff = diffs.dropna().median()
            if median_diff > pd.Timedelta(0):
                large_gaps = (diffs > median_diff * 3).sum()
                if large_gaps > 0:
                    issues.append(f"{large_gaps} gaps detected (>3x median interval)")

    # Report
    if issues:
        action = "raised" if strict else ("fixed" if fix else "detected")
        for issue in issues:
            logger.warning(f"OHLCV validation: {issue} [{action}]")
        if strict:
            raise DataValidationError(f"Data validation failed: {'; '.join(issues)}")

    return df
