"""Reusable sidebar filter controls."""

from datetime import date, timedelta
import streamlit as st
import pandas as pd

from config import STRATEGIES


def strategy_filter(df: pd.DataFrame, key_prefix: str = "") -> list:
    options = sorted(df["strategy"].dropna().unique()) if not df.empty else list(STRATEGIES.keys())
    return st.sidebar.multiselect("Strategy", options, default=options, key=f"{key_prefix}_strat")


def symbol_filter(df: pd.DataFrame, key_prefix: str = "") -> list:
    options = sorted(df["symbol"].dropna().unique()) if not df.empty else ["BTC", "ETH", "NQ"]
    return st.sidebar.multiselect("Symbol", options, default=options, key=f"{key_prefix}_sym")


def source_filter(key_prefix: str = "") -> str:
    return st.sidebar.radio("Source", ["All", "Live", "Backtest"], key=f"{key_prefix}_src")


def direction_filter(key_prefix: str = "") -> list:
    return st.sidebar.multiselect("Direction", ["Long", "Short"], default=["Long", "Short"], key=f"{key_prefix}_dir")


def date_range_filter(df: pd.DataFrame, key_prefix: str = ""):
    if df.empty or "entry_time" not in df.columns or df["entry_time"].isna().all():
        return None, None
    min_d = df["entry_time"].min().date()
    max_d = df["entry_time"].max().date()
    col1, col2 = st.sidebar.columns(2)
    start = col1.date_input("From", min_d, min_value=min_d, max_value=max_d, key=f"{key_prefix}_from")
    end = col2.date_input("To", max_d, min_value=min_d, max_value=max_d, key=f"{key_prefix}_to")
    return start, end


def session_filter(key_prefix: str = "") -> list:
    options = ["Asian", "London", "New York", "Off-Hours", "Unknown"]
    return st.sidebar.multiselect("Session", options, default=options, key=f"{key_prefix}_sess")


def apply_filters(df: pd.DataFrame, strategies=None, symbols=None, directions=None,
                   date_start=None, date_end=None, sessions=None) -> pd.DataFrame:
    """Apply sidebar filter selections to a trades DataFrame."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if strategies:
        mask &= df["strategy"].isin(strategies)
    if symbols:
        mask &= df["symbol"].isin(symbols)
    if directions:
        mask &= df["direction"].isin(directions)
    if date_start and "entry_time" in df.columns:
        mask &= df["entry_time"].dt.date >= date_start
    if date_end and "entry_time" in df.columns:
        mask &= df["entry_time"].dt.date <= date_end
    if sessions:
        mask &= df["session"].isin(sessions)
    return df[mask].reset_index(drop=True)
