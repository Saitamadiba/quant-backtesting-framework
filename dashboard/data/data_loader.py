"""Unified data loader: combine live VPS + backtest sources with caching."""

import pandas as pd
import streamlit as st

from config import TRADE_SCHEMA_COLS
from data.schema_normalizer import load_all_live_trades, load_all_backtest_trades


@st.cache_data(ttl=60)
def get_live_trades() -> pd.DataFrame:
    return load_all_live_trades()


@st.cache_data(ttl=300)
def get_backtest_trades() -> pd.DataFrame:
    return load_all_backtest_trades()


def get_all_trades(source_filter: str = "All") -> pd.DataFrame:
    """Load and merge all trade data with deduplication.

    source_filter: "All", "Live", or "Backtest"
    """
    frames = []

    if source_filter in ("All", "Live"):
        live = get_live_trades()
        if not live.empty:
            frames.append(live)

    if source_filter in ("All", "Backtest"):
        bt = get_backtest_trades()
        if not bt.empty:
            frames.append(bt)

    if not frames:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    df = pd.concat(frames, ignore_index=True)

    # Dedup: prefer Live over Backtest for same (strategy, symbol, entry_time, entry_price)
    dedup_cols = ["strategy", "symbol", "entry_time", "entry_price"]
    existing = [c for c in dedup_cols if c in df.columns]
    if existing and "source" in df.columns:
        source_priority = {"Live": 0, "Backtest": 1}
        df["_priority"] = df["source"].map(source_priority).fillna(2)
        df = df.sort_values("_priority").drop_duplicates(subset=existing, keep="first")
        df = df.drop(columns=["_priority"])

    # Ensure datetime types
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Exclude open trades — they have no exit/PnL and break cumsum/stats
    if "is_open" in df.columns:
        df = df[~df["is_open"].fillna(False)]

    # Sort by entry time
    df = df.sort_values("entry_time", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=60)
def get_open_positions(symbol: str | None = None) -> pd.DataFrame:
    """Return only currently open positions from live VPS data."""
    live = get_live_trades()
    if live.empty or "is_open" not in live.columns:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)
    df = live[live["is_open"] == True].copy()  # noqa: E712
    if symbol and not df.empty:
        df = df[df["symbol"] == symbol]
    return df.reset_index(drop=True)


@st.cache_data(ttl=300)
def get_aggregated_stats(df: pd.DataFrame = None) -> pd.DataFrame:
    """Aggregate per-strategy-symbol summary stats."""
    if df is None:
        df = get_all_trades()
    if df.empty:
        return pd.DataFrame()

    groups = df.groupby(["strategy", "symbol", "source"])

    def _stats(g):
        total = len(g)
        wins = (g["pnl_usd"] > 0).sum()
        losses = (g["pnl_usd"] < 0).sum()
        win_rate = wins / total if total > 0 else 0
        gross_profit = g.loc[g["pnl_usd"] > 0, "pnl_usd"].sum()
        gross_loss = g.loc[g["pnl_usd"] < 0, "pnl_usd"].abs().sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0
        total_r = g["r_multiple"].sum() if "r_multiple" in g else 0
        avg_r = g["r_multiple"].mean() if "r_multiple" in g else 0

        # Max drawdown on cumulative PnL
        cum = g["pnl_usd"].cumsum()
        peak = cum.cummax()
        dd = (peak - cum)
        max_dd = dd.max() if not dd.empty else 0

        return pd.Series({
            "Trades": total,
            "Win Rate": win_rate,
            "Total PnL": g["pnl_usd"].sum(),
            "Total R": total_r,
            "Avg R": avg_r,
            "Profit Factor": profit_factor,
            "Max DD": max_dd,
        })

    return groups.apply(_stats).reset_index()
