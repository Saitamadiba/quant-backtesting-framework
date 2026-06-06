"""Shadow-trade schema normalisers — extracted from 23_Shadow_Trades.py so
the column-alias contracts can be regression-tested without pulling in the
full streamlit page (which has import-time side effects).

Four different writers populate the shadow ecosystem:

  1. ``shared/shadow_tracker.py``  (LR + MM bots)
       table=shadow_trades, canonical schema
  2. ``HyroTrader/lrr_shadow_scanner.py``  (LRR)
       table=lrr_signals, different column names
  3. ``HyroTrader/manual_trades_sync.py``  (Manual)
       table=manual_trades, partial schema (no gate-related fields)
  4. ``HyroTrader/momentum_4h_<asset>_shadow.py``  (Momentum 4H shadow)
       table=shadow_trades but different column names — handled separately

Each per-writer normaliser maps to the canonical schema the Shadow Trades
page reads from. Mappings are **non-destructive** — original columns are
retained so per-strategy Recent Shadows tabs can still pick the native
name.
"""
from __future__ import annotations

import pandas as pd


def normalise_lrr(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``lrr_signals`` columns onto the canonical dashboard schema.

    LRR scanner uses a different shape (table=lrr_signals, single multi-
    asset DB, per-row asset, ``entry/exit_ts_utc`` instead of ``opened/
    closed_at_utc``). The LRR schema also already populates ``exit_reason``
    row-for-row with the same value as ``status`` ('TP'/'SL'/'OPEN') — so
    drop the redundant ``status`` column to avoid duplicate-collision in
    downstream groupbys.

    Aliasing applied (added 2026-06-06):
      - ``regime`` → ``regime_gate`` (Asset-regime tab keys off the latter)
      - ``sl`` → ``stop_loss``, ``tp`` → ``take_profit`` (Recent Shadows
        + RR what-if key off the canonical names)
    """
    if "status" in df.columns and "exit_reason" in df.columns:
        df = df.drop(columns=["status"])
    rename = {
        "entry_ts_utc":  "opened_at_utc",
        "exit_ts_utc":   "closed_at_utc",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Aliases — non-destructive. Skip if the canonical column already exists
    # (defends against future schema where both fields ship).
    if "regime" in df.columns and "regime_gate" not in df.columns:
        df["regime_gate"] = df["regime"]
    if "sl" in df.columns and "stop_loss" not in df.columns:
        df["stop_loss"] = df["sl"]
    if "tp" in df.columns and "take_profit" not in df.columns:
        df["take_profit"] = df["tp"]
    return df


def candidate_filter_columns(df: pd.DataFrame,
                              base_cols: list[str],
                              *, max_unique: int = 50,
                              max_unique_int: int = 24) -> list[str]:
    """Return columns that are sensible to expose as multiselect filters.

    A column qualifies when:
      - it appears in ``base_cols`` (the strategy's visible column list)
      - it has at least 2 distinct non-null values (a 1-value column is
        useless as a filter — there's nothing to narrow down)
      - it has at most ``max_unique`` unique values (otherwise the
        multiselect becomes a noise generator: think ``entry_price`` with
        500 distinct floats)
      - it's categorical-ish: object/string/bool/category, OR a
        low-cardinality int (hour_et 0-23, htf_agree 0/1, cross_count 0-9)

    Used by the Shadow Trades browser's per-tab filter strip. Extracted
    so the rule for "what's a filter-friendly column" can be regression-
    tested without firing up streamlit.
    """
    out: list[str] = []
    for c in base_cols:
        if c not in df.columns:
            continue
        n_unique = df[c].dropna().nunique()
        if n_unique < 2 or n_unique > max_unique:
            continue
        dt = df[c].dtype
        if (dt == object
                or pd.api.types.is_bool_dtype(dt)
                or pd.api.types.is_string_dtype(dt)
                or dt.name == "category"
                or (pd.api.types.is_integer_dtype(dt)
                    and n_unique <= max_unique_int)):
            out.append(c)
    return out


def normalise_manual(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``manual_trades`` columns onto the canonical dashboard schema.

    The manual_trades_sync writer uses canonical ``opened_at_utc`` /
    ``closed_at_utc`` already, so the only step is deriving ``asset`` from
    ``symbol`` (BTCUSDT → BTC).

    Honest-NaN policy: manual trades intentionally have no
    ``block_reason`` (nothing rejected them), ``session`` (not enforced),
    or ``r_multiple`` (no SL/TP attached). These remain NaN so the
    page's strategy-aware Recent Shadows tab can hide the inapplicable
    columns for the Manual sub-view.
    """
    if "symbol" in df.columns:
        df = df.copy()
        df["asset"] = (df["symbol"].astype(str)
                       .str.replace("USDT", "", regex=False))
    return df
