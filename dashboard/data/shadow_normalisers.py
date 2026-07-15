"""Shadow-trade schema normalisers — extracted from 23_Shadow_Trades.py so
the column-alias contracts can be regression-tested without pulling in the
full streamlit page (which has import-time side effects).

Many different writers populate the shadow ecosystem. The canonical schema
the page reads is the one ``shared/shadow_tracker.py`` (LR + MM bots)
writes: ``opened_at_utc / closed_at_utc / direction(BUY|SELL) /
entry_price / stop_loss / take_profit / exit_price / exit_reason(TP|SL|
TIME_EXIT|OPEN) / r_multiple``. Every other writer gets a bridge here:

  1. ``HyroTrader/lrr_shadow_scanner.py``       table=lrr_signals
  2. ``HyroTrader/manual_trades_sync.py``       table=manual_trades
  3. ``HyroTrader/depth_paper_bot.py`` +
     ``HyroTrader/lrr_paper_bot.py``            table=paper_trades
  4. ``HyroTrader/depth_policy_shadow.py``      table=policy_trades
  5. ``ofcs_shadow/paper_bot.py``               table=ofcs_paper_trades
  6. ``HyroTrader/momentum_4h_<asset>_shadow.py`` table=shadow_trades,
     compact column names (entry/sl/tp)
  7. ``HyroTrader/ifvg_sweep_shadow.py``        table=shadow_trades,
     compact names + per-row symbol/timeframe
  8. ``HyroTrader/asia_basket_shadow.py``       table=basket_nights
     (one row per Asia NIGHT, not per trade)
  9. ``HyroTrader/bull_put_spread_bybit.py``    table=shadow_trades
     (options spreads — R defined as pnl / max_loss)
 10. ``HyroTrader/fvg_shadow_sim`` funded sims  table=trades
 11. ``Momentum_Mastery/BTC/btc_momentum_mastery_v2_shadow.py``
     table=trades (full-bot schema, partial-exit experiment)

Each per-writer normaliser maps to the canonical schema. Column mappings
are **non-destructive** — original columns are retained so per-strategy
browser tabs can still pick the native name. Direction VALUES are
canonicalised in place (LONG→BUY, SHORT→SELL, bullish→BUY, bearish→SELL)
because the RR what-if replay treats anything ≠ BUY as a short.

``SHADOW_DB_SPECS`` at the bottom is the single dispatch table the page
loader uses: filename → (table, normaliser).
"""
from __future__ import annotations

import pandas as pd

# Direction canonicalisation — the page's replay machinery assumes BUY/SELL.
_SIDE_MAP = {"LONG": "BUY", "SHORT": "SELL",
             "long": "BUY", "short": "SELL",
             "bullish": "BUY", "bearish": "SELL"}


def _canon_direction(df: pd.DataFrame, src: str = "direction") -> pd.DataFrame:
    """Map LONG/SHORT/bullish/bearish direction values onto BUY/SELL.

    In-place value mapping (not a column alias): the canonical column IS
    ``direction``; only its vocabulary differs across writers. Unknown
    values pass through untouched.
    """
    if src in df.columns:
        df[src] = df[src].map(lambda v: _SIDE_MAP.get(str(v), v)
                              if pd.notna(v) else v)
    return df


def _alias(df: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    """Non-destructive column alias: copy src → dst unless dst exists."""
    if src in df.columns and dst not in df.columns:
        df[dst] = df[src]
    return df


def _open_where_null(df: pd.DataFrame) -> pd.DataFrame:
    """Rows still open carry a NULL/empty exit_reason in several books —
    the page counts open shadows via ``exit_reason == 'OPEN'``."""
    if "exit_reason" in df.columns:
        _blank = df["exit_reason"].isna() | (df["exit_reason"].astype(str)
                                             .str.strip() == "")
        df.loc[_blank, "exit_reason"] = "OPEN"
    else:
        df["exit_reason"] = "OPEN"
    return df


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
    # LRR writes LONG/SHORT; the RR what-if replay assumes BUY/SELL (added
    # 2026-07-06 — before this, LRR longs replayed inverted).
    return _canon_direction(df)


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


# ── Shadow-fleet gap closure normalisers (2026-07-06) ────────────────────────

def normalise_paper_book(df: pd.DataFrame) -> pd.DataFrame:
    """``depth_paper_bot.py`` + ``lrr_paper_bot.py`` books (table=paper_trades).

    Both writers share one skeleton: canonical ``opened_at_utc``/
    ``closed_at_utc`` already present, compact ``sl``/``tp`` names, a
    ``status`` OPEN/CLOSED lifecycle (exit_reason NULL while open), and the
    fee-modeled pair ``r_gross``/``r_net``.  ``r_multiple`` aliases **r_net**
    — the number that decides promotion is net of the 0.10R round-trip toll,
    never the gross print.  The depth book's per-row ``strategy`` column
    (MM/LR/LRR — which detector family produced the signal) is renamed to
    ``family`` so the page-level ``strategy`` label doesn't clobber it; the
    loader folds it into the bot label ("DP MM BTC").
    """
    df = df.copy()
    if "strategy" in df.columns:
        df = df.rename(columns={"strategy": "family"})
    df = _alias(df, "sl", "stop_loss")
    df = _alias(df, "tp", "take_profit")
    df = _alias(df, "r_net", "r_multiple")
    df = _canon_direction(df)
    return _open_where_null(df)


def normalise_depth_policy(df: pd.DataFrame) -> pd.DataFrame:
    """``depth_policy_shadow.py`` counterfactual book (table=policy_trades).

    Every row is CLOSED by construction (it scores an already-resolved
    source trade under the candidate exit policy).  ``r_multiple`` aliases
    ``policy_net`` (fee-adjusted policy outcome); ``native_net`` stays
    alongside so policy-vs-native is one subtraction.  ``policy_exit``
    {HARD, TRAIL, TP} maps onto the canonical exit vocabulary as
    HARD→SL (the −0.40R hard stop is the policy's stop-loss), TRAIL→
    TIME_EXIT (closest canonical bucket for "trailed out"), TP→TP — the
    native ``policy_exit`` column is kept for the honest per-exit split.
    """
    df = df.copy()
    if "strategy" in df.columns:
        df = df.rename(columns={"strategy": "family"})
    df = _alias(df, "entry", "entry_price")
    df = _alias(df, "sl", "stop_loss")
    df = _alias(df, "tp", "take_profit")
    df = _alias(df, "policy_net", "r_multiple")
    if "policy_exit" in df.columns and "exit_reason" not in df.columns:
        df["exit_reason"] = df["policy_exit"].map(
            {"HARD": "SL", "TRAIL": "TIME_EXIT", "TP": "TP"})
    df = _canon_direction(df)
    return _open_where_null(df)


def normalise_ofcs(df: pd.DataFrame) -> pd.DataFrame:
    """``ofcs_shadow/paper_bot.py`` book (table=ofcs_paper_trades).

    Bridges: ``entry_ts``→``opened_at_utc``, ``resolved_at``→
    ``closed_at_utc``, ``realized_r``→``r_multiple``, exit vocabulary
    TIME→TIME_EXIT, LONG/SHORT→BUY/SELL, ``regime5``→``regime_gate``.
    """
    df = df.copy()
    df = df.rename(columns={k: v for k, v in {
        "entry_ts": "opened_at_utc", "resolved_at": "closed_at_utc",
    }.items() if k in df.columns})
    df = _alias(df, "entry", "entry_price")
    df = _alias(df, "sl", "stop_loss")
    df = _alias(df, "tp", "take_profit")
    df = _alias(df, "realized_r", "r_multiple")
    df = _alias(df, "regime5", "regime_gate")
    if "exit_reason" in df.columns:
        df["exit_reason"] = df["exit_reason"].replace({"TIME": "TIME_EXIT"})
    df = _canon_direction(df)
    return _open_where_null(df)


def normalise_momentum4h(df: pd.DataFrame) -> pd.DataFrame:
    """``momentum_4h_<asset>_shadow.py`` books (table=shadow_trades).

    Canonical timestamps already; compact ``entry``/``sl``/``tp`` names and
    LONG/SHORT directions bridged.  ``r_multiple``/``exit_reason`` are
    written by the forward OHLC walk (TP/SL, TIME→TIME_EXIT); NULL = still
    open.
    """
    df = df.copy()
    df = _alias(df, "entry", "entry_price")
    df = _alias(df, "sl", "stop_loss")
    df = _alias(df, "tp", "take_profit")
    if "exit_reason" in df.columns:
        df["exit_reason"] = df["exit_reason"].replace({"TIME": "TIME_EXIT"})
    df = _canon_direction(df)
    return _open_where_null(df)


def normalise_ifvg(df: pd.DataFrame) -> pd.DataFrame:
    """``ifvg_sweep_shadow.py`` (+ the earlier NQ signal shadow) books.

    Same compact shape as the momentum shadows, plus a per-row ``symbol``
    (the DB is multi-cell: symbol × timeframe) which becomes ``asset`` so
    the MULTI loader splits bots per symbol.
    """
    df = normalise_momentum4h(df)
    if "symbol" in df.columns and "asset" not in df.columns:
        df["asset"] = df["symbol"].astype(str)
    return df


def normalise_asia_basket(df: pd.DataFrame) -> pd.DataFrame:
    """``asia_basket_shadow.py`` ledger (table=basket_nights).

    NOT per-trade: one row = one Asia NIGHT of the bounded basket.
    ``r_multiple`` aliases ``mean_r`` (the night's mean per-leg R — the
    quantity the 1%-cap math multiplies); ``night_ret_pct`` (= 1% ×
    mean_r) rides alongside.  A night with all legs resolved is stamped
    TIME_EXIT (nights close by the clock, not by a target); a night still
    filling is OPEN.  No direction/entry columns — the basket is a bundle,
    not a single position.
    """
    df = df.copy()
    if "night" in df.columns:
        df["opened_at_utc"] = pd.to_datetime(df["night"], errors="coerce",
                                             utc=True)
    df = _alias(df, "mean_r", "r_multiple")
    if {"closed_legs", "n_legs"}.issubset(df.columns):
        _done = (df["n_legs"] > 0) & (df["closed_legs"] >= df["n_legs"])
        df["exit_reason"] = _done.map({True: "TIME_EXIT", False: "OPEN"})
    return _open_where_null(df)


def normalise_bullput(df: pd.DataFrame) -> pd.DataFrame:
    """``bull_put_spread_bybit.py`` paper books (table=shadow_trades).

    Options credit spreads — no linear SL/TP geometry, so R is defined as
    ``realized_pnl / max_loss`` (risk 1 unit = the spread's max loss).
    Exit vocabulary: PROFIT_TAKE→TP (the 50%-of-credit harvest),
    SL_TOUCH→SL (short strike touched), EXPIRY→TIME_EXIT.
    """
    df = df.copy()
    df = df.rename(columns={k: v for k, v in {
        "timestamp": "opened_at_utc", "exit_timestamp": "closed_at_utc",
    }.items() if k in df.columns})
    if {"realized_pnl", "max_loss"}.issubset(df.columns):
        _ml = pd.to_numeric(df["max_loss"], errors="coerce")
        df["r_multiple"] = (pd.to_numeric(df["realized_pnl"], errors="coerce")
                            / _ml.where(_ml > 0))
    if "exit_reason" in df.columns:
        df["exit_reason"] = df["exit_reason"].replace(
            {"PROFIT_TAKE": "TP", "SL_TOUCH": "SL", "EXPIRY": "TIME_EXIT"})
    return _open_where_null(df)


def normalise_fvg_funded(df: pd.DataFrame) -> pd.DataFrame:
    """FVG funded-context sim books (table=trades).

    ``r_multiple`` = ``realized_pnl / risk_amount`` (net of commission —
    the sim books commission separately, realized_pnl already carries it).
    Exit vocabulary is prose ("Stop Loss"/"Take Profit") → SL/TP.
    """
    df = df.copy()
    df = df.rename(columns={k: v for k, v in {
        "timestamp": "opened_at_utc",
    }.items() if k in df.columns})
    if {"realized_pnl", "risk_amount"}.issubset(df.columns):
        _risk = pd.to_numeric(df["risk_amount"], errors="coerce")
        df["r_multiple"] = (pd.to_numeric(df["realized_pnl"], errors="coerce")
                            / _risk.where(_risk > 0))
    if "exit_reason" in df.columns:
        df["exit_reason"] = df["exit_reason"].replace(
            {"Stop Loss": "SL", "Take Profit": "TP", "Time Exit": "TIME_EXIT"})
    df = _canon_direction(df)
    return _open_where_null(df)


def normalise_mm_v2_shadow(df: pd.DataFrame) -> pd.DataFrame:
    """``btc_momentum_mastery_v2_shadow.py`` (table=trades, full-bot schema).

    The partial-exit experiment runs the whole MM v2 bot in shadow, so its
    DB is the live-bot trades schema: ``signal_type`` carries the side,
    R must be reconstructed as ``realized_pnl / (|entry − stop| × size)``
    (partial exits make per-leg R fuzzy; this is the blended-trade R).
    """
    df = df.copy()
    df = df.rename(columns={k: v for k, v in {
        "timestamp": "opened_at_utc", "exit_timestamp": "closed_at_utc",
    }.items() if k in df.columns})
    df = _alias(df, "signal_type", "direction")
    df = _canon_direction(df)
    need = {"realized_pnl", "entry_price", "stop_loss", "position_size"}
    if need.issubset(df.columns) and "r_multiple" not in df.columns:
        _risk = ((pd.to_numeric(df["entry_price"], errors="coerce")
                  - pd.to_numeric(df["stop_loss"], errors="coerce")).abs()
                 * pd.to_numeric(df["position_size"], errors="coerce"))
        df["r_multiple"] = (pd.to_numeric(df["realized_pnl"], errors="coerce")
                            / _risk.where(_risk > 0))
    if "exit_reason" in df.columns:
        df["exit_reason"] = df["exit_reason"].replace(
            {"Stop Loss": "SL", "Take Profit": "TP", "Time Exit": "TIME_EXIT",
             "STOP_LOSS": "SL", "HARD_STOP": "SL", "TAKE_PROFIT": "TP",
             "PARTIAL_TP": "TP", "TIME": "TIME_EXIT"})
    return _open_where_null(df)


def normalise_mm15m(df: pd.DataFrame) -> pd.DataFrame:
    """``HyroTrader/mm_15m_shadow.py`` (table=signals) — the MM 15m
    forward-shadow (MM_15M_SHADOW_SPEC.md, Tier-3 record-only).

    Native schema: confirm_bar_utc (signal bar) / exit_utc (market exit bar)
    / tp1+tp2 ladder / outcome in {loss, breakeven, win_tp1, win_tp2,
    timeout} / r_net (taker-cost R, the headline number). ``closed_at_utc``
    natively holds the BOOKING wall-clock, so the market exit ts overrides
    it for the canonical column. ``breakeven`` is a stop-hit at the moved
    stop → SL; the native ``outcome`` column is retained for detail tabs.
    """
    df = df.copy()
    df = _alias(df, "confirm_bar_utc", "opened_at_utc")
    if "exit_utc" in df.columns:
        if "closed_at_utc" in df.columns:
            df["closed_at_utc"] = df["exit_utc"].where(
                df["exit_utc"].notna(), df["closed_at_utc"])
        else:
            df["closed_at_utc"] = df["exit_utc"]
    df = _canon_direction(df)
    df = _alias(df, "tp1", "take_profit")
    df = _alias(df, "r_net", "r_multiple")
    if "outcome" in df.columns and "exit_reason" not in df.columns:
        df["exit_reason"] = df["outcome"].map(
            {"loss": "SL", "breakeven": "SL", "win_tp1": "TP",
             "win_tp2": "TP", "timeout": "TIME_EXIT"})
    return _open_where_null(df)


def normalise_mm5m(df: pd.DataFrame) -> pd.DataFrame:
    """``HyroTrader/mm_5m_shadow.py`` (table=signals) — the MM 5m
    maker-scalper forward-shadow (MM_5M_SHADOW_SPEC.md, Tier-3 record-only,
    touch-gated limit fills).

    Same shape as the 15m arm plus the limit state machine: CANCELLED rows
    (TTL expired, never filled) are NOT trades — dropped here so KPIs read
    fills only; PENDING/FILLED rows count as OPEN. ``limit_price`` is the
    entry; ``r_net`` (asym maker/taker) is the headline R.
    """
    df = df.copy()
    if "status" in df.columns:
        df = df[df["status"] != "CANCELLED"].copy()
    df = _alias(df, "confirm_bar_utc", "opened_at_utc")
    if "exit_utc" in df.columns:
        if "closed_at_utc" in df.columns:
            df["closed_at_utc"] = df["exit_utc"].where(
                df["exit_utc"].notna(), df["closed_at_utc"])
        else:
            df["closed_at_utc"] = df["exit_utc"]
    df = _canon_direction(df)
    df = _alias(df, "limit_price", "entry_price")
    df = _alias(df, "tp1", "take_profit")
    df = _alias(df, "r_net", "r_multiple")
    if "outcome" in df.columns and "exit_reason" not in df.columns:
        df["exit_reason"] = df["outcome"].map(
            {"loss": "SL", "breakeven": "SL", "win_tp1": "TP",
             "win_tp2": "TP", "timeout": "TIME_EXIT"})
    return _open_where_null(df)


# ── Loader dispatch table ─────────────────────────────────────────────────────
# filename → (table, normaliser|None). Files absent here read the canonical
# shared/shadow_tracker.py shape: table=shadow_trades, no bridge needed.
SHADOW_DB_SPECS: dict[str, tuple[str, object]] = {
    "lrr_shadow_trades.db":      ("lrr_signals",       normalise_lrr),
    "manual_trades.db":          ("manual_trades",     normalise_manual),
    "depth_paper_book.db":       ("paper_trades",      normalise_paper_book),
    "lrr_paper_book.db":         ("paper_trades",      normalise_paper_book),
    "ofcs_paper_book.db":        ("ofcs_paper_trades", normalise_ofcs),
    "depth_policy_book.db":      ("policy_trades",     normalise_depth_policy),
    "momentum_4h_sol_shadow.db": ("shadow_trades",     normalise_momentum4h),
    "momentum_4h_xrp_shadow.db": ("shadow_trades",     normalise_momentum4h),
    "momentum_4h_ltc_shadow.db": ("shadow_trades",     normalise_momentum4h),
    "ifvg_sweep_shadow.db":      ("shadow_trades",     normalise_ifvg),
    "ifvg_nq_signal_shadow.db":  ("shadow_trades",     normalise_ifvg),
    "asia_basket_shadow.db":     ("basket_nights",     normalise_asia_basket),
    "mm_btc_partial_shadow.db":  ("trades",            normalise_mm_v2_shadow),
    "bullput_btc_shadow.db":     ("shadow_trades",     normalise_bullput),
    "bullput_eth_shadow.db":     ("shadow_trades",     normalise_bullput),
    "fvg_btc_funded_shadow.db":  ("trades",            normalise_fvg_funded),
    "fvg_nq_funded_shadow.db":   ("trades",            normalise_fvg_funded),
    "mm_15m_shadow.db":          ("signals",           normalise_mm15m),
    "mm_5m_shadow.db":           ("signals",           normalise_mm5m),
}


def table_for(local_name: str) -> str:
    """The SQL table the page should read for this synced file."""
    return SHADOW_DB_SPECS.get(local_name, ("shadow_trades", None))[0]


def normaliser_for(local_name: str):
    """The schema bridge for this synced file (None = canonical already)."""
    return SHADOW_DB_SPECS.get(local_name, ("shadow_trades", None))[1]
