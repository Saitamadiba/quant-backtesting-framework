"""Page 3: Searchable, filterable trade log."""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Trade Journal", page_icon="📓", layout="wide")
st.title("📓 Trade Journal")

from data.data_loader import get_all_trades
from components.filters import (
    strategy_filter, symbol_filter, direction_filter,
    date_range_filter, session_filter, source_filter, apply_filters,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
src = source_filter(key_prefix="tj")
df_all = get_all_trades(source_filter=src)

strategies = strategy_filter(df_all, key_prefix="tj")
symbols = symbol_filter(df_all, key_prefix="tj")
directions = direction_filter(key_prefix="tj")
sessions = session_filter(key_prefix="tj")
date_start, date_end = date_range_filter(df_all, key_prefix="tj")

# PnL range filter
st.sidebar.subheader("P&L Range")
if not df_all.empty and "pnl_usd" in df_all.columns:
    pnl_min = float(df_all["pnl_usd"].min())
    pnl_max = float(df_all["pnl_usd"].max())
    if pnl_min < pnl_max:
        pnl_range = st.sidebar.slider("P&L ($)", pnl_min, pnl_max, (pnl_min, pnl_max), key="tj_pnl")
    else:
        pnl_range = (pnl_min, pnl_max)
else:
    pnl_range = None

# ── Filter ────────────────────────────────────────────────────────────────────
df = apply_filters(
    df_all, strategies=strategies, symbols=symbols, directions=directions,
    date_start=date_start, date_end=date_end, sessions=sessions,
)

# Apply PnL range
if pnl_range and not df.empty:
    df = df[(df["pnl_usd"] >= pnl_range[0]) & (df["pnl_usd"] <= pnl_range[1])]

# ── Stats bar ─────────────────────────────────────────────────────────────────
st.caption("Quick summary of your filtered trade set. Use the sidebar filters to isolate specific strategies, sessions, or date ranges and study their characteristics.")
c1, c2, c3 = st.columns(3)
c1.metric("Filtered Trades", f"{len(df):,}",
          help=f"Number of trades matching your current filter criteria. Currently {len(df):,}.")
if not df.empty:
    wr = (df['pnl_usd'] > 0).mean()
    c2.metric("Win Rate", f"{wr:.1%}",
              help=f"Win rate of filtered trades. Currently {wr:.1%}. Compare this against overall win rate to see if your filter is isolating better or worse subsets.")
    tot = df['pnl_usd'].sum()
    c3.metric("Total P&L", f"${tot:,.2f}",
              help=f"Total P&L of filtered trades. Currently ${tot:,.2f}. Useful for measuring the dollar impact of specific filters (e.g., a particular session or direction).")

st.markdown("---")

# ── Table ─────────────────────────────────────────────────────────────────────
st.caption("Full trade log with per-trade details. The 'Max DD %' column shows the maximum adverse excursion (worst intra-trade drawdown) as a percentage of entry price. Trades highlighted in red had drawdowns of 6% or more — these represent excessive risk exposure and should be reviewed for stop-loss adequacy.")
if not df.empty:
    display_cols = [
        "strategy", "symbol", "source", "direction", "entry_time", "exit_time",
        "entry_price", "exit_price", "pnl_usd", "r_multiple", "session", "exit_reason",
    ]
    display = df[[c for c in display_cols if c in df.columns]].copy()

    # Compute max drawdown % per trade from MAE or price data
    if "mae" in df.columns and "entry_price" in df.columns:
        # MAE = max adverse excursion in $ or price units
        mae_vals = df["mae"].fillna(0).abs()
        entry_vals = df["entry_price"].replace(0, pd.NA)
        display["Max DD %"] = (mae_vals / entry_vals * 100).round(2)
    elif "entry_price" in df.columns and "exit_price" in df.columns:
        # Fallback: use trade PnL as proxy for drawdown
        entry_vals = df["entry_price"].replace(0, pd.NA)
        # For losing trades, the loss itself is a lower bound on drawdown
        loss_pct = (df["pnl_usd"].clip(upper=0).abs() / entry_vals * 100).fillna(0)
        display["Max DD %"] = loss_pct.round(2)
    else:
        display["Max DD %"] = 0.0

    # Color PnL and highlight high-drawdown trades
    def _style_row(row):
        styles = [""] * len(row)
        col_names = list(row.index)

        # Color PnL column
        if "pnl_usd" in col_names:
            pnl_idx = col_names.index("pnl_usd")
            val = row["pnl_usd"]
            if pd.notna(val):
                if val > 0:
                    styles[pnl_idx] = "color: #4CAF50"
                elif val < 0:
                    styles[pnl_idx] = "color: #F44336"

        # Flash entire row red if Max DD >= 6%
        if "Max DD %" in col_names:
            dd_val = row["Max DD %"]
            if pd.notna(dd_val) and dd_val >= 6.0:
                styles = ["background-color: rgba(244,67,54,0.15); color: #F44336"] * len(row)

        return styles

    styled = display.style.apply(_style_row, axis=1)
    st.caption(
        "**Column Reference** — "
        "**strategy**: trading strategy name | "
        "**symbol**: trading pair (e.g. BTCUSD) | "
        "**source**: 'live' (VPS) or 'backtest' (simulated) | "
        "**direction**: long or short | "
        "**entry_time / exit_time**: trade open/close timestamps (UTC) | "
        "**entry_price / exit_price**: prices at open and close | "
        "**pnl_usd**: realized profit/loss in USD (green = profit, red = loss) | "
        "**r_multiple**: risk-adjusted return (1R = risked amount, negative = loss) | "
        "**session**: market session at entry (Asian, London, New York) | "
        "**exit_reason**: why the trade closed (TP, SL, trailing, manual) | "
        "**Max DD %**: worst intra-trade drawdown as % of entry price (rows highlighted red >= 6%)"
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

    # Export
    csv = display.to_csv(index=False)
    st.download_button("Download CSV", csv, "trades.csv", "text/csv")

    # ── Data-Driven Commentary ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Analysis & Recommendations")

    points = []

    # Sample size context
    total = len(df)
    points.append(f"Showing **{total} trades** matching your current filters.")

    # Largest winner / loser
    if "pnl_usd" in df.columns and total > 0:
        best_idx = df["pnl_usd"].idxmax()
        worst_idx = df["pnl_usd"].idxmin()
        best = df.loc[best_idx]
        worst = df.loc[worst_idx]
        points.append(
            f"**Largest winner**: ${best['pnl_usd']:,.2f} "
            f"({best.get('strategy', '?')} {best.get('symbol', '?')} {best.get('direction', '?')}, "
            f"{best.get('entry_time', 'N/A')}). "
            f"**Largest loser**: ${worst['pnl_usd']:,.2f} "
            f"({worst.get('strategy', '?')} {worst.get('symbol', '?')})."
        )

        # Asymmetry check
        avg_win = df.loc[df["pnl_usd"] > 0, "pnl_usd"].mean() if (df["pnl_usd"] > 0).any() else 0
        avg_loss = df.loc[df["pnl_usd"] < 0, "pnl_usd"].mean() if (df["pnl_usd"] < 0).any() else 0
        if avg_loss != 0:
            ratio = abs(avg_win / avg_loss)
            if ratio < 1.0:
                points.append(
                    f"Average win (${avg_win:,.2f}) is smaller than average loss (${avg_loss:,.2f}), "
                    f"giving a **{ratio:.2f}x** win/loss ratio. You need a win rate above "
                    f"**{1/(1+ratio):.0%}** to be profitable with this ratio."
                )
            elif ratio > 2.0:
                points.append(
                    f"Average win (${avg_win:,.2f}) is **{ratio:.1f}x** the average loss — "
                    f"strong risk/reward asymmetry."
                )

    # High-drawdown trade analysis
    if "Max DD %" in display.columns:
        high_dd = display[display["Max DD %"] >= 6.0]
        if not high_dd.empty:
            pct_high = len(high_dd) / total * 100
            points.append(
                f"**{len(high_dd)} trades ({pct_high:.0f}%)** had intra-trade drawdowns of 6% or more "
                f"(highlighted red above). These represent excessive risk exposure — review stop-loss "
                f"placement for these setups."
            )

    # Exit reason breakdown
    if "exit_reason" in df.columns:
        exit_counts = df["exit_reason"].value_counts()
        top_exit = exit_counts.index[0] if not exit_counts.empty else None
        if top_exit:
            pct = exit_counts.iloc[0] / total * 100
            points.append(
                f"Most common exit reason: **{top_exit}** ({pct:.0f}% of trades). "
                + ("High stop-loss exit rate suggests stops may be too tight." if "stop" in str(top_exit).lower() and pct > 50
                   else "")
            )

    # Duration outliers
    if "duration_minutes" in df.columns:
        dur = df["duration_minutes"].dropna()
        if len(dur) > 5:
            median_dur = dur.median()
            long_trades = dur[dur > median_dur * 5]
            if not long_trades.empty:
                points.append(
                    f"**{len(long_trades)} trades** lasted more than 5x the median duration "
                    f"({median_dur:.0f} min). Extended trade duration often indicates missed exit signals "
                    f"or time-decay in the setup."
                )

    # Streak analysis
    if "pnl_usd" in df.columns and total >= 10:
        wins_losses = (df["pnl_usd"] > 0).astype(int).values
        max_loss_streak = 0
        current_streak = 0
        for wl in wins_losses:
            if wl == 0:
                current_streak += 1
                max_loss_streak = max(max_loss_streak, current_streak)
            else:
                current_streak = 0
        if max_loss_streak >= 5:
            points.append(
                f"Longest losing streak: **{max_loss_streak} consecutive losses**. "
                f"Ensure your position sizing can absorb streaks of this length without "
                f"exceeding drawdown limits."
            )

    for p in points:
        st.markdown(f"- {p}")

else:
    st.info("No trades match the current filters.")
