"""Page 1: All-bot summary overview."""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Overview", page_icon="📋", layout="wide")
st.title("📋 Overview")

from data.data_loader import get_all_trades, get_aggregated_stats
from components.kpi_cards import kpi_row
from components.charts import strategy_comparison_bar, cumulative_pnl_line
from components.filters import source_filter
from config import DB_STRATEGY_MAP
from data.vps_sync import get_cached_db_status

# ── VPS Status Bar ────────────────────────────────────────────────────────────
st.caption("Shows whether each bot's trade database has been synced from the VPS. Green means data is available locally; red means you need to sync. Stale data leads to stale decisions.")
db_status = get_cached_db_status()
cols = st.columns(len(DB_STRATEGY_MAP))
for i, (db_file, (strategy, symbol)) in enumerate(DB_STRATEGY_MAP.items()):
    info = db_status.get(db_file, {})
    if info.get("exists"):
        cols[i].success(f"{strategy} {symbol} — {info.get('size_kb', '?')}KB", icon="🟢")
    else:
        cols[i].error(f"{strategy} {symbol} — not synced", icon="🔴")

st.markdown("---")

# ── Filters ───────────────────────────────────────────────────────────────────
src = source_filter(key_prefix="ov")

# ── Load Data ─────────────────────────────────────────────────────────────────
df = get_all_trades(source_filter=src)

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.caption("Top-line health check of your trading operation. If win rate drops below 40% or avg R turns negative, investigate individual strategies immediately.")
kpi_row(df)

st.markdown("---")

# ── Per-Strategy Summary Table ────────────────────────────────────────────────
st.subheader("Strategy Summary")
st.caption("Compare all strategies side-by-side. Look for which strategies are carrying their weight (positive Profit Factor, high Total R) and which are dragging overall performance down.")
stats = get_aggregated_stats(df)
if not stats.empty:
    display = stats.copy()
    display["Win Rate"] = display["Win Rate"].apply(lambda x: f"{x:.1%}")
    display["Profit Factor"] = display["Profit Factor"].apply(
        lambda x: f"{x:.2f}" if x != float("inf") else "∞"
    )
    display["Total PnL"] = display["Total PnL"].apply(lambda x: f"${x:,.2f}")
    display["Max DD"] = display["Max DD"].apply(lambda x: f"${x:,.2f}")
    display["Avg R"] = display["Avg R"].apply(lambda x: f"{x:.2f}")
    display["Total R"] = display["Total R"].apply(lambda x: f"{x:.1f}")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strategy": st.column_config.TextColumn(
                "Strategy",
                help="The trading strategy name (e.g., SBS, QFL). Each row represents one strategy-symbol combination.",
            ),
            "Symbol": st.column_config.TextColumn(
                "Symbol",
                help="The trading pair this strategy trades (e.g., BTCUSD, ETHUSD).",
            ),
            "Source": st.column_config.TextColumn(
                "Source",
                help="Data source — 'live' for real VPS trades, 'backtest' for simulated historical results.",
            ),
            "Trades": st.column_config.NumberColumn(
                "Trades",
                help="Total number of closed trades for this strategy. Low trade counts mean statistics are less reliable.",
            ),
            "Win Rate": st.column_config.TextColumn(
                "Win Rate",
                help="Percentage of trades that closed in profit. Above 50% is good, but must be evaluated alongside Avg R and Profit Factor.",
            ),
            "Profit Factor": st.column_config.TextColumn(
                "Profit Factor",
                help="Gross profit divided by gross loss. Above 1.0 means the strategy is net profitable. Above 2.0 is strong.",
            ),
            "Total PnL": st.column_config.TextColumn(
                "Total PnL",
                help="Total realized profit/loss in USD for this strategy. The bottom-line dollar result.",
            ),
            "Max DD": st.column_config.TextColumn(
                "Max DD",
                help="Maximum drawdown — the largest peak-to-trough decline in equity. Measures worst-case pain.",
            ),
            "Avg R": st.column_config.TextColumn(
                "Avg R",
                help="Average R-multiple per trade. Positive means winners outpace losers on a risk-adjusted basis. Negative means the opposite.",
            ),
            "Total R": st.column_config.TextColumn(
                "Total R",
                help="Sum of all R-multiples. The strategy's total contribution in risk units — higher is better.",
            ),
        },
    )
else:
    st.info("No aggregated stats available.")

# ── Charts ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.caption("Side-by-side strategy comparison — quickly spot which strategy dominates and which underperforms relative to its peers.")
    if not stats.empty:
        st.plotly_chart(strategy_comparison_bar(stats), key="ov_strat_bar")
with c2:
    st.caption("Cumulative P&L over time — an upward-sloping line means consistent profitability. Flat or declining regions signal drawdown periods worth investigating.")
    st.plotly_chart(cumulative_pnl_line(df), key="ov_cum_pnl")

# ── Recent Trades Feed ────────────────────────────────────────────────────────
st.subheader("Recent Trades")
st.caption("The last 10 trades across all bots. Use this as a real-time pulse check — if you see a streak of losses, drill into the strategy or session responsible.")
if not df.empty:
    recent = df.head(10)[["strategy", "symbol", "source", "direction", "entry_time",
                           "pnl_usd", "r_multiple", "session", "exit_reason"]]
    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
        column_config={
            "strategy": st.column_config.TextColumn(
                "strategy",
                help="Which trading strategy generated this trade.",
            ),
            "symbol": st.column_config.TextColumn(
                "symbol",
                help="The trading pair (e.g., BTCUSD, ETHUSD).",
            ),
            "source": st.column_config.TextColumn(
                "source",
                help="'live' = real VPS execution, 'backtest' = simulated.",
            ),
            "direction": st.column_config.TextColumn(
                "direction",
                help="Trade direction — 'long' (betting price goes up) or 'short' (betting price goes down).",
            ),
            "entry_time": st.column_config.DatetimeColumn(
                "entry_time",
                help="Timestamp when the trade was opened (UTC).",
            ),
            "pnl_usd": st.column_config.NumberColumn(
                "pnl_usd",
                help="Realized profit or loss in USD for this trade. Green = profit, red = loss.",
            ),
            "r_multiple": st.column_config.NumberColumn(
                "r_multiple",
                help="How many R (risk units) this trade returned. 1R = you risked $X and made $X. Negative means a loss.",
            ),
            "session": st.column_config.TextColumn(
                "session",
                help="Which trading session was active at entry: Asian, London, or New York.",
            ),
            "exit_reason": st.column_config.TextColumn(
                "exit_reason",
                help="Why the trade was closed (e.g., take-profit, stop-loss, trailing stop, manual).",
            ),
        },
    )
else:
    st.info("No trades to display.")

# ── Data-Driven Commentary ────────────────────────────────────────────────────
if not df.empty:
    st.markdown("---")
    st.subheader("Analysis & Recommendations")

    total_pnl = df["pnl_usd"].sum()
    overall_wr = (df["pnl_usd"] > 0).mean()
    avg_r = df["r_multiple"].mean() if "r_multiple" in df.columns else 0
    n_strategies = df["strategy"].nunique()

    points = []

    # Portfolio-level assessment
    if total_pnl > 0:
        points.append(f"The portfolio is net profitable at **${total_pnl:,.2f}**. However, verify this isn't driven by a single strategy — diversified returns are more resilient.")
    else:
        points.append(f"The portfolio is net negative at **${total_pnl:,.2f}**. Identify the worst-performing strategy on the Strategy Deep Dive page and consider pausing it.")

    if overall_wr < 0.35:
        points.append(f"Overall win rate of **{overall_wr:.1%}** is low. This is acceptable only if your average winner is 2-3x your average loser (check Avg R). If Avg R is also low, the system has no edge.")
    elif overall_wr > 0.55:
        points.append(f"Overall win rate of **{overall_wr:.1%}** is strong. Ensure this isn't inflated by many small wins masking occasional large losses — check the drawdown page.")

    if avg_r < 0:
        points.append(f"Average R-multiple of **{avg_r:.2f}** is negative — losers are larger than winners on average. Tighten stop-losses or widen take-profit targets.")
    elif avg_r > 0.5:
        points.append(f"Average R-multiple of **{avg_r:.2f}** is healthy — your winners meaningfully outpace your losers.")

    # Per-strategy winners and losers
    strat_pnl = df.groupby("strategy")["pnl_usd"].sum().sort_values()
    worst = strat_pnl.index[0]
    best = strat_pnl.index[-1]
    if strat_pnl.iloc[0] < 0:
        points.append(f"**{worst}** is the weakest strategy (${strat_pnl.iloc[0]:,.2f}). Review its entry criteria on the Strategy Deep Dive page.")
    if strat_pnl.iloc[-1] > 0:
        points.append(f"**{best}** is the top performer (${strat_pnl.iloc[-1]:,.2f}). Consider whether it warrants a larger capital allocation.")

    # Recent performance trend
    recent_20 = df.head(20)
    if len(recent_20) >= 10:
        recent_wr = (recent_20["pnl_usd"] > 0).mean()
        if recent_wr < overall_wr - 0.1:
            points.append(f"Recent 20-trade win rate ({recent_wr:.0%}) is notably below the overall average ({overall_wr:.0%}) — performance may be deteriorating.")

    for p in points:
        st.markdown(f"- {p}")
