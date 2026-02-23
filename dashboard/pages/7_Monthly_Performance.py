"""Page 6: Calendar heatmaps & monthly breakdowns."""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Monthly Performance", page_icon="📅", layout="wide")
st.title("📅 Monthly Performance")

from data.data_loader import get_all_trades
from components.filters import strategy_filter, source_filter, apply_filters
from components.charts import monthly_calendar_heatmap, monthly_pnl_bar

# ── Sidebar ───────────────────────────────────────────────────────────────────
src = source_filter(key_prefix="mp")
df_all = get_all_trades(source_filter=src)
strategies = strategy_filter(df_all, key_prefix="mp")
df = apply_filters(df_all, strategies=strategies)

# ── Calendar Heatmap ──────────────────────────────────────────────────────────
st.subheader("Calendar Heatmap (Total R)")
st.caption("Each cell represents one month, colored by total R-multiple earned. Green months are profitable, red months lost money. Quickly identify seasonal patterns — some strategies may underperform during low-volatility summer months or over-perform during high-volatility periods.")
st.plotly_chart(monthly_calendar_heatmap(df), key="mp_cal_heatmap")

# ── Monthly P&L Bar ───────────────────────────────────────────────────────────
st.subheader("Monthly P&L")
st.caption("Dollar profit or loss per month. Consistent green bars indicate a reliable income stream. Alternating red and green suggests high variance — consider position sizing adjustments.")
st.plotly_chart(monthly_pnl_bar(df), key="mp_pnl_bar")

# ── Monthly Metrics Table ─────────────────────────────────────────────────────
st.subheader("Monthly Breakdown")
st.caption("Detailed monthly stats table. Pay attention to Profit Factor (above 1.0 is profitable) and Win Rate trends. A declining Win Rate over recent months may indicate regime change.")
if not df.empty and "entry_time" in df.columns:
    tmp = df.copy()
    tmp["month"] = tmp["entry_time"].dt.to_period("M").astype(str)

    monthly = tmp.groupby("month").agg(
        Trades=("pnl_usd", "count"),
        Win_Rate=("pnl_usd", lambda x: (x > 0).mean()),
        Total_PnL=("pnl_usd", "sum"),
        Total_R=("r_multiple", "sum"),
        Avg_R=("r_multiple", "mean"),
    ).reset_index()

    # Profit Factor per month
    def _pf(g):
        gp = g[g > 0].sum()
        gl = g[g < 0].abs().sum()
        return gp / gl if gl > 0 else float("inf") if gp > 0 else 0

    monthly["Profit_Factor"] = tmp.groupby("month")["pnl_usd"].apply(_pf).values

    display = monthly.copy()
    display["Win_Rate"] = display["Win_Rate"].apply(lambda x: f"{x:.1%}")
    display["Total_PnL"] = display["Total_PnL"].apply(lambda x: f"${x:,.2f}")
    display["Total_R"] = display["Total_R"].apply(lambda x: f"{x:.1f}")
    display["Avg_R"] = display["Avg_R"].apply(lambda x: f"{x:.2f}")
    display["Profit_Factor"] = display["Profit_Factor"].apply(
        lambda x: f"{x:.2f}" if x != float("inf") else "∞"
    )
    display.columns = ["Month", "Trades", "Win Rate", "Total P&L", "Total R", "Avg R", "Profit Factor"]
    st.dataframe(display, use_container_width=True, hide_index=True, column_config={
        "Month": st.column_config.TextColumn("Month", help="Calendar month (YYYY-MM). Each row summarises all trades closed within that month."),
        "Trades": st.column_config.TextColumn("Trades", help="Total number of closed trades in this month. More trades means higher statistical confidence."),
        "Win Rate": st.column_config.TextColumn("Win Rate", help="Percentage of trades that closed with a positive P&L. Above 50% is generally good; above 60% is excellent."),
        "Total P&L": st.column_config.TextColumn("Total P&L", help="Sum of dollar profit and loss for all trades in this month."),
        "Total R": st.column_config.TextColumn("Total R", help="Sum of R-multiples earned. Each R represents one unit of risk, so +5R means you earned 5x your risked amount."),
        "Avg R": st.column_config.TextColumn("Avg R", help="Average R-multiple per trade. Positive means the average trade is profitable after accounting for risk."),
        "Profit Factor": st.column_config.TextColumn("Profit Factor", help="Gross profit divided by gross loss. Above 1.0 is profitable; above 1.5 is strong; above 2.0 is excellent."),
    })

    # Win rate trend line
    st.caption("Win rate trend over time. A steadily declining trend is a red flag — your strategy's edge may be fading and you should consider re-optimizing or pausing the bot.")
    import plotly.graph_objects as go
    fig = go.Figure()
    wr = monthly["Win_Rate"] if "Win_Rate" in monthly.columns else tmp.groupby("month")["pnl_usd"].apply(lambda x: (x > 0).mean())
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=tmp.groupby("month")["pnl_usd"].apply(lambda x: (x > 0).mean()).values,
        mode="lines+markers", line=dict(color="#4CAF50"),
        hovertemplate="Month: %{x}<br>Win Rate: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(title="Monthly Win Rate Trend", template="plotly_dark", height=300,
                      yaxis_tickformat=".0%")
    st.plotly_chart(fig, key="mp_wr_trend")

    # ── Data-Driven Commentary ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Analysis & Recommendations")

    points = []
    n_months = len(monthly)
    points.append(f"Data spans **{n_months} months** of trading activity.")

    # Best and worst months
    if n_months >= 2:
        best_month_idx = monthly["Total_R"].idxmax()
        worst_month_idx = monthly["Total_R"].idxmin()
        best_m = monthly.loc[best_month_idx]
        worst_m = monthly.loc[worst_month_idx]
        points.append(
            f"**Best month**: {best_m['month']} ({best_m['Total_R']:.1f}R, "
            f"${best_m['Total_PnL']:,.2f}, {best_m['Win_Rate']:.0%} win rate). "
            f"**Worst month**: {worst_m['month']} ({worst_m['Total_R']:.1f}R, "
            f"${worst_m['Total_PnL']:,.2f})."
        )

    # Consecutive losing months
    losing_months = (monthly["Total_R"] < 0).values
    max_losing_streak = 0
    current_streak = 0
    for lm in losing_months:
        if lm:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0
    if max_losing_streak >= 2:
        points.append(
            f"Longest streak of losing months: **{max_losing_streak}**. "
            f"Multiple consecutive red months indicate a potential regime shift "
            f"or parameter decay requiring re-optimization."
        )
    elif n_months >= 3:
        points.append("No streak of consecutive losing months — consistent performance.")

    # Profit Factor trend
    if n_months >= 3:
        pf_values = monthly["Profit_Factor"].replace([float("inf")], pd.NA).dropna()
        if len(pf_values) >= 3:
            recent_pf = pf_values.iloc[-2:].mean()
            earlier_pf = pf_values.iloc[:-2].mean()
            if recent_pf < earlier_pf * 0.7 and earlier_pf > 0:
                points.append(
                    f"Recent Profit Factor ({recent_pf:.2f}) has declined significantly "
                    f"from earlier average ({earlier_pf:.2f}). The strategy's edge may be "
                    f"weakening — investigate whether market conditions have changed."
                )

    # Win rate trend
    if n_months >= 4:
        wr_values = monthly["Win_Rate"].values
        if len(wr_values) >= 4:
            recent_wr = wr_values[-2:].mean()
            earlier_wr = wr_values[:-2].mean()
            if recent_wr < earlier_wr - 0.10:
                points.append(
                    f"Win rate has dropped from **{earlier_wr:.0%}** (earlier) to "
                    f"**{recent_wr:.0%}** (recent). A 10%+ decline is an early warning "
                    f"signal — run a Monte Carlo simulation on Page 8 with updated data."
                )
            elif recent_wr > earlier_wr + 0.05:
                points.append(
                    f"Win rate has improved from **{earlier_wr:.0%}** to **{recent_wr:.0%}** "
                    f"in recent months — the strategy may be entering a favorable regime."
                )

    # Average trades per month
    avg_trades = monthly["Trades"].mean()
    points.append(
        f"Average **{avg_trades:.0f} trades/month**. "
        + ("This is a healthy sample for monthly analysis."
           if avg_trades >= 10
           else "Low trade frequency means monthly stats have high variance — "
                "consider longer evaluation windows.")
    )

    # Consistency: what % of months are profitable
    profitable_months = (monthly["Total_R"] > 0).sum()
    pct_profitable = profitable_months / n_months if n_months > 0 else 0
    points.append(
        f"**{profitable_months}/{n_months} months ({pct_profitable:.0%})** are profitable. "
        + ("Excellent consistency."
           if pct_profitable >= 0.75
           else "Below 50% profitable months indicates high variance — "
                "review position sizing and strategy filters."
           if pct_profitable < 0.5
           else "Reasonable consistency, with room for improvement.")
    )

    for p in points:
        st.markdown(f"- {p}")

else:
    st.info("No trade data available.")
