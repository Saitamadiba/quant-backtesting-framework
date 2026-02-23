"""Page 5: Asia/London/NY session performance breakdown."""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Session Analysis", page_icon="🕐", layout="wide")
st.title("🕐 Session Analysis")

from data.data_loader import get_all_trades
from components.filters import strategy_filter, source_filter, apply_filters
from components.charts import session_strategy_heatmap

# ── Sidebar ───────────────────────────────────────────────────────────────────
src = source_filter(key_prefix="sa")
df_all = get_all_trades(source_filter=src)
strategies = strategy_filter(df_all, key_prefix="sa")
df = apply_filters(df_all, strategies=strategies)

# ── Session KPI Cards ─────────────────────────────────────────────────────────
st.subheader("Session Overview")
st.caption("Performance breakdown by trading session (Asian, London, New York). Each session has different liquidity and volatility characteristics. Identify which sessions your strategies thrive in and avoid trading sessions where your edge disappears.")
if not df.empty:
    session_cols = st.columns(3)
    for i, session in enumerate(["Asian", "London", "New York"]):
        s = df[df["session"] == session]
        total = len(s)
        wr = (s["pnl_usd"] > 0).mean() if total else 0
        avg_r = s["r_multiple"].mean() if total else 0
        total_r = s["r_multiple"].sum() if total else 0
        with session_cols[i]:
            st.metric(f"{session}", f"{total} trades",
                      help=f"Total trades taken during the {session} session. Compare volume across sessions to see where your bots are most active.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Win Rate", f"{wr:.1%}",
                      help="Percentage of trades that closed in profit during this session. Below 40% warrants investigation.")
            c2.metric("Avg R", f"{avg_r:.2f}",
                      help="Average R-multiple per trade in this session. Positive means winners outpace losers on average.")
            c3.metric("Total R", f"{total_r:.1f}",
                      help="Sum of all R-multiples. This is the session's total contribution to your portfolio in risk units.")
else:
    st.info("No trade data available.")

st.markdown("---")

# ── Grouped Bar: Per-session by strategy ──────────────────────────────────────
st.subheader("Per-Session Performance by Strategy")
st.caption("See how each strategy performs within each session. A strategy may be profitable overall but lose money in a specific session — this chart exposes that mismatch.")
if not df.empty:
    session_stats = df.groupby(["session", "strategy"]).agg(
        trades=("pnl_usd", "count"),
        win_rate=("pnl_usd", lambda x: (x > 0).mean()),
        total_r=("r_multiple", "sum"),
    ).reset_index()

    fig = px.bar(
        session_stats, x="session", y="total_r", color="strategy",
        barmode="group", title="Total R by Session & Strategy",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Strategy: %{data.name}<br>Total R: %{y:.2f}<extra></extra>"
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, key="sa_session_bar")

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.subheader("Win Rate Heatmap: Strategy x Session")
st.caption("A color-coded matrix of win rates across strategies and sessions. Dark green cells are high-conviction setups; red cells are danger zones where you should consider disabling that strategy during that session.")
st.plotly_chart(session_strategy_heatmap(df), key="sa_heatmap")

# ── Hour-of-Day Analysis ─────────────────────────────────────────────────────
st.subheader("Hour-of-Day Analysis")
st.caption("Granular breakdown by hour of entry (UTC). Helps you fine-tune your bot's active trading windows. If certain hours consistently show low win rates, consider adding time-of-day filters to your bots.")
if not df.empty and "entry_time" in df.columns:
    tmp = df.copy()
    tmp["hour"] = tmp["entry_time"].dt.hour
    hourly = tmp.groupby("hour").agg(
        trades=("pnl_usd", "count"),
        win_rate=("pnl_usd", lambda x: (x > 0).mean()),
        avg_r=("r_multiple", "mean"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(hourly, x="hour", y="trades", title="Trade Distribution by Hour",
                     color_discrete_sequence=["#2196F3"])
        fig.update_traces(
            hovertemplate="Hour: %{x}:00 UTC<br>Trades: %{y}<extra></extra>"
        )
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, key="sa_hour_dist")
    with c2:
        fig = px.line(hourly, x="hour", y="win_rate", title="Win Rate by Hour",
                      color_discrete_sequence=["#4CAF50"])
        fig.update_traces(
            hovertemplate="Hour: %{x}:00 UTC<br>Win Rate: %{y:.1%}<extra></extra>"
        )
        fig.update_layout(template="plotly_dark", height=350, yaxis_tickformat=".0%")
        st.plotly_chart(fig, key="sa_hour_wr")

# ── Data-Driven Commentary ────────────────────────────────────────────────────
if not df.empty:
    st.markdown("---")
    st.subheader("Analysis & Recommendations")

    points = []

    # Identify best and worst sessions
    session_perf = {}
    for session in ["Asian", "London", "New York"]:
        s = df[df["session"] == session]
        if len(s) >= 3:
            session_perf[session] = {
                "trades": len(s),
                "wr": (s["pnl_usd"] > 0).mean(),
                "total_r": s["r_multiple"].sum(),
                "avg_r": s["r_multiple"].mean(),
                "pnl": s["pnl_usd"].sum(),
            }

    if session_perf:
        best = max(session_perf, key=lambda k: session_perf[k]["total_r"])
        worst = min(session_perf, key=lambda k: session_perf[k]["total_r"])
        bp = session_perf[best]
        wp = session_perf[worst]

        points.append(f"**{best}** is your strongest session with **{bp['total_r']:.1f}R** total ({bp['wr']:.0%} win rate across {bp['trades']} trades). Consider concentrating capital during this window.")

        if wp["total_r"] < 0:
            points.append(f"**{worst}** is actively losing money (**{wp['total_r']:.1f}R**, ${wp['pnl']:.2f}). Disabling your bots during this session would immediately improve portfolio performance.")
        elif wp["total_r"] < bp["total_r"] * 0.3:
            points.append(f"**{worst}** contributes minimally (**{wp['total_r']:.1f}R** vs {best}'s {bp['total_r']:.1f}R). The risk exposure may not be justified.")

    # Strategy-session mismatches
    if not df.empty:
        for strat in df["strategy"].unique():
            strat_df = df[df["strategy"] == strat]
            for session in ["Asian", "London", "New York"]:
                ss = strat_df[strat_df["session"] == session]
                if len(ss) >= 5:
                    wr = (ss["pnl_usd"] > 0).mean()
                    total_r = ss["r_multiple"].sum()
                    if total_r < -2:
                        points.append(f"**{strat}** is bleeding during **{session}** ({total_r:.1f}R, {wr:.0%} WR). Consider adding a session filter to this bot's config to skip {session} trades.")

    # Hour-level insights
    if "entry_time" in df.columns:
        tmp = df.copy()
        tmp["hour"] = tmp["entry_time"].dt.hour
        hr_stats = tmp.groupby("hour").agg(
            trades=("pnl_usd", "count"),
            avg_r=("r_multiple", "mean"),
        )
        if len(hr_stats) >= 5:
            worst_hours = hr_stats[hr_stats["avg_r"] < -0.3]
            if not worst_hours.empty and len(worst_hours) <= 4:
                hrs = ", ".join(f"{h}:00" for h in worst_hours.index)
                points.append(f"Hours with consistently negative Avg R: **{hrs}**. Adding a time-of-day blackout for these hours could reduce losses.")

    for p in points:
        st.markdown(f"- {p}")
