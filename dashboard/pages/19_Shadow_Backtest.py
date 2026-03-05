"""Shadow Backtest — live-vs-adapter signal alignment for all strategies."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config import STRATEGIES, STRATEGY_COLORS, BOT_SERVICES
from data.wfo_loader import list_shadow_results, load_shadow_result, get_latest_shadow

st.set_page_config(page_title="Shadow Backtest", layout="wide")
st.title("Shadow Backtest Analysis")
st.caption("Compare live bot signals against WFO adapter reconstructions")

# ── Sidebar ───────────────────────────────────────────────────────────────────
shadow_results = list_shadow_results()

strategies_available = sorted({r["strategy"] for r in shadow_results}) if shadow_results else []
all_strategies = sorted({s for s in STRATEGIES})

sel_strategy = st.sidebar.selectbox(
    "Strategy",
    strategies_available if strategies_available else all_strategies,
)

symbols_available = sorted(
    {r["symbol"] for r in shadow_results if r["strategy"] == sel_strategy}
) if shadow_results else []
all_symbols = STRATEGIES.get(sel_strategy, {}).get("symbols", ["BTC", "ETH"])

sel_symbol = st.sidebar.selectbox(
    "Symbol",
    symbols_available if symbols_available else all_symbols,
)

# ── All Bots Grid ─────────────────────────────────────────────────────────────
st.subheader("Running Bots Overview")

bot_cols = st.columns(min(len(BOT_SERVICES), 5))
for i, (svc_name, svc_info) in enumerate(BOT_SERVICES.items()):
    col = bot_cols[i % len(bot_cols)]
    strat = svc_info["strategy"]
    sym = svc_info["symbol"]
    shadow = get_latest_shadow(strat, sym)
    color = STRATEGY_COLORS.get(strat, "#888")

    with col:
        st.markdown(
            f"<div style='border:2px solid {color}; border-radius:8px; "
            f"padding:12px; margin-bottom:8px; text-align:center;'>"
            f"<b>{strat}</b><br>{sym}<br>"
            + (f"<span style='color:#4CAF50'>Match: {shadow['match_rate']:.1%}</span>"
               if shadow else "<span style='color:#888'>No shadow data</span>")
            + "</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Load Selected Shadow Result ───────────────────────────────────────────────
data = get_latest_shadow(sel_strategy, sel_symbol)

if not data:
    st.warning(
        f"No shadow backtest results for {sel_strategy} / {sel_symbol}. "
        "Run a shadow backtest first."
    )

    # Offer to run shadow backtest
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backtrader_framework.optimization.shadow_backtest import ShadowBacktest
        if st.button(f"Run Shadow Backtest for {sel_strategy} / {sel_symbol}"):
            with st.spinner("Running shadow backtest..."):
                result = ShadowBacktest().run(sel_strategy, sel_symbol, "15m")
                st.success(f"Complete! Match rate: {result.get('match_rate', 0):.1%}")
                st.rerun()
    except ImportError:
        pass

    st.stop()

# ── Summary Metrics ───────────────────────────────────────────────────────────
st.subheader(f"Shadow Results: {sel_strategy} / {sel_symbol}")

m1, m2, m3, m4, m5 = st.columns(5)
match_rate = data.get("match_rate", 0)
m1.metric("Match Rate", f"{match_rate:.1%}",
          delta="Good" if match_rate > 0.2 else "Low",
          delta_color="normal" if match_rate > 0.2 else "inverse")
m2.metric("Live Trades", data.get("n_live_trades", 0))
m3.metric("Shadow Trades", data.get("n_shadow_trades", 0))
m4.metric("Matched", data.get("n_matched", 0))
m5.metric("Phantom", data.get("n_phantom", 0))

# ── Signal Timeline ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Signal Timeline")

matched = data.get("matched_trades", [])
missed = data.get("missed_trades", [])
phantom = data.get("phantom_trades", [])

fig_timeline = go.Figure()

# Live matched trades
if matched:
    live_times = []
    live_dirs = []
    for m in matched:
        lt = m.get("live_time") or m.get("live", {}).get("entry_time", "")
        d = m.get("live_direction") or m.get("live", {}).get("direction", "LONG")
        if lt:
            live_times.append(lt)
            live_dirs.append(1 if "LONG" in str(d).upper() else -1)
    if live_times:
        fig_timeline.add_trace(go.Scatter(
            x=live_times, y=live_dirs,
            mode="markers", name="Matched (Live)",
            marker=dict(color="#4CAF50", size=10, symbol="circle"),
        ))

    shadow_times = []
    shadow_dirs = []
    for m in matched:
        st_val = m.get("shadow_time") or m.get("shadow", {}).get("entry_time", "")
        d = m.get("shadow_direction") or m.get("shadow", {}).get("direction", "LONG")
        if st_val:
            shadow_times.append(st_val)
            shadow_dirs.append(1 if "LONG" in str(d).upper() else -1)
    if shadow_times:
        fig_timeline.add_trace(go.Scatter(
            x=shadow_times, y=shadow_dirs,
            mode="markers", name="Matched (Shadow)",
            marker=dict(color="#4CAF50", size=10, symbol="diamond"),
        ))

# Missed trades (live only)
if missed:
    miss_times = []
    miss_dirs = []
    for m in missed:
        t = m.get("entry_time") or m.get("time", "")
        d = m.get("direction", "LONG")
        if t:
            miss_times.append(t)
            miss_dirs.append(1 if "LONG" in str(d).upper() else -1)
    if miss_times:
        fig_timeline.add_trace(go.Scatter(
            x=miss_times, y=miss_dirs,
            mode="markers", name="Missed (Live only)",
            marker=dict(color="#F44336", size=8, symbol="circle-open"),
        ))

# Phantom trades (shadow only)
if phantom:
    ph_times = []
    ph_dirs = []
    for p in phantom:
        t = p.get("entry_time") or p.get("time", "")
        d = p.get("direction", "LONG")
        if t:
            ph_times.append(t)
            ph_dirs.append(1 if "LONG" in str(d).upper() else -1)
    if ph_times:
        fig_timeline.add_trace(go.Scatter(
            x=ph_times, y=ph_dirs,
            mode="markers", name="Phantom (Shadow only)",
            marker=dict(color="#FF9800", size=8, symbol="diamond-open"),
        ))

fig_timeline.update_layout(
    template="plotly_dark", height=400,
    xaxis_title="Time", yaxis_title="Direction",
    yaxis=dict(tickvals=[-1, 1], ticktext=["SHORT", "LONG"]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_timeline, use_container_width=True)

# ── Match Breakdown ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Match Breakdown")

bc1, bc2 = st.columns(2)

# By direction
with bc1:
    st.markdown("**By Direction**")
    dir_counts = {"LONG": {"Matched": 0, "Missed": 0, "Phantom": 0},
                  "SHORT": {"Matched": 0, "Missed": 0, "Phantom": 0}}
    for m in matched:
        d = str(m.get("live_direction") or m.get("live", {}).get("direction", "LONG")).upper()
        key = "LONG" if "LONG" in d else "SHORT"
        dir_counts[key]["Matched"] += 1
    for m in missed:
        d = str(m.get("direction", "LONG")).upper()
        key = "LONG" if "LONG" in d else "SHORT"
        dir_counts[key]["Missed"] += 1
    for p in phantom:
        d = str(p.get("direction", "LONG")).upper()
        key = "LONG" if "LONG" in d else "SHORT"
        dir_counts[key]["Phantom"] += 1

    dir_rows = []
    for direction, counts in dir_counts.items():
        for status, count in counts.items():
            dir_rows.append({"Direction": direction, "Status": status, "Count": count})
    df_dir = pd.DataFrame(dir_rows)

    fig_dir = go.Figure()
    status_colors = {"Matched": "#4CAF50", "Missed": "#F44336", "Phantom": "#FF9800"}
    for status in ["Matched", "Missed", "Phantom"]:
        subset = df_dir[df_dir["Status"] == status]
        fig_dir.add_trace(go.Bar(
            x=subset["Direction"], y=subset["Count"],
            name=status, marker_color=status_colors[status],
        ))
    fig_dir.update_layout(template="plotly_dark", height=300, barmode="group")
    st.plotly_chart(fig_dir, use_container_width=True)

# By time of day (rough session proxy)
with bc2:
    st.markdown("**By Time of Day (ET)**")
    tod_counts = {"Asian (19-03)": 0, "London (03-08)": 0, "New York (08-16)": 0, "Off-hours": 0}

    def _classify_hour(time_str):
        try:
            h = pd.Timestamp(time_str).hour
            if 19 <= h or h < 3:
                return "Asian (19-03)"
            elif 3 <= h < 8:
                return "London (03-08)"
            elif 8 <= h < 16:
                return "New York (08-16)"
            return "Off-hours"
        except Exception:
            return "Off-hours"

    all_trades = []
    for m in matched:
        t = m.get("live_time") or m.get("live", {}).get("entry_time", "")
        if t:
            all_trades.append(("Matched", t))
    for m in missed:
        t = m.get("entry_time") or m.get("time", "")
        if t:
            all_trades.append(("Missed", t))

    session_status = {}
    for status, t in all_trades:
        sess = _classify_hour(t)
        session_status.setdefault(sess, {"Matched": 0, "Missed": 0})
        session_status[sess][status] += 1

    if session_status:
        sess_rows = []
        for sess, counts in session_status.items():
            for s, c in counts.items():
                sess_rows.append({"Session": sess, "Status": s, "Count": c})
        df_sess = pd.DataFrame(sess_rows)
        fig_sess = go.Figure()
        for s in ["Matched", "Missed"]:
            sub = df_sess[df_sess["Status"] == s]
            fig_sess.add_trace(go.Bar(
                x=sub["Session"], y=sub["Count"],
                name=s, marker_color=status_colors[s],
            ))
        fig_sess.update_layout(template="plotly_dark", height=300, barmode="group")
        st.plotly_chart(fig_sess, use_container_width=True)
    else:
        st.info("No time data available for session breakdown.")

# ── Aggregate Stats ───────────────────────────────────────────────────────────
agg = data.get("aggregate", {})
if agg:
    st.markdown("---")
    with st.expander("Aggregate Statistics", expanded=False):
        agg_rows = [{"Metric": k, "Value": v} for k, v in agg.items()]
        st.dataframe(pd.DataFrame(agg_rows), use_container_width=True)

# ── Re-run Button ─────────────────────────────────────────────────────────────
st.markdown("---")
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from backtrader_framework.optimization.shadow_backtest import ShadowBacktest
    if st.button(f"Re-run Shadow Backtest for {sel_strategy} / {sel_symbol}"):
        with st.spinner("Running shadow backtest..."):
            result = ShadowBacktest().run(sel_strategy, sel_symbol, "15m")
            st.success(f"Complete! Match rate: {result.get('match_rate', 0):.1%}")
            st.rerun()
except ImportError:
    st.caption("Shadow backtest module not available for re-run.")
