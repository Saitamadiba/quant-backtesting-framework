"""Trading Bot Performance Dashboard — Entry point."""

import streamlit as st

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from datetime import datetime
from config import REFRESH_OPTIONS, DB_STRATEGY_MAP, VPS_CACHE_DIR
from data.vps_sync import sync_all_vps_data, get_cached_db_status

# ── Auto-Refresh ──────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    refresh_label = st.sidebar.selectbox("Auto-Refresh", list(REFRESH_OPTIONS.keys()), index=0)
    interval = REFRESH_OPTIONS[refresh_label]
    if interval > 0:
        st_autorefresh(interval=interval, key="auto_refresh")
except ImportError:
    st.sidebar.info("Install streamlit-autorefresh for auto-refresh support.")

# ── VPS Sync Controls ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("VPS Data Sync")

if st.sidebar.button("Sync from VPS", type="primary"):
    with st.sidebar.status("Syncing trade DBs + ML training data...", expanded=True) as status:
        results = sync_all_vps_data()
        ok = sum(1 for r in results.values() if r.get("status") == "ok")
        fail = sum(1 for r in results.values() if r.get("status") != "ok")
        status.update(label=f"Sync done: {ok} ok, {fail} failed", state="complete")
        st.session_state["last_sync"] = datetime.now().strftime("%H:%M:%S")
        st.session_state["sync_results"] = results

# Show last sync time
last_sync = st.session_state.get("last_sync", "Never")
st.sidebar.caption(f"Last sync: {last_sync}")

# Per-bot status dots
db_status = get_cached_db_status()
status_line = ""
for db_file, (strategy, symbol) in DB_STRATEGY_MAP.items():
    info = db_status.get(db_file, {})
    dot = "🟢" if info.get("exists") else "🔴"
    status_line += f"{dot} {strategy} {symbol}  "
st.sidebar.caption(status_line)

# ML training data status
from config import VPS_ML_FILES
ml_ok = sum(1 for f in VPS_ML_FILES if db_status.get(f, {}).get("exists"))
ml_total = len(VPS_ML_FILES)
st.sidebar.caption(f"ML training data: {ml_ok}/{ml_total} synced")

# ── Landing Page ──────────────────────────────────────────────────────────────
st.title("📊 Trading Bot Performance Dashboard")
st.markdown(
    "Monitor live bots (FVG, Liquidity Raid, Momentum Mastery) and backtest results (SBS) "
    "across BTC, ETH, and NQ from a single interface."
)
st.markdown("**Navigate** using the sidebar pages.")

# Quick stats
from data.data_loader import get_all_trades

df = get_all_trades()
if not df.empty:
    st.caption("A snapshot of your entire trading operation — live bots and backtests combined. Use this to quickly confirm all data sources are feeding in and nothing is stale.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", f"{len(df):,}",
              help="Total number of trades across all strategies, symbols, and data sources (live + backtest).")
    c2.metric("Live Trades", f"{(df['source'] == 'Live').sum():,}",
              help="Trades executed by your live VPS bots. These are real-money results synced from the VPS databases.")
    c3.metric("Backtest Trades", f"{(df['source'] == 'Backtest').sum():,}",
              help="Trades from historical backtests (SBS, DuckDB). Useful for validating strategies before going live.")
    c4.metric("Strategies", f"{df['strategy'].nunique()}",
              help="Number of distinct strategies with trade data. You should see FVG, Liquidity Raid, Momentum Mastery, and SBS if all sources are loaded.")
else:
    st.warning(
        "No trade data loaded. Click **Sync from VPS** to pull live data, "
        "or ensure local backtest files exist."
    )
