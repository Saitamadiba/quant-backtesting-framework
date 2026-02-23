"""Page 10: Strategy Logic Explainer — plain-English breakdown of each strategy."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Strategy Explainer", page_icon="📖", layout="wide")
st.title("📖 Strategy Logic Explainer")
st.caption(
    "A plain-English reference guide for every strategy in the portfolio. "
    "Each section explains the core logic, entry/exit conditions, risk management rules, "
    "and key indicators used — so you can understand exactly what your bots are doing and why."
)

from config import STRATEGIES
from data.data_loader import get_all_trades

# Load live data for performance context
df_all = get_all_trades()

# ── Strategy selector ─────────────────────────────────────────────────────────
strategy = st.selectbox(
    "Select Strategy",
    list(STRATEGIES.keys()),
    help="Choose a strategy to see its full logic breakdown.",
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Mini performance summary for context
# ══════════════════════════════════════════════════════════════════════════════
def _show_live_summary(strat_name: str):
    """Show a compact live performance summary for the strategy."""
    sdf = df_all[df_all["strategy"] == strat_name]
    if sdf.empty:
        st.info(f"No live trade data for {strat_name} yet.")
        return

    total = len(sdf)
    wins = (sdf["pnl_usd"] > 0).sum()
    wr = wins / total if total else 0
    total_pnl = sdf["pnl_usd"].sum()
    avg_r = sdf["r_multiple"].mean() if "r_multiple" in sdf.columns else 0
    total_r = sdf["r_multiple"].sum() if "r_multiple" in sdf.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades", total)
    c2.metric("Win Rate", f"{wr:.1%}")
    c3.metric("Total P&L", f"${total_pnl:,.2f}")
    c4.metric("Avg R", f"{avg_r:.2f}")
    c5.metric("Total R", f"{total_r:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Entry logic flow diagram
# ══════════════════════════════════════════════════════════════════════════════
def _flow_diagram(steps: list[dict]):
    """Render a numbered step-by-step flow."""
    for i, step in enumerate(steps, 1):
        icon = step.get("icon", "")
        st.markdown(
            f"**Step {i}{(' ' + icon) if icon else ''}:** {step['label']}  \n"
            f"> {step['detail']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Visualize entry/exit on a synthetic price chart
# ══════════════════════════════════════════════════════════════════════════════
def _entry_exit_diagram(
    title: str,
    prices: list[float],
    entry_idx: int,
    sl: float,
    tp: float,
    direction: str = "Long",
    annotations: list[dict] | None = None,
):
    """Draw a simple line chart showing entry, SL, and TP levels."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=prices, mode="lines", name="Price",
        line=dict(color="white", width=2),
    ))

    # Entry marker
    fig.add_trace(go.Scatter(
        x=[entry_idx], y=[prices[entry_idx]],
        mode="markers+text", name="Entry",
        marker=dict(size=12, color="#2196F3", symbol="triangle-up" if direction == "Long" else "triangle-down"),
        text=["Entry"], textposition="top center",
    ))

    # SL line
    fig.add_hline(y=sl, line_dash="dash", line_color="#F44336",
                  annotation_text="Stop Loss", annotation_position="bottom right")
    # TP line
    fig.add_hline(y=tp, line_dash="dash", line_color="#4CAF50",
                  annotation_text="Take Profit", annotation_position="top right")

    if annotations:
        for ann in annotations:
            fig.add_annotation(**ann)

    fig.update_layout(
        title=title, template="plotly_dark", height=350,
        showlegend=False, xaxis_title="Bars", yaxis_title="Price",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FVG (Fair Value Gap) Strategy
# ══════════════════════════════════════════════════════════════════════════════
if strategy == "FVG":
    st.header("FVG — Fair Value Gap Strategy")
    st.caption("An ICT (Inner Circle Trader) concept: identifies imbalances in price delivery where institutional orders leave 'gaps' in the candle structure.")

    # Live summary
    st.subheader("Live Performance Snapshot")
    _show_live_summary("FVG")
    st.markdown("---")

    # ── Core Concept ──────────────────────────────────────────────────
    st.subheader("Core Concept")
    st.markdown("""
A **Fair Value Gap (FVG)** is a 3-candle pattern where the high of candle 1 does not overlap with
the low of candle 3, leaving a price gap in the middle candle. This gap represents an inefficiency
in price delivery — institutional order flow moved price so aggressively that it left behind
unfilled orders.

**Bullish FVG**: High of candle 1 < Low of candle 3 (gap above)
**Bearish FVG**: Low of candle 1 > High of candle 3 (gap below)

The bot detects these gaps across **multiple timeframes** (15m, 1H, 4H, Daily) and looks for
price to retrace back into the gap zone for an entry.
""")

    # Visual: FVG pattern
    bullish_fvg = [100, 101, 99, 97, 96, 100, 103, 105, 103, 101, 98, 100, 102, 104, 106]
    fig = _entry_exit_diagram(
        "Bullish FVG Entry Example",
        bullish_fvg, entry_idx=10, sl=95.5, tp=107, direction="Long",
        annotations=[
            dict(x=3, y=96, text="FVG Zone", showarrow=True, arrowhead=2, font=dict(color="#FF9800")),
            dict(x=10, y=98, text="Retracement into gap", showarrow=True, arrowhead=2),
        ],
    )
    st.plotly_chart(fig, key="fvg_diagram")

    # ── Entry Logic ──────────────────────────────────────────────────
    st.subheader("Entry Logic (Step by Step)")
    _flow_diagram([
        {"label": "Scan for FVG patterns", "icon": "🔍",
         "detail": "The bot scans 15m, 1H, 4H, and Daily candles for 3-candle FVG patterns. "
                   "Minimum gap size: 0.01% (5m) to 0.08% (Daily)."},
        {"label": "Multi-timeframe confluence check", "icon": "📊",
         "detail": "FVGs aligned across timeframes score higher. Daily FVGs get 3x weight, "
                   "4H gets 75/100, 1H gets 50/100, 15m gets 25/100."},
        {"label": "Volume confirmation", "icon": "📈",
         "detail": "The sweep candle must have volume > 1.2x the 20-period average — "
                   "confirming institutional participation, not random noise."},
        {"label": "Composite score filter", "icon": "✅",
         "detail": "Only accept signals with composite score >= 3.0 or standalone strength >= 4.0. "
                   "This filters out weak, low-probability setups."},
        {"label": "Enter on gap retracement", "icon": "🎯",
         "detail": "When price retraces back into the FVG zone, enter in the direction of the gap "
                   "(long for bullish FVG, short for bearish FVG)."},
    ])

    # ── Exit Logic ───────────────────────────────────────────────────
    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Details |
|-----------|---------|
| **Take Profit** | 1:2 Risk-to-Reward ratio (TP = Entry +/- 2x Risk) |
| **Stop Loss** | Placed at the opposite side of the FVG zone (minimum 0.02% gap) |
| **Position Sizing** | Adaptive: targets 15% annualized volatility, half-Kelly criterion, 0.25x-2.0x scaling |
""")

    # ── Risk Management ──────────────────────────────────────────────
    st.subheader("Risk Management Rules")
    rm_cols = st.columns(4)
    rm_cols[0].metric("Max Daily Trades", "5", help="Prevents overtrading during choppy sessions.")
    rm_cols[1].metric("Max Drawdown Cap", "8%", help="Bot halts if account drawdown hits 8%.")
    rm_cols[2].metric("Daily Loss Limit", "4%", help="Bot stops taking new trades after 4% daily loss.")
    rm_cols[3].metric("Max SL Distance", "1.5%", help="Prop firm safety rule — no stop further than 1.5% from entry.")

    # ── Key Indicators ───────────────────────────────────────────────
    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR (14-period)**: Used for stop-loss sizing and take-profit calculation
- **Volume (20-period rolling avg)**: Confirms institutional participation in FVG formation
- **Multi-Timeframe Priority**: Daily (100) > 4H (75) > 1H (50) > 15m (25) > 5m (10)
- **Confluence Score**: Weighted composite score (max 10.0) combining all timeframe FVGs
""")

    # ── Strengths & Weaknesses ───────────────────────────────────────
    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Multi-timeframe analysis reduces false signals
- Targets institutional order flow inefficiencies
- Works across all sessions (not limited to kill zones)
- Adaptive position sizing via Kelly criterion
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Complex scoring system can miss obvious setups
- FVGs fill quickly in trending markets (reduced opportunity)
- Requires sufficient volatility to create gap patterns
- No trailing stop — full 1:2 R:R or nothing
""")


# ══════════════════════════════════════════════════════════════════════════════
# Liquidity Raid Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "Liquidity Raid":
    st.header("Liquidity Raid Strategy")
    st.caption("Exploits institutional liquidity sweeps: smart money raids stop-loss clusters at session highs/lows, then reverses. The bot waits for the sweep, confirms the reversal, and enters in the opposite direction.")

    st.subheader("Live Performance Snapshot")
    _show_live_summary("Liquidity Raid")
    st.markdown("---")

    # ── Core Concept ──────────────────────────────────────────────────
    st.subheader("Core Concept")
    st.markdown("""
**Liquidity raids** are a market microstructure phenomenon where institutional players push price
beyond a known level (session high/low, swing point) to trigger clustered stop-loss orders. This
"liquidity grab" gives them the fill they need, after which price reverses sharply.

The strategy uses a **state machine** to track the lifecycle of each liquidity level:
1. **WAITING**: Session high/low identified, watching for a sweep
2. **SWEEP_DETECTED**: Price breaks beyond the level (stops triggered)
3. **TRADED**: Entry taken after price reclaims the swept level

**Key insight**: The strategy only trades during **kill zones** (London 3-5am ET, NY 8-10:30am ET)
when institutional activity is highest.
""")

    # Visual: Liquidity sweep + reversal
    sweep_prices = [100, 101, 102, 103, 102.5, 103, 104, 104.5, 103, 101.5, 99, 97, 96, 98, 100, 102, 104, 106, 108]
    fig = _entry_exit_diagram(
        "Bearish Liquidity Sweep -> Long Entry",
        sweep_prices, entry_idx=14, sl=95, tp=109, direction="Long",
        annotations=[
            dict(x=7, y=104.5, text="Session High", showarrow=True, arrowhead=2, font=dict(color="#FF9800")),
            dict(x=12, y=96, text="Liquidity Sweep (stops hit)", showarrow=True, arrowhead=2, font=dict(color="#F44336")),
            dict(x=14, y=100, text="Reclaim + Entry", showarrow=True, arrowhead=2, font=dict(color="#2196F3")),
        ],
    )
    st.plotly_chart(fig, key="lr_diagram")

    # ── Entry Logic ──────────────────────────────────────────────────
    st.subheader("Entry Logic (Step by Step)")
    _flow_diagram([
        {"label": "Identify session levels", "icon": "📍",
         "detail": "Track Asian (19:00-00:00 ET), London (03:00-08:00 ET), and NY (08:00-16:00 ET) "
                   "session highs and lows as liquidity targets."},
        {"label": "Wait for kill zone", "icon": "⏰",
         "detail": "Only look for entries during London (03:00-05:00 ET) or NY (08:00-10:30 ET) kill zones — "
                   "this is when institutional flow is densest."},
        {"label": "Detect the sweep", "icon": "💥",
         "detail": "Price breaks beyond a session high/low. The state machine transitions from "
                   "WAITING -> SWEEP_DETECTED."},
        {"label": "Wait for price reclaim", "icon": "↩️",
         "detail": "After the sweep, price must close back inside the swept level — proving the "
                   "sweep was a liquidity grab, not a genuine breakout."},
        {"label": "Confirm with daily bias", "icon": "📊",
         "detail": "EMA stack (50 > 100 > 200 for longs, inverse for shorts) on the daily chart "
                   "must align. Multi-timeframe confluence score: Daily 40% + 4H 30%."},
        {"label": "5-minute precision entry (optional)", "icon": "🔬",
         "detail": "On 15m signal, optionally wait up to 6 5M candles for a tighter entry — "
                   "reducing stop-loss distance and improving R:R."},
    ])

    # ── Exit Logic ───────────────────────────────────────────────────
    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Details |
|-----------|---------|
| **Take Profit** | Dynamic 1.5x-2.5x R:R based on volatility (high vol = tighter, low vol = wider) |
| **Stop Loss** | ATR-based (2.5x ATR) or sweep-based (1.0 ATR buffer beyond sweep level) |
| **Trailing Stop** | Stepped: only moves on new high-water marks, 0.5x ATR threshold |
| **Breakeven Stop** | Moves to breakeven once 1R profit is achieved |
| **Time Exit** | Auto-close after 6 hours without SL/TP hit |
| **Volatility Adaptive** | High-vol widens SL by 25%, low-vol tightens by 20% |
""")

    # ── Risk Management ──────────────────────────────────────────────
    st.subheader("Risk Management Rules")
    rm_cols = st.columns(4)
    rm_cols[0].metric("Risk Per Trade", "1%", help="Fixed 1% of account equity risked per trade.")
    rm_cols[1].metric("Max Daily Loss", "4%", help="Bot pauses after 4% account loss in a day.")
    rm_cols[2].metric("Max Drawdown", "8%", help="Hard stop — bot disables if DD exceeds 8%.")
    rm_cols[3].metric("Re-entry Attempts", "2", help="Up to 2 re-entries after stop-out, 4-candle cooldown.")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR (14-period)**: Dynamic position sizing, stop-loss, and trailing stop calculation
- **EMA Stack (50/100/200)**: Daily directional bias — Long: 50>100>200, Short: 50<100<200
- **Swing Points**: 5-candle lookback for high/low identification
- **Volatility Percentile**: 20-candle rolling percentile for adaptive SL adjustment
- **Price Structure**: Higher Highs/Higher Lows (bullish) vs Lower Highs/Lower Lows (bearish)
""")

    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Targets institutional order flow (high edge potential)
- State machine prevents re-trading exhausted levels
- Dynamic R:R adapts to market volatility
- Trailing stop + breakeven lock in profits progressively
- 5M precision entry improves risk/reward
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Kill zone restriction limits trading opportunities
- Session-level detection can be late if Asian session is thin
- Re-entry system risks doubling down on failed setups
- Requires accurate session time calibration
""")


# ══════════════════════════════════════════════════════════════════════════════
# Momentum Mastery Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "Momentum Mastery":
    st.header("Momentum Mastery Strategy")
    st.caption("Combines daily EMA directional bias with kill-zone liquidity sweeps and strict confirmation candle quality filters. Designed for high-conviction, low-frequency entries.")

    st.subheader("Live Performance Snapshot")
    _show_live_summary("Momentum Mastery")
    st.markdown("---")

    # ── Core Concept ──────────────────────────────────────────────────
    st.subheader("Core Concept")
    st.markdown("""
Momentum Mastery is a **trend-following + sweep** hybrid. It first establishes the prevailing
trend direction using a daily EMA stack, then waits for a counter-trend liquidity sweep during
a kill zone, and finally enters only when a **high-quality confirmation candle** proves the
sweep has reversed.

The strategy is deliberately **conservative**: it uses fractal invalidation, candle body quality
filters, and volume percentile thresholds to avoid false signals. This means fewer trades but
higher win rates per entry.
""")

    # Visual: EMA bias + sweep + confirmation
    mm_prices = [100, 101, 102, 103, 104, 103.5, 104, 105, 104, 103, 101, 99.5, 98, 97, 99, 101, 103, 105, 107, 109]
    fig = _entry_exit_diagram(
        "Momentum Mastery: EMA Bias + Sweep + Confirmation Entry",
        mm_prices, entry_idx=15, sl=96.5, tp=111, direction="Long",
        annotations=[
            dict(x=7, y=105, text="Daily EMA Bullish Bias", showarrow=False, font=dict(color="#4CAF50", size=12)),
            dict(x=13, y=97, text="Liquidity Sweep (session low)", showarrow=True, arrowhead=2, font=dict(color="#F44336")),
            dict(x=15, y=101, text="Bullish Confirmation Candle", showarrow=True, arrowhead=2, font=dict(color="#2196F3")),
        ],
    )
    st.plotly_chart(fig, key="mm_diagram")

    st.subheader("Entry Logic (Step by Step)")
    _flow_diagram([
        {"label": "Establish daily directional bias", "icon": "📈",
         "detail": "Check Daily EMA stack: LONG requires 50 EMA > 100 EMA > 200 EMA. "
                   "SHORT requires 50 < 100 < 200. Neutral stack = no trades."},
        {"label": "Wait for kill zone window", "icon": "⏰",
         "detail": "London kill zone (03:00-05:00 ET) or NY kill zone (08:00-10:30 ET). "
                   "Trades outside these windows are ignored."},
        {"label": "Detect liquidity sweep", "icon": "💥",
         "detail": "For longs: price sweeps session lows (stops triggered below). "
                   "For shorts: price sweeps session highs."},
        {"label": "Confirmation candle quality check", "icon": "✅",
         "detail": "The confirmation candle must pass 3 tests: (1) correct direction (bullish for longs), "
                   "(2) body >= 50% of total range, (3) volume above 60th percentile of recent 20 candles."},
        {"label": "Fractal invalidation check", "icon": "🛡️",
         "detail": "If 2+ Williams Fractals appear after the sweep, OR an opposite-direction fractal "
                   "is detected, the setup is invalidated. This prevents trading stale or ambiguous structures."},
    ])

    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Details |
|-----------|---------|
| **Take Profit** | Fixed 1:2 R:R ratio (TP = Entry +/- 2x Risk distance) |
| **Stop Loss** | At the sweep level with minimum 1.0x ATR distance |
| **Fractal Invalidation** | Trade canceled if fractal structure invalidates after entry |
| **No trailing stop** | Strict 1:2 R:R — either hit TP or SL |
""")

    st.subheader("Risk Management Rules")
    rm_cols = st.columns(3)
    rm_cols[0].metric("Risk Per Trade", "1% (fixed)", help="No dynamic scaling — flat 1% per trade.")
    rm_cols[1].metric("Max Daily Trades", "5", help="Prevents overtrading even during active sessions.")
    rm_cols[2].metric("Level Exhaustion", "Active", help="Tracks and skips levels that produced losses today.")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **EMA Stack (50/100/200)**: Daily chart directional bias — the primary trend filter
- **ATR (14-period)**: Minimum stop-loss distance enforcement (1.0x ATR)
- **Williams Fractals**: 5-candle high/low fractals for structure invalidation
- **Candle Body Ratio**: Body/range >= 50% ensures strong directional conviction
- **Volume Percentile**: 60th percentile threshold filters weak confirmation candles
""")

    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Multiple confirmation layers reduce false signals
- Fractal invalidation catches deteriorating setups early
- Fixed 1:2 R:R keeps risk/reward simple and predictable
- Volume + body quality filters ensure institutional participation
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Very selective — low trade frequency can mean missed opportunities
- Kill zone restriction limits entry windows
- No trailing stop means giving back unrealized profit on reversals
- Fixed position sizing doesn't capitalize on high-conviction setups
""")


# ══════════════════════════════════════════════════════════════════════════════
# SBS (Swing Break System / Smart Block Structure) Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "SBS":
    st.header("SBS — Swing Break System / Smart Block Structure")
    st.caption("Also known as Smart Block Structure. The most complex strategy in the portfolio: combines ICT liquidity sweeps with Break of Structure (BOS) confirmation, Fibonacci retracements for precision entries, and a multi-TP trailing stop system for graduated profit-taking.")

    st.subheader("Live Performance Snapshot")
    _show_live_summary("SBS")
    st.markdown("---")

    # ── Core Concept ──────────────────────────────────────────────────
    st.subheader("Core Concept")
    st.markdown("""
**SBS** stands for **Swing Break System** (describing what it does — trading swing breaks)
and **Smart Block Structure** (describing what it targets — smart money order blocks).

The strategy identifies institutional order blocks through a 4-stage process:

1. **Liquidity Sweep**: Price breaks beyond a recent high/low, triggering clustered stops
2. **Break of Structure (BOS)**: After the sweep, price reverses and breaks a key structure level
   — confirming that smart money has shifted direction
3. **Fibonacci Retracement**: The bot calculates Fibonacci levels (0.0, 0.236, 0.5, 0.618, 1.0)
   between the sweep level and the post-BOS swing point
4. **Precision Entry**: Price retraces to the 0.618 Fibonacci level ("golden pocket") and shows
   a rejection — this is the entry point

The strategy uses a **multi-take-profit system** that scales out at each Fibonacci level,
progressively trailing the stop loss to lock in gains.
""")

    # Visual: SBS full lifecycle
    sbs_prices = [100, 101, 102, 103, 104, 105, 104, 103, 101, 99, 97, 96,
                  98, 100, 103, 106, 108, 105, 103, 101, 100, 102, 104, 107, 109, 111, 113]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sbs_prices, mode="lines", name="Price",
                             line=dict(color="white", width=2)))

    # Annotations for each stage
    fig.add_annotation(x=11, y=96, text="1. Liquidity Sweep", showarrow=True, arrowhead=2,
                       font=dict(color="#F44336", size=11))
    fig.add_annotation(x=15, y=106, text="2. Break of Structure", showarrow=True, arrowhead=2,
                       font=dict(color="#FF9800", size=11))
    fig.add_annotation(x=20, y=100, text="3. 0.618 Fib Retest", showarrow=True, arrowhead=2,
                       font=dict(color="#9C27B0", size=11))
    fig.add_trace(go.Scatter(x=[20], y=[100], mode="markers", name="Entry",
                             marker=dict(size=12, color="#2196F3", symbol="triangle-up")))

    # Fib levels
    fib_levels = {"1.0 (Sweep)": 96, "0.618": 100, "0.5": 102, "0.236": 105.5, "0.0 (Swing)": 108}
    for label, level in fib_levels.items():
        fig.add_hline(y=level, line_dash="dot", line_color="rgba(156,39,176,0.3)",
                      annotation_text=label, annotation_position="left")

    # TP markers
    fig.add_annotation(x=23, y=105.5, text="TP1 (0.236)", showarrow=True, font=dict(color="#4CAF50"))
    fig.add_annotation(x=25, y=108, text="TP2 (0.0)", showarrow=True, font=dict(color="#4CAF50"))

    fig.update_layout(title="SBS Full Trade Lifecycle: Sweep -> BOS -> Fib Entry -> Multi-TP",
                      template="plotly_dark", height=400, showlegend=False,
                      xaxis_title="Bars", yaxis_title="Price")
    st.plotly_chart(fig, key="sbs_diagram")

    st.subheader("Entry Logic (Step by Step)")
    _flow_diagram([
        {"label": "Detect liquidity sweep", "icon": "💥",
         "detail": "Price breaks 0.1% beyond recent 50-candle range high (bearish) or low (bullish), "
                   "triggering clustered stop orders."},
        {"label": "Confirm Break of Structure (BOS)", "icon": "🔄",
         "detail": "After the sweep, price must close beyond the recent opposite-side structure level. "
                   "For longs: close > recent high after a low sweep. This confirms smart money reversal."},
        {"label": "Calculate Fibonacci levels", "icon": "📐",
         "detail": "Establish swing range: 1.0 = sweep level, 0.0 = post-BOS swing. "
                   "Key levels: 0.618 (golden pocket entry), 0.5 (midpoint), 0.236 (TP1)."},
        {"label": "Wait for 0.618 retracement", "icon": "🎯",
         "detail": "Price must retrace to the 0.618 Fibonacci level. A second liquidity grab beyond "
                   "this level is the ideal trigger. Optional 15m rejection candle for precision."},
        {"label": "15m confirmation or 1H fallback", "icon": "🔬",
         "detail": "Best entry: 15m candle wicks beyond 0.618 level and closes back (rejection pattern). "
                   "Fallback: if no 15m confirmation within 4 hours, enter at next 1H open."},
    ])

    st.subheader("Exit Conditions — Multi-TP Trailing System")
    st.markdown("""
| Level | Action | Trailing SL Moves To |
|-------|--------|---------------------|
| **TP1** (0.236 Fib) | Partial profit taken | SL trails to 0.5 level |
| **TP2** (0.0 / Swing) | More profit taken | SL trails to 0.236 level |
| **TP3** (Recent S/R) | Remaining position closed | SL trails to 0.0 level |
| **Stop Loss** | Full exit | At 1.0 Fib (sweep level) |
| **Re-entry** | Up to 2 attempts | 1-10 candle window after stop |
""")

    st.subheader("Risk Management Rules")
    rm_cols = st.columns(4)
    rm_cols[0].metric("Position Sizing", "ATR-based", help="Dynamic sizing using ATR + Kelly criterion + confidence weighting.")
    rm_cols[1].metric("Setup Age Limit", "50 candles", help="Setups older than 50 candles are discarded as stale.")
    rm_cols[2].metric("Re-entry Cooldown", "1-10 bars", help="Minimum wait between re-entry attempts.")
    rm_cols[3].metric("Confidence Threshold", "0.5+", help="Minimum signal confidence score required for entry.")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR (14-period)**: Risk/reward calculations and dynamic position sizing
- **Fibonacci Retracements (0.0, 0.236, 0.5, 0.618, 1.0)**: Core entry/exit methodology
- **Swing Point Detection**: Identifies high/low structures for Fibonacci range
- **Break of Structure**: 0.1% threshold confirms market structure shift
- **Manipulation Score**: Filters suspicious price action (avoids manipulated moves)
- **Recent S/R Levels**: 50-candle lookback for TP3 placement
""")

    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Multi-TP system maximizes profit extraction from winning trades
- Fibonacci entry provides precise risk/reward definition
- 15m confirmation reduces false entries significantly
- Trailing SL progressively eliminates downside risk
- Confidence-weighted sizing allocates more capital to better setups
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Most complex strategy — more parameters = more potential for overfitting
- Multi-TP exits mean partial positions can be left in adverse moves
- Requires patient waiting for Fibonacci retrace (misses V-reversals)
- 1H timeframe means slower signal generation than 15m strategies
- Setup age limit (50 candles) can be too generous in fast markets
""")


# ══════════════════════════════════════════════════════════════════════════════
# Comparative Summary
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Strategy Comparison Matrix")
st.caption("Side-by-side comparison of all strategies to understand their different approaches, risk profiles, and market conditions where they excel.")

comparison_data = {
    "Aspect": [
        "Primary Entry Signal",
        "Execution Timeframe",
        "Exit Method",
        "Risk Per Trade",
        "Stop Loss Method",
        "Session Dependency",
        "Trailing Stop",
        "Re-entry After Stop",
        "ML Filter",
        "Complexity",
    ],
    "FVG": [
        "3-candle gap pattern",
        "15m + HTF confluence",
        "Fixed 1:2 R:R",
        "1% + vol scaling",
        "FVG boundary",
        "All sessions",
        "No",
        "No",
        "Optional",
        "High",
    ],
    "Liquidity Raid": [
        "Session sweep + reversal",
        "15m / 5m precision",
        "Dynamic 1.5-2.5x R:R",
        "1% fixed",
        "ATR or sweep-based",
        "Kill zones only",
        "Yes (stepped)",
        "2 attempts",
        "Yes",
        "High",
    ],
    "Momentum Mastery": [
        "EMA bias + sweep + confirm",
        "15m signals",
        "Fixed 1:2 R:R",
        "1% fixed",
        "Sweep + min 1.0 ATR",
        "Kill zones only",
        "No",
        "No",
        "Not yet",
        "Medium",
    ],
    "SBS": [
        "Sweep + BOS + Fib entry",
        "1H + 15m confirmation",
        "Multi-TP trailing SL",
        "ATR + confidence",
        "Fibonacci 1.0 level",
        "All sessions",
        "Yes (Fib-based)",
        "2 attempts",
        "Manipulation score",
        "Very High",
    ],
}

st.dataframe(
    pd.DataFrame(comparison_data).set_index("Aspect"),
    use_container_width=True,
    column_config={
        "FVG": st.column_config.TextColumn("FVG", help="Fair Value Gap strategy approach for this aspect. FVG targets institutional price imbalances using multi-timeframe 3-candle gap patterns."),
        "Liquidity Raid": st.column_config.TextColumn("Liquidity Raid", help="Liquidity Raid strategy approach for this aspect. Exploits institutional stop-hunts at session highs/lows during kill zones, then enters on the reversal."),
        "Momentum Mastery": st.column_config.TextColumn("Momentum Mastery", help="Momentum Mastery strategy approach for this aspect. Combines daily EMA trend bias with kill-zone liquidity sweeps and strict confirmation candle filters."),
        "SBS": st.column_config.TextColumn("SBS", help="Swing Break System / Smart Block Structure strategy approach for this aspect. The most complex strategy: uses sweep + BOS + Fibonacci retracement entries with a multi-TP trailing stop system."),
    },
)

# ── Live Performance Comparison ───────────────────────────────────────────────
if not df_all.empty:
    st.subheader("Live Performance Comparison")
    st.caption("How each strategy is actually performing with real money. Compare against each strategy's design goals above.")

    rows = []
    for strat in STRATEGIES:
        sdf = df_all[df_all["strategy"] == strat]
        if sdf.empty:
            continue
        total = len(sdf)
        wins = (sdf["pnl_usd"] > 0).sum()
        wr = wins / total if total else 0
        pnl = sdf["pnl_usd"].sum()
        avg_r = sdf["r_multiple"].mean() if "r_multiple" in sdf.columns else 0
        gp = sdf.loc[sdf["pnl_usd"] > 0, "pnl_usd"].sum()
        gl = sdf.loc[sdf["pnl_usd"] < 0, "pnl_usd"].abs().sum()
        pf = gp / gl if gl > 0 else float("inf")

        rows.append({
            "Strategy": strat,
            "Trades": total,
            "Win Rate": f"{wr:.1%}",
            "Total P&L": f"${pnl:,.2f}",
            "Avg R": f"{avg_r:.2f}",
            "Profit Factor": f"{pf:.2f}" if pf != float("inf") else "---",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config={
            "Strategy": st.column_config.TextColumn("Strategy", help="Name of the trading strategy. See detailed breakdowns above for each strategy's full logic."),
            "Trades": st.column_config.TextColumn("Trades", help="Total number of live trades executed by this strategy. More trades means higher statistical confidence in the results."),
            "Win Rate": st.column_config.TextColumn("Win Rate", help="Percentage of trades that closed with a positive P&L. Compare this to each strategy's design expectations above."),
            "Total P&L": st.column_config.TextColumn("Total P&L", help="Cumulative dollar profit or loss from all live trades. The bottom-line measure of strategy performance."),
            "Avg R": st.column_config.TextColumn("Avg R", help="Average R-multiple per trade. Positive means the average trade earns more than it risks. Above 0.3R is solid."),
            "Profit Factor": st.column_config.TextColumn("Profit Factor", help="Gross profit divided by gross loss. Above 1.0 is profitable; above 1.5 is strong; above 2.0 is excellent. '---' means no losing trades yet."),
        })
