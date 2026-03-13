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
    c1.metric("Trades", total, help="Total closed trades for this strategy across all live bots.")
    c2.metric("Win Rate", f"{wr:.1%}", help="Percentage of trades that closed in profit. Evaluate alongside Avg R — a 30% win rate with 3:1 R:R is highly profitable.")
    c3.metric("Total P&L", f"${total_pnl:,.2f}", help="Cumulative realized profit/loss in USD across all trades.")
    c4.metric("Avg R", f"{avg_r:.2f}", help="Average R-multiple per trade. Positive means winners outpace losers on a risk-adjusted basis. > 0.2R is a strong edge.")
    c5.metric("Total R", f"{total_r:.1f}", help="Sum of all R-multiples. The total risk-adjusted return — 10R means you earned 10x your risk unit.")


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
# Helper: Pipeline diagram
# ══════════════════════════════════════════════════════════════════════════════
def _pipeline_diagram(steps: list[str], final: str = "EXECUTE"):
    """Render a horizontal signal pipeline with arrows."""
    cols = []
    for i, step in enumerate(steps):
        cols.append(step)
        if i < len(steps) - 1:
            cols.append("->")
    cols.append("->")
    cols.append(f"**{final}**")
    st.markdown(" ".join(cols))


# ══════════════════════════════════════════════════════════════════════════════
# FVG (Fair Value Gap) Strategy
# ══════════════════════════════════════════════════════════════════════════════
if strategy == "FVG":
    st.header("FVG — Fair Value Gap Strategy")
    st.caption("An ICT (Inner Circle Trader) concept: identifies imbalances in price delivery where institutional orders leave 'gaps' in the candle structure. Enhanced with HMM regime gating, WFO signal scoring, and midpoint entry precision.")

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
price to retrace back into the gap zone for an entry at the **midpoint** of the gap.
""")

    # Visual: FVG pattern
    bullish_fvg = [100, 101, 99, 97, 96, 100, 103, 105, 103, 101, 98, 100, 102, 104, 106]
    fig = _entry_exit_diagram(
        "Bullish FVG Midpoint Entry Example",
        bullish_fvg, entry_idx=10, sl=95.5, tp=107, direction="Long",
        annotations=[
            dict(x=3, y=96, text="FVG Zone", showarrow=True, arrowhead=2, font=dict(color="#FF9800")),
            dict(x=10, y=98, text="Midpoint entry", showarrow=True, arrowhead=2),
        ],
    )
    st.plotly_chart(fig, key="fvg_diagram")

    # ── Recent Improvements ──────────────────────────────────────────
    st.subheader("Recent Improvements")
    with st.expander("HMM Hard Gate (Crypto Only)", expanded=True):
        st.markdown("""
**What**: A Hidden Markov Model classifies the market into **volatile** (momentum bullish/bearish)
or **calm** (ranging/choppy/unknown) states. The bot **skips signal generation entirely** during
calm regimes for BTC and ETH.

**Why**: The profitability study showed materially better expectancy in one HMM state — a 63% improvement in expectancy. By gating out calm periods, the bot
only trades when its edge is strongest.

**Scope**: Applies to crypto (BTC, ETH) only. NQ is not gated because it trades during defined
equity sessions with naturally higher institutional activity.

| Regime | Action |
|--------|--------|
| `momentum_bullish` | Trade normally |
| `momentum_bearish` | Trade normally |
| `ranging` | Skip signal generation |
| `choppy` | Skip signal generation |
| `unknown` | Skip signal generation |
""")

    with st.expander("WFO Signal Scoring (9 Components)"):
        st.markdown("""
**Walk-Forward Optimized** signal scoring grades every FVG setup across 9 weighted components
before execution. Weights were calibrated on historical profitability data and adapt online
via an ML Weight Adapter.

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| `gap_size` | 0.18 | FVG gap size relative to ATR |
| `volume` | 0.13 | Sweep candle volume vs 20-bar average |
| `ema_align` | 0.13 | EMA stack alignment with trade direction |
| `rsi_align` | 0.09 | RSI alignment (not overbought for longs, not oversold for shorts) |
| `struct_bias` | 0.09 | Price structure (HH/HL or LH/LL) alignment |
| `displacement` | 0.13 | Strength of the move that created the FVG |
| `sweep` | 0.10 | Whether a liquidity sweep preceded the gap |
| `hmm_align` | 0.10 | HMM regime alignment with trade direction |
| `rv_align` | 0.05 | Realized volatility percentile alignment |

**Minimum confidence**: 0.45 for standard FVGs, 0.40 for inverted FVGs (iFVGs).

The **ML Weight Adapter** retrains every 50 trade outcomes using ridge logistic regression,
clamping weight multipliers to [0.5x, 2.0x] of the WFO baseline. Requires 100+ trades and
CV accuracy > 0.52 before first adaptation.
""")

    with st.expander("Midpoint Entry + Study-Matched SL"):
        st.markdown("""
**Midpoint Entry**: All retest entries target the **exact midpoint** of the FVG zone,
not the zone edges. This was validated by the profitability study as the optimal
entry location that balances fill probability against risk distance.

**Study-Matched Stop Loss**: Stop loss is placed at the opposite side of the FVG zone
with an ATR buffer, calibrated from the study's optimal SL distance findings.

```
Entry = (FVG High + FVG Low) / 2
SL = FVG zone boundary + ATR buffer (opposite side)
TP = ATR-regime-scaled R:R ratio
```
""")

    with st.expander("Session Filter Bypass (Crypto 24/7)"):
        st.markdown("""
The session filter (London/NY hours restriction) is enforced **only for NQ futures**.
BTC and ETH trade **24/7** because the profitability study showed no statistically
significant edge improvement from session filtering on crypto pairs — institutional
flow in crypto is distributed across all sessions, unlike equity futures.

| Asset | Session Rule |
|-------|-------------|
| BTC | 24/7 (no session filter) |
| ETH | 24/7 (no session filter) |
| NQ | London + NY sessions only |
""")

    with st.expander("iFVG Detection (Inverted Fair Value Gaps)"):
        st.markdown("""
**Phase 2** of the FVG strategy adds detection and trading of **inverted FVGs** — gaps
that form when a previous FVG zone is fully invalidated and price creates a new gap
in the opposite direction.

iFVGs have their own WFO scorer with higher HMM and RV alignment weights
(`hmm_align: 0.15`, `rv_align: 0.15`) and a lower minimum confidence threshold (0.40)
because the base win rate for iFVG patterns is 51.8%.
""")

    # ── Entry Logic ──────────────────────────────────────────────────
    st.subheader("Entry Logic (Signal Pipeline)")
    st.markdown("""
`FVG Detected` -> `HMM Gate` -> `Regime Filter` -> `Session Filter (NQ)` -> `FVG Invalidation Check`
-> `Proximity Gate` -> `Rejection Candle` -> `WFO Score (>= 0.45)` -> `Risk Gate` -> **MIDPOINT ENTRY**
""")

    _flow_diagram([
        {"label": "HMM regime gate (crypto only)", "icon": "🧠",
         "detail": "Bot checks HMM state. If regime is ranging/choppy/unknown, the entire signal "
                   "generation cycle is skipped. Only volatile (momentum) states proceed."},
        {"label": "Scan for FVG patterns across timeframes", "icon": "🔍",
         "detail": "Scans 15m, 1H, 4H, and Daily candles for 3-candle FVG patterns. "
                   "Minimum gap size: 0.01% (5m) to 0.08% (Daily). Multi-TF confluence weighted."},
        {"label": "FVG invalidation + proximity check", "icon": "📊",
         "detail": "Tracked FVGs are checked for invalidation (price closed through zone). "
                   "Price must be within configured proximity of the gap midpoint."},
        {"label": "WFO confidence scoring", "icon": "📐",
         "detail": "9-component weighted score must exceed 0.45 threshold. Components: gap_size, "
                   "volume, ema_align, rsi_align, struct_bias, displacement, sweep, hmm_align, rv_align."},
        {"label": "Midpoint entry execution", "icon": "🎯",
         "detail": "Entry at FVG midpoint = (gap_high + gap_low) / 2. Stop loss at opposite zone boundary "
                   "with ATR buffer. Take profit at ATR-regime-scaled R:R ratio."},
    ])

    # ── Exit Logic ───────────────────────────────────────────────────
    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Details |
|-----------|---------|
| **Take Profit** | ATR-regime-scaled R:R (momentum: 1.5x target, choppy: 0.8x target) |
| **Stop Loss** | FVG zone boundary + ATR buffer (study-matched distance) |
| **Partial TP** | Partial exit at TP1 reduces position, remainder trails |
| **Trailing Stop** | Activated after partial TP hit |
| **Stale Exit** | Auto-close after 50 candles without SL/TP hit |
| **Position Sizing** | Volatility-adjusted: targets 15% annualized vol, half-Kelly, 0.25x-2.0x |
""")

    # ── Risk Management ──────────────────────────────────────────────
    st.subheader("Risk Management Rules")
    rm_cols = st.columns(4)
    rm_cols[0].metric("Max Daily Trades", "5", help="Prevents overtrading during choppy sessions.")
    rm_cols[1].metric("Prop Firm DD Cap", "8%", help="Bot halts if account drawdown hits 8%.")
    rm_cols[2].metric("Daily Loss Limit", "4%", help="Bot stops taking new trades after 4% daily loss.")
    rm_cols[3].metric("Max SL Distance", "1.5%", help="Prop firm safety rule — no stop further than 1.5% from entry.")

    st.markdown("""
**Additional safeguards:**
- **Traded zone expiry**: Prevents re-trading the same FVG zone within a configurable window (persisted across restarts)
- **Momentum validator**: Auto-disables momentum entries if edge disappears over recent trades
- **Adaptive risk manager**: Self-adjusts confidence thresholds, TP allocation, and R:R targets based on performance
- **Premature entry analyzer**: Monitors stopped trades for 24h — if price reached TP after SL, tightens entry rules
""")

    # ── Key Indicators ───────────────────────────────────────────────
    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR (14-period)**: Stop-loss sizing, take-profit calculation, volatility regime detection
- **HMM Regime (4-state)**: Momentum Bullish, Momentum Bearish, Ranging, Choppy — gates signal generation
- **WFO Signal Score (9 components)**: Weighted composite confidence (0.0-1.0) with ML-adaptive weights
- **EMA Stack (Fast/Slow)**: Trend direction alignment for WFO `ema_align` component
- **Volume (20-period rolling avg)**: Confirms institutional participation in FVG formation
- **Multi-Timeframe Priority**: Daily (100) > 4H (75) > 1H (50) > 15m (25) > 5m (10)
- **RSI (14-period)**: Overbought/oversold alignment for WFO `rsi_align` component
- **Realized Volatility Percentile**: ATR rank over 50 bars for WFO `rv_align` component
""")

    # ── Strengths & Weaknesses ───────────────────────────────────────
    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- HMM gate eliminates 63% lower-expectancy calm-regime trades (crypto)
- WFO scoring with ML-adaptive weights continuously improves signal quality
- Midpoint entry provides precise, study-validated entry location
- Multi-timeframe analysis reduces false signals
- Adaptive risk manager self-tunes from trade outcomes
- iFVG detection captures inverted gap opportunities
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- HMM gate can miss early-trend entries before regime transitions
- 9-component WFO scoring adds latency to signal evaluation
- Midpoint entry may not fill if price reverses at zone boundary
- Complex scoring system has more parameters to calibrate
- iFVG signals have lower base win rate (51.8%) than standard FVGs
""")


# ══════════════════════════════════════════════════════════════════════════════
# Liquidity Raid Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "Liquidity Raid":
    st.header("Liquidity Raid Strategy")
    st.caption("Exploits institutional liquidity sweeps: smart money raids stop-loss clusters at session highs/lows, then reverses. Enhanced with gamma regime integration, WFO counter-trend scoring, continuous sweep quality grading, and multi-timeframe entry confirmation.")

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
1. **MONITORING**: Session high/low identified, watching for a sweep
2. **DETECTED**: Price breaks beyond the level (stops triggered)
3. **CONFIRMED**: Reversal confirmed with displacement and FVG
4. **EXPIRED**: No confirmation within timeout — move on

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

    # ── Recent Improvements ──────────────────────────────────────────
    st.subheader("Recent Improvements")
    with st.expander("Gamma Regime Integration (Options Microstructure)", expanded=True):
        st.markdown("""
**What**: The bot queries the **options gamma regime** module for real-time IV (implied volatility),
DVOL, and gamma flip data. This is used for:

1. **IV-Adaptive Sweep Depth**: Different minimum sweep depth thresholds per IV regime
2. **Gamma R:R Adjustment**: Take-profit levels scaled by gamma regime after standard dynamic R:R

| IV Regime | Min Sweep Depth | R:R Adjustment |
|-----------|----------------|----------------|
| LOW (IV < 40) | Baseline | Standard R:R |
| MEDIUM (IV 40-60) | Higher threshold | 0.75x R:R (tighter) |
| HIGH (IV > 60) | Highest threshold | Wider R:R (more room) |

**Fallback**: If the gamma module is unavailable, defaults to standard thresholds.
""")

    with st.expander("WFO Counter-Trend Scoring (4 Components)"):
        st.markdown("""
The Liquidity Raid uses a **counter-trend** WFO scorer — unlike the FVG's trend-alignment approach,
this scorer grades sweeps by their mean-reversion quality.

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| `sweep_depth_atr` | 0.50 | How deep the sweep penetrated vs ATR (dominant signal) |
| `counter_struct` | 0.20 | Counter-trend price structure quality |
| `counter_htf` | 0.15 | Higher timeframe alignment against the sweep direction |
| `struct_conf` | 0.15 | Structural confirmation of reversal |

**Minimum confidence**: 0.25 (lower threshold because sweep quality scoring provides additional filtering).

**DVOL Regime Gate**: When DVOL is in the medium range (45-65), the R:R target is scaled by 0.75x
to account for lower trending potential in moderate volatility.
""")

    with st.expander("Continuous Sweep Quality Scoring"):
        st.markdown("""
Replaces binary sweep detection with a **0-100 continuous score** across four dimensions:

1. **Depth** (0.5-1.0): How far price penetrated the session level, normalized by ATR
2. **Volume** (0.3-1.0): Sweep candle volume vs 20-bar average
3. **Time Decay** (0.2-1.0): Freshness penalty — 1.0 at 0-1 candles, degrades to 0.2 at 12+ candles
4. **Confirmation** (0.2-1.0): Body ratio and direction match quality

| Grade | Score Range | Action |
|-------|-------------|--------|
| A+ (Excellent) | 80-100 | Trade with full size |
| A (Very Good) | 65-80 | Trade with standard size |
| B (Good) | 50-65 | Trade with reduced size |
| C (Fair) | 35-50 | Consider skipping |
| D (Poor) | 0-35 | Skip |
""")

    with st.expander("Multi-Timeframe Entry Confirmation"):
        st.markdown("""
**MTF Analysis** scores Daily and 4H alignment with the trade direction:

- **Daily aligned**: EMA stack on Daily chart agrees with trade direction
- **4H aligned**: 4H price structure supports the reversal thesis

Two modes:
- **Soft** (default): Informs confidence and logs alignment, but doesn't reject signals
- **Hard**: Rejects counter-trend signals below a score threshold

MTF score, quality label, and alignment flags are included in every signal for ML tracking.
""")

    with st.expander("5M Pending Execution System"):
        st.markdown("""
After a 15M sweep is detected, a **pending entry** waits for 5M confirmation before triggering.

**5M Confirmation requires:**
- Correct directional close (bullish for longs, bearish for shorts)
- Candle range > minimum ATR threshold
- Entry placed at a pullback from the 5M candle range

**Max wait**: Limited to a configurable number of 5M candles. If no confirmation arrives,
the pending sweep expires.
""")

    # ── Entry Logic ──────────────────────────────────────────────────
    st.subheader("Entry Logic (Signal Pipeline)")
    st.markdown("""
`Session Break` -> `Kill Zone` -> `Volume Confirm` -> `Sweep Detected` -> `Depth Score (IV-adaptive)`
-> `Confirmation Candle` -> `MTF Check` -> `SL/TP Calc` -> `WFO Score (>= 0.25)` -> `DVOL R:R Scale` -> **EXECUTE**
""")

    _flow_diagram([
        {"label": "Identify session levels", "icon": "📍",
         "detail": "Track Asian (19:00-00:00 ET), London (03:00-08:00 ET), and NY (08:00-16:00 ET) "
                   "session highs and lows as liquidity targets."},
        {"label": "Kill zone + gamma regime check", "icon": "⏰",
         "detail": "Only enter during London (03:00-05:00 ET) or NY (08:00-10:30 ET) kill zones. "
                   "Fetch gamma regime data for IV-adaptive thresholds."},
        {"label": "Detect sweep + quality scoring", "icon": "💥",
         "detail": "Price breaks beyond session level. Sweep quality scored 0-100 across depth, "
                   "volume, time decay, and confirmation dimensions. Grade A+ to D."},
        {"label": "IV-adaptive depth validation", "icon": "📊",
         "detail": "Sweep depth must exceed a minimum ATR threshold that varies by IV regime: "
                   "lower threshold in low IV, higher in high IV environments."},
        {"label": "Confirmation + MTF alignment", "icon": "✅",
         "detail": "Confirmation candle in correct direction. MTF analysis scores Daily and 4H "
                   "alignment. Graduated short sizing applied for ETH."},
        {"label": "WFO counter-trend scoring", "icon": "📐",
         "detail": "4-component WFO score must exceed 0.25. DVOL medium regime (45-65) triggers "
                   "0.75x R:R scaling. ML Weight Adapter adjusts weights online."},
    ])

    # ── Exit Logic ───────────────────────────────────────────────────
    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Details |
|-----------|---------|
| **Take Profit** | Dynamic 1.5x-2.5x R:R, adjusted by gamma regime and DVOL |
| **Stop Loss** | ATR-based (2.5x ATR) or sweep-based (1.0 ATR buffer beyond sweep level) |
| **Trailing Stop** | Stepped: only moves on new high-water marks, 0.5x ATR threshold |
| **Breakeven Stop** | Moves to breakeven once 1R profit is achieved |
| **Time Exit** | Auto-close after 6 hours without SL/TP hit |
| **Volatility Adaptive** | High-vol widens SL by 25%, low-vol tightens by 20% |
| **Deep Sweep Bonus** | SL distance reduced for exceptionally deep sweeps |
""")

    # ── Risk Management ──────────────────────────────────────────────
    st.subheader("Risk Management Rules")
    rm_cols = st.columns(4)
    rm_cols[0].metric("Risk Per Trade", "1%", help="Fixed 1% of account equity risked per trade.")
    rm_cols[1].metric("Max Daily Loss", "4%", help="Bot pauses after 4% account loss in a day.")
    rm_cols[2].metric("Max Drawdown", "8%", help="Hard stop — bot disables if DD exceeds 8%.")
    rm_cols[3].metric("Re-entry Attempts", "2", help="Up to 2 re-entries after stop-out, 4-candle cooldown.")

    st.markdown("""
**Additional safeguards:**
- **Volatility-adjusted sizing**: `VolatilityAdjustedSizer` scales position based on current vol regime
- **Prop firm SL cap**: Maximum SL distance enforced to stay within prop firm rules
- **Graduated short sizing (ETH)**: Instead of hard-blocking shorts, position size scales down in bullish structure
- **Re-entry SL widening**: Re-entry trades get wider stops to account for post-stop volatility
- **Quant metrics tracked**: Sharpe, Sortino, Calmar, Profit Factor updated in real-time
""")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR (14-period)**: Dynamic position sizing, SL, trailing stop, and sweep depth normalization
- **EMA Stack (50/100/200)**: Daily directional bias — Long: 50>100>200, Short: 50<100<200
- **Sweep Quality Score (0-100)**: 4-dimension continuous grade replacing binary detection
- **Gamma Regime**: IV percentile, DVOL, gamma flip level from options microstructure
- **WFO Counter-Trend Score**: 4-component mean-reversion confidence (0.0-1.0)
- **MTF Alignment**: Daily + 4H structural alignment scoring
- **Volatility Percentile**: 20-candle rolling percentile for adaptive SL adjustment
""")

    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Gamma regime adds options microstructure edge to pure price action
- Continuous sweep quality scoring eliminates low-quality sweeps
- IV-adaptive thresholds prevent trading noise in extreme volatility
- WFO counter-trend scoring validates reversal quality
- MTF confirmation reduces false entries
- Graduated short sizing preserves opportunities while managing risk
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Kill zone restriction limits trading opportunities to ~5 hours/day
- Gamma data dependency adds a failure point (mitigated by fallback)
- Low WFO threshold (0.25) relies heavily on sweep quality scorer
- Session-level detection can be late if Asian session is thin
- 5M execution delay can miss fast V-reversals
""")


# ══════════════════════════════════════════════════════════════════════════════
# Momentum Mastery Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "Momentum Mastery":
    st.header("Momentum Mastery Strategy")
    st.caption("Combines daily EMA directional bias with kill-zone liquidity sweeps and strict confirmation candle quality filters. Enhanced with WFO 6-component scoring, ATR-regime R:R scaling, adaptive risk management, and premature entry analysis.")

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

    # ── Recent Improvements ──────────────────────────────────────────
    st.subheader("Recent Improvements")
    with st.expander("WFO 6-Component Momentum Scoring", expanded=True):
        st.markdown("""
| Component | Weight | What It Measures |
|-----------|--------|------------------|
| `sweep_depth` | 0.30 | Depth of liquidity sweep normalized by ATR (dominant) |
| `volume_conf` | 0.15 | Confirmation candle volume quality |
| `ema_align` | 0.15 | EMA stack alignment with trade direction |
| `confirm_quality` | 0.15 | Confirmation candle body ratio and displacement |
| `struct_bias` | 0.15 | Price structure (HH/HL or LH/LL) alignment |
| `atr_bonus` | 0.10 | ATR regime bonus (volatile = 1.0, quiet = -0.5) |

**Minimum confidence**: 0.35. A rejected signal resets sweep state entirely — no "almost good enough" entries.

The ML Weight Adapter adjusts component weights based on trade outcomes, retraining every 50 trades.
""")

    with st.expander("ATR-Regime R:R Scaling"):
        st.markdown("""
The R:R ratio is dynamically scaled based on the **ATR volatility regime** of recent bars:

| ATR Regime | R:R Adjustment | Rationale |
|------------|---------------|-----------|
| QUIET | -0.5 R:R reduction | Lower trending potential in quiet markets |
| NORMAL | Standard R:R | Baseline behavior |
| VOLATILE | +1.0 R:R bonus | Higher trending potential justifies wider targets |

This prevents over-reaching in quiet markets (tight TP) while capitalizing on volatile
conditions (wider TP).
""")

    with st.expander("Re-Entry System"):
        st.markdown("""
After a stop-out, the bot stores the stopped trade details and checks for **re-entry
opportunities** before looking for fresh sweeps each cycle.

**Re-entry conditions:**
- Within configurable candle window of original stop
- Past minimum cooldown period
- Attempts below maximum re-entry limit
- Price within 1.5 ATR of original sweep zone
- Fresh confirmation candle present

The sweep state is restored programmatically, allowing the bot to capitalize on setups
that were "right direction, wrong timing."
""")

    with st.expander("Adaptive Risk Manager + Premature Entry Analyzer"):
        st.markdown("""
**Adaptive Risk Manager**: Self-adjusts R:R targets and entry thresholds based on historical
trade outcomes. After every trade, it recalibrates:
- Confidence threshold (tighter after losses, looser after wins)
- TP allocation (shifts weight toward TP levels with highest hit rates)
- R:R targets (calibrated from MFE percentiles: p50 -> TP1, p75 -> TP2)

**Premature Entry Analyzer**: Monitors stopped-out trades for 24 hours to check if price
eventually reached the take profit level. If the "premature entry rate" exceeds thresholds:
- >30%: Raises requirements for correlated entry conditions
- >50%: Applies system-wide stricter entry rules

This feedback loop diagnoses "right direction, wrong timing" patterns and tightens
entry criteria accordingly.
""")

    st.subheader("Entry Logic (Signal Pipeline)")
    st.markdown("""
`Kill Zone` -> `Session Levels` -> `EMA Bias` -> `ATR Regime` -> `Regime Filter`
-> `Re-entry Check` -> `Sweep Detection` -> `Fractal Check` -> `Confirm Candle`
-> `Hybrid SL Calc` -> `WFO Score (>= 0.35)` -> `Adaptive Gate` -> `ML Filter` -> **EXECUTE**
""")

    _flow_diagram([
        {"label": "Establish daily directional bias", "icon": "📈",
         "detail": "Softened EMA filter produces a quantified trend_strength (0.0-1.0). "
                   "LONG: 50 > 100 > 200 EMA. SHORT: 50 < 100 < 200. Neutral = no trades."},
        {"label": "ATR regime classification", "icon": "📊",
         "detail": "ATR percentile classifies market as QUIET, NORMAL, or VOLATILE. "
                   "R:R ratio scales accordingly. Blocked regimes can be hard-filtered out."},
        {"label": "Liquidity sweep detection", "icon": "💥",
         "detail": "Price sweeps session low (longs) or high (shorts). Volume confirmation "
                   "required. Sweep age limit prevents stale setups."},
        {"label": "Fractal invalidation check", "icon": "🛡️",
         "detail": "Count counter-trend fractals since sweep candle. Too many fractals = "
                   "invalidated structure. Prevents entering stale or broken setups."},
        {"label": "Confirmation candle quality", "icon": "✅",
         "detail": "Body >= min ATR ratio, correct direction, displacement past sweep price. "
                   "Volume percentile threshold filters weak confirmations."},
        {"label": "WFO 6-component scoring", "icon": "📐",
         "detail": "6 weighted components scored. Minimum 0.35 confidence. "
                   "Rejection resets sweep state entirely — no partial entries."},
    ])

    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Details |
|-----------|---------|
| **Take Profit** | ATR-regime-scaled R:R (QUIET: reduced, VOLATILE: boosted) |
| **Stop Loss** | Hybrid: sweep-based primary -> ATR fallback -> floor (min) -> cap (max) |
| **Partial Exit** | At configurable partial R:R level, remainder trails |
| **Trailing Stop** | Activated after partial exit |
| **Staleness Exit** | Time/candle-based exit for positions that stall |
""")

    st.subheader("Risk Management Rules")
    rm_cols = st.columns(4)
    rm_cols[0].metric("Risk Per Trade", "1% (vol-adjusted)", help="Volatility-adjusted position sizing.")
    rm_cols[1].metric("Max Daily Trades", "5", help="Prevents overtrading even during active sessions.")
    rm_cols[2].metric("SL Floor + Cap", "Active", help="Minimum and maximum SL distance enforced.")
    rm_cols[3].metric("Re-entry System", "Active", help="Allows re-entry after stop-out with fresh confirmation.")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **EMA Stack (50/100/200)**: Daily chart directional bias with softened filter (trend strength 0-1)
- **ATR (14-period)**: R:R scaling, SL sizing, sweep depth normalization, regime classification
- **WFO Score (6 components)**: Momentum-optimized signal confidence (0.0-1.0)
- **Williams Fractals**: 5-candle high/low fractals for structure invalidation
- **Candle Body Ratio**: Body/range threshold ensures strong directional conviction
- **Volume Percentile**: Filters weak confirmation candles
- **ML Trade Filter**: Probability >= 0.45 gate after adaptive threshold
""")

    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- ATR-regime scaling optimizes R:R for current market conditions
- WFO hard rejection (reset sweep state) prevents marginal entries
- Re-entry system captures "right direction, wrong timing" setups
- Adaptive risk manager self-tunes from every trade outcome
- Premature entry analyzer diagnoses systematic timing errors
- Fractal invalidation catches deteriorating setups early
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Very selective — low trade frequency can mean missed opportunities
- Kill zone restriction limits entry windows to ~5 hours/day
- Sweep state reset on WFO rejection can miss borderline good setups
- Multiple adaptive systems increase parameter complexity
- Re-entry in same direction may compound directional risk
""")


# ══════════════════════════════════════════════════════════════════════════════
# SBS (Swing Break System / Smart Block Structure) Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "SBS":
    st.header("SBS — Swing Break System / Smart Block Structure")
    st.caption("The most complex strategy in the portfolio: combines ICT liquidity sweeps with Break of Structure (BOS) confirmation, Fibonacci retracements for precision entries, and a multi-TP trailing stop system. Enhanced with WFO 5-component scoring (highest confidence threshold at 0.48) and a pending entry queue with 15M confirmation.")

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
4. **Second Liquidity Grab**: Price retraces to the 0.618 Fibonacci level ("golden pocket")
   and shows a rejection — a second grab beyond this level confirms high-conviction entry

The strategy uses a **pending entry queue** — setups are detected on the 1H timeframe but
execution waits for **15M confirmation** (or falls back to 1H after 4 hours).
""")

    # Visual: SBS full lifecycle
    sbs_prices = [100, 101, 102, 103, 104, 105, 104, 103, 101, 99, 97, 96,
                  98, 100, 103, 106, 108, 105, 103, 101, 100, 102, 104, 107, 109, 111, 113]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sbs_prices, mode="lines", name="Price",
                             line=dict(color="white", width=2)))

    fig.add_annotation(x=11, y=96, text="1. Liquidity Sweep", showarrow=True, arrowhead=2,
                       font=dict(color="#F44336", size=11))
    fig.add_annotation(x=15, y=106, text="2. Break of Structure", showarrow=True, arrowhead=2,
                       font=dict(color="#FF9800", size=11))
    fig.add_annotation(x=20, y=100, text="3. 0.618 Fib + 2nd Grab", showarrow=True, arrowhead=2,
                       font=dict(color="#9C27B0", size=11))
    fig.add_trace(go.Scatter(x=[20], y=[100], mode="markers", name="Entry",
                             marker=dict(size=12, color="#2196F3", symbol="triangle-up")))

    fib_levels = {"1.0 (Sweep)": 96, "0.618": 100, "0.5": 102, "0.236": 105.5, "0.0 (Swing)": 108}
    for label, level in fib_levels.items():
        fig.add_hline(y=level, line_dash="dot", line_color="rgba(156,39,176,0.3)",
                      annotation_text=label, annotation_position="left")

    fig.add_annotation(x=23, y=105.5, text="TP1 (0.236)", showarrow=True, font=dict(color="#4CAF50"))
    fig.add_annotation(x=25, y=108, text="TP2 (0.0)", showarrow=True, font=dict(color="#4CAF50"))

    fig.update_layout(title="SBS Full Trade Lifecycle: Sweep -> BOS -> 2nd Grab -> Multi-TP",
                      template="plotly_dark", height=400, showlegend=False,
                      xaxis_title="Bars", yaxis_title="Price")
    st.plotly_chart(fig, key="sbs_diagram")

    # ── Recent Improvements ──────────────────────────────────────────
    st.subheader("Recent Improvements")
    with st.expander("WFO 5-Component Scoring (Highest Threshold)", expanded=True):
        st.markdown("""
SBS uses the **highest WFO confidence threshold** across all 4 strategies (0.48), reflecting
its higher inherent complexity and the need for stronger signal quality.

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| `sweep_depth` | 0.30 | Liquidity sweep depth normalized by ATR |
| `ema_align` | 0.20 | EMA stack alignment with trade direction |
| `rsi_align` | 0.15 | RSI alignment (momentum confirmation) |
| `volume_conf` | 0.15 | Volume on sweep and confirmation candles |
| `struct_bias` | 0.20 | Break of Structure quality and price structure |

**Minimum confidence**: 0.48 — the highest across all strategies.

**No DVOL/ATR gating**: SBS relies on the Fibonacci structure and multi-TP system
for risk management rather than regime-based gating.
""")

    with st.expander("Pending Entry Queue (1H Detection -> 15M Confirmation)"):
        st.markdown("""
SBS's most distinctive feature: **setup detection and execution are decoupled**.

**Detection Phase** (1H timeframe):
- Sweep + BOS + Fibonacci calculation + second grab detection
- WFO scoring applied
- Setup added to `pending_entries` queue (NOT immediately executed)

**Confirmation Phase** (15M timeframe):
- Each pending entry is checked against the last 16 x 15M candles (4 hours)
- **15M confirmation**: Candle wicks through the grab level and closes back through it
  with a bullish/bearish body OR strong wick rejection (wick > 0.5x body)
- **4H fallback**: If no 15M confirmation within 4 hours, enter at next 1H open

**Deduplication**: Duplicate pending entries for the same direction + price level are filtered.
Entries older than 2 hours are discarded.
""")

    with st.expander("Second Liquidity Grab Trigger"):
        st.markdown("""
Instead of entering on the first retracement to the 0.618 Fib level, SBS specifically
waits for a **second sweep** beyond that level:

1. After BOS, price retraces toward the sweep origin
2. A candle that **wicks beyond the 0.618 Fib level** and **closes back through it**
   is the actual entry trigger
3. This creates a high-conviction "second grab" confirmation

This double-grab pattern filters out weak retracements and ensures the entry zone
has been tested by institutional order flow before the bot commits capital.
""")

    st.subheader("Entry Logic (Signal Pipeline)")
    st.markdown("""
`1H Sweep` -> `BOS Confirm` -> `Swing Point Found` -> `Fib Levels Calc` -> `2nd Grab Detected`
-> `WFO Score (>= 0.48)` -> `Add to Pending Queue` -> `15M Confirmation (or 4H fallback)` -> **EXECUTE**
""")

    _flow_diagram([
        {"label": "Detect liquidity sweep (1H)", "icon": "💥",
         "detail": "Price breaks 0.1% beyond recent 50-candle range high (bearish) or low (bullish), "
                   "triggering clustered stop orders."},
        {"label": "Confirm Break of Structure (BOS)", "icon": "🔄",
         "detail": "After the sweep, price must close beyond the recent opposite-side structure level. "
                   "For longs: close > recent high after a low sweep."},
        {"label": "Calculate Fibonacci levels", "icon": "📐",
         "detail": "Swing range: 1.0 = sweep level, 0.0 = post-BOS swing point. "
                   "Key levels: 0.618 (entry zone), 0.5 (trail SL 1), 0.236 (TP1), 0.0 (TP2)."},
        {"label": "Detect second liquidity grab", "icon": "🎯",
         "detail": "Price must wick beyond the 0.618 Fib level and close back through it. "
                   "This second grab confirms institutional commitment to the reversal."},
        {"label": "WFO 5-component scoring", "icon": "📐",
         "detail": "5 weighted components scored. Minimum 0.48 confidence — the highest threshold "
                   "across all strategies. Failed signals are not queued."},
        {"label": "15M confirmation or 1H fallback", "icon": "🔬",
         "detail": "Setup enters pending queue. Best: 15M candle wicks through grab level and closes "
                   "back (rejection). Fallback: 1H entry after 4 hours without 15M confirmation."},
    ])

    st.subheader("Exit Conditions — Multi-TP Fibonacci Trailing System")
    st.markdown("""
| Level | Action | Trailing SL Moves To |
|-------|--------|---------------------|
| **TP1** (0.236 Fib) | Partial profit taken | SL trails to 0.5 level |
| **TP2** (0.0 / Swing) | More profit taken | SL trails to 0.236 level |
| **TP3** (Recent S/R) | Remaining position closed | SL trails to 0.0 level |
| **Stop Loss** | Full exit | At 1.0 Fib (sweep level) |
| **Re-entry** | Up to 2 attempts | Configurable window after stop |
""")

    st.subheader("Risk Management Rules")
    rm_cols = st.columns(4)
    rm_cols[0].metric("Position Sizing", "ATR-based", help="Dynamic sizing using ATR + confidence weighting.")
    rm_cols[1].metric("WFO Threshold", "0.48", help="Highest confidence threshold across all strategies.")
    rm_cols[2].metric("Pending Queue", "4h max", help="Setups expire from pending queue after 4 hours.")
    rm_cols[3].metric("MFE/MAE Tracking", "Active", help="Every active trade's MFE/MAE updated each loop.")

    st.markdown("""
**Additional safeguards:**
- **Fibonacci-anchored SL**: Stop placed at structurally meaningful level (sweep origin), not arbitrary ATR distance
- **Multi-TP cascade**: Progressive trailing through Fib levels eliminates downside risk as trade matures
- **WFO outcome recording**: Trade results feed back to ML Weight Adapter for continuous weight optimization
- **Daily/weekly Telegram reports**: Automated performance summaries via background scheduler
- **Max error auto-restart**: Bot auto-restarts after configurable consecutive error count
""")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR (14-period)**: Risk/reward calculations and dynamic position sizing
- **Fibonacci Retracements (0.0, 0.236, 0.5, 0.618, 1.0)**: Core entry/exit methodology
- **WFO Score (5 components)**: Signal confidence with 0.48 threshold (highest)
- **Break of Structure**: 0.1% threshold confirms market structure shift
- **Swing Point Detection**: Identifies high/low structures for Fibonacci range
- **EMA Stack**: Trend direction alignment for WFO `ema_align` component
- **RSI**: Momentum confirmation for WFO `rsi_align` component
- **Recent S/R Levels**: 50-candle lookback for TP3 placement
""")

    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Pending entry queue decouples detection from execution (higher precision)
- Second grab trigger provides high-conviction institutional confirmation
- 15M confirmation reduces false entries significantly
- Multi-TP Fibonacci cascade maximizes profit extraction
- Highest WFO threshold (0.48) ensures strongest signal quality
- Trailing SL through Fib levels progressively eliminates risk
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Most complex strategy — more parameters = more calibration needed
- Pending queue can miss fast V-reversals that don't retrace to 0.618
- 4H fallback entry sacrifices precision when 15M confirmation fails
- Multi-TP exits leave partial positions exposed to adverse moves
- 1H detection timeframe means slower setup generation
- No DVOL/ATR regime gating — relies entirely on Fibonacci structure
""")


# ══════════════════════════════════════════════════════════════════════════════
# Comparative Summary
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Strategy Comparison Matrix")
st.caption("Side-by-side comparison of all strategies including recent improvements. Shows how each strategy approaches signal scoring, regime filtering, and risk management differently.")

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
        "WFO Scoring",
        "WFO Threshold",
        "WFO Components",
        "Regime Gating",
        "Gamma Integration",
        "ML Filter",
        "Adaptive Learning",
        "Complexity",
    ],
    "FVG": [
        "3-candle gap + midpoint entry",
        "15m + HTF confluence",
        "ATR-regime-scaled R:R",
        "1% + vol scaling",
        "FVG boundary + ATR buffer",
        "24/7 crypto, sessions NQ",
        "After partial TP",
        "No",
        "9 components",
        "0.45 (FVG), 0.40 (iFVG)",
        "gap, vol, ema, rsi, struct, disp, sweep, hmm, rv",
        "HMM hard gate (crypto)",
        "No",
        "ML Weight Adapter",
        "Adaptive Risk + Premature Analyzer",
        "High",
    ],
    "Liquidity Raid": [
        "Session sweep + reversal",
        "15m / 5m precision",
        "Dynamic 1.5-2.5x R:R + gamma",
        "1% fixed",
        "ATR or sweep-based + IV-adaptive",
        "Kill zones only",
        "Yes (stepped)",
        "2 attempts",
        "4 components (counter-trend)",
        "0.25",
        "sweep_depth, counter_struct, counter_htf, struct_conf",
        "DVOL medium R:R scaling",
        "IV-adaptive depth + R:R",
        "ML Weight Adapter",
        "Quant metrics (Sharpe/Sortino)",
        "High",
    ],
    "Momentum Mastery": [
        "EMA bias + sweep + confirm",
        "15m signals",
        "ATR-regime-scaled R:R",
        "1% vol-adjusted",
        "Hybrid: sweep -> ATR -> floor -> cap",
        "Kill zones only",
        "After partial exit",
        "Yes (with cooldown)",
        "6 components (momentum)",
        "0.35",
        "sweep_depth, vol, ema, confirm, struct, atr_bonus",
        "ATR regime filter + blocked regimes",
        "No",
        "ML Filter (>= 0.45) + Weight Adapter",
        "Adaptive Risk + Premature Analyzer",
        "Medium-High",
    ],
    "SBS": [
        "Sweep + BOS + Fib 2nd grab",
        "1H detect, 15m confirm",
        "Multi-TP Fib trailing SL",
        "ATR + confidence",
        "Fibonacci 1.0 level (sweep)",
        "All sessions",
        "Yes (Fib cascade)",
        "2 attempts",
        "5 components",
        "0.48 (highest)",
        "sweep_depth, ema, rsi, vol, struct_bias",
        "No regime gate",
        "No",
        "ML Weight Adapter",
        "WFO outcome feedback",
        "Very High",
    ],
}

st.dataframe(
    pd.DataFrame(comparison_data).set_index("Aspect"),
    use_container_width=True,
    column_config={
        "FVG": st.column_config.TextColumn("FVG", help="Fair Value Gap strategy. Targets institutional price imbalances using multi-timeframe 3-candle gap patterns with HMM regime gating and midpoint entry."),
        "Liquidity Raid": st.column_config.TextColumn("Liquidity Raid", help="Exploits institutional stop-hunts with gamma regime integration, continuous sweep quality scoring, and counter-trend WFO validation."),
        "Momentum Mastery": st.column_config.TextColumn("Momentum Mastery", help="Trend-following + sweep hybrid with 6-component WFO scoring, ATR-regime R:R scaling, and adaptive risk management."),
        "SBS": st.column_config.TextColumn("SBS", help="Most complex strategy: sweep + BOS + Fibonacci entries with pending queue, 15M confirmation, and the highest WFO confidence threshold (0.48)."),
    },
)

# ── Shared Infrastructure ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Shared Infrastructure")
st.caption("All strategies share these cross-cutting systems:")

infra_cols = st.columns(3)
with infra_cols[0]:
    st.markdown("**WFO Signal Scorer**")
    st.markdown("""
    Single shared module with per-strategy configs. ML Weight Adapter
    uses ridge logistic regression, retrains every 50 outcomes, requires
    CV accuracy > 0.52. Weight multipliers clamped to [0.5x, 2.0x].
    """)
with infra_cols[1]:
    st.markdown("**ML Data Collection**")
    st.markdown("""
    All bots log entry/exit/MFE/MAE to SQLite via `MLIntegration`.
    70+ features per trade including WFO confidence, regime state,
    gamma data, sweep quality, and market context.
    """)
with infra_cols[2]:
    st.markdown("**Telegram Reporting**")
    st.markdown("""
    Real-time trade alerts, daily summaries, weekly performance
    reports, heartbeat pings, and error alerts. All bots use
    either sync or async Telegram integration.
    """)

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
            "Strategy": st.column_config.TextColumn("Strategy", help="Name of the trading strategy."),
            "Trades": st.column_config.TextColumn("Trades", help="Total number of live trades executed."),
            "Win Rate": st.column_config.TextColumn("Win Rate", help="Percentage of trades with positive P&L."),
            "Total P&L": st.column_config.TextColumn("Total P&L", help="Cumulative dollar profit or loss."),
            "Avg R": st.column_config.TextColumn("Avg R", help="Average R-multiple per trade. Above 0.3R is solid."),
            "Profit Factor": st.column_config.TextColumn("Profit Factor", help="Gross profit / gross loss. Above 1.5 is strong."),
        })
