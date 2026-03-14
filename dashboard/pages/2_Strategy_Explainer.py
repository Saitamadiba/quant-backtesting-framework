"""Page 10: Strategy Logic Explainer — plain-English breakdown of each strategy."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Strategy Explainer", page_icon="📖", layout="wide")
st.title("📖 Strategy Logic Explainer")
st.caption(
    "A plain-English reference guide for every strategy in the portfolio. "
    "Each section explains the core concepts, market mechanics, and the type of edge "
    "each strategy seeks — designed as an educational overview."
)

from config import STRATEGIES
from data.data_loader import get_all_trades

# Load live data for performance context
df_all = get_all_trades()

# ── Strategy selector ─────────────────────────────────────────────────────────
strategy = st.selectbox(
    "Select Strategy",
    list(STRATEGIES.keys()),
    help="Choose a strategy to see its conceptual breakdown.",
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
    c2.metric("Win Rate", f"{wr:.1%}", help="Percentage of trades that closed in profit.")
    c3.metric("Total P&L", f"${total_pnl:,.2f}", help="Cumulative realized profit/loss in USD across all trades.")
    c4.metric("Avg R", f"{avg_r:.2f}", help="Average R-multiple per trade. Positive means winners outpace losers on a risk-adjusted basis.")
    c5.metric("Total R", f"{total_r:.1f}", help="Sum of all R-multiples — the total risk-adjusted return.")


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
    st.caption("An ICT (Inner Circle Trader) concept: identifies imbalances in price delivery where institutional orders leave 'gaps' in the candle structure, then enters at optimal retracement levels within the gap zone.")

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

The bot detects these gaps across **multiple timeframes** and looks for price to retrace back
into the gap zone for an entry at a calculated **optimal level** within the gap.
""")

    # Visual: FVG pattern
    bullish_fvg = [100, 101, 99, 97, 96, 100, 103, 105, 103, 101, 98, 100, 102, 104, 106]
    fig = _entry_exit_diagram(
        "Bullish FVG Retracement Entry Example",
        bullish_fvg, entry_idx=10, sl=95.5, tp=107, direction="Long",
        annotations=[
            dict(x=3, y=96, text="FVG Zone", showarrow=True, arrowhead=2, font=dict(color="#FF9800")),
            dict(x=10, y=98, text="Retracement entry", showarrow=True, arrowhead=2),
        ],
    )
    st.plotly_chart(fig, key="fvg_diagram")

    # ── Key Components ──────────────────────────────────────────────
    st.subheader("Key Components")
    with st.expander("Regime Filtering (Crypto Only)", expanded=True):
        st.markdown("""
**What**: A regime classification model categorises the market into **volatile** (momentum) or
**calm** (ranging/choppy) states. The bot **skips signal generation entirely** during calm
regimes for crypto assets.

**Why**: Analysis showed that volatile regimes produce significantly higher expectancy than calm
regimes. By gating out calm periods, the bot only trades when its edge is strongest.

**Scope**: Applies to crypto assets only. Equity index futures are not gated because they
trade during defined sessions with naturally higher institutional activity.
""")

    with st.expander("Walk-Forward Optimized Signal Scoring"):
        st.markdown("""
**Walk-Forward Optimized (WFO)** signal scoring grades every FVG setup across multiple weighted
components before execution. Weights are calibrated on historical profitability data and
can adapt online via a machine learning adapter.

The scoring system evaluates factors such as:
- **Gap characteristics** — size and quality of the FVG relative to recent volatility
- **Volume confirmation** — whether institutional participation is evident
- **Trend alignment** — EMA stack and momentum indicators agreeing with direction
- **Market structure** — whether the broader price structure supports the trade
- **Regime context** — volatility and market state alignment

A minimum composite confidence score must be exceeded before any trade is taken.
The ML adapter periodically retrains from trade outcomes, adjusting component weights
within bounded ranges to prevent overfitting.
""")

    with st.expander("Optimal Entry + Volatility-Scaled Stop"):
        st.markdown("""
**Entry Location**: Entries target a calculated optimal level within the FVG zone, balancing
fill probability against risk distance. This was validated through backtesting as the
best entry location for the pattern.

**Volatility-Scaled Stop Loss**: Stop loss is placed relative to the FVG zone boundary with
a volatility buffer, ensuring stops adapt to current market conditions.

**Take Profit**: TP targets are scaled by the prevailing volatility regime — wider in
trending conditions, tighter in ranging markets.
""")

    with st.expander("Session Filtering"):
        st.markdown("""
Session filtering is enforced **only for equity index futures**. Crypto assets trade
**24/7** because analysis showed no statistically significant edge improvement from
session filtering on crypto pairs — institutional flow in crypto is distributed across
all sessions, unlike equity futures which concentrate around market opens.
""")

    with st.expander("Inverted FVG (iFVG) Detection"):
        st.markdown("""
The strategy also detects and trades **inverted FVGs** — gaps that form when a previous
FVG zone is fully invalidated and price creates a new gap in the opposite direction.

iFVGs use their own scoring configuration with adjusted weights that emphasize regime
alignment, reflecting the different statistical characteristics of inverted patterns.
""")

    # ── Entry Logic ──────────────────────────────────────────────────
    st.subheader("Entry Logic (Conceptual Pipeline)")
    st.markdown("""
`FVG Detected` -> `Regime Gate` -> `Session Filter` -> `Invalidation Check`
-> `Proximity Gate` -> `Rejection Candle` -> `Signal Score` -> `Risk Gate` -> **ENTRY**
""")

    _flow_diagram([
        {"label": "Regime gate (crypto only)", "icon": "🧠",
         "detail": "Bot checks the market regime classification. If the market is in a calm/ranging "
                   "state, the entire signal generation cycle is skipped."},
        {"label": "Scan for FVG patterns across timeframes", "icon": "🔍",
         "detail": "Scans multiple timeframes for 3-candle FVG patterns. Higher timeframe gaps "
                   "receive greater weight. Minimum gap size thresholds vary by timeframe."},
        {"label": "FVG invalidation + proximity check", "icon": "📊",
         "detail": "Tracked FVGs are checked for invalidation (price closed through zone). "
                   "Price must be within a configured proximity of the gap before entry."},
        {"label": "WFO confidence scoring", "icon": "📐",
         "detail": "Multi-component weighted score must exceed a minimum threshold. Components "
                   "evaluate gap quality, volume, trend alignment, structure, and regime context."},
        {"label": "Entry execution", "icon": "🎯",
         "detail": "Entry at calculated optimal level within the FVG zone. Stop loss at zone "
                   "boundary with volatility buffer. Take profit scaled by regime."},
    ])

    # ── Exit Logic ───────────────────────────────────────────────────
    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Description |
|-----------|-------------|
| **Take Profit** | Volatility-regime-scaled target — wider in trends, tighter in ranges |
| **Stop Loss** | FVG zone boundary with volatility buffer |
| **Partial TP** | Partial exit at first target, remainder trails |
| **Trailing Stop** | Activated after partial TP hit |
| **Stale Exit** | Auto-close after configurable time without SL/TP hit |
| **Position Sizing** | Volatility-adjusted sizing with bounded multipliers |
""")

    # ── Risk Management ──────────────────────────────────────────────
    st.subheader("Risk Management Approach")
    st.markdown("""
- **Daily trade limit** — prevents overtrading during choppy sessions
- **Account drawdown cap** — bot halts if drawdown exceeds a configured threshold
- **Daily loss limit** — stops new trades after a daily loss threshold is breached
- **Maximum stop distance** — enforces a ceiling on stop-loss distance
- **Traded zone expiry** — prevents re-trading the same FVG zone within a cooldown window
- **Adaptive risk manager** — self-adjusts thresholds and TP allocation based on recent performance
- **Premature entry analyzer** — monitors stopped trades to detect timing issues and tighten entry rules
""")

    # ── Key Indicators ───────────────────────────────────────────────
    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR**: Stop-loss sizing, take-profit calculation, volatility regime detection
- **Regime Classification**: Gates signal generation based on market state
- **WFO Signal Score**: Weighted composite confidence with ML-adaptive weights
- **EMA Stack**: Trend direction alignment
- **Volume**: Confirms institutional participation in FVG formation
- **Multi-Timeframe Priority**: Higher timeframes carry more weight
- **RSI**: Overbought/oversold alignment
- **Volatility Percentile**: ATR rank for regime classification
""")

    # ── Strengths & Weaknesses ───────────────────────────────────────
    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Regime gating eliminates low-expectancy calm-market trades
- WFO scoring with adaptive weights continuously improves signal quality
- Calculated entry location provides precise, validated positioning
- Multi-timeframe analysis reduces false signals
- Adaptive risk manager self-tunes from trade outcomes
- iFVG detection captures additional gap opportunities
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Regime gate can miss early-trend entries before state transitions
- Multi-component scoring adds latency to signal evaluation
- Entry may not fill if price reverses at zone boundary
- Complex scoring system has more parameters to calibrate
""")


# ══════════════════════════════════════════════════════════════════════════════
# Liquidity Raid Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "Liquidity Raid":
    st.header("Liquidity Raid Strategy")
    st.caption("Exploits institutional liquidity sweeps: smart money raids stop-loss clusters at session highs/lows, then reverses. Enhanced with options microstructure data, continuous sweep quality grading, and multi-timeframe entry confirmation.")

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
3. **CONFIRMED**: Reversal confirmed with displacement and structure
4. **EXPIRED**: No confirmation within timeout — move on

The strategy only trades during **high-activity windows** (kill zones) when institutional
participation is at its peak.
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

    # ── Key Components ──────────────────────────────────────────────
    st.subheader("Key Components")
    with st.expander("Options Microstructure Integration (Gamma/IV Regime)", expanded=True):
        st.markdown("""
**What**: The bot queries an **options microstructure module** for real-time implied volatility (IV),
DVOL, and gamma flip data. This is used for:

1. **IV-Adaptive Sweep Depth**: Different minimum sweep depth thresholds depending on the
   current IV regime — sweeps need to be more significant in high-volatility environments
   to filter noise
2. **Gamma-Informed R:R**: Take-profit levels can be adjusted based on the gamma regime
   to account for expected price behavior near key options levels

The system classifies IV into tiers (low, medium, high) and adjusts sweep depth requirements
and risk/reward targets accordingly. If the options module is unavailable, the bot falls
back to standard thresholds.
""")

    with st.expander("Walk-Forward Counter-Trend Scoring"):
        st.markdown("""
The Liquidity Raid uses a **counter-trend** WFO scorer — unlike trend-following approaches,
this scorer grades sweeps by their **mean-reversion quality**.

Key scoring dimensions include:
- **Sweep depth** — how deep the sweep penetrated relative to recent volatility (dominant signal)
- **Counter-trend structure** — quality of the price structure supporting reversal
- **Higher timeframe alignment** — whether higher timeframes agree with the reversal thesis
- **Structural confirmation** — evidence that the reversal is confirmed by market structure

Because the sweep quality scorer provides additional filtering, the WFO threshold can be
set lower than trend-following strategies while maintaining signal quality.
""")

    with st.expander("Continuous Sweep Quality Scoring"):
        st.markdown("""
Replaces binary sweep detection with a **continuous quality score** across multiple dimensions:

1. **Depth**: How far price penetrated the session level, normalized by volatility
2. **Volume**: Sweep candle volume relative to recent average
3. **Time Decay**: Freshness penalty — recent sweeps score higher, stale sweeps degrade
4. **Confirmation**: Body ratio and directional match quality

Sweeps are assigned letter grades (A+ through D) based on composite score.
Higher grades receive larger position sizes; lower grades are skipped entirely.
""")

    with st.expander("Multi-Timeframe Entry Confirmation"):
        st.markdown("""
**MTF Analysis** scores alignment across higher timeframes with the trade direction:

- **Daily alignment**: EMA stack and structure on the daily chart
- **4H alignment**: Intermediate timeframe price structure

Two modes are available:
- **Soft** (default): Informs confidence and is logged, but doesn't reject signals
- **Hard**: Rejects counter-trend signals below a score threshold

MTF scores and alignment flags are included in every signal for tracking purposes.
""")

    with st.expander("Pending Execution System"):
        st.markdown("""
After a sweep is detected on the primary timeframe, a **pending entry** waits for
lower-timeframe confirmation before triggering.

**Confirmation requires:**
- Correct directional close (bullish for longs, bearish for shorts)
- Candle range exceeding a minimum volatility threshold
- Entry placed at a pullback from the confirmation candle range

If no confirmation arrives within a configurable window, the pending sweep expires.
""")

    # ── Entry Logic ──────────────────────────────────────────────────
    st.subheader("Entry Logic (Conceptual Pipeline)")
    st.markdown("""
`Session Break` -> `Kill Zone` -> `Volume Confirm` -> `Sweep Detected` -> `Depth Score (IV-adaptive)`
-> `Confirmation Candle` -> `MTF Check` -> `SL/TP Calc` -> `Signal Score` -> `Regime R:R Scale` -> **EXECUTE**
""")

    _flow_diagram([
        {"label": "Identify session levels", "icon": "📍",
         "detail": "Track session highs and lows from defined time windows (Asian, London, NY) "
                   "as liquidity targets."},
        {"label": "Kill zone + regime check", "icon": "⏰",
         "detail": "Only enter during high-activity kill zone windows. Fetch options regime data "
                   "for IV-adaptive thresholds."},
        {"label": "Detect sweep + quality scoring", "icon": "💥",
         "detail": "Price breaks beyond session level. Sweep quality scored across depth, volume, "
                   "time decay, and confirmation dimensions."},
        {"label": "IV-adaptive depth validation", "icon": "📊",
         "detail": "Sweep depth must exceed a minimum volatility-normalized threshold that varies "
                   "by the current IV regime."},
        {"label": "Confirmation + MTF alignment", "icon": "✅",
         "detail": "Confirmation candle in correct direction. MTF analysis scores higher-timeframe "
                   "alignment with the reversal thesis."},
        {"label": "WFO counter-trend scoring", "icon": "📐",
         "detail": "Multi-component counter-trend score must exceed threshold. Options regime may "
                   "scale R:R targets. ML adapter adjusts weights online."},
    ])

    # ── Exit Logic ───────────────────────────────────────────────────
    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Description |
|-----------|-------------|
| **Take Profit** | Dynamic R:R target adjusted by options regime and volatility |
| **Stop Loss** | Volatility-based or sweep-based, whichever provides better structure |
| **Trailing Stop** | Stepped trailing — only moves on new high-water marks |
| **Breakeven Stop** | Moves to breakeven once a profit threshold is achieved |
| **Time Exit** | Auto-close after a maximum holding period without SL/TP hit |
| **Volatility Adaptive** | SL widens in high-vol, tightens in low-vol environments |
""")

    # ── Risk Management ──────────────────────────────────────────────
    st.subheader("Risk Management Approach")
    st.markdown("""
- **Fixed risk per trade** — consistent percentage of account equity risked
- **Daily loss limit** — bot pauses after a configured daily loss threshold
- **Account drawdown cap** — hard stop if drawdown exceeds maximum
- **Re-entry system** — limited re-entries after stop-out with cooldown period
- **Volatility-adjusted sizing** — position scales with current volatility regime
- **Graduated short sizing** — instead of blocking shorts outright, size scales down in bullish structure
- **Re-entry SL widening** — re-entry trades get wider stops to account for post-stop volatility
""")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR**: Dynamic position sizing, SL, trailing stop, and sweep depth normalization
- **EMA Stack**: Daily directional bias
- **Sweep Quality Score**: Multi-dimension continuous grade replacing binary detection
- **Options Regime**: IV percentile, DVOL, gamma flip level from options microstructure
- **WFO Counter-Trend Score**: Mean-reversion confidence scoring
- **MTF Alignment**: Higher-timeframe structural alignment scoring
- **Volatility Percentile**: Rolling percentile for adaptive SL adjustment
""")

    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Options microstructure data adds a unique edge to pure price action
- Continuous sweep quality scoring eliminates low-quality setups
- IV-adaptive thresholds prevent trading noise in extreme volatility
- WFO counter-trend scoring validates reversal quality
- MTF confirmation reduces false entries
- Graduated short sizing preserves opportunities while managing risk
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Kill zone restriction limits trading to high-activity windows only
- Options data dependency adds a failure point (mitigated by fallback)
- Session-level detection can be late if early sessions are thin
- Lower-timeframe execution delay can miss fast V-reversals
""")


# ══════════════════════════════════════════════════════════════════════════════
# Momentum Mastery Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "Momentum Mastery":
    st.header("Momentum Mastery Strategy")
    st.caption("Combines daily EMA directional bias with kill-zone liquidity sweeps and strict confirmation candle quality filters. A trend-following hybrid enhanced with adaptive risk management and premature entry analysis.")

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
filters, and volume thresholds to avoid false signals. This means fewer trades but
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

    # ── Key Components ──────────────────────────────────────────────
    st.subheader("Key Components")
    with st.expander("WFO Momentum Scoring", expanded=True):
        st.markdown("""
The Momentum Mastery WFO scorer evaluates signals across multiple weighted components
tuned for **trend-following with sweep confirmation**:

Key scoring dimensions include:
- **Sweep depth** — depth of the liquidity sweep normalized by volatility (dominant signal)
- **Volume confirmation** — quality of volume on the confirmation candle
- **Trend alignment** — EMA stack agreement with trade direction
- **Confirmation quality** — candle body ratio and displacement strength
- **Market structure** — whether price structure (higher highs/lows or lower highs/lows) supports the trade
- **Volatility bonus** — regime-based adjustment favoring volatile conditions

A rejected signal resets sweep state entirely — there are no "almost good enough" entries.
The ML adapter adjusts component weights based on trade outcomes.
""")

    with st.expander("ATR-Regime R:R Scaling"):
        st.markdown("""
The R:R ratio is dynamically scaled based on the **ATR volatility regime** of recent bars:

- **Quiet regime**: R:R target is reduced — lower trending potential in quiet markets
- **Normal regime**: Standard R:R baseline
- **Volatile regime**: R:R target is boosted — higher trending potential justifies wider targets

This prevents over-reaching in quiet markets (tight TP) while capitalizing on volatile
conditions (wider TP).
""")

    with st.expander("Re-Entry System"):
        st.markdown("""
After a stop-out, the bot stores the stopped trade details and checks for **re-entry
opportunities** before looking for fresh sweeps each cycle.

**Re-entry conditions:**
- Within a configurable candle window of the original stop
- Past minimum cooldown period
- Attempts below maximum re-entry limit
- Price within a volatility-defined distance of the original sweep zone
- Fresh confirmation candle present

The sweep state is restored programmatically, allowing the bot to capitalize on setups
that were "right direction, wrong timing."
""")

    with st.expander("Adaptive Risk Manager + Premature Entry Analyzer"):
        st.markdown("""
**Adaptive Risk Manager**: Self-adjusts R:R targets and entry thresholds based on historical
trade outcomes. After every trade, it recalibrates:
- Confidence threshold — tighter after losses, looser after wins
- TP allocation — shifts weight toward TP levels with highest hit rates
- R:R targets — calibrated from favorable excursion percentiles

**Premature Entry Analyzer**: Monitors stopped-out trades to check if price eventually
reached the take profit level. If the "premature entry rate" exceeds configurable
thresholds, it progressively tightens entry requirements.

This feedback loop diagnoses "right direction, wrong timing" patterns and adjusts
entry criteria accordingly.
""")

    st.subheader("Entry Logic (Conceptual Pipeline)")
    st.markdown("""
`Kill Zone` -> `Session Levels` -> `EMA Bias` -> `ATR Regime` -> `Regime Filter`
-> `Re-entry Check` -> `Sweep Detection` -> `Fractal Check` -> `Confirm Candle`
-> `SL Calc` -> `Signal Score` -> `Adaptive Gate` -> `ML Filter` -> **EXECUTE**
""")

    _flow_diagram([
        {"label": "Establish daily directional bias", "icon": "📈",
         "detail": "EMA filter produces a quantified trend strength score. "
                   "Requires EMA stack alignment for directional trades. Neutral = no trades."},
        {"label": "ATR regime classification", "icon": "📊",
         "detail": "ATR percentile classifies market as quiet, normal, or volatile. "
                   "R:R ratio scales accordingly."},
        {"label": "Liquidity sweep detection", "icon": "💥",
         "detail": "Price sweeps a session level. Volume confirmation required. "
                   "Sweep age limit prevents stale setups."},
        {"label": "Fractal invalidation check", "icon": "🛡️",
         "detail": "Counter-trend fractals are counted since the sweep candle. Too many fractals "
                   "invalidate the structure. Prevents entering stale or broken setups."},
        {"label": "Confirmation candle quality", "icon": "✅",
         "detail": "Candle body meets minimum quality thresholds, correct direction, displacement "
                   "past sweep price. Volume filter removes weak confirmations."},
        {"label": "WFO momentum scoring", "icon": "📐",
         "detail": "Multi-component weighted score must exceed threshold. "
                   "Rejection resets sweep state entirely — no partial entries."},
    ])

    st.subheader("Exit Conditions")
    st.markdown("""
| Condition | Description |
|-----------|-------------|
| **Take Profit** | ATR-regime-scaled R:R (reduced in quiet, boosted in volatile) |
| **Stop Loss** | Hybrid: sweep-based primary -> volatility fallback -> floor -> cap |
| **Partial Exit** | At configurable level, remainder trails |
| **Trailing Stop** | Activated after partial exit |
| **Staleness Exit** | Time/candle-based exit for positions that stall |
""")

    st.subheader("Risk Management Approach")
    st.markdown("""
- **Volatility-adjusted risk per trade** — position sizing adapts to current conditions
- **Daily trade limit** — prevents overtrading even during active sessions
- **SL floor + cap** — minimum and maximum stop-loss distance enforced
- **Re-entry system** — allows re-entry after stop-out with fresh confirmation and cooldown
- **Adaptive risk manager** — self-tunes thresholds from every trade outcome
- **Premature entry analyzer** — diagnoses systematic timing errors and tightens criteria
""")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **EMA Stack**: Daily chart directional bias with softened filter (trend strength score)
- **ATR**: R:R scaling, SL sizing, sweep depth normalization, regime classification
- **WFO Score**: Momentum-optimized signal confidence
- **Williams Fractals**: High/low fractals for structure invalidation
- **Candle Body Ratio**: Quality threshold ensures strong directional conviction
- **Volume Percentile**: Filters weak confirmation candles
- **ML Trade Filter**: Probability gate after adaptive threshold
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
- Kill zone restriction limits entry windows
- Sweep state reset on WFO rejection can miss borderline good setups
- Multiple adaptive systems increase parameter complexity
- Re-entry in same direction may compound directional risk
""")


# ══════════════════════════════════════════════════════════════════════════════
# SBS (Swing Break System / Smart Block Structure) Strategy
# ══════════════════════════════════════════════════════════════════════════════
elif strategy == "SBS":
    st.header("SBS — Swing Break System / Smart Block Structure")
    st.caption("The most complex strategy in the portfolio: combines ICT liquidity sweeps with Break of Structure (BOS) confirmation, Fibonacci retracements for precision entries, and a multi-TP trailing stop system with a pending entry queue and lower-timeframe confirmation.")

    st.subheader("Live Performance Snapshot")
    _show_live_summary("SBS")
    st.markdown("---")

    # ── Core Concept ──────────────────────────────────────────────────
    st.subheader("Core Concept")
    st.markdown("""
**SBS** stands for **Swing Break System** (describing what it does — trading swing breaks)
and **Smart Block Structure** (describing what it targets — smart money order blocks).

The strategy identifies institutional order blocks through a multi-stage process:

1. **Liquidity Sweep**: Price breaks beyond a recent high/low, triggering clustered stops
2. **Break of Structure (BOS)**: After the sweep, price reverses and breaks a key structure level
   — confirming that smart money has shifted direction
3. **Fibonacci Retracement**: The bot calculates Fibonacci levels between the sweep level and
   the post-BOS swing point
4. **Second Liquidity Grab**: Price retraces to a key Fibonacci level and shows a rejection —
   a second grab confirms high-conviction entry

The strategy uses a **pending entry queue** — setups are detected on a higher timeframe but
execution waits for **lower-timeframe confirmation** (or falls back after a timeout).
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
    fig.add_annotation(x=20, y=100, text="3. Fib Retracement + 2nd Grab", showarrow=True, arrowhead=2,
                       font=dict(color="#9C27B0", size=11))
    fig.add_trace(go.Scatter(x=[20], y=[100], mode="markers", name="Entry",
                             marker=dict(size=12, color="#2196F3", symbol="triangle-up")))

    # Conceptual Fib levels (illustrative)
    fib_levels = {"Sweep Level": 96, "Key Fib": 100, "Mid Fib": 102, "Upper Fib": 105.5, "Swing Point": 108}
    for label, level in fib_levels.items():
        fig.add_hline(y=level, line_dash="dot", line_color="rgba(156,39,176,0.3)",
                      annotation_text=label, annotation_position="left")

    fig.add_annotation(x=23, y=105.5, text="TP1", showarrow=True, font=dict(color="#4CAF50"))
    fig.add_annotation(x=25, y=108, text="TP2", showarrow=True, font=dict(color="#4CAF50"))

    fig.update_layout(title="SBS Full Trade Lifecycle: Sweep -> BOS -> 2nd Grab -> Multi-TP",
                      template="plotly_dark", height=400, showlegend=False,
                      xaxis_title="Bars", yaxis_title="Price")
    st.plotly_chart(fig, key="sbs_diagram")

    # ── Key Components ──────────────────────────────────────────────
    st.subheader("Key Components")
    with st.expander("WFO Signal Scoring (Highest Threshold)", expanded=True):
        st.markdown("""
SBS uses the **highest WFO confidence threshold** across all strategies, reflecting
its higher inherent complexity and the need for stronger signal quality.

Key scoring dimensions include:
- **Sweep depth** — liquidity sweep depth normalized by volatility
- **Trend alignment** — EMA stack agreement with trade direction
- **Momentum confirmation** — RSI and momentum indicator alignment
- **Volume confirmation** — volume quality on sweep and confirmation candles
- **Structural quality** — Break of Structure quality and price structure analysis

SBS relies on the Fibonacci structure and multi-TP system for risk management
rather than regime-based gating used by other strategies.
""")

    with st.expander("Pending Entry Queue (Detection -> Confirmation)"):
        st.markdown("""
SBS's most distinctive feature: **setup detection and execution are decoupled**.

**Detection Phase** (higher timeframe):
- Sweep + BOS + Fibonacci calculation + second grab detection
- WFO scoring applied
- Setup added to pending entries queue (NOT immediately executed)

**Confirmation Phase** (lower timeframe):
- Each pending entry is checked against recent lower-timeframe candles
- **Lower-TF confirmation**: candle wicks through the grab level and closes back
  with a directional body or strong wick rejection
- **Timeout fallback**: if no lower-TF confirmation within the timeout window,
  enter at the next higher-TF candle open

**Deduplication**: Duplicate pending entries for the same direction + price level are
filtered. Stale entries beyond a maximum age are discarded.
""")

    with st.expander("Second Liquidity Grab Trigger"):
        st.markdown("""
Instead of entering on the first retracement to a key Fibonacci level, SBS specifically
waits for a **second sweep** beyond that level:

1. After BOS, price retraces toward the sweep origin
2. A candle that **wicks beyond the key Fib level** and **closes back through it**
   is the actual entry trigger
3. This creates a high-conviction "second grab" confirmation

This double-grab pattern filters out weak retracements and ensures the entry zone
has been tested by institutional order flow before the bot commits capital.
""")

    st.subheader("Entry Logic (Conceptual Pipeline)")
    st.markdown("""
`Sweep Detected` -> `BOS Confirmed` -> `Swing Point Found` -> `Fib Levels Calculated`
-> `2nd Grab Detected` -> `Signal Score (highest threshold)` -> `Add to Pending Queue`
-> `Lower-TF Confirmation (or timeout fallback)` -> **EXECUTE**
""")

    _flow_diagram([
        {"label": "Detect liquidity sweep", "icon": "💥",
         "detail": "Price breaks beyond the recent range high or low, triggering "
                   "clustered stop orders."},
        {"label": "Confirm Break of Structure (BOS)", "icon": "🔄",
         "detail": "After the sweep, price must close beyond the recent opposite-side structure "
                   "level, confirming a directional shift."},
        {"label": "Calculate Fibonacci levels", "icon": "📐",
         "detail": "Fibonacci retracement levels are calculated between the sweep level and the "
                   "post-BOS swing point. Key levels define entry zone, TP targets, and trailing SL anchors."},
        {"label": "Detect second liquidity grab", "icon": "🎯",
         "detail": "Price must wick beyond the key Fibonacci level and close back through it, "
                   "confirming institutional commitment to the reversal."},
        {"label": "WFO scoring (highest threshold)", "icon": "📐",
         "detail": "Multi-component score must exceed the portfolio's highest confidence threshold. "
                   "Failed signals are not queued."},
        {"label": "Lower-TF confirmation or fallback", "icon": "🔬",
         "detail": "Setup enters pending queue. Best case: lower-timeframe candle shows wick rejection "
                   "at the grab level. Fallback: higher-TF entry after timeout."},
    ])

    st.subheader("Exit Conditions — Multi-TP Fibonacci Trailing System")
    st.markdown("""
| Level | Action | Trailing SL Behavior |
|-------|--------|---------------------|
| **TP1** (upper Fibonacci level) | Partial profit taken | SL trails to mid-level |
| **TP2** (swing point) | More profit taken | SL trails further |
| **TP3** (recent S/R level) | Remaining position closed | SL at structure level |
| **Stop Loss** | Full exit | At sweep level |
| **Re-entry** | Limited attempts | Configurable window after stop |
""")

    st.subheader("Risk Management Approach")
    st.markdown("""
- **Volatility-based position sizing** — dynamic sizing using ATR and confidence weighting
- **Highest WFO threshold** — most stringent signal quality requirements across all strategies
- **Pending queue timeout** — setups expire from the pending queue after a maximum age
- **MFE/MAE tracking** — every active trade's favorable/adverse excursion updated each cycle
- **Fibonacci-anchored SL** — stop placed at structurally meaningful level, not arbitrary distance
- **Multi-TP cascade** — progressive trailing through Fib levels eliminates risk as trade matures
- **WFO outcome recording** — trade results feed back to ML adapter for continuous optimization
""")

    st.subheader("Key Technical Indicators")
    st.markdown("""
- **ATR**: Risk/reward calculations and dynamic position sizing
- **Fibonacci Retracements**: Core entry/exit methodology — key levels define the entire trade plan
- **WFO Score**: Signal confidence with highest threshold across all strategies
- **Break of Structure**: Confirms market structure shift after sweep
- **Swing Point Detection**: Identifies high/low structures for Fibonacci range
- **EMA Stack**: Trend direction alignment
- **RSI**: Momentum confirmation
- **Recent S/R Levels**: Historical support/resistance for TP placement
""")

    st.subheader("Strengths & Weaknesses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        st.markdown("""
- Pending entry queue decouples detection from execution (higher precision)
- Second grab trigger provides high-conviction institutional confirmation
- Lower-timeframe confirmation reduces false entries significantly
- Multi-TP Fibonacci cascade maximizes profit extraction
- Highest WFO threshold ensures strongest signal quality
- Trailing SL through Fib levels progressively eliminates risk
""")
    with c2:
        st.markdown("**Weaknesses**")
        st.markdown("""
- Most complex strategy — more parameters = more calibration needed
- Pending queue can miss fast V-reversals that don't retrace to key Fib level
- Timeout fallback entry sacrifices precision when lower-TF confirmation fails
- Multi-TP exits leave partial positions exposed to adverse moves
- Higher-TF detection means slower setup generation
""")


# ══════════════════════════════════════════════════════════════════════════════
# Comparative Summary
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Strategy Comparison Matrix")
st.caption("Side-by-side comparison of all strategies. Shows how each strategy approaches signal detection, regime filtering, and risk management differently.")

comparison_data = {
    "Aspect": [
        "Primary Edge",
        "Entry Signal",
        "Execution Approach",
        "Exit Method",
        "Stop Loss Method",
        "Session Dependency",
        "Trailing Stop",
        "Re-entry After Stop",
        "Signal Scoring",
        "Regime Filtering",
        "Options Integration",
        "Adaptive Learning",
        "Relative Complexity",
    ],
    "FVG": [
        "Price delivery inefficiency",
        "3-candle gap + retracement",
        "Multi-TF confluence",
        "Volatility-regime-scaled R:R",
        "Gap boundary + volatility buffer",
        "24/7 crypto, sessions for futures",
        "After partial TP",
        "No",
        "Multi-component (gap, volume, trend, structure, regime)",
        "Hard gate on calm regimes (crypto)",
        "No",
        "ML weight adapter + adaptive risk manager",
        "High",
    ],
    "Liquidity Raid": [
        "Stop-hunt reversal",
        "Session sweep + reversal",
        "Lower-TF precision entry",
        "Dynamic R:R + options-adjusted",
        "Volatility-based or sweep-based",
        "Kill zones only",
        "Yes (stepped)",
        "Limited attempts with cooldown",
        "Counter-trend (sweep depth, structure, HTF, confirmation)",
        "IV regime scaling",
        "IV-adaptive depth + R:R scaling",
        "ML weight adapter + quant metrics",
        "High",
    ],
    "Momentum Mastery": [
        "Trend + sweep hybrid",
        "EMA bias + sweep + confirmation",
        "Signal-based execution",
        "ATR-regime-scaled R:R",
        "Hybrid: sweep -> volatility -> floor -> cap",
        "Kill zones only",
        "After partial exit",
        "Yes (with cooldown)",
        "Momentum-optimized (sweep, volume, trend, confirmation, structure, volatility)",
        "ATR regime filter",
        "No",
        "ML filter + weight adapter + adaptive risk + premature analyzer",
        "Medium-High",
    ],
    "SBS": [
        "Structure break + Fibonacci",
        "Sweep + BOS + Fib 2nd grab",
        "Pending queue with lower-TF confirmation",
        "Multi-TP Fib trailing SL cascade",
        "Fibonacci-anchored (sweep level)",
        "All sessions",
        "Yes (Fib cascade)",
        "Limited attempts",
        "Highest confidence threshold",
        "No regime gate (relies on Fibonacci structure)",
        "No",
        "ML weight adapter + WFO outcome feedback",
        "Very High",
    ],
}

st.dataframe(
    pd.DataFrame(comparison_data).set_index("Aspect"),
    use_container_width=True,
    column_config={
        "FVG": st.column_config.TextColumn("FVG", help="Fair Value Gap strategy. Targets institutional price imbalances using multi-timeframe gap patterns with regime gating."),
        "Liquidity Raid": st.column_config.TextColumn("Liquidity Raid", help="Exploits institutional stop-hunts with options microstructure integration and continuous sweep quality scoring."),
        "Momentum Mastery": st.column_config.TextColumn("Momentum Mastery", help="Trend-following + sweep hybrid with adaptive risk management and premature entry analysis."),
        "SBS": st.column_config.TextColumn("SBS", help="Most complex strategy: sweep + BOS + Fibonacci entries with pending queue and the highest signal confidence threshold."),
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
    Single shared module with per-strategy configurations. An ML weight
    adapter uses regularized logistic regression and periodically retrains
    from trade outcomes, keeping weight adjustments within bounded ranges
    to prevent overfitting.
    """)
with infra_cols[1]:
    st.markdown("**ML Data Collection**")
    st.markdown("""
    All bots log entry/exit/MFE/MAE to SQLite. Dozens of features per
    trade are recorded including signal confidence, regime state,
    sweep quality, and market context for offline analysis.
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
    st.caption("How each strategy is actually performing with real money.")

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
            "Avg R": st.column_config.TextColumn("Avg R", help="Average R-multiple per trade."),
            "Profit Factor": st.column_config.TextColumn("Profit Factor", help="Gross profit / gross loss."),
        })
