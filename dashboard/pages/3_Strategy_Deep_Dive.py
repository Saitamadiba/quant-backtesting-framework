"""Page 2: Strategy Deep Dive — Architecture-guide-style analysis per strategy.

Presents the same depth of analysis as the ARCHITECTURE_GUIDE.html but
integrated into the dashboard with live data context.  Covers: architecture,
modules, signal pipeline, file map, learning loop, and asset differences.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Strategy Deep Dive", page_icon="🔍", layout="wide")
st.title("🔍 Strategy Deep Dive")
st.caption(
    "Architecture-level analysis for each running strategy. Covers the engine, "
    "modules, signal pipeline, file interactions, learning loops, and asset-specific "
    "differences — the full technical blueprint of each bot."
)

from data.data_loader import get_all_trades
from components.kpi_cards import strategy_kpis
from components.charts import (
    r_multiple_histogram, cumulative_pnl_line, rolling_win_rate,
    exit_reason_donut, mfe_mae_scatter,
)
from components.filters import source_filter, symbol_filter, date_range_filter, apply_filters
from config import STRATEGIES

# ── Sidebar ───────────────────────────────────────────────────────────────────
strategy = st.sidebar.selectbox("Strategy", list(STRATEGIES.keys()), key="dd_strat")
src = source_filter(key_prefix="dd")
df_all = get_all_trades(source_filter=src)
symbols = symbol_filter(df_all[df_all["strategy"] == strategy], key_prefix="dd")
date_start, date_end = date_range_filter(df_all, key_prefix="dd")

# ── Filter ────────────────────────────────────────────────────────────────────
df = apply_filters(
    df_all, strategies=[strategy], symbols=symbols,
    date_start=date_start, date_end=date_end,
)

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.subheader(f"{strategy} Performance")
st.caption("Core performance metrics. Profit Factor above 1.5 and positive Avg R indicate a healthy edge.")
strategy_kpis(df)
st.markdown("---")

# ── Tab layout: Architecture + Analytics ─────────────────────────────────────
tab_arch, tab_analytics = st.tabs(["Architecture Deep Dive", "Analytics & Charts"])


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:

    # ══════════════════════════════════════════════════════════════════
    # FVG Architecture
    # ══════════════════════════════════════════════════════════════════
    if strategy == "FVG":
        st.header("FVG — Fair Value Gap: Architecture Blueprint")

        # ── The Big Picture ──────────────────────────────────────────
        st.subheader("The Big Picture")
        st.info(
            "**The Pothole Analogy:** Imagine a highway where cars drive at a steady pace. "
            "Occasionally, a stretch of road gets skipped — traffic leaps from point A to point C, "
            "leaving point B untouched. That gap is like a *Fair Value Gap* in the market: price "
            "moved so fast it left an 'unfilled' zone. The FVG bot detects these gaps across "
            "multiple timeframes and trades the retracement back into the midpoint."
        )

        st.markdown("""
All three FVG bots (BTC, ETH, NQ) share a common engine in `core/`. Think of it as the
"franchise playbook" — the recipe is identical, only the ingredients (data sources, sessions,
risk parameters) change per asset.
""")

        # ── Core Modules ────────────────────────────────────────────
        st.subheader("Core Modules — The Shared Engine")
        modules = {
            "config_base.py": ("Master configuration with **1,174 parameters**: risk limits, session windows, FVG rules, "
                              "ML thresholds, TP/SL levels, prop firm caps.", "@dataclass", "774 lines"),
            "bot_base.py": ("The **abstract engine**. Full trading cycle: HMM gate -> detect FVGs -> WFO score -> "
                           "midpoint entry -> manage positions -> learn from results.", "ABC", "~2,300 lines"),
            "regime_detector.py": ("HMM-based classifier. **4 regimes** (momentum_bullish, momentum_bearish, "
                                  "ranging, choppy). Gates signal generation for crypto.", "5 factors", "798 lines"),
            "adaptive_risk_manager.py": ("**Self-learning** module. After every trade, adjusts confidence thresholds, "
                                        "TP allocation, and R:R targets based on historical performance.", "3 learning loops", "660 lines"),
            "premature_entry_analyzer.py": ("Detects 'right direction, wrong timing' errors. Monitors stopped-out "
                                           "trades for 24h to see if price eventually reached target.", "9 conditions", "827 lines"),
            "telegram_notifier.py": ("Real-time trade alerts, daily summaries, heartbeat pings. "
                                    "Plain text for maximum reliability.", "HTTP POST", "144 lines"),
        }

        cols = st.columns(3)
        for i, (name, (desc, pattern, lines)) in enumerate(modules.items()):
            with cols[i % 3]:
                st.markdown(f"**`{name}`**")
                st.markdown(f"{desc}")
                st.caption(f"{pattern} | {lines}")

        # ── Inheritance ──────────────────────────────────────────────
        st.subheader("Inheritance — One Playbook, Three Players")
        st.markdown("""
```
FVGConfigBase (@dataclass, 1,174 params)
  ├── BTCConfig (+14 overrides)
  ├── ETHConfig (+18 overrides)
  └── NQConfig  (+22 overrides)

FVGBotBase (ABC, 3 @abstractmethod)
  ├── BTCFVGBot (259 lines)
  ├── ETHFVGBot (262 lines)
  └── NQFVGBot  (470 lines — largest, 5-source data cascade)
```

**Savings:** ~10,000 lines of duplicated code eliminated. Each asset wrapper only
configures data sources, session windows, and risk overrides.
""")

        # ── Signal Pipeline ──────────────────────────────────────────
        st.subheader("Complete Signal Pipeline")
        pipeline_steps = [
            ("1. Reset & Validate", "Reset daily metrics, validate balance, check prop firm DD limits"),
            ("2. Fetch Data", "Multi-timeframe fetch (all configured TFs at once)"),
            ("3. HMM Hard Gate", "Skip entire cycle if regime is ranging/choppy/unknown (crypto only)"),
            ("4. Vol Scaling", "Update volatility scaling for position sizing"),
            ("5. Momentum Exhaustion", "Enhanced momentum exhaustion detection"),
            ("6. Phase 1: FVG Retest", "Check tracked FVGs: invalidation -> proximity -> rejection -> WFO (>= 0.45) -> midpoint entry"),
            ("7. Phase 2: iFVG", "Inverted FVG detection and trading (separate WFO scorer, >= 0.40)"),
            ("8. Track New FVGs", "Regime filter -> weekend filter -> session filter (NQ) -> sweep/displacement -> queue"),
            ("9. Monitor Positions", "SL/TP/trailing/partial/time-based exits"),
            ("10. Report & Heartbeat", "Scheduled reporting, heartbeat check"),
        ]
        for step, detail in pipeline_steps:
            st.markdown(f"**{step}**: {detail}")

        # ── Asset Differences ────────────────────────────────────────
        st.subheader("Asset Differences at a Glance")
        asset_data = {
            "Property": [
                "Primary Timeframe", "Data Source", "Sessions Traded",
                "HMM Gate", "Session Filter", "Position Monitoring",
                "Wrapper Size", "Data Fallbacks",
            ],
            "BTC": [
                "5m", "Binance WebSocket", "24/7 (all sessions)",
                "Active (crypto)", "Bypassed", "WebSocket (instant)",
                "259 lines", "None needed",
            ],
            "ETH": [
                "5m", "Binance + Coinbase", "24/7 (all sessions)",
                "Active (crypto)", "Bypassed", "WebSocket (instant)",
                "262 lines", "Coinbase fallback",
            ],
            "NQ": [
                "15m (no 5m data)", "Alpaca + 4 fallbacks", "NY sessions only",
                "Not active (equity)", "London + NY enforced", "Polling (15s loop)",
                "470 lines", "Yahoo, AlphaVantage, NASDAQ, TAAPI",
            ],
        }
        st.dataframe(pd.DataFrame(asset_data).set_index("Property"), use_container_width=True)

        # ── File Interaction Map ─────────────────────────────────────
        st.subheader("File Interaction Map")
        st.markdown("""
```
FVG_Strategy/
├── core/                          # Shared engine — hub for all asset bots
│   ├── bot_base.py (HUB) ──────→ imports all other core modules
│   │   ├── config_base.py         # 1,174 parameters
│   │   ├── regime_detector.py     # HMM 4-state classifier
│   │   ├── adaptive_risk_manager  # Self-learning risk tuning
│   │   ├── premature_entry_analyzer # Entry timing feedback
│   │   └── telegram_notifier.py   # Alert delivery
│   └── __init__.py
├── BTC/                           # Thin wrapper → inherits from core
│   ├── fvg_btc.py (entry point)
│   ├── binance_websocket_fetcher  # Asset-specific data source
│   ├── fvg_detector.py            # FVG pattern detection
│   ├── ml_trade_filter.py         # ML gating
│   ├── ml_data_collector.py       # 70+ feature ML logging
│   ├── ml_integration_helper.py   # Simplified ML wrapper
│   ├── wfo_signal_scorer.py       # 9-component WFO scoring
│   └── ... 8 more modules
├── ETH/                           # Same structure, Coinbase fallback
└── NQ/                            # Same structure, 5-source cascade
```
""")

        # ── Learning Loop ────────────────────────────────────────────
        st.subheader("Learning & Adaptation Loop")
        st.info(
            "**The Chess Engine:** After every game, a chess engine reviews what worked and what "
            "didn't. These bots do the same — every closed trade feeds back into three learning "
            "systems that update parameters for the next trade."
        )

        learn_cols = st.columns(3)
        with learn_cols[0]:
            st.markdown("**Adaptive Risk Manager**")
            st.markdown("""
- **Confidence threshold** (2.0-5.0): Up on losing streaks, down on winning streaks
- **TP allocation**: Shifts weight toward TP levels with highest hit rates
- **R:R targets**: Calibrated from MFE percentiles (p50 -> TP1, p75 -> TP2)
""")
        with learn_cols[1]:
            st.markdown("**Premature Entry Analyzer**")
            st.markdown("""
- Tracks 9 entry conditions; correlates with premature entries
- If premature rate > 30%: raises requirements for correlated conditions
- If > 50%: applies system-wide stricter entry rules
""")
        with learn_cols[2]:
            st.markdown("**WFO Weight Adapter**")
            st.markdown("""
- Ridge logistic regression on trade outcomes
- Retrains every 50 completed trades
- Requires CV accuracy > 0.52 before adaptation
- Weight multipliers clamped to [0.5x, 2.0x] of baseline
""")

    # ══════════════════════════════════════════════════════════════════
    # Liquidity Raid Architecture
    # ══════════════════════════════════════════════════════════════════
    elif strategy == "Liquidity Raid":
        st.header("Liquidity Raid: Architecture Blueprint")

        st.subheader("The Big Picture")
        st.info(
            "**The Fake Sale Analogy:** A store announces a 70% OFF flash sale. A crowd rushes in. "
            "But once everyone is inside, the sale was a trick — the real bargains are around the corner. "
            "In markets, this is a *liquidity sweep*: price breaks a session level to trigger stops, "
            "then reverses. The bot detects these fake breakouts and trades the reversal."
        )

        st.markdown("""
Unlike FVG's monolithic `bot_base.py`, the Liquidity Raid strategy splits responsibilities into
**14 focused modules**, each owning one part of the process.
""")

        # ── Core Modules ────────────────────────────────────────────
        st.subheader("Core Modules — 14 Specialized Components")
        modules = {
            "strategy.py": ("The **signal engine**. Detects session-level sweeps, scores quality, "
                          "confirms with displacement, FVG, and WFO scoring.", "Core signal logic", "~800 lines"),
            "position_manager.py": ("**Position lifecycle** from open to close. Tracks P&L, trailing stops, "
                                  "partial TPs, breakeven moves.", "Full lifecycle", "809 lines"),
            "session_manager.py": ("Tracks **session highs/lows** (Asian, London, NY). These levels are "
                                  "the 'bait' that sweeps target.", "DST-aware", "352 lines"),
            "sweep_quality_scorer.py": ("Grades sweeps by **4 continuous dimensions**: depth-to-ATR, "
                                       "volume ratio, time decay, and confirmation quality. Score 0-100.", "A+ to D grades", "141 lines"),
            "sweep_state.py": ("A **state machine** (IntEnum) tracking sweep progression: "
                             "MONITORING -> DETECTED -> CONFIRMED -> EXPIRED.", "IntEnum pattern", "~50 lines"),
            "technical_analysis.py": ("Computes **indicators**: EMAs, RSI, ATR, market structure "
                                    "(higher highs/lower lows).", "272 lines", "272 lines"),
            "gamma_regime.py": ("**Options microstructure** data: IV percentile, DVOL, gamma flip level. "
                              "Used for IV-adaptive sweep depth and R:R adjustment.", "External module", "shared"),
            "database_manager.py": ("**Trade persistence** in SQLite. Schema includes WFO metadata "
                                  "(confidence, components, regime_gate, dvol_percentile).", "SQLite", "~200 lines"),
            "wfo_signal_scorer.py": ("**Counter-trend** WFO scoring: sweep_depth (0.50), counter_struct (0.20), "
                                   "counter_htf (0.15), struct_conf (0.15). Min confidence 0.25.", "4 components", "shared"),
        }

        cols = st.columns(3)
        for i, (name, (desc, pattern, lines)) in enumerate(modules.items()):
            with cols[i % 3]:
                st.markdown(f"**`{name}`**")
                st.markdown(f"{desc}")
                st.caption(f"{pattern} | {lines}")

        # ── State Machine ────────────────────────────────────────────
        st.subheader("State Machine — Sweep Lifecycle")
        st.markdown("""
```
MONITORING ──[level breached]──> DETECTED ──[reversal confirmed]──> CONFIRMED ──> TRADED
                                     │
                                     └──[no confirm / timeout]──> EXPIRED
```

Each session level (Asia High, Asia Low, London High, London Low, etc.) is tracked independently.
Once a level transitions to TRADED or EXPIRED, it cannot be re-used — preventing overtrading
exhausted levels.
""")

        # ── Signal Pipeline ──────────────────────────────────────────
        st.subheader("Complete Signal Pipeline")
        pipeline_steps = [
            ("1. Fetch 15M Bars", "Candle data retrieval"),
            ("2. Position Management", "Check SL/TP/trailing/time-exit for open positions"),
            ("3. Daily Guards", "Trade limit check, position cap"),
            ("4. 5M Pending Check", "Check pending sweep for 5M confirmation or timeout"),
            ("5. Gamma Regime", "Fetch IV, DVOL, gamma flip data"),
            ("6. Session Levels", "Update Asia/London/NY high and low levels"),
            ("7. Indicators", "EMA50, EMA200, ATR, price structure"),
            ("8. Directional Bias", "Price structure primary, EMA secondary"),
            ("9. Kill Zone Gate", "London or NY only"),
            ("10. Re-entry Check", "Priority over fresh sweeps"),
            ("11. Volume Confirm", "Sweep candle volume vs average"),
            ("12. Sweep Detection", "State machine + price reclaim"),
            ("13. Depth Validation", "IV-adaptive threshold (LOW/MED/HIGH)"),
            ("14. Short Filters", "London High filter, graduated sizing (ETH)"),
            ("15. Confirmation Candle", "Direction + body quality"),
            ("16. MTF Check", "Daily + 4H alignment scoring"),
            ("17. SL/TP Calculation", "Sweep-based primary, ATR fallback"),
            ("18. WFO Score", ">= 0.25 counter-trend confidence"),
            ("19. DVOL Scaling", "Medium DVOL (45-65) -> 0.75x R:R"),
            ("20. Signal Emission", "Signal dict + ML entry log"),
        ]
        for step, detail in pipeline_steps:
            st.markdown(f"**{step}**: {detail}")

        # ── Asset Differences ────────────────────────────────────────
        st.subheader("Asset Differences")
        asset_data = {
            "Property": [
                "Data Source", "Short Handling", "Session Bias",
                "Confirmation Delay", "Gamma Asset", "Wrapper",
            ],
            "BTC": [
                "Binance WebSocket", "Standard filtering",
                "All sessions weighted equally", "None",
                "BTC", "lr_btc_v2.py (thin)",
            ],
            "ETH": [
                "Binance WebSocket", "Graduated sizing (not hard-blocked)",
                "London/NY differentiated multipliers", "Optional short confirmation delay",
                "ETH", "lr_eth_v2.py (thin)",
            ],
        }
        st.dataframe(pd.DataFrame(asset_data).set_index("Property"), use_container_width=True)

        # ── File Map ─────────────────────────────────────────────────
        st.subheader("File Interaction Map")
        st.markdown("""
```
Liquidity_Raid/
├── core/                              # Shared engine — 14 modules
│   ├── bot_base.py (HUB) ──────────→ orchestrates all modules
│   │   ├── strategy.py                # Signal generation engine
│   │   ├── position_manager.py        # Position lifecycle (809 lines)
│   │   ├── session_manager.py         # Session high/low tracking
│   │   ├── sweep_quality_scorer.py    # 4-dimension continuous scoring
│   │   ├── sweep_state.py             # State machine (IntEnum)
│   │   ├── technical_analysis.py      # Indicators
│   │   ├── config_base.py             # 35 settings groups
│   │   ├── database_manager.py        # Trade persistence + WFO metadata
│   │   ├── ml_data_collector.py       # 70+ feature logging
│   │   ├── ml_integration_helper.py   # Simplified ML wrapper
│   │   ├── gamma_regime.py            # Options microstructure (shared)
│   │   ├── wfo_signal_scorer.py       # Counter-trend scoring (shared)
│   │   └── telegram_notifier.py       # Alert delivery
│   └── __init__.py
├── BTC_V2/                            # Thin BTC wrapper
│   └── lr_btc_v2.py (entry point)
└── ETH_V2/                            # Thin ETH wrapper + graduated sizing
    └── lr_eth_v2.py (entry point)
```
""")

        # ── Learning ─────────────────────────────────────────────────
        st.subheader("Learning & Adaptation")
        learn_cols = st.columns(3)
        with learn_cols[0]:
            st.markdown("**WFO Weight Adapter**")
            st.markdown("Online ridge regression adapts component weights every 50 outcomes. "
                       "Counter-trend scoring evolves as the market changes.")
        with learn_cols[1]:
            st.markdown("**Quant Metrics**")
            st.markdown("Sharpe, Sortino, Calmar, and Profit Factor updated in real-time "
                       "for continuous strategy health monitoring.")
        with learn_cols[2]:
            st.markdown("**ML Feature Logging**")
            st.markdown("70+ features per trade including gamma regime, sweep quality score, "
                       "WFO components, MTF alignment for post-hoc analysis.")

    # ══════════════════════════════════════════════════════════════════
    # Momentum Mastery Architecture
    # ══════════════════════════════════════════════════════════════════
    elif strategy == "Momentum Mastery":
        st.header("Momentum Mastery: Architecture Blueprint")

        st.subheader("The Big Picture")
        st.info(
            "**The Surfer Analogy:** A surfer doesn't fight the ocean — they identify the wave "
            "(daily EMA trend), wait for the pullback (liquidity sweep), and paddle in when the "
            "wave starts rising again (confirmation candle). Momentum Mastery does the same: "
            "ride the trend, exploit the pullback, confirm the reversal."
        )

        st.markdown("""
Momentum Mastery follows the same **thin wrapper** architecture as Liquidity Raid: a shared
`core/` engine with asset-specific BTC and ETH wrappers. The strategy is deliberately
**conservative** — fewer trades, higher conviction per entry.
""")

        # ── Core Modules ────────────────────────────────────────────
        st.subheader("Core Modules")
        modules = {
            "strategy.py": ("**Signal engine**. Softened EMA bias -> sweep detection -> fractal check -> "
                          "confirmation candle -> WFO 6-component scoring.", "Core logic", "~600 lines"),
            "bot_base.py": ("**Orchestration hub**. Manages the full cycle: fetch -> strategy -> adaptive gate -> "
                          "ML filter -> execute -> learn.", "Orchestration", "~800 lines"),
            "adaptive_risk_manager.py": ("**Self-tuning** R:R targets and entry thresholds from trade history. "
                                        "Win streak -> lower threshold. Loss streak -> higher.", "3 loops", "~500 lines"),
            "premature_entry_analyzer.py": ("Post-SL monitoring: did price reach TP after stop? "
                                           "Feeds timing diagnostics back for entry tightening.", "24h monitor", "~400 lines"),
            "technical_analysis.py": ("EMA stack, ATR, ATR percentile, Williams Fractals, "
                                    "candle quality metrics.", "Static methods", "~300 lines"),
            "config_base.py": ("Configuration with ATR regime settings, fractal params, "
                             "kill zone windows, re-entry rules.", "@dataclass", "~400 lines"),
        }

        cols = st.columns(3)
        for i, (name, (desc, pattern, lines)) in enumerate(modules.items()):
            with cols[i % 3]:
                st.markdown(f"**`{name}`**")
                st.markdown(f"{desc}")
                st.caption(f"{pattern} | {lines}")

        # ── Signal Pipeline ──────────────────────────────────────────
        st.subheader("Complete Signal Pipeline")
        pipeline_steps = [
            ("1. Kill Zone Check", "London (03:00-05:00 ET) or NY (08:00-10:30 ET)"),
            ("2. Asian Session Filter", "Optional block during Asian session"),
            ("3. Session Levels", "Update liquidity levels (session highs/lows)"),
            ("4. Indicators + Fractals", "EMA stack, ATR, ATR percentile, Williams fractals"),
            ("5. Directional Bias", "Softened EMA filter -> trend_strength (0.0-1.0)"),
            ("6. ATR-Scaled R:R", "QUIET: -0.5, NORMAL: 0, VOLATILE: +1.0 R:R adjustment"),
            ("7. Regime Filter", "Blocked volatility regimes -> skip"),
            ("8. Re-entry Check", "Restores sweep state if valid re-entry opportunity"),
            ("9. Sweep Detection", "Liquidity level swept + volume confirm + age filter"),
            ("10. Fractal Invalidation", "Count counter-trend fractals since sweep -> invalidate if too many"),
            ("11. Confirmation Candle", "Direction + body >= min ATR ratio + displacement past sweep"),
            ("12. Hybrid SL", "Sweep-based primary -> ATR fallback -> floor (min) -> cap (max)"),
            ("13. WFO 6-Component Score", ">= 0.35 confidence; rejection = full sweep state reset"),
            ("14. Adaptive Threshold", "Adaptive Risk Manager gate"),
            ("15. ML Trade Filter", ">= 0.45 probability gate"),
            ("16. Signal Emission", "Signal dict with confidence, wfo_components, atr_regime"),
        ]
        for step, detail in pipeline_steps:
            st.markdown(f"**{step}**: {detail}")

        # ── ATR Regime Scaling ───────────────────────────────────────
        st.subheader("ATR Regime R:R Scaling")
        st.markdown("""
| ATR Regime | R:R Adjustment | WFO `atr_bonus` Score |
|------------|---------------|----------------------|
| QUIET | -0.5 R:R (tighter TP) | -0.5 (penalty) |
| NORMAL | Standard R:R | 0.0 (neutral) |
| VOLATILE | +1.0 R:R (wider TP) | 1.0 (bonus) |

The ATR regime is determined from the ATR percentile of recent bars, creating a
continuous adaptation to current market volatility conditions.
""")

        # ── Asset Differences ────────────────────────────────────────
        st.subheader("Asset Differences")
        asset_data = {
            "Property": ["Data Source", "Kill Zones", "Re-entry", "Wrapper"],
            "BTC": ["Binance WebSocket", "London + NY", "Active", "btc_momentum_mastery_v2.py"],
            "ETH": ["Binance WebSocket", "London + NY", "Active", "eth_momentum_mastery_v2.py"],
        }
        st.dataframe(pd.DataFrame(asset_data).set_index("Property"), use_container_width=True)

        # ── File Map ─────────────────────────────────────────────────
        st.subheader("File Interaction Map")
        st.markdown("""
```
Momentum_Mastery/
├── core/                              # Shared engine
│   ├── bot_base.py (HUB) ──────────→ orchestrates all modules
│   │   ├── strategy.py                # 6-component WFO signal engine
│   │   ├── adaptive_risk_manager.py   # Self-tuning risk from outcomes
│   │   ├── premature_entry_analyzer   # 24h post-SL monitoring
│   │   ├── technical_analysis.py      # EMA, ATR, fractals, candle quality
│   │   ├── config_base.py             # ATR regime, fractal, re-entry config
│   │   ├── database_manager.py        # Trade persistence + WFO metadata
│   │   ├── ml_data_collector.py       # 70+ feature logging
│   │   ├── ml_integration_helper.py   # ML wrapper
│   │   ├── wfo_signal_scorer.py       # 6-component momentum scoring (shared)
│   │   └── telegram_notifier.py       # Alert delivery
│   └── __init__.py
├── BTC/                               # Thin BTC wrapper
│   └── btc_momentum_mastery_v2.py
└── ETH/                               # Thin ETH wrapper
    └── eth_momentum_mastery_v2.py
```
""")

        # ── Learning Loop ────────────────────────────────────────────
        st.subheader("Learning & Adaptation Loop")
        st.markdown("""
Momentum Mastery has the **most comprehensive learning system** with three feedback loops:

1. **Adaptive Risk Manager**: Adjusts confidence threshold, TP allocation, R:R targets after
   every trade. Win streaks lower the bar; loss streaks raise it.

2. **Premature Entry Analyzer**: Monitors stopped trades for 24h. If price reached TP after SL:
   - >30% premature rate -> tighten correlated entry conditions
   - >50% premature rate -> system-wide stricter rules

3. **WFO Weight Adapter**: Ridge logistic regression adapts the 6 component weights every
   50 outcomes. Clamped to [0.5x, 2.0x] of baseline. Requires CV accuracy > 0.52.

These three systems create a **continuous self-improvement cycle**: every trade makes the
next trade's entry criteria more refined.
""")

    # ══════════════════════════════════════════════════════════════════
    # SBS Architecture
    # ══════════════════════════════════════════════════════════════════
    elif strategy == "SBS":
        st.header("SBS — Swing Break System: Architecture Blueprint")

        st.subheader("The Big Picture")
        st.info(
            "**The Sting Operation:** Think of a police sting: (1) set the bait (liquidity level), "
            "(2) wait for the target to take it (sweep), (3) confirm they committed (BOS), "
            "(4) let them come back for more (Fibonacci retracement), (5) catch them at the "
            "second attempt (second grab at 0.618). SBS runs this sting on every timeframe."
        )

        st.markdown("""
SBS is the **most complex strategy** in the portfolio. It decouples setup detection (1H) from
execution (15M pending confirmation), uses Fibonacci-anchored entries and exits, and requires
the highest WFO confidence threshold (0.48) across all strategies.
""")

        # ── Core Modules ────────────────────────────────────────────
        st.subheader("Core Modules")
        modules = {
            "bot_base.py (core)": ("**Async orchestration hub**. Main trading loop, ML integration, "
                                  "MFE/MAE tracking, daily/weekly Telegram reporting, auto-restart.", "asyncio", "~450 lines"),
            "sbs_strategy.py (btc)": ("**BTC signal engine**. Sweep + BOS + Fibonacci + 2nd grab detection. "
                                     "Pending entry queue with 15M confirmation. WFO 5-component scoring.", "Full pipeline", "~700 lines"),
            "sbs_strategy.py (eth)": ("**ETH signal engine**. Same architecture as BTC with ETH-specific "
                                     "config overrides (timeframes, ATR settings).", "ETH variant", "~700 lines"),
            "data_manager.py": ("Data fetching via REST API. Multi-timeframe candle retrieval "
                              "with quality validation.", "REST-based", "~200 lines"),
            "telegram_bot.py": ("Async Telegram integration. Signal alerts with entry/SL/TP/R:R, "
                              "status updates, error alerts.", "aiohttp", "~300 lines"),
            "ml_data_collector.py": ("70+ feature ML logging with WFO metadata: confidence, "
                                   "components, regime_gate, atr_at_entry.", "SQLite", "~400 lines"),
        }

        cols = st.columns(3)
        for i, (name, (desc, pattern, lines)) in enumerate(modules.items()):
            with cols[i % 3]:
                st.markdown(f"**`{name}`**")
                st.markdown(f"{desc}")
                st.caption(f"{pattern} | {lines}")

        # ── Pending Entry System ─────────────────────────────────────
        st.subheader("Pending Entry Queue — Decoupled Detection & Execution")
        st.markdown("""
SBS's most distinctive architectural feature:

```
1H ANALYSIS                     PENDING QUEUE                    15M CONFIRMATION
───────────                     ─────────────                    ────────────────
Sweep detected         ──>     Setup queued with:       ──>     Check last 16 x 15M candles:
BOS confirmed                   - Direction                      - Wick through grab level?
Fibonacci calculated             - Fib levels                    - Close back through it?
2nd grab detected                - WFO score (>= 0.48)          - Bullish/bearish body?
                                 - Market context                - Strong wick rejection?
                                 - Max wait: 4 hours
                                                         ──>     YES: Execute at 15M level
                                                         ──>     NO (4h): Fallback to 1H entry
                                                         ──>     EXPIRED: Discard
```

**Why**: Decoupling detection from execution allows the bot to wait for the optimal
micro-timeframe entry point, significantly improving the risk/reward ratio compared
to entering directly on the 1H signal.
""")

        # ── Fibonacci Structure ──────────────────────────────────────
        st.subheader("Fibonacci Entry/Exit Structure")
        st.markdown("""
| Fib Level | Role | Trading Action |
|-----------|------|---------------|
| **1.0** | Sweep level | Initial stop loss placement |
| **0.618** | Golden pocket | Entry zone (second grab trigger) |
| **0.5** | Midpoint | Trailing SL after TP1 hit |
| **0.236** | First target | **TP1** — partial exit, trail SL to 0.5 |
| **0.0** | Swing high/low | **TP2** — main target, trail SL to 0.236 |
| **Beyond 0.0** | S/R level | **TP3** — remaining position, trail to 0.0 |

The Fibonacci cascade creates a **self-tightening risk profile**: as each TP is hit,
the trailing stop moves to the previous TP level, progressively eliminating downside risk
while allowing remaining position to capture extended moves.
""")

        # ── Signal Pipeline ──────────────────────────────────────────
        st.subheader("Complete Signal Pipeline")
        pipeline_steps = [
            ("1. Pending Entry Check", "Check queue for 15M confirmation or 4H fallback"),
            ("2. Re-entry Check", "Opportunity to re-enter after stop-out"),
            ("3. 1H SBS Analysis", "Process recent 24 candles (expanded from 6 for ML training)"),
            ("4. Indicators", "EMA, ATR, RSI, Fibonacci levels, bias check"),
            ("5. First Sweep Detection", "Session low (LONG) or session high (SHORT) breach"),
            ("6. BOS Confirmation", "Close beyond opposite session structure"),
            ("7. Swing Point Found", "Highest high (bullish) or lowest low (bearish) after sweep"),
            ("8. Fibonacci Calculation", "Levels from sweep (1.0) to swing (0.0)"),
            ("9. Second Grab Detection", "Wick beyond 0.618 Fib + close back through it"),
            ("10. WFO 5-Component Score", ">= 0.48 confidence (highest threshold)"),
            ("11. Add to Pending Queue", "Setup queued with Fib levels + WFO metadata"),
            ("12. 15M Confirmation", "Wick through grab level + rejection body"),
            ("13. Signal Emission", "Multi-TP Fibonacci structure + WFO metadata"),
        ]
        for step, detail in pipeline_steps:
            st.markdown(f"**{step}**: {detail}")

        # ── Asset Differences ────────────────────────────────────────
        st.subheader("Asset Differences")
        asset_data = {
            "Property": ["Data Source", "Primary TF", "Confirmation TF",
                        "Sessions", "WFO Config", "Wrapper"],
            "BTC": ["Binance REST", "1H", "15M", "All sessions",
                   "SBS_CONFIG (shared)", "sbs_btc.py"],
            "ETH": ["Binance REST", "1H", "15M", "All sessions",
                   "SBS_CONFIG (shared)", "sbs_eth.py"],
        }
        st.dataframe(pd.DataFrame(asset_data).set_index("Property"), use_container_width=True)

        # ── File Map ─────────────────────────────────────────────────
        st.subheader("File Interaction Map")
        st.markdown("""
```
SBS/bots/
├── core/                              # Shared infrastructure
│   ├── bot_base.py (HUB) ──────────→ async orchestration
│   │   ├── data_manager.py            # REST data fetching
│   │   ├── telegram_bot.py            # Async alerts (aiohttp)
│   │   ├── ml_integration_helper.py   # ML wrapper
│   │   ├── ml_data_collector.py       # 70+ feature logging + WFO
│   │   └── daily_telegram_reporter.py # Background scheduler
│   └── __init__.py
├── btc/                               # BTC-specific
│   ├── sbs_btc.py (entry point)
│   ├── sbs_strategy.py                # Full SBS pipeline + pending queue
│   ├── config.py                      # BTC overrides
│   └── wfo_signal_scorer.py           # 5-component scoring (shared)
└── eth/                               # ETH-specific
    ├── sbs_eth.py (entry point)
    ├── sbs_strategy.py                # Full SBS pipeline + pending queue
    ├── config.py                      # ETH overrides
    └── wfo_signal_scorer.py           # 5-component scoring (shared)
```
""")

        # ── Learning ─────────────────────────────────────────────────
        st.subheader("Learning & Adaptation")
        st.markdown("""
SBS focuses on **WFO outcome feedback** as its primary learning mechanism:

1. **WFO Weight Adapter**: After each trade exit, the outcome (win/loss) and component scores
   are fed back to the adapter. Ridge logistic regression retrains weights every 50 outcomes.

2. **ML MFE/MAE Tracking**: Every active trade's Maximum Favorable/Adverse Excursion is updated
   each main loop iteration — providing rich data for post-hoc analysis of optimal SL/TP placement.

3. **Prediction Logger**: When available, reconciles ML predictions with actual outcomes
   (win/loss/breakeven, R-multiple, P&L) for model performance scoring.

4. **Daily/Weekly Reporter**: Background scheduler sends automated performance summaries via
   Telegram, enabling passive monitoring of strategy health.
""")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TAB (preserved from original page)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("Performance Analytics")

    # ── Charts ────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.caption("R-Multiple distribution shows how trade outcomes cluster. Right-skewed = genuine edge.")
        st.plotly_chart(r_multiple_histogram(df), key="dd_r_hist")
    with c2:
        st.caption("Cumulative P&L should trend upward steadily. Sudden drops = drawdown events.")
        st.plotly_chart(cumulative_pnl_line(df), key="dd_cum_pnl")

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Exit reason breakdown reveals how trades end.")
        st.plotly_chart(exit_reason_donut(df), key="dd_exit_donut")
    with c4:
        st.caption("Rolling win rate smooths noise and reveals trends.")
        st.plotly_chart(rolling_win_rate(df), key="dd_rolling_wr")

    # ── MFE/MAE Analysis ─────────────────────────────────────────────
    if not df.empty and "mfe" in df.columns and "mae" in df.columns:
        has_mfe_mae = df["mfe"].notna().sum() >= 5 and df["mae"].notna().sum() >= 5
    else:
        has_mfe_mae = False

    if has_mfe_mae:
        st.markdown("---")
        st.subheader("MFE / MAE Analysis")
        st.caption("Max Favorable Excursion vs Max Adverse Excursion. Points above the diagonal = more upside than downside.")

        col_scatter, col_metrics = st.columns([2, 1])
        with col_scatter:
            st.plotly_chart(mfe_mae_scatter(df), key="dd_mfe_mae")

        with col_metrics:
            valid = df.dropna(subset=["mfe", "mae"])
            avg_mfe = valid["mfe"].abs().mean()
            avg_mae = valid["mae"].abs().mean()

            edge_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0
            st.metric("Edge Ratio", f"{edge_ratio:.2f}",
                      help="Avg MFE / Avg MAE. Above 1.0 = trades move further in your favor.")

            avg_pnl = valid["pnl_usd"].mean()
            entry_eff = avg_pnl / avg_mfe if avg_mfe > 0 else 0
            st.metric("Entry Efficiency", f"{entry_eff:.2f}",
                      help="Avg PnL / Avg MFE. Closer to 1.0 = capturing most of the available move.")

            losers = valid[valid["pnl_usd"] < 0]
            if not losers.empty:
                avg_loss = losers["pnl_usd"].abs().mean()
                avg_mae_losers = losers["mae"].abs().mean()
                stop_eff = avg_loss / avg_mae_losers if avg_mae_losers > 0 else 0
                st.metric("Stop Efficiency", f"{stop_eff:.2f}",
                          help="Avg Loss / Avg MAE on losers. Closer to 1.0 = stops well-placed.")

            st.markdown("---")
            if edge_ratio >= 1.5:
                st.success(f"Edge ratio of {edge_ratio:.2f} is excellent.")
            elif edge_ratio >= 1.0:
                st.info(f"Edge ratio of {edge_ratio:.2f} is positive.")
            else:
                st.warning(f"Edge ratio of {edge_ratio:.2f} is below 1.0 — review entry timing.")

    # ── Regime-Conditional Performance ────────────────────────────────
    if not df.empty and len(df) >= 5:
        st.markdown("---")
        st.subheader("Performance by Market Regime")
        st.caption("Breaks down performance by market condition using ADX and EMA indicators on 1h Binance candles.")

        try:
            from data.binance_helpers import fetch_binance_candles, calculate_indicators, classify_regime

            min_date = df["entry_time"].min()
            max_date = df["entry_time"].max()
            regime_days = max(7, (max_date - min_date).days + 5)

            regime_symbols = df["symbol"].unique()
            regime_sym = regime_symbols[0] if len(regime_symbols) == 1 else "BTC"

            with st.spinner(f"Fetching {regime_sym} 1h candles for regime classification..."):
                candles = fetch_binance_candles(regime_sym, "1h", regime_days)

            if candles is not None and not candles.empty:
                candles = calculate_indicators(candles)
                candles["regime"] = candles.apply(classify_regime, axis=1)

                trade_df = df.copy()
                trade_df = trade_df.sort_values("entry_time")
                candle_regime = candles[["regime"]].copy()
                candle_regime.index = candle_regime.index.tz_localize(None) if candle_regime.index.tz else candle_regime.index

                if trade_df["entry_time"].dt.tz is not None:
                    trade_df["entry_time"] = trade_df["entry_time"].dt.tz_localize(None)

                merged = pd.merge_asof(
                    trade_df.sort_values("entry_time"),
                    candle_regime.reset_index().rename(columns={"Timestamp": "entry_time"}),
                    on="entry_time",
                    direction="backward",
                )

                if "regime" in merged.columns and merged["regime"].notna().sum() > 0:
                    regime_stats = []
                    for regime in ["Trending Up", "Trending Down", "Ranging"]:
                        r_df = merged[merged["regime"] == regime]
                        if r_df.empty:
                            continue
                        n = len(r_df)
                        wr = (r_df["pnl_usd"] > 0).mean()
                        avg_r = r_df["r_multiple"].mean() if "r_multiple" in r_df.columns else 0
                        total_r = r_df["r_multiple"].sum() if "r_multiple" in r_df.columns else 0
                        gp = r_df.loc[r_df["pnl_usd"] > 0, "pnl_usd"].sum()
                        gl = r_df.loc[r_df["pnl_usd"] < 0, "pnl_usd"].abs().sum()
                        pf = gp / gl if gl > 0 else float("inf")
                        regime_stats.append({
                            "Regime": regime,
                            "Trades": n,
                            "Win Rate": f"{wr:.1%}",
                            "Avg R": f"{avg_r:.2f}",
                            "Total R": f"{total_r:.1f}",
                            "Profit Factor": f"{pf:.2f}" if pf < 100 else "Inf",
                        })

                    if regime_stats:
                        st.dataframe(pd.DataFrame(regime_stats), use_container_width=True, hide_index=True)

                        best = max(regime_stats, key=lambda x: float(x["Avg R"]))
                        worst = min(regime_stats, key=lambda x: float(x["Avg R"]))
                        if best["Regime"] != worst["Regime"]:
                            st.markdown(
                                f"- **Best regime:** {best['Regime']} (Avg R: {best['Avg R']}, "
                                f"WR: {best['Win Rate']})\n"
                                f"- **Worst regime:** {worst['Regime']} (Avg R: {worst['Avg R']}, "
                                f"WR: {worst['Win Rate']})"
                            )
                    else:
                        st.info("No regime data could be matched to trades.")
                else:
                    st.info("Could not classify regimes for the trade period.")
            else:
                st.info("Could not fetch Binance data for regime analysis.")
        except Exception as e:
            st.warning(f"Regime analysis unavailable: {e}")

    # ── Long vs Short ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Long vs Short")
    st.caption("Compares directional bias performance.")
    if not df.empty:
        for d in ("Long", "Short"):
            sub = df[df["direction"] == d]
            if sub.empty:
                continue
            wr = (sub["pnl_usd"] > 0).mean()
            avg_r = sub["r_multiple"].mean() if "r_multiple" in sub.columns else 0
            st.write(f"**{d}**: {len(sub)} trades | Win Rate {wr:.1%} | Avg R {avg_r:.2f}")

    # ── Data-Driven Commentary ────────────────────────────────────────
    if not df.empty:
        st.markdown("---")
        st.subheader(f"Analysis & Recommendations for {strategy}")

        total = len(df)
        wins = (df["pnl_usd"] > 0).sum()
        wr = wins / total if total else 0
        total_pnl = df["pnl_usd"].sum()
        avg_r = df["r_multiple"].mean() if "r_multiple" in df.columns else 0

        gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
        gross_loss = df.loc[df["pnl_usd"] < 0, "pnl_usd"].abs().sum()
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        points = []

        if pf == float("inf"):
            points.append("No losing trades recorded — either the sample is too small or risk is masked.")
        elif pf >= 2.0:
            points.append(f"Profit Factor of **{pf:.2f}** is excellent — a strong edge.")
        elif pf >= 1.5:
            points.append(f"Profit Factor of **{pf:.2f}** is solid with a meaningful edge.")
        elif pf >= 1.0:
            points.append(f"Profit Factor of **{pf:.2f}** is marginal — thin edge, watch commissions.")
        else:
            points.append(f"Profit Factor of **{pf:.2f}** is below 1.0 — strategy is losing money.")

        if wr < 0.40 and avg_r > 0.3:
            points.append(f"Low win rate ({wr:.0%}) but positive Avg R ({avg_r:.2f}) — trend-following profile.")
        elif wr < 0.40 and avg_r <= 0:
            points.append(f"Low win rate ({wr:.0%}) + negative Avg R ({avg_r:.2f}) — needs overhaul.")
        elif wr > 0.55 and avg_r < 0:
            points.append(f"High win rate ({wr:.0%}) but negative Avg R ({avg_r:.2f}) — tighten stops.")

        longs = df[df["direction"] == "Long"]
        shorts = df[df["direction"] == "Short"]
        if len(longs) >= 5 and len(shorts) >= 5:
            long_wr = (longs["pnl_usd"] > 0).mean()
            short_wr = (shorts["pnl_usd"] > 0).mean()
            if abs(long_wr - short_wr) > 0.15:
                better = "Longs" if long_wr > short_wr else "Shorts"
                worse = "Shorts" if better == "Longs" else "Longs"
                points.append(f"**{better}** significantly outperform **{worse}** — consider adjusting filters.")

        if "exit_reason" in df.columns:
            exit_counts = df["exit_reason"].value_counts(normalize=True)
            for reason, pct in exit_counts.items():
                if "stop" in str(reason).lower() and pct > 0.5:
                    points.append(f"**{pct:.0%}** exit via stop-loss — stops may be too tight or entries poorly timed.")
                elif "time" in str(reason).lower() and pct > 0.3:
                    points.append(f"**{pct:.0%}** exit by timeout — strategy isn't reaching targets frequently enough.")

        if total < 30:
            points.append(f"Only **{total} trades** — collect at least 30 before making strategy changes.")

        for p in points:
            st.markdown(f"- {p}")
