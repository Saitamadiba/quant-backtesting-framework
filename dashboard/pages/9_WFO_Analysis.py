"""Walk-Forward Optimization Analysis — learn, run, and review WFO backtests."""

import time
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="WFO Analysis", layout="wide")
st.title("Walk-Forward Optimization Analysis")
st.caption(
    "Understand WFO parameters, run backtests on any strategy, and review results. "
    "The gold standard for validating that a trading edge is real — not curve-fitted."
)

# Ensure imports
_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from config import STRATEGY_COLORS, STRATEGIES, RESEARCH_STRATEGIES
from components.charts import REGIME_COLORS
from data.wfo_loader import (
    list_wfo_results, load_wfo_result, get_latest_wfo,
    delete_wfo_result, delete_all_wfo_results,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_ci(ci):
    """Format a [lower, upper] confidence interval."""
    if isinstance(ci, (list, tuple)) and len(ci) >= 2:
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    return "N/A"


def _bget(d, *keys, default=0):
    """Nested get for bayesian_edge dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def _fmt_time(seconds):
    """Format seconds as human-readable duration."""
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# ── Strategy name mapping ─────────────────────────────────────────────────────
# STRATEGIES config uses display names ("Liquidity Raid"), adapters use code
# names ("LiquidityRaid").  These maps bridge the two.
_STRATEGY_TO_ADAPTER = {
    "FVG": "FVG",
    "Liquidity Raid": "LiquidityRaid",
    "Momentum Mastery": "MomentumMastery",
    "SBS": "SBS",
}
_ADAPTER_TO_STRATEGY = {v: k for k, v in _STRATEGY_TO_ADAPTER.items()}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A: What Is Walk-Forward Optimization?
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("**What is Walk-Forward Optimization? — and Why It Matters**", expanded=False):
    st.markdown("""
### The Problem with Traditional Backtesting

A traditional backtest optimizes parameters on the **same** data it tests on. This is like memorizing
the answer key before an exam — you'll score perfectly on that exact test but fail miserably on a new
one. In trading, this is called **overfitting**: the strategy learned the noise, not the signal.

### How WFO Solves This

Walk-Forward Optimization splits your data into alternating **in-sample (IS)** and **out-of-sample
(OOS)** windows:

1. **Train** on the IS window — find the best parameters
2. **Test** on the next OOS window — see how those parameters perform on *unseen* data
3. **Slide forward** and repeat across the entire dataset

The OOS results are stitched together to form a **composite equity curve** that represents
how your strategy would have performed in a truly forward-looking way.

### The Dress Rehearsal Metaphor

Think of WFO as a series of dress rehearsals for live trading. Each window is a performance cycle:
you rehearse (IS) then perform in front of an audience (OOS). If you deliver great rehearsals but
bomb every show, the strategy is overfitted. If the shows are consistently decent — even if not as
polished as rehearsal — you have a **real edge**.

### Key Diagnostic Metrics

| Metric | What It Tells You | Good Value |
|--------|-------------------|------------|
| **Overfit Ratio** | IS performance / OOS performance. High = overfit | < 2.0 |
| **PBO** | Probability of Backtest Overfitting. Statistical test | < 50% |
| **P(Edge > 0)** | Bayesian probability that expectancy is positive | > 70% |
| **Param Stability** | How much optimal params change across windows | > 0.7 |
""")

    # Visual: IS/OOS sliding window diagram
    fig_wfo = go.Figure()
    n_windows = 5
    for i in range(n_windows):
        x_is_start = i * 2
        x_is_end = x_is_start + 3
        x_oos_start = x_is_end
        x_oos_end = x_oos_start + 1

        fig_wfo.add_shape(
            type="rect", x0=x_is_start, x1=x_is_end, y0=i, y1=i + 0.7,
            fillcolor="rgba(33, 150, 243, 0.6)", line=dict(color="#2196F3"),
        )
        fig_wfo.add_annotation(
            x=(x_is_start + x_is_end) / 2, y=i + 0.35,
            text=f"Train (IS)", showarrow=False, font=dict(color="white", size=11),
        )
        fig_wfo.add_shape(
            type="rect", x0=x_oos_start, x1=x_oos_end, y0=i, y1=i + 0.7,
            fillcolor="rgba(76, 175, 80, 0.6)", line=dict(color="#4CAF50"),
        )
        fig_wfo.add_annotation(
            x=(x_oos_start + x_oos_end) / 2, y=i + 0.35,
            text="OOS", showarrow=False, font=dict(color="white", size=11),
        )

    fig_wfo.update_layout(
        template="plotly_dark", height=280,
        xaxis=dict(title="Time (bars)", showgrid=False),
        yaxis=dict(title="Window", tickvals=list(range(n_windows)),
                   ticktext=[f"W{i+1}" for i in range(n_windows)]),
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_wfo, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B: Parameter Guide — Why Each One Matters
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("**Parameter Guide — Why Each Setting Matters**", expanded=False):
    st.markdown("""
Every WFO parameter controls a trade-off. Understanding them is the difference between a rigorous
validation and a false sense of security.
""")

    params_data = [
        {
            "Parameter": "Train Window (IS bars)",
            "Technical": "Number of bars used for in-sample parameter optimization. "
                         "Default: 500 (4h) = ~83 days. Auto-scales by timeframe.",
            "Plain English": "How much history the optimizer sees when choosing the best parameters.",
            "Metaphor": "The textbook you study from — too thin and you miss important patterns, "
                        "too thick and you memorize noise instead of learning principles.",
            "Trade-off": "Larger = more stable but slower to adapt. Smaller = adapts faster but noisier.",
        },
        {
            "Parameter": "Test Window (OOS bars)",
            "Technical": "Number of bars for out-of-sample validation per window. "
                         "Default: 100 (4h) = ~17 days.",
            "Plain English": "How long each 'exam' lasts before the optimizer re-tunes.",
            "Metaphor": "The exam duration — too short gives noisy grades that fluctuate wildly, "
                        "too long means you're using stale study material.",
            "Trade-off": "Larger OOS = more reliable per-window stats but fewer windows total.",
        },
        {
            "Parameter": "Step Size",
            "Technical": "How many bars the window advances each iteration. "
                         "Controls overlap between consecutive OOS windows.",
            "Plain English": "The gap between re-optimizations — how often you update your playbook.",
            "Metaphor": "Like updating your GPS route — recalculate every mile vs every 10 miles. "
                        "Too frequent wastes effort; too infrequent misses turns.",
            "Trade-off": "Smaller step = more windows (better stats) but more overlap.",
        },
        {
            "Parameter": "Anchored Mode",
            "Technical": "If True, IS window always starts from bar 0 (expanding). "
                         "If False, IS window rolls forward (fixed width).",
            "Plain English": "Whether you study all history or just recent history.",
            "Metaphor": "An anchored investor builds on all experience — like a veteran coach. "
                        "A rolling investor focuses on the last season — like a day trader. "
                        "Crypto regimes shift hard, so anchored often wins. Futures evolve faster.",
            "Trade-off": "Anchored = more data but older data may be irrelevant. Rolling = fresher but noisier.",
        },
        {
            "Parameter": "Optimization Metric",
            "Technical": "Target function for IS optimization: expectancy, profit_factor, sharpe, or total_r.",
            "Plain English": "What the optimizer tries to maximize on each training window.",
            "Metaphor": "Your report card criteria — optimizing for Sharpe is like aiming for "
                        "consistent B+ grades. Optimizing total_r is going for an A+ but risking an F.",
            "Trade-off": "Expectancy balances win rate and payoff. Sharpe penalizes variance. "
                         "Total R maximizes raw profit. Profit Factor = gross profit / gross loss.",
        },
        {
            "Parameter": "Regime Adaptive",
            "Technical": "Uses HMM to classify market into regimes (trending, ranging, volatile). "
                         "Optimizes separate parameters per regime during IS.",
            "Plain English": "Different playbooks for different market moods — trending markets "
                             "need different settings than choppy ones.",
            "Metaphor": "A football team with different plays for offense, defense, and special "
                        "teams. One-size-fits-all works in calm markets, but adaptive mode thrives "
                        "when conditions change.",
            "Trade-off": "More flexible but needs more data per regime. Risk: overfitting to regime labels.",
        },
        {
            "Parameter": "Transaction Costs",
            "Technical": "Spread + commission + slippage per trade. "
                         "Crypto: ~0.065% round trip. Futures (NQ): ~0.04%.",
            "Plain English": "The invisible tax on every trade — the house edge.",
            "Metaphor": "Like friction in a machine. A strategy that looks profitable in a "
                        "frictionless world may bleed money once you account for the real cost "
                        "of entering and exiting positions.",
            "Trade-off": "Higher costs = fewer viable strategies. Underestimating costs = false edge.",
        },
    ]

    st.dataframe(
        pd.DataFrame(params_data),
        use_container_width=True, hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
            "Technical": st.column_config.TextColumn("Technical", width="large"),
            "Plain English": st.column_config.TextColumn("Plain English", width="large"),
            "Metaphor": st.column_config.TextColumn("Metaphor", width="large"),
            "Trade-off": st.column_config.TextColumn("Trade-off", width="large"),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C: Quant Industry Standards by Asset Type
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("**Quant Industry Standards by Asset Type**", expanded=False):
    crypto_col, futures_col = st.columns(2)

    with crypto_col:
        st.markdown("""
### Crypto (BTC, ETH)

**Window Configuration:**
- **Train**: 500-1000 bars (4h), **anchored** preferred
- **OOS**: 100-200 bars (4h) = 17-33 days
- **Scaling**: 15m uses 16x multiplier (8000 IS / 1600 OOS bars)
- Anchored wins because crypto regime shifts are structural (halving cycles, regulatory shocks)

**Transaction Costs:**
| Component | Value |
|-----------|-------|
| Spread | 0.01% |
| Commission | 0.055% |
| Slippage | 0.01% |
| **Round trip** | **~0.065%** |

**Optimization Target:**
- **Expectancy** (recommended) — balances win rate and payoff asymmetry
- Crypto strategies often have low win rates (25-40%) but high R:R (2-4x)
- Expectancy = (Win% x Avg Win) - (Loss% x Avg Loss) captures this correctly

**Minimum Standards:**
- Min 20 OOS trades per window
- Min 50 total OOS trades across all windows
- Regime-adaptive mode recommended (crypto has pronounced trending/ranging cycles)

**Reference:** Pardo (2008) *The Evaluation and Optimization of Trading Strategies* recommends
60-80% IS / 20-40% OOS ratio — our default 500/100 (83%/17%) is within range.
""")

    with futures_col:
        st.markdown("""
### Equity Futures (NQ)

**Window Configuration:**
- **Train**: 500 bars (4h), **rolling** preferred
- **OOS**: 100 bars (4h) = 17 days
- Rolling wins because equity microstructure evolves faster (market makers adapt, algos update)
- Session-aware: most signals fire during NY (13:30-20:00 UTC)

**Transaction Costs:**
| Component | Value |
|-----------|-------|
| Spread | 0.01% |
| Commission | 0.02% |
| Slippage | 0.01% |
| **Round trip** | **~0.04%** |

**Optimization Target:**
- **Sharpe** (institutional standard) — risk-adjusted return is king
- Futures strategies typically have higher win rates (40-55%) but tighter R:R (1.5-2.5x)
- Sharpe penalizes erratic returns, favoring consistent strategies

**Minimum Standards:**
- Min 15 OOS trades per window
- Min 50 total OOS trades
- Macro event sensitivity: Fed days, FOMC, NFP can invalidate entire windows
- Consider excluding high-impact event days from optimization

**Reference:** Bailey & Lopez de Prado (2014) *Probability of Backtest Overfitting* —
PBO < 50% is the institutional acceptance threshold. Our Bayesian edge estimator provides
P(Edge > 0) as a complementary significance test.
""")

    st.markdown("---")
    st.markdown("""
### Optimum WFO Settings by Timeframe

The WFO engine **auto-scales** window sizes to preserve ~83 calendar days of in-sample
and ~17 calendar days of out-of-sample data regardless of timeframe. The bar counts
change, but the economic exposure per window stays constant.
""")

    tf_crypto, tf_futures = st.columns(2)

    with tf_crypto:
        st.markdown("""
#### Crypto (BTC, ETH) — Anchored + Expectancy

| Timeframe | IS Bars | OOS Bars | Step | ≈ IS Days | ≈ OOS Days | Min Data |
|-----------|---------|----------|------|-----------|------------|----------|
| **4h**    | 500     | 100      | 100  | 83 days   | 17 days    | 250 days |
| **1h**    | 2,000   | 400      | 400  | 83 days   | 17 days    | 250 days |
| **15m**   | 8,000   | 1,600    | 1,600| 83 days   | 17 days    | 250 days |
| **5m**    | 24,000  | 4,800    | 4,800| 83 days   | 17 days    | 250 days |

- **4h** — Best for swing/position strategies (FVG, SBS). Most stable parameters.
- **1h** — Balanced for intraday strategies. Good signal-to-noise ratio.
- **15m** — Scalping territory (Liquidity Raid). More trades, more noise.
- **5m** — HFT-lite. Requires massive data (250+ days at 288 bars/day). Use with caution.
- **Anchored mode** recommended — crypto regime shifts are structural (halving, regulations).
- **Expectancy metric** recommended — captures asymmetric R:R typical of crypto strategies.
""")

    with tf_futures:
        st.markdown("""
#### Equity Futures (NQ) — Rolling + Sharpe

| Timeframe | IS Bars | OOS Bars | Step | ≈ IS Days | ≈ OOS Days | Min Data |
|-----------|---------|----------|------|-----------|------------|----------|
| **4h**    | 500     | 100      | 100  | 87 days   | 17 days    | 260 days |
| **1h**    | 2,000   | 400      | 400  | 87 days   | 17 days    | 260 days |
| **15m**   | 8,000   | 1,600    | 1,600| 87 days   | 17 days    | 260 days |
| **5m**    | 24,000  | 4,800    | 4,800| 87 days   | 17 days    | 260 days |

- **4h** — Best for macro/swing strategies on NQ. Fewer bars/day (~5.75) than crypto.
- **1h** — Standard institutional timeframe. NY session captures most signals.
- **15m** — Active intraday. Good for Liquidity Raid / Momentum Mastery on NQ.
- **5m** — Micro-scalping. Very sensitive to spread and slippage costs.
- **Rolling mode** recommended — equity microstructure evolves faster (algos adapt, MM update).
- **Sharpe metric** recommended — institutional standard for risk-adjusted return.
""")

    st.markdown("---")
    st.markdown("""
### Universal Quality Gates

These thresholds apply regardless of asset type:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| **PBO** | < 50% | Less than coin-flip chance the backtest is overfitted |
| **Overfit Ratio** | < 2.0 | IS performance is no more than 2x OOS performance |
| **P(Edge > 0)** | > 70% | Bayesian posterior probability of positive expectancy |
| **Parameter Stability** | > 0.7 | Optimal params don't wildly shift between windows |
| **Min OOS Trades** | > 50 | Enough data for statistical significance |
| **Min Windows** | > 10 | Enough windows for cross-validation reliability |
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D: Run WFO Backtest
# ══════════════════════════════════════════════════════════════════════════════

st.header("Run WFO Backtest")

# ── Sidebar: Strategy / Symbol / Timeframe ────────────────────────────────────
from backtrader_framework.optimization.strategy_adapters import ADAPTER_REGISTRY

st.sidebar.header("WFO Configuration")

# Use display names from STRATEGIES (matches running bots)
strategy_name = st.sidebar.selectbox(
    "Strategy", list(RESEARCH_STRATEGIES.keys()), key="wfo_strategy",
    help="The trading strategy to optimize. Each strategy has its own "
         "parameter space and signal logic. Choose the one running on your "
         "target asset.",
)

# Symbols available for the selected strategy
_all_symbols = STRATEGIES[strategy_name]["symbols"]
symbol = st.sidebar.selectbox(
    "Symbol", _all_symbols, key="wfo_symbol",
    help="Asset to backtest on. Different assets have different transaction "
         "costs, volatility profiles, and data availability. BTC/ETH are "
         "24/7 crypto; NQ is equity futures with session hours.",
)

# Resolve adapter from display name
_adapter_key = _STRATEGY_TO_ADAPTER.get(strategy_name, strategy_name)
_adapter_cls = ADAPTER_REGISTRY.get(_adapter_key)
_adapter = _adapter_cls() if _adapter_cls else None
_default_tfs = _adapter.default_timeframes if _adapter else ["4h"]
_all_tfs = ["5m", "15m", "1h", "4h"]
_default_idx = _all_tfs.index(_default_tfs[0]) if _default_tfs[0] in _all_tfs else 0

timeframe = st.sidebar.selectbox(
    "Timeframe", _all_tfs, index=_default_idx, key="wfo_timeframe",
    help=f"Candlestick period for the backtest. Default for {strategy_name}: "
         f"{', '.join(_default_tfs)}. Smaller timeframes (5m) generate more "
         f"trades but need far more data and compute time. 4h is the most "
         f"stable; 15m is good for scalping strategies.",
)

# ── Advanced Settings ──────────────────────────────────────────────────────────
from backtrader_framework.optimization.wfo_engine import WFOConfig as _WFOCfg

_tf_defaults = _WFOCfg.for_timeframe(timeframe)

with st.expander("Advanced WFO Settings"):
    adv_c1, adv_c2, adv_c3 = st.columns(3)
    with adv_c1:
        train_bars = st.number_input(
            "IS Window (bars)", min_value=200, max_value=80000,
            value=_tf_defaults.train_window_bars, step=100, key="wfo_train",
            help=f"In-sample training window: {_tf_defaults.train_window_bars} bars "
                 f"for {timeframe} (~83 calendar days). The optimizer searches for "
                 f"the best parameters on this data. Larger = more stable but "
                 f"slower to adapt. Smaller = adapts faster but noisier.",
        )
        test_bars = st.number_input(
            "OOS Window (bars)", min_value=50, max_value=16000,
            value=_tf_defaults.test_window_bars, step=50, key="wfo_test",
            help=f"Out-of-sample test window: {_tf_defaults.test_window_bars} bars "
                 f"for {timeframe} (~17 calendar days). Parameters found in IS are "
                 f"tested here on unseen data. Larger = more reliable per-window "
                 f"stats but fewer total windows.",
        )
    with adv_c2:
        step_bars = st.number_input(
            "Step Size (bars)", min_value=50, max_value=8000,
            value=_tf_defaults.step_bars, step=50, key="wfo_step",
            help=f"How far the window slides forward each iteration: "
                 f"{_tf_defaults.step_bars} bars for {timeframe}. Smaller step = "
                 f"more overlapping windows (better statistics, longer runtime). "
                 f"Larger step = fewer windows (faster, but less reliable).",
        )
        _is_crypto = symbol in ("BTC", "ETH")
        anchored = st.checkbox(
            "Anchored (expanding window)",
            value=_is_crypto,  # True for crypto, False for futures
            key="wfo_anchored",
            help="Anchored: IS window always starts from bar 0 and expands "
                 "over time, using all available history. Best for crypto where "
                 "regime shifts are structural. Rolling: IS window has fixed "
                 "width and slides forward, focusing on recent data. Best for "
                 "futures where market microstructure evolves faster.",
        )
    with adv_c3:
        _metrics = ["expectancy", "profit_factor", "sharpe", "total_r"]
        _default_metric_idx = 0 if _is_crypto else 2  # expectancy for crypto, sharpe for futures
        metric = st.selectbox(
            "Optimization Metric",
            _metrics,
            index=_default_metric_idx,
            key="wfo_metric",
            help="The objective function the optimizer maximizes on each IS window. "
                 "Expectancy = (Win% x AvgWin) - (Loss% x AvgLoss), best for "
                 "crypto's asymmetric R:R. Sharpe = risk-adjusted return, the "
                 "institutional standard for futures. Profit Factor = gross "
                 "profit / gross loss. Total R = raw cumulative R-multiple.",
        )
        max_combos = st.number_input(
            "Max Param Combos", min_value=100, max_value=5000,
            value=1000, step=100, key="wfo_max_combos",
            help="Upper limit on parameter combinations tested per window. "
                 "The engine uses random Sobol sampling (150 combos by default) "
                 "for efficiency. This cap applies when using full grid search "
                 "mode. Higher = more thorough but slower. 1000 is a safe default; "
                 "reduce to 500 for faster runs or increase to 3000 for exhaustive search.",
        )

regime_adaptive = st.checkbox(
    "Regime-Adaptive Mode",
    value=False,
    key="wfo_regime_adaptive",
    help="When enabled, the engine uses a Hidden Markov Model (HMM) to classify "
         "each window into regimes (trending up, trending down, ranging, volatile) "
         "and optimizes separate parameters for each regime. This lets the strategy "
         "use aggressive settings in trends and conservative settings in chop. "
         "Recommended for crypto (pronounced regime cycles). Requires more data "
         "per regime to avoid overfitting to regime labels.",
)

# ── Pre-run Workload Estimate ─────────────────────────────────────────────────
_TF_SCALE_EST = {'5m': 48, '15m': 16, '30m': 8, '1h': 4, '2h': 2, '4h': 1}
_scale_est = _TF_SCALE_EST.get(timeframe, 1)
_embargo_est = 168 * _scale_est
_bars_per_window_est = train_bars + _embargo_est + test_bars
_BARS_PER_DAY = {'5m': 288, '15m': 96, '1h': 24, '4h': 6}
_bpd = _BARS_PER_DAY.get(timeframe, 24)
_is_crypto = symbol in ("BTC", "ETH")
_est_data_days = 548 if _is_crypto else 365  # ~1.5yr crypto, ~1yr futures
_est_total_bars = int(_bpd * _est_data_days)
_est_windows = max(0, (_est_total_bars - _bars_per_window_est) // step_bars)
_est_combos = 150  # random Sobol grid default
_est_total_runs = _est_windows * _est_combos

if _est_windows > 0:
    st.info(
        f"**Estimated workload** (assuming ~{_est_data_days / 365:.1f} years of {timeframe} data):  \n"
        f"~**{_est_windows}** windows x **{_est_combos}** param combos = "
        f"**{_est_total_runs:,}** backtests  \n"
        f"Actual ETA will be displayed once the run starts and the first windows complete."
    )
else:
    st.warning(
        f"With {train_bars:,} IS + {_embargo_est:,} embargo + {test_bars:,} OOS = "
        f"**{_bars_per_window_est:,}** bars per window, you need at least "
        f"**{_bars_per_window_est + step_bars:,}** bars of {timeframe} data for {symbol}."
    )

# ── Run Button ─────────────────────────────────────────────────────────────────
if st.button("Run WFO", type="primary", key="wfo_run"):
    from backtrader_framework.optimization.wfo_engine import (
        WFOEngine, WFOConfig, TransactionCosts, RegimeAdaptiveWFO,
    )
    from backtrader_framework.optimization.persistence import save_wfo_result

    if not _adapter:
        st.error(f"No adapter for strategy: {strategy_name}")
    else:
        config = WFOConfig(
            train_window_bars=train_bars,
            test_window_bars=test_bars,
            step_bars=step_bars,
            anchored=anchored,
            optimization_metric=metric,
            max_param_combos=max_combos,
            costs=TransactionCosts.for_asset(symbol),
        )

        if regime_adaptive:
            engine = RegimeAdaptiveWFO(_adapter, config)
        else:
            engine = WFOEngine(_adapter, config)

        mode_label = "Regime-Adaptive WFO" if regime_adaptive else "WFO"
        with st.status(f"Running {mode_label}: {strategy_name} on {symbol} {timeframe}...",
                       expanded=True) as status:
            progress_bar = st.progress(0.0)
            log_area = st.empty()
            eta_area = st.empty()

            t0 = time.time()

            def _progress_cb(pct, msg):
                progress_bar.progress(min(pct, 1.0))
                elapsed = time.time() - t0
                elapsed_str = _fmt_time(elapsed)
                if pct > 0.12:  # Window optimization phase (0.10-0.90)
                    window_pct = (pct - 0.10) / 0.80
                    if window_pct > 0.01:
                        total_est = elapsed / window_pct
                        remaining = max(0, total_est - elapsed)
                        remaining_str = _fmt_time(remaining)
                        log_area.caption(msg)
                        eta_area.markdown(
                            f"**Elapsed:** {elapsed_str} &nbsp;|&nbsp; "
                            f"**Remaining:** ~{remaining_str} &nbsp;|&nbsp; "
                            f"**Progress:** {pct:.0%}"
                        )
                    else:
                        log_area.caption(f"{msg}  |  Elapsed: {elapsed_str}")
                        eta_area.empty()
                else:
                    log_area.caption(f"{msg}  |  Elapsed: {elapsed_str}")
                    eta_area.empty()
            if regime_adaptive:
                result = engine.run(symbol, timeframe, progress_callback=_progress_cb,
                                    run_standard=True)
            else:
                result = engine.run(symbol, timeframe, progress_callback=_progress_cb)
            elapsed = time.time() - t0

            if result.get('error'):
                status.update(label=f"Failed: {result['error']}", state="error")
                st.error(f"WFO failed: {result['error']}")
            else:
                filepath = save_wfo_result(result)
                status.update(
                    label=f"Done in {_fmt_time(elapsed)} — "
                          f"{result.get('oos_n_trades', 0)} OOS trades across "
                          f"{result.get('n_windows', 0)} windows",
                    state="complete",
                )
                eta_area.empty()
                st.session_state["wfo_page_result"] = result
                # Clear cache so the new result shows in the list
                list_wfo_results.clear()
                load_wfo_result.clear()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E: Results Viewer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("Results")

results = list_wfo_results()

tab_list, tab_analysis = st.tabs(["Saved Results", "Analysis"])

# ── Tab 1: Saved Results List ─────────────────────────────────────────────────
with tab_list:
    if not results:
        st.info("No WFO results yet. Run a backtest above or from the ML Training page.")
    else:
        idx_rows = []
        for i, r in enumerate(results):
            idx_rows.append({
                "": i,
                "Strategy": r["strategy"],
                "Symbol": r["symbol"],
                "Timeframe": r["timeframe"],
                "Timestamp": r["timestamp"],
            })
        st.dataframe(
            pd.DataFrame(idx_rows).drop(columns=[""]),
            use_container_width=True, hide_index=True,
        )

        _sel_c1, _sel_c2 = st.columns([4, 1])
        with _sel_c1:
            sel_idx = st.selectbox(
                "Select result to view",
                range(len(results)),
                format_func=lambda i: (
                    f"{results[i]['strategy']} / {results[i]['symbol']} / "
                    f"{results[i]['timeframe']} — {results[i]['timestamp']}"
                ),
                key="wfo_result_select",
            )
        with _sel_c2:
            st.write("")
            if st.button("Load", key="wfo_load_result"):
                st.session_state["wfo_page_result"] = load_wfo_result(results[sel_idx]["path"])
                st.rerun()

        # Delete controls
        _del_c1, _del_c2, _del_c3 = st.columns([1, 1, 2])
        with _del_c1:
            if st.button("Delete Selected", key="wfo_del_selected"):
                if delete_wfo_result(results[sel_idx]["path"]):
                    st.toast("Deleted.")
                    st.rerun()
        with _del_c2:
            if st.button("Delete All", key="wfo_del_all"):
                st.session_state["wfo_confirm_del_all"] = True

        if st.session_state.get("wfo_confirm_del_all"):
            st.warning(f"Delete all {len(results)} WFO results?")
            _cc1, _cc2 = st.columns(2)
            if _cc1.button("Confirm", key="wfo_del_confirm"):
                n = delete_all_wfo_results()
                st.session_state["wfo_confirm_del_all"] = False
                st.toast(f"Deleted {n} files.")
                st.rerun()
            if _cc2.button("Cancel", key="wfo_del_cancel"):
                st.session_state["wfo_confirm_del_all"] = False
                st.rerun()


# ── Tab 2: Full Analysis ──────────────────────────────────────────────────────
with tab_analysis:
    data = st.session_state.get("wfo_page_result")

    if not data and results:
        # Auto-load first result if none selected
        data = load_wfo_result(results[0]["path"])

    if not data:
        st.info("Select a result from the Saved Results tab or run a new backtest.")
    else:
        _strat_name_raw = data.get("strategy_name", "?")
        _strat_display = _ADAPTER_TO_STRATEGY.get(_strat_name_raw, _strat_name_raw)
        _sym = data.get("symbol", "?")
        _tf = data.get("timeframe", "?")
        _strat_color = STRATEGY_COLORS.get(_strat_display, "#888")
        st.markdown(
            f'<h2 style="margin-bottom:0.2em">'
            f'<span style="color:{_strat_color}">{_strat_display}</span>'
            f' &mdash; {_sym} / {_tf}</h2>',
            unsafe_allow_html=True,
        )

        oos = data.get("oos_stats", {})
        bayesian = data.get("bayesian_edge", {})
        mc = data.get("monte_carlo", {})
        windows = data.get("windows", [])

        # ── Header Metrics ────────────────────────────────────────────────
        st.caption(
            "These are the headline diagnostics from the out-of-sample (OOS) "
            "windows — the data the optimizer never saw during training. "
            "Green deltas are good; red deltas flag potential problems."
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(
            "OOS Win Rate", f"{oos.get('win_rate', 0):.1%}",
            help="Percentage of OOS trades that were profitable. "
                 "Crypto strategies: 25-45% is typical with high R:R. "
                 "Futures: 40-55% is typical with tighter R:R.",
        )
        c2.metric(
            "OOS Sharpe", f"{oos.get('sharpe_per_trade', 0):.2f}",
            help="Risk-adjusted return per trade on unseen data. "
                 "> 0.5 is decent, > 1.0 is strong, > 2.0 is exceptional. "
                 "Negative = the strategy lost money OOS.",
        )
        c3.metric(
            "Overfit Ratio",
            f"{data.get('overfit_ratio', 0):.2f}",
            delta="Good" if data.get("overfit_ratio", 999) < 2.0 else "High",
            delta_color="normal" if data.get("overfit_ratio", 999) < 2.0 else "inverse",
            help="IS performance / OOS performance. Measures how much "
                 "the strategy degrades on unseen data. "
                 "< 2.0 is acceptable. > 3.0 means heavy overfitting — "
                 "the strategy memorized noise instead of learning signal.",
        )
        _pbo_raw = data.get("pbo", 0)
        _pbo_val = _pbo_raw.get("pbo", 0) if isinstance(_pbo_raw, dict) else _pbo_raw
        c4.metric(
            "PBO",
            f"{_pbo_val:.1%}",
            delta="Low" if _pbo_val < 0.5 else "High",
            delta_color="normal" if _pbo_val < 0.5 else "inverse",
            help="Probability of Backtest Overfitting (Bailey & Lopez de Prado). "
                 "The chance that the best in-sample param set is actually a loser OOS. "
                 "< 50% is the institutional acceptance threshold. "
                 "< 25% is strong. > 60% means the optimization is likely worthless.",
        )
        _p_edge = _bget(bayesian, "expectancy", "p_positive", default=None)
        c5.metric(
            "P(Edge > 0)",
            f"{_p_edge:.1%}" if _p_edge is not None else "N/A",
            help="Bayesian posterior probability that the strategy has positive "
                 "expectancy (wins more than it loses, on average). "
                 "> 70% is encouraging, > 90% is strong, > 95% is very strong. "
                 "< 50% means the edge is probably not real.",
        )
        c6.metric(
            "OOS Trades", f"{oos.get('n_trades', 0)}",
            help="Total trades executed across all OOS windows. "
                 "Need at least 50 for statistical significance, "
                 "100+ for reliable conclusions. Fewer trades = wider confidence intervals.",
        )

        # ── Parameter Stability Heatmap ───────────────────────────────────
        st.markdown("---")
        st.subheader("Parameter Stability Across WFO Windows")
        st.caption(
            "Each column is a WFO window; each row is a strategy parameter. "
            "**Uniform horizontal bands** (same color across windows) mean the "
            "optimizer consistently picks similar values — a sign of robust signal. "
            "**Patchy, random-looking rows** mean the parameter is unstable and "
            "the optimizer is chasing noise. Stability > 0.7 is the institutional target."
        )

        if windows:
            param_keys = sorted(
                {k for w in windows if w.get("best_params") for k in w["best_params"]}
            )
            if param_keys:
                heatmap_data = []
                window_labels = []
                for w in windows:
                    bp = w.get("best_params", {})
                    heatmap_data.append([bp.get(k, np.nan) for k in param_keys])
                    window_labels.append(f"W{w.get('id', '?')}")

                arr = np.array(heatmap_data, dtype=float).T
                arr_norm = arr.copy()
                for i in range(arr_norm.shape[0]):
                    row = arr_norm[i]
                    rmin, rmax = np.nanmin(row), np.nanmax(row)
                    if rmax > rmin:
                        arr_norm[i] = (row - rmin) / (rmax - rmin)
                    else:
                        arr_norm[i] = 0.5

                text_vals = [[f"{arr[i, j]:.4g}" for j in range(arr.shape[1])]
                             for i in range(arr.shape[0])]

                fig_hm = go.Figure(data=go.Heatmap(
                    z=arr_norm, x=window_labels, y=param_keys,
                    text=text_vals, texttemplate="%{text}",
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Normalized"),
                ))
                fig_hm.update_layout(
                    template="plotly_dark",
                    height=max(250, len(param_keys) * 50),
                    xaxis_title="WFO Window", yaxis_title="Parameter",
                )
                st.plotly_chart(fig_hm, use_container_width=True)

                stability = data.get("param_stability")
                if isinstance(stability, dict):
                    mean_stab = stability.get("mean_stability", 0)
                    if mean_stab != float("-inf") and mean_stab != float("inf"):
                        st.info(
                            f"Parameter stability: **{mean_stab:.3f}** mean "
                            f"({stability.get('fragile_windows', 0)} fragile windows "
                            f"out of {stability.get('n_windows', 0)})"
                        )
                    else:
                        st.info(
                            f"Parameter stability: {stability.get('fragile_windows', 0)} "
                            f"fragile windows out of {stability.get('n_windows', 0)}"
                        )
                elif stability is not None:
                    st.info(f"Parameter stability score: **{stability:.3f}**")
            else:
                st.info("No parameter data in WFO windows.")
        else:
            st.info("No WFO windows found.")

        # ── Cross-Asset Parameter Comparison ──────────────────────────────
        st.markdown("---")
        st.subheader("Cross-Asset Parameter Comparison")
        st.caption(
            f"Strategy: {_strat_display} | Timeframe: {_tf}  \n"
            "Compares the most recent optimal parameters across all symbols this "
            "strategy trades. **Similar values across assets** suggest the strategy "
            "captures a universal market pattern (structural edge). **Divergent values** "
            "mean the strategy adapts to each asset's microstructure — valid, but "
            "harder to generalize. If one asset has wildly different params, question "
            "whether the strategy truly works on that asset or is just curve-fitted."
        )

        all_symbols = STRATEGIES.get(_strat_display, {}).get("symbols", [_sym])
        cross_data = {}
        for sym_cmp in all_symbols:
            r = get_latest_wfo(_strat_name_raw, sym_cmp, _tf)
            if r and r.get("windows"):
                last_w = r["windows"][-1]
                cross_data[sym_cmp] = last_w.get("best_params", {})

        if len(cross_data) > 1:
            all_params = sorted({k for bp in cross_data.values() for k in bp})
            rows = []
            for sym_cmp, bp in cross_data.items():
                for p in all_params:
                    rows.append({"Symbol": sym_cmp, "Parameter": p, "Value": bp.get(p, 0)})
            df_cross = pd.DataFrame(rows)

            symbol_colors = {"BTC": "#F7931A", "ETH": "#627EEA", "NQ": "#00C853"}
            fig_cross = px.bar(
                df_cross, x="Parameter", y="Value", color="Symbol",
                barmode="group", template="plotly_dark",
            )
            for trace in fig_cross.data:
                if trace.name in symbol_colors:
                    trace.marker.color = symbol_colors[trace.name]
            fig_cross.update_layout(height=400)
            st.plotly_chart(fig_cross, use_container_width=True)

            for p in all_params:
                vals = {s: bp.get(p) for s, bp in cross_data.items() if bp.get(p) is not None}
                if len(vals) > 1:
                    items = [f"{s}: {v:.4g}" for s, v in vals.items()]
                    st.caption(f"**{p}** — " + " | ".join(items))
        elif len(cross_data) == 1:
            st.info(f"Only {list(cross_data.keys())[0]} has results for {_strat_display}/{_tf}.")
        else:
            st.info(f"No WFO results found for other symbols with {_strat_display}/{_tf}.")

        # ── Regime Performance ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Regime Performance Breakdown")
        st.caption(
            "How the strategy performs in each HMM-detected market regime. "
            "The **Win Rate** chart shows reliability; the **Expectancy** chart "
            "shows profitability per trade in R-multiples.  \n"
            "A strong strategy has positive expectancy in most regimes. "
            "If it only works in one regime (e.g., Trending Up) but loses in "
            "others, consider pairing it with a regime filter or using "
            "regime-adaptive mode to switch parameters automatically."
        )

        regime_data = data.get("regime_analysis", {})
        if regime_data:
            regime_rows = []
            for regime_name, stats in regime_data.items():
                if isinstance(stats, dict):
                    regime_rows.append({
                        "Regime": regime_name.replace("_", " ").title(),
                        "Win Rate": stats.get("win_rate", 0),
                        "Expectancy": stats.get("mean_r", stats.get("expectancy", 0)),
                        "N Trades": stats.get("n_trades", 0),
                    })

            if regime_rows:
                df_regime = pd.DataFrame(regime_rows)
                regime_colors_map = {
                    "Trending Up": "rgba(76, 175, 80, 0.8)",
                    "Trending Down": "rgba(244, 67, 54, 0.8)",
                    "Ranging": "rgba(158, 158, 158, 0.8)",
                    "Volatile": "rgba(255, 152, 0, 0.8)",
                }
                colors = [regime_colors_map.get(r, "#888") for r in df_regime["Regime"]]

                fig_reg = go.Figure()
                fig_reg.add_trace(go.Bar(
                    x=df_regime["Regime"], y=df_regime["Win Rate"],
                    name="Win Rate", marker_color=colors,
                    text=[f"{v:.1%}" for v in df_regime["Win Rate"]],
                    textposition="auto",
                ))
                fig_reg.update_layout(
                    template="plotly_dark", height=350,
                    yaxis_title="Win Rate", yaxis_tickformat=".0%",
                )
                st.plotly_chart(fig_reg, use_container_width=True)

                fig_exp = go.Figure()
                fig_exp.add_trace(go.Bar(
                    x=df_regime["Regime"], y=df_regime["Expectancy"],
                    marker_color=colors,
                    text=[f"{v:.3f}R" for v in df_regime["Expectancy"]],
                    textposition="auto",
                ))
                fig_exp.update_layout(
                    template="plotly_dark", height=300,
                    yaxis_title="Expectancy (R)",
                )
                st.plotly_chart(fig_exp, use_container_width=True)
            else:
                st.info("Regime data present but no parseable stats.")
        else:
            st.info("No regime analysis in this WFO result.")

        # ── Bayesian Edge ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Bayesian Edge Estimation")
        st.caption(
            "Bayesian analysis treats the strategy's edge as uncertain and "
            "estimates a **probability distribution** rather than a single number. "
            "This guards against small-sample illusions: a strategy with 5 wins "
            "out of 6 trades looks amazing (83% WR), but Bayesian shrinkage "
            "pulls it toward a realistic ~60%. The posterior chart below shows "
            "the full range of plausible win rates — the wider the curve, the "
            "less certain the estimate."
        )

        if bayesian:
            bc1, bc2, bc3, bc4 = st.columns(4)
            _wr_post = _bget(bayesian, "win_rate", "posterior_mean")
            _mr_post = _bget(bayesian, "mean_r", "posterior_mean")
            _kelly = _bget(bayesian, "kelly_fraction", "posterior_mean")
            _shrink = _bget(bayesian, "shrinkage", "wr_shrinkage")
            bc1.metric(
                "Win Rate (Posterior)", f"{_wr_post:.1%}",
                help="Bayesian-adjusted win rate after shrinking toward the prior (50%). "
                     "More conservative than the raw OOS win rate — especially with "
                     "few trades. This is the win rate you should use for position sizing.",
            )
            bc2.metric(
                "Mean R (Posterior)", f"{_mr_post:.3f}",
                help="Average R-multiple per trade after Bayesian adjustment. "
                     "Positive = profitable on average. > 0.2R is decent for crypto, "
                     "> 0.1R is decent for futures. This accounts for uncertainty "
                     "in the sample mean.",
            )
            bc3.metric(
                "Kelly Fraction", f"{_kelly:.1%}",
                help="Optimal position size as a fraction of capital (Kelly Criterion). "
                     "Full Kelly is aggressive — most practitioners use half-Kelly (divide by 2). "
                     "> 10% suggests a strong edge. < 2% means the edge is marginal. "
                     "Negative = no edge, do not trade.",
            )
            bc4.metric(
                "Shrinkage", f"{_shrink:.2f}",
                help="How much the Bayesian prior pulled the raw win rate toward 50%. "
                     "Range 0-1: 0 = no shrinkage (lots of data, confident estimate), "
                     "1 = full shrinkage (few trades, reverts to prior). "
                     "< 0.3 means enough data for a reliable estimate.",
            )

            _p_exp = _bget(bayesian, "expectancy", "p_positive", default=None)
            if _p_exp is not None:
                _color = "success" if _p_exp > 0.7 else "warning" if _p_exp > 0.5 else "error"
                getattr(st, _color)(
                    f"**P(Expectancy > 0) = {_p_exp:.1%}** — "
                    + ("Strong evidence of edge" if _p_exp > 0.95
                       else "Moderate evidence" if _p_exp > 0.7
                       else "Weak evidence")
                )

            # Posterior visualization
            st.caption(
                "The chart below shows the **prior** (dashed gray — the neutral assumption "
                "before seeing any data) and the **posterior** (blue — updated belief after "
                "observing OOS trades). The red line marks 50% break-even. "
                "If the blue peak sits clearly right of the red line, the strategy "
                "has a statistically meaningful win rate edge."
            )
            wr = _wr_post if _wr_post else 0.5
            n = bayesian.get("n_trades", 50)
            alpha_post = max(1, int(wr * n))
            beta_post = max(1, n - alpha_post)
            x = np.linspace(0.01, 0.99, 200)
            try:
                from scipy.stats import beta as beta_dist
                y_prior = beta_dist.pdf(x, 5, 5)
                y_post = beta_dist.pdf(x, alpha_post, beta_post)
                fig_bayes = go.Figure()
                fig_bayes.add_trace(go.Scatter(
                    x=x, y=y_prior, mode="lines", name="Prior (neutral)",
                    line=dict(color="gray", dash="dash"),
                ))
                fig_bayes.add_trace(go.Scatter(
                    x=x, y=y_post, mode="lines", name="Posterior",
                    fill="tozeroy", line=dict(color="#2196F3"),
                ))
                fig_bayes.add_vline(
                    x=0.5, line_dash="dot", line_color="red",
                    annotation_text="Break-even",
                )
                fig_bayes.update_layout(
                    template="plotly_dark", height=300,
                    xaxis_title="Win Rate", yaxis_title="Density",
                    title="Win Rate Posterior Distribution",
                )
                st.plotly_chart(fig_bayes, use_container_width=True)
            except ImportError:
                st.info("Install scipy for posterior visualization.")
        else:
            st.info("No Bayesian edge data in this WFO result.")

        # ── Monte Carlo ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Monte Carlo Simulation Results")
        st.caption(
            "Monte Carlo randomly reshuffles the OOS trade sequence thousands of "
            "times to answer: *\"If these same trades happened in a different order, "
            "how would the equity curve look?\"* This reveals the range of possible "
            "outcomes and worst-case drawdowns you should prepare for. "
            "The confidence intervals below show the 95% plausible range for each metric."
        )

        if mc and mc.get("valid"):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "P(Profitable)",
                f"{mc.get('p_profitable', 0):.1%}",
                delta="Strong" if mc.get("p_profitable", 0) > 0.7 else "Weak",
                delta_color="normal" if mc.get("p_profitable", 0) > 0.7 else "inverse",
                help="Percentage of Monte Carlo simulations that ended profitable. "
                     "> 70% is encouraging — the strategy is profitable in most "
                     "orderings of its trades. < 50% means profitability depends "
                     "heavily on lucky trade sequencing.",
            )
            mc2.metric(
                "Expectancy CI", _format_ci(mc.get("expectancy_ci")),
                help="95% confidence interval for expectancy (average R per trade). "
                     "If the lower bound is positive, the strategy is profitable "
                     "even in pessimistic scenarios. If it crosses zero, the edge "
                     "is uncertain.",
            )
            mc3.metric(
                "Max Drawdown CI", _format_ci(mc.get("max_drawdown_ci")),
                help="95% confidence interval for the worst peak-to-trough drawdown "
                     "in R-multiples. Use the upper bound for risk management — "
                     "this is the drawdown you should psychologically and financially "
                     "prepare for. Multiply by your R-size to get dollar drawdown.",
            )

            ci_names = ["win_rate_ci", "mean_r_ci", "expectancy_ci", "sharpe_ci",
                        "max_drawdown_ci"]
            ci_labels = ["Win Rate", "Mean R", "Expectancy", "Sharpe", "Max DD"]
            ci_rows = []
            for name, label in zip(ci_names, ci_labels):
                ci = mc.get(name)
                if isinstance(ci, (list, tuple)) and len(ci) >= 2:
                    ci_rows.append({
                        "Metric": label, "Lower": ci[0], "Upper": ci[1],
                        "Mid": (ci[0] + ci[1]) / 2,
                    })

            if ci_rows:
                st.caption(
                    "The bars below show the midpoint of each metric; the error whiskers "
                    "show the 95% confidence range. **Tight whiskers** = high certainty. "
                    "**Wide whiskers** = the outcome depends heavily on trade order. "
                    "Pay special attention to whether the Expectancy bar's lower whisker "
                    "stays above zero — that's the acid test for a real edge."
                )
                df_ci = pd.DataFrame(ci_rows)
                fig_ci = go.Figure()
                fig_ci.add_trace(go.Bar(
                    x=df_ci["Metric"], y=df_ci["Mid"],
                    error_y=dict(
                        type="data", symmetric=False,
                        array=(df_ci["Upper"] - df_ci["Mid"]).tolist(),
                        arrayminus=(df_ci["Mid"] - df_ci["Lower"]).tolist(),
                    ),
                    marker_color="#4CAF50",
                ))
                fig_ci.update_layout(
                    template="plotly_dark", height=350,
                    title=f"Monte Carlo {int(mc.get('confidence', 0.95) * 100)}% "
                          f"Confidence Intervals",
                    yaxis_title="Value",
                )
                st.plotly_chart(fig_ci, use_container_width=True)
        else:
            st.info("No Monte Carlo data or insufficient trades.")

        # ── Window Details ────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("Window-by-Window Details", expanded=False):
            st.caption(
                "Each row is one WFO window. **IS** = in-sample (training), "
                "**OOS** = out-of-sample (validation). Compare IS vs OOS columns: "
                "if IS Total R is high but OOS Total R is near zero or negative, "
                "that window was overfitted. Look for windows where OOS Total R "
                "is consistently positive — those represent periods where the "
                "strategy had a genuine edge. The best_params columns show what "
                "the optimizer selected for each window."
            )
            if windows:
                rows = []
                for w in windows:
                    row = {
                        "Window": w.get("id", "?"),
                        "Train": w.get("train_period", ""),
                        "Test": w.get("test_period", ""),
                        "Regime": w.get("regime", ""),
                        "OOS Trades": w.get("oos_trades", 0),
                        "OOS Total R": round(w.get("oos_total_r", 0), 3),
                        "IS Trades": w.get("is_trades", 0),
                        "IS Total R": round(w.get("is_total_r", 0), 3),
                    }
                    bp = w.get("best_params", {})
                    for k, v in bp.items():
                        row[k] = v
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No window data.")
