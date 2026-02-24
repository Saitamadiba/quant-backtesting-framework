"""Page 8: Monte Carlo Backtest & Prediction vs Reality Comparison."""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Monte Carlo Backtest", page_icon="🎲", layout="wide")
st.title("🎲 Monte Carlo Backtest & Validation")

import sys as _sys
_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in _sys.path:
    _sys.path.insert(0, str(_BASE))

from config import (
    BINANCE_REST_BASE, BINANCE_SYMBOL_MAP,
    BACKTEST_RESULTS_DIR, STRATEGIES, INITIAL_BALANCE,
)
from data.data_loader import get_all_trades
from data.binance_helpers import fetch_binance_candles, calculate_indicators
from components.charts import mc_distribution_chart, mc_equity_fan
from backtrader_framework.optimization.persistence import list_wfo_results, load_wfo_result

BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# How Monte Carlo Backtesting Works
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("**How Monte Carlo Backtesting Works — and Why It Matters**", expanded=False):
    st.markdown("""
### The Problem with Traditional Backtesting

A traditional backtest runs your strategy over historical data and produces **one equity curve** — a
single path showing what *would have happened*. This is dangerously misleading because:

1. **Your actual trade sequence won't repeat.** The specific order of wins and losses in historical data
   is one of millions of possible orderings. A different ordering of the same trades can produce
   dramatically different drawdowns and final returns.
2. **One path tells you nothing about risk.** A single equity curve can't tell you the probability of
   ruin, the range of expected drawdowns, or how wide the band of possible outcomes really is.
3. **It invites overfitting.** When you optimize to one path, you're fitting to noise — the specific
   order and timing of historical events that won't repeat.

### What Monte Carlo Simulation Does

Monte Carlo backtesting solves this by generating **thousands of alternative histories**:

1. **Run a base backtest** to produce a set of trade outcomes (e.g., 50 trades with known P&L each)
2. **Resample with replacement** — randomly draw trades from that pool to create a new sequence
3. **Repeat 10,000 times** — each run produces a different equity path
4. **Analyze the distribution** of final returns and maximum drawdowns across all paths

This produces a **probability distribution** instead of a single number. Instead of "the strategy
returned 25%", you get "the strategy returns between 8% and 42% with 90% confidence, with a
worst-case drawdown of 18%."

### Why This Matters for Position Sizing

The 5th percentile return and 95th percentile drawdown are the numbers that actually matter for risk
management. If your 5th percentile shows a -15% return, you need to size positions so that a 15%
loss is survivable. If you sized based on the mean return alone, you'd be dangerously over-leveraged.
""")

    # Visual: single path vs fan
    st.markdown("#### One Path vs Monte Carlo Distribution")
    demo_fig = go.Figure()
    np.random.seed(42)
    base_trades = np.random.normal(50, 200, 40)
    # Single path
    single_eq = 10000 + np.cumsum(base_trades)
    demo_fig.add_trace(go.Scatter(
        y=[10000] + single_eq.tolist(), mode="lines", name="Single Backtest Path",
        line=dict(color="#F44336", width=2),
    ))
    # MC fan (5 sample paths)
    for i in range(8):
        shuffled = np.random.choice(base_trades, size=len(base_trades), replace=True)
        mc_eq = 10000 + np.cumsum(shuffled)
        demo_fig.add_trace(go.Scatter(
            y=[10000] + mc_eq.tolist(), mode="lines",
            name=f"MC Path {i+1}" if i < 2 else None, showlegend=(i < 2),
            line=dict(color="rgba(33,150,243,0.3)", width=1),
        ))
    demo_fig.update_layout(
        template="plotly_dark", height=300,
        title="Same Trades, Different Orderings = Different Outcomes",
        xaxis_title="Trade #", yaxis_title="Equity ($)",
    )
    st.plotly_chart(demo_fig, use_container_width=True, key="mc_demo")

    st.markdown("""
> The red line is the historical path. The blue lines are alternative Monte Carlo paths using the
> exact same trades in different order. Notice how the same set of trades can produce both higher
> and lower final equity, and very different drawdown experiences.
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Our Backtesting Design: Anti-Overfitting Architecture
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("**Our Backtesting Design: How We Avoid Overfitting**", expanded=False):
    st.markdown("""
### The Overfitting Problem

**Overfitting** is the #1 killer of backtested strategies. It happens when a strategy is tuned to
perform well on historical data but fails on new data because it learned the *noise* instead of the
*signal*. Signs of an overfit strategy:

- Spectacular backtest results that collapse in live trading
- Performance that degrades immediately when you extend the test period
- Hundreds of parameters that were "optimized" on the same data they were tested on

### Our Anti-Overfitting Defenses

This backtesting system uses **7 layers of defense** against overfitting:
""")

    st.markdown("#### 1. Walk-Forward Analysis (WFA)")
    st.markdown("""
The gold standard for preventing overfitting. Instead of training and testing on the same data:

- **Split data into sequential windows**: 500 bars for training, 100 bars for out-of-sample (OOS) testing
- **Only OOS results count** — in-sample results are tracked for comparison only
- **Anchored (expanding) mode**: training window grows over time, mimicking real production
- **Overfit ratio**: calculated as OOS performance / In-Sample performance
  - Ratio ~1.0 = robust (OOS matches IS)
  - Ratio < 0.5 = severe overfitting (OOS is half of IS)

This means the strategy is always tested on data it has **never seen during development**.
""")

    st.markdown("#### 2. No Look-Ahead Bias")
    st.markdown("""
Every technical indicator (EMA, ATR, RSI, ADX) is **recalculated per window** using only data
available at decision time. This prevents a subtle but devastating bias where future price data
leaks into indicator calculations.

```
Old (biased):  df['EMA50'] = df['Close'].ewm(span=50).mean()  # Uses ALL data including future
New (correct): Calculate on the window slice only — never pass future data
```
""")

    st.markdown("#### 3. Transaction Cost Modeling")
    st.markdown("""
Every trade is penalized with realistic execution costs calibrated per asset class:

| Cost Component | BTC/ETH (Perp Futures) | NQ (CME Micro) |
|----------------|----------------------|----------------|
| Spread (half) | 0.01% | 0.01% |
| Taker fee | 0.055% | 0.02% |
| Slippage | 0.01% | 0.01% |
| **Total round-trip** | **~0.15%** | **~0.08%** |

Costs are based on actual Bybit/Binance perp futures and CME micro futures fee schedules.
Naive backtests that ignore costs can show a profitable strategy that actually loses money after
friction. Our implementation deducts costs from **every trade** in R-multiple terms.
""")

    st.markdown("#### 4. Fixed Parameters (No Per-Window Optimization)")
    st.markdown("""
Strategy parameters are **hardcoded, not optimized**:

- `lookback_period = 20`, `sweep_tolerance = 0.2%`, `min_confidence = 0.48`
- These values are the same across **all WFA windows** — no curve-fitting per period
- Parameters were derived from market microstructure theory, not from data mining

This is the most important defense: if you optimize parameters to historical data, you're
guaranteed to overfit. Fixed parameters that work across all time periods indicate a genuine edge.
""")

    st.markdown("#### 5. Statistical Validation Suite")
    st.markdown("""
Results are subjected to rigorous hypothesis testing before being trusted:

| Test | What It Checks | Passing Criteria |
|------|---------------|-----------------|
| **Binomial test** | Is win rate significantly > 50%? | p-value < 0.05 |
| **One-sample t-test** | Is mean R significantly > 0? | p-value < 0.05 |
| **Wilson score CI** | 95% confidence interval on win rate | Lower bound > 40% |
| **Serial correlation** | Are trade outcomes independent? | Autocorrelation < 0.15 |
| **Power analysis** | Do we have enough trades? | Min N for 80% power |

If any test fails, the results should be treated as inconclusive — not as evidence of an edge.
""")

    st.markdown("#### 6. Regime-Aware Analysis")
    st.markdown("""
Results are broken down by market regime (trending up, trending down, ranging, volatile) using
ADX and EMA indicators. This reveals whether the strategy:

- Works in **all** regimes (robust edge)
- Only works in **trending** regimes (vulnerable to range-bound markets)
- Fails in **volatile** regimes (need circuit breakers)

A strategy that only backtest-profits in one regime is much more fragile than one that works
across multiple conditions.
""")

    st.markdown("#### 7. Bootstrap Permutation (This Page)")
    st.markdown("""
The Monte Carlo simulation on this page uses **bootstrap resampling with replacement**:

- Draws N trades randomly from the historical pool (with replacement)
- Each draw creates a different sequence of outcomes
- Repeated 10,000 times to build a full probability distribution

This tests whether the strategy's edge survives when trade ordering is randomized — if it does,
the edge is in the individual trade quality, not in lucky sequencing.
""")

    st.markdown("---")
    st.markdown("#### Naive Backtesting vs Our Implementation")

    comparison_data = {
        "Design Choice": [
            "Parameter optimization",
            "Look-ahead bias",
            "Transaction costs",
            "Out-of-sample testing",
            "Trade ordering",
            "Overfitting measurement",
            "Statistical testing",
            "Regime analysis",
            "Trade duration limits",
            "Data source",
        ],
        "Naive Backtest": [
            "Curve-fit to entire dataset",
            "Future data leaks into indicators",
            "Ignored — inflates results",
            "None — train and test on same data",
            "Tested in historical order only",
            "Not measured",
            "None — just looks at returns",
            "Ignored — all periods lumped together",
            "Infinite hold allowed",
            "Current market data (survivorship bias)",
        ],
        "Our Implementation": [
            "Fixed parameters across all windows",
            "Indicators recalculated per window slice only",
            "Asset-calibrated costs (BTC/ETH: 0.15% RT, NQ: 0.08% RT) per trade",
            "Walk-Forward: only OOS results count",
            "10,000 bootstrap permutations",
            "OOS/IS ratio calculated per window",
            "Binomial, t-test, Wilson CI, power analysis",
            "Broken down by trending/ranging/volatile",
            "Max 168 bars per trade",
            "Pre-collected DuckDB historical data",
        ],
    }
    st.dataframe(pd.DataFrame(comparison_data).set_index("Design Choice"),
                 use_container_width=True)

    st.markdown("""
> **Bottom line**: Every design choice above exists to answer one question:
> *"Would this strategy still be profitable if we hadn't seen this specific historical data?"*
> If the answer is yes across all these checks, the edge is likely real.
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section A: Backtest Configuration
# ══════════════════════════════════════════════════════════════════════════════
st.header("A. Backtest Configuration")

# ── Backtrader strategy name mapping ────────────────────────────────────────
_BT_STRATEGY_MAP = {
    "FVG": "FVG",
    "SBS": "SBS",
    "Liquidity Raid": "LiquidityRaid",
    "Momentum Mastery": "MomentumMastery",
}
# Default timeframes per strategy (from backtrader framework)
_BT_DEFAULT_TF = {
    "FVG": "15m",
    "SBS": "4h",
    "Liquidity Raid": "15m",
    "Momentum Mastery": "15m",
}

c1, c2 = st.columns(2)
strategy = c1.selectbox("Strategy", ["FVG", "SBS", "Liquidity Raid", "Momentum Mastery"])
symbol = c2.selectbox("Symbol", ["BTC", "ETH", "NQ"])

# ── Auto-detect real trades and WFO results ────────────────────────────────
_all_trades_preview = get_all_trades()
_real_count = len(
    _all_trades_preview[
        (_all_trades_preview["strategy"] == strategy)
        & (_all_trades_preview["symbol"] == symbol)
    ]
)

# ── Auto-detect WFO OOS results ───────────────────────────────────────────
_bt_name = _BT_STRATEGY_MAP.get(strategy, strategy)
_all_wfo = list_wfo_results()
_wfo_matches = [
    r for r in _all_wfo
    if r['strategy'] == _bt_name and r['symbol'] == symbol
]

_data_sources = ["Real Trades", "Synthetic Backtest", "WFO OOS Trades"]

if _wfo_matches:
    st.success(
        f"**{len(_wfo_matches)} WFO result(s)** available for {strategy} {symbol}. "
        f"'WFO OOS Trades' uses rigorously validated out-of-sample trades (recommended)."
    )
    _default_source = 2  # WFO OOS
elif _real_count >= 3:
    st.success(
        f"**{_real_count} real {strategy} {symbol} trades available.** "
        f"'Real Trades' mode will Monte Carlo reshuffle these actual outcomes."
    )
    _default_source = 0  # Real Trades
else:
    st.info(
        f"Only {_real_count} real trade(s) for {strategy} {symbol} — not enough for Monte Carlo. "
        f"Use 'Synthetic Backtest' to run the actual {strategy} strategy on historical DuckDB candles."
    )
    _default_source = 1  # Synthetic Backtest

data_source = st.radio(
    "Data Source",
    _data_sources,
    index=_default_source,
    horizontal=True,
    help="**WFO OOS Trades** (Recommended) = Monte Carlo on Walk-Forward validated out-of-sample trades. "
         "These trades were tested on data the strategy never saw during optimization — the most rigorous source. "
         "**Real Trades** = Monte Carlo on your actual bot trade outcomes. "
         "**Synthetic Backtest** = single-pass backtest on historical data (least rigorous).",
)

c3, c4, c5 = st.columns(3)

_all_tf_options = ["15m", "1h", "4h"]

if data_source == "Synthetic Backtest":
    default_tf = _BT_DEFAULT_TF.get(strategy, "15m")
    default_idx = _all_tf_options.index(default_tf) if default_tf in _all_tf_options else 0
    timeframe = c3.selectbox("Timeframe", _all_tf_options, index=default_idx,
                             help=f"Default for {strategy}: {default_tf}. Data comes from local DuckDB.")
elif data_source == "WFO OOS Trades":
    default_tf = _BT_DEFAULT_TF.get(strategy, "4h")
    default_idx = _all_tf_options.index(default_tf) if default_tf in _all_tf_options else 3
    timeframe = c3.selectbox("Timeframe", _all_tf_options, index=default_idx,
                             help="Select any timeframe. If no WFO result exists, one will be run automatically.")
else:
    timeframe = c3.selectbox("Timeframe", ["All"], disabled=True)

num_sims = c4.slider("Monte Carlo Simulations", 100, 50_000, 10_000, step=100)

if data_source == "Synthetic Backtest":
    lookback_days = None
    c5.caption("Uses full DuckDB historical data")
elif data_source == "WFO OOS Trades":
    lookback_days = None
    c5.caption("Uses WFO out-of-sample trades")
else:
    lookback_days = None

# ── WFO result picker ─────────────────────────────────────────────────────
_selected_wfo = None
_risk_per_trade = INITIAL_BALANCE * 0.01  # default 1%
_wfo_needs_run = False

if data_source == "WFO OOS Trades":
    # Filter WFO results by strategy, symbol, and timeframe
    _wfo_filtered = [
        r for r in _wfo_matches
        if r['timeframe'] == timeframe
    ]

    if _wfo_filtered:
        _wfo_labels = {
            i: f"{r['filename']}  ({r['timestamp']})"
            for i, r in enumerate(_wfo_filtered)
        }
        _wfo_idx = st.selectbox(
            "WFO Result",
            list(_wfo_labels.keys()),
            format_func=lambda i: _wfo_labels[i],
            key="mc_wfo_select",
        )
        _selected_wfo = _wfo_filtered[_wfo_idx]
    else:
        st.info(
            f"No saved WFO result for **{strategy}/{symbol}/{timeframe}**. "
            f"A WFO optimization will be run automatically when you click 'Run Backtest'."
        )
        _wfo_needs_run = True

    _risk_per_trade = st.slider(
        "Risk per Trade ($)",
        min_value=50, max_value=500, value=int(INITIAL_BALANCE * 0.01), step=10,
        help="Dollar amount of 1R (one risk unit). WFO trades are in R-multiples — "
             "this converts them to dollar PnLs for the simulation. "
             f"Default: 1% of ${INITIAL_BALANCE:,} = ${int(INITIAL_BALANCE * 0.01)}.",
    )

# Block bootstrap slider
block_size = st.slider(
    "Block Size",
    min_value=1, max_value=10, value=1,
    help="Block size for bootstrap resampling. 1 = IID (independent trades). >1 = block bootstrap preserving consecutive trade clusters to account for regime effects.",
)

run_btn = st.button("Run Backtest", type="primary")


def monte_carlo_simulation(trade_pnls: list, num_runs: int,
                           initial_capital: float = 10_000,
                           block_size: int = 1) -> dict:
    """Run Monte Carlo permutation on trade PnLs with optional block bootstrap."""
    if len(trade_pnls) < 3:
        return {"returns": [], "drawdowns": [], "equity_paths": []}

    pnls = np.array(trade_pnls)
    n = len(pnls)
    final_returns = []
    max_drawdowns = []
    equity_paths = []

    for _ in range(num_runs):
        if block_size <= 1:
            # IID bootstrap (original behavior)
            shuffled = np.random.choice(pnls, size=n, replace=True)
        else:
            # Block bootstrap: preserve consecutive trade clusters
            n_blocks = int(np.ceil(n / block_size))
            blocks = []
            max_start = max(1, n - block_size + 1)
            for _ in range(n_blocks):
                start = np.random.randint(0, max_start)
                blocks.extend(pnls[start:start + block_size].tolist())
            shuffled = np.array(blocks[:n])

        equity = initial_capital + np.cumsum(shuffled)
        equity = np.insert(equity, 0, initial_capital)

        final_ret = (equity[-1] / initial_capital - 1) * 100
        final_returns.append(final_ret)

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1)
        max_drawdowns.append(dd.max() * 100)

        if len(equity_paths) < 200:  # store up to 200 for fan chart
            equity_paths.append(equity.tolist())

    return {
        "returns": final_returns,
        "drawdowns": max_drawdowns,
        "equity_paths": equity_paths,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section B-C: Execute & Display
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    with st.status("Running backtest...", expanded=True) as status:
        trade_pnls = []

        if data_source == "WFO OOS Trades":
            # Run WFO if no saved result exists for this combo
            if _wfo_needs_run:
                st.write(f"Running WFO optimization for **{strategy}/{symbol}/{timeframe}** — this may take a few minutes...")
                try:
                    import importlib
                    import backtrader_framework.optimization.strategy_adapters as _adapters_mod
                    importlib.reload(_adapters_mod)
                    ADAPTER_REGISTRY = _adapters_mod.ADAPTER_REGISTRY
                    from backtrader_framework.optimization.wfo_engine import RegimeAdaptiveWFO, WFOConfig, TransactionCosts
                    from backtrader_framework.optimization.persistence import save_wfo_result

                    if _bt_name not in ADAPTER_REGISTRY:
                        st.error(
                            f"Strategy '{_bt_name}' not found in adapter registry. "
                            f"Available: {list(ADAPTER_REGISTRY.keys())}"
                        )
                        st.stop()
                    adapter = ADAPTER_REGISTRY[_bt_name]()
                    config = WFOConfig.for_timeframe(timeframe, costs=TransactionCosts.for_asset(symbol))
                    engine = RegimeAdaptiveWFO(adapter, config)

                    def _progress(pct, msg):
                        st.write(f"  [{pct:.0%}] {msg}")

                    wfo_data = engine.run(symbol, timeframe, progress_callback=_progress)
                    if not wfo_data or not wfo_data.get('oos_stats', {}).get('valid', False):
                        st.error(f"WFO produced no valid results for {strategy}/{symbol}/{timeframe}.")
                        st.stop()

                    filepath = save_wfo_result(wfo_data)
                    st.write(f"WFO complete — saved to **{Path(filepath).name}**")

                    # Store for metadata saving later
                    _selected_wfo = {
                        'filepath': filepath,
                        'filename': Path(filepath).name,
                        'timestamp': '',
                    }
                except Exception as e:
                    st.error(f"WFO optimization failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
            else:
                st.write(f"Loading WFO OOS result: **{_selected_wfo['filename']}**...")
                wfo_data = load_wfo_result(_selected_wfo['filepath'])

            oos_equity = wfo_data.get('oos_equity', [])
            oos_stats = wfo_data.get('oos_stats', {})

            if not oos_equity:
                st.error("WFO result has no OOS trades.")
                st.stop()

            # Convert R-multiples to dollar PnLs
            trade_pnls = [t['r'] * _risk_per_trade for t in oos_equity]

            # Show WFO metadata
            _wr = oos_stats.get('win_rate', 0)
            _mr = oos_stats.get('mean_r', 0)
            _sh = oos_stats.get('sharpe_per_trade', 0)
            _or = wfo_data.get('overfit_ratio')
            _sig = oos_stats.get('mean_r_significant', False)
            _ds = wfo_data.get('data_start', '?')[:10]
            _de = wfo_data.get('data_end', '?')[:10]
            _nw = wfo_data.get('n_windows', '?')

            st.write(
                f"Loaded **{len(trade_pnls)} OOS trades** from {_nw} WFO windows | "
                f"Win Rate: {_wr:.1%} | Mean R: {_mr:+.4f} | Sharpe: {_sh:.3f} | "
                f"Data: {_ds} to {_de}"
            )
            if _or is not None and np.isfinite(_or):
                _or_label = "good" if 0.5 <= abs(_or) <= 1.5 else "suspect"
                st.write(f"Overfit Ratio: {_or:.3f} ({_or_label}) | Significant: {'Yes' if _sig else 'No'}")

        elif data_source == "Real Trades":
            st.write("Loading real trade data...")
            all_trades = get_all_trades()
            filtered = all_trades[
                (all_trades["strategy"] == strategy) &
                (all_trades["symbol"] == symbol)
            ]
            if filtered.empty:
                st.error(f"No trades found for {strategy} {symbol}. Try WFO OOS Trades or Synthetic Backtest mode.")
                st.stop()
            trade_pnls = filtered.sort_values("entry_time")["pnl_usd"].tolist()
            st.write(f"Loaded {len(trade_pnls)} real trades.")
        else:
            # Synthetic: run the actual strategy via backtrader on DuckDB data
            bt_name = _BT_STRATEGY_MAP.get(strategy)
            st.write(f"Running **{strategy}** strategy on {symbol} {timeframe} via backtrader framework...")
            try:
                import warnings as _w
                _w.filterwarnings("ignore")
                from backtrader_framework.runners.single_backtest import run_backtest as _bt_run

                bt_result = _bt_run(
                    strategy_name=bt_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    initial_cash=INITIAL_BALANCE,
                )
                bt_trades = bt_result["trades"]
                bt_summary = bt_result["summary"]

                if not bt_trades:
                    st.error(f"Backtest produced 0 trades for {strategy} {symbol} {timeframe}. The strategy found no setups in the DuckDB data.")
                    st.stop()

                # Convert percent PnLs to dollar amounts (using initial balance as base)
                trade_pnls = [t["pnl_percent"] / 100 * INITIAL_BALANCE for t in bt_trades if t.get("exit_price")]
                st.write(
                    f"Backtest complete: **{len(trade_pnls)} trades** | "
                    f"Win Rate: {bt_summary['win_rate']:.1f}% | "
                    f"Total R: {bt_summary['total_r']:.2f} | "
                    f"Data: {bt_summary['start_date'][:10]} → {bt_summary['end_date'][:10]}"
                )
            except ImportError as exc:
                st.error(f"Could not load backtrader framework: {exc}")
                st.stop()
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

        # ── Auto-fallback: if synthetic produced too few trades, use real trades ──
        if len(trade_pnls) < 3:
            if data_source == "Synthetic Backtest" and _real_count >= 3:
                st.warning(
                    f"Synthetic backtest produced only **{len(trade_pnls)} trade(s)** — "
                    f"these strategies are highly selective on historical data. "
                    f"Auto-switching to **{_real_count} real {strategy} {symbol} trades** for Monte Carlo."
                )
                all_trades = get_all_trades()
                filtered = all_trades[
                    (all_trades["strategy"] == strategy) &
                    (all_trades["symbol"] == symbol)
                ]
                trade_pnls = filtered.sort_values("entry_time")["pnl_usd"].tolist()
                data_source = "Real Trades"  # update for save metadata
            else:
                st.warning("Too few trades for Monte Carlo. Sync more VPS data or try a different strategy/symbol.")
                st.stop()

        # Monte Carlo with block bootstrap
        bs_label = f" (block size {block_size})" if block_size > 1 else ""
        st.write(f"Running {num_sims:,} Monte Carlo simulations{bs_label}...")
        mc_results = monte_carlo_simulation(
            trade_pnls, num_sims,
            initial_capital=INITIAL_BALANCE,
            block_size=block_size,
        )
        status.update(label="Backtest complete!", state="complete")

    # ── KPI Row ───────────────────────────────────────────────────────────
    st.subheader("Results")
    st.caption("Summary statistics from the Monte Carlo simulation. The 5th and 95th percentiles define the realistic range of outcomes — plan your risk management around the 5th percentile (worst realistic case), not the mean.")
    returns = np.array(mc_results["returns"])
    drawdowns = np.array(mc_results["drawdowns"])

    c1, c2, c3, c4, c5 = st.columns(5)
    sr = (returns > 0).mean()
    c1.metric("Success Rate", f"{sr:.1%}",
              help=f"Percentage of Monte Carlo runs that ended profitably. Currently {sr:.1%}. Above 80% indicates a robust strategy; below 60% is concerning.")
    mr = returns.mean()
    c2.metric("Mean Return", f"{mr:.1f}%",
              help=f"Average return across all simulations. Currently {mr:.1f}%. This is the 'expected' outcome, but individual runs can vary widely.")
    med = np.median(returns)
    c3.metric("Median Return", f"{med:.1f}%",
              help=f"The middle value of all simulated returns. Currently {med:.1f}%. More robust than mean as it's not skewed by outliers.")
    p5 = np.percentile(returns, 5)
    c4.metric("5th %ile", f"{p5:.1f}%",
              help=f"Worst-case scenario (95% confidence). Currently {p5:.1f}%. This is what you should prepare for — if this number is deeply negative, consider reducing position size.")
    p95 = np.percentile(returns, 95)
    c5.metric("95th %ile", f"{p95:.1f}%",
              help=f"Best-case scenario (95% confidence). Currently {p95:.1f}%. Don't plan around this — it's the optimistic tail.")

    cd1, cd2 = st.columns(2)
    mdd = drawdowns.mean()
    cd1.metric("Mean Max DD", f"{mdd:.1f}%",
               help=f"Average maximum drawdown across all simulations. Currently {mdd:.1f}%. This tells you the typical pain you'll experience. Keep position sizing so this stays below your tolerance.")
    p95dd = np.percentile(drawdowns, 95)
    cd2.metric("95th %ile DD", f"{p95dd:.1f}%",
               help=f"Worst-case drawdown (95% confidence). Currently {p95dd:.1f}%. If this exceeds 25%, consider tightening stops or reducing risk per trade.")

    # Trade stats
    pnls = np.array(trade_pnls)
    wins = (pnls > 0).sum()
    st.caption(
        f"Base trades: {len(pnls)} | Win rate: {wins/len(pnls):.1%} | "
        f"Avg trade: ${pnls.mean():,.2f} | Profit factor: "
        f"{pnls[pnls>0].sum() / abs(pnls[pnls<0].sum()):.2f}" if (pnls < 0).any()
        else f"Base trades: {len(pnls)} | All winning"
    )
    if block_size > 1:
        st.caption(f"Block bootstrap with block size {block_size} — preserves serial correlation from regime clusters.")

    # ── Charts ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Distribution of final returns across all simulations. The wider the spread, the more uncertain the outcome. A distribution centered well above zero with a thin left tail is ideal.")
        st.plotly_chart(mc_distribution_chart(mc_results["returns"]), key="mc_dist")
    with col2:
        st.caption("Multiple simulated equity paths overlaid. The spread shows the range of possible portfolio trajectories. If many paths dip significantly below the starting capital, your strategy has meaningful blow-up risk.")
        st.plotly_chart(mc_equity_fan(mc_results["equity_paths"]), key="mc_fan")

    # ── Save Results ──────────────────────────────────────────────────────
    # ── Backtest Results Commentary ──────────────────────────────────
    st.markdown("---")
    st.subheader("Analysis & Recommendations")
    bt_points = []

    if sr >= 0.90:
        bt_points.append(f"**{sr:.0%} success rate** is exceptionally high — the strategy is robust across most Monte Carlo reshuffles. Position sizing can be moderately aggressive.")
    elif sr >= 0.70:
        bt_points.append(f"**{sr:.0%} success rate** is solid — the majority of simulations end profitably. However, {1-sr:.0%} of paths end in a loss, so risk management is still essential.")
    elif sr >= 0.50:
        bt_points.append(f"**{sr:.0%} success rate** is marginal — nearly half the simulations lose money. The strategy's edge is thin and highly dependent on trade ordering. Consider tightening entry filters.")
    else:
        bt_points.append(f"**{sr:.0%} success rate** is below 50% — the strategy is more likely to lose than win. This does not support live deployment.")

    if p5 < -15:
        bt_points.append(f"The 5th percentile return is **{p5:.1f}%** — in the worst realistic scenario, you could lose over 15% of capital. Size positions so that this worst case stays within your risk tolerance.")
    elif p5 > 0:
        bt_points.append(f"Even the 5th percentile return is positive (**{p5:.1f}%**) — the strategy is profitable in 95%+ of simulations.")

    if p95dd > 25:
        bt_points.append(f"95th percentile max drawdown of **{p95dd:.1f}%** is severe. Most traders cannot psychologically handle a 25%+ drawdown. Reduce risk per trade or add a drawdown circuit breaker.")
    elif mdd > 15:
        bt_points.append(f"Mean max drawdown of **{mdd:.1f}%** is significant. Ensure your account can handle this without triggering margin calls or emotional tilt.")

    spread = p95 - p5
    bt_points.append(f"Return spread (5th to 95th percentile): **{spread:.0f}** percentage points. {'Wide spread indicates high variance — results are unpredictable.' if spread > 40 else 'Moderate spread — outcomes are reasonably predictable.' if spread > 20 else 'Tight spread — consistent outcomes.'}")

    if data_source == "WFO OOS Trades":
        bt_points.append(
            "Using **WFO out-of-sample trades** — the most rigorous data source. "
            "These trades were validated on data the strategy never saw during optimization, "
            "eliminating look-ahead bias and overfitting."
        )
    elif data_source == "Real Trades":
        bt_points.append("Using **real trade data** — these results reflect your actual strategy performance.")
    else:
        bt_points.append(f"Using **synthetic backtest** — the actual {strategy} strategy logic was run on historical DuckDB candles via backtrader. These results approximate live conditions but may differ due to execution differences.")

    for p in bt_points:
        st.markdown(f"- {p}")

    # ── Save Results ──────────────────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]

    if data_source == "WFO OOS Trades":
        _save_tf = timeframe
    elif data_source == "Synthetic Backtest":
        _save_tf = timeframe
    else:
        _save_tf = "real"

    result_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": _save_tf,
        "data_source": data_source,
        "block_size": block_size,
        "num_simulations": num_sims,
        "lookback_days": lookback_days or 0,
        "num_trades": len(trade_pnls),
        "results": {
            "success_rate": float((returns > 0).mean()),
            "mean_return": float(returns.mean()),
            "median_return": float(np.median(returns)),
            "p5_return": float(np.percentile(returns, 5)),
            "p95_return": float(np.percentile(returns, 95)),
            "mean_max_dd": float(drawdowns.mean()),
            "p95_max_dd": float(np.percentile(drawdowns, 95)),
            "base_win_rate": float(wins / len(pnls)),
            "base_profit_factor": float(
                pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum())
            ) if (pnls < 0).any() else 999,
        },
        "trade_pnls": [float(p) for p in trade_pnls],
    }

    # Add WFO metadata if applicable
    if data_source == "WFO OOS Trades" and _selected_wfo:
        result_data["wfo_filepath"] = _selected_wfo['filepath']
        result_data["wfo_filename"] = _selected_wfo['filename']
        result_data["risk_per_trade"] = _risk_per_trade
        result_data["wfo_oos_stats"] = {
            k: oos_stats.get(k) for k in [
                'n_trades', 'win_rate', 'mean_r', 'sharpe_per_trade',
                'profit_factor', 'expectancy', 'mean_r_significant',
            ]
        }
        result_data["wfo_overfit_ratio"] = wfo_data.get('overfit_ratio')

    result_path = BACKTEST_RESULTS_DIR / f"mc_{run_id}_{strategy}_{symbol}_{data_source.replace(' ', '_')}.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    st.success(f"Results saved: {result_path.name}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section D: Historical Runs
# ══════════════════════════════════════════════════════════════════════════════
st.header("Historical Backtest Runs")
st.caption("Archive of all previous Monte Carlo runs. Use this to compare how strategy expectations have changed over time as you collect more data or adjust parameters.")
result_files = sorted(BACKTEST_RESULTS_DIR.glob("mc_*.json"), reverse=True)

if result_files:
    rows = []
    for fp in result_files[:20]:
        try:
            with open(fp) as f:
                d = json.load(f)
            rows.append({
                "Run ID": d["run_id"],
                "Time": d["timestamp"][:19],
                "Strategy": d["strategy"],
                "Symbol": d["symbol"],
                "Source": d.get("data_source", "Synthetic"),
                "TF": d.get("timeframe", "?"),
                "Sims": d["num_simulations"],
                "Trades": d["num_trades"],
                "Success Rate": f"{d['results']['success_rate']:.1%}",
                "Mean Return": f"{d['results']['mean_return']:.1f}%",
                "file": fp.name,
            })
        except Exception:
            continue
    if rows:
        st.dataframe(pd.DataFrame(rows).drop(columns=["file"]), use_container_width=True, hide_index=True, column_config={
            "Run ID": st.column_config.TextColumn("Run ID", help="Unique 8-character identifier for this backtest run. Use it to reference specific runs when comparing."),
            "Time": st.column_config.TextColumn("Time", help="Date and time when this Monte Carlo simulation was executed."),
            "Strategy": st.column_config.TextColumn("Strategy", help="Which trading strategy was backtested in this run."),
            "Symbol": st.column_config.TextColumn("Symbol", help="The trading pair (BTC or ETH) used for this backtest."),
            "Source": st.column_config.TextColumn("Source", help="Data source: Real Trades (actual bot outcomes) or Synthetic (EMA cross proxy)."),
            "TF": st.column_config.TextColumn("TF", help="Candle timeframe used (e.g. 15m, 1h, 4h). 'real' means actual trade data was used."),
            "Sims": st.column_config.TextColumn("Sims", help="Number of Monte Carlo simulations run. Higher = more reliable probability estimates. 10,000 is a good default."),
            "Trades": st.column_config.TextColumn("Trades", help="Number of base trades used for Monte Carlo reshuffling."),
            "Success Rate": st.column_config.TextColumn("Success Rate", help="Percentage of Monte Carlo runs that ended profitably. Above 80% is robust; below 60% is concerning."),
            "Mean Return": st.column_config.TextColumn("Mean Return", help="Average return across all simulated equity paths. This is the expected outcome, but individual runs can vary widely."),
        })
else:
    st.info("No past backtest runs found. Run one above!")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section E: Prediction vs Reality
# ══════════════════════════════════════════════════════════════════════════════
st.header("Prediction vs Reality")
st.caption("The most important validation step. This compares what your Monte Carlo simulation predicted against what actually happened in live trading. If actual performance consistently falls outside the predicted range, it means either market conditions have changed or the backtest assumptions were flawed — both require action.")

if result_files:
    selected_run = st.selectbox(
        "Select MC Run to Compare",
        [f.stem for f in result_files[:20]],
    )
    selected_path = BACKTEST_RESULTS_DIR / f"{selected_run}.json"

    if selected_path.exists():
        with open(selected_path) as f:
            mc_data = json.load(f)

        mc_res = mc_data["results"]
        mc_strategy = mc_data["strategy"]
        mc_symbol = mc_data["symbol"]

        # Load actual live performance
        live_df = get_all_trades(source_filter="Live")
        actual = live_df[
            (live_df["strategy"] == mc_strategy) & (live_df["symbol"] == mc_symbol)
        ]

        if not actual.empty:
            actual_pnl = actual["pnl_usd"].sum()
            actual_return = (actual_pnl / INITIAL_BALANCE) * 100
            actual_wr = (actual["pnl_usd"] > 0).mean()

            # Fix 2: Actual max DD using account equity
            cum = actual.sort_values("entry_time")["pnl_usd"].cumsum()
            cum_equity = INITIAL_BALANCE + cum
            peak_equity = cum_equity.cummax()
            actual_max_dd = ((peak_equity - cum_equity) / peak_equity).max() * 100

            gp = actual.loc[actual["pnl_usd"] > 0, "pnl_usd"].sum()
            gl = actual.loc[actual["pnl_usd"] < 0, "pnl_usd"].abs().sum()
            actual_pf = gp / gl if gl > 0 else 999

            # Comparison table
            comparison = pd.DataFrame([
                {
                    "Metric": "Return (%)",
                    "MC Predicted (5th-95th)": f"{mc_res['p5_return']:.1f}% — {mc_res['p95_return']:.1f}%",
                    "Actual": f"{actual_return:.1f}%",
                    "Within Range": "Yes" if mc_res["p5_return"] <= actual_return <= mc_res["p95_return"] else "No",
                },
                {
                    "Metric": "Max Drawdown (%)",
                    "MC Predicted (5th-95th)": f"0% — {mc_res['p95_max_dd']:.1f}%",
                    "Actual": f"{actual_max_dd:.1f}%",
                    "Within Range": "Yes" if actual_max_dd <= mc_res["p95_max_dd"] else "No",
                },
                {
                    "Metric": "Win Rate",
                    "MC Predicted (5th-95th)": f"{mc_res.get('base_win_rate', 0)*100:.0f}% (base)",
                    "Actual": f"{actual_wr:.1%}",
                    "Within Range": "~",
                },
                {
                    "Metric": "Profit Factor",
                    "MC Predicted (5th-95th)": f"{mc_res.get('base_profit_factor', 0):.2f} (base)",
                    "Actual": f"{actual_pf:.2f}",
                    "Within Range": "~",
                },
            ])

            def _color_range(val):
                if val == "Yes":
                    return "background-color: #1B5E20; color: white"
                elif val == "No":
                    return "background-color: #B71C1C; color: white"
                return ""

            st.caption("How to read this table: Each row compares a predicted metric range (from Monte Carlo) against the actual live result. Green 'Yes' means the actual value falls within the predicted 5th-95th percentile range — the backtest was well-calibrated. Red 'No' means the actual value is outside the predicted range — live conditions differ materially from the backtest. A tilde (~) means the metric is shown for reference but not range-tested.")
            st.dataframe(
                comparison.style.map(_color_range, subset=["Within Range"]),
                use_container_width=True, hide_index=True,
            )

            # ── Prediction vs Reality Commentary ──────────────────────
            st.markdown("**Validation Assessment:**")
            pvr_points = []

            in_range_count = sum(1 for _, r in comparison.iterrows() if r["Within Range"] == "Yes")
            total_metrics = sum(1 for _, r in comparison.iterrows() if r["Within Range"] in ("Yes", "No"))
            if total_metrics > 0:
                if in_range_count == total_metrics:
                    pvr_points.append("All metrics fall within the predicted Monte Carlo range — your backtest assumptions are well-calibrated to live conditions.")
                elif in_range_count == 0:
                    pvr_points.append("**No metrics** fall within the predicted range. The backtest is not representative of live trading. Common causes: overfitting, slippage, regime change, or data-snooping bias.")
                else:
                    pvr_points.append(f"**{in_range_count}/{total_metrics}** metrics are within range. Investigate the out-of-range metrics below.")

            if mc_res["p5_return"] > actual_return:
                pvr_points.append(f"Actual return ({actual_return:.1f}%) is **below** the 5th percentile ({mc_res['p5_return']:.1f}%). This is a statistically significant underperformance — live conditions are materially worse than backtested.")
            elif actual_return > mc_res["p95_return"]:
                pvr_points.append(f"Actual return ({actual_return:.1f}%) **exceeds** the 95th percentile ({mc_res['p95_return']:.1f}%). While positive, this suggests the live sample may be too small or benefitting from favorable conditions that won't persist.")

            if actual_max_dd > mc_res["p95_max_dd"]:
                pvr_points.append(f"Actual max drawdown ({actual_max_dd:.1f}%) exceeds predicted 95th percentile ({mc_res['p95_max_dd']:.1f}%). **Reduce position sizes** — your risk exposure is higher than the backtest anticipated.")

            for p in pvr_points:
                st.markdown(f"- {p}")

            # Overlay chart: actual equity vs MC bands
            if "equity_paths" in mc_data or "trade_pnls" in mc_data:
                st.subheader("Actual vs Predicted Equity")
                st.caption("Your actual equity curve overlaid on the Monte Carlo base path. If the actual line diverges significantly below the predicted path, it's an early warning that live conditions differ from backtest assumptions.")
                fig = go.Figure()

                # MC bands (resample from stored trade pnls)
                mc_pnls = mc_data.get("trade_pnls", [])
                if mc_pnls:
                    mc_equity = INITIAL_BALANCE + np.cumsum(mc_pnls)
                    fig.add_trace(go.Scatter(
                        y=[INITIAL_BALANCE] + mc_equity.tolist(),
                        mode="lines", name="MC Base",
                        line=dict(color="rgba(33,150,243,0.5)", dash="dash"),
                        hovertemplate="Trade #%{x}<br>MC Predicted Equity: $%{y:,.2f}<extra>MC Base</extra>",
                    ))

                # Actual equity
                act_sorted = actual.sort_values("entry_time")
                act_equity = INITIAL_BALANCE + act_sorted["pnl_usd"].cumsum()
                fig.add_trace(go.Scatter(
                    x=act_sorted["entry_time"], y=act_equity,
                    mode="lines", name="Actual",
                    line=dict(color="#4CAF50", width=2),
                    hovertemplate="Date: %{x}<br>Actual Equity: $%{y:,.2f}<extra>Actual</extra>",
                ))

                fig.update_layout(
                    title=f"{mc_strategy} {mc_symbol} — Predicted vs Actual",
                    template="plotly_dark", height=400,
                )
                st.plotly_chart(fig, key="mc_pred_vs_real")
        else:
            st.info(f"No live trades found for {mc_strategy} {mc_symbol}. Sync VPS data first.")
else:
    st.info("Run a Monte Carlo backtest first, then compare with live performance here.")
