# Quantitative Trading Framework

End-to-end backtesting, optimization, and analytics framework for systematic crypto trading strategies. Built for BTC/ETH across multiple timeframes (15m, 1h, 4h) with a focus on rigorous out-of-sample validation and risk management.

In plain terms: this system tests trading strategies against historical data, finds the best settings for those strategies, and then validates that the settings actually work on data the system has never seen before &mdash; not just on the data it was trained on. This distinction (in-sample vs. out-of-sample) is the single most important concept in quantitative trading, because strategies that only work on past data they were fitted to are worthless in live markets.

> **Note:** Strategy signal generation logic has been removed from this public repository to protect proprietary IP. The framework architecture, optimization engine, analytics dashboard, and all supporting infrastructure are fully included.

## Architecture

```
backtrader_framework/
  strategies/         Base strategy class + strategy stubs (Backtrader)
  indicators/         FVG detector, sweep detector, session tracker
  optimization/       Walk-forward optimization, Bayesian tuning, Monte Carlo,
                      SHAP analysis, portfolio optimization, stress testing
  data/               DuckDB OHLCV manager, data feeds, candle store
  runners/            Single backtest & strategy comparison runners
  config/             Global settings (sessions, indicators, timeframes)

dashboard/            16-page Streamlit analytics application
  pages/              Overview, Deep Dive, Trade Journal, Equity Curves,
                      Session Analysis, Monthly Performance, ML Training,
                      Monte Carlo, Portfolio, Meta Strategy, SHAP,
                      Bayesian Tuning, Stress Testing, Cross-Asset Robustness
  components/         Plotly charts, KPI cards, filters, WFO tab
  data/               Binance API helpers, VPS sync, data loading, schema normalization
```

## Key Components

### Walk-Forward Optimization Engine (`wfo_engine.py`)

Walk-forward optimization (WFO) is a method for testing trading strategies that prevents overfitting &mdash; the common trap where a strategy looks great on historical data but fails in live trading because it memorized past patterns instead of learning generalizable rules. WFO works by repeatedly splitting data into a "training" window (where parameters are optimized) and a "test" window (where the optimized parameters are validated on unseen data). Only the test-window results count toward the final performance assessment.

This engine supports regime-adaptive optimization with rolling or anchored windows:

- **Timeframe-scaled configuration** &mdash; `WFOConfig.for_timeframe()` automatically scales window sizes by timeframe. A 15-minute chart has 16x more bars than a 4-hour chart for the same calendar period, so window sizes must scale accordingly to cover equivalent real-time durations.
- **Regime detection** &mdash; Markets behave differently when trending, ranging, or volatile. The engine classifies each period into a regime and can optimize separate parameter sets for each, so the strategy adapts its behavior to current market conditions rather than using a one-size-fits-all setting.
- **Grid search modes** &mdash; Full grid (test every combination), random sampling (test a representative subset), or Bayesian optimization via Optuna (intelligently focus on promising regions of the parameter space).
- **Transaction cost modeling** &mdash; Every simulated trade deducts realistic costs for spread (the gap between buy and sell prices), commission (exchange fees), and slippage (price movement between order and fill). Without this, backtests overstate profitability.
- **Statistical validation** &mdash; Overfit ratio (how much worse test results are vs. training results), Sharpe significance testing (whether risk-adjusted returns are statistically meaningful or just luck), and Monte Carlo permutation tests (shuffling trade order to check if results depend on sequence).

### WFO: Practical Lessons and Performance Tuning

Building and running walk-forward optimization on high-frequency data (15m candles, 180K+ bars &mdash; roughly a year of price data sampled every 15 minutes) exposed several non-obvious issues. This section documents the problems encountered and the solutions that brought WFO runtime from 18+ hours down to minutes.

#### Problem 1: Inflated Losses

Early WFO runs showed significantly worse loss metrics than the live bot &mdash; trades that should have been small losses or breakeven appeared as full-size losers. Root causes:

- **Missing exit logic in the adapter.** The live bot uses trailing stops (moving the stop-loss in the direction of profit to lock in gains), partial take-profits (closing a portion of the position at intermediate targets), and time-based exits (closing stale trades that haven't moved). These mechanisms collectively rescue losing trades or protect partial gains. The WFO trade simulator initially only modeled hard SL/TP exits (stop-loss or take-profit, nothing in between), so every trade that didn't hit take-profit was recorded as a full stop-loss hit &mdash; producing inflated average loss sizes.
- **Transaction costs applied twice.** An early bug deducted trading costs from both the signal generator's entry price and the trade simulator's fill price. In effect, every trade was being charged double the real friction, making the whole system look less profitable than it actually was.
- **Overly granular windows.** Stepping every 100 bars on 15m data (advancing the test window by only ~25 hours each time) created 1,794 overlapping windows. With so many tiny windows, each in-sample optimization had very little data to work with, causing the "best" parameters to overfit to noise &mdash; random fluctuations in a small sample. The result was poor out-of-sample performance that appeared as inflated losses. Widening the step size to 1&ndash;4 weeks of bars reduced the window count to ~60&ndash;120, giving each optimization enough data to find genuinely robust parameters.

#### Problem 2: Strategy Selectivity Layers Not Captured

The live bot doesn't trade every signal it detects &mdash; it runs each potential trade through multiple quality filters before committing capital. The WFO adapter initially implemented only the core signal generation (detecting price sweeps of key levels and structural breaks) but omitted several of these selectivity layers, causing it to take trades the live bot would have rejected:

- **Sweep quality scoring** &mdash; The live bot scores each liquidity sweep on how deep price penetrated beyond the level relative to recent volatility (depth-to-ATR ratio), how strongly price rejected that level (wick quality &mdash; a long "tail" on the candle suggests strong rejection), and whether volume confirmed the move. The adapter initially used a flat depth threshold, which meant it accepted shallow, unconvincing sweeps that the live bot would filter out.
- **Multi-timeframe alignment (MTF)** &mdash; The live bot checks that the trade direction agrees with the higher-timeframe trend. For example, a long signal on the 15-minute chart is only valid if the 4-hour trend is also bullish (EMA50 above EMA200). Without this filter, the adapter generated counter-trend signals &mdash; trying to buy in a downtrend or sell in an uptrend &mdash; which inflated the loss count.
- **Gamma regime gating** &mdash; "Gamma" here refers to how options market-makers hedge their positions, which creates predictable price behavior. In high-gamma environments, price tends to be "pinned" and mean-revert, making directional strategies less effective. The live bot suppresses signals during these choppy, mean-reverting periods. The adapter had no such filter, so it fired signals into unfavorable conditions where the strategy has no edge.
- **Session/killzone enforcement** &mdash; Crypto markets trade 24/7, but liquidity and volatility concentrate during specific windows: the London open (2&ndash;5 AM ET), the New York open (8&ndash;11 AM ET), and their overlap. The live bot only trades during these "killzones" using a session manager with DST-aware timezone handling. The adapter's session logic was simplified, allowing signals during low-liquidity periods where fills are worse and false breakouts more common.

The fix was to progressively add these layers to the adapter as soft-scoring components. Each layer contributes a portion of an overall confidence score: sweep depth (0&ndash;0.40), structure bias (0&ndash;0.25), higher-timeframe alignment (0&ndash;0.20), and structure confidence (0&ndash;0.15). The maximum possible score is 1.0, and the `min_confidence` parameter (itself optimizable during WFO) controls the selectivity threshold &mdash; how many of these layers must agree before a signal is accepted.

#### Problem 3: Runtime &mdash; Making WFO Finish in Minutes

A full grid of 576 parameter combinations across 1,794 windows on 180K bars takes 18+ hours. That's 576 settings being tested against nearly 1,800 slices of data, each requiring trade simulation. Four levers bring this down to minutes:

**1. Cut the grid.** Testing all 576 combinations exhaustively is unnecessary &mdash; most of the parameter space is uninteresting. Two alternatives:

- *Random search:* Sample 50&ndash;150 combinations randomly. Research shows random search finds near-optimal settings with a fraction of the evaluations because it explores diverse regions of the space rather than marching through a uniform grid.
- *Two-stage search:* Run a coarse grid first (wide spacing between values), identify the top 5&ndash;10 performing regions, then do a fine-grained search only within those promising areas.

Either approach delivers a 4&ndash;20x speedup. The engine supports this natively via `grid_mode='random'` and `random_samples=150` in `WFOConfig`.

**2. Widen window / step sizes.** Stepping every 100 bars on 15m data means re-running the entire optimization process every 25 hours of market time &mdash; far too frequently. Each step produces only marginally different data, so most of the computation is redundant. Pragmatic WFO settings for 15m crypto:

| Parameter | Too Granular | Recommended | What This Means |
|-----------|-------------|-------------|-----------------|
| Step forward | 100 bars (25h) | 1&ndash;4 weeks of bars | How often you re-optimize |
| OOS window | 100 bars (25h) | 1&ndash;4 weeks | How much unseen data you test on |
| IS window | 500 bars (5 days) | 3&ndash;12 months | How much training data you optimize on |

This produces dozens of windows instead of thousands &mdash; a 10&ndash;50x speedup. The factory method `WFOConfig.for_timeframe()` applies these scaling factors automatically based on the chosen timeframe.

**3. Precompute indicators once.** Technical indicators (moving averages, ATR, RSI, etc.) are mathematical transformations of raw price data. If these are recalculated for every window or every parameter combination, the same arithmetic runs thousands of times over the same data. The correct pattern:

- Compute all indicator columns once in pandas/NumPy over the full dataset before the optimization loop starts
- For each window, slice the pre-computed arrays (essentially free with NumPy views &mdash; no data is copied)
- Run only the signal logic (the lightweight "does this bar meet entry criteria?" check) per parameter set

The engine's `IndicatorEngine.add_all()` handles the one-time computation. Strategy adapters receive a fully-prepared DataFrame and should never recalculate indicators internally.

**4. Parallelize parameter combos.** Modern computers have multiple CPU cores (typically 4&ndash;16), but by default Python uses only one. The 576 parameter combinations are completely independent of each other &mdash; testing one set of parameters doesn't affect the results of another &mdash; making this an ideal candidate for parallel execution using `multiprocessing`. On an 8-core machine, this yields a near-8x speedup. Combined with random search (150 combos instead of 576) and wider windows (60 windows instead of 1,794), total runtime drops from 18 hours to 5&ndash;15 minutes.

### Strategy Adapter Pattern (`strategy_adapters/`)

Strategy adapters are lightweight, stateless signal generators that work on raw pandas DataFrames. "Stateless" means they don't remember anything between calls &mdash; given the same price data and parameters, they always produce the same signals. This makes them safe to run in parallel and easy to test.

```python
class StrategyAdapter(ABC):
    def get_param_space(self) -> List[ParamSpec]: ...
    def generate_signals(self, df, params, start, end) -> List[Signal]: ...
```

Each adapter defines its own parameter space (what settings can be tuned and their valid ranges) and signal generation logic independently of the backtesting engine. This separation enables fast vectorized optimization without Backtrader overhead &mdash; signals are generated using array operations on entire DataFrames rather than processing one candle at a time.

### Monte Carlo Simulation (`pages/9_Monte_Carlo_Backtest.py`)

Monte Carlo simulation answers the question: "Given the trades my strategy produces, what range of outcomes could I realistically expect?" It works by randomly reshuffling the order of trades thousands of times to see how much the final result depends on the specific sequence versus the underlying edge.

Three data source modes:
- **Real Trades** &mdash; Reshuffles actual bot trade PnLs synced from the live VPS server
- **Synthetic Backtest** &mdash; Runs a single-pass backtest, then reshuffles the resulting trades
- **WFO OOS Trades** (recommended) &mdash; Uses only out-of-sample trades from walk-forward validation, ensuring the simulation reflects genuinely forward-tested performance rather than overfitted results

10,000 simulations with optional block bootstrap (reshuffling in chunks rather than individual trades to preserve any serial correlation &mdash; the tendency for winning or losing streaks). Outputs probability distributions for returns, drawdowns, and ruin risk (the chance of losing a catastrophic percentage of capital).

### Analytics Dashboard (Streamlit, 16 pages)

A web-based analytics application for exploring strategy performance from every angle. Each page focuses on a different aspect of the analysis:

| Page | Description |
|------|-------------|
| Overview | Multi-strategy KPI dashboard with real-time VPS sync |
| Strategy Deep Dive | Per-strategy analysis with regime breakdowns |
| Trade Journal | Filterable trade log with MFE/MAE analysis (how far each trade moved in your favor vs. against you before closing) |
| Equity Curves | Multi-strategy equity with strategy changelog overlays |
| Session Analysis | Performance by trading session (Asia/London/NY) |
| Monthly Performance | Calendar heatmaps and monthly P&L tracking |
| ML Training | Feature engineering and model training pipeline |
| Monte Carlo Backtest | Probability distributions from trade reshuffling |
| Portfolio | Multi-strategy portfolio construction and correlation |
| Meta Strategy | Dynamic strategy selection based on market regime (automatically picks which strategy to run based on current conditions) |
| SHAP Analysis | Feature importance and model interpretability (which inputs matter most for predictions and why) |
| Bayesian Tuning | Optuna-based hyperparameter optimization (smart parameter search that learns from previous trials) |
| Stress Testing | Synthetic adverse scenarios and tail risk analysis (what happens during extreme market events) |
| Cross-Asset Robustness | Walk-forward validation across assets and timeframes (does the strategy work on ETH too, not just BTC?) |

### Data Pipeline

- **DuckDB** for high-performance local OHLCV storage &mdash; stores Open, High, Low, Close, Volume price data for BTC/ETH at 15m/1h/4h intervals. DuckDB is an embedded analytical database optimized for fast column-based queries on large datasets.
- **Binance REST API** for real-time market data (public endpoints, no API key required)
- **VPS sync** for live trade database retrieval over SSH &mdash; pulls the latest trade records from the remote server where bots run 24/7
- **Schema normalization** across heterogeneous trade database formats &mdash; different bots store trades in different table structures; this layer translates them into a common format for unified analysis

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Backtesting | Backtrader, custom vectorized engine |
| Data | DuckDB, SQLite, pandas, numpy |
| Optimization | Optuna (Bayesian), scipy, custom WFO engine |
| ML | scikit-learn, XGBoost, SHAP |
| Dashboard | Streamlit, Plotly |
| APIs | Binance REST/WebSocket, yfinance |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file for VPS connectivity (optional &mdash; needed only for live trade sync):
```
VPS_HOST=your.vps.ip
VPS_PORT=22
VPS_USER=trader
```

Populate DuckDB with OHLCV data:
```python
from backtrader_framework.data.duckdb_manager import DuckDBManager
db = DuckDBManager()
db.backfill("BTC", "15m", days=365)
```

Launch the dashboard:
```bash
cd dashboard
streamlit run app.py
```

## Live Performance Snapshot

One of the strategies in this framework &mdash; **Liquidity Raid on BTC (15m)** &mdash; has been running live on a VPS since late November 2025. Below are sanitized performance metrics (strategy logic is not disclosed).

| Metric | Value | What It Means |
|--------|-------|---------------|
| Period | Nov 2025 &ndash; Feb 2026 (~3 months) | Duration of live trading |
| Closed Trades | 88 | Total completed round-trip trades |
| Win Rate | 56.8% | Percentage of trades that made money |
| Profit Factor | 1.67 | Gross profits / gross losses &mdash; above 1.0 means profitable overall |
| Mean R-Multiple | +0.24R | Average return per unit of risk &mdash; risking $100 per trade yields $24 average profit |
| Annualized Sharpe | 3.41 | Risk-adjusted return metric &mdash; above 2.0 is considered excellent |
| Max Drawdown | $556 | Largest peak-to-trough decline &mdash; the worst losing streak in dollar terms |
| Cumulative PnL | +$2,144 | Total profit after all costs |

**Direction split:** Long 59% / Short 41% of trades &mdash; profitable in both directions, meaning the strategy isn't just riding a bull market.

**Exit discipline breakdown:**

| Exit Type | Trades | PnL | What Happened |
|-----------|--------|-----|---------------|
| Take Profit | 18 (20%) | +$3,193 | Price hit the profit target |
| Trailing Stop | 14 (16%) | +$1,532 | Locked in gains as price moved favorably, then exited when it pulled back |
| Time Exit | 13 (15%) | +$489 | Trade was open too long without reaching target &mdash; closed to free up capital |
| Stop Loss | 43 (49%) | &minus;$3,071 | Price moved against the trade and hit the predefined loss limit |

Losses are capped by hard stop-losses, while trailing stops and time-based exits contribute significant edge &mdash; a hallmark of disciplined risk management. The key insight: nearly half of all trades are losers, but the winners are larger than the losers, producing net profit.

**Monthly progression:**

| Month | Trades | Win Rate | PnL |
|-------|--------|----------|-----|
| Nov 2025 | 2 | 50% | +$99 |
| Dec 2025 | 51 | 53% | +$435 |
| Jan 2026 | 25 | 64% | +$707 |
| Feb 2026 | 10 | 60% | +$903 |

Win rate and per-trade expectancy have improved over time as the walk-forward optimization framework feeds parameter updates back into the live bot &mdash; a concrete example of the system's optimization loop working in production.

## Project Status

This framework powers live trading bots deployed on a VPS (virtual private server running 24/7 in a data center). Strategies run continuously on BTC and ETH with automated Telegram reporting for trade alerts, daily performance summaries, and system health monitoring. The optimization and analytics infrastructure shown here is the same tooling used for strategy development, validation, and monitoring in production.
