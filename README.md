# Quantitative Trading Framework

End-to-end backtesting, optimization, and analytics framework for systematic crypto trading strategies. Built for BTC/ETH across multiple timeframes (15m, 1h, 4h) with a focus on rigorous out-of-sample validation and risk management.

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

Regime-adaptive walk-forward optimization with rolling or anchored windows:

- **Timeframe-scaled configuration** &mdash; `WFOConfig.for_timeframe()` automatically scales window sizes by timeframe (15m gets 16x the bars of 4h for equivalent time coverage)
- **Regime detection** &mdash; Classifies market conditions (trending/ranging/volatile) and optimizes parameters per regime
- **Grid search modes** &mdash; Full grid, random sampling, or Bayesian optimization via Optuna
- **Transaction cost modeling** &mdash; Configurable spread, commission, and slippage per asset
- **Statistical validation** &mdash; Overfit ratio, Sharpe significance testing, Monte Carlo permutation tests

### WFO: Practical Lessons and Performance Tuning

Building and running walk-forward optimization on high-frequency data (15m, 180K+ bars) exposed several non-obvious issues. This section documents the problems encountered and the solutions that brought WFO runtime from 18+ hours down to minutes.

#### Problem 1: Inflated Losses

Early WFO runs showed significantly worse loss metrics than the live bot. Root causes:

- **Missing exit logic in the adapter.** The live bot uses trailing stops, partial take-profits, and time-based exits that collectively salvage losing trades or lock in partial gains. The WFO trade simulator initially only modeled hard SL/TP exits, producing inflated average loss sizes.
- **Transaction costs applied twice.** An early bug deducted costs from both the signal generator's entry price and the trade simulator's fill price, doubling effective friction on every trade.
- **Overly granular windows.** Stepping every 100 bars on 15m data (~25 hours) created 1,794 overlapping windows. Each IS window's "best" parameters overfitted to noise, producing poor OOS performance that looked like inflated losses. Widening the step size to 1&ndash;4 weeks of bars cut window count to ~60&ndash;120 and stabilized OOS results.

#### Problem 2: Strategy Selectivity Layers Not Captured

The WFO adapter initially implemented only the core signal generation (sweep detection + BOS confirmation) but omitted several selectivity layers present in the live bot:

- **Sweep quality scoring** &mdash; The live bot scores each sweep on depth-to-ATR ratio, wick rejection quality, and volume confirmation. The adapter initially used a flat depth threshold, admitting low-quality sweeps that the live bot would reject.
- **Multi-timeframe alignment** &mdash; The live bot checks that the signal direction agrees with the higher-timeframe EMA trend (e.g., 4H EMA50 > EMA200 for longs). Without this filter the adapter generated counter-trend signals that inflated the loss count.
- **Gamma regime gating** &mdash; In production, signals are suppressed during high-gamma (choppy/mean-reverting) regimes. The adapter had no regime filter, so it fired signals into unfavorable conditions.
- **Session/killzone enforcement** &mdash; The adapter's killzone logic was simplified relative to the live bot's session manager, which tracks Asia/London/NY boundaries with DST-aware timezone handling.

The fix was to progressively add these layers to the adapter as soft-scoring components: sweep depth contributes 0&ndash;0.40, structure bias 0&ndash;0.25, HTF alignment 0&ndash;0.20, and structure confidence 0&ndash;0.15. The `min_confidence` parameter (optimizable) then controls overall selectivity.

#### Problem 3: Runtime &mdash; Making WFO Finish in Minutes

A full grid of 576 parameter combinations across 1,794 windows on 180K bars takes 18+ hours. Four levers bring this down to minutes:

**1. Cut the grid.** 576 combos is large for walk-forward. Use random search (50&ndash;150 samples) or a two-stage approach: coarse grid first, then refine around the top 5&ndash;10 regions. This alone is a 4&ndash;20x speedup. The engine supports this via `grid_mode='random'` and `random_samples=150` in `WFOConfig`.

**2. Widen window / step sizes.** Stepping every 100 bars on 15m means re-optimizing every 25 hours &mdash; far too granular. Pragmatic WFO settings for 15m crypto:

| Parameter | Too Granular | Recommended |
|-----------|-------------|-------------|
| Step forward | 100 bars (25h) | 1&ndash;4 weeks of bars |
| OOS window | 100 bars (25h) | 1&ndash;4 weeks |
| IS window | 500 bars (5 days) | 3&ndash;12 months |

This produces dozens of windows instead of thousands &mdash; a 10&ndash;50x speedup. `WFOConfig.for_timeframe()` applies these scaling factors automatically.

**3. Precompute indicators once.** If indicators are recalculated per window or per parameter combination, runtime explodes. The correct pattern:

- Compute all indicator columns once in pandas/NumPy over the full dataset
- Slice arrays per window (zero-copy with NumPy views)
- Run signal logic cheaply per parameter set on pre-sliced arrays

The engine's `IndicatorEngine.add_all()` runs once before the window loop. Strategy adapters receive a pre-computed DataFrame and should never recalculate indicators internally.

**4. Parallelize parameter combos.** 576 combinations is ideal for `multiprocessing`. If the engine runs single-threaded, most CPU cores sit idle. On an 8-core machine, parallelizing the IS grid search yields a near-8x speedup. Combined with random search (150 combos) and wider windows, total runtime drops from 18 hours to 5&ndash;15 minutes.

### Strategy Adapter Pattern (`strategy_adapters/`)

Lightweight, stateless signal generators that work on raw pandas DataFrames:

```python
class StrategyAdapter(ABC):
    def get_param_space(self) -> List[ParamSpec]: ...
    def generate_signals(self, df, params, start, end) -> List[Signal]: ...
```

Each adapter defines its parameter space and signal generation independently of the backtesting engine, enabling fast vectorized optimization without Backtrader overhead.

### Monte Carlo Simulation (`pages/9_Monte_Carlo_Backtest.py`)

Three data source modes for Monte Carlo reshuffling:
- **Real Trades** &mdash; Reshuffles actual bot trade PnLs synced from VPS
- **Synthetic Backtest** &mdash; Single-pass backtest, then reshuffles
- **WFO OOS Trades** (recommended) &mdash; Uses only out-of-sample trades from walk-forward validation

10,000 simulations with optional block bootstrap to preserve serial correlation. Outputs probability distributions for returns, drawdowns, and ruin risk.

### Analytics Dashboard (Streamlit, 16 pages)

| Page | Description |
|------|-------------|
| Overview | Multi-strategy KPI dashboard with real-time VPS sync |
| Strategy Deep Dive | Per-strategy analysis with regime breakdowns |
| Trade Journal | Filterable trade log with MFE/MAE analysis |
| Equity Curves | Multi-strategy equity with strategy changelog overlays |
| Session Analysis | Performance by trading session (Asia/London/NY) |
| Monthly Performance | Calendar heatmaps and monthly P&L tracking |
| ML Training | Feature engineering and model training pipeline |
| Monte Carlo Backtest | Probability distributions from trade reshuffling |
| Portfolio | Multi-strategy portfolio construction and correlation |
| Meta Strategy | Dynamic strategy selection based on market regime |
| SHAP Analysis | Feature importance and model interpretability |
| Bayesian Tuning | Optuna-based hyperparameter optimization |
| Stress Testing | Synthetic adverse scenarios and tail risk analysis |
| Cross-Asset Robustness | Walk-forward validation across assets and timeframes |

### Data Pipeline

- **DuckDB** for high-performance local OHLCV storage (BTC/ETH, 15m/1h/4h)
- **Binance REST API** for real-time market data (public endpoints)
- **VPS sync** for live trade database retrieval over SSH
- **Schema normalization** across heterogeneous trade database formats

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

| Metric | Value |
|--------|-------|
| Period | Nov 2025 &ndash; Feb 2026 (~3 months) |
| Closed Trades | 88 |
| Win Rate | 56.8% |
| Profit Factor | 1.67 |
| Mean R-Multiple | +0.24R |
| Annualized Sharpe | 3.41 |
| Max Drawdown | $556 |
| Cumulative PnL | +$2,144 |

**Direction split:** Long 59% / Short 41% of trades &mdash; profitable in both directions.

**Exit discipline breakdown:**

| Exit Type | Trades | PnL |
|-----------|--------|-----|
| Take Profit | 18 (20%) | +$3,193 |
| Trailing Stop | 14 (16%) | +$1,532 |
| Time Exit | 13 (15%) | +$489 |
| Stop Loss | 43 (49%) | &minus;$3,071 |

Losses are capped by hard stop-losses, while trailing stops and time-based exits contribute significant edge &mdash; a hallmark of disciplined risk management.

**Monthly progression:**

| Month | Trades | Win Rate | PnL |
|-------|--------|----------|-----|
| Nov 2025 | 2 | 50% | +$99 |
| Dec 2025 | 51 | 53% | +$435 |
| Jan 2026 | 25 | 64% | +$707 |
| Feb 2026 | 10 | 60% | +$903 |

Win rate and per-trade expectancy have improved over time as the walk-forward optimization framework feeds parameter updates back into the live bot.

## Project Status

This framework powers live trading bots deployed on a VPS. Strategies run 24/7 on BTC and ETH with automated Telegram reporting. The optimization and analytics infrastructure shown here is the same tooling used for strategy development, validation, and monitoring in production.
