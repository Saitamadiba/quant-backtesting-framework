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

## Project Status

This framework powers live trading bots deployed on a VPS. Strategies run 24/7 on BTC and ETH with automated Telegram reporting. The optimization and analytics infrastructure shown here is the same tooling used for strategy development, validation, and monitoring in production.
