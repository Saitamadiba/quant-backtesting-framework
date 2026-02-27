# Quantitative Trading Framework

A risk-first backtesting and portfolio analytics platform for systematic crypto strategies. Built to answer one question: **does this strategy have a genuine, statistically validated edge — or is it overfitted noise?**

> **Note:** Strategy signal generation logic has been removed from this public repository to protect proprietary IP. The framework architecture, optimization engine, analytics dashboard, and all supporting infrastructure are fully included.

---

## The Problem

Most retail backtests are worthless. A strategy is optimized on historical data, produces an impressive equity curve, and then loses money in production. The root cause is almost always the same: **in-sample overfitting disguised as performance**.

A strategy trained on 2024 BTC price action will naturally "learn" patterns specific to that year — the ETF approval rally, the halving anticipation, the summer consolidation. These patterns won't repeat. A naive backtest that trains and tests on the same data reports high Sharpe ratios because it is literally grading its own homework.

This framework exists to eliminate that failure mode through four layers of defense:

1. **Walk-forward optimization** — Parameter tuning and performance evaluation happen on strictly separated data. The strategy never sees its test data during optimization.
2. **Monte Carlo simulation** — Trade sequences are reshuffled thousands of times to separate genuine edge from favorable sequencing.
3. **Regime-conditional validation** — Performance is measured per market regime (trending, ranging, volatile), exposing strategies that only work in one environment.
4. **Stress testing under synthetic tail events** — Historical crash scenarios (Luna/3AC, FTX collapse) are scaled and applied to test whether the strategy survives conditions it has never encountered.

---

## Why Walk-Forward Optimization

Static backtesting optimizes parameters on a dataset and then reports performance on that same dataset. Walk-forward optimization (WFO) does something fundamentally different: it repeatedly splits time-series data into a training window (in-sample) and a validation window (out-of-sample), optimizes on the training data, and evaluates only on the unseen validation window.

**Why this matters:**

- **Overfitting detection is built in.** The overfit ratio (IS performance / OOS performance) directly measures how much the strategy degrades on unseen data. A ratio near 1.0 means the parameters generalize; a ratio above 2.0 means the strategy memorized noise.
- **Parameters adapt to structural change.** Rolling windows allow the optimizer to discover that the best parameters for a trending market (Q4 2024) differ from those in a ranging market (Q3 2024). A single optimization across both periods would find a compromise that works poorly in either.
- **Statistical significance is testable.** Each OOS window produces independent performance samples. With 30+ windows, Sharpe ratios can be tested for significance using t-tests and confidence intervals rather than accepted on faith.

### Implementation

The engine (`wfo_engine.py`) supports:

- **Timeframe-scaled windows** — `WFOConfig.for_timeframe()` automatically scales IS/OOS window sizes. A 15m chart has 16x more bars than a 4h chart for the same calendar period; window sizes scale accordingly.
- **Regime detection** — Each window is classified as trending (up/down), ranging, or volatile based on ADX, ATR percentile, and EMA alignment. Parameter sets are tracked per regime.
- **Grid search modes** — Full grid, random sampling (50-150 combinations for 4-20x speedup), or Bayesian optimization via Optuna.
- **Transaction cost modeling** — Every simulated trade deducts spread (0.01%), commission (0.055%), and slippage (0.01%), calibrated per asset. Without this, backtests overstate profitability by 15-40%.
- **Combinatorial purged cross-validation** — Embargo zones between training and test sets prevent information leakage from overlapping trades.

### Performance

Full WFO on 180K bars (1 year of 15m data) with 576 parameter combinations runs in 5-15 minutes through four optimizations: precomputed indicators (compute once, slice by window), random grid sampling, widened window steps (weeks instead of hours), and multiprocessing across CPU cores. Initial implementation took 18+ hours.

---

## Why Monte Carlo Simulation

A single backtest produces a single equity curve — one path through history. But that path is heavily influenced by trade ordering. A strategy with 55% win rate and 1.5:1 reward-to-risk can produce equity curves ranging from +40% to -15% simply by reshuffling the order of the same trades. The question isn't "what happened?" but **"what distribution of outcomes is consistent with this strategy's edge?"**

Monte Carlo simulation (10,000 iterations with optional block bootstrap) answers this by constructing the full probability distribution of outcomes.

**Key risk metrics computed:**

| Metric | What It Measures |
|--------|-----------------|
| **Value at Risk (95%)** | The loss threshold exceeded only 5% of the time — a standard institutional risk measure |
| **Conditional VaR (CVaR)** | Expected loss *given* that VaR is breached — captures tail severity, not just tail probability |
| **Max Drawdown Distribution** | Percentile bands (p50, p75, p90, p95) for peak-to-trough decline |
| **Probability of Ruin** | Likelihood of account drawdown exceeding a catastrophic threshold |
| **Sharpe Ratio CI** | Confidence interval on risk-adjusted returns — distinguishes a 2.0 Sharpe from a noisy 0.8 |
| **Win Rate CI** | Wilson binomial confidence interval — a 60% win rate on 30 trades has a very different CI than on 300 trades |

**Why block bootstrap?** Standard Monte Carlo assumes trades are independent. In practice, crypto strategies exhibit serial correlation — winning streaks during trends, losing streaks during chop. Block bootstrap preserves these autocorrelation structures by reshuffling in chunks rather than individual trades, producing more realistic tail estimates.

**Three data source modes:**

- **WFO OOS Trades** (recommended) — Only out-of-sample trades from walk-forward validation. This is the gold standard: Monte Carlo on already-validated data.
- **Real Trades** — Reshuffles actual PnLs from live VPS bots.
- **Synthetic Backtest** — Single-pass backtest followed by reshuffling.

---

## Why Regime Detection

A strategy that averages +0.3R per trade may be earning +0.8R in trending markets and -0.4R in ranging markets. Aggregate metrics hide this. Regime-conditional analysis exposes it.

**Market regime classification:**

| Regime | Detection Criteria | Implication |
|--------|-------------------|-------------|
| **Trending (up/down)** | ADX > 30, EMA50/200 alignment | Momentum strategies have edge; mean-reversion strategies lose |
| **Ranging** | ADX <= 30 | Mean-reversion works; trend-following generates whipsaws |
| **Volatile** | ATR > 1.8x 20-bar average | Wider stops required; position sizing must decrease |

**IV/DVOL regime gating:** For crypto assets, implied volatility (DVOL) provides forward-looking regime information from the options market. The FVG strategy uses DVOL direction gating — filtering long-only setups in low-IV environments and short-only in high-IV environments — a signal that backward-looking indicators like ATR cannot capture.

**Why this matters for capital allocation:** A portfolio running four strategies should not allocate equally to all four during a volatile regime if three of them have negative expectancy in volatility. Regime detection enables dynamic allocation — increasing exposure to strategies with demonstrated edge in the current environment and reducing exposure to those without.

---

## Stress Testing and Tail Risk

Backtests only cover market conditions that have already occurred. Stress testing asks: **what happens under conditions that haven't?**

### Synthetic Scenario Engine

Five scenario types, calibrated against historical crypto crashes:

| Scenario | Mechanics | Calibration |
|----------|-----------|-------------|
| **Historical Crash Scaling** | Reproduces actual crash dynamics at 1x-3x magnitude | May 2021 (-53%), FTX Nov 2022 (-27%), Luna/3AC Jun 2022 (-37%) |
| **Volatility Spike** | ATR multiplied by 2x-5x for sustained periods | Captures environments where normal stop distances are insufficient |
| **Flash Crash** | -10% to -15% in 3 bars with 50-70% recovery | Tests whether the strategy is stopped out before the recovery |
| **Prolonged Drawdown** | 200-bar downward drift with dead-cat bounces | Simulates multi-week bear conditions with false recovery signals |
| **V-Shaped Recovery** | Crash followed by mirror recovery | Tests whether the strategy re-enters after a drawdown |

### Survival Scoring

Each strategy receives a composite survival score (0-100) per scenario:

| Component | Weight | Scoring |
|-----------|--------|---------|
| Max drawdown depth | 40% | 100 at -3R, 0 at -15R |
| Consecutive losses | 20% | 100 at <3, 0 at >=10 |
| Recovery speed | 20% | Trades needed to restore pre-drawdown equity |
| Net R during event | 20% | 100 at >=0R, 0 at <=-5R |

A strategy that scores above 70 across all scenario types at 2x magnitude demonstrates genuine robustness. A strategy that scores 90 on historical data but 20 under volatility spikes has a hidden fragility.

---

## Portfolio Construction

Individual strategy validation is necessary but insufficient. Portfolio-level risk requires understanding how strategies interact.

**Allocation methods:**

| Method | Approach |
|--------|----------|
| **Risk Parity** | Equal risk contribution — each strategy contributes the same marginal portfolio volatility |
| **Mean-Variance (Markowitz)** | Maximum Sharpe allocation with Ledoit-Wolf shrinkage for covariance stability |
| **Kelly Criterion** | Optimal geometric growth sizing, fractional (0.25x) for practical drawdown limits |
| **Max Sharpe** | Maximize portfolio-level risk-adjusted returns subject to allocation constraints |

**Correlation analysis:** Pairwise strategy correlations are computed on aligned daily equity curves. High correlation between two strategies means they draw down together — the diversification benefit is illusory. The framework surfaces this through a correlation matrix and diversification ratio.

---

## Architecture

```
backtrader_framework/
  strategies/         Strategy base class and adapter pattern
  indicators/         FVG detector, liquidity sweep detector, session tracker
  optimization/       WFO engine, Monte Carlo, SHAP analysis, Bayesian tuning,
                      portfolio optimization, stress testing, regime detection,
                      drawdown analysis, CPCV, cross-asset robustness
  data/               DuckDB OHLCV manager, data feeds, validation
  analyzers/          Sortino, Calmar, Omega ratio analyzers
  config/             Sessions, kill zones, indicator defaults

dashboard/            16-page Streamlit analytics application
  pages/              Overview, Strategy Deep Dive, Trade Journal, Equity Curves,
                      Session Analysis, Monthly Performance, ML Training,
                      Monte Carlo, Deploy, Portfolio, Meta Strategy, SHAP,
                      Bayesian Tuning, Stress Testing, Cross-Asset Robustness
  components/         Plotly charts, KPI cards, filters
  data/               VPS sync, data loading, schema normalization
```

### Strategy Adapter Pattern

Strategies are implemented as stateless signal generators operating on pandas DataFrames. Given the same data and parameters, they produce identical signals — making them safe for parallel execution and deterministic testing.

```python
class StrategyAdapter(ABC):
    def get_param_space(self) -> List[ParamSpec]: ...
    def generate_signals(self, df, params, start, end) -> List[Signal]: ...
```

Each adapter defines its parameter space and signal logic independently of the backtesting engine. This separation enables vectorized optimization without event-driven overhead.

### Data Pipeline

- **DuckDB** — Embedded analytical database for local OHLCV storage (BTC/ETH at 15m/1h/4h, ~1 year)
- **Binance REST API** — Real-time market data (public endpoints)
- **VPS sync** — SSH-based retrieval of live trade databases from production servers
- **Schema normalization** — Unified trade format across heterogeneous bot database schemas

---

## Analytics Dashboard

A 16-page Streamlit application for strategy analysis, optimization, and monitoring:

| Page | Purpose |
|------|---------|
| **Overview** | Multi-strategy KPI dashboard with real-time VPS sync |
| **Strategy Explainer** | Signal generation rules, parameter space, confidence scoring |
| **Strategy Deep Dive** | WFO results with regime-specific breakdowns |
| **Trade Journal** | Filterable trade log with MFE/MAE analysis |
| **Equity Curves** | Multi-strategy equity with changelog overlays |
| **Session Analysis** | Performance by trading session (Asia/London/NY) |
| **Monthly Performance** | Calendar heatmaps and monthly P&L tracking |
| **ML Training** | Feature engineering and model training pipeline |
| **Monte Carlo** | Probability distributions, VaR/CVaR, equity fan charts |
| **Deploy** | Bot deployment and service management |
| **Portfolio** | Multi-strategy allocation and correlation analysis |
| **Meta Strategy** | Dynamic strategy selection by market regime |
| **SHAP Analysis** | Feature importance and model interpretability |
| **Bayesian Tuning** | Optuna hyperparameter optimization |
| **Stress Testing** | Synthetic scenarios and survival scoring |
| **Cross-Asset Robustness** | Walk-forward validation across BTC/ETH/NQ |

---

## Live Performance

One of the strategies — **Liquidity Raid on BTC (15m)** — has been running live since November 2025. Below are sanitized metrics (strategy logic is not disclosed).

| Metric | Value |
|--------|-------|
| Period | Nov 2025 - Feb 2026 (~3 months) |
| Closed Trades | 88 |
| Win Rate | 56.8% |
| Profit Factor | 1.67 |
| Mean R-Multiple | +0.24R |
| Annualized Sharpe | 3.41 |
| Max Drawdown | $556 |
| Cumulative PnL | +$2,144 |

**Direction split:** Long 59% / Short 41% — profitable in both directions, not dependent on market trend.

**Exit discipline:**

| Exit Type | Trades | PnL |
|-----------|--------|-----|
| Take Profit | 18 (20%) | +$3,193 |
| Trailing Stop | 14 (16%) | +$1,532 |
| Time Exit | 13 (15%) | +$489 |
| Stop Loss | 43 (49%) | -$3,071 |

Nearly half of all trades are losers. The edge comes from asymmetric payoffs — winners are larger than losers — combined with trailing stops and time-based exits that extract value from trades that don't reach full targets. This is the expected distribution for a positive-expectancy system.

**Monthly progression:**

| Month | Trades | Win Rate | PnL |
|-------|--------|----------|-----|
| Nov 2025 | 2 | 50% | +$99 |
| Dec 2025 | 51 | 53% | +$435 |
| Jan 2026 | 25 | 64% | +$707 |
| Feb 2026 | 10 | 60% | +$903 |

Win rate and per-trade expectancy have improved as walk-forward optimization feeds parameter updates back into the live bot.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Backtesting | Backtrader, custom vectorized engine |
| Data | DuckDB, SQLite, pandas, NumPy |
| Optimization | Optuna (Bayesian), SciPy, custom WFO engine |
| ML | scikit-learn, XGBoost, SHAP |
| Dashboard | Streamlit, Plotly |
| APIs | Binance REST/WebSocket |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional `.env` for live trade sync:
```
VPS_HOST=your.vps.ip
VPS_PORT=22
VPS_USER=trader
```

Populate local OHLCV data:
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

---

## Project Status

This framework powers live trading bots deployed on a VPS running 24/7. Four strategy families (8 bots) run continuously on BTC and ETH with automated Telegram reporting for trade alerts, daily performance summaries, and system health monitoring. The optimization and analytics infrastructure shown here is the same tooling used for strategy development, validation, and monitoring in production.
