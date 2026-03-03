# Quantitative Trading Framework

A risk-first backtesting and portfolio analytics platform for systematic crypto strategies. Built to answer one question: **does this strategy have a genuine, statistically validated edge — or is it overfitted noise?**

> **Note:** Strategy signal generation logic has been removed from this public repository to protect proprietary IP. The framework architecture, optimization engine, analytics dashboard, and all supporting infrastructure are fully included.

---

## Documentation

The full strategy validation report — covering WFO architecture, statistical testing, Monte Carlo analysis, regime detection, and live-faithful adapter design — is available as an interactive Jupyter notebook:

**[WFO_Strategy_Validation_Report.ipynb](WFO_Strategy_Validation_Report.ipynb)**

The notebook includes code snippets, engine quality metrics, and architectural details for every component of the validation pipeline.

---

## Quick Start

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

## Project Status

This framework powers live trading bots deployed on a VPS running 24/7. Four strategy families (8 bots) run continuously on BTC and ETH with automated Telegram reporting for trade alerts, daily performance summaries, and system health monitoring. The optimization and analytics infrastructure shown here is the same tooling used for strategy development, validation, and monitoring in production.
