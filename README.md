# Quantitative Trading Framework

A risk-first research and execution stack for systematic crypto strategies, built
around one question: **does this strategy have a genuine, statistically validated
edge — or is it overfitted noise?**

Most of the candidates I have put through it did not. That is the point. This
repository is the apparatus that decides, and the discipline that makes the
verdict trustworthy.

> **Start here:** **[How I Try To Fool Myself](docs/RESEARCH_METHOD.md)** — a field
> guide to **47 documented research traps**, drawn from roughly sixty
> investigations, the large majority of which refuted their own hypothesis. It is
> the most honest summary of how this framework is actually used.

---

## Scale

| | |
|---|---|
| Strategy engines under walk-forward validation | **8 families** (liquidity-sweep, fair-value-gap, momentum, volatility-edge, and variants) |
| Walk-forward runs on record | **560+** result sets |
| Live books deployed | **40+**, across a three-tier risk architecture |
| Continuous operation | 24/7 on a hardened VPS, cron-driven, with Telegram alerting |
| Research investigations documented | **~60**, each with a pre-registered kill/keep bar |

Signal-generation logic, live configurations, research outputs, and market-data
stores are held in a separate private repository. What is here is the
**framework, the validation machinery, the analytics layer, and the method** —
which is the part that transfers.

---

## What actually makes this rigorous

The validation layer is the substance of the project, not a wrapper around a
backtest loop.

**Overfitting detection.** Probability of Backtest Overfitting (PBO), deflated
Sharpe ratio, and combinatorial purged cross-validation (`cpcv.py`) run on every
walk-forward result. When they report PBO ≈ 0.56 and deflated-Sharpe p ≈ 0.94,
the honest summary is *suggestive, not proven* — and the report says so rather
than leading with the mean.

**Multiplicity control.** Permutation-based family-wise error bars (signed max-t
for directional claims), Bonferroni where appropriate, half-split stability
checks on survivors, and day-clustered standard errors with an effective-n
haircut — because simultaneous same-direction positions are one bet wearing
several name tags.

**Direct controls, not just p-value corrections.** The habit that killed the most
candidates: build the cheapest control that takes the same shape of risk at the
same times *without* the signal. Random-bar brackets, highest-volatility
substitutes, placebo event grids, mirrored brackets, drawdown-matched
counterfactuals. Details and case studies in the
[method guide](docs/RESEARCH_METHOD.md#2-controls-the-cheapest-thing-that-mimics-the-trade-without-the-signal).

**Cost decomposition before any verdict.** Gross-versus-toll is separated on
every candidate. An entry must earn gross above the execution toll or better
fills cannot save it — and the fee-in-R wall is computed *a priori*, per
timeframe, before any code is written.

**Regime awareness.** HMM-based regime detection (`hmm_regime.py`) plus a
five-state classifier, with per-regime performance reported separately and
thresholds calibrated **per timeframe** — a scale-dependent threshold applied at
the wrong sampling interval is a documented failure mode in this codebase, not a
hypothetical one.

**Attribution and portfolio construction.** SHAP-based feature attribution,
Bayesian edge estimation, drawdown analysis, cross-asset robustness screens, and
a portfolio optimizer for allocating across validated strategies.

---

## Risk architecture

Every deployed book sits in exactly one of three tiers, and the tier determines
what risk machinery is mandatory:

| Tier | What it is | Risk requirement |
|---|---|---|
| **1** | Places real exchange orders | Must drive the canonical risk guard — circuit breaker enforced every cycle (daily drawdown, max loss, consecutive losses → flatten and halt), native stop on every order, notional and aggregate-exposure caps, fail-closed on missing config |
| **2** | Paper books on a virtual challenge account | Same ruleset applied to virtual equity, so paper P&L answers "would this pass a funded evaluation" |
| **3** | Record-only shadow detectors | **Dimensionless by design** — R-multiples with a defined stop and target, no sizing, no halts. Bolting position sizing onto a measurement instrument corrupts the measurement |

Two supporting principles that took incidents to learn: a halted bot must keep
**accruing data** (it records what it would have taken, in shadow, so a halt
costs money but not information), and sizing and risk enforcement must consume
*literally the same equity function* — two correct implementations of the same
intent will drift.

Deployment carries its own guards: diff-first deploys that refuse to overwrite a
newer remote copy, checksum verification, dry-run smoke tests, isolated
order-only API keys, and a strict separation between the non-privileged account
the bots run as and the administrative account.

---

## Repository layout

```
backtrader_framework/     # engine, WFO, optimization, statistics, data layer
  optimization/           # PBO, CPCV, permutation bars, Monte Carlo, SHAP,
                          # regime models, portfolio optimizer, Numba kernels
  optimization/
    strategy_adapters/    # live-faithful adapters (interfaces public,
                          # signal logic private)
dashboard/                # Streamlit analytics, trade monitoring, reporting
quant_skills/             # reusable research utilities
docs/RESEARCH_METHOD.md   # the 47-trap method guide
<strategy>/               # per-strategy interface + config scaffolding
```

The **[WFO Strategy Validation Report](WFO_Strategy_Validation_Report.ipynb)** is
an interactive walkthrough of the validation pipeline: architecture, statistical
testing, Monte Carlo analysis, regime detection, and live-faithful adapter design.

---

## Quick start

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Load OHLCV data and read it back:

```python
from backtrader_framework.data.duckdb_manager import DuckDBManager
from backtrader_framework.optimization.data_fetcher import DataFetcher

db = DuckDBManager()
db.initialize_schema()

df = DataFetcher.fetch("BTC", "15m")          # market data
db.insert_ohlcv(df, symbol="BTC", timeframe="15m")

bars = db.get_ohlcv("BTC", "15m", start_date="2024-01-01")
```

Launch the analytics dashboard:

```bash
streamlit run dashboard/app.py
```

An optional `.env` enables live trade sync in the dashboard; see
`dashboard/.env.example` for the shape. No credentials are committed to this
repository, and none are required to run the backtesting or validation layers.

---

## Tech stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Backtesting | Backtrader, plus a custom vectorized engine with Numba kernels |
| Data | DuckDB, SQLite, pandas, NumPy |
| Optimization | Custom walk-forward engine, Optuna (Bayesian), SciPy |
| Statistics | PBO, deflated Sharpe, CPCV, permutation tests, bootstrap, Monte Carlo |
| ML | scikit-learn, XGBoost, SHAP, HMM regime models |
| Dashboard | Streamlit, Plotly |
| Market data | Binance REST/WebSocket |
| Live execution | Bybit (private execution layer) |

---

## A note on what is claimed here

This README makes no performance claim, deliberately. The live books span
profitable, break-even, and deliberately-negative record-only detectors, and
several exist specifically to keep measuring a hypothesis that has already been
refuted — a black box on a plane that has already landed.

What I will claim is the process: pre-registered kill/keep bars fixed before data
arrives, family-wise error control, direct controls alongside the corrections,
cost decomposition before every verdict, explicit era boundaries when live
behaviour changes, and retractions written next to the original claim when a
finding of mine fails.

The [method guide](docs/RESEARCH_METHOD.md) is where that is demonstrated rather
than asserted.
