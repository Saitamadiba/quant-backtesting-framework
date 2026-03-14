"""
Academic Presentation Report
=============================

Project report aligned to the Python Software Engineering course evaluation
grid (10 criteria).  Each tab maps directly to one grading category.

Covers four algorithmic trading strategies (FVG, Liquidity Raid, Momentum
Mastery, SBS) deployed as autonomous bots on a cloud VPS, monitored through
a 24-page Streamlit dashboard.

All content is static (no database dependency) — the page loads instantly.
"""

import streamlit as st

st.set_page_config(page_title="Academic Report", page_icon="\U0001F393", layout="wide")

st.markdown("""
<style>
    .report-title { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem; }
    .report-subtitle { font-size: 1.15rem; color: #888; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="report-title">Algorithmic Trading Infrastructure in Python</p>'
    '<p class="report-subtitle">'
    'Multi-strategy autonomous trading system &mdash; '
    'Python Software Engineering Course Project &mdash; Nelson Onu'
    '</p>',
    unsafe_allow_html=True,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  01 — CONCEPTION DU PROJET                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_01_conception():
    st.header("01 — Project Conception")

    # ── Ambition & Originality ──
    st.subheader("Ambition & Originality")
    st.markdown("""
This project builds a **complete, production-grade algorithmic trading
infrastructure** — not a toy prototype.  Four distinct strategies run
autonomously 24/7 on a cloud server, analyse live market data, send
real-time trade alerts via Telegram, log every decision to a database for
machine-learning refinement, and report performance through a 24-page
interactive dashboard.

What sets this project apart:

- **Real money at stake** — the bots monitor live cryptocurrency and
  futures markets (BTC, ETH, NQ) on Binance and Yahoo Finance.
- **Multi-strategy architecture** — four strategies capture fundamentally
  different market behaviours (gap-filling, liquidity hunting, momentum,
  structure-break-and-retest).
- **Walk-Forward Optimisation (WFO)** — parameters are not curve-fitted;
  they are validated on out-of-sample data using rolling windows.
- **Monte Carlo simulation** — statistical confidence intervals quantify
  the range of possible outcomes before risking capital.
- **Self-improving bots** — an adaptive risk manager adjusts risk-reward
  targets based on its own recent performance.
""")

    # ── Feature Description ──
    st.subheader("Description of Features")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Trading Bots (Backend)**")
        st.markdown("""
- 4 strategies (FVG, Liquidity Raid, Momentum Mastery, SBS)
- 9 bot instances across 5 assets (BTC, ETH, NQ)
- Real-time market data via WebSocket and REST APIs
- Automated signal detection, scoring, and alerting
- Walk-Forward Optimised (WFO) signal scoring engine
- HMM-based regime gating (trade only in favourable regimes)
- Gamma regime overlay (options implied volatility)
- Adaptive risk manager (self-tuning R:R from trade history)
- ML feature logging for every trade (50+ features)
- Telegram alerts with rich signal formatting
- SQLite trade persistence with schema migrations
- Auto-restart on crash (systemd + `os.execv` fallback)
""")
    with col2:
        st.markdown("**Dashboard (Frontend)**")
        st.markdown("""
- 24 Streamlit pages for analysis and monitoring
- Portfolio overview with aggregated KPIs
- Per-strategy deep dives with architecture diagrams
- Equity curves with drawdown overlays and regime bands
- Trade journal with multi-filter slider/dropdown controls
- Session analysis (Asian/London/New York performance)
- Monte Carlo simulation with fan charts
- WFO analysis with in-sample/out-of-sample comparison
- ML training pipeline and SHAP feature importance
- Stress testing (drawdown/volatility/gap scenarios)
- One-click VPS deployment and service management
- Real-time VPS database sync via SCP
""")

    # ── Process Description ──
    st.subheader("Description of Processes")
    st.markdown("""
The project follows a **research-to-production pipeline**:
""")
    st.code("""
Research Phase                    Production Phase
─────────────                     ────────────────
1. Backtest strategy idea    ──→  5. Deploy bot to VPS (systemd)
2. WFO parameter validation  ──→  6. Bot runs 24/7, sends Telegram alerts
3. Monte Carlo risk profiling──→  7. Trades logged to SQLite on VPS
4. Package as autonomous bot ──→  8. Dashboard syncs DBs, shows analytics
                                  9. Adaptive risk manager self-tunes
                                 10. ML features collected for refinement
""", language="text")

    st.markdown("**Each bot iteration follows this cycle:**")
    st.code("""
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  Initialize  │────→│     Main Loop        │────→│   Shutdown   │
│              │     │                      │     │              │
│ Load config  │     │ 1. Fetch market data │     │ Close WS     │
│ Test API     │     │ 2. Compute indicators│     │ Final Telegram│
│ Start WS     │     │ 3. Detect setups     │     │ Log uptime   │
│ Init Telegram│     │ 4. WFO score (0-1)   │     └──────────────┘
│ Init ML log  │     │ 5. Regime gate       │            ▲
│ Start sched. │     │ 6. Signal? → Alert   │            │
└──────────────┘     │ 7. Log to SQLite     │     Max errors (10)
                     │ 8. Sleep + heartbeat │     ┌──────────────┐
                     │ 9. Repeat            │────→│ Auto-Restart │
                     └──────────────────────┘     └──────────────┘
""", language="text")

    # ── Workflow Modelling ──
    st.subheader("Workflow Modelling")
    st.markdown("**Complete data flow from market to dashboard:**")
    st.code("""
                          ┌─────────────────────┐
                          │    Market APIs       │
                          │  Binance / Coinbase  │
                          │  Yahoo Finance       │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    DataManager       │
                          │  WebSocket + REST    │
                          │  Multi-source        │
                          │  fallback chain      │
                          └──────────┬──────────┘
                                     │
          ┌───────────────┼───────────────┼───────────────┐
          │               │               │               │
  ┌───────▼──────┐┌───────▼──────┐┌───────▼──────┐┌───────▼──────┐
  │ FVG Engine   ││ Sweep Detect ││ Momentum 6F  ││ SBS Engine   │
  │ 3-candle gap ││ State machine││ Confluence   ││ ML-scored    │
  └───────┬──────┘└───────┬──────┘└───────┬──────┘└───────┬──────┘
          │               │               │               │
  ┌───────▼──────┐┌───────▼──────┐┌───────▼──────┐┌───────▼──────┐
  │ WFO (9 comp) ││ WFO (4 comp) ││ WFO (6 comp) ││ WFO (4 comp) │
  └───────┬──────┘└───────┬──────┘└───────┬──────┘└───────┬──────┘
          │               │               │               │
          └───────────────┼───────────────┼───────────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
          ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
          │  Telegram    │  │  ML Logger   │  │   SQLite     │
          │  Alert       │  │  50+ features│  │  Trades DB   │
          └──────────────┘  └──────────────┘  └───────┬──────┘
                                                      │
                                              ┌───────▼──────┐
                                              │  SCP Sync    │
                                              │  VPS → Local │
                                              └───────┬──────┘
                                                      │
                                              ┌───────▼──────┐
                                              │  Dashboard   │
                                              │  24 Streamlit│
                                              │  pages       │
                                              └──────────────┘
""", language="text")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  02 — STRUCTURE LOGIQUE DE L'APPLICATION                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_02_structure():
    st.header("02 — Application Structure")

    # ── Workflow ──
    st.subheader("Application Workflow")
    st.markdown("""
The application is divided into three layers, each with a clear
responsibility:

| Layer | Role | Technology |
|-------|------|------------|
| **Bot Layer** | Autonomous trading agents running on VPS | Python asyncio + threading |
| **Data Layer** | Persistence, sync, aggregation | SQLite (per-bot) + DuckDB (analytics) |
| **Presentation Layer** | Interactive dashboard for analysis | Streamlit + Plotly |

Each layer communicates through well-defined interfaces — the bots write to
SQLite databases, the dashboard reads them via SCP sync, and DuckDB aggregates
all sources into a unified analytics store.
""")

    # ── External Modules ──
    st.subheader("External Modules Inventory")
    st.markdown("""
The project uses a carefully chosen set of external packages, each solving
a specific problem that Python's standard library cannot address:
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Core Libraries**")
        st.markdown("""
| Package | Purpose |
|---------|---------|
| `pandas` (2.0+) | DataFrame-based data manipulation |
| `numpy` (1.24+) | Numerical array operations |
| `plotly` (5.18+) | Interactive financial charts |
| `streamlit` (1.30+) | Web dashboard framework |
| `requests` (2.31+) | HTTP client for REST APIs |
| `python-dotenv` | Load `.env` files for secrets |
| `pytz` | Timezone-aware datetime handling |
""")
    with c2:
        st.markdown("**Specialist Libraries**")
        st.markdown("""
| Package | Purpose |
|---------|---------|
| `backtrader` (1.9+) | Event-driven backtesting engine |
| `duckdb` (0.9+) | Columnar analytics database |
| `optuna` (3.4+) | Bayesian hyperparameter tuning |
| `scikit-learn` (1.3+) | ML model training |
| `xgboost` (1.7+) | Gradient-boosted tree models |
| `shap` (0.43+) | ML feature importance |
| `scipy` (1.10+) | Statistical testing |
| `yfinance` | Yahoo Finance data (NQ futures) |
""")

    # ── Class System ──
    st.subheader("Class System (Modelling)")
    st.markdown("""
A common misconception would be that the config class inherits from the bot
class (or vice versa) in a single chain.  In reality, the project uses
**two parallel inheritance chains** that are completely independent of each
other:

- **Chain 1 — Configuration:** `FVGConfigBase` is a `@dataclass` that
  defines 40+ parameters with sensible defaults.  Each asset (BTC, ETH, NQ)
  creates a small subclass that **inherits** all those defaults and overrides
  only the 5-10 fields that differ (API keys, symbol name, thresholds).

- **Chain 2 — Bot Logic:** `FVGBotBase` is an abstract class (`ABC`) that
  contains 2,000+ lines of shared trading logic — signal detection, position
  monitoring, Telegram alerts, ML logging.  Each asset creates a thin
  subclass that **inherits** all that logic and implements only the 3
  abstract methods the base class requires.

These two chains never touch each other through inheritance.  Instead, they
are joined by **composition** — the bot wrapper instantiates its
corresponding config object and passes it to the base class constructor:
""")
    st.code("""
  CHAIN 1 — Configuration (inheritance)    CHAIN 2 — Bot Logic (inheritance)
  ─────────────────────────────────────    ──────────────────────────────────
  FVGConfigBase (@dataclass, 40+ fields)   FVGBotBase (ABC, 2,000+ lines)
        │                                        │
        ├──→ BTCConfig (overrides 5 fields)      ├──→ BTCFVGBot (implements 3 methods)
        ├──→ ETHConfig (overrides 5 fields)      ├──→ ETHFVGBot (implements 3 methods)
        └──→ NQConfig  (overrides 8 fields)      └──→ NQFVGBot  (implements 3 methods)

  These two chains are INDEPENDENT — neither inherits from the other.
  They are linked by COMPOSITION at the asset wrapper level:

  class BTCFVGBot(FVGBotBase):       # ← inherits from Chain 2
      def __init__(self):
          super().__init__(BTCConfig())  # ← passes Chain 1 object IN
                           ▲
                           │
               BTCConfig inherits from FVGConfigBase (Chain 1)
               BTCFVGBot inherits from FVGBotBase   (Chain 2)
               The link between them is composition, not inheritance.

  Same pattern for Liquidity Raid (LRConfigBase + LRBotBase),
  Momentum Mastery (MMConfigBase + MMBotBase), and SBS (SBSBotBase).
""", language="text")

    st.markdown("""
**Why two chains instead of one?**  Because configuration and behaviour are
two separate concerns.  A config object is a passive data container (what
values to use), while a bot object is an active agent (what actions to
perform).  Keeping them in separate hierarchies means:

- You can **test a config** without instantiating a bot (useful for
  validation).
- You can **swap configs** at runtime — the same `BTCFVGBot` class could
  accept a test config with paper-trading settings or a production config
  with real API keys.
- Adding a new asset only requires two small subclasses (one per chain),
  not a single monolithic class that mixes data and logic.

**Inheritance vs composition in plain terms:**
- *Inheritance* says "**is-a**" — `BTCConfig` *is a* `FVGConfigBase` with
  some overrides.
- *Composition* says "**has-a**" — `BTCFVGBot` *has a* `BTCConfig` that it
  uses to read parameters.
""")

    st.markdown("**Real code** from `FVG_Strategy/BTC/fvg_btc.py` — "
                "the actual asset wrapper file:")
    st.code("""
# FVG_Strategy/BTC/fvg_btc.py — the complete thin wrapper

from core.config_base import FVGConfigBase
from core.bot_base import FVGBotBase

# ── Chain 1: Configuration inheritance ──
class Config(FVGConfigBase):
    \"\"\"BTC overrides — inherits 40+ defaults, changes only what differs.\"\"\"
    SYMBOL = "BTC-USD"
    PRIMARY_TIMEFRAME = "5m"
    TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
    TRADE_ASIAN_SESSION = False
    TELEGRAM_BOT_TOKEN = os.getenv("FVG_BTC_TELEGRAM_TOKEN", "")

# ── Chain 2: Bot logic inheritance ──
class BTCFVGBot(FVGBotBase):
    \"\"\"BTC bot — inherits 2,000+ lines of logic, implements 3 methods.\"\"\"

    def _init_data_fetchers(self):            # Abstract method #1
        from binance_websocket_fetcher import BinanceWebsocketFetcher
        self.binance_fetcher = BinanceWebsocketFetcher(symbol="BTC-USD")

    def _get_multi_timeframe_data(self):      # Abstract method #2
        return self.binance_fetcher.get_all_timeframes()

    def send_startup_message(self):           # Abstract method #3
        self.telegram.send("BTC FVG Bot started!")

# ── Composition: bot receives config as constructor argument ──
if __name__ == "__main__":
    bot = BTCFVGBot(config=Config())   # Chain 2 object receives Chain 1 object
    asyncio.run(bot.run())
""", language="python")

    st.markdown("""
**Key classes and their roles:**

| Class | Module | Responsibility | Pattern |
|-------|--------|---------------|---------|
| `FVGBotBase` | `FVG_Strategy/core/bot_base.py` | Trading orchestration | ABC + Template Method |
| `FVGConfigBase` | `FVG_Strategy/core/config_base.py` | 40+ parameters | @dataclass |
| `SweepState` | `the internal strategy core` | Sweep lifecycle | Enum state machine |
| `TechnicalAnalysis` | `*/core/technical_analysis.py` | Indicator math | @staticmethod utility |
| `DatabaseManager` | `*/core/database_manager.py` | Trade persistence | SQLite + migrations |
| `QuantMetrics` | `the internal strategy core` | Sharpe, Kelly, CVaR | Running accumulators |
| `DataManager` | `*/core/data_manager.py` | API data fetching | Fallback chain |
| `AdaptiveRiskManager` | `FVG_Strategy/core/adaptive_risk_manager.py` | Self-tuning R:R | SQLite bootstrap |
| `RegimeDetector` | `FVG_Strategy/core/regime_detector.py` | Market classification | Multi-factor scoring |
| `SBSBotBase` | `SBS/bots/core/bot_base.py` | SBS trading orchestration | ABC + asyncio |
| `WFOSignalScorer` | `wfo_signal_scorer.py` | Signal quality rating | Weighted component model |
""")

    # ── Data Management ──
    st.subheader("Data Management System")
    st.markdown("""
The project uses a **dual-database architecture**:

| Database | Engine | Location | Purpose |
|----------|--------|----------|---------|
| Trade DBs | SQLite | VPS (per-bot) | Record every trade with 20+ columns |
| ML Training DB | SQLite | VPS (per-bot) | 50+ ML features per trade entry |
| Analytics DB | DuckDB | Local machine | Aggregate all bots for dashboard queries |
| Manual Trades | SQLite | Local | User's discretionary trade journal |

**Data flow:** VPS SQLite files are synced to the local machine via SCP.
The dashboard's data loader merges them into DuckDB for fast analytical
queries (groupby, window functions, percentile calculations).
""")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  03 — METHODE DE TRAVAIL                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_03_methode():
    st.header("03 — Work Methodology")

    # ── File Organisation ──
    st.subheader("File Organisation")
    st.code("""
Backtesting/                         # Project root (Git repository)
├── FVG_Strategy/
│   ├── core/                        # Shared base classes
│   │   ├── bot_base.py              # FVGBotBase (ABC) — 2,000+ lines
│   │   ├── config_base.py           # FVGConfigBase (@dataclass)
│   │   ├── regime_detector.py       # HMM-proxy market regime
│   │   ├── adaptive_risk_manager.py # Self-tuning R:R
│   │   ├── telegram_notifier.py     # Telegram alerts
│   │   └── premature_entry_analyzer.py
│   ├── BTC/                         # Thin wrapper + asset-specific fetchers
│   ├── ETH/
│   └── NQ/
│
├── Liquidity_Raid/
│   ├── core/
│   │   ├── bot_base.py              # LRBotBase (ABC)
│   │   ├── config_base.py           # LRConfigBase — 80+ fields, 35 sections
│   │   ├── sweep_state.py           # SweepState (Enum)
│   │   ├── technical_analysis.py    # 8 @staticmethod methods
│   │   ├── database_manager.py      # SQLite + migrations
│   │   ├── quant_metrics.py         # Sharpe, Kelly, CVaR
│   │   ├── session_manager.py       # Kill zone time management
│   │   ├── position_manager.py      # Position lifecycle
│   │   ├── mtf_analysis.py          # Multi-timeframe scoring
│   │   ├── data_manager.py          # API fetcher with fallback
│   │   └── ml_integration_helper.py # ML feature logging
│   ├── BTC_V2/
│   └── ETH_V2/
│
├── Momentum_Mastery/
│   ├── core/
│   │   ├── bot_base.py              # MMBotBase (ABC)
│   │   ├── config_base.py           # 20+ configuration sections
│   │   ├── technical_analysis.py    # 10 @staticmethod methods
│   │   └── mtf_analysis.py
│   ├── BTC/
│   └── ETH/
│
├── SBS/
│   ├── bots/
│   │   └── core/
│   │       ├── bot_base.py          # SBSBotBase (ABC, asyncio)
│   │       └── ml_integration_helper.py
│   ├── BTC/
│   └── ETH/
│
├── dashboard/
│   ├── app.py                       # Streamlit entry point
│   ├── config.py                    # Centralised paths & registry
│   ├── data/
│   │   ├── data_loader.py           # Unified trade loader
│   │   ├── schema_normalizer.py     # 4 normalizers → unified schema
│   │   ├── vps_sync.py              # SCP database sync
│   │   ├── binance_helpers.py       # Public market data & indicators
│   │   └── binance_trading.py       # Authenticated trading API
│   ├── components/
│   │   ├── charts.py                # 17 Plotly chart builders
│   │   ├── kpi_cards.py             # KPI metric card helpers
│   │   └── filters.py               # Reusable sidebar filters
│   └── pages/                       # 24 Streamlit pages (auto-discovered)
│
├── backtrader_framework/
│   ├── optimization/
│   │   ├── wfo_engine.py            # Walk-Forward Optimisation
│   │   └── statistics.py            # Backtest statistical analysis
│   ├── data/
│   │   ├── duckdb_manager.py        # DuckDB analytics schema
│   │   └── validation.py            # OHLCV data validation
│   └── tests/                       # pytest test suite
│       ├── test_validation.py
│       ├── test_indicators.py
│       ├── test_risk_management.py
│       └── test_cpcv.py
│
├── wfo_signal_scorer.py             # WFO scoring engine (shared)
├── requirements.txt                 # External dependencies
├── pyproject.toml                   # Project metadata (PEP 621)
└── .gitignore                       # 99 rules (secrets, DBs, strategies)
""", language="text")

    # ── Local/Server Workspaces ──
    st.subheader("Local / Server Workspaces")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Local Machine (Development)**")
        st.markdown("""
- Code editing, testing, and backtesting
- WFO optimisation and Monte Carlo simulation
- Dashboard development (`streamlit run Home.py`)
- Git version control (all commits on local)
- `requirements.txt` and `pyproject.toml` define dependencies
- `.env` file stores local API keys

**Path:** `~/Desktop/Quant/Backtesting/`
""")
    with c2:
        st.markdown("**VPS (Production Server)**")
        st.markdown("""
- 9 bot instances as `systemd` services (auto-restart)
- Each bot has its own working directory and SQLite DB
- Deployed via **SCP** file transfer + SSH service restart
- `.env` file stores production API keys (different from local)
- Logs stored per-bot for forensic debugging

**Host:** `YOUR_VPS:2222` (user: `trader`)
**Path:** `/home/trader/trading_bots/`
""")

    st.code("""
Deployment workflow:

  Local Machine                           VPS
  ─────────────                           ───
  Edit code
       │
       ▼
  Test locally    ──── SCP ────────────→  Receive files
       │                                       │
       ▼                                       ▼
  Git commit      ──── SSH ────────────→  Restart systemd service
                                               │
                                               ▼
                                          Bot runs 24/7
                                               │
                  ◄─── SCP (databases) ───     │
                                               ▼
  Dashboard reads                         Writes to SQLite
  synced DBs
""", language="text")

    # ── Code Reusability ──
    st.subheader("Code Organisation & Reusability")
    st.markdown("""
The project follows the **DRY principle** (Don't Repeat Yourself) at
architecture scale:

| Component | Shared (base) | Per-asset (wrapper) | Reuse ratio |
|-----------|--------------|---------------------|-------------|
| FVG Bot | `bot_base.py` — 2,000+ lines | BTC/ETH/NQ — ~300 lines each | **87%** shared |
| LR Bot | `bot_base.py` — 1,500+ lines | BTC/ETH — ~260 lines each | **85%** shared |
| MM Bot | `bot_base.py` — 1,200+ lines | BTC/ETH — ~300 lines each | **80%** shared |
| SBS Bot | `bot_base.py` — 1,000+ lines | BTC/ETH — ~200 lines each | **83%** shared |
| Config | `config_base.py` — all defaults | Asset wrapper — 5-10 overrides | **95%** shared |

The `wfo_signal_scorer.py` module is shared across **all** strategies — each
strategy passes its own component configuration, but the scoring engine is
written once.
""")

    # ── Naming & Documentation ──
    st.subheader("Naming Conventions & Documentation")
    st.markdown("""
- **Classes**: `PascalCase` — `FVGBotBase`, `SweepState`, `QuantMetrics`
- **Functions/methods**: `snake_case` — `calculate_kelly()`, `get_daily_ema_bias()`
- **Constants**: `UPPER_SNAKE` — `MAX_DAILY_TRADES`, `ATR_MULTIPLIER`
- **Private methods**: `_prefix` — `_execute_retest_trade()`, `_get_current_dvol()`
- **Modules**: `snake_case.py` — `bot_base.py`, `sweep_state.py`

Every module starts with a **docstring** explaining its purpose, and every
class documents its responsibilities and constructor arguments.  Type hints
(`Dict[str, float]`, `Optional[pd.DataFrame]`, `Tuple[bool, str]`) are used
throughout for machine-readable documentation.
""")

    # ── Pure Functions ──
    st.subheader("Pure Functions")
    st.markdown("""
All technical indicator calculations are implemented as **pure functions**
using `@staticmethod`.  A pure function:

1. Always returns the same output for the same input (deterministic).
2. Has no side effects (doesn't modify external state).
3. Doesn't depend on instance variables (`self` is not needed).
""")
    st.code("""
# the internal strategy core

class TechnicalAnalysis:
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        \"\"\"Pure function: same input → same output, no side effects.\"\"\"
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        \"\"\"Pure: depends only on input DataFrame and period.\"\"\"
        high, low, close = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
""", language="python")
    st.markdown("""
The FVG bot has **8** pure static methods, Liquidity Raid has **8**, and
Momentum Mastery has **10** — grouped in `TechnicalAnalysis` utility classes
for namespace organisation without instance overhead.
""")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  04 — PROGRAMMATION PYTHON                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_04_python():
    st.header("04 — Python Programming")

    # ── Functions & Variables ──
    st.subheader("Function & Variable Declarations")

    st.markdown("**Type-annotated function signatures** document inputs and outputs:")
    st.code("""
# the internal strategy core
def is_in_killzone(self) -> Tuple[bool, str]:
    \"\"\"Returns (is_active, session_name) — caller knows the return shape.\"\"\"

# the internal strategy core
def get_open_trade(self) -> Optional[Dict]:
    \"\"\"Returns a trade dict or None — Optional forces caller to check.\"\"\"

# FVG_Strategy/core/bot_base.py
def _get_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
    \"\"\"Returns {'1H': df_1h, '4H': df_4h, '1D': df_daily}.\"\"\"
""", language="python")

    st.markdown("**List and dict comprehensions** for declarative data transforms:")
    st.code("""
# the internal strategy core — Kelly criterion

wins = [r for r in self.r_multiples if r > 0]    # Filter winners
losses = [abs(r) for r in self.r_multiples if r < 0]  # Filter losers (absolute)
win_rate = len(wins) / len(self.r_multiples)

# Profit factor — one-liner that reads as a mathematical definition
gross_profits = sum(p for p in self.pnls if p > 0)
gross_losses = abs(sum(p for p in self.pnls if p < 0))
""", language="python")

    st.markdown("**Generator expressions** for lazy, memory-efficient evaluation:")
    st.code("""
# Generator inside all() — short-circuits on first False
all_aligned = all(
    ema_fast > ema_slow
    for ema_fast, ema_slow in zip(fast_emas, slow_emas)
)
# Stops evaluating at first misalignment — O(1) best case

# Sum with generator (no intermediate list created):
total_risk = sum(
    trade['risk_amount'] for trade in active_trades.values()
    if trade['status'] == 'open'
)
""", language="python")

    st.divider()

    # ── Classes, Methods, Inheritance ──
    st.subheader("Classes, Methods & Inheritance")

    st.markdown("**ABC + @abstractmethod** — Template Method pattern:")
    st.code("""
# FVG_Strategy/core/bot_base.py

from abc import ABC, abstractmethod

class FVGBotBase(ABC):
    \"\"\"Subclasses MUST implement three abstract methods.
    Forgetting one raises TypeError at import time, not at runtime.\"\"\"

    @abstractmethod
    def _init_data_fetchers(self):
        \"\"\"Wire up Binance (BTC/ETH) or Yahoo (NQ) data sources.\"\"\"
        ...

    @abstractmethod
    def _get_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        \"\"\"Fetch OHLCV for all configured timeframes.\"\"\"
        ...

    @abstractmethod
    def send_startup_message(self):
        \"\"\"Asset-branded Telegram greeting.\"\"\"
        ...

    # 2,000+ lines of shared logic inherited by all asset wrappers
""", language="python")

    st.markdown("**@dataclass** — declarative configuration with deferred evaluation:")
    st.code("""
# the internal strategy core

from dataclasses import dataclass, field
import os

@dataclass
class LRConfigBase:
    \"\"\"80+ fields across 35 sections. Auto-generates __init__, __repr__.\"\"\"

    BINANCE_API: str = "https://api.binance.com/api/v3"
    TIMEFRAME: str = "15m"
    RISK_REWARD_RATIO: float = 0.5      # Calibrated from WFO MFE analysis
    MAX_DAILY_TRADES: int = 5

    # Secrets — deferred evaluation via default_factory
    # Env var is read at instantiation, NOT at import time
    TELEGRAM_BOT_TOKEN: str = field(
        default_factory=lambda: os.getenv("LR_BTC_TELEGRAM_TOKEN", "")
    )
""", language="python")

    st.markdown("**Enum** — type-safe state machine:")
    st.code("""
# the internal strategy core

from enum import Enum

class SweepState(Enum):
    \"\"\"Finite state machine for sweep lifecycle.
    WAITING → SWEEP_DETECTED → TRADED\"\"\"
    WAITING = "waiting"
    SWEEP_DETECTED = "detected"
    TRADED = "traded"

# SweepState.WAITNG → AttributeError (catches typos at write time)
""", language="python")

    st.divider()

    # ── Built-in Modules ──
    st.subheader("Built-In Modules Used")
    st.markdown("""
These modules come with Python — no `pip install` needed:

| Module | Usage in Project | Example |
|--------|-----------------|---------|
| `abc` | Abstract base classes | `FVGBotBase(ABC)`, `@abstractmethod` |
| `dataclasses` | Config declarations | `@dataclass`, `field(default_factory=...)` |
| `enum` | State machines | `SweepState(Enum)` |
| `typing` | Type annotations | `Dict[str, float]`, `Optional[pd.DataFrame]` |
| `asyncio` | Non-blocking event loop | `await asyncio.sleep()`, `async def run()` |
| `threading` | Daemon background tasks | Reporter scheduler in separate thread |
| `logging` | Structured log system | Per-module loggers, file + console handlers |
| `sqlite3` | Trade database | `CREATE TABLE`, `ALTER TABLE` migrations |
| `os` | Environment variables | `os.getenv("TELEGRAM_TOKEN")` |
| `sys` | Path manipulation | `sys.path.insert(0, ...)` for module resolution |
| `json` | Data serialisation | WFO components stored as JSON text in SQLite |
| `datetime` | Timestamps | Session time management, trade timing |
| `time` | Performance timing | `time.time()` for iteration benchmarking |
| `traceback` | Error diagnostics | Full stack traces in error handlers |
| `signal` | Graceful shutdown | `SIGTERM` handling for systemd stop |
| `math` | Numerical functions | `math.sqrt()` for Sharpe ratio |
| `hashlib` | API auth | `HMAC-SHA256` signature for Binance |
| `hmac` | API auth | Message authentication codes |
""")

    st.divider()

    # ── Standard Libraries ──
    st.subheader("Standard Library Patterns")
    st.markdown("""
**asyncio** — non-blocking I/O for concurrent API calls:
""")
    st.code("""
# SBS/bots/core/bot_base.py

async def run(self):
    \"\"\"Main loop — non-blocking sleep between iterations.\"\"\"
    self.running = True
    while self.running:
        await self.main_loop_iteration()
        await self.sleep_with_heartbeat()  # yields control, doesn't block

async def sleep_with_heartbeat(self):
    \"\"\"Sleep in 5-minute chunks, sending heartbeats between.\"\"\"
    slept = 0
    while slept < self.config.check_interval_minutes * 60:
        await asyncio.sleep(300)   # Non-blocking — event loop free
        slept += 300
        await self.telegram_bot.send_heartbeat()
""", language="python")

    st.markdown("**sqlite3** — schema evolution with EAFP migrations:")
    st.code("""
# the internal strategy core

def init_database(self):
    conn = sqlite3.connect(self.db_file)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, signal_type TEXT, entry_price REAL, ...
    )''')

    # Migration: add new columns to old databases
    # EAFP = "Easier to Ask Forgiveness than Permission"
    for col, col_type in [
        ('confidence', 'REAL'),
        ('wfo_components_json', 'TEXT'),
        ('regime_gate', 'TEXT'),
    ]:
        try:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists — silently continue
    conn.commit()
""", language="python")

    st.markdown("**logging** — structured, per-module observability:")
    st.code("""
# FVG_Strategy/core/bot_base.py

import logging
logger = logging.getLogger(__name__)  # Creates 'core.bot_base' logger

def setup_logging(config):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)    # File: capture everything

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console: only important

    root = logging.getLogger()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

# Output: 2026-03-09 14:30:01 - core.bot_base - INFO - Price: $84,521
""", language="python")

    st.divider()

    # ── Scalability & Extensibility ──
    st.subheader("Scalability & Extensibility")
    st.markdown("""
The architecture is designed so that **adding a new asset takes 30 minutes**:

1. Create a new directory (e.g., `FVG_Strategy/SOL/`)
2. Write a thin wrapper (~300 lines) that subclasses `FVGBotBase`
3. Override the 3 abstract methods (data source, multi-TF fetch, startup message)
4. Create a `SOLConfig(FVGConfigBase)` with the 5-10 fields that differ

The base class, WFO scorer, ML logger, Telegram notifier, and all analytics
are inherited automatically.  The dashboard's `config.py` registry needs one
line added to recognise the new bot.

**Graceful degradation** ensures extensibility doesn't break existing code:
""")
    st.code("""
# FVG_Strategy/core/bot_base.py — 9 optional modules

try:
    from wfo_signal_scorer import WFOSignalScorer
except ImportError:
    WFOSignalScorer = None    # Feature disabled, bot still runs

try:
    from ml_trade_filter import MLTradeFilter
except ImportError:
    MLTradeFilter = None      # No ML filter, bot still runs

# At runtime:
if WFOSignalScorer is not None:
    score = self.wfo_scorer.score(features)
    if score < self.config.WFO_THRESHOLD:
        return  # Skip low-quality setup
# If WFOSignalScorer not installed → this block never executes
""", language="python")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  05 — INTERFACE UTILISATEUR (STREAMLIT instead of FLASK)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_05_ui():
    st.header("05 — User Interface (Streamlit)")

    st.info("""
**Note:** This project uses **Streamlit** instead of Flask.  Streamlit is a
Python-native web framework designed for data applications — it provides
interactive widgets, reactive state management, and Plotly chart integration
out of the box, without requiring HTML, CSS, or JavaScript.  For a data-heavy
trading dashboard, Streamlit is more productive than Flask + Jinja2 + Bootstrap.
""")

    # ── Libraries ──
    st.subheader("UI Libraries & Components")
    st.markdown("""
| Streamlit Feature | Flask Equivalent | Usage in Project |
|-------------------|-----------------|------------------|
| `st.sidebar` | Bootstrap navbar | Strategy/symbol/date filters |
| `st.columns()` | Bootstrap grid | Multi-column KPI layouts |
| `st.tabs()` | Bootstrap nav-tabs | Section navigation within pages |
| `st.metric()` | Custom HTML cards | Win rate, Sharpe, P&L display |
| `st.dataframe()` | DataTables.js | Interactive sortable trade tables |
| `st.plotly_chart()` | Chart.js / Plotly.js | Equity curves, histograms, heatmaps |
| `st.selectbox()` | `<select>` dropdown | Strategy/symbol selectors |
| `st.slider()` | Range slider plugin | P&L range filter |
| `st.multiselect()` | Multi-select dropdown | Strategy combination selector |
| `st.checkbox()` | `<input type=checkbox>` | Toggle overlays (benchmark, regime) |
| `st.button()` | `<button>` | Deploy to VPS, restart service |
| `st.progress()` | Bootstrap progress bar | Bulk deployment progress |
| `st.expander()` | Bootstrap accordion | Collapsible detail sections |
| `st.form()` | `<form>` | Manual trade entry |
| `st.session_state` | Flask session | Persist state across reruns |
| `st.set_page_config()` | Flask template base | Page title, icon, layout |
| Auto-discovery pages | Flask Blueprint | `pages/` directory = automatic routing |
""")

    # ── Functionalities ──
    st.subheader("Dashboard Functionalities (24 Pages)")
    st.markdown("""
| # | Page | Key Widgets Used |
|---|------|-----------------|
| 1 | Overview | `st.metric`, `st.columns`, `st.plotly_chart` |
| 2 | Strategy Explainer | `st.tabs`, `st.expander`, `st.dataframe` |
| 3 | Strategy Deep Dive | `st.tabs`, `st.code`, `st.columns` |
| 3b | Live Logs | `st.code`, `st.sidebar.slider`, `st_autorefresh` |
| 4 | Trade Journal | `st.sidebar.slider`, `st.dataframe(height=600)` |
| 5 | Equity Curves | `st.sidebar.checkbox`, `st.plotly_chart` |
| 6 | Session Analysis | `st.plotly_chart` (heatmap), `st.columns` |
| 7 | Monthly Performance | `st.plotly_chart` (calendar heatmap) |
| 8 | ML Training | `st.button`, `st.spinner`, `st.code` |
| 9 | WFO Analysis | `st.selectbox`, `st.plotly_chart` (scatter) |
| 10 | Monte Carlo | `st.plotly_chart` (fan chart), `st.metric` |
| 11 | Deploy Bots | `st.button`, `st.progress`, `st.session_state` |
| 12 | Portfolio | `st.multiselect`, `st.plotly_chart` |
| 13 | Meta Strategy | `st.multiselect`, `st.expander` |
| 13b | Trade Monitor | `st_autorefresh`, `st.metric`, `st.session_state` |
| 14 | SHAP Analysis | `st.plotly_chart` (waterfall), `st.selectbox` |
| 15 | Bayesian Tuning | `st.form`, `st.slider`, `st.button` |
| 16 | Stress Testing | `st.selectbox`, `st.plotly_chart` |
| 17 | Cross Asset | `st.dataframe`, `st.plotly_chart` |
| 18 | ML Performance | `st.tabs`, `st.metric`, `st.plotly_chart` |
| 19 | Shadow Backtest | `st.button`, `st.spinner` |
| 20 | Quant Research Lab | `st.tabs`, `st.code`, `st.plotly_chart` |
| 21 | Academic Report | This page |
| 22 | Dashboard Presentation | `st.tabs`, `st.code`, `st.markdown` |
""")

    # ── Usability ──
    st.subheader("Usability Features")
    st.code("""
# Reusable filter pattern — every analytics page uses this

# dashboard/components/filters.py
def source_filter(key_prefix=""):
    return st.sidebar.radio("Data Source", ["All", "Live", "Backtest"],
                            key=f"{key_prefix}_source")

# dashboard/pages/4_Trade_Journal.py — interactive slider
pnl_range = st.sidebar.slider(
    "P&L ($)", pnl_min, pnl_max, (pnl_min, pnl_max),
    key="tj_pnl", help="Filter trades by profit/loss range"
)

# dashboard/pages/5_Equity_Curves.py — toggle overlays
show_benchmark = st.sidebar.checkbox("Buy & Hold Benchmark", value=False)
show_regimes = st.sidebar.checkbox("Market Regimes", value=False)

# dashboard/pages/11_Deploy_Bots.py — persistent state
if "deploy_results" not in st.session_state:
    st.session_state["deploy_results"] = {}

if st.button("Deploy FVG BTC", type="primary"):
    with st.spinner("Uploading via SCP..."):
        result = deploy_file_to_vps(local_path, remote_path)
    st.session_state["deploy_results"]["FVG BTC"] = result
""", language="python")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  06 — MODULES EXTERNES                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_06_modules():
    st.header("06 — External Modules & API Connections")

    # ── External Libraries ──
    st.subheader("External Libraries")
    st.markdown("""
From `requirements.txt` and `pyproject.toml`:

| Library | Version | Category | Why Not Standard Library? |
|---------|---------|----------|--------------------------|
| `pandas` | 2.0+ | Data | DataFrame operations, `groupby`, `ewm`, `resample` |
| `numpy` | 1.24+ | Numerical | Vectorised array ops, `percentile`, `polyfit` |
| `plotly` | 5.18+ | Viz | Interactive financial charts (zoom, hover, pan) |
| `streamlit` | 1.30+ | Web UI | Reactive dashboard framework (no JS needed) |
| `requests` | 2.31+ | HTTP | REST API calls to Binance, Coinbase, Telegram |
| `backtrader` | 1.9+ | Backtesting | Event-driven historical simulation engine |
| `duckdb` | 0.9+ | Analytics DB | Columnar OLAP engine for fast aggregations |
| `optuna` | 3.4+ | ML | Bayesian hyperparameter optimisation |
| `scikit-learn` | 1.3+ | ML | Logistic regression, model pipelines |
| `xgboost` | 1.7+ | ML | Gradient-boosted trees for trade filtering |
| `shap` | 0.43+ | ML | Feature importance (SHAP values) |
| `scipy` | 1.10+ | Statistics | Confidence intervals, t-tests |
| `yfinance` | 0.2+ | Data | Yahoo Finance API for NQ futures |
| `python-dotenv` | 1.0+ | Config | Load `.env` files for secrets |
| `pytz` | 2023+ | Time | Timezone-aware datetime conversions |
""")

    # ── Remote API Connections ──
    st.subheader("Remote API Connections")

    st.markdown("**1. Binance REST API** — market data and authenticated trading:")
    st.code("""
# dashboard/data/binance_trading.py — HMAC-SHA256 authenticated requests

import hmac, hashlib, requests

BINANCE_REST_BASE = "https://api.binance.us/api/v3"
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

def _sign(params: dict) -> dict:
    \"\"\"Add timestamp and HMAC-SHA256 signature for Binance auth.\"\"\"
    params["timestamp"] = int(time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(
        API_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    params["signature"] = sig
    return params

def _get(endpoint, params=None, signed=True):
    if signed:
        params = _sign(dict(params or {}))
    resp = requests.get(
        f"{BINANCE_REST_BASE}/{endpoint}",
        params=params,
        headers={"X-MBX-APIKEY": API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
""", language="python")

    st.markdown("**2. Binance WebSocket** — real-time price streaming:")
    st.code("""
# Momentum_Mastery/BTC/binance_websocket_fetcher.py

class BinanceWebsocketFetcher:
    def __init__(self, symbol="BTC-USD"):
        self.ws_base = "wss://stream.binance.us:9443/ws"
        self.candle_cache = {}       # {symbol_tf: DataFrame}
        self.cache_lock = threading.Lock()  # Thread-safe access
        self.live_prices = {}

    def initialize_with_history(self, symbol, timeframes, limit=500):
        \"\"\"Fetch REST history, then start WebSocket stream.\"\"\"
        df = self._fetch_from_rest(symbol, timeframe, limit)
        self._start_websocket_stream(symbol, timeframes)
""", language="python")

    st.markdown("**3. Telegram Bot API** — trade alert delivery:")
    st.code("""
# FVG_Strategy/core/telegram_notifier.py

class TelegramNotifierV3:
    def __init__(self, bot_token, chat_id):
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, message: str) -> bool:
        resp = requests.post(
            f"{self.base_url}/sendMessage",
            json={'chat_id': self.chat_id, 'text': message},
            timeout=10,
        )
        return resp.status_code == 200
""", language="python")

    st.markdown("**4. Multi-source data fallback chain:**")
    st.code("""
# FVG_Strategy/BTC/coinbase_binance_btc_fetcher.py

class CoinbaseBinanceBTCFetcher:
    def get_candles(self, symbol, timeframe, limit=500):
        # 1. Try Coinbase Pro (free, no auth)
        try:
            df = self._fetch_from_coinbase(timeframe, limit)
            if df is not None: return df
        except Exception:
            self.logger.warning("Coinbase failed, trying Binance...")

        # 2. Try Binance (very reliable)
        try:
            df = self._fetch_from_binance(timeframe, limit)
            if df is not None: return df
        except Exception:
            self.logger.warning("Binance failed, trying TaapiIO...")

        # 3. Fallback to TaapiIO (requires API key)
        if self.taapi_api_key:
            return self._fetch_from_taapi(timeframe, limit)
""", language="python")

    st.info("""
**Note on Maps/GPS and Audio/Video:** These criteria are not applicable to
this project.  Instead, we extensively use **real-time financial data feeds**
(WebSocket streaming, REST API polling) and **interactive financial
visualisations** (Plotly candlestick charts, equity curves, heatmaps) which
represent the domain-equivalent data acquisition and presentation layer.
""")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  07 — TRAITEMENT DES DONNEES (NUMPY, PANDAS, VISUALISATION)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_07_data():
    st.header("07 — Data Processing with NumPy & Pandas")

    # ── Data Collection ──
    st.subheader("Data Collection Pipeline")
    st.markdown("""
Market data flows through a multi-stage pipeline before reaching the
strategy engine:

| Stage | Tool | Operation |
|-------|------|-----------|
| 1. Fetch | `requests` / WebSocket | Raw JSON from Binance/Coinbase/Yahoo |
| 2. Parse | `pd.DataFrame()` | Convert JSON arrays to typed DataFrames |
| 3. Clean | `pd.to_datetime()`, `dropna()` | Fix timestamps, handle missing values |
| 4. Enrich | `ewm()`, `rolling()`, `concat()` | Add EMA, ATR, RSI indicators |
| 5. Resample | `resample('4h').agg()` | Build higher-timeframe candles |
| 6. Store | `sqlite3` / `duckdb` | Persist for dashboard analytics |
""")

    # ── Pandas ──
    st.subheader("Pandas Operations (Real Examples)")

    st.markdown("**Exponential weighted moving average** for trend detection:")
    st.code("""
# the internal strategy core
return data.ewm(span=period, adjust=False).mean()
""", language="python")

    st.markdown("**concat + max** for True Range calculation (ATR foundation):")
    st.code("""
# Momentum_Mastery/core/technical_analysis.py
tr1 = high - low                    # Intra-bar range
tr2 = abs(high - close)             # Gap up from previous close
tr3 = abs(low - close)              # Gap down from previous close
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  # Largest of three
atr = tr.rolling(window=14).mean()  # Smoothed average
""", language="python")

    st.markdown("**groupby + agg** for per-strategy statistics:")
    st.code("""
# dashboard/data/data_loader.py
groups = df.groupby(["strategy", "symbol", "source"])

def _stats(g):
    total = len(g)
    wins = (g["pnl_usd"] > 0).sum()
    gross_profit = g.loc[g["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = g.loc[g["pnl_usd"] < 0, "pnl_usd"].abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    # Max drawdown from cumulative P&L
    cum = g["pnl_usd"].cumsum()
    peak = cum.cummax()
    max_dd = (peak - cum).max()
    return pd.Series({...})
""", language="python")

    st.markdown("**resample** for higher-timeframe candle construction:")
    st.code("""
# backtrader_framework/optimization/wfo_engine.py
ohlcv_4h = df[['Open','High','Low','Close']].resample('4h').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
}).dropna()
# Build 4H EMA from resampled data, then forward-fill to 15M index
df['HTF_EMA50'] = htf_ema50.reindex(df.index, method='ffill')
""", language="python")

    st.markdown("**cumsum + cummax** for drawdown analysis:")
    st.code("""
# dashboard/pages/5_Equity_Curves.py
cum = s["pnl_usd"].cumsum()
cum_equity = INITIAL_BALANCE + cum
peak_equity = cum_equity.cummax()
drawdown = peak_equity - cum_equity
max_dd = drawdown.max()
""", language="python")

    st.divider()

    # ── NumPy ──
    st.subheader("NumPy Operations (Real Examples)")

    st.markdown("**polyfit + polyval** for equity curve R-squared:")
    st.code("""
# dashboard/pages/5_Equity_Curves.py
x = np.arange(len(cum_equity), dtype=float)
y = cum_equity.values.astype(float)
coeffs = np.polyfit(x, y, 1)          # Linear regression
y_pred = np.polyval(coeffs, x)        # Predicted values
ss_res = np.sum((y - y_pred) ** 2)    # Residual sum of squares
ss_tot = np.sum((y - y.mean()) ** 2)  # Total sum of squares
equity_r2 = 1 - (ss_res / ss_tot)     # R-squared (0 to 1)
""", language="python")

    st.markdown("**percentile** for Monte Carlo confidence intervals:")
    st.code("""
# dashboard/components/charts.py
arr = np.array(equity_paths[:200])
median_path = np.median(arr, axis=0)
p5_path = np.percentile(arr, 5, axis=0)     # 5th percentile (worst case)
p95_path = np.percentile(arr, 95, axis=0)   # 95th percentile (best case)
""", language="python")

    st.markdown("**where** for conditional vectorised operations:")
    st.code("""
# backtrader_framework/optimization/wfo_engine.py
bar_range = df['High'] - df['Low']
df['CloseVsRange'] = np.where(
    bar_range > 0,
    (df['Close'] - df['Low']) / bar_range,  # Normalised position in bar
    0.5                                      # Default if zero-range bar
)
""", language="python")

    st.markdown("**maximum.accumulate** for running maximum (drawdown calc):")
    st.code("""
# backtrader_framework/optimization/statistics.py
cumulative_r = np.cumsum(r_values)
running_max = np.maximum.accumulate(cumulative_r)
drawdowns = running_max - cumulative_r
max_drawdown_r = float(np.max(drawdowns))
""", language="python")

    st.divider()

    # ── Visualisation ──
    st.subheader("Data Visualisation with Plotly")
    st.markdown("""
The `dashboard/components/charts.py` module provides **17 reusable chart
builders**.  All charts use `plotly_dark` template for consistency.
""")

    st.markdown("**Chart types used across the dashboard:**")
    st.markdown("""
| Chart Type | Plotly Class | Dashboard Page | Purpose |
|------------|-------------|----------------|---------|
| Line chart | `go.Scatter` | Equity Curves | Cumulative P&L over time |
| Histogram | `go.Histogram` | Trade Journal | R-multiple distribution |
| Heatmap | `go.Heatmap` | Session Analysis | Strategy x session win rates |
| Bar chart | `px.bar` | Overview | Total P&L by strategy and symbol |
| Donut chart | `go.Pie(hole=0.4)` | Deep Dive | Exit reason breakdown |
| Scatter plot | `go.Scatter(mode="markers")` | Deep Dive | MFE vs MAE analysis |
| Fan chart | `go.Scatter` (fill) | Monte Carlo | Confidence band equity paths |
| Regime bands | `fig.add_shape(type="rect")` | Equity Curves | Background regime colouring |
| Vertical lines | `fig.add_vline()` | Equity Curves | Strategy deployment markers |
""")

    st.markdown("**Example — interactive equity curve with regime overlays:**")
    st.code("""
# dashboard/components/charts.py

def equity_curve(df, initial_balance=10000, regime_df=None):
    fig = go.Figure()

    # Add regime background bands (semi-transparent rectangles)
    if regime_df is not None:
        for start, end, regime in regime_blocks:
            fig.add_shape(type="rect", x0=start, x1=end,
                          y0=0, y1=1, yref="paper",
                          fillcolor=REGIME_COLORS[regime], line_width=0)

    # Plot each strategy as a line
    for strat in df["strategy"].unique():
        s = df[df["strategy"] == strat].sort_values("entry_time")
        equity = initial_balance + s["pnl_usd"].cumsum()
        fig.add_trace(go.Scatter(
            x=s["entry_time"], y=equity, name=strat, mode="lines",
            line=dict(color=STRATEGY_COLORS[strat]),
        ))

    fig.update_layout(template="plotly_dark", height=500)
    return fig
""", language="python")

    # ── API Testing ──
    st.subheader("API Accessibility Testing")
    st.markdown("""
The bots use a **test-on-startup** pattern: during initialisation, they
fetch a small batch of candles from the API and validate the response
before entering the main loop.  If the test fails, the bot raises an
exception and sends a Telegram error alert.
""")
    st.code("""
# FVG_Strategy/core/bot_base.py — initialization

logger.info("Testing data connection...")
test_data = self.data_manager.get_current_data()

if test_data is None or test_data.empty:
    raise Exception("Failed to fetch initial market data")

if not self.data_manager.validate_data_quality(test_data):
    raise Exception("Data quality validation failed")

logger.info(f"Connection OK — {len(test_data)} candles loaded")
""", language="python")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  08 — REST API / COMMUNICATION SERVEUR / SGBD                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_08_server():
    st.header("08 — Server Communication & Database")

    # ── REST API Routes ──
    st.subheader("REST API Routes Consumed")
    st.markdown("""
The project **consumes** external REST APIs (it does not expose its own):

| API | Base URL | Endpoints Used | Auth Method |
|-----|----------|---------------|-------------|
| **Binance US** | `api.binance.us/api/v3` | `/klines` (candles), `/account`, `/order` | HMAC-SHA256 signature |
| **Binance Global** | `api.binance.com/api/v3` | `/klines` (candles) | None (public) |
| **Coinbase Pro** | `api.exchange.coinbase.com` | `/products/{pair}/candles` | None (public) |
| **TaapiIO** | `api.taapi.io` | `/candles`, `/rsi`, `/ema` | API key in query |
| **Yahoo Finance** | via `yfinance` | NQ=F futures data | None (public) |
| **Telegram** | `api.telegram.org/bot{token}` | `/sendMessage` | Bot token in URL |
""")

    # ── Authentication ──
    st.subheader("Authentication System")
    st.markdown("""
The project uses **three levels of authentication**:
""")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**API Keys (Binance)**")
        st.markdown("""
HMAC-SHA256 signing:
- Timestamp added to every request
- Query string hashed with secret key
- Signature appended as parameter
- API key sent in `X-MBX-APIKEY` header
""")
    with c2:
        st.markdown("**Bot Tokens (Telegram)**")
        st.markdown("""
Token-in-URL pattern:
- Bot token issued by @BotFather
- Chat ID identifies the destination
- HTTPS ensures transport encryption
- One token per bot instance
""")
    with c3:
        st.markdown("**Environment Variables**")
        st.markdown("""
Secret management:
- `.env` files on local and VPS
- `python-dotenv` loads at startup
- `os.getenv()` with empty defaults
- `.gitignore` excludes all `.env` files
""")

    # ── SGBD ──
    st.subheader("SGBD — Database Management Systems")
    st.markdown("""
**Two database engines** serve different purposes:

**SQLite** (per-bot, on VPS):
- One database per bot instance (e.g., `btc_liquidity_raid_v2.db`)
- Stores trade records, ML training features, adaptive risk state
- Schema migrations via `ALTER TABLE ADD COLUMN` (EAFP pattern)
- Zero configuration — embedded, no server process needed

**DuckDB** (analytics, local):
- Columnar storage optimised for analytical queries (OLAP)
- Aggregates all 9 bot databases into one analytics store
- Supports SQL window functions, percentiles, complex joins
- Used by the dashboard for cross-strategy portfolio analysis
""")

    # ── Data Modelling ──
    st.subheader("Data Modelling — Schema Design")

    st.markdown("**Trade record schema** (SQLite — per-bot):")
    st.code("""
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,              -- ISO 8601 entry time
    signal_type TEXT,            -- 'LONG' or 'SHORT'
    entry_price REAL,
    stop_loss REAL,
    take_profit REAL,
    exit_price REAL,
    position_size REAL,
    realized_pnl REAL,          -- Dollars
    realized_pnl_pct REAL,      -- Percentage
    status TEXT,                 -- 'open' or 'closed'
    killzone TEXT,               -- 'Asian', 'London', 'New York'
    sweep_type TEXT,             -- 'ASIA_HIGH', 'LONDON_LOW', etc.
    reason TEXT,                 -- Signal reason
    exit_timestamp TEXT,
    exit_reason TEXT,            -- 'TP', 'SL', 'TIME_EXIT', 'TRAILING'
    gamma_regime TEXT,           -- 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    confidence REAL,             -- WFO score (0-1)
    wfo_components_json TEXT,    -- JSON: 9 component scores
    regime_gate TEXT,            -- 'PASSED', 'CALM_SKIP', etc.
    dvol_percentile REAL,
    atr_percentile REAL,
    is_reentry INTEGER DEFAULT 0
);
""", language="sql")

    st.markdown("**ML training feature schema** (50+ columns per trade):")
    st.code("""
-- Subset of ml_training_data table columns:
entry_price, stop_loss, take_profit, direction,
signal_score, confluence_score, displacement_score,
regime, regime_strength, gamma_regime,
iv_percentile, distance_to_gamma_flip_pct,
sweep_depth_atr, sweep_volume_ratio, sweep_velocity,
asia_high, asia_low, london_high, london_low,
current_drawdown_pct, equity_curve_slope,
wfo_confidence, wfo_components_json, hmm_state,
-- ... plus MFE (max favourable excursion) and MAE tracking
mfe_price, mae_price, mfe_r, mae_r,
exit_price, exit_reason, pnl_dollars, r_multiple
""", language="sql")

    st.markdown("**DuckDB analytics schema** (unified across all bots):")
    st.code("""
-- backtrader_framework/data/duckdb_manager.py

CREATE TABLE IF NOT EXISTS backtest_trades (
    trade_id VARCHAR PRIMARY KEY,
    strategy_name VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    direction VARCHAR NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    entry_price DOUBLE NOT NULL,
    exit_time TIMESTAMP,
    exit_price DOUBLE,
    stop_loss DOUBLE NOT NULL,
    pnl_percent DOUBLE,
    r_multiple DOUBLE,
    mfe_percent DOUBLE,           -- Max Favourable Excursion
    mae_percent DOUBLE,           -- Max Adverse Excursion
    session VARCHAR,
    bars_held INTEGER
);
""", language="sql")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  09 — DEVOPS (GIT, DEPLOYMENT, TESTING)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_09_devops():
    st.header("09 — DevOps & Deployment")

    # ── Git ──
    st.subheader("Git Version Control")
    st.markdown("""
The project is managed as a **Git repository** with a single `main` branch.
The `.gitignore` file (99 rules) protects sensitive and large files:
""")
    st.code("""
# .gitignore (key rules)

# Secrets & Credentials
.env, .env.*, *.pem, *.key, credentials.json

# Proprietary Strategy Code (live trading logic)
FVG_Strategy/
Liquidity_Raid/
Momentum_Mastery/
SBS/

# Database Files (too large for Git)
*.db, *.sqlite3, *.duckdb

# ML Models and Results
*.pkl, *.joblib, *.h5, *.onnx
ml_models/
backtest_results/

# Deployment Scripts (contain server IPs)
deploy_to_vps.sh, upload_to_vps.sh
""", language="gitignore")

    st.markdown("""
**Recent commit history** (sample):
```
abe55b3  Add WFO signal scorer module and ML performance dashboard page
e6de3da  Inline WFO component classes to eliminate cross-package imports
1536dd3  Add SBS bots to dashboard, fix sync and concat warnings
56030f8  Rewrite README for institutional framing
abce15c  Institutional-grade upgrade: 60+ fixes across 6 domains
```
""")

    # ── Deployment ──
    st.subheader("Server Deployment")

    st.markdown("""
Bots are deployed to a **Linux VPS** running as **systemd services**:
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Deployment method: SCP + SSH**")
        st.code("""
# SBS/scripts/deploy_to_vps.sh

# 1. Package files (exclude secrets)
tar -czvf "/tmp/$PACKAGE" \\
    --exclude='*.pyc' \\
    --exclude='.env' \\
    --exclude='venv' \\
    bots/ requirements.txt

# 2. Upload via SCP
scp "/tmp/$PACKAGE" \\
    trader@YOUR_VPS:/tmp/

# 3. Deploy on VPS via SSH
ssh trader@YOUR_VPS << EOF
    tar -xzvf "/tmp/$PACKAGE"
    pip install -r requirements.txt
EOF
""", language="bash")
    with c2:
        st.markdown("**systemd service management**")
        st.code("""
# 9 bot services running on VPS:
#   fvg-btc, fvg-eth, fvg-nq
#   lr-btc, lr-eth
#   mm-btc, mm-eth
#   sbs-btc, sbs-eth

# Restart a bot after deployment:
sudo systemctl restart fvg-btc

# Check bot status:
sudo systemctl status fvg-btc

# View live logs:
journalctl -u fvg-btc -f

# Key systemd features:
# - Restart=always (auto-restart on crash)
# - WorkingDirectory per bot
# - Environment file for secrets
""", language="bash")

    st.markdown("""
The dashboard's **Deploy Bots page** (page 11) provides a GUI for this
process — one-click SCP upload, progress bar for bulk deployment, and
service restart buttons with status feedback via `st.session_state`.
""")

    st.divider()

    # ── Testing ──
    st.subheader("Testing")

    st.markdown("""
The `backtrader_framework/tests/` directory contains a **pytest** test suite:

| Test Module | What It Tests | Key Assertions |
|-------------|---------------|----------------|
| `test_validation.py` | OHLCV data quality checks | NaN handling, High >= Low, no duplicate timestamps |
| `test_indicators.py` | RSI and ATR calculations | RSI always in [0, 100], ATR always positive |
| `test_risk_management.py` | Position sizing, drawdown, daily limits | NAV-based sizing, drawdown from peak |
| `test_cpcv.py` | Combinatorial purged cross-validation | No data leakage between folds |
""")

    st.code("""
# backtrader_framework/tests/test_validation.py

import pytest
import pandas as pd
import numpy as np

class TestValidateOHLCV:
    def _make_df(self, n=100):
        \"\"\"Create a valid OHLCV DataFrame for testing.\"\"\"
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n)) * 0.3
        low = close - np.abs(np.random.randn(n)) * 0.3
        return pd.DataFrame({
            'Open': close, 'High': high, 'Low': low,
            'Close': close, 'Volume': np.random.rand(n) * 1000
        }, index=dates)

    def test_valid_data_passes(self):
        result = validate_ohlcv(self._make_df())
        assert len(result) == 100

    def test_nan_values_fixed(self):
        df = self._make_df()
        df.iloc[5, 0] = np.nan
        result = validate_ohlcv(df, fix=True)
        assert not result['Open'].isna().any()

    def test_strict_mode_raises(self):
        df = self._make_df()
        df.iloc[5, 0] = np.nan
        with pytest.raises(DataValidationError):
            validate_ohlcv(df, strict=True)

    def test_high_low_violation_fixed(self):
        df = self._make_df()
        df.iloc[10, 1] = df.iloc[10, 2] - 1  # High < Low
        result = validate_ohlcv(df, fix=True)
        assert (result['High'] >= result['Low']).all()
""", language="python")

    st.info("""
**Note on Docker:** This project uses **systemd** for process management
instead of Docker.  systemd provides equivalent capabilities for our use case:
auto-restart on crash, environment variable injection, log management via
`journalctl`, and per-service working directories.  Since each bot is a single
Python process (not a microservice architecture), containerisation would add
complexity without proportional benefit.
""")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  10 — FONCTIONNALITES DU PROJET                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_10_fonctionnalites():
    st.header("10 — Project Showcase & Future Development")

    # ── Proof of Work ──
    st.subheader("Proof of Work")
    st.markdown("""
This is not a theoretical project — **the bots are running live right now:**

| Evidence | Description |
|----------|-------------|
| **9 systemd services** | `fvg-btc`, `fvg-eth`, `fvg-nq`, `lr-btc`, `lr-eth`, `mm-btc`, `mm-eth`, `sbs-btc`, `sbs-eth` — all active on VPS |
| **Trade databases** | SQLite files growing daily with real signal records |
| **Telegram alerts** | Real-time notifications delivered to private channels |
| **Dashboard sync** | VPS databases pulled to local machine for live analytics |
| **Git history** | 23+ commits documenting iterative development |
| **WFO validation** | Parameters validated on 4,376 out-of-sample trades |
| **Monte Carlo** | 10,000 simulated equity paths before deployment |

The **Deploy Bots** dashboard page (page 11) allows one-click deployment of
updated code to the VPS, with progress tracking and service restart buttons.
""")

    # ── Incremental Usage ──
    st.subheader("Incremental Usage")
    st.markdown("""
The 24 dashboard pages represent an **incremental feature set** that grew
organically as the project evolved:

| Phase | Pages Added | Purpose |
|-------|-------------|---------|
| **Phase 1** — Core | Overview, Trade Journal, Equity Curves | Basic portfolio monitoring |
| **Phase 2** — Analysis | Session Analysis, Monthly Performance | Identify when and where bots perform |
| **Phase 3** — Optimisation | WFO Analysis, Monte Carlo, Bayesian Tuning | Validate and tune parameters |
| **Phase 4** — ML | ML Training, SHAP Analysis, ML Performance | Machine learning pipeline |
| **Phase 5** — Operations | Deploy Bots, Strategy Explainer, Deep Dive | Production management |
| **Phase 6** — Advanced | Stress Testing, Cross Asset, Quant Research Lab | Institutional analytics |
| **Phase 7** — Live Ops | Live Logs, Trade Monitor | Real-time VPS monitoring |
| **Phase 8** — Reporting | Academic Report, Dashboard Presentation | Documentation & presentation |
""")

    # ── Forms & Advanced Interfaces ──
    st.subheader("Forms & Advanced Interfaces")
    st.markdown("""
**Interactive controls used across the dashboard:**

- **Sidebar filters** — Radio buttons for data source (Live/Backtest/All),
  dropdowns for strategy and symbol selection, date range pickers
- **Sliders** — P&L range filter on Trade Journal page
- **Multi-select** — Combine multiple WFO results on Meta Strategy page
- **Checkboxes** — Toggle equity curve overlays (benchmark, regime bands)
- **Buttons with state** — Deploy buttons remember their result across
  Streamlit reruns via `st.session_state`
- **Progress bars** — Bulk deployment shows real-time upload progress
- **Forms** — Bayesian tuning page uses `st.form()` for parameter input
- **Column configs** — DataFrames styled with `st.column_config` for
  formatted numbers, currency display, and tooltips
""")

    # ── Future Development ──
    st.subheader("Future Development Options")
    st.markdown("""
| Feature | Description | Status |
|---------|-------------|--------|
| **Live order execution** | Connect Binance trading API to place real orders | API code ready, awaiting capital allocation |
| **ML trade filter** | XGBoost model gates entries based on 50+ features | Model trained, shadow-testing in progress |
| **Options overlay** | Real-time gamma/vanna exposure from Deribit | Prototype in FVG bot, expanding to LR/MM |
| **Multi-asset correlation** | Cross-asset regime detection (BTC+ETH+NQ) | Dashboard page exists, bot integration pending |
| **Mobile dashboard** | Streamlit Cloud deployment for phone access | Local-only currently, Cloud deployment planned |
| **Docker containerisation** | Migrate from systemd to Docker Compose | Would simplify multi-server deployment |
| **CI/CD pipeline** | GitHub Actions for automated testing + deployment | Currently manual SCP; GH Actions in roadmap |
""")

    # ── Commercial / Professional Exploitation ──
    st.subheader("Commercial & Professional Potential")
    st.markdown("""
This project demonstrates skills directly applicable to:

| Industry | Application |
|----------|-------------|
| **Quantitative finance** | Strategy development, backtesting, WFO, risk management |
| **FinTech** | Real-time data pipelines, API integration, dashboard development |
| **Data engineering** | Multi-source ETL, database design, schema migrations |
| **DevOps** | Server deployment, process management, monitoring |
| **Software architecture** | ABC patterns, DRY/SOLID principles, graceful degradation |
| **Machine learning** | Feature engineering, model training, SHAP interpretability |

The codebase architecture (base class + thin wrappers) could be packaged as a
**framework for building custom trading bots** — a user would only need to
implement the abstract methods for their strategy, inheriting all infrastructure.
""")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  11 — BACKTESTING & QUANTITATIVE METHODS                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tab_11_backtesting():
    st.header("11 — Backtesting & Quantitative Methods")

    st.markdown("""
This section covers the **quantitative backtesting infrastructure** that
validates every strategy before deployment.  The system implements
Walk-Forward Optimisation (WFO), Monte Carlo simulation, Hidden Markov Model
regime detection, implied volatility gating, and Bayesian optimisation — all
built from first principles in Python.
""")

    # ══════════════════════════════════════════════════════════════════════
    #  11.1 — WALK-FORWARD OPTIMISATION (WFO) ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════

    st.subheader("11.1 — Walk-Forward Optimisation (WFO)")

    st.markdown("""
**Walk-Forward Optimisation** prevents curve-fitting by splitting historical
data into sequential **in-sample (IS)** and **out-of-sample (OOS)** windows.
Parameters are optimised on IS data, then validated on unseen OOS data.
The engine supports both **anchored** (expanding) and **rolling** windows.
""")

    st.code("""
Anchored (expanding) window — IS grows, OOS slides forward:

  ┌─────────── IS Window 1 ──────────┐ embargo ┌── OOS 1 ──┐
  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│▒▒▒▒▒▒▒▒▒▒▒│
  0                                 500       668          768

  ┌──────────────── IS Window 2 ─────────────────┐ embargo ┌── OOS 2 ──┐
  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│▒▒▒▒▒▒▒▒▒▒▒│
  0                                             600       768          868

  ▓ = In-Sample (train)   ░ = Embargo gap   ▒ = Out-of-Sample (validate)

  The embargo gap = max_trade_bars (168 bars on 15m ≈ 42 hours).
  This prevents IS trades from "leaking" exit prices into OOS territory.
""", language="text")

    st.markdown("**`WFOConfig` — the dataclass controlling the entire pipeline:**")
    st.code("""
# backtrader_framework/optimization/wfo_engine.py

@dataclass
class WFOConfig:
    # Window sizing
    train_window_bars: int = 500      # Bars in each IS window
    test_window_bars: int = 100       # Bars in each OOS window
    step_bars: int = 100              # Slide forward by N bars per iteration
    anchored: bool = True             # Expanding (True) or rolling (False) window
    min_train_bars: int = 250         # Minimum IS size before optimisation starts
    min_ema_warmup: int = 250         # Bars needed for EMA200 to stabilise
    min_trades_per_window: int = 3    # Reject param combos with fewer IS trades

    # Transaction costs (round-trip model)
    costs: TransactionCosts = field(default_factory=TransactionCosts)
    max_trade_bars: int = 168         # Force-close trades after N bars (= embargo)

    # Optimisation
    optimization_metric: str = 'expectancy'  # 'expectancy' | 'profit_factor' | 'sharpe' | 'total_r'
    max_param_combos: int = 1000      # Cap total grid search combos
    grid_mode: str = 'auto'           # 'full' | 'random' (Sobol) | 'auto'
    random_samples: int = 150         # Combos to sample in random mode

    # Advanced
    use_bayesian: bool = False        # Use Optuna TPE instead of grid search
    use_hmm_regime: bool = False      # HMM regime detection per window
    hmm_position_sizing: bool = False # Scale OOS R-multiples by HMM probability
    parallel: bool = True             # Multiprocess combo evaluation
    n_workers: int = 0                # 0 = auto (cpu_count - 1)
""", language="python")

    st.markdown("**The `for_timeframe()` factory** scales window sizes so that each window covers "
                "the same calendar duration regardless of bar frequency:")
    st.code("""
# 4h bars: 100 bars = 17 days of market data
# 15m bars: need 16× more bars for the same 17 days

@classmethod
def for_timeframe(cls, tf: str, **overrides) -> 'WFOConfig':
    multiplier = {'15m': 16, '30m': 8, '1h': 4, '4h': 1, '1d': 1}[tf]
    return cls(
        train_window_bars=500 * multiplier,
        test_window_bars=100 * multiplier,
        step_bars=100 * multiplier,
        max_trade_bars=168 * multiplier,
        **overrides,
    )
""", language="python")

    st.markdown("""
**High-level execution flow:**
""")
    st.code("""
WalkForwardOptimizer.run_optimization(symbol, timeframe)
│
├── 1. Fetch OHLCV from DuckDB
├── 2. IndicatorEngine.calculate(df) → adds 30+ indicator columns
├── 3. _generate_windows(df) → list of (IS_start, IS_end, OOS_start, OOS_end)
│
├── 4. FOR EACH WINDOW:
│   ├── _optimize_window()
│   │   ├── IN-SAMPLE: evaluate all param combos (grid or Bayesian)
│   │   │   ├── adapter.generate_signals() or adapter.execute_signals()
│   │   │   ├── TradeSimulator.simulate() per signal
│   │   │   └── score_trades() → select best combo
│   │   │
│   │   └── OUT-OF-SAMPLE: apply best params to unseen data
│   │       ├── Generate signals with best_params
│   │       ├── Simulate trades
│   │       └── (optional) HMM regime position sizing
│   │
│   └── Collect OOS trades
│
└── 5. _compile_results()
    ├── Deduplicate overlapping OOS trades
    ├── StatisticalTests.run_all() → win rate, Sharpe, profit factor, CIs
    ├── Overfit ratio = OOS_mean_R / IS_mean_R
    ├── Deflated Sharpe Ratio (adjusts for multiple testing)
    ├── MonteCarloAnalysis.bootstrap_ci() → 10,000 resamples
    └── MonteCarloAnalysis.equity_fan() → 1,000 equity paths
""", language="text")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    #  11.2 — WFO SCORING SYSTEM
    # ══════════════════════════════════════════════════════════════════════

    st.subheader("11.2 — WFO Scoring System")

    st.markdown("""
Every parameter combination is scored on its **in-sample trades** using one of
four metrics.  The metric operates on **R-multiples** (profit/loss normalised
by initial risk), not raw dollar P&L — this makes scores comparable across
assets with different price levels.
""")

    st.code("""
# backtrader_framework/optimization/parallel.py

def score_trades(trades: list, metric: str) -> float:
    \"\"\"Score a set of trades using the given optimization metric.\"\"\"
    if len(trades) < 2:
        return -float('inf')                  # Reject trivial samples

    r_values = [t.r_multiple_after_costs for t in trades]

    if metric == 'expectancy':
        # E = WR × avg_win + (1 - WR) × avg_loss
        wins = [r for r in r_values if r > 0]
        losses = [r for r in r_values if r <= 0]
        wr = len(wins) / len(r_values)
        avg_w = float(np.mean(wins)) if wins else 0
        avg_l = float(np.mean(losses)) if losses else 0
        return wr * avg_w + (1 - wr) * avg_l

    elif metric == 'profit_factor':
        # PF = gross_profit / |gross_loss|
        gp = sum(r for r in r_values if r > 0)
        gl = abs(sum(r for r in r_values if r < 0))
        return gp / gl if gl > 0 else 0

    elif metric == 'sharpe':
        # Sharpe = mean(R) / std(R)
        m = np.mean(r_values)
        s = np.std(r_values, ddof=1)
        return m / s if s > 0 else 0

    else:  # total_r
        return sum(r_values)
""", language="python")

    st.markdown("""
| Metric | Formula | Best for |
|--------|---------|----------|
| **Expectancy** | `WR × avg_win + (1 - WR) × avg_loss` | General-purpose (default) |
| **Profit Factor** | `gross_profit / \\|gross_loss\\|` | Conservative strategies |
| **Sharpe** | `mean(R) / std(R, ddof=1)` | Risk-adjusted returns |
| **Total R** | `sum(R-multiples)` | Aggressive: maximise raw P&L |
""")

    st.markdown("**Transaction cost model** — applied to every simulated trade:")
    st.code("""
# backtrader_framework/optimization/wfo_engine.py

@dataclass
class TransactionCosts:
    spread_pct: float = 0.0005       # Bid-ask spread
    commission_pct: float = 0.001    # Exchange fee
    slippage_pct: float = 0.0003     # Price impact

    @classmethod
    def for_asset(cls, symbol: str) -> 'TransactionCosts':
        if symbol in ('BTC', 'ETH'):
            return cls(spread_pct=0.0001, commission_pct=0.00055, slippage_pct=0.0001)
        if symbol == 'NQ':
            return cls(spread_pct=0.0001, commission_pct=0.0002, slippage_pct=0.0001)
        return cls()
""", language="python")

    st.markdown("**Trade Simulator** — walks bar-by-bar from entry, applying SL/TP/trailing:")
    st.code("""
# backtrader_framework/optimization/wfo_engine.py — TradeSimulator.simulate()

# Simplified structure (actual code is 150+ lines):
for i in range(entry_bar + 1, entry_bar + max_bars):
    h, lo = highs[i], lows[i]

    # Track maximum favourable/adverse excursion
    mfe = max(mfe, (h - entry) / risk)      # Best unrealised profit
    mae = max(mae, (entry - lo) / risk)      # Worst unrealised loss

    # TP1 hit → move stop-loss to breakeven (lock in 0R floor)
    if not tp1_hit and h >= take_profit_1:
        tp1_hit = True
        stop_loss = entry_price              # Breakeven

    # ATR trailing stop (after TP1)
    if tp1_hit and atrs is not None:
        trail = high_water - trail_atr_mult * atrs[i]
        stop_loss = max(stop_loss, trail)    # Only ratchet up

    # Exit conditions
    if lo <= stop_loss:
        exit_price = stop_loss               # Loss or breakeven
        break
    if h >= take_profit_2:
        exit_price = take_profit_2           # Full target hit
        break

# Final R-multiple (after costs):
raw_r = (exit_price - entry_price) / risk
cost_r = (entry_commission + exit_costs) / risk
r_after_costs = raw_r - cost_r
""", language="python")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    #  11.3 — MONTE CARLO SIMULATION
    # ══════════════════════════════════════════════════════════════════════

    st.subheader("11.3 — Monte Carlo Simulation")

    st.markdown("""
After WFO produces a set of out-of-sample trades, Monte Carlo simulation
quantifies the **range of possible outcomes** by resampling those trades
thousands of times.  This answers questions like: *"If I traded this strategy
1,000 times, what's the worst drawdown I might see?"*

The implementation uses **block bootstrap resampling**: rather than shuffling
individual trades (which destroys serial correlation), it samples contiguous
blocks to preserve regime clustering.
""")

    st.code("""
Algorithm: Block Bootstrap Monte Carlo

Input:  R-multiples from OOS trades = [r₁, r₂, ..., rₙ]
Config: n_resamples = 10,000, block_size = 3

FOR each of 10,000 simulations:
    1. Build a resampled trade sequence of length n:
       - Pick a random start index
       - Copy block_size consecutive trades
       - Repeat until n trades are collected

    2. Compute 6 metrics on this resampled sequence:
       - Mean R-multiple
       - Win rate (% of R > 0)
       - Expectancy (WR × avg_win + (1 - WR) × avg_loss)
       - Profit factor (gross_wins / |gross_losses|)
       - Sharpe ratio (mean / std)
       - Max drawdown (peak-to-trough of cumulative R)

OUTPUT: For each metric, extract the 2.5th and 97.5th percentile
        → 95% confidence interval
""", language="text")

    st.markdown("**Core implementation** (with optional Numba JIT acceleration):")
    st.code("""
# backtrader_framework/optimization/wfo_engine.py — MonteCarloAnalysis

@staticmethod
def bootstrap_ci(trades, n_resamples=10000, confidence=0.95, block_size=1):
    r_values = np.array([t.r_multiple_after_costs for t in trades])
    n = len(r_values)

    # Pre-allocate metric arrays
    mean_rs = np.empty(n_resamples)
    win_rates = np.empty(n_resamples)
    expectancies = np.empty(n_resamples)
    profit_factors = np.empty(n_resamples)
    sharpes = np.empty(n_resamples)
    max_dds = np.empty(n_resamples)

    if HAS_NUMBA:
        # Numba kernel: tight loop, ~100x faster than pure Python
        _mc_bootstrap_ci_kernel(r_values, n_resamples, block_size,
                                mean_rs, win_rates, expectancies,
                                profit_factors, sharpes, max_dds)
    else:
        # Pure Python fallback (identical logic)
        for i in range(n_resamples):
            sample = r_values[np.random.randint(0, n, size=n)]
            mean_rs[i] = np.mean(sample)
            # ... compute other 5 metrics ...

    # Extract percentile-based confidence intervals
    alpha = 1 - confidence    # 0.05
    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100  # 2.5, 97.5

    return {
        'p_profitable': float(np.mean(mean_rs > 0)),
        'mean_r_ci':       (np.percentile(mean_rs, lo),       np.percentile(mean_rs, hi)),
        'win_rate_ci':     (np.percentile(win_rates, lo),     np.percentile(win_rates, hi)),
        'expectancy_ci':   (np.percentile(expectancies, lo),  np.percentile(expectancies, hi)),
        'profit_factor_ci':(np.percentile(profit_factors, lo),np.percentile(profit_factors, hi)),
        'sharpe_ci':       (np.percentile(sharpes, lo),       np.percentile(sharpes, hi)),
        'max_drawdown_ci': (np.percentile(max_dds, lo),       np.percentile(max_dds, hi)),
    }
""", language="python")

    st.markdown("**Equity fan chart** — visualises the distribution of possible equity paths:")
    st.code("""
# MonteCarloAnalysis.equity_fan()

@staticmethod
def equity_fan(trades, n_paths=1000, block_size=1):
    r_values = np.array([t.r_multiple_after_costs for t in trades])
    n = len(r_values)
    all_paths = np.empty((n_paths, n))

    # Generate 1,000 resampled cumulative equity paths
    for i in range(n_paths):
        sample = r_values[np.random.randint(0, n, size=n)]
        all_paths[i] = np.cumsum(sample)

    # Extract percentile bands at each trade index
    return {
        'percentiles': {
            '5':  np.percentile(all_paths, 5, axis=0).tolist(),   # Worst 5%
            '25': np.percentile(all_paths, 25, axis=0).tolist(),
            '50': np.percentile(all_paths, 50, axis=0).tolist(),  # Median
            '75': np.percentile(all_paths, 75, axis=0).tolist(),
            '95': np.percentile(all_paths, 95, axis=0).tolist(),  # Best 5%
        },
        'actual_equity': np.cumsum(r_values).tolist(),
        'median_max_dd': float(np.median(per_path_max_drawdowns)),
    }
""", language="python")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    #  11.4 — HMM REGIME DETECTION
    # ══════════════════════════════════════════════════════════════════════

    st.subheader("11.4 — HMM Regime Detection (Hidden Markov Model)")

    st.markdown("""
The WFO engine includes a **2-state Gaussian Hidden Markov Model** that
classifies the market into *calm* and *volatile* regimes.  This is used to
gate entries (only trade during volatile regimes) and scale position sizes
by regime probability.

**Three layers of information leakage prevention:**

| Layer | Protection | Why it matters |
|-------|-----------|----------------|
| 1. IS-only fitting | HMM is trained exclusively on in-sample data | OOS bars are never seen during model fitting |
| 2. IS-only standardisation | Feature mean/std computed from IS only | Prevents OOS statistics from leaking into normalisation |
| 3. Forward-filter only | OOS inference uses `P(state_t \\| x_1:t)` | No backward pass — each bar only uses data up to that point |
""")

    st.code("""
# backtrader_framework/optimization/hmm_regime.py

class GaussianHMM:
    \"\"\"2-state multivariate Gaussian HMM with diagonal covariance.\"\"\"

    def fit(self, X: np.ndarray) -> 'GaussianHMM':
        T, D = X.shape

        # ── K-Means Initialisation ──
        # Split by median of first feature (RealizedVol)
        med = np.median(X[:, 0])
        self.mu[0] = np.mean(X[X[:, 0] <= med], axis=0)   # Calm centre
        self.mu[1] = np.mean(X[X[:, 0] > med], axis=0)    # Volatile centre

        # Sticky transition matrix: 90% stay in same state
        self.A = np.full((2, 2), 0.05)
        np.fill_diagonal(self.A, 0.90)

        # ── Baum-Welch EM (up to 100 iterations) ──
        for iteration in range(self.max_iter):
            # Forward pass: log α(t,k) = log P(x_1:t, state_t=k)
            for t in range(1, T):
                for j in range(K):
                    log_alpha[t, j] = logsumexp(
                        log_alpha[t-1] + log(A[:, j])
                    ) + log_emission[t, j]

            # Backward pass: log β(t,k) = log P(x_t+1:T | state_t=k)
            for t in range(T-2, -1, -1):
                for i in range(K):
                    log_beta[t, i] = logsumexp(
                        log(A[i, :]) + log_emission[t+1] + log_beta[t+1]
                    )

            # Posterior: γ(t,k) = P(state_t=k | x_1:T)
            gamma = softmax(log_alpha + log_beta, axis=1)

            # M-step: update μ, σ, A, π from γ-weighted statistics
            for k in range(K):
                self.mu[k] = Σ(γ[:,k] * X) / Σ(γ[:,k])
                self.sigma[k] = sqrt(Σ(γ[:,k] * (X - μ[k])²) / Σ(γ[:,k]))

        # Relabel: state 0 = calm (lower variance in RealizedVol dimension)
        if self.mu[1, vol_dim] < self.mu[0, vol_dim]:
            swap(state_0, state_1)
""", language="python")

    st.markdown("**Forward filter** — causal inference for OOS (no future leakage):")
    st.code("""
def forward_filter(self, X: np.ndarray) -> np.ndarray:
    \"\"\"P(state_t | x_1:t) — uses only past and current data.\"\"\"
    log_alpha = np.zeros((T, K))
    log_alpha[0] = log(π) + log_emission(X[0])

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = logsumexp(
                log_alpha[t-1] + log(A[:, j])
            ) + log_emission(X[t], j)

        # Normalise at each step (prevents underflow)
        log_alpha[t] -= logsumexp(log_alpha[t])

    return exp(log_alpha)  # (T, 2) → state probabilities per bar
""", language="python")

    st.markdown("**Position sizing by regime probability:**")
    st.markdown("""
| Condition | Size Multiplier | Interpretation |
|-----------|----------------|----------------|
| P(calm) ≥ 0.70 | **1.0×** | High confidence calm → full size |
| P(calm) ≥ 0.55 | **0.7×** | Moderate calm → reduced size |
| P(volatile) ≥ 0.60 | **0.3×** | Volatile regime → defensive |
| Otherwise | **0.5×** | Uncertain → half size |

**Features used:** `[LogReturn, RealizedVol20]` — both computed from price data only.
""")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    #  11.5 — DVOL (DERIBIT IMPLIED VOLATILITY INDEX)
    # ══════════════════════════════════════════════════════════════════════

    st.subheader("11.5 — DVOL (Deribit Implied Volatility Index)")

    st.markdown("""
**DVOL** is an external implied volatility index derived from Bitcoin options
on the Deribit exchange — analogous to the VIX for the S&P 500.  It measures
the market's *expectation* of future volatility, as opposed to realised
volatility which looks backward.

The WFO engine loads hourly historical DVOL data and merges it into the
strategy DataFrame via **forward-fill**, bridging the frequency gap between
hourly DVOL and 15-minute strategy bars.
""")

    st.code("""
# backtrader_framework/optimization/indicators.py — DVOL loading

import json

dvol_path = 'Liquidity_Raid/Research/btc_dvol_hourly.json'
# JSON format: [{"timestamp": 1619820000000, "dvol": 75.25}, ...]

with open(dvol_path) as f:
    dvol_raw = json.load(f)

dvol_df = pd.DataFrame(dvol_raw)
dvol_df['timestamp'] = pd.to_datetime(
    dvol_df['timestamp'], unit='ms', utc=True
).dt.tz_localize(None)
dvol_df = dvol_df.set_index('timestamp').sort_index()

# Forward-fill: each 15m bar inherits the most recent hourly DVOL
df['DVOL'] = dvol_df['dvol'].reindex(df.index, method='ffill')
""", language="python")

    st.markdown("**IV regime classification and directional gating:**")
    st.code("""
# backtrader_framework/optimization/strategy_adapters/fvg_adapter.py

# Classify the current bar's IV regime
if dvol < 45:
    iv_regime = 'LOW'
elif dvol < 65:
    iv_regime = 'MED'
else:
    iv_regime = 'HIGH'

# RETEST signals — directional gate:
if iv_regime == 'LOW' and direction == 'SHORT':
    continue    # Low IV → suppress short entries
if iv_regime == 'HIGH' and direction == 'LONG':
    continue    # High IV → suppress long entries

# iFVG signals — reversed logic (works best in high volatility):
if iv_regime == 'LOW' and direction == 'LONG':
    continue    # Low IV → only short iFVGs allowed
""", language="python")

    st.markdown("""
| DVOL Range | Regime | Retest Gate | iFVG Gate | Study Result |
|-----------|--------|-------------|-----------|-------------|
| < 45 | **LOW** | Block SHORT | Block LONG | Suppress counter-trend |
| 45 – 65 | **MED** | Allow all | Allow all | positive in regime (per internal DVOL study) |
| ≥ 65 | **HIGH** | Block LONG | Allow all | positive for iFVG (per internal DVOL study) |
""")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    #  11.6 — ADX (AVERAGE DIRECTIONAL INDEX)
    # ══════════════════════════════════════════════════════════════════════

    st.subheader("11.6 — ADX (Average Directional Index)")

    st.markdown("""
**ADX** (Average Directional Index) is a Wilder indicator that measures
**trend strength** on a 0–100 scale — it does not indicate direction, only
how strongly the market is trending.  ADX > 25 signals a meaningful trend;
ADX < 20 suggests consolidation.

The computation follows Wilder's original 5-step process:
""")

    st.markdown("""
**Step 1 — Directional Movement (DM):**
- `+DM = High[today] - High[yesterday]` (if positive and > −DM, else 0)
- `−DM = Low[yesterday] - Low[today]` (if positive and > +DM, else 0)

**Step 2 — True Range (TR):**
- `TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)`

**Step 3 — Smoothed averages** using Wilder's EMA (α = 1/14):
- `ATR = EWM(TR, α=1/14)`
- `+DI = 100 × EWM(+DM, α=1/14) / ATR`
- `−DI = 100 × EWM(−DM, α=1/14) / ATR`

**Step 4 — Directional Movement Index:**
- `DX = 100 × |+DI − −DI| / (+DI + −DI)`

**Step 5 — ADX (smoothed DX):**
- `ADX = EWM(DX, α=1/14)`
""")

    st.code("""
# backtrader_framework/optimization/indicators.py

@staticmethod
def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']

    # Step 1: Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Step 2: True Range
    tr = pd.concat([
        high - low,
        np.abs(high - close.shift()),
        np.abs(low - close.shift())
    ], axis=1).max(axis=1)

    # Steps 3-4: Wilder's EMA smoothing (alpha = 1/period)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period,
                                   adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period,
                                     adjust=False).mean() / (atr + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    # Step 5: Smooth DX into ADX
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
""", language="python")

    st.markdown("""
**Integration with WFO regime classification:**

| ADX Value | ATR vs Average | Classification |
|-----------|---------------|---------------|
| ADX > 30 | — | `trending_up` or `trending_down` (based on EMA position) |
| ADX ≤ 30 | ATR > 1.8× average | `volatile` (expansion without trend) |
| ADX ≤ 30 | ATR ≤ 1.8× average | `ranging` (consolidation) |
""")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    #  11.7 — BAYESIAN OPTIMISATION (OPTUNA / TPE)
    # ══════════════════════════════════════════════════════════════════════

    st.subheader("11.7 — Bayesian Optimisation (Optuna / TPE)")

    st.markdown("""
As an alternative to grid search, the WFO engine supports **Bayesian
optimisation** via Optuna's **Tree-structured Parzen Estimator (TPE)**.

**Why Bayesian over grid search?**

| Aspect | Grid Search | Bayesian (TPE) |
|--------|------------|----------------|
| Exploration | Exhaustive (or Sobol quasi-random) | Guided by surrogate model |
| Efficiency | Tests every combo blindly | Focuses on promising regions |
| Combos needed | 150+ for 5-parameter space | 50–100 trials often suffice |
| Overhead | Near-zero | Small per-trial overhead |
| Best for | Low-dimensional, fast combos | High-dimensional, slow combos |

**TPE principle:** Instead of modelling `f(params)` directly, TPE builds two
density estimators — `l(params)` for "good" parameter regions and `g(params)`
for "bad" regions — then suggests the next trial by maximising `l(x)/g(x)`.
""")

    st.code("""
# backtrader_framework/optimization/bayesian_tuner.py

import optuna

def tune_strategy_params(self, param_specs, objective_fn):
    def objective(trial):
        params = {}
        for spec in param_specs:
            params[spec.name] = suggest_from_param_spec(trial, spec)
        return objective_fn(params)    # Returns score (e.g. expectancy)

    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10,           # Random exploration before TPE kicks in
    )
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        objective,
        n_trials=100,                  # Max evaluations per window
        timeout=60,                    # Hard time limit (seconds)
    )
    return study.best_params, study.best_value
""", language="python")

    st.markdown("**Parameter suggestion** supports int, float, log-scale, and stepped ranges:")
    st.code("""
# backtrader_framework/optimization/param_grid.py

def suggest_from_param_spec(trial, spec: ParamSpec):
    if spec.param_type == 'int':
        return trial.suggest_int(spec.name, int(spec.min_val),
                                 int(spec.max_val), step=int(spec.step))
    elif spec.log_scale and spec.min_val > 0:
        # Log-scale for parameters spanning orders of magnitude
        return trial.suggest_float(spec.name, spec.min_val,
                                   spec.max_val, log=True)
    elif spec.step > 0:
        return trial.suggest_float(spec.name, spec.min_val,
                                   spec.max_val, step=spec.step)
    else:
        return trial.suggest_float(spec.name, spec.min_val, spec.max_val)
""", language="python")

    st.markdown("""
**Sobol quasi-random sampling** (used in `grid_mode='random'`):

When grid search has too many combos (> `max_param_combos`), the engine
switches to **Sobol sequences** — a low-discrepancy quasi-random sampling
method that provides better coverage of the parameter space than pseudo-random
sampling.  The default parameters are always included as the first sample.
""")
    st.code("""
# backtrader_framework/optimization/param_grid.py

from scipy.stats.qmc import Sobol

def generate_random_grid(param_specs, n_samples=150, seed=42):
    grid = [get_defaults(param_specs)]    # Always include defaults

    # Sobol sequence for uniform coverage (power-of-2 samples)
    m = int(np.ceil(np.log2(max(n_samples - 1, 2))))
    sampler = Sobol(d=len(param_specs), scramble=True, seed=seed)
    unit_samples = sampler.random_base2(m)[:n_samples - 1]

    # Scale [0, 1] samples to parameter ranges
    for sample in unit_samples:
        params = {}
        for i, spec in enumerate(param_specs):
            if spec.log_scale and spec.min_val > 0:
                val = exp(log(spec.min_val) + sample[i] *
                          (log(spec.max_val) - log(spec.min_val)))
            else:
                val = spec.min_val + sample[i] * (spec.max_val - spec.min_val)
            params[spec.name] = round_to_step(val, spec)
        grid.append(params)
    return grid
""", language="python")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN — 11 TABS ALIGNED TO EVALUATION GRID                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    tabs = st.tabs([
        "01 Conception",
        "02 Structure",
        "03 Methodology",
        "04 Python",
        "05 UI",
        "06 Ext. Modules",
        "07 Data & Viz",
        "08 Server & DB",
        "09 DevOps",
        "10 Showcase",
        "11 Backtesting",
    ])

    with tabs[0]:
        tab_01_conception()
    with tabs[1]:
        tab_02_structure()
    with tabs[2]:
        tab_03_methode()
    with tabs[3]:
        tab_04_python()
    with tabs[4]:
        tab_05_ui()
    with tabs[5]:
        tab_06_modules()
    with tabs[6]:
        tab_07_data()
    with tabs[7]:
        tab_08_server()
    with tabs[8]:
        tab_09_devops()
    with tabs[9]:
        tab_10_fonctionnalites()
    with tabs[10]:
        tab_11_backtesting()


main()
