"""
Dashboard & Project Presentation
==================================

Professional presentation and learning material for the Python Software
Engineering course.  Uses the Streamlit dashboard and its components as the
primary object of study, while also introducing the trading bot project and
backtesting infrastructure.

This page is designed to be read top-to-bottom as a walkthrough.  Each tab
builds on the previous one, from high-level architecture to low-level code.
"""

import streamlit as st

st.set_page_config(
    page_title="Dashboard Presentation",
    page_icon="\U0001F4DA",
    layout="wide",
)

st.markdown("""
<style>
    .pres-title { font-size: 2.6rem; font-weight: 700; margin-bottom: 0.2rem; }
    .pres-sub   { font-size: 1.15rem; color: #888; margin-bottom: 2rem; }
    .code-label { font-size: 0.85rem; color: #aaa; margin-bottom: -0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="pres-title">Algorithmic Trading Dashboard &mdash; Code Walkthrough</p>'
    '<p class="pres-sub">'
    'Python Software Engineering &mdash; '
    'Streamlit dashboard as the primary object of study &mdash; Nelson Onu'
    '</p>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_01_overview():
    st.header("1 — Project Overview")

    st.markdown("""
This project builds a **production-grade algorithmic trading infrastructure**
in Python.  It consists of three main layers:

1. **Trading Bots** — four autonomous strategies that monitor live markets
   24/7, detect trading opportunities, send Telegram alerts, and log every
   decision to a database.
2. **Backtesting Framework** — a historical simulation engine (Backtrader)
   with Walk-Forward Optimisation and Monte Carlo analysis to validate
   strategies before deploying them.
3. **Streamlit Dashboard** — a 24-page interactive web application that
   aggregates data from all bots, visualises performance, and provides
   operational controls (deployment, sync, manual trading).

**This presentation focuses on layer 3 (the dashboard)**, treating it as a
complete Python web application worthy of study.  We will examine every
module, every component, and every page — from the entry point to the
individual chart builders.
""")

    st.subheader("High-Level Architecture")
    st.code("""
    ┌──────────────────────────────────────────────────────────┐
    │                    MARKET APIS                           │
    │  Binance REST/WS  ·  Coinbase  ·  Yahoo Finance         │
    └──────────────────────────┬───────────────────────────────┘
                               │
    ┌──────────────────────────▼───────────────────────────────┐
    │              TRADING BOTS  (VPS — Linux)                 │
    │                                                          │
    │  FVG Bot (BTC/ETH/NQ)     Liquidity Raid (BTC/ETH)      │
    │  Momentum Mastery (BTC/ETH)   SBS (BTC/ETH)             │
    │                                                          │
    │  Each bot:                                               │
    │   - Fetches market data via DataManager                  │
    │   - Runs strategy analysis (indicators + pattern detect) │
    │   - Scores signals via WFO Signal Scorer (0.0 → 1.0)    │
    │   - Sends Telegram alert if score > threshold            │
    │   - Logs trade to SQLite + ML training database          │
    │   - Sleeps, then repeats                                 │
    └──────────────────────────┬───────────────────────────────┘
                               │  SCP / rsync (SSH)
    ┌──────────────────────────▼───────────────────────────────┐
    │              STREAMLIT DASHBOARD  (Local)                │
    │                                                          │
    │  app.py ──→ config.py ──→ data/ ──→ components/ ──→     │
    │            24 pages in pages/                             │
    │                                                          │
    │  Reads synced SQLite DBs, normalises to unified schema,  │
    │  renders interactive charts, KPIs, and filters.          │
    └──────────────────────────────────────────────────────────┘
""", language="text")

    st.subheader("What You Will Learn")
    st.markdown("""
| Tab | Topic | Key Concepts |
|-----|-------|-------------|
| 2 | Streamlit Fundamentals | Page auto-discovery, session state, reactive reruns |
| 3 | Configuration System | Centralised registry, environment variables, `pathlib` |
| 4 | Data Pipeline | `@st.cache_data`, schema normalisation, deduplication |
| 5 | VPS Operations | `subprocess.run`, rsync/SCP, DB protection, systemd |
| 6 | API Layer | REST clients, HMAC-SHA256 auth, data fallback chains |
| 7 | Component Library | Plotly chart builders, KPI cards, reusable filters |
| 8 | Dashboard Pages | Walkthrough of all 24 pages — widgets, patterns, code |
| 9 | Bot Architecture | ABC, `@dataclass`, Enum, asyncio, composition |
| 10 | Synthesis | Key takeaways, Python patterns catalogue |
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — STREAMLIT FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════

def tab_02_streamlit():
    st.header("2 — How Streamlit Works")

    st.markdown("""
**Streamlit** is a Python-native web framework for data applications.  Unlike
Flask (which requires HTML templates, CSS, and JavaScript), Streamlit lets you
build interactive web pages using only Python.  Every time a user interacts
with a widget (clicks a button, moves a slider), the **entire script reruns
from top to bottom**.  This is called the **reactive execution model**.
""")

    st.info("""
**Key mental model:** A Streamlit app is not a long-running server like Flask.
It is a Python script that gets re-executed on every user interaction.  The
framework captures the output of each `st.xxx()` call and renders it in the
browser.
""")

    # ── Entry point ──
    st.subheader("Entry Point: app.py (Home Page)")
    st.markdown("""
The dashboard starts with `app.py` — this is the file you run with
`streamlit run app.py`.  It serves as the landing page and sets up global
configuration.
""")
    st.code("""
# dashboard/app.py — 86 lines

import streamlit as st

# 1. Page configuration — must be the FIRST Streamlit call
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",               # Use full browser width
    initial_sidebar_state="expanded",
)

# 2. Auto-refresh — optional third-party widget
from streamlit_autorefresh import st_autorefresh
refresh_label = st.sidebar.selectbox("Auto-Refresh", ["Off", "30s", "1m"])
# st_autorefresh triggers a full page rerun at the specified interval

# 3. VPS sync controls in the sidebar
if st.sidebar.button("Sync from VPS", type="primary"):
    with st.sidebar.status("Syncing...", expanded=True) as status:
        results = sync_all_vps_data()     # SCP files from VPS
        status.update(label="Done", state="complete")
    st.session_state["last_sync"] = datetime.now()
    # session_state persists across reruns — the sync time survives

# 4. Quick stats on the landing page
from data.data_loader import get_all_trades
df = get_all_trades()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Trades", f"{len(df):,}")
c2.metric("Live Trades", f"{(df['source'] == 'Live').sum():,}")
""", language="python")

    st.markdown("""
**Key patterns in this file:**
- `st.set_page_config()` — must be the first Streamlit call; sets title, icon,
  and layout mode.
- `st.sidebar.button()` — places a button in the sidebar; returns `True` on
  the rerun where it was clicked.
- `st.session_state` — a dictionary that persists across reruns.  Without it,
  every rerun would lose all state (because the script re-executes from scratch).
- `st.columns(4)` — creates a 4-column layout; each column's `.metric()` call
  renders a KPI card with title, value, and optional delta.
""")

    st.divider()

    # ── Page auto-discovery ──
    st.subheader("Page Auto-Discovery")
    st.markdown("""
Streamlit automatically discovers Python files in the `pages/` directory and
adds them to the sidebar navigation.  The file naming convention determines
the sort order and display name:

```
pages/
├── 1_Overview.py              → "Overview"
├── 2_Strategy_Explainer.py    → "Strategy Explainer"
├── 3_Strategy_Deep_Dive.py    → "Strategy Deep Dive"
├── 3b_Live_Logs.py            → "Live Logs"
├── 4_Trade_Journal.py         → "Trade Journal"
├── 5_Equity_Curves.py         → "Equity Curves"
├── ...
├── 13_Meta_Strategy.py        → "Meta Strategy"
├── 13b_Trade_Monitor.py       → "Trade Monitor"
├── 14_SHAP_Analysis.py        → "SHAP Analysis"
├── ...
├── 21_Academic_Report.py      → "Academic Report"
└── 22_Dashboard_Presentation.py → This page
```

The leading number sets the order.  Underscores become spaces.  **No router
configuration needed** — this is equivalent to Flask Blueprints but fully
automatic.

Each page file is a standalone Python script.  It calls `st.set_page_config()`
at the top, imports its data sources and components, and renders its content
using `st.xxx()` calls.
""")

    st.divider()

    # ── Session state ──
    st.subheader("Session State: Persisting Data Across Reruns")
    st.markdown("""
Because Streamlit re-executes the entire script on every interaction, local
variables are lost between reruns.  `st.session_state` solves this:
""")
    st.code("""
# Problem: without session state, clicking "Deploy" loses the result
if st.button("Deploy"):
    result = deploy_file_to_vps(...)  # This runs...
    st.write(result)                  # ...but on next rerun, it's gone

# Solution: store in session_state
if st.button("Deploy"):
    result = deploy_file_to_vps(...)
    st.session_state["deploy_result"] = result  # Persists!

# On every rerun, check for stored result
if "deploy_result" in st.session_state:
    st.write(st.session_state["deploy_result"])  # Still there!
""", language="python")

    st.markdown("""
**Where we use session state:**
- `app.py` — stores the last VPS sync time
- Deploy Bots page — stores deployment results per bot
- Trade Monitor page — stores refresh counters
- Bayesian Tuning page — stores optimization results
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CONFIGURATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def tab_03_config():
    st.header("3 — Configuration System: config.py")

    st.markdown("""
The file `dashboard/config.py` (252 lines) is the **single source of truth**
for every path, mapping, and constant in the dashboard.  No other file
hardcodes paths or strategy names — everything references `config.py`.

This is a critical architectural decision: when a new bot is added, you
update **one file** and the entire dashboard recognises it.
""")

    st.subheader("File Structure")
    st.code("""
# dashboard/config.py — 252 lines, 12 sections

from pathlib import Path
from dotenv import load_dotenv
import os

# Section 1: Base paths (derived from __file__)
BASE_DIR = Path(__file__).resolve().parent.parent    # → /Backtesting
DASHBOARD_DIR = BASE_DIR / "dashboard"
VPS_CACHE_DIR = DASHBOARD_DIR / "databases"

# Section 2: VPS connection (from .env)
VPS_HOST = os.getenv("VPS_HOST", "")
VPS_PORT = int(os.getenv("VPS_PORT", "22"))
VPS_USER = os.getenv("VPS_USER", "trader")

# Section 3: Remote DB mapping (local filename → VPS path)
VPS_DB_FILES = {
    "fvg_btc.db": f"{VPS_REMOTE_BASE}/FVG_Strategy/BTC/btc-usd_enhanced_v3_trades.db",
    "fvg_eth.db": f"{VPS_REMOTE_BASE}/FVG_Strategy/ETH/eth-usd_enhanced_v3_trades.db",
    "lr_btc.db":  f"{VPS_REMOTE_BASE}/Liquidity_Raid/BTC_V2/btc_liquidity_raid_v2.db",
    # ... 8 total
}

# Section 4: Strategy registry
STRATEGIES = {
    "FVG":              {"color": "#2196F3", "symbols": ["BTC", "ETH", "NQ"]},
    "Liquidity Raid":   {"color": "#FF9800", "symbols": ["BTC", "ETH"]},
    "Momentum Mastery": {"color": "#4CAF50", "symbols": ["BTC", "ETH"]},
    "SBS":              {"color": "#9C27B0", "symbols": ["BTC", "ETH", "NQ"]},
}

# Section 5: Database → strategy+symbol mapping
DB_STRATEGY_MAP = {
    "fvg_btc.db": ("FVG", "BTC"),
    "lr_btc.db":  ("Liquidity Raid", "BTC"),
    # ... used by schema_normalizer to tag each trade
}

# Section 6: Systemd service names
BOT_SERVICES = {
    "fvg-btc": {"strategy": "FVG", "symbol": "BTC"},
    "lr-btc":  {"strategy": "Liquidity Raid", "symbol": "BTC"},
    # ... used by Deploy Bots page
}

# Section 7: Trading sessions (Eastern Time)
SESSIONS = {
    "Asian":    {"start": 19, "end": 3},    # 7PM-3AM ET
    "London":   {"start": 3,  "end": 8},    # 3AM-8AM ET
    "New York": {"start": 8,  "end": 16},   # 8AM-4PM ET
}

# Section 8: Unified trade schema (22 columns)
TRADE_SCHEMA_COLS = [
    "trade_id", "strategy", "symbol", "timeframe", "source",
    "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "stop_loss", "take_profit",
    "pnl_usd", "pnl_pct", "r_multiple",
    "session", "exit_reason", "duration_minutes",
    "running_balance", "mfe", "mae", "is_open",
]
""", language="python")

    st.subheader("Key Python Patterns")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**`pathlib.Path`** — cross-platform path handling")
        st.code("""
# Instead of string concatenation:
#   path = "/Users/me/Desktop/" + "trading/" + "data.db"
# Use Path objects:
BASE_DIR = Path(__file__).resolve().parent.parent
VPS_CACHE_DIR = BASE_DIR / "dashboard" / "databases"
# The / operator joins path segments.
# .resolve() converts to absolute path.
# .parent goes up one directory.
""", language="python")

    with c2:
        st.markdown("**`python-dotenv`** — secret management")
        st.code("""
# .env file (NOT in Git):
#   VPS_HOST=YOUR_VPS
#   BINANCE_API_KEY=abc123...

from dotenv import load_dotenv
load_dotenv()  # Reads .env into os.environ

VPS_HOST = os.getenv("VPS_HOST", "")
# Default "" means: if no .env, don't crash.
# This is the Twelve-Factor App pattern:
# configuration via environment variables.
""", language="python")

    st.markdown("""
**Why a centralised config matters:**
- Adding a new bot requires editing **one dict** (`VPS_DB_FILES`), not
  searching through 24 page files.
- Every page imports from `config` — no hardcoded paths anywhere.
- The `TRADE_SCHEMA_COLS` list defines the "contract" between data producers
  (normalizers) and data consumers (pages).  If a column is missing, the
  normalizer adds it as `None`.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def tab_04_data():
    st.header("4 — Data Pipeline: Loading, Normalising & Caching")

    st.markdown("""
The data pipeline is the most complex part of the dashboard.  It solves a
fundamental problem: **four different bot families store trades in four
different SQLite schemas**, and the dashboard needs to display them all in a
single unified table.

The pipeline has three stages:
""")

    st.code("""
Stage 1: NORMALISE (schema_normalizer.py)
  FVG SQLite  ──→  normalize_fvg()   ──→  22-column DataFrame
  LR  SQLite  ──→  normalize_lr_mm() ──→  22-column DataFrame
  MM  SQLite  ──→  normalize_lr_mm() ──→  22-column DataFrame
  SBS SQLite  ──→  normalize_sbs_live() → 22-column DataFrame
  SBS CSV     ──→  normalize_sbs_csv()  → 22-column DataFrame

Stage 2: CACHE (data_loader.py)
  get_live_trades()     ──→  @st.cache_data(ttl=60)   # 1-min cache
  get_backtest_trades() ──→  @st.cache_data(ttl=300)  # 5-min cache

Stage 3: MERGE (data_loader.py)
  get_all_trades()      ──→  pd.concat + dedup + sort
""", language="text")

    # ── Schema Normaliser ──
    st.subheader("Stage 1: Schema Normalisation")
    st.markdown("""
The file `dashboard/data/schema_normalizer.py` (453 lines) contains one
normalizer function per database format.  Each function reads a source-specific
schema and maps it to the 22 unified columns.

**The problem it solves:**
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**FVG SQLite schema** (source)")
        st.code("""
# enhanced_trades table
trade_id, direction, entry_timestamp,
exit_timestamp, entry_price, exit_price,
net_pnl, risk_amount,
max_favorable_excursion,
max_adverse_excursion,
market_session, exit_reason, ...
""", language="sql")

    with c2:
        st.markdown("**LR/MM SQLite schema** (source)")
        st.code("""
# trades table
id, timestamp, signal_type,
exit_timestamp, entry_price, exit_price,
realized_pnl, realized_pnl_pct,
position_size, status,
killzone, sweep_type, exit_reason, ...
""", language="sql")

    st.markdown("**Unified output** (22 columns, same for every source):")
    st.code("""
trade_id, strategy, symbol, timeframe, source,
direction, entry_time, exit_time,
entry_price, exit_price, stop_loss, take_profit,
pnl_usd, pnl_pct, r_multiple,
session, exit_reason, duration_minutes,
running_balance, mfe, mae, is_open
""", language="text")

    st.markdown("**Real code from `normalize_fvg()`** — key transformations:")
    st.code("""
# dashboard/data/schema_normalizer.py — normalize_fvg()

def normalize_fvg(db_path: Path, strategy: str, symbol: str) -> pd.DataFrame:
    # 1. Read from SQLite
    with sqlite3.connect(str(db_path)) as conn:
        raw = pd.read_sql_query("SELECT * FROM enhanced_trades", conn)

    # 2. Data quality filter: drop bogus trades
    raw = raw[~raw["trade_id"].str.startswith("FIXED", na=False)]

    # 3. Direction mapping (source uses lowercase, we want titlecase)
    dir_map = {"bullish": "Long", "bearish": "Short", "long": "Long", "short": "Short"}
    df["direction"] = raw["direction"].str.strip().str.lower().map(dir_map)
    #   .str.strip()  — remove whitespace
    #   .str.lower()  — normalise case
    #   .map(dict)    — vectorised lookup (much faster than apply)

    # 4. Timestamp parsing
    df["entry_time"] = pd.to_datetime(raw["entry_timestamp"], errors="coerce")
    #   errors="coerce" turns unparseable strings into NaT (not-a-time)
    #   instead of raising an exception

    # 5. R-multiple calculation
    risk = raw["risk_amount"].astype(float)
    df["r_multiple"] = df["pnl_usd"] / risk.replace(0, float("nan"))
    #   .replace(0, nan) prevents division by zero

    # 6. Session classification from market_session column
    session_map = {
        "asian": "Asian", "london": "London",
        "new_york": "New York", "ny": "New York",
    }
    df["session"] = raw["market_session"].str.strip().str.lower().map(session_map)

    # 7. Detect open trades (no exit time = still open)
    df["is_open"] = df["exit_time"].isna()
    df.loc[df["is_open"], ["pnl_usd", "r_multiple"]] = None  # Can't compute PnL yet

    return _ensure_schema(df)  # Guarantee all 22 columns exist
""", language="python")

    st.success("""
**Why this matters:** Without normalisation, every dashboard page would need
`if strategy == "FVG": ... elif strategy == "LR": ...` conditionals.  With
normalisation, pages just call `get_all_trades()` and work with a uniform
DataFrame — they don't know or care about the source format.
""")

    st.divider()

    # ── Caching & Merging ──
    st.subheader("Stages 2 & 3: Caching, Merging, Deduplication")
    st.code("""
# dashboard/data/data_loader.py — 121 lines

@st.cache_data(ttl=60)          # Cache result for 60 seconds
def get_live_trades() -> pd.DataFrame:
    return load_all_live_trades()  # Reads all 8 VPS DBs, normalises

@st.cache_data(ttl=300)         # Cache result for 5 minutes
def get_backtest_trades() -> pd.DataFrame:
    return load_all_backtest_trades()  # Reads SBS CSV

def get_all_trades(source_filter: str = "All") -> pd.DataFrame:
    frames = []
    if source_filter in ("All", "Live"):
        live = get_live_trades()            # Hits cache if fresh
        if not live.empty:
            frames.append(live)
    if source_filter in ("All", "Backtest"):
        bt = get_backtest_trades()          # Hits cache if fresh
        if not bt.empty:
            frames.append(bt)

    if not frames:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    df = pd.concat(frames, ignore_index=True)

    # Dedup: same trade appears in both Live and Backtest → keep Live
    source_priority = {"Live": 0, "Backtest": 1}
    df["_priority"] = df["source"].map(source_priority)
    df = df.sort_values("_priority").drop_duplicates(
        subset=["strategy", "symbol", "entry_time", "entry_price"],
        keep="first",              # Keeps Live (priority 0) over Backtest (1)
    )
    df = df.drop(columns=["_priority"])

    # Ensure datetime types
    for col in ("entry_time", "exit_time"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Exclude open trades (no exit = no PnL)
    if "is_open" in df.columns:
        df = df[~df["is_open"].fillna(False)]

    return df.sort_values("entry_time", ascending=False)
""", language="python")

    st.markdown("""
**Key Pandas operations demonstrated:**
| Operation | Code | Purpose |
|-----------|------|---------|
| `pd.concat()` | `pd.concat(frames, ignore_index=True)` | Vertically stack DataFrames from different sources |
| `.map()` | `df["source"].map({"Live": 0, "Backtest": 1})` | Vectorised dictionary lookup for priority sorting |
| `.drop_duplicates()` | `df.drop_duplicates(subset=[...], keep="first")` | Remove duplicate trades, keeping highest priority |
| `pd.to_datetime()` | `pd.to_datetime(df[col], errors="coerce")` | Parse strings to datetime; invalid → NaT |
| Boolean indexing | `df[~df["is_open"].fillna(False)]` | Filter out rows where `is_open` is True |
""")

    st.info("""
**`@st.cache_data` explained:** This decorator caches the function's return
value.  On the next rerun (triggered by any user interaction), Streamlit checks
if the function's arguments have changed and if the `ttl` (time-to-live) has
expired.  If both are unchanged, the cached result is returned instantly —
the SQLite files are not re-read.  This makes page loads fast even with
large databases.
""")

    # ── DST-Aware Session Classification ──
    st.subheader("Bonus: DST-Aware Session Classification")
    st.code("""
# dashboard/data/schema_normalizer.py

from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")  # DST-aware (UTC-5 winter, UTC-4 summer)
_UTC = ZoneInfo("UTC")

def _utc_to_et_hour(ts) -> Optional[int]:
    \"\"\"Convert UTC timestamp to Eastern Time hour, handling DST.\"\"\"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_UTC)  # Assume UTC if naive
    return ts.astimezone(_ET).hour    # Correct hour regardless of DST

def _classify_session(hour) -> str:
    for name, times in SESSIONS.items():
        start, end = times["start"], times["end"]
        if start > end:               # Asian wraps midnight (19→3)
            if hour >= start or hour < end:
                return name
        else:                         # London/NY don't wrap
            if start <= hour < end:
                return name
    return "Off-Hours"
""", language="python")
    st.markdown("""
**Why `zoneinfo` instead of hardcoded UTC-5:** Eastern Time is UTC-5 in
winter but UTC-4 in summer (daylight saving).  A hardcoded offset would
misclassify sessions for half the year.  `ZoneInfo("America/New_York")`
handles DST transitions automatically.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — VPS OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def tab_05_vps():
    st.header("5 — VPS Operations: Sync, Deploy & Manage")

    st.markdown("""
The file `dashboard/data/vps_sync.py` (228 lines) handles all communication
between the local dashboard and the remote VPS server where the bots run.
It provides three main capabilities: **sync** (pull data), **deploy** (push
code), and **manage** (start/stop services).
""")

    # ── Sync ──
    st.subheader("Database Sync: rsync with SCP Fallback")
    st.code("""
# dashboard/data/vps_sync.py

def sync_single_file(local_name: str, remote_path: str) -> Dict:
    \"\"\"Rsync a single DB file from VPS.\"\"\"
    VPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)  # Create dir if needed
    local_path = VPS_CACHE_DIR / local_name

    remote = f"{VPS_USER}@{VPS_HOST}:{remote_path}"
    cmd = [
        "rsync", "-az", "--checksum", "--timeout=15",
        "-e", f"ssh {' '.join(_ssh_args())}",
        remote, str(local_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            local_path.touch()            # Update mtime for freshness tracking
            size_kb = round(local_path.stat().st_size / 1024, 1)
            return {"file": local_name, "status": "ok", "size_kb": size_kb}
        return {"file": local_name, "status": "error", "error": result.stderr}
    except FileNotFoundError:
        return _scp_fallback(local_name, remote_path)  # rsync not installed
    except subprocess.TimeoutExpired:
        return {"file": local_name, "status": "timeout"}
""", language="python")

    st.markdown("""
**Key patterns:**
- **`subprocess.run()`** — runs an external command (rsync/scp) as a child
  process.  `capture_output=True` captures stdout/stderr.  `timeout=30`
  kills the process after 30 seconds.
- **rsync with `--checksum`** — compares file contents (not just size/time)
  to detect changes.  More reliable than scp for incremental syncs.
- **Fallback chain** — if rsync isn't installed (`FileNotFoundError`), falls
  back to scp.  The function always returns a status dict, never crashes.
""")

    st.divider()

    # ── Deploy ──
    st.subheader("Code Deployment with DB Protection")
    st.code("""
# dashboard/data/vps_sync.py

PROTECTED_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}

def deploy_file_to_vps(local_path: str, remote_path: str) -> Dict:
    local = Path(local_path)

    # Safety check: NEVER overwrite database files on VPS
    if local.suffix.lower() in PROTECTED_EXTENSIONS:
        return {
            "status": "error",
            "error": f"BLOCKED: refusing to deploy {local.name} — "
                     f"database files must not be overwritten on VPS",
        }

    # Back up remote files before deploying
    remote_dir = str(Path(remote_path).parent)
    run_ssh_command(f"bash {VPS_BACKUP_SCRIPT} {remote_dir}")

    # Upload via SCP
    remote = f"{VPS_USER}@{VPS_HOST}:{remote_path}"
    cmd = ["scp"] + _scp_args() + [str(local), remote]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        return {"status": "ok", "size_kb": round(local.stat().st_size / 1024, 1)}
    return {"status": "error", "error": result.stderr}
""", language="python")

    st.warning("""
**DB Protection is critical:** If you accidentally deploy a `.py` file that
happens to have a `.db` extension, or if a file selection bug picks the wrong
file, the `PROTECTED_EXTENSIONS` set prevents overwriting live trade databases
that contain irreplaceable data.  This is a **safety net**, not convenience.
""")

    st.divider()

    # ── Service Management ──
    st.subheader("Service Management via SSH")
    st.code("""
# dashboard/data/vps_sync.py

def manage_bot_service(service_name: str, action: str) -> Dict:
    \"\"\"Start/stop/restart a bot service on VPS.\"\"\"
    if action not in ("start", "stop", "restart"):
        return {"success": False, "error": f"Invalid action: {action}"}

    # Safety: back up databases before restart or stop
    if action in ("restart", "stop"):
        backup_bot_dbs(service_name)

    result = run_ssh_command(
        f"sudo systemctl {action} {service_name}.service"
    )
    return {"success": result["returncode"] == 0}

def get_bot_service_status(service_name: str) -> str:
    result = run_ssh_command(
        f"systemctl is-active {service_name}.service 2>/dev/null"
    )
    return result["stdout"].strip()  # "active", "inactive", or "failed"
""", language="python")

    st.markdown("""
The dashboard's Deploy Bots page (page 11) wraps these functions with
`st.button()` and `st.progress()` widgets, giving users a **one-click GUI**
for operations that would otherwise require SSH terminal access.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — API LAYER
# ══════════════════════════════════════════════════════════════════════════════

def tab_06_api():
    st.header("6 — API Layer: Market Data & Authenticated Trading")

    st.markdown("""
The dashboard interacts with external APIs through two modules:
- `binance_helpers.py` (344 lines) — **public** market data + indicator
  calculation + ICT pattern detection
- `binance_trading.py` (241 lines) — **authenticated** trading operations
  (account info, order placement)
""")

    # ── Public API ──
    st.subheader("Public API: binance_helpers.py")
    st.code("""
# dashboard/data/binance_helpers.py

def fetch_binance_candles(sym: str, tf: str, days: int) -> pd.DataFrame:
    \"\"\"Fetch historical OHLCV from Binance REST API.\"\"\"
    binance_sym = BINANCE_SYMBOL_MAP.get(sym, f"{sym}USDT")
    url = f"{BINANCE_REST_BASE}/klines"

    all_candles = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Pagination: Binance limits to 1000 candles per request
    while start_time < end_time:
        params = {
            "symbol": binance_sym, "interval": tf,
            "startTime": start_time, "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()

        if not data:
            break
        all_candles.extend(data)
        start_time = data[-1][6] + 1    # close_time + 1ms (advance cursor)
        if len(data) < 1000:
            break                        # Last page

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "Close_Time", "Quote_Volume", "Trades",
        "Taker_Buy_Base", "Taker_Buy_Quote", "Ignore",
    ])
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = df[col].astype(float)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    return df.set_index("Timestamp")
""", language="python")

    st.markdown("""
**Key techniques:**
- **Paginated fetching** — the `while` loop advances `start_time` by 1ms past
  the last candle's close time, requesting 1000 candles at a time until all
  data is retrieved.
- **`requests.get()` with `timeout`** — prevents the dashboard from hanging
  indefinitely if the API is slow.
- **Millisecond timestamps** — Binance uses Unix timestamps in milliseconds.
  `pd.to_datetime(df["Timestamp"], unit="ms")` converts them to Python datetimes.
""")

    st.markdown("**Indicator calculation** — adding EMA, ATR, and ADX:")
    st.code("""
# dashboard/data/binance_helpers.py

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()                               # Don't modify the original

    # EMA: Exponential Weighted Moving Average
    df["EMA50"]  = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    df["Bullish_Bias"] = df["EMA50"] > df["EMA200"]

    # ATR: Average True Range (14-period)
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    # ADX: Average Directional Index (trend strength)
    # Uses Wilder's smoothing (ewm with alpha=1/14)
    plus_dm = high.diff().where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = (-low.diff()).where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)                       # True Range = max of three

    atr14 = tr.ewm(alpha=1/14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    df["ADX"] = dx.ewm(alpha=1/14).mean()

    return df
""", language="python")

    st.divider()

    # ── ICT Patterns ──
    st.subheader("ICT Pattern Detection")
    st.markdown("""
The module also detects **ICT (Inner Circle Trader) patterns** — institutional
trading concepts used by the FVG and Liquidity Raid strategies:
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fair Value Gap (FVG)**")
        st.code("""
def detect_fvgs(df) -> list[dict]:
    \"\"\"3-candle imbalance pattern.\"\"\"
    for i in range(1, len(df) - 1):
        # Bullish FVG: candle i-1 high
        # < candle i+1 low (price gap)
        if highs[i-1] < lows[i+1]:
            top = lows[i+1]
            bottom = highs[i-1]

            # Check if gap has been filled
            future = lows[i+2:]
            if future.min() > bottom:
                fvgs.append({
                    "time": times[i],
                    "top": top,
                    "bottom": bottom,
                    "type": "bullish",
                })
""", language="python")

    with c2:
        st.markdown("**Order Block (OB)**")
        st.code("""
def detect_order_blocks(df, lookback=20):
    \"\"\"Last candle before a structure break.\"\"\"
    for i in range(lookback, len(df)-1):
        swing_high = highs[i-lookback:i].max()

        # Bullish OB: bearish candle before
        # a break of swing high
        if closes[i] < opens[i]:        # Bearish
            if highs[i+1] > swing_high: # Breaks high
                # Check if zone is
                # still unmitigated
                future = lows[i+1:]
                if future.min() > closes[i]:
                    obs.append({...})
""", language="python")

    st.divider()

    # ── Authenticated API ──
    st.subheader("Authenticated Trading: binance_trading.py")
    st.code("""
# dashboard/data/binance_trading.py — HMAC-SHA256 authentication

import hmac, hashlib

API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

def _sign(params: dict) -> dict:
    \"\"\"Add timestamp and HMAC-SHA256 signature for Binance auth.

    How HMAC-SHA256 works:
    1. Add current timestamp (milliseconds) to prevent replay attacks
    2. Build query string: "symbol=BTCUSDT&side=BUY&timestamp=1710000000"
    3. Hash the query string using the API secret as key
    4. Append the hash as 'signature' parameter
    5. Binance recomputes the hash server-side — if it matches, request is authentic
    \"\"\"
    params["timestamp"] = int(time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(
        API_SECRET.encode(),    # Key
        query.encode(),         # Message
        hashlib.sha256,         # Hash algorithm
    ).hexdigest()               # → 64-char hex string
    params["signature"] = sig
    return params

def _get(endpoint: str, params=None, signed=True) -> dict:
    params = dict(params or {})
    if signed:
        params = _sign(params)
    resp = requests.get(
        f"{BINANCE_REST_BASE}/{endpoint}",
        params=params,
        headers={"X-MBX-APIKEY": API_KEY},  # API key in header
        timeout=10,
    )
    resp.raise_for_status()   # Raises HTTPError if 4xx/5xx
    return resp.json()
""", language="python")

    st.markdown("""
**Why HMAC-SHA256?**
- The API key identifies *who* is making the request.
- The signature proves the request hasn't been tampered with in transit.
- The timestamp prevents replaying old requests.
- The secret key is never sent over the network — only the hash.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — COMPONENT LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def tab_07_components():
    st.header("7 — Component Library: Charts, KPIs & Filters")

    st.markdown("""
The `dashboard/components/` directory contains three modules of **reusable
building blocks** that are imported by the page files.  This is the
dashboard's equivalent of a UI component library.
""")

    # ── Charts ──
    st.subheader("charts.py — 17 Plotly Chart Builders (486 lines)")
    st.markdown("""
Every chart in the dashboard is built by a function in `charts.py`.  Each
function takes a DataFrame, builds a Plotly `Figure`, and returns it.  The
page file then passes it to `st.plotly_chart()`.

| Function | Chart Type | Used On |
|----------|-----------|---------|
| `strategy_comparison_bar()` | Grouped bar | Overview |
| `monthly_pnl_bar()` | Color-coded bar | Monthly Performance |
| `cumulative_pnl_line()` | Multi-line | Overview |
| `equity_curve()` | Line + regime bands | Equity Curves |
| `drawdown_chart()` | Filled area | Equity Curves |
| `rolling_win_rate()` | Smoothed line | Equity Curves |
| `r_multiple_histogram()` | Histogram | Trade Journal |
| `duration_histogram()` | Histogram | Trade Journal |
| `exit_reason_donut()` | Pie (hole=0.4) | Deep Dive |
| `session_strategy_heatmap()` | Heatmap | Session Analysis |
| `monthly_calendar_heatmap()` | Heatmap | Monthly Performance |
| `mfe_mae_scatter()` | Scatter | Deep Dive |
| `mc_distribution_chart()` | Histogram + vlines | Monte Carlo |
| `mc_equity_fan()` | Fan chart | Monte Carlo |
| `_add_regime_bands()` | Background rectangles | Equity Curves |
| `_add_change_markers()` | Vertical dotted lines | Equity Curves |
| `_get_color()` | Color lookup | All charts |
""")

    st.markdown("**Example: equity curve with regime overlay and benchmark**")
    st.code("""
# dashboard/components/charts.py — equity_curve()

def equity_curve(df, initial_balance=10000, benchmark=None,
                 regime_df=None, strategy_changes=None):
    fig = go.Figure()

    # 1. Add regime background bands (semi-transparent rectangles)
    _add_regime_bands(fig, regime_df)

    # 2. Plot each strategy as a line
    for strat in df["strategy"].unique():
        s = df[df["strategy"] == strat].sort_values("entry_time")
        equity = initial_balance + s["pnl_usd"].cumsum()
        fig.add_trace(go.Scatter(
            x=s["entry_time"], y=equity, name=strat, mode="lines",
            line=dict(color=STRATEGY_COLORS.get(strat, "#607D8B")),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Equity: $%{y:,.2f}<extra></extra>"
            ),
        ))

    # 3. Benchmark overlay (buy-and-hold comparison)
    if benchmark is not None and "Close" in benchmark.columns:
        close = benchmark["Close"]
        bm_equity = initial_balance * (close / close.iloc[0])
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=bm_equity, name="Buy & Hold",
            line=dict(color="#FFD700", dash="dash"),
        ))

    # 4. Strategy deployment markers
    _add_change_markers(fig, strategy_changes or [])

    fig.update_layout(template="plotly_dark", height=450)
    return fig
""", language="python")

    st.markdown("""
**Design pattern:** Every chart function follows the same structure:
1. Guard clause: `if df.empty: return go.Figure()` — never crash on empty data
2. Create `go.Figure()` instance
3. Add traces (lines, bars, markers)
4. Configure layout (template, height, axis labels)
5. Return the figure (caller decides where to render it)

All charts use `template="plotly_dark"` for visual consistency.
""")

    st.divider()

    # ── KPI Cards ──
    st.subheader("kpi_cards.py — Metric Card Renderers (83 lines)")
    st.code("""
# dashboard/components/kpi_cards.py

def kpi_row(df: pd.DataFrame, cols=None):
    \"\"\"Render a row of 4 top-level KPI metric cards.\"\"\"
    if df.empty:
        st.info("No trade data available.")
        return

    cols = cols or st.columns(4)
    total = len(df)
    wins = (df["pnl_usd"] > 0).sum()
    win_rate = wins / total if total > 0 else 0
    total_pnl = df["pnl_usd"].sum()
    avg_r = df["r_multiple"].mean()

    cols[0].metric("Total Trades", f"{total:,}",
        help="Total number of completed trades...")
    cols[1].metric("Win Rate", f"{win_rate:.1%}",
        help="Percentage of trades that closed in profit...")
    cols[2].metric("Total P&L", f"${total_pnl:,.2f}",
        help="Sum of all trade profits and losses in USD...")
    cols[3].metric("Avg R-Multiple", f"{avg_r:.2f}R",
        help="Average reward-to-risk ratio per trade...")
""", language="python")

    st.markdown("""
**`st.metric()` features used:**
- **Title + value** — the card displays a label and a formatted number
- **`help`** — adds a tooltip (?) icon with explanatory text
- **Format strings** — `f"{total:,}"` adds comma separators (10,000), `f"{win_rate:.1%}"` formats as percentage
""")

    st.divider()

    # ── Filters ──
    st.subheader("filters.py — Reusable Sidebar Filters (69 lines)")
    st.code("""
# dashboard/components/filters.py

def strategy_filter(df, key_prefix="") -> list:
    options = sorted(df["strategy"].dropna().unique())
    return st.sidebar.multiselect("Strategy", options, default=options,
                                   key=f"{key_prefix}_strat")

def source_filter(key_prefix="") -> str:
    return st.sidebar.radio("Source", ["All", "Live", "Backtest"],
                             key=f"{key_prefix}_src")

def date_range_filter(df, key_prefix=""):
    min_d = df["entry_time"].min().date()
    max_d = df["entry_time"].max().date()
    col1, col2 = st.sidebar.columns(2)
    start = col1.date_input("From", min_d, key=f"{key_prefix}_from")
    end   = col2.date_input("To",   max_d, key=f"{key_prefix}_to")
    return start, end

def apply_filters(df, strategies=None, symbols=None, directions=None,
                   date_start=None, date_end=None, sessions=None):
    \"\"\"Apply all sidebar selections to a trades DataFrame.\"\"\"
    mask = pd.Series(True, index=df.index)  # Start with all True
    if strategies:
        mask &= df["strategy"].isin(strategies)
    if symbols:
        mask &= df["symbol"].isin(symbols)
    if directions:
        mask &= df["direction"].isin(directions)
    if date_start:
        mask &= df["entry_time"].dt.date >= date_start
    if date_end:
        mask &= df["entry_time"].dt.date <= date_end
    return df[mask].reset_index(drop=True)
""", language="python")

    st.markdown("""
**Why `key_prefix`?**  Streamlit requires unique widget keys.  If two pages
both create a "Strategy" multiselect, they'd collide.  The `key_prefix`
parameter (e.g., `"tj"` for Trade Journal, `"eq"` for Equity Curves) ensures
uniqueness: `"tj_strat"` vs `"eq_strat"`.

**Why `apply_filters()` uses boolean masking?**  Instead of chaining
`.query()` calls or multiple `df[df["x"] == y]` filters, it builds a single
boolean Series (`mask`) using `&=` (bitwise AND assignment).  This is:
- Faster (one pass through the DataFrame)
- Cleaner (one function call instead of 6 filter lines)
- Composable (each filter is optional — `None` means "don't filter")
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 8 — DASHBOARD PAGES WALKTHROUGH
# ══════════════════════════════════════════════════════════════════════════════

def tab_08_pages():
    st.header("8 — Dashboard Pages: All 24 Pages Explained")

    st.markdown("""
Each page follows a consistent structure:

```python
# 1. Imports
import streamlit as st
from data.data_loader import get_all_trades
from components.charts import equity_curve
from components.filters import strategy_filter, apply_filters

# 2. Page config
st.set_page_config(page_title="...", page_icon="...", layout="wide")

# 3. Sidebar filters
src = source_filter(key_prefix="xx")
strategies = strategy_filter(df, key_prefix="xx")
df = apply_filters(df, strategies=strategies)

# 4. KPIs (summary metrics at the top)
kpi_row(df)

# 5. Charts (Plotly figures)
st.plotly_chart(equity_curve(df))

# 6. Data-driven commentary (algorithmic insights)
if win_rate < 0.35:
    st.markdown("Win rate is concerning...")
```
""")

    st.divider()

    # ── Core Analytics ──
    st.subheader("Core Analytics (Pages 1-7, 3b)")

    pages_core = [
        ("1 — Overview", "212 lines", "All-bot summary dashboard",
         "Aggregates data from all 9 bots into a single view. Shows VPS sync "
         "status dots (green/red per bot), portfolio KPIs, strategy comparison "
         "bar chart, cumulative P&L line, recent trades feed, and algorithmic "
         "commentary that analyses win rate trends and strategy performance.",
         """# Page 1 — key patterns:

# VPS status dots using st.columns + conditional styling
cols = st.columns(len(DB_STRATEGY_MAP))
for i, (db_file, (strategy, symbol)) in enumerate(DB_STRATEGY_MAP.items()):
    info = db_status.get(db_file, {})
    if info.get("exists"):
        cols[i].success(f"{strategy} {symbol}", icon="🟢")
    else:
        cols[i].error(f"{strategy} {symbol} — not synced", icon="🔴")

# Data-driven commentary (algorithmic analysis)
strat_pnl = df.groupby("strategy")["pnl_usd"].sum().sort_values()
worst = strat_pnl.index[0]
best = strat_pnl.index[-1]
st.markdown(f"**{worst}** is the weakest strategy (${strat_pnl.iloc[0]:,.2f})")
"""),
        ("2 — Strategy Explainer", "1,127 lines", "Visual strategy documentation",
         "Uses `st.tabs()` to create one tab per strategy (FVG, Liquidity Raid, "
         "Momentum Mastery, SBS). Each tab explains the strategy in plain English "
         "with ASCII flow diagrams, parameter tables, and `st.expander()` sections "
         "for advanced concepts.",
         """# Page 2 — uses st.tabs for strategy navigation
tab_fvg, tab_lr, tab_mm, tab_sbs = st.tabs(["FVG", "Liquidity Raid", ...])

with tab_fvg:
    st.markdown("**How FVG detection works:**")
    st.code(\"\"\"
    Candle i-1:  ┌──┐
    Candle i:    ┌────────┐  ← Strong momentum candle
    Candle i+1:       ┌──┐
                 GAP ↑    ← Fair Value Gap (price imbalance)
    \"\"\", language="text")
    with st.expander("Advanced: WFO Signal Scoring"):
        st.markdown("The bot scores each setup using 9 weighted components...")
"""),
        ("3 — Strategy Deep Dive", "892 lines", "Per-strategy performance analysis",
         "Lets users select a single strategy and see its detailed performance: "
         "strategy-specific KPIs (6 metrics), R-multiple histogram, MFE vs MAE "
         "scatter plot, exit reason donut chart, and duration histogram.",
         """# Page 3 — per-strategy deep dive
strat = st.selectbox("Select Strategy", df["strategy"].unique())
filtered = df[df["strategy"] == strat]

# 6 KPIs using strategy_kpis() helper
strategy_kpis(filtered)

# Charts in two columns
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(r_multiple_histogram(filtered))
with c2:
    st.plotly_chart(exit_reason_donut(filtered))
"""),
        ("3b — Live Logs", "186 lines", "Real-time VPS bot log viewer",
         "Streams systemd journal logs from the VPS for each of the 9 bot "
         "services. Sidebar controls: bot selector (`st.selectbox` over "
         "`BOT_SERVICES`), line count slider (25-500), grep filter, log level "
         "multiselect, and auto-refresh interval. Shows service status with "
         "green/red/yellow indicators. VPS reachability guard with `st.stop()`.",
         """# Page 3b — live logs via SSH
cmd = f"journalctl -u {selected_service}.service --no-pager -n {num_lines}"

# Server-side grep for performance
if grep_pattern.strip():
    safe_pattern = re.sub(r"[;&|`$(){}\\\\\\\"']\", \"\", grep_pattern)
    cmd += f" | grep -i -- {safe_pattern!r}"

result = run_ssh_command(cmd)

# Client-side log level filtering
level_re = re.compile(rf"(?:{level_pattern})(?:\\s*-|\\s*:)")
filtered = [ln for ln in lines if level_re.search(ln)]
st.code(display_logs, language="log", line_numbers=True)
"""),
        ("4 — Trade Journal", "229 lines", "Filterable trade log with styling",
         "Full trade table with sidebar filters (strategy, symbol, direction, "
         "date range, session, P&L slider). Rows with > 6% drawdown are "
         "highlighted red. Includes CSV export and algorithmic commentary "
         "(largest winner/loser, win/loss ratio, streak analysis).",
         """# Page 4 — interactive P&L slider + styled DataFrame

# P&L range filter using st.sidebar.slider
pnl_range = st.sidebar.slider("P&L ($)", pnl_min, pnl_max,
                                (pnl_min, pnl_max), key="tj_pnl")
df = df[(df["pnl_usd"] >= pnl_range[0]) & (df["pnl_usd"] <= pnl_range[1])]

# Styled DataFrame with conditional row highlighting
def _style_row(row):
    styles = [""] * len(row)
    if row.get("Max DD %", 0) >= 6.0:
        styles = ["background-color: rgba(244,67,54,0.15)"] * len(row)
    return styles

styled = display.style.apply(_style_row, axis=1)
st.dataframe(styled, height=600)

# CSV export button
csv = display.to_csv(index=False)
st.download_button("Download CSV", csv, "trades.csv", "text/csv")
"""),
        ("5 — Equity Curves", "282 lines", "Equity, drawdown, and risk metrics",
         "Plots equity curves with optional overlays: buy-and-hold benchmark "
         "(checkbox), market regime bands (checkbox), strategy deployment markers. "
         "Calculates Sharpe, Calmar, Recovery Factor, Equity R-squared, and "
         "win-rate trend using NumPy `polyfit`.",
         """# Page 5 — checkbox-controlled overlays + NumPy calculations

# Sidebar checkboxes control what gets overlaid
show_benchmark = st.sidebar.checkbox("Buy & Hold Benchmark", value=False)
show_regimes = st.sidebar.checkbox("Market Regimes", value=False)

# Fetch overlay data only if needed (lazy loading)
if show_benchmark or show_regimes:
    candles = fetch_binance_candles("BTC", "1h", overlay_days)
    if show_benchmark:
        benchmark_df = candles[["Close"]]
    if show_regimes:
        candles = calculate_indicators(candles)
        candles["regime"] = candles.apply(classify_regime, axis=1)

# Equity R² using NumPy polyfit
x = np.arange(len(cum_equity), dtype=float)
y = cum_equity.values.astype(float)
coeffs = np.polyfit(x, y, 1)          # Linear regression
y_pred = np.polyval(coeffs, x)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
equity_r2 = 1 - (ss_res / ss_tot)     # R² (0 to 1)
"""),
        ("6 — Session Analysis", "162 lines", "Performance by trading session",
         "Heatmap showing win rate for each strategy x session combination "
         "(Asian, London, New York). Per-session KPI cards. Identifies which "
         "time windows produce the best results.",
         """# Page 6 — heatmap from components/charts.py
st.plotly_chart(session_strategy_heatmap(df))

# Per-session KPIs
for session in ["Asian", "London", "New York"]:
    session_kpis(df, session)
"""),
        ("7 — Monthly Performance", "183 lines", "Calendar heatmap and monthly bars",
         "Year x Month heatmap coloured by total R-multiple. Monthly P&L bar "
         "chart with green/red colouring. Helps identify seasonal patterns.",
         """# Page 7 — calendar heatmap
st.plotly_chart(monthly_calendar_heatmap(df))
st.plotly_chart(monthly_pnl_bar(df))
"""),
    ]

    for name, lines, subtitle, description, code in pages_core:
        with st.expander(f"**{name}** ({lines}) — {subtitle}"):
            st.markdown(description)
            st.code(code, language="python")

    st.divider()

    # ── Optimisation & ML ──
    st.subheader("Optimisation & ML (Pages 8-14, 13b)")

    pages_opt = [
        ("8 — ML Training", "976 lines", "Machine learning pipeline UI",
         "Provides a GUI for training XGBoost models on trade features. "
         "Users select strategy, adjust hyperparameters via `st.slider()`, "
         "and click Train to run the ML pipeline. Shows feature importance "
         "plots and model accuracy metrics."),
        ("9 — WFO Analysis", "1,166 lines", "Walk-Forward Optimisation results",
         "Visualises WFO results: in-sample vs out-of-sample performance, "
         "parameter stability across walk-forward windows, equity curves per "
         "fold. Uses `st.selectbox()` for strategy selection and `st.tabs()` "
         "for analysis sections."),
        ("10 — Monte Carlo", "1,029 lines", "Monte Carlo simulation",
         "Runs 10,000 equity path simulations by shuffling trade order. Shows "
         "fan chart (confidence bands), return distribution histogram with "
         "percentile markers, and risk metrics (P(ruin), expected drawdown)."),
        ("11 — Deploy Bots", "126 lines", "One-click VPS deployment",
         "GUI for deploying code to VPS: file selection via `st.selectbox()`, "
         "deploy button, progress bar for bulk operations, service restart "
         "buttons with status feedback via `st.session_state`."),
        ("12 — Portfolio", "328 lines", "Multi-strategy portfolio analysis",
         "Correlation matrix between strategies, portfolio allocation pie "
         "chart, combined equity curve. Uses `st.multiselect()` to choose "
         "which strategies to include."),
        ("13 — Meta Strategy", "437 lines", "Regime-based strategy selection",
         "Combines WFO results across strategies to suggest which strategy "
         "to trade in each market regime. Uses `st.multiselect()` and "
         "`st.expander()` for regime-strategy mapping."),
        ("13b — Trade Monitor", "666 lines", "Real-time trade and position monitor",
         "Auto-refreshing page that shows live open positions, recent trade "
         "activity, and bot heartbeat status. Uses `st_autorefresh` for "
         "periodic polling, `st.metric()` for position P&L, and "
         "`st.session_state` for refresh counters."),
        ("14 — SHAP Analysis", "421 lines", "ML feature importance",
         "Displays SHAP (SHapley Additive exPlanations) waterfall plots "
         "showing which features most influence trade outcomes. Uses "
         "`st.selectbox()` for model selection."),
    ]

    for name, lines, subtitle, description in pages_opt:
        with st.expander(f"**{name}** ({lines}) — {subtitle}"):
            st.markdown(description)

    st.divider()

    # ── Operations & Advanced ──
    st.subheader("Operations & Advanced (Pages 15-22)")

    pages_ops = [
        ("15 — Bayesian Tuning", "344 lines", "Optuna hyperparameter search",
         "Uses `st.form()` for parameter bounds input and `st.button()` to "
         "launch Optuna optimization. Results displayed as parameter importance "
         "and optimization history plots."),
        ("16 — Stress Testing", "408 lines", "Drawdown/volatility scenarios",
         "Simulates extreme market conditions (2x volatility, consecutive "
         "losses, gap events) and projects their impact on portfolio equity."),
        ("17 — Cross Asset", "522 lines", "Multi-asset correlation",
         "Fetches BTC and ETH price data, computes rolling correlation, and "
         "displays side-by-side equity curves for cross-asset comparison."),
        ("18 — ML Performance", "624 lines", "ML model scoring dashboard",
         "Tracks prediction accuracy over time. Uses `st.tabs()` for different "
         "metrics (accuracy, precision, recall, F1). Plots calibration curves."),
        ("19 — Shadow Backtest", "304 lines", "Paper-trading validator",
         "Runs strategies in shadow mode (no real money) and compares results "
         "against live performance to detect strategy drift."),
        ("20 — Quant Research Lab", "658 lines", "Exploratory analysis",
         "Interactive research tools: custom indicator testing, statistical "
         "tests, regime analysis. Uses `st.tabs()` and `st.code()` for "
         "code-alongside-results layout."),
        ("21 — Academic Report", "1,765 lines", "Course evaluation grid",
         "10-tab report aligned to the Python course evaluation criteria. "
         "Covers conception, structure, methodology, Python patterns, UI, "
         "external modules, data processing, server/DB, DevOps, and showcase."),
        ("22 — This Presentation", "1,870 lines", "Dashboard code walkthrough",
         "This page — comprehensive walkthrough of every dashboard module "
         "with real code snippets and explanations."),
    ]

    for name, lines, subtitle, description in pages_ops:
        with st.expander(f"**{name}** ({lines}) — {subtitle}"):
            st.markdown(description)

    st.divider()

    st.subheader("Page Statistics")
    st.markdown("""
| Metric | Value |
|--------|-------|
| Total pages | 24 |
| Total lines of page code | ~14,900 |
| Largest page | Dashboard Presentation (1,838 lines) |
| Smallest page | Deploy Bots (126 lines) |
| Average page size | ~620 lines |
| Unique Streamlit widgets used | 19 types |
| Unique Plotly chart types | 9 types |
| Reusable components imported | charts (17), filters (7), KPIs (3) |
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 9 — BOT ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def tab_09_bots():
    st.header("9 — Bot Architecture: The Backend That Feeds the Dashboard")

    st.markdown("""
The dashboard exists to visualise data produced by the trading bots.
Understanding the bot architecture explains **why** the data looks the way
it does and **why** the schema normaliser needs to handle different formats.
""")

    # ── Two-Chain Composition ──
    st.subheader("Two-Chain Composition Pattern")
    st.markdown("""
Each bot family (FVG, Liquidity Raid, Momentum Mastery, SBS) uses **two
parallel inheritance chains** joined by **composition**:
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
             BTCConfig IS-A FVGConfigBase (Chain 1 — inheritance)
             BTCFVGBot IS-A FVGBotBase   (Chain 2 — inheritance)
             BTCFVGBot HAS-A BTCConfig   (composition — not inheritance)
""", language="text")

    st.markdown("""
**Why two chains instead of one?**
- **Separation of concerns** — configuration (data) and behaviour (logic) are
  independent concepts.  Mixing them in one class would violate the Single
  Responsibility Principle.
- **Testability** — you can create a `TestConfig` with fake API keys and pass
  it to a real bot for testing.
- **Flexibility** — the same bot class could accept different configs (paper
  trading vs live trading) without any code changes.
""")

    st.divider()

    # ── ABC Pattern ──
    st.subheader("ABC + @abstractmethod (Template Method Pattern)")
    st.code("""
# FVG_Strategy/core/bot_base.py

from abc import ABC, abstractmethod

class FVGBotBase(ABC):
    \"\"\"2,000+ lines of shared trading logic.
    Subclasses MUST implement 3 abstract methods.\"\"\"

    @abstractmethod
    def _init_data_fetchers(self):
        \"\"\"Connect to the right data source (Binance for BTC/ETH,
        Yahoo for NQ).  Each asset needs a different API client.\"\"\"
        ...

    @abstractmethod
    def _get_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        \"\"\"Fetch OHLCV candles for all configured timeframes.\"\"\"
        ...

    @abstractmethod
    def send_startup_message(self):
        \"\"\"Send asset-branded 'I'm alive' message via Telegram.\"\"\"
        ...

    # Concrete methods — inherited by all asset wrappers:
    async def run(self): ...                  # Main trading loop
    async def _check_for_signals(self): ...   # Signal detection
    def _compute_indicators(self): ...        # EMA, ATR, RSI
    def _score_signal(self): ...              # WFO scoring
    async def _send_telegram_alert(self): ... # Alert delivery
    def _log_trade_to_db(self): ...           # SQLite persistence
""", language="python")

    st.markdown("""
**Why ABC?**  If someone creates `class SOLFVGBot(FVGBotBase)` but forgets
to implement `_init_data_fetchers()`, Python raises `TypeError` **at import
time** — not at runtime when the bot tries to fetch data.  This catches bugs
before deployment.
""")

    st.divider()

    # ── @dataclass ──
    st.subheader("@dataclass for Configuration")
    st.code("""
# the internal strategy core

from dataclasses import dataclass, field
import os

@dataclass
class LRConfigBase:
    \"\"\"80+ fields across 35 sections.\"\"\"

    # Simple defaults
    BINANCE_API: str = "https://api.binance.com/api/v3"
    TIMEFRAME: str = "15m"
    RISK_REWARD_RATIO: float = 0.5
    MAX_DAILY_TRADES: int = 5

    # Deferred secret evaluation via default_factory
    TELEGRAM_BOT_TOKEN: str = field(
        default_factory=lambda: os.getenv("LR_BTC_TELEGRAM_TOKEN", "")
    )
    # The lambda is called at instantiation time, NOT at class definition time.
    # This means:
    # - The .env file only needs to exist when the bot starts, not when
    #   the module is imported.
    # - Unit tests can set environment variables before creating a config.
    # - Multiple bot instances can use different tokens.
""", language="python")

    st.markdown("""
**What `@dataclass` generates for you:**
- `__init__()` — constructor accepting all 80+ fields as keyword arguments
- `__repr__()` — readable string representation for debugging
- `__eq__()` — compare two configs for equality
- No boilerplate — 3 lines replace what would be a 200-line `__init__`
""")

    st.divider()

    # ── Enum State Machine ──
    st.subheader("Enum State Machine")
    st.code("""
# the internal strategy core

from enum import Enum

class SweepState(Enum):
    \"\"\"Finite state machine for the sweep detection lifecycle.\"\"\"
    WAITING = "waiting"            # Monitoring for sweep
    SWEEP_DETECTED = "detected"    # Sweep found, waiting for confirmation
    TRADED = "traded"              # Entry taken, trade is active

# Usage in bot:
if self.state == SweepState.WAITING:
    if self._detect_sweep(candle):
        self.state = SweepState.SWEEP_DETECTED    # Transition

elif self.state == SweepState.SWEEP_DETECTED:
    if self._confirm_entry(candle):
        self.state = SweepState.TRADED            # Transition

# Why Enum instead of strings?
# SweepState.WAITNG → AttributeError (typo caught at write time!)
# "waitng" == "waiting" → False (typo silently passes at runtime)
""", language="python")

    st.divider()

    # ── asyncio ──
    st.subheader("asyncio: Non-Blocking Event Loop")
    st.code("""
# SBS/bots/core/bot_base.py

import asyncio

class SBSBotBase(ABC):
    async def run(self):
        \"\"\"Main event loop — runs indefinitely.\"\"\"
        self.running = True
        while self.running:
            await self.main_loop_iteration()   # Fetch data, analyse, alert
            await self.sleep_with_heartbeat()  # Non-blocking sleep

    async def sleep_with_heartbeat(self):
        \"\"\"Sleep in 5-minute chunks, sending heartbeats between.\"\"\"
        sleep_duration = self.config.check_interval_minutes * 60
        slept = 0
        while slept < sleep_duration and self.running:
            await asyncio.sleep(300)           # Yields control for 5 min
            slept += 300
            await self.telegram_bot.send_heartbeat()

# Why async instead of time.sleep()?
# time.sleep(300) BLOCKS the thread — nothing else can run.
# await asyncio.sleep(300) YIELDS control — the event loop can
# handle other tasks (like responding to Telegram commands).
""", language="python")

    st.divider()

    # ── Graceful Degradation ──
    st.subheader("Graceful Degradation Pattern")
    st.code("""
# FVG_Strategy/core/bot_base.py — 9 optional module imports

try:
    from wfo_signal_scorer import WFOSignalScorer
except ImportError:
    WFOSignalScorer = None        # Feature disabled

try:
    from gamma_overlay import GammaOverlay
except ImportError:
    GammaOverlay = None           # Feature disabled

try:
    from ml_trade_filter import MLTradeFilter
except ImportError:
    MLTradeFilter = None          # Feature disabled

# At runtime — check before using:
if WFOSignalScorer is not None:
    score = self.wfo_scorer.score(features)
    if score < self.config.WFO_THRESHOLD:
        return  # Skip low-quality setup

# If the module isn't installed, this block never runs.
# The bot still works — it just doesn't have WFO scoring.
# This means you can deploy a minimal bot first and add
# advanced features incrementally.
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 10 — SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════

def tab_10_synthesis():
    st.header("10 — Synthesis: Key Takeaways")

    st.subheader("Python Patterns Catalogue")
    st.markdown("""
Every pattern below is used **in production** in this project, not as a
textbook exercise:

| Pattern | Where Used | What It Does |
|---------|-----------|-------------|
| `pathlib.Path` | `config.py` | Cross-platform path handling with `/` operator |
| `os.getenv()` + `dotenv` | `config.py`, `binance_trading.py` | Twelve-Factor App secret management |
| `@st.cache_data(ttl=N)` | `data_loader.py` | Memoize expensive DB reads for N seconds |
| `pd.concat()` + dedup | `data_loader.py` | Merge multi-source data with priority resolution |
| `pd.to_datetime(errors="coerce")` | `schema_normalizer.py` | Parse dates safely (bad data → NaT, not crash) |
| `.str.strip().str.lower().map()` | `schema_normalizer.py` | Chain string cleaning + vectorised lookup |
| `subprocess.run(timeout=N)` | `vps_sync.py` | Run external commands with safety timeout |
| `try/except FileNotFoundError` | `vps_sync.py` | Fallback chain (rsync → scp) |
| `HMAC-SHA256` | `binance_trading.py` | Cryptographic request signing |
| `go.Figure()` + `go.Scatter/Bar/Heatmap` | `charts.py` | Plotly chart construction |
| `st.columns()` / `st.tabs()` | All pages | Multi-column and tabbed layouts |
| `st.session_state` | `app.py`, Deploy page | Persist state across Streamlit reruns |
| `st.sidebar.checkbox()` | Equity Curves page | Toggle optional overlays |
| `.style.apply()` | Trade Journal | Conditional row highlighting |
| `st.download_button()` | Trade Journal | CSV export |
| `ABC` + `@abstractmethod` | Bot base classes | Enforce method contracts |
| `@dataclass` + `field(default_factory)` | Config base classes | Declarative config with deferred secrets |
| `Enum` | `sweep_state.py` | Type-safe state machines |
| `async/await` | Bot main loops | Non-blocking I/O for concurrent tasks |
| `try/except ImportError` | Bot base classes | Graceful degradation for optional modules |
| `logging.getLogger(__name__)` | All bot modules | Per-module structured logging |
| `sqlite3` + `ALTER TABLE` | `database_manager.py` | Schema migrations (EAFP pattern) |
| `threading.Thread(daemon=True)` | Bot base classes | Background scheduler threads |
| List/dict comprehensions | `quant_metrics.py` | Declarative data transforms |
| Generator expressions | `mtf_analysis.py` | Lazy, memory-efficient evaluation |
| `@staticmethod` | `technical_analysis.py` | Pure functions grouped by namespace |
| `zoneinfo.ZoneInfo` | `schema_normalizer.py` | DST-aware timezone conversion |
""")

    st.divider()

    st.subheader("Architecture Decisions Summary")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Decisions Made**")
        st.markdown("""
- **Streamlit over Flask** — reactive execution model eliminates frontend
  code; every page is pure Python.
- **SQLite over PostgreSQL** — zero-config embedded database; one file per
  bot, trivially synced via SCP.
- **Composition over deep inheritance** — config and bot logic are separate
  concerns joined by constructor injection.
- **Centralised config** — `config.py` is the single source of truth; adding
  a bot means editing one dictionary.
- **Schema normalisation** — each data source has its own normalizer; pages
  never deal with source-specific formats.
- **Caching with TTL** — `@st.cache_data` prevents redundant DB reads while
  ensuring data freshness.
""")

    with c2:
        st.markdown("**Trade-Offs Accepted**")
        st.markdown("""
- **No Docker** — systemd is simpler for single-process bots; Docker would
  add complexity without proportional benefit.
- **No CI/CD** — deployment is manual SCP + SSH; GitHub Actions would automate
  this but isn't critical for a solo developer.
- **No WebSocket in dashboard** — the dashboard polls via cached DB reads
  instead of real-time streaming; sufficient for minute-level analytics.
- **Single branch** — `main` only; feature branches would add process
  overhead for a solo project.
- **rsync dependency** — falls back to scp if rsync isn't installed, but
  rsync is preferred for incremental syncs.
""")

    st.divider()

    st.subheader("Dashboard File Inventory")
    st.code("""
dashboard/                           Lines
├── app.py                              86    # Entry point, VPS sync sidebar
├── config.py                          252    # Paths, registries, constants
├── data/
│   ├── data_loader.py                 121    # Caching + merge + dedup
│   ├── schema_normalizer.py           453    # 4 normalizers → unified schema
│   ├── vps_sync.py                    228    # rsync/SCP, deploy, services
│   ├── binance_helpers.py             344    # Candles, indicators, ICT patterns
│   └── binance_trading.py             241    # HMAC-SHA256 auth, orders
├── components/
│   ├── charts.py                      486    # 17 Plotly chart builders
│   ├── kpi_cards.py                    83    # KPI metric card helpers
│   └── filters.py                      69    # 7 reusable sidebar filters
└── pages/                              24 pages
    ├── 1_Overview.py                  212
    ├── 2_Strategy_Explainer.py      1,127
    ├── 3_Strategy_Deep_Dive.py        892
    ├── 3b_Live_Logs.py                186    # VPS systemd log viewer
    ├── 4_Trade_Journal.py             229
    ├── 5_Equity_Curves.py             282
    ├── 6_Session_Analysis.py          162
    ├── 7_Monthly_Performance.py       183
    ├── 8_ML_Training.py               976
    ├── 9_WFO_Analysis.py            1,166
    ├── 10_Monte_Carlo_Backtest.py   1,029
    ├── 11_Deploy_Bots.py              126
    ├── 12_Portfolio.py                328
    ├── 13_Meta_Strategy.py            437
    ├── 13b_Trade_Monitor.py           666    # Real-time position monitor
    ├── 14_SHAP_Analysis.py            421
    ├── 15_Bayesian_Tuning.py          344
    ├── 16_Stress_Testing.py           408
    ├── 17_Cross_Asset.py              522
    ├── 18_ML_Performance.py           624
    ├── 19_Shadow_Backtest.py          304
    ├── 20_Quant_Research_Lab.py       658
    ├── 21_Academic_Report.py        1,765
    └── 22_Dashboard_Presentation.py 1,870
                                    ───────
    Total dashboard code:          ~18,800 lines
""", language="text")

    st.divider()

    st.subheader("Conclusion")
    st.markdown("""
This project demonstrates that **Python alone** — without JavaScript, without
Java, without C++ — is sufficient to build a complete, production-grade
trading infrastructure:

1. **Data acquisition** — REST APIs, WebSocket streams, multi-source fallback
2. **Data processing** — Pandas DataFrames, NumPy arrays, schema normalisation
3. **Persistence** — SQLite per-bot, DuckDB for analytics, CSV for backtests
4. **Visualisation** — 17 Plotly chart types, interactive dashboards
5. **Web interface** — 24 Streamlit pages, reactive widgets, session state
6. **Deployment** — systemd services, SCP/rsync sync, SSH service management
7. **Machine learning** — XGBoost, SHAP, Walk-Forward Optimisation
8. **Architecture** — ABC, dataclass, Enum, composition, graceful degradation

The Streamlit dashboard ties it all together as a **single pane of glass**
for monitoring, analysis, and operations — built entirely in Python.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — RENDER ALL TABS
# ══════════════════════════════════════════════════════════════════════════════

def main():
    tabs = st.tabs([
        "1. Overview",
        "2. Streamlit",
        "3. Config",
        "4. Data Pipeline",
        "5. VPS Ops",
        "6. API Layer",
        "7. Components",
        "8. Pages",
        "9. Bot Arch.",
        "10. Synthesis",
    ])

    with tabs[0]:
        tab_01_overview()
    with tabs[1]:
        tab_02_streamlit()
    with tabs[2]:
        tab_03_config()
    with tabs[3]:
        tab_04_data()
    with tabs[4]:
        tab_05_vps()
    with tabs[5]:
        tab_06_api()
    with tabs[6]:
        tab_07_components()
    with tabs[7]:
        tab_08_pages()
    with tabs[8]:
        tab_09_bots()
    with tabs[9]:
        tab_10_synthesis()


main()
