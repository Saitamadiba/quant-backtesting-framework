"""Shared WFO and shadow backtest result loading with Streamlit caching."""

import json
import os
import re
from pathlib import Path

import streamlit as st

from config import BASE_DIR

RESULTS_DIR = BASE_DIR / "backtrader_framework" / "optimization" / "results"

# Filename patterns — timestamp is optional
_WFO_RE = re.compile(
    r"^wfo_(?P<rest>.+)\.json$"
)
_SHADOW_RE = re.compile(
    r"^shadow_(?P<strategy>.+?)_(?P<symbol>[A-Z]+)_(?P<ts>\d{8}(?:_\d{6})?)\.json$"
)

# Extract timestamp from end of filename
_TS_RE = re.compile(r"_(\d{8}(?:_\d{6})?)$")


@st.cache_data(ttl=300)
def list_wfo_results() -> list[dict]:
    """Scan wfo_*.json, read JSON headers for metadata, sorted by timestamp desc."""
    results = []
    if not RESULTS_DIR.exists():
        return results
    for p in RESULTS_DIR.glob("wfo_*.json"):
        if not p.name.startswith("wfo_"):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            strategy = d.get("strategy_name", "")
            symbol = d.get("symbol", "")
            timeframe = d.get("timeframe", "")
            ts = d.get("run_timestamp", "")
            if not strategy or not symbol:
                continue
            # Use file mtime as fallback sort key
            mtime = os.path.getmtime(p)
            results.append({
                "path": str(p),
                "strategy": strategy,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": ts or str(int(mtime)),
                "mtime": mtime,
            })
        except (json.JSONDecodeError, KeyError):
            continue
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return results


@st.cache_data(ttl=300)
def load_wfo_result(path: str) -> dict:
    """Load a single WFO result JSON."""
    with open(path) as f:
        return json.load(f)


def get_latest_wfo(strategy: str, symbol: str, timeframe: str) -> dict | None:
    """Return the most recent WFO result for a strategy/symbol/timeframe combo."""
    for r in list_wfo_results():
        if (r["strategy"].lower() == strategy.lower()
                and r["symbol"] == symbol
                and r["timeframe"] == timeframe):
            return load_wfo_result(r["path"])
    return None


@st.cache_data(ttl=300)
def list_shadow_results() -> list[dict]:
    """Scan shadow_*.json files and return metadata dicts."""
    results = []
    if not RESULTS_DIR.exists():
        return results
    for p in RESULTS_DIR.glob("shadow_*.json"):
        m = _SHADOW_RE.match(p.name)
        if not m:
            continue
        results.append({
            "path": str(p),
            "strategy": m.group("strategy").replace("_", " "),
            "symbol": m.group("symbol"),
            "timestamp": m.group("ts"),
        })
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results


@st.cache_data(ttl=300)
def load_shadow_result(path: str) -> dict:
    """Load a single shadow backtest result JSON."""
    with open(path) as f:
        return json.load(f)


def get_latest_shadow(strategy: str, symbol: str) -> dict | None:
    """Return the most recent shadow result for a strategy/symbol combo."""
    for r in list_shadow_results():
        if (r["strategy"].lower() == strategy.lower()
                and r["symbol"] == symbol):
            return load_shadow_result(r["path"])
    return None
