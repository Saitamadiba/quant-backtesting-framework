"""Page 25: Macro Context — regime backdrop for the strategies.

Three source tiers:
  - **Market macro** (yfinance, keyless): DXY, VIX, US 10Y, Gold, S&P 500.
  - **FRED** (needs a free API key in dashboard/.env as FRED_API_KEY):
    CPI, Fed Funds, M2, unemployment, 10Y Treasury.
  - **World Bank** (keyless): US headline inflation + GDP growth.

This is *context*, not a trading signal — a place to glance at the macro
weather your bots are trading inside of (e.g. is IV elevated because VIX is
spiking? is the dollar trending?).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config import FRED_API_KEY

st.set_page_config(page_title="Macro Context", page_icon="🌍", layout="wide")
st.title("🌍 Macro Context")
st.caption("Macro weather behind the strategies — implied-vol drivers, rates, dollar, liquidity. "
           "Context only, not a signal.")


# ──────────────────────────────────────────────────────────────────────────────
# Market macro (yfinance — keyless)
# ──────────────────────────────────────────────────────────────────────────────
_MARKET = {
    "Dollar (DXY)": "DX-Y.NYB",
    "VIX": "^VIX",
    "US 10Y Yield": "^TNX",
    "Gold": "GC=F",
    "S&P 500": "^GSPC",
}


@st.cache_data(ttl=900, show_spinner=False)
def _yf_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    return df[["Close"]].dropna() if df is not None and not df.empty else pd.DataFrame()


st.subheader("Market macro")
st.caption("Daily, ~6-month window. Source: Yahoo Finance.")
mcols = st.columns(len(_MARKET))
market_series: dict[str, pd.Series] = {}
for i, (label, ticker) in enumerate(_MARKET.items()):
    df = _yf_history(ticker)
    if df.empty:
        mcols[i].metric(label, "—", help=f"No data for {ticker}")
        continue
    s = df["Close"]
    market_series[label] = s
    last = s.iloc[-1]
    prev = s.iloc[-2] if len(s) > 1 else last
    chg = (last - prev) / prev * 100 if prev else 0.0
    # ^TNX is yield ×10; show as a percent
    disp = f"{last/10:.2f}%" if ticker == "^TNX" else f"{last:,.2f}"
    mcols[i].metric(label, disp, f"{chg:+.2f}% d/d")

if market_series:
    pick = st.selectbox("Chart series", list(market_series.keys()), key="macro_mkt_pick")
    s = market_series[pick]
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", name=pick))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), title=f"{pick} — 6mo")
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# FRED (needs key)
# ──────────────────────────────────────────────────────────────────────────────
_FRED = {
    "CPI (index)": "CPIAUCSL",
    "Fed Funds Rate (%)": "FEDFUNDS",
    "M2 Money Supply ($B)": "M2SL",
    "Unemployment (%)": "UNRATE",
    "10Y Treasury (%)": "DGS10",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fred_series(series_id: str, key: str) -> pd.Series:
    start = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series_id, "api_key": key, "file_type": "json",
                "observation_start": start},
        timeout=15,
    )
    r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = [(o["date"], float(o["value"])) for o in obs if o.get("value") not in (".", "", None)]
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(vals, index=pd.to_datetime(idx)).sort_index()


st.markdown("---")
st.subheader("FRED — US macro fundamentals")
if not FRED_API_KEY:
    st.info("FRED is not configured. Add a free key to `dashboard/.env` as "
            "`FRED_API_KEY=...` (get one at fred.stlouisfed.org/docs/api/api_key.html), "
            "then reload. World Bank + market-macro above work without it.")
else:
    st.caption("Trailing 2 years. Source: Federal Reserve (FRED).")
    fcols = st.columns(len(_FRED))
    fred_series: dict[str, pd.Series] = {}
    for i, (label, sid) in enumerate(_FRED.items()):
        try:
            s = _fred_series(sid, FRED_API_KEY)
        except Exception as e:
            fcols[i].metric(label, "—", help=f"fetch failed: {e}")
            continue
        if s.empty:
            fcols[i].metric(label, "—")
            continue
        fred_series[label] = s
        last = s.iloc[-1]
        prev = s.iloc[-2] if len(s) > 1 else last
        fcols[i].metric(label, f"{last:,.2f}", f"{last - prev:+.2f} vs prior")
    if fred_series:
        fpick = st.selectbox("Chart series", list(fred_series.keys()), key="macro_fred_pick")
        s = fred_series[fpick]
        fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", name=fpick))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), title=f"{fpick} — 2y")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# World Bank (keyless)
# ──────────────────────────────────────────────────────────────────────────────
_WB = {
    "US Inflation (CPI, % annual)": "FP.CPI.TOTL.ZG",
    "US GDP Growth (% annual)": "NY.GDP.MKTP.KD.ZG",
}


@st.cache_data(ttl=86400, show_spinner=False)
def _worldbank(indicator: str, country: str = "US") -> pd.Series:
    r = requests.get(
        f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}",
        params={"format": "json", "per_page": "30"}, timeout=15,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.Series(dtype=float)
    rows = [(d["date"], d["value"]) for d in payload[1] if d.get("value") is not None]
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(vals, index=idx).sort_index()


st.markdown("---")
st.subheader("World Bank — long-run US fundamentals")
st.caption("Annual. Source: World Bank Open Data (keyless).")
wcols = st.columns(len(_WB))
for i, (label, ind) in enumerate(_WB.items()):
    try:
        s = _worldbank(ind)
    except Exception as e:
        wcols[i].metric(label, "—", help=f"fetch failed: {e}")
        continue
    if s.empty:
        wcols[i].metric(label, "—")
        continue
    last_year, last_val = s.index[-1], s.iloc[-1]
    wcols[i].metric(label, f"{last_val:.2f}%", help=f"latest: {last_year}")

st.caption("Tiers: market macro (Yahoo Finance) · FRED (Federal Reserve) · World Bank Open Data. "
           "Integration 2 of 3 (QuantLib → macro → broker).")
