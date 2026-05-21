"""Page 26: Broker — Alpaca (READ-ONLY paper).

A read-only window into an Alpaca **paper** account: connection health,
account equity / buying power, and open positions.  It deliberately does
**not** place, modify, or cancel orders — order routing is a separate,
explicit step for whenever you commit to live execution.

Credentials come from dashboard/.env (ALPACA_API_KEY / ALPACA_SECRET_KEY) —
never hardcoded.  Uses Alpaca's REST v2 endpoints via plain requests (no SDK
dependency); add `alpaca-py` later if/when you wire order placement.
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, ALPACA_PAPER

st.set_page_config(page_title="Broker — Alpaca", page_icon="🏦", layout="wide")
st.title("🏦 Broker — Alpaca")

# Unmissable safety banner.
st.warning("**READ-ONLY.** This panel never places, modifies, or cancels orders. "
           "It only displays account state from the "
           f"{'PAPER' if ALPACA_PAPER else 'LIVE'} endpoint.")

if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
    st.info(
        "Alpaca is not configured. Add **paper** keys to `dashboard/.env`:\n\n"
        "```\nALPACA_API_KEY=...\nALPACA_SECRET_KEY=...\nALPACA_PAPER=true\n```\n\n"
        "Generate paper keys at app.alpaca.markets (Paper Trading → API Keys). "
        "Don't paste them into chat — put them in the `.env` file."
    )
    st.stop()


_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


@st.cache_data(ttl=30, show_spinner=False)
def _get(path: str) -> tuple[int, object]:
    """GET an Alpaca v2 endpoint. Returns (status_code, json|error_text)."""
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/{path}", headers=_HEADERS, timeout=15)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body
    except Exception as e:
        return -1, str(e)


col_l, col_r = st.columns([3, 1])
with col_r:
    if st.button("↻ Refresh", use_container_width=True):
        _get.clear()
with col_l:
    st.caption(f"Endpoint: `{ALPACA_BASE_URL}`  ·  auto-cached 30s.")

# ── Connection + account ─────────────────────────────────────────────────────
status, acct = _get("account")
if status != 200 or not isinstance(acct, dict):
    st.error(f"Connection failed (HTTP {status}). "
             f"Check your keys / paper-vs-live setting.\n\n`{acct}`")
    st.stop()

st.success(f"Connected — account `{acct.get('account_number', '?')}` · "
           f"status **{acct.get('status', '?')}** · currency {acct.get('currency', 'USD')}")


def _f(key: str) -> float:
    try:
        return float(acct.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


k1, k2, k3, k4 = st.columns(4)
k1.metric("Equity", f"${_f('equity'):,.2f}")
k2.metric("Cash", f"${_f('cash'):,.2f}")
k3.metric("Buying power", f"${_f('buying_power'):,.2f}")
k4.metric("Portfolio value", f"${_f('portfolio_value'):,.2f}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Long mkt value", f"${_f('long_market_value'):,.2f}")
k6.metric("Short mkt value", f"${_f('short_market_value'):,.2f}")
k7.metric("Daytrade count", str(acct.get("daytrade_count", "—")))
k8.metric("PDT flag", "Yes" if acct.get("pattern_day_trader") else "No")

if acct.get("trading_blocked") or acct.get("account_blocked"):
    st.error("⚠ Account has a trading/account block flag set — check Alpaca dashboard.")


# ── Positions ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Open positions")
pstatus, positions = _get("positions")
if pstatus != 200 or not isinstance(positions, list):
    st.warning(f"Could not load positions (HTTP {pstatus}).")
elif not positions:
    st.info("No open positions in this account.")
else:
    rows = []
    for p in positions:
        try:
            rows.append({
                "Symbol": p.get("symbol"),
                "Side": p.get("side"),
                "Qty": float(p.get("qty", 0)),
                "Avg entry": float(p.get("avg_entry_price", 0)),
                "Current": float(p.get("current_price", 0) or 0),
                "Mkt value": float(p.get("market_value", 0) or 0),
                "Unreal P&L": float(p.get("unrealized_pl", 0) or 0),
                "Unreal %": float(p.get("unrealized_plpc", 0) or 0) * 100,
            })
        except (TypeError, ValueError):
            continue
    pdf = pd.DataFrame(rows)
    st.dataframe(
        pdf.round(4), use_container_width=True, hide_index=True,
        column_config={
            "Unreal P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Unreal %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    tot_pl = pdf["Unreal P&L"].sum()
    st.metric("Total unrealized P&L", f"${tot_pl:,.2f}")


# ── Recent orders (read-only) ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Recent orders (read-only)")
ostatus, orders = _get("orders?status=all&limit=20&direction=desc")
if ostatus != 200 or not isinstance(orders, list):
    st.caption("Orders unavailable.")
elif not orders:
    st.info("No recent orders.")
else:
    orows = [{
        "Submitted": o.get("submitted_at", "")[:19].replace("T", " "),
        "Symbol": o.get("symbol"),
        "Side": o.get("side"),
        "Type": o.get("type"),
        "Qty": o.get("qty"),
        "Status": o.get("status"),
        "Filled avg": o.get("filled_avg_price"),
    } for o in orders]
    st.dataframe(pd.DataFrame(orows), use_container_width=True, hide_index=True)

st.caption("Read-only Alpaca REST v2 (account / positions / orders). "
           "Integration 3 of 3 (QuantLib → macro → broker). "
           "Order routing intentionally omitted — add `alpaca-py` + an explicit "
           "confirm-gated ticket when you go live.")
