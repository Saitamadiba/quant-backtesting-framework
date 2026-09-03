"""Page 26: Broker — Alpaca (READ-ONLY paper, the analyst-drift seat's own account).

A read-only window into the Alpaca **paper** account the `analyst_drift_paper`
seat trades: connection health, equity / buying power, open positions, resting
orders. It deliberately does **not** place, modify, or cancel orders.

Where the numbers come from (2026-09-03): the VPS, using the SAME credential
file the bot itself loads (`Momentum_Mastery/core/.env`, same candidate order),
so this panel shows *that bot's* account and not whatever key happens to sit in
`dashboard/.env`. The old local-key path is kept as a fallback for a manual
paper login, but it is not the seat — the local key had in fact expired (401)
while the seat kept trading, which is exactly the confusion this removes.
Reading through the VPS also inherits the bot's own guard: a key without the
`PK` paper prefix is refused, never queried.
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, ALPACA_PAPER
from data.fleet_live import fetch_alpaca_only

st.set_page_config(page_title="Broker — Alpaca", page_icon="🏦", layout="wide")
st.title("🏦 Broker — Alpaca")

st.warning("**READ-ONLY · PAPER.** This panel never places, modifies, or cancels orders. "
           "It shows the analyst-drift seat's account as the seat itself sees it.")

src = st.radio(
    "Source", ["VPS — the seat's own key (recommended)", "Local dashboard/.env key"],
    horizontal=True,
    help="The seat authenticates on the VPS. A local key is a different login unless "
         "you copied the seat's key there — and a different login is a different book.",
)


@st.cache_data(ttl=45, show_spinner="Reading the Alpaca seat via the VPS…")
def _vps_snapshot() -> dict:
    return fetch_alpaca_only()


col_l, col_r = st.columns([3, 1])
with col_r:
    if st.button("↻ Refresh", use_container_width=True):
        _vps_snapshot.clear()

# ══════════════════════════════════════════════════════════════════════════════
#  Path A — the seat's account, read on the VPS
# ══════════════════════════════════════════════════════════════════════════════
if src.startswith("VPS"):
    with col_l:
        st.caption("VPS → paper-api.alpaca.markets with the bot's own `PK…` key · cached 45s.")
    raw = _vps_snapshot()
    if not raw.get("ok"):
        st.error(f"Could not reach the VPS collector: `{raw.get('error', 'unknown')}`")
        st.stop()
    accts = [a for a in (raw.get("balances") or {}).get("accounts", [])
             if a.get("venue") == "alpaca-paper"]
    if not accts:
        st.error("The collector returned no Alpaca account — check `data/fleet_live.ALPACA`.")
        st.stop()
    a = accts[0]
    if not a.get("ok"):
        st.error(f"Alpaca seat did not answer: `{a.get('error', 'unknown')}`\n\n"
                 "If this says the key is not a PAPER key, that is the guard doing its job.")
        st.stop()

    w = a.get("wallet", {})
    st.success(f"Connected — seat **{', '.join(a.get('seats', []))}** · account `{a.get('uid', '…')}` "
               f"· status **{w.get('status', '?')}**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Equity", f"${w.get('equity', 0):,.2f}")
    k2.metric("Cash", f"${w.get('wallet_balance', 0):,.2f}")
    k3.metric("Buying power", f"${w.get('available', 0):,.2f}")
    k4.metric("Day Δ (vs last close)", f"${w.get('day_change', 0):,.2f}",
              delta=f"{w.get('day_change', 0):,.2f}")
    if w.get("blocked"):
        st.error("⚠ Alpaca reports a trading block on this account.")

    st.markdown("---")
    st.subheader("Open positions")
    pos = a.get("positions", [])
    if not pos:
        st.info("Flat — no open equity position. (The seat is MOO→MOC: it is normally "
                "flat outside the US session.)")
    else:
        pdf = pd.DataFrame(pos)[["symbol", "side", "size", "avg_price", "mark_price",
                                 "value", "upnl"]]
        st.dataframe(pdf, use_container_width=True, hide_index=True, column_config={
            "size": st.column_config.NumberColumn("Qty", format="%.0f"),
            "avg_price": st.column_config.NumberColumn("Avg entry", format="$%.2f"),
            "mark_price": st.column_config.NumberColumn("Last", format="$%.2f"),
            "value": st.column_config.NumberColumn("Mkt value", format="$%.0f"),
            "upnl": st.column_config.NumberColumn("Unreal P&L", format="$%.2f"),
        })
        st.metric("Total unrealized P&L", f"${pdf['upnl'].sum():,.2f}")

    st.markdown("---")
    st.subheader("Resting orders (read-only)")
    ords = a.get("orders", [])
    if not ords:
        st.info("No open order.")
    else:
        st.dataframe(pd.DataFrame(ords)[["symbol", "side", "order_type", "qty", "price",
                                         "status", "created"]],
                     use_container_width=True, hide_index=True)
    st.caption("The seat's closed trades and its rolling PnL are on **🛰️ Live Fleet** "
               "(family *Analyst Drift*, Tier 2 — a $ book with no stop-defined R).")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  Path B — a local paper key (legacy panel)
# ══════════════════════════════════════════════════════════════════════════════
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
    st.info(
        "No local Alpaca key. Add **paper** keys to `dashboard/.env`:\n\n"
        "```\nALPACA_API_KEY=...\nALPACA_SECRET_KEY=...\nALPACA_PAPER=true\n```\n\n"
        "Don't paste them into chat — put them in the `.env` file. Or just use the VPS path above."
    )
    st.stop()

if not ALPACA_PAPER:
    st.error("ALPACA_PAPER is false — this panel refuses to read a LIVE account.")
    st.stop()

_HEADERS = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}


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


with col_l:
    st.caption(f"Endpoint: `{ALPACA_BASE_URL}` · local key · auto-cached 30s.")

status, acct = _get("account")
if status != 200 or not isinstance(acct, dict):
    st.error(f"Connection failed (HTTP {status}). The local key may be expired or revoked — "
             f"the seat's own account is on the VPS path above.\n\n`{acct}`")
    st.stop()

acct_no = str(acct.get("account_number", ""))
st.success(f"Connected — account `…{acct_no[-4:]}` · status **{acct.get('status', '?')}** "
           f"· currency {acct.get('currency', 'USD')}")


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
if acct.get("trading_blocked") or acct.get("account_blocked"):
    st.error("⚠ Account has a trading/account block flag set — check the Alpaca dashboard.")

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
                "Symbol": p.get("symbol"), "Side": p.get("side"),
                "Qty": float(p.get("qty", 0)),
                "Avg entry": float(p.get("avg_entry_price", 0)),
                "Current": float(p.get("current_price", 0) or 0),
                "Mkt value": float(p.get("market_value", 0) or 0),
                "Unreal P&L": float(p.get("unrealized_pl", 0) or 0),
            })
        except (TypeError, ValueError):
            continue
    pdf = pd.DataFrame(rows)
    st.dataframe(pdf.round(4), use_container_width=True, hide_index=True)
    st.metric("Total unrealized P&L", f"${pdf['Unreal P&L'].sum():,.2f}")

st.markdown("---")
st.subheader("Recent orders (read-only)")
ostatus, orders = _get("orders?status=all&limit=20&direction=desc")
if ostatus != 200 or not isinstance(orders, list):
    st.warning(f"Could not load orders (HTTP {ostatus}).")
elif not orders:
    st.info("No orders.")
else:
    st.dataframe(pd.DataFrame(orders)[["symbol", "side", "type", "qty", "filled_qty",
                                       "status", "submitted_at"]],
                 use_container_width=True, hide_index=True)
