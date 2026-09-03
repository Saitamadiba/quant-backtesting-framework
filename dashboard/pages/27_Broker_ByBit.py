"""Page 27: Broker — ByBit (READ-ONLY, via VPS) — any of the fleet's accounts.

A read-only window into ONE ByBit sub-account at a time: wallet equity by
coin, open positions, resting orders. Pick the account; the page reads it with
the very key the seat on that account uses.

Why it changed (2026-09-03): the original panel read a single account through
`HyroTrader/.env`, and that key has been dead since 2026-09 (retCode 10003) —
the page was a window onto a wall. The fleet now trades **18 distinct ByBit
sub-accounts**, so the panel reuses the Live Fleet collector: every seat's env
file is scanned on the VPS, keys are deduplicated by the account UID the
exchange reports (several seats share one sub), and the account you pick is
read in depth. ByBit geo-blocks non-tunnelled hosts, so nothing here talks to
ByBit directly — the VPS does, through the Zurich proxy, with GET calls only. No
key is held locally and no order is ever placed, modified, or cancelled.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data.fleet_live import account_detail, accounts_frame, fetch_accounts_only, seat_status_frame

st.set_page_config(page_title="Broker — ByBit", page_icon="🟡", layout="wide")
st.title("🟡 Broker — ByBit")


@st.cache_data(ttl=45, show_spinner="Reading the fleet's accounts via the VPS…")
def _accounts() -> dict:
    return fetch_accounts_only()


cols = st.columns([3, 1])
with cols[1]:
    if st.button("↻ Refresh", use_container_width=True):
        _accounts.clear()
with cols[0]:
    st.caption("VPS → Zurich proxy → ByBit, GET-only, one wallet read per distinct account · cached 45s.")

raw = _accounts()
if not raw.get("ok"):
    st.error(f"Could not reach the VPS collector.\n\n`{raw.get('error', 'unknown error')}`")
    st.stop()

acc = accounts_frame(raw)
acc = acc[acc["venue"] == "bybit"] if not acc.empty else acc
if acc.empty:
    st.error("No ByBit account answered — every key failed or the tunnel is down. "
             "The key ↔ account map below says which.")
    st.dataframe(seat_status_frame(raw), use_container_width=True, hide_index=True)
    st.stop()

ok = acc[acc["status"] == "ok"]
st.info(f"**READ-ONLY · DEMO.** {len(ok)}/{len(acc)} accounts answered · "
        f"total equity ${ok['equity'].sum():,.0f} · uPnL ${ok['upnl'].sum():,.2f}. "
        "Display only — no order routing.")

# ── pick an account ───────────────────────────────────────────────────────────
acc = acc.sort_values("equity", ascending=False)
labels = {r.uid: f"{r.seats}  ·  UID {r.uid}  ·  ${r.equity:,.0f}" for r in acc.itertuples()}
uid = st.selectbox("Account", list(labels.keys()), format_func=labels.get,
                   help="One row per distinct sub-account. Seats sharing an account share "
                        "a position and a stop — two nameplates on one mailbox.")
a = account_detail(raw, uid)
if not a or not a.get("ok"):
    st.error(f"Account {uid} did not answer: `{(a or {}).get('error', 'unknown')}`")
    st.stop()

# ── Wallet ────────────────────────────────────────────────────────────────────
w = a.get("wallet", {})
st.success(f"Connected — UID `{uid}` · {a.get('bybit_env', '?')} · UNIFIED · "
           f"seats: **{', '.join(a.get('seats', []))}**")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total equity", f"${w.get('equity', 0):,.2f}")
k2.metric("Wallet balance", f"${w.get('wallet_balance', 0):,.2f}")
k3.metric("Available", f"${w.get('available', 0):,.2f}")
k4.metric("Perp uPnL", f"${w.get('upnl', 0):,.2f}", delta=f"{w.get('upnl', 0):,.2f}")

coins = w.get("coins", [])
if coins:
    st.caption("Balances by coin — demo seats are seeded with several coins, which is why "
               "equity sits above the USDT wallet balance.")
    st.dataframe(pd.DataFrame(coins), use_container_width=True, hide_index=True, column_config={
        "equity": st.column_config.NumberColumn("Equity", format="%.4f"),
        "wallet_balance": st.column_config.NumberColumn("Wallet", format="%.4f"),
        "usd_value": st.column_config.NumberColumn("USD value", format="$%.2f"),
        "upnl": st.column_config.NumberColumn("uPnL", format="$%.4f"),
    })

# ── Positions ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Open positions")
positions = a.get("positions", [])
if not positions:
    st.info("No open positions on this account.")
else:
    pdf = pd.DataFrame(positions)
    st.dataframe(pdf, use_container_width=True, hide_index=True, column_config={
        "size": st.column_config.NumberColumn("Size", format="%.4f"),
        "avg_price": st.column_config.NumberColumn("Avg entry", format="%.5f"),
        "mark_price": st.column_config.NumberColumn("Mark", format="%.5f"),
        "upnl": st.column_config.NumberColumn("uPnL", format="$%.2f"),
        "value": st.column_config.NumberColumn("Notional", format="$%.0f"),
        "leverage": st.column_config.NumberColumn("Lev", format="%.0fx"),
        "sl": st.column_config.NumberColumn("SL", format="%.5f"),
        "tp": st.column_config.NumberColumn("TP", format="%.5f"),
    })
    st.metric("Total unrealized PnL", f"${pdf['upnl'].sum():,.2f}")

# ── Open orders ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Resting orders (read-only)")
orders = a.get("orders", [])
if not orders:
    st.info("No open orders on this account.")
else:
    st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)

with st.expander("Key ↔ account map — every key the fleet holds (hashed; no secret leaves the VPS)"):
    st.dataframe(seat_status_frame(raw), use_container_width=True, hide_index=True)

st.caption("Read-only via the Live Fleet collector (`data/fleet_collector_remote.py`, streamed "
           "to the VPS over stdin — never deployed). Order routing is intentionally absent: the "
           "seats place orders through the guard-gated `bybit_execution`, not this dashboard.")
