"""Page 27: Broker — ByBit (READ-ONLY, via VPS).

A read-only window into the ByBit account behind the HyroTrader funded
challenge: wallet equity / balances, open positions, open orders.

ByBit geo-blocks non-tunnelled hosts, so — unlike the Alpaca panel — this
page never talks to ByBit directly. It SSHes to the VPS and runs the
read-only `HyroTrader/dashboard_snapshot.py`, which uses the validated
`bybit_client` through the Zurich proxy. No keys are held locally and no
orders are ever placed/modified/cancelled.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from data.vps_sync import run_ssh_command

st.set_page_config(page_title="Broker — ByBit", page_icon="🟡", layout="wide")
st.title("🟡 Broker — ByBit")

_SNAPSHOT_CMD = (
    "cd /home/trader/trading_bots/HyroTrader && "
    "/home/trader/trading_bots/venv/bin/python3 dashboard_snapshot.py"
)


@st.cache_data(ttl=30, show_spinner="Fetching ByBit snapshot via VPS…")
def _fetch_snapshot() -> dict:
    res = run_ssh_command(_SNAPSHOT_CMD)
    if res.get("returncode") != 0:
        return {"ok": False, "error": f"SSH failed: {res.get('stderr') or res.get('returncode')}"}
    out = res.get("stdout", "").strip()
    # The script prints a single JSON line; tolerate any leading noise.
    line = out.splitlines()[-1] if out else ""
    try:
        return json.loads(line)
    except Exception as e:
        return {"ok": False, "error": f"bad snapshot output: {e} · raw={out[:200]!r}"}


cols = st.columns([3, 1])
with cols[1]:
    if st.button("↻ Refresh", use_container_width=True):
        _fetch_snapshot.clear()
with cols[0]:
    st.caption("Fetched via VPS → Zurich proxy → ByBit. Auto-cached 30s.")

snap = _fetch_snapshot()
env = snap.get("env", "?")

if env == "mainnet":
    st.warning("**READ-ONLY · LIVE (mainnet).** This panel only displays account state — "
               "it never places, modifies, or cancels orders.")
else:
    st.info(f"**READ-ONLY · {env.upper()}.** HyroTrader funded-challenge account "
            f"({'demo dry-run' if env == 'demo' else env}). Display only — no order routing.")

if not snap.get("ok"):
    st.error(f"Could not fetch ByBit snapshot.\n\n`{snap.get('error', 'unknown error')}`\n\n"
             "Checks: VPS reachable? `bybit-proxy` service up? `HyroTrader/.env` keys valid?")
    st.stop()


# ── Wallet ────────────────────────────────────────────────────────────────────
w = snap.get("wallet", {})
st.success(f"Connected — ByBit {env} · UNIFIED account")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total equity", f"${w.get('totalEquity', 0):,.2f}")
k2.metric("Wallet balance", f"${w.get('totalWalletBalance', 0):,.2f}")
k3.metric("Available", f"${w.get('totalAvailableBalance', 0):,.2f}")
k4.metric("Perp uPnL", f"${w.get('totalPerpUPL', 0):,.2f}")

coins = w.get("coins", [])
if coins:
    st.caption("Balances by coin")
    st.dataframe(
        pd.DataFrame(coins).round(4), use_container_width=True, hide_index=True,
        column_config={
            "usdValue": st.column_config.NumberColumn(format="$%.2f"),
            "unrealisedPnl": st.column_config.NumberColumn(format="$%.4f"),
        },
    )


# ── Positions ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Open positions")
positions = snap.get("positions", [])
if not positions:
    st.info("No open positions.")
else:
    pdf = pd.DataFrame(positions)
    st.dataframe(
        pdf.round(4), use_container_width=True, hide_index=True,
        column_config={
            "unrealisedPnl": st.column_config.NumberColumn("Unreal PnL", format="$%.4f"),
            "positionValue": st.column_config.NumberColumn("Notional", format="$%.2f"),
        },
    )
    st.metric("Total unrealized PnL", f"${pdf['unrealisedPnl'].sum():,.4f}")


# ── Open orders ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Open orders (read-only)")
orders = snap.get("orders", [])
if not orders:
    st.info("No open orders.")
else:
    st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)

st.caption("Read-only via `HyroTrader/dashboard_snapshot.py` on the VPS "
           "(bybit_client v5 → Zurich proxy). Mirrors the Alpaca panel; order routing "
           "intentionally omitted — the live bots (`lr-bybit` / `mm-bybit`) place orders "
           "through the guard-gated `bybit_execution`, not this dashboard.")
