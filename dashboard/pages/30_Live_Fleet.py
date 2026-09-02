"""Page 30: Live Fleet — every bot, right now: balances, running entries, 7-day PnL.

One page that answers the three questions you actually ask at the desk:

  1. **What's in the accounts?** Equity, free margin and unrealised PnL for every
     ByBit sub-account the fleet trades — plus which seats share one account and
     which keys have stopped answering. (Two seats on one sub-account are two
     nameplates on one mailbox: the labels differ, every letter lands in the same
     slot.)
  2. **What's running right now?** Live exchange positions (the ground truth for
     money at risk) beside each bot's own book — filled legs and orders still
     working at the exchange but not yet filled.
  3. **What did they do this week?** Realized PnL over a rolling window, per bot
     and per day, in dollars for the seats that place orders and in R for the
     paper and shadow books.

Everything is read in ONE read-only SSH round trip (`data/fleet_live.py` pipes
`data/fleet_collector_remote.py` to the VPS python). Every statement is a SELECT,
every database is opened `mode=ro`, and no ByBit key is ever held locally — this
page can read the fleet and cannot change it.

Reading the numbers honestly, per the house standard:
  • **Tier 1** (live/funded/demo orders) — realized **$** and R. The money line.
  • **Tier 2** (paper / virtual book) — net **R** on a virtual $100k; the dollars
    are simulated, so read R, not a bank balance.
  • **Tier 3** (shadow / record-only) — dimensionless **cumulative R** by design.
    A deep-negative shadow is a confirmed-dead detector still faithfully
    recording — the black box on a plane that already landed, not a bug.
  • R is net of the fee/slip toll unless the label says *gross*.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Live Fleet", page_icon="🛰️", layout="wide")

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go

from data.fleet_live import (
    accounts_frame, books_frame, daily_frame, exchange_orders_frame,
    exchange_positions_frame, fetch_raw, open_frame, reconcile, seat_status_frame,
    trades_frame,
)
from data.fleet_registry import TIER_NAMES

TIER_COLOR = {1: "#2e7d32", 2: "#1565c0", 3: "#8d6e63"}


# ══════════════════════════════════════════════════════════════════════════════
#  Fetch
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def load(days: int, with_balances: bool) -> dict:
    return fetch_raw(days=days, with_balances=with_balances, timeout=150)


st.title("🛰️ Live Fleet")
st.caption(
    "Every bot on the VPS in one read-only pull: account balances, the positions "
    "running right now, and what the books banked over the window. "
    "One phone call to the warehouse, not forty ledgers couriered home."
)

c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.4, 1.4])
days = c1.selectbox("Window", [1, 3, 7, 14, 30], index=2,
                    format_func=lambda d: f"{d}d", help="Rolling lookback for the PnL half of the page.")
with_bal = c2.toggle("Balances", value=True,
                     help="Read each seat's ByBit equity + live positions (GET-only, ~5s). "
                          "Turn off for a faster, database-only refresh.")
if c3.button("↻ Refresh now", use_container_width=True, type="primary"):
    load.clear()
try:
    from streamlit_autorefresh import st_autorefresh
    every = c4.selectbox("Auto-refresh", ["Off", "1m", "5m", "15m"], index=0)
    if every != "Off":
        st_autorefresh(interval={"1m": 60_000, "5m": 300_000, "15m": 900_000}[every],
                       key="fleet_autorefresh")
except ImportError:
    pass

with st.spinner("Reading the fleet over SSH…"):
    raw = load(days, with_bal)

if not raw.get("ok"):
    st.error(f"Fleet snapshot unavailable — **{raw.get('error', 'unknown error')}**")
    st.caption("This page fails closed: no VPS, no numbers. Nothing was written or restarted. "
               "Check `dashboard/.env` (VPS_HOST/PORT/USER) and that the VPS is reachable.")
    st.stop()

books = books_frame(raw)
trades = trades_frame(raw, days=days)
daily = daily_frame(raw)
running = open_frame(raw)
accounts = accounts_frame(raw)
positions = exchange_positions_frame(raw)
orders = exchange_orders_frame(raw)
seats = seat_status_frame(raw)

n_fail = sum(1 for v in raw.get("results", {}).values() if not v.get("ok"))
st.caption(
    f"VPS clock **{raw.get('server_utc', '?')} UTC** · {len(books)} books read · "
    f"{n_fail} query error(s) · balances "
    f"{'read in ' + str(raw.get('balances', {}).get('elapsed_s', '?')) + 's' if with_bal else 'skipped'} · "
    f"cached 60s (local {datetime.now().strftime('%H:%M:%S')})"
)
if n_fail:
    with st.expander(f"⚠️ {n_fail} book quer(y/ies) failed — these bots are missing below"):
        st.write({k: v.get("error") for k, v in raw["results"].items() if not v.get("ok")})


# ══════════════════════════════════════════════════════════════════════════════
#  1 · Balances
# ══════════════════════════════════════════════════════════════════════════════
st.header("1 · Balances")
st.caption(
    "One row per distinct exchange **account**, not per bot — several seats often "
    "share one sub-account, and asking once per seat would double-count the money."
)

if not with_bal:
    st.info("Balances are switched off. Flip the **Balances** toggle to read them.")
elif accounts.empty:
    st.warning("No ByBit account answered. Every seat's key failed, or the tunnel is down.")
else:
    ok = accounts[accounts["status"] == "ok"]
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Total equity", f"${ok['equity'].sum():,.0f}",
              help="Sum of `totalEquity` across every account that answered — coins included, "
                   "so on demo seats it sits above the USDT wallet balance.")
    b2.metric("Wallet balance", f"${ok['wallet_balance'].sum():,.0f}",
              help="Settled cash, before unrealised PnL.")
    b3.metric("Available", f"${ok['available'].sum():,.0f}",
              help="Free margin — what is not already pledged against an open position.")
    b4.metric("Unrealised PnL", f"${ok['upnl'].sum():,.0f}",
              delta=f"{ok['upnl'].sum():,.0f}",
              help="Open-position PnL, marked to market. Not yet money: it moves every tick.")
    dead = accounts[accounts["status"] != "ok"]
    b5.metric("Accounts reporting", f"{len(ok)}/{len(accounts)}",
              delta=None if dead.empty else f"-{len(dead)} silent",
              delta_color="inverse",
              help="An account that does not answer is a bot flying blind — a dead key, "
                   "a revoked permission, or an IP the exchange no longer recognises.")

    st.dataframe(
        accounts.sort_values("equity", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "uid": st.column_config.TextColumn("Account UID", width="small"),
            "seats": st.column_config.TextColumn("Seats trading it", width="large"),
            "n_seats": st.column_config.NumberColumn("Seats", width="small"),
            "env": st.column_config.TextColumn("Env", width="small"),
            "equity": st.column_config.NumberColumn("Equity", format="$%.0f"),
            "wallet_balance": st.column_config.NumberColumn("Wallet", format="$%.0f"),
            "available": st.column_config.NumberColumn("Available", format="$%.0f"),
            "upnl": st.column_config.NumberColumn("uPnL", format="$%.2f"),
            "positions": st.column_config.NumberColumn("Open pos", width="small"),
            "status": st.column_config.TextColumn("Status"),
        },
    )

    shared = accounts[accounts["n_seats"] > 1]
    if not shared.empty:
        st.warning(
            "**Shared sub-accounts:** " +
            " · ".join(f"`{r.uid}` ← {r.seats}" for r in shared.itertuples()) +
            "  \nTwo seats on one account are one position, one stop and two claim "
            "tickets — the ledger reports two books, the exchange only ever had one."
        )
    bad_seats = seats[seats["status"] != "ok"] if not seats.empty else pd.DataFrame()
    if not bad_seats.empty:
        st.error(
            f"**{len(bad_seats)} key(s) not answering** — the seats behind them are dark: " +
            " · ".join(f"{r.seats} ({r.status})" for r in bad_seats.itertuples())
        )
    with st.expander("Key ↔ account map (hashed keys — no secret leaves the VPS)"):
        st.dataframe(seats, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  2 · Running entries
# ══════════════════════════════════════════════════════════════════════════════
st.header("2 · Running entries")
st.caption(
    "What is live at this second: the exchange's own view first — that is the "
    "money genuinely at risk — then each bot's book, which also shows orders "
    "still *working* (placed, resting, not yet filled)."
)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Exchange positions", f"{len(positions)}",
          help="Non-zero positions ByBit reports across every account that answered. "
               "The ground truth — a bot book can be wrong, a fill cannot.")
r2.metric("Notional at risk", f"${positions['value'].sum():,.0f}" if not positions.empty else "$0",
          help="Sum of position value. Not your loss if it goes wrong — that is the stop distance — "
               "but it is the size the exchange has on your behalf.")
r3.metric("Open uPnL", f"${positions['upnl'].sum():,.2f}" if not positions.empty else "$0.00",
          help="Mark-to-market on those positions.")
t1_open = int(running[(running["tier"] == 1) & (running["state"] == "FILLED")].shape[0])
r4.metric("Orders resting", f"{len(orders)}",
          delta=f"{t1_open} book legs" if t1_open != len(positions) else None,
          delta_color="off",
          help="Unfilled orders the exchange is holding right now. The delta shows what the "
               "Tier-1 books believe they hold filled — if that disagrees with the position "
               "count on the left, it is a reconciliation problem, not a rounding one.")

if not positions.empty:
    st.subheader("Exchange positions", divider="gray")
    p = positions.copy()
    p["drift_%"] = ((p["mark_price"] - p["avg_price"]) / p["avg_price"] * 100).where(p["avg_price"] > 0)
    st.dataframe(
        p.sort_values("upnl"), use_container_width=True, hide_index=True,
        column_config={
            "account": st.column_config.TextColumn("Account", width="medium"),
            "uid": st.column_config.TextColumn("UID", width="small"),
            "symbol": "Symbol", "side": "Side",
            "size": st.column_config.NumberColumn("Size", format="%.4f"),
            "avg_price": st.column_config.NumberColumn("Avg entry", format="%.5f"),
            "mark_price": st.column_config.NumberColumn("Mark", format="%.5f"),
            "drift_%": st.column_config.NumberColumn("Move %", format="%.2f%%"),
            "upnl": st.column_config.NumberColumn("uPnL", format="$%.2f"),
            "value": st.column_config.NumberColumn("Notional", format="$%.0f"),
            "leverage": st.column_config.NumberColumn("Lev", format="%.0fx"),
            "sl": st.column_config.NumberColumn("SL", format="%.5f"),
            "tp": st.column_config.NumberColumn("TP", format="%.5f"),
        },
    )
    naked = p[p["sl"] == 0]
    if not naked.empty:
        who = ", ".join(sorted(set(naked["account"].astype(str))))
        st.warning(
            f"**{len(naked)} position(s) carry no native stop** at the exchange — {who}. "
            "A delta-neutral carry leg is hedged by its twin and is meant to run without "
            "one; a directional seat is not. Anywhere else, a stop that lives only inside a "
            "running process dies with the process."
        )
elif with_bal:
    st.info("Flat — no open position on any account that answered.")

if with_bal:
    st.subheader("Orders resting at the exchange", divider="gray")
    if orders.empty:
        st.info("No unfilled order is resting on any account that answered.")
    else:
        st.dataframe(
            orders[["account", "symbol", "side", "order_type", "qty", "price",
                    "status", "sl", "tp", "reduce_only", "placed", "age_h"]],
            use_container_width=True, hide_index=True,
            column_config={
                "account": st.column_config.TextColumn("Account", width="medium"),
                "symbol": "Symbol", "side": "Side",
                "order_type": st.column_config.TextColumn("Type", width="small"),
                "qty": st.column_config.NumberColumn("Qty", format="%.4f"),
                "price": st.column_config.NumberColumn("Limit", format="%.5f"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "sl": st.column_config.NumberColumn("SL", format="%.5f"),
                "tp": st.column_config.NumberColumn("TP", format="%.5f"),
                "reduce_only": st.column_config.CheckboxColumn("Reduce-only"),
                "placed": st.column_config.DatetimeColumn("Placed (UTC)", format="MM-DD HH:mm"),
                "age_h": st.column_config.NumberColumn("Age (h)", format="%.1f"),
            },
        )
        st.caption(
            f"{len(orders)} live order(s). This is the venue's own list — the truth about "
            "what can still fill. A reduce-only order is an exit already parked in the book."
        )

st.subheader("Bot books — filled and working legs", divider="gray")
tier_pick = st.multiselect(
    "Tiers", [1, 2, 3], default=[1, 2],
    format_func=lambda t: TIER_NAMES[t],
    help="Tier 3 shadow legs are recorded, not traded — they carry no money and are "
         "off by default so the real book stays legible.",
)
rv = running[running["tier"].isin(tier_pick)] if tier_pick else running.iloc[0:0]
if rv.empty:
    st.info("No running legs in the selected tiers.")
else:
    show = rv[["bot", "state", "symbol", "side", "entry", "sl", "tp", "qty",
               "risk_usd", "since", "age_h"]]
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config={
            "bot": st.column_config.TextColumn("Bot", width="medium"),
            "state": st.column_config.TextColumn("State", width="small"),
            "symbol": "Symbol", "side": "Side",
            "entry": st.column_config.NumberColumn("Entry", format="%.5f"),
            "sl": st.column_config.NumberColumn("SL", format="%.5f"),
            "tp": st.column_config.NumberColumn("TP", format="%.5f"),
            "qty": st.column_config.NumberColumn("Qty", format="%.4f"),
            "risk_usd": st.column_config.NumberColumn("Risk", format="$%.0f"),
            "since": st.column_config.DatetimeColumn("Since (UTC)", format="MM-DD HH:mm"),
            "age_h": st.column_config.NumberColumn("Age (h)", format="%.1f"),
        },
    )
    st.caption(
        f"{len(rv)} leg(s) · **FILLED** = the bot believes it is in the trade · "
        "**WORKING** = the book armed an order in the last 6h. A resting maker order "
        "costs nothing until it fills — but it is a claim on the seat's risk budget. "
        "For what is genuinely live, trust the exchange tables above."
    )

# ── reconciliation: the book's word against its own account ─────────────────
if with_bal:
    orphans = reconcile(raw, running)
    if not orphans.empty:
        st.error(
            f"**{len(orphans)} Tier-1 book leg(s) the exchange does not confirm.** "
            "The seat still counts these as open — against its risk budget and its "
            "consecutive-loss count — while the venue holds nothing in that symbol. "
            "A ticket stub for a seat nobody is sitting in."
        )
        st.dataframe(
            orphans[["bot", "symbol", "side", "entry", "sl", "tp", "since", "age_h"]],
            use_container_width=True, hide_index=True,
            column_config={
                "bot": st.column_config.TextColumn("Bot", width="medium"),
                "symbol": "Symbol", "side": "Side",
                "entry": st.column_config.NumberColumn("Entry", format="%.5f"),
                "sl": st.column_config.NumberColumn("SL", format="%.5f"),
                "tp": st.column_config.NumberColumn("TP", format="%.5f"),
                "since": st.column_config.DatetimeColumn("Opened (UTC)", format="YYYY-MM-DD HH:mm"),
                "age_h": st.column_config.NumberColumn("Age (h)", format="%.0f"),
            },
        )
        st.caption(
            "Read with care before acting: a seat whose key is dark cannot be checked "
            "against its own account, and an account that answered but holds nothing is "
            "the stronger signal. This page never closes or edits anything — it reports."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  3 · Performance over the window
# ══════════════════════════════════════════════════════════════════════════════
st.header(f"3 · Last {days} days")

t1 = books[books["tier"] == 1]
t2 = books[books["tier"] == 2]
t3 = books[books["tier"] == 3]
w_trades = int(books["n_7d"].sum())
w_pnl = float(t1["pnl_usd_7d"].fillna(0).sum())
w_r1 = float(t1["sum_r_7d"].fillna(0).sum())
w_r2 = float(t2["sum_r_7d"].fillna(0).sum())
w_r3 = float(t3["sum_r_7d"].fillna(0).sum())
t1_tr = trades[trades["tier"] == 1]
w_wr = float((t1_tr["r"] > 0).mean()) if len(t1_tr) and t1_tr["r"].notna().any() else float("nan")

k = st.columns(5)
k[0].metric("Tier-1 realized", f"${w_pnl:,.0f}", delta=f"{w_pnl:,.0f}",
            help="Closed PnL booked by the order-placing seats over the window. "
                 "This is the money line — everything else is a rehearsal or an instrument.")
k[1].metric("Tier-1 ΣR", f"{w_r1:+.2f}R", delta=f"{w_r1:+.2f}",
            help="Same trades in risk units: R = PnL ÷ the dollars risked on that trade. "
                 "Net of the fee/slip toll unless a label says gross.")
k[2].metric("Trades closed", f"{w_trades:,}",
            help=f"Across all three tiers in the last {days} days, counted on the VPS "
                 f"(not from the capped ledger below). Tier 1: {int(t1['n_7d'].sum()):,}.")
k[3].metric("Tier-1 win rate", "—" if w_wr != w_wr else f"{w_wr:.0%}",
            help="Share of Tier-1 closes with R > 0. A high win rate with negative ΣR "
                 "still loses money — the losers are simply bigger.")
k[4].metric("Tier-2 / Tier-3 ΣR", f"{w_r2:+.1f} / {w_r3:+.1f}R",
            help="Paper book and shadow recorders. Tier-3 R is dimensionless by design "
                 "(no sizing) — an edge-measuring instrument, not a P&L.")

# ── per-day PnL ──────────────────────────────────────────────────────────────
if not daily.empty:
    by_day = daily.groupby(["day", "tier"], as_index=False).agg(
        r=("sum_r", "sum"), pnl=("sum_pnl", "sum"), n=("n", "sum"))

    gc1, gc2 = st.columns(2)
    with gc1:
        st.subheader("Realized $ per day — Tier 1", divider="gray")
        dd = by_day[by_day["tier"] == 1]
        if dd.empty or dd["pnl"].abs().sum() == 0:
            st.info("No Tier-1 dollars booked in the window.")
        else:
            fig = go.Figure(go.Bar(
                x=dd["day"], y=dd["pnl"],
                marker_color=["#2e7d32" if v >= 0 else "#c62828" for v in dd["pnl"]],
                hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="realized $", xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each bar is one UTC day's closed PnL from the seats that place real orders.")
    with gc2:
        st.subheader("Cumulative R by tier", divider="gray")
        fig = go.Figure()
        for tier in sorted(by_day["tier"].unique()):
            dd = by_day[by_day["tier"] == tier].sort_values("day")
            fig.add_trace(go.Scatter(
                x=dd["day"], y=dd["r"].cumsum(),
                name=TIER_NAMES[int(tier)].split("·")[0].strip(),
                mode="lines+markers", line=dict(color=TIER_COLOR[int(tier)], width=2)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="cumulative R", xaxis_title=None,
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Every closed trade in the window, counted on the VPS. R stacks across "
                   "tiers on one axis — but only the Tier-1 line is money.")

# ── per-bot table ────────────────────────────────────────────────────────────
st.subheader("Per-bot scoreboard", divider="gray")
h1, h2 = st.columns([1, 3])
hide_empty = h1.toggle("Hide silent books", value=True,
                       help="Hide books with no trade ever and nothing running — dead or "
                            "not-yet-armed seats. Turn off to audit the full roster.")
view = books.copy()
if hide_empty:
    view = view[(view["n"] > 0) | (view["open_n"] > 0) | (view["working_n"] > 0)]

tabs = st.tabs([TIER_NAMES[t] for t in (1, 2, 3)])
for tab, tier in zip(tabs, (1, 2, 3)):
    with tab:
        v = view[view["tier"] == tier].sort_values("n_7d", ascending=False)
        if v.empty:
            st.info("No book in this tier matches the filter.")
            continue
        cols = ["bot", "n_7d", "mean_r_7d", "sum_r_7d"]
        cfg = {
            "bot": st.column_config.TextColumn("Bot", width="medium"),
            "n_7d": st.column_config.NumberColumn(f"n ({days}d)", width="small"),
            "mean_r_7d": st.column_config.NumberColumn(f"mean R ({days}d)", format="%.3f"),
            "sum_r_7d": st.column_config.NumberColumn(f"ΣR ({days}d)", format="%+.2f"),
            "win_rate_7d": st.column_config.ProgressColumn(
                f"WR ({days}d)", format="%.0f%%", min_value=0, max_value=1),
            "pnl_usd_7d": st.column_config.NumberColumn(f"$ ({days}d)", format="$%.0f"),
            "n": st.column_config.NumberColumn("n (life)", width="small"),
            "mean_r": st.column_config.NumberColumn("mean R (life)", format="%.3f"),
            "sum_r": st.column_config.NumberColumn("ΣR (life)", format="%+.1f"),
            "pnl_usd": st.column_config.NumberColumn("$ (life)", format="$%.0f"),
            "open_n": st.column_config.NumberColumn("open", width="small"),
            "working_n": st.column_config.NumberColumn("working", width="small"),
            "note": st.column_config.TextColumn("Caveat", width="large"),
        }
        cols += ["win_rate_7d"]
        if tier != 3:
            cols += ["pnl_usd_7d"]
        cols += ["open_n", "working_n", "n", "mean_r", "sum_r"]
        if tier != 3:
            cols += ["pnl_usd"]
        cols += ["note"]
        st.dataframe(v[cols], use_container_width=True, hide_index=True, column_config=cfg)
        if tier == 1:
            st.caption("Dollars are real closes on funded/demo seats. R is net of the "
                       "fee/slip toll — the ~0.15–0.35R crossing charge an entry must "
                       "out-earn before anything is left over.")
        elif tier == 2:
            st.caption("A virtual $100k book: the dollars are simulated and uncapped, so "
                       "read the R column. This tier answers *would this pass a challenge*.")
        else:
            st.caption("Dimensionless by design — no sizing, so no dollars. A deep-negative "
                       "line here is a confirmed-dead detector still faithfully recording: "
                       "the black box on a plane that already landed.")

# ── trade ledger ─────────────────────────────────────────────────────────────
st.subheader(f"Trade ledger — every close in the last {days} days", divider="gray")
if trades.empty:
    st.info("No book closed a trade in the window.")
else:
    f1, f2, f3 = st.columns([1.2, 2, 1])
    ledger_tiers = f1.multiselect("Tier", [1, 2, 3], default=[1],
                                  format_func=lambda t: f"Tier {t}", key="ledger_tiers")
    pool = trades[trades["tier"].isin(ledger_tiers)] if ledger_tiers else trades
    bot_pick = f2.multiselect("Bot", sorted(pool["bot"].unique()), default=[],
                              help="Empty = every bot in the selected tiers.")
    if bot_pick:
        pool = pool[pool["bot"].isin(bot_pick)]
    only_losers = f3.toggle("Losers only", value=False)
    if only_losers:
        pool = pool[pool["r"] < 0]

    st.dataframe(
        pool[["ts", "bot", "symbol", "side", "entry", "exit_px", "r", "pnl"]],
        use_container_width=True, hide_index=True, height=420,
        column_config={
            "ts": st.column_config.DatetimeColumn("Closed (UTC)", format="MM-DD HH:mm"),
            "bot": st.column_config.TextColumn("Bot", width="medium"),
            "symbol": "Symbol", "side": "Side",
            "entry": st.column_config.NumberColumn("Entry", format="%.5f"),
            "exit_px": st.column_config.NumberColumn("Exit", format="%.5f"),
            "r": st.column_config.NumberColumn("R", format="%+.3f"),
            "pnl": st.column_config.NumberColumn("PnL", format="$%.2f"),
        },
    )
    s_r = pool["r"].sum(skipna=True)
    s_p = pool["pnl"].sum(skipna=True)
    st.caption(
        f"{len(pool):,} close(s) shown · ΣR **{s_r:+.2f}** · Σ$ **{s_p:,.2f}** "
        "(dollars only where the book prices in dollars — Tier-3 rows carry none). "
        "The ledger keeps the newest 500 closes **per book**, so a chatty shadow "
        "recorder is trimmed here; the scoreboard and charts above count every row."
    )
    st.download_button(
        "⬇ Download this ledger (CSV)",
        pool.to_csv(index=False).encode(),
        file_name=f"fleet_trades_{days}d_{datetime.now(timezone.utc):%Y%m%dT%H%M}Z.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Read-only by construction: SELECT-only SQL, `mode=ro` databases, GET-only exchange "
    "calls, no key held locally. Books the session recap also prints are marked "
    "`in_recap` in `data/fleet_registry.py`, so this page and the terminal scoreboard "
    "can be reconciled line by line."
)
