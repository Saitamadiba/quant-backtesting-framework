"""Page 29: Funded Accounts — would the bots have passed a HyroTrader challenge?

Re-runs the selected strategies' real trade history from a chosen date as if it
had been taken on a HyroTrader Trial account, applies their rule set, and reports
PASS / BREACH / IN PROGRESS — at any of the seven account sizes. The rule engine
lives in data/funded_sim.py (unit-tested); this page is the cockpit around it.

Plain-English picture of the five rules:
  • Profit target (+5%)   — the finish line: bank +5% in closed PnL.
  • Max loss (−10%)       — the trapdoor: equity through 90% and the seat is gone.
  • Daily drawdown (5%)   — a daily leash: high-to-low in one UTC day ≤ 5%.
  • Min trading days (5)  — proof you showed up: ≥5 days with a real trade.
  • Stop-loss obligation  — a seatbelt on every trade, risking ≤ 3%.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import VPS_CACHE_DIR, VPS_KNIFE_DB_FILES
from data.data_loader import get_all_trades
from data.vps_sync import sync_vps_databases, sync_single_file
from data.funded_sim import (
    ACCOUNT_SIZES, HYRO_TRIAL, prepare_trades, simulate, compare_sizes,
)

# Knife funded arms — VPS-only `funded_trades` tables, read here directly so the
# maker/taker split survives (the unified loader folds them into one "Knife"
# family). The other fleet seats (desk, retest, OFCS, depth, …) arrive through
# get_all_trades() as their own strategies once the fleet books are synced.
# We read every present knife funded DB and split rows by their `entry_mode`
# (maker post-only limit vs taker market), since the arms can co-exist in one DB
# (e.g. the $100k demo) and older rows pre-date the entry_mode column (= maker).
KNIFE_DBS = ["knife_funded_maker.db", "knife_funded_taker.db", "knife_funded_100k.db",
             "knife_funded_10k.db", "knife_funded_maker2.db", "knife_funded_ethmstop.db"]
KNIFE_FEE_BPS = {"maker": 4.0, "taker": 11.0}  # round-trip: ~2 / ~5.5 bps a side
# Configured funded roster (KNIFE_FUNDED_SYMBOLS, USDT stripped). NB: no ETH.
KNIFE_FUNDED_UNIVERSE = ["ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT",
                         "LINK", "LTC", "SOL", "XRP"]
# Favored holdout mean-R per asset (frozen v1, OOS break ≥ 2025-07-01): the edge
# map that explains the roster — alts carry the knife edge, the majors don't.
KNIFE_FAVORED_R = {"XRP": 0.845, "AVAX": 0.775, "BCH": 0.711, "SOL": 0.617,
                   "DOT": 0.584, "LTC": 0.505, "BNB": 0.379, "ADA": 0.352,
                   "ETH": 0.140, "LINK": 0.037, "DOGE": -0.131, "BTC": -0.166}


@st.cache_data(ttl=60, show_spinner=False)
def load_knife_funded() -> pd.DataFrame:
    """Map the knife `funded_trades` rows into the sim schema, split maker/taker.

    Entry is the armed level (post-only limit / break level); only filled
    positions are real trades. Each arm carries its round-trip fee so the taker
    fee drag shows. Trades are de-duplicated across DBs by fill identity.
    """
    raw = []
    for fn in KNIFE_DBS:
        p = VPS_CACHE_DIR / fn
        if not p.exists():
            continue
        try:
            with sqlite3.connect(p) as conn:
                t = pd.read_sql_query("SELECT * FROM funded_trades", conn)
        except Exception:
            continue
        if not t.empty and "level" in t.columns and "filled_at_utc" in t.columns:
            raw.append(t[t["filled_at_utc"].notna()])
    if not raw:
        return pd.DataFrame()
    t = pd.concat(raw, ignore_index=True)
    dedup = [c for c in ("asset", "direction", "filled_at_utc", "level") if c in t.columns]
    if dedup:
        t = t.drop_duplicates(subset=dedup)
    mode = (t["entry_mode"].fillna("maker").str.lower()
            if "entry_mode" in t.columns else pd.Series("maker", index=t.index))
    mode = mode.where(mode.isin(["maker", "taker"]), "maker")
    asset = t["asset"] if "asset" in t.columns else t.get("symbol")
    return pd.DataFrame({
        "strategy": mode.map(lambda m: f"Knife {m.capitalize()}"),
        "symbol": asset.astype(str).str.replace("USDT", "", regex=False),
        "direction": t["direction"],
        "entry_price": pd.to_numeric(t["level"], errors="coerce"),
        "stop_loss": pd.to_numeric(t["sl"], errors="coerce"),
        "take_profit": pd.to_numeric(t.get("tp"), errors="coerce"),
        "exit_price": pd.to_numeric(t.get("exit_price"), errors="coerce"),
        "exit_reason": t.get("exit_reason"),
        # ISO8601 (not inferred) so rows with/without microseconds both parse —
        # pandas infers the format from the first row and would NaT the rest.
        "entry_time": pd.to_datetime(t["filled_at_utc"], errors="coerce", utc=True,
                                     format="ISO8601"),
        "exit_time": pd.to_datetime(t.get("closed_at_utc"), errors="coerce", utc=True,
                                    format="ISO8601"),
        "pnl_usd": pd.to_numeric(t.get("pnl_usd"), errors="coerce"),
        "source": "Live",
        "fee_bps": mode.map(KNIFE_FEE_BPS).astype(float),
    })

st.set_page_config(page_title="Funded Accounts", page_icon="💰", layout="wide")
st.title("💰 Funded Accounts — pass a HyroTrader challenge?")
st.caption(
    "Pick a start date, the strategies to include, and an account size. The page "
    "re-runs those real trades under HyroTrader Trial rules and tells you whether "
    "the account would have been funded, breached, or still in progress — with the "
    "exact number first and the plain-English picture as a chaser."
)


# ── Sidebar: data refresh + assumptions ──────────────────────────────────────
with st.sidebar:
    st.subheader("Data")
    if st.button("⟳ Update trades from VPS", use_container_width=True):
        with st.spinner("Syncing bot trade DBs from VPS…"):
            res = sync_vps_databases()
            for ln, rp in VPS_KNIFE_DB_FILES.items():
                res[ln] = sync_single_file(ln, rp)
        ok = sum(1 for r in res.values() if r.get("status") == "ok")
        st.success(f"Synced {ok}/{len(res)} DBs")
        for n, r in res.items():
            if r.get("status") != "ok":
                st.write(f"❌ {n}: {r.get('status')} {r.get('error', '')}")
        st.cache_data.clear()
    st.caption("Pulls each bot's trade DB into `dashboard/databases/`, then "
               "rebuilds the unified trade list.")


# ── Load + top controls ──────────────────────────────────────────────────────
source = st.radio("Trade source", ["Live", "All", "Backtest"], horizontal=True,
                  help="Live = real fills only (the honest funded view). "
                       "Backtest includes simulated trades.")
df_all = get_all_trades(source)
# Normalise the unified loader's timestamps to tz-aware UTC BEFORE folding in the
# knife arms (which are already tz-aware). Concatenating a tz-naive column with a
# tz-aware one yields an object column that pd.to_datetime later coerces to NaT —
# silently dropping the knife maker rows. Unify first to avoid that.
for _c in ("entry_time", "exit_time"):
    if _c in df_all.columns:
        df_all[_c] = pd.to_datetime(df_all[_c], errors="coerce", utc=True)
# Fold in the knife maker/taker funded arms (VPS-only; appear after a Sync).
if source in ("Live", "All"):
    knife = load_knife_funded()
    if not knife.empty:
        df_all = pd.concat([df_all, knife], ignore_index=True)
if df_all.empty:
    st.info("No trades loaded. Click **⟳ Update trades from VPS** in the sidebar "
            "(the knife maker/taker arms appear once their DBs are synced).")
    st.stop()

df_all = df_all.dropna(subset=["entry_time"])
min_d, max_d = df_all["entry_time"].min().date(), df_all["entry_time"].max().date()

c1, c2, c3 = st.columns([1.1, 1.6, 1.3])
start = c1.date_input("Review trades from", value=min_d,
                      min_value=min_d, max_value=max_d)
all_strats = sorted(df_all["strategy"].dropna().unique())
strats = c2.multiselect("Strategies (selectable / modifiable)", all_strats,
                        default=all_strats)
syms = sorted(df_all["symbol"].dropna().unique())
sel_syms = c3.multiselect("Symbols", syms, default=syms)

s1, s2, s3 = st.columns([1, 1, 1])
balance = s1.selectbox("Account size", ACCOUNT_SIZES, index=1,
                       format_func=lambda x: f"${x:,.0f}")
risk_display = s2.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25,
                         help="Target risk per position as a % of the account. "
                         "HyroTrader caps this at 3%.")
risk_pct = risk_display / 100.0
max_lev = s3.slider("Max leverage (notional cap)", 1.0, 5.0, 3.0, 0.5,
                    help="Caps position notional. Mirrors HyroTrader's ~3× "
                    "notional backstop and tames trades whose logged stop sat "
                    "almost on the entry.")

# ── Filter ───────────────────────────────────────────────────────────────────
start_ts = pd.Timestamp(start, tz="UTC")
mask = (df_all["entry_time"] >= start_ts) & df_all["strategy"].isin(strats) \
    & df_all["symbol"].isin(sel_syms)
df = df_all[mask].copy()
if df.empty or not strats:
    st.warning("No trades match the current date / strategy / symbol filters.")
    st.stop()

prepared = prepare_trades(df)
if prepared.empty:
    st.warning("None of the filtered trades are simulatable (need entry, stop, "
               "and a recoverable exit).")
    st.stop()

res = simulate(prepared, balance, risk_pct=risk_pct, max_leverage=max_lev)


# ── Verdict banner ───────────────────────────────────────────────────────────
v = res.verdict
vmsg = {
    "PASS": ("✅", st.success, "FUNDED — target reached with no breach"),
    "BREACH": ("❌", st.error, "BREACHED — account would have been closed"),
    "IN PROGRESS": ("⏳", st.warning, "IN PROGRESS — no breach, target not yet met"),
}.get(v, ("ℹ️", st.info, v))
icon, box, label = vmsg
when = (f" on **{pd.Timestamp(res.terminal_date).date()}**"
        if res.terminal_date is not None else "")
n_clamped = int(res.trades["_clamped"].sum()) if "_clamped" in res.trades else 0
clamp_note = (f"  ·  {n_clamped} trade(s) guarded as data outliers (R capped to "
              "±2/+10)" if n_clamped else "")
box(f"{icon} **{v}** — {label}{when}. {res.terminal_event}"
    + f"  ·  {res.coverage}/{res.n_total} trades simulatable "
      f"({res.coverage / max(res.n_total, 1):.0%})" + clamp_note + ".")


# ── Rule scorecard ───────────────────────────────────────────────────────────
st.subheader("Rule scorecard")


def _fmt(rule):
    f, val, lim = rule["fmt"], rule["value"], rule["limit"]
    if f == "pct":
        return f"{val:.2%}", f"{lim:.0%}"
    return f"{int(val)}", f"{int(lim)}"


cols = st.columns(5)
order = ["Profit target", "Max loss", "Daily drawdown",
         "Min trading days", "Stop-loss obligation"]
helps = {
    "Profit target": "Best cumulative closed PnL vs the +5% finish line.",
    "Max loss": "Worst equity drop from start vs the −10% trapdoor.",
    "Daily drawdown": "Worst single-day high-to-low vs the 5% daily leash.",
    "Min trading days": "Distinct days with a qualifying trade (≥5% notional, "
                        "|PnL|≥1%) vs the 5-day minimum.",
    "Stop-loss obligation": "Positions missing a stop or risking >3% (must be 0).",
}
for col, name in zip(cols, order):
    rule = res.rules[name]
    cur, lim = _fmt(rule)
    ok = rule["status"]
    col.metric(f"{'✅' if ok else '❌'} {name}", cur,
               delta=f"limit {lim}", delta_color="off",
               help=helps[name])

m = res.meta
mc = st.columns(4)
mc[0].metric("Final equity", f"${m['final_equity_usd']:,.0f}",
             delta=f"{m['total_pnl_usd'] / balance:+.2%}")
mc[1].metric("Net PnL", f"${m['total_pnl_usd']:,.0f}")
mc[2].metric("Profit target", f"${m['target_usd']:,.0f}")
mc[3].metric("Max-loss floor", f"${m['floor_usd']:,.0f}")


# ── Equity curve ─────────────────────────────────────────────────────────────
st.subheader("Equity curve")
tr = res.trades
fig = go.Figure()
fig.add_trace(go.Scatter(x=tr["ts"], y=tr["equity_$"], mode="lines",
                         name="Equity", line=dict(color="#2196F3", width=2)))
fig.add_hline(y=balance, line_dash="dot", line_color="#888", annotation_text="start")
fig.add_hline(y=balance * (1 + HYRO_TRIAL.profit_target), line_dash="dash",
              line_color="#4CAF50", annotation_text="+5% target")
fig.add_hline(y=m["floor_usd"], line_dash="dash", line_color="#F44336",
              annotation_text="−10% floor")
if res.terminal_date is not None and v in ("PASS", "BREACH"):
    yter = tr.loc[tr["ts"] <= pd.Timestamp(res.terminal_date), "equity_$"]
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(res.terminal_date)],
        y=[yter.iloc[-1] if len(yter) else balance],
        mode="markers", name=v,
        marker=dict(size=13, symbol="star",
                    color="#4CAF50" if v == "PASS" else "#F44336")))
fig.update_layout(height=420, margin=dict(t=30, b=20),
                  yaxis_title="account equity ($)", xaxis_title="")
st.plotly_chart(fig, use_container_width=True)


# ── Daily drawdown ───────────────────────────────────────────────────────────
st.subheader("Daily drawdown — the 5% leash")
daily = res.daily
if not daily.empty:
    daily = daily.copy()
    daily["color"] = np.where(daily["breached"], "#F44336", "#90CAF9")
    figdd = go.Figure()
    figdd.add_trace(go.Bar(x=pd.to_datetime(daily["date"]), y=daily["dd_pct"],
                           marker_color=daily["color"], name="daily DD"))
    figdd.add_hline(y=HYRO_TRIAL.daily_drawdown, line_dash="dash",
                    line_color="#F44336", annotation_text="5% limit")
    figdd.update_layout(height=300, margin=dict(t=20, b=20),
                        yaxis_title="high-to-low (% of account)",
                        yaxis_tickformat=".0%")
    st.plotly_chart(figdd, use_container_width=True)
    n_breach = int(daily["breached"].sum())
    st.caption(
        f"{n_breach} day(s) breached the 5% daily leash. "
        "Note: built from **closed-trade** equity, so it captures realized "
        "intraday swings but not unrealized open-position wiggles — a real "
        "HyroTrader monitor watching live equity could trip a hair earlier."
    )


# ── Account-size comparison ──────────────────────────────────────────────────
st.subheader("Across account sizes")
st.caption(
    "Same trades, every HyroTrader size. The rules are percentage-based, so the "
    "**verdict is identical at every size** — the dollars scale, the pass/fail "
    "doesn't. A bot either clears the funded rules or it doesn't."
)
cmp = compare_sizes(prepared, risk_pct=risk_pct, max_leverage=max_lev)
if not cmp.empty:
    disp = cmp.copy()
    for cc in ("Final equity", "Net PnL $", "Profit target $", "Max loss limit $"):
        disp[cc] = disp[cc].map(lambda x: f"${x:,.0f}")
    disp["Return %"] = disp["Return %"].map(lambda x: f"{x:+.2%}")
    disp["Worst daily DD %"] = disp["Worst daily DD %"].map(lambda x: f"{x:.2%}")
    st.dataframe(disp, hide_index=True, use_container_width=True)


# ── Knife: maker vs taker, side-by-side ──────────────────────────────────────
st.subheader("🔪 Knife: maker vs taker")
st.caption(
    "Both knife funded arms over the **same window and account size**. The maker "
    "rests a post-only limit at the level (≈4 bps round-trip); the taker crosses "
    "the spread at the break (≈11 bps). The **fee drag** is the R the fee alone "
    "eats — the structural reason a taker entry starts underwater on a ~0.5% stop."
)
_knife = load_knife_funded()
if not _knife.empty:
    _knife = _knife[_knife["entry_time"] >= start_ts]
if _knife.empty:
    st.info("No knife funded data in range. Click **⟳ Update trades from VPS** in "
            "the sidebar — the taker arm appears once its DB is synced.")
else:
    def _arm_sim(sub, with_fee):
        d = sub if with_fee else sub.assign(fee_bps=0.0)
        return simulate(prepare_trades(d), balance, risk_pct=risk_pct,
                        max_leverage=max_lev)

    drag = {}
    kcols = st.columns(2)
    for col, arm in zip(kcols, ["Knife Maker", "Knife Taker"]):
        sub = _knife[_knife["strategy"] == arm]
        with col:
            st.markdown(f"#### {arm.replace('Knife ', '')} arm")
            if sub.empty:
                st.info(f"No **{arm}** trades in range — sync the arm's DB to compare.")
                continue
            net, gross = _arm_sim(sub, True), _arm_sim(sub, False)
            if net.trades.empty:
                st.info("No simulatable trades.")
                continue
            nR, gR = net.trades["R"], gross.trades["R"]
            fee_cost = gross.meta["total_pnl_usd"] - net.meta["total_pnl_usd"]
            drag_r = float((gR - nR).mean())
            drag[arm] = (drag_r, fee_cost)
            vbox = {"PASS": st.success, "BREACH": st.error,
                    "IN PROGRESS": st.warning}.get(net.verdict, st.info)
            vbox(f"**{net.verdict}** · {net.terminal_event or '—'}")
            a, b = st.columns(2)
            a.metric("Trades", net.coverage)
            b.metric("Win rate", f"{(nR > 0).mean():.0%}")
            a.metric("Mean R (net)", f"{nR.mean():+.3f}",
                     delta=f"{-drag_r:+.3f} R fee")
            b.metric("Net PnL", f"${net.meta['total_pnl_usd']:,.0f}")
            a.metric("Total R", f"{nR.sum():+.1f}")
            b.metric("Fee drag", f"−${fee_cost:,.0f}",
                     help="Dollars the round-trip fee removed over the window.")
    if "Knife Maker" in drag and "Knife Taker" in drag:
        mr, tr = drag["Knife Maker"][0], drag["Knife Taker"][0]
        ratio = f"~{tr / mr:.1f}× " if mr > 1e-9 else ""
        st.caption(
            f"Taker fee drag **{tr:+.3f} R/trade** vs maker **{mr:+.3f} R/trade** — "
            f"the taker pays {ratio}the fee per trade. On a ~0.5%-stop knife where "
            "the favored edge is only ~+0.34R, that gap is the whole edge."
        )
    elif "Knife Taker" not in drag:
        st.caption("Only the **maker** arm is synced. Sync a taker `funded_trades` "
                   "DB to see the side-by-side fee comparison.")

    # ── universe coverage — why some symbols (BTC/ETH) are absent ─────────
    with st.expander("🌐 Knife universe coverage — roster vs what actually traded"):
        counts = _knife.groupby("symbol").size()
        syms = list(dict.fromkeys(KNIFE_FUNDED_UNIVERSE + counts.index.tolist()))
        rows = []
        for s in syms:
            note = ""
            if s == "BTC":
                note = ("In the roster, but the knife rarely flags a *favored* BTC "
                        "break — majors are too deep/efficient to fade.")
            elif s == "ETH":
                note = "Not in the funded roster (KNIFE_FUNDED_SYMBOLS) at all."
            rows.append({
                "symbol": s,
                "in roster": "✓" if s in KNIFE_FUNDED_UNIVERSE else "—",
                "trades in range": int(counts.get(s, 0)),
                "favored R (OOS)": KNIFE_FAVORED_R.get(s, float("nan")),
                "note": note,
            })
        cov = pd.DataFrame(rows).sort_values("favored R (OOS)", ascending=False,
                                             na_position="last")
        cov["favored R (OOS)"] = cov["favored R (OOS)"].map(
            lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
        st.dataframe(cov, hide_index=True, use_container_width=True)
        st.caption(
            "**ETH** isn't in `KNIFE_FUNDED_SYMBOLS`; **BTC** is, but almost never "
            "qualifies (≈9 breaks / 0 favored in the live window, and the worst "
            "asset at −0.17R). The knife fades over-extended stop-runs, and the "
            "deep, efficient majors rarely overshoot enough to fade — the edge "
            "lives in the jumpier alts (XRP +0.85R, AVAX +0.78R). So the absence "
            "is by design, not missing data."
        )


# ── Sized trade ledger ───────────────────────────────────────────────────────
st.subheader("Sized trade ledger")
st.caption(
    "Every filtered trade re-sized to this account: notional ≥5% of balance, "
    "risk target ≤3%, capped at the leverage limit. `R` is the price-based "
    "risk-multiple (−1 = stop hit)."
)
ledger = res.trades.copy()
ledger["date"] = ledger["ts"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M")
view_cols = {
    "date": "date", "strategy": "strategy", "symbol": "symbol",
    "direction": "direction", "notional_$": "notional $", "risk_$": "risk $",
    "R": "R", "pnl_$": "PnL $", "equity_$": "equity $", "qualifies": "qual day",
}
have = {k: v for k, v in view_cols.items() if k in ledger.columns}
led = ledger[list(have)].rename(columns=have)
for cc in ("notional $", "risk $", "PnL $", "equity $"):
    if cc in led.columns:
        led[cc] = led[cc].round(0)
led["R"] = led["R"].round(2)
st.dataframe(led, hide_index=True, use_container_width=True, height=340)
st.download_button(
    "⬇ Download sized ledger (CSV)",
    led.to_csv(index=False).encode(),
    file_name=f"funded_{int(balance)}_{'-'.join(strats)}.csv", mime="text/csv")


# ── Rules reference ──────────────────────────────────────────────────────────
with st.expander("📋 HyroTrader Trial rules + how this page models them"):
    st.markdown(
        f"""
| Rule | HyroTrader requirement | How the sim applies it |
|---|---|---|
| **Profit target** | +5% of initial balance in closed PnL | cumulative realized PnL ≥ ${HYRO_TRIAL.profit_target:.0%} |
| **Max loss** | equity must stay ≥ 90% of start | breach if equity ever drops below the −10% floor |
| **Daily drawdown** | high-to-low in one UTC day ≤ 5% | per-day high-to-low of closed-trade equity ≤ 5% |
| **Min trading days** | ≥5 days, each with a real trade | distinct days with a trade ≥5% notional **and** \\|PnL\\| ≥1% |
| **Stop loss** | every position; risk ≤ 3% | flags any trade missing a stop or risking >3% |

**Sizing.** `notional = clip(risk% ÷ stop-distance, 5%, leverage-cap) × balance`,
then `PnL = notional × price-move`. Returns come from **price geometry**
(entry / stop / exit / direction), not the bots' differently-scaled logged R —
so heterogeneous bots compare apples-to-apples.

**Knife arms.** The **Knife Maker** and **Knife Taker** funded arms are folded in
from their VPS `funded_trades` tables (entry at the armed level; only filled
positions). They carry their round-trip fee (maker ≈ 4 bps, taker ≈ 11 bps) so
the taker fee drag shows; the other bots are modeled fee-free (their logs don't
separate fees). They appear in the strategy list once their DBs are synced.

**Honest caveats.** (1) Daily drawdown and max-loss use *closed-trade* equity;
true intraday unrealized swings aren't in the local trade logs, so a live monitor
could trip marginally earlier. (2) Missing exit prices are reconstructed from the
exit reason (stop→stop price, take-profit→target). (3) The verdict is size-
invariant by construction — the dollars scale, the rules are percentages.
"""
    )
