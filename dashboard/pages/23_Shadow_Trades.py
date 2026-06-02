"""Page 23: Shadow Trades — practice vs theory, per gate.

Every signal the LR live path rejects via a gate is recorded as a paper
trade in a per-bot ``<sym>_shadow_trades.db``.  Each shadow carries a
``block_reason`` so we can confront live R-multiples per gate against the
WFO claim that justified it.

Hybrid paper-bot extension (2026-05-31): the 7 no-Deribit-data LR paper bots
(XRP + BNB/DOGE/AVAX/LINK/DOT/BCH) route off-window signals here too, with
``block_reason='OFF_WINDOW'``, plus ``BTC_REGIME_BLOCK`` when the BTC-regime
gate fires.  This page now supports per-bot filtering, BTC-regime asof tagging
(via duckdb_data/trading_data.duckdb), per-session / per-regime / per-MTF
breakdowns, and a what-if RR analysis (replay each trade against klines at
reward 1.5R / 2R / 3R, fixed 1R risk).

Gates currently shadowed:

* ``OFF_WINDOW`` — signal generated outside the live 12-16 ET window.
* ``BTC_REGIME_BLOCK`` — BTC was in a losing regime for the asset.
* ``IV_GATE_{LOW|MED|HIGH}`` — DVOL bucket on the IV-block list (mostly LR ETH).
* ``LONDON_SHORTS_BAN`` — London-High SHORT (hardcoded 0%-WR rule on BTC/SOL).
* ``REGIME_BLOCK_<regime>`` — strategy disabled in regime (e.g. NQ ranging).
* ``COUNTER_TREND_LONG_IN_TRENDING_DOWN`` / ``..._SHORT_IN_TRENDING_UP``.
* ``SUPPRESSION_SHORT`` — SHORT in gamma SUPPRESSION.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    SHADOW_DB_STRATEGY_MAP,
    VPS_CACHE_DIR,
    VPS_SHADOW_DB_FILES,
    DUCKDB_PATH,
)
from data.vps_sync import sync_single_file


st.set_page_config(page_title="Shadow Trades", page_icon="🌑", layout="wide")
st.title("🌑 Shadow Trades — practice vs theory")

st.markdown(
    "Every signal the LR live path **rejects** via a gate is tracked here as a "
    "paper trade.  Compare the **live R-multiple per gate** to the WFO claim "
    "that put the gate there.  When a row's `total R` disagrees with the cited "
    "WFO finding for more than a couple of weeks, the gate is the suspect."
)


# ── Sync controls (sidebar) ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Sync")
    if st.button("⟳ Sync shadow DBs from VPS", use_container_width=True):
        results: dict[str, dict] = {}
        with st.spinner("Syncing shadow DBs..."):
            for local_name, remote_path in VPS_SHADOW_DB_FILES.items():
                results[local_name] = sync_single_file(local_name, remote_path)
        ok = sum(1 for r in results.values() if r.get("status") == "ok")
        st.success(f"Synced {ok}/{len(results)}")
        for name, r in results.items():
            if r.get("status") != "ok":
                st.write(f"❌ {name}: {r.get('status')} {r.get('error','')}")
    st.caption(
        "Shadow DBs live next to each bot's trade DB on the VPS.  "
        "Sync pulls the latest snapshot into `dashboard/databases/`."
    )


# ── Loaders ──────────────────────────────────────────────────────────────────
def _load_one(local_name: str) -> pd.DataFrame:
    p = VPS_CACHE_DIR / local_name
    if not p.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(p) as conn:
            df = pd.read_sql_query("SELECT * FROM shadow_trades", conn)
    except Exception as e:
        st.warning(f"Could not read {local_name}: {e}")
        return pd.DataFrame()
    if df.empty:
        return df
    strat, sym = SHADOW_DB_STRATEGY_MAP.get(local_name, ("?", "?"))
    df["strategy"] = strat
    df["symbol"] = sym
    _abbr = "".join(w[0] for w in strat.split()) if strat != "?" else "?"
    df["bot"] = f"{_abbr} {sym}"
    for c in ("opened_at_utc", "closed_at_utc"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


frames = [_load_one(name) for name in VPS_SHADOW_DB_FILES]
frames = [f for f in frames if not f.empty]
all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── BTC regime asof tagging (cached) ─────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _btc_regime_series_from_duckdb() -> pd.Series:
    """Return a pd.Series indexed by UTC timestamp with the BTC new5 regime
    label per 15m bar.  Uses the shared regime_classifier so the labels match
    the bots' BTCRegimeMonitor exactly."""
    try:
        import duckdb
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        df = con.execute("""
            SELECT timestamp, open AS Open, high AS High, low AS Low,
                   close AS Close, volume AS Volume
            FROM ohlcv_data
            WHERE symbol = 'BTC' AND timeframe = '15m'
            ORDER BY timestamp
        """).fetchdf()
        con.close()
    except Exception as e:
        st.warning(f"BTC regime tagging unavailable (duckdb read failed: {e})")
        return pd.Series(dtype="object")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    try:
        import sys, pathlib
        # repo root sits two levels up from dashboard/pages/
        _root = pathlib.Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from regime_classifier import compute_features, classify_rule_based
        lc = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        feats = compute_features(lc[["open", "high", "low", "close", "volume"]])
        labels = classify_rule_based(feats).dropna()
        return labels
    except Exception as e:
        st.warning(f"regime_classifier not available: {e}")
        return pd.Series(dtype="object")


def _asof_tag_btc_regime(ts: pd.Series) -> pd.Series:
    btc = _btc_regime_series_from_duckdb()
    if btc.empty:
        return pd.Series([pd.NA] * len(ts), index=ts.index, dtype="object")
    # ensure tz-aware UTC for both sides
    ts_utc = pd.to_datetime(ts, utc=True, errors="coerce")
    idx = btc.index.searchsorted(ts_utc.fillna(btc.index.min()), side="right") - 1
    out = pd.Series(
        np.where(idx >= 0, btc.iloc[np.clip(idx, 0, None)].to_numpy(), pd.NA),
        index=ts.index, dtype="object",
    )
    out[ts_utc.isna()] = pd.NA
    return out


if not all_df.empty and "opened_at_utc" in all_df.columns:
    all_df["btc_regime"] = _asof_tag_btc_regime(all_df["opened_at_utc"])


# ── Sidebar bot filter ───────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("Filter")
    bot_options = ["All bots"] + (
        sorted(all_df["bot"].unique().tolist()) if not all_df.empty else []
    )
    sel_bot = st.selectbox("Bot", bot_options, index=0)


view_df = all_df if sel_bot == "All bots" else all_df[all_df["bot"] == sel_bot]


# ── Cache status row ─────────────────────────────────────────────────────────
st.subheader("Cache status")
status_cols = st.columns(min(len(VPS_SHADOW_DB_FILES), 8))
for i, (name, (strat, sym)) in enumerate(SHADOW_DB_STRATEGY_MAP.items()):
    p = VPS_CACHE_DIR / name
    _abbr = "".join(w[0] for w in strat.split()) if strat != "?" else "?"
    col = status_cols[i % len(status_cols)]
    if p.exists():
        size_kb = round(p.stat().st_size / 1024, 1)
        age_min = int((datetime.now().timestamp() - p.stat().st_mtime) / 60)
        col.success(f"{_abbr} {sym} · {size_kb} KB · {age_min} min", icon="🟢")
    else:
        col.error(f"{_abbr} {sym} · not synced", icon="🔴")


if view_df.empty:
    st.info(
        f"No shadow data for **{sel_bot}**.  Click **⟳ Sync shadow DBs from VPS** "
        "in the sidebar, or pick a different bot."
    )
    st.stop()


# ── Summary helper ───────────────────────────────────────────────────────────
def _summarise(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return df
    closed_mask = df["exit_reason"].isin(["TP", "SL", "TIME_EXIT"])
    closed = df[closed_mask]
    open_df = df[df["exit_reason"] == "OPEN"]
    rows = []
    if group_col is None:
        iter_ = [("All", closed, open_df)]
    else:
        keys = sorted(set(df[group_col].dropna().astype(str).unique()))
        iter_ = [
            (k, closed[closed[group_col].astype(str) == k],
                 open_df[open_df[group_col].astype(str) == k])
            for k in keys
        ]
    for k, g, og in iter_:
        n = len(g)
        tp = int((g["exit_reason"] == "TP").sum())
        sl = int((g["exit_reason"] == "SL").sum())
        te = int((g["exit_reason"] == "TIME_EXIT").sum())
        wr_tp = round(tp / n * 100, 1) if n else 0.0
        avg_r = round(g["r_multiple"].mean(), 3) if n else None
        tot_r = round(g["r_multiple"].sum(), 2) if n else 0.0
        tot_pnl_1pct = round(g["pnl_at_1pct"].sum(), 2) if (n and "pnl_at_1pct" in g.columns) else 0.0
        rows.append({
            (group_col or "scope"): k,
            "open": len(og),
            "closed": n,
            "TP": tp,
            "SL": sl,
            "TIME_EXIT": te,
            "TP %": wr_tp,
            "avg R": avg_r,
            "total R": tot_r,
            "total $@1%": tot_pnl_1pct,
        })
    return pd.DataFrame(rows)


# ── Headline KPIs (respects bot filter) ──────────────────────────────────────
overall = _summarise(view_df)
k1, k2, k3, k4 = st.columns(4)
ov = overall.iloc[0]
k1.metric("Open shadows", int(ov["open"]))
k2.metric("Closed shadows", int(ov["closed"]))
k3.metric("Avg R", f"{ov['avg R']:+.3f}" if ov["avg R"] is not None else "—")
k4.metric("Total R", f"{ov['total R']:+.2f}")

st.markdown("---")


# ── Per-bot (only meaningful when not already filtered to one) ──────────────
if sel_bot == "All bots":
    st.subheader("Per bot")
    per_bot = _summarise(view_df, group_col="bot").rename(columns={"bot": "Bot"})
    st.dataframe(per_bot, use_container_width=True, hide_index=True)


# ── Per gate (the headline practice-vs-theory view) ──────────────────────────
st.subheader("By gate (`block_reason`) — *practice vs theory*")
st.caption(
    "Each row is a gate's cumulative live shadow performance.  Compare to "
    "the WFO claim that justified the gate in the strategy code."
)
per_reason = (
    _summarise(view_df, group_col="block_reason")
    .rename(columns={"block_reason": "Block reason"})
    .sort_values("closed", ascending=False)
)
st.dataframe(per_reason, use_container_width=True, hide_index=True)


# ── Cumulative R per gate (chart) ────────────────────────────────────────────
closed_chart = view_df[view_df["exit_reason"].isin(["TP", "SL", "TIME_EXIT"])].copy()
if not closed_chart.empty and "r_multiple" in closed_chart.columns:
    closed_chart = closed_chart.dropna(subset=["r_multiple", "closed_at_utc"])
    if not closed_chart.empty:
        closed_chart = closed_chart.sort_values("closed_at_utc")
        closed_chart["cum_R"] = (
            closed_chart.groupby("block_reason")["r_multiple"].cumsum()
        )
        fig = px.line(
            closed_chart, x="closed_at_utc", y="cum_R",
            color="block_reason",
            title=f"Cumulative R per gate — {sel_bot}",
            labels={"closed_at_utc": "closed at (UTC)",
                    "cum_R": "cumulative R", "block_reason": "gate"},
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ── Breakdown tabs (session / asset regime / BTC regime / MTF band) ──────────
st.subheader("Breakdowns")
st.caption(
    "Slice the shadow book by entry session, asset's own new5 regime, BTC's "
    "new5 regime at entry (asof from duckdb), and MTF score band.  Each row "
    "reports closed-trade stats only."
)

# Pre-compute MTF band
if "mtf_score" in view_df.columns:
    view_df = view_df.copy()
    view_df["mtf_band"] = pd.cut(
        view_df["mtf_score"],
        bins=[-0.001, 40, 55, 70, 85, 100],
        labels=["<40", "40-55", "55-70", "70-85", "85-100"],
    ).astype(str)

tab_sess, tab_arg, tab_btc, tab_mtf = st.tabs([
    "Session", "Asset regime", "BTC regime", "MTF band",
])
with tab_sess:
    if "session" in view_df.columns:
        st.dataframe(
            _summarise(view_df, group_col="session")
            .rename(columns={"session": "Session"})
            .sort_values("closed", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No `session` column in this shadow DB.")
with tab_arg:
    if "regime_gate" in view_df.columns:
        st.dataframe(
            _summarise(view_df, group_col="regime_gate")
            .rename(columns={"regime_gate": "Asset regime"})
            .sort_values("closed", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No `regime_gate` column in this shadow DB.")
with tab_btc:
    if "btc_regime" in view_df.columns:
        st.dataframe(
            _summarise(view_df, group_col="btc_regime")
            .rename(columns={"btc_regime": "BTC regime"})
            .sort_values("closed", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "BTC regime computed via `regime_classifier.classify_rule_based` on "
            "BTC 15m from `duckdb_data/trading_data.duckdb`, asof-tagged to each "
            "trade's `opened_at_utc` (strict no-lookahead)."
        )
    else:
        st.info("BTC regime tagging is unavailable (duckdb not loadable).")
with tab_mtf:
    if "mtf_band" in view_df.columns:
        st.dataframe(
            _summarise(view_df, group_col="mtf_band")
            .rename(columns={"mtf_band": "MTF band"})
            .sort_values("closed", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No `mtf_score` column in this shadow DB.")


# ── RR What-If (replay against klines, fixed 1R risk, vary reward) ──────────
st.markdown("---")
st.subheader("RR what-if — same SL, different rewards")
st.caption(
    "For each closed shadow trade, replay forward against the asset's 15m "
    "klines (duckdb) with **fixed 1R risk** (the original SL) and three "
    "candidate rewards (**1.5R / 2R / 3R**).  Whichever the price path hits "
    "first wins — SL → −1R, target → +reward, neither within 2688 bars "
    "(~28 days) → TIME_EXIT at the last close.  Lets you see whether your "
    "current TP setting is leaving money on the table for each bucket."
)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_klines(symbol: str) -> pd.DataFrame:
    try:
        import duckdb
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        df = con.execute(f"""
            SELECT timestamp, high, low, close
            FROM ohlcv_data
            WHERE symbol = '{symbol}' AND timeframe = '15m'
            ORDER BY timestamp
        """).fetchdf()
        con.close()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def _replay_one(klines: pd.DataFrame, opened_at, entry: float, sl: float,
                direction: str, target_r: float, max_bars: int = 2688):
    """Return (outcome_r, outcome_reason) for a hypothetical trade with the
    given SL and target_r (multiple of risk).  outcome_reason ∈ {'TP','SL','TIME_EXIT'}."""
    if klines.empty or pd.isna(opened_at) or pd.isna(entry) or pd.isna(sl):
        return None, None
    risk = abs(entry - sl)
    if risk <= 0:
        return None, None
    is_long = str(direction).upper() == "BUY"
    target = entry + target_r * risk if is_long else entry - target_r * risk
    # walk forward from the first bar STRICTLY AFTER opened_at
    idx = klines.index.searchsorted(opened_at, side="right")
    end = min(idx + max_bars, len(klines))
    if idx >= end:
        return None, None
    sub = klines.iloc[idx:end]
    highs = sub["high"].to_numpy()
    lows = sub["low"].to_numpy()
    for i in range(len(sub)):
        if is_long:
            tp_hit = highs[i] >= target
            sl_hit = lows[i] <= sl
        else:
            tp_hit = lows[i] <= target
            sl_hit = highs[i] >= sl
        # Conservative tie-break: if BOTH hit on the same bar, assume SL first
        # (worst case for the trader — matches the WFO simulator convention).
        if sl_hit and not tp_hit:
            return -1.0, "SL"
        if tp_hit and not sl_hit:
            return float(target_r), "TP"
        if tp_hit and sl_hit:
            return -1.0, "SL"
    return 0.0, "TIME_EXIT"  # didn't hit either within max_bars


@st.cache_data(ttl=1800, show_spinner=False)
def _rr_whatif(view_records: tuple, target_rs: tuple, max_bars: int = 2688) -> pd.DataFrame:
    """Compute outcomes per reward level for the supplied trades.

    view_records: tuple of (bot, symbol, opened_at_utc, direction, entry_price,
                            stop_loss, exit_reason) — hashable for caching.
    """
    rows = []
    # Lazy klines per symbol
    klines_cache: dict[str, pd.DataFrame] = {}
    for bot, sym, opened_at, direction, entry, sl, exit_reason in view_records:
        if exit_reason not in ("TP", "SL", "TIME_EXIT"):
            continue
        if sym not in klines_cache:
            klines_cache[sym] = _load_klines(sym)
        kl = klines_cache[sym]
        if kl.empty:
            continue
        opened_at_dt = pd.to_datetime(opened_at, utc=True)
        for tr in target_rs:
            r_val, reason = _replay_one(kl, opened_at_dt, entry, sl, direction, tr, max_bars)
            if r_val is None:
                continue
            rows.append({
                "bot": bot, "target_R": tr,
                "outcome_R": r_val, "outcome": reason,
            })
    return pd.DataFrame(rows)


target_rs = (1.5, 2.0, 3.0)
closed = view_df[view_df["exit_reason"].isin(["TP", "SL", "TIME_EXIT"])].copy()
records = []
for _, row in closed.iterrows():
    records.append((
        row.get("bot"), row.get("symbol"), row.get("opened_at_utc"),
        row.get("direction"), row.get("entry_price"),
        row.get("stop_loss"), row.get("exit_reason"),
    ))

if not records:
    st.info("No closed shadow trades yet for the selected bot — nothing to replay.")
else:
    with st.spinner(f"Replaying {len(records)} trades against duckdb klines…"):
        whatif = _rr_whatif(tuple(records), target_rs)

    if whatif.empty:
        st.warning(
            "Replay produced no rows.  Likely cause: the asset's klines aren't "
            "in `trading_data.duckdb` (only BTC/ETH/SOL/NQ/XRP/LTC/ADA + 6 majors are)."
        )
    else:
        # Aggregate per target_R
        agg = (
            whatif.groupby("target_R")
            .agg(
                n=("outcome_R", "count"),
                wins=("outcome", lambda x: (x == "TP").sum()),
                losses=("outcome", lambda x: (x == "SL").sum()),
                time_exits=("outcome", lambda x: (x == "TIME_EXIT").sum()),
                mean_R=("outcome_R", "mean"),
                total_R=("outcome_R", "sum"),
            )
            .round({"mean_R": 3, "total_R": 2})
        )
        agg["WR %"] = (agg["wins"] / agg["n"] * 100).round(1)
        agg = agg.reset_index()
        agg["target_R"] = agg["target_R"].map(lambda v: f"{v:g}R")
        agg = agg.rename(columns={"target_R": "Reward (R)"})
        # Also include the bot's ACTUAL outcome as a reference row
        actual = pd.DataFrame([{
            "Reward (R)": "actual",
            "n": int(closed["r_multiple"].notna().sum()),
            "wins": int((closed["exit_reason"] == "TP").sum()),
            "losses": int((closed["exit_reason"] == "SL").sum()),
            "time_exits": int((closed["exit_reason"] == "TIME_EXIT").sum()),
            "mean_R": round(closed["r_multiple"].mean(), 3) if not closed.empty else None,
            "total_R": round(closed["r_multiple"].sum(), 2) if not closed.empty else 0.0,
            "WR %": round((closed["exit_reason"] == "TP").mean() * 100, 1) if not closed.empty else 0.0,
        }])
        combined = pd.concat([actual, agg], ignore_index=True)[
            ["Reward (R)", "n", "wins", "losses", "time_exits", "WR %", "mean_R", "total_R"]
        ]
        st.dataframe(combined, use_container_width=True, hide_index=True)
        st.caption(
            "**Reading the table:** the **actual** row is what each bot actually "
            "booked.  The 1.5/2/3R rows are the simulated outcomes if you'd kept "
            "the same SL but moved TP — same trades, same entries, just a "
            "different target.  Higher reward earns more per win but trades "
            "fewer wins for more time-exits."
        )


# ── Recent shadows ───────────────────────────────────────────────────────────
with st.expander("Recent shadows (last 50)"):
    cols = [
        "bot", "opened_at_utc", "direction", "block_reason",
        "session", "regime_gate", "btc_regime", "mtf_score",
        "entry_price", "stop_loss", "take_profit",
        "exit_price", "exit_reason",
        "bars_held", "r_multiple", "pnl_at_1pct",
    ]
    have = [c for c in cols if c in view_df.columns]
    recent = view_df.sort_values("opened_at_utc", ascending=False).head(50).loc[:, have]
    st.dataframe(recent, use_container_width=True, hide_index=True)

st.caption(
    "Source: `Liquidity_Raid/<sym>_V2/<sym>_shadow_trades.db` on the VPS, "
    "synced into `dashboard/databases/`.  Filled by `shared/shadow_tracker.py` "
    "whenever a strategy gate rejects a signal."
)
