"""Page 23: Shadow Trades — practice vs theory, per gate.

Every signal the LR live path rejects via a gate is recorded as a paper
trade in a per-bot ``<sym>_shadow_trades.db``.  Each shadow carries a
``block_reason`` so we can confront live R-multiples per gate against the
WFO claim that justified it.

Gates currently shadowed:

* ``OFF_WINDOW`` — SOL signal generated outside the live 12-16 ET window.
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

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    SHADOW_DB_STRATEGY_MAP,
    VPS_CACHE_DIR,
    VPS_SHADOW_DB_FILES,
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
        for name, r in results.items():
            icon = "✅" if r.get("status") == "ok" else "❌"
            st.write(f"{icon} {name}: {r.get('status')}")
    st.divider()
    st.caption(
        "Shadow DBs live next to each bot's live trade DB on the VPS.  "
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
    # "Liquidity Raid" -> "LR", "Momentum Mastery" -> "MM" (initials of each word)
    _abbr = "".join(w[0] for w in strat.split()) if strat != "?" else "?"
    df["bot"] = f"{_abbr} {sym}"
    for c in ("opened_at_utc", "closed_at_utc"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


frames = [_load_one(name) for name in VPS_SHADOW_DB_FILES]
frames = [f for f in frames if not f.empty]
all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Cache status row ─────────────────────────────────────────────────────────
st.subheader("Cache status")
status_cols = st.columns(len(VPS_SHADOW_DB_FILES))
for i, (name, (strat, sym)) in enumerate(SHADOW_DB_STRATEGY_MAP.items()):
    p = VPS_CACHE_DIR / name
    _abbr = "".join(w[0] for w in strat.split()) if strat != "?" else "?"
    if p.exists():
        size_kb = round(p.stat().st_size / 1024, 1)
        age_min = int((datetime.now().timestamp() - p.stat().st_mtime) / 60)
        status_cols[i].success(
            f"{_abbr} {sym} · {size_kb} KB · {age_min} min old", icon="🟢"
        )
    else:
        status_cols[i].error(f"{_abbr} {sym} · not synced", icon="🔴")


if all_df.empty:
    st.info(
        "No shadow data cached yet.  Click **⟳ Sync shadow DBs from VPS** in "
        "the sidebar to pull the latest data."
    )
    st.stop()


# ── Summary helper ───────────────────────────────────────────────────────────
def _summarise(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """Aggregate shadow rows.  ``group_col`` of ``None`` returns one row."""
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
        tot_pnl_1pct = round(g["pnl_at_1pct"].sum(), 2) if n else 0.0
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


# ── Headline KPIs ────────────────────────────────────────────────────────────
overall = _summarise(all_df)
k1, k2, k3, k4 = st.columns(4)
ov = overall.iloc[0]
k1.metric("Open shadows", int(ov["open"]))
k2.metric("Closed shadows", int(ov["closed"]))
k3.metric("Avg R", f"{ov['avg R']:+.3f}" if ov["avg R"] is not None else "—")
k4.metric("Total R (live shadow)", f"{ov['total R']:+.2f}")

st.markdown("---")

# ── Per-bot ──────────────────────────────────────────────────────────────────
st.subheader("Per bot")
per_bot = _summarise(all_df, group_col="bot").rename(columns={"bot": "Bot"})
st.dataframe(per_bot, use_container_width=True, hide_index=True)


# ── Per gate (the headline practice-vs-theory view) ──────────────────────────
st.subheader("By gate (block_reason) — *practice vs theory*")
st.caption(
    "Each row is a gate's cumulative live shadow performance.  Compare to "
    "the WFO claim that justified the gate in the strategy code.  If "
    "`total R` here disagrees materially with the cited expectation over a "
    "few weeks of fresh data, the gate is the suspect."
)
per_reason = (
    _summarise(all_df, group_col="block_reason")
    .rename(columns={"block_reason": "Block reason"})
    .sort_values("closed", ascending=False)
)
st.dataframe(per_reason, use_container_width=True, hide_index=True)


# ── Per gate × bot ───────────────────────────────────────────────────────────
st.subheader("By gate × bot")
combo = all_df.copy()
combo["scope"] = combo["bot"].astype(str) + " · " + combo["block_reason"].astype(str)
per_combo = (
    _summarise(combo, group_col="scope")
    .sort_values("closed", ascending=False)
)
st.dataframe(per_combo, use_container_width=True, hide_index=True)


# ── Cumulative R per gate (chart) ────────────────────────────────────────────
closed_chart = all_df[all_df["exit_reason"].isin(["TP", "SL", "TIME_EXIT"])].copy()
if not closed_chart.empty and "r_multiple" in closed_chart.columns:
    closed_chart = closed_chart.dropna(subset=["r_multiple", "closed_at_utc"])
    if not closed_chart.empty:
        closed_chart = closed_chart.sort_values("closed_at_utc")
        closed_chart["cum_R"] = (
            closed_chart.groupby("block_reason")["r_multiple"].cumsum()
        )
        fig = px.line(
            closed_chart,
            x="closed_at_utc", y="cum_R",
            color="block_reason",
            title="Cumulative R per gate (live shadow)",
            labels={
                "closed_at_utc": "closed at (UTC)",
                "cum_R": "cumulative R",
                "block_reason": "gate",
            },
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ── Recent shadows ───────────────────────────────────────────────────────────
with st.expander("Recent shadows (last 50)"):
    recent = (
        all_df.sort_values("opened_at_utc", ascending=False)
        .head(50)
        .loc[:, [
            "bot", "opened_at_utc", "direction", "block_reason",
            "entry_price", "stop_loss", "take_profit",
            "exit_price", "exit_reason",
            "bars_held", "r_multiple", "pnl_at_1pct",
        ]]
    )
    st.dataframe(recent, use_container_width=True, hide_index=True)

st.caption(
    "Source: `Liquidity_Raid/<sym>_V2/<sym>_shadow_trades.db` on the VPS, "
    "synced into `dashboard/databases/`.  Filled by "
    "`the internal strategy core` whenever a strategy gate "
    "rejects a signal (deployed 2026-05-20)."
)
