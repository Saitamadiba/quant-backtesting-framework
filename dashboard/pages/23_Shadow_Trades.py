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
from data.shadow_normalisers import normalise_lrr as _normalise_lrr  # noqa: F401
from data.shadow_normalisers import normalise_manual as _normalise_manual  # noqa: F401


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
# Normalisers live in `data/shadow_normalisers.py` so the schema-bridge
# contracts can be regression-tested independently of the page.

def _load_one(local_name: str) -> pd.DataFrame:
    p = VPS_CACHE_DIR / local_name
    if not p.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(p) as conn:
            # Per-file branch: different DBs use different table names.
            if local_name == "lrr_shadow_trades.db":
                df = pd.read_sql_query("SELECT * FROM lrr_signals", conn)
                df = _normalise_lrr(df)
            elif local_name == "manual_trades.db":
                df = pd.read_sql_query("SELECT * FROM manual_trades", conn)
                df = _normalise_manual(df)
            else:
                df = pd.read_sql_query("SELECT * FROM shadow_trades", conn)
    except Exception as e:
        st.warning(f"Could not read {local_name}: {e}")
        return pd.DataFrame()
    if df.empty:
        return df
    strat, sym = SHADOW_DB_STRATEGY_MAP.get(local_name, ("?", "?"))
    df["strategy"] = strat
    _abbr = "".join(w[0] for w in strat.split()) if strat != "?" else "?"
    if sym == "MULTI" and "asset" in df.columns:
        # Multi-asset DB: per-row symbol. For ``manual_trades`` the bot
        # label also reflects the strategy_tag the sync auto-tagger chose
        # (e.g. "M FVG BTC", "M LR BTC", "M ad-hoc BTC") so the bot filter
        # in the sidebar can drill down to the source strategy or to
        # truly ad-hoc trades.
        df["symbol"] = df["asset"].astype(str)
        if local_name == "manual_trades.db" and "strategy_tag" in df.columns:
            def _mk_bot(row: pd.Series) -> str:
                tag = str(row.get("strategy_tag") or "ad-hoc")
                # Strip trailing asset tail from the tag if present
                # (e.g. "FVG BTC" → "FVG"); keeps the bot label compact.
                tag_short = tag.replace(f" {row['asset']}", "").strip()
                return f"{_abbr} {tag_short} {row['asset']}".strip()
            df["bot"] = df.apply(_mk_bot, axis=1)
        else:
            df["bot"] = df["symbol"].apply(lambda a: f"{_abbr} {a}")
    else:
        df["symbol"] = sym
        df["bot"] = f"{_abbr} {sym}"
    for c in ("opened_at_utc", "closed_at_utc"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def _collapse_setups(df: pd.DataFrame, gap_min: float = 60.0) -> pd.DataFrame:
    """Collapse re-detections of one swept level into a single DISTINCT SETUP.

    The live LR path re-emits the same unconsumed sweep on every 15m bar (and
    from scratch after each bot restart), so historically one setup wrote 4-12
    near-identical rows — inflating every per-gate ``n`` 4-12x and, where a
    setup won, dragging mean-R toward that one outcome counted many times (BTC
    raw +0.217 → +-0.217 once deduped).  This keeps one row per real setup so
    R / WR / n reflect distinct trade ideas, not poll cadence.

    Resolution order per LR row:
      1. ``setup_id`` (written by the dedup_shadow_setups backfill) when present.
      2. else chain ``(symbol, block_reason, sweep_type, direction)`` on time —
         consecutive rows whose gap <= ``gap_min`` are the same setup.
    Rows the post-fix tracker writes are already one-per-setup → no-op.  LRR
    rows (own UNIQUE-constraint dedup, no ``block_reason``) pass straight through.
    """
    if df.empty or "opened_at_utc" not in df.columns:
        return df
    df = df.copy()
    has_reason = "block_reason" in df.columns
    is_lr = df["block_reason"].notna() if has_reason else pd.Series(False, index=df.index)
    lr, rest = df[is_lr].copy(), df[~is_lr]
    if lr.empty:
        return df
    lr = lr.sort_values("opened_at_utc")
    keys = [k for k in ("symbol", "block_reason", "sweep_type", "direction")
            if k in lr.columns]
    # 1) backfilled setup_id (per-DB int) — exact agreement with the VPS DBs.
    if "setup_id" in lr.columns and lr["setup_id"].notna().any():
        lr["_setup"] = lr["setup_id"]
        # rows the backfill missed (NULL — e.g. written post-backfill) fall back
        # to a per-row unique id so each counts as its own setup.
        _miss = lr["_setup"].isna()
        if _miss.any() and "shadow_id" in lr.columns:
            lr.loc[_miss, "_setup"] = "sid:" + lr.loc[_miss, "shadow_id"].astype(str)
        grp = ["symbol", "_setup"] if "symbol" in lr.columns else ["_setup"]
    else:
        # 2) gap-cluster fallback (stale local copies that pre-date the backfill).
        # Sort by key then time so same-key rows are contiguous; a new setup
        # starts whenever the key changes OR the gap to the prior same-key row
        # exceeds gap_min.  cumsum yields a globally-unique setup id.
        lr = lr.sort_values(keys + ["opened_at_utc"])
        _same_key = (lr[keys] == lr[keys].shift()).all(axis=1)
        _dt = lr["opened_at_utc"].diff().dt.total_seconds() / 60.0
        lr["_setup"] = (~_same_key | _dt.isna() | (_dt > gap_min)).cumsum()
        grp = ["_setup"]
    sort_col = "shadow_id" if "shadow_id" in lr.columns else "opened_at_utc"
    lr = (lr.sort_values(sort_col)
            .drop_duplicates(subset=grp, keep="first")
            .drop(columns=["_setup"]))
    return pd.concat([lr, rest], ignore_index=True)


frames = [_load_one(name) for name in VPS_SHADOW_DB_FILES]
frames = [f for f in frames if not f.empty]
all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── Distinct-setup collapse (default ON) ─────────────────────────────────────
# One swept level == one setup.  Off shows every raw re-detection (inflated n).
with st.sidebar:
    st.divider()
    _collapse = st.checkbox(
        "Collapse re-detections → 1 row per setup", value=True,
        help="The live LR path re-logs the same unconsumed sweep every 15m bar "
             "and after every restart. ON counts each setup once (trustworthy "
             "R / WR / n); OFF shows the raw, duplicate-inflated rows.",
    )
if _collapse and not all_df.empty:
    _n_raw = len(all_df)
    all_df = _collapse_setups(all_df)
    _n_dropped = _n_raw - len(all_df)
    if _n_dropped > 0:
        st.sidebar.caption(
            f"Collapsed {_n_raw} raw rows → {len(all_df)} setups "
            f"({_n_dropped} re-detections hidden)."
        )


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
    # Sentinel "MULTI" = single DB holds multiple assets. Render as "(multi)"
    # in the cache-status badge; per-asset rows in the body still split correctly.
    _sym_label = "(multi)" if sym == "MULTI" else sym
    col = status_cols[i % len(status_cols)]
    if p.exists():
        size_kb = round(p.stat().st_size / 1024, 1)
        age_min = int((datetime.now().timestamp() - p.stat().st_mtime) / 60)
        col.success(f"{_abbr} {_sym_label} · {size_kb} KB · {age_min} min", icon="🟢")
    else:
        col.error(f"{_abbr} {_sym_label} · not synced", icon="🔴")


if view_df.empty:
    st.info(
        f"No shadow data for **{sel_bot}**.  Click **⟳ Sync shadow DBs from VPS** "
        "in the sidebar, or pick a different bot."
    )
    st.stop()


# ── Data quality panel ───────────────────────────────────────────────────────
# Per-DB row count + column population %. Shows whether None cells are due
# to (a) the DB being empty (sync failure / bot silent), (b) the column not
# existing for that schema, or (c) historical rows pre-dating the column.

with st.expander("🔍 Data quality — per-source row counts + column population"):
    _qcols = ["block_reason", "session", "regime_gate", "mtf_score",
              "exit_reason", "r_multiple", "pnl_at_1pct"]
    _q_rows = []
    for _name, (_strat, _sym) in SHADOW_DB_STRATEGY_MAP.items():
        _p = VPS_CACHE_DIR / _name
        _row = {"source": _name, "strategy": _strat,
                "synced": "yes" if _p.exists() else "no",
                "rows": 0}
        if _p.exists():
            try:
                with sqlite3.connect(_p) as _c:
                    _tbl = ("lrr_signals" if _name == "lrr_shadow_trades.db"
                            else "manual_trades" if _name == "manual_trades.db"
                            else "shadow_trades")
                    _n = _c.execute(f"SELECT COUNT(*) FROM {_tbl}").fetchone()[0]
                    _row["rows"] = _n
                    _info = {r[1] for r in
                             _c.execute(f"PRAGMA table_info({_tbl})").fetchall()}
                    for _col in _qcols:
                        if _col not in _info:
                            _row[_col] = "n/a (column absent)"
                        elif _n == 0:
                            _row[_col] = "—"
                        else:
                            _filled = _c.execute(
                                f"SELECT COUNT({_col}) FROM {_tbl}"
                            ).fetchone()[0]
                            _row[_col] = f"{int(100*_filled/_n)}%"
            except Exception as _e:
                _row["rows"] = f"read error: {_e}"
        _q_rows.append(_row)
    _qdf = pd.DataFrame(_q_rows)
    st.dataframe(_qdf, use_container_width=True, hide_index=True)
    st.caption(
        "**Reading the panel:** `n/a (column absent)` = the column doesn't "
        "exist in that schema (e.g. LRR has `regime` not `regime_gate`, "
        "manual_trades has none of the gate-related fields). A percentage "
        "= column exists but only that fraction of rows have it filled "
        "(pre-2026-05-23 LR rows pre-date the regime-gate logger and show "
        "NaN by design). `—` = source is empty (0 rows). `synced=no` = "
        "click ⟳ Sync in the sidebar."
    )


# ── Summary helper ───────────────────────────────────────────────────────────
def _summarise(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return df
    closed_mask = df["exit_reason"].isin(["TP", "SL", "TIME_EXIT"])
    closed = df[closed_mask]
    open_df = df[df["exit_reason"] == "OPEN"]
    rows = []
    # group_col may be missing entirely (e.g. LRR-only filter has no
    # block_reason) — surface that to the caller rather than silently
    # returning a malformed frame.
    if group_col is not None and group_col not in df.columns:
        return pd.DataFrame()
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
        # Use NaN (rendered as "—" by Streamlit) instead of 0 when there are
        # no closed rows — avoids the misleading "0.000 / 0.00" cells.
        _has_r = n and "r_multiple" in g.columns and g["r_multiple"].notna().any()
        avg_r = round(g["r_multiple"].mean(), 3) if _has_r else float("nan")
        tot_r = round(g["r_multiple"].sum(), 2) if _has_r else float("nan")
        _has_pnl = (n and "pnl_at_1pct" in g.columns
                    and g["pnl_at_1pct"].notna().any())
        tot_pnl_1pct = (round(g["pnl_at_1pct"].sum(), 2)
                        if _has_pnl else float("nan"))
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


# (internal forward-shadow tracker; sanitised from public commit history)

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


# ── Shadow trades browser — full pagination, per-column filters, reorder ────
st.subheader("Shadow trades — full browser")
st.caption(
    "Every row in the synced shadow DBs, grouped by strategy. **All rows** "
    "are shown (no row cap). Use the **🔍 Column filters** expander to "
    "narrow down by `bot` / `direction` / `exit_reason` / `block_reason` / "
    "anything categorical. Use the **📋 Columns to show** picker to control "
    "which columns appear AND the order (selection order = display order). "
    "Click any column header for native sort."
)

# Column lists per strategy — derived from the actual writer schema.
# Each list is ordered by importance for that strategy's reader.
_COMMON_HEAD = ["bot", "opened_at_utc", "direction", "exit_reason",
                "r_multiple"]
_COMMON_TAIL = ["entry_price", "stop_loss", "take_profit", "exit_price",
                "bars_held", "btc_regime"]

_STRATEGY_COLS = {
    "Liquidity Raid": _COMMON_HEAD + [
        "block_reason", "session", "regime_gate", "mtf_score",
        "pnl_at_1pct",
    ] + _COMMON_TAIL,
    "LR Paper":       _COMMON_HEAD + [
        "block_reason", "session", "regime_gate", "mtf_score",
        "pnl_at_1pct",
    ] + _COMMON_TAIL,
    "Momentum Mastery": _COMMON_HEAD + [
        "block_reason", "session", "regime_gate", "mtf_score",
        "pnl_at_1pct",
    ] + _COMMON_TAIL,
    "LRR Shadow": _COMMON_HEAD + [
        "vol_ratio", "wick_ratio", "prior_move_pct", "hour_et",
        "is_best_combo", "mtf_score", "regime",
        "dvol_band", "vrp_bucket", "htf_agree", "htf_trend",
        "liquidity_class", "cross_count", "ml_p", "ml_pass",
        "entry_price", "sl", "tp", "exit_price", "bars_held",
        "btc_regime",
    ],
    "Manual":     _COMMON_HEAD + [
        "symbol", "strategy_tag",
        "entry_price", "exit_price", "btc_regime",
    ],
}

# Display friendly hyphens for NaN / None in dataframe cells.
def _render(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Use object dtype + fillna() so visible cells render as "—" rather than
    # NaN/None. Numeric columns keep their numeric dtype for the rendered
    # frame because Streamlit handles NaN -> blank gracefully in the live UI;
    # we only stringify text-like columns here.
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].where(out[c].notna(), "—")
    return out


_strategies_present = (view_df["strategy"].dropna().unique().tolist()
                       if "strategy" in view_df.columns else [])
_tab_labels = [s for s in
               ("Liquidity Raid", "LR Paper", "Momentum Mastery",
                "LRR Shadow", "Manual")
               if s in _strategies_present]


from dashboard.data.shadow_normalisers import candidate_filter_columns as \
    _candidate_filter_columns  # noqa: E402  (defined here to keep tab scope)


def _render_strategy_table(strat: str, sub: pd.DataFrame) -> None:
    """Render the strategy's tab content: filters + column picker + table."""
    if sub.empty:
        st.info(f"No rows for {strat}.")
        return

    # Strategy-specific column list, intersected with what's actually present
    # AND that has at least one non-NaN value (honest-NaN policy from
    # 2026-06-06 cleanup — see internal note).
    base_cols = [c for c in _STRATEGY_COLS.get(strat, _COMMON_HEAD)
                 if c in sub.columns and sub[c].notna().any()]
    if not base_cols:
        st.info(f"All columns are 100% empty for {strat}.")
        return

    # ── 1. Per-column filters (expander, default collapsed) ────────────────
    filter_cols = _candidate_filter_columns(sub, base_cols)
    _filt_key_prefix = f"sb_v2_{strat.replace(' ', '_')}"
    with st.expander(f"🔍 Column filters ({len(filter_cols)} available)",
                     expanded=False):
        if not filter_cols:
            st.caption("No filterable columns in this strategy's view.")
        else:
            st.caption(
                "Each filter starts with **all values selected** (no filter "
                "applied). Deselect values to narrow down. "
                "Filters are AND-combined across columns. Click "
                "**Reset filters** to start over."
            )
            # Reset button — clears every per-column session_state key for
            # this tab. Streamlit re-runs the script after a button click so
            # the multiselects below pick up the cleared keys.
            if st.button("Reset filters", key=f"{_filt_key_prefix}_reset"):
                for c in filter_cols:
                    _k = f"{_filt_key_prefix}_filt_{c}"
                    if _k in st.session_state:
                        del st.session_state[_k]
            # Layout filters in a 3-column grid so a strategy with many
            # categorical columns (LRR has ~10) stays scannable.
            ncols = 3
            for i in range(0, len(filter_cols), ncols):
                row = st.columns(ncols)
                for j, col in enumerate(filter_cols[i:i+ncols]):
                    with row[j]:
                        # Use string conversion so multiselect can deal with
                        # mixed-dtype columns (NaN + int + str).
                        opts = sorted(
                            sub[col].dropna().astype(str).unique().tolist()
                        )
                        sel = st.multiselect(
                            col, opts, default=opts,
                            key=f"{_filt_key_prefix}_filt_{col}",
                        )
                        if sel and set(sel) != set(opts):
                            sub = sub[sub[col].astype(str).isin(sel)]

    if sub.empty:
        st.warning("No rows match the current filter combination.")
        return

    # ── 2. Column picker — controls visibility AND order ────────────────────
    # Streamlit's multiselect returns options in the order the user selects
    # them. We treat that as the column display order. Default = base_cols
    # which is the curated importance order from _STRATEGY_COLS.
    st.markdown(
        "**📋 Columns to show** — selection order = display order. "
        "Deselect any column to hide it; reorder by deselecting then re-selecting."
    )
    show_cols = st.multiselect(
        "Columns to show",
        options=base_cols,
        default=base_cols,
        key=f"{_filt_key_prefix}_cols",
        label_visibility="collapsed",
    )
    if not show_cols:
        st.warning(
            "Select at least one column to display "
            "(or use the legend above to add columns)."
        )
        return

    # ── 3. Sort key — default by opened_at_utc desc; user picks otherwise ──
    sort_options = ["(no sort — preserve insertion order)"] + show_cols
    _default_sort = ("opened_at_utc" if "opened_at_utc" in show_cols
                     else sort_options[0])
    sort_col = st.selectbox(
        "Sort by", sort_options,
        index=sort_options.index(_default_sort),
        key=f"{_filt_key_prefix}_sortcol",
    )
    sort_asc = st.checkbox(
        "Ascending", value=False,
        key=f"{_filt_key_prefix}_sortasc",
        help="(Or click the column header in the table for native sort.)",
    )

    # ── 4. Final assembly — ALL ROWS (no head() cap) ────────────────────────
    table = sub.copy()
    if sort_col != "(no sort — preserve insertion order)":
        table = table.sort_values(sort_col, ascending=sort_asc,
                                   na_position="last")
    table = table.loc[:, show_cols]

    st.caption(
        f"**Showing {len(table):,} rows.** "
        f"Click column headers for native sort. "
        f"Drag column edges to resize. Sort/filter/reorder is persisted "
        f"in the Streamlit session — refresh resets to defaults."
    )
    # Tall height so the user can scroll through hundreds of rows without
    # the page jumping. Streamlit handles internal virtualisation.
    st.dataframe(
        _render(table),
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # ── 5. Strategy-specific caption — honest-NaN explainer ─────────────────
    if strat == "LRR Shadow":
        st.caption(
            "LRR rows have no `block_reason` / `session` / `pnl_at_1pct` "
            "columns — the LRR scanner doesn't apply a gate, doesn't "
            "track a session label, and the DB stores R-only outcomes."
        )
    elif strat == "Manual":
        st.caption(
            "Manual trades have no `block_reason` (nothing rejected "
            "them), no `session` (not enforced), no `r_multiple` "
            "(manual trades don't carry an SL/TP in the broker DB)."
        )
    elif strat in ("Liquidity Raid", "LR Paper"):
        _rg_pct = int(100 * sub["regime_gate"].notna().mean()) \
                  if "regime_gate" in sub.columns else 0
        st.caption(
            f"`regime_gate` is populated on {_rg_pct}% of {strat} rows. "
            "Pre-2026-05-23 rows pre-date the regime-gate logger "
            "(written by `the internal strategy core`) and "
            "show NaN by design."
        )


if _tab_labels:
    _tabs = st.tabs([f"{s} ({len(view_df[view_df['strategy']==s])})"
                    for s in _tab_labels])
    for tab, strat in zip(_tabs, _tab_labels):
        with tab:
            _render_strategy_table(strat,
                                    view_df[view_df["strategy"] == strat])
else:
    st.info("No shadow rows for the current filter.")

st.caption(
    "Sources: `Liquidity_Raid/<sym>_V2/<sym>_shadow_trades.db` (LR/LR Paper) — "
    "`Momentum_Mastery/<sym>/<sym>_shadow_trades.db` (MM) — "
    "`HyroTrader/lrr_shadow_trades.db` (LRR) — "
    "`HyroTrader/manual_trades.db` (Manual). "
    "All synced into `dashboard/databases/` on **⟳ Sync** above."
)
