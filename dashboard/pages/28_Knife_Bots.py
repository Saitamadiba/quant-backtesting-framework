"""Page 28: Knife Bots — the falling-knife fade, its three arms, and its metrics.

The "knife" catches an over-extended liquidity sweep — an 8-bar price extreme
that spikes through a level — and fades it back, betting the spike is a
stop-run that snaps back (catching the falling knife by the handle, not the
blade). This page does three things the user asked for:

  1. Summarises the performance of the three live arms — the ORIGINAL
     forward-shadow detector, the MAKER funded arm, and the TAKER funded arm.
  2. Breaks down the strategy + every microstructure metric in plain English
     with a metaphor a non-quant could repeat.
  3. Tells the honest story per metric: why we use it, the shortfall we hit,
     and the solution we found (L2 absorption, spoof/replenish, OI, etc.).

Live numbers come from the VPS knife DBs (Sync button). When those aren't
synced, the page falls back to the local offline research corpus (the frozen
spec + the 947-episode forward scout) so the strategy story always renders.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    VPS_CACHE_DIR,
    VPS_KNIFE_DB_FILES,
    KNIFE_SPEC_PATH,
    KNIFE_SCOUT_PARQUET,
    KNIFE_EPISODES_PARQUET,
    KNIFE_FROZEN_SCORE_CUT as CUT,
)
from data.vps_sync import sync_single_file

st.set_page_config(page_title="Knife Bots", page_icon="🔪", layout="wide")
st.title("🔪 Knife Bots — fading the falling knife")
st.caption(
    "An over-extended liquidity sweep (an 8-bar price extreme that spikes "
    "*through* a level) is a stop-run that often snaps back. The knife fades it. "
    "This page summarises the three live arms and explains every metric in "
    "plain English — the precise number first, the picture as a chaser."
)


# ══════════════════════════════════════════════════════════════════════════════
#  Loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_spec() -> dict:
    try:
        return json.loads(KNIFE_SPEC_PATH.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_scout() -> pd.DataFrame:
    """The local 947-episode forward scout (BTC/ETH/SOL, May–Jun 2026, all OOS).
    Carries v1_score + r_net — the cleanest local proof the SELECTION works."""
    try:
        return pd.read_parquet(KNIFE_SCOUT_PARQUET)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_episodes_all() -> pd.DataFrame:
    """The full knife detector population across ALL 12 assets, scored by the
    frozen model (feature_lab/score_knife_episodes.py). Carries is_holdout so the
    browser can separate dev (in-sample) from holdout (honest OOS)."""
    try:
        return pd.read_parquet(KNIFE_EPISODES_PARQUET)
    except Exception:
        return pd.DataFrame()


def load_knife_db(local_name: str) -> pd.DataFrame:
    """Read a synced knife DB from the dashboard cache; '' if not synced yet.
    Schema-tolerant: tries the known table, then any table with rows."""
    p = VPS_CACHE_DIR / local_name
    if not p.exists():
        return pd.DataFrame()
    table = "episodes" if "shadow" in local_name else "funded_trades"
    try:
        with sqlite3.connect(p) as conn:
            names = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()
            tbl = table if table in names else (names[0] if names else None)
            if tbl is None:
                return pd.DataFrame()
            return pd.read_sql_query(f"SELECT * FROM '{tbl}'", conn)
    except Exception as e:
        st.warning(f"Could not read {local_name}: {e}")
        return pd.DataFrame()


def _col(df: pd.DataFrame, *cands):
    """First column present from candidates, else None (schema tolerance)."""
    for c in cands:
        if c in df.columns:
            return c
    return None


def _et_session(h) -> str:
    """ET hour → trading session bucket (crypto trades 24/7, so 4 buckets)."""
    if pd.isna(h):
        return "n/a"
    h = int(h)
    if 8 <= h < 16:
        return "New York"
    if 3 <= h < 8:
        return "London"
    if 16 <= h < 19:
        return "NY-Late"
    return "Asian"  # 19–23, 0–2


def normalise_knife(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Bridge the 3 knife schemas (scout/shadow episodes/funded) into one frame.

    Mirrors the shadow-page normaliser idea: each source uses different column
    names, so map them to a common contract the breakdown machinery can share.
    """
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    a = _col(df, "asset", "symbol")
    out["asset"] = df[a].astype(str).str.replace("USDT", "", regex=False) if a else "?"
    d = _col(df, "direction")
    out["direction"] = (
        df[d].astype(str).str.upper().map(
            lambda x: "LONG" if x in ("LONG", "BUY", "1", "1.0") else "SHORT")
        if d else "?"
    )
    rc = _col(df, "r_net", "r_multiple")
    out["r"] = pd.to_numeric(df[rc], errors="coerce") if rc else np.nan
    sc = _col(df, "v1_score", "score", "knife_score")
    out["score"] = pd.to_numeric(df[sc], errors="coerce") if sc else np.nan
    fc = _col(df, "favored")
    if fc is not None:
        out["favored"] = pd.to_numeric(df[fc], errors="coerce").fillna(0).astype(bool)
    elif sc is not None:
        out["favored"] = out["score"] >= CUT
    else:
        out["favored"] = False
    tc = _col(df, "break_time", "ts_break", "fill_ts", "filled_at_utc",
              "placed_at_utc", "entry_ts", "detected_at_utc")
    out["ts"] = pd.to_datetime(df[tc], errors="coerce", utc=True) if tc else pd.NaT
    hc = _col(df, "hour_et")
    if hc is not None:
        out["hour_et"] = pd.to_numeric(df[hc], errors="coerce")
    elif tc is not None:
        out["hour_et"] = out["ts"].dt.tz_convert("America/New_York").dt.hour
    else:
        out["hour_et"] = np.nan
    out["session"] = out["hour_et"].map(_et_session)
    rg = _col(df, "regime5", "regime")
    out["regime"] = df[rg].astype(str) if rg else "n/a"
    dv = _col(df, "dvol_band")
    out["dvol_band"] = df[dv].astype(str) if dv else "n/a"
    er = _col(df, "exit_reason")
    out["exit_reason"] = df[er].astype(str) if er else "n/a"
    bh = _col(df, "bars_held")
    out["bars_held"] = pd.to_numeric(df[bh], errors="coerce") if bh else np.nan
    pc = _col(df, "pnl_usd")
    out["pnl_usd"] = pd.to_numeric(df[pc], errors="coerce") if pc else np.nan
    ho = _col(df, "is_holdout")
    out["is_holdout"] = pd.to_numeric(df[ho], errors="coerce") if ho else np.nan
    out["source"] = source
    return out


def summarise_knife(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Grouped stats table: n, favored, WR, avg/total R, avg bars — the knife
    analog of the shadow page's per-gate `_summarise`."""
    if df.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> dict:
        res = g.dropna(subset=["r"])
        n = len(res)
        wins = int((res["r"] > 0).sum()) if n else 0
        has_pnl = "pnl_usd" in g.columns and g["pnl_usd"].notna().any()
        return {
            "n": len(g),
            "resolved": n,
            "favored": int(g["favored"].sum()),
            "WR %": round(100 * wins / n, 1) if n else float("nan"),
            "avg R": round(res["r"].mean(), 3) if n else float("nan"),
            "total R": round(res["r"].sum(), 2) if n else float("nan"),
            "avg bars": round(res["bars_held"].mean(), 0)
                        if n and res["bars_held"].notna().any() else float("nan"),
            "net $": round(g["pnl_usd"].sum(), 0) if has_pnl else float("nan"),
        }

    if group_col is None:
        row = _agg(df)
        row = {"scope": "All", **row}
        return pd.DataFrame([row])
    rows = []
    for k, g in df.groupby(group_col):
        rows.append({group_col: str(k), **_agg(g)})
    res = pd.DataFrame(rows)
    # drop net $ column when nothing has it (offline corpus / shadow)
    if res["net $"].isna().all():
        res = res.drop(columns=["net $"])
    return res.sort_values("total R", ascending=False)


def working_frame(source_key: str) -> pd.DataFrame:
    """Load + normalise one data source into the common frame."""
    if source_key == "corpus":
        return normalise_knife(load_episodes_all(), "Offline corpus")
    db = {"shadow": "knife_shadow.db", "maker": "knife_funded_maker.db",
          "taker": "knife_funded_taker.db"}.get(source_key)
    return normalise_knife(load_knife_db(db), source_key) if db else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar — sync
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.subheader("Sync")
    if st.button("⟳ Sync knife DBs from VPS", use_container_width=True):
        res = {}
        with st.spinner("Syncing knife DBs…"):
            for ln, rp in VPS_KNIFE_DB_FILES.items():
                res[ln] = sync_single_file(ln, rp)
        ok = sum(1 for r in res.values() if r.get("status") == "ok")
        st.success(f"Synced {ok}/{len(res)}")
        for n, r in res.items():
            if r.get("status") != "ok":
                st.write(f"❌ {n}: {r.get('status')} {r.get('error', '')}")
        st.cache_data.clear()
    st.caption(
        "Knife DBs live next to the bots on the VPS "
        "(`HyroTrader/knife_*.db`). Sync pulls the latest snapshot into "
        "`dashboard/databases/`. Until then, the page shows the offline "
        "research corpus."
    )

spec = load_spec()
episodes = load_episodes_all()

# Frozen-model headline numbers (from spec + dev report), shown everywhere.
sel = spec.get("selection", {})
DEV_AUC = sel.get("pooled_auc", 0.7376)
DEV_MR_SEL = sel.get("pooled_mr_sel", 0.459)
DEV_N_SEL = sel.get("pooled_n_sel", 430)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — performance of the three arms
# ══════════════════════════════════════════════════════════════════════════════
st.header("1 · The three arms")
st.markdown(
    "One detector, three ways to actually take the trade. The detector's *edge "
    "is selection* — it loses on every break taken blindly and only makes money "
    "on the slice it flags **favored** (frozen score ≥ "
    f"`{CUT:.3f}`). Think of it as a metal detector on a beach: useless if you "
    "dig every spot, valuable only where it pings."
)


def episodes_kpis(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    rc = _col(df, "r_net", "r_multiple")
    fav = _col(df, "favored")
    resolved = df[df[rc].notna()] if rc else df.iloc[0:0]
    out = {"n": len(df), "resolved": len(resolved)}
    if fav:
        out["favored"] = int(pd.to_numeric(df[fav], errors="coerce").fillna(0).sum())
    if rc and len(resolved):
        out["wr"] = float((resolved[rc] > 0).mean())
        out["mr"] = float(resolved[rc].mean())
        out["tot"] = float(resolved[rc].sum())
        if fav:
            fr = resolved[pd.to_numeric(resolved[fav], errors="coerce") == 1]
            out["mr_fav"] = float(fr[rc].mean()) if len(fr) else None
    return out


def funded_kpis(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None
    rc = _col(df, "r_multiple", "r_net")
    pnl = _col(df, "pnl_usd")
    filled = _col(df, "filled_at_utc")
    fav = _col(df, "favored")
    closed = df[df[rc].notna()] if rc else df.iloc[0:0]
    out = {"n": len(df)}
    if filled:
        out["fills"] = int(df[filled].notna().sum())
    if fav:
        out["favored"] = int(pd.to_numeric(df[fav], errors="coerce").fillna(0).sum())
    out["closed"] = len(closed)
    if rc and len(closed):
        out["wr"] = float((closed[rc] > 0).mean())
        out["mr"] = float(closed[rc].mean())
        out["tot"] = float(closed[rc].sum())
    if pnl and df[pnl].notna().any():
        out["pnl"] = float(pd.to_numeric(df[pnl], errors="coerce").sum())
    return out


def render_arm(title, icon, kpis, offline_note):
    st.subheader(f"{icon} {title}")
    if not kpis:
        st.info(offline_note)
        return
    cols = st.columns(6)
    cols[0].metric("Signals", f"{kpis.get('n', 0):,}")
    if "fills" in kpis:
        cols[1].metric("Fills", f"{kpis['fills']:,}")
    elif "resolved" in kpis:
        cols[1].metric("Resolved", f"{kpis['resolved']:,}")
    cols[2].metric("Favored", f"{kpis.get('favored', 0):,}",
                   help=f"Frozen score ≥ {CUT:.3f} — the slice the model would take.")
    if "wr" in kpis:
        cols[3].metric("Win rate", f"{kpis['wr']:.1%}")
        cols[4].metric("Mean R", f"{kpis['mr']:+.3f}",
                       help="Per-trade R-multiple under the maker cost model.")
        last = f"${kpis['pnl']:,.0f}" if "pnl" in kpis else f"{kpis.get('tot', 0):+.1f}R"
        cols[5].metric("Net $ / R", last)
        if kpis.get("mr_fav") is not None:
            st.caption(f"Favored-only mean R: **{kpis['mr_fav']:+.3f}** "
                       "(the slice the gate would actually keep).")
    else:
        cols[3].metric("Closed", f"{kpis.get('closed', 0):,}")
        st.caption("No resolved outcomes yet — fills still open.")


shadow_df = load_knife_db("knife_shadow.db")
maker_df = load_knife_db("knife_funded_maker.db")
taker_df = load_knife_db("knife_funded_taker.db")

t_orig, t_maker, t_taker = st.tabs(
    ["① Original (forward-shadow)", "② Maker arm (funded)", "③ Taker arm (funded)"]
)
with t_orig:
    render_arm(
        "Original detector — read-only forward shadow", "🛰️",
        episodes_kpis(shadow_df),
        "Not synced. **Offline benchmark (frozen dev walk-forward):** the "
        f"selection lifts a take-everything **−0.63R** baseline to **{DEV_MR_SEL:+.3f}R** "
        f"on the favored top-decile (n={DEV_N_SEL}), AUC **{DEV_AUC:.3f}**, and "
        "cleared its holdout at **+0.36R (P≈0.9997)**. Sync to see live shadow fills.",
    )
with t_maker:
    render_arm(
        "Maker arm — post-only limit at the level", "🅼",
        funded_kpis(maker_df),
        "Not synced. **Offline reality (live maker bridge, 2026-06-14):** "
        "**−$3,569 / 17% WR / 0-of-27 favored captured.** The selection is real "
        "but the *fill* broke it — a passive limit fills the slow losers and the "
        "fast favored reverts gap straight through it. See §4 for the fix menu.",
    )
with t_taker:
    render_arm(
        "Taker arm — market at the break", "🆃",
        funded_kpis(taker_df),
        "Not synced. **Offline reality:** a market entry pays ~**0.46R** round-trip "
        "fee on a 0.5×ATR stop, but the favored edge is only ~**+0.34R** — so a "
        "taker entry starts **underwater and can't clear** (every v3 taker arm ≈ "
        "−1R). This demo arm exists only to test whether *L2 absorption at entry* "
        "can pay for that fee.",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — offline research corpus (always available)
# ══════════════════════════════════════════════════════════════════════════════
st.header("2 · The edge is selection (offline proof)")
st.markdown(
    "Take **every** knife and you bleed. Take only the **favored** slice and it "
    "flips positive. Two independent windows say the same thing — the same "
    "fingerprint at two crime scenes."
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Frozen dev walk-forward** (2021 → 2025-H1, K=120s HGB)")
    dev_tbl = pd.DataFrame({
        "fold": ["F2023", "F2024", "F2025-H1", "Pooled"],
        "AUC": [0.765, 0.721, 0.719, DEV_AUC],
        "take-all mR": [-0.729, -0.579, -0.525, -0.626],
        "favored mR": [0.439, 0.542, 0.343, DEV_MR_SEL],
        "favored n": [136, 185, 109, DEV_N_SEL],
    })
    st.dataframe(dev_tbl, hide_index=True, use_container_width=True)
    st.caption(
        f"Favored = frozen-model top decile (score ≥ {CUT:.3f}). **Positive in "
        "all three folds** — not one lucky window. Holdout (≥2025-07) cleared at "
        "+0.36R, P≈0.9997."
    )

with c2:
    st.markdown("**Holdout — all 12 assets** (break ≥ 2025-07-01, out-of-sample)")
    hold = (episodes[episodes["is_holdout"] == 1]
            if not episodes.empty and "is_holdout" in episodes else pd.DataFrame())
    if not hold.empty and "r_net" in hold and "v1_score" in hold:
        fav = hold[hold["v1_score"] >= CUT]
        unf = hold[hold["v1_score"] < CUT]
        comp = pd.DataFrame({
            "slice": ["Take-all", "Favored", "Unfavored"],
            "n": [len(hold), len(fav), len(unf)],
            "win rate": [(hold.r_net > 0).mean(), (fav.r_net > 0).mean(),
                         (unf.r_net > 0).mean()],
            "mean R": [hold.r_net.mean(), fav.r_net.mean(), unf.r_net.mean()],
        })
        comp_disp = comp.copy()
        comp_disp["win rate"] = comp_disp["win rate"].map(lambda x: f"{x:.1%}")
        comp_disp["mean R"] = comp_disp["mean R"].map(lambda x: f"{x:+.3f}")
        st.dataframe(comp_disp, hide_index=True, use_container_width=True)
        fig = px.bar(comp, x="slice", y="mean R", color="slice",
                     color_discrete_map={"Take-all": "#9E9E9E", "Favored": "#4CAF50",
                                         "Unfavored": "#F44336"},
                     title="Mean R by slice — holdout, all 12 assets")
        fig.add_hline(y=0, line_dash="dot", line_color="#555")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Favored **{fav.r_net.mean():+.3f}R / {(fav.r_net>0).mean():.0%} WR** "
            f"(n={len(fav)}) vs unfavored **{unf.r_net.mean():+.3f}R** — the gate "
            "keeps the snap-backs and drops the bleeders. Caveat: favored is only "
            f"~{len(fav)/len(hold):.0%} of breaks, so it's a sniper, not a soldier. "
            "Per-asset detail (all 12 alts) is in the browser below."
        )
    else:
        st.info("Scored all-asset corpus not found — run "
                "`feature_lab/score_knife_episodes.py`.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — performance browser (shadow-trades-level granularity)
# ══════════════════════════════════════════════════════════════════════════════
st.header("3 · Performance browser")
st.markdown(
    "The same depth the Shadow-Trades page gives the gated book — sliced by "
    "asset, session, regime, DVOL band and direction — plus a **score-cut "
    "what-if** that moves the favored threshold and watches the economics "
    "respond. Pick a source: live arms need a Sync; the offline corpus "
    "(all 12 assets) is always here."
)

KNIFE_LABELS = {
    "knife_shadow.db": "Original (shadow)",
    "knife_funded_maker.db": "Maker (funded)",
    "knife_funded_taker.db": "Taker (funded)",
    "knife_funded_100k.db": "Maker $100k",
}
_bcols = st.columns(len(KNIFE_LABELS))
for _i, (_fn, _lab) in enumerate(KNIFE_LABELS.items()):
    _p = VPS_CACHE_DIR / _fn
    if _p.exists():
        _kb = round(_p.stat().st_size / 1024, 1)
        _age = int((datetime.now().timestamp() - _p.stat().st_mtime) / 60)
        _bcols[_i].success(f"{_lab} · {_kb} KB · {_age}m", icon="🟢")
    else:
        _bcols[_i].error(f"{_lab} · not synced", icon="🔴")

_SRC = {
    "Offline corpus — all 12 assets (15.6k ep.)": "corpus",
    "Original — shadow (live)": "shadow",
    "Maker — funded (live)": "maker",
    "Taker — funded (live)": "taker",
}
src_label = st.radio("Data source", list(_SRC), horizontal=True)
wf = working_frame(_SRC[src_label])

if wf.empty:
    st.info(
        f"**{src_label}** has no local data yet. Click **⟳ Sync knife DBs** in "
        "the sidebar, or switch to the offline corpus."
    )
else:
    # Dev vs holdout window (offline corpus only). Default to honest OOS.
    if "is_holdout" in wf.columns and wf["is_holdout"].notna().any():
        win = st.radio(
            "Episode window",
            ["Holdout (OOS — honest)", "Dev (in-sample — optimistic)", "All"],
            horizontal=True,
            help="The frozen model trained on the dev rows (break < 2025-07-01), so "
                 "their favored numbers are in-sample and far rosier than reality "
                 "(favored ≈ +1.3R / 99% WR). Holdout (≥ 2025-07-01) is the only "
                 "out-of-sample evidence (favored ≈ +0.36R / 64% WR).",
        )
        if win.startswith("Holdout"):
            wf = wf[wf["is_holdout"] == 1]
        elif win.startswith("Dev"):
            wf = wf[wf["is_holdout"] == 0]
            st.warning("Dev rows are **in-sample** — the frozen model trained on "
                       "them, so these favored numbers are optimistic. Use "
                       "**Holdout** for honest out-of-sample performance.")
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    _assets = sorted(wf["asset"].unique())
    f_assets = fc1.multiselect("Asset", _assets, default=_assets)
    f_dir = fc2.multiselect("Direction", ["LONG", "SHORT"], default=["LONG", "SHORT"])
    _regs = sorted(wf["regime"].unique())
    f_reg = fc3.multiselect("Regime", _regs, default=_regs)
    _dvs = sorted(wf["dvol_band"].unique())
    f_dv = fc4.multiselect("DVOL band", _dvs, default=_dvs)
    f_fav = fc5.selectbox("Slice", ["All breaks", "Favored only", "Unfavored only"])

    q = wf[wf.asset.isin(f_assets) & wf.direction.isin(f_dir)
           & wf.regime.isin(f_reg) & wf.dvol_band.isin(f_dv)]
    if f_fav == "Favored only":
        q = q[q.favored]
    elif f_fav == "Unfavored only":
        q = q[~q.favored]

    if q.empty:
        st.warning("No rows match the current filters.")
    else:
        res = q.dropna(subset=["r"])
        n = len(res)
        wins = int((res.r > 0).sum()) if n else 0
        kc = st.columns(6)
        kc[0].metric("Breaks", f"{len(q):,}")
        kc[1].metric("Favored", f"{int(q.favored.sum()):,}")
        kc[2].metric("Resolved", f"{n:,}")
        kc[3].metric("Win rate", f"{wins / n:.1%}" if n else "—")
        kc[4].metric("Avg R", f"{res.r.mean():+.3f}" if n else "—")
        if q["pnl_usd"].notna().any():
            kc[5].metric("Net $", f"${q['pnl_usd'].sum():,.0f}")
        else:
            kc[5].metric("Total R", f"{res.r.sum():+.1f}" if n else "—")

        st.markdown("**Breakdowns** — resolved-trade stats per slice "
                    "(sorted by total R).")
        _tabs = st.tabs(["Asset", "Session", "Regime", "DVOL band",
                         "Direction", "Exit reason"])
        for _tab, _g in zip(_tabs, ["asset", "session", "regime", "dvol_band",
                                    "direction", "exit_reason"]):
            with _tab:
                st.dataframe(
                    summarise_knife(q, _g).rename(
                        columns={_g: _g.replace("_", " ").title()}),
                    use_container_width=True, hide_index=True,
                )

        ec = res.dropna(subset=["ts"]).sort_values("ts")
        if not ec.empty and ec["ts"].notna().any():
            ec = ec.copy()
            ec["grp"] = np.where(ec.favored, "Favored", "Unfavored")
            ec["cum_R"] = ec.groupby("grp")["r"].cumsum()
            fig = px.line(
                ec, x="ts", y="cum_R", color="grp",
                color_discrete_map={"Favored": "#4CAF50", "Unfavored": "#F44336"},
                title="Cumulative R over time — favored vs unfavored",
                labels={"ts": "time (UTC)", "cum_R": "cumulative R", "grp": ""},
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#555")
            fig.update_layout(height=380, margin=dict(t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # ── score-cut what-if (the knife analog of the RR what-if) ─────────
        if q["score"].notna().any():
            st.markdown("---")
            st.subheader("Score-cut what-if — move the favored threshold")
            st.caption(
                "Slide the favored cut and watch how many breaks survive and "
                "what they earn. Higher cut = fewer, cleaner trades (a sniper); "
                "lower cut = more trades, more bleed."
            )
            sq = q.dropna(subset=["score", "r"])
            lo, hi = float(sq.score.min()), float(sq.score.max())
            cut = st.slider("Favored score ≥", round(lo, 3), round(hi, 3),
                            value=float(min(max(CUT, lo), hi)), step=0.01)
            kept = sq[sq.score >= cut]
            wc = st.columns(4)
            wc[0].metric("Kept", f"{len(kept):,}",
                         f"{len(kept) / len(sq):.0%} of breaks" if len(sq) else "")
            wc[1].metric("Win rate", f"{(kept.r > 0).mean():.1%}" if len(kept) else "—")
            wc[2].metric("Avg R", f"{kept.r.mean():+.3f}" if len(kept) else "—")
            wc[3].metric("Total R", f"{kept.r.sum():+.1f}" if len(kept) else "—")
            grid = np.round(np.linspace(lo, hi, 25), 3)
            curve = pd.DataFrame([
                {"cut": c,
                 "avg R": sq[sq.score >= c].r.mean() if (sq.score >= c).any() else np.nan,
                 "kept": int((sq.score >= c).sum())}
                for c in grid])
            fig2 = px.line(curve, x="cut", y="avg R", title="Avg R vs favored cut",
                           labels={"cut": "score cut", "avg R": "avg R of kept"})
            fig2.add_hline(y=0, line_dash="dot", line_color="#555")
            fig2.add_vline(x=CUT, line_dash="dash", line_color="#4CAF50",
                           annotation_text="frozen cut")
            fig2.update_layout(height=320, margin=dict(t=50, b=20))
            st.plotly_chart(fig2, use_container_width=True)
            dq = sq.copy()
            dq["decile"] = pd.qcut(dq.score.rank(method="first"), 10,
                                   labels=False, duplicates="drop")
            dec = dq.groupby("decile").agg(
                n=("r", "size"), avg_R=("r", "mean"),
                WR=("r", lambda x: (x > 0).mean()),
                score_lo=("score", "min"), score_hi=("score", "max"))
            dec["WR"] = dec["WR"].map(lambda x: f"{x:.0%}")
            dec["avg_R"] = dec["avg_R"].map(lambda x: f"{x:+.3f}")
            st.caption("Score-decile calibration — does a higher score really "
                       "buy a higher R? (A clean staircase means the gate is honest.)")
            st.dataframe(dec.round(3).reset_index(), use_container_width=True,
                         hide_index=True)

        # ── episode browser ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Episode browser")
        _allcols = ["ts", "asset", "direction", "session", "regime", "dvol_band",
                    "score", "favored", "r", "exit_reason", "bars_held",
                    "pnl_usd", "source"]
        showcols = [c for c in _allcols if c in q.columns]
        pick = st.multiselect("Columns", showcols,
                              default=[c for c in showcols if c != "source"])
        if pick:
            brow = q[pick].sort_values("ts", ascending=False) if "ts" in pick else q[pick]
            st.dataframe(brow, use_container_width=True, hide_index=True, height=360)
        st.download_button(
            "⬇ Download filtered episodes (CSV)",
            q[showcols].to_csv(index=False).encode(),
            file_name=f"knife_{_SRC[src_label]}_filtered.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — how the knife works
# ══════════════════════════════════════════════════════════════════════════════
st.header("4 · How the knife works")
st.markdown(
    """
**The setup — an 8-bar extreme break.** Price prints a fresh 8-bar high or low
and spikes *through* it. In plain terms: the market just ran the obvious stops
sitting beyond a recent extreme. That stop-run is the "knife" — a fast, emotional
spike that frequently over-shoots and snaps back, like a rubber band yanked too
far.

**The trade — fade it back.** Enter against the spike (short the high-break,
long the low-break) and target the snap-back to the level, with a tight stop
just beyond the extreme. The payoff is asymmetric in time: the good ones revert
within seconds; the bad ones keep running and hit the tight stop.

**The geometry — maker cost model.** Stops sit ~0.5×ATR away (≈0.2–0.5% of
price). Costs are booked as a **maker** round-trip (entry 2 bps, take-profit
2 bps, stop 5.5 bps, plus funding) — because, as §4 shows, the fee math only
works if we never pay the taker spread.

**The gate — the frozen detector.** Not every break is worth fading. A frozen
HistGradient-Boosting model scores each break from the order-flow in the first
**120 seconds** after entry and flags the top decile **favored**. Everything
below the cut is left alone. That 120-second window is both the model's power
*and* its Achilles' heel (§4).
"""
)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — the metrics dictionary (plain English + metaphor + reason/short/sol)
# ══════════════════════════════════════════════════════════════════════════════
st.header("5 · The metrics — what, why, the shortfall, the fix")
st.markdown(
    "Every metric below feeds (or was tested for) the favored gate. For each: "
    "**what it measures**, the **picture**, **why we use it**, the **shortfall** "
    "we hit, and the **solution** we found."
)

METRICS = [
    dict(
        name="L2 absorption", icon="🧱", feat="absorb_ratio_k120, absorb_size/opp/imb",
        what="Volume traded *through* the swept level vs volume that slams *into* "
             "it and stalls. A high ratio means aggressive orders keep hitting the "
             "level but price barely moves.",
        metaphor="A sea-wall: the waves (aggressive volume) keep crashing, but the "
                 "water barely rises — someone big is soaking it up.",
        why="Absorption is the classic reversal tell — a resting whale eating the "
            "stop-run is exactly who you want to fade alongside.",
        short="L2 depth was **never recorded historically** — the order book isn't "
              "in the tape, so the frozen v1 model can only proxy absorption from "
              "trade prints, not see the resting wall itself.",
        sol="A read-only **depth-logger** now records L2 forward, and **record-only** "
            "`absorb_size/opp/imb` columns were added to the funded arms (2026-06-14). "
            "The taker demo arm exists to test whether entry-time absorption is "
            "strong enough to pay the 0.46R taker fee.",
    ),
    dict(
        name="Penetration depth", icon="📏", feat="pen_depth_atr_k120, pen_speed_k120",
        what="How far past the level the spike reached, in ATR units, and how fast "
             "it got there.",
        metaphor="A diving board: the deeper and faster it bends, the harder it "
                 "snaps back.",
        why="Over-extension mean-reverts — a deeper, faster sweep is a more "
            "stretched rubber band, so a higher revert probability.",
        short="The frozen model only *reads* depth at fill+120s; it doesn't **use** "
              "it to choose the entry, so deep favored reverts still gapped past a "
              "limit resting at the level.",
        sol="The **A4 'penetration-depth maker'** redesign places the limit *below* "
            "the low by k×ATR, so the fill itself becomes the filter — it only fills "
            "on a deep sweep, at a better price, with no taker fee and no 120s wait.",
    ),
    dict(
        name="Into vs opposing volume", icon="⚖️", feat="into_vol_k120, opp_vol_k120, large_share",
        what="Volume pushing *into* the extreme (believers chasing the break) vs "
             "volume on the *opposite* side (early revert buyers), plus the share "
             "of large prints.",
        metaphor="A tug-of-war: lots of opposing-side volume means the revert team "
                 "already grabbed the rope.",
        why="A break with weak conviction (low into-volume, rising opposing volume) "
            "is a fakeout dressed as a breakout — prime fade material.",
        short="Raw volume is noisy bar-to-bar and asset-dependent; absolute "
              "thresholds don't transfer across BTC/ETH/SOL.",
        sol="Fed as **ratios and large-print share** into the gradient-boosting "
            "model rather than hard cutoffs, so the model learns per-regime context "
            "instead of a brittle line in the sand.",
    ),
    dict(
        name="Order-book imbalance", icon="📊", feat="imbalance_k120, q_imb_mean/end",
        what="L1 resting size on the bid vs ask, |buy−sell|/(buy+sell), measured "
             "across the window and at its end.",
        metaphor="A crowd leaning one way on a boat — the lean tells you which way "
                 "it's about to tip.",
        why="A book leaning toward the revert side is resting demand ready to catch "
            "the knife.",
        short="L1 quotes only existed from the bookticker archive (~May 2026 on) — "
              "**no historical L1**, so imbalance couldn't enter the frozen dev model.",
        sol="Scouted as **v2 quote features** on the forward window. Verdict: "
            "right-signed but **not significant** (favored-with-mp-drift +0.527 vs "
            "−0.035R, P≈0.88) — registered for forward confirmation, not yet wired.",
    ),
    dict(
        name="Microprice drift", icon="🧭", feat="q_mp_drift_atr, mp_drift_Δ",
        what="Drift of the size-weighted mid-price (microprice) from fill toward the "
             "revert, in ATR units.",
        metaphor="A compass needle: which way the 'true' price is already turning "
                 "before the trades print.",
        why="If the weighted mid is already drifting back toward the level, the "
            "revert has quietly begun — a leading edge over trade prints.",
        short="At **entry time** (the fastest knives) microprice drift had **≈zero "
              "predictive power** in the v3 short-window scout — the signal arrives "
              "after the fast knives are already gone.",
        sol="Demoted from an entry trigger to a **120s-window confirmer** only; the "
            "honest read is that no entry-time microprice signal clears the taker fee.",
    ),
    dict(
        name="Spoof / replenish", icon="🎭", feat="q_replenish",
        what="How fast the swept side's resting size comes *back* after the spike — "
             "replenished size at fill+Δ vs the pre-break baseline.",
        metaphor="A storefront after a smash-and-grab: real shops restock fast; a "
                 "fake front (spoof) stays empty.",
        why="A wall that vanishes on approach was a **spoof** — its 'support' is "
            "fake and the revert won't hold. Quick replenish means a genuine bid.",
        short="Spoofing is only visible in **live L2** and is adversarial — it "
              "appears precisely to fool models, so it can't be learned from history.",
        sol="Captured forward via the depth-logger's `replenish` feature; treated as "
            "a **veto/penalty** in scouting (don't fade into a wall that just "
            "evaporated) rather than a standalone alpha.",
    ),
    dict(
        name="Open interest / funding", icon="💰", feat="oi_delta_pct, funding_z",
        what="Change in open interest around the break, and the funding rate as a "
             "z-score vs the prior 90 funding prints.",
        metaphor="A crowded elevator: when everyone has piled onto one side "
                 "(extreme funding), it doesn't take much to tip it back.",
        why="Extreme positive funding = crowded longs paying to stay long → a "
            "stop-run lower has fuel to revert (longs flushed, then snap back).",
        short="OI was **dead at the 5-minute horizon** (tested in the LRR provider "
              "program) — too slow to time a seconds-scale knife.",
        sol="Kept only as **slow context** (`funding_z` is a frozen model feature; "
            "`oi_delta_pct` is recorded for analysis), never as a trigger. It tilts "
            "the odds, it doesn't pull the trigger.",
    ),
    dict(
        name="VPIN (flow toxicity)", icon="☣️", feat="vpin_bvc (record-only)",
        what="Volume-synchronised probability of informed trading — bulk-classified "
             "buy/sell imbalance over recent volume buckets (Easley/López de Prado).",
        metaphor="A Geiger counter for 'smart money': it clicks when the flow is "
                 "toxic — informed traders, not noise.",
        why="High toxicity means someone *knows* something — fading into informed "
            "flow is catching the knife by the blade.",
        short="VPIN is a **correlated proxy** for the volatility-expansion edge the "
              "strategy already trades, not independent alpha (LRR conditioning "
              "battery: right-signed but couldn't flip a negative host).",
        sol="Logged **record-only** in the shadow episodes (`vpin_bvc`) for "
            "post-hoc study; deliberately **not** a model input to avoid double-"
            "counting an edge already priced in.",
    ),
    dict(
        name="Retrace & stall", icon="⏱️", feat="retrace_frac_k120, stall_secs_k120",
        what="What fraction of the spike has already retraced inside the window, "
             "and how many seconds price stalled at the extreme before moving.",
        metaphor="A sprinter at the line: a long stall at the extreme is a runner "
                 "out of breath — the next step is a stumble back.",
        why="Early retrace + a long stall at the high/low is exhaustion — the spike "
            "ran out of buyers and is about to fade.",
        short="These are **within-window** features (fill+120s), so they're part of "
              "the same latency problem — they confirm a revert you may already have "
              "missed entering.",
        sol="Drove the **B1 (shorter-K)** and **B4 (early time-stop)** redesigns — "
            "read the stall/retrace at fill+30–60s instead of 120s, or cut fast if "
            "no revert by t∈{10,20}s, turning −1R losers into −0.2 to −0.4R cuts.",
    ),
]

for m in METRICS:
    with st.expander(f"{m['icon']} **{m['name']}**  ·  `{m['feat']}`"):
        st.markdown(f"**What it measures.** {m['what']}")
        st.markdown(f"> 🖼️ {m['metaphor']}")
        st.markdown(f"**Why we use it.** {m['why']}")
        st.markdown(f"**Shortfall.** {m['short']}")
        st.markdown(f"**Solution.** {m['sol']}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — the journey: shortfalls & solutions at the strategy level
# ══════════════════════════════════════════════════════════════════════════════
st.header("6 · The journey — what broke, what we did")

st.markdown(
    """
**① The cost anchor that prunes everything.** Stops sit ~0.5×ATR ≈ 0.2–0.5% of
price. A round-trip **taker** fee is ~11 bps ≈ **0.46R** on that stop, while the
favored edge is only ~**+0.34R**. So a taker entry begins **0.46R underwater and
can't clear** — every taker arm in testing ran ≈ −1R.
**→ Solution:** *maker fills only.* The whole redesign is constrained to passive
entries that pay no spread.

**② The live maker bridge failed anyway.** First funded run: **−$3,569 / 17% WR /
0-of-27 favored captured.** Two break-points: the **fill mechanism** (a passive
limit resting *at* the level fills the slow losers and the fast favored reverts
gap straight through it — you catch the falling knives and miss the bouncing
balls) and the **decision latency** (the favored verdict lands at fill+120s, long
after the fastest reverts are over).
**→ Solutions on the bench:** **A4** penetration-depth maker (let the fill depth
*be* the filter), **B1** shorter-K model (verdict at fill+30–60s), **B4** early
momentum time-stop (cut in 10–20s instead of waiting 120s).

**③ Can the score at least *size* another strategy?** We tested the frozen knife
score as a sizing band on the OFCS host (the regime-admitted LRR signals) — H8,
16,848 signals. Verdict: **definitive NULL** (favored +0.099R, Δ+0.248,
P_block 0.86 < 0.99). The regime-admit step had already eaten the knife's edge on
that host.
**→ Solution:** the knife stays a **standalone** signal; it is *not* wired into
OFCS as a modifier.

**④ Where it stands.** The detector's selection edge is **real and forward-
confirmed** (favored +0.34R vs unfavored −0.86R out-of-sample). The open problem
is purely **execution** — fill + latency — which is exactly what the maker/taker
demo arms and the A4/B1/B4 scouts are built to solve. Nothing here is live with
real capital until a redesign clears its pre-declared kill condition on the
$100k demo (shadow can't model fills — the 2026-06-14 lesson).
"""
)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — model card
# ══════════════════════════════════════════════════════════════════════════════
st.header("7 · Frozen model card")
if spec:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", spec.get("model", "hgb").upper())
    c2.metric("K-window", f"{spec.get('k_seconds', 120)}s")
    c3.metric("Dev AUC", f"{DEV_AUC:.3f}")
    c4.metric("Favored cut", f"{CUT:.3f}",
              help="Frozen score threshold (top decile, q90).")
    st.caption(
        f"Label: **{spec.get('label', 'r_net>0 (maker cost model)')}** · "
        f"dev n={spec.get('n_dev', 12774):,} ({spec.get('dev_range', ['', ''])[0][:10]} → "
        f"{spec.get('dev_cutoff', '')[:10]}) · seed {spec.get('seed', '')} · "
        "research-only, frozen (never retrained)."
    )
    nf = spec.get("numeric_features", [])
    cf = spec.get("categorical_features", [])
    st.markdown(
        f"**{len(nf)} numeric + {len(cf)} categorical features.** "
        f"Numeric: `{', '.join(nf)}`. Categorical: `{', '.join(cf)}`."
    )
    st.caption(
        "Sample-weighted by same-asset average uniqueness (López de Prado) so "
        "overlapping, time-clustered episodes don't over-count — grading each "
        "trade by how much genuinely new information it carries."
    )
else:
    st.info("Frozen spec not found locally (feature_lab/_ml_imports/knife_detector_v1_spec.json).")
