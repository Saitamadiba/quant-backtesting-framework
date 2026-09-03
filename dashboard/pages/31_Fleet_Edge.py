"""Page 31: Fleet Edge — every book's edge as a POSTERIOR, not a running total.

WS1 of the fleet quant plan. The scoreboard on page 30 answers "what has this
book made?"; this page answers the question that decides anything: **"how sure
are we that its mean R clears the fee/slip toll — and how many independent bets
is that confidence actually built on?"**

Three rules are baked in, and they are the whole point:

1. **The bar is the toll, not zero.** A book recording R net of fees has already
   paid the crossing charge, so its bar is 0.0R. A book recording GROSS R still
   owes it — bar `+toll` (0.25R by default, the fleet's measured 0.15–0.35R
   midpoint). Sorting is by P(mean R > bar).
2. **Effective n, applied by aggregation.** Fills sharing (family, direction,
   15-minute bucket) are one bet wearing several name tags, and so is a resting
   level a scanner re-stamps every cycle. The posterior is fitted on ONE R per
   bet — the cluster's mean — so a replicated setup can neither narrow the
   interval nor drag the point estimate. It is the difference between counting
   forty till receipts and counting the one sale they all record.
3. **The prior is skeptical and centred on the toll.** Two lucky trades read as
   "prior dominates", never as an edge.

*Picture: a witness statement with an error bar. Ten trades is a hunch, two
hundred is testimony — and ten witnesses who all heard it from one person are
still one witness.*

This is a READING instrument. It has no kill bar, gates no trade and moves no
size; a book that reads well here is a nominee for a pre-registered forward
shadow, never a deployment.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Fleet Edge", page_icon="🎚️", layout="wide")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data.fleet_edge import (
    DEFAULT_TOLL_R, MIN_TRADES, family_posteriors, fleet_posteriors, verdict,
)
from data.fleet_registry import TIER_NAMES

TIER_COLOR = {1: "#2e7d32", 2: "#1565c0", 3: "#8d6e63"}
VERDICT_COLOR = {"evidence for": "#2e7d32", "evidence against": "#c62828",
                 "undecided": "#ef6c00", "prior dominates": "#8d6e63",
                 "no read": "#9e9e9e"}


@st.cache_data(ttl=300, show_spinner="Fitting posteriors over the fleet…")
def load(toll: float, prior: str, ci: float) -> tuple:
    books = fleet_posteriors(toll=toll, prior=prior)
    fams = family_posteriors(toll=toll, prior=prior)
    return books, fams


# ══════════════════════════════════════════════════════════════════════════════
#  Controls
# ══════════════════════════════════════════════════════════════════════════════
st.title("🎚️ Fleet Edge — posteriors, not running totals")

c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.4, 1.6])
with c1:
    toll = st.slider("Fee/slip toll (R)", 0.0, 0.50, DEFAULT_TOLL_R, 0.05,
                     help="What one round trip costs. Applied as the bar a GROSS "
                          "book must clear; net books already paid it.")
with c2:
    prior_name = st.selectbox("Prior", ["toll", "skeptical", "uninformative"], index=0,
                              help="'toll' = the fleet's null: a seat that pays the "
                                   "crossing charge and has no edge (μ₀ = −0.25R).")
with c3:
    tiers = st.multiselect("Tiers", [1, 2, 3], default=[1, 2, 3],
                           format_func=lambda t: f"{t} · {TIER_NAMES.get(t, '')}")
with c4:
    view = st.radio("Grain", ["Per book", "Per family"], horizontal=True)

books, fams = load(float(toll), prior_name, 0.90)
if books.empty:
    st.error("No fleet books readable locally — sync the fleet databases first "
             "(page 30 pulls them in one rsync batch).")
    st.stop()

df = (books if view == "Per book" else fams).copy()
df = df[df["tier"].isin(tiers)] if not df.empty else df
readable = df[df["status"] == "ok"].copy()
if not readable.empty:
    readable["verdict"] = readable.apply(verdict, axis=1)

# ══════════════════════════════════════════════════════════════════════════════
#  Headline
# ══════════════════════════════════════════════════════════════════════════════
if readable.empty:
    st.warning("Nothing with two or more closed trades in the selected tiers.")
else:
    v = readable["verdict"].value_counts()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Books read", f"{len(readable)}")
    m2.metric("Evidence FOR", f"{int(v.get('evidence for', 0))}",
              help="P(mean R > bar) ≥ 90% on ≥5 effective bets. A nominee, not a deployment.")
    m3.metric("Evidence AGAINST", f"{int(v.get('evidence against', 0))}",
              help="P(mean R > bar) ≤ 10%.")
    m4.metric("Undecided", f"{int(v.get('undecided', 0))}")
    m5.metric("Prior dominates", f"{int(v.get('prior dominates', 0))}",
              help=f"Fewer than {MIN_TRADES} effective bets — the prior is the answer.")

    haircut = readable["n"].sum() / max(readable["n_eff"].sum(), 1)
    st.caption(
        f"**{int(readable['n'].sum()):,} closed rows → {int(readable['n_eff'].sum()):,} "
        f"independent bets** ({haircut:.1f}× clustering haircut). Bar = "
        f"{toll:.2f}R for gross books, 0.00R for net ones. Sorted by P(mean R > bar). "
        "*Ten fills in one 15-minute bucket are one bet wearing ten name tags.*"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  The posterior strip
# ══════════════════════════════════════════════════════════════════════════════
if not readable.empty:
    st.subheader("Posterior mean R — point, 90% interval, and the bar it must clear")
    plot = readable.sort_values("post_mean_r").tail(45)
    fig = go.Figure()
    for _, r in plot.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["ci_lo"], r["ci_hi"]], y=[r["label"], r["label"]],
            mode="lines", line=dict(color=TIER_COLOR.get(int(r["tier"]), "#666"), width=6),
            opacity=0.35, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[r["post_mean_r"]], y=[r["label"]], mode="markers",
            marker=dict(size=11, color=VERDICT_COLOR.get(r["verdict"], "#666"),
                        line=dict(width=1, color="#333")),
            showlegend=False,
            hovertemplate=(f"<b>{r['label']}</b><br>"
                           f"posterior mean {r['post_mean_r']:+.3f}R<br>"
                           f"90% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]<br>"
                           f"n {int(r['n'])} → n_eff {int(r['n_eff'])}<br>"
                           f"bar {r['bar_r']:.2f}R · P(>bar) {r['p_above_bar']:.1%}"
                           "<extra></extra>")))
        fig.add_trace(go.Scatter(
            x=[r["bar_r"]], y=[r["label"]], mode="markers",
            marker=dict(size=9, symbol="line-ns-open", color="#111"),
            hoverinfo="skip", showlegend=False))
    fig.add_vline(x=0.0, line=dict(color="#111", width=1, dash="dot"))
    fig.update_layout(height=max(420, 20 * len(plot)), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="mean R (posterior)", yaxis_title=None,
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bar colour = tier (green Tier 1, blue Tier 2, brown Tier 3); dot colour = "
               "verdict; the black tick is the bar that book has to clear. A wide bar is "
               "not a bad book — it is a book we have not watched long enough.")

    # ── the table ────────────────────────────────────────────────────────────
    st.subheader("Every readable book")
    show = readable[["label", "tier", "gross", "n", "n_eff", "sum_r", "raw_mean_r",
                     "bet_mean_r", "post_mean_r", "ci_lo", "ci_hi", "bar_r", "p_above_0",
                     "p_above_bar", "post_win_rate", "verdict", "last_ts"]].copy()
    show = show.rename(columns={
        "label": "Book", "tier": "T", "gross": "Gross", "n": "rows", "n_eff": "bets",
        "sum_r": "ΣR", "raw_mean_r": "row mean R", "bet_mean_r": "bet mean R",
        "post_mean_r": "post mean R", "ci_lo": "CI lo", "ci_hi": "CI hi", "bar_r": "bar",
        "p_above_0": "P(>0)", "p_above_bar": "P(>bar)", "post_win_rate": "post WR",
        "verdict": "verdict", "last_ts": "last trade"})
    st.dataframe(
        show.style.format({"ΣR": "{:+.1f}", "row mean R": "{:+.3f}", "bet mean R": "{:+.3f}",
                           "post mean R": "{:+.3f}", "CI lo": "{:+.3f}", "CI hi": "{:+.3f}",
                           "bar": "{:.2f}", "P(>0)": "{:.1%}", "P(>bar)": "{:.1%}",
                           "post WR": "{:.1%}"}),
        use_container_width=True, hide_index=True, height=520)

    st.caption(
        "**row mean R → bet mean R** is the replication haircut: how much of a book's "
        "average was one wager counted many times. **bet mean R → post mean R** is the "
        "shrinkage: how far the skeptical prior pulls a thin book back toward the toll. "
        "A book whose three columns agree has earned its number; one where they diverge "
        "has not yet said enough to be believed."
    )

# ══════════════════════════════════════════════════════════════════════════════
#  What could not be read
# ══════════════════════════════════════════════════════════════════════════════
unread = df[df["status"] != "ok"]
if not unread.empty:
    with st.expander(f"⚠️ {len(unread)} book(s) with no posterior — absent, not zero"):
        st.dataframe(unread[["label", "tier", "n", "status"]]
                     .rename(columns={"label": "Book", "tier": "T", "n": "rows",
                                      "status": "why"}),
                     use_container_width=True, hide_index=True)
        st.caption("A book that cannot be read must show as unread. Rendering it as 0.0R "
                   "would put a silent seat in the same column as a flat one.")

# ══════════════════════════════════════════════════════════════════════════════
#  How to read this page
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("How to read this page (and how it can mislead you)"):
    st.markdown(f"""
**The posterior is about the MEAN, not the next trade.** A 90% interval of
[−0.05, +0.20] says the book's long-run average R is probably in that range —
individual trades still scatter a full R either side.

**P(>bar) is not a p-value and not a permission slip.** It is the posterior mass
above the toll under a skeptical prior. A book at 95% is a *nominee* for a
pre-registered forward shadow; nothing on this page changes sizing or deploys a
seat.

**Effective n is where the honesty lives.** Two collapses are applied before the
fit: fills in the same (family, direction, 15-minute bucket) become one bet, and
a resting level re-stamped by a scanner every cycle becomes one bet however many
rows it wrote. Each bet then contributes ONE R — the cluster's mean — so the
point estimate and the interval both stop counting replicas. Worked example from
this fleet: `halt-shadow(era2)` writes 1,258 rows over 134 distinct levels and
reads **+0.672R at a 90% win rate** on the rows — impossible at 1:1 RR — and
**+0.03R** on the bets. Same data, one wager counted forty times.

**Gross books owe the toll.** Labels carrying *(gross…)* are charged {toll:.2f}R:
they have to earn the crossing charge back before they are worth anything. Net
books are already charged and their bar is 0.

**Eras are never pooled.** Recorder eras are separate books in the registry
(`ofcs-paper(gross,era1)` vs `(net,era2)`), and this page reads them apart —
an honest-clock fix that flips a book's sign must never be averaged away.

**What this page cannot tell you.** It reads recorded outcomes, so it inherits
whatever the recorder does: an optimistic shadow resolver, a book whose fills
never happened at the printed price, or a first-passage bracket asymmetry will
all show up here as "edge". When a Tier-3 recorder reads far better than every
live seat in its own family, suspect the recorder before believing the edge.
""")
