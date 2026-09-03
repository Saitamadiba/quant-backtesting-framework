"""Page 16: Cross-Asset Robustness — test if strategy edge is structural."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Cross-Asset Robustness", page_icon="🌐", layout="wide")
st.title("🌐 Cross-Asset Robustness")
st.caption(
    "Test whether a strategy's edge is structural (works across multiple assets) "
    "or asset-specific (fragile). Compare OOS performance across BTC, ETH, NQ."
)

# Ensure imports work
_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))


# ══════════════════════════════════════════════════════════════════════════════
# MODE — the backtest's cross-asset story, or the LIVE market's (WS2)
# ══════════════════════════════════════════════════════════════════════════════
# The WFO sections below stop early whenever no results are loaded, so the fleet
# atlas lives ahead of them: it depends on the local price store and the Deribit
# index, never on a WFO run.
_mode = st.radio(
    "View", ["WFO robustness (backtests)", "Fleet — the live market (WS2)"],
    horizontal=True, key="cross_asset_mode",
    help="The second view measures the market the fleet actually trades in: how "
         "correlated its assets are, whether BTC leads them, and whether implied "
         "volatility changes any of it.",
)

if _mode.startswith("Fleet"):
    import numpy as _np
    from data.vol_atlas import (FLEET, beta_by_band, correlation_matrix, fleet_band_fwer,
                                fleet_fills_with_dvol, fleet_r_by_band, lead_lag,
                                rolling_correlation, skew_direction)

    st.caption(
        "Everything here is computed on THIS machine from the local 15-minute "
        "store and Deribit's public volatility index — nothing is cached on the "
        "VPS. *An atlas, not a gate: it tells you where the ground is soft, it "
        "does not tell you where to step.*"
    )

    t_corr, t_lead, t_band, t_fleet, t_skew = st.tabs(
        ["Correlation", "Lead / lag", "Beta by DVOL band", "Fleet R by band", "Skew"])

    # ── correlation ──────────────────────────────────────────────────────────
    with t_corr:
        win = st.select_slider("Correlation window (days)", [3, 7, 14, 30], value=7,
                               key="ws2_corr_win")
        with st.spinner("Correlating 15-minute returns…"):
            roll = rolling_correlation(window_days=win)
            mat = correlation_matrix(window_days=win)
        if roll.empty:
            st.error("No local 15m bars — nothing to correlate.")
        else:
            cur = float(roll["mean_rho"].iloc[-1])
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean pairwise ρ, latest window", f"{cur:.3f}")
            c2.metric("Median over all windows", f"{roll['mean_rho'].median():.3f}")
            c3.metric("Range", f"{roll['mean_rho'].min():.2f} – {roll['mean_rho'].max():.2f}")
            fig = go.Figure(go.Scatter(x=roll["window_end"], y=roll["mean_rho"],
                                       mode="lines", line=dict(color="#42A5F5", width=1.4),
                                       name="mean ρ"))
            fig.add_trace(go.Scatter(x=roll["window_end"], y=roll["max_rho"], mode="lines",
                                     line=dict(color="#EF5350", width=1, dash="dot"),
                                     name="max pair ρ"))
            fig.update_layout(template="plotly_dark", height=320, yaxis_title="ρ",
                              margin=dict(l=10, r=10, t=30, b=10),
                              title=f"Mean pairwise correlation of 15m returns, "
                                    f"{win}-day trailing window")
            st.plotly_chart(fig, use_container_width=True, key="ws2_roll_corr")
            st.warning(
                f"**This is the sizing number, not a signal.** In the median week the "
                f"fleet's {len(FLEET)} assets move together at ρ ≈ "
                f"{roll['mean_rho'].median():.2f}, and the worst week reached "
                f"{roll['mean_rho'].max():.2f}. *Two seats long two different alts in a "
                "0.9-correlation week are one bet wearing two tickets* — the effective-n "
                "haircut on page 31 applies within a family; this is the same problem "
                "ACROSS them."
            )
            if not mat.empty:
                fig2 = go.Figure(go.Heatmap(
                    z=mat.values, x=mat.columns.tolist(), y=mat.index.tolist(),
                    text=[[f"{v:.2f}" for v in row] for row in mat.values],
                    texttemplate="%{text}", textfont=dict(size=10),
                    colorscale="RdBu_r", zmin=-1, zmax=1))
                fig2.update_layout(template="plotly_dark",
                                   height=max(340, 34 * len(mat)),
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   title=f"Latest {win}-day correlation matrix")
                st.plotly_chart(fig2, use_container_width=True, key="ws2_corr_mat")

    # ── lead / lag ───────────────────────────────────────────────────────────
    with t_lead:
        with st.spinner("Cross-correlating…"):
            ll = lead_lag()
        if ll.empty:
            st.error("No local 15m bars.")
        else:
            show = ll[["symbol", "n", "rho_0", "btc_leads_1", "alt_leads_1",
                       "btc_leads_2", "alt_leads_2"]]
            st.dataframe(show.style.format({"rho_0": "{:.4f}", "btc_leads_1": "{:+.4f}",
                                            "alt_leads_1": "{:+.4f}", "btc_leads_2": "{:+.4f}",
                                            "alt_leads_2": "{:+.4f}"}),
                         use_container_width=True, hide_index=True)
            n = int(ll["n"].max())
            se = 1 / _np.sqrt(n)
            biggest = float(_np.abs(ll[["btc_leads_1", "alt_leads_1"]].to_numpy()).max())
            st.success(
                f"**Nothing leads anything.** Contemporaneous ρ runs "
                f"{ll['rho_0'].min():.2f}–{ll['rho_0'].max():.2f}, but at one bar ahead the "
                f"largest |ρ| in either direction is {biggest:.4f} — against a standard "
                f"error of {se:.4f} at n={n:,}. Detectable, and worth roughly nothing: "
                f"|ρ| that small explains under {100*biggest**2:.2g}% of the next bar's "
                "variance, while a round trip costs 0.15–0.35R. *A whisper you can only "
                "hear in a soundproof room, and the door charges admission.*"
            )
            st.caption(
                "Both directions are shown deliberately. If anything the alts lead BTC "
                "slightly more often than the reverse — the opposite of the folk story — "
                "and the two-bar column is NEGATIVE for every asset both ways, which is "
                "bid-ask bounce and mean reversion, not information flow."
            )

    # ── beta by band ─────────────────────────────────────────────────────────
    with t_band:
        with st.spinner("Fitting beta inside each DVOL band…"):
            bb = beta_by_band()
        if bb.empty:
            st.error("No local 15m bars or no DVOL series.")
        else:
            order = [b for b in ("P_LOW", "P_MID", "P_HIGH") if b in set(bb["band"])]
            for metric, title, fmt in (("beta", "Beta to BTC", "{:.3f}"),
                                       ("r2", "R² — share of the alt's move that IS BTC", "{:.3f}"),
                                       ("resid_vol_annual_pct", "Residual vol (annualised %)", "{:.1f}")):
                piv = bb.pivot(index="symbol", columns="band", values=metric)[order]
                st.markdown(f"**{title}**")
                # No `background_gradient` here: it needs matplotlib, which this
                # dashboard does not install — the styler raises at render time
                # rather than degrading, and nothing else on the page depends on it.
                st.dataframe(piv.style.format(fmt), use_container_width=True)
            piv_r2 = bb.pivot(index="symbol", columns="band", values="r2")
            if {"P_LOW", "P_HIGH"} <= set(piv_r2.columns):
                rose = int((piv_r2["P_HIGH"] > piv_r2["P_LOW"]).sum())
                st.warning(
                    f"**Beta barely moves; R² rises in {rose} of {len(piv_r2)} assets** "
                    f"(mean {piv_r2['P_LOW'].mean():.3f} → {piv_r2['P_HIGH'].mean():.3f}). "
                    "Diversification does not fail because the alts start swinging harder "
                    "relative to BTC — it fails because a larger *share* of each alt's move "
                    "becomes BTC's move. *The orchestra doesn't change instruments when it "
                    "gets loud; it stops playing separate parts.* Concentration risk is "
                    "highest exactly when the book is most exposed."
                )

    # ── fleet R by band, with the family-wise bar ────────────────────────────
    with t_fleet:
        st.caption(
            "Every fleet fill tagged with the DVOL state that had **closed** by its "
            "entry (Deribit stamps a bar at its open, so a fill at 10:30 sees the "
            "09:00 bar and no later). R is per BET, not per row — the page-31 haircut."
        )
        with st.spinner("Joining fills to the DVOL tape…"):
            fills = fleet_fills_with_dvol()
        if fills.empty:
            st.error("No fleet books readable locally.")
        else:
            band_col = st.radio("Split by", ["pct_band", "abs_band", "d24h_sign"],
                                horizontal=True, key="ws2_bandcol",
                                format_func=lambda c: {"pct_band": "DVOL percentile band",
                                                       "abs_band": "Absolute band (the live gate's)",
                                                       "d24h_sign": "DVOL rising vs falling"}[c])
            tbl = fleet_r_by_band(band_col=band_col, fills=fills)
            st.dataframe(tbl.style.format({"mean_r": "{:+.3f}", "sum_r": "{:+.1f}", "t": "{:+.2f}"}),
                         use_container_width=True, hide_index=True, height=420)
            if st.button("Run the family-wise bar (permutation, ~10s)", key="ws2_fwer"):
                with st.spinner("Permuting band labels in day-sized blocks…"):
                    fw = fleet_band_fwer(band_col=band_col, fills=fills)
                st.dataframe(fw.style.format({"max_abs_t": "{:.2f}", "p_fwer": "{:.4f}",
                                              "alpha_bonferroni": "{:.4f}"}),
                             use_container_width=True, hide_index=True)
                tested = int(fw["p_fwer"].notna().sum())
                cleared = int(fw["clears"].sum())
                refused = len(fw) - tested
                st.info(
                    f"**{cleared} of {tested} testable families clear the bar**"
                    + (f"; {refused} refused — too few distinct day-blocks for the "
                       "permutation null to mean anything." if refused else ".")
                )
                st.caption(
                    "The statistic is a **contrast** — each band against the family's own "
                    "mean — so a book that simply loses in every band scores zero here. "
                    "Labels are permuted in day-sized blocks because DVOL barely moves "
                    "inside a day; shuffling fill by fill would compare real, clustered "
                    "data against a null that never clusters and hand back significance "
                    "for free."
                )

    # ── skew ─────────────────────────────────────────────────────────────────
    with t_skew:
        a = st.radio("Asset", ["BTC", "ETH"], horizontal=True, key="ws2_skew_asset")
        res = skew_direction(a)
        if res.get("status") != "ok":
            st.info(f"Not measurable yet: {res.get('status')}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Observations", f"{res['n']:,}")
            c2.metric("mean 24h return | RR25 > 0", f"{res['mean_fwd_bps_rr_pos']:+.0f} bps")
            c3.metric("mean 24h return | RR25 ≤ 0", f"{res['mean_fwd_bps_rr_neg']:+.0f} bps")
            st.error(
                f"**Read this against the baseline, not against 50%.** Unconditionally, "
                f"price touches +{res['bracket_pct']:.0f}% before −{res['bracket_pct']:.0f}% "
                f"only {res['up_rate_unconditional']:.1%} of the time on this tape — the "
                f"BRACKET is already asymmetric before any signal is applied. Conditional "
                f"on RR25 the rate is {res['up_rate_rr_pos']:.1%} (positive skew) versus "
                f"{res['up_rate_rr_neg']:.1%} (negative). *On a driftless path any bracket "
                "looks like a prediction; the mirror is what tells you whether you found "
                "one.* The sample is ~5 weeks of rr25 — far too short for a verdict."
            )

    st.stop()


from backtrader_framework.optimization.persistence import list_wfo_results
from backtrader_framework.optimization.cross_asset_robustness import (
    CrossAssetAnalyzer,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. COVERAGE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

st.header("1. Coverage Matrix")
st.caption("Which strategy/asset combinations have WFO results.")

saved = list_wfo_results()
if not saved:
    st.warning("No saved WFO results found. Run WFO optimization first.")
    st.stop()

analyzer = CrossAssetAnalyzer()
coverage = analyzer.get_coverage_matrix()

strategies = coverage['strategies']
assets = coverage['assets']

# Build coverage table
cov_rows = []
for strat in strategies:
    row = {'Strategy': strat}
    for asset in assets:
        row[asset] = '✅' if coverage['coverage'][strat].get(asset) else '❌'
    cov_rows.append(row)

st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)

if coverage['missing']:
    st.info(
        f"{len(coverage['missing'])} missing combination(s). "
        "Use the buttons below to run WFO for missing combos."
    )

    # Run missing buttons
    for strat, asset, tf in coverage['missing']:
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            run_key = f"run_wfo_{strat}_{asset}_{tf}"
            if st.button(f"Run {strat}/{asset}/{tf}", key=run_key):
                st.session_state[f'_running_{run_key}'] = True

        with col_status:
            if st.session_state.get(f'_running_{run_key}'):
                try:
                    with st.spinner(f"Running WFO for {strat}/{asset}/{tf}..."):
                        fp = analyzer.run_missing_wfo(
                            strat, asset, tf,
                            progress_callback=lambda p, m: None,
                        )
                    st.success(f"Done! Saved to {Path(fp).name}")
                    st.session_state[f'_running_{run_key}'] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
                    st.session_state[f'_running_{run_key}'] = False


# ══════════════════════════════════════════════════════════════════════════════
# 2. SELECT RESULTS & ANALYZE
# ══════════════════════════════════════════════════════════════════════════════

st.header("2. Select Results")

# Build display labels
options = {}
for i, r in enumerate(saved):
    label = f"{r['strategy']}  /  {r['symbol']}  /  {r['timeframe']}  —  {r['timestamp']}"
    options[i] = label

# Default: select latest result per unique strategy/symbol/timeframe combo
seen = set()
defaults = []
for i, r in enumerate(saved):
    key = (r['strategy'], r['symbol'], r['timeframe'])
    if key not in seen:
        seen.add(key)
        defaults.append(i)

selected_indices = st.multiselect(
    "Select WFO results to compare (pick same strategy across different assets)",
    list(options.keys()),
    format_func=lambda i: options[i],
    default=defaults,
    key="cross_asset_select",
    help="Select one WFO result per strategy/asset combination. The tool compares performance of the same strategy across different assets.",
)

if len(selected_indices) < 2:
    st.info("Select at least 2 WFO results to analyze cross-asset robustness.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if st.button("Analyze Cross-Asset Robustness", type="primary"):
    filepaths = [saved[i]['filepath'] for i in selected_indices]

    try:
        with st.spinner("Loading WFO results and computing robustness..."):
            ca = CrossAssetAnalyzer()
            ca.load_results(filepaths)
            result = ca.analyze_all()

        st.session_state['cross_asset_result'] = result
        st.session_state['cross_asset_analyzer'] = ca

        n_analyzed = len(result['strategies'])
        if n_analyzed == 0:
            st.warning(
                "No strategies have 2+ assets loaded. "
                "Select results for the same strategy on different assets."
            )
        else:
            st.success(f"Analysis complete! {n_analyzed} strategy(ies) evaluated.")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if 'cross_asset_result' not in st.session_state:
    st.info("Select results above and click 'Analyze Cross-Asset Robustness' to start.")
    st.stop()

result = st.session_state['cross_asset_result']
strat_results = result.get('strategies', {})
cross_strat = result.get('cross_strategy', {})

if not strat_results:
    st.warning("No strategies with 2+ assets to analyze.")
    st.stop()


# ── Strategy × Asset Heatmap ──────────────────────────────────────────────

st.header("3. Strategy × Asset Performance")

METRIC_MAP = {
    'Sharpe': 'sharpe',
    'Mean R': 'mean_r',
    'Win Rate': 'win_rate',
    'Expectancy': 'expectancy',
    'Profit Factor': 'profit_factor',
}

metric_choice = st.selectbox(
    "Metric to display", list(METRIC_MAP.keys()), index=0,
    key="cross_asset_metric",
    help="Choose which metric to visualize in the heatmap. Sharpe = risk-adjusted return. Win Rate = % profitable trades. Mean R = average R-multiple per trade.",
)
metric_attr = METRIC_MAP[metric_choice]

# Build heatmap data
all_assets = sorted(set(
    asset for rob in strat_results.values() for asset in rob.assets
))
all_strats = sorted(strat_results.keys())

z_data = []
text_data = []
for strat in all_strats:
    row_z = []
    row_t = []
    rob = strat_results[strat]
    for asset in all_assets:
        ar = rob.asset_results.get(asset)
        if ar:
            val = getattr(ar, metric_attr)
            row_z.append(val)
            if metric_attr == 'win_rate':
                row_t.append(f"{val:.1%}")
            elif metric_attr in ('mean_r', 'expectancy'):
                row_t.append(f"{val:+.4f}")
            else:
                row_t.append(f"{val:.3f}")
        else:
            row_z.append(None)
            row_t.append("—")
    z_data.append(row_z)
    text_data.append(row_t)

fig_hm = go.Figure(data=go.Heatmap(
    z=z_data,
    x=all_assets,
    y=all_strats,
    text=text_data,
    texttemplate="%{text}",
    textfont=dict(size=14, color='white'),
    colorscale='RdYlGn',
    colorbar=dict(title=metric_choice),
    hovertemplate="Strategy: %{y}<br>Asset: %{x}<br>Value: %{text}<extra></extra>",
))
fig_hm.update_layout(
    template="plotly_dark",
    height=max(200, 70 * len(all_strats)),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_hm, use_container_width=True, key="cross_asset_heatmap")


# ── Per-Strategy Deep Dive ────────────────────────────────────────────────

st.header("4. Per-Strategy Deep Dive")

tab_names = list(strat_results.keys())
tabs = st.tabs(tab_names)

EQUITY_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']

for tab, strat_name in zip(tabs, tab_names):
    with tab:
        rob = strat_results[strat_name]

        # KPI row
        k1, k2, k3 = st.columns(3)
        k1.metric("Robustness Score", f"{rob.robustness_score:.0f}/100",
                  help="Composite 0-100 score. 80+ = strategy works across multiple assets (structural edge). <50 = asset-specific (may be curve-fitted to one market).")
        k2.metric("Grade", rob.robustness_grade,
                  help="Letter grade for cross-asset robustness. A = excellent structural edge across all tested assets. F = only works on one asset.")
        k3.metric("Assets Tested", len(rob.assets),
                  help="Number of different assets this strategy was evaluated on. More assets tested = more confident robustness assessment.")

        st.info(rob.verdict)

        # Metrics table
        with st.expander("Side-by-Side Metrics", expanded=True):
            st.caption("Compares key metrics (Sharpe, win rate, mean R, drawdown) for the same strategy across assets. Similar values = universal pattern. Divergent = asset-specific adaptation.")
            df_metrics = pd.DataFrame(rob.metrics_table)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # Equity curves
        with st.expander("Equity Curves", expanded=True):
            st.caption("OOS equity curves overlaid by asset. High visual correlation = strategy doesn't adapt well to different markets.")
            fig_eq = go.Figure()
            for i, (asset, ar) in enumerate(sorted(rob.asset_results.items())):
                if not ar.oos_equity:
                    continue
                times = pd.to_datetime([e['time'] for e in ar.oos_equity])
                cum_r = [e['cumulative_r'] for e in ar.oos_equity]
                fig_eq.add_trace(go.Scatter(
                    x=times, y=cum_r, mode='lines',
                    name=asset,
                    line=dict(color=EQUITY_COLORS[i % len(EQUITY_COLORS)], width=2),
                ))

            fig_eq.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_eq.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="Date",
                yaxis_title="Cumulative R-Multiple",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_eq, use_container_width=True, key=f"eq_{strat_name}")

        # Equity correlation
        if rob.equity_correlation is not None:
            with st.expander("Equity Curve Correlation", expanded=False):
                st.caption("Correlation matrix of returns across assets. >0.5 = moves together (less diversification benefit). <0.1 = independent streams (good for portfolio construction).")
                corr = rob.equity_correlation
                fig_corr = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    text=[[f"{v:.3f}" for v in row] for row in corr.values],
                    texttemplate="%{text}",
                    textfont=dict(size=14),
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1,
                    colorbar=dict(title="Correlation"),
                ))
                fig_corr.update_layout(
                    template="plotly_dark",
                    height=max(250, 80 * len(corr)),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_{strat_name}")

        # Regime comparison
        with st.expander("Regime Comparison", expanded=True):
            st.caption("Mean R-multiple per market regime across assets. Reveals if the edge is regime-dependent (e.g., only works in trends).")
            regimes = ['trending_up', 'trending_down', 'ranging', 'volatile']
            regime_assets = sorted(rob.asset_results.keys())

            # Build heatmap: regime × asset → mean_r
            rz = []
            rt = []
            for regime in regimes:
                row_z = []
                row_t = []
                for asset in regime_assets:
                    data = rob.regime_comparison.get(regime, {}).get(asset, {})
                    mr = data.get('mean_r', 0.0)
                    nt = data.get('n_trades', 0)
                    row_z.append(mr)
                    row_t.append(f"{mr:+.3f}\n({nt}t)")
                rz.append(row_z)
                rt.append(row_t)

            fig_regime = go.Figure(go.Heatmap(
                z=rz,
                x=regime_assets,
                y=[r.replace('_', ' ').title() for r in regimes],
                text=rt,
                texttemplate="%{text}",
                textfont=dict(size=12),
                colorscale='RdYlGn',
                colorbar=dict(title="Mean R"),
                hovertemplate="Regime: %{y}<br>Asset: %{x}<br>Mean R: %{z:.4f}<extra></extra>",
            ))
            fig_regime.update_layout(
                template="plotly_dark",
                height=max(200, 60 * len(regimes)),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_regime, use_container_width=True, key=f"regime_{strat_name}")

        # Direction breakdown
        with st.expander("Direction Breakdown", expanded=False):
            st.caption("LONG vs SHORT performance by asset. Asymmetric win rates suggest the strategy has a directional bias.")
            dir_rows = []
            for asset, ar in sorted(rob.asset_results.items()):
                for direction in ['LONG', 'SHORT']:
                    dd = ar.direction_breakdown.get(direction, {})
                    dir_rows.append({
                        'Asset': asset,
                        'Direction': direction,
                        'Trades': dd.get('n_trades', 0),
                        'Win Rate': f"{dd.get('win_rate', 0):.1%}",
                        'Mean R': f"{dd.get('mean_r', 0):+.4f}",
                        'Total R': f"{dd.get('total_r', 0):+.2f}",
                    })
            if dir_rows:
                st.dataframe(
                    pd.DataFrame(dir_rows),
                    use_container_width=True, hide_index=True,
                )


# ── Robustness Scores ─────────────────────────────────────────────────────

st.header("5. Robustness Scores")

# Score cards
score_cols = st.columns(len(strat_results))
for col, (strat_name, rob) in zip(score_cols, sorted(
    strat_results.items(), key=lambda x: -x[1].robustness_score,
)):
    with col:
        st.metric(strat_name, f"{rob.robustness_score:.0f}/100")
        grade_colors = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
        st.write(f"{grade_colors.get(rob.robustness_grade, '')} Grade **{rob.robustness_grade}**")
        st.caption(rob.verdict)

# Sub-score breakdown
with st.expander("Score Breakdown", expanded=True):
    for strat_name, rob in sorted(strat_results.items()):
        st.subheader(strat_name)
        comp = rob.component_scores
        labels = list(comp.keys())
        values = list(comp.values())
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=values,
            y=[l.replace('_', ' ').title() for l in labels],
            orientation='h',
            marker_color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B'],
            text=[f"{v:.0f}" for v in values],
            textposition='inside',
            hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            height=200,
            xaxis=dict(range=[0, 100], title="Score"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key=f"scores_{strat_name}")

# Comparison table
with st.expander("Comparison Table", expanded=False):
    comp_rows = []
    for strat_name, rob in sorted(
        strat_results.items(), key=lambda x: -x[1].robustness_score,
    ):
        comp_rows.append({
            'Strategy': strat_name,
            'Assets': ', '.join(rob.assets),
            'Score': f"{rob.robustness_score:.0f}",
            'Grade': rob.robustness_grade,
            'Avg Sharpe': f"{np.mean([ar.sharpe for ar in rob.asset_results.values()]):.3f}",
            'Avg Win Rate': f"{np.mean([ar.win_rate for ar in rob.asset_results.values()]):.1%}",
            'Verdict': rob.verdict,
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ── Cross-Asset Correlation ───────────────────────────────────────────────

all_equity_labels = []
all_equity_series = {}

for strat_name, rob in strat_results.items():
    for asset, ar in rob.asset_results.items():
        if not ar.oos_equity:
            continue
        label = f"{strat_name}_{asset}"
        times = pd.to_datetime([e['time'] for e in ar.oos_equity])
        cum_r = pd.Series(
            [e['cumulative_r'] for e in ar.oos_equity],
            index=times, dtype=float,
        )
        cum_r = cum_r[~cum_r.index.duplicated(keep='last')].sort_index()
        daily_cum = cum_r.resample('D').last().ffill().fillna(0.0)
        daily_ret = daily_cum.diff().fillna(0.0)
        all_equity_series[label] = daily_ret
        all_equity_labels.append(label)

if len(all_equity_series) >= 2:
    st.header("6. Cross-Asset Correlation")

    eq_df = pd.DataFrame(all_equity_series).dropna()
    if len(eq_df) >= 10:
        full_corr = eq_df.corr()

        fig_fc = go.Figure(go.Heatmap(
            z=full_corr.values,
            x=[c.replace('_', ' ') for c in full_corr.columns],
            y=[c.replace('_', ' ') for c in full_corr.index],
            text=[[f"{v:.2f}" for v in row] for row in full_corr.values],
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
        ))
        fig_fc.update_layout(
            template="plotly_dark",
            height=max(300, 70 * len(full_corr)),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_fc, use_container_width=True, key="full_correlation")

        # Highlight pairs
        pairs_high = []
        pairs_low = []
        cols = full_corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = full_corr.iloc[i, j]
                pair = f"{cols[i]} / {cols[j]}"
                if val > 0.5:
                    pairs_high.append((pair, val))
                elif abs(val) < 0.1:
                    pairs_low.append((pair, val))

        if pairs_high:
            st.warning("Highly correlated pairs (>0.5): " + ", ".join(
                f"**{p}** ({v:.2f})" for p, v in pairs_high
            ))
        if pairs_low:
            st.success("Uncorrelated pairs (<0.1): " + ", ".join(
                f"**{p}** ({v:.2f})" for p, v in pairs_low
            ))


# ── Recommendations ───────────────────────────────────────────────────────

st.header("7. Recommendations")

recommendations = cross_strat.get('recommendations', [])
if recommendations:
    for rec in recommendations:
        st.write(f"- {rec}")
else:
    st.info("Run analysis with more strategy/asset combinations for recommendations.")

if cross_strat.get('best_overall'):
    st.caption(f"Best overall performance: **{cross_strat['best_overall']}**")
if cross_strat.get('most_robust'):
    st.caption(f"Most robust strategy: **{cross_strat['most_robust']}**")
