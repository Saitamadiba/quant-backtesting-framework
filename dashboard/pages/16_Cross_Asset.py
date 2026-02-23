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
        k1.metric("Robustness Score", f"{rob.robustness_score:.0f}/100")
        k2.metric("Grade", rob.robustness_grade)
        k3.metric("Assets Tested", len(rob.assets))

        st.info(rob.verdict)

        # Metrics table
        with st.expander("Side-by-Side Metrics", expanded=True):
            df_metrics = pd.DataFrame(rob.metrics_table)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # Equity curves
        with st.expander("Equity Curves", expanded=True):
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
