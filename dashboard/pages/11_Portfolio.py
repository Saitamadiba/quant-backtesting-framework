"""Page 11: Portfolio-Level Optimization."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Portfolio Optimization", page_icon="📊", layout="wide")
st.title("📊 Portfolio Optimization")
st.caption(
    "Combine multiple WFO results to analyze portfolio-level performance. "
    "Select strategy/symbol/timeframe combinations, choose an allocation method, "
    "and see the combined Sharpe, drawdown, correlation, and diversification metrics."
)

# Ensure imports work
_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from backtrader_framework.optimization.persistence import list_wfo_results
from backtrader_framework.optimization.portfolio_optimizer import PortfolioOptimizer

# ══════════════════════════════════════════════════════════════════════════════
# 1. SELECT WFO RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.header("1. Select WFO Results")

saved = list_wfo_results()
if len(saved) < 2:
    st.warning("Need at least 2 saved WFO results. Run WFO optimizations first.")
    st.stop()

# Build display labels
options = {}
for i, r in enumerate(saved):
    label = f"{r['strategy']}  /  {r['symbol']}  /  {r['timeframe']}  —  {r['timestamp']}"
    options[i] = label

selected_indices = st.multiselect(
    "Select WFO results to combine (2 or more)",
    list(options.keys()),
    format_func=lambda i: options[i],
    default=[0, 1] if len(options) >= 2 else [],
    key="portfolio_select",
)

if len(selected_indices) < 2:
    st.info("Select at least 2 WFO results to build a portfolio.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.header("2. Configuration")

col_method, col_kelly = st.columns([3, 1])
with col_method:
    method = st.selectbox(
        "Allocation Method",
        ['risk_parity', 'equal', 'kelly', 'max_sharpe'],
        index=0,
        format_func=lambda m: {
            'equal': 'Equal Weight (1/N baseline)',
            'risk_parity': 'Risk Parity (inverse-volatility)',
            'kelly': 'Kelly Criterion (fractional)',
            'max_sharpe': 'Mean-Variance (maximize Sharpe)',
        }[m],
        key="portfolio_method",
    )
with col_kelly:
    frac_kelly = st.slider(
        "Fractional Kelly",
        min_value=0.10, max_value=1.0, value=0.25, step=0.05,
        key="portfolio_frac_kelly",
        help="Kelly fraction multiplier. 0.25 = quarter Kelly (industry standard).",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

if st.button("Optimize Portfolio", type="primary"):
    filepaths = [saved[i]['filepath'] for i in selected_indices]

    try:
        with st.spinner("Loading & aligning equity curves..."):
            optimizer = PortfolioOptimizer(fractional_kelly=frac_kelly)

            # Run primary method
            result = optimizer.optimize(filepaths, method=method)

            # Also run all methods for comparison
            comparison = optimizer.compare_all_methods(filepaths)

        st.session_state['portfolio_result'] = result
        st.session_state['portfolio_comparison'] = comparison
        st.success("Portfolio optimization complete!")

    except Exception as e:
        st.error(f"Portfolio optimization failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if 'portfolio_result' not in st.session_state:
    st.stop()

result = st.session_state['portfolio_result']
comparison = st.session_state.get('portfolio_comparison', {})
ps = result.portfolio_stats

# ── KPI Row ─────────────────────────────────────────────────────

st.header("3. Portfolio Results")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Sharpe (Annual)", f"{ps['sharpe_annual']:.3f}")
k2.metric("Max Drawdown", f"{ps['max_drawdown_r']:.2f} R")
k3.metric("Total R", f"{ps['total_r']:.2f}")
k4.metric("Diversification Ratio", f"{result.diversification_ratio:.2f}")
k5.metric("Overlap Days", f"{ps['n_days']}")

st.caption(f"Period: {ps['date_start']} to {ps['date_end']}  |  Method: {result.allocation_method}")

# ── Allocation Weights Table ────────────────────────────────────

st.subheader("Allocation Weights")

weight_rows = []
for label, stats in result.component_stats.items():
    weight_rows.append({
        'Component': label,
        'Strategy': stats['strategy'],
        'Symbol': stats['symbol'],
        'Timeframe': stats['timeframe'],
        'Weight': f"{stats['weight']:.1%}",
        'OOS Trades': stats['n_trades'],
        'Total R': f"{stats['total_r']:.2f}",
        'Annual Sharpe': f"{stats['annual_sharpe']:.3f}",
        'Mean R/Trade': f"{stats['mean_r_per_trade']:.4f}",
    })

weight_df = pd.DataFrame(weight_rows)
st.dataframe(weight_df, use_container_width=True, hide_index=True)

if result.kelly_fractions:
    with st.expander("Kelly Criterion Fractions (informational)"):
        kelly_rows = []
        for label, frac in result.kelly_fractions.items():
            r_vals = [e['r'] for e in next(
                c for c in result.components if c.label == label
            ).oos_equity]
            mean_r = np.mean(r_vals) if r_vals else 0
            kelly_rows.append({
                'Component': label,
                'Mean R': f"{mean_r:.4f}",
                'Kelly Fraction': f"{frac:.1%}",
                'Note': 'Positive edge' if mean_r > 0 else 'No edge (fallback to equal)',
            })
        st.dataframe(pd.DataFrame(kelly_rows), use_container_width=True, hide_index=True)


# ── Correlation Heatmap ─────────────────────────────────────────

st.subheader("Correlation Matrix")
st.caption(
    "Pairwise Pearson correlation of daily R returns. "
    "Low correlation (near 0) between components means better diversification."
)

corr = result.correlation_matrix
fig_corr = go.Figure(go.Heatmap(
    z=corr.values,
    x=[c.replace('_', ' ') for c in corr.columns],
    y=[c.replace('_', ' ') for c in corr.index],
    text=np.round(corr.values, 3),
    texttemplate="%{text}",
    textfont={"size": 14},
    colorscale='RdBu_r',
    zmin=-1, zmax=1,
    showscale=True,
    colorbar=dict(title="Correlation"),
))
fig_corr.update_layout(
    template="plotly_dark",
    height=max(350, 100 * len(corr)),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_corr, use_container_width=True, key="portfolio_corr")

avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
st.caption(f"Average off-diagonal correlation: **{avg_corr:.3f}**")


# ── Combined Equity Curve ──────────────────────────────────────

st.subheader("Combined Equity Curve")

fig_eq = go.Figure()

# Individual component equity curves (thin, dashed)
colors = px.colors.qualitative.Set2
for i, comp in enumerate(result.components):
    eq = comp.oos_equity
    times = pd.to_datetime([e['time'] for e in eq])
    cum_r = [e['cumulative_r'] for e in eq]
    w = result.weights.get(comp.label, 0)
    fig_eq.add_trace(go.Scatter(
        x=times,
        y=cum_r,
        mode='lines',
        name=f"{comp.label.replace('_', ' ')} ({w:.0%})",
        line=dict(color=colors[i % len(colors)], width=1.5, dash='dot'),
        opacity=0.6,
    ))

# Combined portfolio equity curve (thick gold)
port_times = pd.to_datetime([e['time'] for e in result.combined_equity])
port_cum = [e['cumulative_r'] for e in result.combined_equity]
fig_eq.add_trace(go.Scatter(
    x=port_times,
    y=port_cum,
    mode='lines',
    name=f'Portfolio ({result.allocation_method})',
    line=dict(color='#FFD700', width=3),
))

fig_eq.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
fig_eq.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Date",
    yaxis_title="Cumulative R-Multiple",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig_eq, use_container_width=True, key="portfolio_equity")


# ── Method Comparison Table ─────────────────────────────────────

if comparison:
    st.subheader("Allocation Method Comparison")
    st.caption("All 4 methods compared side-by-side on the same data.")

    comp_rows = []
    for m_name, m_result in comparison.items():
        ms = m_result.portfolio_stats
        comp_rows.append({
            'Method': {
                'equal': 'Equal Weight',
                'risk_parity': 'Risk Parity',
                'kelly': 'Kelly Criterion',
                'max_sharpe': 'Max Sharpe (MV)',
            }.get(m_name, m_name),
            'Sharpe (Annual)': f"{ms['sharpe_annual']:.3f}",
            'Max DD (R)': f"{ms['max_drawdown_r']:.2f}",
            'Total R': f"{ms['total_r']:.2f}",
            'Div. Ratio': f"{m_result.diversification_ratio:.2f}",
            'Weights': ', '.join(
                f"{k.split('_')[0]}={v:.0%}" for k, v in m_result.weights.items()
            ),
        })

    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ── Monte Carlo ─────────────────────────────────────────────────

if result.monte_carlo and result.monte_carlo.get('valid'):
    st.subheader("Monte Carlo Analysis (Portfolio)")

    mc = result.monte_carlo

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("P(Profitable)", f"{mc['p_profitable']:.1%}")
    mc2.metric("5th %ile Final R", f"{mc['pct_5_final_r']:.2f}")
    mc3.metric("95th %ile Max DD", f"{mc['pct_95_max_dd']:.2f} R")

    with st.expander("Confidence Intervals", expanded=False):
        ci_rows = [
            {
                'Metric': 'Final R',
                f"{mc['confidence']:.0%} CI Lower": f"{mc['final_r_ci'][0]:.2f}",
                f"{mc['confidence']:.0%} CI Upper": f"{mc['final_r_ci'][1]:.2f}",
                'Median': f"{mc['median_final_r']:.2f}",
            },
            {
                'Metric': 'Max Drawdown (R)',
                f"{mc['confidence']:.0%} CI Lower": f"{mc['max_dd_ci'][0]:.2f}",
                f"{mc['confidence']:.0%} CI Upper": f"{mc['max_dd_ci'][1]:.2f}",
                'Median': '—',
            },
            {
                'Metric': 'Sharpe (Daily)',
                f"{mc['confidence']:.0%} CI Lower": f"{mc['sharpe_ci'][0]:.4f}",
                f"{mc['confidence']:.0%} CI Upper": f"{mc['sharpe_ci'][1]:.4f}",
                'Median': '—',
            },
        ]
        st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

    st.caption(f"Based on {mc['n_resamples']:,} bootstrap resamples of daily portfolio returns.")
