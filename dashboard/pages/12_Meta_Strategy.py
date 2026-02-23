"""Page 12: Meta-Strategy Selector — predict best strategy from market state."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Meta-Strategy Selector", page_icon="🧠", layout="wide")
st.title("🧠 Meta-Strategy Selector")
st.caption(
    "Train a classifier to predict which strategy will perform best given current "
    "market conditions. Compare dynamic allocation vs. static baselines."
)

# Ensure imports work
_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from backtrader_framework.optimization.persistence import list_wfo_results
from backtrader_framework.optimization.meta_strategy_selector import MetaStrategySelector

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
    "Select WFO results to combine (2+ strategies, ideally same symbol)",
    list(options.keys()),
    format_func=lambda i: options[i],
    default=[0, 1] if len(options) >= 2 else [],
    key="meta_select",
)

if len(selected_indices) < 2:
    st.info("Select at least 2 WFO results to build a meta-strategy selector.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.header("2. Configuration")

col_sym, col_tf, col_fwd = st.columns(3)

# Infer symbol from selected results
selected_symbols = set(saved[i]['symbol'] for i in selected_indices)
with col_sym:
    symbol = st.selectbox(
        "Symbol for market features",
        sorted(selected_symbols),
        key="meta_symbol",
        help="OHLCV data will be loaded from DuckDB for this symbol.",
    )

with col_tf:
    timeframe = st.selectbox(
        "Feature timeframe",
        ['4h', '1h', '15m'],
        index=0,
        key="meta_timeframe",
        help="Timeframe for computing market features (4h recommended).",
    )

with col_fwd:
    lookforward = st.slider(
        "Lookforward window (days)",
        min_value=1, max_value=30, value=7, step=1,
        key="meta_lookforward",
        help="How many days ahead to evaluate strategy performance for labeling.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN & BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

if st.button("Train & Backtest", type="primary"):
    filepaths = [saved[i]['filepath'] for i in selected_indices]

    try:
        with st.spinner("Loading OHLCV data from DuckDB..."):
            ohlcv = MetaStrategySelector.load_ohlcv_from_duckdb(symbol, timeframe)

        with st.spinner("Building dataset (aligning returns + features)..."):
            selector = MetaStrategySelector(lookforward_days=lookforward)
            dataset = selector.build_dataset(filepaths, ohlcv)

        with st.spinner("Training classifier..."):
            train_stats = selector.train(dataset)

        with st.spinner("Running walk-forward backtest..."):
            backtest_result = selector.backtest(dataset, min_train_days=60, retrain_every=20)

        with st.spinner("Computing regime × strategy heatmap..."):
            regime_heatmap = selector.get_regime_strategy_heatmap(dataset)

        st.session_state['meta_train_stats'] = train_stats
        st.session_state['meta_backtest'] = backtest_result
        st.session_state['meta_regime_heatmap'] = regime_heatmap
        st.session_state['meta_wfo_filepaths'] = filepaths
        st.session_state['meta_dataset_info'] = {
            'n_rows': len(dataset),
            'date_start': str(dataset.index[0].date()),
            'date_end': str(dataset.index[-1].date()),
            'strategies': selector.strategy_labels,
        }
        st.success("Training & backtesting complete!")

    except Exception as e:
        st.error(f"Meta-strategy selector failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if 'meta_train_stats' not in st.session_state:
    st.stop()

train_stats = st.session_state['meta_train_stats']
backtest_result = st.session_state.get('meta_backtest', {})
regime_heatmap = st.session_state.get('meta_regime_heatmap', {})
ds_info = st.session_state.get('meta_dataset_info', {})


# ── Training Results ──────────────────────────────────────────────

st.header("3. Training Results")

st.caption(
    f"Dataset: {ds_info.get('n_rows', '?')} days "
    f"({ds_info.get('date_start', '?')} to {ds_info.get('date_end', '?')})  |  "
    f"Strategies: {', '.join(ds_info.get('strategies', []))}"
)

# KPI row
tr_c1, tr_c2, tr_c3, tr_c4 = st.columns(4)
tr_c1.metric("Test Accuracy", f"{train_stats['accuracy']:.1%}")
tr_c2.metric(
    "CV Score",
    f"{train_stats['cv_mean']:.1%}",
    delta=f"±{train_stats['cv_std']:.1%}",
)
tr_c3.metric("Classes", train_stats['n_classes'])
tr_c4.metric("Train / Test", f"{train_stats['n_train']} / {train_stats['n_test']}")

# Confusion matrix
with st.expander("Confusion Matrix", expanded=True):
    cm = np.array(train_stats['confusion_matrix'])
    classes = train_stats['classes']

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=[c.replace('_', ' ') for c in classes],
        y=[c.replace('_', ' ') for c in classes],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Count"),
    ))
    fig_cm.update_layout(
        template="plotly_dark",
        height=max(300, 80 * len(classes)),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_cm, use_container_width=True, key="meta_cm")

# Feature importance
with st.expander("Feature Importance (Top 15)", expanded=True):
    feat_imp = train_stats['feature_importances'][:15]
    names = [f[0].replace('_', ' ') for f in feat_imp]
    values = [f[1] for f in feat_imp]

    fig_fi = go.Figure(go.Bar(
        x=values[::-1],
        y=names[::-1],
        orientation='h',
        marker_color='#4CAF50',
    ))
    fig_fi.update_layout(
        template="plotly_dark",
        height=max(300, 28 * len(names)),
        xaxis_title="Importance",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True, key="meta_fi")

# Class distribution
with st.expander("Class Distribution", expanded=False):
    dist_c1, dist_c2 = st.columns(2)
    with dist_c1:
        st.write("**Training set:**")
        dist_train = train_stats.get('class_distribution_train', {})
        for cls, cnt in sorted(dist_train.items(), key=lambda x: -x[1]):
            st.write(f"- {cls.replace('_', ' ')}: {cnt}")
    with dist_c2:
        st.write("**Test set:**")
        dist_test = train_stats.get('class_distribution_test', {})
        for cls, cnt in sorted(dist_test.items(), key=lambda x: -x[1]):
            st.write(f"- {cls.replace('_', ' ')}: {cnt}")


# ── Backtest Comparison ───────────────────────────────────────────

if backtest_result.get('valid'):
    st.header("4. Backtest Comparison")

    # KPI row
    bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
    bt_c1.metric(
        "Meta (Soft) Sharpe",
        f"{backtest_result['meta_soft']['sharpe_annual']:.3f}",
    )
    bt_c2.metric(
        "Equal Weight Sharpe",
        f"{backtest_result['equal_weight']['sharpe_annual']:.3f}",
    )
    improvement = (
        backtest_result['meta_soft']['total_r'] - backtest_result['equal_weight']['total_r']
    )
    bt_c3.metric(
        "Meta vs Equal (R)",
        f"{improvement:+.1f}",
        delta=f"{'better' if improvement > 0 else 'worse'}",
        delta_color="normal" if improvement > 0 else "inverse",
    )
    bt_c4.metric(
        "Prediction Accuracy",
        f"{backtest_result['prediction_accuracy']:.1%}",
    )

    # Equity curves
    with st.expander("Equity Curves", expanded=True):
        dates = pd.to_datetime(backtest_result['dates'])

        fig_eq = go.Figure()

        styles = {
            'meta_hard': ('Meta-Selector (Hard)', '#FFD700', 2.5, None),
            'meta_soft': ('Meta-Selector (Soft)', '#2196F3', 3, None),
            'equal_weight': ('Equal Weight', '#9E9E9E', 2, 'dash'),
            'best_single': ('Best Single (Oracle)', '#4CAF50', 1.5, 'dot'),
        }

        for key, (name, color, width, dash) in styles.items():
            cum = backtest_result[key]['cumulative']
            fig_eq.add_trace(go.Scatter(
                x=dates, y=cum, mode='lines', name=name,
                line=dict(color=color, width=width, dash=dash),
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
        st.plotly_chart(fig_eq, use_container_width=True, key="meta_equity")

    # Metrics comparison table
    with st.expander("Detailed Metrics", expanded=True):
        comp_rows = []
        for key in ['meta_hard', 'meta_soft', 'equal_weight', 'best_single']:
            m = backtest_result[key]
            comp_rows.append({
                'Method': m['name'],
                'Sharpe (Annual)': f"{m['sharpe_annual']:.3f}",
                'Total R': f"{m['total_r']:.2f}",
                'Max DD (R)': f"{m['max_drawdown']:.2f}",
                'Win Rate': f"{m['win_rate']:.1%}",
                'Mean R/Day': f"{m['mean_r']:.4f}",
                'Days': m['n_periods'],
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Strategy selection timeline
    timeline = backtest_result.get('selection_timeline', [])
    if timeline:
        with st.expander("Strategy Selection Timeline", expanded=False):
            # Build stacked area data: date → strategy probability
            strat_labels = backtest_result.get('strategy_labels', [])
            all_labels = strat_labels + ['none']

            timeline_dates = pd.to_datetime([t['date'] for t in timeline])
            prob_data = {label: [] for label in all_labels}
            for t in timeline:
                probs = t.get('probabilities', {})
                for label in all_labels:
                    prob_data[label].append(probs.get(label, 0.0))

            fig_tl = go.Figure()
            colors_tl = px.colors.qualitative.Set2
            for i, label in enumerate(all_labels):
                fig_tl.add_trace(go.Scatter(
                    x=timeline_dates,
                    y=prob_data[label],
                    mode='lines',
                    name=label.replace('_', ' '),
                    stackgroup='one',
                    line=dict(width=0.5, color=colors_tl[i % len(colors_tl)]),
                ))

            fig_tl.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Date",
                yaxis_title="Allocation Probability",
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_tl, use_container_width=True, key="meta_timeline")


# ── Regime × Strategy Heatmap ─────────────────────────────────────

if regime_heatmap.get('valid'):
    st.header("5. Regime × Strategy Preferences")
    st.caption(
        "How the classifier allocates across strategies in each market regime. "
        "Higher values indicate the selector prefers that strategy in that regime."
    )

    regimes = regime_heatmap['regimes']
    strategies = regime_heatmap['strategies']
    matrix = regime_heatmap['matrix']

    z_data = []
    text_data = []
    for regime in regimes:
        row_z = []
        row_t = []
        for strat in strategies:
            val = matrix.get(regime, {}).get(strat, 0)
            row_z.append(val)
            row_t.append(f"{val:.0%}")
        z_data.append(row_z)
        text_data.append(row_t)

    fig_rs = go.Figure(go.Heatmap(
        z=z_data,
        x=[s.replace('_', ' ') for s in strategies],
        y=[r.replace('_', ' ') for r in regimes],
        text=text_data,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale='YlOrRd',
        zmin=0, zmax=0.6,
        showscale=True,
        colorbar=dict(title="Selection %"),
    ))
    fig_rs.update_layout(
        template="plotly_dark",
        height=max(250, 70 * len(regimes)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_rs, use_container_width=True, key="meta_regime_strat")

    # Key insight
    for regime in regimes:
        strats = matrix.get(regime, {})
        best_strat = max(strats, key=strats.get) if strats else None
        if best_strat and strats.get(best_strat, 0) > 0.3:
            st.info(
                f"In **{regime.replace('_', ' ')}** markets, the selector "
                f"prefers **{best_strat.replace('_', ' ')}** ({strats[best_strat]:.0%} of the time)."
            )
