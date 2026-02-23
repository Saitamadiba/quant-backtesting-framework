"""Page 14: Bayesian Hyperparameter Tuning with Optuna."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Bayesian Tuning", page_icon="🎯", layout="wide")
st.title("🎯 Bayesian Hyperparameter Tuning")
st.caption(
    "Use Optuna's Bayesian optimization (TPE) to find optimal hyperparameters "
    "for ML classifiers. Replaces manual grid search with intelligent exploration."
)

# Ensure imports work
_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREREQUISITES
# ══════════════════════════════════════════════════════════════════════════════

st.header("1. Prerequisites")

has_meta = (
    'meta_train_stats' in st.session_state
    and st.session_state.get('meta_dataset_info')
)

if not has_meta:
    st.warning(
        "No meta-strategy data available. Run the **Meta-Strategy Selector** page "
        "(page 12) with 'Train & Backtest' first to generate training data."
    )
    st.stop()

ds_info = st.session_state['meta_dataset_info']
train_stats = st.session_state['meta_train_stats']

baseline_acc = train_stats.get('accuracy', 0)
baseline_cv = train_stats.get('cv_mean', 0)

st.success(
    f"Meta-strategy data loaded  |  "
    f"{ds_info.get('n_rows', '?')} samples  |  "
    f"Baseline accuracy: {baseline_acc:.1%}  |  "
    f"Baseline CV: {baseline_cv:.1%}"
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.header("2. Tuning Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    n_trials = st.slider(
        "Number of trials", min_value=10, max_value=200, value=50, step=10,
        help="More trials = better optimization but slower. 50 is usually enough.",
    )

with col2:
    sampler = st.selectbox(
        "Sampler",
        ['tpe', 'random', 'cmaes'],
        index=0,
        format_func=lambda x: {'tpe': 'TPE (Recommended)', 'random': 'Random', 'cmaes': 'CMA-ES'}[x],
        help="TPE (Tree-structured Parzen Estimator) is the default Bayesian method.",
    )

with col3:
    scoring = st.selectbox(
        "Scoring metric",
        ['accuracy', 'f1_weighted'],
        index=0,
        help="Metric to optimize during cross-validation.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. RUN TUNING
# ══════════════════════════════════════════════════════════════════════════════

if st.button("Run Bayesian Tuning", type="primary"):
    try:
        from backtrader_framework.optimization.bayesian_tuner import OptunaTuner, TunerConfig
        from backtrader_framework.optimization.meta_strategy_selector import MetaStrategySelector

        if not OptunaTuner.is_available():
            st.error("Optuna is not installed. Run: `pip install optuna`")
            st.stop()

        # Reload the selector and dataset
        filepaths = st.session_state.get('meta_wfo_filepaths', [])
        if not filepaths:
            st.error(
                "WFO file paths not found in session state. "
                "Please re-run the Meta-Strategy Selector page first."
            )
            st.stop()

        symbol = st.session_state.get('meta_symbol', 'BTC')
        timeframe = st.session_state.get('meta_timeframe', '4h')
        lookforward = st.session_state.get('meta_lookforward', 7)

        with st.spinner("Loading OHLCV data..."):
            ohlcv = MetaStrategySelector.load_ohlcv_from_duckdb(symbol, timeframe)

        with st.spinner("Building dataset..."):
            selector = MetaStrategySelector(lookforward_days=lookforward)
            dataset = selector.build_dataset(filepaths, ohlcv)

        config = TunerConfig(
            n_trials=n_trials,
            sampler=sampler,
            scoring_metric=scoring,
        )

        with st.spinner(f"Running {n_trials} Optuna trials..."):
            train_result = selector.train(
                dataset,
                use_bayesian_tuning=True,
                tuner_config=config,
            )

        st.session_state['bayesian_result'] = train_result.get('bayesian_tuning', {})
        st.session_state['bayesian_train_stats'] = train_result
        st.session_state['bayesian_baseline'] = {
            'accuracy': baseline_acc,
            'cv_mean': baseline_cv,
        }
        st.success("Bayesian tuning complete!")

    except Exception as e:
        st.error(f"Tuning failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if 'bayesian_result' not in st.session_state:
    st.info("Configure settings above and click 'Run Bayesian Tuning' to start.")
    st.stop()

bay = st.session_state['bayesian_result']
bay_train = st.session_state.get('bayesian_train_stats', {})
baseline = st.session_state.get('bayesian_baseline', {})

if not bay.get('valid'):
    st.warning("No valid tuning results.")
    st.stop()


# ── KPI Row ──────────────────────────────────────────────────────────────────

st.header("3. Results Summary")

k1, k2, k3, k4 = st.columns(4)
tuned_acc = bay_train.get('accuracy', 0)
tuned_cv = bay_train.get('cv_mean', 0)
base_acc = baseline.get('accuracy', 0)
base_cv = baseline.get('cv_mean', 0)

k1.metric("Tuned Accuracy", f"{tuned_acc:.1%}", f"{tuned_acc - base_acc:+.1%}")
k2.metric("Tuned CV Mean", f"{tuned_cv:.1%}", f"{tuned_cv - base_cv:+.1%}")
k3.metric("Trials Completed", bay.get('n_trials_completed', 0))
k4.metric("Trials Pruned", bay.get('n_trials_pruned', 0))


# ── Best Parameters ──────────────────────────────────────────────────────────

st.header("4. Best Parameters Found")

best_params = bay.get('best_params', bay_train.get('bayesian_tuning', {}).get('best_params', {}))
if not best_params and 'best_params' in bay:
    best_params = bay['best_params']

DEFAULTS = {
    'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2,
    'min_samples_leaf': 1, 'max_features': 'sqrt',
    'class_weight': 'balanced',
}

if best_params:
    rows = []
    for param, value in best_params.items():
        default = DEFAULTS.get(param, '—')
        rows.append({
            'Parameter': param,
            'Tuned Value': str(value),
            'Default': str(default),
            'Changed': '✓' if str(value) != str(default) else '',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Optimization History ─────────────────────────────────────────────────────

st.header("5. Optimization History")

history = bay.get('optimization_history', [])
convergence = bay.get('convergence', [])

if history:
    with st.expander("Trial Scores & Convergence", expanded=True):
        trial_nums = [h['trial_number'] for h in history]
        trial_vals = [h['value'] for h in history]

        fig_hist = go.Figure()

        # Individual trials
        fig_hist.add_trace(go.Scatter(
            x=trial_nums, y=trial_vals,
            mode='markers',
            marker=dict(size=6, color='#2196F3', opacity=0.5),
            name='Trial Score',
            hovertemplate="Trial %{x}: %{y:.4f}<extra></extra>",
        ))

        # Convergence line (running best)
        if convergence:
            conv_x = [c['trial'] for c in convergence]
            conv_y = [c['best_value'] for c in convergence]
            fig_hist.add_trace(go.Scatter(
                x=conv_x, y=conv_y,
                mode='lines',
                line=dict(color='#4CAF50', width=2),
                name='Best So Far',
                hovertemplate="Best at trial %{x}: %{y:.4f}<extra></extra>",
            ))

        # Mark best trial
        best_trial = bay.get('best_trial_number')
        best_value = bay.get('best_value', bay.get('best_score'))
        if best_trial is not None and best_value is not None:
            fig_hist.add_trace(go.Scatter(
                x=[best_trial], y=[best_value],
                mode='markers',
                marker=dict(size=14, color='#FF9800', symbol='star'),
                name=f'Best (trial {best_trial})',
                hovertemplate=f"Best: trial {best_trial}, score {best_value:.4f}<extra></extra>",
            ))

        fig_hist.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Trial Number",
            yaxis_title="Objective Score",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="bay_history")


# ── Parameter Importance ──────────────────────────────────────────────────────

st.header("6. Parameter Importance")

param_imp = bay.get('param_importances', {})
if param_imp:
    with st.expander("Which hyperparameters matter most?", expanded=True):
        names = list(param_imp.keys())
        values = list(param_imp.values())

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=values[::-1], y=names[::-1],
            orientation='h',
            marker_color='#9C27B0',
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            template="plotly_dark",
            height=max(250, 35 * len(names)),
            xaxis_title="Importance",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True, key="bay_imp")
else:
    st.info("Parameter importance not available (need more completed trials).")


# ── Trial Duration ────────────────────────────────────────────────────────────

if history:
    durations = [h.get('duration_seconds') for h in history if h.get('duration_seconds')]
    if durations:
        avg_dur = np.mean(durations)
        total_dur = sum(durations)
        st.caption(
            f"Average trial: {avg_dur:.2f}s  |  "
            f"Total time: {total_dur:.1f}s  |  "
            f"{len(history)} trials completed"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. APPLY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.header("7. Apply Results")
st.caption(
    "Apply the tuned hyperparameters to the meta-strategy selector. "
    "This updates pages 12 (Meta-Strategy) and 13 (SHAP Analysis)."
)

if st.button("Apply Tuned Model to Session"):
    st.session_state['meta_train_stats'] = bay_train
    st.success(
        "Tuned model applied! Pages 12 and 13 now reflect the Bayesian-tuned model."
    )
