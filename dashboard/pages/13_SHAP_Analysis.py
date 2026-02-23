"""Page 13: SHAP Feature Importance Analysis."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="SHAP Analysis", page_icon="🔍", layout="wide")
st.title("🔍 SHAP Feature Importance Analysis")
st.caption(
    "Understand *why* ML models make specific predictions using SHAP values. "
    "See which features drive predictions, how feature values interact, "
    "and get actionable trading insights."
)

# Ensure imports work
_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from backtrader_framework.optimization.shap_analysis import (
    FEATURE_CATEGORIES, CATEGORY_COLORS,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA SOURCE
# ══════════════════════════════════════════════════════════════════════════════

st.header("1. Data Source")

# Check if meta-strategy results are in session state
has_meta = (
    'meta_train_stats' in st.session_state
    and st.session_state['meta_train_stats'].get('shap_analysis')
)

if not has_meta:
    st.warning(
        "No SHAP analysis data available. Run the **Meta-Strategy Selector** page first "
        "(page 12) with 'Train & Backtest' to generate SHAP values."
    )
    st.info(
        "The SHAP analysis is automatically computed when you train the meta-strategy "
        "selector. Go to page 12, select WFO results, and click 'Train & Backtest'."
    )
    st.stop()

shap_data = st.session_state['meta_train_stats']['shap_analysis']
train_stats = st.session_state['meta_train_stats']
ds_info = st.session_state.get('meta_dataset_info', {})

if not shap_data.get('valid'):
    st.error("SHAP analysis data is invalid.")
    st.stop()

st.success(
    f"SHAP analysis loaded from Meta-Strategy Selector  |  "
    f"{shap_data['n_samples']} samples, {shap_data['n_features']} features  |  "
    f"Explainer: {shap_data['explainer_type']}"
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

st.header("2. Global Feature Importance")

with st.expander("SHAP vs Gini Importance", expanded=True):
    gi = shap_data['global_importance']
    gini_dict = dict(train_stats.get('feature_importances', []))

    # Top 15
    top_gi = gi[:15]
    names = [g['feature'].replace('_', ' ') for g in top_gi]
    shap_vals = [g['importance'] for g in top_gi]
    gini_vals = [gini_dict.get(g['feature'], 0) for g in top_gi]
    categories = [g.get('category', 'Other') for g in top_gi]
    colors = [CATEGORY_COLORS.get(cat, '#9E9E9E') for cat in categories]

    fig_gi = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("SHAP Importance (mean |SHAP|)", "Gini Importance"),
        horizontal_spacing=0.05,
    )

    fig_gi.add_trace(go.Bar(
        x=shap_vals[::-1], y=names[::-1], orientation='h',
        marker_color=colors[::-1], name='SHAP',
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ), row=1, col=1)

    fig_gi.add_trace(go.Bar(
        x=gini_vals[::-1], y=names[::-1], orientation='h',
        marker_color=[c + '80' for c in colors[::-1]], name='Gini',
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ), row=1, col=2)

    fig_gi.update_layout(
        template="plotly_dark",
        height=max(400, 30 * len(names)),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_gi, use_container_width=True, key="shap_gi")

    # Category legend
    cat_counts = {}
    for g in top_gi:
        cat = g.get('category', 'Other')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    legend_parts = [
        f"**{cat}** ({cnt})" for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])
    ]
    st.caption("Feature categories: " + " | ".join(legend_parts))


# ══════════════════════════════════════════════════════════════════════════════
# 3. BEESWARM SUMMARY PLOT
# ══════════════════════════════════════════════════════════════════════════════

st.header("3. SHAP Summary (Beeswarm)")
st.caption(
    "Each dot is a sample. X-axis = SHAP value (impact on prediction). "
    "Color = feature value (red = high, blue = low). "
    "If red dots cluster right, high feature values push prediction positive."
)

with st.expander("SHAP Beeswarm Plot", expanded=True):
    shap_sample = np.array(shap_data.get('shap_values_sample', []))
    X_sample = np.array(shap_data.get('X_sample', []))
    feat_names = shap_data.get('feature_names', [])

    if shap_sample.size > 0 and X_sample.size > 0:
        # Rank features by mean |SHAP|
        mean_abs = np.mean(np.abs(shap_sample), axis=0)
        top_n = min(15, len(feat_names))
        top_idx = np.argsort(mean_abs)[::-1][:top_n]

        fig_bee = go.Figure()
        for plot_rank, fi in enumerate(top_idx):
            fi = int(fi)
            shap_col = shap_sample[:, fi]
            feat_col = X_sample[:, fi]

            # Normalize feature values for color
            fmin, fmax = np.nanmin(feat_col), np.nanmax(feat_col)
            if fmax > fmin:
                feat_norm = (feat_col - fmin) / (fmax - fmin)
            else:
                feat_norm = np.zeros_like(feat_col)

            # Jitter y-axis for visibility
            jitter = np.random.RandomState(fi).uniform(-0.3, 0.3, len(shap_col))

            fig_bee.add_trace(go.Scatter(
                x=shap_col,
                y=[top_n - 1 - plot_rank + j for j in jitter],
                mode='markers',
                marker=dict(
                    size=4,
                    color=feat_norm,
                    colorscale='RdBu_r',
                    showscale=(plot_rank == 0),
                    colorbar=dict(title="Feature Value", len=0.5) if plot_rank == 0 else None,
                    opacity=0.6,
                ),
                name=feat_names[fi].replace('_', ' '),
                hovertemplate=(
                    f"{feat_names[fi].replace('_', ' ')}<br>"
                    "SHAP: %{x:.4f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

        fig_bee.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

        ytick_vals = list(range(top_n))
        ytick_labels = [feat_names[int(top_idx[top_n - 1 - i])].replace('_', ' ') for i in range(top_n)]

        fig_bee.update_layout(
            template="plotly_dark",
            height=max(400, 35 * top_n),
            xaxis_title="SHAP Value (impact on prediction)",
            yaxis=dict(tickvals=ytick_vals, ticktext=ytick_labels),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_bee, use_container_width=True, key="shap_bee")
    else:
        st.info("No SHAP sample data available for beeswarm plot.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE DEPENDENCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

st.header("4. Feature Dependence")
st.caption(
    "How individual feature values affect predictions. "
    "Each point = one sample. Color = strongest interacting feature."
)

dep_data = shap_data.get('dependence', {})
if dep_data:
    dep_features = list(dep_data.keys())[:4]  # Top 4

    with st.expander("Dependence Plots (Top 4 Features)", expanded=True):
        cols = st.columns(2)
        for i, fname in enumerate(dep_features):
            d = dep_data[fname]
            col = cols[i % 2]

            with col:
                vals = d['values']
                shap_vals = d['shap_values']
                interact = d.get('interaction_feature', '')

                fig_dep = go.Figure()
                fig_dep.add_trace(go.Scatter(
                    x=vals, y=shap_vals,
                    mode='markers',
                    marker=dict(size=5, color='#2196F3', opacity=0.5),
                    hovertemplate=f"{fname}: " + "%{x:.3f}<br>SHAP: %{y:.4f}<extra></extra>",
                ))
                fig_dep.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_dep.update_layout(
                    template="plotly_dark",
                    height=280,
                    title=dict(text=fname.replace('_', ' '), font=dict(size=13)),
                    xaxis_title=fname.replace('_', ' '),
                    yaxis_title="SHAP value",
                    margin=dict(l=10, r=10, t=35, b=10),
                )
                st.plotly_chart(fig_dep, use_container_width=True, key=f"shap_dep_{i}")
                if interact:
                    st.caption(
                        f"Strongest interaction: {interact.replace('_', ' ')} "
                        f"(strength: {d.get('interaction_strength', 0):.2f})"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. ACTIONABLE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

insights = shap_data.get('insights', [])
if insights:
    st.header("5. Actionable Insights")
    st.caption(
        "Key patterns discovered from SHAP values. These are data-driven observations "
        "about which feature values most strongly affect model predictions."
    )

    for i, ins in enumerate(insights[:10]):
        strength = ins.get('strength', 0)
        ins_type = ins.get('type', '')

        if strength > 0.02:
            st.success(f"**{i+1}.** {ins['insight']}")
        elif strength > 0.01:
            st.warning(f"**{i+1}.** {ins['insight']}")
        else:
            st.info(f"**{i+1}.** {ins['insight']}")

    if len(insights) > 10:
        with st.expander(f"All {len(insights)} insights"):
            for i, ins in enumerate(insights[10:], start=11):
                st.write(f"{i}. {ins['insight']} (strength: {ins['strength']:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PER-CLASS BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

per_class = shap_data.get('per_class')
if per_class:
    st.header("6. Per-Class Feature Drivers")
    st.caption(
        "Which features most strongly drive the model toward predicting each class. "
        "This reveals what market conditions favor each strategy."
    )

    for cname, features in per_class.items():
        with st.expander(f"What drives '{cname.replace('_', ' ')}' predictions?", expanded=False):
            rows = []
            for f in features:
                rows.append({
                    'Feature': f['feature'].replace('_', ' '),
                    'Direction': f['direction'],
                    'Mean SHAP': f"{f['mean_shap']:+.4f}",
                    'Abs Importance': f"{f['abs_importance']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Interpretation
            top = features[0]
            if top['direction'] == 'positive':
                st.info(
                    f"The model predicts **{cname.replace('_', ' ')}** most strongly when "
                    f"**{top['feature'].replace('_', ' ')}** is high."
                )
            else:
                st.info(
                    f"The model predicts **{cname.replace('_', ' ')}** most strongly when "
                    f"**{top['feature'].replace('_', ' ')}** is low."
                )


# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

feature_summary = shap_data.get('feature_summary', {})
if feature_summary:
    with st.expander("Full Feature Summary Table", expanded=False):
        rows = []
        for fname, s in sorted(
            feature_summary.items(), key=lambda x: x[1]['mean_abs_shap'], reverse=True
        ):
            rows.append({
                'Feature': fname.replace('_', ' '),
                'Category': s.get('category', 'Other'),
                'Mean |SHAP|': f"{s['mean_abs_shap']:.4f}",
                'Std SHAP': f"{s['std_shap']:.4f}",
                'Positive Effect %': f"{s['positive_effect_frac']:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
