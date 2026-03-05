"""WFO Analysis — parameter deep-dive per strategy, asset, and timeframe."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from config import STRATEGY_COLORS, STRATEGIES
from components.charts import REGIME_COLORS
from data.wfo_loader import list_wfo_results, load_wfo_result, get_latest_wfo


def _format_ci(ci):
    """Format a [lower, upper] confidence interval."""
    if isinstance(ci, (list, tuple)) and len(ci) >= 2:
        return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    return "N/A"


def _bget(d, *keys, default=0):
    """Nested get for bayesian_edge dicts (e.g. d['win_rate']['posterior_mean'])."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


st.set_page_config(page_title="WFO Analysis", layout="wide")
st.title("Walk-Forward Optimization Analysis")

# ── Sidebar Selectors ─────────────────────────────────────────────────────────
results = list_wfo_results()
if not results:
    st.warning("No WFO results found in `backtrader_framework/optimization/results/`.")
    st.stop()

strategies = sorted({r["strategy"] for r in results})
sel_strategy = st.sidebar.selectbox("Strategy", strategies)

symbols = sorted({r["symbol"] for r in results if r["strategy"] == sel_strategy})
sel_symbol = st.sidebar.selectbox("Symbol", symbols)

timeframes = sorted(
    {r["timeframe"] for r in results
     if r["strategy"] == sel_strategy and r["symbol"] == sel_symbol}
)
sel_tf = st.sidebar.selectbox("Timeframe", timeframes)

# Load the latest result for this combo
data = get_latest_wfo(sel_strategy, sel_symbol, sel_tf)
if not data:
    st.error("Could not load WFO result.")
    st.stop()

oos = data.get("oos_stats", {})
bayesian = data.get("bayesian_edge", {})
mc = data.get("monte_carlo", {})
windows = data.get("windows", [])

# ── Header Metrics ────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("OOS Win Rate", f"{oos.get('win_rate', 0):.1%}")
c2.metric("OOS Sharpe", f"{oos.get('sharpe_per_trade', 0):.2f}")
c3.metric(
    "Overfit Ratio",
    f"{data.get('overfit_ratio', 0):.2f}",
    delta="Good" if data.get("overfit_ratio", 999) < 2.0 else "High",
    delta_color="normal" if data.get("overfit_ratio", 999) < 2.0 else "inverse",
)
_pbo_raw = data.get("pbo", 0)
_pbo_val = _pbo_raw.get("pbo", 0) if isinstance(_pbo_raw, dict) else _pbo_raw
c4.metric(
    "PBO",
    f"{_pbo_val:.1%}",
    delta="Low" if _pbo_val < 0.5 else "High",
    delta_color="normal" if _pbo_val < 0.5 else "inverse",
)
_p_edge = _bget(bayesian, "expectancy", "p_positive", default=None)
c5.metric(
    "P(Edge > 0)",
    f"{_p_edge:.1%}" if _p_edge is not None else "N/A",
)
c6.metric("OOS Trades", f"{oos.get('n_trades', 0)}")

# ── Parameter Stability Heatmap ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Parameter Stability Across WFO Windows")

if windows:
    param_keys = sorted(
        {k for w in windows if w.get("best_params") for k in w["best_params"]}
    )
    if param_keys:
        heatmap_data = []
        window_labels = []
        for w in windows:
            bp = w.get("best_params", {})
            heatmap_data.append([bp.get(k, np.nan) for k in param_keys])
            window_labels.append(f"W{w.get('id', '?')}")

        arr = np.array(heatmap_data, dtype=float).T
        # Normalize each param row to 0-1 for comparable coloring
        arr_norm = arr.copy()
        for i in range(arr_norm.shape[0]):
            row = arr_norm[i]
            rmin, rmax = np.nanmin(row), np.nanmax(row)
            if rmax > rmin:
                arr_norm[i] = (row - rmin) / (rmax - rmin)
            else:
                arr_norm[i] = 0.5

        # Custom text showing actual values
        text_vals = [[f"{arr[i, j]:.4g}" for j in range(arr.shape[1])]
                     for i in range(arr.shape[0])]

        fig_hm = go.Figure(data=go.Heatmap(
            z=arr_norm,
            x=window_labels,
            y=param_keys,
            text=text_vals,
            texttemplate="%{text}",
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Normalized"),
        ))
        fig_hm.update_layout(
            template="plotly_dark",
            height=max(250, len(param_keys) * 50),
            xaxis_title="WFO Window",
            yaxis_title="Parameter",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Stability score
        stability = data.get("param_stability")
        if isinstance(stability, dict):
            mean_stab = stability.get("mean_stability", 0)
            if mean_stab != float("-inf") and mean_stab != float("inf"):
                st.info(f"Parameter stability: **{mean_stab:.3f}** mean "
                        f"({stability.get('fragile_windows', 0)} fragile windows "
                        f"out of {stability.get('n_windows', 0)})")
            else:
                st.info(f"Parameter stability: {stability.get('fragile_windows', 0)} "
                        f"fragile windows out of {stability.get('n_windows', 0)}")
        elif stability is not None:
            st.info(f"Parameter stability score: **{stability:.3f}** "
                    f"(lower = more stable across windows)")
    else:
        st.info("No parameter data in WFO windows.")
else:
    st.info("No WFO windows found.")

# ── Cross-Asset Parameter Comparison ──────────────────────────────────────────
st.markdown("---")
st.subheader("Cross-Asset Parameter Comparison")
st.caption(f"Strategy: {sel_strategy} | Timeframe: {sel_tf}")

all_symbols = STRATEGIES.get(sel_strategy, {}).get("symbols", symbols)
cross_data = {}
for sym in all_symbols:
    r = get_latest_wfo(sel_strategy, sym, sel_tf)
    if r and r.get("windows"):
        # Use the last window's best params as "current optimal"
        last_w = r["windows"][-1]
        cross_data[sym] = last_w.get("best_params", {})

if len(cross_data) > 1:
    all_params = sorted({k for bp in cross_data.values() for k in bp})
    rows = []
    for sym, bp in cross_data.items():
        for p in all_params:
            rows.append({"Symbol": sym, "Parameter": p, "Value": bp.get(p, 0)})
    df_cross = pd.DataFrame(rows)

    fig_cross = px.bar(
        df_cross, x="Parameter", y="Value", color="Symbol",
        barmode="group",
        color_discrete_map={s: STRATEGY_COLORS.get(sel_strategy, "#888")
                            for s in cross_data},
        template="plotly_dark",
    )
    # Use distinct colors per symbol
    symbol_colors = {"BTC": "#F7931A", "ETH": "#627EEA", "NQ": "#00C853"}
    for trace in fig_cross.data:
        if trace.name in symbol_colors:
            trace.marker.color = symbol_colors[trace.name]
    fig_cross.update_layout(height=400)
    st.plotly_chart(fig_cross, use_container_width=True)

    # Callout text
    for p in all_params:
        vals = {s: bp.get(p) for s, bp in cross_data.items() if bp.get(p) is not None}
        if len(vals) > 1:
            items = [f"{s}: {v:.4g}" for s, v in vals.items()]
            st.caption(f"**{p}** — " + " | ".join(items))
elif len(cross_data) == 1:
    st.info(f"Only {list(cross_data.keys())[0]} has WFO results for {sel_strategy}/{sel_tf}.")
else:
    st.info(f"No WFO results found for other symbols with {sel_strategy}/{sel_tf}.")

# ── Regime Performance ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Regime Performance Breakdown")

regime_data = data.get("regime_analysis", {})
if regime_data:
    regime_rows = []
    for regime_name, stats in regime_data.items():
        if isinstance(stats, dict):
            regime_rows.append({
                "Regime": regime_name.replace("_", " ").title(),
                "Win Rate": stats.get("win_rate", 0),
                "Expectancy": stats.get("mean_r", stats.get("expectancy", 0)),
                "N Trades": stats.get("n_trades", 0),
            })

    if regime_rows:
        df_regime = pd.DataFrame(regime_rows)

        fig_reg = go.Figure()
        regime_colors_map = {
            "Trending Up": "rgba(76, 175, 80, 0.8)",
            "Trending Down": "rgba(244, 67, 54, 0.8)",
            "Ranging": "rgba(158, 158, 158, 0.8)",
            "Volatile": "rgba(255, 152, 0, 0.8)",
        }
        colors = [regime_colors_map.get(r, "#888") for r in df_regime["Regime"]]

        fig_reg.add_trace(go.Bar(
            x=df_regime["Regime"], y=df_regime["Win Rate"],
            name="Win Rate", marker_color=colors,
            text=[f"{v:.1%}" for v in df_regime["Win Rate"]],
            textposition="auto",
        ))
        fig_reg.update_layout(
            template="plotly_dark", height=350,
            yaxis_title="Win Rate", yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # Expectancy subplot
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Bar(
            x=df_regime["Regime"], y=df_regime["Expectancy"],
            marker_color=colors,
            text=[f"{v:.3f}R" for v in df_regime["Expectancy"]],
            textposition="auto",
        ))
        fig_exp.update_layout(
            template="plotly_dark", height=300,
            yaxis_title="Expectancy (R)",
        )
        st.plotly_chart(fig_exp, use_container_width=True)
    else:
        st.info("Regime data present but no parseable stats.")
else:
    st.info("No regime analysis in this WFO result.")

# ── Bayesian Edge ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Bayesian Edge Estimation")

if bayesian:
    bc1, bc2, bc3, bc4 = st.columns(4)
    _wr_post = _bget(bayesian, "win_rate", "posterior_mean")
    _mr_post = _bget(bayesian, "mean_r", "posterior_mean")
    _kelly = _bget(bayesian, "kelly_fraction", "posterior_mean")
    _shrink = _bget(bayesian, "shrinkage", "wr_shrinkage")
    bc1.metric("Win Rate (Posterior)", f"{_wr_post:.1%}")
    bc2.metric("Mean R (Posterior)", f"{_mr_post:.3f}")
    bc3.metric("Kelly Fraction", f"{_kelly:.1%}")
    bc4.metric("Shrinkage", f"{_shrink:.2f}")

    # P(Edge > 0) from expectancy posterior
    _p_exp = _bget(bayesian, "expectancy", "p_positive", default=None)
    if _p_exp is not None:
        color = "success" if _p_exp > 0.7 else "warning" if _p_exp > 0.5 else "error"
        getattr(st, color)(
            f"**P(Expectancy > 0) = {_p_exp:.1%}** — "
            f"{'Strong evidence of edge' if _p_exp > 0.95 else 'Moderate evidence' if _p_exp > 0.7 else 'Weak evidence'}"
        )

    # Posterior visualization: Beta distribution for win rate
    wr = _wr_post if _wr_post else 0.5
    n = bayesian.get("n_trades", 50)
    alpha_post = max(1, int(wr * n))
    beta_post = max(1, n - alpha_post)
    x = np.linspace(0.01, 0.99, 200)
    try:
        from scipy.stats import beta as beta_dist
        y_prior = beta_dist.pdf(x, 5, 5)  # Neutral prior
        y_post = beta_dist.pdf(x, alpha_post, beta_post)
        fig_bayes = go.Figure()
        fig_bayes.add_trace(go.Scatter(
            x=x, y=y_prior, mode="lines", name="Prior (neutral)",
            line=dict(color="gray", dash="dash"),
        ))
        fig_bayes.add_trace(go.Scatter(
            x=x, y=y_post, mode="lines", name="Posterior",
            fill="tozeroy", line=dict(color="#2196F3"),
        ))
        fig_bayes.add_vline(x=0.5, line_dash="dot", line_color="red",
                            annotation_text="Break-even")
        fig_bayes.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="Win Rate", yaxis_title="Density",
            title="Win Rate Posterior Distribution",
        )
        st.plotly_chart(fig_bayes, use_container_width=True)
    except ImportError:
        st.info("Install scipy for posterior visualization.")
else:
    st.info("No Bayesian edge data in this WFO result.")

# ── Monte Carlo ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Monte Carlo Simulation Results")

if mc and mc.get("valid"):
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        "P(Profitable)",
        f"{mc.get('p_profitable', 0):.1%}",
        delta="Strong" if mc.get("p_profitable", 0) > 0.7 else "Weak",
        delta_color="normal" if mc.get("p_profitable", 0) > 0.7 else "inverse",
    )
    mc2.metric("Expectancy CI", _format_ci(mc.get("expectancy_ci")))
    mc3.metric("Max Drawdown CI", _format_ci(mc.get("max_drawdown_ci")))

    # CI range chart
    ci_names = ["win_rate_ci", "mean_r_ci", "expectancy_ci", "sharpe_ci", "max_drawdown_ci"]
    ci_labels = ["Win Rate", "Mean R", "Expectancy", "Sharpe", "Max DD"]
    ci_rows = []
    for name, label in zip(ci_names, ci_labels):
        ci = mc.get(name)
        if isinstance(ci, (list, tuple)) and len(ci) >= 2:
            ci_rows.append({"Metric": label, "Lower": ci[0], "Upper": ci[1],
                            "Mid": (ci[0] + ci[1]) / 2})

    if ci_rows:
        df_ci = pd.DataFrame(ci_rows)
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Bar(
            x=df_ci["Metric"], y=df_ci["Mid"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(df_ci["Upper"] - df_ci["Mid"]).tolist(),
                arrayminus=(df_ci["Mid"] - df_ci["Lower"]).tolist(),
            ),
            marker_color="#4CAF50",
        ))
        fig_ci.update_layout(
            template="plotly_dark", height=350,
            title=f"Monte Carlo {int(mc.get('confidence', 0.95) * 100)}% Confidence Intervals",
            yaxis_title="Value",
        )
        st.plotly_chart(fig_ci, use_container_width=True)
else:
    st.info("No Monte Carlo data or insufficient trades.")

# ── Window Details ────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Window-by-Window Details", expanded=False):
    if windows:
        rows = []
        for w in windows:
            row = {
                "Window": w.get("id", "?"),
                "Train": w.get("train_period", ""),
                "Test": w.get("test_period", ""),
                "Regime": w.get("regime", ""),
                "OOS Trades": w.get("oos_trades", 0),
                "OOS Total R": round(w.get("oos_total_r", 0), 3),
                "IS Trades": w.get("is_trades", 0),
                "IS Total R": round(w.get("is_total_r", 0), 3),
            }
            bp = w.get("best_params", {})
            for k, v in bp.items():
                row[k] = v
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No window data.")

# ── All Available Results ─────────────────────────────────────────────────────
st.markdown("---")
with st.expander("All WFO Results Index", expanded=False):
    idx_rows = []
    for r in results:
        idx_rows.append({
            "Strategy": r["strategy"],
            "Symbol": r["symbol"],
            "Timeframe": r["timeframe"],
            "Timestamp": r["timestamp"],
        })
    st.dataframe(pd.DataFrame(idx_rows), use_container_width=True)
