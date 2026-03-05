"""Quant Research Lab — interactive explainers for quantitative tools."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from data.wfo_loader import get_latest_wfo

st.set_page_config(page_title="Quant Research Lab", layout="wide")
st.title("Quant Research Lab")
st.caption(
    "Plain-English explanations of the quantitative tools powering our strategies. "
    "Each section includes interactive visualizations and real-world metaphors."
)

# ── Tool Selector ─────────────────────────────────────────────────────────────
tool = st.sidebar.radio(
    "Select Tool",
    [
        "Walk-Forward Optimization",
        "HMM Regime Detection",
        "Bayesian Edge Estimation",
        "Monte Carlo Simulation",
        "Optuna Optimization",
        "DVOL / Implied Volatility",
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1: Walk-Forward Optimization
# ═══════════════════════════════════════════════════════════════════════════════
def _render_wfo():
    st.header("Walk-Forward Optimization (WFO)")

    st.info(
        '**Metaphor:** "Training for a marathon by running different routes '
        '— then racing a route you\'ve never seen."\n\n'
        "WFO trains your strategy on one chunk of data, then tests it on "
        "the next unseen chunk. It repeats this across the entire dataset. "
        "If the strategy only works on data it's already seen, WFO exposes that."
    )

    st.markdown("### How It Works")
    st.markdown(
        "1. **Split** historical data into overlapping train/test windows\n"
        "2. **Optimize** parameters on each training window\n"
        "3. **Test** those parameters on the out-of-sample window\n"
        "4. **Stitch** all out-of-sample results into one equity curve\n\n"
        "The key insight: the strategy *never* sees the test data during optimization. "
        "This mimics real trading where the future is always unknown."
    )

    # Interactive visualization: train/test windows
    n_windows = st.slider("Number of WFO windows", 3, 8, 5, key="wfo_nwin")
    total_bars = 1000
    train_pct = 0.7

    fig = go.Figure()
    window_size = total_bars // n_windows
    train_size = int(window_size * train_pct)
    test_size = window_size - train_size

    for i in range(n_windows):
        start = i * window_size
        train_end = start + train_size
        test_end = train_end + test_size

        fig.add_shape(
            type="rect", x0=start, x1=train_end, y0=i - 0.35, y1=i + 0.35,
            fillcolor="rgba(33, 150, 243, 0.6)", line=dict(width=0),
        )
        fig.add_shape(
            type="rect", x0=train_end, x1=test_end, y0=i - 0.35, y1=i + 0.35,
            fillcolor="rgba(76, 175, 80, 0.6)", line=dict(width=0),
        )
        fig.add_annotation(
            x=(start + train_end) / 2, y=i, text="TRAIN",
            showarrow=False, font=dict(color="white", size=11),
        )
        fig.add_annotation(
            x=(train_end + test_end) / 2, y=i, text="TEST",
            showarrow=False, font=dict(color="white", size=11),
        )

    fig.update_layout(
        template="plotly_dark", height=50 + n_windows * 60,
        xaxis_title="Data (bars)", yaxis_title="Window",
        yaxis=dict(tickvals=list(range(n_windows)),
                   ticktext=[f"W{i+1}" for i in range(n_windows)]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Real data: IS vs OOS performance
    wfo = get_latest_wfo("FVG", "BTC", "15m")
    if wfo and wfo.get("windows"):
        st.markdown("### Real WFO Results: FVG / BTC / 15m")
        windows = wfo["windows"]
        w_ids = [f"W{w.get('id', i)}" for i, w in enumerate(windows)]
        is_r = [w.get("is_total_r", 0) for w in windows]
        oos_r = [w.get("oos_total_r", 0) for w in windows]

        fig_real = go.Figure()
        fig_real.add_trace(go.Bar(x=w_ids, y=is_r, name="In-Sample R",
                                  marker_color="#2196F3"))
        fig_real.add_trace(go.Bar(x=w_ids, y=oos_r, name="Out-of-Sample R",
                                  marker_color="#4CAF50"))
        fig_real.update_layout(
            template="plotly_dark", height=350, barmode="group",
            yaxis_title="Total R-Multiple",
        )
        st.plotly_chart(fig_real, use_container_width=True)

        overfit = wfo.get("overfit_ratio", 0)
        st.caption(
            f"Overfit ratio: **{overfit:.2f}** "
            f"({'healthy (< 2.0)' if overfit < 2.0 else 'concerning (> 2.0)'}). "
            "A ratio close to 1.0 means out-of-sample performance matches in-sample."
        )

    st.success(
        "**Why it matters:** WFO is our primary defense against overfitting. "
        "Any parameter set that only works on historical data it was trained on "
        "gets immediately exposed when tested on unseen data."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2: HMM Regime Detection
# ═══════════════════════════════════════════════════════════════════════════════
def _render_hmm():
    st.header("HMM Regime Detection")

    st.info(
        '**Metaphor:** "A weather forecaster that classifies the market as '
        "'sunny' (calm) or 'stormy' (volatile) — and adjusts your umbrella size.\"\n\n"
        "A Hidden Markov Model (HMM) assumes the market switches between hidden "
        "'states' (regimes) that we can't directly observe, but whose effects "
        "(returns and volatility) we can measure."
    )

    st.markdown("### The Two Market Regimes")

    # Generate synthetic two-regime data
    np.random.seed(42)
    calm_returns = np.random.normal(0.001, 0.01, 500)
    volatile_returns = np.random.normal(-0.0005, 0.035, 500)

    x = np.linspace(-0.15, 0.15, 300)

    # Manual Gaussian PDFs
    def _norm_pdf(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    calm_pdf = _norm_pdf(x, 0.001, 0.01)
    volatile_pdf = _norm_pdf(x, -0.0005, 0.035)

    fig_hmm = go.Figure()
    fig_hmm.add_trace(go.Scatter(
        x=x * 100, y=calm_pdf, mode="lines", name="Calm Regime",
        fill="tozeroy", line=dict(color="#4CAF50"),
        fillcolor="rgba(76, 175, 80, 0.3)",
    ))
    fig_hmm.add_trace(go.Scatter(
        x=x * 100, y=volatile_pdf, mode="lines", name="Volatile Regime",
        fill="tozeroy", line=dict(color="#F44336"),
        fillcolor="rgba(244, 67, 54, 0.3)",
    ))
    fig_hmm.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Return (%)", yaxis_title="Probability Density",
        title="Return Distributions by Regime",
    )
    st.plotly_chart(fig_hmm, use_container_width=True)

    st.markdown("### Position Sizing by Regime")
    sizing_data = pd.DataFrame({
        "Regime": ["Calm", "Reduced", "Uncertain", "Defensive"],
        "Size Multiplier": [1.0, 0.7, 0.5, 0.3],
        "Description": [
            "Full position — market is well-behaved",
            "Slightly elevated vol — reduce exposure",
            "Mixed signals — half position",
            "Storm mode — minimal exposure",
        ],
    })

    fig_sizing = go.Figure()
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    fig_sizing.add_trace(go.Bar(
        x=sizing_data["Regime"], y=sizing_data["Size Multiplier"],
        marker_color=colors,
        text=[f"{v:.0%}" for v in sizing_data["Size Multiplier"]],
        textposition="auto",
    ))
    fig_sizing.update_layout(
        template="plotly_dark", height=300,
        yaxis_title="Position Size Multiplier",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_sizing, use_container_width=True)

    st.success(
        "**Why it matters:** During volatile regimes, our bots automatically "
        "reduce position sizes to 30% of normal. This protects capital during "
        "adverse conditions while staying fully invested during calm markets."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3: Bayesian Edge Estimation
# ═══════════════════════════════════════════════════════════════════════════════
def _render_bayesian():
    st.header("Bayesian Edge Estimation")

    st.info(
        '**Metaphor:** "A detective who starts with a hunch (prior) and updates '
        "beliefs as evidence accumulates. With 10 trades, the hunch dominates. "
        'After 200 trades, the evidence speaks for itself."'
    )

    st.markdown("### How Bayesian Updating Works")
    st.markdown(
        "We start with a **prior belief** about our strategy's win rate "
        "(e.g., 50% — no edge assumed). As trades come in, the posterior "
        "distribution sharpens around the true win rate.\n\n"
        "- **Few trades** → posterior is wide and uncertain\n"
        "- **Many trades** → posterior narrows, giving high confidence\n"
        "- **Key output**: P(edge > 0) = probability the strategy genuinely profits"
    )

    # Interactive: slider for trades and win rate
    col1, col2 = st.columns(2)
    with col1:
        n_trades = st.slider("Number of trades", 10, 300, 50, key="bayes_n")
    with col2:
        true_wr = st.slider("True win rate (%)", 30, 70, 55, key="bayes_wr")

    n_wins = int(n_trades * true_wr / 100)

    # Prior: Beta(5, 5) — neutral
    alpha_prior, beta_prior = 5, 5
    # Posterior: Beta(alpha_prior + wins, beta_prior + losses)
    alpha_post = alpha_prior + n_wins
    beta_post = beta_prior + (n_trades - n_wins)

    x = np.linspace(0.01, 0.99, 300)

    # Manual Beta PDF approximation
    from math import lgamma
    def _beta_pdf(x, a, b):
        log_norm = lgamma(a + b) - lgamma(a) - lgamma(b)
        return np.exp(log_norm + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x))

    y_prior = np.array([_beta_pdf(xi, alpha_prior, beta_prior) for xi in x])
    y_post = np.array([_beta_pdf(xi, alpha_post, beta_post) for xi in x])

    fig_bayes = go.Figure()
    fig_bayes.add_trace(go.Scatter(
        x=x, y=y_prior, mode="lines", name="Prior (no belief)",
        line=dict(color="gray", dash="dash"),
    ))
    fig_bayes.add_trace(go.Scatter(
        x=x, y=y_post, mode="lines", name=f"Posterior ({n_trades} trades)",
        fill="tozeroy", line=dict(color="#2196F3"),
        fillcolor="rgba(33, 150, 243, 0.3)",
    ))
    fig_bayes.add_vline(x=0.5, line_dash="dot", line_color="red",
                        annotation_text="Break-even (50%)")
    fig_bayes.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Win Rate", yaxis_title="Belief Density",
        title="Prior vs Posterior Win Rate Distribution",
    )
    st.plotly_chart(fig_bayes, use_container_width=True)

    # P(edge > 0) = P(win_rate > 0.5)
    # Approximate with simple integration
    p_edge = sum(1 for xi, yi in zip(x, y_post) if xi > 0.5) / len(x)
    # Better: use CDF approximation
    above_50 = y_post[x > 0.5].sum()
    total = y_post.sum()
    p_edge = above_50 / total if total > 0 else 0.5

    st.metric("P(Edge > 0)", f"{p_edge:.1%}")
    st.caption(
        f"After {n_trades} trades with {true_wr}% win rate, "
        f"there is a **{p_edge:.1%}** probability that this strategy "
        "has a genuine positive edge."
    )

    st.success(
        "**Why it matters:** Bayesian estimation tells us how confident we "
        "should be in our strategy's edge. A strategy with 60% win rate over "
        "30 trades is much less convincing than 55% over 300 trades. Bayesian "
        "analysis quantifies exactly how convincing."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4: Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════════
def _render_monte_carlo():
    st.header("Monte Carlo Simulation")

    st.info(
        '**Metaphor:** "Rolling the dice 10,000 times with your actual trade '
        "results to see the full range of possible outcomes. Your backtest "
        "is ONE path — Monte Carlo shows you ALL possible paths.\""
    )

    st.markdown("### Equity Path Fan Chart")
    st.markdown(
        "We take your actual trade results, randomly reshuffle their order "
        "thousands of times, and plot the resulting equity curves. This shows:\n\n"
        "- **Best case**: everything goes right early\n"
        "- **Worst case**: drawdowns cluster at the start\n"
        "- **Confidence bands**: where 95% of outcomes fall"
    )

    # Generate synthetic equity paths
    n_paths = st.slider("Number of simulations", 100, 5000, 1000, key="mc_paths")
    confidence = st.slider("Confidence level (%)", 80, 99, 95, key="mc_conf")

    np.random.seed(42)
    # Simulate trade results: slight positive edge
    trade_results = np.random.normal(0.005, 0.02, 100)

    paths = np.zeros((n_paths, len(trade_results) + 1))
    paths[:, 0] = 10000  # Starting equity
    for i in range(n_paths):
        shuffled = np.random.choice(trade_results, size=len(trade_results), replace=True)
        for j, r in enumerate(shuffled):
            paths[i, j + 1] = paths[i, j] * (1 + r)

    x_axis = list(range(len(trade_results) + 1))
    lower_pct = (100 - confidence) / 2
    upper_pct = 100 - lower_pct

    median_path = np.median(paths, axis=0)
    lower_path = np.percentile(paths, lower_pct, axis=0)
    upper_path = np.percentile(paths, upper_pct, axis=0)
    worst_path = np.min(paths, axis=0)
    best_path = np.max(paths, axis=0)

    fig_mc = go.Figure()

    # Confidence band
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=upper_path, mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=lower_path, mode="lines",
        fill="tonexty", fillcolor="rgba(33, 150, 243, 0.2)",
        line=dict(width=0), name=f"{confidence}% CI",
    ))

    # Key paths
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=median_path, mode="lines",
        name="Median", line=dict(color="#4CAF50", width=2),
    ))
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=worst_path, mode="lines",
        name="Worst Case", line=dict(color="#F44336", width=1, dash="dash"),
    ))
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=best_path, mode="lines",
        name="Best Case", line=dict(color="#2196F3", width=1, dash="dash"),
    ))

    fig_mc.update_layout(
        template="plotly_dark", height=450,
        xaxis_title="Trade Number", yaxis_title="Equity ($)",
        title=f"Monte Carlo Equity Fan ({n_paths:,} simulations)",
        yaxis_tickformat="$,.0f",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Max drawdown distribution
    max_dds = []
    for i in range(n_paths):
        peak = np.maximum.accumulate(paths[i])
        dd = (paths[i] - peak) / peak
        max_dds.append(dd.min())

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Histogram(
        x=np.array(max_dds) * 100, nbinsx=50,
        marker_color="#F44336", opacity=0.7,
    ))
    dd_threshold = np.percentile(max_dds, 100 - confidence) * 100
    fig_dd.add_vline(x=dd_threshold, line_dash="dash", line_color="#FFC107",
                     annotation_text=f"{confidence}% worst: {dd_threshold:.1f}%")
    fig_dd.update_layout(
        template="plotly_dark", height=300,
        xaxis_title="Max Drawdown (%)", yaxis_title="Frequency",
        title="Max Drawdown Distribution",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.success(
        "**Why it matters:** Your backtest shows one specific ordering of trades. "
        "Monte Carlo reveals the full range of possible equity paths. "
        f"At {confidence}% confidence, the worst drawdown is "
        f"**{dd_threshold:.1f}%** — this is what you should size your "
        "account to survive."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 5: Optuna (Bayesian Hyperparameter Optimization)
# ═══════════════════════════════════════════════════════════════════════════════
def _render_optuna():
    st.header("Optuna — Bayesian Hyperparameter Optimization")

    st.info(
        '**Metaphor:** "A smart treasure hunter who learns from each dig where '
        "to search next. Grid search digs everywhere exhaustively. Random search "
        "scatters holes randomly. Bayesian optimization clusters digs near the "
        'most promising areas — finding treasure 5-10x faster."'
    )

    st.markdown("### Three Optimization Strategies Compared")

    # Generate synthetic 2D optimization landscape
    np.random.seed(42)
    x_range = np.linspace(0, 10, 100)
    y_range = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x_range, y_range)
    # Multi-modal surface with one clear optimum
    Z = (np.sin(X) * np.cos(Y / 2) + 0.5 * np.exp(-((X - 7) ** 2 + (Y - 3) ** 2) / 4)
         - 0.3 * np.exp(-((X - 3) ** 2 + (Y - 7) ** 2) / 3))

    # Contour plot of the landscape
    fig_landscape = go.Figure(data=go.Contour(
        x=x_range, y=y_range, z=Z,
        colorscale="Viridis", showscale=True,
        colorbar=dict(title="Objective"),
    ))
    fig_landscape.update_layout(
        template="plotly_dark", height=400,
        xaxis_title="Parameter A", yaxis_title="Parameter B",
        title="Optimization Landscape (darker = better)",
    )
    st.plotly_chart(fig_landscape, use_container_width=True)

    # Compare search strategies
    method = st.radio(
        "Search method",
        ["Grid Search", "Random Search", "Bayesian (Optuna)"],
        horizontal=True, key="optuna_method",
    )

    n_evals = st.slider("Number of evaluations", 10, 100, 30, key="optuna_n")

    if method == "Grid Search":
        side = int(np.sqrt(n_evals))
        gx = np.linspace(0, 10, side)
        gy = np.linspace(0, 10, side)
        pts_x, pts_y = np.meshgrid(gx, gy)
        pts_x, pts_y = pts_x.flatten()[:n_evals], pts_y.flatten()[:n_evals]
        desc = "Evaluates a uniform grid — thorough but slow, wastes time on poor regions."
    elif method == "Random Search":
        pts_x = np.random.uniform(0, 10, n_evals)
        pts_y = np.random.uniform(0, 10, n_evals)
        desc = "Randomly samples — better coverage than grid but no learning."
    else:
        # Simulate Bayesian: cluster near optimum with some exploration
        pts_x = np.concatenate([
            np.random.uniform(0, 10, n_evals // 3),
            np.random.normal(7, 1.5, n_evals // 3),
            np.random.normal(7, 0.5, n_evals - 2 * (n_evals // 3)),
        ])
        pts_y = np.concatenate([
            np.random.uniform(0, 10, n_evals // 3),
            np.random.normal(3, 1.5, n_evals // 3),
            np.random.normal(3, 0.5, n_evals - 2 * (n_evals // 3)),
        ])
        pts_x = np.clip(pts_x, 0, 10)
        pts_y = np.clip(pts_y, 0, 10)
        desc = ("Explores broadly at first, then focuses on promising regions. "
                "Finds the optimum with fewer evaluations.")

    # Evaluate points
    try:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((x_range, y_range), Z.T)
        pts_z = interp(np.column_stack([pts_x, pts_y]))
    except ImportError:
        pts_z = np.zeros(len(pts_x))

    fig_search = go.Figure(data=go.Contour(
        x=x_range, y=y_range, z=Z,
        colorscale="Viridis", showscale=False, opacity=0.5,
    ))
    fig_search.add_trace(go.Scatter(
        x=pts_x, y=pts_y, mode="markers",
        marker=dict(
            color=pts_z, colorscale="RdYlGn", size=8,
            line=dict(width=1, color="white"),
        ),
        name="Evaluated points",
    ))
    fig_search.update_layout(
        template="plotly_dark", height=400,
        xaxis_title="Parameter A", yaxis_title="Parameter B",
        title=f"{method}: {n_evals} evaluations",
    )
    st.plotly_chart(fig_search, use_container_width=True)
    st.caption(desc)

    best_idx = np.argmax(pts_z) if len(pts_z) > 0 else 0
    st.metric("Best objective found",
              f"{pts_z[best_idx]:.3f}" if len(pts_z) > 0 else "N/A")

    st.success(
        "**Why it matters:** With 20+ parameters to tune, grid search would take "
        "millions of evaluations. Optuna's TPE (Tree-structured Parzen Estimator) "
        "finds near-optimal parameters in 50-200 trials by learning which regions "
        "of parameter space are most promising."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 6: DVOL / Implied Volatility Gates
# ═══════════════════════════════════════════════════════════════════════════════
def _render_dvol():
    st.header("DVOL — Implied Volatility Gates")

    st.info(
        '**Metaphor:** "An insurance price gauge. When insurance premiums are '
        "expensive, the market expects turbulence ahead. DVOL is the crypto "
        'equivalent of the VIX — it tells you how scared the market is."'
    )

    st.markdown("### What is DVOL?")
    st.markdown(
        "**DVOL** (Deribit Volatility Index) measures the market's expectation "
        "of future Bitcoin volatility, derived from options prices.\n\n"
        "- **Low DVOL (< 40):** Market is calm — tight stops, aggressive entries\n"
        "- **Medium DVOL (40-70):** Normal conditions — standard parameters\n"
        "- **High DVOL (> 70):** Fear/turbulence expected — wider stops, "
        "more depth tolerance"
    )

    # Synthetic DVOL time series
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    dvol_base = 50 + 15 * np.sin(np.linspace(0, 4 * np.pi, 365))
    dvol = dvol_base + np.random.normal(0, 5, 365)
    dvol = np.clip(dvol, 20, 120)

    # Synthetic price data
    returns = np.random.normal(0.001, 0.02, 365)
    # Make returns more volatile when DVOL is high
    returns = returns * (1 + (dvol - 50) / 100)
    price = 50000 * np.cumprod(1 + returns)

    threshold = st.slider("DVOL threshold for regime shift", 30, 90, 55, key="dvol_thresh")

    fig_dvol = go.Figure()

    # DVOL line
    fig_dvol.add_trace(go.Scatter(
        x=dates, y=dvol, mode="lines", name="DVOL",
        line=dict(color="#FF9800", width=2),
    ))

    # Threshold line
    fig_dvol.add_hline(y=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold: {threshold}")

    # Shade high-vol periods
    in_high = dvol > threshold
    start = None
    for i in range(len(dates)):
        if in_high[i] and start is None:
            start = dates[i]
        elif not in_high[i] and start is not None:
            fig_dvol.add_vrect(
                x0=start, x1=dates[i],
                fillcolor="rgba(244, 67, 54, 0.1)", line_width=0,
            )
            start = None

    fig_dvol.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Date", yaxis_title="DVOL",
        title="DVOL Time Series with Regime Bands",
    )
    st.plotly_chart(fig_dvol, use_container_width=True)

    # Impact on strategy parameters
    st.markdown("### How DVOL Adjusts Our Strategies")
    adj_data = pd.DataFrame({
        "DVOL Range": ["< 40 (Calm)", "40-55 (Normal)", "55-70 (Elevated)", "> 70 (High)"],
        "Min Sweep Depth": ["0.20 ATR", "0.30 ATR", "0.40 ATR", "0.50 ATR"],
        "R:R Scale": ["1.0x", "1.0x", "1.2x", "1.5x"],
        "Interpretation": [
            "Tight markets — small sweeps are significant",
            "Normal — standard parameters",
            "Elevated fear — need deeper sweeps to confirm",
            "Panic — only trade very deep sweeps with wider targets",
        ],
    })
    st.dataframe(adj_data, use_container_width=True, hide_index=True)

    # Signal count by DVOL regime
    low_signals = np.sum(dvol < 40)
    mid_signals = np.sum((dvol >= 40) & (dvol < threshold))
    high_signals = np.sum(dvol >= threshold)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["Calm (< 40)", f"Normal (40-{threshold})", f"Elevated (> {threshold})"],
        y=[low_signals, mid_signals, high_signals],
        marker_color=["#4CAF50", "#FFC107", "#F44336"],
        text=[low_signals, mid_signals, high_signals],
        textposition="auto",
    ))
    fig_bar.update_layout(
        template="plotly_dark", height=300,
        yaxis_title="Days in Regime",
        title="Time Spent in Each DVOL Regime",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.success(
        "**Why it matters:** DVOL gates prevent our strategies from trading "
        "with parameters calibrated for calm markets during volatile periods. "
        "When DVOL spikes, we automatically require deeper sweeps and set wider "
        "targets — adapting to the new reality rather than fighting it."
    )


# ── Render Selected Tool ──────────────────────────────────────────────────────
RENDERERS = {
    "Walk-Forward Optimization": _render_wfo,
    "HMM Regime Detection": _render_hmm,
    "Bayesian Edge Estimation": _render_bayesian,
    "Monte Carlo Simulation": _render_monte_carlo,
    "Optuna Optimization": _render_optuna,
    "DVOL / Implied Volatility": _render_dvol,
}

RENDERERS[tool]()
