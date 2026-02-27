"""Walk-Forward Optimization tab for the ML Training dashboard page."""

import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _get_adapter(strategy_name: str):
    """Import and return the strategy adapter."""
    from backtrader_framework.optimization.strategy_adapters import ADAPTER_REGISTRY
    cls = ADAPTER_REGISTRY.get(strategy_name)
    return cls() if cls else None


def _get_symbols():
    return ["BTC", "ETH", "NQ"]


def _get_timeframes():
    return ["15m", "1h", "4h"]


def render_wfo_tab():
    """Render the full WFO tab inside a Streamlit container."""

    st.header("Walk-Forward Optimization")
    st.caption(
        "Optimize strategy parameters on in-sample windows and validate on "
        "out-of-sample windows. This reveals whether a strategy's edge is "
        "real or just curve-fitted to historical data."
    )

    # ── W1: Configuration & Run ──────────────────────────────────────────
    st.subheader("Configuration")

    from backtrader_framework.optimization.strategy_adapters import ADAPTER_REGISTRY

    col_s, col_sym, col_tf = st.columns(3)
    with col_s:
        strategy_name = st.selectbox(
            "Strategy", list(ADAPTER_REGISTRY.keys()),
            key="wfo_strategy",
        )
    with col_sym:
        symbol = st.selectbox("Symbol", _get_symbols(), key="wfo_symbol")
    with col_tf:
        adapter = _get_adapter(strategy_name)
        default_tfs = adapter.default_timeframes if adapter else ["4h"]
        all_tfs = _get_timeframes()
        default_idx = all_tfs.index(default_tfs[0]) if default_tfs[0] in all_tfs else 0
        timeframe = st.selectbox("Timeframe", all_tfs, index=default_idx, key="wfo_tf")

    # Timeframe-scaled defaults (so 15m doesn't create 1800 windows)
    from backtrader_framework.optimization.wfo_engine import WFOConfig as _WFOCfg
    _tf_defaults = _WFOCfg.for_timeframe(timeframe)

    # Advanced settings
    with st.expander("Advanced Settings"):
        adv_c1, adv_c2, adv_c3 = st.columns(3)
        with adv_c1:
            train_bars = st.number_input(
                "IS Window (bars)", min_value=200, max_value=80000,
                value=_tf_defaults.train_window_bars, step=100, key="wfo_train",
                help=f"In-sample window size. Auto-scaled for {timeframe}: {_tf_defaults.train_window_bars} bars.",
            )
            test_bars = st.number_input(
                "OOS Window (bars)", min_value=50, max_value=16000,
                value=_tf_defaults.test_window_bars, step=50, key="wfo_test",
                help=f"Out-of-sample window size. Auto-scaled for {timeframe}: {_tf_defaults.test_window_bars} bars.",
            )
        with adv_c2:
            step_bars = st.number_input(
                "Step Size (bars)", min_value=50, max_value=8000,
                value=_tf_defaults.step_bars, step=50, key="wfo_step",
                help=f"How far to advance each window. Auto-scaled for {timeframe}: {_tf_defaults.step_bars} bars.",
            )
            anchored = st.checkbox(
                "Anchored", value=True, key="wfo_anchored",
                help="If checked, IS window always starts from the beginning (expanding window).",
            )
        with adv_c3:
            metric = st.selectbox(
                "Optimization Metric",
                ["expectancy", "profit_factor", "sharpe", "total_r"],
                key="wfo_metric",
                help="Metric used to select best params on each IS window.",
            )
            max_combos = st.number_input(
                "Max Param Combos", min_value=100, max_value=5000,
                value=1000, step=100, key="wfo_max_combos",
            )

    regime_adaptive = st.checkbox(
        "Regime-Adaptive Mode",
        value=False,
        key="wfo_regime_adaptive",
        help="Optimize separate parameters per market regime (trending, ranging, volatile) "
             "during IS, then switch between them during OOS based on detected regime.",
    )

    # Run button
    run_col, _, load_col = st.columns([2, 4, 2])
    with run_col:
        run_clicked = st.button("Run WFO", type="primary", key="wfo_run")

    # ── Execute WFO ──────────────────────────────────────────────────────
    if run_clicked:
        _run_wfo(strategy_name, symbol, timeframe,
                 train_bars, test_bars, step_bars, anchored, metric, max_combos,
                 regime_adaptive=regime_adaptive)

    # ── W2: Display Results ──────────────────────────────────────────────
    if "wfo_result" in st.session_state and st.session_state["wfo_result"]:
        _render_results(st.session_state["wfo_result"])

    # ── W3: Saved Results ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Saved Results")
    _render_saved_results()


def _run_wfo(strategy_name, symbol, timeframe,
             train_bars, test_bars, step_bars, anchored, metric, max_combos,
             regime_adaptive=False):
    """Execute WFO and store result in session state."""
    from backtrader_framework.optimization.wfo_engine import WFOEngine, WFOConfig, TransactionCosts, RegimeAdaptiveWFO
    from backtrader_framework.optimization.persistence import save_wfo_result

    adapter = _get_adapter(strategy_name)
    if not adapter:
        st.error(f"No adapter for strategy: {strategy_name}")
        return

    config = WFOConfig(
        train_window_bars=train_bars,
        test_window_bars=test_bars,
        step_bars=step_bars,
        anchored=anchored,
        optimization_metric=metric,
        max_param_combos=max_combos,
        costs=TransactionCosts.for_asset(symbol),
    )

    if regime_adaptive:
        engine = RegimeAdaptiveWFO(adapter, config)
    else:
        engine = WFOEngine(adapter, config)

    mode_label = "Regime-Adaptive WFO" if regime_adaptive else "WFO"
    with st.status(f"Running {mode_label}: {strategy_name} on {symbol} {timeframe}...", expanded=True) as status:
        progress_bar = st.progress(0.0)
        log_area = st.empty()

        def progress_cb(pct, msg):
            progress_bar.progress(min(pct, 1.0))
            log_area.caption(msg)

        t0 = time.time()
        if regime_adaptive:
            result = engine.run(symbol, timeframe, progress_callback=progress_cb, run_standard=True)
        else:
            result = engine.run(symbol, timeframe, progress_callback=progress_cb)
        elapsed = time.time() - t0

        if result.get('error'):
            status.update(label=f"Failed: {result['error']}", state="error")
            st.error(f"WFO failed: {result['error']}")
            return

        # Save
        filepath = save_wfo_result(result)
        result['_saved_path'] = filepath

        status.update(
            label=f"Done in {elapsed:.1f}s — {result.get('oos_n_trades', 0)} OOS trades across {result.get('n_windows', 0)} windows",
            state="complete",
        )

    st.session_state["wfo_result"] = result


def _render_results(result: dict):
    """Render WFO results: KPIs, equity curve, breakdowns, param analysis."""
    st.markdown("---")
    st.subheader(f"Results: {result['strategy_name']} — {result['symbol']} {result['timeframe']}")

    oos = result.get('oos_stats', {})
    is_stats = result.get('is_stats', {})
    overfit = result.get('overfit_ratio')

    # ── KPI Row ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("OOS Trades", result.get('oos_n_trades', 0))
    k2.metric("Win Rate", f"{oos.get('win_rate', 0):.1%}" if oos.get('valid') else "N/A")
    k3.metric("Mean R", f"{oos.get('mean_r', 0):.3f}" if oos.get('valid') else "N/A")
    k4.metric("Profit Factor", f"{oos.get('profit_factor', 0):.2f}" if oos.get('valid') else "N/A")
    k5.metric("Sharpe/Trade", f"{oos.get('sharpe_per_trade', 0):.2f}" if oos.get('valid') else "N/A")

    if overfit is not None:
        color = "normal" if 0.5 <= overfit <= 1.2 else "inverse"
        k6.metric("Overfit Ratio", f"{overfit:.2f}",
                  help="OOS Mean R / IS Mean R. 0.5-1.0 = robust; <0.3 = overfitting; >1.0 = conservative IS")
    else:
        k6.metric("Overfit Ratio", "N/A")

    # ── Regime-Adaptive: Comparison & Per-Regime Params ────────────────
    if result.get('regime_adaptive'):
        _render_regime_adaptive_section(result)

    # ── Statistical significance ─────────────────────────────────────────
    if oos.get('valid'):
        sig_msgs = []
        if oos.get('win_rate_significant'):
            sig_msgs.append(f"Win rate **{oos['win_rate']:.1%}** is statistically > 50% (p={oos.get('binomial_p_value', 1):.4f})")
        if oos.get('mean_r_significant'):
            sig_msgs.append(f"Mean R **{oos['mean_r']:.3f}** is statistically > 0 (p={oos.get('t_p_value', 1):.4f})")

        ci = oos.get('win_rate_ci_95')
        if ci:
            sig_msgs.append(f"95% CI for win rate: [{ci[0]:.1%}, {ci[1]:.1%}]")
        ci_r = oos.get('mean_r_ci_95')
        if ci_r:
            sig_msgs.append(f"95% CI for mean R: [{ci_r[0]:.3f}, {ci_r[1]:.3f}]")

        if sig_msgs:
            with st.expander("Statistical Significance", expanded=False):
                for m in sig_msgs:
                    st.markdown(f"- {m}")
                n_min = oos.get('min_n_for_significance', 999)
                if n_min > oos.get('n_trades', 0):
                    st.warning(f"Need ~{n_min} trades for reliable significance (have {oos['n_trades']})")

    # ── IS vs OOS Comparison ─────────────────────────────────────────────
    if is_stats.get('valid') and oos.get('valid'):
        with st.expander("IS vs OOS Comparison", expanded=True):
            comp_df = pd.DataFrame({
                'Metric': ['Trades', 'Win Rate', 'Mean R', 'Profit Factor', 'Expectancy', 'Max DD (R)'],
                'In-Sample': [
                    is_stats.get('n_trades', 0),
                    f"{is_stats.get('win_rate', 0):.1%}",
                    f"{is_stats.get('mean_r', 0):.3f}",
                    f"{is_stats.get('profit_factor', 0):.2f}",
                    f"{is_stats.get('expectancy', 0):.3f}",
                    f"{is_stats.get('max_drawdown_r', 0):.2f}",
                ],
                'Out-of-Sample': [
                    oos.get('n_trades', 0),
                    f"{oos.get('win_rate', 0):.1%}",
                    f"{oos.get('mean_r', 0):.3f}",
                    f"{oos.get('profit_factor', 0):.2f}",
                    f"{oos.get('expectancy', 0):.3f}",
                    f"{oos.get('max_drawdown_r', 0):.2f}",
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── OOS Equity Curve ─────────────────────────────────────────────────
    equity_data = result.get('oos_equity', [])
    if equity_data:
        st.subheader("OOS Equity Curve")
        eq_df = pd.DataFrame(equity_data)
        eq_df['time'] = pd.to_datetime(eq_df['time'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df['time'], y=eq_df['cumulative_r'],
            mode='lines+markers',
            line=dict(color='#4CAF50', width=2),
            marker=dict(
                size=6,
                color=['#4CAF50' if r > 0 else '#F44336' for r in eq_df['r']],
            ),
            hovertemplate=(
                "Time: %{x}<br>"
                "Cumulative R: %{y:.2f}<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            template="plotly_dark", height=350,
            xaxis_title="Date", yaxis_title="Cumulative R-Multiple",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="wfo_equity_curve")

    # ── Monte Carlo Confidence Intervals ────────────────────────────────
    if result.get('monte_carlo') or result.get('equity_fan'):
        _render_monte_carlo_section(result)

    # ── Drawdown Analysis & Auto-Disable Rules ────────────────────────
    if result.get('drawdown_analysis'):
        _render_drawdown_section(result)

    # ── Execution Timing Analysis ────────────────────────────────────────
    if result.get('timing_analysis'):
        _render_timing_section(result)

    # ── Regime Breakdown ─────────────────────────────────────────────────
    regime_data = result.get('regime_analysis', {})
    dir_data = result.get('direction_analysis', {})

    if regime_data or dir_data:
        bd_c1, bd_c2 = st.columns(2)

        if regime_data:
            with bd_c1:
                st.markdown("**Performance by Regime**")
                regime_rows = []
                for regime, stats in regime_data.items():
                    regime_rows.append({
                        'Regime': regime,
                        'Trades': stats['n_trades'],
                        'Win Rate': f"{stats['win_rate']:.1%}",
                        'Mean R': f"{stats['mean_r']:.3f}",
                        'Total R': f"{stats['total_r']:.2f}",
                    })
                if regime_rows:
                    st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

        if dir_data:
            with bd_c2:
                st.markdown("**Performance by Direction**")
                dir_rows = []
                for direction, stats in dir_data.items():
                    if stats.get('n_trades', 0) > 0:
                        dir_rows.append({
                            'Direction': direction,
                            'Trades': stats['n_trades'],
                            'Win Rate': f"{stats['win_rate']:.1%}",
                            'Mean R': f"{stats['mean_r']:.3f}",
                            'Total R': f"{stats['total_r']:.2f}",
                        })
                if dir_rows:
                    st.dataframe(pd.DataFrame(dir_rows), use_container_width=True, hide_index=True)

    # ── Window-by-Window Table ───────────────────────────────────────────
    windows = result.get('windows', [])
    if windows:
        with st.expander(f"Window Details ({len(windows)} windows)", expanded=False):
            win_rows = []
            for w in windows:
                win_rows.append({
                    'Window': w['id'] + 1,
                    'Train Period': w['train_period'],
                    'Test Period': w['test_period'],
                    'Regime': w['regime'],
                    'IS Trades': w['is_trades'],
                    'IS Total R': f"{w['is_total_r']:.2f}",
                    'OOS Trades': w['oos_trades'],
                    'OOS Total R': f"{w['oos_total_r']:.2f}",
                })
            st.dataframe(pd.DataFrame(win_rows), use_container_width=True, hide_index=True)

    # ── Parameter Analysis ───────────────────────────────────────────────
    param_hist = result.get('param_history', [])
    if param_hist and len(param_hist) > 1:
        st.subheader("Parameter Analysis")

        # Extract all param names from first entry
        param_names = list(param_hist[0]['best_params'].keys())
        n_windows = len(param_hist)

        # Build param matrix
        param_matrix = {}
        for pname in param_names:
            param_matrix[pname] = [ph['best_params'].get(pname, 0) for ph in param_hist]

        # Heatmap
        st.markdown("**Best Parameters per Window**")

        # Normalize each param to [0,1] for the heatmap
        z_data = []
        text_data = []
        for pname in param_names:
            vals = param_matrix[pname]
            vmin, vmax = min(vals), max(vals)
            rng = vmax - vmin if vmax != vmin else 1
            z_data.append([(v - vmin) / rng for v in vals])
            text_data.append([f"{v}" for v in vals])

        fig_heat = go.Figure(go.Heatmap(
            z=z_data,
            x=[f"W{i+1}" for i in range(n_windows)],
            y=param_names,
            text=text_data,
            texttemplate="%{text}",
            colorscale='Viridis',
            showscale=False,
        ))
        fig_heat.update_layout(
            template="plotly_dark", height=max(200, 40 * len(param_names)),
            xaxis_title="Window",
        )
        st.plotly_chart(fig_heat, use_container_width=True, key="wfo_param_heatmap")

        # Param stability (coefficient of variation)
        st.markdown("**Parameter Stability**")
        stability_rows = []
        for pname in param_names:
            vals = param_matrix[pname]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            cv = std_v / abs(mean_v) if abs(mean_v) > 1e-10 else 0
            stability_rows.append({
                'Parameter': pname,
                'Mean': f"{mean_v:.4f}",
                'Std Dev': f"{std_v:.4f}",
                'CV': f"{cv:.2f}",
                'Stable': 'Yes' if cv < 0.3 else 'No',
                'Range': f"[{min(vals)}, {max(vals)}]",
            })
        stab_df = pd.DataFrame(stability_rows)
        st.dataframe(stab_df, use_container_width=True, hide_index=True)

        unstable = [r['Parameter'] for r in stability_rows if r['Stable'] == 'No']
        if unstable:
            st.warning(
                f"Parameters with high variance across windows: **{', '.join(unstable)}**. "
                f"High CV (>0.3) suggests the optimal value shifts with market conditions — "
                f"consider narrowing the search range or using a different metric."
            )
        else:
            st.success("All parameters are stable across windows (CV < 0.3).")


def _render_monte_carlo_section(result: dict):
    """Render Monte Carlo bootstrap CIs, equity fan chart, and key stats."""

    mc = result.get('monte_carlo', {})
    fan = result.get('equity_fan', {})
    oos = result.get('oos_stats', {})

    st.subheader("Monte Carlo Analysis")

    # ── Key MC Metrics ───────────────────────────────────────────────
    if mc.get('valid'):
        mc_c1, mc_c2, mc_c3 = st.columns(3)
        with mc_c1:
            p_prof = mc.get('p_profitable', 0)
            st.metric(
                "P(Profitable)",
                f"{p_prof:.1%}",
                help=f"Probability that reshuffled OOS trades yield positive mean R ({mc.get('n_resamples', 0):,} resamples)",
            )
        with mc_c2:
            if fan.get('valid'):
                st.metric(
                    "5th %ile Final R",
                    f"{fan.get('pct_5_final_r', 0):.2f}",
                    help="Worst-case cumulative R at 5th percentile of resampled paths",
                )
        with mc_c3:
            if fan.get('valid'):
                st.metric(
                    "95th %ile Max DD",
                    f"{fan.get('pct_95_max_dd', 0):.2f}R",
                    help="Worst-case maximum drawdown (in R) at 95th percentile",
                )

    # ── CI Comparison Table ──────────────────────────────────────────
    if mc.get('valid') and oos.get('valid'):
        with st.expander("Parametric vs Bootstrap 95% CI", expanded=True):
            def _fmt_ci(ci, fmt=".3f"):
                if ci:
                    return f"[{ci[0]:{fmt}}, {ci[1]:{fmt}}]"
                return "N/A"

            def _fmt_ci_pct(ci):
                if ci:
                    return f"[{ci[0]:.1%}, {ci[1]:.1%}]"
                return "N/A"

            ci_df = pd.DataFrame({
                'Metric': ['Mean R', 'Win Rate', 'Expectancy', 'Profit Factor', 'Sharpe/Trade', 'Max Drawdown (R)'],
                'Parametric 95% CI': [
                    _fmt_ci(oos.get('mean_r_ci_95')),
                    _fmt_ci_pct(oos.get('win_rate_ci_95')),
                    'N/A',  # No parametric CI for expectancy
                    'N/A',  # No parametric CI for PF
                    'N/A',  # No parametric CI for sharpe
                    'N/A',  # No parametric CI for max DD
                ],
                'Bootstrap 95% CI': [
                    _fmt_ci(mc.get('mean_r_ci')),
                    _fmt_ci_pct(mc.get('win_rate_ci')),
                    _fmt_ci(mc.get('expectancy_ci')),
                    _fmt_ci(mc.get('profit_factor_ci'), ".2f"),
                    _fmt_ci(mc.get('sharpe_ci'), ".2f"),
                    _fmt_ci(mc.get('max_drawdown_ci'), ".2f"),
                ],
            })
            st.dataframe(ci_df, use_container_width=True, hide_index=True)

            st.caption(
                f"Bootstrap: {mc.get('n_resamples', 0):,} resamples, "
                f"{mc.get('confidence', 0.95):.0%} confidence. "
                "Bootstrap CIs are distribution-free and typically wider than parametric."
            )

    # ── Equity Fan Chart ─────────────────────────────────────────────
    if fan.get('valid'):
        with st.expander("Equity Fan Chart", expanded=True):
            pcts = fan.get('percentiles', {})
            actual = fan.get('actual_equity', [])
            n_trades = fan.get('n_trades', 0)
            x = list(range(1, n_trades + 1))

            fig = go.Figure()

            # 5th-95th percentile band (light)
            if '5' in pcts and '95' in pcts:
                fig.add_trace(go.Scatter(
                    x=x + x[::-1],
                    y=pcts['95'] + pcts['5'][::-1],
                    fill='toself',
                    fillcolor='rgba(33, 150, 243, 0.1)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='5th-95th %ile',
                    showlegend=True,
                    hoverinfo='skip',
                ))

            # 25th-75th percentile band (darker)
            if '25' in pcts and '75' in pcts:
                fig.add_trace(go.Scatter(
                    x=x + x[::-1],
                    y=pcts['75'] + pcts['25'][::-1],
                    fill='toself',
                    fillcolor='rgba(33, 150, 243, 0.25)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='25th-75th %ile',
                    showlegend=True,
                    hoverinfo='skip',
                ))

            # Median path
            if '50' in pcts:
                fig.add_trace(go.Scatter(
                    x=x, y=pcts['50'],
                    mode='lines',
                    line=dict(color='#2196F3', width=2, dash='dash'),
                    name='Median Path',
                ))

            # Actual OOS equity
            if actual:
                fig.add_trace(go.Scatter(
                    x=x, y=actual,
                    mode='lines',
                    line=dict(color='#4CAF50', width=3),
                    name='Actual OOS',
                ))

            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            fig.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_title="Trade #",
                yaxis_title="Cumulative R-Multiple",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig, use_container_width=True, key="wfo_equity_fan")

            # Summary row
            fan_c1, fan_c2, fan_c3 = st.columns(3)
            fan_c1.caption(f"Median final R: **{fan.get('median_final_r', 0):.2f}**")
            fan_c2.caption(f"5th %ile: **{fan.get('pct_5_final_r', 0):.2f}** | 95th %ile: **{fan.get('pct_95_final_r', 0):.2f}**")
            fan_c3.caption(f"Median max DD: **{fan.get('median_max_dd', 0):.2f}R** | 95th %ile DD: **{fan.get('pct_95_max_dd', 0):.2f}R**")


def _render_drawdown_section(result: dict):
    """Render drawdown analysis: underwater plot, streaks, rolling DD, auto-disable rules."""

    dd = result.get('drawdown_analysis', {})
    if not dd.get('valid'):
        return

    st.subheader("Drawdown Analysis & Auto-Disable Rules")

    # ── KPI Row ───────────────────────────────────────────────────────
    depth = dd.get('depth_stats', {})
    streaks = dd.get('streak_stats', {})
    thresholds = dd.get('thresholds', {})

    dd_c1, dd_c2, dd_c3, dd_c4 = st.columns(4)
    dd_c1.metric("Max Drawdown", f"{depth.get('max', 0):.1f} R")
    dd_c2.metric("Max Loss Streak", f"{streaks.get('max_length', 0)} trades")
    dd_c3.metric(
        "Pause Threshold (20-trade)",
        f"{thresholds.get('rolling_20_dd', {}).get('value', '—')} R",
    )
    dd_c4.metric(
        "Pause Streak Limit",
        f"{thresholds.get('max_consecutive_losses', {}).get('value', '—')} losses",
    )

    # ── Underwater Plot ───────────────────────────────────────────────
    underwater = dd.get('underwater', [])
    times = dd.get('times', [])
    if underwater and times:
        import plotly.graph_objects as go

        fig_uw = go.Figure()
        fig_uw.add_trace(go.Scatter(
            x=list(range(len(underwater))),
            y=underwater,
            fill='tozeroy',
            fillcolor='rgba(239,83,80,0.3)',
            line=dict(color='#EF5350', width=1.5),
            name='Drawdown',
            hovertemplate='Trade %{x}: %{y:.2f}R<extra></extra>',
        ))

        # Add threshold line
        rolling_thresh = thresholds.get('rolling_20_dd', {}).get('value')
        if rolling_thresh:
            fig_uw.add_hline(
                y=-rolling_thresh, line_dash="dash", line_color="#FF9800",
                annotation_text=f"Pause threshold: -{rolling_thresh}R",
                annotation_position="bottom right",
            )

        fig_uw.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Trade #",
            yaxis_title="Drawdown (R)",
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_uw, use_container_width=True, key="wfo_underwater")

    # ── Rolling 20-Trade Max DD ──────────────────────────────────────
    rolling = dd.get('rolling_dd', {}).get('window_20', {})
    if rolling.get('values'):
        fig_roll = go.Figure()
        vals = rolling['values']
        fig_roll.add_trace(go.Scatter(
            x=list(range(len(vals))),
            y=vals,
            mode='lines',
            line=dict(color='#FF7043', width=1.5),
            name='Rolling 20-trade Max DD',
            hovertemplate='Window start %{x}: %{y:.2f}R<extra></extra>',
        ))

        p95 = rolling.get('p95', 0)
        fig_roll.add_hline(
            y=p95, line_dash="dot", line_color="rgba(255,255,255,0.4)",
            annotation_text=f"95th pctile: {p95:.1f}R",
        )

        rolling_thresh = thresholds.get('rolling_20_dd', {}).get('value')
        if rolling_thresh:
            fig_roll.add_hline(
                y=rolling_thresh, line_dash="dash", line_color="#FF9800",
                annotation_text=f"Threshold: {rolling_thresh}R",
            )

        fig_roll.update_layout(
            template="plotly_dark",
            height=280,
            xaxis_title="Window Start (Trade #)",
            yaxis_title="Max DD in 20-Trade Window (R)",
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_roll, use_container_width=True, key="wfo_rolling_dd")

    # ── Auto-Disable Rules Table ─────────────────────────────────────
    with st.expander("Calibrated Auto-Disable Rules", expanded=True):
        rules = []
        r20 = thresholds.get('rolling_20_dd', {})
        if r20:
            rules.append({
                'Rule': 'Rolling 20-trade DD',
                'Threshold': f"{r20.get('value', '—')} R",
                'Calibration': r20.get('calibration', ''),
            })
        mcl = thresholds.get('max_consecutive_losses', {})
        if mcl:
            rules.append({
                'Rule': 'Consecutive losses',
                'Threshold': f"{mcl.get('value', '—')} losses",
                'Calibration': mcl.get('calibration', ''),
            })
        cd = thresholds.get('cooldown_trades', {})
        if cd:
            rules.append({
                'Rule': 'Cooldown after trigger',
                'Threshold': f"{cd.get('value', '—')} trades",
                'Calibration': cd.get('calibration', ''),
            })
        mc95 = thresholds.get('mc_max_dd_95', {})
        if mc95:
            rules.append({
                'Rule': 'Monte Carlo 95% max DD',
                'Threshold': f"{mc95.get('value', '—')} R",
                'Calibration': mc95.get('calibration', ''),
            })

        if rules:
            st.dataframe(
                pd.DataFrame(rules), use_container_width=True, hide_index=True,
            )

        confidence = thresholds.get('confidence', 'unknown')
        n_trades = thresholds.get('n_trades', 0)
        st.caption(f"Calibration confidence: **{confidence}** ({n_trades} OOS trades)")

    # ── Top Drawdown Episodes ────────────────────────────────────────
    episodes = dd.get('episodes', [])
    if episodes:
        with st.expander(f"Drawdown Episodes ({len(episodes)} total)"):
            sorted_eps = sorted(episodes, key=lambda e: e['depth'], reverse=True)
            ep_rows = []
            for ep in sorted_eps[:10]:
                ep_rows.append({
                    'Depth (R)': f"{ep['depth']:.2f}",
                    'Start': f"Trade #{ep['start_idx']}",
                    'Trough': f"Trade #{ep['trough_idx']}",
                    'Trades to Trough': ep['duration_to_trough'],
                    'Recovery Trades': ep['recovery_trades'] if ep['recovered'] else 'Not recovered',
                    'Total Duration': ep['total_duration'],
                })
            st.dataframe(
                pd.DataFrame(ep_rows), use_container_width=True, hide_index=True,
            )

    # ── Loss Streak Distribution ─────────────────────────────────────
    streak_data = dd.get('streaks', {})
    if streak_data.get('lengths'):
        with st.expander("Loss Streak Distribution"):
            lengths = streak_data['lengths']
            # Histogram
            from collections import Counter
            counts = Counter(lengths)
            x_vals = sorted(counts.keys())
            y_vals = [counts[x] for x in x_vals]

            fig_streak = go.Figure(go.Bar(
                x=x_vals, y=y_vals,
                marker_color='#EF5350',
                text=y_vals, textposition='auto',
            ))
            fig_streak.update_layout(
                template="plotly_dark",
                height=250,
                xaxis_title="Consecutive Losses",
                yaxis_title="Frequency",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_streak, use_container_width=True, key="wfo_streak_dist")

            ss = dd.get('streak_stats', {})
            st.caption(
                f"Max streak: **{ss.get('max_length', 0)}** | "
                f"Mean: **{ss.get('mean_length', 0):.1f}** | "
                f"Max R lost in streak: **{ss.get('max_r_lost', 0):.2f}R**"
            )

    # ── Recovery Patterns ────────────────────────────────────────────
    recovery = dd.get('recovery', {})
    if recovery:
        with st.expander("Recovery Patterns"):
            rec_rows = []
            for depth, info in recovery.items():
                rec_rows.append({
                    'DD Depth': depth,
                    'Occurrences': info['n_occurrences'],
                    'Recovered': info['n_recovered'],
                    'Recovery Rate': f"{info['recovery_rate']:.0%}",
                    'Avg Trades to Recover': (
                        f"{info['avg_trades_to_recover']:.0f}"
                        if info['avg_trades_to_recover'] is not None else '—'
                    ),
                })
            st.dataframe(
                pd.DataFrame(rec_rows), use_container_width=True, hide_index=True,
            )


def _render_timing_section(result: dict):
    """Render execution timing analysis: hourly, session, day-of-week breakdowns."""

    ta = result.get('timing_analysis', {})
    if not ta.get('valid'):
        return

    st.subheader("Execution Timing Analysis")

    recs = ta.get('recommendations', {})
    confidence = recs.get('confidence', 'low')

    # ── KPI Row ───────────────────────────────────────────────────────
    best_sessions = recs.get('best_sessions', [])
    worst_sessions = recs.get('worst_sessions', [])
    best_hours_list = recs.get('best_hours', [])

    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    t_c1.metric(
        "Best Session",
        best_sessions[0]['session'] if best_sessions else "N/A",
        delta=f"{best_sessions[0]['mean_r']:.3f} R/trade" if best_sessions else None,
    )
    t_c2.metric(
        "Worst Session",
        worst_sessions[0]['session'] if worst_sessions else "N/A",
        delta=f"{worst_sessions[0]['mean_r']:.3f} R/trade" if worst_sessions else None,
        delta_color="inverse",
    )
    t_c3.metric(
        "Best Hour (UTC)",
        f"{best_hours_list[0]['hour']:02d}:00" if best_hours_list else "N/A",
        delta=f"{best_hours_list[0]['mean_r']:.3f} R/trade" if best_hours_list else None,
    )
    t_c4.metric(
        "Confidence",
        confidence.title(),
        help=f"Based on {ta.get('n_trades', 0)} OOS trades",
    )

    # ── Session Performance ───────────────────────────────────────────
    session_data = ta.get('session', {})
    if session_data:
        with st.expander("Session Performance", expanded=True):
            sess_rows = []
            for session_name in ['Asia', 'London', 'Overlap', 'NY', 'Quiet']:
                m = session_data.get(session_name, {})
                if m.get('n_trades', 0) > 0:
                    sess_rows.append({
                        'Session': session_name,
                        'Trades': m['n_trades'],
                        'Win Rate': f"{m['win_rate']:.1%}",
                        'Mean R': f"{m['mean_r']:.4f}",
                        'Total R': f"{m['total_r']:.2f}",
                        'MFE/MAE': f"{m.get('mfe_mae_ratio', 0):.2f}" if m.get('mfe_mae_ratio') else "N/A",
                    })
            if sess_rows:
                st.dataframe(pd.DataFrame(sess_rows), use_container_width=True, hide_index=True)

    # ── Hourly Performance Chart ──────────────────────────────────────
    hourly_data = ta.get('hourly', {})
    if hourly_data:
        with st.expander("Hourly Performance (UTC)", expanded=True):
            hours = list(range(24))
            mean_rs = [hourly_data.get(h, hourly_data.get(str(h), {})).get('mean_r', 0) for h in hours]
            n_trades = [hourly_data.get(h, hourly_data.get(str(h), {})).get('n_trades', 0) for h in hours]
            win_rates = [hourly_data.get(h, hourly_data.get(str(h), {})).get('win_rate', 0) for h in hours]

            colors = ['#4CAF50' if r > 0 else '#F44336' for r in mean_rs]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"{h:02d}" for h in hours],
                y=mean_rs,
                marker_color=colors,
                text=[f"n={n}" for n in n_trades],
                textposition='outside',
                hovertemplate=(
                    "Hour: %{x} UTC<br>"
                    "Mean R: %{y:.4f}<br>"
                    "Win Rate: %{customdata:.1%}<br>"
                    "<extra></extra>"
                ),
                customdata=win_rates,
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig.update_layout(
                template="plotly_dark", height=350,
                xaxis_title="Hour (UTC)", yaxis_title="Mean R per Trade",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True, key="wfo_hourly_perf")

    # ── Direction x Hour Heatmap ──────────────────────────────────────
    dir_hour = ta.get('direction_hour_heatmap', {})
    if dir_hour:
        with st.expander("Direction × Hour Heatmap", expanded=False):
            directions = ['LONG', 'SHORT']
            hours = list(range(24))

            z_data = []
            text_data = []
            for d in directions:
                d_data = dir_hour.get(d, {})
                row_z = []
                row_t = []
                for h in hours:
                    m = d_data.get(h, d_data.get(str(h), {}))
                    wr = m.get('win_rate', 0)
                    n = m.get('n_trades', 0)
                    row_z.append(wr if n >= 3 else None)
                    row_t.append(f"{wr:.0%} (n={n})" if n > 0 else "")
                z_data.append(row_z)
                text_data.append(row_t)

            fig_heat = go.Figure(go.Heatmap(
                z=z_data,
                x=[f"{h:02d}" for h in hours],
                y=directions,
                text=text_data,
                texttemplate="%{text}",
                colorscale='RdYlGn',
                zmin=0.3, zmax=0.7,
                colorbar=dict(title="Win Rate"),
            ))
            fig_heat.update_layout(
                template="plotly_dark", height=200,
                xaxis_title="Hour (UTC)",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_heat, use_container_width=True, key="wfo_dir_hour_heat")

    # ── Day-of-Week Performance ───────────────────────────────────────
    dow_data = ta.get('day_of_week', {})
    if dow_data:
        with st.expander("Day-of-Week Performance", expanded=False):
            dow_rows = []
            for day_idx in range(7):
                m = dow_data.get(day_idx, dow_data.get(str(day_idx), {}))
                if m.get('n_trades', 0) > 0:
                    dow_rows.append({
                        'Day': m.get('day_name', str(day_idx)),
                        'Trades': m['n_trades'],
                        'Win Rate': f"{m['win_rate']:.1%}",
                        'Mean R': f"{m['mean_r']:.4f}",
                        'Total R': f"{m['total_r']:.2f}",
                    })
            if dow_rows:
                st.dataframe(pd.DataFrame(dow_rows), use_container_width=True, hide_index=True)

    # ── Execution Quality by Hour ─────────────────────────────────────
    eq_data = ta.get('execution_quality', {})
    if eq_data:
        with st.expander("Execution Quality by Hour (MFE/MAE Ratio)", expanded=False):
            hours = list(range(24))
            ratios = []
            for h in hours:
                m = eq_data.get(h, eq_data.get(str(h), {}))
                r = m.get('mfe_mae_ratio')
                n = m.get('n_trades', 0)
                ratios.append(r if r is not None and n >= 3 else None)

            valid_hours = [h for h, r in zip(hours, ratios) if r is not None]
            valid_ratios = [r for r in ratios if r is not None]

            if valid_ratios:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Bar(
                    x=[f"{h:02d}" for h in valid_hours],
                    y=valid_ratios,
                    marker_color=['#4CAF50' if r > 1.0 else '#FF9800' if r > 0.5 else '#F44336'
                                  for r in valid_ratios],
                    hovertemplate="Hour %{x} UTC: MFE/MAE = %{y:.2f}<extra></extra>",
                ))
                fig_eq.add_hline(
                    y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                    annotation_text="MFE = MAE",
                )
                fig_eq.update_layout(
                    template="plotly_dark", height=300,
                    xaxis_title="Hour (UTC)", yaxis_title="Mean MFE/MAE Ratio",
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_eq, use_container_width=True, key="wfo_exec_quality")

                st.caption(
                    "MFE/MAE > 1.0 means trades get more favorable excursion than adverse. "
                    "Higher = better execution quality for that hour."
                )

    # ── Hold Duration Analysis ────────────────────────────────────────
    hold = ta.get('hold_duration', {})
    if hold.get('valid'):
        with st.expander("Hold Duration Analysis", expanded=False):
            dist = hold.get('distribution', {})
            hd_c1, hd_c2, hd_c3 = st.columns(3)
            hd_c1.metric("Mean Duration", f"{dist.get('mean', 0):.1f} bars")
            hd_c2.metric("Median Duration", f"{dist.get('median', 0):.0f} bars")
            hd_c3.metric("Max Duration", f"{dist.get('max', 0)} bars")

            by_outcome = hold.get('by_outcome', {})
            if by_outcome:
                out_rows = []
                for outcome, stats in by_outcome.items():
                    out_rows.append({
                        'Outcome': outcome,
                        'Count': stats['n'],
                        'Mean Bars': f"{stats['mean']:.1f}",
                        'Median Bars': f"{stats['median']:.0f}",
                    })
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

            corr = hold.get('duration_r_correlation', {})
            if corr:
                st.caption(
                    f"Duration vs R correlation: r = {corr['pearson_r']:.3f} "
                    f"(p = {corr['p_value']:.4f}) — "
                    f"{'significant' if corr['significant'] else 'not significant'}"
                )

    # ── Timing Filter Recommendations ─────────────────────────────────
    if recs.get('suggested_filters') or recs.get('best_hours') or recs.get('worst_hours'):
        with st.expander("Timing Filter Recommendations", expanded=True):
            filters = recs.get('suggested_filters', [])
            if filters:
                for f in filters:
                    if f['type'] == 'avoid_hours':
                        hours_str = ', '.join(f"{h:02d}:00" for h in f['hours_utc'])
                        st.warning(f"Consider avoiding entries at: **{hours_str} UTC** — {f['rationale']}")
                    elif f['type'] == 'avoid_session':
                        st.warning(f"Consider avoiding the **{f['session']}** session — {f['rationale']}")

            if recs.get('best_hours'):
                best = recs['best_hours'][:3]
                hours_str = ', '.join(
                    f"{h['hour']:02d}:00 ({h['mean_r']:.3f}R, n={h['n_trades']})" for h in best
                )
                st.success(f"Best entry hours: {hours_str}")

            if confidence == 'low':
                st.info(
                    "Low confidence: fewer than 100 OOS trades. "
                    "Timing patterns may not be reliable. "
                    "Run more WFO windows or use finer timeframes to increase trade count."
                )


def _render_regime_adaptive_section(result: dict):
    """Render regime-adaptive comparison, per-regime params, and timeline."""

    # ── Standard vs Adaptive Comparison ──────────────────────────────
    comparison = result.get('regime_vs_standard')
    if comparison:
        st.markdown("#### Standard vs Regime-Adaptive Comparison")
        std_stats = result.get('standard_oos_stats', {})
        ada_stats = result.get('oos_stats', {})

        comp_c1, comp_c2, comp_c3 = st.columns(3)
        with comp_c1:
            st.metric(
                "Standard Mean R",
                f"{comparison['standard_oos_mean_r']:.3f}",
                help=f"{comparison['standard_oos_n_trades']} trades",
            )
        with comp_c2:
            st.metric(
                "Adaptive Mean R",
                f"{comparison['adaptive_oos_mean_r']:.3f}",
                help=f"{comparison['adaptive_oos_n_trades']} trades",
            )
        with comp_c3:
            imp = comparison['improvement_pct']
            st.metric(
                "Improvement",
                f"{imp:+.1f}%",
                delta=f"{imp:+.1f}%",
                delta_color="normal" if imp > 0 else "inverse",
            )

        # Side-by-side KPI table
        if std_stats.get('valid') and ada_stats.get('valid'):
            comp_df = pd.DataFrame({
                'Metric': ['Trades', 'Win Rate', 'Mean R', 'Profit Factor', 'Expectancy', 'Max DD (R)'],
                'Standard WFO': [
                    std_stats.get('n_trades', 0),
                    f"{std_stats.get('win_rate', 0):.1%}",
                    f"{std_stats.get('mean_r', 0):.3f}",
                    f"{std_stats.get('profit_factor', 0):.2f}",
                    f"{std_stats.get('expectancy', 0):.3f}",
                    f"{std_stats.get('max_drawdown_r', 0):.2f}",
                ],
                'Regime-Adaptive WFO': [
                    ada_stats.get('n_trades', 0),
                    f"{ada_stats.get('win_rate', 0):.1%}",
                    f"{ada_stats.get('mean_r', 0):.3f}",
                    f"{ada_stats.get('profit_factor', 0):.2f}",
                    f"{ada_stats.get('expectancy', 0):.3f}",
                    f"{ada_stats.get('max_drawdown_r', 0):.2f}",
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Per-Regime Optimal Parameters ────────────────────────────────
    best_by_regime = result.get('best_params_by_regime', {})
    if best_by_regime:
        with st.expander("Per-Regime Optimal Parameters", expanded=True):
            # Build a table: rows = param names, columns = regimes
            all_params = set()
            for params in best_by_regime.values():
                all_params.update(params.keys())
            all_params = sorted(all_params)

            regime_param_data = {'Parameter': all_params}
            for regime, params in sorted(best_by_regime.items()):
                regime_param_data[regime] = [params.get(p, '-') for p in all_params]

            st.dataframe(
                pd.DataFrame(regime_param_data),
                use_container_width=True, hide_index=True,
            )

            # Check if params actually differ across regimes
            if len(best_by_regime) > 1:
                param_sets = list(best_by_regime.values())
                all_same = all(p == param_sets[0] for p in param_sets[1:])
                if all_same:
                    st.warning(
                        "All regimes selected identical parameters. "
                        "Regime switching adds no value for this configuration."
                    )

    # ── Regime-Switching Timeline ────────────────────────────────────
    timeline = result.get('regime_switching_timeline', [])
    if timeline:
        with st.expander("Regime-Switching Timeline", expanded=False):
            regime_colors = {
                'trending_up': '#4CAF50',
                'trending_down': '#F44336',
                'ranging': '#2196F3',
                'volatile': '#FF9800',
                'unknown': '#9E9E9E',
            }

            fig = go.Figure()

            for entry in timeline:
                w_id = entry['window_id']
                dist = entry.get('regime_distribution', {})
                total_bars = sum(dist.values()) if dist else 1

                # Stacked bar for each window showing regime distribution
                cumulative = 0
                for regime in ['trending_up', 'trending_down', 'ranging', 'volatile', 'unknown']:
                    count = dist.get(regime, 0)
                    if count > 0:
                        pct = count / total_bars
                        fig.add_trace(go.Bar(
                            x=[pct],
                            y=[f"W{w_id + 1}"],
                            orientation='h',
                            name=regime,
                            marker_color=regime_colors.get(regime, '#9E9E9E'),
                            showlegend=(w_id == 0),
                            legendgroup=regime,
                            hovertemplate=f"{regime}: {count} bars ({pct:.0%})<extra></extra>",
                        ))

            fig.update_layout(
                barmode='stack',
                template='plotly_dark',
                height=max(200, 30 * len(timeline)),
                xaxis_title='Regime Distribution',
                yaxis_title='Window',
                xaxis=dict(tickformat='.0%'),
                legend_title='Regime',
            )
            st.plotly_chart(fig, use_container_width=True, key="wfo_regime_timeline")


def _render_saved_results():
    """Render the saved results section with load capability."""
    from backtrader_framework.optimization.persistence import list_wfo_results, load_wfo_result

    saved = list_wfo_results()
    if not saved:
        st.info("No saved WFO results yet. Run an optimization above to generate one.")
        return

    options = [f"{r['strategy']} {r['symbol']} {r['timeframe']} — {r['timestamp']}" for r in saved]
    selected_idx = st.selectbox(
        "Load saved result", range(len(options)),
        format_func=lambda i: options[i],
        key="wfo_saved_select",
    )

    if st.button("Load", key="wfo_load_saved"):
        result = load_wfo_result(saved[selected_idx]['filepath'])
        st.session_state["wfo_result"] = result
        st.rerun()
