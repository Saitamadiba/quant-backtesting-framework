"""Page 15: Synthetic Stress Testing."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Stress Testing", page_icon="💥", layout="wide")
st.title("💥 Synthetic Stress Testing")
st.caption(
    "Test strategy survival under synthetic worst-case scenarios. "
    "Generate crash paths at 1x-3x magnitude and measure drawdown, "
    "consecutive losses, and recovery."
)

# Ensure imports work
_BASE = Path(__file__).resolve().parent.parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from backtrader_framework.optimization.persistence import list_wfo_results
from backtrader_framework.optimization.synthetic_scenarios import (
    StressTester, StressTestConfig, ScenarioType, KNOWN_CRASHES,
)
from backtrader_framework.optimization.strategy_adapters import ADAPTER_REGISTRY


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.header("1. Configuration")

saved = list_wfo_results()
if not saved:
    st.warning("No saved WFO results found. Run WFO optimization first.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    strategies = list(ADAPTER_REGISTRY.keys())
    strategy = st.selectbox("Strategy", strategies, index=0, key="stress_strategy")

with col2:
    symbols = sorted(set(s['symbol'] for s in saved))
    symbol = st.selectbox("Symbol", symbols, index=0, key="stress_symbol")

with col3:
    timeframes = sorted(set(s['timeframe'] for s in saved))
    timeframe = st.selectbox("Timeframe", timeframes, index=0, key="stress_timeframe")

# Filter WFO results
filtered = [
    s for s in saved
    if s['strategy'] == strategy and s['symbol'] == symbol and s['timeframe'] == timeframe
]

if not filtered:
    st.warning(
        f"No WFO results for {strategy}/{symbol}/{timeframe}. "
        "Select a different combination or run WFO optimization first."
    )
    st.stop()

wfo_filepath = filtered[0]['filepath']
st.caption(f"Using WFO result: `{filtered[0]['filename']}`")

# Scenario selection
st.subheader("Scenarios")
col_a, col_b = st.columns(2)

with col_a:
    use_historical = st.checkbox("Historical Crash Scaling", value=True)
    use_vol_spike = st.checkbox("Volatility Spike", value=True)
    use_flash = st.checkbox("Flash Crash", value=True)

with col_b:
    use_drawdown = st.checkbox("Prolonged Drawdown", value=True)
    use_v_recovery = st.checkbox("V-Shaped Recovery", value=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. RUN STRESS TESTS
# ══════════════════════════════════════════════════════════════════════════════

if st.button("Run Stress Tests", type="primary"):
    # Build scenario list based on checkboxes
    scenarios = []

    if use_historical:
        crash_keys = ['may_2021', 'nov_2022']
        for key in crash_keys:
            crash = KNOWN_CRASHES.get(key)
            if crash is None:
                continue
            for mag in [1.0, 1.5, 2.0]:
                scenarios.append({
                    'type': ScenarioType.HISTORICAL_CRASH_SCALED,
                    'name': f"{crash.name} ({mag}x)",
                    'magnitude': mag,
                    'crash_key': key,
                })

    if use_vol_spike:
        for mult in [2.0, 3.0, 5.0]:
            scenarios.append({
                'type': ScenarioType.VOLATILITY_SPIKE,
                'name': f"Vol Spike {mult}x",
                'magnitude': mult,
                'vol_multiplier': mult,
            })

    if use_flash:
        for drop in [-0.10, -0.15]:
            scenarios.append({
                'type': ScenarioType.FLASH_CRASH,
                'name': f"Flash Crash {drop:.0%}",
                'magnitude': abs(drop) / 0.10,
                'flash_drop_pct': drop,
            })

    if use_drawdown:
        scenarios.append({
            'type': ScenarioType.PROLONGED_DRAWDOWN,
            'name': "Prolonged Drawdown",
            'magnitude': 1.0,
        })

    if use_v_recovery:
        scenarios.append({
            'type': ScenarioType.V_SHAPED_RECOVERY,
            'name': "V-Recovery (-30%)",
            'magnitude': 1.0,
            'crash_depth_pct': -0.30,
        })

    if not scenarios:
        st.error("Select at least one scenario type.")
        st.stop()

    # Build ScenarioConfig objects
    from backtrader_framework.optimization.synthetic_scenarios import ScenarioConfig

    sc_configs = []
    for sc in scenarios:
        kwargs = {
            'scenario_type': sc['type'],
            'name': sc['name'],
            'magnitude': sc.get('magnitude', 1.0),
        }
        if 'crash_key' in sc:
            kwargs['crash_key'] = sc['crash_key']
        if 'vol_multiplier' in sc:
            kwargs['vol_multiplier'] = sc['vol_multiplier']
        if 'flash_drop_pct' in sc:
            kwargs['flash_drop_pct'] = sc['flash_drop_pct']
        if 'crash_depth_pct' in sc:
            kwargs['crash_depth_pct'] = sc['crash_depth_pct']
        sc_configs.append(ScenarioConfig(**kwargs))

    config = StressTestConfig(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy,
        wfo_filepath=wfo_filepath,
        scenarios=sc_configs,
    )

    try:
        with st.spinner(f"Running {len(sc_configs)} stress scenarios..."):
            tester = StressTester(config)
            results = tester.run()

        st.session_state['stress_results'] = results
        st.success(
            f"Stress testing complete! "
            f"{len(results['scenarios'])} scenarios evaluated."
        )
    except Exception as e:
        st.error(f"Stress testing failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 3. DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if 'stress_results' not in st.session_state:
    st.info("Configure settings above and click 'Run Stress Tests' to start.")
    st.stop()

results = st.session_state['stress_results']
baseline = results.get('baseline', {})
scenarios = results.get('scenarios', [])
matrix = results.get('survival_matrix', {})

if not scenarios:
    st.warning("No scenario results.")
    st.stop()


# ── Baseline KPIs ────────────────────────────────────────────────────────────

st.header("2. Baseline Performance")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Strategy", results.get('strategy', ''))
b2.metric("Baseline Trades", baseline.get('n_trades', 0))
b3.metric("Baseline Win Rate", f"{baseline.get('win_rate', 0):.1%}")
b4.metric("Baseline Mean R", f"{baseline.get('mean_r', 0):+.3f}")


# ── Survival Heatmap ─────────────────────────────────────────────────────────

st.header("3. Survival Heatmap")
st.caption(
    "Green (80+) = strategy handles this stress well. "
    "Yellow (50-80) = vulnerable. Red (<50) = high risk of significant losses."
)

if matrix.get('types') and matrix.get('magnitudes') and matrix.get('scores'):
    type_labels = [t.replace('_', ' ').title() for t in matrix['types']]
    mag_labels = [f"{m}x" for m in matrix['magnitudes']]
    z_data = matrix['scores']

    fig_hm = go.Figure(data=go.Heatmap(
        z=z_data,
        x=mag_labels,
        y=type_labels,
        colorscale=[
            [0, '#D32F2F'],
            [0.5, '#FFC107'],
            [1.0, '#4CAF50'],
        ],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=14, color='white'),
        colorbar=dict(title="Score"),
        hovertemplate="Type: %{y}<br>Magnitude: %{x}<br>Score: %{z:.0f}<extra></extra>",
    ))
    fig_hm.update_layout(
        template="plotly_dark",
        height=max(250, 50 * len(type_labels)),
        xaxis_title="Magnitude",
        yaxis_title="Scenario Type",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_hm, use_container_width=True, key="stress_heatmap")


# ── Price Charts ─────────────────────────────────────────────────────────────

st.header("4. Scenario Price Charts")

scenarios_with_price = [s for s in scenarios if s.get('price_data')]
if scenarios_with_price:
    tab_names = [s['name'] for s in scenarios_with_price]
    tabs = st.tabs(tab_names)

    for tab, sc in zip(tabs, scenarios_with_price):
        with tab:
            pd_data = sc['price_data']
            fig_price = go.Figure()

            # Original price
            if 'original_close' in pd_data:
                fig_price.add_trace(go.Scatter(
                    x=pd_data.get('original_times', []),
                    y=pd_data['original_close'],
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
                    name='Original',
                ))

            # Synthetic price
            fig_price.add_trace(go.Scatter(
                x=pd_data.get('synthetic_times', []),
                y=pd_data['synthetic_close'],
                mode='lines',
                line=dict(color='#FF5722', width=2),
                name='Synthetic',
            ))

            fig_price.update_layout(
                template="plotly_dark",
                height=350,
                title=dict(text=sc['name'], font=dict(size=13)),
                yaxis_title="Price",
                margin=dict(l=10, r=10, t=35, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_price, use_container_width=True, key=f"price_{sc['name']}")

            # Metrics for this scenario
            sm = sc.get('survival_metrics', {})
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Survival Score", f"{sm.get('survival_score', 0):.0f}/100")
            mc2.metric("Max DD (R)", f"{sm.get('max_drawdown_r', 0):.2f}")
            mc3.metric("Consec Losses", sm.get('consecutive_losses', 0))
            mc4.metric("Trades (Event)", sc.get('trades_during_event', 0))
            mc5.metric("Net R (Event)", f"{sm.get('net_r_during_event', 0):+.2f}")


# ── Detailed Comparison Table ────────────────────────────────────────────────

st.header("5. Detailed Comparison")

rows = []
for sc in scenarios:
    sm = sc.get('survival_metrics', {})
    rows.append({
        'Scenario': sc.get('name', ''),
        'Type': sc.get('type', '').replace('_', ' ').title(),
        'Magnitude': f"{sc.get('magnitude', 1.0)}x",
        'Trades': sc.get('n_trades', 0),
        'Event Trades': sc.get('trades_during_event', 0),
        'Max DD (R)': f"{sm.get('max_drawdown_r', 0):.2f}",
        'Consec Losses': sm.get('consecutive_losses', 0),
        'Recovery': sm.get('recovery_trades', 0),
        'Net R (Event)': f"{sm.get('net_r_during_event', 0):+.3f}",
        'Worst Trade': f"{sm.get('worst_single_trade_r', 0):.3f}",
        'Survival Score': f"{sm.get('survival_score', 0):.0f}",
    })

if rows:
    df_table = pd.DataFrame(rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)


# ── Summary ──────────────────────────────────────────────────────────────────

st.header("6. Summary")

scores = [
    sc.get('survival_metrics', {}).get('survival_score', 0)
    for sc in scenarios
]
if scores:
    avg_score = np.mean(scores)
    min_score = min(scores)
    worst_scenario = scenarios[scores.index(min_score)]

    if avg_score >= 70:
        st.success(
            f"Average survival score: **{avg_score:.0f}/100** — "
            f"strategy shows good resilience across stress scenarios."
        )
    elif avg_score >= 40:
        st.warning(
            f"Average survival score: **{avg_score:.0f}/100** — "
            f"strategy is vulnerable to some stress conditions."
        )
    else:
        st.error(
            f"Average survival score: **{avg_score:.0f}/100** — "
            f"strategy is fragile under stress conditions."
        )

    st.caption(
        f"Weakest scenario: **{worst_scenario.get('name', '')}** "
        f"(score: {min_score:.0f}/100)"
    )
