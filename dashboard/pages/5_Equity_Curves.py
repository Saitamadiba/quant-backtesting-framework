"""Page 4: Equity curves & drawdown analysis."""

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Equity Curves", page_icon="📈", layout="wide")
st.title("📈 Equity Curves")

from data.data_loader import get_all_trades
from components.charts import equity_curve, drawdown_chart, rolling_win_rate
from components.filters import strategy_filter, source_filter, apply_filters
from config import INITIAL_BALANCE, STRATEGY_CHANGELOG

# ── Sidebar ───────────────────────────────────────────────────────────────────
src = source_filter(key_prefix="eq")
st.sidebar.markdown("**Overlays**")
show_benchmark = st.sidebar.checkbox("Buy & Hold Benchmark", value=False, key="eq_benchmark")
show_regimes = st.sidebar.checkbox("Market Regimes", value=False, key="eq_regimes",
                                   help="Shade background by market regime (Trending Up / Down / Ranging) using ADX + EMA on 1h candles.")
show_changes = st.sidebar.checkbox("Strategy Deployments", value=True, key="eq_changes",
                                   help="Show vertical markers where strategy code changes were deployed.")
df_all = get_all_trades(source_filter=src)
strategies = strategy_filter(df_all, key_prefix="eq")
df = apply_filters(df_all, strategies=strategies)

# ── Fetch overlay data (benchmark + regimes share the same Binance call) ────
benchmark_df = None
regime_df = None

if (show_benchmark or show_regimes) and not df.empty:
    from data.binance_helpers import fetch_binance_candles, calculate_indicators, classify_regime

    min_date = df["entry_time"].min()
    max_date = df["entry_time"].max()
    overlay_days = max(7, (max_date - min_date).days + 5)

    bm_symbols = df["symbol"].unique()
    bm_sym = "ETH" if len(bm_symbols) == 1 and bm_symbols[0] == "ETH" else "BTC"

    with st.spinner(f"Fetching {bm_sym} 1h candles for overlays..."):
        candles = fetch_binance_candles(bm_sym, "1h", overlay_days)

    if candles is not None and not candles.empty:
        if show_benchmark:
            benchmark_df = candles[["Close"]].copy()
            benchmark_df.index.name = "Timestamp"

        if show_regimes:
            candles_ind = calculate_indicators(candles)
            candles_ind["regime"] = candles_ind.apply(classify_regime, axis=1)
            regime_df = candles_ind[["regime"]].copy()

# Strategy changes to pass
change_markers = STRATEGY_CHANGELOG if show_changes else []

# ── Equity Curve ──────────────────────────────────────────────────────────────
st.caption("The equity curve shows cumulative profit over time for each strategy. Colored background bands indicate market regimes; vertical dotted lines mark strategy deployments so you can see before/after impact on performance.")
st.plotly_chart(
    equity_curve(
        df, initial_balance=INITIAL_BALANCE, benchmark=benchmark_df,
        regime_df=regime_df, strategy_changes=change_markers,
    ),
    key="eq_equity",
)

# ── Drawdown ──────────────────────────────────────────────────────────────────
st.caption("The drawdown chart shows how far your equity has fallen from its peak at each point in time. Regime bands and deployment markers are mirrored from the equity curve for easy cross-reference.")
st.plotly_chart(
    drawdown_chart(df, regime_df=regime_df, strategy_changes=change_markers),
    key="eq_dd",
)

# ── Max Drawdown Table ────────────────────────────────────────────────────────
st.subheader("Max Drawdown by Strategy")
st.caption("Maximum peak-to-trough decline for each strategy, in both dollar and percentage terms. This is the worst pain you've experienced. If Max DD exceeds your risk tolerance, reduce position size. Hover over column headers for metric definitions.")
if not df.empty:
    rows = []
    for strat in df["strategy"].unique():
        s = df[df["strategy"] == strat].sort_values("entry_time")
        cum = s["pnl_usd"].cumsum()

        # --- Fix 2: Use account equity (INITIAL_BALANCE + cum PnL) ---
        cum_equity = INITIAL_BALANCE + cum
        peak_equity = cum_equity.cummax()
        dd = peak_equity - cum_equity
        max_dd = dd.max()
        max_dd_idx = dd.idxmax() if not dd.empty else None
        max_dd_date = s.loc[max_dd_idx, "entry_time"] if max_dd_idx is not None and max_dd_idx in s.index else None

        # DD% relative to peak account equity
        peak_at_dd = peak_equity.loc[max_dd_idx] if max_dd_idx is not None and max_dd_idx in peak_equity.index else INITIAL_BALANCE
        max_dd_pct = (max_dd / peak_at_dd * 100) if peak_at_dd > 0 else 0

        # --- Fix 1: Sharpe with actual trade frequency ---
        r = s["r_multiple"].dropna()
        total_pnl = s["pnl_usd"].sum()
        if len(r) >= 2 and "entry_time" in s.columns:
            trading_days = (s["entry_time"].max() - s["entry_time"].min()).days or 1
            trades_per_year = len(r) / (trading_days / 365.25)
            sharpe = (r.mean() / r.std()) * (trades_per_year ** 0.5)
        else:
            sharpe = 0
            trading_days = 1

        # --- Fix 3: Calmar (annualized) vs Recovery (non-annualized) ---
        if max_dd > 0 and trading_days > 0:
            annualized_return = total_pnl * (365.25 / trading_days)
            calmar = annualized_return / max_dd
            recovery = total_pnl / max_dd
        else:
            calmar = 0
            recovery = 0

        # --- Fix 9: Equity R² and WR Trend ---
        equity_r2 = 0.0
        wr_slope = 0.0
        if len(cum_equity) >= 5:
            x = np.arange(len(cum_equity), dtype=float)
            y = cum_equity.values.astype(float)
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            equity_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if len(r) >= 5:
            win_loss = (s["pnl_usd"] > 0).astype(float).values
            x_wr = np.arange(len(win_loss), dtype=float)
            wr_coeffs = np.polyfit(x_wr, win_loss, 1)
            wr_slope = wr_coeffs[0]  # slope per trade

        rows.append({
            "Strategy": strat,
            "Max DD ($)": max_dd,
            "Max DD (%)": max_dd_pct,
            "Max DD Date": max_dd_date.strftime("%Y-%m-%d") if pd.notna(max_dd_date) else "N/A",
            "Sharpe Ratio": sharpe,
            "Calmar Ratio": calmar,
            "Recovery Factor": recovery,
            "Equity R²": equity_r2,
            "WR Trend": wr_slope,
            "Trades": len(s),
        })

    dd_df = pd.DataFrame(rows)

    st.dataframe(
        dd_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strategy": st.column_config.TextColumn(
                "Strategy",
                help="The trading strategy name.",
            ),
            "Max DD ($)": st.column_config.NumberColumn(
                "Max DD ($)",
                help="Maximum drawdown in dollars — the largest peak-to-trough decline in account equity. This is the worst-case historical loss you would have experienced.",
                format="$%.2f",
            ),
            "Max DD (%)": st.column_config.NumberColumn(
                "Max DD (%)",
                help="Maximum drawdown as a percentage of peak account equity (initial balance + cumulative P&L). Above 20% is aggressive; above 30% is dangerous.",
                format="%.1f%%",
            ),
            "Max DD Date": st.column_config.TextColumn(
                "Max DD Date",
                help="The date when the maximum drawdown occurred. Cross-reference with market events to understand what caused it.",
            ),
            "Sharpe Ratio": st.column_config.NumberColumn(
                "Sharpe Ratio",
                help="Risk-adjusted return annualized using actual trade frequency (not assuming 1 trade/day). Below 0.5 = poor, 0.5-1.0 = mediocre, 1.0-2.0 = good, above 2.0 = excellent.",
                format="%.2f",
            ),
            "Calmar Ratio": st.column_config.NumberColumn(
                "Calmar Ratio",
                help="Annualized return divided by max drawdown. Measures return efficiency relative to worst-case pain. Above 1.0 is acceptable; above 3.0 is excellent.",
                format="%.2f",
            ),
            "Recovery Factor": st.column_config.NumberColumn(
                "Recovery Factor",
                help="Total (non-annualized) P&L divided by Max Drawdown. Above 1.0 means profits exceed the worst drawdown.",
                format="%.2f",
            ),
            "Equity R²": st.column_config.NumberColumn(
                "Equity R²",
                help="R-squared of a linear fit to the equity curve. Above 0.8 = consistent growth. Below 0.5 = erratic, driven by a few large trades.",
                format="%.3f",
            ),
            "WR Trend": st.column_config.NumberColumn(
                "WR Trend",
                help="Slope of win rate over time (per trade). Positive = improving hit rate. Negative = degrading edge. Near zero = stable.",
                format="%.5f",
            ),
            "Trades": st.column_config.NumberColumn(
                "Trades",
                help="Total number of trades for this strategy. Metrics from fewer than 30 trades should be treated with caution — the sample is too small for statistical reliability.",
            ),
        },
    )

    # ── Data-Driven Commentary ────────────────────────────────────────────
    st.subheader("Analysis & Recommendations")

    for row in rows:
        strat = row["Strategy"]
        sharpe = row["Sharpe Ratio"]
        calmar = row["Calmar Ratio"]
        max_dd_pct_val = row["Max DD (%)"]
        max_dd_val = row["Max DD ($)"]
        recovery = row["Recovery Factor"]
        eq_r2 = row["Equity R²"]
        wr_trend = row["WR Trend"]
        trades = row["Trades"]

        points = []

        # Sharpe assessment
        if sharpe >= 2.0:
            points.append(f"Excellent risk-adjusted performance (Sharpe {sharpe:.2f}). The strategy generates consistent returns relative to volatility.")
        elif sharpe >= 1.0:
            points.append(f"Solid Sharpe ratio of {sharpe:.2f} — returns are reasonable for the risk taken.")
        elif sharpe >= 0.5:
            points.append(f"Sharpe of {sharpe:.2f} is below average. Consider tightening entry criteria or reducing exposure to cut volatility.")
        elif sharpe > 0:
            points.append(f"Weak Sharpe of {sharpe:.2f} — the strategy barely compensates for the risk. Review whether this strategy deserves capital allocation.")
        else:
            points.append(f"Negative Sharpe ({sharpe:.2f}) — the strategy is destroying value on a risk-adjusted basis. Consider pausing or reparameterizing.")

        # Drawdown assessment
        if max_dd_pct_val > 30:
            points.append(f"Max drawdown of {max_dd_pct_val:.1f}% is dangerously high. Reduce position size by at least {int(max_dd_pct_val / 15)}x or add a circuit breaker that halts trading after 3 consecutive losses.")
        elif max_dd_pct_val > 20:
            points.append(f"Max drawdown of {max_dd_pct_val:.1f}% is aggressive. Most professional traders target max DD below 20%. Consider halving position sizes.")
        elif max_dd_pct_val > 10:
            points.append(f"Max drawdown of {max_dd_pct_val:.1f}% is within acceptable bounds but worth monitoring.")
        elif max_dd_pct_val > 0:
            points.append(f"Max drawdown of {max_dd_pct_val:.1f}% is well-controlled.")

        # Calmar assessment
        if calmar >= 3.0:
            points.append(f"Calmar ratio of {calmar:.2f} is excellent — annualized returns far exceed worst-case drawdown.")
        elif calmar < 1.0 and calmar > 0:
            points.append(f"Calmar ratio of {calmar:.2f} is below 1.0 — annualized returns don't yet justify the drawdown risk.")

        # Recovery assessment
        if recovery < 0:
            points.append("The strategy is net negative — it has not recovered from drawdowns. Immediate review needed.")
        elif recovery < 1.0:
            points.append(f"Recovery factor of {recovery:.2f} means total profits haven't yet exceeded the worst drawdown. The strategy needs more time or better edge to justify the risk.")
        elif recovery >= 3.0:
            points.append(f"Recovery factor of {recovery:.2f} is strong — profits significantly exceed the worst drawdown.")

        # Equity R² assessment
        if eq_r2 >= 0.8:
            points.append(f"Equity R² of {eq_r2:.3f} indicates very consistent growth — returns are evenly distributed, not driven by a few lucky trades.")
        elif eq_r2 >= 0.5:
            points.append(f"Equity R² of {eq_r2:.3f} is moderate — some consistency but with notable bumps. Review the largest individual trades for outsized influence.")
        elif eq_r2 > 0 and trades >= 10:
            points.append(f"Equity R² of {eq_r2:.3f} is low — the equity curve is erratic. Profits may be concentrated in a few trades rather than evenly earned.")

        # WR Trend assessment
        if trades >= 20:
            if wr_trend < -0.002:
                points.append(f"Win rate trend is declining ({wr_trend:.5f}/trade) — the strategy's edge may be degrading. Monitor closely and consider pausing if the trend continues.")
            elif wr_trend > 0.002:
                points.append(f"Win rate trend is improving ({wr_trend:+.5f}/trade) — the strategy appears to be gaining effectiveness.")

        # Sample size warning
        if trades < 30:
            points.append(f"Only {trades} trades — all metrics should be treated as preliminary. Need 30+ trades for statistical significance.")

        if points:
            with st.expander(f"**{strat}**", expanded=True):
                for p in points:
                    st.markdown(f"- {p}")

# ── Rolling Win Rate ──────────────────────────────────────────────────────────
st.subheader("Rolling Win Rate (20-trade)")
st.caption("A smoothed view of your win rate over the last 20 trades. Tracks whether your strategy's hit rate is improving, stable, or deteriorating. A declining trend here is an early warning signal — investigate before it impacts your P&L significantly.")
st.plotly_chart(rolling_win_rate(df), key="eq_rolling_wr")
