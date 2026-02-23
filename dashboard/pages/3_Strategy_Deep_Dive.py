"""Page 2: Per-strategy detailed analysis."""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Strategy Deep Dive", page_icon="🔍", layout="wide")
st.title("🔍 Strategy Deep Dive")

from data.data_loader import get_all_trades
from components.kpi_cards import strategy_kpis
from components.charts import (
    r_multiple_histogram, duration_histogram, cumulative_pnl_line,
    exit_reason_donut, rolling_win_rate, mfe_mae_scatter,
)
from components.filters import source_filter, symbol_filter, date_range_filter, apply_filters
from config import STRATEGIES

# ── Sidebar ───────────────────────────────────────────────────────────────────
strategy = st.sidebar.selectbox("Strategy", list(STRATEGIES.keys()), key="dd_strat")
src = source_filter(key_prefix="dd")
df_all = get_all_trades(source_filter=src)
symbols = symbol_filter(df_all[df_all["strategy"] == strategy], key_prefix="dd")
date_start, date_end = date_range_filter(df_all, key_prefix="dd")

# ── Filter ────────────────────────────────────────────────────────────────────
df = apply_filters(
    df_all, strategies=[strategy], symbols=symbols,
    date_start=date_start, date_end=date_end,
)

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.subheader(f"{strategy} Performance")
st.caption("Core performance metrics for this strategy in isolation. A Profit Factor above 1.5 and positive Avg R indicate a healthy edge. Compare these against the Overview page to see how this strategy contributes to the portfolio.")
strategy_kpis(df)
st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.caption("R-Multiple distribution shows how your trade outcomes cluster. A right-skewed distribution (more mass on the positive side) confirms a genuine edge. Watch for fat left tails — they indicate blow-up risk.")
    st.plotly_chart(r_multiple_histogram(df), key="dd_r_hist")
with c2:
    st.caption("Trade duration tells you how long capital is tied up per trade. Shorter durations with good R mean higher capital efficiency. Extremely long trades may indicate missed exits.")
    st.plotly_chart(duration_histogram(df), key="dd_dur_hist")

c3, c4 = st.columns(2)
with c3:
    st.caption("Cumulative P&L should trend upward steadily. Sudden drops indicate drawdown events — cross-reference with the session and monthly pages to find what caused them.")
    st.plotly_chart(cumulative_pnl_line(df), key="dd_cum_pnl")
with c4:
    st.caption("Exit reason breakdown reveals how trades end. A high proportion of stop-loss exits may mean your stops are too tight. Many time-based exits suggest the strategy isn't finding clean setups.")
    st.plotly_chart(exit_reason_donut(df), key="dd_exit_donut")

# ── MFE/MAE Analysis (Fix 8) ────────────────────────────────────────────────
if not df.empty and "mfe" in df.columns and "mae" in df.columns:
    has_mfe_mae = df["mfe"].notna().sum() >= 5 and df["mae"].notna().sum() >= 5
else:
    has_mfe_mae = False

if has_mfe_mae:
    st.markdown("---")
    st.subheader("MFE / MAE Analysis")
    st.caption("Max Favorable Excursion (MFE) vs Max Adverse Excursion (MAE) shows how far each trade moved in your favor and against you. Points above the diagonal = more upside than downside. Green points above the line are wins that ran well; red points near the line are losses that almost worked.")

    col_scatter, col_metrics = st.columns([2, 1])
    with col_scatter:
        st.plotly_chart(mfe_mae_scatter(df), key="dd_mfe_mae")

    with col_metrics:
        valid = df.dropna(subset=["mfe", "mae"])
        avg_mfe = valid["mfe"].abs().mean()
        avg_mae = valid["mae"].abs().mean()

        # Edge Ratio
        edge_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0
        st.metric("Edge Ratio", f"{edge_ratio:.2f}",
                  help="Avg MFE / Avg MAE. Above 1.0 = trades move further in your favor than against you. Good trade selection.")

        # Entry Efficiency
        avg_pnl = valid["pnl_usd"].mean()
        entry_eff = avg_pnl / avg_mfe if avg_mfe > 0 else 0
        st.metric("Entry Efficiency", f"{entry_eff:.2f}",
                  help="Avg PnL / Avg MFE. Closer to 1.0 = you're capturing most of the available move. Low values mean you're leaving profit on the table.")

        # Stop Efficiency
        losers = valid[valid["pnl_usd"] < 0]
        if not losers.empty:
            avg_loss = losers["pnl_usd"].abs().mean()
            avg_mae_losers = losers["mae"].abs().mean()
            stop_eff = avg_loss / avg_mae_losers if avg_mae_losers > 0 else 0
            st.metric("Stop Efficiency", f"{stop_eff:.2f}",
                      help="Avg Loss / Avg MAE on losing trades. Closer to 1.0 = stops are well-placed. Much above 1.0 = impossible (check data). Much below 1.0 = you're exiting before the worst point.")

        # Commentary
        st.markdown("---")
        if edge_ratio >= 1.5:
            st.success(f"Edge ratio of {edge_ratio:.2f} is excellent — trades consistently move more in your favor than against.")
        elif edge_ratio >= 1.0:
            st.info(f"Edge ratio of {edge_ratio:.2f} is positive — favorable excursions exceed adverse on average.")
        else:
            st.warning(f"Edge ratio of {edge_ratio:.2f} is below 1.0 — trades move more against you than in your favor. Review entry timing.")

# ── Regime-Conditional Performance (Fix 7) ───────────────────────────────────
if not df.empty and len(df) >= 5:
    st.markdown("---")
    st.subheader("Performance by Market Regime")
    st.caption("Breaks down performance by market condition (Trending Up, Trending Down, Ranging) using ADX and EMA indicators on 1h Binance candles. Reveals whether the strategy works in all conditions or is regime-dependent.")

    try:
        from data.binance_helpers import fetch_binance_candles, calculate_indicators, classify_regime

        # Determine trade date range
        min_date = df["entry_time"].min()
        max_date = df["entry_time"].max()
        regime_days = max(7, (max_date - min_date).days + 5)

        # Pick the primary symbol for regime data
        regime_symbols = df["symbol"].unique()
        regime_sym = regime_symbols[0] if len(regime_symbols) == 1 else "BTC"

        with st.spinner(f"Fetching {regime_sym} 1h candles for regime classification..."):
            candles = fetch_binance_candles(regime_sym, "1h", regime_days)

        if candles is not None and not candles.empty:
            candles = calculate_indicators(candles)
            candles["regime"] = candles.apply(classify_regime, axis=1)

            # Merge trades with nearest candle regime
            trade_df = df.copy()
            trade_df = trade_df.sort_values("entry_time")
            candle_regime = candles[["regime"]].copy()
            candle_regime.index = candle_regime.index.tz_localize(None) if candle_regime.index.tz else candle_regime.index

            # Ensure trade entry_time is tz-naive too
            if trade_df["entry_time"].dt.tz is not None:
                trade_df["entry_time"] = trade_df["entry_time"].dt.tz_localize(None)

            merged = pd.merge_asof(
                trade_df.sort_values("entry_time"),
                candle_regime.reset_index().rename(columns={"Timestamp": "entry_time"}),
                on="entry_time",
                direction="backward",
            )

            if "regime" in merged.columns and merged["regime"].notna().sum() > 0:
                regime_stats = []
                for regime in ["Trending Up", "Trending Down", "Ranging"]:
                    r_df = merged[merged["regime"] == regime]
                    if r_df.empty:
                        continue
                    n = len(r_df)
                    wr = (r_df["pnl_usd"] > 0).mean()
                    avg_r = r_df["r_multiple"].mean() if "r_multiple" in r_df.columns else 0
                    total_r = r_df["r_multiple"].sum() if "r_multiple" in r_df.columns else 0
                    gp = r_df.loc[r_df["pnl_usd"] > 0, "pnl_usd"].sum()
                    gl = r_df.loc[r_df["pnl_usd"] < 0, "pnl_usd"].abs().sum()
                    pf = gp / gl if gl > 0 else float("inf")
                    regime_stats.append({
                        "Regime": regime,
                        "Trades": n,
                        "Win Rate": f"{wr:.1%}",
                        "Avg R": f"{avg_r:.2f}",
                        "Total R": f"{total_r:.1f}",
                        "Profit Factor": f"{pf:.2f}" if pf < 100 else "Inf",
                    })

                if regime_stats:
                    st.dataframe(pd.DataFrame(regime_stats), use_container_width=True, hide_index=True)

                    # Commentary on best/worst regime
                    best = max(regime_stats, key=lambda x: float(x["Avg R"]))
                    worst = min(regime_stats, key=lambda x: float(x["Avg R"]))
                    if best["Regime"] != worst["Regime"]:
                        st.markdown(
                            f"- **Best regime:** {best['Regime']} (Avg R: {best['Avg R']}, "
                            f"WR: {best['Win Rate']})\n"
                            f"- **Worst regime:** {worst['Regime']} (Avg R: {worst['Avg R']}, "
                            f"WR: {worst['Win Rate']})"
                        )
                else:
                    st.info("No regime data could be matched to trades.")
            else:
                st.info("Could not classify regimes for the trade period.")
        else:
            st.info("Could not fetch Binance data for regime analysis.")
    except Exception as e:
        st.warning(f"Regime analysis unavailable: {e}")

# ── Long vs Short ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Long vs Short")
st.caption("Compares directional bias. Some strategies perform better in one direction — if Shorts have a significantly worse win rate, consider filtering them out or tightening criteria.")
if not df.empty:
    for d in ("Long", "Short"):
        sub = df[df["direction"] == d]
        if sub.empty:
            continue
        wr = (sub["pnl_usd"] > 0).mean()
        avg_r = sub["r_multiple"].mean() if "r_multiple" in sub.columns else 0
        st.write(f"**{d}**: {len(sub)} trades | Win Rate {wr:.1%} | Avg R {avg_r:.2f}")

# ── Rolling Win Rate ──────────────────────────────────────────────────────────
st.caption("Rolling win rate smooths out noise and reveals trends. A declining trend over the last 20-50 trades is an early warning that market conditions may have shifted away from this strategy's edge.")
st.plotly_chart(rolling_win_rate(df), key="dd_rolling_wr")

# ── Data-Driven Commentary ────────────────────────────────────────────────────
if not df.empty:
    st.markdown("---")
    st.subheader(f"Analysis & Recommendations for {strategy}")

    total = len(df)
    wins = (df["pnl_usd"] > 0).sum()
    wr = wins / total if total else 0
    total_pnl = df["pnl_usd"].sum()
    avg_r = df["r_multiple"].mean() if "r_multiple" in df.columns else 0
    total_r = df["r_multiple"].sum() if "r_multiple" in df.columns else 0
    avg_dur = df["duration_minutes"].mean() if "duration_minutes" in df.columns else 0

    gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = df.loc[df["pnl_usd"] < 0, "pnl_usd"].abs().sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    points = []

    # Profit Factor assessment
    if pf == float("inf"):
        points.append("No losing trades recorded — either the sample is too small or risk is being masked. Monitor closely as losses will eventually occur.")
    elif pf >= 2.0:
        points.append(f"Profit Factor of **{pf:.2f}** is excellent — gross profits are {pf:.1f}x gross losses. This is a strong edge.")
    elif pf >= 1.5:
        points.append(f"Profit Factor of **{pf:.2f}** is solid. The strategy has a meaningful edge that should persist if market conditions remain similar.")
    elif pf >= 1.0:
        points.append(f"Profit Factor of **{pf:.2f}** is marginal. The strategy is profitable but the edge is thin — commissions or slippage increases could erode it. Consider tightening entry filters.")
    else:
        points.append(f"Profit Factor of **{pf:.2f}** is below 1.0 — the strategy is losing money. Either the edge has disappeared or entry/exit rules need reworking.")

    # Win rate + R relationship
    if wr < 0.40 and avg_r > 0.3:
        points.append(f"Low win rate ({wr:.0%}) but positive Avg R ({avg_r:.2f}) suggests a trend-following profile — few winners but they're large. This is sustainable if you manage the psychological drawdown of frequent losses.")
    elif wr < 0.40 and avg_r <= 0:
        points.append(f"Low win rate ({wr:.0%}) combined with negative Avg R ({avg_r:.2f}) is a losing formula. The strategy needs a fundamental overhaul — either improve entry accuracy or widen reward targets.")
    elif wr > 0.55 and avg_r < 0:
        points.append(f"High win rate ({wr:.0%}) but negative Avg R ({avg_r:.2f}) means small frequent wins are overwhelmed by occasional large losses. Tighten your stop-losses — the big losses are destroying the edge.")

    # Direction bias
    longs = df[df["direction"] == "Long"]
    shorts = df[df["direction"] == "Short"]
    if len(longs) >= 5 and len(shorts) >= 5:
        long_wr = (longs["pnl_usd"] > 0).mean()
        short_wr = (shorts["pnl_usd"] > 0).mean()
        if abs(long_wr - short_wr) > 0.15:
            better = "Longs" if long_wr > short_wr else "Shorts"
            worse = "Shorts" if better == "Longs" else "Longs"
            points.append(f"**{better}** significantly outperform **{worse}** ({max(long_wr, short_wr):.0%} vs {min(long_wr, short_wr):.0%} win rate). Consider disabling the weaker direction or applying stricter entry filters.")

    # Duration analysis
    if pd.notna(avg_dur) and avg_dur > 0:
        if avg_dur > 1440:  # > 1 day
            points.append(f"Average trade duration is **{avg_dur/60:.1f} hours** — capital is locked up for extended periods. Ensure the R-multiple justifies the holding time, or add a time-based exit to free capital sooner.")
        elif avg_dur < 5:
            points.append(f"Average duration of **{avg_dur:.0f} minutes** is extremely short. Verify these aren't erroneous fills and that commissions/slippage aren't eating into your edge.")

    # Exit reason analysis
    if "exit_reason" in df.columns:
        exit_counts = df["exit_reason"].value_counts(normalize=True)
        for reason, pct in exit_counts.items():
            if "stop" in str(reason).lower() and pct > 0.5:
                points.append(f"**{pct:.0%}** of trades exit via stop-loss. This is very high — either stops are too tight or entries are poorly timed. Try widening stops by 0.5x ATR and backtesting the result.")
            elif "time" in str(reason).lower() and pct > 0.3:
                points.append(f"**{pct:.0%}** of trades exit by timeout. The strategy isn't reaching its targets frequently enough — consider adjusting take-profit levels or entry timing.")

    # Sample size
    if total < 30:
        points.append(f"Only **{total} trades** — these observations are preliminary. Collect at least 30 trades before making strategy changes.")

    for p in points:
        st.markdown(f"- {p}")
