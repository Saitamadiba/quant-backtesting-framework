"""Reusable Plotly chart builders."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from config import STRATEGY_COLORS, INITIAL_BALANCE


# ── Overlay helpers ──────────────────────────────────────────────────────────

REGIME_COLORS = {
    "Trending Up":   "rgba(76, 175, 80, 0.10)",   # green
    "Trending Down": "rgba(244, 67, 54, 0.10)",    # red
    "Ranging":       "rgba(158, 158, 158, 0.08)",  # gray
}


def _add_regime_bands(fig, regime_df: pd.DataFrame) -> None:
    """Add semi-transparent background rectangles for market regime periods.

    Args:
        fig: Plotly Figure to add shapes to.
        regime_df: DataFrame with DatetimeIndex and a 'regime' column.
                   Should be pre-classified via classify_regime().
    """
    if regime_df is None or regime_df.empty or "regime" not in regime_df.columns:
        return

    # Consolidate contiguous blocks of the same regime
    regimes = regime_df["regime"]
    blocks = []
    prev_regime = None
    block_start = None

    for ts, regime in regimes.items():
        if regime != prev_regime:
            if prev_regime is not None:
                blocks.append((block_start, ts, prev_regime))
            block_start = ts
            prev_regime = regime
        last_ts = ts
    if prev_regime is not None:
        blocks.append((block_start, last_ts, prev_regime))

    # Add shapes — limit to avoid performance issues on huge datasets
    for start, end, regime in blocks[-500:]:
        color = REGIME_COLORS.get(regime, "rgba(100,100,100,0.05)")
        fig.add_shape(
            type="rect",
            x0=start, x1=end,
            y0=0, y1=1,
            yref="paper",
            fillcolor=color,
            line_width=0,
            layer="below",
        )

    # Add a single invisible trace per regime for the legend
    for regime, color in REGIME_COLORS.items():
        # Check if this regime actually appears
        if regime in regimes.values:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color.replace("0.10", "0.5").replace("0.08", "0.4"),
                            symbol="square"),
                name=regime, showlegend=True,
            ))


def _add_change_markers(fig, changes: list, active_strategies: list = None) -> None:
    """Add vertical lines with annotations for strategy change deployments.

    Args:
        fig: Plotly Figure.
        changes: List of dicts from STRATEGY_CHANGELOG.
        active_strategies: If provided, only show changes relevant to these strategies.
    """
    if not changes:
        return

    for ch in changes:
        # Filter to relevant strategies
        ch_strats = ch.get("strategies", ["ALL"])
        if active_strategies and "ALL" not in ch_strats:
            if not any(s in active_strategies for s in ch_strats):
                continue

        date_val = pd.Timestamp(ch["date"])
        label = ch.get("label", "Change")
        color = ch.get("color", "#FFFFFF")

        fig.add_shape(
            type="line", x0=date_val, x1=date_val,
            y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color=color, width=1.5),
        )
        fig.add_annotation(
            x=date_val, y=1, yref="paper",
            text=label, showarrow=False,
            font=dict(size=10, color=color),
            yshift=10,
        )


def _get_color(strategy: str) -> str:
    """Return the theme color for a strategy, falling back to blue-grey."""
    return STRATEGY_COLORS.get(strategy, "#607D8B")


# ── Bar Charts ────────────────────────────────────────────────────────────────

def strategy_comparison_bar(stats_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing strategy metrics."""
    if stats_df.empty:
        return go.Figure()
    fig = px.bar(
        stats_df, x="strategy", y="Total PnL", color="symbol",
        barmode="group", title="Total P&L by Strategy & Symbol",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra>%{fullData.name}</extra>",
    )
    fig.update_layout(template="plotly_dark", height=400)
    return fig


def monthly_pnl_bar(df: pd.DataFrame) -> go.Figure:
    """Monthly P&L bar chart."""
    if df.empty or "entry_time" not in df.columns:
        return go.Figure()
    tmp = df.copy()
    tmp["month"] = tmp["entry_time"].dt.to_period("M").astype(str)
    monthly = tmp.groupby("month")["pnl_usd"].sum().reset_index()
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in monthly["pnl_usd"]]
    fig = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["pnl_usd"], marker_color=colors,
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(title="Monthly P&L", template="plotly_dark", height=400,
                      xaxis_title="Month", yaxis_title="P&L ($)")
    return fig


# ── Line Charts ───────────────────────────────────────────────────────────────

def cumulative_pnl_line(df: pd.DataFrame) -> go.Figure:
    """Cumulative P&L line per strategy."""
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    for strat in df["strategy"].unique():
        s = df[df["strategy"] == strat].sort_values("entry_time")
        cum = s["pnl_usd"].cumsum()
        fig.add_trace(go.Scatter(
            x=s["entry_time"], y=cum, name=strat, mode="lines",
            line=dict(color=_get_color(strat)),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Cumulative P&L: $%{y:,.2f}<extra></extra>"
            ),
        ))
    fig.update_layout(title="Cumulative P&L", template="plotly_dark", height=400,
                      xaxis_title="Date", yaxis_title="Cumulative P&L ($)")
    return fig


def equity_curve(df: pd.DataFrame, initial_balance: float = 10000,
                 benchmark: pd.DataFrame = None,
                 regime_df: pd.DataFrame = None,
                 strategy_changes: list = None) -> go.Figure:
    """Multi-strategy equity curve with optional benchmark, regime bands, and change markers."""
    if df.empty:
        return go.Figure()
    fig = go.Figure()

    # Regime background bands (rendered first so they sit behind everything)
    _add_regime_bands(fig, regime_df)

    active_strategies = list(df["strategy"].unique())

    for strat in active_strategies:
        s = df[df["strategy"] == strat].sort_values("entry_time")
        equity = initial_balance + s["pnl_usd"].cumsum()
        fig.add_trace(go.Scatter(
            x=s["entry_time"], y=equity, name=strat, mode="lines",
            line=dict(color=_get_color(strat)),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Equity: $%{y:,.2f}<extra></extra>"
            ),
        ))

    # Buy-and-hold benchmark overlay
    if benchmark is not None and not benchmark.empty and "Close" in benchmark.columns:
        close = benchmark["Close"]
        bm_equity = initial_balance * (close / close.iloc[0])
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=bm_equity, name="Buy & Hold",
            mode="lines",
            line=dict(color="#FFD700", width=2, dash="dash"),
            hovertemplate=(
                "<b>Buy & Hold</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Equity: $%{y:,.2f}<extra></extra>"
            ),
        ))

    # Strategy change deployment markers
    _add_change_markers(fig, strategy_changes or [], active_strategies)

    fig.update_layout(title="Equity Curves", template="plotly_dark", height=450,
                      xaxis_title="Date", yaxis_title="Balance ($)")
    return fig


def drawdown_chart(df: pd.DataFrame, regime_df: pd.DataFrame = None,
                   strategy_changes: list = None) -> go.Figure:
    """Drawdown (underwater) plot per strategy — uses account equity, not raw PnL."""
    if df.empty:
        return go.Figure()
    fig = go.Figure()

    _add_regime_bands(fig, regime_df)

    active_strategies = list(df["strategy"].unique())
    for strat in active_strategies:
        s = df[df["strategy"] == strat].sort_values("entry_time")
        cum = s["pnl_usd"].cumsum()
        # Fix: compute DD% from account equity, not raw cumulative PnL
        cum_equity = INITIAL_BALANCE + cum
        peak_equity = cum_equity.cummax()
        dd_pct = ((cum_equity - peak_equity) / peak_equity) * 100
        fig.add_trace(go.Scatter(
            x=s["entry_time"], y=dd_pct, name=strat, mode="lines",
            fill="tozeroy", line=dict(color=_get_color(strat)),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Drawdown: %{y:.1f}%<extra></extra>"
            ),
        ))

    _add_change_markers(fig, strategy_changes or [], active_strategies)

    fig.update_layout(title="Drawdown (%)", template="plotly_dark", height=350,
                      xaxis_title="Date", yaxis_title="Drawdown %")
    return fig


def rolling_win_rate(df: pd.DataFrame, window: int = 20) -> go.Figure:
    """Rolling win-rate over last N trades."""
    if df.empty or len(df) < window:
        return go.Figure()
    fig = go.Figure()
    for strat in df["strategy"].unique():
        s = df[df["strategy"] == strat].sort_values("entry_time")
        wins = (s["pnl_usd"] > 0).rolling(window).mean()
        fig.add_trace(go.Scatter(
            x=s["entry_time"], y=wins, name=strat, mode="lines",
            line=dict(color=_get_color(strat)),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Win Rate: %{y:.1%}<extra></extra>"
            ),
        ))
    fig.update_layout(title=f"Rolling Win Rate ({window}-trade)", template="plotly_dark",
                      height=350, yaxis_tickformat=".0%")
    return fig


# ── Histograms ────────────────────────────────────────────────────────────────

def r_multiple_histogram(df: pd.DataFrame) -> go.Figure:
    """R-multiple distribution histogram."""
    if df.empty or "r_multiple" not in df.columns:
        return go.Figure()
    r = df["r_multiple"].dropna()
    fig = go.Figure(go.Histogram(
        x=r, marker_color="#2196F3", nbinsx=40,
        hovertemplate="R-Multiple: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="white")
    fig.update_layout(title="R-Multiple Distribution", template="plotly_dark", height=350,
                      xaxis_title="R-Multiple", yaxis_title="Count")
    return fig


def duration_histogram(df: pd.DataFrame) -> go.Figure:
    """Trade duration distribution."""
    if df.empty or "duration_minutes" not in df.columns:
        return go.Figure()
    dur = df["duration_minutes"].dropna()
    fig = go.Figure(go.Histogram(
        x=dur, marker_color="#FF9800", nbinsx=30,
        hovertemplate="Duration: %{x:.0f} min<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Trade Duration Distribution", template="plotly_dark", height=350,
                      xaxis_title="Duration (min)", yaxis_title="Count")
    return fig


# ── Pie / Donut ───────────────────────────────────────────────────────────────

def exit_reason_donut(df: pd.DataFrame) -> go.Figure:
    """Exit reason donut chart."""
    if df.empty or "exit_reason" not in df.columns:
        return go.Figure()
    counts = df["exit_reason"].fillna("Unknown").value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.4,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(title="Exit Reasons", template="plotly_dark", height=350)
    return fig


# ── Heatmaps ──────────────────────────────────────────────────────────────────

def session_strategy_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: Strategies x Sessions (color = win rate)."""
    if df.empty:
        return go.Figure()
    pivot = df.groupby(["strategy", "session"]).apply(
        lambda g: (g["pnl_usd"] > 0).mean()
    ).unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdYlGn", zmin=0, zmax=1,
        text=[[f"{v:.0%}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> during <b>%{x}</b><br>Win Rate: %{text}<extra></extra>",
    ))
    fig.update_layout(title="Win Rate: Strategy x Session", template="plotly_dark", height=350)
    return fig


def monthly_calendar_heatmap(df: pd.DataFrame) -> go.Figure:
    """Year x Month calendar heatmap of total R."""
    if df.empty or "entry_time" not in df.columns or "r_multiple" not in df.columns:
        return go.Figure()
    tmp = df.copy()
    tmp["year"] = tmp["entry_time"].dt.year
    tmp["month"] = tmp["entry_time"].dt.month
    pivot = tmp.groupby(["year", "month"])["r_multiple"].sum().unstack(fill_value=0)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Ensure all 12 months
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0
    pivot = pivot[sorted(pivot.columns)]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=month_labels[:len(pivot.columns)],
        y=[str(y) for y in pivot.index],
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.1f}R" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} %{x}</b><br>Total R: %{text}<extra></extra>",
    ))
    fig.update_layout(title="Monthly Performance (Total R)", template="plotly_dark", height=300)
    return fig


# ── MFE/MAE Scatter ──────────────────────────────────────────────────────────

def mfe_mae_scatter(df: pd.DataFrame) -> go.Figure:
    """MFE vs MAE scatter plot, colored by win/loss, with MFE=MAE diagonal."""
    if df.empty:
        return go.Figure()

    mfe = df["mfe"].dropna()
    mae = df["mae"].dropna()
    # Only use rows with both values
    valid = df.dropna(subset=["mfe", "mae"])
    if valid.empty:
        return go.Figure()

    is_win = valid["pnl_usd"] > 0
    fig = go.Figure()

    # Wins
    wins = valid[is_win]
    if not wins.empty:
        fig.add_trace(go.Scatter(
            x=wins["mae"].abs(), y=wins["mfe"].abs(),
            mode="markers", name="Win",
            marker=dict(color="#4CAF50", size=6, opacity=0.7),
            hovertemplate="MAE: %{x:.2f}<br>MFE: %{y:.2f}<extra>Win</extra>",
        ))

    # Losses
    losses = valid[~is_win]
    if not losses.empty:
        fig.add_trace(go.Scatter(
            x=losses["mae"].abs(), y=losses["mfe"].abs(),
            mode="markers", name="Loss",
            marker=dict(color="#F44336", size=6, opacity=0.7),
            hovertemplate="MAE: %{x:.2f}<br>MFE: %{y:.2f}<extra>Loss</extra>",
        ))

    # Diagonal line (MFE = MAE)
    max_val = max(valid["mfe"].abs().max(), valid["mae"].abs().max(), 1)
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="MFE=MAE",
        line=dict(color="white", dash="dot", width=1),
        showlegend=False,
    ))

    fig.update_layout(
        title="MFE vs MAE",
        template="plotly_dark", height=400,
        xaxis_title="MAE (Max Adverse Excursion)",
        yaxis_title="MFE (Max Favorable Excursion)",
    )
    return fig


# ── Monte Carlo Charts ────────────────────────────────────────────────────────

def mc_distribution_chart(returns: list) -> go.Figure:
    """Monte Carlo return distribution histogram with percentile markers."""
    if not returns:
        return go.Figure()
    arr = np.array(returns)
    p5, p50, p95 = np.percentile(arr, [5, 50, 95])
    fig = go.Figure(go.Histogram(
        x=arr, nbinsx=80, marker_color="#2196F3",
        hovertemplate="Return: %{x:.1f}%<br>Frequency: %{y}<extra></extra>",
    ))
    for val, label, color in [(p5, "5th %ile", "#F44336"), (p50, "Median", "#FFFFFF"), (p95, "95th %ile", "#4CAF50")]:
        fig.add_vline(x=val, line_dash="dash", line_color=color, annotation_text=f"{label}: {val:.1f}%")
    fig.update_layout(title="Monte Carlo Return Distribution", template="plotly_dark", height=400,
                      xaxis_title="Return (%)", yaxis_title="Frequency")
    return fig


def mc_equity_fan(equity_paths: list, percentiles: dict = None) -> go.Figure:
    """Fan chart showing sample equity paths + confidence bands."""
    if not equity_paths:
        return go.Figure()
    fig = go.Figure()
    # Plot sample paths (up to 100)
    for i, path in enumerate(equity_paths[:100]):
        fig.add_trace(go.Scatter(
            y=path, mode="lines", line=dict(color="rgba(33,150,243,0.1)", width=0.5),
            showlegend=False, hoverinfo="skip",
        ))
    # Median path
    arr = np.array(equity_paths[:200])
    if arr.shape[0] >= 5:
        median_path = np.median(arr, axis=0)
        p5_path = np.percentile(arr, 5, axis=0)
        p95_path = np.percentile(arr, 95, axis=0)
        fig.add_trace(go.Scatter(
            y=median_path, mode="lines", name="Median",
            line=dict(color="#FFFFFF", width=2),
            hovertemplate="Trade #%{x}<br>Median Equity: $%{y:,.0f}<extra>Median</extra>",
        ))
        fig.add_trace(go.Scatter(
            y=p5_path, mode="lines", name="5th %ile",
            line=dict(color="#F44336", width=1, dash="dot"),
            hovertemplate="Trade #%{x}<br>5th %ile: $%{y:,.0f}<extra>Worst case</extra>",
        ))
        fig.add_trace(go.Scatter(
            y=p95_path, mode="lines", name="95th %ile",
            line=dict(color="#4CAF50", width=1, dash="dot"),
            hovertemplate="Trade #%{x}<br>95th %ile: $%{y:,.0f}<extra>Best case</extra>",
        ))
    # Confidence bands if provided
    if percentiles:
        for label, vals in percentiles.items():
            fig.add_trace(go.Scatter(y=vals, mode="lines", name=label,
                                     line=dict(width=2)))
    fig.update_layout(title="Equity Fan Chart", template="plotly_dark", height=400,
                      xaxis_title="Trade #", yaxis_title="Equity")
    return fig
