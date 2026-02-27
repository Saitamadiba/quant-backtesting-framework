"""Page 17: ML Filter Performance Scoring & Attribution."""

import sys
import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ML Performance", page_icon="🎯", layout="wide")
st.title("🎯 ML Filter Performance")

from config import (
    ML_PREDICTIONS_DB, ML_PERFORMANCE_SCORER, BASE_DIR,
    STRATEGY_COLORS, VPS_CACHE_DIR,
)

# Ensure project root is importable for ml_performance_scorer
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from ml_performance_scorer import (
        PredictionLogger,
        MLAttributionMetrics,
        DriftDetector,
    )
    _SCORER_AVAILABLE = True
except ImportError as exc:
    _SCORER_AVAILABLE = False
    _SCORER_ERROR = str(exc)

# ── Helpers ──────────────────────────────────────────────────────────────────

DARK_LAYOUT = dict(
    template="plotly_dark",
    height=400,
    margin=dict(l=40, r=20, t=40, b=40),
)


def _load_predictions_df(strategy=None, symbol=None) -> pd.DataFrame:
    """Load reconciled predictions directly from SQLite (avoids PredictionLogger init issues)."""
    db_path = ML_PREDICTIONS_DB
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(db_path))
        query = """
            SELECT
                p.trade_id, p.strategy, p.symbol,
                p.predicted_prob, p.threshold, p.was_filtered,
                p.filter_reason, p.timestamp AS prediction_ts,
                o.actual_outcome, o.r_multiple, o.pnl_usd, o.reconciled_at
            FROM predictions p
            INNER JOIN outcomes o ON p.trade_id = o.trade_id
            WHERE 1=1
        """
        params = []
        if strategy:
            query += " AND p.strategy = ?"
            params.append(strategy)
        if symbol:
            query += " AND p.symbol = ?"
            params.append(symbol)
        query += " ORDER BY p.timestamp"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _load_all_predictions() -> pd.DataFrame:
    """Load all predictions (including unreconciled) for overview stats."""
    db_path = ML_PREDICTIONS_DB
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY timestamp", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── Pre-flight checks ────────────────────────────────────────────────────────

if not _SCORER_AVAILABLE:
    st.error(f"Could not import ml_performance_scorer: {_SCORER_ERROR}")
    st.info("Ensure ml_performance_scorer.py exists at the project root and its dependencies (numpy, pandas, sklearn) are installed.")
    st.stop()

if not ML_PREDICTIONS_DB.exists():
    st.warning("No ML predictions database found.")
    st.caption(
        f"Expected at: `{ML_PREDICTIONS_DB}`\n\n"
        "The ML filter must be active on live bots with a PredictionLogger attached. "
        "Once trades are logged and reconciled, this page will display performance metrics."
    )
    st.stop()

# ── Load data ────────────────────────────────────────────────────────────────

all_preds = _load_all_predictions()
reconciled = _load_predictions_df()

if all_preds.empty:
    st.warning("No ML predictions have been logged yet.")
    st.caption("Attach a PredictionLogger to your MLTradeFilter instances on the live bots. Each filter decision will be recorded automatically.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")
    strategies = sorted(all_preds["strategy"].dropna().unique().tolist())
    selected_strategy = st.selectbox(
        "Strategy", ["All"] + strategies, key="ml_perf_strat"
    )
    symbols = sorted(all_preds["symbol"].dropna().unique().tolist())
    selected_symbol = st.selectbox(
        "Symbol", ["All"] + symbols, key="ml_perf_sym"
    )

strat_filter = None if selected_strategy == "All" else selected_strategy
sym_filter = None if selected_symbol == "All" else selected_symbol

# Re-load with filters
df = _load_predictions_df(strategy=strat_filter, symbol=sym_filter)

# ══════════════════════════════════════════════════════════════════════════════
# Section A: Overview KPIs
# ══════════════════════════════════════════════════════════════════════════════

st.header("A. Overview")
st.caption(
    "High-level stats on ML filter activity. Reconciled predictions have known "
    "outcomes (win/loss); unreconciled are still pending or were never closed."
)

# Filter all_preds by sidebar too
ap = all_preds.copy()
if strat_filter:
    ap = ap[ap["strategy"] == strat_filter]
if sym_filter:
    ap = ap[ap["symbol"] == sym_filter]

total_logged = len(ap)
total_filtered = int(ap["was_filtered"].sum()) if "was_filtered" in ap.columns else 0
total_passed = total_logged - total_filtered
total_reconciled = len(df)
pct_reconciled = total_reconciled / total_logged if total_logged > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Predictions", f"{total_logged:,}")
c2.metric("Filtered (Blocked)", f"{total_filtered:,}")
c3.metric("Passed (Traded)", f"{total_passed:,}")
c4.metric("Reconciled", f"{total_reconciled:,} ({pct_reconciled:.0%})")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section B: Attribution Metrics (needs reconciled data)
# ══════════════════════════════════════════════════════════════════════════════

if df.empty:
    st.info("No reconciled predictions available for the current filter selection. Metrics require both a prediction and an outcome for each trade.")
    st.stop()

metrics = MLAttributionMetrics(df)
all_metrics = metrics.compute_all()

st.header("B. Classification Performance")
st.caption(
    "How well does the ML filter separate winners from losers? "
    "Accuracy = fraction of correct filter decisions. Precision = of all trades we blocked, "
    "how many were truly losers? Recall = of all losers, how many did we catch?"
)

acc = all_metrics["accuracy"]
pr = all_metrics["precision_recall"]
brier = all_metrics["brier_score"]
ece = all_metrics["expected_calibration_error"]
roc = all_metrics["roc_auc"]
pr_auc = all_metrics["pr_auc"]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{acc:.1%}")
c2.metric("Precision", f"{pr['precision']:.1%}")
c3.metric("Recall", f"{pr['recall']:.1%}")
c4.metric("Brier Score", f"{brier:.4f}", help="Lower is better. 0 = perfect, 0.25 = random.")
c5.metric("ECE", f"{ece:.4f}", help="Expected Calibration Error. Lower = better calibrated.")

roc_str = f"{roc:.3f}" if not np.isnan(roc) else "N/A"
c6.metric("ROC-AUC", roc_str, help="Area under ROC curve. 0.5 = random, 1.0 = perfect.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section C: Calibration Plot
# ══════════════════════════════════════════════════════════════════════════════

st.header("C. Calibration")
st.caption(
    "Compares the model's predicted probability with the actual win rate in each "
    "probability bin. A perfectly calibrated model follows the diagonal. "
    "Points above the diagonal mean the model is under-confident; below means over-confident."
)

cal_table = all_metrics["calibration_table"]

if not cal_table.empty and cal_table["count"].sum() > 0:
    cal_plot = cal_table[cal_table["count"] > 0].copy()

    fig_cal = go.Figure()

    # Perfect calibration line
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="gray", width=1),
        name="Perfect Calibration", showlegend=True,
    ))

    # Actual calibration
    fig_cal.add_trace(go.Scatter(
        x=cal_plot["mean_predicted_prob"],
        y=cal_plot["actual_win_rate"],
        mode="lines+markers",
        marker=dict(size=cal_plot["count"].clip(upper=30), color="#2196F3"),
        line=dict(color="#2196F3", width=2),
        name="Model",
        text=[f"n={int(c)}" for c in cal_plot["count"]],
        hovertemplate="Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}<extra></extra>",
    ))

    fig_cal.update_layout(
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Actual Win Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        **DARK_LAYOUT,
    )
    st.plotly_chart(fig_cal, use_container_width=True, key="calibration_plot")

    # Show raw table
    with st.expander("Calibration Table"):
        display_cal = cal_table.copy()
        display_cal["mean_predicted_prob"] = display_cal["mean_predicted_prob"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
        display_cal["actual_win_rate"] = display_cal["actual_win_rate"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
        st.dataframe(display_cal, use_container_width=True, hide_index=True)
else:
    st.info("Insufficient data to build a calibration plot.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section D: ROC Curve
# ══════════════════════════════════════════════════════════════════════════════

st.header("D. ROC Curve")
st.caption(
    "Receiver Operating Characteristic curve. Shows the trade-off between true "
    "positive rate (catching losers) and false positive rate (incorrectly blocking winners) "
    "at various probability thresholds. AUC > 0.5 means the model has predictive power."
)

try:
    from sklearn.metrics import roc_curve

    df_roc = df.copy()
    df_roc["label"] = (df_roc["actual_outcome"] == "win").astype(int)

    if df_roc["label"].nunique() > 1:
        fpr, tpr, thresholds = roc_curve(df_roc["label"], df_roc["predicted_prob"])

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line=dict(dash="dash", color="gray", width=1),
            name="Random (AUC=0.50)", showlegend=True,
        ))
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            line=dict(color="#FF9800", width=2),
            name=f"Model (AUC={roc:.3f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig_roc, use_container_width=True, key="roc_curve")
    else:
        st.info("ROC curve requires both wins and losses in the reconciled data.")
except ImportError:
    st.warning("scikit-learn is required for ROC curves. Install with: `pip install scikit-learn`")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section E: Economic Value
# ══════════════════════════════════════════════════════════════════════════════

st.header("E. Economic Value")
st.caption(
    "The bottom line: how much did the ML filter save (by blocking losing trades) "
    "versus how much it cost (by blocking trades that would have won)? "
    "Net ML Edge = Dollars Saved - Dollars Missed."
)

econ = all_metrics["economic_value"]

col1, col2, col3 = st.columns(3)
col1.metric(
    "Saved (Filtered Losers)",
    f"${econ['dollars_saved_filtering_losers']:,.2f}",
    help="Total absolute loss avoided by correctly filtering losing trades.",
)
col2.metric(
    "Missed (Filtered Winners)",
    f"${econ['dollars_missed_filtering_winners']:,.2f}",
    help="Total profit missed by incorrectly filtering winning trades.",
)
net_usd = econ["net_ml_edge_usd"]
net_color = "normal" if net_usd >= 0 else "inverse"
col3.metric(
    "Net ML Edge",
    f"${net_usd:,.2f}",
    delta=f"{econ['net_ml_edge_r']:.2f}R",
    delta_color=net_color,
)

# R-multiple breakdown
with st.expander("R-Multiple Breakdown"):
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("R Saved", f"{econ['r_saved']:.2f}R")
    rc2.metric("R Missed", f"{econ['r_missed']:.2f}R")
    rc3.metric("Net R Edge", f"{econ['net_ml_edge_r']:.2f}R")

# Waterfall chart
fig_waterfall = go.Figure(go.Waterfall(
    name="Economic Value",
    orientation="v",
    x=["Saved (Blocked Losers)", "Missed (Blocked Winners)", "Net ML Edge"],
    y=[econ["dollars_saved_filtering_losers"],
       -econ["dollars_missed_filtering_winners"],
       econ["net_ml_edge_usd"]],
    measure=["relative", "relative", "total"],
    connector=dict(line=dict(color="rgba(150,150,150,0.4)")),
    increasing=dict(marker=dict(color="#4CAF50")),
    decreasing=dict(marker=dict(color="#F44336")),
    totals=dict(marker=dict(color="#2196F3")),
    text=[f"${econ['dollars_saved_filtering_losers']:,.0f}",
          f"-${econ['dollars_missed_filtering_winners']:,.0f}",
          f"${econ['net_ml_edge_usd']:,.0f}"],
    textposition="outside",
))
fig_waterfall.update_layout(
    yaxis_title="USD",
    showlegend=False,
    **DARK_LAYOUT,
)
st.plotly_chart(fig_waterfall, use_container_width=True, key="econ_waterfall")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section F: Prediction Distribution
# ══════════════════════════════════════════════════════════════════════════════

st.header("F. Prediction Distribution")
st.caption(
    "Histogram of predicted win probabilities split by actual outcome. "
    "A useful model should show wins skewing right (higher probability) "
    "and losses skewing left."
)

df_hist = df.copy()
df_hist["Outcome"] = df_hist["actual_outcome"].str.capitalize()

fig_dist = px.histogram(
    df_hist,
    x="predicted_prob",
    color="Outcome",
    nbins=20,
    barmode="overlay",
    opacity=0.6,
    color_discrete_map={"Win": "#4CAF50", "Loss": "#F44336", "Breakeven": "#9E9E9E"},
    labels={"predicted_prob": "Predicted Win Probability"},
)
fig_dist.update_layout(**DARK_LAYOUT)

# Add threshold line
threshold = df["threshold"].mode().iloc[0] if not df["threshold"].empty else 0.30
fig_dist.add_vline(
    x=threshold, line_dash="dash", line_color="#FF9800",
    annotation_text=f"Threshold ({threshold:.0%})",
    annotation_position="top right",
)
st.plotly_chart(fig_dist, use_container_width=True, key="pred_distribution")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section G: Per-Strategy Breakdown
# ══════════════════════════════════════════════════════════════════════════════

if selected_strategy == "All" and df["strategy"].nunique() > 1:
    st.header("G. Per-Strategy Breakdown")
    st.caption("Metrics computed independently for each strategy that has reconciled predictions.")

    strat_rows = []
    for strat_name in sorted(df["strategy"].unique()):
        strat_df = df[df["strategy"] == strat_name]
        if len(strat_df) < 5:
            continue
        sm = MLAttributionMetrics(strat_df)
        sa = sm.compute_all()
        econ_s = sa["economic_value"]
        strat_rows.append({
            "Strategy": strat_name,
            "N Trades": len(strat_df),
            "Accuracy": f"{sa['accuracy']:.1%}",
            "Precision": f"{sa['precision_recall']['precision']:.1%}",
            "Recall": f"{sa['precision_recall']['recall']:.1%}",
            "Brier": f"{sa['brier_score']:.4f}",
            "ROC-AUC": f"{sa['roc_auc']:.3f}" if not np.isnan(sa['roc_auc']) else "N/A",
            "Net Edge ($)": f"${econ_s['net_ml_edge_usd']:,.2f}",
            "Net Edge (R)": f"{econ_s['net_ml_edge_r']:.2f}R",
        })

    if strat_rows:
        st.dataframe(
            pd.DataFrame(strat_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Each strategy needs at least 5 reconciled predictions for a breakdown.")

    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section H: Drift Detection
# ══════════════════════════════════════════════════════════════════════════════

st.header("H. Model Drift Detection")
st.caption(
    "Rolling accuracy over the last N reconciled predictions. Monitors whether "
    "the ML filter's effectiveness is stable, improving, or degrading over time. "
    "A sustained drop below 50% indicates the model may need retraining."
)

drift_window = st.slider(
    "Rolling window size", min_value=10, max_value=100, value=30,
    step=5, key="drift_window",
)

df_drift = df.copy()
df_drift["prediction_ts"] = pd.to_datetime(df_drift["prediction_ts"], errors="coerce")
df_drift = df_drift.sort_values("prediction_ts").reset_index(drop=True)

# Compute rolling accuracy
df_drift["correct"] = (
    (df_drift["was_filtered"].astype(bool) & (df_drift["actual_outcome"] != "win"))
    | (~df_drift["was_filtered"].astype(bool) & (df_drift["actual_outcome"] == "win"))
).astype(int)

df_drift["rolling_accuracy"] = (
    df_drift["correct"].rolling(window=drift_window, min_periods=max(5, drift_window // 3)).mean()
)

fig_drift = go.Figure()
fig_drift.add_trace(go.Scatter(
    x=df_drift["prediction_ts"],
    y=df_drift["rolling_accuracy"],
    mode="lines",
    line=dict(color="#2196F3", width=2),
    name=f"Rolling Accuracy (w={drift_window})",
    hovertemplate="%{x}<br>Accuracy: %{y:.1%}<extra></extra>",
))

# Threshold lines
fig_drift.add_hline(y=0.50, line_dash="dash", line_color="#F44336",
                     annotation_text="50% (random)", annotation_position="bottom right")
fig_drift.add_hline(y=0.60, line_dash="dot", line_color="#4CAF50",
                     annotation_text="60% (target)", annotation_position="top right")

fig_drift.update_layout(
    xaxis_title="Time",
    yaxis_title="Rolling Accuracy",
    yaxis=dict(range=[0, 1]),
    **DARK_LAYOUT,
)
st.plotly_chart(fig_drift, use_container_width=True, key="drift_chart")

# Trend assessment
detector = DriftDetector(window_size=drift_window)
for _, row in df_drift.iterrows():
    detector.update(bool(row["correct"]))

current_acc = detector.rolling_accuracy()
current_trend = detector.trend()
is_drifting = detector.is_drifting(threshold=0.50)

tc1, tc2, tc3 = st.columns(3)
tc1.metric("Current Rolling Accuracy", f"{current_acc:.1%}")
trend_icons = {"improving": "+", "degrading": "-", "stable": ""}
tc2.metric("Trend", current_trend.capitalize(),
           delta=trend_icons.get(current_trend, ""),
           delta_color="normal" if current_trend != "degrading" else "inverse")
if is_drifting:
    tc3.error("DRIFT DETECTED")
else:
    tc3.success("No drift")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section I: Verdict
# ══════════════════════════════════════════════════════════════════════════════

st.header("I. Verdict")
st.caption("Overall assessment of whether the ML filter is adding value to your trading operation.")

n_trades = len(df)
net_edge = econ["net_ml_edge_usd"]
net_edge_r = econ["net_ml_edge_r"]

# Build verdict
verdict_parts = []
verdict_color = "green"

if n_trades < 30:
    verdict_parts.append(f"Insufficient sample size ({n_trades} trades). Need 30+ reconciled predictions for reliable assessment.")
    verdict_color = "orange"
else:
    # Accuracy assessment
    if acc >= 0.55:
        verdict_parts.append(f"Filter accuracy ({acc:.1%}) is above random chance.")
    else:
        verdict_parts.append(f"Filter accuracy ({acc:.1%}) is near or below random chance.")
        verdict_color = "red"

    # Economic value
    if net_edge > 0:
        verdict_parts.append(f"Net economic value is positive (${net_edge:,.2f} / {net_edge_r:.2f}R).")
    else:
        verdict_parts.append(f"Net economic value is negative (${net_edge:,.2f} / {net_edge_r:.2f}R).")
        verdict_color = "red"

    # Calibration
    if ece < 0.10:
        verdict_parts.append("Model is well-calibrated (ECE < 10%).")
    elif ece < 0.20:
        verdict_parts.append(f"Model calibration is acceptable (ECE = {ece:.1%}).")
    else:
        verdict_parts.append(f"Model is poorly calibrated (ECE = {ece:.1%}). Consider recalibrating.")

    # Drift
    if is_drifting:
        verdict_parts.append("Model drift detected. Consider retraining.")
        verdict_color = "red"

# Final verdict
providing_edge = acc >= 0.55 and net_edge > 0 and n_trades >= 30

if providing_edge:
    verdict_header = "ML filter IS providing a statistically meaningful edge."
elif n_trades < 30:
    verdict_header = "Insufficient data to determine ML filter effectiveness."
else:
    verdict_header = "ML filter is NOT providing a clear edge. Review model or threshold."

verdict_text = "\n\n".join([f"**{verdict_header}**"] + [f"- {v}" for v in verdict_parts])

if verdict_color == "green":
    st.success(verdict_text)
elif verdict_color == "orange":
    st.warning(verdict_text)
else:
    st.error(verdict_text)

# ── Raw Data ─────────────────────────────────────────────────────────────────

with st.expander("Raw Reconciled Predictions"):
    display_df = df.copy()
    for col in ["predicted_prob", "threshold"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    for col in ["pnl_usd"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    for col in ["r_multiple"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}R" if pd.notna(x) else "N/A")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
