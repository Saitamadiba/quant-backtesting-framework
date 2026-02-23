"""Page 7: ML Training & Bot Management."""

import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ML Training", page_icon="🤖", layout="wide")
st.title("🤖 ML Training & Bot Management")

from config import (
    ML_TRAINING_DATA, ML_TRAINING_SCRIPT, ML_MODELS_DIR,
    ML_TRAINING_DB, ML_ROOT_TRAINING_SCRIPT,
    BOT_SERVICES, VPS_HOST, VPS_PORT, VPS_USER,
)
from data.vps_sync import get_bot_service_status, manage_bot_service
from components.wfo_tab import render_wfo_tab

# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_ml, tab_wfo = st.tabs(["ML Training & Models", "Walk-Forward Optimization"])

with tab_wfo:
    try:
        render_wfo_tab()
    except Exception as e:
        st.error(f"Error loading WFO tab: {e}")
        import traceback
        st.code(traceback.format_exc())

with tab_ml:
    # ══════════════════════════════════════════════════════════════════════
    # Section A: ML Data Readiness
    # ══════════════════════════════════════════════════════════════════════
    st.header("A. ML Data Readiness")
    st.caption("Shows how many trades have been collected per strategy/symbol/timeframe for ML model training. The ML trade filter needs a minimum sample size (30 trades) to learn reliable patterns. Green means ready for training; amber means collecting; red means insufficient data.")

    MIN_TRADES = 30

    # Check root-level ML training database (ml_training_data.db - 95MB, all strategies)
    db_found = False
    if ML_TRAINING_DB.exists():
        try:
            conn = sqlite3.connect(str(ML_TRAINING_DB))
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()

            if "ml_training_data" in tables:
                ml_db_data = pd.read_sql_query(
                    "SELECT strategy_name, symbol, timeframe, COUNT(*) as count "
                    "FROM ml_training_data GROUP BY strategy_name, symbol, timeframe", conn
                )
                total_db = pd.read_sql_query("SELECT COUNT(*) as c FROM ml_training_data", conn)["c"].iloc[0]

                st.caption(f"Source: `ml_training_data.db` ({ML_TRAINING_DB.stat().st_size / 1024 / 1024:.0f}MB)")
                for _, row in ml_db_data.iterrows():
                    count = row["count"]
                    label = f"{row.get('strategy_name', '?')} {row.get('symbol', '?')} {row.get('timeframe', '?')}"
                    pct = min(count / MIN_TRADES, 1.0)
                    if count >= MIN_TRADES:
                        st.progress(pct, text=f"🟢 {label}: {count} trades (ready)")
                    elif count >= 15:
                        st.progress(pct, text=f"🟡 {label}: {count}/{MIN_TRADES} trades")
                    else:
                        st.progress(pct, text=f"🔴 {label}: {count}/{MIN_TRADES} trades")

                if total_db >= MIN_TRADES:
                    st.success(f"Sufficient data collected ({total_db:,} records). Consider retraining ML models.")
                db_found = True
            conn.close()
        except Exception as e:
            st.error(f"Error reading ML DB: {e}")

    # Fallback to SBS CSV training data
    if not db_found and ML_TRAINING_DATA.exists():
        try:
            ml_data = pd.read_csv(str(ML_TRAINING_DATA))
            total_records = len(ml_data)
            st.caption(f"Source: `ml_training_data.csv`")

            if "symbol" in ml_data.columns and "timeframe" in ml_data.columns:
                groups = ml_data.groupby(["symbol", "timeframe"]).size().reset_index(name="count")
            else:
                groups = pd.DataFrame({"symbol": ["All"], "timeframe": ["All"], "count": [total_records]})

            for _, row in groups.iterrows():
                count = row["count"]
                label = f"{row['symbol']} {row['timeframe']}"
                pct = min(count / MIN_TRADES, 1.0)
                if count >= MIN_TRADES:
                    st.progress(pct, text=f"🟢 {label}: {count} trades (ready)")
                elif count >= 15:
                    st.progress(pct, text=f"🟡 {label}: {count}/{MIN_TRADES} trades")
                else:
                    st.progress(pct, text=f"🔴 {label}: {count}/{MIN_TRADES} trades")

            if total_records >= MIN_TRADES:
                st.success(f"Sufficient data collected ({total_records} records). Consider retraining ML models.")
            else:
                st.warning(f"Only {total_records} records. Need at least {MIN_TRADES} for training.")
        except Exception as e:
            st.error(f"Error reading ML training CSV: {e}")
    elif not db_found:
        st.warning("No ML training data found.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # Section A2: Database Contents Inspector
    # ══════════════════════════════════════════════════════════════════════
    st.header("A2. Database Contents")
    st.caption(
        "Detailed view of what data the bots are recording into the ML training database. "
        "Columns with high null rates indicate missing data collection — the ML model "
        "can only learn from features that are actually populated."
    )

    if ML_TRAINING_DB.exists():
        try:
            _ins_conn = sqlite3.connect(str(ML_TRAINING_DB))
            _ins_tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'",
                _ins_conn,
            )["name"].tolist()

            # ── Database overview ────────────────────────────────────
            db_size_mb = ML_TRAINING_DB.stat().st_size / 1024 / 1024
            _table_counts = {}
            for _t in _ins_tables:
                _table_counts[_t] = pd.read_sql_query(f"SELECT COUNT(*) as c FROM [{_t}]", _ins_conn)["c"].iloc[0]

            overview_cols = st.columns([1, 2])
            with overview_cols[0]:
                st.metric("Database Size", f"{db_size_mb:.1f} MB")
                st.metric("Tables", len(_ins_tables))
                st.metric("Total Records", f"{sum(_table_counts.values()):,}")
            with overview_cols[1]:
                _overview_df = pd.DataFrame([
                    {"Table": t, "Rows": f"{_table_counts[t]:,}", "Description": {
                        "ml_training_data": "Trade outcomes + 75 features per trade",
                        "ml_features": "Engineered features for model training",
                        "market_conditions": "Market context snapshots at trade time",
                        "trade_price_tracking": "Tick-by-tick price during open trades",
                    }.get(t, "")} for t in _ins_tables
                ])
                st.dataframe(_overview_df, use_container_width=True, hide_index=True)

            # ── Per-table schema + completeness ──────────────────────
            _inspect_table = st.selectbox(
                "Inspect table", _ins_tables, key="ml_inspect_table",
            )

            if _inspect_table and _table_counts.get(_inspect_table, 0) > 0:
                _schema = pd.read_sql_query(f"PRAGMA table_info([{_inspect_table}])", _ins_conn)
                _total_rows = _table_counts[_inspect_table]

                # Compute null counts per column
                _null_parts = []
                for _, col_row in _schema.iterrows():
                    col_name = col_row["name"]
                    _null_parts.append(
                        f"SUM(CASE WHEN [{col_name}] IS NULL THEN 1 ELSE 0 END) AS [{col_name}]"
                    )
                _null_query = f"SELECT {', '.join(_null_parts)} FROM [{_inspect_table}]"
                _null_row = pd.read_sql_query(_null_query, _ins_conn).iloc[0]

                schema_rows = []
                for _, col_row in _schema.iterrows():
                    col_name = col_row["name"]
                    nulls = int(_null_row[col_name])
                    filled = _total_rows - nulls
                    pct = filled / _total_rows * 100 if _total_rows > 0 else 0
                    schema_rows.append({
                        "Column": col_name,
                        "Type": col_row["type"],
                        "Filled": f"{filled:,} / {_total_rows:,}",
                        "Completeness": pct,
                    })

                _schema_df = pd.DataFrame(schema_rows)

                # Split into filled vs sparse
                well_filled = _schema_df[_schema_df["Completeness"] >= 50].copy()
                sparse = _schema_df[_schema_df["Completeness"] < 50].copy()

                tab_all, tab_filled, tab_sparse, tab_sample = st.tabs([
                    f"All Columns ({len(_schema_df)})",
                    f"Well-Populated ({len(well_filled)})",
                    f"Sparse / Empty ({len(sparse)})",
                    "Sample Records",
                ])

                with tab_all:
                    st.dataframe(
                        _schema_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Completeness": st.column_config.ProgressColumn(
                                "Completeness", min_value=0, max_value=100, format="%.0f%%",
                            ),
                        },
                    )

                with tab_filled:
                    if not well_filled.empty:
                        st.dataframe(
                            well_filled,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Completeness": st.column_config.ProgressColumn(
                                    "Completeness", min_value=0, max_value=100, format="%.0f%%",
                                ),
                            },
                        )
                    else:
                        st.info("No columns are 50%+ populated.")

                with tab_sparse:
                    if not sparse.empty:
                        st.warning(
                            f"**{len(sparse)} columns** are less than 50% populated. "
                            f"The ML model cannot learn from empty features — either the bots "
                            f"need to be updated to record these fields, or these columns should "
                            f"be excluded from training."
                        )
                        st.dataframe(
                            sparse,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Completeness": st.column_config.ProgressColumn(
                                    "Completeness", min_value=0, max_value=100, format="%.0f%%",
                                ),
                            },
                        )
                    else:
                        st.success("All columns are well-populated.")

                with tab_sample:
                    _sample_n = min(10, _total_rows)
                    _sample = pd.read_sql_query(
                        f"SELECT * FROM [{_inspect_table}] ORDER BY ROWID DESC LIMIT {_sample_n}",
                        _ins_conn,
                    )
                    st.caption(f"Most recent {_sample_n} records from `{_inspect_table}`")
                    st.dataframe(_sample, use_container_width=True, hide_index=True)

            _ins_conn.close()
        except Exception as e:
            st.error(f"Error inspecting database: {e}")
    else:
        st.info("ML training database not found.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # Section B: Current Model Performance
    # ══════════════════════════════════════════════════════════════════════
    st.header("B. Current Model Performance")
    st.caption("Displays the currently trained ML model's classification metrics. These tell you how well the model distinguishes good trades from bad ones. High accuracy alone isn't enough — check Precision (are predicted good trades actually good?) and Recall (are you catching all good trades?).")

    model_meta_path = ML_MODELS_DIR / "model_metadata.json"
    if model_meta_path.exists():
        try:
            with open(model_meta_path) as f:
                metadata = json.load(f)

            c1, c2, c3, c4, c5 = st.columns(5)
            acc = metadata.get('accuracy', 0)
            c1.metric("Accuracy", f"{acc:.1%}",
                      help=f"Proportion of all predictions (take/skip) that were correct. Currently {acc:.1%} — above 60% is acceptable, above 70% is strong.")
            prec = metadata.get('precision', 0)
            c2.metric("Precision", f"{prec:.1%}",
                      help=f"Of all trades the model said 'TAKE', what percentage actually won? Currently {prec:.1%}. High precision means fewer false positives (bad trades let through).")
            rec = metadata.get('recall', 0)
            c3.metric("Recall", f"{rec:.1%}",
                      help=f"Of all actual winning trades, what percentage did the model correctly predict? Currently {rec:.1%}. Low recall means the model is too conservative and skipping good trades.")
            f1 = metadata.get('f1_score', 0)
            c4.metric("F1 Score", f"{f1:.2f}",
                      help=f"Harmonic mean of Precision and Recall — a balanced measure. Currently {f1:.2f}. Above 0.65 is good; below 0.50 suggests the model needs retraining.")
            auc = metadata.get('auc_roc', 0)
            c5.metric("AUC-ROC", f"{auc:.2f}",
                      help=f"Area Under the ROC Curve — measures the model's ability to separate winners from losers across all thresholds. Currently {auc:.2f}. 0.5 = random guessing, 1.0 = perfect separation.")

            # Feature importance
            feat_imp = metadata.get("feature_importance", {})
            if feat_imp:
                st.caption("Feature importance from the Random Forest model. These are the market features that most influence trade outcome predictions. Use this to understand what drives your strategy's edge.")
                top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
                fig = px.bar(
                    x=[v for _, v in top_features], y=[k for k, _ in top_features],
                    orientation="h", title="Top 10 Feature Importance",
                    labels={"x": "Importance", "y": "Feature"},
                    color_discrete_sequence=["#2196F3"],
                )
                fig.update_layout(template="plotly_dark", height=350, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

            # Model age
            trained_at = metadata.get("trained_at", "Unknown")
            st.caption(f"Model last trained: {trained_at}")

            # ── Model Performance Commentary ──────────────────────────
            st.markdown("**Model Health Assessment:**")
            model_points = []

            # Accuracy assessment
            if acc >= 0.70:
                model_points.append(f"Accuracy of **{acc:.1%}** is strong — the model correctly classifies most trades.")
            elif acc >= 0.55:
                model_points.append(f"Accuracy of **{acc:.1%}** is acceptable but not exceptional. Monitor for drift.")
            else:
                model_points.append(f"Accuracy of **{acc:.1%}** is below the useful threshold. The model is barely better than coin-flipping — retrain with more data or better features.")

            # Precision vs Recall trade-off
            if prec > 0 and rec > 0:
                if prec > rec + 0.15:
                    model_points.append(f"Precision ({prec:.1%}) >> Recall ({rec:.1%}): The model is conservative — it rarely approves bad trades, but misses many good ones. You're leaving money on the table.")
                elif rec > prec + 0.15:
                    model_points.append(f"Recall ({rec:.1%}) >> Precision ({prec:.1%}): The model catches most good trades but also lets through too many losers. Tighten the TAKE threshold (currently 0.60) to improve precision.")
                else:
                    model_points.append(f"Precision ({prec:.1%}) and Recall ({rec:.1%}) are balanced — a healthy trade-off.")

            # AUC-ROC assessment
            if auc >= 0.75:
                model_points.append(f"AUC-ROC of **{auc:.2f}** indicates strong separation between winning and losing trade signals.")
            elif auc >= 0.60:
                model_points.append(f"AUC-ROC of **{auc:.2f}** — moderate discriminative power. The model adds value but isn't decisive.")
            elif auc > 0:
                model_points.append(f"AUC-ROC of **{auc:.2f}** is near random — the model is not effectively distinguishing good trades from bad. Consider adding new features or collecting more training data.")

            # Model staleness
            if trained_at != "Unknown":
                try:
                    trained_dt = pd.to_datetime(trained_at)
                    days_old = (pd.Timestamp.now() - trained_dt).days
                    if days_old > 30:
                        model_points.append(f"Model is **{days_old} days old**. Market regimes shift — retrain at least monthly to maintain relevance.")
                    elif days_old > 14:
                        model_points.append(f"Model is **{days_old} days old**. Consider retraining if recent live performance differs from predictions.")
                except Exception:
                    pass

            for p in model_points:
                st.markdown(f"- {p}")

        except Exception as e:
            st.error(f"Error reading model metadata: {e}")
    else:
        st.info("No trained model found. Use training controls below to train one.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # Section B2: ML Filter Impact — Before vs After
    # ══════════════════════════════════════════════════════════════════════
    st.header("B2. ML Filter Impact — Did It Actually Help?")
    st.caption(
        "The critical question: is the ML filter improving or hurting your bots? "
        "This section analyzes completed trades from the ML training database to compare "
        "what performance looks like across different ML confidence levels. "
        "If the filter isn't providing measurable lift, it may be adding complexity without benefit."
    )

    # ── Which bots have ML models? ────────────────────────────────────
    st.subheader("ML Deployment Status")
    st.caption("Shows which strategy/symbol combinations have a trained ML model and are actively using ML-based trade filtering on the VPS.")

    _ml_filter_paths = {
        "FVG BTC": Path(__file__).resolve().parent.parent.parent / "FVG_Strategy" / "BTC" / "ml_trade_filter.py",
        "FVG ETH": Path(__file__).resolve().parent.parent.parent / "FVG_Strategy" / "ETH" / "ml_trade_filter.py",
        "FVG NQ": Path(__file__).resolve().parent.parent.parent / "FVG_Strategy" / "NQ" / "ml_trade_filter.py",
        "LR BTC": Path(__file__).resolve().parent.parent.parent / "Liquidity_Raid" / "BTC_V2" / "ml_trade_filter.py",
        "LR ETH": Path(__file__).resolve().parent.parent.parent / "Liquidity_Raid" / "ETH_V2" / "ml_trade_filter.py",
        "MM BTC": Path(__file__).resolve().parent.parent.parent / "Momentum_Mastery" / "BTC" / "ml_trade_filter.py",
        "MM ETH": Path(__file__).resolve().parent.parent.parent / "Momentum_Mastery" / "ETH" / "ml_trade_filter.py",
        "SBS BTC": Path(__file__).resolve().parent.parent.parent / "SBS" / "bots" / "btc" / "ml_trade_filter.py",
        "SBS ETH": Path(__file__).resolve().parent.parent.parent / "SBS" / "bots" / "eth" / "ml_trade_filter.py",
    }

    _model_pkl_paths = {
        "FVG BTC": Path(__file__).resolve().parent.parent.parent / "FVG_Strategy" / "BTC" / "ml_models",
        "FVG ETH": Path(__file__).resolve().parent.parent.parent / "FVG_Strategy" / "ETH" / "ml_models",
        "LR BTC": Path(__file__).resolve().parent.parent.parent / "Liquidity_Raid" / "BTC_V2" / "ml_models",
        "LR ETH": Path(__file__).resolve().parent.parent.parent / "Liquidity_Raid" / "ETH_V2" / "ml_models",
        "SBS": ML_MODELS_DIR,
    }

    deploy_cols = st.columns(3)
    for i, (label, fpath) in enumerate(_ml_filter_paths.items()):
        col_idx = i % 3
        has_filter = fpath.exists()
        # Check for trained model file
        model_dir = _model_pkl_paths.get(label.rsplit(" ", 1)[0] + " " + label.rsplit(" ", 1)[-1],
                                          _model_pkl_paths.get(label.split()[0], None))
        has_model = False
        if model_dir and model_dir.exists():
            has_model = any(model_dir.glob("*.pkl"))
        elif ML_MODELS_DIR.exists():
            has_model = any(ML_MODELS_DIR.glob("*.pkl"))

        if has_filter and has_model:
            deploy_cols[col_idx].success(f"**{label}** — Filter + Model", icon="🟢")
        elif has_filter:
            deploy_cols[col_idx].warning(f"**{label}** — Filter only (no model)", icon="🟡")
        else:
            deploy_cols[col_idx].error(f"**{label}** — No ML filter", icon="🔴")

    st.markdown("---")

    # ── Before vs After Analysis from ml_training_data.db ─────────────
    st.subheader("Trade Outcome Analysis by ML Confidence")
    st.caption(
        "The ML filter assigns a confidence score (0-1) to each potential trade. "
        "Trades above **0.60** are classified TAKE (strong), **0.45-0.60** as GOOD, "
        "**0.30-0.45** as MARGINAL, and below **0.30** as SKIP. "
        "This analysis shows whether higher-confidence trades actually perform better — "
        "if they don't, the model isn't learning useful patterns."
    )

    _ml_trade_analysis_done = False

    if ML_TRAINING_DB.exists():
        try:
            conn = sqlite3.connect(str(ML_TRAINING_DB))
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()

            if "ml_training_data" in tables:
                # Load completed trades with outcomes
                ml_trades = pd.read_sql_query(
                    "SELECT strategy_name, symbol, timeframe, outcome, "
                    "pnl_dollars, pnl_percent, risk_reward_actual, "
                    "exit_reason, mfe_percent, mae_percent "
                    "FROM ml_training_data WHERE outcome IS NOT NULL",
                    conn,
                )

                # Load features and compute ML confidence via trained model
                ml_features_exist = "ml_features" in tables
                ml_confidence = pd.DataFrame()
                if ml_features_exist:
                    try:
                        import pickle
                        # Load all feature columns + outcome
                        ml_feat_df = pd.read_sql_query(
                            "SELECT trade_id, is_win, hour_utc, day_of_week, is_weekend, "
                            "is_killzone, direction_num, strategy_type, asset, is_crypto, "
                            "planned_rr, stop_distance_pct, regime_num, recent_wr, "
                            "consecutive_losses, pnl_dollars "
                            "FROM ml_features WHERE is_win IS NOT NULL",
                            conn,
                        )
                        # Try to load model and compute predictions
                        model_pkl = ML_MODELS_DIR / "win_predictor.pkl"
                        meta_json = ML_MODELS_DIR / "model_metadata.json"
                        if model_pkl.exists() and meta_json.exists() and not ml_feat_df.empty:
                            with open(model_pkl, "rb") as f:
                                model = pickle.load(f)
                            with open(meta_json, "r") as f:
                                meta = json.load(f)
                            feature_cols = meta.get("features", [])
                            # Build feature matrix in model's expected order
                            X = ml_feat_df[feature_cols].fillna(-1).values
                            probs = model.predict_proba(X)
                            win_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                            ml_confidence = ml_feat_df[["trade_id", "is_win", "pnl_dollars"]].copy()
                            ml_confidence["confidence"] = win_probs
                    except Exception as feat_err:
                        st.caption(f"Could not compute ML confidence: {feat_err}")

                conn.close()

                if not ml_trades.empty:
                    _ml_trade_analysis_done = True
                    total_ml = len(ml_trades)
                    wins = (ml_trades["outcome"] == "win").sum()
                    losses = (ml_trades["outcome"] == "loss").sum()
                    baseline_wr = wins / total_ml if total_ml > 0 else 0
                    baseline_pnl = ml_trades["pnl_dollars"].sum()
                    baseline_avg_pnl = ml_trades["pnl_dollars"].mean()

                    gp = ml_trades.loc[ml_trades["pnl_dollars"] > 0, "pnl_dollars"].sum()
                    gl = ml_trades.loc[ml_trades["pnl_dollars"] < 0, "pnl_dollars"].abs().sum()
                    baseline_pf = gp / gl if gl > 0 else float("inf")

                    avg_rr = ml_trades["risk_reward_actual"].dropna().mean()

                    # ── Baseline (All Trades) KPIs ────────────────────
                    st.markdown("#### Baseline: All Recorded Trades (No ML Filter)")
                    bl_cols = st.columns(5)
                    bl_cols[0].metric("Total Trades", f"{total_ml}",
                                      help="Total trades logged in the ML training database with known outcomes.")
                    bl_cols[1].metric("Win Rate", f"{baseline_wr:.1%}",
                                      help=f"{wins} wins out of {total_ml} trades.")
                    bl_cols[2].metric("Total P&L", f"${baseline_pnl:,.2f}",
                                      help="Sum of all realized P&L across all ML-tracked trades.")
                    bl_cols[3].metric("Profit Factor", f"{baseline_pf:.2f}" if baseline_pf != float("inf") else "---",
                                      help="Gross profit / Gross loss. Above 1.0 = profitable.")
                    bl_cols[4].metric("Avg R:R", f"{avg_rr:.2f}" if pd.notna(avg_rr) else "---",
                                      help="Average realized Risk:Reward ratio across all trades.")

                    # ── Confidence Bucket Analysis ──────────────────
                    if not ml_confidence.empty and "confidence" in ml_confidence.columns:
                        st.markdown("#### ML Confidence vs Actual Outcome")
                        st.caption(
                            "Shows whether the model's predicted confidence actually correlates with "
                            "real trade outcomes. If higher-confidence buckets have higher win rates, "
                            "the model is learning useful patterns."
                        )

                        # Bucket by confidence
                        bins = [0, 0.30, 0.45, 0.60, 1.0]
                        labels_conf = ["SKIP (<30%)", "MARGINAL (30-45%)", "GOOD (45-60%)", "STRONG (>60%)"]
                        ml_confidence["bucket"] = pd.cut(
                            ml_confidence["confidence"], bins=bins, labels=labels_conf, include_lowest=True
                        )

                        bucket_stats = ml_confidence.groupby("bucket", observed=True).agg(
                            trades=("is_win", "count"),
                            wins=("is_win", "sum"),
                            pnl=("pnl_dollars", "sum"),
                        ).reset_index()
                        bucket_stats["wr"] = bucket_stats["wins"] / bucket_stats["trades"]
                        bucket_stats["wr"] = bucket_stats["wr"].fillna(0)

                        if len(bucket_stats) >= 2:
                            fig_conf = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=("Win Rate by Confidence", "P&L by Confidence"),
                            )
                            colors = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"]
                            fig_conf.add_trace(go.Bar(
                                x=bucket_stats["bucket"].astype(str),
                                y=bucket_stats["wr"],
                                marker_color=colors[:len(bucket_stats)],
                                text=[f"{wr:.0%}" for wr in bucket_stats["wr"]],
                                textposition="outside",
                            ), row=1, col=1)
                            fig_conf.add_trace(go.Bar(
                                x=bucket_stats["bucket"].astype(str),
                                y=bucket_stats["pnl"],
                                marker_color=[
                                    "#4CAF50" if p > 0 else "#F44336" for p in bucket_stats["pnl"]
                                ],
                                text=[f"${p:,.0f}" for p in bucket_stats["pnl"]],
                                textposition="outside",
                            ), row=1, col=2)
                            fig_conf.add_hline(
                                y=baseline_wr, line_dash="dash",
                                line_color="rgba(255,255,255,0.3)",
                                row=1, col=1, annotation_text=f"Baseline {baseline_wr:.0%}",
                            )
                            fig_conf.update_layout(
                                template="plotly_dark", height=350, showlegend=False,
                            )
                            fig_conf.update_yaxes(tickformat=".0%", row=1, col=1)
                            st.plotly_chart(fig_conf, use_container_width=True, key="ml_confidence_buckets")

                            # Summary table
                            disp_bucket = bucket_stats.copy()
                            disp_bucket["wr"] = disp_bucket["wr"].apply(lambda x: f"{x:.1%}")
                            disp_bucket["pnl"] = disp_bucket["pnl"].apply(lambda x: f"${x:,.2f}")
                            disp_bucket.columns = ["Confidence", "Trades", "Wins", "Total P&L", "Win Rate"]
                            st.dataframe(disp_bucket, use_container_width=True, hide_index=True)

                            # Assess model quality
                            strong = bucket_stats[bucket_stats["bucket"] == "STRONG (>60%)"]
                            skip = bucket_stats[bucket_stats["bucket"] == "SKIP (<30%)"]
                            if not strong.empty and not skip.empty:
                                strong_wr = strong["wr"].iloc[0]
                                skip_wr = skip["wr"].iloc[0]
                                if strong_wr > skip_wr + 0.1:
                                    st.success(
                                        f"Model shows discrimination: STRONG trades win at "
                                        f"{strong_wr:.0%} vs SKIP at {skip_wr:.0%}."
                                    )
                                else:
                                    st.warning(
                                        f"Model shows weak discrimination: STRONG ({strong_wr:.0%}) "
                                        f"vs SKIP ({skip_wr:.0%}). Consider retraining with more data."
                                    )
                        else:
                            st.info("Not enough confidence buckets to display analysis.")

                    # ── Per-Strategy Breakdown ────────────────────────
                    st.markdown("#### Performance by Strategy")
                    strat_groups = ml_trades.groupby("strategy_name")
                    strat_rows = []
                    for strat, grp in strat_groups:
                        n = len(grp)
                        w = (grp["outcome"] == "win").sum()
                        wr = w / n if n > 0 else 0
                        pnl = grp["pnl_dollars"].sum()
                        g_p = grp.loc[grp["pnl_dollars"] > 0, "pnl_dollars"].sum()
                        g_l = grp.loc[grp["pnl_dollars"] < 0, "pnl_dollars"].abs().sum()
                        pf = g_p / g_l if g_l > 0 else float("inf")
                        avg_r = grp["risk_reward_actual"].dropna().mean()
                        avg_mfe = grp["mfe_percent"].dropna().mean()
                        avg_mae = grp["mae_percent"].dropna().mean()
                        strat_rows.append({
                            "Strategy": strat,
                            "Trades": n,
                            "Wins": w,
                            "Win Rate": wr,
                            "Total P&L": pnl,
                            "Profit Factor": pf,
                            "Avg R:R": avg_r,
                            "Avg MFE %": avg_mfe,
                            "Avg MAE %": avg_mae,
                        })

                    strat_df = pd.DataFrame(strat_rows)
                    if not strat_df.empty:
                        # Visual: side-by-side WR and PF comparison
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Win Rate by Strategy", "Profit Factor by Strategy"),
                        )
                        fig.add_trace(go.Bar(
                            x=strat_df["Strategy"], y=strat_df["Win Rate"],
                            marker_color=["#4CAF50" if wr > 0.5 else "#FF9800" if wr > 0.35 else "#F44336"
                                           for wr in strat_df["Win Rate"]],
                            text=[f"{wr:.0%}" for wr in strat_df["Win Rate"]],
                            textposition="outside",
                        ), row=1, col=1)
                        fig.add_trace(go.Bar(
                            x=strat_df["Strategy"],
                            y=[min(pf, 5) for pf in strat_df["Profit Factor"]],
                            marker_color=["#4CAF50" if pf > 1.5 else "#FF9800" if pf > 1 else "#F44336"
                                           for pf in strat_df["Profit Factor"]],
                            text=[f"{pf:.2f}" if pf < 100 else "---" for pf in strat_df["Profit Factor"]],
                            textposition="outside",
                        ), row=1, col=2)
                        fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                                      row=1, col=1, annotation_text="50% breakeven")
                        fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                                      row=1, col=2, annotation_text="Breakeven")
                        fig.update_layout(template="plotly_dark", height=350, showlegend=False)
                        fig.update_yaxes(tickformat=".0%", row=1, col=1)
                        st.plotly_chart(fig, use_container_width=True, key="ml_strat_comparison")

                        # Table
                        display_strat = strat_df.copy()
                        display_strat["Win Rate"] = display_strat["Win Rate"].apply(lambda x: f"{x:.1%}")
                        display_strat["Total P&L"] = display_strat["Total P&L"].apply(lambda x: f"${x:,.2f}")
                        display_strat["Profit Factor"] = display_strat["Profit Factor"].apply(
                            lambda x: f"{x:.2f}" if x < 100 else "---")
                        display_strat["Avg R:R"] = display_strat["Avg R:R"].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else "---")
                        display_strat["Avg MFE %"] = display_strat["Avg MFE %"].apply(
                            lambda x: f"{x:.2f}%" if pd.notna(x) else "---")
                        display_strat["Avg MAE %"] = display_strat["Avg MAE %"].apply(
                            lambda x: f"{x:.2f}%" if pd.notna(x) else "---")
                        st.dataframe(display_strat, use_container_width=True, hide_index=True)

                    # ── Exit Reason Analysis (filter effectiveness proxy) ─
                    st.markdown("#### Exit Reason Breakdown")
                    st.caption(
                        "How trades are ending. A high proportion of TP1/TP2/TP3 hits indicates "
                        "the entry filter is selecting good setups. Mostly stop-loss exits suggest "
                        "the filter isn't discriminating well enough."
                    )
                    if "exit_reason" in ml_trades.columns:
                        exit_counts = ml_trades["exit_reason"].value_counts()
                        if not exit_counts.empty:
                            fig_exit = go.Figure(go.Pie(
                                labels=exit_counts.index,
                                values=exit_counts.values,
                                hole=0.4,
                                marker=dict(colors=px.colors.qualitative.Set2),
                            ))
                            fig_exit.update_layout(
                                title="Exit Reason Distribution",
                                template="plotly_dark", height=350,
                            )
                            st.plotly_chart(fig_exit, use_container_width=True, key="ml_exit_pie")

                    # ── MFE/MAE Efficiency Analysis ───────────────────
                    st.markdown("#### Trade Efficiency: MFE vs MAE")
                    st.caption(
                        "**MFE** (Max Favorable Excursion) = the best unrealized P&L during the trade. "
                        "**MAE** (Max Adverse Excursion) = the worst intra-trade drawdown. "
                        "Ideal trades have high MFE and low MAE. If MAE consistently exceeds MFE, "
                        "the ML filter is allowing entries at poor locations."
                    )
                    mfe_vals = ml_trades["mfe_percent"].dropna()
                    mae_vals = ml_trades["mae_percent"].dropna()
                    if len(mfe_vals) >= 5 and len(mae_vals) >= 5:
                        fig_eff = go.Figure()
                        fig_eff.add_trace(go.Histogram(
                            x=mfe_vals, name="MFE %", opacity=0.7,
                            marker_color="#4CAF50", nbinsx=30,
                        ))
                        fig_eff.add_trace(go.Histogram(
                            x=mae_vals.apply(lambda x: -abs(x)), name="MAE % (negative)",
                            opacity=0.7, marker_color="#F44336", nbinsx=30,
                        ))
                        fig_eff.update_layout(
                            title="MFE vs MAE Distribution",
                            template="plotly_dark", height=300,
                            barmode="overlay",
                            xaxis_title="Excursion %",
                        )
                        st.plotly_chart(fig_eff, use_container_width=True, key="ml_mfe_mae")

                        avg_mfe = mfe_vals.mean()
                        avg_mae = mae_vals.mean()
                        efficiency = avg_mfe / avg_mae if avg_mae > 0 else float("inf")
                        st.caption(
                            f"Avg MFE: **{avg_mfe:.2f}%** | Avg MAE: **{avg_mae:.2f}%** | "
                            f"Efficiency Ratio: **{efficiency:.2f}** "
                            f"({'Good — winners run further than losers dip' if efficiency > 1.5 else 'Marginal — entries could be tighter' if efficiency > 1.0 else 'Poor — adverse excursions exceed favorable ones'})"
                        )

                    # ── Win/Loss by Time of Day (ML should filter bad hours)
                    st.markdown("#### Outcome by Hour of Day")
                    st.caption(
                        "If the ML model is learning correctly, it should be filtering out trades "
                        "during hours that historically lose money. Check if the model is actually "
                        "blocking trades during low-performing hours."
                    )
                    # Get hour from original data if available
                    if ML_TRAINING_DB.exists():
                        try:
                            conn2 = sqlite3.connect(str(ML_TRAINING_DB))
                            hour_data = pd.read_sql_query(
                                "SELECT hour_of_day, outcome, pnl_dollars "
                                "FROM ml_training_data WHERE outcome IS NOT NULL AND hour_of_day IS NOT NULL",
                                conn2,
                            )
                            conn2.close()
                            if not hour_data.empty and len(hour_data) >= 10:
                                hourly_stats = hour_data.groupby("hour_of_day").agg(
                                    trades=("outcome", "count"),
                                    wins=("outcome", lambda x: (x == "win").sum()),
                                    pnl=("pnl_dollars", "sum"),
                                ).reset_index()
                                hourly_stats["wr"] = hourly_stats["wins"] / hourly_stats["trades"]

                                fig_hour = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=("Trade Count by Hour", "Win Rate by Hour"),
                                )
                                fig_hour.add_trace(go.Bar(
                                    x=hourly_stats["hour_of_day"], y=hourly_stats["trades"],
                                    marker_color="#2196F3",
                                ), row=1, col=1)
                                fig_hour.add_trace(go.Bar(
                                    x=hourly_stats["hour_of_day"], y=hourly_stats["wr"],
                                    marker_color=[
                                        "#4CAF50" if wr > 0.5 else "#FF9800" if wr > 0.35 else "#F44336"
                                        for wr in hourly_stats["wr"]
                                    ],
                                ), row=1, col=2)
                                fig_hour.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                                                   row=1, col=2)
                                fig_hour.update_layout(template="plotly_dark", height=300, showlegend=False)
                                fig_hour.update_yaxes(tickformat=".0%", row=1, col=2)
                                fig_hour.update_xaxes(title_text="Hour (UTC)", row=1, col=1)
                                fig_hour.update_xaxes(title_text="Hour (UTC)", row=1, col=2)
                                st.plotly_chart(fig_hour, use_container_width=True, key="ml_hourly")

                                # Flag bad hours the model should be filtering
                                bad_hours = hourly_stats[
                                    (hourly_stats["wr"] < 0.3) & (hourly_stats["trades"] >= 5)
                                ]
                                if not bad_hours.empty:
                                    hrs = ", ".join(f"{int(h)}:00" for h in bad_hours["hour_of_day"])
                                    st.warning(
                                        f"Hours with <30% win rate (5+ trades): **{hrs}**. "
                                        f"The ML model should be filtering these — check if `is_killzone` "
                                        f"and `hour_utc` features are properly weighted."
                                    )
                        except Exception:
                            pass

                    # ── Overall Commentary ────────────────────────────
                    st.markdown("---")
                    st.markdown("#### ML Impact Assessment")
                    impact_points = []

                    if baseline_wr < 0.35:
                        impact_points.append(
                            f"Baseline win rate of **{baseline_wr:.1%}** across {total_ml} trades is low. "
                            f"The ML filter needs to significantly improve this or ensure the avg winner "
                            f"(${ml_trades.loc[ml_trades['pnl_dollars'] > 0, 'pnl_dollars'].mean():,.2f}) "
                            f"far exceeds the avg loser (${ml_trades.loc[ml_trades['pnl_dollars'] < 0, 'pnl_dollars'].mean():,.2f}) "
                            f"to remain profitable."
                        )
                    elif baseline_wr > 0.45:
                        impact_points.append(
                            f"Baseline win rate of **{baseline_wr:.1%}** is healthy even without ML filtering. "
                            f"The ML model should focus on precision — filtering the worst 20% of trades "
                            f"could meaningfully lift Profit Factor without sacrificing volume."
                        )

                    if baseline_pf < 1.0:
                        impact_points.append(
                            f"Profit Factor of **{baseline_pf:.2f}** is below breakeven. The ML filter "
                            f"is critical — without effective filtering, the strategy loses money. "
                            f"Retrain with focus on identifying the losing trade characteristics."
                        )
                    elif baseline_pf < 1.5:
                        impact_points.append(
                            f"Profit Factor of **{baseline_pf:.2f}** is marginally profitable. "
                            f"ML filtering could push this above 1.5 by eliminating low-confidence trades."
                        )
                    elif baseline_pf >= 2.0:
                        impact_points.append(
                            f"Profit Factor of **{baseline_pf:.2f}** is already strong. ML filtering should "
                            f"focus on position sizing adjustment (size up on high-confidence, down on marginal) "
                            f"rather than outright trade blocking."
                        )

                    # Check model quality from metadata
                    if model_meta_path.exists():
                        try:
                            with open(model_meta_path) as f:
                                meta = json.load(f)
                            m_auc = meta.get("auc_roc", 0)
                            m_prec = meta.get("precision", 0)
                            m_rec = meta.get("recall", 0)

                            if m_auc < 0.55:
                                impact_points.append(
                                    f"**Model AUC-ROC is {m_auc:.2f}** (near random). The model is not "
                                    f"learning meaningful patterns. Consider: (1) collecting more data, "
                                    f"(2) engineering better features (add volatility regime, correlation metrics), "
                                    f"(3) rebalancing the training set (currently {baseline_wr:.0%} wins vs "
                                    f"{1-baseline_wr:.0%} losses — try SMOTE oversampling)."
                                )
                            if m_prec == 0 and m_rec == 0:
                                impact_points.append(
                                    "**Precision and Recall are both 0%** — the model is predicting "
                                    "almost all trades as losses. This is a class imbalance problem. "
                                    "The training data has more losses than wins, causing the model to "
                                    "default to predicting 'loss' for everything. Fix by: (1) SMOTE oversampling, "
                                    "(2) class weights in RandomForest (`class_weight='balanced'`), or "
                                    "(3) lowering the TAKE threshold from 0.60 to 0.50."
                                )
                        except Exception:
                            pass

                    # Feature importance insight
                    if model_meta_path.exists():
                        try:
                            with open(model_meta_path) as f:
                                meta = json.load(f)
                            fi = meta.get("feature_importance", {})
                            if fi:
                                top_3 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
                                top_names = ", ".join(f"**{k}** ({v:.1%})" for k, v in top_3)
                                impact_points.append(
                                    f"Top 3 features driving predictions: {top_names}. "
                                    f"If these are mostly time-based features (hour, day), the model is "
                                    f"learning session patterns. If they're market-based (regime, ATR), "
                                    f"it's learning volatility patterns — both are valid edges."
                                )
                        except Exception:
                            pass

                    for p in impact_points:
                        st.markdown(f"- {p}")

            else:
                conn.close()
        except Exception as e:
            st.error(f"Error analyzing ML trades: {e}")

    if not _ml_trade_analysis_done:
        st.info(
            "No completed trades found in the ML training database. "
            "Trades are logged automatically as your bots run on the VPS. "
            "Once you have 30+ completed trades, this section will show detailed "
            "before/after analysis."
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # Section C: Training Controls
    # ══════════════════════════════════════════════════════════════════════
    st.header("C. Training Controls")
    st.caption("Trigger a full ML model retrain. This re-reads all collected trade data, engineers features, trains a Random Forest classifier, and saves the model. Retraining takes 1-5 minutes depending on data size. After training, restart your bots to apply the updated trade filter.")

    # Determine best available training script
    training_script = None
    if ML_ROOT_TRAINING_SCRIPT.exists():
        training_script = ML_ROOT_TRAINING_SCRIPT
        st.caption(f"Script: `ml_model_training.py` (root-level, all strategies)")
    elif ML_TRAINING_SCRIPT.exists():
        training_script = ML_TRAINING_SCRIPT
        st.caption(f"Script: `{ML_TRAINING_SCRIPT.name}` (SBS research)")

    if training_script:
        if st.button("Train ML Models", type="primary"):
            with st.status("Training ML models...", expanded=True) as status:
                try:
                    result = subprocess.run(
                        ["python", str(training_script)],
                        capture_output=True, text=True, timeout=600,
                        cwd=str(training_script.parent),
                    )
                    st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
                    if result.returncode == 0:
                        status.update(label="Training complete!", state="complete")
                    else:
                        st.error(result.stderr[-2000:] if result.stderr else "Training failed.")
                        status.update(label="Training failed", state="error")
                except subprocess.TimeoutExpired:
                    st.error("Training timed out (10 min limit).")
                    status.update(label="Timed out", state="error")
                except Exception as e:
                    st.error(f"Error: {e}")
                    status.update(label="Error", state="error")
    else:
        st.warning("No training script found.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # Section D: VPS Bot Management
    # ══════════════════════════════════════════════════════════════════════
    st.header("D. VPS Bot Management")
    st.caption("Live status of all trading bots running on your VPS. Green means the bot is active and trading; red means it's stopped. Use Start/Stop/Restart to manage bots remotely. After ML retraining, restart bots so they pick up the new model.")

    st.caption(f"VPS: {VPS_USER}@{VPS_HOST}:{VPS_PORT}")

    if st.button("Refresh Bot Status"):
        st.session_state["_refresh_bots"] = True

    cols_per_row = 3
    service_items = list(BOT_SERVICES.items())

    for i in range(0, len(service_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (svc, info) in enumerate(service_items[i:i + cols_per_row]):
            with cols[j]:
                status = get_bot_service_status(svc)
                if status == "active":
                    st.success(f"**{info['strategy']} {info['symbol']}**  \n`{svc}` — Running", icon="🟢")
                elif status in ("inactive", "dead"):
                    st.error(f"**{info['strategy']} {info['symbol']}**  \n`{svc}` — Stopped", icon="🔴")
                else:
                    st.warning(f"**{info['strategy']} {info['symbol']}**  \n`{svc}` — {status}", icon="🟡")

                bc1, bc2, bc3 = st.columns(3)
                if bc1.button("Start", key=f"start_{svc}"):
                    r = manage_bot_service(svc, "start")
                    st.toast(f"Start {svc}: {'OK' if r['success'] else r['stderr']}")
                    st.rerun()
                if bc2.button("Stop", key=f"stop_{svc}"):
                    r = manage_bot_service(svc, "stop")
                    st.toast(f"Stop {svc}: {'OK' if r['success'] else r['stderr']}")
                    st.rerun()
                if bc3.button("Restart", key=f"restart_{svc}"):
                    r = manage_bot_service(svc, "restart")
                    st.toast(f"Restart {svc}: {'OK' if r['success'] else r['stderr']}")
                    st.rerun()
