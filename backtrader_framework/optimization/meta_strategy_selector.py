"""
Meta-Strategy Selector — predict which strategy performs best given market state.

Trains a classifier on market features → best-performing strategy label,
then backtests dynamic allocation vs. static baselines. Uses existing
MarketFeatureEngine (20 features) + RegimeDetector as inputs, and
forward-looking OOS strategy returns as labels.

Usage:
    selector = MetaStrategySelector(lookforward_days=7)
    dataset = selector.build_dataset(wfo_filepaths, ohlcv_df)
    train_result = selector.train(dataset)
    backtest_result = selector.backtest(dataset)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter

from .persistence import load_wfo_result
from .market_features import MarketFeatureEngine, MARKET_FEATURE_NAMES
from .wfo_engine import IndicatorEngine, RegimeDetector


# Regime classes for one-hot encoding
REGIME_CLASSES = ['ranging', 'trending_up', 'trending_down', 'volatile']


class MetaStrategySelector:
    """Predict best strategy from market features; backtest dynamic allocation."""

    def __init__(self, lookforward_days: int = 7):
        self.lookforward_days = lookforward_days
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.strategy_labels: Optional[List[str]] = None
        self.training_stats: Optional[Dict] = None
        self._tuned_rf_params: Optional[Dict] = None

    # ══════════════════════════════════════════════════════════════════
    # 1. BUILD DATASET
    # ══════════════════════════════════════════════════════════════════

    def build_dataset(
        self,
        wfo_filepaths: List[str],
        ohlcv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build training dataset from WFO results + OHLCV data.

        Args:
            wfo_filepaths: Paths to WFO result JSON files (2+ strategies).
            ohlcv_df: OHLCV DataFrame with columns Open/High/Low/Close/Volume
                      and a DatetimeIndex (from DuckDB or other source).

        Returns:
            DataFrame with market features, strategy forward returns, and label.
        """
        if len(wfo_filepaths) < 2:
            raise ValueError("Need at least 2 WFO results for meta-selection.")

        # ── Load strategy daily returns ──────────────────────────────
        strategy_daily = {}  # label → Series(date → daily_r)
        strategy_meta = {}   # label → {strategy, symbol, timeframe}

        for fp in wfo_filepaths:
            result = load_wfo_result(fp)
            label = f"{result['strategy_name']}_{result['symbol']}_{result['timeframe']}"

            # Deduplicate labels
            if label in strategy_daily:
                i = 1
                while f"{label}_{i}" in strategy_daily:
                    i += 1
                label = f"{label}_{i}"

            strategy_meta[label] = {
                'strategy': result['strategy_name'],
                'symbol': result['symbol'],
                'timeframe': result['timeframe'],
            }

            # Build daily R series from OOS equity
            oos = result.get('oos_equity', [])
            if not oos:
                continue

            trades_df = pd.DataFrame(oos)
            trades_df['date'] = pd.to_datetime(trades_df['time']).dt.date
            daily_r = trades_df.groupby('date')['r'].sum()
            daily_r.index = pd.to_datetime(daily_r.index)
            strategy_daily[label] = daily_r

        if len(strategy_daily) < 2:
            raise ValueError("Need at least 2 strategies with OOS trades.")

        self.strategy_labels = sorted(strategy_daily.keys())

        # ── Build daily returns DataFrame ────────────────────────────
        # Outer join: fill missing days with 0 (no trades = 0 return)
        daily_df = pd.DataFrame(strategy_daily)
        daily_df = daily_df.sort_index()
        daily_df = daily_df.fillna(0.0)

        # ── Compute market features at daily resolution ──────────────
        indicator_df = IndicatorEngine.calculate(ohlcv_df)

        # Get last bar index for each calendar date
        indicator_df_with_date = indicator_df.copy()
        indicator_df_with_date['_date'] = indicator_df_with_date.index.date
        last_bar_per_day = indicator_df_with_date.groupby('_date').apply(
            lambda g: g.index[-1], include_groups=False,
        )

        # Compute features at each day's last bar
        feature_rows = []
        for date_val, bar_ts in last_bar_per_day.items():
            bar_idx = indicator_df.index.get_loc(bar_ts)
            if isinstance(bar_idx, slice):
                bar_idx = bar_idx.stop - 1

            features = MarketFeatureEngine.compute_at_bar(indicator_df, bar_idx)
            regime = RegimeDetector.classify(indicator_df, bar_idx)

            row = {'date': pd.Timestamp(date_val)}
            row.update(features)
            # One-hot regime
            for rc in REGIME_CLASSES:
                row[f'regime_{rc}'] = 1.0 if regime == rc else 0.0
            feature_rows.append(row)

        features_df = pd.DataFrame(feature_rows).set_index('date')

        # ── Compute forward returns ──────────────────────────────────
        fwd_cols = {}
        for label in self.strategy_labels:
            col = f'fwd_{label}'
            fwd_cols[col] = daily_df[label].rolling(
                window=self.lookforward_days, min_periods=1
            ).sum().shift(-self.lookforward_days)

        fwd_df = pd.DataFrame(fwd_cols, index=daily_df.index)

        # ── Assign label = best strategy ─────────────────────────────
        fwd_only = fwd_df[[f'fwd_{l}' for l in self.strategy_labels]]

        def pick_label(row):
            vals = row.values
            if np.all(np.isnan(vals)):
                return np.nan
            if np.all(vals <= 0):
                return 'none'
            best_idx = np.nanargmax(vals)
            return self.strategy_labels[best_idx]

        labels = fwd_only.apply(pick_label, axis=1)
        labels.name = 'label'

        # ── Merge everything ─────────────────────────────────────────
        dataset = features_df.join(fwd_df, how='inner').join(
            labels, how='inner'
        )

        # Drop rows with NaN label (end of data, no forward window)
        dataset = dataset.dropna(subset=['label'])
        # Drop rows where label is NaN string
        dataset = dataset[dataset['label'].notna()]

        # Drop rows with excessive NaN features (>50%)
        n_features = len(MARKET_FEATURE_NAMES) + len(REGIME_CLASSES)
        feature_cols = MARKET_FEATURE_NAMES + [f'regime_{rc}' for rc in REGIME_CLASSES]
        existing_feat_cols = [c for c in feature_cols if c in dataset.columns]
        nan_frac = dataset[existing_feat_cols].isna().sum(axis=1) / len(existing_feat_cols)
        dataset = dataset[nan_frac < 0.5]

        # Fill remaining NaN features with 0
        dataset[existing_feat_cols] = dataset[existing_feat_cols].fillna(0.0)

        self.feature_names = existing_feat_cols

        return dataset

    # ══════════════════════════════════════════════════════════════════
    # 2. TRAIN
    # ══════════════════════════════════════════════════════════════════

    def train(
        self,
        dataset: pd.DataFrame,
        test_size: float = 0.2,
        use_bayesian_tuning: bool = False,
        tuner_config=None,
    ) -> Dict:
        """
        Train a RandomForest classifier on the dataset.

        Uses chronological split (NOT random) and TimeSeriesSplit for CV.

        Args:
            dataset: Output of build_dataset().
            test_size: Fraction held out for evaluation.
            use_bayesian_tuning: If True, use Optuna to find best RF hyperparams.
            tuner_config: Optional TunerConfig for Bayesian tuning.

        Returns:
            Dict with accuracy, confusion_matrix, feature_importances,
            cv_scores, classification_report, class_distribution.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.metrics import (
            accuracy_score, confusion_matrix, classification_report,
        )

        if self.feature_names is None:
            raise ValueError("Call build_dataset() first.")

        X = dataset[self.feature_names].values
        y = dataset['label'].values

        # Chronological split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Bayesian hyperparameter tuning (optional)
        bayesian_result = None
        self._tuned_rf_params = None

        if use_bayesian_tuning:
            try:
                from .bayesian_tuner import OptunaTuner, TunerConfig
                tuner = OptunaTuner(tuner_config or TunerConfig())
                bayesian_result = tuner.tune_classifier(X_train, y_train)
                rf_params = dict(bayesian_result['best_params'])
                rf_params['random_state'] = 42
                rf_params['n_jobs'] = -1
                self._tuned_rf_params = rf_params
            except Exception:
                rf_params = None  # Fall back to defaults

        if self._tuned_rf_params:
            self.model = RandomForestClassifier(**self._tuned_rf_params)
        else:
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight='balanced', random_state=42, n_jobs=-1,
            )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        report = classification_report(
            y_test, y_pred, labels=self.model.classes_, output_dict=True,
            zero_division=0,
        )

        # Time-series CV on training set
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, split_idx // 30)))
        cv_rf_params = dict(self._tuned_rf_params) if self._tuned_rf_params else {
            'n_estimators': 100, 'max_depth': 10,
            'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1,
        }
        cv_scores = cross_val_score(
            RandomForestClassifier(**cv_rf_params),
            X_train, y_train, cv=tscv, scoring='accuracy',
        )

        # Feature importance
        importances = self.model.feature_importances_
        feat_imp = sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda x: x[1], reverse=True,
        )

        self.training_stats = {
            'accuracy': round(accuracy, 4),
            'cv_mean': round(float(np.mean(cv_scores)), 4),
            'cv_std': round(float(np.std(cv_scores)), 4),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_classes': len(self.model.classes_),
            'classes': self.model.classes_.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'feature_importances': feat_imp,
            'class_distribution_train': dict(Counter(y_train)),
            'class_distribution_test': dict(Counter(y_test)),
        }

        # Bayesian tuning results (if applicable)
        if bayesian_result:
            self.training_stats['bayesian_tuning'] = bayesian_result

        # SHAP analysis (optional, non-blocking)
        try:
            from .shap_analysis import SHAPAnalyzer
            shap_result = SHAPAnalyzer.analyze(
                self.model, X_test, self.feature_names,
                class_names=self.model.classes_.tolist(),
            )
            self.training_stats['shap_analysis'] = shap_result
        except Exception:
            pass  # shap not installed or computation failed

        return self.training_stats

    # ══════════════════════════════════════════════════════════════════
    # 3. PREDICT
    # ══════════════════════════════════════════════════════════════════

    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Predict best strategy from current market features.

        Args:
            features: Dict of feature_name → value (from MarketFeatureEngine).

        Returns:
            Dict with predicted_strategy, probabilities, confidence.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        prob_dict = dict(zip(self.model.classes_.tolist(), proba.tolist()))

        return {
            'predicted_strategy': pred,
            'probabilities': {k: round(v, 4) for k, v in prob_dict.items()},
            'confidence': round(float(max(proba)), 4),
        }

    # ══════════════════════════════════════════════════════════════════
    # 4. BACKTEST
    # ══════════════════════════════════════════════════════════════════

    def backtest(
        self,
        dataset: pd.DataFrame,
        min_train_days: int = 60,
        retrain_every: int = 20,
    ) -> Dict:
        """
        Walk-forward backtest of the meta-selector.

        Args:
            dataset: Output of build_dataset().
            min_train_days: Minimum days before first prediction.
            retrain_every: Retrain model every N days (not every day, for speed).

        Returns:
            Dict with equity curves, metrics, selection timeline.
        """
        from sklearn.ensemble import RandomForestClassifier

        if self.feature_names is None:
            raise ValueError("Call build_dataset() first.")

        X = dataset[self.feature_names].values
        y = dataset['label'].values
        dates = dataset.index.tolist()

        # Strategy actual daily returns (for computing portfolio returns)
        fwd_cols = [f'fwd_{l}' for l in self.strategy_labels]
        # We need actual daily returns, not forward sums
        # Reconstruct daily returns from the dataset
        # The forward returns are already in the dataset from build_dataset
        # But for backtest we need the actual return that happened on each day
        # Strategy daily returns are not stored directly, re-derive:
        strat_return_cols = []
        for label in self.strategy_labels:
            col = f'fwd_{label}'
            if col in dataset.columns:
                strat_return_cols.append(col)

        # For the backtest, we use the forward returns as the "truth"
        # Each day's forward return = sum of next N days of that strategy
        # The meta-selector picks which strategy to follow

        n = len(dataset)
        if n < min_train_days + 10:
            return {'valid': False, 'reason': f'Insufficient data: {n} days'}

        # Walk-forward
        meta_hard_returns = []
        meta_soft_returns = []
        equal_weight_returns = []
        best_single_returns = []
        selection_timeline = []
        bt_dates = []

        current_model = None
        last_train_idx = -1

        for i in range(min_train_days, n):
            # Retrain periodically
            if current_model is None or (i - last_train_idx) >= retrain_every:
                if self._tuned_rf_params:
                    current_model = RandomForestClassifier(**self._tuned_rf_params)
                else:
                    current_model = RandomForestClassifier(
                        n_estimators=100, max_depth=10,
                        class_weight='balanced', random_state=42, n_jobs=-1,
                    )
                current_model.fit(X[:i], y[:i])
                last_train_idx = i

            # Predict
            x_today = X[i:i+1]
            pred = current_model.predict(x_today)[0]
            proba = current_model.predict_proba(x_today)[0]
            prob_dict = dict(zip(current_model.classes_.tolist(), proba.tolist()))

            # Get actual forward returns for each strategy
            actual_fwd = {}
            for label in self.strategy_labels:
                col = f'fwd_{label}'
                actual_fwd[label] = dataset[col].iloc[i] if col in dataset.columns else 0.0

            # Meta-hard: 100% to predicted strategy (skip if "none")
            if pred == 'none':
                meta_hard_r = 0.0
            else:
                meta_hard_r = actual_fwd.get(pred, 0.0)

            # Meta-soft: probability-weighted
            meta_soft_r = 0.0
            for label in self.strategy_labels:
                p = prob_dict.get(label, 0.0)
                meta_soft_r += p * actual_fwd.get(label, 0.0)
            # Subtract "none" probability (allocated to cash = 0 return)
            none_p = prob_dict.get('none', 0.0)
            # Already handled: none contributes 0

            # Equal weight
            strat_returns = [actual_fwd.get(l, 0.0) for l in self.strategy_labels]
            equal_r = np.mean(strat_returns) if strat_returns else 0.0

            # Best single (oracle)
            best_r = max(strat_returns) if strat_returns else 0.0

            meta_hard_returns.append(meta_hard_r)
            meta_soft_returns.append(meta_soft_r)
            equal_weight_returns.append(equal_r)
            best_single_returns.append(best_r)
            bt_dates.append(dates[i])

            selection_timeline.append({
                'date': str(dates[i]),
                'predicted': pred,
                'probabilities': {k: round(v, 4) for k, v in prob_dict.items()},
                'actual_best': self.strategy_labels[np.argmax(strat_returns)] if strat_returns else 'none',
                'correct': pred == (self.strategy_labels[np.argmax(strat_returns)] if strat_returns and max(strat_returns) > 0 else 'none'),
            })

        # Compute metrics for each approach
        def _compute_metrics(returns, name):
            arr = np.array(returns)
            cum = np.cumsum(arr)
            total_r = float(np.sum(arr))
            n_periods = len(arr)

            # Max drawdown
            peak = np.maximum.accumulate(cum)
            dd = cum - peak
            max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

            # Sharpe (annualized from daily, assuming ~365 trading days for crypto)
            mean_r = np.mean(arr) if n_periods > 0 else 0.0
            std_r = np.std(arr, ddof=1) if n_periods > 1 else 1.0
            sharpe = float(mean_r / std_r * np.sqrt(365)) if std_r > 0 else 0.0

            # Win rate (periods with positive return)
            wins = np.sum(arr > 0)
            win_rate = float(wins / n_periods) if n_periods > 0 else 0.0

            return {
                'name': name,
                'total_r': round(total_r, 4),
                'sharpe_annual': round(sharpe, 4),
                'max_drawdown': round(max_dd, 4),
                'win_rate': round(win_rate, 4),
                'n_periods': n_periods,
                'mean_r': round(float(mean_r), 6),
                'cumulative': cum.tolist(),
            }

        # Accuracy of hard predictions
        correct = sum(1 for s in selection_timeline if s['correct'])
        total_preds = len(selection_timeline)

        results = {
            'valid': True,
            'n_test_days': total_preds,
            'prediction_accuracy': round(correct / total_preds, 4) if total_preds > 0 else 0.0,
            'meta_hard': _compute_metrics(meta_hard_returns, 'Meta-Selector (Hard)'),
            'meta_soft': _compute_metrics(meta_soft_returns, 'Meta-Selector (Soft)'),
            'equal_weight': _compute_metrics(equal_weight_returns, 'Equal Weight'),
            'best_single': _compute_metrics(best_single_returns, 'Best Single (Oracle)'),
            'dates': [str(d) for d in bt_dates],
            'selection_timeline': selection_timeline,
            'strategy_labels': self.strategy_labels,
        }

        return results

    # ══════════════════════════════════════════════════════════════════
    # 5. REGIME × STRATEGY HEATMAP
    # ══════════════════════════════════════════════════════════════════

    def get_regime_strategy_heatmap(self, dataset: pd.DataFrame) -> Dict:
        """
        Cross-tabulate regime × best-performing strategy.

        Returns dict with matrix data for heatmap visualization.
        """
        if self.model is None or self.feature_names is None:
            return {'valid': False}

        # Determine regime for each row
        regime_cols = [f'regime_{rc}' for rc in REGIME_CLASSES]
        existing_regime_cols = [c for c in regime_cols if c in dataset.columns]

        if not existing_regime_cols:
            return {'valid': False, 'reason': 'No regime columns'}

        # Get regime label for each row
        regimes = []
        for _, row in dataset.iterrows():
            regime_vals = {rc: row.get(f'regime_{rc}', 0) for rc in REGIME_CLASSES}
            best_regime = max(regime_vals, key=regime_vals.get)
            regimes.append(best_regime)

        dataset_copy = dataset.copy()
        dataset_copy['_regime'] = regimes

        # Cross-tab: regime × predicted label
        X = dataset_copy[self.feature_names].values
        predictions = self.model.predict(X)
        dataset_copy['_predicted'] = predictions

        # Count matrix
        all_strategies = self.strategy_labels + ['none']
        matrix = {}
        for regime in REGIME_CLASSES:
            regime_mask = dataset_copy['_regime'] == regime
            regime_preds = dataset_copy.loc[regime_mask, '_predicted']
            counts = Counter(regime_preds)
            total = sum(counts.values())
            matrix[regime] = {}
            for strat in all_strategies:
                matrix[regime][strat] = round(counts.get(strat, 0) / total, 4) if total > 0 else 0.0

        return {
            'valid': True,
            'regimes': REGIME_CLASSES,
            'strategies': all_strategies,
            'matrix': matrix,  # regime → strategy → fraction
        }

    # ══════════════════════════════════════════════════════════════════
    # 6. HELPER: LOAD OHLCV FROM DUCKDB
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def load_ohlcv_from_duckdb(
        symbol: str,
        timeframe: str = '4h',
        db_path: str = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from local DuckDB.

        Returns DataFrame with DatetimeIndex and OHLCV columns.
        """
        import duckdb

        if db_path is None:
            from pathlib import Path
            db_path = str(
                Path(__file__).resolve().parent.parent.parent
                / 'duckdb_data' / 'trading_data.duckdb'
            )

        con = duckdb.connect(db_path, read_only=True)
        try:
            df = con.execute(
                "SELECT timestamp, open AS Open, high AS High, low AS Low, "
                "close AS Close, volume AS Volume "
                "FROM ohlcv_data WHERE symbol = ? AND timeframe = ? "
                "ORDER BY timestamp",
                [symbol, timeframe],
            ).fetchdf()
        finally:
            con.close()

        if df.empty:
            raise ValueError(f"No OHLCV data for {symbol}/{timeframe}")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df
