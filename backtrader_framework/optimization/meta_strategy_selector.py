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

Faithfulness review 2026-09-03 (fleet plan WS5) — two leaks, one mis-count:
  * OVERLAPPING-LABEL LEAK. The label at day t is the best strategy over days
    t+1..t+H. Training on rows up to t-1 therefore trains on labels that already
    contain the returns of days t..t+H-2 — the very returns the day-t prediction
    is then scored on. Any persistent feature lets the model "remember" them.
    `train()` and `backtest()` now leave an embargo of H rows between the last
    training label and the first evaluated day (TimeSeriesSplit(gap=H) in CV).
  * H-FOLD OVER-COUNT. The backtest scored EVERY day's forward H-day sum as that
    day's return and summed them — each realized day counted H times, with the
    Sharpe annualised as if the series were daily and independent. The walk-
    forward now decides at day t, holds the pick for H days, and books the
    DAILY returns actually earned (stored per strategy at build time), so total
    R is the R a follower would have banked and the Sharpe is a daily Sharpe.
  * The regime × strategy heatmap is computed in-sample and is labelled so.
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
        # Daily REALIZED returns per strategy — what a follower actually books.
        daily_cols = {f'daily_{label}': daily_df[label] for label in self.strategy_labels}
        daily_ret_df = pd.DataFrame(daily_cols, index=daily_df.index)

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
            daily_ret_df, how='left'
        ).join(labels, how='inner')

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
        H = int(self.lookforward_days)

        # Chronological split WITH an embargo of H rows: the last H training
        # labels overlap the first test days' returns, so they are dropped.
        split_idx = int(len(X) * (1 - test_size))
        train_end = max(split_idx - H, 1)
        X_train, X_test = X[:train_end], X[split_idx:]
        y_train, y_test = y[:train_end], y[split_idx:]

        # Bayesian hyperparameter tuning (optional)
        bayesian_result = None
        self._tuned_rf_params = None

        if use_bayesian_tuning:
            try:
                from .bayesian_tuner import OptunaTuner, TunerConfig
                cfg = tuner_config or TunerConfig()
                cfg.embargo = max(cfg.embargo, H)      # never tune on leaked folds
                tuner = OptunaTuner(cfg)
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

        # Time-series CV on training set — with the same H-row embargo (gap)
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, train_end // 30)), gap=H)
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
            'embargo_rows': H,
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

        # Daily realized returns per strategy (stored by build_dataset). Fall
        # back to the forward sums with a stride of H if an old dataset lacks them.
        H = int(self.lookforward_days)
        daily_cols = [f'daily_{l}' for l in self.strategy_labels]
        have_daily = all(c in dataset.columns for c in daily_cols)
        if have_daily:
            daily_mat = dataset[daily_cols].fillna(0.0).values          # (n, S)
        else:
            daily_mat = None
            fwd_mat = dataset[[f'fwd_{l}' for l in self.strategy_labels]].fillna(0.0).values

        # Walk-forward: decide at day i on a model trained with an H-row embargo,
        # HOLD the pick for the next H days, book each of those days once.
        meta_hard_returns, meta_soft_returns = [], []
        equal_weight_returns, best_single_returns = [], []
        selection_timeline, bt_dates = [], []

        current_model = None
        last_train_idx = -1
        i = min_train_days
        while i < n:
            train_end = i - H                       # labels up to here are fully realized
            if train_end < 10:
                i += H
                continue
            if current_model is None or (i - last_train_idx) >= retrain_every:
                if self._tuned_rf_params:
                    current_model = RandomForestClassifier(**self._tuned_rf_params)
                else:
                    current_model = RandomForestClassifier(
                        n_estimators=100, max_depth=10,
                        class_weight='balanced', random_state=42, n_jobs=-1,
                    )
                current_model.fit(X[:train_end], y[:train_end])
                last_train_idx = i

            x_today = X[i:i + 1]
            pred = current_model.predict(x_today)[0]
            proba = current_model.predict_proba(x_today)[0]
            prob_dict = dict(zip(current_model.classes_.tolist(), proba.tolist()))

            # The block of days this decision governs: i+1 .. i+H (clipped)
            lo, hi = i + 1, min(i + H, n - 1)
            if have_daily:
                block = daily_mat[lo:hi + 1]                 # (h, S) daily returns
            else:  # legacy fallback: one forward sum per block, no daily detail
                block = fwd_mat[i:i + 1]
            per_strat = block.sum(axis=0)                    # realized over the block
            strat_returns = {l: float(per_strat[k]) for k, l in enumerate(self.strategy_labels)}

            hard_r = 0.0 if pred == 'none' else strat_returns.get(pred, 0.0)
            soft_r = sum(prob_dict.get(l, 0.0) * strat_returns[l] for l in self.strategy_labels)
            equal_r = float(np.mean(list(strat_returns.values())))
            best_r = float(max(strat_returns.values()))

            # Book the block DAY BY DAY so the equity curve and Sharpe are daily.
            if have_daily and len(block) > 0:
                w_soft = np.array([prob_dict.get(l, 0.0) for l in self.strategy_labels])
                k_hard = (self.strategy_labels.index(pred) if pred in self.strategy_labels else None)
                for d in range(len(block)):
                    meta_hard_returns.append(float(block[d, k_hard]) if k_hard is not None else 0.0)
                    meta_soft_returns.append(float(block[d] @ w_soft))
                    equal_weight_returns.append(float(block[d].mean()))
                    best_single_returns.append(float(block[d].max()))
                    bt_dates.append(dates[lo + d])
            else:
                meta_hard_returns.append(hard_r); meta_soft_returns.append(soft_r)
                equal_weight_returns.append(equal_r); best_single_returns.append(best_r)
                bt_dates.append(dates[i])

            actual_best = max(strat_returns, key=strat_returns.get)
            correct_target = actual_best if best_r > 0 else 'none'
            selection_timeline.append({
                'date': str(dates[i]),
                'predicted': pred,
                'probabilities': {k: round(v, 4) for k, v in prob_dict.items()},
                'actual_best': actual_best,
                'correct': pred == correct_target,
                'block_days': int(hi - lo + 1),
            })
            i += H                                          # next decision after the hold

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

            # Sharpe annualised from DAILY, non-overlapping bookings (crypto ~365d).
            # Before 2026-09-03 the series was H-day sums taken every day — H×
            # overlapping — and this number was not a Sharpe of anything real.
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
            'n_decisions': total_preds,
            'n_test_days': len(bt_dates),
            'hold_days': H,
            'embargo_rows': H,
            'booking': 'daily, hold-for-horizon' if have_daily else 'per-block forward sum (legacy dataset)',
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
        Cross-tabulate regime × PREDICTED strategy over the whole dataset.

        In-sample and descriptive by construction (the model has seen these
        rows); read it as "what the model would say in each regime", not as
        an out-of-sample performance claim.

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
            'in_sample': True,
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
