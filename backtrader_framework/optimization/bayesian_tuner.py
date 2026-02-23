"""
Bayesian Hyperparameter Tuning via Optuna.

Provides two tuning modes:
1. tune_classifier() — optimize sklearn classifier hyperparameters (RF, GB)
2. tune_strategy_params() — optimize WFO strategy parameters (replaces grid search)

All Optuna imports are lazy. Module loads without optuna installed.

Usage:
    from bayesian_tuner import OptunaTuner, TunerConfig
    tuner = OptunaTuner(TunerConfig(n_trials=50))
    result = tuner.tune_classifier(X_train, y_train)
    print(result['best_params'])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import logging
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TunerConfig:
    """Configuration for Bayesian tuning."""
    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    sampler: str = 'tpe'          # 'tpe', 'cmaes', 'random'
    pruner: str = 'median'        # 'median', 'halving', 'none'
    direction: str = 'maximize'
    n_cv_splits: int = 5
    scoring_metric: str = 'accuracy'  # 'accuracy', 'f1_weighted'
    random_seed: int = 42
    n_startup_trials: int = 10
    show_progress_bar: bool = False


# Default search spaces for classifier tuning
RF_SEARCH_SPACE = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'step': 50},
    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
    'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
    'class_weight': {'type': 'categorical', 'choices': ['balanced', 'balanced_subsample']},
}

GB_SEARCH_SPACE = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'step': 50},
    'max_depth': {'type': 'int', 'low': 3, 'high': 15},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
}


class OptunaTuner:
    """
    Bayesian hyperparameter tuner wrapping Optuna.

    Two use cases:
    1. ML classifier hyperparameter tuning (meta-strategy RF)
    2. Strategy parameter optimization (WFO alternative to grid/random)
    """

    def __init__(self, config: Optional[TunerConfig] = None):
        self.config = config or TunerConfig()
        self.study = None
        self.best_params: Optional[Dict] = None
        self.best_value: Optional[float] = None

    # ══════════════════════════════════════════════════════════════
    # MODE 1: Classifier Hyperparameter Tuning
    # ══════════════════════════════════════════════════════════════

    def tune_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        classifier_type: str = 'random_forest',
        search_space: Optional[Dict] = None,
    ) -> Dict:
        """
        Tune sklearn classifier hyperparameters using Bayesian optimization.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training labels (n_samples,).
            classifier_type: 'random_forest' or 'gradient_boosting'.
            search_space: Optional override of default search space.

        Returns:
            Dict with best_params, best_score, and visualization data.
        """
        import optuna
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if search_space is None:
            search_space = RF_SEARCH_SPACE if classifier_type == 'random_forest' else GB_SEARCH_SPACE

        n_splits = min(self.config.n_cv_splits, max(2, len(X_train) // 30))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        def objective(trial):
            params = self._suggest_from_search_space(trial, search_space)
            params['random_state'] = 42
            params['n_jobs'] = -1

            if classifier_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(**params)
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                params.pop('n_jobs', None)
                clf = GradientBoostingClassifier(**params)

            scores = cross_val_score(
                clf, X_train, y_train, cv=tscv,
                scoring=self.config.scoring_metric,
            )
            return float(np.mean(scores))

        sampler = self._create_sampler(
            self.config.sampler, self.config.random_seed, self.config.n_startup_trials,
        )
        pruner = self._create_pruner(self.config.pruner)

        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
        )
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            show_progress_bar=self.config.show_progress_bar,
        )

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # Remove non-model params before returning
        result_params = dict(self.best_params)
        result_params.pop('random_state', None)
        result_params.pop('n_jobs', None)

        return {
            'best_params': result_params,
            'best_score': self.best_value,
            'classifier_type': classifier_type,
            **self.get_visualization_data(),
        }

    # ══════════════════════════════════════════════════════════════
    # MODE 2: Strategy Parameter Tuning (WFO replacement)
    # ══════════════════════════════════════════════════════════════

    def tune_strategy_params(
        self,
        param_specs: List,
        objective_fn: Callable[[Dict], float],
    ) -> Dict:
        """
        Bayesian optimization over strategy parameter space.

        Args:
            param_specs: List of ParamSpec from the strategy adapter.
            objective_fn: fn(params_dict) → score (higher is better).

        Returns:
            Dict with best_params, best_score, and visualization data.
        """
        import optuna
        from .param_grid import suggest_from_param_spec

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {}
            for spec in param_specs:
                params[spec.name] = suggest_from_param_spec(trial, spec)
            return objective_fn(params)

        sampler = self._create_sampler(
            self.config.sampler, self.config.random_seed, self.config.n_startup_trials,
        )
        # No pruning for strategy mode (single evaluation, no intermediate values)
        pruner = self._create_pruner('none')

        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
        )
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            show_progress_bar=self.config.show_progress_bar,
        )

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # Cast param types back to match ParamSpec
        result_params = dict(self.best_params)
        spec_map = {s.name: s for s in param_specs}
        for k, v in result_params.items():
            if k in spec_map and spec_map[k].param_type == 'int':
                result_params[k] = int(round(v))

        return {
            'best_params': result_params,
            'best_score': self.best_value,
            **self.get_visualization_data(),
        }

    # ══════════════════════════════════════════════════════════════
    # RESULT EXTRACTION
    # ══════════════════════════════════════════════════════════════

    def get_visualization_data(self) -> Dict:
        """Extract visualization data from the completed study."""
        if self.study is None:
            return {'valid': False, 'reason': 'No study completed'}

        import optuna

        history = []
        best_so_far = -float('inf') if self.config.direction == 'maximize' else float('inf')
        convergence = []

        for trial in self.study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            history.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'duration_seconds': (
                    (trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_complete and trial.datetime_start else None
                ),
            })
            if self.config.direction == 'maximize':
                best_so_far = max(best_so_far, trial.value)
            else:
                best_so_far = min(best_so_far, trial.value)
            convergence.append({
                'trial': trial.number,
                'best_value': best_so_far,
            })

        # Parameter importances
        try:
            importances = optuna.importance.get_param_importances(self.study)
            param_importances = {k: round(float(v), 6) for k, v in importances.items()}
        except Exception:
            param_importances = {}

        n_complete = len([
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        n_pruned = len([
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ])

        return {
            'valid': True,
            'optimization_history': history,
            'convergence': convergence,
            'param_importances': param_importances,
            'n_trials_completed': n_complete,
            'n_trials_pruned': n_pruned,
            'best_trial_number': self.study.best_trial.number if self.study.best_trial else None,
        }

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _suggest_from_search_space(trial, search_space: Dict) -> Dict:
        """Suggest parameter values from a search space definition."""
        params = {}
        for name, spec in search_space.items():
            if spec['type'] == 'int':
                kwargs = {'name': name, 'low': spec['low'], 'high': spec['high']}
                if 'step' in spec:
                    kwargs['step'] = spec['step']
                params[name] = trial.suggest_int(**kwargs)
            elif spec['type'] == 'float':
                kwargs = {'name': name, 'low': spec['low'], 'high': spec['high']}
                if spec.get('log'):
                    kwargs['log'] = True
                if 'step' in spec and not spec.get('log'):
                    kwargs['step'] = spec['step']
                params[name] = trial.suggest_float(**kwargs)
            elif spec['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, spec['choices'])
        return params

    @staticmethod
    def _create_sampler(sampler_name: str, seed: int, n_startup: int):
        """Create Optuna sampler by name."""
        import optuna
        if sampler_name == 'tpe':
            return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
        elif sampler_name == 'cmaes':
            return optuna.samplers.CmaEsSampler(seed=seed)
        elif sampler_name == 'random':
            return optuna.samplers.RandomSampler(seed=seed)
        raise ValueError(f"Unknown sampler: {sampler_name}")

    @staticmethod
    def _create_pruner(pruner_name: str):
        """Create Optuna pruner by name."""
        import optuna
        if pruner_name == 'median':
            return optuna.pruners.MedianPruner(n_warmup_steps=5)
        elif pruner_name == 'halving':
            return optuna.pruners.SuccessiveHalvingPruner()
        elif pruner_name == 'none':
            return optuna.pruners.NopPruner()
        raise ValueError(f"Unknown pruner: {pruner_name}")

    @staticmethod
    def is_available() -> bool:
        """Check if optuna is installed."""
        try:
            import optuna  # noqa: F401
            return True
        except ImportError:
            return False
