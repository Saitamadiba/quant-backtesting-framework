"""
WFO-Aligned Signal Scorer — Shared module for all trading bots.

Replaces per-bot hardcoded scoring with WFO-validated confidence formulas.
Includes regime gating (DVOL + ATR) and ML-driven weight adaptation.
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """Output of WFOSignalScorer.score()."""
    confidence: float
    components: Dict[str, float]
    regime_gate: str  # PASS, SKIP, LONG_ONLY, SHORT_ONLY
    gate_reason: str
    min_confidence: float
    passed: bool  # confidence >= min_confidence AND regime allows direction

    def to_dict(self) -> dict:
        return {
            'confidence': round(self.confidence, 4),
            'components': {k: round(v, 4) for k, v in self.components.items()},
            'regime_gate': self.regime_gate,
            'gate_reason': self.gate_reason,
            'passed': self.passed,
        }


@dataclass
class StrategyScoreConfig:
    """Configuration for a strategy's WFO-validated scoring."""
    name: str
    components: Dict[str, float]  # component_name -> WFO baseline weight
    min_confidence: float
    regime_gates: Dict[str, Any] = field(default_factory=dict)

    @property
    def component_names(self) -> List[str]:
        return list(self.components.keys())

    @property
    def baseline_weights(self) -> Dict[str, float]:
        return dict(self.components)


# ---------------------------------------------------------------------------
# Pre-built strategy configs (WFO-validated)
# ---------------------------------------------------------------------------

LR_CONFIG = StrategyScoreConfig(
    name='LiquidityRaid',
    components={
        'sweep_depth_atr': 0.50,
        'counter_struct': 0.20,
        'counter_htf': 0.15,
        'struct_conf': 0.15,
    },
    min_confidence=0.25,
    regime_gates={
        'atr_pctile_max': 0.80,
        'dvol_med_range': (45, 65),
    },
)

FVG_CONFIG = StrategyScoreConfig(
    name='FVG',
    components={
        'gap_size': 0.20,
        'volume': 0.15,
        'ema_align': 0.15,
        'rsi_align': 0.10,
        'struct_bias': 0.10,
        'displacement': 0.15,
        'sweep': 0.15,
    },
    min_confidence=0.45,
    regime_gates={
        'dvol_low_max': 45,
        'dvol_med_range': (45, 65),
        'dvol_high_min': 65,
    },
)

MM_CONFIG = StrategyScoreConfig(
    name='MomentumMastery',
    components={
        'sweep_depth': 0.30,
        'volume_conf': 0.15,
        'ema_align': 0.15,
        'confirm_quality': 0.15,
        'struct_bias': 0.15,
        'atr_bonus': 0.10,
    },
    min_confidence=0.35,
    regime_gates={
        'atr_regime_rr': {'QUIET': -0.5, 'NORMAL': 0.0, 'VOLATILE': 1.0},
    },
)

SBS_CONFIG = StrategyScoreConfig(
    name='SBS',
    components={
        'sweep_depth': 0.30,
        'ema_align': 0.20,
        'rsi_align': 0.15,
        'volume_conf': 0.15,
        'struct_bias': 0.20,
    },
    min_confidence=0.48,
    regime_gates={},
)


# ---------------------------------------------------------------------------
# ML Weight Adapter
# ---------------------------------------------------------------------------

class MLWeightAdapter:
    """
    Ridge logistic regression on component scores -> adapted weights.

    Learns which WFO confidence components are most predictive of trade
    outcome in the current regime, then adjusts weights accordingly.
    Weights are clamped to [0.5x, 2.0x] of WFO baseline to prevent
    catastrophic divergence.
    """

    MIN_TRADES = 100        # minimum trades before first adaptation
    RETRAIN_EVERY = 50      # retrain every N new outcomes
    MIN_CV_ACCURACY = 0.52  # cross-validated accuracy must beat this
    CLAMP_MIN = 0.5         # min multiplier of baseline weight
    CLAMP_MAX = 2.0         # max multiplier of baseline weight
    MAX_BUFFER = 500        # rolling buffer size

    def __init__(self, config: StrategyScoreConfig, weights_dir: str = 'ml_models/wfo_weights'):
        self.config = config
        self.weights_dir = weights_dir
        self._component_names = config.component_names
        self._baseline = config.baseline_weights
        self._adapted_weights: Optional[Dict[str, float]] = None
        self._buffer: deque = deque(maxlen=self.MAX_BUFFER)
        self._trades_since_retrain = 0
        self._total_trades = 0
        self._cv_accuracy = 0.0
        self._last_retrain_ts = 0

        self._load_weights()

    @property
    def active_weights(self) -> Dict[str, float]:
        """Return adapted weights if available, else WFO baseline."""
        return self._adapted_weights if self._adapted_weights else self._baseline

    @property
    def is_adapted(self) -> bool:
        return self._adapted_weights is not None

    def record_outcome(self, component_scores: Dict[str, float], won: bool):
        """Record a trade outcome for future adaptation."""
        scores = [component_scores.get(c, 0.0) for c in self._component_names]
        self._buffer.append((scores, 1 if won else 0))
        self._total_trades += 1
        self._trades_since_retrain += 1

        if (self._total_trades >= self.MIN_TRADES
                and self._trades_since_retrain >= self.RETRAIN_EVERY):
            self._retrain()

    def _retrain(self):
        """Fit Ridge Logistic Regression and extract adapted weights."""
        self._trades_since_retrain = 0

        if len(self._buffer) < self.MIN_TRADES:
            return

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler

            X = np.array([s for s, _ in self._buffer])
            y = np.array([o for _, o in self._buffer])

            if len(np.unique(y)) < 2:
                logger.debug(f"[MLWeightAdapter:{self.config.name}] Only one class in buffer, skipping retrain")
                return

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)

            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            cv_accuracy = cv_scores.mean()
            self._cv_accuracy = cv_accuracy

            if cv_accuracy < self.MIN_CV_ACCURACY:
                logger.info(
                    f"[MLWeightAdapter:{self.config.name}] CV accuracy {cv_accuracy:.3f} "
                    f"< {self.MIN_CV_ACCURACY}, keeping baseline weights"
                )
                self._adapted_weights = None
                self._save_weights()
                return

            model.fit(X_scaled, y)
            coefficients = model.coef_[0]

            raw_weights = np.abs(coefficients)
            if raw_weights.sum() == 0:
                self._adapted_weights = None
                self._save_weights()
                return

            normalized = raw_weights / raw_weights.sum()

            adapted = {}
            for i, name in enumerate(self._component_names):
                baseline_w = self._baseline[name]
                adapted_w = normalized[i]
                lo = baseline_w * self.CLAMP_MIN
                hi = baseline_w * self.CLAMP_MAX
                adapted[name] = float(np.clip(adapted_w, lo, hi))

            total = sum(adapted.values())
            if total > 0:
                adapted = {k: v / total for k, v in adapted.items()}

            self._adapted_weights = adapted
            self._last_retrain_ts = int(time.time())

            logger.info(
                f"[MLWeightAdapter:{self.config.name}] Retrained — "
                f"CV accuracy: {cv_accuracy:.3f}, "
                f"adapted weights: {self._format_weights(adapted)}"
            )
            self._save_weights()

        except Exception as e:
            logger.warning(f"[MLWeightAdapter:{self.config.name}] Retrain failed: {e}")

    def _format_weights(self, weights: Dict[str, float]) -> str:
        return ', '.join(f"{k}={v:.3f}" for k, v in weights.items())

    def _save_weights(self):
        """Persist adapted weights to JSON."""
        try:
            os.makedirs(self.weights_dir, exist_ok=True)
            path = os.path.join(self.weights_dir, f"{self.config.name.lower()}_weights.json")
            data = {
                'strategy': self.config.name,
                'baseline_weights': self._baseline,
                'adapted_weights': self._adapted_weights,
                'total_trades': self._total_trades,
                'buffer_size': len(self._buffer),
                'cv_accuracy': round(self._cv_accuracy, 4),
                'last_retrain_ts': self._last_retrain_ts,
                'is_adapted': self.is_adapted,
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[MLWeightAdapter:{self.config.name}] Failed to save weights: {e}")

    def _load_weights(self):
        """Load previously adapted weights from JSON."""
        path = os.path.join(self.weights_dir, f"{self.config.name.lower()}_weights.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._adapted_weights = data.get('adapted_weights')
            self._total_trades = data.get('total_trades', 0)
            self._cv_accuracy = data.get('cv_accuracy', 0.0)
            self._last_retrain_ts = data.get('last_retrain_ts', 0)
            if self._adapted_weights:
                logger.info(
                    f"[MLWeightAdapter:{self.config.name}] Loaded adapted weights "
                    f"(trades={self._total_trades}, cv={self._cv_accuracy:.3f})"
                )
        except Exception as e:
            logger.warning(f"[MLWeightAdapter:{self.config.name}] Failed to load weights: {e}")

    def get_status(self) -> dict:
        """Return current adapter status for logging."""
        return {
            'strategy': self.config.name,
            'is_adapted': self.is_adapted,
            'total_trades': self._total_trades,
            'trades_until_retrain': max(0, self.RETRAIN_EVERY - self._trades_since_retrain),
            'cv_accuracy': round(self._cv_accuracy, 4),
            'active_weights': self.active_weights,
        }


# ---------------------------------------------------------------------------
# WFO Signal Scorer
# ---------------------------------------------------------------------------

class WFOSignalScorer:
    """
    Shared scorer using WFO-validated confidence formulas.
    Each strategy has its own config defining components, weights, and regime gates.
    """

    def __init__(self, config: StrategyScoreConfig, adapter: Optional[MLWeightAdapter] = None):
        self.config = config
        self.adapter = adapter

    def score(self, features: Dict[str, float], direction: str,
              dvol: Optional[float] = None, atr_pctile: Optional[float] = None) -> ScoreResult:
        """
        Score a signal using WFO-validated formula.

        Args:
            features: Dict of component_name -> raw score (0-1 scale)
            direction: 'LONG' or 'SHORT'
            dvol: Current DVOL value (optional, for regime gating)
            atr_pctile: Current ATR percentile (optional, for regime gating)

        Returns:
            ScoreResult with confidence, components, and regime gate status
        """
        # Step 1: Check regime gates
        gate, reason = self._check_regime_gates(direction, dvol, atr_pctile)

        if gate == 'SKIP':
            return ScoreResult(
                confidence=0.0,
                components={},
                regime_gate='SKIP',
                gate_reason=reason,
                min_confidence=self.config.min_confidence,
                passed=False,
            )

        if gate == 'LONG_ONLY' and direction == 'SHORT':
            return ScoreResult(
                confidence=0.0,
                components={},
                regime_gate='LONG_ONLY',
                gate_reason=reason,
                min_confidence=self.config.min_confidence,
                passed=False,
            )

        if gate == 'SHORT_ONLY' and direction == 'LONG':
            return ScoreResult(
                confidence=0.0,
                components={},
                regime_gate='SHORT_ONLY',
                gate_reason=reason,
                min_confidence=self.config.min_confidence,
                passed=False,
            )

        # Step 2: Get active weights (ML-adapted or WFO baseline)
        weights = self.adapter.active_weights if self.adapter else self.config.baseline_weights

        # Step 3: Compute weighted component scores
        component_scores = {}
        confidence = 0.0
        for comp_name, weight in weights.items():
            raw_score = features.get(comp_name, 0.0)
            raw_score = max(0.0, min(1.0, raw_score))  # clamp to [0, 1]
            weighted = raw_score * weight
            component_scores[comp_name] = weighted
            confidence += weighted

        confidence = max(0.0, min(1.0, confidence))

        passed = confidence >= self.config.min_confidence
        if gate in ('LONG_ONLY', 'SHORT_ONLY'):
            # Direction already validated above, gate is informational
            pass

        return ScoreResult(
            confidence=confidence,
            components=component_scores,
            regime_gate=gate,
            gate_reason=reason,
            min_confidence=self.config.min_confidence,
            passed=passed,
        )

    def _check_regime_gates(self, direction: str,
                            dvol: Optional[float],
                            atr_pctile: Optional[float]) -> Tuple[str, str]:
        """
        Check regime gates for this strategy's config.

        Returns:
            (gate_status, reason) where gate_status is one of:
            PASS, SKIP, LONG_ONLY, SHORT_ONLY
        """
        gates = self.config.regime_gates
        if not gates:
            return 'PASS', ''

        # ATR percentile hard gate (LR: skip if > 0.80)
        if 'atr_pctile_max' in gates and atr_pctile is not None:
            if atr_pctile > gates['atr_pctile_max']:
                return 'SKIP', f"ATR_pctile {atr_pctile:.2f} > {gates['atr_pctile_max']}"

        # FVG DVOL direction gates
        if dvol is not None:
            if 'dvol_low_max' in gates and 'dvol_high_min' in gates:
                # FVG-style: LOW=LONG_ONLY, MED=SKIP, HIGH=SHORT_ONLY
                dvol_low = gates['dvol_low_max']
                dvol_high = gates['dvol_high_min']
                dvol_med = gates.get('dvol_med_range', (dvol_low, dvol_high))

                if dvol < dvol_low:
                    return 'LONG_ONLY', f"DVOL {dvol:.1f} < {dvol_low} (low IV → LONG only)"
                elif dvol_med[0] <= dvol < dvol_med[1]:
                    return 'SKIP', f"DVOL {dvol:.1f} in [{dvol_med[0]}, {dvol_med[1]}) (medium IV → skip)"
                elif dvol >= dvol_high:
                    return 'SHORT_ONLY', f"DVOL {dvol:.1f} >= {dvol_high} (high IV → SHORT only)"

            elif 'dvol_med_range' in gates:
                # LR-style: medium DVOL → stricter but not skip (handled by caller via gate_reason)
                dvol_med = gates['dvol_med_range']
                if dvol_med[0] <= dvol < dvol_med[1]:
                    return 'PASS', f"DVOL_MED:{dvol:.1f} in [{dvol_med[0]}, {dvol_med[1]})"

        return 'PASS', ''

    def get_rr_adjustment(self, dvol: Optional[float] = None,
                          atr_pctile: Optional[float] = None,
                          atr_regime: Optional[str] = None) -> float:
        """
        Get R:R multiplier adjustment based on regime.

        Returns:
            Additive R:R adjustment (e.g., -0.5 for QUIET, +1.0 for VOLATILE)
        """
        gates = self.config.regime_gates

        # MM: ATR regime adjusts R:R
        if 'atr_regime_rr' in gates and atr_regime:
            return gates['atr_regime_rr'].get(atr_regime, 0.0)

        # LR: medium DVOL reduces R:R
        if 'dvol_med_range' in gates and dvol is not None:
            dvol_med = gates['dvol_med_range']
            if dvol_med[0] <= dvol < dvol_med[1]:
                return -0.25  # 0.75x R:R = base + (-0.25 * base) handled by caller

        return 0.0

    def is_dvol_med(self, dvol: Optional[float]) -> bool:
        """Check if DVOL is in the medium (cautious) range for this config."""
        if dvol is None:
            return False
        dvol_med = self.config.regime_gates.get('dvol_med_range')
        if dvol_med:
            return dvol_med[0] <= dvol < dvol_med[1]
        return False
