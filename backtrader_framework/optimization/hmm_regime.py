"""
HMM-based regime detection for Walk-Forward Optimization.

Implements a 2-state multivariate Gaussian HMM with 3 layers of leakage
prevention per the WFO methodology article:

1. HMM is fit ONLY on in-sample (IS) data — never sees OOS bars.
2. Feature standardisation uses IS-computed mean/std — no OOS stats leakage.
3. OOS inference uses forward-filtering ONLY — no backward pass/smoothing.

The forward filter computes P(state_t | x_1:t), i.e. the regime probability
at bar t uses only data up to and including bar t.  This is the causal,
non-anticipatory analog of the Viterbi or forward-backward algorithms.

Classes:
    GaussianHMM       — Manual 2-state multivariate Gaussian HMM (Baum-Welch EM).
    HMMRegimeAssessor  — WFO integration wrapper for IS fit / OOS forward-filter.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1-D array."""
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return -np.inf
    return a_max + np.log(np.sum(np.exp(a - a_max)))


class GaussianHMM:
    """2-state multivariate Gaussian HMM with diagonal covariance.

    Fit via Baum-Welch EM on (T, D) observations.  After fitting,
    states are relabelled so that state 0 = "calm" (lowest variance in
    the first feature dimension, which is typically RealizedVol).
    """

    def __init__(self, n_states: int = 2, max_iter: int = 100, tol: float = 1e-4):
        self.K = n_states
        self.max_iter = max_iter
        self.tol = tol

        # Parameters (set by fit)
        self.mu: Optional[np.ndarray] = None      # (K, D)
        self.sigma: Optional[np.ndarray] = None    # (K, D)  diagonal std devs
        self.A: Optional[np.ndarray] = None        # (K, K)  transition matrix
        self.pi: Optional[np.ndarray] = None       # (K,)    initial distribution
        self.D: int = 0
        self.fitted: bool = False

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Compute log N(x_t | mu_k, diag(sigma_k^2)) for all t, k.

        Returns array of shape (T, K).
        """
        T = X.shape[0]
        log_B = np.zeros((T, self.K))
        for k in range(self.K):
            diff = X - self.mu[k]  # (T, D)
            var = np.maximum(self.sigma[k] ** 2, 1e-12)
            log_B[:, k] = -0.5 * np.sum(
                np.log(2 * np.pi * var) + diff ** 2 / var, axis=1
            )
        return log_B

    def fit(self, X: np.ndarray) -> 'GaussianHMM':
        """Fit HMM via Baum-Welch EM on observations X of shape (T, D)."""
        T, D = X.shape
        self.D = D
        K = self.K

        # --- K-means init ---
        # Split by median of first feature (typically volatility)
        med = np.median(X[:, 0])
        mask_lo = X[:, 0] <= med
        mask_hi = ~mask_lo

        self.mu = np.zeros((K, D))
        self.sigma = np.zeros((K, D))

        if np.sum(mask_lo) > 0:
            self.mu[0] = np.mean(X[mask_lo], axis=0)
            self.sigma[0] = np.maximum(np.std(X[mask_lo], axis=0), 1e-6)
        else:
            self.mu[0] = np.mean(X, axis=0) - 0.1
            self.sigma[0] = np.maximum(np.std(X, axis=0), 1e-6)

        if np.sum(mask_hi) > 0:
            self.mu[1] = np.mean(X[mask_hi], axis=0)
            self.sigma[1] = np.maximum(np.std(X[mask_hi], axis=0), 1e-6)
        else:
            self.mu[1] = np.mean(X, axis=0) + 0.1
            self.sigma[1] = np.maximum(np.std(X, axis=0), 1e-6)

        # Sticky transitions
        self.A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.A, 0.9)
        self.pi = np.full(K, 1.0 / K)

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            log_B = self._log_emission(X)

            # --- Forward pass ---
            log_alpha = np.zeros((T, K))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]

            for t in range(1, T):
                for j in range(K):
                    log_alpha[t, j] = (
                        _logsumexp(log_alpha[t - 1] + np.log(self.A[:, j] + 1e-300))
                        + log_B[t, j]
                    )

            loglik = _logsumexp(log_alpha[-1])

            # --- Backward pass ---
            log_beta = np.zeros((T, K))
            for t in range(T - 2, -1, -1):
                for i in range(K):
                    log_beta[t, i] = _logsumexp(
                        np.log(self.A[i, :] + 1e-300) + log_B[t + 1] + log_beta[t + 1]
                    )

            # --- Posterior gamma ---
            log_gamma = log_alpha + log_beta
            for t in range(T):
                log_gamma[t] -= _logsumexp(log_gamma[t])
            gamma = np.exp(log_gamma)

            # --- Xi ---
            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = (
                            log_alpha[t, i]
                            + np.log(self.A[i, j] + 1e-300)
                            + log_B[t + 1, j]
                            + log_beta[t + 1, j]
                        )
                norm = _logsumexp(xi[t].ravel())
                xi[t] -= norm
            xi = np.exp(xi)

            # --- Convergence ---
            if abs(loglik - prev_ll) < self.tol and iteration > 0:
                break
            prev_ll = loglik

            # --- M-step ---
            self.pi = gamma[0] / np.sum(gamma[0])
            self.pi = np.clip(self.pi, 1e-6, 1.0)
            self.pi /= self.pi.sum()

            for i in range(K):
                denom = np.sum(gamma[:-1, i])
                if denom > 1e-12:
                    for j in range(K):
                        self.A[i, j] = np.sum(xi[:, i, j]) / denom
                self.A[i] = np.clip(self.A[i], 1e-6, 1.0)
                self.A[i] /= self.A[i].sum()

            for k in range(K):
                g_sum = np.sum(gamma[:, k])
                if g_sum > 1e-12:
                    self.mu[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / g_sum
                    diff = X - self.mu[k]
                    self.sigma[k] = np.sqrt(
                        np.sum(gamma[:, k : k + 1] * diff ** 2, axis=0) / g_sum
                    )
                    self.sigma[k] = np.maximum(self.sigma[k], 1e-6)

        # --- Re-label: state 0 = calm (lowest RealizedVol mean, feature dim 1) ---
        vol_dim = min(1, D - 1)  # second feature if available, else first
        if self.mu[1, vol_dim] < self.mu[0, vol_dim]:
            self.mu = self.mu[::-1].copy()
            self.sigma = self.sigma[::-1].copy()
            self.A = self.A[::-1, ::-1].copy()
            self.pi = self.pi[::-1].copy()

        self.fitted = True
        return self

    def forward_filter(self, X: np.ndarray) -> np.ndarray:
        """Forward-filter ONLY: P(state_t | x_1:t).

        Returns array of shape (T, K) with filtered state probabilities.
        No backward pass — safe for OOS use without future leakage.
        """
        if not self.fitted:
            raise ValueError("HMM not fitted. Call fit() first.")

        T = X.shape[0]
        log_B = self._log_emission(X)

        log_alpha = np.zeros((T, self.K))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]
        # Normalise
        log_alpha[0] -= _logsumexp(log_alpha[0])

        for t in range(1, T):
            for j in range(self.K):
                log_alpha[t, j] = (
                    _logsumexp(log_alpha[t - 1] + np.log(self.A[:, j] + 1e-300))
                    + log_B[t, j]
                )
            log_alpha[t] -= _logsumexp(log_alpha[t])

        return np.exp(log_alpha)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding — for IS labelling only."""
        if not self.fitted:
            raise ValueError("HMM not fitted. Call fit() first.")

        T = X.shape[0]
        log_B = self._log_emission(X)

        log_delta = np.zeros((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)
        log_delta[0] = np.log(self.pi + 1e-300) + log_B[0]

        for t in range(1, T):
            for j in range(self.K):
                candidates = log_delta[t - 1] + np.log(self.A[:, j] + 1e-300)
                psi[t, j] = int(np.argmax(candidates))
                log_delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        # Back-track
        labels = np.zeros(T, dtype=int)
        labels[-1] = int(np.argmax(log_delta[-1]))
        for t in range(T - 2, -1, -1):
            labels[t] = psi[t + 1, labels[t + 1]]

        return labels


class HMMRegimeAssessor:
    """WFO integration wrapper for HMM regime assessment.

    Provides IS fit / OOS forward-filter with leakage prevention:
    - Features: [LogReturn, RealizedVol20] from IndicatorEngine
    - Standardisation with IS-computed mean/std
    - Forward-filter-only OOS inference
    - Graduated position sizing by regime probability
    """

    # Sizing tiers (article recommendation)
    FULL_SIZE_THRESHOLD = 0.70     # P(calm) >= 0.70 → 1.0x
    REDUCED_SIZE_THRESHOLD = 0.55  # P(calm) >= 0.55 → 0.7x
    VOLATILE_THRESHOLD = 0.60      # P(volatile) >= 0.60 → 0.3x

    FEATURES = ['LogReturn', 'RealizedVol20']

    def __init__(self, n_states: int = 2):
        self.n_states = n_states
        self.hmm = GaussianHMM(n_states=n_states)
        self.is_mean: Optional[np.ndarray] = None
        self.is_std: Optional[np.ndarray] = None

    def _extract_features(self, df) -> Optional[np.ndarray]:
        """Extract [LogReturn, RealizedVol20] features from DataFrame."""
        missing = [f for f in self.FEATURES if f not in df.columns]
        if missing:
            logger.warning(f"HMM features missing: {missing}")
            return None

        X = df[self.FEATURES].values.copy()

        # Drop NaN rows
        mask = ~np.isnan(X).any(axis=1)
        if mask.sum() < 50:
            logger.warning(f"HMM: only {mask.sum()} valid rows after NaN drop")
            return None

        return X[mask]

    def fit_on_is(self, is_df) -> bool:
        """Fit HMM on in-sample data. Returns True if successful."""
        X = self._extract_features(is_df)
        if X is None:
            return False

        # Compute and store IS standardisation parameters
        self.is_mean = np.mean(X, axis=0)
        self.is_std = np.maximum(np.std(X, axis=0), 1e-8)

        # Standardise
        X_std = (X - self.is_mean) / self.is_std

        try:
            self.hmm.fit(X_std)
            return True
        except Exception as e:
            logger.warning(f"HMM fit failed: {e}")
            return False

    def filter_oos(self, oos_df) -> Optional[np.ndarray]:
        """Forward-filter OOS data using IS-fitted standardisation.

        Returns (T, K) array of filtered state probabilities, or None on failure.
        """
        if not self.hmm.fitted or self.is_mean is None:
            return None

        X_raw = oos_df[self.FEATURES].values.copy()
        nan_mask = np.isnan(X_raw).any(axis=1)
        # Fill NaN with IS mean (conservative: no info from OOS)
        for i in range(X_raw.shape[1]):
            X_raw[nan_mask, i] = self.is_mean[i]

        # Standardise with IS stats (leakage prevention layer 2)
        X_std = (X_raw - self.is_mean) / self.is_std

        try:
            return self.hmm.forward_filter(X_std)
        except Exception as e:
            logger.warning(f"HMM forward filter failed: {e}")
            return None

    @staticmethod
    def get_size_multiplier(state_probs: np.ndarray) -> float:
        """Convert state probabilities at a single bar to a position size multiplier.

        State 0 = calm, State 1 = volatile (by convention from GaussianHMM relabelling).

        Sizing tiers (from article):
            P(calm) >= 0.70 → 1.0  (full size)
            P(calm) >= 0.55 → 0.7  (reduced)
            P(volatile) >= 0.60 → 0.3  (defensive)
            else → 0.5  (uncertain)
        """
        p_calm = state_probs[0]
        p_volatile = state_probs[1] if len(state_probs) > 1 else 1 - p_calm

        if p_calm >= HMMRegimeAssessor.FULL_SIZE_THRESHOLD:
            return 1.0
        elif p_calm >= HMMRegimeAssessor.REDUCED_SIZE_THRESHOLD:
            return 0.7
        elif p_volatile >= HMMRegimeAssessor.VOLATILE_THRESHOLD:
            return 0.3
        else:
            return 0.5

    def get_assessment(self, is_df, oos_df) -> Dict:
        """Full HMM assessment: fit on IS, forward-filter OOS.

        Returns dict with HMM parameters, OOS regime distribution, sizing stats.
        """
        result = {
            'fitted': False,
            'n_states': self.n_states,
        }

        if not self.fit_on_is(is_df):
            result['error'] = 'fit_failed'
            return result

        result['fitted'] = True
        result['hmm_means'] = self.hmm.mu.tolist()
        result['hmm_stds'] = self.hmm.sigma.tolist()
        result['transition_matrix'] = self.hmm.A.tolist()

        # Forward-filter OOS
        probs = self.filter_oos(oos_df)
        if probs is None:
            result['oos_filtered'] = False
            return result

        result['oos_filtered'] = True
        result['oos_n_bars'] = probs.shape[0]

        # Regime distribution in OOS
        calm_probs = probs[:, 0]
        result['oos_mean_p_calm'] = round(float(np.mean(calm_probs)), 4)
        result['oos_pct_calm'] = round(float(np.mean(calm_probs >= 0.5)) * 100, 1)

        # Position sizing distribution
        size_mults = np.array([self.get_size_multiplier(p) for p in probs])
        result['mean_size_mult'] = round(float(np.mean(size_mults)), 4)
        result['sizing_distribution'] = {
            'full_1.0': round(float(np.mean(size_mults == 1.0)) * 100, 1),
            'reduced_0.7': round(float(np.mean(size_mults == 0.7)) * 100, 1),
            'uncertain_0.5': round(float(np.mean(size_mults == 0.5)) * 100, 1),
            'defensive_0.3': round(float(np.mean(size_mults == 0.3)) * 100, 1),
        }

        return result
