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
    GaussianHMM       — Manual K-state multivariate Gaussian HMM (Baum-Welch EM).
    HMMRegimeAssessor  — WFO integration wrapper for IS fit / OOS forward-filter.

Faithfulness review 2026-09-03 (fleet plan WS3):
  * K > 2 used to be silently broken — the init only seeded states 0 and 1 and
    the relabel only compared them, so a 3-state model carried a zero-mean,
    1e-6-sigma ghost state. Init now splits on quantiles of the volatility
    feature for any K and states are relabelled by ascending volatility mean.
  * The "calm = state 0" convention assumed the volatility feature sat in
    column 1 ([LogReturn, RealizedVol20]). Fleet features put realized vol
    first, so the volatility column is now an explicit `vol_feature_index`.
  * The OOS forward filter restarted from the fitted initial distribution `pi`
    at the first OOS bar — a seam where the filter forgets everything the IS
    window knew. `forward_filter(init_state_probs=...)` lets a caller carry the
    last IS posterior across, and `HMMRegimeAssessor` does so automatically.
  * The forward/backward/xi recursions are vectorised over states (K is small,
    T is not); results are identical, rolling refits stop taking minutes.
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

    def __init__(self, n_states: int = 2, max_iter: int = 100, tol: float = 1e-4,
                 vol_feature_index: int = 1):
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.K = n_states
        self.max_iter = max_iter
        self.tol = tol
        # Column of X that carries the volatility measure. States are ordered by
        # their mean on this column: state 0 = calmest, state K-1 = most volatile.
        self.vol_feature_index = vol_feature_index

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
        X = np.asarray(X, dtype=float)
        T, D = X.shape
        self.D = D
        K = self.K
        v = min(self.vol_feature_index, D - 1)

        # --- Quantile init on the volatility feature (any K) ---
        edges = np.quantile(X[:, v], np.linspace(0, 1, K + 1))
        self.mu = np.zeros((K, D))
        self.sigma = np.zeros((K, D))
        glob_mu, glob_sd = np.mean(X, axis=0), np.maximum(np.std(X, axis=0), 1e-6)
        for k in range(K):
            lo, hi = edges[k], edges[k + 1]
            mask = (X[:, v] >= lo) & ((X[:, v] <= hi) if k == K - 1 else (X[:, v] < hi))
            if mask.sum() >= 2:
                self.mu[k] = np.mean(X[mask], axis=0)
                self.sigma[k] = np.maximum(np.std(X[mask], axis=0), 1e-6)
            else:  # degenerate slice — nudge off the global mean so states differ
                self.mu[k] = glob_mu + (k - (K - 1) / 2) * 0.1 * glob_sd
                self.sigma[k] = glob_sd

        # Sticky transitions
        self.A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.A, 0.9)
        self.pi = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            log_B = self._log_emission(X)
            log_A = np.log(self.A + 1e-300)

            # --- Forward pass (vectorised over states) ---
            log_alpha = np.zeros((T, K))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]
            for t in range(1, T):
                m = log_alpha[t - 1][:, None] + log_A            # (K_from, K_to)
                mx = m.max(axis=0)
                log_alpha[t] = mx + np.log(np.exp(m - mx).sum(axis=0)) + log_B[t]
            loglik = _logsumexp(log_alpha[-1])

            # --- Backward pass ---
            log_beta = np.zeros((T, K))
            for t in range(T - 2, -1, -1):
                m = log_A + (log_B[t + 1] + log_beta[t + 1])[None, :]   # (K_from, K_to)
                mx = m.max(axis=1)
                log_beta[t] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))

            # --- Posterior gamma ---
            log_gamma = log_alpha + log_beta
            log_gamma -= log_gamma.max(axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            gamma /= gamma.sum(axis=1, keepdims=True)

            # --- Xi (T-1, K, K) ---
            log_xi = (log_alpha[:-1, :, None] + log_A[None, :, :]
                      + (log_B[1:] + log_beta[1:])[:, None, :])
            log_xi -= log_xi.reshape(T - 1, -1).max(axis=1)[:, None, None]
            xi = np.exp(log_xi)
            xi /= xi.reshape(T - 1, -1).sum(axis=1)[:, None, None]

            # --- Convergence ---
            if abs(loglik - prev_ll) < self.tol and iteration > 0:
                break
            prev_ll = loglik

            # --- M-step ---
            self.pi = np.clip(gamma[0] / np.sum(gamma[0]), 1e-6, 1.0)
            self.pi /= self.pi.sum()

            denom = gamma[:-1].sum(axis=0)                       # (K,)
            A_new = xi.sum(axis=0)                               # (K, K)
            for i in range(K):
                if denom[i] > 1e-12:
                    self.A[i] = A_new[i] / denom[i]
                self.A[i] = np.clip(self.A[i], 1e-6, 1.0)
                self.A[i] /= self.A[i].sum()

            for k in range(K):
                g_sum = np.sum(gamma[:, k])
                if g_sum > 1e-12:
                    self.mu[k] = np.sum(gamma[:, k:k + 1] * X, axis=0) / g_sum
                    diff = X - self.mu[k]
                    self.sigma[k] = np.sqrt(np.sum(gamma[:, k:k + 1] * diff ** 2, axis=0) / g_sum)
                    self.sigma[k] = np.maximum(self.sigma[k], 1e-6)

        # --- Re-label: states in ascending order of the volatility feature's mean ---
        order = np.argsort(self.mu[:, v])
        self.mu = self.mu[order].copy()
        self.sigma = self.sigma[order].copy()
        self.A = self.A[np.ix_(order, order)].copy()
        self.pi = self.pi[order].copy()

        self.fitted = True
        return self

    def forward_filter(self, X: np.ndarray,
                       init_state_probs: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward-filter ONLY: P(state_t | x_1:t).

        Returns array of shape (T, K) with filtered state probabilities.
        No backward pass — safe for OOS use without future leakage.

        `init_state_probs` is the state distribution believed to hold just
        BEFORE the first row of X (e.g. the last filtered posterior of the IS
        window). It is propagated one step through A before the first emission,
        so an OOS segment continues the IS filter instead of restarting from pi.
        """
        if not self.fitted:
            raise ValueError("HMM not fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        log_B = self._log_emission(X)
        log_A = np.log(self.A + 1e-300)

        log_alpha = np.zeros((T, self.K))
        if init_state_probs is None:
            start = np.log(self.pi + 1e-300)
        else:
            prev = np.asarray(init_state_probs, dtype=float)
            prev = prev / prev.sum()
            start = np.log(prev @ self.A + 1e-300)               # one transition step
        log_alpha[0] = start + log_B[0]
        log_alpha[0] -= _logsumexp(log_alpha[0])

        for t in range(1, T):
            m = log_alpha[t - 1][:, None] + log_A
            mx = m.max(axis=0)
            log_alpha[t] = mx + np.log(np.exp(m - mx).sum(axis=0)) + log_B[t]
            log_alpha[t] -= _logsumexp(log_alpha[t])

        return np.exp(log_alpha)

    def transition_summary(self) -> Dict:
        """The switch-sequencing view of a fitted model: P(A→B), expected dwell
        time per state (1/(1−A_ii) bars), and the stationary distribution."""
        if not self.fitted:
            raise ValueError("HMM not fitted. Call fit() first.")
        A = self.A
        dwell = 1.0 / np.maximum(1.0 - np.diag(A), 1e-9)
        evals, evecs = np.linalg.eig(A.T)
        stat = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
        stat = stat / stat.sum()
        return {
            'transition_matrix': A.tolist(),
            'expected_dwell_bars': dwell.tolist(),
            'stationary': stat.tolist(),
            'state_order': 'ascending volatility (state 0 = calmest)',
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding — for IS labelling only."""
        if not self.fitted:
            raise ValueError("HMM not fitted. Call fit() first.")

        T = X.shape[0]
        log_B = self._log_emission(X)

        log_delta = np.zeros((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)
        log_delta[0] = np.log(self.pi + 1e-300) + log_B[0]

        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            cand = log_delta[t - 1][:, None] + log_A            # (K_from, K_to)
            psi[t] = np.argmax(cand, axis=0)
            log_delta[t] = cand[psi[t], np.arange(self.K)] + log_B[t]

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

    def __init__(self, n_states: int = 2, features: Optional[List[str]] = None,
                 vol_feature: Optional[str] = None):
        self.n_states = n_states
        self.features = list(features) if features else list(self.FEATURES)
        vol_feature = vol_feature or ('RealizedVol20' if 'RealizedVol20' in self.features
                                      else self.features[-1])
        if vol_feature not in self.features:
            raise ValueError(f"vol_feature {vol_feature!r} not in features {self.features}")
        self.hmm = GaussianHMM(n_states=n_states,
                               vol_feature_index=self.features.index(vol_feature))
        self.is_mean: Optional[np.ndarray] = None
        self.is_std: Optional[np.ndarray] = None
        self.is_last_posterior: Optional[np.ndarray] = None   # carried into OOS

    def _extract_features(self, df) -> Optional[np.ndarray]:
        """Extract [LogReturn, RealizedVol20] features from DataFrame."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.warning(f"HMM features missing: {missing}")
            return None

        X = df[self.features].values.copy()

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
            # Where the IS window left the filter — the OOS segment starts here.
            self.is_last_posterior = self.hmm.forward_filter(X_std)[-1]
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

        X_raw = oos_df[self.features].values.copy()
        nan_mask = np.isnan(X_raw).any(axis=1)
        # Fill NaN with IS mean (conservative: no info from OOS)
        for i in range(X_raw.shape[1]):
            X_raw[nan_mask, i] = self.is_mean[i]

        # Standardise with IS stats (leakage prevention layer 2)
        X_std = (X_raw - self.is_mean) / self.is_std

        try:
            return self.hmm.forward_filter(X_std, init_state_probs=self.is_last_posterior)
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
        # NOTE: these multipliers are the article's heuristic for the research
        # WFO. On the live fleet the HyroTrader standard forbids dimming size
        # below the 1% floor — a regime verdict may gate a seat ON/OFF, never
        # shrink it. Use forward_filter() + a pre-registered rule for that.
        p_calm = state_probs[0]
        p_volatile = state_probs[-1] if len(state_probs) > 1 else 1 - p_calm

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
        result['switch_sequencing'] = self.hmm.transition_summary()

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
