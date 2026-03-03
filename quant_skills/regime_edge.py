"""
Regime-edge decomposition skill for quantitative strategy auditing.

Decomposes strategy performance across market regimes to determine whether
edge is universal or confined to specific conditions.  Uses both a manual
2-state Gaussian HMM (with hmmlearn as optional accelerator) and price-regime
labels supplied by the backtester.

Key question answered: "Is the edge real in all market conditions, or does it
only work in one regime that may not persist?"

Analyses included:
    - Gaussian HMM regime detection (EM + Viterbi, manual fallback)
    - Per-regime edge statistics (mean R, win rate, expectancy)
    - Regime transition matrix and stationary distribution
    - Price-regime decomposition (ranging, trending, volatile)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import TradeData, SkillResult, _grade_from_score


class RegimeEdgeDecomposer:
    """Stateless analyser that decomposes strategy edge across regimes."""

    MIN_TRADES = 50

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(data: Any, n_regimes: int = 2) -> Dict[str, Any]:
        """Run regime-edge decomposition.

        Parameters
        ----------
        data : array-like, list[dict], or TradeData
            Trade data in any format accepted by ``TradeData.from_auto``.
        n_regimes : int
            Number of latent regimes for HMM detection (default 2).

        Returns
        -------
        dict
            JSON-serialisable result following the SkillResult schema.
        """
        td = TradeData.from_auto(data)

        if td.n < RegimeEdgeDecomposer.MIN_TRADES:
            return SkillResult(
                skill_name="regime_edge",
                valid=False,
                reason=f"Insufficient trades: {td.n} < {RegimeEdgeDecomposer.MIN_TRADES}",
            ).to_dict()

        r = td.r_multiples

        # --- HMM-based regime detection ---
        hmm_result = RegimeEdgeDecomposer.hmm_regime_detection(r, n_regimes=n_regimes)
        hmm_labels = hmm_result.get("regime_labels", [])
        hmm_edge = {}
        if hmm_labels:
            hmm_edge = RegimeEdgeDecomposer.per_regime_edge(r, hmm_labels)

        # --- Price-regime decomposition (if available) ---
        price_regime_edge: Dict[str, Any] = {}
        if td.regimes is not None and any(rg is not None for rg in td.regimes):
            price_labels = [rg if rg is not None else "unknown" for rg in td.regimes]
            price_regime_edge = RegimeEdgeDecomposer.per_regime_edge(r, price_labels)

        # --- Scoring ---
        score = RegimeEdgeDecomposer._compute_score(hmm_edge, price_regime_edge)
        grade = _grade_from_score(score)
        verdict = RegimeEdgeDecomposer._build_verdict(score, hmm_edge, price_regime_edge)

        return SkillResult(
            skill_name="regime_edge",
            valid=True,
            score=float(score),
            grade=grade,
            verdict=verdict,
            metrics={
                "hmm": hmm_result,
                "hmm_regime_edge": hmm_edge,
                "price_regime_edge": price_regime_edge,
            },
            diagnostics={
                "n_trades": int(td.n),
                "n_hmm_regimes": n_regimes,
                "has_price_regimes": bool(price_regime_edge),
            },
        ).to_dict()

    # ------------------------------------------------------------------
    #  HMM regime detection
    # ------------------------------------------------------------------

    @staticmethod
    def hmm_regime_detection(
        r_multiples: np.ndarray, n_regimes: int = 2
    ) -> Dict[str, Any]:
        """Detect latent regimes in R-multiples using a Gaussian HMM.

        Tries ``hmmlearn.GaussianHMM`` first.  Falls back to a manual
        2-state EM implementation (~80 lines) that uses the forward-backward
        algorithm for inference and Viterbi for decoding.

        Parameters
        ----------
        r_multiples : ndarray
            Array of R-multiples.
        n_regimes : int
            Number of hidden states (default 2).

        Returns
        -------
        dict with regime_params, regime_labels, time_in_regime,
        transition_matrix, stationary_distribution, regime_weighted_edge, bic.
        """
        n = len(r_multiples)

        # --- Try hmmlearn first -------------------------------------------
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]

            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
            )
            X = r_multiples.reshape(-1, 1)
            model.fit(X)
            labels = model.predict(X).tolist()
            means = model.means_.flatten().tolist()
            stds = [float(np.sqrt(v)) for v in model.covars_.flatten()]
            trans = model.transmat_.tolist()
            loglik = float(model.score(X) * n)
            n_params = n_regimes * 2 + n_regimes * (n_regimes - 1)
            bic_val = -2 * loglik + n_params * np.log(n)

            return RegimeEdgeDecomposer._package_hmm_result(
                r_multiples, labels, means, stds, trans, bic_val, n_regimes,
            )
        except ImportError:
            pass

        # --- Manual 2-state Gaussian HMM ----------------------------------
        if n_regimes != 2:
            return {
                "regime_params": {},
                "regime_labels": [],
                "time_in_regime": {},
                "transition_matrix": [],
                "stationary_distribution": [],
                "regime_weighted_edge": None,
                "bic": None,
                "method": "manual_em_unavailable",
                "note": "Manual EM only supports 2 regimes; install hmmlearn for more.",
            }

        labels, means, stds, trans, loglik = RegimeEdgeDecomposer._manual_em_2state(
            r_multiples
        )
        n_params = 2 * 2 + 2 * 1  # 2 means + 2 vars + 2 trans probs (row-stochastic)
        bic_val = -2 * loglik + n_params * np.log(n)

        result = RegimeEdgeDecomposer._package_hmm_result(
            r_multiples, labels, means, stds, trans, bic_val, 2,
        )
        result["method"] = "manual_em"
        return result

    @staticmethod
    def _manual_em_2state(r: np.ndarray):
        """Manual 2-state Gaussian HMM via Baum-Welch EM.

        Implements the full forward-backward algorithm for E-step,
        closed-form M-step, and Viterbi decoding.

        Returns
        -------
        labels : list[int]
        means : list[float]
        stds : list[float]
        trans : list[list[float]]
        loglik : float
        """
        n = len(r)
        K = 2

        # --- Initialisation -----------------------------------------------
        med = float(np.median(r))
        mu = np.array([
            float(np.mean(r[r > med])) if np.sum(r > med) > 0 else med + 0.1,
            float(np.mean(r[r <= med])) if np.sum(r <= med) > 0 else med - 0.1,
        ])
        overall_std = float(np.std(r, ddof=0))
        sigma = np.array([max(overall_std, 1e-6)] * K)
        A = np.array([[0.9, 0.1], [0.1, 0.9]])  # transition matrix
        pi = np.array([0.5, 0.5])  # initial state distribution

        prev_ll = -np.inf
        MAX_ITER = 100
        TOL = 1e-4

        def _log_gaussian(x: np.ndarray, m: float, s: float) -> np.ndarray:
            """Log N(x | m, s^2)."""
            return -0.5 * np.log(2 * np.pi) - np.log(max(s, 1e-12)) - 0.5 * ((x - m) / max(s, 1e-12)) ** 2

        for iteration in range(MAX_ITER):
            # --- E-step: forward-backward ---------------------------------
            # Emission log-probs: (n, K)
            log_B = np.column_stack([_log_gaussian(r, mu[k], sigma[k]) for k in range(K)])

            # Forward pass (log-space for numerical stability)
            log_alpha = np.zeros((n, K))
            log_alpha[0] = np.log(pi + 1e-300) + log_B[0]

            for t in range(1, n):
                for j in range(K):
                    log_alpha[t, j] = (
                        _logsumexp_vec(log_alpha[t - 1] + np.log(A[:, j] + 1e-300))
                        + log_B[t, j]
                    )

            loglik = float(_logsumexp_vec(log_alpha[-1]))

            # Backward pass (log-space)
            log_beta = np.zeros((n, K))
            for t in range(n - 2, -1, -1):
                for i in range(K):
                    vals = np.log(A[i, :] + 1e-300) + log_B[t + 1] + log_beta[t + 1]
                    log_beta[t, i] = _logsumexp_vec(vals)

            # Posterior: gamma_t(k) = P(S_t = k | r_{1:T})
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp_vec(log_gamma[0])  # normalise (per row below)
            for t in range(n):
                log_gamma[t] -= _logsumexp_vec(log_gamma[t])
            gamma = np.exp(log_gamma)

            # Xi: P(S_t=i, S_{t+1}=j | r_{1:T})
            xi = np.zeros((n - 1, K, K))
            for t in range(n - 1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = (
                            log_alpha[t, i]
                            + np.log(A[i, j] + 1e-300)
                            + log_B[t + 1, j]
                            + log_beta[t + 1, j]
                        )
                # Normalise
                log_norm = _logsumexp_2d(xi[t])
                xi[t] -= log_norm
            xi = np.exp(xi)

            # --- Convergence check ----------------------------------------
            if abs(loglik - prev_ll) < TOL and iteration > 0:
                break
            prev_ll = loglik

            # --- M-step ---------------------------------------------------
            pi = gamma[0] / np.sum(gamma[0])
            pi = np.clip(pi, 1e-6, 1.0)
            pi /= pi.sum()

            for i in range(K):
                denom_a = np.sum(gamma[:-1, i])
                if denom_a > 1e-12:
                    for j in range(K):
                        A[i, j] = np.sum(xi[:, i, j]) / denom_a
                A[i] = np.clip(A[i], 1e-6, 1.0)
                A[i] /= A[i].sum()

            for k in range(K):
                gamma_sum = np.sum(gamma[:, k])
                if gamma_sum > 1e-12:
                    mu[k] = np.sum(gamma[:, k] * r) / gamma_sum
                    sigma[k] = np.sqrt(
                        np.sum(gamma[:, k] * (r - mu[k]) ** 2) / gamma_sum
                    )
                    sigma[k] = max(sigma[k], 1e-6)

        # --- Viterbi decoding ---------------------------------------------
        log_delta = np.zeros((n, K))
        psi = np.zeros((n, K), dtype=int)
        log_delta[0] = np.log(pi + 1e-300) + log_B[0]
        for t in range(1, n):
            for j in range(K):
                candidates = log_delta[t - 1] + np.log(A[:, j] + 1e-300)
                psi[t, j] = int(np.argmax(candidates))
                log_delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        # Back-track
        labels = [0] * n
        labels[-1] = int(np.argmax(log_delta[-1]))
        for t in range(n - 2, -1, -1):
            labels[t] = int(psi[t + 1, labels[t + 1]])

        # --- Re-label so regime 0 has higher mean (the "good" regime) ---
        if mu[1] > mu[0]:
            labels = [1 - lbl for lbl in labels]
            mu = mu[::-1]
            sigma = sigma[::-1]
            A = A[::-1, ::-1]
            pi = pi[::-1]

        return (
            labels,
            [round(float(m), 6) for m in mu],
            [round(float(s), 6) for s in sigma],
            [[round(float(a), 6) for a in row] for row in A],
            float(loglik),
        )

    @staticmethod
    def _package_hmm_result(
        r_multiples: np.ndarray,
        labels: list,
        means: list,
        stds: list,
        trans: list,
        bic_val: float,
        n_regimes: int,
    ) -> Dict[str, Any]:
        """Build standardised HMM result dict from fitted parameters."""
        n = len(r_multiples)
        label_arr = np.array(labels)

        # Regime params
        regime_params = {}
        for k in range(n_regimes):
            count_k = int(np.sum(label_arr == k))
            regime_params[str(k)] = {
                "mean": round(float(means[k]), 4),
                "std": round(float(stds[k]), 4),
                "label": "favourable" if k == 0 else f"adverse_{k}" if n_regimes > 2 else "adverse",
                "n_trades": count_k,
            }

        # Time in regime
        time_in_regime = {}
        for k in range(n_regimes):
            time_in_regime[str(k)] = round(float(np.sum(label_arr == k) / n), 4)

        # Stationary distribution from transition matrix
        trans_np = np.array(trans)
        stat_dist = RegimeEdgeDecomposer._stationary_from_trans(trans_np)

        # Regime-weighted edge
        weighted_edge = 0.0
        for k in range(n_regimes):
            frac = time_in_regime.get(str(k), 0.0)
            weighted_edge += frac * means[k]

        return {
            "regime_params": regime_params,
            "regime_labels": labels,
            "time_in_regime": time_in_regime,
            "transition_matrix": trans if isinstance(trans, list) else [
                [round(float(x), 6) for x in row] for row in trans
            ],
            "stationary_distribution": [round(float(s), 6) for s in stat_dist],
            "regime_weighted_edge": round(float(weighted_edge), 4),
            "bic": round(float(bic_val), 2) if bic_val is not None else None,
        }

    # ------------------------------------------------------------------
    #  Per-regime edge
    # ------------------------------------------------------------------

    @staticmethod
    def per_regime_edge(r_multiples: np.ndarray, labels: list) -> Dict[str, Any]:
        """Compute edge statistics for each regime label.

        Parameters
        ----------
        r_multiples : ndarray
            Array of R-multiples.
        labels : list
            Regime labels aligned to r_multiples (same length).

        Returns
        -------
        dict keyed by label with mean_r, win_rate, expectancy, n, std_r.
        """
        labels_arr = np.array(labels)
        unique_labels = sorted(set(str(lbl) for lbl in labels))
        result: Dict[str, Any] = {}

        for lbl in unique_labels:
            mask = np.array([str(l) == lbl for l in labels_arr])
            subset = r_multiples[mask]
            n_sub = len(subset)
            if n_sub == 0:
                continue
            mean_r = float(np.mean(subset))
            std_r = float(np.std(subset, ddof=1)) if n_sub > 1 else 0.0
            win_rate = float(np.mean(subset > 0))
            wins = subset[subset > 0]
            losses = subset[subset <= 0]
            avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0
            expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

            result[str(lbl)] = {
                "mean_r": round(mean_r, 4),
                "win_rate": round(win_rate, 4),
                "expectancy": round(float(expectancy), 4),
                "n": int(n_sub),
                "std_r": round(std_r, 4),
            }

        return result

    # ------------------------------------------------------------------
    #  Transition matrix utilities
    # ------------------------------------------------------------------

    @staticmethod
    def regime_transition_matrix(labels: list) -> Dict[str, Any]:
        """Compute transition matrix from a label sequence.

        A[i,j] = count(labels[t]==i and labels[t+1]==j) / count(labels[t]==i)

        Returns
        -------
        dict with transition_matrix and stationary_distribution.
        """
        unique = sorted(set(labels))
        k_map = {u: i for i, u in enumerate(unique)}
        K = len(unique)
        counts = np.zeros((K, K))

        for t in range(len(labels) - 1):
            i = k_map[labels[t]]
            j = k_map[labels[t + 1]]
            counts[i, j] += 1

        # Normalise rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        trans = counts / row_sums

        stat_dist = RegimeEdgeDecomposer._stationary_from_trans(trans)

        return {
            "transition_matrix": [[round(float(x), 6) for x in row] for row in trans],
            "state_labels": [str(u) for u in unique],
            "stationary_distribution": [round(float(s), 6) for s in stat_dist],
        }

    @staticmethod
    def _stationary_from_trans(A: np.ndarray) -> np.ndarray:
        """Compute stationary distribution as the left eigenvector with eigenvalue 1."""
        try:
            eigenvalues, eigenvectors = np.linalg.eig(A.T)
            # Find eigenvector closest to eigenvalue 1
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stat = np.real(eigenvectors[:, idx])
            stat = np.abs(stat)
            total = stat.sum()
            if total > 1e-12:
                stat = stat / total
            else:
                stat = np.ones(len(stat)) / len(stat)
            return stat
        except Exception:
            K = A.shape[0]
            return np.ones(K) / K

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(
        hmm_edge: Dict[str, Any],
        price_edge: Dict[str, Any],
    ) -> float:
        """Derive a 0-100 score from regime-edge decomposition.

        Scoring guide:
            85+   All regimes profitable
            65    Dominant regime (>50% of time) profitable
            40    Only minor regime profitable
            25    No regime profitable but some near-zero
            <15   All regimes negative
        """
        # Combine both edge sources (HMM has priority)
        all_edges = {}
        all_edges.update(price_edge)
        all_edges.update(hmm_edge)

        if not all_edges:
            return 30.0

        means = []
        weights = []
        for key, stats in all_edges.items():
            means.append(stats.get("mean_r", 0.0))
            weights.append(stats.get("n", 1))

        total_n = sum(weights)
        if total_n == 0:
            return 30.0

        fracs = [w / total_n for w in weights]
        n_profitable = sum(1 for m in means if m > 0)
        n_total = len(means)

        # Dominant regime = the one with the most trades
        dominant_idx = int(np.argmax(weights))
        dominant_mean = means[dominant_idx]
        dominant_frac = fracs[dominant_idx]

        # All regimes profitable
        if n_profitable == n_total and n_total > 0:
            min_mean = min(means)
            if min_mean > 0.1:
                return min(95.0, 85.0 + min_mean * 10)
            return 85.0

        # Dominant regime profitable
        if dominant_mean > 0 and dominant_frac > 0.5:
            # Scale by how strong the dominant edge is
            base = 65.0
            if dominant_mean > 0.2:
                base += 10.0
            # Penalise if minor regime is deeply negative
            worst_mean = min(means)
            if worst_mean < -0.3:
                base -= 10.0
            return min(80.0, base)

        # Only minor regime profitable
        if n_profitable > 0:
            return 40.0

        # No regime profitable but some near zero
        best_mean = max(means)
        if best_mean > -0.05:
            return 25.0

        # All regimes negative
        return max(5.0, 15.0 + best_mean * 10)  # deeper negative => lower score

    @staticmethod
    def _build_verdict(
        score: float,
        hmm_edge: Dict[str, Any],
        price_edge: Dict[str, Any],
    ) -> str:
        """One-liner human-readable verdict."""
        if score >= 85:
            return "Edge is positive across all detected regimes - robust strategy."
        if score >= 65:
            return (
                "Edge exists in the dominant market regime but may weaken or "
                "reverse in less frequent conditions."
            )
        if score >= 40:
            return (
                "Edge is regime-dependent: only profitable in a minor regime.  "
                "Performance is fragile if market conditions shift."
            )
        if score >= 20:
            return (
                "No regime shows clear profitability.  Strategy lacks "
                "consistent edge in any detected market state."
            )
        return (
            "All regimes show negative expectancy.  Strategy is "
            "unprofitable regardless of market conditions."
        )


# ======================================================================
#  Module-level helpers (log-sum-exp in pure numpy)
# ======================================================================

def _logsumexp_vec(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp for a 1-D array."""
    c = float(np.max(x))
    if not np.isfinite(c):
        return float(-np.inf)
    return c + float(np.log(np.sum(np.exp(x - c))))


def _logsumexp_2d(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp for a 2-D array (flattened)."""
    return _logsumexp_vec(x.ravel())
