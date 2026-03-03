"""
Edge Existence Validator — statistical tests for genuine trading edge.

Determines whether a strategy's out-of-sample returns reflect a real,
statistically significant edge or could be explained by chance.  Uses a
non-parametric sign-flip permutation test (no distributional assumptions),
complemented by power analysis and multiple-comparison correction so that
results generalise across multi-strategy portfolios.

All public methods are stateless ``@staticmethod`` callables that return
plain JSON-serialisable dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import TradeData, SkillResult, _grade_from_score


class EdgeExistenceValidator:
    """Non-parametric statistical tests for genuine trading edge."""

    MIN_TRADES = 20

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(
        data: Any,
        n_permutations: int = 10_000,
        comparisons: int = 1,
        alpha: float = 0.05,
    ) -> dict:
        """Run all edge-existence tests and return a scored result dict.

        Parameters
        ----------
        data : TradeData | list[dict] | ndarray
            Trade data in any format accepted by ``TradeData.from_auto``.
        n_permutations : int
            Number of sign-flip permutations (default 10 000).
        comparisons : int
            Number of strategies being tested simultaneously (for
            multiple-comparison correction).  Set to 1 when evaluating a
            single strategy in isolation.
        alpha : float
            Family-wise error rate for significance tests.

        Returns
        -------
        dict
            ``SkillResult.to_dict()`` with metrics, diagnostics, score
            and grade.
        """
        td = TradeData.from_auto(data)

        if td.n < EdgeExistenceValidator.MIN_TRADES:
            return SkillResult(
                skill_name="edge_existence",
                valid=False,
                reason=(
                    f"Insufficient trades: {td.n} < "
                    f"{EdgeExistenceValidator.MIN_TRADES} minimum"
                ),
            ).to_dict()

        r = td.r_multiples

        # --- Sub-analyses ---
        perm = EdgeExistenceValidator.permutation_test(
            r, n_permutations=n_permutations
        )
        power = EdgeExistenceValidator.power_analysis(r, alpha=alpha)

        # Multiple-comparison correction (only meaningful when comparisons > 1)
        p_values = [perm["p_value_positive"]]
        if comparisons > 1:
            p_values = p_values * comparisons  # placeholder list
        mc = EdgeExistenceValidator.multi_comparison_correct(
            p_values, alpha=alpha
        )

        # --- Scoring ---
        score = EdgeExistenceValidator._compute_score(perm, power)
        grade = _grade_from_score(score)
        verdict = EdgeExistenceValidator._build_verdict(perm, power, score)

        return SkillResult(
            skill_name="edge_existence",
            valid=True,
            score=score,
            grade=grade,
            verdict=verdict,
            metrics={
                "observed_mean_r": perm["observed_mean_r"],
                "permutation_p_positive": perm["p_value_positive"],
                "permutation_p_negative": perm["p_value_negative"],
                "significant_positive_edge": perm["significant_positive"],
                "significant_negative_edge": perm["significant_negative"],
                "cohens_d": power["cohens_d"],
                "achieved_power": power["achieved_power"],
                "n_for_80pct_power": power["n_for_80pct_power"],
                "n_for_90pct_power": power["n_for_90pct_power"],
                "n_for_95pct_power": power["n_for_95pct_power"],
            },
            diagnostics={
                "permutation_test": perm,
                "power_analysis": power,
                "multi_comparison": mc,
            },
        ).to_dict()

    # ------------------------------------------------------------------
    #  Permutation test
    # ------------------------------------------------------------------

    @staticmethod
    def permutation_test(
        r_multiples: np.ndarray,
        n_permutations: int = 10_000,
        alpha: float = 0.05,
    ) -> dict:
        """Sign-flip permutation test for a symmetric-around-zero null.

        H0: The true expected R-multiple is zero (signs are arbitrary).
        For each permutation, multiply each R by a random sign drawn
        uniformly from {-1, +1} and compute the permuted mean.

        Parameters
        ----------
        r_multiples : ndarray
            1-D array of R-multiples.
        n_permutations : int
            Number of random sign-flip permutations.
        alpha : float
            Significance threshold.

        Returns
        -------
        dict
            Observed mean, null distribution statistics, two-sided
            p-values, significance flags, and null percentiles.
        """
        r = np.asarray(r_multiples, dtype=np.float64)
        n = len(r)
        observed_mean = float(np.mean(r))

        rng = np.random.default_rng()
        # (n_permutations, n) matrix of random signs
        signs = rng.choice([-1, 1], size=(n_permutations, n))
        perm_means = np.mean(signs * r, axis=1)

        null_mean = float(np.mean(perm_means))
        null_std = float(np.std(perm_means, ddof=0))

        # Right-tail: P(perm_mean >= observed_mean)
        p_positive = float(
            (np.sum(perm_means >= observed_mean) + 1) / (n_permutations + 1)
        )
        # Left-tail: P(perm_mean <= observed_mean)
        p_negative = float(
            (np.sum(perm_means <= observed_mean) + 1) / (n_permutations + 1)
        )

        null_percentiles = {
            str(q): float(np.percentile(perm_means, q))
            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        return {
            "observed_mean_r": round(observed_mean, 6),
            "null_mean": round(null_mean, 6),
            "null_std": round(null_std, 6),
            "p_value_positive": round(p_positive, 6),
            "p_value_negative": round(p_negative, 6),
            "significant_positive": bool(p_positive < alpha),
            "significant_negative": bool(p_negative < alpha),
            "null_percentiles": null_percentiles,
            "n_permutations": n_permutations,
        }

    # ------------------------------------------------------------------
    #  Power analysis
    # ------------------------------------------------------------------

    @staticmethod
    def power_analysis(
        r_multiples: np.ndarray,
        alpha: float = 0.05,
    ) -> dict:
        """Cohen's d effect size, achieved power, and required sample sizes.

        Parameters
        ----------
        r_multiples : ndarray
            1-D array of R-multiples.
        alpha : float
            Significance level for two-sided z-test.

        Returns
        -------
        dict
            ``cohens_d``, ``achieved_power``, ``n_for_80pct_power``,
            ``n_for_90pct_power``, ``n_for_95pct_power``.
        """
        from scipy.stats import norm  # lazy import

        r = np.asarray(r_multiples, dtype=np.float64)
        n = len(r)
        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1))

        if std_r < 1e-12:
            # Degenerate case — zero variance
            return {
                "cohens_d": float("inf") if abs(mean_r) > 0 else 0.0,
                "achieved_power": 1.0 if abs(mean_r) > 0 else 0.0,
                "n_for_80pct_power": 1 if abs(mean_r) > 0 else None,
                "n_for_90pct_power": 1 if abs(mean_r) > 0 else None,
                "n_for_95pct_power": 1 if abs(mean_r) > 0 else None,
            }

        d = abs(mean_r) / std_r
        z_alpha2 = norm.ppf(1 - alpha / 2)

        # Achieved power: Phi(|mean|*sqrt(n)/std - z_{alpha/2})
        achieved_power = float(
            norm.cdf(abs(mean_r) * np.sqrt(n) / std_r - z_alpha2)
        )

        # Required n for target power levels
        def _n_required(beta_target: float) -> Optional[int]:
            if d < 1e-12:
                return None
            z_beta = norm.ppf(1 - beta_target)
            n_req = ((z_alpha2 + z_beta) / d) ** 2
            return max(2, int(np.ceil(n_req)))

        return {
            "cohens_d": round(d, 6),
            "achieved_power": round(achieved_power, 6),
            "n_for_80pct_power": _n_required(0.20),
            "n_for_90pct_power": _n_required(0.10),
            "n_for_95pct_power": _n_required(0.05),
        }

    # ------------------------------------------------------------------
    #  Multiple-comparison correction
    # ------------------------------------------------------------------

    @staticmethod
    def multi_comparison_correct(
        p_values: list,
        alpha: float = 0.05,
    ) -> dict:
        """Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg corrections.

        Parameters
        ----------
        p_values : list[float]
            Raw p-values from independent tests.
        alpha : float
            Family-wise error rate.

        Returns
        -------
        dict
            Adjusted p-values and significance flags for each method.
        """
        m = len(p_values)
        pv = np.asarray(p_values, dtype=np.float64)

        # --- Bonferroni ---
        bonf_adj = np.minimum(pv * m, 1.0)
        bonf_sig = (bonf_adj < alpha).tolist()

        # --- Holm-Bonferroni (step-down) ---
        order = np.argsort(pv)
        holm_adj = np.empty(m, dtype=np.float64)
        cumulative_max = 0.0
        for rank, idx in enumerate(order):
            adjusted = pv[idx] * (m - rank)
            cumulative_max = max(cumulative_max, adjusted)
            holm_adj[idx] = min(cumulative_max, 1.0)
        holm_sig = (holm_adj < alpha).tolist()

        # --- Benjamini-Hochberg (step-up) ---
        bh_adj = np.empty(m, dtype=np.float64)
        order_asc = np.argsort(pv)
        cumulative_min = 1.0
        for i in range(m - 1, -1, -1):
            idx = order_asc[i]
            rank = i + 1  # 1-based
            adjusted = pv[idx] * m / rank
            cumulative_min = min(cumulative_min, adjusted)
            bh_adj[idx] = min(cumulative_min, 1.0)
        bh_sig = (bh_adj < alpha).tolist()

        return {
            "n_tests": m,
            "alpha": alpha,
            "raw_p_values": [round(float(p), 6) for p in pv],
            "bonferroni": {
                "adjusted": [round(float(p), 6) for p in bonf_adj],
                "significant": bonf_sig,
            },
            "holm_bonferroni": {
                "adjusted": [round(float(p), 6) for p in holm_adj],
                "significant": holm_sig,
            },
            "benjamini_hochberg": {
                "adjusted": [round(float(p), 6) for p in bh_adj],
                "significant": bh_sig,
            },
        }

    # ------------------------------------------------------------------
    #  Internal scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(perm: dict, power: dict) -> float:
        """Map permutation-test and power results to a 0-100 score.

        Scoring rubric
        --------------
        * Positive edge confirmed at p < 0.01 with power > 0.9  -> 85-100
        * Positive edge at p < 0.05 with power > 0.8            -> 65-84
        * Mean R ~ 0, not significant                            -> 35-50
        * Negative edge confirmed                                -> 0-15
        """
        p_pos = perm["p_value_positive"]
        p_neg = perm["p_value_negative"]
        mean_r = perm["observed_mean_r"]
        achieved = power["achieved_power"]

        # --- Positive edge ---
        if perm["significant_positive"]:
            if p_pos < 0.01 and achieved > 0.9:
                # Strong positive edge — scale 85-100 by power
                return float(85 + 15 * min(1.0, (achieved - 0.9) / 0.1))
            if p_pos < 0.01 and achieved > 0.8:
                return float(75 + 10 * (achieved - 0.8) / 0.1)
            if p_pos < 0.05 and achieved > 0.8:
                return float(65 + 10 * (achieved - 0.8) / 0.2)
            if p_pos < 0.05:
                return float(55 + 10 * min(1.0, achieved / 0.8))
            # Barely significant
            return float(50 + 5 * min(1.0, achieved / 0.5))

        # --- Not significant either way ---
        if not perm["significant_negative"]:
            # Lean slightly based on sign of mean_r
            base = 40.0
            if mean_r > 0:
                base += min(10.0, 10.0 * (1.0 - p_pos))
            else:
                base -= min(10.0, 10.0 * (1.0 - p_neg))
            return float(max(20.0, min(55.0, base)))

        # --- Negative edge confirmed ---
        if perm["significant_negative"]:
            if p_neg < 0.01 and achieved > 0.8:
                return float(max(0, 5 - 5 * min(1.0, abs(mean_r))))
            if p_neg < 0.01:
                return float(max(0, 10 - 5 * min(1.0, abs(mean_r))))
            return float(max(5, 15 - 10 * min(1.0, abs(mean_r))))

        return 40.0  # fallback

    @staticmethod
    def _build_verdict(perm: dict, power: dict, score: float) -> str:
        """Human-readable verdict string."""
        mean_r = perm["observed_mean_r"]
        p_pos = perm["p_value_positive"]
        achieved = power["achieved_power"]

        if perm["significant_positive"]:
            strength = "strong" if p_pos < 0.01 else "moderate"
            return (
                f"Statistically significant POSITIVE edge detected "
                f"({strength}, p={p_pos:.4f}, mean R={mean_r:+.4f}, "
                f"power={achieved:.2f})."
            )
        if perm["significant_negative"]:
            p_neg = perm["p_value_negative"]
            return (
                f"Statistically significant NEGATIVE edge detected "
                f"(p={p_neg:.4f}, mean R={mean_r:+.4f}). "
                f"Strategy is worse than random."
            )
        return (
            f"No statistically significant edge detected "
            f"(p_positive={p_pos:.4f}, mean R={mean_r:+.4f}, "
            f"power={achieved:.2f}). More data may be needed."
        )
