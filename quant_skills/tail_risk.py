"""
Tail-risk analysis skill for quantitative strategy auditing.

Examines distributional properties of R-multiples to detect fat tails,
volatility clustering, and Value-at-Risk model disagreement.  Heavy tails
invalidate the normal-distribution assumptions baked into naive position-sizing
and many risk metrics, so catching them early is critical.

Analyses included:
    - Distributional shape (skewness, kurtosis, Jarque-Bera normality test)
    - Hill tail-index estimator (tail heaviness from order statistics)
    - GARCH(1,1) volatility-clustering detection
    - VaR comparison (Normal, Historical, Cornish-Fisher) + CVaR
    - Q-Q plot data for front-end visualisation
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import TradeData, SkillResult, _grade_from_score


class TailRiskAnalyzer:
    """Stateless analyser for tail-risk properties of trade returns."""

    MIN_TRADES = 30

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(data: Any, var_confidence: float = 0.95) -> Dict[str, Any]:
        """Run the full tail-risk analysis suite.

        Parameters
        ----------
        data : array-like, list[dict], or TradeData
            Trade data in any format accepted by ``TradeData.from_auto``.
        var_confidence : float
            Confidence level for VaR/CVaR calculations (default 0.95).

        Returns
        -------
        dict
            JSON-serialisable result following the SkillResult schema.
        """
        td = TradeData.from_auto(data)

        if td.n < TailRiskAnalyzer.MIN_TRADES:
            return SkillResult(
                skill_name="tail_risk",
                valid=False,
                reason=f"Insufficient trades: {td.n} < {TailRiskAnalyzer.MIN_TRADES}",
            ).to_dict()

        r = td.r_multiples

        # Sub-analyses
        tail_stats = TailRiskAnalyzer.tail_statistics(r)
        hill = TailRiskAnalyzer.hill_estimator(r)
        garch = TailRiskAnalyzer.garch_fit(r)
        var_comp = TailRiskAnalyzer.var_comparison(r, confidence=var_confidence)
        qq = TailRiskAnalyzer.qq_data(r)

        # --- Scoring (0-100) -----------------------------------------
        score = TailRiskAnalyzer._compute_score(tail_stats, hill, garch, var_comp)
        grade = _grade_from_score(score)
        verdict = TailRiskAnalyzer._build_verdict(score, tail_stats, garch, var_comp)

        return SkillResult(
            skill_name="tail_risk",
            valid=True,
            score=float(score),
            grade=grade,
            verdict=verdict,
            metrics={
                "tail_statistics": tail_stats,
                "hill_estimator": hill,
                "garch": garch,
                "var_comparison": var_comp,
            },
            diagnostics={
                "qq_data": qq,
                "n_trades": int(td.n),
            },
        ).to_dict()

    # ------------------------------------------------------------------
    #  Sub-analyses
    # ------------------------------------------------------------------

    @staticmethod
    def tail_statistics(r_multiples: np.ndarray) -> Dict[str, Any]:
        """Compute distributional shape statistics and Jarque-Bera normality test.

        Formulae
        --------
        skew        = E[(X-mu)^3] / sigma^3
        kurt        = E[(X-mu)^4] / sigma^4
        excess_kurt = kurt - 3
        SE_skew     = sqrt(6/n)
        SE_kurt     = sqrt(24/n)
        z_skew      = skew / SE_skew
        z_kurt      = excess_kurt / SE_kurt
        JB          = (n/6) * (skew^2 + excess_kurt^2 / 4)   ~  chi2(2)

        Returns
        -------
        dict with skewness, excess_kurtosis, jarque_bera_stat, jarque_bera_p,
        z_skew, z_kurt, is_normal (JB p > 0.05).
        """
        from scipy.stats import chi2 as _chi2  # lazy import

        n = len(r_multiples)
        mu = np.mean(r_multiples)
        sigma = np.std(r_multiples, ddof=0)
        if sigma < 1e-12:
            return {
                "skewness": 0.0,
                "excess_kurtosis": 0.0,
                "jarque_bera_stat": 0.0,
                "jarque_bera_p": 1.0,
                "z_skew": 0.0,
                "z_kurt": 0.0,
                "is_normal": True,
            }

        centered = r_multiples - mu
        skew = float(np.mean(centered ** 3) / sigma ** 3)
        kurt = float(np.mean(centered ** 4) / sigma ** 4)
        excess_kurt = kurt - 3.0

        se_skew = float(np.sqrt(6.0 / n))
        se_kurt = float(np.sqrt(24.0 / n))
        z_skew = skew / se_skew if se_skew > 1e-12 else 0.0
        z_kurt = excess_kurt / se_kurt if se_kurt > 1e-12 else 0.0

        jb_stat = (n / 6.0) * (skew ** 2 + excess_kurt ** 2 / 4.0)
        jb_p = float(1.0 - _chi2.cdf(jb_stat, df=2))

        return {
            "skewness": round(float(skew), 4),
            "excess_kurtosis": round(float(excess_kurt), 4),
            "jarque_bera_stat": round(float(jb_stat), 4),
            "jarque_bera_p": round(float(jb_p), 6),
            "z_skew": round(float(z_skew), 4),
            "z_kurt": round(float(z_kurt), 4),
            "is_normal": bool(jb_p > 0.05),
        }

    @staticmethod
    def hill_estimator(r_multiples: np.ndarray) -> Dict[str, Any]:
        """Hill tail-index estimator from upper order statistics of |r|.

        Sorts |r| in descending order and computes Hill estimates over
        a range of thresholds k to identify a stable plateau.

        Parameters
        ----------
        r_multiples : ndarray
            Array of R-multiples.

        Returns
        -------
        dict with tail_index, tail_exponent (alpha=1/H), finite_variance,
        and hill_plot (list of [k, H_k] pairs).
        """
        absvals = np.abs(r_multiples)
        absvals = absvals[absvals > 1e-12]  # drop zeros
        n = len(absvals)

        if n < 10:
            return {
                "tail_index": None,
                "tail_exponent": None,
                "finite_variance": None,
                "hill_plot": [],
            }

        sorted_r = np.sort(absvals)[::-1]  # descending
        k_min = max(5, int(0.05 * n))
        k_max = int(0.30 * n)
        k_max = max(k_max, k_min + 1)  # ensure at least one point

        hill_plot: List[List[float]] = []
        hill_values: List[float] = []

        for k in range(k_min, k_max + 1):
            # H_k = (1/k) * sum(ln(r_(i) / r_(k+1))) for i = 1..k
            # r_(i) are sorted_r[0..k-1], r_(k+1) is sorted_r[k]
            if k >= n:
                break
            anchor = sorted_r[k]
            if anchor < 1e-12:
                continue
            log_ratios = np.log(sorted_r[:k] / anchor)
            h_k = float(np.mean(log_ratios))
            if h_k > 1e-12:
                hill_values.append(h_k)
                hill_plot.append([int(k), round(h_k, 6)])

        if not hill_values:
            return {
                "tail_index": None,
                "tail_exponent": None,
                "finite_variance": None,
                "hill_plot": [],
            }

        tail_index = float(np.median(hill_values))
        tail_exponent = 1.0 / tail_index if tail_index > 1e-12 else float("inf")
        finite_variance = bool(tail_exponent > 2.0)

        return {
            "tail_index": round(tail_index, 4),
            "tail_exponent": round(float(tail_exponent), 4),
            "finite_variance": finite_variance,
            "hill_plot": hill_plot,
        }

    @staticmethod
    def garch_fit(r_multiples: np.ndarray) -> Dict[str, Any]:
        """Fit GARCH(1,1) and test for volatility clustering.

        Uses ``scipy.optimize.minimize`` (L-BFGS-B) to maximise the
        conditional Gaussian log-likelihood:

            sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}
            LL       = -0.5 * sum( log(sigma2_t) + eps2_t / sigma2_t )

        Also computes the Ljung-Box Q statistic on squared returns as a
        model-free fallback for detecting autocorrelation in volatility.

        Returns
        -------
        dict with garch_params (omega, alpha, beta, persistence),
        has_clustering, ljung_box_stat, ljung_box_p, sq_autocorr_lag1.
        """
        from scipy.optimize import minimize as _minimize  # lazy import
        from scipy.stats import chi2 as _chi2

        eps = r_multiples - np.mean(r_multiples)
        eps2 = eps ** 2
        n = len(eps)
        var_eps = float(np.var(eps))

        # --- Ljung-Box on squared returns ---------------------------------
        m = min(10, n // 5)
        m = max(m, 1)
        sq_centered = eps2 - np.mean(eps2)
        denom = float(np.sum(sq_centered ** 2))
        rho_sq: List[float] = []
        for k in range(1, m + 1):
            if denom < 1e-15:
                rho_sq.append(0.0)
            else:
                rho_sq.append(float(np.sum(sq_centered[:n - k] * sq_centered[k:]) / denom))

        lb_stat = 0.0
        for k_idx, rho_k in enumerate(rho_sq):
            k_val = k_idx + 1
            lb_stat += (rho_k ** 2) / (n - k_val)
        lb_stat *= n * (n + 2)
        lb_p = float(1.0 - _chi2.cdf(lb_stat, df=m))
        sq_ac_lag1 = rho_sq[0] if rho_sq else 0.0

        # --- GARCH(1,1) MLE via L-BFGS-B ---------------------------------
        garch_params = {"omega": None, "alpha": None, "beta": None, "persistence": None}
        garch_converged = False

        if var_eps > 1e-12 and n >= 20:
            def _neg_loglik(params: np.ndarray) -> float:
                omega, alpha, beta = params
                sigma2 = np.empty(n)
                sigma2[0] = var_eps
                for t in range(1, n):
                    sigma2[t] = omega + alpha * eps2[t - 1] + beta * sigma2[t - 1]
                    if sigma2[t] < 1e-12:
                        sigma2[t] = 1e-12
                ll = -0.5 * np.sum(np.log(sigma2) + eps2 / sigma2)
                return -ll  # minimise negative LL

            x0 = np.array([0.05 * var_eps, 0.10, 0.85])
            bounds = [(1e-6, None), (1e-6, 0.5), (1e-6, 0.99)]

            try:
                res = _minimize(
                    _neg_loglik, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-10},
                )
                if res.success or res.fun < _neg_loglik(x0):
                    omega_hat, alpha_hat, beta_hat = res.x
                    persistence = alpha_hat + beta_hat
                    if persistence < 1.0:
                        garch_params = {
                            "omega": round(float(omega_hat), 6),
                            "alpha": round(float(alpha_hat), 4),
                            "beta": round(float(beta_hat), 4),
                            "persistence": round(float(persistence), 4),
                        }
                        garch_converged = True
            except Exception:
                pass  # fall through to LB-only result

        has_clustering = False
        if garch_converged and garch_params["persistence"] is not None:
            has_clustering = garch_params["persistence"] > 0.5
        if lb_p < 0.05:
            has_clustering = True

        return {
            "garch_params": garch_params,
            "garch_converged": garch_converged,
            "has_clustering": has_clustering,
            "ljung_box_stat": round(float(lb_stat), 4),
            "ljung_box_p": round(float(lb_p), 6),
            "sq_autocorr_lag1": round(float(sq_ac_lag1), 4),
        }

    @staticmethod
    def var_comparison(r_multiples: np.ndarray, confidence: float = 0.95) -> Dict[str, Any]:
        """Compare Normal, Historical, and Cornish-Fisher VaR estimates.

        Formulae
        --------
        Normal VaR :  mu - z_alpha * sigma
        Historical :  percentile(r, (1-conf)*100)
        Cornish-Fisher z_cf :
            z + (z^2-1)*S/6 + (z^3-3z)*K/24 - (2z^3-5z)*S^2/36
        CF VaR :  mu - z_cf * sigma
        CVaR   :  mean(r[r < historical_var])

        Returns
        -------
        dict with normal_var, historical_var, cornish_fisher_var, cvar,
        most_conservative, normal_underestimate_pct.
        """
        from scipy.stats import norm as _norm  # lazy import

        mu = float(np.mean(r_multiples))
        sigma = float(np.std(r_multiples, ddof=1))
        if sigma < 1e-12:
            sigma = 1e-12
        z_alpha = float(_norm.ppf(1.0 - confidence))  # negative for left tail

        # Normal VaR (left-tail loss, expressed as the R-multiple threshold)
        normal_var = mu + z_alpha * sigma  # z_alpha is negative

        # Historical VaR
        historical_var = float(np.percentile(r_multiples, (1.0 - confidence) * 100.0))

        # Cornish-Fisher VaR
        centered = r_multiples - mu
        s = float(np.mean(centered ** 3) / sigma ** 3)  # skewness
        k = float(np.mean(centered ** 4) / sigma ** 4) - 3.0  # excess kurtosis
        z = z_alpha
        z_cf = (z + (z ** 2 - 1) * s / 6.0
                + (z ** 3 - 3 * z) * k / 24.0
                - (2 * z ** 3 - 5 * z) * s ** 2 / 36.0)
        cf_var = mu + z_cf * sigma

        # CVaR (Expected Shortfall) — historical
        tail = r_multiples[r_multiples <= historical_var]
        cvar = float(np.mean(tail)) if len(tail) > 0 else float(historical_var)

        # Most conservative (most negative = worst-case)
        vars_dict = {
            "normal": normal_var,
            "historical": historical_var,
            "cornish_fisher": cf_var,
        }
        most_conservative = min(vars_dict, key=vars_dict.get)

        # How much does Normal VaR underestimate the tail?
        if abs(historical_var) > 1e-12:
            underest_pct = ((normal_var - historical_var) / abs(historical_var)) * 100.0
        else:
            underest_pct = 0.0

        return {
            "confidence": confidence,
            "normal_var": round(float(normal_var), 4),
            "historical_var": round(float(historical_var), 4),
            "cornish_fisher_var": round(float(cf_var), 4),
            "cvar": round(float(cvar), 4),
            "most_conservative": most_conservative,
            "normal_underestimate_pct": round(float(underest_pct), 2),
        }

    @staticmethod
    def qq_data(r_multiples: np.ndarray) -> Dict[str, Any]:
        """Generate Q-Q plot data comparing empirical quantiles to N(0,1).

        Returns a list of [theoretical, empirical] pairs suitable for
        front-end scatter-plot rendering.
        """
        from scipy.stats import norm as _norm  # lazy import

        n = len(r_multiples)
        sorted_r = np.sort(r_multiples)
        # Standardise to zero mean / unit variance for comparison
        mu = np.mean(r_multiples)
        sigma = np.std(r_multiples, ddof=1)
        if sigma < 1e-12:
            sigma = 1.0
        standardised = (sorted_r - mu) / sigma
        theoretical = _norm.ppf((np.arange(1, n + 1) - 0.5) / n)

        pairs = [
            [round(float(t), 4), round(float(e), 4)]
            for t, e in zip(theoretical, standardised)
        ]
        return {"qq_pairs": pairs, "n": int(n)}

    # ------------------------------------------------------------------
    #  Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(
        tail_stats: Dict[str, Any],
        hill: Dict[str, Any],
        garch: Dict[str, Any],
        var_comp: Dict[str, Any],
    ) -> float:
        """Derive a 0-100 score from the sub-analyses.

        Scoring guide:
            80+   Near-normal, no clustering, VaR models agree
            60-80 Mild fat tails (excess_kurt < 2), some clustering
            40-60 Moderate fat tails (excess_kurt 2-5)
            20-40 Heavy tails (excess_kurt > 5), strong clustering
            <20   Extreme tails, VaR unreliable
        """
        score = 80.0  # start from "healthy" baseline

        # --- Kurtosis penalty ----
        ek = abs(tail_stats.get("excess_kurtosis", 0.0))
        if ek < 1.0:
            pass  # near-normal
        elif ek < 2.0:
            score -= 10.0
        elif ek < 5.0:
            score -= 25.0
        elif ek < 10.0:
            score -= 40.0
        else:
            score -= 55.0

        # --- Skewness penalty (negative skew is worse for traders) ----
        skew = tail_stats.get("skewness", 0.0)
        if skew < -1.0:
            score -= 10.0
        elif skew < -0.5:
            score -= 5.0
        # Positive skew is desirable, small bonus
        if skew > 0.5:
            score += 5.0

        # --- Non-normality from JB ----
        if not tail_stats.get("is_normal", True):
            score -= 5.0

        # --- Hill estimator ----
        if hill.get("finite_variance") is False:
            score -= 15.0
        elif hill.get("tail_exponent") is not None:
            alpha = hill["tail_exponent"]
            if alpha < 3.0:
                score -= 10.0  # moderate tail heaviness

        # --- GARCH clustering ----
        if garch.get("has_clustering"):
            persistence = (garch.get("garch_params") or {}).get("persistence")
            if persistence is not None and persistence > 0.9:
                score -= 15.0
            elif persistence is not None and persistence > 0.7:
                score -= 10.0
            else:
                score -= 5.0

        # --- VaR model disagreement ----
        underest = abs(var_comp.get("normal_underestimate_pct", 0.0))
        if underest > 50.0:
            score -= 10.0
        elif underest > 25.0:
            score -= 5.0

        return float(max(0.0, min(100.0, score)))

    @staticmethod
    def _build_verdict(
        score: float,
        tail_stats: Dict[str, Any],
        garch: Dict[str, Any],
        var_comp: Dict[str, Any],
    ) -> str:
        """One-liner human-readable verdict."""
        if score >= 80:
            return "Return distribution is near-normal with well-behaved tails."
        if score >= 60:
            return (
                "Mild fat tails detected.  Normal-distribution assumptions "
                "are approximately valid but position sizing should use "
                "conservative VaR estimates."
            )
        if score >= 40:
            return (
                "Moderate fat tails present.  Normal VaR underestimates "
                f"true risk by ~{abs(var_comp.get('normal_underestimate_pct', 0)):.0f}%.  "
                "Use historical or Cornish-Fisher VaR for sizing."
            )
        if score >= 20:
            clustering = " with volatility clustering" if garch.get("has_clustering") else ""
            return (
                f"Heavy tails (excess kurtosis "
                f"{tail_stats.get('excess_kurtosis', '?')}){clustering}.  "
                "Standard risk models are unreliable; consider robust "
                "position-sizing and tail-risk hedging."
            )
        return (
            "Extreme tail risk.  Distribution has very heavy tails that "
            "invalidate most parametric risk measures.  Treat all VaR "
            "numbers as severe underestimates."
        )
