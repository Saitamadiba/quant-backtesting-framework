"""
Stationarity Analyzer — detect regime changes and parameter drift.

Answers the question: is the strategy's edge *stable* over time, or is it
degrading, improving, or flipping between regimes?  Uses four complementary
lenses:

1. **Kalman filter** — tracks the time-varying expected R-multiple with
   uncertainty bands.
2. **CUSUM change-point detection** — flags abrupt mean-shifts via Page's
   algorithm.
3. **Rolling KS test** — checks distributional stability across time
   windows.
4. **Parameter convergence** — measures whether the cumulative mean is
   converging or drifting.

All public methods are stateless ``@staticmethod`` callables that return
plain JSON-serialisable dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import TradeData, SkillResult, _grade_from_score


class StationarityAnalyzer:
    """Regime-change and parameter-drift diagnostics for R-multiple streams."""

    MIN_TRADES = 50

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(
        data: Any,
        window_size: int = 50,
    ) -> dict:
        """Run all stationarity tests and return a scored result dict.

        Parameters
        ----------
        data : TradeData | list[dict] | ndarray
            Trade data in any format accepted by ``TradeData.from_auto``.
        window_size : int
            Window size for the rolling KS test (default 50).

        Returns
        -------
        dict
            ``SkillResult.to_dict()`` with metrics, diagnostics, score
            and grade.
        """
        td = TradeData.from_auto(data)

        if td.n < StationarityAnalyzer.MIN_TRADES:
            return SkillResult(
                skill_name="stationarity",
                valid=False,
                reason=(
                    f"Insufficient trades: {td.n} < "
                    f"{StationarityAnalyzer.MIN_TRADES} minimum"
                ),
            ).to_dict()

        r = td.r_multiples

        # --- Sub-analyses ---
        kalman = StationarityAnalyzer.kalman_filter_edge(r)
        cusum = StationarityAnalyzer.cusum_changepoints(r)
        ks = StationarityAnalyzer.rolling_ks_test(r, window_size=window_size)
        convergence = StationarityAnalyzer.parameter_convergence(r)

        # --- Scoring ---
        score = StationarityAnalyzer._compute_score(
            kalman, cusum, ks, convergence
        )
        grade = _grade_from_score(score)
        verdict = StationarityAnalyzer._build_verdict(
            kalman, cusum, ks, convergence, score
        )

        return SkillResult(
            skill_name="stationarity",
            valid=True,
            score=score,
            grade=grade,
            verdict=verdict,
            metrics={
                "n_changepoints": cusum["n_changepoints"],
                "edge_trend": kalman["edge_trend"],
                "initial_edge": kalman["initial_edge"],
                "final_edge": kalman["final_edge"],
                "total_drift": kalman["total_drift"],
                "pct_time_positive": kalman["pct_time_positive"],
                "convergence_ratio": convergence["convergence_ratio"],
                "is_converging": convergence["is_converging"],
                "drift_slope": convergence["drift_slope"],
                "is_drifting": convergence["is_drifting"],
                "pct_unstable_windows": ks["pct_unstable"],
            },
            diagnostics={
                "kalman_filter": kalman,
                "cusum": cusum,
                "rolling_ks": ks,
                "parameter_convergence": convergence,
            },
        ).to_dict()

    # ------------------------------------------------------------------
    #  Kalman filter
    # ------------------------------------------------------------------

    @staticmethod
    def kalman_filter_edge(r_multiples: np.ndarray) -> dict:
        """Univariate Kalman filter for time-varying expected R-multiple.

        State model
        -----------
        mu_t = mu_{t-1} + w_t,  w_t ~ N(0, Q)
        r_t  = mu_t     + v_t,  v_t ~ N(0, R)

        Initialisation: mu_0 = r[0], P_0 = var(r).
        Q = max(1e-6, var(diff(r)) / 2).
        R = var(r) - Q, clamped to >= 0.1 * var(r).

        Returns
        -------
        dict
            Filtered edge trajectory, confidence bands, trend
            classification, and fraction of time the edge was positive.
        """
        r = np.asarray(r_multiples, dtype=np.float64)
        n = len(r)
        var_r = float(np.var(r, ddof=0))

        # Noise covariances
        if n > 1:
            diff_r = np.diff(r)
            Q = max(1e-6, float(np.var(diff_r, ddof=0)) / 2.0)
        else:
            Q = 1e-6
        R = max(0.1 * var_r, var_r - Q) if var_r > 0 else 1e-6

        # Kalman pass
        mu = np.empty(n, dtype=np.float64)
        P = np.empty(n, dtype=np.float64)
        mu[0] = r[0]
        P[0] = var_r if var_r > 0 else 1.0

        for t in range(1, n):
            # Predict
            mu_pred = mu[t - 1]
            P_pred = P[t - 1] + Q
            # Update
            K = P_pred / (P_pred + R)
            mu[t] = mu_pred + K * (r[t] - mu_pred)
            P[t] = (1.0 - K) * P_pred

        ci_upper = mu + 2.0 * np.sqrt(P)
        ci_lower = mu - 2.0 * np.sqrt(P)

        initial_edge = float(mu[0])
        final_edge = float(mu[-1])
        total_drift = float(final_edge - initial_edge)
        pct_positive = float(np.mean(mu > 0))

        # Trend classification based on last-quarter vs first-quarter
        q1_end = max(1, n // 4)
        q4_start = n - max(1, n // 4)
        early_mean = float(np.mean(mu[:q1_end]))
        late_mean = float(np.mean(mu[q4_start:]))
        std_mu = float(np.std(mu, ddof=0))
        threshold = 0.1 * std_mu if std_mu > 0 else 1e-6

        if late_mean - early_mean > threshold:
            edge_trend = "improving"
        elif early_mean - late_mean > threshold:
            edge_trend = "deteriorating"
        else:
            edge_trend = "stable"

        return {
            "filtered_edge": [round(float(x), 6) for x in mu],
            "ci_upper": [round(float(x), 6) for x in ci_upper],
            "ci_lower": [round(float(x), 6) for x in ci_lower],
            "initial_edge": round(initial_edge, 6),
            "final_edge": round(final_edge, 6),
            "total_drift": round(total_drift, 6),
            "edge_trend": edge_trend,
            "pct_time_positive": round(pct_positive, 4),
        }

    # ------------------------------------------------------------------
    #  CUSUM change-point detection
    # ------------------------------------------------------------------

    @staticmethod
    def cusum_changepoints(
        r_multiples: np.ndarray,
        reference_mean: Optional[float] = None,
    ) -> dict:
        """Page's CUSUM algorithm for mean-shift detection.

        Parameters
        ----------
        r_multiples : ndarray
            1-D array of R-multiples.
        reference_mean : float | None
            In-control mean.  Defaults to ``mean(r)``.

        Returns
        -------
        dict
            Change-point indices, CUSUM traces, and current-regime info.
        """
        r = np.asarray(r_multiples, dtype=np.float64)
        n = len(r)
        mu_0 = reference_mean if reference_mean is not None else float(np.mean(r))
        sigma = float(np.std(r, ddof=1)) if n > 1 else 1.0
        if sigma < 1e-12:
            sigma = 1.0

        k = 0.5 * sigma   # allowance — detect ~1-sigma shifts
        h = 5.0 * sigma   # threshold

        s_plus = np.zeros(n, dtype=np.float64)
        s_minus = np.zeros(n, dtype=np.float64)
        changepoints: list[int] = []

        for t in range(1, n):
            s_plus[t] = max(0.0, s_plus[t - 1] + (r[t] - mu_0 - k))
            s_minus[t] = max(0.0, s_minus[t - 1] - (r[t] - mu_0 + k))

            if s_plus[t] > h or s_minus[t] > h:
                changepoints.append(int(t))
                s_plus[t] = 0.0
                s_minus[t] = 0.0

        # Current regime stats
        regime_start = int(changepoints[-1]) if changepoints else 0
        regime_r = r[regime_start:]
        current_regime_mean = float(np.mean(regime_r)) if len(regime_r) > 0 else 0.0

        return {
            "changepoints": changepoints,
            "n_changepoints": len(changepoints),
            "cusum_plus": [round(float(x), 6) for x in s_plus],
            "cusum_minus": [round(float(x), 6) for x in s_minus],
            "current_regime_start": regime_start,
            "current_regime_mean": round(current_regime_mean, 6),
            "reference_mean": round(mu_0, 6),
            "threshold_h": round(h, 6),
            "allowance_k": round(k, 6),
        }

    # ------------------------------------------------------------------
    #  Rolling KS test
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_ks_test(
        r_multiples: np.ndarray,
        window_size: int = 50,
    ) -> dict:
        """Distributional stability via non-overlapping KS tests.

        Splits the R-multiple series into non-overlapping windows of
        ``window_size`` and runs a two-sample Kolmogorov-Smirnov test
        of each window against the full sample.  Bonferroni correction
        is applied for multiple windows.

        Parameters
        ----------
        r_multiples : ndarray
            1-D array of R-multiples.
        window_size : int
            Size of each non-overlapping window (default 50).

        Returns
        -------
        dict
            KS statistics and p-values per window, instability metrics.
        """
        from scipy.stats import ks_2samp  # lazy import

        r = np.asarray(r_multiples, dtype=np.float64)
        n = len(r)

        n_windows = n // window_size
        if n_windows < 2:
            return {
                "n_windows": max(1, n_windows),
                "ks_stats": [],
                "ks_pvalues": [],
                "n_unstable": 0,
                "pct_unstable": 0.0,
                "worst_window": None,
                "window_size": window_size,
            }

        ks_stats: list[float] = []
        ks_pvalues: list[float] = []

        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window_data = r[start:end]
            stat, pval = ks_2samp(window_data, r)
            ks_stats.append(round(float(stat), 6))
            ks_pvalues.append(round(float(pval), 6))

        # Bonferroni correction
        alpha_corrected = 0.05 / n_windows
        unstable = [p < alpha_corrected for p in ks_pvalues]
        n_unstable = sum(unstable)
        pct_unstable = round(float(n_unstable / n_windows), 4)

        worst_idx = int(np.argmax(ks_stats))
        worst_window = {
            "index": worst_idx,
            "start_trade": worst_idx * window_size,
            "ks_stat": ks_stats[worst_idx],
            "p_value": ks_pvalues[worst_idx],
        }

        return {
            "n_windows": n_windows,
            "ks_stats": ks_stats,
            "ks_pvalues": ks_pvalues,
            "n_unstable": n_unstable,
            "pct_unstable": pct_unstable,
            "worst_window": worst_window,
            "window_size": window_size,
        }

    # ------------------------------------------------------------------
    #  Parameter convergence
    # ------------------------------------------------------------------

    @staticmethod
    def parameter_convergence(r_multiples: np.ndarray) -> dict:
        """Cumulative mean convergence and drift detection.

        Computes the running cumulative mean of R-multiples and checks
        whether the second half is more stable than the first half
        (convergence) and whether there is a systematic linear drift.

        Returns
        -------
        dict
            Cumulative means, convergence ratio, drift statistics.
        """
        from scipy.stats import linregress  # lazy import

        r = np.asarray(r_multiples, dtype=np.float64)
        n = len(r)

        cum_mean = np.cumsum(r) / np.arange(1, n + 1)

        half = n // 2
        if half < 2:
            return {
                "cumulative_means": [round(float(x), 6) for x in cum_mean],
                "convergence_ratio": 1.0,
                "is_converging": False,
                "drift_slope": 0.0,
                "drift_p_value": 1.0,
                "is_drifting": False,
            }

        std_first = float(np.std(cum_mean[:half], ddof=0))
        std_second = float(np.std(cum_mean[half:], ddof=0))
        convergence_ratio = (
            round(std_second / std_first, 6) if std_first > 1e-12 else 1.0
        )
        is_converging = convergence_ratio < 1.0

        # Linear drift in cumulative mean
        x = np.arange(n, dtype=np.float64)
        slope, intercept, r_val, p_val, stderr = linregress(x, cum_mean)
        is_drifting = bool(p_val < 0.05)

        return {
            "cumulative_means": [round(float(x), 6) for x in cum_mean],
            "convergence_ratio": convergence_ratio,
            "is_converging": is_converging,
            "drift_slope": round(float(slope), 8),
            "drift_p_value": round(float(p_val), 6),
            "is_drifting": is_drifting,
        }

    # ------------------------------------------------------------------
    #  Internal scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(
        kalman: dict,
        cusum: dict,
        ks: dict,
        convergence: dict,
    ) -> float:
        """Map sub-analysis results to a 0-100 score.

        Scoring rubric
        --------------
        * 0 changepoints + converging + no drift + stable Kalman: 80+
        * 1 changepoint, mostly stable: 60-80
        * 2-3 changepoints, some drift: 40-60
        * Significant drift, many unstable windows: 20-40
        * Multiple breaks, severe deterioration: <20
        """
        n_cp = cusum["n_changepoints"]
        pct_unstable = ks["pct_unstable"]
        is_converging = convergence["is_converging"]
        is_drifting = convergence["is_drifting"]
        edge_trend = kalman["edge_trend"]
        pct_positive = kalman["pct_time_positive"]

        # Start from base score and adjust
        score = 60.0

        # --- Changepoints penalty ---
        if n_cp == 0:
            score += 15.0
        elif n_cp == 1:
            score += 5.0
        elif n_cp <= 3:
            score -= 10.0
        else:
            score -= 25.0

        # --- Window instability penalty ---
        score -= 20.0 * pct_unstable

        # --- Convergence bonus/penalty ---
        if is_converging:
            score += 8.0
        else:
            # Diverging cumulative mean
            cr = convergence["convergence_ratio"]
            if cr > 2.0:
                score -= 15.0
            elif cr > 1.5:
                score -= 8.0
            else:
                score -= 3.0

        # --- Drift penalty ---
        if is_drifting:
            score -= 10.0

        # --- Kalman trend adjustment ---
        if edge_trend == "improving":
            score += 5.0
        elif edge_trend == "deteriorating":
            score -= 10.0

        # --- Edge positivity bonus ---
        if pct_positive >= 0.9:
            score += 5.0
        elif pct_positive < 0.5:
            score -= 10.0

        return float(max(0.0, min(100.0, score)))

    @staticmethod
    def _build_verdict(
        kalman: dict,
        cusum: dict,
        ks: dict,
        convergence: dict,
        score: float,
    ) -> str:
        """Human-readable verdict string."""
        n_cp = cusum["n_changepoints"]
        trend = kalman["edge_trend"]
        pct_pos = kalman["pct_time_positive"]

        parts: list[str] = []

        if n_cp == 0:
            parts.append("No structural breaks detected.")
        elif n_cp == 1:
            parts.append(f"1 regime change detected at trade {cusum['changepoints'][0]}.")
        else:
            parts.append(f"{n_cp} regime changes detected.")

        parts.append(f"Edge trend: {trend}.")

        if convergence["is_drifting"]:
            parts.append(
                f"Significant parameter drift (slope={convergence['drift_slope']:.6f})."
            )
        else:
            parts.append("No significant parameter drift.")

        if convergence["is_converging"]:
            parts.append("Cumulative mean is converging.")
        else:
            parts.append(
                f"Cumulative mean NOT converging "
                f"(ratio={convergence['convergence_ratio']:.2f})."
            )

        parts.append(f"Edge positive {pct_pos:.0%} of the time.")

        return " ".join(parts)
