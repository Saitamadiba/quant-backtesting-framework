"""
Signal-decay analysis skill for quantitative strategy auditing.

Determines whether the trading signal degrades over time, identifies the
optimal holding period, and checks for serial dependence in returns.  A
decaying signal means the entry criterion loses value the longer a position
is held -- critical information for trade-management rules.

Analyses included:
    - MFE / MAE time profiles  (excursion efficiency by holding period)
    - Optimal holding period    (which bin of bars_held maximises R)
    - Signal autocorrelation    (ACF + Ljung-Box independence test)
    - Edge half-life            (rolling-mean OU estimation of reversion speed)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import TradeData, SkillResult, _grade_from_score


class SignalDecayAnalyzer:
    """Stateless analyser for signal-decay properties of trade returns."""

    MIN_TRADES = 30

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(data: Any) -> Dict[str, Any]:
        """Run the full signal-decay analysis suite.

        Parameters
        ----------
        data : array-like, list[dict], or TradeData
            Trade data in any format accepted by ``TradeData.from_auto``.

        Returns
        -------
        dict
            JSON-serialisable result following the SkillResult schema.
        """
        td = TradeData.from_auto(data)

        if td.n < SignalDecayAnalyzer.MIN_TRADES:
            return SkillResult(
                skill_name="signal_decay",
                valid=False,
                reason=f"Insufficient trades: {td.n} < {SignalDecayAnalyzer.MIN_TRADES}",
            ).to_dict()

        r = td.r_multiples

        # --- Sub-analyses ------------------------------------------------
        mfe_mae_result: Dict[str, Any] = {}
        if (
            td.mfe is not None
            and td.mae is not None
            and td.bars_held is not None
            and np.any(td.bars_held > 0)
        ):
            mfe_mae_result = SignalDecayAnalyzer.mfe_mae_time_profiles(
                r, td.mfe, td.mae, td.bars_held,
            )

        holding_result: Dict[str, Any] = {}
        if td.bars_held is not None and np.any(td.bars_held > 0):
            holding_result = SignalDecayAnalyzer.optimal_holding_period(r, td.bars_held)

        acf_result = SignalDecayAnalyzer.signal_autocorrelation(r)

        half_life_result = SignalDecayAnalyzer.edge_half_life(r)

        # --- Scoring -----------------------------------------------------
        score = SignalDecayAnalyzer._compute_score(
            mfe_mae_result, holding_result, acf_result, half_life_result,
        )
        grade = _grade_from_score(score)
        verdict = SignalDecayAnalyzer._build_verdict(
            score, holding_result, acf_result, half_life_result,
        )

        return SkillResult(
            skill_name="signal_decay",
            valid=True,
            score=float(score),
            grade=grade,
            verdict=verdict,
            metrics={
                "mfe_mae_profiles": mfe_mae_result,
                "optimal_holding": holding_result,
                "signal_autocorrelation": acf_result,
                "edge_half_life": half_life_result,
            },
            diagnostics={
                "n_trades": int(td.n),
                "has_mfe_mae": bool(mfe_mae_result),
                "has_bars_held": bool(holding_result),
            },
        ).to_dict()

    # ------------------------------------------------------------------
    #  MFE / MAE time profiles
    # ------------------------------------------------------------------

    @staticmethod
    def mfe_mae_time_profiles(
        r_multiples: np.ndarray,
        mfe: np.ndarray,
        mae: np.ndarray,
        bars_held: np.ndarray,
    ) -> Dict[str, Any]:
        """Bin trades by holding period and compute MFE/MAE profiles.

        Trades are sorted into ~6-8 quantile-based bins by ``bars_held``.
        For each bin the mean MFE, mean MAE, mean R, and the MFE/MAE ratio
        are calculated.

        Returns
        -------
        dict with bins (list of bin dicts) and optimal_mfe_mae_ratio_bin.
        """
        n = len(r_multiples)
        if n == 0:
            return {"bins": [], "optimal_mfe_mae_ratio_bin": None}

        # Determine bin edges using quantiles (target 6-8 bins)
        n_bins = min(8, max(3, n // 10))
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(bars_held, quantiles))

        # If too few unique edges, fall back to equal-width
        if len(edges) < 3:
            edges = np.linspace(float(np.min(bars_held)), float(np.max(bars_held)) + 1, n_bins + 1)

        bins_list: List[Dict[str, Any]] = []
        best_ratio = -np.inf
        best_bin_label: str | None = None

        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i < len(edges) - 2:
                mask = (bars_held >= lo) & (bars_held < hi)
            else:
                mask = (bars_held >= lo) & (bars_held <= hi)

            subset_r = r_multiples[mask]
            subset_mfe = mfe[mask]
            subset_mae = mae[mask]
            n_sub = int(np.sum(mask))

            if n_sub == 0:
                continue

            mean_mfe = float(np.mean(subset_mfe))
            mean_mae = float(np.mean(np.abs(subset_mae)))
            mean_r = float(np.mean(subset_r))
            ratio = mean_mfe / max(mean_mae, 0.01)

            bin_label = f"{lo:.0f}-{hi:.0f}"
            bins_list.append({
                "bin_label": bin_label,
                "bars_lo": round(float(lo), 1),
                "bars_hi": round(float(hi), 1),
                "n": n_sub,
                "mean_mfe": round(mean_mfe, 4),
                "mean_mae": round(mean_mae, 4),
                "mean_r": round(mean_r, 4),
                "mfe_mae_ratio": round(float(ratio), 4),
            })

            if ratio > best_ratio:
                best_ratio = ratio
                best_bin_label = bin_label

        return {
            "bins": bins_list,
            "optimal_mfe_mae_ratio_bin": best_bin_label,
        }

    # ------------------------------------------------------------------
    #  Optimal holding period
    # ------------------------------------------------------------------

    @staticmethod
    def optimal_holding_period(
        r_multiples: np.ndarray,
        bars_held: np.ndarray,
    ) -> Dict[str, Any]:
        """Find the holding-period bin with the best mean R.

        Returns
        -------
        dict with bins, optimal_bin, optimal_mean_r, is_any_bin_profitable.
        """
        n = len(r_multiples)
        if n == 0:
            return {
                "bins": [],
                "optimal_bin": None,
                "optimal_mean_r": None,
                "is_any_bin_profitable": False,
            }

        n_bins = min(8, max(3, n // 10))
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(bars_held, quantiles))
        if len(edges) < 3:
            edges = np.linspace(float(np.min(bars_held)), float(np.max(bars_held)) + 1, n_bins + 1)

        bins_list: List[Dict[str, Any]] = []
        best_mean = -np.inf
        best_label: str | None = None
        any_profitable = False

        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i < len(edges) - 2:
                mask = (bars_held >= lo) & (bars_held < hi)
            else:
                mask = (bars_held >= lo) & (bars_held <= hi)

            subset_r = r_multiples[mask]
            n_sub = int(np.sum(mask))
            if n_sub == 0:
                continue

            mean_r = float(np.mean(subset_r))
            win_rate = float(np.mean(subset_r > 0))

            bin_label = f"{lo:.0f}-{hi:.0f}"
            bins_list.append({
                "bin_label": bin_label,
                "bars_lo": round(float(lo), 1),
                "bars_hi": round(float(hi), 1),
                "n": n_sub,
                "mean_r": round(mean_r, 4),
                "win_rate": round(win_rate, 4),
            })

            if mean_r > 0:
                any_profitable = True
            if mean_r > best_mean:
                best_mean = mean_r
                best_label = bin_label

        return {
            "bins": bins_list,
            "optimal_bin": best_label,
            "optimal_mean_r": round(float(best_mean), 4) if np.isfinite(best_mean) else None,
            "is_any_bin_profitable": any_profitable,
        }

    # ------------------------------------------------------------------
    #  Signal autocorrelation
    # ------------------------------------------------------------------

    @staticmethod
    def signal_autocorrelation(
        r_multiples: np.ndarray, max_lag: int = 20
    ) -> Dict[str, Any]:
        """ACF at lags 1..max_lag with Ljung-Box Q-test for independence.

        Formulae
        --------
        rho_k = corrcoef(r[:-k], r[k:])  (sample autocorrelation at lag k)
        bartlett_ci = 1.96 / sqrt(n)
        Q(m) = n(n+2) * sum( rho_k^2 / (n-k) ) for k=1..m,  Q ~ chi2(m)

        Returns
        -------
        dict with lag_correlations, ljung_box_stat, ljung_box_p,
        trades_independent.
        """
        from scipy.stats import chi2 as _chi2  # lazy import

        n = len(r_multiples)
        max_lag = min(max_lag, n // 3)  # ensure enough data
        max_lag = max(max_lag, 1)

        bartlett_ci = 1.96 / np.sqrt(n) if n > 0 else 0.0

        # Compute ACF
        r_centered = r_multiples - np.mean(r_multiples)
        var_r = float(np.sum(r_centered ** 2))

        lag_corrs: List[Dict[str, Any]] = []
        rho_values: List[float] = []

        for lag in range(1, max_lag + 1):
            if lag >= n:
                break
            if var_r < 1e-15:
                rho = 0.0
            else:
                rho = float(np.sum(r_centered[:n - lag] * r_centered[lag:]) / var_r)

            rho_values.append(rho)
            lag_corrs.append({
                "lag": int(lag),
                "rho": round(rho, 4),
                "significant": bool(abs(rho) > bartlett_ci),
            })

        # Ljung-Box Q statistic
        m = len(rho_values)
        lb_stat = 0.0
        for k_idx, rho_k in enumerate(rho_values):
            k_val = k_idx + 1
            if n - k_val > 0:
                lb_stat += (rho_k ** 2) / (n - k_val)
        lb_stat *= n * (n + 2)
        lb_p = float(1.0 - _chi2.cdf(lb_stat, df=max(m, 1)))

        return {
            "lag_correlations": lag_corrs,
            "bartlett_ci": round(float(bartlett_ci), 4),
            "ljung_box_stat": round(float(lb_stat), 4),
            "ljung_box_p": round(float(lb_p), 6),
            "trades_independent": bool(lb_p > 0.05),
            "n_lags_tested": int(m),
        }

    # ------------------------------------------------------------------
    #  Edge half-life
    # ------------------------------------------------------------------

    @staticmethod
    def edge_half_life(
        r_multiples: np.ndarray, window_size: int = 50
    ) -> Dict[str, Any]:
        """Rolling edge with Ornstein-Uhlenbeck half-life estimation.

        Computes a rolling mean of R-multiples, then fits an OLS regression:

            delta_edge[t] = a + b * edge[t-1] + epsilon

        If b < 0 the edge mean-reverts with:
            theta     = -b
            half_life = ln(2) / theta

        Returns
        -------
        dict with rolling_edge, half_life_trades, decay_rate, model_r_squared,
        is_mean_reverting.
        """
        n = len(r_multiples)
        window = min(window_size, n // 2)
        window = max(window, 5)

        # Rolling mean edge
        rolling_edge: List[float] = []
        for i in range(n):
            start = max(0, i - window + 1)
            rolling_edge.append(float(np.mean(r_multiples[start:i + 1])))

        rolling_arr = np.array(rolling_edge)

        # Need at least 10 points for meaningful regression
        if n < 10:
            return {
                "rolling_edge": [round(float(x), 4) for x in rolling_edge],
                "half_life_trades": None,
                "decay_rate": None,
                "model_r_squared": None,
                "is_mean_reverting": False,
            }

        # OLS: delta_edge = a + b * edge_prev
        edge_prev = rolling_arr[:-1]
        delta_edge = np.diff(rolling_arr)

        # X = [ones, edge_prev], y = delta_edge
        X = np.column_stack([np.ones(len(edge_prev)), edge_prev])
        try:
            # Solve via normal equations: beta = (X'X)^-1 X'y
            XtX = X.T @ X
            Xty = X.T @ delta_edge
            beta = np.linalg.solve(XtX, Xty)
            a, b = float(beta[0]), float(beta[1])

            # R-squared
            y_hat = X @ beta
            ss_res = float(np.sum((delta_edge - y_hat) ** 2))
            ss_tot = float(np.sum((delta_edge - np.mean(delta_edge)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
            r_squared = max(0.0, min(1.0, r_squared))

            if b < 0:
                theta = -b
                half_life = float(np.log(2) / theta) if theta > 1e-12 else float("inf")
                is_mr = True
            else:
                theta = 0.0
                half_life = float("inf")
                is_mr = False

        except np.linalg.LinAlgError:
            a, b, theta = 0.0, 0.0, 0.0
            half_life = float("inf")
            r_squared = 0.0
            is_mr = False

        return {
            "rolling_edge": [round(float(x), 4) for x in rolling_edge],
            "half_life_trades": round(half_life, 1) if np.isfinite(half_life) else None,
            "decay_rate": round(float(theta), 6),
            "model_r_squared": round(float(r_squared), 4),
            "is_mean_reverting": is_mr,
        }

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(
        mfe_mae: Dict[str, Any],
        holding: Dict[str, Any],
        acf: Dict[str, Any],
        half_life: Dict[str, Any],
    ) -> float:
        """Derive a 0-100 score from signal-decay sub-analyses.

        Scoring guide:
            80+   Stable edge, optimal holding positive, low autocorrelation
            60    Some bins positive, slow decay
            40    All bins negative but shallow
            20    All bins deeply negative, fast decay
            <15   No signal at any horizon
        """
        score = 50.0  # neutral starting point

        # --- Optimal holding period ---
        if holding:
            if holding.get("is_any_bin_profitable"):
                opt_r = holding.get("optimal_mean_r")
                if opt_r is not None:
                    if opt_r > 0.3:
                        score += 25.0
                    elif opt_r > 0.1:
                        score += 15.0
                    elif opt_r > 0:
                        score += 10.0
                # Bonus if multiple bins profitable
                bins_data = holding.get("bins", [])
                n_prof = sum(1 for b in bins_data if b.get("mean_r", 0) > 0)
                n_total = len(bins_data)
                if n_total > 0 and n_prof / n_total > 0.5:
                    score += 5.0
            else:
                # No bin profitable
                opt_r = holding.get("optimal_mean_r")
                if opt_r is not None and opt_r > -0.05:
                    score -= 10.0
                elif opt_r is not None and opt_r > -0.2:
                    score -= 20.0
                else:
                    score -= 30.0

        # --- MFE/MAE profiles ---
        if mfe_mae:
            bins_data = mfe_mae.get("bins", [])
            ratios = [b.get("mfe_mae_ratio", 1.0) for b in bins_data if b.get("n", 0) > 0]
            if ratios:
                best_ratio = max(ratios)
                if best_ratio > 2.0:
                    score += 5.0
                elif best_ratio < 0.8:
                    score -= 5.0

        # --- Autocorrelation ---
        if acf:
            if acf.get("trades_independent"):
                score += 5.0
            else:
                # Serial dependence detected
                sig_lags = sum(
                    1 for lc in acf.get("lag_correlations", [])
                    if lc.get("significant")
                )
                if sig_lags > 3:
                    score -= 10.0
                else:
                    score -= 5.0

        # --- Half-life ---
        if half_life:
            hl = half_life.get("half_life_trades")
            if half_life.get("is_mean_reverting") and hl is not None:
                if hl < 20:
                    score -= 10.0  # fast decay is bad
                elif hl < 50:
                    score -= 5.0
                # Slow mean-reversion is neutral (edge persists for a while)

        return float(max(0.0, min(100.0, score)))

    @staticmethod
    def _build_verdict(
        score: float,
        holding: Dict[str, Any],
        acf: Dict[str, Any],
        half_life: Dict[str, Any],
    ) -> str:
        """One-liner human-readable verdict."""
        if score >= 80:
            return (
                "Signal is stable with positive edge across multiple holding "
                "periods and no significant serial dependence."
            )
        if score >= 60:
            parts = ["Signal shows moderate stability."]
            if holding and holding.get("optimal_bin"):
                parts.append(
                    f"Best holding period is {holding['optimal_bin']} bars "
                    f"(mean R = {holding.get('optimal_mean_r', '?')})."
                )
            return " ".join(parts)
        if score >= 40:
            if holding and not holding.get("is_any_bin_profitable"):
                return (
                    "No holding-period bin is profitable on average.  "
                    "Signal may be too weak or entry timing is poor."
                )
            return (
                "Signal shows some decay.  Edge is marginal and "
                "concentrated in specific holding periods."
            )
        if score >= 20:
            hl = half_life.get("half_life_trades") if half_life else None
            hl_str = f" (half-life ~{hl:.0f} trades)" if hl and np.isfinite(hl) else ""
            return (
                f"Signal is decaying{hl_str}.  Negative expectancy at most "
                "holding horizons suggests fundamental weakness."
            )
        return (
            "No detectable signal at any horizon.  Returns are "
            "indistinguishable from noise across all analyses."
        )
