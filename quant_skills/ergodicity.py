"""
Ergodicity Analyzer — time-average vs ensemble-average growth diagnostics.

Most trading metrics assume ergodicity: that an ensemble average (expected
value across many parallel bets) equals the time average (what a single
account experiences sequentially).  Leveraged, compounded returns violate
this assumption.  This module quantifies the gap and derives safe position
sizing:

1. **Geometric vs arithmetic growth** — computes the log-growth curve
   g(f) = E[ln(1 + f*R)] over a grid of risk fractions, identifies the
   optimal Kelly leverage and the critical fraction where growth turns
   negative.
2. **Kelly with uncertainty** — bootstraps the Kelly fraction to produce
   confidence intervals, since point-estimate Kelly is notoriously
   overfit on small samples.
3. **Ruin probability** — Monte Carlo simulation of sequential capital
   trajectories to estimate the probability of drawdown below a ruin
   threshold.

All public methods are stateless ``@staticmethod`` callables that return
plain JSON-serialisable dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .base import TradeData, SkillResult, _grade_from_score


class ErgodicityAnalyzer:
    """Time-average growth, Kelly sizing, and ruin probability diagnostics."""

    MIN_TRADES = 30

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(
        data: Any,
        initial_capital: float = 10_000.0,
        risk_per_trade_pct: float = 1.0,
        n_bootstrap: int = 5_000,
    ) -> dict:
        """Run all ergodicity analyses and return a scored result dict.

        Parameters
        ----------
        data : TradeData | list[dict] | ndarray
            Trade data in any format accepted by ``TradeData.from_auto``.
        initial_capital : float
            Starting equity for ruin simulations (default 10 000).
        risk_per_trade_pct : float
            Percentage of capital risked per trade (default 1%).
        n_bootstrap : int
            Bootstrap iterations for Kelly CI (default 5 000).

        Returns
        -------
        dict
            ``SkillResult.to_dict()`` with metrics, diagnostics, score
            and grade.
        """
        td = TradeData.from_auto(data)

        if td.n < ErgodicityAnalyzer.MIN_TRADES:
            return SkillResult(
                skill_name="ergodicity",
                valid=False,
                reason=(
                    f"Insufficient trades: {td.n} < "
                    f"{ErgodicityAnalyzer.MIN_TRADES} minimum"
                ),
            ).to_dict()

        r = td.r_multiples
        risk_frac = risk_per_trade_pct / 100.0

        # --- Sub-analyses ---
        growth = ErgodicityAnalyzer.geometric_vs_arithmetic(r)
        kelly = ErgodicityAnalyzer.kelly_with_uncertainty(
            r, n_bootstrap=n_bootstrap
        )
        ruin = ErgodicityAnalyzer.ruin_probability(
            r,
            risk_fraction=risk_frac,
            initial_capital=initial_capital,
        )

        # --- Scoring ---
        score = ErgodicityAnalyzer._compute_score(growth, kelly, ruin)
        grade = _grade_from_score(score)
        verdict = ErgodicityAnalyzer._build_verdict(growth, kelly, ruin, score)

        return SkillResult(
            skill_name="ergodicity",
            valid=True,
            score=score,
            grade=grade,
            verdict=verdict,
            metrics={
                "arithmetic_mean": growth["arithmetic_mean"],
                "is_ergodic": growth["is_ergodic"],
                "optimal_f": growth["optimal_f"],
                "max_growth_rate": growth["max_growth_rate"],
                "critical_f": growth["critical_f"],
                "kelly_point_estimate": kelly["point_estimate"],
                "kelly_ci_95": kelly["ci_95"],
                "kelly_conservative": kelly["conservative"],
                "kelly_recommended": kelly["recommended"],
                "pct_positive_bootstraps": kelly["pct_positive_bootstraps"],
                "p_ruin": ruin["p_ruin"],
                "median_final_capital": ruin["median_final_capital"],
            },
            diagnostics={
                "geometric_vs_arithmetic": growth,
                "kelly_with_uncertainty": kelly,
                "ruin_probability": ruin,
            },
        ).to_dict()

    # ------------------------------------------------------------------
    #  Geometric vs arithmetic growth
    # ------------------------------------------------------------------

    @staticmethod
    def geometric_vs_arithmetic(
        r_multiples: np.ndarray,
        risk_fractions: Optional[np.ndarray] = None,
    ) -> dict:
        """Compare time-average (geometric) to ensemble-average (arithmetic) growth.

        For each risk fraction *f*, the expected log-growth rate is:
            g(f) = mean( ln(1 + f * r_i) )

        The strategy is ergodic if and only if g(f) > 0 for some viable f.
        The optimal f (full Kelly) maximises g(f).

        Parameters
        ----------
        r_multiples : ndarray
            1-D array of R-multiples.
        risk_fractions : ndarray | None
            Grid of risk fractions to evaluate.  Defaults to
            ``np.linspace(0.01, 0.50, 50)``.

        Returns
        -------
        dict
            Arithmetic mean, growth curve, optimal f, max growth rate,
            whether strategy is ergodic, and the critical f.
        """
        r = np.asarray(r_multiples, dtype=np.float64)
        arithmetic_mean = float(np.mean(r))

        if risk_fractions is None:
            fractions = np.linspace(0.01, 0.50, 50)
        else:
            fractions = np.asarray(risk_fractions, dtype=np.float64)

        growth_curve: list[dict] = []
        best_g = -np.inf
        optimal_f = 0.0
        critical_f: Optional[float] = None
        prev_positive = False

        for f in fractions:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_returns = np.log1p(f * r)
            # Guard against -inf from ln(0) when 1 + f*r <= 0
            finite_mask = np.isfinite(log_returns)
            if not np.all(finite_mask):
                # If any trade would cause total ruin at this f, g = -inf
                g = -np.inf
            else:
                g = float(np.mean(log_returns))

            growth_curve.append({
                "f": round(float(f), 4),
                "g_f": round(g, 8) if np.isfinite(g) else None,
            })

            if np.isfinite(g) and g > best_g:
                best_g = g
                optimal_f = float(f)

            # Track where growth transitions from positive to negative
            is_positive = np.isfinite(g) and g > 0
            if prev_positive and not is_positive and critical_f is None:
                critical_f = round(float(f), 4)
            prev_positive = is_positive

        is_ergodic = bool(best_g > 0)

        return {
            "arithmetic_mean": round(arithmetic_mean, 6),
            "growth_curve": growth_curve,
            "optimal_f": round(optimal_f, 4),
            "max_growth_rate": round(best_g, 8) if np.isfinite(best_g) else None,
            "is_ergodic": is_ergodic,
            "critical_f": critical_f,
        }

    # ------------------------------------------------------------------
    #  Kelly with uncertainty
    # ------------------------------------------------------------------

    @staticmethod
    def kelly_with_uncertainty(
        r_multiples: np.ndarray,
        n_bootstrap: int = 5_000,
    ) -> dict:
        """Bootstrap the Kelly fraction to produce confidence intervals.

        For each bootstrap resample, computes:
            wins  = sample[sample > 0]
            losses = sample[sample < 0]
            wr    = len(wins) / len(sample)
            kelly = wr - (1 - wr) / (mean(wins) / mean(|losses|))

        Parameters
        ----------
        r_multiples : ndarray
            1-D array of R-multiples.
        n_bootstrap : int
            Number of bootstrap iterations (default 5 000).

        Returns
        -------
        dict
            Point estimate, 95% and 80% confidence intervals,
            conservative estimate (5th percentile), fractional Kelly
            recommendations, and fraction of positive bootstrap draws.
        """
        r = np.asarray(r_multiples, dtype=np.float64)
        n = len(r)

        def _kelly_single(sample: np.ndarray) -> float:
            wins = sample[sample > 0]
            losses = sample[sample < 0]
            if len(wins) == 0 or len(losses) == 0:
                return 0.0
            wr = len(wins) / len(sample)
            avg_win = float(np.mean(wins))
            avg_loss = float(np.mean(np.abs(losses)))
            if avg_loss < 1e-12:
                return 0.0
            win_loss_ratio = avg_win / avg_loss
            return wr - (1.0 - wr) / win_loss_ratio

        point_estimate = _kelly_single(r)

        rng = np.random.default_rng()
        kelly_samples = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            boot = rng.choice(r, size=n, replace=True)
            kelly_samples[i] = _kelly_single(boot)

        ci_95 = [
            round(float(np.percentile(kelly_samples, 2.5)), 6),
            round(float(np.percentile(kelly_samples, 97.5)), 6),
        ]
        ci_80 = [
            round(float(np.percentile(kelly_samples, 10)), 6),
            round(float(np.percentile(kelly_samples, 90)), 6),
        ]
        conservative = float(np.percentile(kelly_samples, 5))
        half_kelly = round(point_estimate / 2.0, 6)
        quarter_kelly = round(point_estimate / 4.0, 6)
        recommended = round(max(0.0, conservative / 4.0), 6)
        pct_positive = float(np.mean(kelly_samples > 0))

        return {
            "point_estimate": round(point_estimate, 6),
            "ci_95": ci_95,
            "ci_80": ci_80,
            "conservative": round(conservative, 6),
            "half_kelly": half_kelly,
            "quarter_kelly": quarter_kelly,
            "recommended": recommended,
            "pct_positive_bootstraps": round(pct_positive, 4),
        }

    # ------------------------------------------------------------------
    #  Ruin probability
    # ------------------------------------------------------------------

    @staticmethod
    def ruin_probability(
        r_multiples: np.ndarray,
        risk_fraction: float = 0.01,
        initial_capital: float = 10_000.0,
        ruin_pct: float = 10.0,
        max_trades: int = 1_000,
        n_simulations: int = 10_000,
    ) -> dict:
        """Monte Carlo ruin estimation via sequential capital simulation.

        For each simulation path, repeatedly draw a random R-multiple
        (with replacement) and compound the capital:
            capital *= (1 + risk_fraction * r)
        Ruin occurs when capital drops below ``initial_capital * ruin_pct / 100``.

        Parameters
        ----------
        r_multiples : ndarray
            Historical R-multiples.
        risk_fraction : float
            Fraction of capital risked per trade (default 0.01 = 1%).
        initial_capital : float
            Starting equity.
        ruin_pct : float
            Ruin threshold as % of initial capital (default 10%).
        max_trades : int
            Horizon in number of trades (default 1 000).
        n_simulations : int
            Number of Monte Carlo paths (default 10 000).

        Returns
        -------
        dict
            Ruin probabilities at several horizons, median trades to
            ruin among ruined paths, and final capital distribution.
        """
        r = np.asarray(r_multiples, dtype=np.float64)
        ruin_level = initial_capital * ruin_pct / 100.0

        rng = np.random.default_rng()

        # Checkpoints for partial-horizon ruin probability
        checkpoints = [t for t in [100, 250, 500] if t <= max_trades]

        ruined = np.zeros(n_simulations, dtype=bool)
        trades_to_ruin = np.full(n_simulations, np.nan, dtype=np.float64)
        final_capitals = np.empty(n_simulations, dtype=np.float64)
        # Track ruin at each checkpoint
        ruined_at_checkpoint: dict[int, int] = {cp: 0 for cp in checkpoints}

        for sim in range(n_simulations):
            capital = initial_capital
            sim_ruined = False
            random_indices = rng.integers(0, len(r), size=max_trades)

            for t in range(max_trades):
                capital *= (1.0 + risk_fraction * r[random_indices[t]])

                if not sim_ruined and capital < ruin_level:
                    sim_ruined = True
                    ruined[sim] = True
                    trades_to_ruin[sim] = t + 1

                # Record checkpoint ruin status
                if (t + 1) in ruined_at_checkpoint and sim_ruined:
                    ruined_at_checkpoint[t + 1] += 1

            final_capitals[sim] = capital

        p_ruin = float(np.mean(ruined))

        # Ruin at checkpoints
        p_ruin_at: dict[str, float] = {}
        for cp in checkpoints:
            p_ruin_at[str(cp)] = round(
                float(ruined_at_checkpoint[cp] / n_simulations), 6
            )

        # Median trades to ruin (among ruined paths)
        ruined_mask = ~np.isnan(trades_to_ruin)
        if np.any(ruined_mask):
            median_ttr = float(np.median(trades_to_ruin[ruined_mask]))
        else:
            median_ttr = None  # type: ignore[assignment]

        return {
            "p_ruin": round(p_ruin, 6),
            "p_ruin_at_trades": p_ruin_at,
            "median_trades_to_ruin": (
                round(median_ttr, 1) if median_ttr is not None else None
            ),
            "median_final_capital": round(float(np.median(final_capitals)), 2),
            "p5_final_capital": round(
                float(np.percentile(final_capitals, 5)), 2
            ),
            "p95_final_capital": round(
                float(np.percentile(final_capitals, 95)), 2
            ),
            "risk_fraction": round(risk_fraction, 6),
            "initial_capital": round(initial_capital, 2),
            "ruin_threshold_pct": round(ruin_pct, 2),
            "max_trades": max_trades,
            "n_simulations": n_simulations,
        }

    # ------------------------------------------------------------------
    #  Internal scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(growth: dict, kelly: dict, ruin: dict) -> float:
        """Map sub-analysis results to a 0-100 score.

        Scoring rubric
        --------------
        * Positive Kelly CI, g > 0, P(ruin) < 5%:  80+
        * Kelly CI includes 0, low ruin:            60
        * Kelly near 0, moderate ruin:              40
        * Kelly negative, high ruin:                20
        * Certain ruin, deeply negative Kelly:      <10
        """
        kelly_pt = kelly["point_estimate"]
        kelly_lo = kelly["ci_95"][0]
        pct_pos_boot = kelly["pct_positive_bootstraps"]
        is_ergodic = growth["is_ergodic"]
        max_g = growth["max_growth_rate"]
        p_ruin = ruin["p_ruin"]

        score = 50.0

        # --- Ergodicity / growth rate ---
        if is_ergodic and max_g is not None and max_g > 0:
            score += 15.0
            # Extra credit for strong growth
            if max_g > 0.005:
                score += 5.0
        else:
            score -= 20.0

        # --- Kelly fraction ---
        if kelly_lo > 0:
            # Entire 95% CI is positive
            score += 15.0
        elif kelly_pt > 0 and pct_pos_boot > 0.8:
            # Mostly positive but CI crosses zero
            score += 5.0
        elif kelly_pt <= 0:
            score -= 15.0
            if kelly_pt < -0.1:
                score -= 10.0

        # --- Ruin probability ---
        if p_ruin < 0.01:
            score += 10.0
        elif p_ruin < 0.05:
            score += 5.0
        elif p_ruin < 0.15:
            pass  # neutral
        elif p_ruin < 0.30:
            score -= 10.0
        elif p_ruin < 0.50:
            score -= 20.0
        else:
            score -= 30.0

        # --- Bootstrap confidence ---
        if pct_pos_boot > 0.95:
            score += 5.0
        elif pct_pos_boot < 0.5:
            score -= 10.0

        return float(max(0.0, min(100.0, score)))

    @staticmethod
    def _build_verdict(
        growth: dict,
        kelly: dict,
        ruin: dict,
        score: float,
    ) -> str:
        """Human-readable verdict string."""
        parts: list[str] = []

        if growth["is_ergodic"]:
            parts.append(
                f"Strategy IS ergodic (max g={growth['max_growth_rate']:.6f} "
                f"at f*={growth['optimal_f']:.2%})."
            )
        else:
            parts.append(
                "Strategy is NOT ergodic: no risk fraction yields positive "
                "time-average growth."
            )

        kelly_pt = kelly["point_estimate"]
        ci = kelly["ci_95"]
        parts.append(
            f"Kelly fraction: {kelly_pt:.3f} "
            f"(95% CI [{ci[0]:.3f}, {ci[1]:.3f}])."
        )
        parts.append(
            f"Recommended fractional Kelly: {kelly['recommended']:.4f}."
        )

        p_ruin = ruin["p_ruin"]
        if p_ruin < 0.01:
            parts.append(f"Ruin risk negligible (P={p_ruin:.4f}).")
        elif p_ruin < 0.05:
            parts.append(f"Ruin risk low (P={p_ruin:.4f}).")
        elif p_ruin < 0.15:
            parts.append(f"Ruin risk moderate (P={p_ruin:.4f}).")
        else:
            parts.append(f"Ruin risk HIGH (P={p_ruin:.4f}). Reduce sizing.")

        return " ".join(parts)
