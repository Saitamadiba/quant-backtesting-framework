"""
Strategy Auditor — orchestrates all quant skills into a single tradability verdict.

Runs EdgeExistenceValidator, StationarityAnalyzer, ErgodicityAnalyzer,
TailRiskAnalyzer, RegimeEdgeDecomposer, and SignalDecayAnalyzer on trade data,
then produces a weighted tradability score (0-100), grade (A-F), and
natural-language assessment.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import TradeData, SkillResult, _grade_from_score
from .edge_existence import EdgeExistenceValidator
from .stationarity import StationarityAnalyzer
from .ergodicity import ErgodicityAnalyzer
from .tail_risk import TailRiskAnalyzer
from .regime_edge import RegimeEdgeDecomposer
from .signal_decay import SignalDecayAnalyzer

logger = logging.getLogger(__name__)

# Skill weight allocation for overall tradability score
SKILL_WEIGHTS: Dict[str, float] = {
    "edge_existence": 0.25,
    "ergodicity": 0.20,
    "stationarity": 0.15,
    "regime_edge": 0.15,
    "signal_decay": 0.15,
    "tail_risk": 0.10,
}

GRADE_VERDICTS = {
    "A": "TRADABLE: Strategy demonstrates robust, stationary edge with viable sizing.",
    "B": "CONDITIONAL: Strategy shows edge with caveats — regime-specific or sizing-sensitive.",
    "C": "MARGINAL: Weak or unstable edge — significant improvements needed before live trading.",
    "D": "DEFICIENT: Edge absent or dominated by noise — not ready for capital deployment.",
    "F": "UNTRADABLE: No statistical evidence of edge — strategy should not be traded.",
}


class StrategyAuditor:
    """Orchestrate all quant skills to produce a comprehensive strategy audit."""

    @staticmethod
    def audit(
        data: Any,
        strategy_name: str = "Unknown",
        symbol: str = "",
        timeframe: str = "",
        initial_capital: float = 10_000,
        risk_per_trade_pct: float = 1.0,
        comparisons: int = 1,
    ) -> Dict[str, Any]:
        """Run full strategy audit.

        Parameters
        ----------
        data : list[dict] | np.ndarray | TradeData
            Trade data in any supported format.
        strategy_name, symbol, timeframe : str
            Metadata labels.
        initial_capital : float
            Starting capital for ruin analysis.
        risk_per_trade_pct : float
            Percent of capital risked per trade.
        comparisons : int
            Number of parallel tests (for multi-comparison correction).

        Returns
        -------
        dict
            Full audit report with per-skill results, overall score, and verdict.
        """
        td = TradeData.from_auto(data)

        skills: Dict[str, Dict] = {}

        # 1 — Edge Existence
        logger.info("Running EdgeExistenceValidator ...")
        try:
            skills["edge_existence"] = EdgeExistenceValidator.analyze(
                td, n_permutations=10_000, comparisons=comparisons
            )
        except Exception as exc:
            logger.warning("EdgeExistenceValidator failed: %s", exc)
            skills["edge_existence"] = {"valid": False, "reason": str(exc), "score": 0, "grade": "F"}

        # 2 — Stationarity
        logger.info("Running StationarityAnalyzer ...")
        try:
            skills["stationarity"] = StationarityAnalyzer.analyze(td, window_size=50)
        except Exception as exc:
            logger.warning("StationarityAnalyzer failed: %s", exc)
            skills["stationarity"] = {"valid": False, "reason": str(exc), "score": 0, "grade": "F"}

        # 3 — Ergodicity
        logger.info("Running ErgodicityAnalyzer ...")
        try:
            skills["ergodicity"] = ErgodicityAnalyzer.analyze(
                td,
                initial_capital=initial_capital,
                risk_per_trade_pct=risk_per_trade_pct,
            )
        except Exception as exc:
            logger.warning("ErgodicityAnalyzer failed: %s", exc)
            skills["ergodicity"] = {"valid": False, "reason": str(exc), "score": 0, "grade": "F"}

        # 4 — Tail Risk
        logger.info("Running TailRiskAnalyzer ...")
        try:
            skills["tail_risk"] = TailRiskAnalyzer.analyze(td)
        except Exception as exc:
            logger.warning("TailRiskAnalyzer failed: %s", exc)
            skills["tail_risk"] = {"valid": False, "reason": str(exc), "score": 0, "grade": "F"}

        # 5 — Regime Edge
        logger.info("Running RegimeEdgeDecomposer ...")
        try:
            skills["regime_edge"] = RegimeEdgeDecomposer.analyze(td, n_regimes=2)
        except Exception as exc:
            logger.warning("RegimeEdgeDecomposer failed: %s", exc)
            skills["regime_edge"] = {"valid": False, "reason": str(exc), "score": 0, "grade": "F"}

        # 6 — Signal Decay
        logger.info("Running SignalDecayAnalyzer ...")
        try:
            skills["signal_decay"] = SignalDecayAnalyzer.analyze(td)
        except Exception as exc:
            logger.warning("SignalDecayAnalyzer failed: %s", exc)
            skills["signal_decay"] = {"valid": False, "reason": str(exc), "score": 0, "grade": "F"}

        # --- Aggregate ---
        weighted_score = 0.0
        for key, weight in SKILL_WEIGHTS.items():
            s = skills.get(key, {}).get("score", 0)
            weighted_score += weight * s

        overall_score = round(max(0, min(100, weighted_score)), 1)
        overall_grade = _grade_from_score(overall_score)

        # Summary table
        summary_table = []
        for key in SKILL_WEIGHTS:
            sk = skills.get(key, {})
            summary_table.append({
                "skill": sk.get("skill_name", key),
                "score": sk.get("score", 0),
                "grade": sk.get("grade", "F"),
                "verdict": sk.get("verdict", ""),
            })

        # Improvement suggestions
        suggestions = _generate_suggestions(skills, td)

        verdict = (
            f"{overall_grade}-rated ({overall_score}/100): "
            f"{GRADE_VERDICTS.get(overall_grade, '')} "
            f"Based on {td.n} OOS trades."
        )

        return {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "audit_timestamp": datetime.utcnow().isoformat(),
            "n_trades": td.n,
            "tradability_score": overall_score,
            "grade": overall_grade,
            "verdict": verdict,
            "skills": skills,
            "summary_table": summary_table,
            "recommendation": _recommendation(overall_grade, td.n),
            "improvement_suggestions": suggestions,
        }

    @staticmethod
    def audit_multi_asset(
        asset_data: Dict[str, Any],
        strategy_name: str = "Unknown",
        initial_capital: float = 10_000,
        risk_per_trade_pct: float = 1.0,
    ) -> Dict[str, Any]:
        """Audit a strategy across multiple assets/timeframes.

        Parameters
        ----------
        asset_data : dict
            Keys are labels like ``'BTC_1h'``, values are trade data in any
            supported format.

        Returns
        -------
        dict
            Per-asset audits plus a cross-asset summary.
        """
        comparisons = len(asset_data)
        per_asset: Dict[str, Dict] = {}
        scores: List[float] = []

        for label, data in asset_data.items():
            parts = label.split("_", 1)
            sym = parts[0] if parts else ""
            tf = parts[1] if len(parts) > 1 else ""
            report = StrategyAuditor.audit(
                data,
                strategy_name=strategy_name,
                symbol=sym,
                timeframe=tf,
                initial_capital=initial_capital,
                risk_per_trade_pct=risk_per_trade_pct,
                comparisons=comparisons,
            )
            per_asset[label] = report
            scores.append(report["tradability_score"])

        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        best_label = max(per_asset, key=lambda k: per_asset[k]["tradability_score"])
        worst_label = min(per_asset, key=lambda k: per_asset[k]["tradability_score"])

        return {
            "strategy_name": strategy_name,
            "n_assets": len(asset_data),
            "average_tradability_score": avg_score,
            "average_grade": _grade_from_score(avg_score),
            "best_asset": best_label,
            "best_score": per_asset[best_label]["tradability_score"],
            "worst_asset": worst_label,
            "worst_score": per_asset[worst_label]["tradability_score"],
            "per_asset": per_asset,
            "cross_asset_verdict": (
                f"Across {len(asset_data)} assets, average tradability = {avg_score}/100 "
                f"({_grade_from_score(avg_score)}). Best: {best_label} "
                f"({per_asset[best_label]['tradability_score']}), "
                f"Worst: {worst_label} ({per_asset[worst_label]['tradability_score']})."
            ),
        }


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _recommendation(grade: str, n_trades: int) -> str:
    if grade == "A":
        return (
            "DEPLOY: Strategy is suitable for live trading with appropriate position sizing. "
            "Monitor stationarity and re-audit quarterly."
        )
    if grade == "B":
        return (
            "CONDITIONAL DEPLOY: Strategy may be viable under specific regimes or with "
            "tighter filters. Paper-trade for 100+ additional trades before committing capital."
        )
    if grade == "C":
        return (
            "HOLD: Strategy has marginal characteristics. Focus on identifying profitable "
            "sub-regimes, improving exit logic, and gathering more out-of-sample data."
        )
    if grade == "D":
        return (
            "DO NOT TRADE: Edge is absent or statistically insignificant. Fundamental "
            "strategy redesign or alternative signal sources are required."
        )
    return (
        "ABANDON OR REDESIGN: All quantitative dimensions indicate this strategy "
        "destroys capital. No viable bet size exists under current signal generation."
    )


def _generate_suggestions(skills: Dict[str, Dict], td: TradeData) -> List[str]:
    suggestions: List[str] = []

    # Edge existence suggestions
    ee = skills.get("edge_existence", {})
    if ee.get("valid") and ee.get("metrics", {}).get("observed_mean_r", 0) < 0:
        suggestions.append(
            "Core edge is negative — consider inverting signal direction "
            "or adding pre-trade filters to improve selectivity."
        )

    # Stationarity suggestions
    st = skills.get("stationarity", {})
    n_cp = st.get("metrics", {}).get("n_changepoints", 0)
    if n_cp >= 2:
        suggestions.append(
            f"Strategy underwent {n_cp} structural breaks — consider regime-conditional "
            "activation (trade only in favourable regimes)."
        )

    # Ergodicity suggestions
    erg = skills.get("ergodicity", {})
    kelly_pt = erg.get("metrics", {}).get("kelly_point_estimate", 0)
    if kelly_pt < 0:
        suggestions.append(
            "Kelly fraction is negative — no viable position size exists. "
            "R:R ratio or win rate must improve before any capital is deployed."
        )

    # Regime suggestions
    re = skills.get("regime_edge", {})
    hmm = re.get("diagnostics", {}).get("hmm_regimes", {})
    regime_params = hmm.get("regime_params", {})
    for rid, rp in regime_params.items():
        if isinstance(rp, dict) and rp.get("mean_r", -1) > 0:
            pct = hmm.get("time_in_regime", {}).get(str(rid), 0)
            suggestions.append(
                f"HMM regime {rid} (mean_r={rp['mean_r']:.3f}, {pct:.0%} of time) "
                "shows positive edge — investigate isolating trades to this regime."
            )

    # Signal decay suggestions
    sd = skills.get("signal_decay", {})
    hold = sd.get("diagnostics", {}).get("optimal_holding", {})
    opt_r = hold.get("optimal_mean_r", -1)
    opt_bin = hold.get("optimal_bin", "")
    if opt_r > -0.05 and opt_bin:
        suggestions.append(
            f"Holding-period bin '{opt_bin}' has the least-negative edge ({opt_r:.3f}R) "
            "— consider tightening time-based exits to this range."
        )

    # Tail risk suggestions
    tr = skills.get("tail_risk", {})
    clustering = tr.get("diagnostics", {}).get("volatility_clustering", {})
    if clustering.get("has_clustering"):
        suggestions.append(
            "Significant volatility clustering detected — losing streaks tend to "
            "cluster. Consider reducing position size after consecutive losses."
        )

    if not suggestions:
        suggestions.append(
            "No actionable improvements identified — strategy fundamentals may need "
            "to be reconsidered from scratch."
        )

    return suggestions
