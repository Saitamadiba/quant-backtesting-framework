"""
Shared data normalization and result containers for quant skills.

Provides TradeData (unified input format) and SkillResult (unified output format)
used by all skill modules.  Accepts WFO oos_equity dicts, raw R-multiple arrays,
or any object with an ``r_multiple_after_costs`` attribute (e.g. TradeResult).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
#  TradeData — normalised input
# ---------------------------------------------------------------------------

@dataclass
class TradeData:
    """Normalised trade data consumed by every quant skill."""

    r_multiples: np.ndarray            # (n,) after-cost R-multiples
    timestamps: Optional[List[Any]]    # entry times (datetime or str)
    directions: Optional[List[str]]    # 'LONG' / 'SHORT'
    outcomes: Optional[List[str]]      # 'win' / 'loss'
    mfe: Optional[np.ndarray]          # max favourable excursion (R units)
    mae: Optional[np.ndarray]          # max adverse excursion (R units)
    bars_held: Optional[np.ndarray]    # holding period in bars
    regimes: Optional[List[str]]       # price-regime labels
    confidences: Optional[np.ndarray]  # signal confidence scores
    n: int = 0

    def __post_init__(self) -> None:
        self.n = len(self.r_multiples)

    # -- factory methods ---------------------------------------------------

    @classmethod
    def from_oos_equity(cls, entries: Sequence[Dict[str, Any]]) -> "TradeData":
        """Parse the ``oos_equity`` list from a WFO result JSON.

        Expected keys per entry: ``r``, ``time``, ``direction``, ``outcome``,
        ``mfe``, ``mae``, ``bars_held``, ``regime``, ``confidence``.
        """
        r = np.array([e["r"] for e in entries], dtype=np.float64)
        ts = [e.get("time") for e in entries]
        dirs_ = [e.get("direction") for e in entries]
        outs = [e.get("outcome") for e in entries]
        mfe = np.array([e.get("mfe", 0.0) for e in entries], dtype=np.float64)
        mae = np.array([e.get("mae", 0.0) for e in entries], dtype=np.float64)
        bh = np.array([e.get("bars_held", 0) for e in entries], dtype=np.float64)
        regimes = [e.get("regime") for e in entries]
        conf = np.array([e.get("confidence", 0.0) for e in entries], dtype=np.float64)
        return cls(
            r_multiples=r, timestamps=ts, directions=dirs_, outcomes=outs,
            mfe=mfe, mae=mae, bars_held=bh, regimes=regimes, confidences=conf,
        )

    @classmethod
    def from_r_multiples(cls, r: Union[np.ndarray, Sequence[float]]) -> "TradeData":
        """Minimal construction from a plain array of R-multiples."""
        arr = np.asarray(r, dtype=np.float64)
        return cls(
            r_multiples=arr, timestamps=None, directions=None, outcomes=None,
            mfe=None, mae=None, bars_held=None, regimes=None, confidences=None,
        )

    @classmethod
    def from_trade_results(cls, trades: Sequence[Any]) -> "TradeData":
        """Convert from objects that expose TradeResult-compatible attributes."""
        r = np.array([t.r_multiple_after_costs for t in trades], dtype=np.float64)
        ts = [getattr(t, "entry_time", None) for t in trades]
        dirs_ = [getattr(t, "direction", None) for t in trades]
        outs = [getattr(t, "outcome", None) for t in trades]
        mfe = np.array([getattr(t, "mfe", 0.0) for t in trades], dtype=np.float64)
        mae = np.array([getattr(t, "mae", 0.0) for t in trades], dtype=np.float64)
        bh = np.array([getattr(t, "bars_held", 0) for t in trades], dtype=np.float64)
        regimes = [getattr(t, "regime", None) for t in trades]
        conf = np.array([getattr(t, "confidence", 0.0) for t in trades], dtype=np.float64)
        return cls(
            r_multiples=r, timestamps=ts, directions=dirs_, outcomes=outs,
            mfe=mfe, mae=mae, bars_held=bh, regimes=regimes, confidences=conf,
        )

    @classmethod
    def from_auto(cls, data: Any) -> "TradeData":
        """Auto-detect input format and convert."""
        if isinstance(data, TradeData):
            return data
        if isinstance(data, np.ndarray):
            return cls.from_r_multiples(data)
        if isinstance(data, list):
            if len(data) == 0:
                return cls.from_r_multiples(np.array([]))
            first = data[0]
            if isinstance(first, dict) and "r" in first:
                return cls.from_oos_equity(data)
            if isinstance(first, (int, float)):
                return cls.from_r_multiples(data)
            if hasattr(first, "r_multiple_after_costs"):
                return cls.from_trade_results(data)
        raise TypeError(f"Cannot auto-detect trade data format from {type(data)}")


# ---------------------------------------------------------------------------
#  SkillResult — normalised output
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    """Structured result returned by every quant skill."""

    skill_name: str
    valid: bool
    reason: str = ""                              # empty if valid
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    score: float = 0.0                            # 0–100
    grade: str = "F"                              # A / B / C / D / F

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict matching existing infrastructure pattern."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "skill_name": self.skill_name,
            "score": round(self.score, 1),
            "grade": self.grade,
            "verdict": self.verdict,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _grade_from_score(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    if score >= 20:
        return "D"
    return "F"
