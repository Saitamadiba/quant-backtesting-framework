"""Base class and dataclasses for strategy adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import pandas as pd


@dataclass
class ParamSpec:
    """Defines one tunable parameter for grid search."""
    name: str
    default: float
    min_val: float
    max_val: float
    step: float
    param_type: str = 'float'  # 'float', 'int'
    log_scale: bool = False     # Sample in log space (for parameters spanning orders of magnitude)


@dataclass
class Signal:
    """A single trade signal produced by a strategy adapter."""
    idx: int
    time: Any  # pd.Timestamp
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    risk: float
    confidence: float
    bias: str  # 'ALIGNED' or 'COUNTER'
    atr: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'idx': self.idx,
            'time': self.time,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'risk': self.risk,
            'confidence': self.confidence,
            'bias': self.bias,
            'atr': self.atr,
        }


class StrategyAdapter(ABC):
    """
    Abstract base for lightweight strategy signal generators.

    Must work on raw pandas DataFrames with no backtrader dependency.
    Must be stateless — no instance state between generate_signals calls.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def default_timeframes(self) -> List[str]:
        ...

    @abstractmethod
    def get_param_space(self) -> List[ParamSpec]:
        ...

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        scan_start_idx: int,
        scan_end_idx: int,
    ) -> List[Signal]:
        """
        Generate trade signals over [scan_start_idx, scan_end_idx).

        df has columns: Open, High, Low, Close, Volume, plus indicators
        from IndicatorEngine (ATR, RSI, EMA20, EMA50, EMA200, etc.)

        MUST use only data up to each bar (no look-ahead).
        """
        ...

    def get_default_params(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.get_param_space()}
