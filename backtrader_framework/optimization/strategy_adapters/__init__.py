"""Strategy adapters for the WFO engine."""

from .sbs_adapter import SBSAdapter
from .fvg_adapter import FVGAdapter
from .liquidity_raid_adapter import LiquidityRaidAdapter
from .mm_adapter import MomentumMasteryAdapter

ADAPTER_REGISTRY = {
    'SBS': SBSAdapter,
    'FVG': FVGAdapter,
    'LiquidityRaid': LiquidityRaidAdapter,
    'MomentumMastery': MomentumMasteryAdapter,
}
