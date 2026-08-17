"""Validation harness — the machinery that decides whether an edge is real.

This package is deliberately free of strategy logic. Everything here operates on
arrays of outcomes, so it can be pointed at any bracketed trading system.

Four groups, in the order they are usually applied:

**1. Direct controls** (:mod:`.controls`) — the cheapest thing that mimics the
trade without the signal. Applied first, because they are cheap and they kill
more candidates than any p-value adjustment:

- :func:`~.controls.random_bar_control` — same bracket, random entries
- :func:`~.controls.mirrored_bracket_control` — first-passage geometry check
- :func:`~.controls.drawdown_matched_control` — controls matched on stress
- :func:`~.controls.placebo_schedule_control` — fake calendars for calendar effects

**2. Cost decomposition** (:mod:`.costs`) — whether there is gross edge above the
execution toll at all:

- :func:`~.costs.fee_in_r_wall` — the a-priori toll, per timeframe
- :func:`~.costs.decompose_gross_net` — gross vs toll, with a verdict
- :func:`~.costs.toll_neutral_threshold` — minimum worthwhile trigger distance

**3. Clustered inference** (:mod:`.clustering`) — because simultaneous
same-direction positions are one bet wearing several name tags:

- :func:`~.clustering.clustered_tstat` — cluster-robust t, with the naive inflation factor
- :func:`~.clustering.effective_sample_size` — the design-effect haircut
- :func:`~.clustering.weighting_disagreement` — obs- vs cluster-weighted means

**4. Multiplicity** (:mod:`.multiplicity`) — for any multi-cell scan, plus the
stability check a survivor must pass:

- :func:`~.multiplicity.signed_permutation_maxt` — family-wise bar, signed
- :func:`~.multiplicity.holm_bonferroni` — step-down correction
- :func:`~.multiplicity.half_split_stability` — does it hold in both halves?

Overfitting diagnostics (PBO, deflated Sharpe, combinatorial purged CV) live in
:mod:`backtrader_framework.optimization.cpcv` and are re-exported here so this
package is a single entry point.

The reasoning behind each tool — and the specific mistake that motivated it — is
documented in ``docs/RESEARCH_METHOD.md``.

Example
-------
>>> from backtrader_framework.validation import (
...     random_bar_control, excess_over_control, decompose_gross_net,
...     clustered_tstat, signed_permutation_maxt,
... )
"""

from .controls import (  # noqa: F401
    BracketOutcome,
    drawdown_matched_control,
    excess_over_control,
    mirrored_bracket_control,
    placebo_pvalue,
    placebo_schedule_control,
    random_bar_control,
    walk_bracket,
)
from .costs import (  # noqa: F401
    decompose_gross_net,
    fee_in_r_wall,
    toll_neutral_threshold,
)
from .clustering import (  # noqa: F401
    clustered_tstat,
    effective_sample_size,
    weighting_disagreement,
)
from .multiplicity import (  # noqa: F401
    half_split_stability,
    holm_bonferroni,
    one_sample_t,
    signed_permutation_maxt,
)

# Re-exported so the validation entry point is one import. Optional: the
# optimization package pulls heavier dependencies, so a missing extra degrades
# to "these two names are unavailable" rather than breaking the whole package.
try:  # pragma: no cover
    from ..optimization.cpcv import (  # noqa: F401
        CPCV,
        deflated_sharpe_ratio,
        minimum_backtest_length,
    )
except Exception:  # pragma: no cover
    CPCV = None  # type: ignore[assignment]
    deflated_sharpe_ratio = None  # type: ignore[assignment]
    minimum_backtest_length = None  # type: ignore[assignment]

__all__ = [
    # controls
    "BracketOutcome",
    "walk_bracket",
    "random_bar_control",
    "excess_over_control",
    "mirrored_bracket_control",
    "drawdown_matched_control",
    "placebo_schedule_control",
    "placebo_pvalue",
    # costs
    "fee_in_r_wall",
    "decompose_gross_net",
    "toll_neutral_threshold",
    # clustering
    "clustered_tstat",
    "effective_sample_size",
    "weighting_disagreement",
    # multiplicity
    "one_sample_t",
    "signed_permutation_maxt",
    "holm_bonferroni",
    "half_split_stability",
    # re-exported overfitting diagnostics
    "CPCV",
    "deflated_sharpe_ratio",
    "minimum_backtest_length",
]
