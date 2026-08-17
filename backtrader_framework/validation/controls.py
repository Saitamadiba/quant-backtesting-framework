"""Direct controls: the cheapest thing that mimics the trade without the signal.

Family-wise error control asks "could this many tests produce this result by
chance?" A *direct control* asks a different and usually more damaging question:
"would something that ignores my signal entirely have done just as well?"

In practice the second question kills more candidates than the first, and kills
them for reasons you can explain. Each control here corresponds to a documented
failure mode — see ``docs/RESEARCH_METHOD.md`` §2.

All functions are pure NumPy/pandas and know nothing about any strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "BracketOutcome",
    "walk_bracket",
    "random_bar_control",
    "mirrored_bracket_control",
    "drawdown_matched_control",
    "placebo_schedule_control",
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared primitive: first-passage of a stop/target bracket
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BracketOutcome:
    """Result of walking one bracketed trade to first passage."""

    r_multiple: float
    bars_held: int
    outcome: str  # "target" | "stop" | "timeout"


def walk_bracket(
    highs: np.ndarray,
    lows: np.ndarray,
    entry_idx: int,
    entry_price: float,
    stop_frac: float,
    target_frac: float,
    direction: int = 1,
    max_bars: int = 500,
) -> BracketOutcome:
    """Walk a fixed bracket forward from ``entry_idx`` until stop or target.

    ``stop_frac`` and ``target_frac`` are distances as fractions of
    ``entry_price``. The R-multiple is measured against the stop distance, so a
    2:1 bracket that reaches target returns +2.0.

    Same-bar ambiguity resolves **against** the trade: if a bar's range spans
    both levels, the stop is booked. Intrabar ordering is unknowable from OHLC,
    and the conservative tie is the only honest one — the optimistic tie is a
    documented source of phantom edge (``RESEARCH_METHOD.md`` §1.2).

    Parameters
    ----------
    direction : +1 for long, -1 for short.

    Returns
    -------
    BracketOutcome
        ``outcome="timeout"`` returns the mark-to-market R at ``max_bars``.
    """
    if stop_frac <= 0 or target_frac <= 0:
        raise ValueError("stop_frac and target_frac must be positive")
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")

    n = len(highs)
    stop_px = entry_price * (1 - direction * stop_frac)
    target_px = entry_price * (1 + direction * target_frac)
    rr = target_frac / stop_frac

    last = min(entry_idx + max_bars, n - 1)
    for i in range(entry_idx + 1, last + 1):
        if direction == 1:
            hit_stop = lows[i] <= stop_px
            hit_target = highs[i] >= target_px
        else:
            hit_stop = highs[i] >= stop_px
            hit_target = lows[i] <= target_px

        if hit_stop:  # conservative tie: checked first
            return BracketOutcome(-1.0, i - entry_idx, "stop")
        if hit_target:
            return BracketOutcome(rr, i - entry_idx, "target")

    # timeout: mark to market in R
    close_like = (highs[last] + lows[last]) / 2.0
    r = direction * (close_like - entry_price) / (entry_price * stop_frac)
    return BracketOutcome(float(r), last - entry_idx, "timeout")


# ─────────────────────────────────────────────────────────────────────────────
# Control 1 — random entry bars, identical bracket
# ─────────────────────────────────────────────────────────────────────────────
def random_bar_control(
    highs: np.ndarray,
    lows: np.ndarray,
    n_trades: int,
    stop_frac: float,
    target_frac: float,
    entry_prices: Optional[np.ndarray] = None,
    direction: int = 1,
    max_bars: int = 500,
    n_replicates: int = 200,
    warmup: int = 0,
    seed: Optional[int] = None,
) -> dict:
    """What would the *same bracket* have earned entered at random bars?

    The single most clarifying control for any bracketed strategy. If a signal's
    R-expectancy matches what random entries earn under the same stop, target
    and holding cap, the signal is selecting nothing — the bracket is doing all
    the work. One volume-pattern study died here with excess ``t = +0.15``
    (``RESEARCH_METHOD.md`` §2.3).

    Draws ``n_replicates`` independent samples of ``n_trades`` random entries so
    the baseline carries its own sampling distribution rather than a single
    point estimate.

    Parameters
    ----------
    entry_prices : optional per-bar entry price. Defaults to the bar midpoint,
        which avoids the close-of-bar look-ahead a naive ``closes[i]`` invites.
    warmup : bars at the start to exclude (indicator warmup parity).

    Returns
    -------
    dict with ``mean_r`` (grand mean), ``replicate_means`` (array),
    ``ci95`` (percentile interval over replicates), and ``n_used``.
    """
    rng = np.random.default_rng(seed)
    n = len(highs)
    if entry_prices is None:
        entry_prices = (np.asarray(highs) + np.asarray(lows)) / 2.0

    lo = max(warmup, 0)
    hi = n - 2
    if hi <= lo:
        raise ValueError("not enough bars after warmup to sample entries")

    replicate_means = np.empty(n_replicates, dtype=float)
    for k in range(n_replicates):
        idx = rng.integers(lo, hi, size=n_trades)
        rs = [
            walk_bracket(
                highs, lows, int(i), float(entry_prices[i]),
                stop_frac, target_frac, direction, max_bars,
            ).r_multiple
            for i in idx
        ]
        replicate_means[k] = float(np.mean(rs))

    return {
        "mean_r": float(replicate_means.mean()),
        "replicate_means": replicate_means,
        "ci95": (
            float(np.percentile(replicate_means, 2.5)),
            float(np.percentile(replicate_means, 97.5)),
        ),
        "n_used": int(n_trades),
    }


def excess_over_control(signal_r: Sequence[float], control: dict) -> dict:
    """Signal mean minus the random-bar baseline, with a z-score.

    The z is computed against the *replicate* spread, i.e. it asks whether the
    signal beats the baseline by more than the baseline's own sampling noise.
    """
    sig = np.asarray(signal_r, dtype=float)
    reps = control["replicate_means"]
    excess = float(sig.mean() - control["mean_r"])
    sd = float(reps.std(ddof=1))
    return {
        "signal_mean_r": float(sig.mean()),
        "control_mean_r": control["mean_r"],
        "excess_r": excess,
        "z_vs_control": float(excess / sd) if sd > 0 else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Control 2 — mirrored bracket (first-passage geometry check)
# ─────────────────────────────────────────────────────────────────────────────
def mirrored_bracket_control(
    highs: np.ndarray,
    lows: np.ndarray,
    entry_indices: Sequence[int],
    entry_prices: Sequence[float],
    stop_frac: float,
    target_frac: float,
    direction: int = 1,
    max_bars: int = 500,
) -> dict:
    """Evaluate the same entries with stop and target distances swapped.

    On a driftless path a bracket's hit probabilities are asymmetric by pure
    first-passage geometry, which reliably manufactures small positive-looking
    results. The diagnostic is that **win rate** moves a lot while
    **R-expectancy** barely does.

    If the R-expectancy is similar in both orientations, you are measuring
    geometry, not prediction. One inversion study died exactly here: the
    measured asymmetry was smaller than a single spread crossing
    (``RESEARCH_METHOD.md`` §2.4).

    Returns
    -------
    dict with ``as_designed`` and ``mirrored`` sub-dicts (mean R, win rate) plus
    ``r_gap`` and ``win_rate_gap``.
    """
    def _run(sf: float, tf: float) -> dict:
        out = [
            walk_bracket(highs, lows, int(i), float(p), sf, tf, direction, max_bars)
            for i, p in zip(entry_indices, entry_prices)
        ]
        rs = np.array([o.r_multiple for o in out], dtype=float)
        wins = np.mean([o.outcome == "target" for o in out]) if out else float("nan")
        return {"mean_r": float(rs.mean()), "win_rate": float(wins), "n": len(rs)}

    designed = _run(stop_frac, target_frac)
    mirrored = _run(target_frac, stop_frac)
    return {
        "as_designed": designed,
        "mirrored": mirrored,
        "r_gap": designed["mean_r"] - mirrored["mean_r"],
        "win_rate_gap": designed["win_rate"] - mirrored["win_rate"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Control 3 — drawdown-matched controls
# ─────────────────────────────────────────────────────────────────────────────
def drawdown_matched_control(
    event_idx: Sequence[int],
    drawdown: np.ndarray,
    tolerance: float = 0.02,
    n_per_event: int = 1,
    exclude_window: int = 0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample control bars whose drawdown matches each event's drawdown.

    Events cluster in stressed conditions. A control matched only on *date*
    therefore compares stress against calm and attributes the difference to the
    event. Matching on realised drawdown removed **68%** of one event-study
    effect (``RESEARCH_METHOD.md`` §2.7).

    Parameters
    ----------
    drawdown : per-bar drawdown (any consistent sign convention).
    tolerance : absolute matching window on the drawdown value.
    exclude_window : bars around each event excluded from the control pool, so
        a "control" is not simply the same episode one bar over.

    Returns
    -------
    Array of control bar indices (may be shorter than requested if the pool is
    exhausted; the shortfall is the caller's signal that matching failed).
    """
    rng = np.random.default_rng(seed)
    dd = np.asarray(drawdown, dtype=float)
    n = len(dd)

    blocked = np.zeros(n, dtype=bool)
    for e in event_idx:
        lo, hi = max(0, e - exclude_window), min(n, e + exclude_window + 1)
        blocked[lo:hi] = True

    picked: list[int] = []
    for e in event_idx:
        target = dd[e]
        pool = np.flatnonzero((~blocked) & (np.abs(dd - target) <= tolerance))
        pool = pool[~np.isin(pool, picked)]
        if pool.size == 0:
            continue
        take = min(n_per_event, pool.size)
        picked.extend(rng.choice(pool, size=take, replace=False).tolist())

    return np.array(sorted(picked), dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Control 4 — placebo schedules
# ─────────────────────────────────────────────────────────────────────────────
def placebo_schedule_control(
    real_events: Sequence[pd.Timestamp],
    candidate_times: Sequence[pd.Timestamp],
    n_placebos: int = 200,
    preserve_weekday: bool = True,
    seed: Optional[int] = None,
) -> list[np.ndarray]:
    """Generate fake event schedules matched on count (and optionally weekday).

    When an effect is tied to a *schedule*, the right null is **other
    schedules** — not shuffled returns. A calendar effect once scored
    ``t = +7.07``; 23 of 24 fake grids scored higher, giving an honest
    ``p = 0.609`` (``RESEARCH_METHOD.md`` §2.2). The confound was the clock:
    a session open, a funding stamp and an expiry shared the same minute.

    The caller scores each returned placebo schedule with the *same* pipeline
    used on the real one, then computes
    ``p = (1 + #{placebo >= real}) / (1 + n_placebos)``.

    Parameters
    ----------
    preserve_weekday : draw placebo dates from the same weekday as the real
        events, so a weekday effect cannot masquerade as an event effect.

    Returns
    -------
    List of arrays of placebo timestamps, one per replicate.
    """
    rng = np.random.default_rng(seed)
    real = pd.DatetimeIndex(real_events)
    cand = pd.DatetimeIndex(candidate_times)
    k = len(real)
    if k == 0:
        raise ValueError("real_events is empty")

    out: list[np.ndarray] = []
    if preserve_weekday:
        by_dow: dict[int, np.ndarray] = {
            d: np.flatnonzero(cand.dayofweek == d) for d in range(7)
        }
        wanted = pd.Series(real.dayofweek).value_counts().to_dict()

    for _ in range(n_placebos):
        if preserve_weekday:
            pick: list[int] = []
            for dow, count in wanted.items():
                pool = by_dow.get(dow, np.array([], dtype=int))
                if pool.size == 0:
                    continue
                pick.extend(
                    rng.choice(pool, size=min(count, pool.size), replace=False).tolist()
                )
            idx = np.array(sorted(pick), dtype=int)
        else:
            idx = np.sort(rng.choice(len(cand), size=min(k, len(cand)), replace=False))
        out.append(cand[idx].to_numpy())
    return out


def placebo_pvalue(real_score: float, placebo_scores: Sequence[float]) -> float:
    """One-sided permutation p-value with the standard +1 correction.

    The +1 in numerator and denominator keeps the p-value from ever being
    exactly zero, which is the honest treatment of a finite null.
    """
    ps = np.asarray(placebo_scores, dtype=float)
    return float((1 + np.sum(ps >= real_score)) / (1 + ps.size))
