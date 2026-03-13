"""
Parallel combo evaluation for Walk-Forward Optimization.

Uses multiprocessing.Pool with an initializer pattern: shared read-only data
(adapter, DataFrame, numpy arrays, config scalars) is set once per worker,
avoiding repeated pickling.  Only the params dict is sent per task.
"""

import logging
import multiprocessing
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
#  Standalone score function (extracted from WFOEngine._score_trades)
# ═════════════════════════════════════════════════════════════

def score_trades(trades: list, metric: str) -> float:
    """Score a set of trades using the given optimization metric.

    Supports 'expectancy', 'profit_factor', 'sharpe', and 'total_r'.
    Returns -inf if fewer than 2 trades.
    """
    if len(trades) < 2:
        return -float('inf')

    r_values = [t.r_multiple_after_costs for t in trades]

    if metric == 'expectancy':
        wins = [r for r in r_values if r > 0]
        losses = [r for r in r_values if r <= 0]
        wr = len(wins) / len(r_values)
        avg_w = float(np.mean(wins)) if wins else 0
        avg_l = float(np.mean(losses)) if losses else 0
        return wr * avg_w + (1 - wr) * avg_l
    elif metric == 'profit_factor':
        gp = sum(r for r in r_values if r > 0)
        gl = abs(sum(r for r in r_values if r < 0))
        return gp / gl if gl > 0 else 0
    elif metric == 'sharpe':
        m = np.mean(r_values)
        s = np.std(r_values, ddof=1)
        return m / s if s > 0 else 0
    else:  # total_r
        return sum(r_values)


# ═════════════════════════════════════════════════════════════
#  Worker process globals (set by _init_worker, read by _evaluate_combo)
# ═════════════════════════════════════════════════════════════

_w_adapter = None
_w_df = None
_w_highs = None
_w_lows = None
_w_closes = None
_w_atrs = None
_w_scan_start = None
_w_scan_end = None
_w_costs = None
_w_max_bars = None
_w_wid = None
_w_regime = None
_w_metric = None


def _init_worker(
    adapter,
    df_bytes: bytes,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs,  # np.ndarray or None
    scan_start: int,
    scan_end: int,
    costs,
    max_bars: int,
    wid: int,
    regime: str,
    metric: str,
):
    """Initialize worker process globals.  Called once per spawned process."""
    global _w_adapter, _w_df, _w_highs, _w_lows, _w_closes, _w_atrs
    global _w_scan_start, _w_scan_end, _w_costs
    global _w_max_bars, _w_wid, _w_regime, _w_metric

    _w_adapter = adapter
    _w_df = pickle.loads(df_bytes)
    _w_highs = highs
    _w_lows = lows
    _w_closes = closes
    _w_atrs = atrs
    _w_scan_start = scan_start
    _w_scan_end = scan_end
    _w_costs = costs
    _w_max_bars = max_bars
    _w_wid = wid
    _w_regime = regime
    _w_metric = metric


# ═════════════════════════════════════════════════════════════
#  Worker function (runs in child process)
# ═════════════════════════════════════════════════════════════

def _evaluate_combo(params: Dict[str, Any]) -> Tuple[Dict[str, Any], float, list]:
    """Evaluate a single parameter combo in a worker process.

    Reads shared data from module globals set by _init_worker().
    Returns (params, score, trades).
    """
    try:
        from .simulator import TradeSimulator

        # Try execute_signals (stateful adapter path) first
        trades = _w_adapter.execute_signals(
            _w_df, params, _w_scan_start, _w_scan_end,
            _w_costs, _w_max_bars, _w_wid,
            is_oos=False, regime=_w_regime,
        )

        if trades is None:
            signals = _w_adapter.generate_signals(
                _w_df, params, _w_scan_start, _w_scan_end,
            )
            trades = []
            for sig in signals:
                trade = TradeSimulator.simulate(
                    sig.to_dict(), _w_df, _w_costs,
                    _w_max_bars, _w_wid,
                    is_oos=False, regime=_w_regime,
                    _highs=_w_highs, _lows=_w_lows,
                    _closes=_w_closes, _atrs=_w_atrs,
                )
                if trade:
                    trades.append(trade)

        score = score_trades(trades, _w_metric)
        return (params, score, trades)

    except Exception as e:
        logger.warning(f"Combo evaluation failed for {params}: {e}")
        return (params, -float('inf'), [])


# ═════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════

def evaluate_combos_parallel(
    adapter,
    train_df: pd.DataFrame,
    train_highs: np.ndarray,
    train_lows: np.ndarray,
    train_closes: np.ndarray,
    train_atrs,  # np.ndarray or None
    scan_start: int,
    scan_end: int,
    costs,
    max_trade_bars: int,
    window_id: int,
    regime: str,
    optimization_metric: str,
    param_grid: List[Dict[str, Any]],
    n_workers: int = 0,
) -> List[Tuple[Dict[str, Any], float, list]]:
    """Evaluate all parameter combos in parallel using multiprocessing.

    Parameters
    ----------
    n_workers : int
        Number of worker processes.  0 = auto (cpu_count - 1).

    Returns
    -------
    List of (params, score, trades) for every combo in param_grid.
    """
    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    # Don't spawn more workers than combos
    n_workers = min(n_workers, len(param_grid))

    # Pre-serialize DataFrame once (shared across all workers via initializer)
    df_bytes = pickle.dumps(train_df, protocol=pickle.HIGHEST_PROTOCOL)

    # Use fork on Unix (fast — workers inherit parent imports, ~4ms overhead)
    # Fall back to spawn on Windows (slower — ~3s overhead per pool creation)
    ctx_name = 'fork' if sys.platform != 'win32' else 'spawn'
    ctx = multiprocessing.get_context(ctx_name)

    init_args = (
        adapter,
        df_bytes,
        train_highs,
        train_lows,
        train_closes,
        train_atrs,
        scan_start,
        scan_end,
        costs,
        max_trade_bars,
        window_id,
        regime,
        optimization_metric,
    )

    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=init_args,
    ) as pool:
        results = pool.map(_evaluate_combo, param_grid)

    return results
