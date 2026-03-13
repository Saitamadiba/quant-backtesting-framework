"""
Numba JIT-compiled kernels for trade simulation and Monte Carlo bootstrap.

Provides accelerated versions of:
  - Trade simulation inner loops (V1 and V2)
  - Block bootstrap resampling
  - Monte Carlo confidence interval computation
  - Equity fan path generation

If numba is not installed, all @njit functions fall back to plain Python
with identical signatures and semantics (slower but correct).
"""

import numpy as np

# ── Numba import with graceful fallback ─────────────────────
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator when numba is not installed."""
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator


# ── Outcome integer constants ───────────────────────────────
# Used inside @njit kernels instead of strings.
# The Python wrapper maps these back to strings via OUTCOME_STRINGS.

OUTCOME_LOSS       = 0
OUTCOME_BREAKEVEN  = 1
OUTCOME_WIN_TP1    = 2
OUTCOME_WIN_TP2    = 3
OUTCOME_WIN_TRAIL  = 4
OUTCOME_WIN_TP     = 5
OUTCOME_TIME_EXIT  = 6
OUTCOME_TIMEOUT    = 7

OUTCOME_STRINGS = {
    0: 'loss',
    1: 'breakeven',
    2: 'win_tp1',
    3: 'win_tp2',
    4: 'win_trail',
    5: 'win_tp',
    6: 'time_exit',
    7: 'timeout',
}


# ═════════════════════════════════════════════════════════════
#  KERNEL 0: Block Bootstrap Resampling (shared primitive)
# ═════════════════════════════════════════════════════════════

@njit(cache=True)
def _block_bootstrap_sample(values, n, block_size, max_block_start, out):
    """Fill pre-allocated ``out`` array with one block bootstrap resample.

    Parameters
    ----------
    values : float64[:]
        Source array to resample from.
    n : int
        Number of elements to produce (== len(out)).
    block_size : int
        Block size (1 = IID bootstrap).
    max_block_start : int
        Maximum valid start index for a block (max(1, len(values) - block_size + 1)).
    out : float64[:]
        Pre-allocated output buffer of length ``n``.
    """
    if block_size <= 1:
        for i in range(n):
            out[i] = values[np.random.randint(0, n)]
    else:
        pos = 0
        n_values = len(values)
        while pos < n:
            start = np.random.randint(0, max_block_start)
            end = start + block_size
            if end > n_values:
                end = n_values
            for j in range(start, end):
                if pos >= n:
                    break
                out[pos] = values[j]
                pos += 1


# ═════════════════════════════════════════════════════════════
#  KERNEL 1: Trade Simulator V1
# ═════════════════════════════════════════════════════════════

@njit(cache=True)
def _simulate_v1_kernel(
    idx, is_long, entry_price, stop_loss, tp1, tp2, risk,
    spread_pct, commission_pct, slippage_pct,
    max_bars, trail_atr_mult,
    highs, lows, closes, atrs, has_atrs, n,
):
    """JIT-compiled inner loop for V1 trade simulation.

    Returns
    -------
    tuple of (outcome_code, exit_price, bars_held, mfe, mae, raw_r, total_cost, final_sl)
        outcome_code is one of the OUTCOME_* integer constants.
        exit_price == -1.0 signals an invalid/skipped trade.
    """
    if risk <= 0.0 or idx + 1 >= n:
        return (OUTCOME_TIMEOUT, -1.0, 0, 0.0, 0.0, 0.0, 0.0, stop_loss)

    # Entry cost
    entry_cost = entry_price * (spread_pct + slippage_pct)
    if is_long:
        effective_entry = entry_price + entry_cost
    else:
        effective_entry = entry_price - entry_cost

    outcome = OUTCOME_TIMEOUT
    exit_price = -1.0
    bars_held = 0
    mfe = 0.0
    mae = 0.0
    tp1_hit = False
    high_water = 0.0
    low_water = np.inf

    end_bar = idx + max_bars
    if end_bar > n:
        end_bar = n

    for i in range(idx + 1, end_bar):
        h = highs[i]
        lo = lows[i]
        bars_held += 1

        if is_long:
            favorable = (h - effective_entry) / risk
            adverse = (effective_entry - lo) / risk
            if favorable > mfe:
                mfe = favorable
            if adverse > mae:
                mae = adverse

            # SL checked first (conservative)
            if lo <= stop_loss:
                if tp1_hit:
                    outcome = OUTCOME_BREAKEVEN
                else:
                    outcome = OUTCOME_LOSS
                exit_price = stop_loss
                break

            if not tp1_hit and h >= tp1:
                tp1_hit = True
                stop_loss = effective_entry  # Breakeven floor
                high_water = h

            if tp1_hit:
                if h > high_water:
                    high_water = h
                # ATR trailing
                if trail_atr_mult > 0.0 and has_atrs and i < len(atrs):
                    trail_level = high_water - trail_atr_mult * atrs[i]
                    if trail_level > stop_loss:
                        stop_loss = trail_level

            if h >= tp2:
                outcome = OUTCOME_WIN_TP2
                exit_price = tp2
                break
        else:
            # SHORT direction
            favorable = (effective_entry - lo) / risk
            adverse = (h - effective_entry) / risk
            if favorable > mfe:
                mfe = favorable
            if adverse > mae:
                mae = adverse

            # SL checked first (conservative)
            if h >= stop_loss:
                if tp1_hit:
                    outcome = OUTCOME_BREAKEVEN
                else:
                    outcome = OUTCOME_LOSS
                exit_price = stop_loss
                break

            if not tp1_hit and lo <= tp1:
                tp1_hit = True
                stop_loss = effective_entry  # Breakeven floor
                low_water = lo

            if tp1_hit:
                if lo < low_water:
                    low_water = lo
                # ATR trailing
                if trail_atr_mult > 0.0 and has_atrs and i < len(atrs):
                    trail_level = low_water + trail_atr_mult * atrs[i]
                    if trail_level < stop_loss:
                        stop_loss = trail_level

            if lo <= tp2:
                outcome = OUTCOME_WIN_TP2
                exit_price = tp2
                break

    # Timeout handling
    if outcome == OUTCOME_TIMEOUT:
        if tp1_hit:
            outcome = OUTCOME_WIN_TP1
            exit_price = tp1
        else:
            last_idx = idx + max_bars - 1
            if last_idx >= n:
                last_idx = n - 1
            exit_price = closes[last_idx]

    if exit_price < 0.0:
        return (OUTCOME_TIMEOUT, -1.0, 0, 0.0, 0.0, 0.0, 0.0, stop_loss)

    # R-multiple
    if is_long:
        raw_r = (exit_price - effective_entry) / risk
    else:
        raw_r = (effective_entry - exit_price) / risk

    # Costs
    entry_comm = entry_price * commission_pct
    exit_cost = exit_price * (spread_pct + commission_pct + slippage_pct)
    total_cost = (entry_comm + exit_cost) / risk if risk > 0.0 else 0.0

    return (outcome, exit_price, bars_held, mfe, mae, raw_r, total_cost, stop_loss)


# ═════════════════════════════════════════════════════════════
#  KERNEL 2: Trade Simulator V2
# ═════════════════════════════════════════════════════════════

@njit(cache=True)
def _simulate_v2_kernel(
    idx, is_long, entry_price, stop_loss, tp, risk,
    spread_pct, commission_pct, slippage_pct,
    max_bars,
    buffer_bars, buffer_mult, be_trigger_r, be_buffer_pct,
    time_exit_bars, trail_atr_mult, trail_step_atr,
    highs, lows, closes, atrs, has_atrs, n,
):
    """JIT-compiled inner loop for V2 trade simulation.

    V2 adds: initial volatility buffer, breakeven trigger, stepped trailing,
    and time-based exit over V1.

    Returns
    -------
    tuple of (outcome_code, exit_price, bars_held, mfe, mae, raw_r, total_cost, final_sl)
    """
    if risk <= 0.0 or idx + 1 >= n:
        return (OUTCOME_TIMEOUT, -1.0, 0, 0.0, 0.0, 0.0, 0.0, stop_loss)

    # Entry cost
    entry_cost = entry_price * (spread_pct + slippage_pct)
    if is_long:
        effective_entry = entry_price + entry_cost
    else:
        effective_entry = entry_price - entry_cost

    # SL distance for buffer
    sl_distance = abs(effective_entry - stop_loss)

    outcome = OUTCOME_TIMEOUT
    exit_price = -1.0
    bars_held = 0
    mfe = 0.0
    mae = 0.0
    be_triggered = False
    trailing_active = False

    if is_long:
        high_water = effective_entry
        low_water = np.inf
    else:
        high_water = 0.0
        low_water = effective_entry

    last_trail_price = effective_entry

    end_bar = idx + max_bars
    if end_bar > n:
        end_bar = n

    for i in range(idx + 1, end_bar):
        h = highs[i]
        lo = lows[i]
        cl = closes[i]
        bars_held += 1

        # Current ATR
        atr_i = 0.0
        if has_atrs and i < len(atrs):
            atr_i = atrs[i]

        # MFE/MAE tracking
        if is_long:
            favorable = (h - effective_entry) / risk
            adverse = (effective_entry - lo) / risk
        else:
            favorable = (effective_entry - lo) / risk
            adverse = (h - effective_entry) / risk
        if favorable > mfe:
            mfe = favorable
        if adverse > mae:
            mae = adverse

        # --- 1. INITIAL BUFFER (first N bars) ---
        in_buffer = bars_held <= buffer_bars and buffer_bars > 0
        if in_buffer:
            virtual_sl_dist = sl_distance * buffer_mult
            if is_long:
                virtual_sl = effective_entry - virtual_sl_dist
                if lo <= virtual_sl:
                    outcome = OUTCOME_LOSS
                    exit_price = virtual_sl
                    break
            else:
                virtual_sl = effective_entry + virtual_sl_dist
                if h >= virtual_sl:
                    outcome = OUTCOME_LOSS
                    exit_price = virtual_sl
                    break
        else:
            # --- 2. BREAKEVEN CHECK ---
            if not be_triggered:
                if is_long:
                    current_r = (h - effective_entry) / risk
                else:
                    current_r = (effective_entry - lo) / risk
                if current_r >= be_trigger_r:
                    be_triggered = True
                    trailing_active = True
                    if is_long:
                        be_level = effective_entry + effective_entry * be_buffer_pct
                        if be_level > stop_loss:
                            stop_loss = be_level
                    else:
                        be_level = effective_entry - effective_entry * be_buffer_pct
                        if be_level < stop_loss:
                            stop_loss = be_level

            # --- 3. STEPPED TRAILING ---
            if trailing_active and trail_atr_mult > 0.0 and atr_i > 0.0:
                if is_long:
                    if h > high_water:
                        high_water = h
                    step_ok = (trail_step_atr <= 0.0 or
                               high_water - last_trail_price >= trail_step_atr * atr_i)
                    if step_ok:
                        trail_level = high_water - trail_atr_mult * atr_i
                        if trail_level > stop_loss:
                            stop_loss = trail_level
                            last_trail_price = high_water
                else:
                    if lo < low_water:
                        low_water = lo
                    step_ok = (trail_step_atr <= 0.0 or
                               last_trail_price - low_water >= trail_step_atr * atr_i)
                    if step_ok:
                        trail_level = low_water + trail_atr_mult * atr_i
                        if trail_level < stop_loss:
                            stop_loss = trail_level
                            last_trail_price = low_water

            # --- 4. SL CHECK (conservative: SL before TP) ---
            if is_long:
                if lo <= stop_loss:
                    if be_triggered:
                        raw_exit_r = (stop_loss - effective_entry) / risk
                        if raw_exit_r > 0.05:
                            outcome = OUTCOME_WIN_TRAIL
                        else:
                            outcome = OUTCOME_BREAKEVEN
                    else:
                        outcome = OUTCOME_LOSS
                    exit_price = stop_loss
                    break
            else:
                if h >= stop_loss:
                    if be_triggered:
                        raw_exit_r = (effective_entry - stop_loss) / risk
                        if raw_exit_r > 0.05:
                            outcome = OUTCOME_WIN_TRAIL
                        else:
                            outcome = OUTCOME_BREAKEVEN
                    else:
                        outcome = OUTCOME_LOSS
                    exit_price = stop_loss
                    break

            # --- 5. TP CHECK ---
            if is_long and h >= tp:
                outcome = OUTCOME_WIN_TP
                exit_price = tp
                break
            elif not is_long and lo <= tp:
                outcome = OUTCOME_WIN_TP
                exit_price = tp
                break

        # --- 6. TIME EXIT ---
        if time_exit_bars > 0 and bars_held >= time_exit_bars:
            if is_long:
                in_profit = cl > effective_entry
            else:
                in_profit = cl < effective_entry
            if not in_profit:
                outcome = OUTCOME_TIME_EXIT
                exit_price = cl
                break

    # Timeout handling
    if outcome == OUTCOME_TIMEOUT:
        last_idx = idx + max_bars - 1
        if last_idx >= n:
            last_idx = n - 1
        exit_price = closes[last_idx]

    if exit_price < 0.0:
        return (OUTCOME_TIMEOUT, -1.0, 0, 0.0, 0.0, 0.0, 0.0, stop_loss)

    # R-multiple
    if is_long:
        raw_r = (exit_price - effective_entry) / risk
    else:
        raw_r = (effective_entry - exit_price) / risk

    # Costs
    entry_comm = entry_price * commission_pct
    exit_cost_val = exit_price * (spread_pct + commission_pct + slippage_pct)
    total_cost = (entry_comm + exit_cost_val) / risk if risk > 0.0 else 0.0

    return (outcome, exit_price, bars_held, mfe, mae, raw_r, total_cost, stop_loss)


# ═════════════════════════════════════════════════════════════
#  KERNEL 3: Monte Carlo Bootstrap CI (full loop)
# ═════════════════════════════════════════════════════════════

@njit(cache=True)
def _mc_bootstrap_ci_kernel(
    r_values, n_resamples, block_size,
    mean_rs, win_rates, expectancies, profit_factors, sharpes, max_dds,
):
    """Run the full Monte Carlo bootstrap loop and fill pre-allocated output arrays.

    Computes 6 metrics per resample: mean_r, win_rate, expectancy,
    profit_factor, sharpe (ddof=1), max_drawdown.
    """
    n = len(r_values)
    max_block_start = max(1, n - block_size + 1)
    sample = np.empty(n)

    for i in range(n_resamples):
        _block_bootstrap_sample(r_values, n, block_size, max_block_start, sample)

        # Mean
        s = 0.0
        for j in range(n):
            s += sample[j]
        mean_val = s / n
        mean_rs[i] = mean_val

        # Win rate, sums for expectancy and profit factor
        n_wins = 0
        sum_w = 0.0
        cnt_w = 0
        sum_l = 0.0
        cnt_l = 0
        gp = 0.0
        gl = 0.0
        for j in range(n):
            v = sample[j]
            if v > 0.0:
                n_wins += 1
                sum_w += v
                cnt_w += 1
                gp += v
            else:
                sum_l += v
                cnt_l += 1
                gl -= v  # gl accumulates absolute losses

        wr = n_wins / n
        win_rates[i] = wr

        avg_w = sum_w / cnt_w if cnt_w > 0 else 0.0
        avg_l = sum_l / cnt_l if cnt_l > 0 else 0.0
        expectancies[i] = wr * avg_w + (1.0 - wr) * avg_l

        profit_factors[i] = gp / gl if gl > 0.0 else 0.0

        # Sharpe (ddof=1)
        ss = 0.0
        for j in range(n):
            diff = sample[j] - mean_val
            ss += diff * diff
        std = (ss / (n - 1)) ** 0.5 if n > 1 else 0.0
        sharpes[i] = mean_val / std if std > 0.0 else 0.0

        # Max drawdown
        cum = 0.0
        running_max = 0.0
        max_dd = 0.0
        for j in range(n):
            cum += sample[j]
            if cum > running_max:
                running_max = cum
            dd = running_max - cum
            if dd > max_dd:
                max_dd = dd
        max_dds[i] = max_dd


# ═════════════════════════════════════════════════════════════
#  KERNEL 4: Equity Fan Paths
# ═════════════════════════════════════════════════════════════

@njit(cache=True)
def _mc_equity_fan_kernel(r_values, n_paths, block_size, all_paths):
    """Fill pre-allocated (n_paths, n) matrix with cumulative R paths."""
    n = len(r_values)
    max_block_start = max(1, n - block_size + 1)
    sample = np.empty(n)

    for i in range(n_paths):
        _block_bootstrap_sample(r_values, n, block_size, max_block_start, sample)
        # Cumulative sum into all_paths[i]
        all_paths[i, 0] = sample[0]
        for j in range(1, n):
            all_paths[i, j] = all_paths[i, j - 1] + sample[j]


# ═════════════════════════════════════════════════════════════
#  KERNEL 5: Standalone MC Block Bootstrap (run_monte_carlo.py)
# ═════════════════════════════════════════════════════════════

@njit(cache=True)
def _mc_block_bootstrap_full(
    pnls, risk_per_trade, initial_capital, n_runs, block_size,
    final_equities, max_drawdowns, mean_rs, win_rates, profit_factors,
):
    """Full Monte Carlo bootstrap loop for the standalone runner.

    Fills 5 pre-allocated output arrays with per-run metrics.
    """
    n = len(pnls)
    max_block_start = max(1, n - block_size + 1)
    sample = np.empty(n)

    for sim in range(n_runs):
        _block_bootstrap_sample(pnls, n, block_size, max_block_start, sample)

        # Equity curve: initial_capital + cumsum(sample)
        equity_final = initial_capital
        peak = initial_capital
        max_dd = 0.0
        for j in range(n):
            equity_final += sample[j]
            if equity_final > peak:
                peak = equity_final
            dd = peak - equity_final
            if dd > max_dd:
                max_dd = dd

        final_equities[sim] = equity_final
        max_drawdowns[sim] = max_dd

        # Trade stats on resampled R-multiples
        n_wins = 0
        sum_r = 0.0
        wins_sum = 0.0
        losses_sum = 0.0
        for j in range(n):
            r_val = sample[j] / risk_per_trade
            sum_r += r_val
            if r_val > 0.0:
                n_wins += 1
                wins_sum += r_val
            else:
                losses_sum -= r_val  # absolute value

        mean_rs[sim] = sum_r / n
        win_rates[sim] = n_wins / n
        profit_factors[sim] = wins_sum / losses_sum if losses_sum > 0.0 else 99.0


# ═════════════════════════════════════════════════════════════
#  KERNEL 6: Dashboard MC Simulation
# ═════════════════════════════════════════════════════════════

@njit(cache=True)
def _mc_dashboard_kernel(
    pnls, initial_capital, n_runs, block_size,
    final_returns, max_drawdowns,
    equity_paths, max_stored_paths,
):
    """Monte Carlo kernel for the Streamlit dashboard page.

    Parameters
    ----------
    pnls : float64[:]
        Trade PnL array.
    initial_capital : float
        Starting capital.
    n_runs : int
        Number of MC simulations.
    block_size : int
        Block bootstrap size.
    final_returns : float64[:]
        Pre-allocated output for final return % per run.
    max_drawdowns : float64[:]
        Pre-allocated output for max drawdown % per run.
    equity_paths : float64[:, :]
        Pre-allocated (max_stored_paths, n+1) matrix for equity paths.
    max_stored_paths : int
        Number of paths to store (typically 200).
    """
    n = len(pnls)
    max_block_start = max(1, n - block_size + 1)
    sample = np.empty(n)

    for run in range(n_runs):
        _block_bootstrap_sample(pnls, n, block_size, max_block_start, sample)

        # Build equity curve inline
        equity_prev = initial_capital
        peak = initial_capital
        max_dd = 0.0

        # Store path if within limit
        store = run < max_stored_paths
        if store:
            equity_paths[run, 0] = initial_capital

        for j in range(n):
            equity_prev += sample[j]
            if store:
                equity_paths[run, j + 1] = equity_prev
            if equity_prev > peak:
                peak = equity_prev
            # Drawdown as percentage of peak
            if peak > 0.0:
                dd_pct = (peak - equity_prev) / peak
            else:
                dd_pct = 0.0
            if dd_pct > max_dd:
                max_dd = dd_pct

        final_returns[run] = (equity_prev / initial_capital - 1.0) * 100.0
        max_drawdowns[run] = max_dd * 100.0
