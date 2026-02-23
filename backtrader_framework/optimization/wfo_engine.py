"""
Generalized Walk-Forward Optimization Engine.

Optimizes strategy parameters per in-sample window and validates on
out-of-sample windows. Strategy-agnostic via the StrategyAdapter interface.

Components copied from SBS/research/analysis/wfa_backtest.py to avoid
cross-package dependency:
- IndicatorEngine, RegimeDetector, TradeSimulator, StatisticalTests
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd
from .strategy_adapters.base_adapter import StrategyAdapter, Signal
from .param_grid import generate_grid, generate_random_grid
from .drawdown_analysis import DrawdownAnalyzer
from .timing_analysis import TimingAnalyzer

# DuckDB path
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DUCKDB_PATH = os.path.join(_BASE, 'duckdb_data', 'trading_data.duckdb')


# ================================================================
#  DATA CLASSES
# ================================================================

@dataclass
class TransactionCosts:
    spread_pct: float = 0.0005
    commission_pct: float = 0.001
    slippage_pct: float = 0.0003

    @property
    def round_trip_cost_pct(self) -> float:
        return 2 * (self.spread_pct + self.commission_pct + self.slippage_pct)

    @classmethod
    def for_asset(cls, symbol: str) -> 'TransactionCosts':
        if symbol == 'NQ':
            return cls(spread_pct=0.0001, commission_pct=0.0002, slippage_pct=0.0001)
        return cls()


@dataclass
class WFOConfig:
    train_window_bars: int = 500
    test_window_bars: int = 100
    step_bars: int = 100
    anchored: bool = True
    min_train_bars: int = 250
    min_ema_warmup: int = 250
    min_trades_per_window: int = 3
    min_total_oos_trades: int = 15
    costs: TransactionCosts = field(default_factory=TransactionCosts)
    max_trade_bars: int = 168
    optimization_metric: str = 'expectancy'
    max_param_combos: int = 1000
    grid_mode: str = 'auto'  # 'full', 'random', 'auto'
    random_samples: int = 150
    max_is_scan_bars: int = 0        # Max bars to scan in IS window (0 = no limit).
                                     # In anchored mode the train window grows to 180k+
                                     # bars. Scanning all of them for every param combo is
                                     # wasteful — recent data is more relevant. Set to e.g.
                                     # 10_000 to scan only the last 10k bars of the IS window.
    use_bayesian: bool = False       # Use Optuna instead of grid search
    bayesian_n_trials: int = 100     # Number of Optuna trials per window
    bayesian_timeout: int = 60       # Timeout per window in seconds

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs) -> 'WFOConfig':
        """Create a WFOConfig scaled to the timeframe.

        The base defaults (500/100/100 bars) were designed for 4h data where
        100 bars ~ 17 days.  For smaller timeframes the same bar counts
        produce thousands of overlapping windows (15m: 1,794 windows) making
        WFO take 18h+.  This factory scales window sizes so every timeframe
        produces roughly the same number of windows (~60-120) and covers
        similar real-time durations.

        Scaling factors (relative to 4h):
            4h  → 1x   (100 bars = 17 days)
            1h  → 4x   (400 bars = 17 days)
            15m → 16x  (1600 bars = 17 days)
        """
        # bars_per_4h_bar: how many bars of this TF fit in one 4h bar
        _TF_SCALE = {'5m': 48, '15m': 16, '30m': 8, '1h': 4, '2h': 2, '4h': 1}
        scale = _TF_SCALE.get(timeframe, 1)

        base_train = kwargs.pop('train_window_bars', 500)
        base_test = kwargs.pop('test_window_bars', 100)
        base_step = kwargs.pop('step_bars', 100)
        base_min_train = kwargs.pop('min_train_bars', 250)
        base_warmup = kwargs.pop('min_ema_warmup', 250)
        base_max_trade = kwargs.pop('max_trade_bars', 168)

        # For sub-4h TFs, also default to random grid for speed
        grid_mode = kwargs.pop('grid_mode', 'random' if scale > 1 else 'auto')

        # For sub-4h TFs, use rolling (non-anchored) windows.
        # Anchored mode creates ever-growing DataFrames (up to 180k rows for 15m)
        # that slow down pandas operations even with scan capping.
        # Rolling windows keep each train_df at constant size (8000 bars for 15m
        # = 83 days — plenty for IS optimization).
        anchored = kwargs.pop('anchored', scale <= 1)

        return cls(
            train_window_bars=base_train * scale,
            test_window_bars=base_test * scale,
            step_bars=base_step * scale,
            min_train_bars=base_min_train * scale,
            min_ema_warmup=base_warmup * scale,
            max_trade_bars=base_max_trade * scale,
            grid_mode=grid_mode,
            anchored=anchored,
            **kwargs,
        )


@dataclass
class TradeResult:
    entry_time: Any
    exit_time: Any
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    outcome: str
    r_multiple: float
    r_multiple_after_costs: float
    bars_held: int
    confidence: float
    bias: str
    mfe: float
    mae: float
    window_id: int
    is_oos: bool
    regime: str
    cost_deducted: float


# ================================================================
#  INDICATOR ENGINE (copied from wfa_backtest.py)
# ================================================================

class IndicatorEngine:

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

        df['Bullish_Bias'] = df['EMA50'] > df['EMA200']
        df['Bearish_Bias'] = df['EMA50'] < df['EMA200']

        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['High_Volume'] = df['Volume'] > df['Volume_SMA'] * 1.5

        df['ADX'] = IndicatorEngine._calculate_adx(df)

        df['Manip_Score'] = 0
        daily_range = (df['High'] - df['Low']) / df['Close']
        avg_range = daily_range.rolling(window=20).mean()
        df.loc[daily_range > avg_range * 2, 'Manip_Score'] = 2
        df.loc[(daily_range > avg_range * 1.5) & (daily_range <= avg_range * 2), 'Manip_Score'] = 1

        # ── ML Feature Support Columns ──────────────────────────
        df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        df['RealizedVol20'] = df['LogReturn'].rolling(window=20).std(ddof=1)

        # ATR percentile ranks (rolling rank / window)
        atr_series = df['ATR']
        df['ATR_Pctile20'] = atr_series.rolling(window=20).apply(
            lambda w: np.sum(w <= w.iloc[-1]) / len(w), raw=False
        )
        df['ATR_Pctile100'] = atr_series.rolling(window=100).apply(
            lambda w: np.sum(w <= w.iloc[-1]) / len(w), raw=False
        )

        # Momentum: 5-bar rate of change
        df['Momentum5'] = df['Close'] / df['Close'].shift(5) - 1

        # Close vs Range: buying pressure (0 = closed at low, 1 = closed at high)
        bar_range = df['High'] - df['Low']
        df['CloseVsRange'] = np.where(
            bar_range > 0, (df['Close'] - df['Low']) / bar_range, 0.5
        )

        # Candle streak: consecutive same-direction closes
        direction = np.sign(df['Close'].values - np.roll(df['Close'].values, 1))
        direction[0] = 0
        streak = np.zeros(len(direction), dtype=float)
        for i in range(1, len(direction)):
            if direction[i] == 0:
                streak[i] = 0
            elif direction[i] == np.sign(streak[i - 1]) or streak[i - 1] == 0:
                streak[i] = streak[i - 1] + direction[i]
            else:
                streak[i] = direction[i]
        df['CandleStreak'] = streak

        return df

    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['High'], df['Low'], df['Close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            np.abs(high - close.shift()),
            np.abs(low - close.shift())
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx.rolling(window=period).mean()


# ================================================================
#  REGIME DETECTOR
# ================================================================

class RegimeDetector:

    @staticmethod
    def classify(df: pd.DataFrame, idx: int) -> str:
        if idx < 50:
            return 'unknown'
        window = df.iloc[max(0, idx - 50):idx + 1]
        adx = window['ADX'].iloc[-1] if 'ADX' in window.columns else 20
        atr = window['ATR'].iloc[-1] if 'ATR' in window.columns else 0
        price = window['Close'].iloc[-1]

        atr_pct = atr / price if price > 0 else 0
        avg_atr_pct = (window['ATR'] / window['Close']).mean() if 'ATR' in window.columns else atr_pct

        if atr_pct > avg_atr_pct * 1.8:
            return 'volatile'
        elif adx > 30:
            if 'EMA50' in window.columns and window['Close'].iloc[-1] > window['EMA50'].iloc[-1]:
                return 'trending_up'
            else:
                return 'trending_down'
        else:
            return 'ranging'


# ================================================================
#  TRADE SIMULATOR
# ================================================================

class TradeSimulator:

    @staticmethod
    def simulate(
        signal: Dict, df: pd.DataFrame, costs: TransactionCosts,
        max_bars: int = 168, window_id: int = 0,
        is_oos: bool = True, regime: str = 'unknown',
        _highs: np.ndarray = None, _lows: np.ndarray = None,
        _closes: np.ndarray = None,
    ) -> Optional[TradeResult]:
        idx = signal['idx']
        direction = signal['direction']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        tp1 = signal['take_profit_1']
        tp2 = signal.get('take_profit_2') or tp1
        risk = signal['risk']

        n = len(df)
        if risk <= 0 or idx + 1 >= n:
            return None

        # Use pre-extracted arrays if provided, else extract
        highs = _highs if _highs is not None else df['High'].values
        lows = _lows if _lows is not None else df['Low'].values
        closes = _closes if _closes is not None else df['Close'].values

        entry_cost = entry_price * (costs.spread_pct + costs.slippage_pct)
        effective_entry = entry_price + entry_cost if direction == 'LONG' else entry_price - entry_cost

        outcome = 'timeout'
        exit_price = None
        bars_held = 0
        mfe = 0.0
        mae = 0.0
        tp1_hit = False
        is_long = direction == 'LONG'

        end_bar = min(idx + max_bars, n)
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

                if not tp1_hit and h >= tp1:
                    tp1_hit = True
                if lo <= stop_loss:
                    outcome = 'loss'
                    exit_price = stop_loss
                    break
                if h >= tp2:
                    outcome = 'win_tp2'
                    exit_price = tp2
                    break
            else:
                favorable = (effective_entry - lo) / risk
                adverse = (h - effective_entry) / risk
                if favorable > mfe:
                    mfe = favorable
                if adverse > mae:
                    mae = adverse

                if not tp1_hit and lo <= tp1:
                    tp1_hit = True
                if h >= stop_loss:
                    outcome = 'loss'
                    exit_price = stop_loss
                    break
                if lo <= tp2:
                    outcome = 'win_tp2'
                    exit_price = tp2
                    break

        if outcome == 'timeout':
            if tp1_hit:
                outcome = 'win_tp1'
                exit_price = tp1
            else:
                last_idx = min(idx + max_bars - 1, n - 1)
                exit_price = closes[last_idx]

        if exit_price is None:
            return None

        if is_long:
            raw_r = (exit_price - effective_entry) / risk
        else:
            raw_r = (effective_entry - exit_price) / risk

        exit_cost = exit_price * (costs.spread_pct + costs.commission_pct + costs.slippage_pct)
        entry_cost_full = entry_price * (costs.spread_pct + costs.commission_pct + costs.slippage_pct)
        total_cost = (entry_cost_full + exit_cost) / risk if risk > 0 else 0

        return TradeResult(
            entry_time=signal['time'],
            exit_time=df.index[min(idx + bars_held, n - 1)],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            outcome=outcome,
            r_multiple=raw_r,
            r_multiple_after_costs=raw_r - total_cost,
            bars_held=bars_held,
            confidence=signal.get('confidence', 0.5),
            bias=signal.get('bias', 'COUNTER'),
            mfe=mfe,
            mae=mae,
            window_id=window_id,
            is_oos=is_oos,
            regime=regime,
            cost_deducted=total_cost,
        )


# ================================================================
#  STATISTICAL TESTS
# ================================================================

class StatisticalTests:

    @staticmethod
    def run_all(trades: List[TradeResult]) -> Dict:
        from scipy import stats as scipy_stats
        if len(trades) < 5:
            return {'valid': False, 'reason': f'Insufficient trades: {len(trades)}', 'n_trades': len(trades)}

        r_values = [t.r_multiple_after_costs for t in trades]
        wins = [t for t in trades if 'win' in t.outcome]
        n = len(trades)
        n_wins = len(wins)

        results = {'valid': True, 'n_trades': n, 'n_wins': n_wins, 'n_losses': n - n_wins}

        win_rate = n_wins / n
        results['win_rate'] = win_rate
        results['win_rate_ci_95'] = StatisticalTests._wilson_ci(n_wins, n, 0.95)

        try:
            binom_p = scipy_stats.binomtest(n_wins, n, 0.5, alternative='greater').pvalue
        except AttributeError:
            binom_p = 1.0
        results['binomial_p_value'] = binom_p
        results['win_rate_significant'] = binom_p < 0.05

        mean_r = float(np.mean(r_values))
        std_r = float(np.std(r_values, ddof=1))
        se_r = std_r / np.sqrt(n)
        ci_r = scipy_stats.t.interval(0.95, n - 1, loc=mean_r, scale=se_r) if se_r > 0 else (mean_r, mean_r)
        results['mean_r'] = mean_r
        results['std_r'] = std_r
        results['mean_r_ci_95'] = (float(ci_r[0]), float(ci_r[1]))

        t_stat, t_p = scipy_stats.ttest_1samp(r_values, 0)
        results['t_statistic'] = float(t_stat)
        results['t_p_value'] = float(t_p / 2)
        results['mean_r_significant'] = (t_p / 2) < 0.05 and mean_r > 0

        results['sharpe_per_trade'] = mean_r / std_r if std_r > 0 else 0

        gross_profit = sum(r for r in r_values if r > 0)
        gross_loss = abs(sum(r for r in r_values if r < 0))
        results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        cumulative_r = np.cumsum(r_values)
        running_max = np.maximum.accumulate(cumulative_r)
        drawdowns = running_max - cumulative_r
        results['max_drawdown_r'] = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        avg_win = float(np.mean([r for r in r_values if r > 0])) if any(r > 0 for r in r_values) else 0
        avg_loss = float(np.mean([r for r in r_values if r <= 0])) if any(r <= 0 for r in r_values) else 0
        results['avg_win_r'] = avg_win
        results['avg_loss_r'] = avg_loss
        results['expectancy'] = win_rate * avg_win + (1 - win_rate) * avg_loss

        if std_r > 0 and mean_r != 0:
            effect_size = abs(mean_r) / std_r
            results['min_n_for_significance'] = int(np.ceil((2.8 / effect_size) ** 2))
        else:
            results['min_n_for_significance'] = 999

        if n > 10:
            binary = [1 if r > 0 else 0 for r in r_values]
            autocorr = float(np.corrcoef(binary[:-1], binary[1:])[0, 1])
            results['serial_correlation'] = autocorr
            results['trades_independent'] = abs(autocorr) < 0.2
        else:
            results['serial_correlation'] = None
            results['trades_independent'] = None

        return results

    @staticmethod
    def _wilson_ci(wins: int, n: int, confidence: float) -> Tuple[float, float]:
        from scipy import stats as scipy_stats
        if n == 0:
            return (0.0, 1.0)
        z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
        p_hat = wins / n
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
        return (max(0, center - margin), min(1, center + margin))


# ================================================================
#  MONTE CARLO ANALYSIS
# ================================================================

class MonteCarloAnalysis:
    """Bootstrap-based confidence intervals and equity path simulation."""

    @staticmethod
    def bootstrap_ci(
        trades: List[TradeResult],
        n_resamples: int = 10000,
        block_size: int = 1,
        confidence: float = 0.95,
    ) -> Dict:
        """
        Compute bootstrap CIs on key metrics by resampling trades.

        Returns dict with percentile-based CIs for mean_r, win_rate,
        expectancy, profit_factor, sharpe, max_drawdown, plus p_profitable.
        """
        r_values = np.array([t.r_multiple_after_costs for t in trades])
        n = len(r_values)
        if n < 5:
            return {'valid': False, 'reason': f'Insufficient trades: {n}'}

        alpha = 1 - confidence
        lo_pct = 100 * alpha / 2
        hi_pct = 100 * (1 - alpha / 2)

        # Pre-allocate metric arrays
        mean_rs = np.empty(n_resamples)
        win_rates = np.empty(n_resamples)
        expectancies = np.empty(n_resamples)
        profit_factors = np.empty(n_resamples)
        sharpes = np.empty(n_resamples)
        max_dds = np.empty(n_resamples)

        max_block_start = max(1, n - block_size + 1)

        for i in range(n_resamples):
            # Resample
            if block_size <= 1:
                sample = r_values[np.random.randint(0, n, size=n)]
            else:
                n_blocks = int(np.ceil(n / block_size))
                blocks = []
                for _ in range(n_blocks):
                    start = np.random.randint(0, max_block_start)
                    blocks.extend(r_values[start:start + block_size].tolist())
                sample = np.array(blocks[:n])

            # Compute metrics on this resample
            mean_rs[i] = np.mean(sample)
            wins = sample > 0
            wr = np.sum(wins) / n
            win_rates[i] = wr

            win_vals = sample[wins]
            loss_vals = sample[~wins]
            avg_w = np.mean(win_vals) if len(win_vals) > 0 else 0.0
            avg_l = np.mean(loss_vals) if len(loss_vals) > 0 else 0.0
            expectancies[i] = wr * avg_w + (1 - wr) * avg_l

            gp = np.sum(sample[sample > 0])
            gl = abs(np.sum(sample[sample < 0]))
            profit_factors[i] = gp / gl if gl > 0 else 0.0

            std = np.std(sample, ddof=1)
            sharpes[i] = mean_rs[i] / std if std > 0 else 0.0

            cum = np.cumsum(sample)
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            max_dds[i] = np.max(dd) if len(dd) > 0 else 0.0

        p_profitable = float(np.mean(mean_rs > 0))

        return {
            'valid': True,
            'n_resamples': n_resamples,
            'confidence': confidence,
            'p_profitable': p_profitable,
            'mean_r_ci': (float(np.percentile(mean_rs, lo_pct)), float(np.percentile(mean_rs, hi_pct))),
            'win_rate_ci': (float(np.percentile(win_rates, lo_pct)), float(np.percentile(win_rates, hi_pct))),
            'expectancy_ci': (float(np.percentile(expectancies, lo_pct)), float(np.percentile(expectancies, hi_pct))),
            'profit_factor_ci': (float(np.percentile(profit_factors, lo_pct)), float(np.percentile(profit_factors, hi_pct))),
            'sharpe_ci': (float(np.percentile(sharpes, lo_pct)), float(np.percentile(sharpes, hi_pct))),
            'max_drawdown_ci': (float(np.percentile(max_dds, lo_pct)), float(np.percentile(max_dds, hi_pct))),
        }

    @staticmethod
    def equity_fan(
        trades: List[TradeResult],
        n_paths: int = 1000,
        block_size: int = 1,
    ) -> Dict:
        """
        Generate resampled equity paths for fan chart visualization.

        Returns percentile bands at each trade index, plus summary stats.
        """
        r_values = np.array([t.r_multiple_after_costs for t in trades])
        n = len(r_values)
        if n < 5:
            return {'valid': False}

        max_block_start = max(1, n - block_size + 1)

        # Generate paths
        all_paths = np.empty((n_paths, n))
        for i in range(n_paths):
            if block_size <= 1:
                sample = r_values[np.random.randint(0, n, size=n)]
            else:
                n_blocks = int(np.ceil(n / block_size))
                blocks = []
                for _ in range(n_blocks):
                    start = np.random.randint(0, max_block_start)
                    blocks.extend(r_values[start:start + block_size].tolist())
                sample = np.array(blocks[:n])
            all_paths[i] = np.cumsum(sample)

        # Compute percentile bands at each trade index
        pct_5 = np.percentile(all_paths, 5, axis=0).tolist()
        pct_25 = np.percentile(all_paths, 25, axis=0).tolist()
        pct_50 = np.percentile(all_paths, 50, axis=0).tolist()
        pct_75 = np.percentile(all_paths, 75, axis=0).tolist()
        pct_95 = np.percentile(all_paths, 95, axis=0).tolist()

        # Final R values across all paths
        final_rs = all_paths[:, -1]

        # Max drawdowns across all paths
        max_dds = []
        for path in all_paths:
            running_max = np.maximum.accumulate(path)
            dd = running_max - path
            max_dds.append(float(np.max(dd)))

        # Actual OOS equity for overlay
        actual_equity = np.cumsum(r_values).tolist()

        # Store up to 200 raw paths for spaghetti overlay
        raw_paths = all_paths[:min(200, n_paths)].tolist()

        return {
            'valid': True,
            'n_trades': n,
            'n_paths': n_paths,
            'percentiles': {
                '5': pct_5, '25': pct_25, '50': pct_50,
                '75': pct_75, '95': pct_95,
            },
            'actual_equity': actual_equity,
            'pct_5_final_r': float(np.percentile(final_rs, 5)),
            'median_final_r': float(np.median(final_rs)),
            'pct_95_final_r': float(np.percentile(final_rs, 95)),
            'pct_95_max_dd': float(np.percentile(max_dds, 95)),
            'median_max_dd': float(np.median(max_dds)),
            'paths': raw_paths,
        }


# ================================================================
#  DATA FETCHER (DuckDB only)
# ================================================================

class DataFetcher:

    @staticmethod
    def fetch(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            import duckdb
        except ImportError:
            print("ERROR: duckdb not installed")
            return None

        if not os.path.exists(DUCKDB_PATH):
            print(f"ERROR: DuckDB not found at {DUCKDB_PATH}")
            return None

        db_symbol = symbol.replace('-USD', '').replace('-', '')

        try:
            conn = duckdb.connect(DUCKDB_PATH, read_only=True)
            df = conn.execute(
                "SELECT timestamp, open as Open, high as High, "
                "low as Low, close as Close, volume as Volume "
                "FROM ohlcv_data WHERE symbol = ? AND timeframe = ? "
                "ORDER BY timestamp",
                [db_symbol, timeframe]
            ).fetchdf()
            conn.close()

            if df.empty:
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)

            return df
        except Exception as e:
            print(f"DuckDB fetch error: {e}")
            return None


# ================================================================
#  WFO ENGINE
# ================================================================

class WFOEngine:
    """
    Generalized Walk-Forward Optimization engine.

    For each IS window, searches over the parameter grid, picks the best
    params by the optimization metric, then applies them to the OOS window.
    """

    def __init__(self, adapter: StrategyAdapter, config: WFOConfig):
        self.adapter = adapter
        self.config = config
        self.all_oos_trades: List[TradeResult] = []
        self.all_is_trades: List[TradeResult] = []
        self.param_history: List[Dict] = []
        self.window_results: List[Dict] = []

    def run(
        self, symbol: str, timeframe: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """Execute full WFO analysis. Returns result dict."""
        cfg = self.config

        # 1. Fetch data
        if progress_callback:
            progress_callback(0.0, f"Fetching {symbol} {timeframe} from DuckDB...")
        raw_df = DataFetcher.fetch(symbol, timeframe)
        if raw_df is None or len(raw_df) < cfg.min_train_bars + cfg.test_window_bars:
            return self._empty_result(symbol, timeframe, 'insufficient_data')

        # 2. Calculate indicators
        if progress_callback:
            progress_callback(0.05, f"Calculating indicators on {len(raw_df)} bars...")
        full_df = IndicatorEngine.calculate(raw_df)
        full_df = full_df.dropna(subset=['EMA200', 'ATR', 'RSI'])

        if len(full_df) < cfg.min_train_bars + cfg.test_window_bars:
            return self._empty_result(symbol, timeframe, 'insufficient_data_after_warmup')

        # 3. Generate param grid
        param_specs = self.adapter.get_param_space()
        if cfg.grid_mode == 'random':
            param_grid = generate_random_grid(param_specs, cfg.random_samples)
        elif cfg.grid_mode == 'full':
            param_grid = generate_grid(param_specs, cfg.max_param_combos)
        else:
            param_grid = generate_grid(param_specs, cfg.max_param_combos)

        if progress_callback:
            progress_callback(0.10, f"Grid: {len(param_grid)} param combos")

        # 4. Generate windows
        windows = self._generate_windows(full_df)
        if not windows:
            return self._empty_result(symbol, timeframe, 'no_valid_windows')

        # 5. Process each window
        self.all_oos_trades = []
        self.all_is_trades = []
        self.param_history = []
        self.window_results = []

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            pct = 0.10 + 0.80 * (w_idx / len(windows))
            if progress_callback:
                progress_callback(pct, f"Window {w_idx + 1}/{len(windows)}: optimizing {len(param_grid)} combos...")

            self._optimize_window(
                full_df, w_idx, train_start, train_end, test_start, test_end, param_grid
            )

        # 6. Compile results
        if progress_callback:
            progress_callback(0.95, "Compiling statistics...")

        result = self._compile_results(symbol, timeframe, full_df)

        if progress_callback:
            progress_callback(1.0, "Done.")

        return result

    def _generate_windows(self, df: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
        cfg = self.config
        windows = []
        total_bars = len(df)

        if cfg.anchored:
            train_start = 0
            test_start = max(cfg.min_train_bars, cfg.train_window_bars)
            while test_start + cfg.test_window_bars <= total_bars:
                windows.append((train_start, test_start, test_start, test_start + cfg.test_window_bars))
                test_start += cfg.step_bars
        else:
            start = 0
            while start + cfg.train_window_bars + cfg.test_window_bars <= total_bars:
                train_end = start + cfg.train_window_bars
                test_end = train_end + cfg.test_window_bars
                windows.append((start, train_end, train_end, test_end))
                start += cfg.step_bars

        return windows

    def _optimize_window(
        self, df, w_id, train_start, train_end, test_start, test_end, param_grid,
    ):
        cfg = self.config
        train_df = df.iloc[train_start:train_end]
        test_df_with_history = df.iloc[train_start:test_end]

        regime = RegimeDetector.classify(df, test_start)

        # Pre-extract arrays for fast simulation
        train_highs = train_df['High'].values
        train_lows = train_df['Low'].values
        train_closes = train_df['Close'].values

        # --- IS Optimization ---
        best_score = -float('inf')
        best_params = self.adapter.get_default_params()
        best_is_trades = []

        scan_start = max(cfg.min_ema_warmup, 50)
        scan_end = len(train_df) - 20

        # Cap IS scan range for anchored mode (avoids scanning 180k bars)
        if cfg.max_is_scan_bars > 0 and (scan_end - scan_start) > cfg.max_is_scan_bars:
            scan_start = scan_end - cfg.max_is_scan_bars

        if scan_end <= scan_start:
            return

        if cfg.use_bayesian:
            # Bayesian optimization via Optuna
            try:
                from .bayesian_tuner import OptunaTuner, TunerConfig

                def _bayesian_objective(params):
                    signals = self.adapter.generate_signals(
                        train_df, params, scan_start, scan_end,
                    )
                    trades = []
                    for sig in signals:
                        trade = TradeSimulator.simulate(
                            sig.to_dict(), train_df, cfg.costs,
                            cfg.max_trade_bars, w_id, is_oos=False, regime=regime,
                            _highs=train_highs, _lows=train_lows, _closes=train_closes,
                        )
                        if trade:
                            trades.append(trade)
                    if len(trades) < cfg.min_trades_per_window:
                        return -float('inf')
                    return self._score_trades(trades)

                tuner = OptunaTuner(TunerConfig(
                    n_trials=cfg.bayesian_n_trials,
                    timeout_seconds=cfg.bayesian_timeout,
                    direction='maximize',
                ))
                param_specs = self.adapter.get_param_space()
                bay_result = tuner.tune_strategy_params(param_specs, _bayesian_objective)
                best_params = bay_result['best_params']
                best_score = bay_result['best_score']

                # Regenerate IS trades with best params
                best_signals = self.adapter.generate_signals(
                    train_df, best_params, scan_start, scan_end,
                )
                for sig in best_signals:
                    trade = TradeSimulator.simulate(
                        sig.to_dict(), train_df, cfg.costs,
                        cfg.max_trade_bars, w_id, is_oos=False, regime=regime,
                        _highs=train_highs, _lows=train_lows, _closes=train_closes,
                    )
                    if trade:
                        best_is_trades.append(trade)
            except ImportError:
                cfg.use_bayesian = False  # Fall through to grid search

        if not cfg.use_bayesian:
            # Standard grid/random search
            for params in param_grid:
                signals = self.adapter.generate_signals(train_df, params, scan_start, scan_end)

                trades = []
                for sig in signals:
                    trade = TradeSimulator.simulate(
                        sig.to_dict(), train_df, cfg.costs,
                        cfg.max_trade_bars, w_id, is_oos=False, regime=regime,
                        _highs=train_highs, _lows=train_lows, _closes=train_closes,
                    )
                    if trade:
                        trades.append(trade)

                score = self._score_trades(trades)
                if score > best_score and len(trades) >= cfg.min_trades_per_window:
                    best_score = score
                    best_params = params
                    best_is_trades = trades

        # --- OOS Evaluation with best params ---
        oos_scan_start = test_start - train_start
        oos_scan_end = min(test_end - train_start, len(test_df_with_history) - 20)

        if oos_scan_end <= oos_scan_start:
            return

        oos_signals = self.adapter.generate_signals(
            test_df_with_history, best_params, oos_scan_start, oos_scan_end,
        )

        oos_highs = test_df_with_history['High'].values
        oos_lows = test_df_with_history['Low'].values
        oos_closes = test_df_with_history['Close'].values

        oos_trades = []
        for sig in oos_signals:
            trade = TradeSimulator.simulate(
                sig.to_dict(), test_df_with_history, cfg.costs,
                cfg.max_trade_bars, w_id, is_oos=True, regime=regime,
                _highs=oos_highs, _lows=oos_lows, _closes=oos_closes,
            )
            if trade:
                oos_trades.append(trade)

        self.all_oos_trades.extend(oos_trades)
        self.all_is_trades.extend(best_is_trades)

        self.param_history.append({
            'window_id': w_id,
            'best_params': best_params,
            'best_score': best_score,
            'n_is_trades': len(best_is_trades),
        })

        self.window_results.append({
            'id': w_id,
            'train_period': f"{df.index[train_start].strftime('%Y-%m-%d')} to {df.index[train_end - 1].strftime('%Y-%m-%d')}",
            'test_period': f"{df.index[test_start].strftime('%Y-%m-%d')} to {df.index[min(test_end - 1, len(df) - 1)].strftime('%Y-%m-%d')}",
            'regime': regime,
            'best_params': best_params,
            'oos_trades': len(oos_trades),
            'oos_total_r': sum(t.r_multiple_after_costs for t in oos_trades),
            'is_trades': len(best_is_trades),
            'is_total_r': sum(t.r_multiple_after_costs for t in best_is_trades),
        })

    def _score_trades(self, trades: List[TradeResult]) -> float:
        if len(trades) < 2:
            return -float('inf')

        r_values = [t.r_multiple_after_costs for t in trades]
        metric = self.config.optimization_metric

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

    def _compile_results(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict:
        oos = self.all_oos_trades
        is_trades = self.all_is_trades

        oos_stats = StatisticalTests.run_all(oos) if oos else {'valid': False, 'n_trades': 0}
        is_stats = StatisticalTests.run_all(is_trades) if is_trades else {'valid': False, 'n_trades': 0}

        overfit_ratio = None
        if oos_stats.get('valid') and is_stats.get('valid'):
            is_mean = is_stats.get('mean_r', 0)
            oos_mean = oos_stats.get('mean_r', 0)
            if is_mean != 0:
                overfit_ratio = oos_mean / is_mean

        # Regime breakdown
        regime_analysis = {}
        for t in oos:
            if t.regime not in regime_analysis:
                regime_analysis[t.regime] = []
            regime_analysis[t.regime].append(t)
        regime_summary = {}
        for regime, trades in regime_analysis.items():
            r_vals = [t.r_multiple_after_costs for t in trades]
            wins = [t for t in trades if 'win' in t.outcome]
            regime_summary[regime] = {
                'n_trades': len(trades),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'mean_r': float(np.mean(r_vals)) if r_vals else 0,
                'total_r': sum(r_vals),
            }

        # Direction breakdown
        direction_analysis = {}
        for direction in ['LONG', 'SHORT']:
            dir_trades = [t for t in oos if t.direction == direction]
            if not dir_trades:
                direction_analysis[direction] = {'n_trades': 0}
                continue
            r_vals = [t.r_multiple_after_costs for t in dir_trades]
            wins = [t for t in dir_trades if 'win' in t.outcome]
            direction_analysis[direction] = {
                'n_trades': len(dir_trades),
                'win_rate': len(wins) / len(dir_trades),
                'mean_r': float(np.mean(r_vals)),
                'total_r': sum(r_vals),
            }

        # OOS equity curve data
        oos_equity = []
        cum_r = 0.0
        for t in sorted(oos, key=lambda x: x.entry_time):
            cum_r += t.r_multiple_after_costs
            oos_equity.append({
                'time': str(t.entry_time),
                'cumulative_r': cum_r,
                'r': t.r_multiple_after_costs,
                'direction': t.direction,
                'outcome': t.outcome,
                'mfe': t.mfe,
                'mae': t.mae,
                'bars_held': t.bars_held,
                'regime': t.regime,
                'confidence': t.confidence,
            })

        # Monte Carlo bootstrap CIs
        monte_carlo = None
        equity_fan = None
        if len(oos) >= 10:
            monte_carlo = MonteCarloAnalysis.bootstrap_ci(oos)
            equity_fan = MonteCarloAnalysis.equity_fan(oos)

        # Drawdown analysis & auto-disable thresholds
        drawdown_analysis = None
        if len(oos) >= 10:
            drawdown_analysis = DrawdownAnalyzer.analyze(oos_equity, monte_carlo)

        # Execution timing analysis
        timing_analysis = None
        if len(oos) >= 10:
            timing_analysis = TimingAnalyzer.analyze(oos, symbol)

        result = {
            'strategy_name': self.adapter.name,
            'symbol': symbol,
            'timeframe': timeframe,
            'run_timestamp': datetime.now().isoformat(),
            'data_start': str(df.index[0]),
            'data_end': str(df.index[-1]),
            'total_bars': len(df),
            'n_windows': len(self.window_results),
            'config': {
                'train_window': self.config.train_window_bars,
                'test_window': self.config.test_window_bars,
                'step': self.config.step_bars,
                'anchored': self.config.anchored,
                'metric': self.config.optimization_metric,
                'max_combos': self.config.max_param_combos,
                'costs_pct': self.config.costs.round_trip_cost_pct,
            },
            'oos_stats': oos_stats,
            'is_stats': is_stats,
            'overfit_ratio': overfit_ratio,
            'regime_analysis': regime_summary,
            'direction_analysis': direction_analysis,
            'windows': self.window_results,
            'param_history': self.param_history,
            'oos_equity': oos_equity,
            'oos_n_trades': len(oos),
            'is_n_trades': len(is_trades),
        }

        if monte_carlo:
            result['monte_carlo'] = monte_carlo
        if equity_fan:
            result['equity_fan'] = equity_fan
        if drawdown_analysis:
            result['drawdown_analysis'] = drawdown_analysis
        if timing_analysis:
            result['timing_analysis'] = timing_analysis

        return result

    def _empty_result(self, symbol: str, timeframe: str, reason: str) -> Dict:
        return {
            'strategy_name': self.adapter.name,
            'symbol': symbol,
            'timeframe': timeframe,
            'error': reason,
            'run_timestamp': datetime.now().isoformat(),
            'oos_stats': {'valid': False, 'n_trades': 0},
            'is_stats': {'valid': False, 'n_trades': 0},
            'overfit_ratio': None,
            'windows': [],
            'param_history': [],
            'oos_equity': [],
            'oos_n_trades': 0,
            'is_n_trades': 0,
        }


# ================================================================
#  REGIME-ADAPTIVE WFO ENGINE
# ================================================================

ALL_REGIMES = ['trending_up', 'trending_down', 'ranging', 'volatile', 'unknown']


class RegimeAdaptiveWFO(WFOEngine):
    """
    Walk-Forward Optimization with regime-adaptive parameter switching.

    Instead of one best param set per window, optimizes separate params per
    regime during IS, then switches between them during OOS based on the
    detected regime at each bar.
    """

    def __init__(self, adapter: StrategyAdapter, config: WFOConfig,
                 min_trades_per_regime: int = 5):
        super().__init__(adapter, config)
        self.min_trades_per_regime = min_trades_per_regime
        self.regime_param_history: List[Dict] = []

    def run(
        self, symbol: str, timeframe: str,
        progress_callback: Optional[Callable] = None,
        run_standard: bool = False,
    ) -> Dict:
        """
        Execute regime-adaptive WFO.

        If run_standard=True, also runs standard WFO for comparison
        (doubles runtime — only needed for overfit-ratio diagnostics).
        """
        cfg = self.config

        # 1. Fetch data
        if progress_callback:
            progress_callback(0.0, f"Fetching {symbol} {timeframe} from DuckDB...")
        raw_df = DataFetcher.fetch(symbol, timeframe)
        if raw_df is None or len(raw_df) < cfg.min_train_bars + cfg.test_window_bars:
            return self._empty_result(symbol, timeframe, 'insufficient_data')

        # 2. Calculate indicators
        if progress_callback:
            progress_callback(0.05, f"Calculating indicators on {len(raw_df)} bars...")
        full_df = IndicatorEngine.calculate(raw_df)
        full_df = full_df.dropna(subset=['EMA200', 'ATR', 'RSI'])

        if len(full_df) < cfg.min_train_bars + cfg.test_window_bars:
            return self._empty_result(symbol, timeframe, 'insufficient_data_after_warmup')

        # 3. Generate param grid
        param_specs = self.adapter.get_param_space()
        if cfg.grid_mode == 'random':
            param_grid = generate_random_grid(param_specs, cfg.random_samples)
        elif cfg.grid_mode == 'full':
            param_grid = generate_grid(param_specs, cfg.max_param_combos)
        else:
            param_grid = generate_grid(param_specs, cfg.max_param_combos)

        if progress_callback:
            progress_callback(0.08, f"Grid: {len(param_grid)} param combos")

        # 4. Generate windows
        windows = self._generate_windows(full_df)
        if not windows:
            return self._empty_result(symbol, timeframe, 'no_valid_windows')

        # 5. Optionally run standard WFO first for comparison
        standard_oos_trades = []
        if run_standard:
            if progress_callback:
                progress_callback(0.10, "Running standard WFO for comparison...")
            std_engine = WFOEngine(self.adapter, self.config)
            std_result = std_engine.run(symbol, timeframe)
            standard_oos_trades = std_engine.all_oos_trades

        # 6. Process each window with regime-adaptive optimization
        self.all_oos_trades = []
        self.all_is_trades = []
        self.param_history = []
        self.window_results = []
        self.regime_param_history = []

        total_steps = len(windows)
        base_pct = 0.50 if run_standard else 0.10

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            pct = base_pct + (0.90 - base_pct) * (w_idx / total_steps)
            if progress_callback:
                progress_callback(pct, f"[Adaptive] Window {w_idx + 1}/{total_steps}: regime optimization...")

            self._optimize_window_regime_adaptive(
                full_df, w_idx, train_start, train_end, test_start, test_end, param_grid
            )

        # 7. Compile results
        if progress_callback:
            progress_callback(0.95, "Compiling regime-adaptive results...")

        result = self._compile_results(symbol, timeframe, full_df)

        # Add regime-adaptive metadata
        result['regime_adaptive'] = True

        # Build best_params_by_regime summary across all windows
        regime_params_agg = {r: [] for r in ALL_REGIMES}
        for rph in self.regime_param_history:
            for regime, params in rph.get('params_by_regime', {}).items():
                regime_params_agg[regime].append(params)

        # Pick the most recent params per regime as representative
        best_by_regime = {}
        for regime, params_list in regime_params_agg.items():
            if params_list:
                best_by_regime[regime] = params_list[-1]
        result['best_params_by_regime'] = best_by_regime

        # Regime-switching timeline: record per-window which regimes were active
        result['regime_switching_timeline'] = self.regime_param_history

        # Standard vs Adaptive comparison
        if standard_oos_trades:
            std_r_vals = [t.r_multiple_after_costs for t in standard_oos_trades]
            ada_r_vals = [t.r_multiple_after_costs for t in self.all_oos_trades]
            std_mean = float(np.mean(std_r_vals)) if std_r_vals else 0.0
            ada_mean = float(np.mean(ada_r_vals)) if ada_r_vals else 0.0
            improvement = ((ada_mean - std_mean) / abs(std_mean) * 100) if std_mean != 0 else 0.0
            result['regime_vs_standard'] = {
                'standard_oos_mean_r': std_mean,
                'standard_oos_n_trades': len(standard_oos_trades),
                'adaptive_oos_mean_r': ada_mean,
                'adaptive_oos_n_trades': len(self.all_oos_trades),
                'improvement_pct': improvement,
            }
            # Also store standard stats for dashboard comparison
            result['standard_oos_stats'] = StatisticalTests.run_all(standard_oos_trades) if standard_oos_trades else {'valid': False, 'n_trades': 0}

        if progress_callback:
            progress_callback(1.0, "Done.")

        return result

    def _optimize_window_regime_adaptive(
        self, df, w_id, train_start, train_end, test_start, test_end, param_grid,
    ):
        """Optimize per-regime params on IS, apply regime-switching on OOS."""
        cfg = self.config
        train_df = df.iloc[train_start:train_end]
        test_df_with_history = df.iloc[train_start:test_end]

        # Pre-extract arrays for train
        train_highs = train_df['High'].values
        train_lows = train_df['Low'].values
        train_closes = train_df['Close'].values

        scan_start = max(cfg.min_ema_warmup, 50)
        scan_end = len(train_df) - 20

        # Cap IS scan range for anchored mode (avoids scanning 180k bars)
        if cfg.max_is_scan_bars > 0 and (scan_end - scan_start) > cfg.max_is_scan_bars:
            scan_start = scan_end - cfg.max_is_scan_bars

        if scan_end <= scan_start:
            return

        # --- Phase 1: IS — score each param combo per regime ---
        # Structure: regime -> list of (params, score, n_trades)
        regime_scores: Dict[str, List[Tuple[Dict, float, int]]] = {r: [] for r in ALL_REGIMES}
        overall_best_score = -float('inf')
        overall_best_params = self.adapter.get_default_params()
        overall_best_trades = []

        for params in param_grid:
            signals = self.adapter.generate_signals(train_df, params, scan_start, scan_end)

            trades = []
            for sig in signals:
                regime = RegimeDetector.classify(df, train_start + sig.idx)
                trade = TradeSimulator.simulate(
                    sig.to_dict(), train_df, cfg.costs,
                    cfg.max_trade_bars, w_id, is_oos=False, regime=regime,
                    _highs=train_highs, _lows=train_lows, _closes=train_closes,
                )
                if trade:
                    trades.append(trade)

            # Overall score (for fallback)
            overall_score = self._score_trades(trades)
            if overall_score > overall_best_score and len(trades) >= cfg.min_trades_per_window:
                overall_best_score = overall_score
                overall_best_params = params
                overall_best_trades = trades

            # Per-regime scoring
            regime_trades: Dict[str, List[TradeResult]] = {r: [] for r in ALL_REGIMES}
            for t in trades:
                regime_trades[t.regime].append(t)

            for regime in ALL_REGIMES:
                rt = regime_trades[regime]
                if len(rt) >= 2:
                    score = self._score_trades(rt)
                    regime_scores[regime].append((params, score, len(rt)))

        # --- Phase 2: Pick best params per regime ---
        params_by_regime: Dict[str, Dict] = {}
        for regime in ALL_REGIMES:
            candidates = regime_scores[regime]
            # Filter candidates with enough trades
            valid = [(p, s, n) for p, s, n in candidates if n >= self.min_trades_per_regime]
            if valid:
                best = max(valid, key=lambda x: x[1])
                params_by_regime[regime] = best[0]
            else:
                # Fallback to overall best
                params_by_regime[regime] = overall_best_params

        # Record IS trades
        self.all_is_trades.extend(overall_best_trades)

        # --- Phase 3: OOS — regime-switching signal generation ---
        oos_abs_start = test_start - train_start
        oos_abs_end = min(test_end - train_start, len(test_df_with_history) - 20)

        if oos_abs_end <= oos_abs_start:
            return

        # Classify each OOS bar into a regime
        oos_regimes = []
        for i in range(oos_abs_start, oos_abs_end):
            abs_idx = train_start + i  # index in full df
            regime = RegimeDetector.classify(df, abs_idx)
            oos_regimes.append((i, regime))

        # Group into contiguous same-regime segments
        segments = []
        if oos_regimes:
            seg_start = oos_regimes[0][0]
            seg_regime = oos_regimes[0][1]
            for i, (bar_idx, regime) in enumerate(oos_regimes[1:], 1):
                if regime != seg_regime:
                    segments.append((seg_start, oos_regimes[i - 1][0] + 1, seg_regime))
                    seg_start = bar_idx
                    seg_regime = regime
            # Last segment
            segments.append((seg_start, oos_regimes[-1][0] + 1, seg_regime))

        # Generate signals per segment with regime-appropriate params
        oos_highs = test_df_with_history['High'].values
        oos_lows = test_df_with_history['Low'].values
        oos_closes = test_df_with_history['Close'].values

        oos_trades = []
        for seg_start_idx, seg_end_idx, seg_regime in segments:
            seg_params = params_by_regime.get(seg_regime, overall_best_params)
            seg_signals = self.adapter.generate_signals(
                test_df_with_history, seg_params, seg_start_idx, seg_end_idx,
            )
            for sig in seg_signals:
                trade = TradeSimulator.simulate(
                    sig.to_dict(), test_df_with_history, cfg.costs,
                    cfg.max_trade_bars, w_id, is_oos=True, regime=seg_regime,
                    _highs=oos_highs, _lows=oos_lows, _closes=oos_closes,
                )
                if trade:
                    oos_trades.append(trade)

        self.all_oos_trades.extend(oos_trades)

        # Record window results
        window_regime = max(
            set(r for _, r in oos_regimes),
            key=lambda r: sum(1 for _, rr in oos_regimes if rr == r),
        ) if oos_regimes else 'unknown'

        self.param_history.append({
            'window_id': w_id,
            'best_params': overall_best_params,  # fallback / overall
            'best_score': overall_best_score,
            'n_is_trades': len(overall_best_trades),
        })

        self.regime_param_history.append({
            'window_id': w_id,
            'params_by_regime': params_by_regime,
            'n_segments': len(segments),
            'regime_distribution': {
                r: sum(1 for _, rr in oos_regimes if rr == r)
                for r in ALL_REGIMES if any(rr == r for _, rr in oos_regimes)
            },
        })

        self.window_results.append({
            'id': w_id,
            'train_period': f"{df.index[train_start].strftime('%Y-%m-%d')} to {df.index[train_end - 1].strftime('%Y-%m-%d')}",
            'test_period': f"{df.index[test_start].strftime('%Y-%m-%d')} to {df.index[min(test_end - 1, len(df) - 1)].strftime('%Y-%m-%d')}",
            'regime': window_regime,
            'best_params': overall_best_params,
            'params_by_regime': params_by_regime,
            'oos_trades': len(oos_trades),
            'oos_total_r': sum(t.r_multiple_after_costs for t in oos_trades),
            'is_trades': len(overall_best_trades),
            'is_total_r': sum(t.r_multiple_after_costs for t in overall_best_trades),
        })
