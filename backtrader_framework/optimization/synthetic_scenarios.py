"""
Synthetic Stress Scenarios.

Generates artificial OHLCV price paths that mimic historical crashes at
configurable magnitudes, then runs strategies through them using existing
WFO-optimized parameters. Answers: "does the strategy survive a crash worse
than anything in history?"

Scenario types:
  - Historical crash scaling (May 2021, Nov 2022, Jun 2022 at 1x-3x)
  - Volatility spike
  - Flash crash
  - Prolonged drawdown
  - V-shaped recovery

Usage:
    from synthetic_scenarios import StressTester, StressTestConfig
    config = StressTestConfig(symbol='BTC', timeframe='4h', strategy_name='SBS')
    config.scenarios = StressTester.generate_default_scenarios()
    results = StressTester(config).run()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioType(Enum):
    HISTORICAL_CRASH_SCALED = "historical_crash_scaled"
    VOLATILITY_SPIKE = "volatility_spike"
    FLASH_CRASH = "flash_crash"
    PROLONGED_DRAWDOWN = "prolonged_drawdown"
    V_SHAPED_RECOVERY = "v_shaped_recovery"


@dataclass
class CrashPeriod:
    """A known historical crash period."""
    name: str
    start_date: str
    end_date: str
    peak_to_trough_pct: float
    description: str


KNOWN_CRASHES = {
    'may_2021': CrashPeriod(
        'May 2021 Crash', '2021-05-08', '2021-05-24', -53.0,
        'China mining ban FUD; BTC ~58k to ~29k',
    ),
    'nov_2022': CrashPeriod(
        'Nov 2022 FTX', '2022-11-05', '2022-11-22', -27.0,
        'FTX insolvency; BTC ~21k to ~15.5k',
    ),
    'jun_2022': CrashPeriod(
        'Jun 2022 Luna/3AC', '2022-06-08', '2022-06-20', -37.0,
        'Luna collapse + 3AC liquidation cascade',
    ),
}


@dataclass
class ScenarioConfig:
    """Configuration for a single stress scenario."""
    scenario_type: ScenarioType
    name: str
    magnitude: float = 1.0
    # Historical crash
    crash_key: Optional[str] = None
    # Volatility spike
    vol_multiplier: float = 3.0
    spike_duration_bars: int = 50
    # Flash crash
    flash_drop_pct: float = -0.15
    flash_bars: int = 3
    # Prolonged drawdown
    drawdown_bars: int = 200
    daily_drift_pct: float = -0.003
    # V-shaped recovery
    crash_depth_pct: float = -0.30
    crash_duration_bars: int = 20
    recovery_duration_bars: int = 20


@dataclass
class StressTestConfig:
    """Top-level configuration for stress testing."""
    symbol: str = 'BTC'
    timeframe: str = '4h'
    strategy_name: str = 'SBS'
    wfo_filepath: Optional[str] = None
    scenarios: List[ScenarioConfig] = field(default_factory=list)
    pre_context_bars: int = 250
    post_context_bars: int = 100


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class StressScenarioGenerator:
    """Generate synthetic OHLCV DataFrames for stress testing."""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self._real_df: Optional[pd.DataFrame] = None

    def _load_real_data(self) -> pd.DataFrame:
        if self._real_df is None:
            from .wfo_engine import DataFetcher
            self._real_df = DataFetcher.fetch(self.symbol, self.timeframe)
            if self._real_df is None:
                raise ValueError(
                    f"No OHLCV data for {self.symbol}/{self.timeframe}"
                )
        return self._real_df

    def generate(self, config: ScenarioConfig) -> Dict:
        """Generate a synthetic OHLCV DataFrame for the given scenario."""
        generators = {
            ScenarioType.HISTORICAL_CRASH_SCALED: self._gen_historical_crash,
            ScenarioType.VOLATILITY_SPIKE: self._gen_volatility_spike,
            ScenarioType.FLASH_CRASH: self._gen_flash_crash,
            ScenarioType.PROLONGED_DRAWDOWN: self._gen_prolonged_drawdown,
            ScenarioType.V_SHAPED_RECOVERY: self._gen_v_shaped_recovery,
        }
        gen_fn = generators.get(config.scenario_type)
        if gen_fn is None:
            raise ValueError(f"Unknown scenario type: {config.scenario_type}")
        return gen_fn(config)

    # ── Historical Crash Scaling ─────────────────────────────────────────

    def _gen_historical_crash(self, config: ScenarioConfig) -> Dict:
        real_df = self._load_real_data()
        crash = KNOWN_CRASHES.get(config.crash_key)
        if crash is None:
            raise ValueError(f"Unknown crash key: {config.crash_key}")

        # Extract crash period
        crash_mask = (real_df.index >= crash.start_date) & (
            real_df.index <= crash.end_date
        )
        crash_df = real_df.loc[crash_mask].copy()
        if len(crash_df) < 5:
            raise ValueError(
                f"Crash period {crash.name} has only {len(crash_df)} bars"
            )

        # Find crash start index in full df
        crash_start_iloc = real_df.index.get_indexer([crash_df.index[0]])[0]

        # Pre-context: real bars before the crash (250 for indicator warmup)
        pre_start = max(0, crash_start_iloc - 250)
        pre_df = real_df.iloc[pre_start:crash_start_iloc].copy()

        # Scale crash returns
        scaled_crash = self._scale_returns(crash_df, config.magnitude)

        # Adjust price level: scaled crash starts from pre_df's last close
        if len(pre_df) > 0:
            scaled_crash = self._adjust_price_level(
                scaled_crash, pre_df.iloc[-1]['Close']
            )

        # Post-context: real bars after the crash, adjusted to scaled level
        crash_end_iloc = real_df.index.get_indexer([crash_df.index[-1]])[0]
        post_end = min(len(real_df), crash_end_iloc + 1 + 100)
        post_df = real_df.iloc[crash_end_iloc + 1:post_end].copy()
        if len(post_df) > 0 and len(scaled_crash) > 0:
            post_df = self._adjust_price_level(
                post_df, scaled_crash.iloc[-1]['Close']
            )

        # Combine
        combined = pd.concat([pre_df, scaled_crash, post_df])
        combined = combined[~combined.index.duplicated(keep='first')]

        event_start_idx = len(pre_df)
        event_end_idx = len(pre_df) + len(scaled_crash)

        # Original (unscaled) for comparison
        orig_end = min(len(real_df), crash_end_iloc + 1 + 100)
        original_df = real_df.iloc[pre_start:orig_end].copy()

        # Metadata
        if len(scaled_crash) > 0:
            peak = scaled_crash['High'].max()
            trough = scaled_crash['Low'].min()
            total_drop = (trough - scaled_crash.iloc[0]['Open']) / scaled_crash.iloc[0]['Open']
        else:
            peak = trough = total_drop = 0.0

        return {
            'df': combined,
            'event_start_idx': event_start_idx,
            'event_end_idx': event_end_idx,
            'original_df': original_df,
            'scenario_config': config,
            'metadata': {
                'peak_price': float(peak),
                'trough_price': float(trough),
                'total_drop_pct': float(total_drop),
                'duration_bars': len(scaled_crash),
                'type': config.scenario_type.value,
            },
        }

    # ── Volatility Spike ─────────────────────────────────────────────────

    def _gen_volatility_spike(self, config: ScenarioConfig) -> Dict:
        real_df = self._load_real_data()

        # Pick a calm period from the middle of the data
        mid = len(real_df) // 2
        total_needed = 250 + config.spike_duration_bars + 100
        start = max(0, mid - total_needed // 2)

        base_df = real_df.iloc[start:start + total_needed].copy()
        if len(base_df) < total_needed:
            base_df = real_df.iloc[:total_needed].copy()

        event_start = 250
        event_end = event_start + config.spike_duration_bars

        # Expand H-L range by vol_multiplier within the spike window
        synth = base_df.copy()
        for i in range(event_start, min(event_end, len(synth))):
            mid_price = (synth.iloc[i]['High'] + synth.iloc[i]['Low']) / 2
            half_range = (synth.iloc[i]['High'] - synth.iloc[i]['Low']) / 2
            new_half = half_range * config.vol_multiplier

            synth.iloc[i, synth.columns.get_loc('High')] = mid_price + new_half
            synth.iloc[i, synth.columns.get_loc('Low')] = mid_price - new_half
            # Keep Close within range
            close = synth.iloc[i]['Close']
            synth.iloc[i, synth.columns.get_loc('Close')] = np.clip(
                close, mid_price - new_half, mid_price + new_half
            )
            open_ = synth.iloc[i]['Open']
            synth.iloc[i, synth.columns.get_loc('Open')] = np.clip(
                open_, mid_price - new_half, mid_price + new_half
            )
            # Scale volume
            synth.iloc[i, synth.columns.get_loc('Volume')] *= config.vol_multiplier

        return {
            'df': synth,
            'event_start_idx': event_start,
            'event_end_idx': event_end,
            'original_df': base_df,
            'scenario_config': config,
            'metadata': {
                'peak_price': float(synth.iloc[event_start:event_end]['High'].max()),
                'trough_price': float(synth.iloc[event_start:event_end]['Low'].min()),
                'total_drop_pct': float(
                    (synth.iloc[event_end - 1]['Close'] - synth.iloc[event_start]['Open'])
                    / synth.iloc[event_start]['Open']
                ) if event_end <= len(synth) else 0.0,
                'duration_bars': config.spike_duration_bars,
                'type': config.scenario_type.value,
            },
        }

    # ── Flash Crash ──────────────────────────────────────────────────────

    def _gen_flash_crash(self, config: ScenarioConfig) -> Dict:
        real_df = self._load_real_data()
        rng = np.random.RandomState(42)

        # Take a calm period
        mid = len(real_df) // 2
        total_needed = 250 + config.flash_bars + 20 + 100
        start = max(0, mid - total_needed // 2)
        base_df = real_df.iloc[start:start + total_needed].copy()

        synth = base_df.copy()
        event_start = 250
        anchor_close = synth.iloc[event_start - 1]['Close']

        # Flash crash bars
        drop_per_bar = config.flash_drop_pct / config.flash_bars
        current_price = anchor_close

        for i in range(config.flash_bars):
            idx = event_start + i
            if idx >= len(synth):
                break

            bar_return = drop_per_bar * (1 + 0.3 * rng.randn())
            new_close = current_price * (1 + bar_return)
            # Wicks extend 1.5x beyond the close move
            wick_ext = abs(current_price - new_close) * 0.5

            synth.iloc[idx, synth.columns.get_loc('Open')] = current_price
            synth.iloc[idx, synth.columns.get_loc('Close')] = new_close
            synth.iloc[idx, synth.columns.get_loc('High')] = current_price + wick_ext * 0.2
            synth.iloc[idx, synth.columns.get_loc('Low')] = new_close - wick_ext
            synth.iloc[idx, synth.columns.get_loc('Volume')] *= 5.0

            current_price = new_close

        # Partial recovery (50-70%)
        recovery_target = anchor_close * (1 + config.flash_drop_pct * 0.4)
        recovery_bars = min(20, len(synth) - event_start - config.flash_bars)
        if recovery_bars > 0:
            recovery_per_bar = (recovery_target - current_price) / recovery_bars
            for i in range(recovery_bars):
                idx = event_start + config.flash_bars + i
                if idx >= len(synth):
                    break
                new_close = current_price + recovery_per_bar * (1 + 0.2 * rng.randn())
                half_range = abs(recovery_per_bar) * 1.5
                synth.iloc[idx, synth.columns.get_loc('Open')] = current_price
                synth.iloc[idx, synth.columns.get_loc('Close')] = new_close
                synth.iloc[idx, synth.columns.get_loc('High')] = max(current_price, new_close) + half_range
                synth.iloc[idx, synth.columns.get_loc('Low')] = min(current_price, new_close) - half_range * 0.3
                synth.iloc[idx, synth.columns.get_loc('Volume')] *= 3.0
                current_price = new_close

        event_end = event_start + config.flash_bars + recovery_bars
        self._enforce_ohlc(synth, event_start, event_end)

        return {
            'df': synth,
            'event_start_idx': event_start,
            'event_end_idx': event_end,
            'original_df': base_df,
            'scenario_config': config,
            'metadata': {
                'peak_price': float(anchor_close),
                'trough_price': float(synth.iloc[event_start:event_end]['Low'].min()),
                'total_drop_pct': float(config.flash_drop_pct),
                'duration_bars': config.flash_bars + recovery_bars,
                'type': config.scenario_type.value,
            },
        }

    # ── Prolonged Drawdown ───────────────────────────────────────────────

    def _gen_prolonged_drawdown(self, config: ScenarioConfig) -> Dict:
        real_df = self._load_real_data()
        rng = np.random.RandomState(42)

        total_needed = 250 + config.drawdown_bars + 100
        # Find an uptrend period to start from
        mid = len(real_df) // 3
        start = max(0, mid)
        base_df = real_df.iloc[start:start + total_needed].copy()

        synth = base_df.copy()
        event_start = 250
        anchor_close = synth.iloc[event_start - 1]['Close']

        # Compute typical bar range from pre-context
        pre_ranges = (synth.iloc[:event_start]['High'] - synth.iloc[:event_start]['Low']).values
        avg_range = float(np.mean(pre_ranges[pre_ranges > 0])) if len(pre_ranges) > 0 else anchor_close * 0.01

        current_price = anchor_close
        for i in range(config.drawdown_bars):
            idx = event_start + i
            if idx >= len(synth):
                break

            # Negative drift with reduced volatility
            drift = config.daily_drift_pct * current_price
            noise = rng.randn() * avg_range * 0.5  # 0.5x normal vol
            new_close = current_price + drift + noise

            # Small dead-cat bounces every ~40 bars
            if i > 0 and i % 40 == 0:
                bounce = abs(drift) * rng.uniform(5, 10)
                new_close = current_price + bounce

            bar_range = avg_range * 0.7
            synth.iloc[idx, synth.columns.get_loc('Open')] = current_price
            synth.iloc[idx, synth.columns.get_loc('Close')] = new_close
            synth.iloc[idx, synth.columns.get_loc('High')] = max(current_price, new_close) + bar_range * 0.3
            synth.iloc[idx, synth.columns.get_loc('Low')] = min(current_price, new_close) - bar_range * 0.3
            synth.iloc[idx, synth.columns.get_loc('Volume')] *= 0.7

            current_price = new_close

        event_end = min(event_start + config.drawdown_bars, len(synth))

        # Adjust post-context
        if event_end < len(synth):
            post_slice = synth.iloc[event_end:].copy()
            adjusted = self._adjust_price_level(post_slice, current_price)
            for col in ['Open', 'High', 'Low', 'Close']:
                synth.iloc[event_end:, synth.columns.get_loc(col)] = adjusted[col].values

        self._enforce_ohlc(synth, event_start, event_end)

        return {
            'df': synth,
            'event_start_idx': event_start,
            'event_end_idx': event_end,
            'original_df': base_df,
            'scenario_config': config,
            'metadata': {
                'peak_price': float(anchor_close),
                'trough_price': float(synth.iloc[event_start:event_end]['Low'].min()),
                'total_drop_pct': float((current_price - anchor_close) / anchor_close),
                'duration_bars': config.drawdown_bars,
                'type': config.scenario_type.value,
            },
        }

    # ── V-Shaped Recovery ────────────────────────────────────────────────

    def _gen_v_shaped_recovery(self, config: ScenarioConfig) -> Dict:
        real_df = self._load_real_data()
        rng = np.random.RandomState(42)

        total_duration = config.crash_duration_bars + config.recovery_duration_bars
        total_needed = 250 + total_duration + 100
        mid = len(real_df) // 2
        start = max(0, mid - total_needed // 2)
        base_df = real_df.iloc[start:start + total_needed].copy()

        synth = base_df.copy()
        event_start = 250
        anchor_close = synth.iloc[event_start - 1]['Close']

        pre_ranges = (synth.iloc[:event_start]['High'] - synth.iloc[:event_start]['Low']).values
        avg_range = float(np.mean(pre_ranges[pre_ranges > 0])) if len(pre_ranges) > 0 else anchor_close * 0.01

        current_price = anchor_close

        # Phase 1: crash
        crash_target = anchor_close * (1 + config.crash_depth_pct)
        for i in range(config.crash_duration_bars):
            idx = event_start + i
            if idx >= len(synth):
                break
            # Front-loaded crash: bigger drops early
            progress = (i + 1) / config.crash_duration_bars
            weight = 2 * (1 - progress) + 0.5  # Higher weight early
            step_return = config.crash_depth_pct / config.crash_duration_bars * weight
            noise = rng.randn() * avg_range * 0.3
            new_close = current_price * (1 + step_return) + noise

            vol_mult = 2.0 + progress  # Vol increases through crash
            bar_range = avg_range * vol_mult

            synth.iloc[idx, synth.columns.get_loc('Open')] = current_price
            synth.iloc[idx, synth.columns.get_loc('Close')] = new_close
            synth.iloc[idx, synth.columns.get_loc('High')] = max(current_price, new_close) + bar_range * 0.3
            synth.iloc[idx, synth.columns.get_loc('Low')] = min(current_price, new_close) - bar_range * 0.5
            synth.iloc[idx, synth.columns.get_loc('Volume')] *= vol_mult

            current_price = new_close

        trough_price = current_price

        # Phase 2: recovery (mirror)
        recovery_target = anchor_close * (1 + config.crash_depth_pct * 0.1)  # Recover to ~90% of original
        recovery_per_bar = (recovery_target - trough_price) / max(1, config.recovery_duration_bars)
        for i in range(config.recovery_duration_bars):
            idx = event_start + config.crash_duration_bars + i
            if idx >= len(synth):
                break
            noise = rng.randn() * avg_range * 0.3
            new_close = current_price + recovery_per_bar + noise

            progress = (i + 1) / config.recovery_duration_bars
            vol_mult = 3.0 - progress * 1.5  # Vol decreases through recovery
            bar_range = avg_range * vol_mult

            synth.iloc[idx, synth.columns.get_loc('Open')] = current_price
            synth.iloc[idx, synth.columns.get_loc('Close')] = new_close
            synth.iloc[idx, synth.columns.get_loc('High')] = max(current_price, new_close) + bar_range * 0.4
            synth.iloc[idx, synth.columns.get_loc('Low')] = min(current_price, new_close) - bar_range * 0.3
            synth.iloc[idx, synth.columns.get_loc('Volume')] *= vol_mult

            current_price = new_close

        event_end = min(event_start + total_duration, len(synth))

        # Adjust post-context
        if event_end < len(synth):
            post_slice = synth.iloc[event_end:].copy()
            adjusted = self._adjust_price_level(post_slice, current_price)
            for col in ['Open', 'High', 'Low', 'Close']:
                synth.iloc[event_end:, synth.columns.get_loc(col)] = adjusted[col].values

        self._enforce_ohlc(synth, event_start, event_end)

        return {
            'df': synth,
            'event_start_idx': event_start,
            'event_end_idx': event_end,
            'original_df': base_df,
            'scenario_config': config,
            'metadata': {
                'peak_price': float(anchor_close),
                'trough_price': float(trough_price),
                'total_drop_pct': float(config.crash_depth_pct),
                'duration_bars': total_duration,
                'type': config.scenario_type.value,
            },
        }

    # ── Utility Methods ──────────────────────────────────────────────────

    @staticmethod
    def _scale_returns(df: pd.DataFrame, magnitude: float) -> pd.DataFrame:
        """Scale OHLC log-returns by a magnitude multiplier."""
        if magnitude == 1.0:
            return df.copy()

        scaled = df.copy()
        closes = df['Close'].values
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]  # First bar anchor

        anchor = closes[0]
        new_prices = {'Open': [], 'High': [], 'Low': [], 'Close': []}

        running_close = anchor
        for i in range(len(df)):
            for col in ['Open', 'High', 'Low', 'Close']:
                orig_val = df.iloc[i][col]
                ref = prev_closes[i] if i > 0 else anchor
                if ref > 0 and orig_val > 0:
                    log_ret = np.log(orig_val / ref)
                    scaled_ret = log_ret * magnitude
                    new_val = running_close * np.exp(scaled_ret)
                else:
                    new_val = orig_val
                new_prices[col].append(new_val)

            running_close = new_prices['Close'][-1]

        for col in ['Open', 'High', 'Low', 'Close']:
            scaled[col] = new_prices[col]

        # Enforce OHLC consistency
        for i in range(len(scaled)):
            o = scaled.iloc[i]['Open']
            c = scaled.iloc[i]['Close']
            h = scaled.iloc[i]['High']
            l = scaled.iloc[i]['Low']
            scaled.iloc[i, scaled.columns.get_loc('High')] = max(h, o, c)
            scaled.iloc[i, scaled.columns.get_loc('Low')] = min(l, o, c)

        # Scale volume proportionally
        orig_abs_ret = np.abs(np.diff(closes, prepend=closes[0]) / np.maximum(closes, 1e-10))
        vol_scale = 1.0 + (magnitude - 1.0) * np.clip(
            orig_abs_ret / (np.max(orig_abs_ret) + 1e-10), 0, 1
        )
        scaled['Volume'] = df['Volume'].values * vol_scale

        return scaled

    @staticmethod
    def _adjust_price_level(df: pd.DataFrame, target_close: float) -> pd.DataFrame:
        """Shift OHLCV so the first bar's Open matches target_close."""
        if len(df) == 0:
            return df.copy()
        adjusted = df.copy()
        ratio = target_close / df.iloc[0]['Open'] if df.iloc[0]['Open'] > 0 else 1.0
        for col in ['Open', 'High', 'Low', 'Close']:
            adjusted[col] = df[col] * ratio
        return adjusted

    @staticmethod
    def _enforce_ohlc(df: pd.DataFrame, start: int, end: int):
        """Enforce H >= max(O,C) and L <= min(O,C) in-place."""
        for i in range(start, min(end, len(df))):
            o = df.iloc[i]['Open']
            c = df.iloc[i]['Close']
            h = df.iloc[i]['High']
            l = df.iloc[i]['Low']
            df.iloc[i, df.columns.get_loc('High')] = max(h, o, c)
            df.iloc[i, df.columns.get_loc('Low')] = min(l, o, c)
            # Ensure Low > 0
            if df.iloc[i]['Low'] <= 0:
                df.iloc[i, df.columns.get_loc('Low')] = min(o, c) * 0.95


# ══════════════════════════════════════════════════════════════════════════════
# STRESS TESTER
# ══════════════════════════════════════════════════════════════════════════════

class StressTester:
    """Run strategies through synthetic stress scenarios."""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self._adapter = None
        self._wfo_result = None
        self._best_params = None

    def run(self) -> Dict:
        """Execute stress tests across all configured scenarios."""
        from .wfo_engine import IndicatorEngine, TradeSimulator, TransactionCosts

        adapter = self._get_adapter()
        best_params = self._get_best_params()
        costs = TransactionCosts.for_asset(self.config.symbol)
        generator = StressScenarioGenerator(self.config.symbol, self.config.timeframe)

        # Baseline on real data
        baseline = self._run_baseline(generator, adapter, best_params, costs)

        # Run each scenario
        scenario_results = []
        for sc in self.config.scenarios:
            result = self._run_single_scenario(
                generator, adapter, best_params, costs, sc,
            )
            scenario_results.append(result)

        # Build survival matrix
        survival_matrix = self._build_survival_matrix(scenario_results)

        return {
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'strategy': self.config.strategy_name,
            'params_used': best_params,
            'baseline': baseline,
            'scenarios': scenario_results,
            'survival_matrix': survival_matrix,
        }

    def _run_single_scenario(
        self, generator, adapter, best_params, costs, sc_config,
    ) -> Dict:
        from .wfo_engine import IndicatorEngine, TradeSimulator

        try:
            scenario_data = generator.generate(sc_config)
        except Exception as e:
            return {
                'name': sc_config.name,
                'type': sc_config.scenario_type.value,
                'magnitude': sc_config.magnitude,
                'error': str(e),
                'n_trades': 0,
                'trades_during_event': 0,
                'survival_metrics': self._empty_survival(),
            }

        df = scenario_data['df']
        event_start = scenario_data['event_start_idx']
        event_end = scenario_data['event_end_idx']

        # Add indicators
        df_ind = IndicatorEngine.calculate(df)

        # Generate signals around the event
        scan_start = max(250, event_start - 50)
        scan_end = min(len(df_ind) - 20, event_end + 50)

        if scan_end <= scan_start:
            return {
                'name': sc_config.name,
                'type': sc_config.scenario_type.value,
                'magnitude': sc_config.magnitude,
                'n_trades': 0,
                'trades_during_event': 0,
                'survival_metrics': self._empty_survival(),
                'price_data': self._extract_price_data(scenario_data),
            }

        signals = adapter.generate_signals(df_ind, best_params, scan_start, scan_end)

        trades = []
        for sig in signals:
            trade = TradeSimulator.simulate(
                sig.to_dict(), df_ind, costs,
                max_bars=168, window_id=0, is_oos=True, regime='stress',
            )
            if trade:
                trades.append(trade)

        # Classify trades by event window
        trades_during = []
        for t in trades:
            # Check if trade entry falls within event window by index
            entry_idx = df_ind.index.get_indexer([t.entry_time])[0]
            if event_start <= entry_idx < event_end:
                trades_during.append(t)

        survival = self._compute_survival_metrics(
            trades, trades_during, scenario_data,
        )

        return {
            'name': sc_config.name,
            'type': sc_config.scenario_type.value,
            'magnitude': sc_config.magnitude,
            'n_trades': len(trades),
            'trades_during_event': len(trades_during),
            'survival_metrics': survival,
            'metadata': scenario_data['metadata'],
            'price_data': self._extract_price_data(scenario_data),
        }

    def _run_baseline(self, generator, adapter, best_params, costs) -> Dict:
        """Run strategy on real (unmodified) data for comparison."""
        from .wfo_engine import IndicatorEngine, TradeSimulator

        real_df = generator._load_real_data()
        df_ind = IndicatorEngine.calculate(real_df)

        # Use same scan range as would be used for a crash scenario
        scan_start = max(250, 250)
        scan_end = min(len(df_ind) - 20, 250 + 200)

        if scan_end <= scan_start:
            return {'n_trades': 0, 'win_rate': 0, 'mean_r': 0, 'max_dd_r': 0}

        signals = adapter.generate_signals(df_ind, best_params, scan_start, scan_end)
        trades = []
        for sig in signals:
            trade = TradeSimulator.simulate(
                sig.to_dict(), df_ind, costs,
                max_bars=168, window_id=0, is_oos=True, regime='baseline',
            )
            if trade:
                trades.append(trade)

        if not trades:
            return {'n_trades': 0, 'win_rate': 0, 'mean_r': 0, 'max_dd_r': 0}

        rs = [t.r_multiple_after_costs for t in trades]
        wins = sum(1 for r in rs if r > 0)
        cum_r = np.cumsum(rs)
        peak = np.maximum.accumulate(cum_r)
        dd = cum_r - peak

        return {
            'n_trades': len(trades),
            'win_rate': round(wins / len(trades), 4),
            'mean_r': round(float(np.mean(rs)), 4),
            'total_r': round(float(np.sum(rs)), 4),
            'max_dd_r': round(float(np.min(dd)), 4) if len(dd) > 0 else 0,
        }

    def _compute_survival_metrics(
        self, all_trades, event_trades, scenario_data,
    ) -> Dict:
        if not all_trades:
            return {
                'survived': True,
                'max_drawdown_r': 0.0,
                'consecutive_losses': 0,
                'recovery_trades': 0,
                'net_r_during_event': 0.0,
                'net_r_total': 0.0,
                'worst_single_trade_r': 0.0,
                'trades_stopped_out': 0,
                'survival_score': 100.0,  # No trades = no losses
            }

        rs = [t.r_multiple_after_costs for t in all_trades]
        event_rs = [t.r_multiple_after_costs for t in event_trades]

        # Max drawdown
        cum_r = np.cumsum(rs)
        peak = np.maximum.accumulate(cum_r)
        dd = cum_r - peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

        # Consecutive losses
        max_consec = 0
        current_consec = 0
        for r in rs:
            if r <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        # Recovery: trades needed after max DD to get back to pre-DD equity
        max_dd_idx = int(np.argmin(dd)) if len(dd) > 0 else 0
        recovery_trades = 0
        if max_dd_idx < len(cum_r) - 1:
            pre_dd_level = float(peak[max_dd_idx])
            for j in range(max_dd_idx + 1, len(cum_r)):
                if cum_r[j] >= pre_dd_level:
                    recovery_trades = j - max_dd_idx
                    break
            else:
                recovery_trades = -1  # Never recovered

        # Stopped out
        stopped = sum(1 for t in all_trades if t.outcome == 'loss')

        survival_score = self._compute_survival_score(
            max_dd, max_consec, recovery_trades, sum(event_rs) if event_rs else 0,
        )

        return {
            'survived': max_dd > -15.0,
            'max_drawdown_r': round(max_dd, 4),
            'consecutive_losses': max_consec,
            'recovery_trades': recovery_trades,
            'net_r_during_event': round(float(sum(event_rs)), 4) if event_rs else 0.0,
            'net_r_total': round(float(sum(rs)), 4),
            'worst_single_trade_r': round(float(min(rs)), 4),
            'trades_stopped_out': stopped,
            'survival_score': round(survival_score, 1),
        }

    @staticmethod
    def _compute_survival_score(
        max_dd: float, consec_losses: int, recovery_trades: int, net_r_event: float,
    ) -> float:
        """Composite survival score (0-100)."""
        # Max drawdown (40%): 100 if DD > -3R, 0 if DD <= -15R
        dd_score = np.clip((max_dd + 15) / 12 * 100, 0, 100)

        # Consecutive losses (20%): 100 if < 3, 0 if >= 10
        consec_score = np.clip((10 - consec_losses) / 7 * 100, 0, 100)

        # Recovery (20%): 100 if < 20 trades, 0 if never recovered
        if recovery_trades == -1:
            rec_score = 0.0
        elif recovery_trades == 0:
            rec_score = 100.0
        else:
            rec_score = np.clip((20 - recovery_trades) / 20 * 100, 0, 100)

        # Net R during event (20%): 100 if >= 0, 0 if <= -5
        r_score = np.clip((net_r_event + 5) / 5 * 100, 0, 100)

        return float(dd_score * 0.4 + consec_score * 0.2 + rec_score * 0.2 + r_score * 0.2)

    @staticmethod
    def _empty_survival() -> Dict:
        return {
            'survived': True,
            'max_drawdown_r': 0.0,
            'consecutive_losses': 0,
            'recovery_trades': 0,
            'net_r_during_event': 0.0,
            'net_r_total': 0.0,
            'worst_single_trade_r': 0.0,
            'trades_stopped_out': 0,
            'survival_score': 100.0,
        }

    @staticmethod
    def _extract_price_data(scenario_data: Dict) -> Dict:
        """Extract price series for dashboard charts (cap size)."""
        df = scenario_data['df']
        orig = scenario_data.get('original_df')

        # Subsample for JSON size
        max_points = 500
        if len(df) > max_points:
            idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
        else:
            idx = np.arange(len(df))

        result = {
            'synthetic_close': df['Close'].iloc[idx].tolist(),
            'synthetic_times': [str(t) for t in df.index[idx]],
            'event_start_idx': scenario_data['event_start_idx'],
            'event_end_idx': scenario_data['event_end_idx'],
        }

        if orig is not None and len(orig) > 0:
            if len(orig) > max_points:
                oidx = np.linspace(0, len(orig) - 1, max_points, dtype=int)
            else:
                oidx = np.arange(len(orig))
            result['original_close'] = orig['Close'].iloc[oidx].tolist()
            result['original_times'] = [str(t) for t in orig.index[oidx]]

        return result

    def _build_survival_matrix(self, scenario_results: List[Dict]) -> Dict:
        """Build heatmap data: type × magnitude → survival_score."""
        type_set = []
        mag_set = set()
        score_map = {}

        for sc in scenario_results:
            t = sc.get('type', '')
            m = sc.get('magnitude', 1.0)
            s = sc.get('survival_metrics', {}).get('survival_score', 0)
            if t not in type_set:
                type_set.append(t)
            mag_set.add(m)
            score_map[(t, m)] = s

        magnitudes = sorted(mag_set)
        scores = []
        for t in type_set:
            row = [score_map.get((t, m), 0) for m in magnitudes]
            scores.append(row)

        return {
            'types': type_set,
            'magnitudes': magnitudes,
            'scores': scores,
        }

    def _get_adapter(self):
        if self._adapter is None:
            from .strategy_adapters import ADAPTER_REGISTRY
            adapter_cls = ADAPTER_REGISTRY.get(self.config.strategy_name)
            if adapter_cls is None:
                raise ValueError(
                    f"Unknown strategy: {self.config.strategy_name}. "
                    f"Available: {list(ADAPTER_REGISTRY.keys())}"
                )
            self._adapter = adapter_cls()
        return self._adapter

    def _get_best_params(self) -> Dict:
        if self._best_params is not None:
            return self._best_params

        from .persistence import load_wfo_result, load_latest_wfo

        wfo = None
        if self.config.wfo_filepath:
            wfo = load_wfo_result(self.config.wfo_filepath)
        else:
            wfo = load_latest_wfo(
                strategy=self.config.strategy_name,
                symbol=self.config.symbol,
            )

        if wfo:
            self._wfo_result = wfo
            # Strategy 1: last window's best params
            param_hist = wfo.get('param_history', [])
            if param_hist:
                self._best_params = param_hist[-1].get('best_params', {})
                return self._best_params

            # Strategy 2: volatile regime params
            by_regime = wfo.get('best_params_by_regime', {})
            if 'volatile' in by_regime:
                self._best_params = by_regime['volatile']
                return self._best_params

        # Fallback: adapter defaults
        self._best_params = self._get_adapter().get_default_params()
        return self._best_params

    @staticmethod
    def generate_default_scenarios(
        crash_keys: Optional[List[str]] = None,
    ) -> List[ScenarioConfig]:
        """Generate a standard battery of stress scenarios."""
        if crash_keys is None:
            crash_keys = ['may_2021', 'nov_2022']

        scenarios = []

        # Historical crashes at 1x, 1.5x, 2x
        for key in crash_keys:
            crash = KNOWN_CRASHES.get(key)
            if crash is None:
                continue
            for mag in [1.0, 1.5, 2.0]:
                scenarios.append(ScenarioConfig(
                    scenario_type=ScenarioType.HISTORICAL_CRASH_SCALED,
                    name=f"{crash.name} ({mag}x)",
                    magnitude=mag,
                    crash_key=key,
                ))

        # Volatility spikes
        for mult in [2.0, 3.0, 5.0]:
            scenarios.append(ScenarioConfig(
                scenario_type=ScenarioType.VOLATILITY_SPIKE,
                name=f"Vol Spike {mult}x",
                magnitude=mult,
                vol_multiplier=mult,
                spike_duration_bars=50,
            ))

        # Flash crashes
        for drop in [-0.10, -0.15]:
            scenarios.append(ScenarioConfig(
                scenario_type=ScenarioType.FLASH_CRASH,
                name=f"Flash Crash {drop:.0%}",
                magnitude=abs(drop) / 0.10,
                flash_drop_pct=drop,
                flash_bars=3,
            ))

        # Prolonged drawdown
        scenarios.append(ScenarioConfig(
            scenario_type=ScenarioType.PROLONGED_DRAWDOWN,
            name="Prolonged Drawdown (200 bars)",
            magnitude=1.0,
            drawdown_bars=200,
            daily_drift_pct=-0.003,
        ))

        # V-shaped recovery
        scenarios.append(ScenarioConfig(
            scenario_type=ScenarioType.V_SHAPED_RECOVERY,
            name="V-Recovery (-30%)",
            magnitude=1.0,
            crash_depth_pct=-0.30,
            crash_duration_bars=20,
            recovery_duration_bars=20,
        ))

        return scenarios
