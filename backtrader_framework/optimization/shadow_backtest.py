"""Shadow Backtest: compare live bot execution against adapter signal replay.

Replays the strategy adapter's signal logic on the exact same candles the
live bot traded, matches each live trade to its simulated counterpart, and
computes per-trade execution gaps.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_fetcher import DataFetcher
from .indicators import IndicatorEngine
from .simulator import TradeSimulator
from .wfo_engine import TransactionCosts, TradeResult
from .strategy_adapters import ADAPTER_REGISTRY

logger = logging.getLogger(__name__)

# Project root
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dashboard strategy name -> adapter registry key
STRATEGY_NAME_MAP = {
    'Liquidity Raid': 'LiquidityRaidLive',
    'LiquidityRaid': 'LiquidityRaidLive',
    'LiquidityRaidLive': 'LiquidityRaidLive',
    'LiquidityRaidV3': 'LiquidityRaidV3',
    'Momentum Mastery': 'MomentumMastery',
    'MomentumMastery': 'MomentumMastery',
    'FVG': 'FVG',
    'SBS': 'SBS',
}

# Dashboard name -> (db_filename_prefix, normalizer_type)
_DB_STRATEGY_MAP = {
    'fvg_btc.db': ('FVG', 'BTC', 'fvg'),
    'fvg_eth.db': ('FVG', 'ETH', 'fvg'),
    'fvg_nq.db': ('FVG', 'NQ', 'fvg'),
    'lr_btc.db': ('Liquidity Raid', 'BTC', 'lr_mm'),
    'lr_eth.db': ('Liquidity Raid', 'ETH', 'lr_mm'),
    'mm_btc.db': ('Momentum Mastery', 'BTC', 'lr_mm'),
    'mm_eth.db': ('Momentum Mastery', 'ETH', 'lr_mm'),
    'sbs.db': ('SBS', 'ALL', 'sbs'),
}

# Unified trade schema columns (matches dashboard/config.py)
TRADE_SCHEMA_COLS = [
    'trade_id', 'strategy', 'symbol', 'timeframe', 'source',
    'direction', 'entry_time', 'exit_time',
    'entry_price', 'exit_price', 'stop_loss', 'take_profit',
    'pnl_usd', 'pnl_pct', 'r_multiple',
    'session', 'exit_reason', 'duration_minutes',
    'running_balance', 'mfe', 'mae', 'is_open',
]


# ── Live Trade Loading ────────────────────────────────────────────────────────

def _normalize_lr_mm_minimal(db_path: Path, strategy: str, symbol: str) -> pd.DataFrame:
    """Minimal LR/MM normalizer (avoids dashboard imports)."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            raw = pd.read_sql_query("SELECT * FROM trades", conn)
    except Exception as e:
        logger.error(f"Failed to load {db_path}: {e}")
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    if raw.empty:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    df = pd.DataFrame()
    df['trade_id'] = raw.get('id', raw.index).astype(str)
    df['strategy'] = strategy
    df['symbol'] = symbol
    df['timeframe'] = '15m'
    df['source'] = 'Live'

    sig_map = {
        'BUY': 'Long', 'SELL': 'Short', 'buy': 'Long', 'sell': 'Short',
        'LONG': 'Long', 'SHORT': 'Short', 'long': 'Long', 'short': 'Short',
    }
    df['direction'] = raw['signal_type'].str.strip().map(sig_map).fillna('Unknown')
    df['entry_time'] = pd.to_datetime(raw['timestamp'], errors='coerce')
    df['exit_time'] = pd.to_datetime(raw.get('exit_timestamp'), errors='coerce')
    df['entry_price'] = raw['entry_price'].astype(float)
    df['exit_price'] = raw.get('exit_price', pd.Series(dtype=float)).astype(float)
    df['stop_loss'] = raw.get('stop_loss', pd.Series(dtype=float)).astype(float)
    df['take_profit'] = raw.get('take_profit', pd.Series(dtype=float)).astype(float)
    df['pnl_usd'] = raw.get('realized_pnl', pd.Series(0.0)).astype(float)
    df['pnl_pct'] = raw.get('realized_pnl_pct', pd.Series(dtype=float)).astype(float)

    risk_per_unit = (df['entry_price'] - df['stop_loss']).abs()
    pnl_per_unit = df['exit_price'] - df['entry_price']
    pnl_per_unit = pnl_per_unit.where(df['direction'] == 'Long', -pnl_per_unit)
    df['r_multiple'] = pnl_per_unit / risk_per_unit.replace(0, float('nan'))

    df['session'] = raw.get('killzone', 'Unknown')
    df['exit_reason'] = raw.get('exit_reason', raw.get('reason', 'Unknown'))
    if df['entry_time'].notna().any() and df['exit_time'].notna().any():
        df['duration_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
    else:
        df['duration_minutes'] = None
    df['running_balance'] = None
    df['mfe'] = None
    df['mae'] = None

    if 'status' in raw.columns:
        df['is_open'] = raw['status'].str.strip().str.lower() == 'open'
    else:
        df['is_open'] = df['exit_time'].isna()

    # Ensure all schema columns exist
    for col in TRADE_SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None

    return df[TRADE_SCHEMA_COLS]


def load_live_trades_for_strategy(
    strategy: str,
    symbol: str,
    vps_cache_dir: str = None,
) -> pd.DataFrame:
    """Load closed live trades for a specific strategy/symbol from VPS cache.

    Args:
        strategy: Strategy name (dashboard or adapter registry format).
        symbol: Trading symbol ('BTC', 'ETH', etc.).
        vps_cache_dir: Path to dashboard/databases/. Auto-detected if None.

    Returns:
        DataFrame with TRADE_SCHEMA_COLS, filtered to closed trades only.
    """
    if vps_cache_dir is None:
        vps_cache_dir = os.path.join(_BASE, 'dashboard', 'databases')
    cache_dir = Path(vps_cache_dir)

    # Normalize strategy name to dashboard format
    name_to_dashboard = {
        'LiquidityRaidLive': 'Liquidity Raid',
        'LiquidityRaid': 'Liquidity Raid',
        'MomentumMastery': 'Momentum Mastery',
    }
    dash_name = name_to_dashboard.get(strategy, strategy)

    frames = []
    for db_file, (db_strat, db_sym, norm_type) in _DB_STRATEGY_MAP.items():
        if db_strat != dash_name:
            continue
        if symbol != 'ALL' and db_sym != 'ALL' and db_sym != symbol:
            continue

        db_path = cache_dir / db_file
        if not db_path.exists():
            logger.warning(f"DB not found: {db_path}")
            continue

        if norm_type == 'lr_mm':
            df = _normalize_lr_mm_minimal(db_path, db_strat, db_sym)
        else:
            logger.info(f"Skipping {db_file} (normalizer '{norm_type}' not implemented in shadow_backtest)")
            continue

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=TRADE_SCHEMA_COLS)

    result = pd.concat(frames, ignore_index=True)

    # Filter to closed trades only
    result = result[~result['is_open'].fillna(False)].copy()
    result = result.dropna(subset=['entry_time', 'exit_time'])

    return result


# ── Shadow Backtest ───────────────────────────────────────────────────────────

class ShadowBacktest:
    """Compare live bot execution against adapter signal replay on the same candles.

    Replays the adapter's generate_signals() / execute_signals() on DuckDB
    candles covering the live trading period, then matches each live trade
    to the nearest shadow trade by time and direction.
    """

    LOOKBACK_BUFFER = 500  # bars for indicator warmup

    def run(
        self,
        strategy: str,
        symbol: str,
        timeframe: str = '15m',
        params: dict = None,
    ) -> dict:
        """Run shadow backtest comparing live execution vs adapter replay.

        Args:
            strategy: Strategy name (dashboard or adapter registry format).
            symbol: Trading symbol ('BTC', 'ETH').
            timeframe: Candle timeframe (default '15m').
            params: Adapter parameters. If None, uses adapter defaults.

        Returns:
            Shadow backtest result dict.
        """
        # 1. Resolve adapter
        adapter_key = STRATEGY_NAME_MAP.get(strategy, strategy)
        if adapter_key not in ADAPTER_REGISTRY:
            return {
                'error': f"Unknown strategy '{strategy}'. "
                         f"Available: {list(ADAPTER_REGISTRY.keys())}",
            }

        # 2. Load live trades
        live_trades = load_live_trades_for_strategy(strategy, symbol)
        if live_trades.empty:
            return {
                'error': f'No closed live trades found for {strategy}/{symbol}',
                'strategy': strategy,
                'symbol': symbol,
            }

        logger.info(f"Loaded {len(live_trades)} closed live trades for {strategy}/{symbol}")

        # 3. Date range
        start = live_trades['entry_time'].min()
        end = live_trades['exit_time'].max()

        # 4. Fetch candles
        df = self._fetch_candles(symbol, timeframe, start, end)
        if df is None or len(df) < 100:
            return {
                'error': f'Insufficient candle data for {symbol}/{timeframe}',
                'strategy': strategy,
                'symbol': symbol,
            }

        logger.info(f"Fetched {len(df)} candles covering {df.index[0]} to {df.index[-1]}")

        # 5. Instantiate adapter
        adapter_cls = ADAPTER_REGISTRY[adapter_key]
        if adapter_key == 'LiquidityRaidLive':
            adapter = adapter_cls(timeframe=timeframe)
        else:
            adapter = adapter_cls()

        # 6. Params
        if params is None:
            params = adapter.get_default_params()

        # 7. Scan range: from first live entry to last live exit
        start_idx = df.index.get_indexer([start], method='nearest')[0]
        end_idx = df.index.get_indexer([end], method='nearest')[0]
        scan_start = max(0, start_idx)
        scan_end = min(len(df), end_idx + 1)

        if scan_end - scan_start < 10:
            return {
                'error': 'Scan range too small',
                'strategy': strategy,
                'symbol': symbol,
            }

        # 8. Generate shadow trades
        shadow_trades = self._generate_shadow_trades(
            adapter, df, params, scan_start, scan_end, symbol, timeframe,
        )

        logger.info(f"Generated {len(shadow_trades)} shadow trades")

        # 9. Match trades
        matched, missed, phantom = self._match_trades(live_trades, shadow_trades, df)

        # 10. Aggregate metrics
        aggregate = self._compute_aggregates(matched)

        return {
            'strategy': strategy,
            'symbol': symbol,
            'timeframe': timeframe,
            'params_used': params,
            'live_period': {
                'start': str(start),
                'end': str(end),
            },
            'n_live_trades': len(live_trades),
            'n_shadow_trades': len(shadow_trades),
            'n_matched': len(matched),
            'n_missed': len(missed),
            'n_phantom': len(phantom),
            'match_rate': round(len(matched) / max(len(live_trades), 1), 4),
            'matched_trades': matched,
            'missed_trades': missed,
            'phantom_trades': phantom,
            'aggregate': aggregate,
        }

    def _fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """Fetch DuckDB candles covering [start - lookback, end] with indicators."""
        raw = DataFetcher.fetch(symbol, timeframe)
        if raw is None or raw.empty:
            return None

        # Add lookback buffer for indicator warmup
        tf_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}.get(timeframe, 15)
        lookback_td = pd.Timedelta(minutes=tf_minutes * self.LOOKBACK_BUFFER)

        # Ensure start is timezone-naive for comparison
        start_naive = start.tz_localize(None) if start.tzinfo else start
        end_naive = end.tz_localize(None) if end.tzinfo else end

        # Also make index timezone-naive if needed
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)

        sliced = raw.loc[start_naive - lookback_td: end_naive].copy()
        if len(sliced) < 100:
            return None

        # Calculate indicators
        df = IndicatorEngine.calculate(sliced)
        df = df.dropna(subset=['ATR'])

        return df

    def _generate_shadow_trades(
        self,
        adapter,
        df: pd.DataFrame,
        params: dict,
        scan_start: int,
        scan_end: int,
        symbol: str,
        timeframe: str,
    ) -> List[TradeResult]:
        """Run the adapter on candle data to produce shadow trades."""
        costs = TransactionCosts.for_asset(symbol)

        # Try execute_signals first (stateful adapters like LiquidityRaidLive)
        trades = adapter.execute_signals(
            df, params, scan_start, scan_end,
            costs, max_bars=168, window_id=0, is_oos=True, regime='unknown',
        )
        if trades is not None:
            return trades

        # Fallback: generate_signals + simulate
        signals = adapter.generate_signals(df, params, scan_start, scan_end)
        trades = []
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        atrs = df['ATR'].values if 'ATR' in df.columns else None

        for sig in signals:
            trade = TradeSimulator.simulate(
                sig.to_dict(), df, costs, max_bars=168,
                window_id=0, is_oos=True, regime='unknown',
                _highs=highs, _lows=lows, _closes=closes, _atrs=atrs,
            )
            if trade is not None:
                trades.append(trade)

        return trades

    def _match_trades(
        self,
        live_trades: pd.DataFrame,
        shadow_trades: List[TradeResult],
        df: pd.DataFrame,
        bar_tolerance: int = 8,
    ) -> tuple:
        """Match live trades to shadow trades by time proximity and direction.
        Default ±8 bars (±2h on 15m) accounts for 5M execution timing.

        Returns:
            (matched_list, missed_list, phantom_list)
        """
        dir_map = {'Long': 'LONG', 'Short': 'SHORT'}

        # Convert shadow trades to dicts with entry bar index
        shadow_entries = []
        for st in shadow_trades:
            st_time = st.entry_time
            if st_time is not None:
                if hasattr(st_time, 'tz') and st_time.tz is not None:
                    st_time = st_time.tz_localize(None)
                idx = df.index.get_indexer([st_time], method='nearest')[0]
            else:
                idx = -1
            shadow_entries.append({
                'trade': st,
                'bar_idx': idx,
                'direction': st.direction,
                'matched': False,
            })

        matched = []
        missed = []
        shadow_used = set()

        for _, live_row in live_trades.iterrows():
            live_time = live_row['entry_time']
            if hasattr(live_time, 'tz') and live_time.tz is not None:
                live_time = live_time.tz_localize(None)
            live_bar = df.index.get_indexer([live_time], method='nearest')[0]
            live_dir = dir_map.get(live_row['direction'], live_row['direction'])

            best_match = None
            best_dist = float('inf')

            for si, se in enumerate(shadow_entries):
                if si in shadow_used:
                    continue
                if se['direction'] != live_dir:
                    continue
                dist = abs(se['bar_idx'] - live_bar)
                if dist <= bar_tolerance and dist < best_dist:
                    best_dist = dist
                    best_match = si

            if best_match is not None:
                shadow_used.add(best_match)
                st = shadow_entries[best_match]['trade']

                live_entry = float(live_row['entry_price'])
                live_exit = float(live_row['exit_price']) if pd.notna(live_row['exit_price']) else None
                live_r = float(live_row['r_multiple']) if pd.notna(live_row['r_multiple']) else None
                shadow_entry = st.entry_price
                shadow_exit = st.exit_price
                shadow_r = st.r_multiple_after_costs

                entry_gap = (live_entry - shadow_entry) / shadow_entry if shadow_entry else 0
                exit_gap = (
                    (live_exit - shadow_exit) / shadow_exit
                    if live_exit and shadow_exit else None
                )
                r_gap = live_r - shadow_r if live_r is not None else None
                outcome_match = (
                    (live_r > 0) == (shadow_r > 0)
                    if live_r is not None else None
                )

                matched.append({
                    'live': {
                        'entry_time': str(live_row['entry_time']),
                        'direction': live_row['direction'],
                        'entry_price': live_entry,
                        'exit_price': live_exit,
                        'r_multiple': live_r,
                        'exit_reason': live_row.get('exit_reason'),
                    },
                    'shadow': {
                        'entry_time': str(st.entry_time),
                        'direction': st.direction,
                        'entry_price': shadow_entry,
                        'exit_price': shadow_exit,
                        'r_multiple': round(shadow_r, 4),
                        'outcome': st.outcome,
                    },
                    'bar_distance': best_dist,
                    'entry_gap_pct': round(entry_gap, 6),
                    'exit_gap_pct': round(exit_gap, 6) if exit_gap is not None else None,
                    'r_gap': round(r_gap, 4) if r_gap is not None else None,
                    'outcome_match': outcome_match,
                })
            else:
                missed.append({
                    'entry_time': str(live_row['entry_time']),
                    'direction': live_row['direction'],
                    'entry_price': float(live_row['entry_price']),
                    'r_multiple': (
                        float(live_row['r_multiple'])
                        if pd.notna(live_row['r_multiple']) else None
                    ),
                    'exit_reason': live_row.get('exit_reason'),
                })

        # Phantom trades (shadow only)
        phantom = []
        for si, se in enumerate(shadow_entries):
            if si not in shadow_used:
                st = se['trade']
                phantom.append({
                    'entry_time': str(st.entry_time),
                    'direction': st.direction,
                    'entry_price': st.entry_price,
                    'r_multiple': round(st.r_multiple_after_costs, 4),
                    'outcome': st.outcome,
                })

        return matched, missed, phantom

    def _compute_aggregates(self, matched: List[dict]) -> dict:
        """Compute aggregate metrics from matched trades."""
        if not matched:
            return {
                'mean_entry_gap_pct': None,
                'mean_exit_gap_pct': None,
                'mean_r_gap': None,
                'total_execution_drag_r': None,
                'outcome_agreement_pct': None,
            }

        entry_gaps = [m['entry_gap_pct'] for m in matched]
        exit_gaps = [m['exit_gap_pct'] for m in matched if m['exit_gap_pct'] is not None]
        r_gaps = [m['r_gap'] for m in matched if m['r_gap'] is not None]
        outcomes = [m['outcome_match'] for m in matched if m['outcome_match'] is not None]

        return {
            'mean_entry_gap_pct': round(float(np.mean(entry_gaps)), 6) if entry_gaps else None,
            'mean_exit_gap_pct': round(float(np.mean(exit_gaps)), 6) if exit_gaps else None,
            'mean_r_gap': round(float(np.mean(r_gaps)), 4) if r_gaps else None,
            'total_execution_drag_r': round(float(np.sum(r_gaps)), 4) if r_gaps else None,
            'outcome_agreement_pct': round(float(np.mean(outcomes)), 4) if outcomes else None,
        }

    def save_results(self, result: dict, output_dir: str = None) -> str:
        """Save shadow backtest results as JSON."""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strat = result.get('strategy', 'unknown').replace(' ', '_')
        sym = result.get('symbol', 'unknown')
        filename = f"shadow_{strat}_{sym}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        return filepath
