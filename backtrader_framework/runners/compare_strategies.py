"""
Strategy Comparison Runner.
Runs all strategies and compares their performance.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backtrader_framework.runners.single_backtest import (
    run_backtest, STRATEGIES, STRATEGY_TIMEFRAMES, print_summary
)
from backtrader_framework.data.duckdb_manager import DuckDBManager


def run_all_strategies(
    symbols: List[str] = ['BTC', 'ETH'],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_cash: float = 10000.0,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run all strategies on all symbols and compare results.

    Args:
        symbols: List of symbols to test
        start_date: Backtest start date
        end_date: Backtest end date
        initial_cash: Starting capital per strategy
        save_results: Whether to save results to DuckDB

    Returns:
        dict: All results and comparison data
    """
    all_results = {}
    all_summaries = []

    strategies = [
        ('FVG', '15m'),
        ('LiquidityRaid', '15m'),
        ('MomentumMastery', '15m'),
        ('SBS', '4h'),
    ]

    print("\n" + "=" * 80)
    print("RUNNING ALL STRATEGY BACKTESTS")
    print("=" * 80)

    for strategy_name, timeframe in strategies:
        for symbol in symbols:
            key = f"{strategy_name}_{symbol}"
            print(f"\nRunning {key}...")

            try:
                result = run_backtest(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=initial_cash
                )

                all_results[key] = result
                all_summaries.append(result['summary'])

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[key] = {'error': str(e)}

    # Create comparison DataFrame
    if all_summaries:
        comparison_df = pd.DataFrame(all_summaries)
        comparison_df = comparison_df.sort_values('total_r', ascending=False)
    else:
        comparison_df = pd.DataFrame()

    # Print comparison table
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)
    print(f"{'Strategy':<20} {'Symbol':<8} {'Trades':<8} {'Win Rate':<10} "
          f"{'Total R':<10} {'Avg R':<8} {'Sharpe':<8} {'Return':<10}")
    print("-" * 100)

    for _, row in comparison_df.iterrows():
        print(f"{row['strategy']:<20} {row['symbol']:<8} {row['total_trades']:<8} "
              f"{row['win_rate']:<10.1f}% {row['total_r']:<10.2f} {row['avg_r']:<8.3f} "
              f"{row['sharpe_ratio']:<8.2f} {row['total_return']:<10.2f}%")

    print("=" * 100)

    # Save results to DuckDB
    if save_results and all_summaries:
        save_backtest_results(all_results)

    return {
        'results': all_results,
        'comparison': comparison_df,
        'summaries': all_summaries,
    }


def save_backtest_results(results: Dict[str, Any]):
    """Save backtest trades to DuckDB."""
    db = DuckDBManager()

    trades_to_insert = []

    for key, result in results.items():
        if 'error' in result:
            continue

        trades = result.get('trades', [])
        for trade in trades:
            trades_to_insert.append(trade)

    if trades_to_insert:
        db.insert_trades(trades_to_insert)
        print(f"\nSaved {len(trades_to_insert)} trades to DuckDB")

    db.close()


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed results for each strategy."""
    for key, result in results.items():
        if 'error' in result:
            print(f"\n{key}: ERROR - {result['error']}")
            continue

        print_summary(result['summary'])


def analyze_by_session(results: Dict[str, Any]) -> pd.DataFrame:
    """Analyze results grouped by trading session."""
    session_data = []

    for key, result in results.items():
        if 'error' in result:
            continue

        for trade in result.get('trades', []):
            session_data.append({
                'strategy': result['summary']['strategy'],
                'symbol': result['summary']['symbol'],
                'session': trade.get('session', 'UNKNOWN'),
                'r_multiple': trade.get('r_multiple', 0),
                'win': 1 if trade.get('r_multiple', 0) > 0 else 0,
            })

    if not session_data:
        return pd.DataFrame()

    df = pd.DataFrame(session_data)

    session_summary = df.groupby(['strategy', 'session']).agg({
        'r_multiple': ['count', 'sum', 'mean'],
        'win': 'sum'
    }).round(3)

    session_summary.columns = ['trades', 'total_r', 'avg_r', 'wins']
    session_summary['win_rate'] = (session_summary['wins'] / session_summary['trades'] * 100).round(1)

    return session_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare all strategies")
    parser.add_argument("--symbols", nargs="+", default=['BTC', 'ETH'],
                        help="Symbols to test")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=10000, help="Initial cash")
    parser.add_argument("--no-save", action="store_true", help="Don't save to DuckDB")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")

    args = parser.parse_args()

    results = run_all_strategies(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        save_results=not args.no_save
    )

    if args.detailed:
        print_detailed_results(results['results'])

    # Show session analysis
    session_df = analyze_by_session(results['results'])
    if not session_df.empty:
        print("\n" + "=" * 80)
        print("PERFORMANCE BY SESSION")
        print("=" * 80)
        print(session_df.to_string())
