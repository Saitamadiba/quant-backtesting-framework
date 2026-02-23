"""
Single Strategy Backtest Runner.
Runs backtest for one strategy with proper configuration.
"""

import backtrader as bt
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Type
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backtrader_framework.data.duckdb_manager import DuckDBManager
from backtrader_framework.strategies.fvg_strategy import FVGStrategy
from backtrader_framework.strategies.liquidity_raid_strategy import LiquidityRaidStrategy
from backtrader_framework.strategies.momentum_mastery_strategy import MomentumMasteryStrategy
from backtrader_framework.strategies.sbs_strategy import SBSStrategy
from backtrader_framework.config.settings import DEFAULT_INITIAL_CASH, DEFAULT_COMMISSION


# Strategy registry
STRATEGIES = {
    'FVG': FVGStrategy,
    'LiquidityRaid': LiquidityRaidStrategy,
    'MomentumMastery': MomentumMasteryStrategy,
    'SBS': SBSStrategy,
}

# Default timeframes for each strategy
STRATEGY_TIMEFRAMES = {
    'FVG': '15m',
    'LiquidityRaid': '15m',
    'MomentumMastery': '15m',
    'SBS': '4h',
}


def run_backtest(
    strategy_name: str,
    symbol: str = 'BTC',
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    commission: float = DEFAULT_COMMISSION,
    plot: bool = False,
    **strategy_params
) -> Dict[str, Any]:
    """
    Run a single strategy backtest.

    Args:
        strategy_name: Strategy name ('FVG', 'LiquidityRaid', 'MomentumMastery', 'SBS')
        symbol: Trading symbol ('BTC', 'ETH')
        timeframe: Candle timeframe (defaults to strategy's preferred timeframe)
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        initial_cash: Starting capital
        commission: Commission rate
        plot: Whether to show chart
        **strategy_params: Strategy-specific parameters

    Returns:
        dict: Backtest results including trades and performance metrics
    """
    # Get strategy class
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")

    strategy_class = STRATEGIES[strategy_name]
    timeframe = timeframe or STRATEGY_TIMEFRAMES[strategy_name]

    # Load data from DuckDB
    db = DuckDBManager()
    df = db.get_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        include_indicators=True
    )
    db.close()

    if df.empty:
        raise ValueError(f"No data found for {symbol} {timeframe}")

    # Prepare DataFrame for Backtrader
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)

    # Create Cerebro engine (stdstats=False to avoid writer errors with custom data)
    cerebro = bt.Cerebro(stdstats=False)

    # Add strategy
    cerebro.addstrategy(strategy_class, **strategy_params)

    # Create data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )

    # Store metadata on data feed (use Backtrader constants for timeframe)
    data._name = symbol

    cerebro.adddata(data)

    # Broker settings
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Print header
    print(f"\n{'=' * 60}")
    print(f"BACKTEST: {strategy_name} on {symbol} {timeframe}")
    print(f"{'=' * 60}")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Data points: {len(df)}")
    print(f"Initial cash: ${initial_cash:,.2f}")
    print(f"Commission: {commission * 100:.2f}%")
    print(f"{'=' * 60}\n")

    # Run backtest
    import warnings
    warnings.filterwarnings('ignore')
    results = cerebro.run()
    strategy = results[0]

    # Extract analyzer results
    trade_analysis = strategy.analyzers.trades.get_analysis()
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()

    # Get trade history from strategy
    trades_history = strategy.trades_history

    # Calculate summary metrics
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    won = trade_analysis.get('won', {}).get('total', 0)
    lost = trade_analysis.get('lost', {}).get('total', 0)
    win_rate = won / total_trades * 100 if total_trades > 0 else 0

    # Calculate total R from trades history
    total_r = sum(t.get('r_multiple', 0) for t in trades_history)
    avg_r = total_r / total_trades if total_trades > 0 else 0

    summary = {
        'strategy': strategy_name,
        'symbol': symbol,
        'timeframe': timeframe,
        'start_date': str(df.index[0]),
        'end_date': str(df.index[-1]),
        'total_trades': total_trades,
        'won': won,
        'lost': lost,
        'win_rate': win_rate,
        'total_r': total_r,
        'avg_r': avg_r,
        'sharpe_ratio': sharpe.get('sharperatio') or 0,
        'max_drawdown': drawdown.get('max', {}).get('drawdown', 0),
        'total_return': (returns.get('rtot') or 0) * 100,
        'final_value': cerebro.broker.getvalue(),
    }

    # Plot if requested
    if plot:
        cerebro.plot()

    return {
        'summary': summary,
        'trades': trades_history,
        'cerebro': cerebro,
    }


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary."""
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {summary['strategy']} on {summary['symbol']} {summary['timeframe']}")
    print(f"{'=' * 60}")
    print(f"Period: {summary['start_date']} to {summary['end_date']}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Won: {summary['won']}, Lost: {summary['lost']}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Total R: {summary['total_r']:.2f}")
    print(f"Average R: {summary['avg_r']:.3f}")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
    print(f"Total Return: {summary['total_return']:.2f}%")
    print(f"Final Value: ${summary['final_value']:,.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single strategy backtest")
    parser.add_argument("--strategy", "-s", required=True, choices=list(STRATEGIES.keys()),
                        help="Strategy to backtest")
    parser.add_argument("--symbol", default="BTC", help="Symbol (BTC, ETH)")
    parser.add_argument("--timeframe", "-tf", help="Timeframe (15m, 1h, 4h)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=10000, help="Initial cash")
    parser.add_argument("--plot", action="store_true", help="Show chart")

    args = parser.parse_args()

    result = run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        plot=args.plot
    )

    print_summary(result['summary'])
