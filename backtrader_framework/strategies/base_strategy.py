"""
Base Strategy Class for Backtrader Framework.
Provides common functionality for all trading strategies.
"""

import backtrader as bt
from datetime import datetime
import pytz
from typing import Optional, Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backtrader_framework.config.settings import (
    ATR_PERIOD, SESSIONS, KILL_ZONES, COOLDOWN_BARS
)


class BaseStrategy(bt.Strategy):
    """
    Base class for all trading strategies.

    Features:
    - Timezone handling (UTC to ET conversion)
    - Session detection (Asia, London, NY)
    - Kill zone filtering
    - MFE/MAE tracking during trades
    - Trade logging with entry features
    - Cooldown management between trades
    - Stop loss and take profit handling
    """

    # Common parameters for all strategies
    params = (
        ('atr_period', ATR_PERIOD),
        ('atr_sl_multiplier', 2.0),
        ('commission', 0.001),
        ('slippage', 0.0005),
        ('cooldown_bars', COOLDOWN_BARS),
        ('max_hold_bars', 100),
        ('risk_per_trade', 0.01),  # 1% risk
    )

    def __init__(self):
        """Initialize base strategy components."""
        # ATR indicator (used by all strategies)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # Timezone setup
        self.utc_tz = pytz.UTC
        self.et_tz = pytz.timezone('America/New_York')

        # Trade tracking
        self.active_trade: Optional[Dict[str, Any]] = None
        self.trades_history: List[Dict[str, Any]] = []
        self.mfe_tracker: Dict[str, float] = {}
        self.mae_tracker: Dict[str, float] = {}

        # Cooldown tracking
        self.last_trade_bar = -100

        # Order tracking
        self.pending_order = None
        self.stop_order = None
        self.profit_order = None

    # ==========================================================================
    # Timezone and Session Methods
    # ==========================================================================

    def get_et_datetime(self, dt: datetime) -> datetime:
        """Convert datetime to ET timezone."""
        if dt.tzinfo is None:
            dt = self.utc_tz.localize(dt)
        return dt.astimezone(self.et_tz)

    def get_et_hour(self, dt: datetime) -> int:
        """Get hour in ET timezone."""
        return self.get_et_datetime(dt).hour

    def get_session(self, dt: datetime) -> str:
        """
        Determine trading session.

        Returns:
            'ASIA', 'LONDON', 'NEW_YORK', or 'OFF_HOURS'
        """
        hour = self.get_et_hour(dt)

        # Asia: 7pm - 3am ET (19-23, 0-2)
        if 19 <= hour or hour < 3:
            return 'ASIA'
        # London: 3am - 8am ET
        elif 3 <= hour < 8:
            return 'LONDON'
        # New York: 8am - 4pm ET
        elif 8 <= hour < 16:
            return 'NEW_YORK'

        return 'OFF_HOURS'

    def is_kill_zone(self, dt: datetime) -> bool:
        """Check if current time is in a trading kill zone."""
        hour = self.get_et_hour(dt)

        # London kill zone: 3-5am ET
        if 3 <= hour < 5:
            return True
        # NY kill zone: 8am - 4pm ET
        if 8 <= hour < 16:
            return True

        return False

    def in_cooldown(self) -> bool:
        """Check if still in cooldown period after last trade."""
        return len(self) - self.last_trade_bar < self.p.cooldown_bars

    # ==========================================================================
    # MFE/MAE Tracking
    # ==========================================================================

    def update_mfe_mae(self):
        """Track Maximum Favorable/Adverse Excursion during active trade."""
        if not self.position or not self.active_trade:
            return

        entry_price = self.active_trade['entry_price']
        trade_id = self.active_trade['id']

        if self.active_trade['direction'] == 'LONG':
            # MFE: highest price since entry
            favorable = self.data.high[0] - entry_price
            adverse = entry_price - self.data.low[0]
        else:
            # SHORT
            favorable = entry_price - self.data.low[0]
            adverse = self.data.high[0] - entry_price

        self.mfe_tracker[trade_id] = max(
            self.mfe_tracker.get(trade_id, 0),
            favorable
        )
        self.mae_tracker[trade_id] = max(
            self.mae_tracker.get(trade_id, 0),
            adverse
        )

    # ==========================================================================
    # Trade Logging
    # ==========================================================================

    def log_trade_entry(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        reason: str = None
    ):
        """
        Log trade entry details.

        Args:
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            reason: Entry reason/signal type
        """
        dt = self.data.datetime.datetime(0)
        trade_id = f"{self.__class__.__name__}_{dt.isoformat()}"

        self.active_trade = {
            'id': trade_id,
            'strategy_name': self.__class__.__name__,
            'symbol': self.data._name if hasattr(self.data, '_name') else 'UNKNOWN',
            'timeframe': getattr(self.data, '_timeframe', '15m'),
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': dt,
            'entry_bar': len(self),
            'stop_loss': stop_loss,
            'take_profit_1': take_profit,
            'take_profit_2': None,
            'atr_at_entry': self.atr[0],
            'session': self.get_session(dt),
            'entry_reason': reason,
        }

        # Initialize MFE/MAE tracking
        self.mfe_tracker[trade_id] = 0
        self.mae_tracker[trade_id] = 0

        # Update cooldown
        self.last_trade_bar = len(self)

        self.log(f"ENTRY {direction}: {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

    def log_trade_exit(self, exit_price: float, exit_reason: str):
        """
        Log trade exit and calculate final metrics.

        Args:
            exit_price: Exit price
            exit_reason: 'STOP_LOSS', 'TAKE_PROFIT', 'TIME_EXIT', etc.
        """
        if not self.active_trade:
            return

        trade = self.active_trade
        trade_id = trade['id']

        trade['exit_price'] = exit_price
        trade['exit_time'] = self.data.datetime.datetime(0)
        trade['exit_reason'] = exit_reason
        trade['bars_held'] = len(self) - trade['entry_bar']

        # Calculate P&L
        entry_price = trade['entry_price']
        risk = abs(entry_price - trade['stop_loss'])

        if trade['direction'] == 'LONG':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        trade['pnl_percent'] = (pnl / entry_price) * 100
        trade['r_multiple'] = pnl / risk if risk > 0 else 0

        # MFE/MAE final values
        mfe = self.mfe_tracker.get(trade_id, 0)
        mae = self.mae_tracker.get(trade_id, 0)

        trade['mfe_price'] = entry_price + mfe if trade['direction'] == 'LONG' else entry_price - mfe
        trade['mae_price'] = entry_price - mae if trade['direction'] == 'LONG' else entry_price + mae
        trade['mfe_percent'] = (mfe / entry_price) * 100
        trade['mae_percent'] = (mae / entry_price) * 100
        trade['mfe_r'] = mfe / risk if risk > 0 else 0
        trade['mae_r'] = mae / risk if risk > 0 else 0

        self.trades_history.append(trade)

        self.log(f"EXIT {trade['direction']}: {exit_price:.2f}, P&L: {trade['pnl_percent']:.2f}%, R: {trade['r_multiple']:.2f}")

        self.active_trade = None

    # ==========================================================================
    # Order Management
    # ==========================================================================

    def cancel_pending_orders(self):
        """Cancel all pending orders."""
        if self.stop_order:
            self.cancel(self.stop_order)
            self.stop_order = None
        if self.profit_order:
            self.cancel(self.profit_order)
            self.profit_order = None

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"BUY EXECUTED: {order.executed.price:.2f}")
            else:
                self.log(f"SELL EXECUTED: {order.executed.price:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")

    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            self.log(f"Trade P&L: Gross={trade.pnl:.2f}, Net={trade.pnlcomm:.2f}")

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def log(self, txt: str):
        """Log message with timestamp."""
        dt = self.data.datetime.datetime(0)
        print(f"{dt.isoformat()} - {txt}")

    def get_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk.

        Args:
            entry_price: Intended entry price
            stop_loss: Stop loss price

        Returns:
            Position size in units
        """
        risk_amount = self.broker.get_cash() * self.p.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            return 0

        size = risk_amount / risk_per_unit
        return size

    # ==========================================================================
    # Main Strategy Loop
    # ==========================================================================

    def next(self):
        """
        Main strategy loop - called for each bar.
        Override in subclass to implement strategy logic.
        """
        # Update MFE/MAE for active trades
        if self.position:
            self.update_mfe_mae()

            # Check for time-based exit
            if self.active_trade:
                bars_held = len(self) - self.active_trade['entry_bar']
                if bars_held >= self.p.max_hold_bars:
                    self.close()
                    self.log_trade_exit(self.data.close[0], 'TIME_EXIT')

    def stop(self):
        """Called when backtest ends."""
        # Print summary
        total_trades = len(self.trades_history)
        if total_trades == 0:
            print(f"\n{self.__class__.__name__}: No trades executed")
            return

        wins = sum(1 for t in self.trades_history if t['r_multiple'] > 0)
        losses = total_trades - wins
        total_r = sum(t['r_multiple'] for t in self.trades_history)
        win_rate = (wins / total_trades) * 100

        print(f"\n{'=' * 50}")
        print(f"{self.__class__.__name__} Summary:")
        print(f"{'=' * 50}")
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins}, Losses: {losses}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total R: {total_r:.2f}")
        print(f"Average R: {total_r / total_trades:.3f}")
        print(f"{'=' * 50}")
