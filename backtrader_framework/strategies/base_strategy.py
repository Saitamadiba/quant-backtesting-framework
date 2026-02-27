"""
Base Strategy Class for Backtrader Framework.
Provides common functionality for all trading strategies.
"""

import backtrader as bt
import logging
from datetime import datetime
import pytz
from typing import Optional, Dict, Any, List

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
        ('max_position_pct', 0.25),  # Max 25% of NAV per position
        ('max_drawdown_pct', 0.20),  # 20% drawdown circuit breaker
        ('max_daily_loss_pct', 0.05),  # 5% daily loss limit
    )

    def __init__(self):
        """Initialize base strategy components."""
        # Logger setup
        self.logger = logging.getLogger(self.__class__.__name__)

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

        # Drawdown circuit breaker
        self.peak_value = self.broker.getvalue()
        self.trading_halted = False

        # Daily loss limit tracking
        self.daily_start_value = self.broker.getvalue()
        self.last_trading_date = None

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

        # Clean up MFE/MAE trackers to prevent memory leak
        if trade_id in self.mfe_tracker:
            del self.mfe_tracker[trade_id]
        if trade_id in self.mae_tracker:
            del self.mae_tracker[trade_id]

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
        """Handle order notifications with proper order reference tracking."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"BUY EXECUTED: {order.executed.price:.2f}, Size: {order.executed.size:.4f}")
            else:
                self.log(f"SELL EXECUTED: {order.executed.price:.2f}, Size: {order.executed.size:.4f}")

            # Track which order completed and clear its reference
            if order == self.pending_order:
                self.pending_order = None
            elif order == self.stop_order:
                self.stop_order = None
                # Stop was hit — cancel the take profit order
                if self.profit_order:
                    self.cancel(self.profit_order)
                    self.profit_order = None
            elif order == self.profit_order:
                self.profit_order = None
                # TP was hit — cancel the stop loss order
                if self.stop_order:
                    self.cancel(self.stop_order)
                    self.stop_order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")
            # Clear reference for the failed order
            if order == self.pending_order:
                self.pending_order = None
            elif order == self.stop_order:
                self.stop_order = None
            elif order == self.profit_order:
                self.profit_order = None

    def notify_trade(self, trade):
        """Handle trade notifications and capture actual close price."""
        if trade.isclosed:
            actual_close_price = trade.price
            self.log(f"Trade P&L: Gross={trade.pnl:.2f}, Net={trade.pnlcomm:.2f}, ClosePrice={actual_close_price:.2f}")

            # Determine exit reason from which order triggered the close
            if self.active_trade:
                # Infer exit reason based on trade outcome
                if self.stop_order is None and self.profit_order is not None:
                    exit_reason = 'STOP_LOSS'
                elif self.profit_order is None and self.stop_order is not None:
                    exit_reason = 'TAKE_PROFIT'
                else:
                    exit_reason = 'CLOSE'
                self.log_trade_exit(actual_close_price, exit_reason)

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def log(self, txt: str):
        """Log message with timestamp."""
        dt = self.data.datetime.datetime(0)
        self.logger.info(f"{dt.isoformat()} - {txt}")

    def get_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk using portfolio NAV.

        Args:
            entry_price: Intended entry price
            stop_loss: Stop loss price

        Returns:
            Position size in units
        """
        nav = self.broker.getvalue()
        risk_amount = nav * self.p.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            return 0

        size = risk_amount / risk_per_unit

        # Cap position size to max_position_pct of NAV
        max_size = nav * self.p.max_position_pct / entry_price
        size = min(size, max_size)

        return size

    # ==========================================================================
    # Main Strategy Loop
    # ==========================================================================

    def next(self):
        """
        Main strategy loop - called for each bar.
        Override in subclass to implement strategy logic.
        """
        # --- Drawdown circuit breaker (FIX 4) ---
        current_value = self.broker.getvalue()
        self.peak_value = max(self.peak_value, current_value)
        drawdown = (self.peak_value - current_value) / self.peak_value
        if drawdown >= self.p.max_drawdown_pct:
            if not self.trading_halted:
                self.log(f"CIRCUIT BREAKER: {drawdown:.1%} drawdown exceeds {self.p.max_drawdown_pct:.1%} limit")
                self.trading_halted = True
            if self.position:
                self.close()
            return

        # --- Daily loss limit (FIX 5) ---
        current_date = self.data.datetime.date(0)
        if self.last_trading_date is None or current_date != self.last_trading_date:
            # New trading day — reset daily tracking
            self.daily_start_value = current_value
            self.last_trading_date = current_date

        daily_loss = (self.daily_start_value - current_value) / self.daily_start_value
        if daily_loss >= self.p.max_daily_loss_pct:
            if self.position:
                self.close()
            return

        # Update MFE/MAE for active trades
        if self.position:
            self.update_mfe_mae()

            # Check for time-based exit (exit logged in notify_trade)
            if self.active_trade:
                bars_held = len(self) - self.active_trade['entry_bar']
                if bars_held >= self.p.max_hold_bars:
                    self.close()

    def stop(self):
        """Called when backtest ends."""
        # Log summary
        total_trades = len(self.trades_history)
        if total_trades == 0:
            self.logger.info(f"{self.__class__.__name__}: No trades executed")
            return

        wins = sum(1 for t in self.trades_history if t['r_multiple'] > 0)
        losses = total_trades - wins
        total_r = sum(t['r_multiple'] for t in self.trades_history)
        win_rate = (wins / total_trades) * 100

        self.logger.info(f"{'=' * 50}")
        self.logger.info(f"{self.__class__.__name__} Summary:")
        self.logger.info(f"{'=' * 50}")
        self.logger.info(f"Total Trades: {total_trades}")
        self.logger.info(f"Wins: {wins}, Losses: {losses}")
        self.logger.info(f"Win Rate: {win_rate:.1f}%")
        self.logger.info(f"Total R: {total_r:.2f}")
        self.logger.info(f"Average R: {total_r / total_trades:.3f}")
        self.logger.info(f"{'=' * 50}")
