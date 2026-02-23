"""
Sweep Detector Indicator for Backtrader.

Detects liquidity sweeps where price wicks beyond a level
and closes back inside, indicating a potential reversal.
"""

import backtrader as bt


class SweepDetector(bt.Indicator):
    """
    Detects liquidity sweeps of recent highs/lows.

    A sweep occurs when:
        1. Price wicks beyond a level (breaks it intrabar)
        2. Price closes back inside the level
        3. Previous bar was cleanly above/below the level

    This indicates that stop losses were triggered (liquidity taken)
    and price may reverse.

    Lines:
        - bullish_sweep: 1 when sweep of lows detected (buy signal)
        - bearish_sweep: 1 when sweep of highs detected (sell signal)
        - sweep_level: The price level that was swept
        - sweep_depth: How far price went beyond the level
    """

    lines = ('bullish_sweep', 'bearish_sweep', 'sweep_level', 'sweep_depth')

    params = (
        ('lookback', 20),       # Bars to look back for high/low
        ('tolerance', 0.002),   # 0.2% tolerance for sweep detection
        ('min_depth_atr', 0.1), # Minimum sweep depth as % of ATR
    )

    plotinfo = dict(
        plot=False,
        subplot=False,
    )

    def __init__(self):
        # ATR for sweep depth validation
        self.atr = bt.indicators.ATR(self.data, period=14)

    def next(self):
        # Reset lines
        self.lines.bullish_sweep[0] = 0
        self.lines.bearish_sweep[0] = 0
        self.lines.sweep_level[0] = 0
        self.lines.sweep_depth[0] = 0

        # Need enough data
        if len(self) < self.p.lookback + 2:
            return

        # Current candle
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        current_close = self.data.close[0]
        current_open = self.data.open[0]

        # Previous candle
        prev_high = self.data.high[-1]
        prev_low = self.data.low[-1]

        # Get recent high/low from lookback (excluding current and previous)
        recent_highs = [self.data.high[-i] for i in range(2, self.p.lookback + 2)]
        recent_lows = [self.data.low[-i] for i in range(2, self.p.lookback + 2)]

        recent_high = max(recent_highs) if recent_highs else current_high
        recent_low = min(recent_lows) if recent_lows else current_low

        atr = self.atr[0] if self.atr[0] > 0 else 1

        # Check for bullish sweep (sweep of lows)
        # Conditions:
        # 1. Current low wicks below recent low
        # 2. Current close is above recent low
        # 3. Previous low was above or near recent low (clean)
        tolerance = recent_low * self.p.tolerance
        min_depth = atr * self.p.min_depth_atr

        if (current_low < recent_low - tolerance and
            current_close > recent_low and
            prev_low >= recent_low * (1 - self.p.tolerance)):

            sweep_depth = recent_low - current_low

            if sweep_depth >= min_depth:
                self.lines.bullish_sweep[0] = 1
                self.lines.sweep_level[0] = recent_low
                self.lines.sweep_depth[0] = sweep_depth
                return

        # Check for bearish sweep (sweep of highs)
        tolerance = recent_high * self.p.tolerance

        if (current_high > recent_high + tolerance and
            current_close < recent_high and
            prev_high <= recent_high * (1 + self.p.tolerance)):

            sweep_depth = current_high - recent_high

            if sweep_depth >= min_depth:
                self.lines.bearish_sweep[0] = 1
                self.lines.sweep_level[0] = recent_high
                self.lines.sweep_depth[0] = sweep_depth


class LiquiditySweepDetector(bt.Indicator):
    """
    Detects sweeps of specific session levels.

    Used with SessionTracker to detect sweeps of Asia/London/NY highs and lows.

    Lines:
        - sweep_type: 1=Asia Low, 2=Asia High, 3=London Low, 4=London High, etc.
        - sweep_direction: 1=Bullish (sweep of low), -1=Bearish (sweep of high)
    """

    lines = ('sweep_type', 'sweep_direction', 'swept_price')

    params = (
        ('tolerance', 0.002),
    )

    plotinfo = dict(
        plot=False,
        subplot=False,
    )

    def __init__(self):
        # Import SessionTracker
        from .session_tracker import SessionTracker
        self.sessions = SessionTracker(self.data)

    def next(self):
        # Reset
        self.lines.sweep_type[0] = 0
        self.lines.sweep_direction[0] = 0
        self.lines.swept_price[0] = 0

        current_high = self.data.high[0]
        current_low = self.data.low[0]
        current_close = self.data.close[0]
        prev_low = self.data.low[-1] if len(self) > 1 else current_low
        prev_high = self.data.high[-1] if len(self) > 1 else current_high

        # Check each session level
        levels = [
            (1, 'LOW', self.sessions.asia_low[0]),
            (2, 'HIGH', self.sessions.asia_high[0]),
            (3, 'LOW', self.sessions.london_low[0]),
            (4, 'HIGH', self.sessions.london_high[0]),
            (5, 'LOW', self.sessions.ny_low[0]),
            (6, 'HIGH', self.sessions.ny_high[0]),
        ]

        for level_type, direction, level_price in levels:
            if level_price <= 0 or level_price == float('inf'):
                continue

            tolerance = level_price * self.p.tolerance

            if direction == 'LOW':
                # Bullish sweep: wick below, close above
                if (current_low < level_price - tolerance and
                    current_close > level_price and
                    prev_low >= level_price * (1 - self.p.tolerance)):
                    self.lines.sweep_type[0] = level_type
                    self.lines.sweep_direction[0] = 1  # Bullish
                    self.lines.swept_price[0] = level_price
                    return
            else:  # HIGH
                # Bearish sweep: wick above, close below
                if (current_high > level_price + tolerance and
                    current_close < level_price and
                    prev_high <= level_price * (1 + self.p.tolerance)):
                    self.lines.sweep_type[0] = level_type
                    self.lines.sweep_direction[0] = -1  # Bearish
                    self.lines.swept_price[0] = level_price
                    return
