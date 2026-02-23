"""
Session Tracker Indicator for Backtrader.

Tracks trading session highs and lows for ICT-style strategies.
Sessions are defined in Eastern Time (ET).
"""

import backtrader as bt
import pytz
from collections import deque


class SessionTracker(bt.Indicator):
    """
    Tracks trading session highs and lows.

    Sessions (ET):
        - Asia: 7pm - 3am ET (19:00 - 03:00)
        - London: 3am - 8am ET (03:00 - 08:00)
        - New York: 8am - 4pm ET (08:00 - 16:00)

    Lines:
        - asia_high: Highest price during Asia session
        - asia_low: Lowest price during Asia session
        - london_high: Highest price during London session
        - london_low: Lowest price during London session
        - ny_high: Highest price during NY session
        - ny_low: Lowest price during NY session
        - current_session: Current session (1=Asia, 2=London, 3=NY, 0=Off)
    """

    lines = (
        'asia_high', 'asia_low',
        'london_high', 'london_low',
        'ny_high', 'ny_low',
        'current_session',
        'prev_day_high', 'prev_day_low'
    )

    params = (
        ('lookback_hours', 48),  # Hours to look back for session levels
    )

    plotinfo = dict(
        plot=False,
        subplot=False,
    )

    def __init__(self):
        self.utc_tz = pytz.UTC
        self.et_tz = pytz.timezone('America/New_York')

        # Store recent bars for session calculation
        self.bar_cache = deque(maxlen=500)

    def get_et_hour(self, dt) -> int:
        """Get hour in ET timezone."""
        if dt.tzinfo is None:
            dt = self.utc_tz.localize(dt)
        return dt.astimezone(self.et_tz).hour

    def get_session_code(self, dt) -> int:
        """
        Get session code for a datetime.

        Returns:
            1 = Asia, 2 = London, 3 = NY, 0 = Off Hours
        """
        hour = self.get_et_hour(dt)

        if 19 <= hour or hour < 3:  # 7pm - 3am ET
            return 1  # ASIA
        elif 3 <= hour < 8:  # 3am - 8am ET
            return 2  # LONDON
        elif 8 <= hour < 16:  # 8am - 4pm ET
            return 3  # NEW_YORK
        return 0  # OFF_HOURS

    def next(self):
        # Current datetime
        dt = self.data.datetime.datetime(0)
        current_session = self.get_session_code(dt)
        self.lines.current_session[0] = current_session

        # Store bar data for lookback
        self.bar_cache.append({
            'datetime': dt,
            'high': self.data.high[0],
            'low': self.data.low[0],
            'session': current_session
        })

        # Calculate session levels from lookback
        lookback_bars = int(self.p.lookback_hours * 4)  # Assuming 15m bars
        lookback_bars = min(lookback_bars, len(self.bar_cache))

        asia_highs, asia_lows = [], []
        london_highs, london_lows = [], []
        ny_highs, ny_lows = [], []
        day_highs, day_lows = [], []

        # Get today's date for prev day calculation
        current_date = dt.date()

        for bar in list(self.bar_cache)[-lookback_bars:]:
            session = bar['session']
            bar_date = bar['datetime'].date()

            if session == 1:  # ASIA
                asia_highs.append(bar['high'])
                asia_lows.append(bar['low'])
            elif session == 2:  # LONDON
                london_highs.append(bar['high'])
                london_lows.append(bar['low'])
            elif session == 3:  # NY
                ny_highs.append(bar['high'])
                ny_lows.append(bar['low'])

            # Previous day levels (any bar from yesterday)
            if bar_date < current_date:
                day_highs.append(bar['high'])
                day_lows.append(bar['low'])

        # Set session levels
        self.lines.asia_high[0] = max(asia_highs) if asia_highs else 0
        self.lines.asia_low[0] = min(asia_lows) if asia_lows else float('inf')
        self.lines.london_high[0] = max(london_highs) if london_highs else 0
        self.lines.london_low[0] = min(london_lows) if london_lows else float('inf')
        self.lines.ny_high[0] = max(ny_highs) if ny_highs else 0
        self.lines.ny_low[0] = min(ny_lows) if ny_lows else float('inf')
        self.lines.prev_day_high[0] = max(day_highs) if day_highs else 0
        self.lines.prev_day_low[0] = min(day_lows) if day_lows else float('inf')


class SessionLevel:
    """
    Data class representing a session level (high or low).
    """

    def __init__(
        self,
        session_name: str,
        level_type: str,  # 'HIGH' or 'LOW'
        price: float,
        formed_time=None
    ):
        self.session_name = session_name
        self.level_type = level_type
        self.price = price
        self.formed_time = formed_time
        self.swept = False
        self.swept_time = None

    def check_sweep(self, high: float, low: float, close: float) -> bool:
        """
        Check if this level was swept.

        A sweep occurs when price wicks beyond the level but closes back.

        Args:
            high: Current candle high
            low: Current candle low
            close: Current candle close

        Returns:
            True if level was swept
        """
        if self.swept:
            return False

        if self.level_type == 'LOW':
            # Sweep of low: wick below, close above
            if low < self.price and close > self.price:
                self.swept = True
                return True
        else:  # HIGH
            # Sweep of high: wick above, close below
            if high > self.price and close < self.price:
                self.swept = True
                return True

        return False

    def __repr__(self):
        return f"SessionLevel({self.session_name} {self.level_type}: {self.price:.2f})"
