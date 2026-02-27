"""
Session Tracker Indicator for Backtrader.

Tracks trading session highs and lows for ICT-style strategies.
Sessions are defined in Eastern Time (ET), with DST-aware conversion
using zoneinfo (no hardcoded UTC-5 offset).
"""

import backtrader as bt
from collections import deque
from datetime import timezone
from zoneinfo import ZoneInfo


# Shared timezone constants
UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


class SessionTracker(bt.Indicator):
    """
    Tracks trading session highs and lows.

    Sessions (ET):
        - Asia: 7pm - 3am ET (19:00 - 03:00)
        - London: 3am - 8am ET (03:00 - 08:00)
        - New York: 8am - 4pm ET (08:00 - 16:00)

    Lines:
        - asia_high: Highest price during most recent completed Asia session
        - asia_low: Lowest price during most recent completed Asia session
        - london_high: Highest price during most recent completed London session
        - london_low: Lowest price during most recent completed London session
        - ny_high: Highest price during most recent completed NY session
        - ny_low: Lowest price during most recent completed NY session
        - current_session: Current session (1=Asia, 2=London, 3=NY, 0=Off)
        - prev_day_high: Yesterday's high
        - prev_day_low: Yesterday's low
    """

    lines = (
        'asia_high', 'asia_low',
        'london_high', 'london_low',
        'ny_high', 'ny_low',
        'current_session',
        'prev_day_high', 'prev_day_low'
    )

    params = (
        ('lookback_hours', 48),       # Hours to look back for session levels
        ('timeframe_minutes', 15),    # Timeframe in minutes (15, 60, 240, etc.)
    )

    plotinfo = dict(
        plot=False,
        subplot=False,
    )

    def __init__(self):
        # Compute bars per hour from timeframe_minutes (no hardcoded assumption)
        bars_per_hour = 60 / self.p.timeframe_minutes
        max_lookback_bars = int(self.p.lookback_hours * bars_per_hour)

        # Store recent bars for session calculation
        self.bar_cache = deque(maxlen=max(max_lookback_bars, 500))

        # Cache for the most recent completed session levels
        self._last_completed = {
            1: {'high': 0, 'low': float('inf')},   # Asia
            2: {'high': 0, 'low': float('inf')},   # London
            3: {'high': 0, 'low': float('inf')},   # NY
        }
        # Track the session code of the previous bar to detect transitions
        self._prev_session = None
        # Accumulate bars for the *current* (in-progress) session
        self._current_session_highs = []
        self._current_session_lows = []
        self._current_session_code = None

    def _to_et(self, dt):
        """Convert a datetime to Eastern Time (DST-aware)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(ET)

    def get_et_hour(self, dt) -> int:
        """Get hour in ET timezone (DST-aware)."""
        return self._to_et(dt).hour

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

        # --- Session transition detection ---
        # When the session code changes, finalize the previous in-progress session
        if self._prev_session is not None and current_session != self._prev_session:
            prev_code = self._prev_session
            if prev_code in (1, 2, 3) and self._current_session_highs:
                self._last_completed[prev_code] = {
                    'high': max(self._current_session_highs),
                    'low': min(self._current_session_lows),
                }
            # Reset accumulation for the new session
            self._current_session_highs = []
            self._current_session_lows = []

        # Accumulate bars for the current active session
        if current_session in (1, 2, 3):
            self._current_session_highs.append(self.data.high[0])
            self._current_session_lows.append(self.data.low[0])

        self._prev_session = current_session

        # --- Set session levels from most recent COMPLETED sessions ---
        self.lines.asia_high[0] = self._last_completed[1]['high']
        self.lines.asia_low[0] = self._last_completed[1]['low']
        self.lines.london_high[0] = self._last_completed[2]['high']
        self.lines.london_low[0] = self._last_completed[2]['low']
        self.lines.ny_high[0] = self._last_completed[3]['high']
        self.lines.ny_low[0] = self._last_completed[3]['low']

        # --- Previous day high/low (only yesterday's bars) ---
        current_et = self._to_et(dt)
        current_date_et = current_et.date()

        bars_per_hour = 60 / self.p.timeframe_minutes
        lookback_bars = int(self.p.lookback_hours * bars_per_hour)
        lookback_bars = min(lookback_bars, len(self.bar_cache))

        yesterday_highs = []
        yesterday_lows = []

        for bar in list(self.bar_cache)[-lookback_bars:]:
            bar_et = self._to_et(bar['datetime'])
            bar_date_et = bar_et.date()

            # Only include bars from exactly yesterday (ET date)
            day_diff = (current_date_et - bar_date_et).days
            if day_diff == 1:
                yesterday_highs.append(bar['high'])
                yesterday_lows.append(bar['low'])

        self.lines.prev_day_high[0] = max(yesterday_highs) if yesterday_highs else 0
        self.lines.prev_day_low[0] = min(yesterday_lows) if yesterday_lows else float('inf')


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
