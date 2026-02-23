"""
Fair Value Gap (FVG) Detector Indicator for Backtrader.

Detects price imbalances (gaps) where price moved too fast,
leaving an unfilled zone that often gets revisited.
"""

import backtrader as bt


class FVGDetector(bt.Indicator):
    """
    Fair Value Gap Detector

    A FVG forms when there's a gap between candle[i-2]'s high/low and candle[i]'s low/high.

    Bullish FVG (gap up):
        - High of candle[i-2] < Low of candle[i]
        - Zone: High[i-2] to Low[i]

    Bearish FVG (gap down):
        - Low of candle[i-2] > High of candle[i]
        - Zone: High[i] to Low[i-2]

    Lines:
        - bullish_fvg: 1 when bullish FVG detected, else 0
        - bearish_fvg: 1 when bearish FVG detected, else 0
        - fvg_top: Top price of the FVG zone
        - fvg_bottom: Bottom price of the FVG zone
        - fvg_size: Size of the FVG in price units
    """

    lines = ('bullish_fvg', 'bearish_fvg', 'fvg_top', 'fvg_bottom', 'fvg_size')

    params = (
        ('min_gap_pct', 0.001),  # 0.1% minimum gap size
    )

    plotinfo = dict(
        plot=False,  # Don't plot by default
        subplot=False,
    )

    def __init__(self):
        pass

    def next(self):
        # Need at least 3 bars
        if len(self) < 3:
            self.lines.bullish_fvg[0] = 0
            self.lines.bearish_fvg[0] = 0
            self.lines.fvg_top[0] = 0
            self.lines.fvg_bottom[0] = 0
            self.lines.fvg_size[0] = 0
            return

        # Candle references:
        # self.data.high[-2] = candle 0 (oldest)
        # self.data.high[-1] = candle 1 (middle)
        # self.data.high[0]  = candle 2 (current/newest)

        high_0 = self.data.high[-2]  # Oldest candle high
        low_0 = self.data.low[-2]    # Oldest candle low
        high_2 = self.data.high[0]   # Current candle high
        low_2 = self.data.low[0]     # Current candle low
        mid_close = self.data.close[-1]  # Middle candle close (for % calculation)

        # Reset lines
        self.lines.bullish_fvg[0] = 0
        self.lines.bearish_fvg[0] = 0
        self.lines.fvg_top[0] = 0
        self.lines.fvg_bottom[0] = 0
        self.lines.fvg_size[0] = 0

        # Bullish FVG: Gap UP - High[0] < Low[2]
        if high_0 < low_2:
            gap_size = low_2 - high_0
            gap_pct = gap_size / mid_close if mid_close > 0 else 0

            if gap_pct >= self.p.min_gap_pct:
                self.lines.bullish_fvg[0] = 1
                self.lines.fvg_top[0] = low_2
                self.lines.fvg_bottom[0] = high_0
                self.lines.fvg_size[0] = gap_size
                return

        # Bearish FVG: Gap DOWN - Low[0] > High[2]
        if low_0 > high_2:
            gap_size = low_0 - high_2
            gap_pct = gap_size / mid_close if mid_close > 0 else 0

            if gap_pct >= self.p.min_gap_pct:
                self.lines.bearish_fvg[0] = 1
                self.lines.fvg_top[0] = low_0
                self.lines.fvg_bottom[0] = high_2
                self.lines.fvg_size[0] = gap_size


class FVGZone:
    """
    Data class to track an active FVG zone.
    Used by strategies to manage multiple FVG zones.
    """

    def __init__(
        self,
        fvg_type: str,
        top: float,
        bottom: float,
        formed_bar: int,
        formed_time=None
    ):
        self.fvg_type = fvg_type  # 'BULLISH' or 'BEARISH'
        self.top = top
        self.bottom = bottom
        self.midpoint = (top + bottom) / 2
        self.size = abs(top - bottom)
        self.formed_bar = formed_bar
        self.formed_time = formed_time
        self.filled = False
        self.fill_pct = 0.0
        self.traded = False

    def update_fill(self, high: float, low: float) -> float:
        """
        Update the fill percentage based on current price action.

        Args:
            high: Current candle high
            low: Current candle low

        Returns:
            Current fill percentage (0.0 to 1.0)
        """
        if self.filled:
            return self.fill_pct

        if self.fvg_type == 'BULLISH':
            # Price entering from above (retracing down into the gap)
            if low <= self.top:
                fill_depth = self.top - max(low, self.bottom)
                self.fill_pct = min(fill_depth / self.size, 1.0) if self.size > 0 else 0
                if low <= self.bottom:
                    self.filled = True
        else:  # BEARISH
            # Price entering from below (retracing up into the gap)
            if high >= self.bottom:
                fill_depth = min(high, self.top) - self.bottom
                self.fill_pct = min(fill_depth / self.size, 1.0) if self.size > 0 else 0
                if high >= self.top:
                    self.filled = True

        return self.fill_pct

    def is_in_entry_zone(self, fill_min: float = 0.3, fill_max: float = 0.7) -> bool:
        """Check if FVG is in the entry zone (optimal fill range)."""
        return not self.filled and fill_min <= self.fill_pct <= fill_max

    def __repr__(self):
        return (f"FVGZone({self.fvg_type}, top={self.top:.2f}, "
                f"bottom={self.bottom:.2f}, fill={self.fill_pct:.1%})")
