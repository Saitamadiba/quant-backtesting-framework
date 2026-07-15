# NY 4H Range Reversal — formal specification (v1, 2026-07-15)

Source: YouTube "4-Hour Range Reversal Scalping" concept, converted to an
objective spec. This is a **fade-the-failed-breakout / liquidity-sweep
reversion** strategy — the same family as LRR, which was refuted at scale
(see memory `lrr-scaleup-refuted-2026-07-14`). This backtest is a Tier-3
style *measurement*, not a promotion candidate: the deliverable is the
per-regime map and the winner/loser anatomy.

## Range construction
- **Trading day** = calendar day in `America/New_York` (DST-aware).
- **Reference range** = first 4H window of the NY day: bars whose NY open
  time falls in `[00:00, 04:00)`. Range High = max(high), Range Low =
  min(low) over that window. Requires ≥ 40 of 48 5m bars, else the day is
  skipped.
- Levels are FIXED for the rest of the NY day; discarded at the next NY
  midnight.
- Ambiguity in the source: "first 4H candle of the NY trading day" could
  also mean 09:30 ET equity open or 18:00 ET futures open. v1 uses 00:00
  NY (the TradingView-with-NY-timezone reading). `range_start_hour` is a
  parameter for sensitivity arms.

## Signals (execution TF = 5m, body closes only)
Per side, an independent state machine over 5m closes inside the trading
window `[04:00 NY, 24:00 NY)`:

1. ARM: a 5m candle **closes** beyond the level (close < Range Low for
   longs, close > Range High for shorts). Wicks do not arm.
2. While armed, track the excursion extreme (lowest low since arming for
   longs; highest high for shorts), inclusive of the re-entry bar.
3. TRIGGER: a later 5m candle **closes back inside** the range
   (Range Low < close < Range High). Enter at that bar's close.
   - If the close crosses to beyond the *opposite* level instead, the
     setup is invalidated (and the opposite side arms).
   - If the NY day ends while still armed → no trade.
4. After a trigger the side re-arms from scratch. No cap on trades/day.
   Overlapping open trades are allowed and booked independently
   (dimensionless R accounting, Tier-3 convention).

## Exits
- **Stop** = excursion extreme (the "lowest/highest point of the breakout").
- **Target** = fixed 2R from entry.
- No partials, no trailing, no break-even (faithful to source).
- Intrabar resolution on 5m: gap-through-stop fills at the open;
  stop+target both touched in one bar → **stop first** (pessimistic).
- Safety: `max_hold_bars` = 576 (48 h); time exits book mark-to-market R.

## Costs
`TransactionCosts.for_asset(symbol)` from the WFO framework: BTC/ETH
0.15 % round trip, SOL 0.21 %, other alts 0.36 % (framework default —
conservative for thin books). Booked as `fee_r = round_trip_pct / stop_pct`
subtracted from gross R. Both gross and net are reported — with 5m-sized
stops the fee toll is the a-priori killer (see the 15m-economic-floor
memory), so the gross/net split is the headline, not a footnote.

## Universe
All symbols with deep local 5m history in `duckdb_data/trading_data.duckdb`:
ADA, AVAX, BCH, BNB, BTC, DOGE, DOT, LINK, LTC, XRP (5–8 years each,
ending 2026-05-29). ETH and SOL have no local 5m bars; they run as a
**15m-execution sensitivity arm** only, clearly flagged.

## Regime attachment
- Own-asset `regime5` (quiet_chop / quiet_trend / normal_chop /
  normal_trend / vol_expansion) via `regime_classifier.py` on 15m bars,
  thresholds percentile-calibrated per asset (p43/p75/p93 of that asset's
  own atr14_pct distribution — the documented calibration recipe).
- BTC `regime5` attached to every trade as the cross-asset gate candidate
  (the LR-family convention).
- Attachment is causal: last 15m bar whose *close time* ≤ entry time.

## Per-trade features recorded (entry-time, causal)
side, entry hour (NY), minutes since range formed, day-of-week,
range_pct, range_vs_atr15m, sweep_depth_pct, sweep_depth_atr,
bars_outside, reentry_pos (0–1 inside range), stop_pct, fee_r,
tp_fits_in_range, tp_room_frac, breakout volume ratio (vs 20-bar avg),
trend_align (entry vs 5m EMA200), dist_ema200_pct, prior signals that day
(total and same-side), own regime5, BTC regime5.
Outcomes: gross_r, net_r, exit reason, bars held, MAE/MFE in R,
same-day resolution flag.

## Known limitations
- 5m OHLC intrabar ambiguity (stop-first is pessimistic by design).
- Full-history percentile calibration of regime thresholds is mildly
  in-sample; acceptable for a descriptive regime map, not for a deployable
  gate.
- Entry fills assumed at bar close + cost model; no queue/latency model.
