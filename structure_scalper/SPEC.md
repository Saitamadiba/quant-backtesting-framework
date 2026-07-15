# HTF-Bias Structure Scalper — formal specification (v1, 2026-07-15)

Source: YouTube "price action scalping" concept (HTF bias → 5m BOS → retest
→ candlestick confirmation → 2R continuation). Every discretionary element
below is formalized; parameters in `StructConfig`. Trend-continuation
family — unlike the fade families, trend has survived live here (MM trend
scoreboard), so the prior is neutral-to-mildly-positive going in.

## Swing detection (both TFs)
Fractal pivot, N = 3 (configurable): swing high at bar *i* iff
`high[i] >= max(high[i-N..i-1])` and `high[i] > max(high[i+1..i+N])`
(mirror for lows). A pivot is only *known* N bars later — all state
machines consume pivots at confirmation time (`shift(N)` + ffill), never
at formation time. No look-ahead.

## HTF bias (1h)
Objective merge of the video's Steps 1–2: bias flips **bullish** when a 1h
candle CLOSES above the most recent confirmed 1h swing high (fresh pivot —
each pivot can trigger once), **bearish** on a close below the most recent
confirmed swing low. Bias persists until the opposite break. Before the
first break: no bias, no trades. The HH/HL vs LH/LL sequence state
(last 2 swing highs + last 2 swing lows) is recorded per trade as
`structure_seq` (up / down / mixed) rather than hard-required — the video's
visual "recent structure" call is not objectively reproducible, so it
becomes a measured feature, and the strict reading (bias ∧ seq agreement)
is recoverable in analysis.

## 5m sequence (per side, active only while HTF bias agrees)
1. **BOS**: 5m close beyond the most recent confirmed 5m swing (high for
   longs / low for shorts; fresh pivots only). Broken level = L.
2. **Retest**: a later bar's extreme returns to within `retest_tol_atr`
   (0.25) × ATR14(5m) of L. Timeout `max_wait_retest` = 48 bars (4h).
3. **Confirmation** (within `max_wait_confirm` = 12 bars of first touch,
   same-bar allowed), any of, long side shown:
   - *Engulfing*: prev bar bearish, this bar bullish, body engulfs prev
     body, body ≥ 30% of bar range.
   - *Pin/hammer*: lower wick ≥ 55% of range, upper wick ≤ 25% of range,
     lower wick ≥ 2× body, close in top half.
   - *Rejection*: bar touched the retest zone, closed back beyond L, close
     in top 35% of range, bullish body.
   First match booked as `confirm_type` (engulf > pin > reject).
4. **Entry** at confirmation close.
5. **Cancel** while waiting: HTF bias flip; opposite 5m structural close
   (close beyond the most recent confirmed opposite swing); either timeout.
6. After entry or cancel the side machine returns to IDLE and requires a
   NEW BOS. Overlapping open trades allowed (independent R bookkeeping,
   Tier-3 convention).

## Exits
- Stop, both variants booked as paired arms for every entry:
  - `confirm` arm — beyond the confirmation candle extreme − 0.25 ATR buffer.
  - `pullback` arm — beyond the pullback extreme since BOS − 0.25 ATR buffer.
- Target fixed 2R. No partials / trailing / BE. Same pessimistic intrabar
  resolver as ny4h (gap fills at open, stop wins ties), `max_hold` 48h.

## Costs / universe / regimes
Identical to ny4h_range_reversal: `TransactionCosts.for_asset`, 10-symbol
local 5m universe + ETH/SOL 15m sensitivity arm, own-asset + BTC regime5
percentile-calibrated on 15m, causal attachment.

## Per-trade features
side, confirm_type, session (london / overlap / ny / off), entry hour NY,
dow, structure_seq, bars_since_htf_break, bos_dist_atr, bos_vol_ratio,
bars_to_retest, retest_depth_atr, retrace_frac (pullback depth vs impulse,
1.0 = exactly to L), bars_to_confirm, confirm_body_frac, confirm_range_atr,
stop_pct, fee_r, dist_ema200_pct, trend_align (5m EMA200), prior signals
that day, regime5, btc_regime5. Outcomes: gross_r, net_r, exit reason,
bars held, MAE/MFE R.

## Faithful-arm choices (measured, not filtered)
No session filter, no ATR/volume/news filters, no risk-model halts — the
video's optional filters become recorded features so the analysis can show
what they WOULD have done. Sessions: London 03–08 NY-time, overlap 08–11,
NY 11–17, off otherwise.
