# Fibonacci BOS Continuation — formal specification (v1, 2026-07-16)

Source: YouTube "Fibonacci scalping" concept = BOS continuation with entry
in the 0.50–0.618 retracement ("golden zone") of the impulse leg, stop
beyond the 1.0 fib (impulse origin), target at the previous swing (0.0
fib). Third study in the YouTube series (after ny4h_range_reversal and
structure_scalper). Key design difference vs structure_scalper: entry
LOCATION is deep in the pullback, not at the broken level after a bounce,
and RR is structural (~1–1.5) instead of fixed 2R.

## Timeframes
Video primary = 1m — NOT available locally (5m is the finest bar in
trading_data.duckdb). Ladder run: exec 5m (10-symbol universe), 15m and 1h
(full 12 incl. ETH/SOL). Bar-denominated geometry constant across TFs.

## Micro trend (exec TF)
Confirmed N=3 fractal pivots (lagged consumption, no look-ahead). Trend =
"up" iff last two swing highs ascend AND last two swing lows ascend;
"down" mirrored; else no trades. This replaces the video's visual call.

## Sequence per side (long shown; active only while trend agrees)
1. **BOS**: close above the most recent confirmed swing high (fresh pivot,
   once per pivot). Impulse origin = most recent confirmed swing low at
   BOS time (fib 1.0). Setup skipped if no origin exists.
2. **Impulse tracking**: impulse high (fib 0.0) = running max high since
   BOS; fib prices recomputed as it extends: f50 = H − 0.50·leg,
   f618 = H − 0.618·leg, f786 = H − 0.786·leg (leg = H − origin).
3. **Golden-zone touch**: bar low ≤ f50. Retrace beyond 0.786 ON A CLOSE
   (close < f786) invalidates the setup. Timeout 96 bars after BOS.
4. **Two entry arms booked from the same setup** (paired measurement):
   - **trigger** (faithful): within 6 bars of the last zone touch, a
     rejection candle — engulfing, pin (same quantified detectors as
     structure_scalper), or momentum candle (body ≥ 60% of range, range ≥
     1.2×ATR14, directional close) — closing above f786. Enter at that
     close (taker).
   - **limit618** ("blind fib" probe): resting limit at f618; fills when
     price first trades through it (gap-through fills at open). Maker-
     style entry; measures the entry-location edge without trigger timing.
     **Causality rule (v1.1)**: the intrabar fill prices off the fib grid
     of the PREVIOUS bar's impulse — a same-bar impulse extension is
     close-time knowledge and only moves the resting order from the next
     bar. (v1 priced fills off the same-bar impulse; the implicit
     high-before-low ordering assumption fabricated the arm's entire
     apparent edge — caught by the ambiguous-fill audit, 2026-07-16.)
5. **Stop** (both arms): origin − 0.25·ATR14 buffer. **Target**: impulse
   high frozen at entry (the "previous swing"). rr_planned recorded
   (~0.8–1.6 typical). Entries with rr_planned < 0.5 are skipped — the
   objective form of the video's "price accelerates → cancel" (a trigger
   candle closing back near the impulse high leaves no room to target).
   No partials / trailing / BE in v1 (video's TP1 is the primary exit).
6. **Cancel pending setups on**: trend flip, opposite fresh BOS close,
   0.786 close-through, timeout. Booked trades always run to SL/TP
   (pessimistic intrabar resolver, 576-bar cap).
7. The video's "trend switch after a stop-out" is NOT hard-coded (our
   HH/HL trend detector already flips on structure); it is measured
   instead via the `prior_was_stopout` feature (most recent same-arm trade
   resolved before this entry ended at the stop).

## Costs
`TransactionCosts.for_asset` round-trip taker as standard net. For the
limit618 arm the analysis also shows a maker-entry counterfactual
(entry-side commission+spread removed ≈ RT/2 bound) — labeled as bound.

## Features per trade
side, arm, trigger_type, entry_fib (retrace frac at entry), leg_atr,
leg_pct, bos_dist_atr, zone_touch_bars (BOS→first zone touch),
retrace_depth_frac (deepest retrace before entry), rr_planned, stop_pct,
fee_r, trigger_vol_ratio, dist_ema200_pct, entry_hour_ny, session, dow,
prior_signals_today, prior_was_stopout, regime5, btc_regime5.
Outcomes: gross_r, net_r, exit_reason, bars_held, mae_r, mfe_r.

## Known limitations
5m coarseness vs the video's 1m intent; limit618 fills assume no queue
loss at the level (idealized upper bound, flag in report); full-history
regime threshold calibration as before.
