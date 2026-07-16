# Predefined-Level Range Fade ("plan → wait → confirm") — spec v1, 2026-07-16

Source: YouTube #5 (Jason Casper) — a probability/execution framework: mark
levels before the session, wait for price, demand confirmation, tiny stop,
huge RR, scratch if no immediate reaction, scale out. This spec is the
faithful mechanical core; the discretionary parts are formalized exactly
as the accompanying critique proposes (5-bar / 0.4-ATR reaction rule etc.).

## Levels (predefined = known before the trading day)
PDH / PDL = previous America/New_York day's high/low (prev day needs ≥200
5m bars). The map is fixed at day start — nothing is drawn intraday. If
the day OPENS beyond a level, that side is skipped (level already broken).

## Episodes (per side, first touch of the day only)
- Touch: bar low ≤ PDL (long side) / bar high ≥ PDH (short side).
- **Arm A `limit`** — resting limit AT the level (predefined ⇒ causally
  clean intrabar fill; gap-through fills at open; fill-then-stop in the
  same bar books −1, pessimistic). Stop = level − 0.35·ATR14(exec) beyond;
  the video's "tiny stop".
- **Arm B `diverge`** — after the touch, ≤24 bars for a bullish (bearish)
  RSI14 divergence: a bar setting a NEW episode extreme whose RSI exceeds
  the RSI at the previous extreme bar. Enter at that bar's close. Stop =
  episode extreme − 0.25·ATR. Episode cancels on: close beyond the level
  by >1.5·ATR (range assumption dead), opposite-level touch, or timeout.
- Target (both arms) = the OPPOSITE prior-day level (buy low, sell high
  across the prior-day range). rr_planned recorded (typically 5–40).
- One trade per (day, side, arm); overlapping trades bookkept
  independently (Tier-3 dimensionless R).

## Exit variants (computed per trade in one resolver walk)
- `r_plain` — stop/target only, pessimistic intrabar (stop first), gap
  fills at open, 576-bar cap.
- `r_scratch` — plain + the reaction rule: 5 bars after entry, if the
  close is < +0.4·ATR(entry) in favor and the trade is still open, exit at
  that close.
- `r_scaleout` — 50% banked at +1R, stop to breakeven, runner to the
  opposite level (composite R = 0.5·1 + 0.5·runner; BE exit = +0.5R).

## Costs
Arm A entry is maker by construction: net_mm = gross − (2bps entry + exit
cost)/stop_pct with exit = 2bps on target-limit fills, one-way taker on
stops/scratches. Arm B entry is taker. Full-taker net also shown for both.
**A-priori fee wall: a 0.35-ATR stop at 5m is ~0.04–0.05% wide → taker
round-trip toll ≈ 2.5–3.7R.** The gross column carries the verdict.

## Universe / TFs
5m (10-symbol universe) + 15m (all 12 incl. ETH/SOL). Regime5 (own+BTC)
attached causally as always.

## Features per trade
side, arm, session, entry_hour_ny, dow, pd_range_atr, pd_range_rel20 (vs
20-day mean), gap_open_pct, bars_since_day_open, sweep_depth_atr (extreme
beyond level at entry), vol_ratio_touch, dist_ema200_pct, rsi_entry,
rr_planned, stop_pct, fee_r, other_side_touched_first, regime5,
btc_regime5. Outcomes: r_plain/r_scratch/r_scaleout gross + nets, exit
reasons, bars held, MAE/MFE.

## Preregistered questions (in order)
1. Does the blind level fade (A) have ANY gross edge at these RRs?
2. Does divergence confirmation (B) add or cost, paired on shared episodes?
3. Does the scratch rule improve expectancy (it is the video's mitigation
   for tiny stops)?
4. Does scale-out beat single-target?
5. Standard regime map + winner/loser anatomy.
