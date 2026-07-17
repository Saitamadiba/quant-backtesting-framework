# Multi-TF SMC (HTF FVG → LTF CHOCH → LTF FVG CE entry) — spec v1, 2026-07-17

Source: YouTube #6 (Casper) — 15m context / 1m execution. No local 1m bars
exist, so the design runs at its RATIO: HTF 1h → LTF 5m (12:1, 10-symbol
crypto universe) and HTF 4h → LTF 15m (16:1, all 12 cryptos + NQ).

## HTF layer — context zones
- **HTF FVG**: 3-candle imbalance. Bullish: `low[i] > high[i-2]`; zone =
  [high[i-2], low[i]] (demand, look LONG on retrace into it). Bearish
  mirrored. Zone is knowable only at the close of candle *i* — it becomes
  active for the LTF machine strictly after that timestamp.
- Zone lifecycle: active until FULLY filled (LTF close beyond the far
  edge), or age > 96 HTF bars, or its trade fires (one episode per zone).
- Minimum zone size: 0.15 × HTF ATR14 (dust filter, recorded).

## LTF layer — the sequence (long side; shorts mirrored)
1. **Touch**: LTF price enters the zone (low ≤ zone top).
2. **CHOCH** within 48 LTF bars of the touch: LTF micro-structure was
   bearish coming in (last-2-swings seq = "down", N=3 confirmed pivots)
   and a LTF close breaks above the most recent confirmed LTF swing high
   (fresh pivot). CHOCH leg = from the episode low to the highest high
   within the CHOCH window (tracked forward as it extends).
3. **Fresh LTF FVG** in the trade direction within 12 bars of the CHOCH.
   Entry level = its midpoint (**CE**, the video's Consequential
   Encroachment). Known at the FVG's 3rd-bar close; the resting limit is
   live from the NEXT bar (the fib618 causality lesson, hard-coded).
4. **Fib filter** (recorded + preregistered split at 0.618, not a hard
   gate): entry_leg_retrace = how far the pullback to CE retraces the
   CHOCH leg.
5. **Fill**: price returns to CE within 24 bars (gap-through fills at
   open; fill-then-stop same bar books −1, pessimistic). Cancel pending
   on: LTF close beyond the LTF-FVG far edge, close back below the HTF
   zone bottom − 1 ATR, or timeout.
6. **Stop**: low of the displacement (middle) candle of the LTF FVG — the
   video's "outside the FVG-producing candle", no buffer. **Target**: 4R
   fixed. Exit variants in one resolver walk:
   - `r_plain` — stop/4R only;
   - `r_be` — after +1R trades, stop → entry (BE exit = 0R; pessimistic
     tie-breaks: stop beats target, BE beats target).
7. 576/192-bar hold cap (LTF), time exits marked-to-market and reported
   SEPARATELY (pdlevel time-exit lesson: decompose by exit reason).

## Costs
Entry is a resting limit at a pre-known level ⇒ maker-entry legitimate:
report net_taker (full RT taker, standard) AND net_mm (2bps maker entry;
2bps on 4R-limit target fills, one-way taker on stops). NQ uses
`TransactionCosts.for_asset('NQ')` (~8bps RT).

## Features per trade
side, session, entry_hour_ny, dow, htf_zone_atr (zone size), zone_age_htf,
zone_penetration_frac at CHOCH, choch_leg_atr, choch_break_margin_atr,
displacement_body_frac, displacement_range_atr, ltf_fvg_atr,
entry_leg_retrace, bars_touch_to_choch, bars_choch_to_fvg, bars_to_fill,
vol_ratio_choch, dist_ema200_pct, stop_pct, fee_r, rr_planned (=4),
regime5, btc_regime5. Outcomes: r_plain/r_be + reasons, bars held,
MAE/MFE.

## Preregistered questions
1. Any gross edge in the full hierarchy at scale? (Prior: joiner +
   structure_scalper say LTF-continuation-after-break is dead; the HTF-FVG
   context is the new variable.)
2. Does the 0.618 fib filter select? (split ≥0.618 vs <0.618)
3. Does BE-at-1R help or hurt? (scale-out hurt at high RR in #5; BE at 4R
   targets is the milder cousin)
4. Crypto vs NQ — does the framework transfer to the asset it was taught on?
5. Standard regime map + winner/loser anatomy.
