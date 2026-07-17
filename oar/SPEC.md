# Opening-Auction Exhaustion Reversal — spec v1 (2026-07-17)

> **VERDICT (2026-07-17): REFUTED for deployment — real selector, fee-walled.**
> The exhaustion gate (opening_vol_ratio) is a GENUINE reversion selector: at a
> fixed 2R target, gross + hit-rate rise monotonically with the gate (gross
> −0.007→+0.217R, win 0.348→0.429, clustered-t up to +3.0, clears a 5-point
> Bonferroni bar), surviving the fixed-rr AND first-passage controls — so it is
> NOT a target-distance or volatility artifact. But net_taker is negative at
> EVERY gate (best −0.15R maker, n=382), the effect is weak (Spearman ρ=0.017)
> and decays in the 2nd half, so it is un-monetizable at our costs. NQ (native)
> confirms the shape after a daily-ATR scale-bug fix. Fade space stays closed;
> sharper than ny4h but same wall. Full write-up: `reports/oar/REPORT.md`.


Source: YouTube (user transcript) — "large opening candle → wait → reversal
candle → fade back into the range." Reviewer's institutional read (adopted):
**Opening-Auction Exhaustion + mean-reversion into the initial balance** —
NOT "manipulation." This is the **fade/reversal** twin of the ORB
continuation strategy (`orb_fvg/`, refuted same day). Same phenomenon
(opening auction), opposite bet: fade the overextended open back to the
opposite edge of the opening range.

## Closure this challenges + the NEW mechanism (CLAUDE.md gate)

Fade/reversal at a visible level is CLOSED across six families — knife, LRR,
NY4H, structure-scalper, PD-level, fib-BOS — and specifically
`ny4h-range-reversal-refuted` already faded the **opening range** and got
gross ≈ 0 (best post-hoc cell +0.047 gross / −0.12 taker = ceiling). So the
honest prior is: dead.

**What is genuinely new and earns the run:**
1. **An exhaustion gate, not a bare fade.** ny4h faded any body-close-back-
   inside. Here the fade only fires when the opening move is an *outsized
   fraction of daily ATR* — `opening_vol_ratio = IB_range / daily_ATR`.
   The thesis is that only *overextended* opens exhaust and revert. This is
   a testable selector ny4h did not have. **Keystone test:** does gross rise
   *monotonically* with the gate threshold? If not, the gate is decorative.
2. **Candlestick-exhaustion triggers** (hammer-break / engulfing) as the
   entry, i.e. an order-flow-absorption proxy, vs ny4h's plain close-back-in.
3. **Target = the opposite edge of the IB** (mean-reversion to a level), not
   a fixed R multiple.
4. **NQ is the native asset** (equities open); our fade refutations are
   mostly 24/7 crypto.

**Honest prior:** ny4h says gross ≈ 0. This run lives or dies on whether the
exhaustion gate (#1) manufactures a monotonic gross lift that survives the
fee toll and a family-wise bar. Tier-3 measurement, NOT a promotion
candidate.

## Initial balance (IB)
- Trading day = calendar day in `America/New_York` (DST-aware).
- IB = first 15m bar at/after 09:30 ET (NQ cash open; NY-open proxy for
  crypto). IB_high, IB_low, ib_range, ib_dir (up if close>open).
- `daily_ATR` = Wilder ATR14 on NY-calendar-day bars (crypto: resampled from
  1h; NQ: native 1d), attached CAUSALLY as the **prior** completed day's ATR.
- `opening_vol_ratio = ib_range / daily_ATR`. Day-level gate:
  `ib_ratio_min` (swept: 0.0 / 0.15 / 0.20 / 0.30 / 0.50). 0 = gate off.

## Signals (long = fade a DOWN extension; short mirrored)
Exec TF = 5m (10 alts) / 15m (ETH, SOL, NQ). Window `(IB close, 16:00 ET)`.

1. **Sweep**: price trades beyond the IB edge — long: a bar's `low < IB_low`
   (new session low below IB); short: `high > IB_high`. Tracks the extension
   extreme.
2. **Exhaustion trigger** at/after the sweep (two arms, run separately):
   - **hammer**: bullish hammer near the extension low — lower_wick ≥
     `wick_mult`×body (default 2.0), body ≤ 0.4×range, upper_wick ≤ 0.4×
     range, and the bar swept (`low ≤ IB_low`). Short = shooting star at the
     high. ENTER on a later bar breaking the hammer HIGH (stop-entry); fill
     at `max(open, hammer_high)` (gap-through fills at open).
   - **engulf**: bullish engulfing of the prior (down) bar, prior or current
     bar having swept below IB_low. ENTER at the engulf **close**.
3. **Stop** = exhaustion extreme ± buffer — long: `min(trigger lows) −
   buffer`; buffer = `stop_buffer_atr × ATR15` (default 0).
4. **Target** = opposite IB edge (long → IB_high; short → IB_low)
   [`target_mode='ib'`], or `entry + side·rr·risk` [`target_mode='fixed'`,
   rr=2 sensitivity]. Require target beyond entry and risk>0.
5. Passive: no trail/partials/BE (faithful). max_trades_per_day = 2. Flat at
   16:00 ET (mark-to-market). Setup resets per side after entry or if price
   closes back beyond the opposite IB edge / session ends.
6. OHLC intrabar: gap-through at open; stop+target same bar → **stop first**
   (pessimistic).

## Costs
`TransactionCosts.for_asset`. Entry (engulf close or stop-entry) + stop =
taker; report `net_taker` AND `net_mm` (the IB-target as a resting limit =
maker exit). `fee_r = round_trip_pct / stop_pct`.

## Per-trade features (entry-time, causal — all PRE-fill)
side, session, entry_hour_ny, dow, opening_vol_ratio, ib_range_pct,
sweep_depth_atr (how far beyond IB the extension reached), sweep_depth_frac
(of IB range), bars_ib_to_sweep, bars_sweep_to_entry, trigger_wick_ratio,
trigger_body_frac, trigger_range_atr, dist_to_target_atr, dist_to_target_r
(= rr_planned), stop_atr, stop_pct, fee_r, dist_ema200_pct, trend_align,
break_vol_ratio, mins_since_open, prior_trades_today, own regime5,
btc_regime5. Outcomes: gross_r, net_taker_r, net_mm_r, exit_reason,
bars_held, mae_r, mfe_r, same_day_resolve.

## Preregistered questions + kill/keep bar
1. **Gross edge at scale?** Prior = ny4h says ≈ 0. KEEP-worthy only if
   pooled **gross > +0.05R**, clustered-|t| > 2, AND net_taker > 0 on the
   native asset (NQ). Else REFUTED / fee-walled.
2. **KEYSTONE — does the exhaustion gate select?** Sweep `ib_ratio_min`
   0→0.5. A real mechanism ⇒ gross rises monotonically with the gate. Flat/
   non-monotonic ⇒ the "only overextended opens revert" thesis is false and
   the gate is decorative. (This is the one lever ny4h lacked.)
3. **hammer vs engulf** trigger — does either select?
4. **NQ (native) vs crypto** transfer.
5. **Target**: IB-opposite vs fixed rr=2.
6. Regime map + winner/loser AUC + family-wise bar (Bonferroni/permutation)
   + half-split — any winning cell must clear the bar or it is ceiling-not-
   edge (the ny4h lesson, applied to its own twin).

## Known limitations
- OHLC intrabar (stop-first pessimistic); no local tick fills.
- Crypto daily_ATR resampled from 1h (no native 1d); crypto opening-auction
  is a 09:30-ET proxy — thesis may not apply to 24/7 markets.
- NQ = QQQ cash-hours proxy, 15m IB, n small; no globex.
- No order-flow/delta — hammer/engulf is the only OHLC absorption proxy.
- Hammer/star geometry thresholds are conventional, not calibrated (swept
  qualitatively via wick_mult sensitivity if the gate shows life).
