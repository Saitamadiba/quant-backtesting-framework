# Opening-Range Breakout + FVG-retest + engulfing — spec v1 (2026-07-17)

> **VERDICT (2026-07-17): REFUTED.** Pooled gross −0.039R, day-clustered
> t = −6.0 (n=29,223, 10 crypto + NQ, 2021–2026) — significantly negative,
> not merely fee-walled. NQ (native) gross ≈ 0 (t −0.16), net −0.50R. No
> rescue from regime, target, or the engulf gate (which only culls the worst
> micro-stop fills). The 3R target hits ~20% < the 25% break-even it needs.
> Full write-up: `reports/orb_fvg/REPORT.md`. Confirms the break-continuation
> closures at the opening-range level; with `ny4h` (the fade side) both
> directions of the opening range are now dead for us.


Source: YouTube "The Only Scalping Strategy You'll Ever Need" (transcript,
user-supplied). The reviewer's own read (correct): this is **not** an FVG
strategy — it is an **Opening-Range Breakout (ORB)** with the FVG used only
as a displacement/retest trigger and an engulfing candle as the order-flow
confirmation. Family: auction-open → initial balance → break → acceptance
→ directional continuation. Native asset = NQ (equities open).

## Closure this challenges + the NEW mechanism (CLAUDE.md gate)

This is a **break-continuation with retest + confirmation** design. Three
standing closures already cover that space:

- `joiner-break-continuation-refuted` (2026-07-14): post-break drift ~0bps
  vs 15.4bps taker hurdle, 0/36 bracket arms positive even GROSS.
- `structure-scalper-refuted` (2026-07-15): SMC BOS-**retest** continuation
  gross ≈ −0.05R, scale-invariant across 5m/15m/1h/4h.
- confirmation-then-enter costs ~0.22–0.25R paired vs a resting limit
  (`fib-bos-refuted`, knife delayed-entry tick result).

**What is genuinely new here, and why it earns a fresh run:**

1. **The level is the opening auction, not an arbitrary structure pivot.**
   joiner/structure-scalper broke *swing/BOS* levels. Here the level is the
   first-candle opening range — the one price reference the auction-theory
   claim says concentrates real institutional volume. Different level-
   selection mechanism ⇒ not pre-answered by a swing-level refutation.
2. **NQ is the native asset.** Our continuation refutations are almost all
   on 24/7 crypto, which has no opening auction at all. NQ at the 09:30 ET
   cash open is where this strategy's thesis actually lives. Testing it on
   NQ is the honest re-open; crypto is the transfer test.
3. **Engulfing = an order-flow-shift gate**, not just "confirmation candle."
   Whether that specific gate selects is an open empirical question.

**Honest prior:** the family is dead more often than not, and the fixed 3R
target + tiny 1-tick stop means the fee/slip toll (~0.15–0.35R at our tiers)
is the a-priori killer at 5m/15m. We expect gross ≈ 0 on crypto and a coin-
flip on NQ. The deliverable is a *measurement* (per-asset, per-regime,
winner/loser anatomy), Tier-3 convention — NOT a promotion candidate.

## Opening range
- **Trading day** = calendar day in `America/New_York` (DST-aware).
- **Opening candle** = the first execution-TF bar at/after the session open.
  - NQ: 09:30 ET (cash open). Exec TF = 15m (only NQ resolution we hold) ⇒
    OR = the 09:30–09:45 bar. Flagged: the source uses a 5m OR; NQ has no
    local 5m, so the OR is one 15m bar — a faithful-as-possible adaptation.
  - Crypto: 09:30 ET **NY-open proxy** (primary). Crypto has no opening
    auction; 09:30 ET is the strongest recurring institutional liquidity
    inflection. `open_hour/open_min` are params; a 00:00 UTC sensitivity arm
    is available. Exec TF = 5m (10 alts with local 5m) ⇒ OR = 09:30–09:35.
    ETH/SOL have no local 5m ⇒ 15m exec arm, clearly flagged.
- OR High = high, OR Low = low of that single opening bar. FIXED for the
  rest of the NY day; discarded at the next session open.
- "No-man's land" = strictly inside [OR Low, OR High] — never an entry zone.

## Signals (per side; long shown, short mirrored)
State machine over execution-TF **body closes**, in the trading window
`(OR close, session_end]`:

1. **BREAKOUT**: a bar **closes** beyond the OR level
   (close > OR High for long; close < OR Low for short). Wicks do not arm.
2. **FVG** (must form after the breakout, in the break direction):
   bullish 3-candle imbalance at bar i ⇒ `low[i] > high[i-2]`; FVG zone =
   `[high[i-2], low[i]]`. Bearish mirrored (`high[i] < low[i-2]`, zone
   `[high[i], low[i-2]]`). Quality filter `fvg_min_atr` (× 15m ATR14) on the
   gap height — 0 = off (baseline); {0.10, 0.25} sensitivity. The FVG must
   sit on the breakout side of the OR level (bullish FVG top ≥ OR High).
3. **RETEST**: a later bar trades back INTO the FVG zone
   (long: `low ≤ FVG_top`; short: `high ≥ FVG_bot`). The bar(s) doing this
   are the retest leg.
4. **ENGULFING** (order-flow-shift confirmation): after ≥1 retest bar, a bar
   whose body **engulfs the prior bar's body** in the trade direction —
   long: `close>open`, `close ≥ open[prev]`, `open ≤ close[prev]`, and the
   prior bar was a down bar (`close[prev] < open[prev]`). Enter at this
   bar's **close**. The engulfed (prior) bar is the "retest candle."
5. Setup **invalidates** (side re-arms from scratch) if, before the engulf:
   a bar closes back beyond the *opposite* OR level, or `max_setup_bars`
   (default 78 exec bars) elapse since breakout, or the FVG is fully filled
   through its far edge without an engulf, or the session ends.
6. **max_trades_per_day = 2** (faithful). Per-side re-arm after any entry.
   Overlapping opens booked independently (dimensionless R, Tier-3).

## Exits
- **Stop** = engulfed (retest) candle's extreme ± `stop_buffer` ("one tick"):
  long stop = `low[engulfed] − buffer`; short = `high[engulfed] + buffer`.
  buffer = `stop_buffer_atr × ATR15` (default 0.0 = literal 1-tick ≈ 0).
  Require risk = |entry − stop| > 0 else skip.
- **Target** = fixed `rr` R from entry (default **3.0**; 2.0 sensitivity).
- No partials, no trail, no break-even (faithful — "don't micromanage").
- **Session flat**: NQ flatten at 16:00 ET; crypto flatten at
  `flat_hour_ny` (default 16:00 ET, the session analog) — books
  mark-to-market R. `max_hold_bars` safety cap too.
- Intrabar on exec-TF OHLC: gap-through fills at open; stop+target both
  touched in one bar → **stop first** (pessimistic).

## Costs
`TransactionCosts.for_asset(symbol)` (BTC/ETH ~0.15% RT, SOL ~0.21%, alts
~0.36% framework default, NQ ~0.08% RT). Entry (engulf close) and stop are
**taker**; report `net_taker` (both-taker, honest baseline) AND `net_mm`
(the 3R target as a resting limit = maker, entry+stop taker) — the ORB
target is a fixed level so a maker-TP is legitimate. `fee_r =
round_trip_pct / stop_pct`. With tiny retest stops the fee wall is the
headline: gross vs net decomposition leads every verdict.

## Per-trade features (entry-time, causal — all PRE-fill)
side, session, entry_hour_ny, dow, or_range_pct, or_range_vs_atr15,
break_close_margin_atr (how far the breakout bar closed past OR),
bars_break_to_fvg, fvg_height_atr, fvg_depth_frac (how deep the retest went
into the gap, 0–1), bars_fvg_to_engulf, engulf_body_ratio (engulf body /
engulfed body), engulf_range_atr, retest_leg_bars, stop_pct, stop_atr,
fee_r, rr_planned, dist_ema200_pct, trend_align (vs exec-TF EMA200),
break_vol_ratio (breakout-bar vol / 20-bar avg), mins_since_open,
prior_trades_today, own regime5, btc_regime5.
Outcomes: gross_r, net_taker_r, net_mm_r, exit_reason, bars_held,
mae_r, mfe_r, same_day_resolve.

## Preregistered questions + kill/keep bar
Written BEFORE running (forward-shadow discipline, applied to a historical
run as pre-commitment on what would count as signal):

1. **Gross edge at scale?** Prior = joiner/structure-scalper say no. KEEP-
   worthy only if pooled **gross > +0.05R** with clustered-t |t|>2 AND it
   survives the fee toll on the native asset (NQ net_taker > 0) — else the
   verdict is REFUTED / fee-walled, consistent with the family.
2. **Does NQ (native) beat crypto (transfer)?** Directional read only.
3. **Does the engulfing gate select?** Compare vs a no-engulf variant
   (enter on first retest-close back in FVG direction). If engulf adds no
   gross, the "order-flow shift" claim is decorative.
4. **Regime map + winner/loser anatomy** (recorded regardless of verdict).
5. Any single winning cell from the parameter/regime scan must clear a
   **family-wise bar** (Bonferroni or permutation max-|t|) and a half-split
   stability check, or it is labelled ceiling-not-edge (the ny4h lesson).

## Known limitations
- OHLC intrabar ambiguity (stop-first pessimistic by design); no tick fills.
- NQ is a QQQ-proxy (cash-hours), 15m OR (not 5m); no overnight/globex.
- Crypto opening-auction is a *proxy* — the thesis may simply not apply.
- Regime thresholds full-history percentile-calibrated (mildly in-sample;
  fine for a descriptive map, not a deployable gate).
- No order-flow/delta/volume-profile data — the transcript's "AI
  enhancements" (delta, footprint, DOM) are out of scope; engulf is the only
  order-flow proxy we can compute from OHLC.
