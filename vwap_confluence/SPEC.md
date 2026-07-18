# VWAP Confluence (Tom Crown) — spec v1 (2026-07-18)

> **VERDICT (2026-07-18): REFUTED for deploy — real feature, not a trigger.**
> Pullback-to-VWAP continuation (arm A) has a REAL, time-stable, all-regime
> positive gross on crypto (+0.0255R, clustered-t +8.4, n=82,614; stable in
> both half-splits) — the strongest gross of the 3 YouTube strategies — and
> VWAP-anchoring BEATS a plain EMA pullback (+0.016R over the MA-control). But
> fee-walled everywhere (net_taker −1.15 @5m / −0.70 @15m; net_mm −0.98/−0.60);
> confluence stacking doesn't lift gross; SD-reversion (D) toll-dead (net
> −1.59); edge is short-side-concentrated. NQ EXCLUDED (corrupt multi-scale
> data, gross −1100 outliers). Vindicates the reviewer: VWAP = core CONTEXTUAL
> feature, NOT a standalone trigger. Full write-up: reports/vwap_confluence/
> REPORT.md.


Source: YouTube (Tom Crown VWAP), user transcript. Reviewer read (adopted):
VWAP is **dynamic fair value = the average institutional cost basis**, traded
as **support/resistance in confluence** — never alone. Two mechanisms live in
the video: (1) **pull back to VWAP in a trend and buy the hold** (VWAP as
dynamic support / re-accumulation at cost basis); (2) when **stretched from
VWAP, revert to it** (fair-value magnet / SD-band reversion).

## Closures this challenges + the NEW mechanism (CLAUDE.md gate)

Unlike the ORB (continuation) and OAR (fade) twins, this straddles two
families with OPPOSITE standing verdicts:

- **The pullback-continuation arm is in the family that LIVES.** MM trend is
  live and *survives* (`mm-trend-scoreboard`: "trend SURVIVES live UNLIKE
  fades; trend entry ESCAPES the adverse-selection that kills fades"). A
  trend-pullback-to-a-dynamic-level entry is the same family. The relevant
  cautions are the *confirmation-candle costs ~0.22–0.25R* result (fib-bos /
  knife delayed-entry) and the 5m/15m fee wall — an entry can be in a live
  family and still be toll-dead at scalp resolution. The reviewer's own table
  scores "Moving Average Pullback 7.5/10" as a *separate, weaker* strategy —
  so the honest question is whether **volume-weighted, session-anchored VWAP**
  as the pullback level beats a plain MA-pullback or plain trend entry.
- **The SD-band reversion arm is in the CLOSED fade space** (fade-at-levels
  dead across six families; ny4h/OAR = opening-range fades). VWAP-as-magnet is
  the "new level," but the prior is dead. Tested for completeness.

**NEW mechanism justifying the run:** the pullback/target level is VWAP —
volume-weighted fair value with session/weekly/monthly anchors and volume-SD
bands — not price-only structure. Prereg keystone: does VWAP-anchoring add
anything over the same entry anchored to a plain EMA (the MA-pullback
control), and does multi-VWAP *confluence* select above VWAP alone?

**Honest prior:** the continuation arm may show a real (trend-family) gross
edge but get fee-walled at 5m/15m and pay the confirmation-candle toll; the
reversion arm is likely dead. Tier-3 measurement, NOT a promotion candidate.
Note: the transcript's order-flow "upgrades" (delta, CVD, footprint, volume
profile) need data we do not hold locally — out of scope; VWAP geometry +
candle + multi-anchor + trend is the computable core.

## VWAP construction (from OHLCV+volume; no ticks needed)
- typical price `tp = (h+l+c)/3`. Session VWAP = cumulative(tp·vol)/cumulative
  (vol) from the session anchor, reset each session. Anchors:
  - **session**: 00:00 UTC daily (crypto); first cash bar 09:30 ET (NQ).
  - **weekly**: Monday 00:00 UTC. **monthly**: 1st 00:00 UTC.
- **Session SD bands**: volume-weighted std of `(tp − vwap)` cumulative within
  the session; `±1/±2/±3 SD` levels (dust-guarded: needs ≥ `min_session_bars`).
- All VWAP/band values at bar i use bars ≤ i (causal, no look-ahead).

## Arms (exec TF = 5m alts / 15m ETH,SOL,NQ; window (anchor+warmup, session end])
**A. Pullback-continuation (the video's core), long shown; short mirrored:**
1. **Bias**: `close > session_vwap` (long) and VWAP slope ≥ 0 over `slope_bars`.
2. **Extension then pullback**: price was ≥ `ext_atr`×ATR above VWAP, then a
   bar pulls back to touch VWAP (`low ≤ session_vwap`) while `close ≥
   session_vwap` (VWAP holds as support), within `max_setup_bars`.
3. **Confirmation** candle at the touch: bullish engulf OR hammer OR a
   close back above VWAP that reclaims the prior bar's high. Enter at close.
4. **Stop** = min(low over the pullback) − `stop_buffer_atr`×ATR (below VWAP
   support). **Target** = fixed `rr` R (default 2) [`target=fixed`] or the
   pre-pullback swing high [`target=swing`]. Passive; session-flat.
**B. A + HTF-trend confluence**: also require exec EMA200 trend-align
   (`entry>ema200` for long).
**C. A + weekly-VWAP alignment**: also require `close > weekly_vwap`
   (session and weekly VWAP agree = multi-participant confluence).
**MA-control**: A with the pullback level = EMA(`ma_len`) instead of VWAP
   (isolates whether VWAP-anchoring beats a plain MA pullback).
**D. SD-band reversion**: price `≥ sd_k`×SD beyond VWAP → fade toward VWAP;
   entry on a reversal candle at the band, stop beyond the extreme, **target =
   session_vwap** (mean-revert to fair value). Both sides.

## Costs
`TransactionCosts.for_asset`. Confirmation entry = taker; stop = taker; report
`net_taker` AND `net_mm` (a fixed/swing target as a resting limit = maker).
`fee_r = round_trip_pct / stop_pct`. Gross-vs-net decomposition leads.

## Per-trade features (entry-time, causal — PRE-fill)
side, session, entry_hour_ny, dow, vwap_dist_atr (entry vs session VWAP),
vwap_slope_atr (per bar), pullback_depth_atr (extension reached before pull),
retrace_frac (pullback vs extension), weekly_align (bool), monthly_align,
htf_trend_align, sd_pos (entry in SD units), touch_wick_atr, confirm_body_ratio,
dist_ema200_pct, stop_atr, stop_pct, fee_r, rr_planned, bars_since_anchor,
break_vol_ratio, regime5, btc_regime5. Outcomes: gross_r, net_taker_r,
net_mm_r, exit_reason, bars_held, mae_r, mfe_r, same_day_resolve.

## Preregistered questions + kill/keep bar
1. **Gross edge in the pullback-continuation core (arm A)?** Prior: trend
   family lives, so a small +gross is plausible. KEEP-worthy only if gross
   > +0.05R, clustered-|t|>2, AND net_taker > 0 on ≥1 tradable slice — else
   fee-walled-but-live (still a null for deployment).
2. **KEYSTONE — does VWAP-anchoring beat a plain MA pullback (MA-control)?**
   If arm A ≈ MA-control, VWAP is not special — it's just a pullback level.
3. **Does confluence STACK select?** A → B → C: does adding HTF-trend or
   weekly-VWAP alignment lift gross monotonically, or is it just fewer trades?
4. **Does SD-band reversion (arm D) survive?** Prior: fade space closed.
5. **NQ (native equity session) vs crypto** transfer.
6. Regime map + winner/loser AUC + family-wise bar (Bonferroni/permutation) +
   half-split. Any winning cell clears the bar or is ceiling-not-edge.

## Known limitations
- OHLC intrabar (stop-first pessimistic); no tick fills. Bar-VWAP uses (h+l+c)
  /3 as intrabar proxy (standard).
- Crypto session VWAP anchored 00:00 UTC (24/7 has no cash open); NQ =
  QQQ-proxy cash session, 15m. Weekly/monthly UTC-anchored.
- No order-flow/delta/CVD/volume-profile — the transcript's flagship
  "institutional upgrades" are unmeasurable here; VWAP geometry + candle +
  multi-anchor + trend is the tested core, stated plainly.
- Candlestick geometry thresholds conventional, not calibrated.
