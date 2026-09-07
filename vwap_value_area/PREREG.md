# Session-VWAP Developing Value-Area Reversion — PREREG (frozen 2026-09-07)

Source: Reddit r/tradingmillionaires post (user-supplied text, 2026-09-07).
Author claims ~$20k/month across 6 funded accounts, NQ primary, 5m entries.

## The strategy as stated (literal reading)

Indicator: **session VWAP with developing value areas**. Author's own mapping:
centre line = "point of control"; first developing value area = ~70% of session
volume; second = ~95%. That is the textbook VWAP volume-SD band construction
(+-1 SD ~= 68.3%, +-2 SD ~= 95.4%). POC == VWAP centre line in his usage.
"Developing" = anchored cumulative, widening through the session.

Entry (fade back to fair value):
1. Price closes BEYOND the 2-SD band (the "95% level").
2. Then a later 5m bar CLOSES back inside the 2-SD envelope.
3. Then a later 5m bar CLOSES back inside the 1-SD envelope ("70%").
   ENTER at that close. Short if the excursion was above; long if below.
"No confirmation, no trade." "Taking the touch instead of the close is what
makes people lose money with this."

Stop: swing high/low of the range that produced the signal. Author reports
**37-75 NQ points**.
Targets: T1 = POC(VWAP); T2 = opposite 1-SD; T3 = opposite 2-SD.
Author reports **RR 1:1.5 to 1:2 sweet spot**, 1:3 only when conditions allow.
Early exit: if long and price crosses ABOVE the POC then CLOSES back below it,
close early (failed acceptance). Symmetric for short.

Filters: ranging days only ("the single biggest determinant"); no NFP / FOMC /
CPI / PPI / PCE / Fed speakers / presidential speeches; no Asia session on NQ;
window = Asia close -> NY close.

## Closure this challenges + the NEW mechanism (CLAUDE.md gate)

This sits inside the **fade-at-a-visible-level** family, which is CLOSED across
six prior families (knife, LRR, NY4H, structure scalper, PD-level, fib BOS).
It is closest to **vwap-confluence arm D** (SD-band fade -> VWAP reversion,
2026-07-18): gross +0.0135R (t+6.8), win 21%, **net_taker -1.59R = toll-dead**.

THREE mechanism differences justify a narrow re-open. All are about the TOLL,
which is `round_trip_cost / stop_distance` -- not about the direction of the bet:

1. **Entry is far deeper, so the stop is far wider.** Arm D entered AT the band
   while still stretched, stopping just beyond the current bar's extreme (stop
   ~= 1 bar range) and targeting a distant VWAP (rr_med ~4.8, win 21%,
   fee_r 0.7-1.2R). This strategy enters only after price has closed back inside
   1 SD -- roughly halfway home -- and stops at the SWING extreme of the whole
   excursion leg. Wider stop + nearer target = a structurally different toll
   exposure. Arm D's death certificate does not transfer automatically.
2. **The instrument is CME futures, not crypto perps.** Every fee-wall closure in
   this lab is calibrated on crypto perp costs (15-36 bps round trip). NQ futures
   round-trip is ~1 index point on a ~24,000 handle = **~0.4 bps**, ~40x smaller.
   A gross edge that is toll-dead on BTC can be live on NQ. This has never been
   tested here because the NQ series is corrupt (see below).
3. **A ranging-day gate applied ex ante.** Arm D had a regime SPLIT (post hoc);
   the author claims the day-type read, made early in the session, is the single
   biggest determinant. That is a testable entry gate, not a reporting slice.

Honest prior: the fade family is dead six times over and the author's account is
an unverifiable self-report with obvious survivorship/selection pressure. The
one thing that could genuinely be different is the cost arithmetic on futures.
Expectation: small-or-zero gross; the interesting question is whether whatever
gross exists clears NQ's ~0.02R toll rather than crypto's ~0.3R toll.

## Pre-registered questions (frozen BEFORE any result is read)

- **C1 (his explicit claim).** Does two-stage close confirmation beat entering on
  the naive touch of the 2-SD band? Paired on the same armed setups.
- **C2.** Is there a gross edge at all, on 13 crypto symbols x 9 years?
- **C3.** Does it survive costs -- on crypto perp tolls AND on realistic NQ
  futures tolls? Report the breakeven round-trip cost.
- **C4 (his explicit claim).** Is the ranging-day read "the single biggest
  determinant"? Measured as edge conditional on a CAUSAL early-session
  efficiency-ratio read, with a circular-shift placebo so the filter is
  FALSIFIABLE (the 2026-06 unfalsifiable-FWER-scan trap).
- **C5 (KEYSTONE).** Does the developing session-VWAP construction beat a generic
  Bollinger (SMA20 +- 2SD) envelope run through the identical state machine? If
  not, "developing value area" is decoration on plain mean reversion.
- **C6 (faithfulness).** Do the implementation's stops land in the author's
  reported 37-75 NQ points, and RR in 1.5-2? If not, the reading is wrong.

## Kill / keep bar (frozen)

- **KILL** if gross <= 0 on the crypto panel with n > 2,000 -- mechanism absent,
  and the NQ self-report is then a small-sample story.
- **RECORD-ONLY** if gross > 0 but below the NQ toll, or if the effect fails the
  half-split / permutation-FWER / placebo bars.
- **ESCALATE to a Tier-3 forward shadow** only if: gross > 0 with day-clustered
  |t| > 3 AND positive in both half-splits AND surviving max-|t| permutation
  across the arm family AND net-positive at the realistic NQ toll AND beating
  the Bollinger control (C5).
- No parameter is tuned to maximise a result; the arm grid below is the whole
  search space and carries a family-wise bar.

## Arm grid (the entire search space -- FWER applied across it)

Confirmation: `S2` strict two-stage | `S1` permissive one-stage | `TOUCH` (control)
Target:       `T1` POC | `T2` opposite 1SD | `T3` opposite 2SD | `SCALE` thirds
Bands:        `VWAP` developing session | `BB` Bollinger control (C5)

## Standards applied (non-negotiable, from CLAUDE.md)

- Every feature labelled PRE-fill vs POST-fill; only PRE-fill features gate.
- Gross vs fee/slip toll decomposed BEFORE any verdict.
- Day-clustered t-stats; effective-n haircut for same-session clustering.
- Half-split stability + permutation max-|t| across the arm family.
- Exit-reason decomposition (the PD-level trap: time-cap marks flatter means).
- Same-bar stop-share audit (the PD-level trap: stops < 0.5x bar ATR die in the
  touch bar).
- Swing-extreme stops are computed ONLY from bars <= entry bar; resolution
  starts at entry+1 (no running-extreme look-ahead).
- NQ series in duckdb is CORRUPT (closes 255 -> 26,321 in one timeframe: QQQ ETF
  prices mixed with index prices). It is NOT used. NQ is re-sourced fresh.
