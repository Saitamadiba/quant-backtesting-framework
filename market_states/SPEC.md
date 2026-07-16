# Five-State Market Framework — formal specification (v1, 2026-07-16)

Source: YouTube #4 — a regime taxonomy (Strong Trend / Wide Trend /
Reversal / Breakout / Range), not an entry strategy. The deliverable is a
CLASSIFIER plus a validation of the framework's central claim: "classify
the market first, then deploy the strategy that historically performs
best in that regime." We test that claim directly on the three recorded
YouTube trade books (ny4h fade, structure_scalper continuation, fib_bos
continuation) and benchmark the taxonomy against our validated regime5.

## Causality rules
All inputs at bar j use data ≤ j: EMAs/ADX/ATR are trailing; pivots are
N=3 fractals consumed at confirmation (+3 bars); Donchian excludes bar j;
the regression channel fits the trailing window ending at j. Event states
(Breakout, Reversal) persist for a fixed window unless re-triggered.

## Per-bar features (exec TF; run on 15m and 1h)
EMA20/50/200, EMA20 slope (5-bar diff), Wilder ADX14, ATR14, ATR
percentile (vs trailing 2000 bars), confirmed-pivot HH/HL sequence label
(up/down/mixed from last 2 swing highs + last 2 swing lows), Donchian
high/low over the prior 48 bars, linear regression over the prior 96
bars: normalized net drift `slope_norm = slope·96/ATR`, residual band
width `(2·resid_std)/ATR`, and residual zero-crossing count (oscillation).

## State definitions (priority order; first match wins)
1. **Breakout** (event, 12-bar window): close beyond the prior 48-bar
   Donchian extreme AND bar range ≥ 1.5·ATR AND body ≥ 55% of range AND
   volume ≥ 1.5×20-bar average. Pre-break compression (Donchian width ≤
   12·ATR) is recorded, not required. Direction recorded.
2. **Reversal** (event, 12-bar window): EMA-stacked trend (20>50>200 or
   mirrored) AND a fresh structural close AGAINST it — close below the
   most recent confirmed swing low in an up-stack (CHOCH), mirrored for
   down. Direction = against the old trend.
3. **Strong Trend**: EMAs fully stacked AND ADX ≥ 25 AND pivot sequence
   agrees (HH/HL for bull stack) AND EMA20 slope sign agrees.
4. **Wide Trend (channel)**: |slope_norm| ≥ 2 AND band width ≥ 3·ATR AND
   residual zero-crossings ≥ 4 in the window (oscillation around a
   drifting line) AND not Strong Trend.
5. **Range**: ADX < 20 AND |slope_norm| < 1 AND ATR percentile < 0.5.
6. **unclassified**: none of the above (reported honestly, not forced
   into the five — the video's "exactly one of five" hides fuzz).

## Validation battery
- Occupancy, median dwell, state-transition matrix (at state changes).
- Forward anatomy per state (next 24 bars): drift (signed by state
  direction where defined), realized vol vs ATR (expansion prediction),
  false-breakout rate (close back inside Donchian within the window).
- **Framework claim test**: per-state gross R of each recorded book
  (ny4h 5m fade; structure 5m confirm + 1h/4h pullback; fib trigger 5m +
  limit618 1h), states attached causally (bar close ≤ entry; 15m states
  for 5m books, 1h states for 1h books). Prescriptions to check:
  fade edge should live in Reversal/Range; continuation edge in Strong
  Trend/Breakout; Range should be no-trade for everything.
- **Benchmark**: eta² (variance in gross R explained by the taxonomy) vs
  our regime5 on the same books.

## Out of scope v1
Confidence scores, ML classifier, adaptive thresholds, HTF blending —
noted as extensions; pointless until the base taxonomy proves it sorts
trades better than what we already run.
